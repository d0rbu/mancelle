"""Feature conditioning editor panel.

Overlay widget for editing SAE feature conditioning on a per-segment basis.
Shows a scrollable list of features with magnitude editing, toggle, and delete.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from typing import Any

from rich.text import Text
from textual.events import Key
from textual.message import Message
from textual.widget import Widget

from scripts.completion_tui.cluster_view import ClusterView
from scripts.completion_tui.state import (
    FeatureCondition,
    Segment,
    display_magnitude_range,
)

DESCRIPTION_LOADING = "..."


def _sort_key(fc: FeatureCondition) -> tuple[bool, int, int]:
    """Sort key: enabled first, then magnitude descending, then feature_id ascending."""
    return (not fc.enabled, -fc.magnitude, fc.feature_id)


_RIGHT_PANEL_WIDTH = 20
_COLUMN_GAP = 2
_DESC_SCROLL_REPEAT_GAP = 0.3  # max seconds between events to count as held
_DESC_SCROLL_TIERS: list[tuple[float, int]] = [
    # (hold_seconds, step_size) — checked in reverse order
    (1.5, 3),
    (1.0, 2),
    (0.0, 1),
]
_SCROLL_MARGIN = 5  # start scrolling when selection is this close to the edge
_MODIFIER_SCROLL_STEP = 5
_HISTOGRAM_LABEL_WIDTH = 3  # "{mag:>2} " — 2-char right-aligned label + space
_HISTOGRAM_BAR_CHAR = "\u2588"  # full block

_INSTRUCTIONS = [
    "\u2191\u2193  navigate (Cmd/Ctrl+\u2191\u2193 fast)",
    "\u2190\u2192  magnitude",
    "0-9 set directly",
    "-   negative",
    "Spc toggle",
    "BS  delete",
    "N   search",
    "Ent detail/cluster",
    "^R  randomize",
    "R   resample",
    "Esc close",
]


class FeatureEditor(Widget):
    """Editor panel for a segment's feature conditioning list.

    Manages its own sorted feature list, selection cursor, digit accumulator,
    and selection history. Does not require being mounted in a Textual app
    for its core logic — the rendering (compose/render) is separate from
    the state mutation methods tested here.
    """

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    class OpenFeatureSearch(Message):
        """Posted when the user presses N to search for new features."""

    class OpenFeatureDetail(Message):
        """Posted when the user presses Enter on a selected feature."""

        def __init__(self, feature_id: int) -> None:
            super().__init__()
            self.feature_id = feature_id

    class OpenClusterMembers(Message):
        """Posted when Enter opens a composite representative's members."""

        def __init__(self, rep_id: int) -> None:
            super().__init__()
            self.rep_id = rep_id

    class FeatureEditorClosed(Message):
        """Posted when the user presses Esc to close the editor."""

    class SwitchSegmentRequested(Message):
        """Posted when the user presses Alt+Left/Right to switch segments."""

        def __init__(self, delta: int) -> None:
            super().__init__()
            self.delta = delta

    class RandomizeFeaturesRequested(Message):
        """Posted when the user presses Ctrl+R to randomize features from another segment."""

    class ResampleRequested(Message):
        """Posted when the user presses R to resample from the same segment."""

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    _NO_DESCRIPTIONS_WARNING = (
        "Set neuronpedia_model_id & neuronpedia_sae_source in config for descriptions"
    )

    def __init__(
        self,
        segment: Segment,
        *,
        num_buckets: int,
        has_negatives: bool,
        descriptions_available: bool = True,
        sampling_label: str | None = None,
        neg_sampling_label: str | None = None,
        cluster_view: ClusterView | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._segment = segment
        self._num_buckets = num_buckets
        self._has_negatives = has_negatives
        self._descriptions_available = descriptions_available
        self._sampling_label = sampling_label
        self._neg_sampling_label = neg_sampling_label
        self._cluster_view = cluster_view
        self._min_mag, self._max_mag = display_magnitude_range(num_buckets, has_negatives)

        # Deep-copy features so edits don't mutate the caller's list
        self._features: list[FeatureCondition] = copy.deepcopy(segment.features)
        self._sort()

        # Selection
        self._selected_idx: int = 0

        # Digit accumulator state machine
        self._accum_active: bool = False
        self._accum_value: int = 0

        # Selection history (undo/redo)
        self._sel_history: list[int] = []
        self._sel_future: list[int] = []

        # Vertical scroll
        self._scroll_offset: int = 0

        # Per-feature horizontal description scroll (feature_id → offset)
        self._desc_scroll: dict[int, int] = {}
        self._last_desc_width: int = 0  # available chars for description text
        self._desc_scroll_hold_start: float = 0.0
        self._desc_scroll_last_time: float = 0.0

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def sorted_features(self) -> list[FeatureCondition]:
        return self._features

    @property
    def selected_idx(self) -> int:
        return self._selected_idx

    @property
    def segment(self) -> Segment:
        return self._segment

    # ------------------------------------------------------------------
    # Description helpers
    # ------------------------------------------------------------------

    def _display_description(self, feature: FeatureCondition) -> str:
        description = feature.description or ""
        if self._cluster_view is None or not self._cluster_view.is_composite(feature.feature_id):
            return description
        composite_prefix = (
            f"[{self._cluster_view.n_members(feature.feature_id)} composite features]"
        )
        return f"{composite_prefix} {description}" if description else composite_prefix

    def visible_feature_ids(self, scroll_height: int) -> list[int]:
        """Return feature IDs currently visible in the scroll viewport.

        *scroll_height* is the number of feature rows that fit (excluding
        the sticky header).  Returns IDs in display order.
        """
        if not self._features or scroll_height <= 0:
            return []
        end = min(self._scroll_offset + scroll_height, len(self._features))
        return [self._features[i].feature_id for i in range(self._scroll_offset, end)]

    def update_description(self, feature_id: int, description: str) -> None:
        """Set *description* on the feature matching *feature_id*.

        No-op if the feature is not in the list (e.g. deleted since the
        request was made).
        """
        for fc in self._features:
            if fc.feature_id == feature_id:
                fc.description = description
                return

    # ------------------------------------------------------------------
    # Magnitude counting (for histogram)
    # ------------------------------------------------------------------

    def _count_by_magnitude(self) -> dict[int, int]:
        """Count enabled features at each magnitude bucket.

        Returns ``{magnitude: enabled_count}`` for every integer in
        ``[_min_mag, _max_mag]``. Disabled features are excluded — the
        histogram reflects only the active conditioning signal.
        """
        counts: dict[int, int] = dict.fromkeys(range(self._min_mag, self._max_mag + 1), 0)
        for fc in self._features:
            if fc.enabled and fc.magnitude in counts:
                counts[fc.magnitude] += 1
        return counts

    def _build_histogram_lines(self) -> list[Text]:
        """Build horizontal-bar histogram lines from enabled feature magnitudes.

        Returns one ``Text`` line per bucket in display order:
        ``-1`` first (if negatives), then ``0`` through ``max_mag``.
        The negative bucket uses red styling.
        """
        counts = self._count_by_magnitude()
        max_count = max(counts.values(), default=0)
        if max_count == 0:
            return []

        max_bar_width = _RIGHT_PANEL_WIDTH - _HISTOGRAM_LABEL_WIDTH

        ordered = list(range(self._min_mag, self._max_mag + 1))

        lines: list[Text] = []
        for mag in ordered:
            count = counts.get(mag, 0)
            label = f"{mag:>2} "
            line = Text()
            line.append(label)

            if count > 0:
                bar_len = max(1, round(count / max_count * max_bar_width))
                is_neg = mag == -1
                style = "red" if is_neg else ""
                line.append(_HISTOGRAM_BAR_CHAR * bar_len, style=style)

            lines.append(line)
        return lines

    def _build_right_panel_lines(self) -> list[Text]:
        """Build all right-panel lines: counter, instructions, histogram."""
        lines: list[Text] = []

        # Active counter
        enabled = sum(1 for f in self._features if f.enabled)
        total = len(self._features)
        lines.append(Text(f"Active: {enabled}/{total}"))

        if self._sampling_label:
            lines.append(Text(self._sampling_label, style="dim"))
        if self._neg_sampling_label:
            lines.append(Text(self._neg_sampling_label, style="dim"))

        # Blank separator
        lines.append(Text())

        # Instructions
        for hint in _INSTRUCTIONS:
            lines.append(Text(hint))

        # Blank separator
        lines.append(Text())

        # Histogram
        lines.extend(self._build_histogram_lines())

        return lines

    # ------------------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------------

    def _sort(self) -> None:
        self._features.sort(key=_sort_key)

    # ------------------------------------------------------------------
    # Description scrolling
    # ------------------------------------------------------------------

    def scroll_description(self, direction: int) -> None:
        """Scroll the selected feature's description horizontally.

        Clamps so the end of the description can reach the right edge of
        the visible area but never scroll beyond it.  Accelerates to
        ``_DESC_SCROLL_STEP_FAST`` after holding for 2+ seconds.
        """
        if not self._features:
            return
        fc = self._features[self._selected_idx]
        desc = self._display_description(fc)
        if not desc:
            return

        now = time.monotonic()
        if now - self._desc_scroll_last_time > _DESC_SCROLL_REPEAT_GAP:
            # Gap too large — reset hold timer
            self._desc_scroll_hold_start = now
        self._desc_scroll_last_time = now

        held_secs = now - self._desc_scroll_hold_start
        step = 1
        for threshold, s in _DESC_SCROLL_TIERS:
            if held_secs >= threshold:
                step = s
                break

        cur = self._desc_scroll.get(fc.feature_id, 0)
        visible = self._last_desc_width if self._last_desc_width > 0 else len(desc)
        max_scroll = max(0, len(desc) - visible)
        new = max(0, min(cur + direction * step, max_scroll))
        if new == 0:
            self._desc_scroll.pop(fc.feature_id, None)
        else:
            self._desc_scroll[fc.feature_id] = new

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def move_selection(self, delta: int) -> None:
        """Move selection by *delta* positions, clamping to bounds.

        Pushing to selection history and resetting the digit accumulator.
        """
        if not self._features:
            return
        self._deactivate_accumulator()
        old = self._selected_idx
        new = max(0, min(len(self._features) - 1, old + delta))
        if new != old:
            self._push_selection_history(old)
            self._selected_idx = new

    def jump_to_first(self) -> None:
        """Jump selection to the first feature."""
        if not self._features:
            return
        self._deactivate_accumulator()
        old = self._selected_idx
        if old != 0:
            self._push_selection_history(old)
            self._selected_idx = 0

    def jump_to_last(self) -> None:
        """Jump selection to the last feature."""
        if not self._features:
            return
        self._deactivate_accumulator()
        old = self._selected_idx
        last = len(self._features) - 1
        if old != last:
            self._push_selection_history(old)
            self._selected_idx = last

    # ------------------------------------------------------------------
    # Magnitude editing
    # ------------------------------------------------------------------

    def adjust_magnitude(self, delta: int) -> None:
        """Increment/decrement selected feature's magnitude with wraparound."""
        if not self._features:
            return
        self._deactivate_accumulator()
        fc = self._features[self._selected_idx]
        new_mag = fc.magnitude + delta
        if new_mag > self._max_mag:
            new_mag = self._min_mag
        elif new_mag < self._min_mag:
            new_mag = self._max_mag
        fc.magnitude = new_mag

    def handle_digit(self, digit: int) -> None:
        """Process a digit keypress through the accumulator state machine."""
        if not self._features:
            return
        if not self._accum_active:
            self._accum_value = digit
            self._accum_active = True
        else:
            candidate = self._accum_value * 10 + digit
            if candidate > self._max_mag:
                candidate = digit
            self._accum_value = candidate
        # Clamp to valid range and apply
        value = max(self._min_mag, min(self._max_mag, self._accum_value))
        self._features[self._selected_idx].magnitude = value

    def handle_minus(self) -> None:
        """Set selected feature's magnitude to -1 if negatives are allowed."""
        if not self._features:
            return
        if not self._has_negatives:
            return
        self._deactivate_accumulator()
        self._features[self._selected_idx].magnitude = -1

    def _deactivate_accumulator(self) -> None:
        self._accum_active = False
        self._accum_value = 0

    # ------------------------------------------------------------------
    # Toggle / Delete
    # ------------------------------------------------------------------

    def toggle_selected(self) -> None:
        """Toggle the enabled flag on the selected feature."""
        if not self._features:
            return
        self._deactivate_accumulator()
        fc = self._features[self._selected_idx]
        fc.enabled = not fc.enabled

    def delete_selected(self) -> None:
        """Remove the selected feature from the list."""
        if not self._features:
            return
        self._deactivate_accumulator()
        del self._features[self._selected_idx]
        # Clamp selection
        if not self._features:
            self._selected_idx = 0
        elif self._selected_idx >= len(self._features):
            self._selected_idx = len(self._features) - 1

    def replace_features(self, features: list[FeatureCondition]) -> None:
        """Replace the entire feature list with a deep copy of *features*."""
        self._deactivate_accumulator()
        self._features = copy.deepcopy(features)
        self._sort()
        self._selected_idx = 0
        self._scroll_offset = 0
        self._desc_scroll.clear()

    # ------------------------------------------------------------------
    # Selection history (undo / redo)
    # ------------------------------------------------------------------

    def _push_selection_history(self, old_idx: int) -> None:
        self._sel_history.append(old_idx)
        self._sel_future.clear()

    def undo_selection(self) -> None:
        """Go back to the previous selection position."""
        if not self._sel_history:
            return
        self._sel_future.append(self._selected_idx)
        idx = self._sel_history.pop()
        # Clamp — deletions may have shrunk the list since this index was stored
        self._selected_idx = min(idx, max(0, len(self._features) - 1))

    def redo_selection(self) -> None:
        """Go forward to the next selection position."""
        if not self._sel_future:
            return
        self._sel_history.append(self._selected_idx)
        idx = self._sel_future.pop()
        self._selected_idx = min(idx, max(0, len(self._features) - 1))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    can_focus = True

    # Index of the first feature line within _build_lines() output.
    _HEADER_LINES = 2  # "Feature Conditioning: Segment N" + blank line

    def _build_header_lines(self) -> list[Text]:
        """Build the sticky header lines (title + blank)."""
        return [
            Text(f"Feature Conditioning: Segment {self._segment.segment_idx}"),
            Text(),
        ]

    def _build_feature_lines(self, *, focused: bool = True) -> list[Text]:
        """Build scrollable feature list lines.

        When *focused* is False the selection indicator is hidden.
        """
        if not self._features:
            return [Text("  (no features)")]
        lines: list[Text] = []
        for i, fc in enumerate(self._features):
            indicator = "\u25b6" if focused and i == self._selected_idx else " "
            prefix = f"{indicator} {fc.feature_id:>5}  [{fc.magnitude:>2}] "
            desc = self._display_description(fc)
            d_scroll = self._desc_scroll.get(fc.feature_id, 0)
            if d_scroll > 0 and desc:
                desc = desc[d_scroll:]
            loading = fc.description == DESCRIPTION_LOADING
            line = Text()
            style = "dim" if not fc.enabled else ""
            line.append(prefix, style=style)
            if desc:
                line.append(desc, style="dim italic" if loading else style)
            lines.append(line)
        if not self._descriptions_available:
            lines.append(Text())
            lines.append(Text(self._NO_DESCRIPTIONS_WARNING, style="dim italic"))
        return lines

    def _build_lines(self) -> list[Text]:
        """Build all content lines as individual Rich Text objects."""
        return self._build_header_lines() + self._build_feature_lines()

    def _ensure_selected_visible(self, height: int) -> None:
        """Adjust scroll offset so the selected feature line stays visible.

        The offset is relative to the feature lines (not the header).
        Keeps a margin of ``_SCROLL_MARGIN`` rows from the top/bottom edge
        so the list scrolls before the cursor hits the very last row.
        """
        if not self._features or height <= 0:
            return
        margin = min(_SCROLL_MARGIN, (height - 1) // 2)
        if self._selected_idx < self._scroll_offset + margin:
            self._scroll_offset = max(0, self._selected_idx - margin)
        elif self._selected_idx >= self._scroll_offset + height - margin:
            self._scroll_offset = self._selected_idx - height + margin + 1

    def render_content(self, height: int, width: int = 0, *, focused: bool = True) -> list[Text]:
        """Return the visible lines for *height* rows.

        The header is sticky (always visible at the top). Only the feature
        list scrolls. When *width* > 0 and wide enough, renders left
        (feature list) and right (counter + instructions + histogram) columns
        side-by-side. When *width* is 0 or too narrow, returns left-column
        only (backward compatible).
        """
        header_lines = self._build_header_lines()
        feature_lines = self._build_feature_lines(focused=focused)

        if height <= 0:
            height = len(header_lines) + len(feature_lines)

        # Scrollable area = height minus sticky header
        scroll_height = max(0, height - len(header_lines))
        self._ensure_selected_visible(scroll_height)
        max_offset = max(0, len(feature_lines) - scroll_height)
        self._scroll_offset = min(self._scroll_offset, max_offset)
        visible_features = feature_lines[self._scroll_offset : self._scroll_offset + scroll_height]

        visible_left = header_lines + visible_features

        # Left-only fallback
        if width <= 0 or width <= _RIGHT_PANEL_WIDTH + _COLUMN_GAP + 10:
            return visible_left

        # Side-by-side rendering with gap
        left_width = width - _RIGHT_PANEL_WIDTH - _COLUMN_GAP
        # "▶ XXXXX  [YY] " = 14 chars prefix per feature line
        self._last_desc_width = max(0, left_width - 14)
        right_lines = self._build_right_panel_lines()

        result: list[Text] = []
        for i in range(height):
            combined = Text()

            # Left column: pad or truncate to left_width
            if i < len(visible_left):
                left_cell = visible_left[i].copy()
                left_cell.truncate(left_width)
                combined.append_text(left_cell)
                pad = left_width - len(left_cell.plain)
                if pad > 0:
                    combined.append(" " * pad)
            else:
                combined.append(" " * left_width)

            combined.append(" " * _COLUMN_GAP)

            # Right column: pad to _RIGHT_PANEL_WIDTH
            if i < len(right_lines):
                right_cell = right_lines[i].copy()
                right_cell.truncate(_RIGHT_PANEL_WIDTH)
                combined.append_text(right_cell)
                pad = _RIGHT_PANEL_WIDTH - len(right_cell.plain)
                if pad > 0:
                    combined.append(" " * pad)
            else:
                combined.append(" " * _RIGHT_PANEL_WIDTH)

            result.append(combined)
        return result

    def render(self) -> Text:
        """Render the feature list with vertical scrolling and right panel."""
        height = self.size.height if self.size.height > 0 else 0
        width = self.size.width if self.size.width > 0 else 0
        visible = self.render_content(height, width, focused=self.has_focus)
        result = Text()
        for i, line in enumerate(visible):
            if i > 0:
                result.append("\n")
            result.append_text(line)
        return result

    # ------------------------------------------------------------------
    # Key event dispatch
    # ------------------------------------------------------------------

    def on_key(self, event: Key) -> None:
        """Route key events to the appropriate handler."""
        key = event.key

        # Digits
        if len(key) == 1 and key.isdigit():
            self.handle_digit(int(key))
            self.refresh()
            event.prevent_default()
            event.stop()
            return

        handler = self._KEY_MAP.get(key)
        if handler is not None:
            handler(self)
            self.refresh()
            event.prevent_default()
            event.stop()

    def _key_up(self) -> None:
        self.move_selection(-1)

    def _key_down(self) -> None:
        self.move_selection(1)

    def _key_fast_up(self) -> None:
        self.move_selection(-_MODIFIER_SCROLL_STEP)

    def _key_fast_down(self) -> None:
        self.move_selection(_MODIFIER_SCROLL_STEP)

    def _key_left(self) -> None:
        self.adjust_magnitude(-1)

    def _key_right(self) -> None:
        self.adjust_magnitude(1)

    def _key_minus(self) -> None:
        self.handle_minus()

    def _key_space(self) -> None:
        self.toggle_selected()

    def _key_backspace(self) -> None:
        self.delete_selected()

    def _key_escape(self) -> None:
        self.post_message(self.FeatureEditorClosed())

    def _key_enter(self) -> None:
        if self._features:
            fid = self._features[self._selected_idx].feature_id
            if self._cluster_view is not None and self._cluster_view.is_composite(fid):
                self.post_message(self.OpenClusterMembers(fid))
            else:
                self.post_message(self.OpenFeatureDetail(fid))

    def _key_n(self) -> None:
        self.post_message(self.OpenFeatureSearch())

    def _key_cmd_up(self) -> None:
        self.jump_to_first()

    def _key_cmd_down(self) -> None:
        self.jump_to_last()

    def _key_undo(self) -> None:
        self.undo_selection()

    def _key_redo(self) -> None:
        self.redo_selection()

    def _key_prev_segment(self) -> None:
        self.post_message(self.SwitchSegmentRequested(-1))

    def _key_next_segment(self) -> None:
        self.post_message(self.SwitchSegmentRequested(1))

    def _key_randomize(self) -> None:
        self.post_message(self.RandomizeFeaturesRequested())

    def _key_resample(self) -> None:
        self.post_message(self.ResampleRequested())

    def _key_desc_scroll_right(self) -> None:
        self.scroll_description(1)

    def _key_desc_scroll_left(self) -> None:
        self.scroll_description(-1)

    _KEY_MAP: dict[str, Callable[[FeatureEditor], None]] = {
        "up": _key_up,
        "down": _key_down,
        "ctrl+up": _key_fast_up,
        "ctrl+down": _key_fast_down,
        "meta+up": _key_fast_up,
        "meta+down": _key_fast_down,
        "command+up": _key_fast_up,
        "command+down": _key_fast_down,
        "left": _key_left,
        "right": _key_right,
        "minus": _key_minus,
        "hyphen_minus": _key_minus,
        "-": _key_minus,
        "space": _key_space,
        "backspace": _key_backspace,
        "escape": _key_escape,
        "enter": _key_enter,
        "n": _key_n,
        "alt+up": _key_cmd_up,
        "alt+down": _key_cmd_down,
        "ctrl+z": _key_undo,
        "ctrl+shift+z": _key_redo,
        "alt+left": _key_prev_segment,
        "alt+right": _key_next_segment,
        "ctrl+left": _key_prev_segment,  # Option+Left on macOS Ghostty
        "ctrl+right": _key_next_segment,  # Option+Right on macOS Ghostty
        "ctrl+a": _key_desc_scroll_left,  # Cmd+Left on macOS Ghostty
        "ctrl+e": _key_desc_scroll_right,  # Cmd+Right on macOS Ghostty
        "ctrl+r": _key_randomize,
        "r": _key_resample,
    }
