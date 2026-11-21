"""Segment selector widget for the completion TUI.

Bottom bar showing segment chips with seed input. Chips display enabled/disabled
state, generation target indicator, and segment color. Posts highlight messages
when selection changes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rich.text import Text
from textual.events import Key
from textual.message import Message
from textual.widget import Widget

from scripts.completion_tui.color_manager import DEFAULT_COLOR, color_for_idx
from scripts.completion_tui.state import Segment


class SegmentSelector(Widget):
    """Bottom bar segment chips with seed input."""

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    class OpenFeatureEditor(Message):
        """Posted when the user presses Enter on a selected segment."""

        def __init__(self, segment_idx: int) -> None:
            super().__init__()
            self.segment_idx = segment_idx

    class SegmentHighlighted(Message):
        """Posted when the segment selection changes."""

        def __init__(self, segment_idx: int) -> None:
            super().__init__()
            self.segment_idx = segment_idx

    class SegmentUnhighlighted(Message):
        """Posted when selection leaves the selector."""

    class EnterSeedInput(Message):
        """Posted when the user wants to focus the seed input widget."""

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    can_focus = True

    def __init__(
        self,
        segments: list[Segment],
        *,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._segments = segments
        self._selected_idx: int = 0
        self._seed: int = seed

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._seed = value

    @property
    def selected_idx(self) -> int:
        return self._selected_idx

    @property
    def segments(self) -> list[Segment]:
        return self._segments

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def navigate(self, delta: int) -> None:
        """Move selection by *delta*, wrapping around."""
        if not self._segments:
            return
        self._selected_idx = (self._selected_idx + delta) % len(self._segments)

    # ------------------------------------------------------------------
    # Toggle
    # ------------------------------------------------------------------

    def toggle_enabled(self) -> None:
        """Toggle the enabled flag on the selected segment."""
        if not self._segments:
            return
        seg = self._segments[self._selected_idx]
        seg.enabled = not seg.enabled

    _SEPARATOR = "  "
    _LEFT_INDICATOR = "< "
    _RIGHT_INDICATOR = " >"

    def _chip_slot_width(self) -> int:
        """Max chip width across all segments (for stable layout during scroll).

        Always includes space for the target marker (*) so layout is
        stable regardless of ``is_generation_target`` state.
        """
        if not self._segments:
            return 0
        max_idx = max(s.segment_idx for s in self._segments)
        return len(f"\u25a0 Seg {max_idx}*")

    def _visible_window(self, width: int) -> tuple[int, int]:
        """Return ``(start, end)`` slice of segments that fits in *width*.

        The window is centred on ``_selected_idx``.  When the full list
        cannot fit, space is reserved for ``< `` / `` >`` indicators.
        """
        n = len(self._segments)
        if n == 0:
            return (0, 0)

        slot_w = self._chip_slot_width()
        sep = len(self._SEPARATOR)

        def _slots_in(space: int) -> int:
            if space <= 0:
                return 0
            return max(1, (space + sep) // (slot_w + sep))

        k = _slots_in(width)
        if k >= n:
            return (0, n)

        # Truncation needed — reserve for both indicators conservatively.
        indicator_budget = len(self._LEFT_INDICATOR) + len(self._RIGHT_INDICATOR)
        k = max(1, min(_slots_in(width - indicator_budget), n))

        # Centre on selected_idx.
        half = k // 2
        start = self._selected_idx - half
        end = start + k

        if start < 0:
            start = 0
            end = min(k, n)
        if end > n:
            end = n
            start = max(0, end - k)

        return (start, end)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_bar(self, width: int, focused: bool = False) -> Text:
        """Render the segment bar into *width* columns as a Rich Text.

        Pure function of widget state + width, split out for testability.
        When *width* is 0 (widget not yet mounted), all segments are shown.
        """
        n = len(self._segments)
        if n == 0:
            return Text()

        if width <= 0:
            start, end = 0, n
        else:
            start, end = self._visible_window(width)

        visible = self._segments[start:end]
        slot_w = self._chip_slot_width()

        result = Text()
        truncated = not (start == 0 and end == n)
        if truncated:
            if start > 0:
                # Color arrow by the segment just off-screen (unless focused).
                left_style = "" if focused else color_for_idx(self._segments[start - 1].color_idx)
                result.append(self._LEFT_INDICATOR, style=left_style)
            else:
                result.append(" " * len(self._LEFT_INDICATOR))

        for vi, seg in enumerate(visible):
            if vi > 0:
                result.append(self._SEPARATOR)
            indicator = "\u25a0" if seg.enabled else "\u2591"
            target = "*" if seg.is_generation_target else ""
            label = f"{indicator} Seg {seg.segment_idx}{target}"
            # Pad to fixed slot width for stable layout during scroll.
            label = label.ljust(slot_w)
            color = color_for_idx(seg.color_idx)
            i = start + vi  # real index
            if focused and i == self._selected_idx:
                result.append(label, style=color)
            elif focused:
                result.append(label, style=DEFAULT_COLOR)
            else:
                result.append(label, style=color)

        if truncated:
            if end < n:
                right_style = "" if focused else color_for_idx(self._segments[end].color_idx)
                result.append(self._RIGHT_INDICATOR, style=right_style)
            else:
                result.append(" " * len(self._RIGHT_INDICATOR))

        return result

    def render(self) -> Text:
        width = self.size.width if self.size.width > 0 else 0
        return self.render_bar(width, focused=self.has_focus)

    # ------------------------------------------------------------------
    # Key event dispatch
    # ------------------------------------------------------------------

    def on_key(self, event: Key) -> None:
        key = event.key
        handler = self._KEY_MAP.get(key)
        if handler is not None:
            handler(self)
            self.refresh()
            event.prevent_default()
            event.stop()

    def _key_left(self) -> None:
        old = self._selected_idx
        self.navigate(-1)
        if self._selected_idx != old:
            self.post_message(self.SegmentHighlighted(self._selected_idx))

    def _key_right(self) -> None:
        old = self._selected_idx
        self.navigate(1)
        if self._selected_idx != old:
            self.post_message(self.SegmentHighlighted(self._selected_idx))

    def _key_space(self) -> None:
        self.toggle_enabled()

    def _key_enter(self) -> None:
        if self._segments:
            self.post_message(self.OpenFeatureEditor(self._selected_idx))

    def _key_escape(self) -> None:
        self.post_message(self.SegmentUnhighlighted())

    def _key_alt_up(self) -> None:
        self.post_message(self.SegmentUnhighlighted())

    def _key_seed(self) -> None:
        self.post_message(self.EnterSeedInput())

    _KEY_MAP: dict[str, Callable[[SegmentSelector], None]] = {
        "left": _key_left,
        "right": _key_right,
        "space": _key_space,
        "enter": _key_enter,
        "escape": _key_escape,
        "alt+up": _key_alt_up,
        "tab": _key_seed,
        "ctrl+right": _key_seed,  # Opt+Right on macOS Ghostty
    }
