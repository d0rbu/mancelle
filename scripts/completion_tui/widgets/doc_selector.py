"""Document selector widget for the completion TUI.

Top bar showing a filterable document list. Typing digits filters documents
by prefix. Supports navigation, segment count adjustment, and confirmation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rich.text import Text
from textual.events import Key
from textual.message import Message
from textual.timer import Timer
from textual.widget import Widget

_FILTER_TIMEOUT_S = 2.0


class DocSelector(Widget):
    """Top bar document picker with digit-prefix filtering.

    When focused, typing digits filters the document list by prefix.
    Left/Right navigates the filtered list. Up/Down adjusts segment count.
    Enter confirms selection and posts DocumentSelected.
    """

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    class DocumentSelected(Message):
        """Posted when the user confirms a document selection."""

        def __init__(self, doc_idx: int, num_segments: int) -> None:
            super().__init__()
            self.doc_idx = doc_idx
            self.num_segments = num_segments

    class SelectionChanged(Message):
        """Posted when the highlighted doc or segment count changes."""

        def __init__(
            self,
            doc_idx: int | None,
            num_segments: int,
            total_segments: int,
            gen_prefilled: bool = False,
        ) -> None:
            super().__init__()
            self.doc_idx = doc_idx
            self.num_segments = num_segments
            self.total_segments = total_segments
            self.gen_prefilled = gen_prefilled

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    can_focus = True

    def __init__(
        self,
        available_doc_ids: list[int],
        total_segments_map: dict[int, int],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._all_doc_ids: list[int] = sorted(available_doc_ids)
        self._total_segments_map: dict[int, int] = dict(total_segments_map)

        # Filter state
        self._filter_text: str = ""
        self._filter_active: bool = False
        self._filtered_docs: list[int] = list(self._all_doc_ids)

        # Selection
        self._selected_idx: int = 0

        # Per-doc segment count overrides (cleared on unfocus)
        self._seg_count_overrides: dict[int, int] = {}

        # Per-doc generation prefill toggle (cleared on unfocus)
        self._gen_prefilled: set[int] = set()

        # The doc currently loaded in the text panel
        self._loaded_doc_idx: int | None = self._all_doc_ids[0] if self._all_doc_ids else None

        # Timer for filter timeout (only used when mounted)
        self._filter_timer: Timer | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def filtered_docs(self) -> list[int]:
        return self._filtered_docs

    @property
    def selected_doc_idx(self) -> int | None:
        if not self._filtered_docs:
            return None
        return self._filtered_docs[self._selected_idx]

    @property
    def selected_idx(self) -> int:
        return self._selected_idx

    @property
    def num_segments_for_selected(self) -> int:
        doc = self.selected_doc_idx
        if doc is None:
            return 0
        if doc in self._seg_count_overrides:
            return self._seg_count_overrides[doc]
        total = self._total_segments_map.get(doc, 1)
        return min(2, total)

    @property
    def total_segments_for_selected(self) -> int:
        doc = self.selected_doc_idx
        if doc is None:
            return 0
        return self._total_segments_map.get(doc, 1)

    @property
    def loaded_doc_idx(self) -> int | None:
        return self._loaded_doc_idx

    @loaded_doc_idx.setter
    def loaded_doc_idx(self, value: int | None) -> None:
        self._loaded_doc_idx = value

    @property
    def gen_prefilled_for_selected(self) -> bool:
        return self.selected_doc_idx in self._gen_prefilled

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _apply_filter(self) -> None:
        """Recompute filtered_docs from filter_text."""
        if not self._filter_text:
            self._filtered_docs = list(self._all_doc_ids)
        else:
            self._filtered_docs = [
                d for d in self._all_doc_ids if str(d).startswith(self._filter_text)
            ]
        # Clamp selection
        if self._filtered_docs:
            self._selected_idx = min(self._selected_idx, len(self._filtered_docs) - 1)
        else:
            self._selected_idx = 0
        # Per-doc overrides persist across filter changes.

    def handle_digit(self, digit: int) -> None:
        """Process a digit keypress: append to filter and recompute."""
        if not self._filter_active:
            # Fresh query (after reset/timeout or first digit)
            self._filter_text = str(digit)
            self._filter_active = True
        else:
            self._filter_text += str(digit)
        self._selected_idx = 0
        self._apply_filter()

    def reset_filter(self) -> None:
        """Reset filter text but keep displayed results (non-digit key)."""
        self._filter_text = ""
        self._filter_active = False

    def apply_timeout(self) -> None:
        """Simulate the 2-second inactivity timeout.

        Resets filter text but keeps the currently displayed results.
        """
        self._filter_text = ""
        self._filter_active = False

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def navigate(self, delta: int) -> None:
        """Move selection within the filtered list with wraparound."""
        if not self._filtered_docs:
            return
        self._selected_idx = (self._selected_idx + delta) % len(self._filtered_docs)

    # ------------------------------------------------------------------
    # Segment count
    # ------------------------------------------------------------------

    def adjust_segments(self, delta: int) -> None:
        """Increment/decrement segment count for the selected doc."""
        doc = self.selected_doc_idx
        if doc is None:
            return
        total = self._total_segments_map.get(doc, 1)
        current = self.num_segments_for_selected
        new = max(1, min(total, current + delta))
        self._seg_count_overrides[doc] = new

    def seed_segment_count(self, doc_idx: int, count: int) -> None:
        """Set the initial segment count for a doc (e.g. from loaded state)."""
        self._seg_count_overrides[doc_idx] = count

    def reset_segment_overrides(self) -> None:
        """Clear all per-doc segment overrides (called on unfocus)."""
        self._seg_count_overrides.clear()
        self._gen_prefilled.clear()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    _PREFIX = "Doc "
    _SEPARATOR = ""
    _LEFT_INDICATOR = "< "
    _RIGHT_INDICATOR = " >"
    _ACCENT_STYLE = "bold #e5c07b"

    def _visible_window(self, width: int) -> tuple[int, int]:
        """Return ``(start, end)`` slice of ``_filtered_docs`` that fits in *width*.

        The window is centred on ``_selected_idx``.  When the full list
        cannot fit, space is reserved for ``< `` / `` >`` indicators
        (2 chars each, 4 chars worst-case for both).
        """
        n = len(self._filtered_docs)
        if n == 0:
            return (0, 0)

        max_digits = max(len(str(d)) for d in self._filtered_docs)
        slot_w = max_digits + 2  # "[42]"
        sep = len(self._SEPARATOR)
        prefix = len(self._PREFIX)

        usable = width - prefix
        if usable <= 0:
            # Absurdly narrow — show just the selected doc.
            return (self._selected_idx, self._selected_idx + 1)

        def _slots_in(space: int) -> int:
            if space <= 0:
                return 0
            # k slots = k*slot_w + (k-1)*sep  →  k = (space+sep) // (slot_w+sep)
            return max(1, (space + sep) // (slot_w + sep))

        k = _slots_in(usable)
        if k >= n:
            return (0, n)

        # Truncation needed — reserve for both indicators conservatively.
        indicator_budget = len(self._LEFT_INDICATOR) + len(self._RIGHT_INDICATOR)
        k = max(1, min(_slots_in(usable - indicator_budget), n))

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

    def render_bar(self, width: int, focused: bool = False) -> Text:
        """Render the doc bar into *width* columns as a Rich Text.

        Pure function of widget state + width, split out for testability.
        When *width* is 0 (widget not yet mounted), all docs are shown.
        Brackets on the selected doc are only styled when *focused*.
        """
        doc = self.selected_doc_idx
        if doc is None:
            return Text("Doc (none)")

        n = len(self._filtered_docs)
        if width <= 0:
            start, end = 0, n
        else:
            start, end = self._visible_window(width)

        visible = self._filtered_docs[start:end]

        # Slot width based on ALL filtered docs for stable layout during scroll.
        max_digits = max(len(str(d)) for d in self._filtered_docs)
        slot_w = max_digits + 2

        result = Text(self._PREFIX)
        truncated = not (start == 0 and end == n)
        if truncated:
            result.append(self._LEFT_INDICATOR if start > 0 else " " * len(self._LEFT_INDICATOR))

        for i, d in enumerate(visible):
            if i > 0 and self._SEPARATOR:
                result.append(self._SEPARATOR)
            s = str(d)
            is_selected = d == doc
            is_loaded = d == self._loaded_doc_idx
            inner = f"[{s}]" if is_selected else s
            total_pad = slot_w - len(inner)
            left_pad = total_pad // 2
            right_pad = total_pad - left_pad
            if left_pad:
                result.append(" " * left_pad)
            bracket_style = self._ACCENT_STYLE if (is_selected and focused) else ""
            number_style = self._ACCENT_STYLE if is_loaded else ""
            if is_selected:
                result.append("[", style=bracket_style)
                result.append(s, style=number_style)
                result.append("]", style=bracket_style)
            else:
                result.append(s, style=number_style)
            if right_pad:
                result.append(" " * right_pad)

        if truncated:
            result.append(self._RIGHT_INDICATOR if end < n else " " * len(self._RIGHT_INDICATOR))

        return result

    def render(self) -> Text:
        width = self.size.width if self.size.width > 0 else 0
        return self.render_bar(width, focused=self.has_focus)

    # ------------------------------------------------------------------
    # Key event dispatch
    # ------------------------------------------------------------------

    def _reset_filter_timer(self) -> None:
        """Schedule the filter timeout. Only works when mounted."""
        if self._filter_timer is not None:
            self._filter_timer.stop()
        self._filter_timer = self.set_timer(_FILTER_TIMEOUT_S, self._fire_filter_timeout)

    def _post_selection_changed(self) -> None:
        """Post a SelectionChanged message with current selection state."""
        self.post_message(
            self.SelectionChanged(
                self.selected_doc_idx,
                self.num_segments_for_selected,
                self.total_segments_for_selected,
                gen_prefilled=self.gen_prefilled_for_selected,
            )
        )

    def _fire_filter_timeout(self) -> None:
        self.apply_timeout()
        self._filter_timer = None
        self.refresh()
        self._post_selection_changed()

    def on_key(self, event: Key) -> None:
        key = event.key

        # Digit keys
        if len(key) == 1 and key.isdigit():
            self.handle_digit(int(key))
            self._reset_filter_timer()
            self.refresh()
            self._post_selection_changed()
            event.prevent_default()
            event.stop()
            return

        handler = self._KEY_MAP.get(key)
        if handler is not None:
            handler(self)
            self.refresh()
            self._post_selection_changed()
            event.prevent_default()
            event.stop()

    def _key_left(self) -> None:
        self.reset_filter()
        self.navigate(-1)

    def _key_right(self) -> None:
        self.reset_filter()
        self.navigate(1)

    def _key_up(self) -> None:
        self.reset_filter()
        self.adjust_segments(-1)

    def _key_down(self) -> None:
        self.reset_filter()
        self.adjust_segments(1)

    def _key_enter(self) -> None:
        doc = self.selected_doc_idx
        if doc is not None:
            self.post_message(self.DocumentSelected(doc, self.num_segments_for_selected))

    def _key_g(self) -> None:
        doc = self.selected_doc_idx
        if doc is None:
            return
        if doc in self._gen_prefilled:
            self._gen_prefilled.discard(doc)
        else:
            self._gen_prefilled.add(doc)

    _KEY_MAP: dict[str, Callable[[DocSelector], None]] = {
        "left": _key_left,
        "right": _key_right,
        "up": _key_up,
        "down": _key_down,
        "enter": _key_enter,
        "g": _key_g,
    }
