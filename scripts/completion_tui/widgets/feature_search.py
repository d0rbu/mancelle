"""Feature search overlay for adding new features to a segment.

Digit-based search: type feature ID digits to filter available features,
navigate results, and add a selected feature to the current segment.
"""

from __future__ import annotations

from typing import Any

from textual.events import Key
from textual.message import Message
from textual.widget import Widget

MAX_SEARCH_RESULTS = 10


class FeatureSearch(Widget):
    """Search overlay for finding features to add to a segment.

    Filters available features by feature ID prefix matching.
    Excludes features already present in the segment.
    """

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    class FeatureAdded(Message):
        """Posted when the user selects a feature to add."""

        def __init__(self, feature_id: int) -> None:
            super().__init__()
            self.feature_id = feature_id

    class FeatureSearchClosed(Message):
        """Posted when the user closes the search overlay."""

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        available_features: dict[int, str],
        excluded_ids: set[int],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._available = available_features
        self._excluded = excluded_ids
        self._query: str = ""
        self._results: list[int] = []
        self._selected_idx: int = 0

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def query(self) -> str:
        return self._query

    @property
    def results(self) -> list[int]:
        return self._results

    @property
    def selected_idx(self) -> int:
        return self._selected_idx

    # ------------------------------------------------------------------
    # Query manipulation
    # ------------------------------------------------------------------

    def handle_digit(self, digit: int) -> None:
        """Append a digit to the search query and refresh results."""
        self._query += str(digit)
        self._update_results()

    def handle_backspace(self) -> None:
        """Remove the last digit from the search query."""
        if self._query:
            self._query = self._query[:-1]
        self._update_results()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def move_selection(self, delta: int) -> None:
        """Move selection by delta, clamping to bounds."""
        if not self._results:
            return
        self._selected_idx = max(0, min(len(self._results) - 1, self._selected_idx + delta))

    # ------------------------------------------------------------------
    # Result filtering
    # ------------------------------------------------------------------

    def _update_results(self) -> None:
        """Recompute results based on current query."""
        if not self._query:
            self._results = []
            self._selected_idx = 0
            return
        matching = [
            fid
            for fid in self._available
            if str(fid).startswith(self._query) and fid not in self._excluded
        ]
        matching.sort()
        self._results = matching[:MAX_SEARCH_RESULTS]
        self._selected_idx = 0

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    can_focus = True

    def render(self) -> str:
        """Render the search overlay as a plain string."""
        lines: list[str] = []
        lines.append(f"Search: {self._query}")
        lines.append("")

        if not self._query:
            pass  # Empty query — no results shown
        elif not self._results:
            lines.append("No matching features")
        else:
            for i, fid in enumerate(self._results):
                indicator = "▶" if i == self._selected_idx else " "
                desc = self._available.get(fid, "")
                lines.append(f"{indicator} {fid:>5}  {desc}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Key event dispatch
    # ------------------------------------------------------------------

    def on_key(self, event: Key) -> None:
        """Route key events to the appropriate handler."""
        key = event.key

        if len(key) == 1 and key.isdigit():
            self.handle_digit(int(key))
            self.refresh()
            event.prevent_default()
            event.stop()
            return

        if key == "backspace":
            self.handle_backspace()
            self.refresh()
        elif key == "up":
            self.move_selection(-1)
            self.refresh()
        elif key == "down":
            self.move_selection(1)
            self.refresh()
        elif key in ("space", "enter"):
            if self._results:
                self.post_message(self.FeatureAdded(self._results[self._selected_idx]))
                self.refresh()
        elif key == "escape":
            self.post_message(self.FeatureSearchClosed())
            self.refresh()
        else:
            return

        event.prevent_default()
        event.stop()
