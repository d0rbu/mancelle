"""Segment bar widget for the completion TUI.

Thin vertical fill bar (1-2 chars wide) overlaid on the right edge of TextPanel.
Visible only when DocSelector is focused. Uses block characters for smooth fill.
"""

from __future__ import annotations

import math
from typing import Any

from textual.widget import Widget


class SegmentBar(Widget):
    """Thin vertical fill bar showing segment load progress.

    Fill = (loaded_segments - 1) / (total_segments - 1), or full if
    total_segments <= 1. Uses block characters: filled from the bottom up.
    """

    can_focus = False

    DEFAULT_CSS = """
    SegmentBar {
        width: 3;
    }
    """

    def __init__(
        self,
        loaded_segments: int = 0,
        total_segments: int = 0,
        visible: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._loaded = loaded_segments
        self._total = total_segments
        self._visible = visible

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fill_level(self) -> float:
        """Return fill fraction in [0.0, 1.0].

        Scale starts at 0: loaded/total, so 1 segment out of 10 shows
        a small fill (0.1) rather than being empty.
        """
        if self._total <= 0:
            return 1.0
        return max(0.0, min(1.0, self._loaded / self._total))

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def show_bar(self) -> None:
        self._visible = True

    def hide_bar(self) -> None:
        self._visible = False

    def update_segments(self, loaded: int, total: int) -> None:
        self._loaded = loaded
        self._total = total

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_bar(self, height: int) -> list[str]:
        """Return a list of exactly *height* strings, each 3 chars wide.

        Top of the list is the top of the bar. Fill grows from bottom.
        Up to 2 label rows (loaded, total) are appended if height permits.
        """
        labels = [f"{self._loaded:^3}", f"{self._total:^3}"]
        label_rows = min(len(labels), max(0, height - 1))
        bar_height = max(1, height - label_rows)

        fill = self.fill_level
        filled_rows = max(0, min(bar_height, math.floor(fill * bar_height + 0.5)))
        empty_rows = bar_height - filled_rows

        lines: list[str] = []
        for _ in range(empty_rows):
            lines.append("\u2591" * 3)  # ░░░
        for _ in range(filled_rows):
            lines.append("\u2588" * 3)  # ███
        lines.extend(labels[:label_rows])
        return lines

    def render(self) -> str:
        if not self._visible:
            return ""
        h = self.size.height if self.size.height > 0 else 7
        lines = self.render_bar(height=h)
        return "\n".join(lines)
