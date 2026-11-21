"""Scrollable submenu for inspecting members of a composite feature."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rich.text import Text
from textual.events import Key
from textual.message import Message
from textual.widget import Widget

_SCROLL_MARGIN = 2
_LOADING_DESCRIPTION = "..."
_MODIFIER_SCROLL_STEP = 5


class ClusterMembersPanel(Widget):
    """Show the original SAE features represented by one composite row."""

    class OpenMemberDetail(Message):
        """Posted when Enter requests detail for the selected member."""

        def __init__(self, member_id: int) -> None:
            super().__init__()
            self.member_id = member_id

    class ClusterMembersClosed(Message):
        """Posted when Escape closes the member submenu."""

    def __init__(
        self,
        rep_id: int,
        members: tuple[int, ...],
        evidence: Mapping[int, float] | None,
        *,
        descriptions_available: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._rep_id = rep_id
        self._members = tuple(sorted(members))
        member_set = set(self._members)
        self._evidence = {
            feature_id: float(percentile)
            for feature_id, percentile in (evidence or {}).items()
            if feature_id in member_set
        }
        initial_description = _LOADING_DESCRIPTION if descriptions_available else ""
        self._descriptions = dict.fromkeys(self._members, initial_description)
        self._selected_idx = 0
        self._scroll_offset = 0

    @property
    def members(self) -> tuple[int, ...]:
        return self._members

    @property
    def representative_id(self) -> int:
        return self._rep_id

    @property
    def selected_idx(self) -> int:
        return self._selected_idx

    def update_description(self, feature_id: int, description: str) -> None:
        """Update a member description, ignoring stale or unrelated results."""
        if feature_id in self._descriptions:
            self._descriptions[feature_id] = description

    def update_evidence(self, evidence: Mapping[int, float]) -> None:
        """Replace member evidence with values computed for the current segment."""
        self._evidence = {
            feature_id: float(percentile)
            for feature_id, percentile in evidence.items()
            if feature_id in self._members
        }

    def visible_feature_ids(self, scroll_height: int) -> list[int]:
        """Return member IDs in the current viewport."""
        if scroll_height <= 0:
            return []
        end = min(self._scroll_offset + scroll_height, len(self._members))
        return list(self._members[self._scroll_offset : end])

    def move_selection(self, delta: int) -> None:
        """Move the member cursor by ``delta``, clamped to the member list."""
        if not self._members:
            return
        self._selected_idx = max(
            0,
            min(len(self._members) - 1, self._selected_idx + delta),
        )

    can_focus = True

    def _build_header_line(self) -> Text:
        return Text(
            f"Cluster @{self._rep_id} — {len(self._members)} members "
            f"({len(self._evidence)} active in this segment)"
        )

    def _build_member_lines(self, *, focused: bool = True) -> list[Text]:
        lines: list[Text] = []
        for index, feature_id in enumerate(self._members):
            indicator = "▶" if focused and index == self._selected_idx else " "
            rep_marker = "*" if feature_id == self._rep_id else " "
            if feature_id in self._evidence:
                evidence = f"[p{self._evidence[feature_id]:.2f} ✓]"
            else:
                evidence = "[  —  · ]"
            description = self._descriptions[feature_id]
            lines.append(Text(f"{indicator} {feature_id:>5}{rep_marker} {evidence} {description}"))
        return lines

    def _ensure_selected_visible(self, height: int) -> None:
        if not self._members or height <= 0:
            return
        margin = min(_SCROLL_MARGIN, (height - 1) // 2)
        if self._selected_idx < self._scroll_offset + margin:
            self._scroll_offset = max(0, self._selected_idx - margin)
        elif self._selected_idx >= self._scroll_offset + height - margin:
            self._scroll_offset = self._selected_idx - height + margin + 1

    def render_content(self, height: int, *, focused: bool = True) -> list[Text]:
        """Render a sticky header and a vertically scrollable member list."""
        header = [self._build_header_line(), Text()]
        rows = self._build_member_lines(focused=focused)
        if height <= 0:
            return header + rows
        scroll_height = max(0, height - len(header))
        self._ensure_selected_visible(scroll_height)
        max_offset = max(0, len(rows) - scroll_height)
        self._scroll_offset = min(self._scroll_offset, max_offset)
        return header + rows[self._scroll_offset : self._scroll_offset + scroll_height]

    def render(self) -> Text:
        height = self.size.height if self.size.height > 0 else 0
        lines = self.render_content(height, focused=self.has_focus)
        result = Text()
        for index, line in enumerate(lines):
            if index > 0:
                result.append("\n")
            result.append_text(line)
        return result

    def on_key(self, event: Key) -> None:
        """Navigate members or post submenu lifecycle messages."""
        if event.key in {"ctrl+up", "meta+up", "command+up"}:
            self.move_selection(-_MODIFIER_SCROLL_STEP)
            self.refresh()
        elif event.key in {"ctrl+down", "meta+down", "command+down"}:
            self.move_selection(_MODIFIER_SCROLL_STEP)
            self.refresh()
        elif event.key == "up":
            self.move_selection(-1)
            self.refresh()
        elif event.key == "down":
            self.move_selection(1)
            self.refresh()
        elif event.key == "enter" and self._members:
            self.post_message(self.OpenMemberDetail(self._members[self._selected_idx]))
        elif event.key == "escape":
            self.post_message(self.ClusterMembersClosed())
        else:
            return
        event.prevent_default()
        event.stop()
