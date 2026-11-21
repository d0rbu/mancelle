from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from scripts.completion_tui.widgets.cluster_members import ClusterMembersPanel
from tests.test_completion_tui._key_dispatch import dispatch_key


class _PanelHarness(App):
    def __init__(self, panel: ClusterMembersPanel) -> None:
        super().__init__()
        self.panel = panel
        self.opened_members: list[int] = []
        self.closed_count = 0

    def compose(self) -> ComposeResult:
        yield self.panel

    def on_cluster_members_panel_open_member_detail(
        self, message: ClusterMembersPanel.OpenMemberDetail
    ) -> None:
        self.opened_members.append(message.member_id)

    def on_cluster_members_panel_cluster_members_closed(
        self, message: ClusterMembersPanel.ClusterMembersClosed
    ) -> None:
        del message
        self.closed_count += 1


def test_header_counts() -> None:
    panel = ClusterMembersPanel(
        12345,
        (33210, 12345, 77777),
        {12345: 0.98, 33210: 0.25, 99999: 0.50},
    )
    assert panel._build_header_line().plain == (
        "Cluster @12345 — 3 members (2 active in this segment)"
    )


def test_row_active_evidence_format() -> None:
    panel = ClusterMembersPanel(12345, (12345,), {12345: 0.984})
    panel.update_description(12345, "mentions of chemical elements")
    assert panel._build_member_lines()[0].plain == (
        "▶ 12345* [p0.98 ✓] mentions of chemical elements"
    )


def test_row_inactive_format() -> None:
    panel = ClusterMembersPanel(12345, (33210, 12345), {12345: 0.98})
    panel.update_description(33210, "periodic table references")
    assert panel._build_member_lines()[1].plain == ("  33210  [  —  · ] periodic table references")


def test_members_ascending() -> None:
    panel = ClusterMembersPanel(20, (31, 20, 7), None)
    assert panel.members == (7, 20, 31)
    assert panel.visible_feature_ids(10) == [7, 20, 31]


def test_enter_posts_member_detail() -> None:
    panel = ClusterMembersPanel(20, (7, 20, 31), None)
    panel.move_selection(1)
    messages = dispatch_key(panel, "enter")
    assert len(messages) == 1
    assert isinstance(messages[0], ClusterMembersPanel.OpenMemberDetail)
    assert messages[0].member_id == 20


def test_modifier_scroll_moves_by_five() -> None:
    panel = ClusterMembersPanel(20, tuple(range(10)), None)
    dispatch_key(panel, "ctrl+down")
    assert panel.selected_idx == 5
    dispatch_key(panel, "command+up")
    assert panel.selected_idx == 0


def test_update_evidence_replaces_current_segment_values() -> None:
    panel = ClusterMembersPanel(20, (20, 21), {20: 0.2, 21: 0.3})
    panel.update_evidence({21: 0.91})
    assert panel._build_header_line().plain.endswith("(1 active in this segment)")
    assert "[  —  · ]" in panel._build_member_lines()[0].plain
    assert "[p0.91 ✓]" in panel._build_member_lines()[1].plain


def test_escape_posts_closed() -> None:
    panel = ClusterMembersPanel(20, (7, 20, 31), None)
    messages = dispatch_key(panel, "escape")
    assert len(messages) == 1
    assert isinstance(messages[0], ClusterMembersPanel.ClusterMembersClosed)


def test_update_description_noop_for_unknown() -> None:
    panel = ClusterMembersPanel(20, (7, 20, 31), None)
    before = panel._descriptions.copy()
    panel.update_description(999, "unknown")
    assert panel._descriptions == before


def test_scroll_clamps() -> None:
    panel = ClusterMembersPanel(0, tuple(range(10)), None)
    panel.move_selection(100)
    assert panel.selected_idx == 9
    panel.render_content(height=5)
    assert panel._scroll_offset == 7
    assert panel.visible_feature_ids(3) == [7, 8, 9]
    panel.move_selection(-100)
    panel.render_content(height=5)
    assert panel.selected_idx == 0
    assert panel._scroll_offset == 0


def test_loading_placeholder_render() -> None:
    panel = ClusterMembersPanel(12345, (12345,), None)
    assert panel._build_member_lines()[0].plain.endswith(" ...")


def test_no_evidence_all_dash() -> None:
    panel = ClusterMembersPanel(20, (7, 20, 31), None)
    for line in panel._build_member_lines():
        assert "[  —  · ]" in line.plain
        assert "p0." not in line.plain


@pytest.mark.anyio
async def test_mounted_panel_real_keypress_dispatch() -> None:
    panel = ClusterMembersPanel(20, (7, 20, 31), None)
    app = _PanelHarness(panel)

    async with app.run_test() as pilot:
        panel.focus()
        await pilot.press("down")
        assert panel.members[panel.selected_idx] == 20
        await pilot.press("up")
        assert panel.members[panel.selected_idx] == 7

        await pilot.press("q")
        assert panel.members[panel.selected_idx] == 7
        assert app.opened_members == []
        assert app.closed_count == 0

        await pilot.press("down", "enter")
        await pilot.pause()
        assert app.opened_members == [20]

        await pilot.press("escape")
        await pilot.pause()
        assert app.closed_count == 1
