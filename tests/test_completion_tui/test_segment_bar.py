"""Tests for the SegmentBar widget.

Covers fill level calculation, visibility control, and render width.
"""

from __future__ import annotations

from scripts.completion_tui.widgets.segment_bar import SegmentBar

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bar(
    loaded_segments: int = 1,
    total_segments: int = 10,
    visible: bool = True,
) -> SegmentBar:
    return SegmentBar(
        loaded_segments=loaded_segments,
        total_segments=total_segments,
        visible=visible,
    )


# ---------------------------------------------------------------------------
# 1. Fill level correct for various segment counts
# ---------------------------------------------------------------------------


class TestFillLevel:
    def test_full_fill_when_all_loaded(self) -> None:
        bar = _make_bar(loaded_segments=10, total_segments=10)
        assert bar.fill_level == 1.0

    def test_small_fill_when_one_loaded(self) -> None:
        bar = _make_bar(loaded_segments=1, total_segments=10)
        # 1/10 = 0.1 — not empty, shows a sliver
        assert bar.fill_level == 0.1

    def test_partial_fill(self) -> None:
        bar = _make_bar(loaded_segments=5, total_segments=10)
        # 5/10 = 0.5
        assert bar.fill_level == 0.5

    def test_full_fill_when_total_is_one(self) -> None:
        bar = _make_bar(loaded_segments=1, total_segments=1)
        assert bar.fill_level == 1.0

    def test_full_fill_when_total_is_zero(self) -> None:
        bar = _make_bar(loaded_segments=0, total_segments=0)
        assert bar.fill_level == 1.0

    def test_half_fill(self) -> None:
        bar = _make_bar(loaded_segments=5, total_segments=10)
        assert bar.fill_level == 0.5

    def test_zero_loaded_with_multiple_total_clamps_to_zero(self) -> None:
        bar = _make_bar(loaded_segments=0, total_segments=10)
        assert bar.fill_level == 0.0


# ---------------------------------------------------------------------------
# 2. Visible only when configured to show
# ---------------------------------------------------------------------------


class TestVisibility:
    def test_visible_when_configured(self) -> None:
        bar = _make_bar(visible=True)
        assert bar.render()

    def test_hidden_when_not_configured(self) -> None:
        bar = _make_bar(visible=False)
        assert bar.render() == ""

    def test_show_and_hide(self) -> None:
        bar = _make_bar(visible=False)
        bar.show_bar()
        assert bar.render()
        bar.hide_bar()
        assert bar.render() == ""


# ---------------------------------------------------------------------------
# 3. Width is 1-2 characters
# ---------------------------------------------------------------------------


class TestRenderWidth:
    def test_width_at_most_three_chars(self) -> None:
        bar = _make_bar(loaded_segments=5, total_segments=10, visible=True)
        rendered = bar.render_bar(height=10)
        for line in rendered:
            assert len(line) <= 3

    def test_uses_block_characters(self) -> None:
        bar = _make_bar(loaded_segments=5, total_segments=10, visible=True)
        rendered = bar.render_bar(height=10)
        all_text = "".join(rendered)
        assert "\u2588" in all_text or "\u2591" in all_text  # █ or ░

    def test_filled_at_bottom_empty_at_top(self) -> None:
        bar = _make_bar(loaded_segments=5, total_segments=10, visible=True)
        rendered = bar.render_bar(height=13)  # 10 bar rows + 3 label rows
        # Bar rows: top = empty, bottom = filled. Last 3 rows are labels.
        bar_rows = rendered[:-3]
        assert "\u2591" in bar_rows[0]  # top = ░
        assert "\u2588" in bar_rows[-1]  # bottom of bar = █

    def test_full_fill_all_filled(self) -> None:
        bar = _make_bar(loaded_segments=10, total_segments=10, visible=True)
        rendered = bar.render_bar(height=8)  # 5 bar rows + 3 label rows
        bar_rows = rendered[:-3]
        for line in bar_rows:
            assert "\u2588" in line  # all █

    def test_zero_fill_all_empty(self) -> None:
        bar = _make_bar(loaded_segments=1, total_segments=10, visible=True)
        rendered = bar.render_bar(height=8)  # 5 bar rows + 3 label rows
        bar_rows = rendered[:-3]
        for line in bar_rows:
            assert "\u2591" in line  # all ░

    def test_render_correct_height(self) -> None:
        bar = _make_bar(loaded_segments=5, total_segments=10, visible=True)
        rendered = bar.render_bar(height=8)
        assert len(rendered) == 8

    def test_label_rows_at_bottom(self) -> None:
        bar = _make_bar(loaded_segments=5, total_segments=10, visible=True)
        rendered = bar.render_bar(height=10)
        # Last 2 rows: loaded count (centred), total count (centred)
        assert rendered[-2].strip() == "5"
        assert rendered[-1].strip() == "10"

    def test_update_segments(self) -> None:
        bar = _make_bar(loaded_segments=1, total_segments=10)
        assert bar.fill_level == 0.1
        bar.update_segments(loaded=10, total=10)
        assert bar.fill_level == 1.0
