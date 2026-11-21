"""Tests for the SegmentSelector widget.

Covers chip rendering, navigation, toggle, enter message,
seed input, highlight messages, and disabled chip indicator.
"""

from __future__ import annotations

from scripts.completion_tui.color_manager import color_for_idx
from scripts.completion_tui.state import Segment
from scripts.completion_tui.widgets.segment_selector import SegmentSelector
from tests.test_completion_tui._key_dispatch import dispatch_key

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_segments(n: int = 4) -> list[Segment]:
    segments = []
    for i in range(n):
        seg = Segment(
            segment_idx=i,
            char_start=i * 10,
            char_end=(i + 1) * 10,
            color_idx=i,
            features=[],
            enabled=True,
            is_generation_target=(i == n - 1),
        )
        segments.append(seg)
    return segments


def _make_selector(
    segments: list[Segment] | None = None,
) -> SegmentSelector:
    if segments is None:
        segments = _make_segments()
    return SegmentSelector(segments=segments)


# ---------------------------------------------------------------------------
# 1. Chips render with correct colors and indicators
# ---------------------------------------------------------------------------


class TestChipRendering:
    def test_enabled_chip_has_filled_indicator(self) -> None:
        sel = _make_selector()
        rendered = sel.render_bar(width=200).plain
        assert "\u25a0 Seg 0" in rendered

    def test_generation_target_has_asterisk(self) -> None:
        sel = _make_selector()
        rendered = sel.render_bar(width=200).plain
        assert "Seg 3*" in rendered

    def test_correct_segment_labels(self) -> None:
        sel = _make_selector()
        rendered = sel.render_bar(width=200).plain
        for segment_idx in range(4):
            assert f"Seg {segment_idx}" in rendered

    def test_non_target_has_no_asterisk(self) -> None:
        sel = _make_selector()
        rendered = sel.render_bar(width=200).plain
        for segment_idx in range(3):
            assert f"Seg {segment_idx}*" not in rendered


# ---------------------------------------------------------------------------
# 2. Left/Right moves selection
# ---------------------------------------------------------------------------


class TestNavigation:
    def test_right_moves_selection(self) -> None:
        sel = _make_selector()
        assert sel.selected_idx == 0
        sel.navigate(1)
        assert sel.selected_idx == 1

    def test_left_moves_selection(self) -> None:
        sel = _make_selector()
        sel.navigate(1)
        sel.navigate(-1)
        assert sel.selected_idx == 0

    def test_right_wraps_at_end(self) -> None:
        segs = _make_segments(2)
        sel = _make_selector(segments=segs)
        sel.navigate(1)
        sel.navigate(1)  # wraps to start
        assert sel.selected_idx == 0

    def test_left_wraps_at_start(self) -> None:
        segs = _make_segments(3)
        sel = _make_selector(segments=segs)
        sel.navigate(-1)  # wraps to end
        assert sel.selected_idx == 2


# ---------------------------------------------------------------------------
# 3. Space toggles enabled/disabled
# ---------------------------------------------------------------------------


class TestToggle:
    def test_space_toggles_enabled(self) -> None:
        sel = _make_selector()
        assert sel.segments[0].enabled is True
        sel.toggle_enabled()
        assert sel.segments[0].enabled is False
        sel.toggle_enabled()
        assert sel.segments[0].enabled is True

    def test_toggle_on_specific_segment(self) -> None:
        sel = _make_selector()
        sel.navigate(1)  # select segment 1
        sel.toggle_enabled()
        assert sel.segments[1].enabled is False
        assert sel.segments[0].enabled is True  # unaffected


# ---------------------------------------------------------------------------
# 4. Enter posts OpenFeatureEditor
# ---------------------------------------------------------------------------


class TestEnterMessage:
    def test_enter_posts_open_feature_editor(self) -> None:
        sel = _make_selector()
        sel.navigate(1)  # select segment 1
        msgs = dispatch_key(sel, "enter")
        assert len(msgs) == 1
        msg = msgs[0]
        assert isinstance(msg, SegmentSelector.OpenFeatureEditor)
        assert msg.segment_idx == 1

    def test_enter_on_first_segment(self) -> None:
        sel = _make_selector()
        msgs = dispatch_key(sel, "enter")
        assert len(msgs) == 1
        assert isinstance(msgs[0], SegmentSelector.OpenFeatureEditor)
        assert msgs[0].segment_idx == 0


# ---------------------------------------------------------------------------
# 5. Opt+Right posts EnterSeedInput message (seed is now its own widget)
# ---------------------------------------------------------------------------


class TestSeedNavigation:
    def test_tab_posts_enter_seed_input(self) -> None:
        sel = _make_selector()
        msgs = dispatch_key(sel, "tab")
        assert len(msgs) == 1
        assert isinstance(msgs[0], SegmentSelector.EnterSeedInput)

    def test_ctrl_right_posts_enter_seed_input(self) -> None:
        sel = _make_selector()
        msgs = dispatch_key(sel, "ctrl+right")
        assert len(msgs) == 1
        assert isinstance(msgs[0], SegmentSelector.EnterSeedInput)


# ---------------------------------------------------------------------------
# 6. SegmentHighlighted posted when selection changes
# ---------------------------------------------------------------------------


class TestHighlightMessages:
    def test_navigate_right_posts_highlighted(self) -> None:
        sel = _make_selector()
        msgs = dispatch_key(sel, "right")
        highlighted = [m for m in msgs if isinstance(m, SegmentSelector.SegmentHighlighted)]
        assert len(highlighted) == 1
        assert highlighted[0].segment_idx == 1

    def test_navigate_left_posts_highlighted(self) -> None:
        sel = _make_selector()
        sel.navigate(1)  # move to 1 first
        msgs = dispatch_key(sel, "left")
        highlighted = [m for m in msgs if isinstance(m, SegmentSelector.SegmentHighlighted)]
        assert len(highlighted) == 1
        assert highlighted[0].segment_idx == 0

    def test_escape_posts_unhighlighted(self) -> None:
        sel = _make_selector()
        msgs = dispatch_key(sel, "escape")
        assert any(isinstance(m, SegmentSelector.SegmentUnhighlighted) for m in msgs)

    def test_no_highlight_when_single_segment(self) -> None:
        segs = _make_segments(1)
        sel = _make_selector(segments=segs)
        # Single segment: wrapping stays at 0
        msgs = dispatch_key(sel, "left")
        highlighted = [m for m in msgs if isinstance(m, SegmentSelector.SegmentHighlighted)]
        assert highlighted == []


# ---------------------------------------------------------------------------
# 7. Disabled segment chip shows the production disabled indicator
# ---------------------------------------------------------------------------


class TestDisabledChip:
    def test_disabled_chip_has_empty_indicator(self) -> None:
        sel = _make_selector()
        sel.toggle_enabled()  # disable segment 0
        rendered = sel.render_bar(width=200).plain
        assert "\u2591 Seg 0" in rendered
        assert "\u25a0 Seg 0" not in rendered

    def test_other_chips_still_filled(self) -> None:
        sel = _make_selector()
        sel.toggle_enabled()  # disable segment 0 only
        rendered = sel.render_bar(width=200).plain
        assert "\u25a0 Seg 1" in rendered


# ---------------------------------------------------------------------------
# 8. Horizontal scrolling / visible window
# ---------------------------------------------------------------------------


class TestVisibleWindow:
    def test_all_fit_returns_full_range(self) -> None:
        sel = _make_selector(_make_segments(3))
        # Wide enough for all 3 chips + separators
        start, end = sel._visible_window(200)
        assert (start, end) == (0, 3)

    def test_narrow_width_truncates(self) -> None:
        sel = _make_selector(_make_segments(10))
        # Each chip ~7 chars + 2 sep = 9 per slot; 30 chars should show ~3
        start, end = sel._visible_window(30)
        assert end - start < 10
        assert end - start >= 1

    def test_window_centres_on_selection(self) -> None:
        sel = _make_selector(_make_segments(10))
        sel.navigate(5)  # select segment 5
        start, end = sel._visible_window(40)
        assert start <= 5 < end

    def test_window_clamps_at_start(self) -> None:
        sel = _make_selector(_make_segments(10))
        # selected_idx == 0, window should start at 0
        start, end = sel._visible_window(40)
        assert start == 0

    def test_window_clamps_at_end(self) -> None:
        sel = _make_selector(_make_segments(10))
        sel.navigate(-1)  # wraps to segment 9
        start, end = sel._visible_window(40)
        assert end == 10

    def test_empty_segments(self) -> None:
        sel = _make_selector([])
        assert sel._visible_window(200) == (0, 0)


class TestRenderBar:
    def test_render_bar_no_indicators_when_all_fit(self) -> None:
        sel = _make_selector(_make_segments(3))
        bar = sel.render_bar(200)
        assert "< " not in bar.plain
        assert " >" not in bar.plain

    def test_render_bar_shows_right_indicator(self) -> None:
        sel = _make_selector(_make_segments(10))
        bar = sel.render_bar(40)
        assert " >" in bar.plain

    def test_render_bar_shows_left_indicator(self) -> None:
        sel = _make_selector(_make_segments(10))
        sel.navigate(-1)  # wraps to last segment
        bar = sel.render_bar(40)
        assert "< " in bar.plain

    def test_render_bar_width_zero_shows_all(self) -> None:
        sel = _make_selector(_make_segments(10))
        bar = sel.render_bar(0)
        assert "Seg 0" in bar.plain
        assert "Seg 9" in bar.plain

    def test_render_bar_selected_gets_segment_color(self) -> None:
        sel = _make_selector(_make_segments(3))
        bar = sel.render_bar(200, focused=True)
        # Selected segment (idx 0) should use its color, not DEFAULT_COLOR
        assert "Seg 0" in bar.plain

    def test_right_arrow_colored_by_next_segment_unfocused(self) -> None:
        sel = _make_selector(_make_segments(10))
        bar = sel.render_bar(40, focused=False)
        # Right arrow should be colored by the first off-screen segment
        _, end = sel._visible_window(40)
        expected_color = color_for_idx(sel.segments[end].color_idx)
        # Find the " >" span and check its style
        for span in bar._spans:
            text = bar.plain[span.start : span.end]
            if ">" in text:
                assert expected_color in str(span.style)
                break
        else:
            raise AssertionError("No styled '>' span found")

    def test_left_arrow_colored_by_prev_segment_unfocused(self) -> None:
        sel = _make_selector(_make_segments(10))
        sel.navigate(-1)  # wraps to segment 9
        bar = sel.render_bar(40, focused=False)
        start, _ = sel._visible_window(40)
        expected_color = color_for_idx(sel.segments[start - 1].color_idx)
        for span in bar._spans:
            text = bar.plain[span.start : span.end]
            if "<" in text:
                assert expected_color in str(span.style)
                break
        else:
            raise AssertionError("No styled '<' span found")

    def test_arrows_uncolored_when_focused(self) -> None:
        sel = _make_selector(_make_segments(10))
        bar = sel.render_bar(40, focused=True)
        # Right arrow should have no color style when focused
        for span in bar._spans:
            text = bar.plain[span.start : span.end]
            if ">" in text:
                # Style should be empty/default
                assert str(span.style) == "" or span.style is None
                break


# ---------------------------------------------------------------------------
# 9. Chip slot width is stable regardless of generation target state
# ---------------------------------------------------------------------------


class TestChipSlotWidthStable:
    def test_same_width_with_and_without_target(self) -> None:
        """Slot width is identical whether or not any segment is a gen target."""
        segs_with_target = _make_segments(4)  # last has is_generation_target=True
        segs_no_target = _make_segments(4)
        for s in segs_no_target:
            s.is_generation_target = False

        sel_with = _make_selector(segs_with_target)
        sel_without = _make_selector(segs_no_target)
        assert sel_with._chip_slot_width() == sel_without._chip_slot_width()

    def test_slot_width_includes_asterisk_space(self) -> None:
        """Slot width always accounts for the '*' marker character."""
        segs = _make_segments(3)
        for s in segs:
            s.is_generation_target = False
        sel = _make_selector(segs)
        # "■ Seg 2*" is 8 chars — slot width should be at least this
        assert sel._chip_slot_width() >= len("\u25a0 Seg 2*")

    def test_render_bar_same_width_both_modes(self) -> None:
        """Rendered bar has same total width with and without gen target."""
        segs_with = _make_segments(3)
        segs_without = _make_segments(3)
        for s in segs_without:
            s.is_generation_target = False

        sel_with = _make_selector(segs_with)
        sel_without = _make_selector(segs_without)
        bar_with = sel_with.render_bar(200)
        bar_without = sel_without.render_bar(200)
        assert len(bar_with.plain) == len(bar_without.plain)
