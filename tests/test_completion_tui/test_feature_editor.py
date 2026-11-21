"""Tests for the FeatureEditor widget.

Tests the feature conditioning editor's sorting, navigation,
magnitude editing (digit accumulator + left/right wraparound),
selection history, toggle/delete, and message posting.
"""

from __future__ import annotations

from rich.text import Text

from scripts.completion_tui.cluster_view import ClusterView
from scripts.completion_tui.state import FeatureCondition, Segment
from scripts.completion_tui.widgets.feature_editor import (
    _COLUMN_GAP,
    _DESC_SCROLL_TIERS,
    _HISTOGRAM_LABEL_WIDTH,
    _RIGHT_PANEL_WIDTH,
    DESCRIPTION_LOADING,
    FeatureEditor,
)
from tests.test_completion_tui._key_dispatch import dispatch_key

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_segment(
    features: list[FeatureCondition],
    segment_idx: int = 0,
) -> Segment:
    return Segment(
        segment_idx=segment_idx,
        char_start=0,
        char_end=100,
        color_idx=0,
        features=features,
    )


def _make_editor(
    features: list[FeatureCondition],
    num_buckets: int = 11,
    has_negatives: bool = True,
    segment_idx: int = 0,
) -> FeatureEditor:
    seg = _make_segment(features, segment_idx=segment_idx)
    return FeatureEditor(seg, num_buckets=num_buckets, has_negatives=has_negatives)


def _selected_feature_id(editor: FeatureEditor) -> int | None:
    if not editor.sorted_features:
        return None
    return editor.sorted_features[editor.selected_idx].feature_id


def _selected_desc_scroll(editor: FeatureEditor) -> int:
    feature_id = _selected_feature_id(editor)
    if feature_id is None:
        return 0
    return editor._desc_scroll.get(feature_id, 0)


# ---------------------------------------------------------------------------
# 1. Features sorted correctly: by magnitude desc, then id asc
# ---------------------------------------------------------------------------


class TestSorting:
    def test_sort_by_magnitude_desc_then_id_asc(self) -> None:
        features = [
            FeatureCondition(feature_id=10, magnitude=3),
            FeatureCondition(feature_id=5, magnitude=7),
            FeatureCondition(feature_id=20, magnitude=7),
            FeatureCondition(feature_id=1, magnitude=1),
        ]
        editor = _make_editor(features)
        ids = [f.feature_id for f in editor.sorted_features]
        # mag 7 first (id 5 before 20), then mag 3 (id 10), then mag 1 (id 1)
        assert ids == [5, 20, 10, 1]


# ---------------------------------------------------------------------------
# 2. Up/Down moves selection
# ---------------------------------------------------------------------------


class TestNavigation:
    def test_down_increments_selection(self) -> None:
        features = [
            FeatureCondition(feature_id=1, magnitude=5),
            FeatureCondition(feature_id=2, magnitude=3),
            FeatureCondition(feature_id=3, magnitude=1),
        ]
        editor = _make_editor(features)
        assert editor.selected_idx == 0
        editor.move_selection(1)
        assert editor.selected_idx == 1
        editor.move_selection(1)
        assert editor.selected_idx == 2

    def test_up_decrements_selection(self) -> None:
        features = [
            FeatureCondition(feature_id=1, magnitude=5),
            FeatureCondition(feature_id=2, magnitude=3),
        ]
        editor = _make_editor(features)
        editor.move_selection(1)  # now at 1
        editor.move_selection(-1)  # back to 0
        assert editor.selected_idx == 0

    def test_modifier_scroll_moves_by_five(self) -> None:
        editor = _make_editor(
            [FeatureCondition(feature_id=i, magnitude=10 - i) for i in range(8)]
        )
        dispatch_key(editor, "ctrl+down")
        assert editor.selected_idx == 5
        dispatch_key(editor, "meta+up")
        assert editor.selected_idx == 0

    def test_down_clamps_at_end(self) -> None:
        features = [
            FeatureCondition(feature_id=1, magnitude=5),
            FeatureCondition(feature_id=2, magnitude=3),
        ]
        editor = _make_editor(features)
        editor.move_selection(1)
        editor.move_selection(1)  # past end
        assert editor.selected_idx == 1

    def test_up_clamps_at_start(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=5)]
        editor = _make_editor(features)
        editor.move_selection(-1)  # past start
        assert editor.selected_idx == 0


# ---------------------------------------------------------------------------
# 3. Left decrements magnitude, Right increments, with wraparound
# ---------------------------------------------------------------------------


class TestLeftRightMagnitude:
    def test_right_increments(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=3)]
        editor = _make_editor(features, num_buckets=11, has_negatives=True)
        editor.adjust_magnitude(1)
        assert editor.sorted_features[0].magnitude == 4

    def test_left_decrements(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=3)]
        editor = _make_editor(features, num_buckets=11, has_negatives=True)
        editor.adjust_magnitude(-1)
        assert editor.sorted_features[0].magnitude == 2

    def test_right_wraps_max_to_min(self) -> None:
        # has_negatives=True, num_buckets=11 → range is (-1, 9)
        features = [FeatureCondition(feature_id=1, magnitude=9)]
        editor = _make_editor(features, num_buckets=11, has_negatives=True)
        editor.adjust_magnitude(1)
        assert editor.sorted_features[0].magnitude == -1

    def test_left_wraps_min_to_max(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=-1)]
        editor = _make_editor(features, num_buckets=11, has_negatives=True)
        editor.adjust_magnitude(-1)
        assert editor.sorted_features[0].magnitude == 9

    def test_wraparound_no_negatives(self) -> None:
        # has_negatives=False, num_buckets=11 → range is (0, 10)
        features = [FeatureCondition(feature_id=1, magnitude=10)]
        editor = _make_editor(features, num_buckets=11, has_negatives=False)
        editor.adjust_magnitude(1)
        assert editor.sorted_features[0].magnitude == 0

    def test_wraparound_no_negatives_left(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=0)]
        editor = _make_editor(features, num_buckets=11, has_negatives=False)
        editor.adjust_magnitude(-1)
        assert editor.sorted_features[0].magnitude == 10


# ---------------------------------------------------------------------------
# 4. Digit input: press 1 → mag 1, press 0 → mag 10
# ---------------------------------------------------------------------------


class TestDigitAccumulator:
    def test_single_digit(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=5)]
        editor = _make_editor(features, num_buckets=11, has_negatives=False)
        # range (0, 10), press '1'
        editor.handle_digit(1)
        assert editor.sorted_features[0].magnitude == 1

    def test_two_digits(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=5)]
        editor = _make_editor(features, num_buckets=11, has_negatives=False)
        # range (0, 10), press '1' then '0'
        editor.handle_digit(1)
        editor.handle_digit(0)
        assert editor.sorted_features[0].magnitude == 10


# ---------------------------------------------------------------------------
# 5. Digit overflow: if max is 9 and value would be 12, reset to 2
# ---------------------------------------------------------------------------


class TestDigitOverflow:
    def test_overflow_resets_to_digit(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=5)]
        # num_buckets=11, has_negatives=True → max_display=9
        editor = _make_editor(features, num_buckets=11, has_negatives=True)
        editor.handle_digit(1)  # mag = 1
        editor.handle_digit(2)  # 1*10+2=12 > 9, reset to 2
        assert editor.sorted_features[0].magnitude == 2


# ---------------------------------------------------------------------------
# 6. Navigate away and back resets accumulator
# ---------------------------------------------------------------------------


class TestAccumulatorReset:
    def test_navigation_resets_accumulator(self) -> None:
        features = [
            FeatureCondition(feature_id=1, magnitude=0),
            FeatureCondition(feature_id=2, magnitude=0),
        ]
        editor = _make_editor(features, num_buckets=11, has_negatives=False)
        # Press 1 on first feature (id=1)
        editor.handle_digit(1)
        assert editor.sorted_features[editor.selected_idx].magnitude == 1
        # Navigate down then back up — accumulator should reset
        editor.move_selection(1)
        editor.move_selection(-1)
        # Press 0 — should be 0, not 10 (accumulator was reset)
        editor.handle_digit(0)
        assert editor.sorted_features[editor.selected_idx].magnitude == 0


# ---------------------------------------------------------------------------
# 7-8. '-' key behavior
# ---------------------------------------------------------------------------


class TestMinusKey:
    def test_minus_sets_negative_one(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=5)]
        editor = _make_editor(features, num_buckets=11, has_negatives=True)
        editor.handle_minus()
        assert editor.sorted_features[0].magnitude == -1

    def test_minus_noop_without_negatives(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=5)]
        editor = _make_editor(features, num_buckets=11, has_negatives=False)
        editor.handle_minus()
        assert editor.sorted_features[0].magnitude == 5


# ---------------------------------------------------------------------------
# 9. Space toggles feature enabled/disabled
# ---------------------------------------------------------------------------


class TestToggle:
    def test_space_toggles_enabled(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=5, enabled=True)]
        editor = _make_editor(features)
        editor.toggle_selected()
        assert editor.sorted_features[0].enabled is False
        editor.toggle_selected()
        assert editor.sorted_features[0].enabled is True


# ---------------------------------------------------------------------------
# 10-11. Backspace deletes feature
# ---------------------------------------------------------------------------


class TestDelete:
    def test_backspace_deletes_and_moves_to_next(self) -> None:
        features = [
            FeatureCondition(feature_id=1, magnitude=5),
            FeatureCondition(feature_id=2, magnitude=3),
            FeatureCondition(feature_id=3, magnitude=1),
        ]
        editor = _make_editor(features)
        # Delete first feature (id=1, mag=5)
        editor.delete_selected()
        assert len(editor.sorted_features) == 2
        # Selection should now be at index 0, pointing to next feature
        assert editor.selected_idx == 0
        assert editor.sorted_features[0].feature_id == 2

    def test_backspace_last_item_adjusts_selection(self) -> None:
        features = [
            FeatureCondition(feature_id=1, magnitude=5),
            FeatureCondition(feature_id=2, magnitude=3),
        ]
        editor = _make_editor(features)
        editor.move_selection(1)  # select last
        editor.delete_selected()
        assert len(editor.sorted_features) == 1
        assert editor.selected_idx == 0

    def test_backspace_on_only_feature_empties_list(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=5)]
        editor = _make_editor(features)
        editor.delete_selected()
        assert len(editor.sorted_features) == 0
        assert editor.selected_idx == 0  # clamped, no crash


# ---------------------------------------------------------------------------
# 12. Selection history: navigate to 3 different features, Cmd+Z goes back
# ---------------------------------------------------------------------------


class TestSelectionHistory:
    def test_undo_redo(self) -> None:
        features = [
            FeatureCondition(feature_id=1, magnitude=9),
            FeatureCondition(feature_id=2, magnitude=7),
            FeatureCondition(feature_id=3, magnitude=5),
            FeatureCondition(feature_id=4, magnitude=3),
        ]
        editor = _make_editor(features)
        # Navigate: 0 → 1 → 2 → 3
        editor.move_selection(1)
        editor.move_selection(1)
        editor.move_selection(1)
        assert editor.selected_idx == 3

        # Undo back: 3 → 2 → 1 → 0
        editor.undo_selection()
        assert editor.selected_idx == 2
        editor.undo_selection()
        assert editor.selected_idx == 1
        editor.undo_selection()
        assert editor.selected_idx == 0

        # Redo forward: 0 → 1 → 2
        editor.redo_selection()
        assert editor.selected_idx == 1
        editor.redo_selection()
        assert editor.selected_idx == 2

    def test_undo_at_start_is_noop(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=5)]
        editor = _make_editor(features)
        editor.undo_selection()  # no history — should be no-op
        assert editor.selected_idx == 0

    def test_redo_at_end_is_noop(self) -> None:
        features = [
            FeatureCondition(feature_id=1, magnitude=5),
            FeatureCondition(feature_id=2, magnitude=3),
        ]
        editor = _make_editor(features)
        editor.move_selection(1)
        editor.redo_selection()  # no future — should be no-op
        assert editor.selected_idx == 1

    def test_undo_after_delete_clamps_index(self) -> None:
        """Stale indices in history must be clamped after deletions."""
        features = [
            FeatureCondition(feature_id=1, magnitude=9),
            FeatureCondition(feature_id=2, magnitude=7),
            FeatureCondition(feature_id=3, magnitude=5),
        ]
        editor = _make_editor(features)
        # Navigate to idx 2
        editor.move_selection(1)
        editor.move_selection(1)
        assert editor.selected_idx == 2
        # Delete twice — list shrinks to 1 item
        editor.delete_selected()
        editor.delete_selected()
        assert len(editor.sorted_features) == 1
        assert editor.selected_idx == 0
        # Undo should not crash (stale idx=1 must be clamped to 0)
        editor.undo_selection()
        assert editor.selected_idx == 0

    def test_navigate_after_undo_clears_redo(self) -> None:
        features = [
            FeatureCondition(feature_id=1, magnitude=9),
            FeatureCondition(feature_id=2, magnitude=7),
            FeatureCondition(feature_id=3, magnitude=5),
        ]
        editor = _make_editor(features)
        editor.move_selection(1)  # → 1
        editor.move_selection(1)  # → 2
        editor.undo_selection()  # → 1
        # New navigation should clear redo stack
        editor.move_selection(1)  # → 2, redo cleared
        editor.redo_selection()  # should be no-op (redo was cleared)
        assert editor.selected_idx == 2


# ---------------------------------------------------------------------------
# 13-15. Messages: Esc, N, Enter
# ---------------------------------------------------------------------------


class TestMessages:
    def test_esc_posts_closed(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=5)]
        editor = _make_editor(features)
        messages = dispatch_key(editor, "escape")
        assert any(isinstance(m, FeatureEditor.FeatureEditorClosed) for m in messages)

    def test_n_posts_search(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=5)]
        editor = _make_editor(features)
        messages = dispatch_key(editor, "n")
        assert any(isinstance(m, FeatureEditor.OpenFeatureSearch) for m in messages)

    def test_enter_posts_detail(self) -> None:
        features = [FeatureCondition(feature_id=42, magnitude=5)]
        editor = _make_editor(features)
        messages = dispatch_key(editor, "enter")
        assert len(messages) == 1
        msg = messages[0]
        assert isinstance(msg, FeatureEditor.OpenFeatureDetail)
        assert msg.feature_id == 42


# ---------------------------------------------------------------------------
# 16. Cmd+Up jumps to first, Cmd+Down to last
# ---------------------------------------------------------------------------


class TestJumpNavigation:
    def test_cmd_up_jumps_to_first(self) -> None:
        features = [
            FeatureCondition(feature_id=1, magnitude=5),
            FeatureCondition(feature_id=2, magnitude=3),
            FeatureCondition(feature_id=3, magnitude=1),
        ]
        editor = _make_editor(features)
        editor.move_selection(1)
        editor.move_selection(1)
        assert editor.selected_idx == 2
        editor.jump_to_first()
        assert editor.selected_idx == 0

    def test_cmd_down_jumps_to_last(self) -> None:
        features = [
            FeatureCondition(feature_id=1, magnitude=5),
            FeatureCondition(feature_id=2, magnitude=3),
            FeatureCondition(feature_id=3, magnitude=1),
        ]
        editor = _make_editor(features)
        editor.jump_to_last()
        assert editor.selected_idx == 2


# ---------------------------------------------------------------------------
# 17. Re-sort after magnitude change: selection follows the feature
# ---------------------------------------------------------------------------


class TestNoResortOnMagnitudeChange:
    """Features are sorted once on import. Magnitude changes don't re-sort."""

    def test_order_stable_after_magnitude_change(self) -> None:
        features = [
            FeatureCondition(feature_id=10, magnitude=9),
            FeatureCondition(feature_id=20, magnitude=5),
            FeatureCondition(feature_id=30, magnitude=1),
        ]
        editor = _make_editor(features, num_buckets=11, has_negatives=False)
        # Initial sort: [10(9), 20(5), 30(1)]
        assert [f.feature_id for f in editor.sorted_features] == [10, 20, 30]

        # Select feature 30 (index 2) and set magnitude to 10
        editor.move_selection(1)
        editor.move_selection(1)
        editor.handle_digit(1)
        editor.handle_digit(0)

        # Order should NOT change — 30 stays at index 2
        assert [f.feature_id for f in editor.sorted_features] == [10, 20, 30]
        assert editor.selected_idx == 2
        assert editor.sorted_features[2].magnitude == 10

    def test_selection_stays_put_after_magnitude_change(self) -> None:
        features = [
            FeatureCondition(feature_id=1, magnitude=8),
            FeatureCondition(feature_id=2, magnitude=5),
            FeatureCondition(feature_id=3, magnitude=3),
        ]
        editor = _make_editor(features, num_buckets=11, has_negatives=True)
        # Select feature 3 (index 2)
        editor.move_selection(1)
        editor.move_selection(1)
        assert _selected_feature_id(editor) == 3

        # Change magnitude — selection should not move
        editor.adjust_magnitude(1)
        assert editor.selected_idx == 2
        assert _selected_feature_id(editor) == 3
        editor.adjust_magnitude(1)
        assert editor.selected_idx == 2


# ---------------------------------------------------------------------------
# 18. Vertical scrolling
# ---------------------------------------------------------------------------


class TestVerticalScrolling:
    def _features(self, n: int) -> list[FeatureCondition]:
        return [FeatureCondition(feature_id=i, magnitude=n - i) for i in range(n)]

    def test_all_fit_no_scroll(self) -> None:
        editor = _make_editor(self._features(3))
        lines = editor.render_content(height=20)
        # header(1) + blank(1) + 3 features = 5
        assert len(lines) == 5
        assert editor._scroll_offset == 0

    def test_scroll_follows_selection_down(self) -> None:
        editor = _make_editor(self._features(10))
        # height=5 means only 5 lines visible
        for _ in range(9):
            editor.move_selection(1)
        lines = editor.render_content(height=5)
        assert len(lines) == 5
        # Selected feature (idx 9) should be visible
        selected_line = editor._HEADER_LINES + editor.selected_idx
        assert editor._scroll_offset <= selected_line < editor._scroll_offset + 5

    def test_scroll_follows_selection_up(self) -> None:
        editor = _make_editor(self._features(10))
        # Go to bottom then back to top
        for _ in range(9):
            editor.move_selection(1)
        editor.render_content(height=5)
        for _ in range(9):
            editor.move_selection(-1)
        lines = editor.render_content(height=5)
        assert len(lines) == 5
        # Selected feature (idx 0) should be visible
        selected_line = editor._HEADER_LINES + editor.selected_idx
        assert editor._scroll_offset <= selected_line < editor._scroll_offset + 5

    def test_scroll_offset_clamps_after_delete(self) -> None:
        editor = _make_editor(self._features(10))
        # Navigate to end and render to set scroll offset high
        for _ in range(9):
            editor.move_selection(1)
        editor.render_content(height=5)
        # Delete several features to shrink the list
        for _ in range(7):
            editor.delete_selected()
        lines = editor.render_content(height=5)
        # Should not crash, offset should be clamped
        assert editor._scroll_offset >= 0
        assert len(lines) <= 5

    def test_height_zero_shows_all(self) -> None:
        editor = _make_editor(self._features(10))
        lines = editor.render_content(height=0)
        # header(1) + blank(1) + 10 features = 12
        assert len(lines) == 12

    def test_selected_feature_text_visible(self) -> None:
        editor = _make_editor(self._features(10))
        for _ in range(7):
            editor.move_selection(1)
        lines = editor.render_content(height=5)
        # The selected feature (id=7) should appear in the visible lines
        visible_text = "\n".join(line.plain for line in lines)
        assert "\u25b6" in visible_text  # ▶ indicator


# ---------------------------------------------------------------------------
# 19. Magnitude counting for histogram
# ---------------------------------------------------------------------------


class TestCountByMagnitude:
    def test_empty_features(self) -> None:
        editor = _make_editor([], num_buckets=11, has_negatives=False)
        counts = editor._count_by_magnitude()
        # All buckets 0..10 present with 0 enabled count.
        # Source semantics: dict[int, int] (enabled-only count per bucket).
        assert len(counts) == 11
        assert all(v == 0 for v in counts.values())

    def test_single_enabled_feature(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=3, enabled=True)]
        editor = _make_editor(features, num_buckets=11, has_negatives=False)
        counts = editor._count_by_magnitude()
        assert counts[3] == 1
        assert counts[0] == 0

    def test_single_disabled_feature(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=3, enabled=False)]
        editor = _make_editor(features, num_buckets=11, has_negatives=False)
        counts = editor._count_by_magnitude()
        # Disabled features are excluded from the histogram (enabled-only count).
        assert counts[3] == 0

    def test_mixed_enabled_disabled_same_magnitude(self) -> None:
        features = [
            FeatureCondition(feature_id=1, magnitude=5, enabled=True),
            FeatureCondition(feature_id=2, magnitude=5, enabled=False),
            FeatureCondition(feature_id=3, magnitude=5, enabled=True),
        ]
        editor = _make_editor(features, num_buckets=11, has_negatives=False)
        counts = editor._count_by_magnitude()
        assert counts[5] == 2

    def test_with_negatives(self) -> None:
        features = [
            FeatureCondition(feature_id=1, magnitude=-1, enabled=True),
            FeatureCondition(feature_id=2, magnitude=0, enabled=True),
            FeatureCondition(feature_id=3, magnitude=9, enabled=False),
        ]
        editor = _make_editor(features, num_buckets=11, has_negatives=True)
        counts = editor._count_by_magnitude()
        # has_negatives=True, num_buckets=11 → range [-1, 9], 11 buckets
        assert len(counts) == 11
        assert counts[-1] == 1
        assert counts[0] == 1
        assert counts[9] == 0

    def test_all_disabled(self) -> None:
        features = [
            FeatureCondition(feature_id=1, magnitude=2, enabled=False),
            FeatureCondition(feature_id=2, magnitude=2, enabled=False),
        ]
        editor = _make_editor(features, num_buckets=11, has_negatives=False)
        counts = editor._count_by_magnitude()
        assert counts[2] == 0

    def test_all_buckets_present(self) -> None:
        """Every integer in [min_mag, max_mag] has an entry, even if zero."""
        features = [FeatureCondition(feature_id=1, magnitude=5)]
        editor = _make_editor(features, num_buckets=11, has_negatives=True)
        counts = editor._count_by_magnitude()
        # Range is [-1, 9] → keys should be -1, 0, 1, ..., 9
        assert sorted(counts.keys()) == list(range(-1, 10))


# ---------------------------------------------------------------------------
# 20. Histogram line rendering
# ---------------------------------------------------------------------------


class TestBuildHistogramLines:
    def test_no_features_returns_empty(self) -> None:
        editor = _make_editor([], num_buckets=11, has_negatives=False)
        lines = editor._build_histogram_lines()
        assert lines == []

    def test_single_bucket_full_width(self) -> None:
        """One feature -> that bucket gets the full bar width."""
        features = [FeatureCondition(feature_id=1, magnitude=3, enabled=True)]
        editor = _make_editor(features, num_buckets=11, has_negatives=False)
        lines = editor._build_histogram_lines()
        # 11 buckets (0..10), one line per bucket
        assert len(lines) == 11
        # Bucket 3 should have a bar, others should have label only
        line_3 = lines[3]  # index 3 = magnitude 3 (0-indexed from 0)
        assert "\u2588" in line_3.plain
        # Bucket 0 should have label only, no bar
        line_0 = lines[0]
        assert "\u2588" not in line_0.plain

    def test_proportional_scaling(self) -> None:
        """Bars scale proportionally to max count."""
        features = [
            FeatureCondition(feature_id=i, magnitude=0, enabled=True) for i in range(10)
        ] + [
            FeatureCondition(feature_id=100, magnitude=1, enabled=True),
        ]
        editor = _make_editor(features, num_buckets=11, has_negatives=False)
        lines = editor._build_histogram_lines()
        bar_0 = lines[0].plain.count("\u2588")
        bar_1 = lines[1].plain.count("\u2588")
        # mag 0 has 10 features, mag 1 has 1 -> bar_0 should be much longer
        assert bar_0 > bar_1
        assert bar_1 >= 1  # minimum 1 char for non-zero

    def test_negative_bucket_at_top_when_present(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=-1, enabled=True)]
        editor = _make_editor(features, num_buckets=11, has_negatives=True)
        lines = editor._build_histogram_lines()
        # First line should be -1
        assert lines[0].plain.startswith("-1")
        # Second line should be 0
        assert lines[1].plain.strip().startswith("0")

    def test_no_negative_bucket_starts_at_zero(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=0, enabled=True)]
        editor = _make_editor(features, num_buckets=11, has_negatives=False)
        lines = editor._build_histogram_lines()
        assert lines[0].plain.strip().startswith("0")

    def test_negative_bucket_colored_red(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=-1, enabled=True)]
        editor = _make_editor(features, num_buckets=11, has_negatives=True)
        lines = editor._build_histogram_lines()
        # The -1 line's bar spans should include red styling
        neg_line = lines[0]
        styles = [span.style for span in neg_line._spans]
        assert any("red" in str(s) for s in styles)

    def test_all_disabled_no_bars(self) -> None:
        """Source drops disabled features from the histogram entirely.

        When every feature is disabled, the max count is 0 and
        ``_build_histogram_lines`` returns an empty list.
        """
        features = [FeatureCondition(feature_id=1, magnitude=3, enabled=False)]
        editor = _make_editor(features, num_buckets=11, has_negatives=False)
        lines = editor._build_histogram_lines()
        assert lines == []

    def test_mixed_enabled_disabled_only_enabled_counted(self) -> None:
        """Disabled features are excluded from the bar; only enabled contribute."""
        features = [
            FeatureCondition(feature_id=1, magnitude=5, enabled=True),
            FeatureCondition(feature_id=2, magnitude=5, enabled=True),
            FeatureCondition(feature_id=3, magnitude=5, enabled=False),
            FeatureCondition(feature_id=4, magnitude=5, enabled=False),
        ]
        editor = _make_editor(features, num_buckets=11, has_negatives=False)
        lines = editor._build_histogram_lines()
        line_5 = lines[5]
        # Bar reflects only the 2 enabled features; no dim styling for disabled.
        assert line_5.plain.count("\u2588") > 0

    def test_minimum_one_char_bar(self) -> None:
        """Bucket with 1 feature when max_count is large still gets 1-char bar."""
        features = [
            FeatureCondition(feature_id=i, magnitude=0, enabled=True) for i in range(100)
        ] + [
            FeatureCondition(feature_id=200, magnitude=5, enabled=True),
        ]
        editor = _make_editor(features, num_buckets=11, has_negatives=False)
        lines = editor._build_histogram_lines()
        line_5 = lines[5]
        assert line_5.plain.count("\u2588") >= 1

    def test_empty_bucket_label_only(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=5, enabled=True)]
        editor = _make_editor(features, num_buckets=11, has_negatives=False)
        lines = editor._build_histogram_lines()
        # Bucket 0 has no features
        line_0 = lines[0]
        assert "\u2588" not in line_0.plain
        # But label is still there
        assert line_0.plain.strip() == "0"

    def test_negative_disabled_not_rendered(self) -> None:
        """Disabled negative features are excluded from the histogram."""
        features = [FeatureCondition(feature_id=1, magnitude=-1, enabled=False)]
        editor = _make_editor(features, num_buckets=11, has_negatives=True)
        lines = editor._build_histogram_lines()
        # Only disabled features → max_count == 0 → empty histogram.
        assert lines == []

    def test_bar_width_respects_panel_width(self) -> None:
        """No bar should exceed _RIGHT_PANEL_WIDTH - _HISTOGRAM_LABEL_WIDTH."""
        features = [FeatureCondition(feature_id=i, magnitude=0, enabled=True) for i in range(50)]
        editor = _make_editor(features, num_buckets=11, has_negatives=False)
        lines = editor._build_histogram_lines()
        max_bar = _RIGHT_PANEL_WIDTH - _HISTOGRAM_LABEL_WIDTH
        for line in lines:
            bar_chars = line.plain.count("\u2588")
            assert bar_chars <= max_bar


# ---------------------------------------------------------------------------
# 21. Right panel assembly
# ---------------------------------------------------------------------------


class TestBuildRightPanelLines:
    def test_counter_line(self) -> None:
        features = [
            FeatureCondition(feature_id=1, magnitude=5, enabled=True),
            FeatureCondition(feature_id=2, magnitude=3, enabled=False),
            FeatureCondition(feature_id=3, magnitude=1, enabled=True),
        ]
        editor = _make_editor(features)
        lines = editor._build_right_panel_lines()
        assert lines[0].plain == "Active: 2/3"

    def test_counter_zero_features(self) -> None:
        editor = _make_editor([])
        lines = editor._build_right_panel_lines()
        assert lines[0].plain == "Active: 0/0"

    def test_blank_after_counter(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=5)]
        editor = _make_editor(features)
        lines = editor._build_right_panel_lines()
        assert lines[1].plain == ""

    def test_instructions_present(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=5)]
        editor = _make_editor(features)
        lines = editor._build_right_panel_lines()
        # Instructions start at line 2 and span 11 entries (current UX).
        instruction_text = " ".join(line.plain for line in lines[2:13])
        assert "navigate" in instruction_text
        assert "magnitude" in instruction_text
        assert "toggle" in instruction_text
        assert "delete" in instruction_text
        assert "search" in instruction_text
        assert "detail" in instruction_text
        assert "close" in instruction_text
        assert "resample" in instruction_text

    def test_instruction_count(self) -> None:
        """There should be exactly 11 instruction lines (current UX)."""
        features = [FeatureCondition(feature_id=1, magnitude=5)]
        editor = _make_editor(features)
        lines = editor._build_right_panel_lines()
        # Line 0: counter, Line 1: blank, Lines 2-12: 11 instructions
        for i in range(2, 13):
            assert lines[i].plain.strip() != ""

    def test_blank_before_histogram(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=5)]
        editor = _make_editor(features)
        lines = editor._build_right_panel_lines()
        # Line 13 should be blank (separator before histogram)
        assert lines[13].plain == ""

    def test_histogram_follows_instructions(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=5)]
        editor = _make_editor(features, num_buckets=11, has_negatives=False)
        lines = editor._build_right_panel_lines()
        # Histogram starts at line 14 (counter + blank + 11 instructions + blank).
        # First histogram line should be bucket 0.
        assert lines[14].plain.strip().startswith("0")

    def test_counter_updates_after_toggle(self) -> None:
        features = [
            FeatureCondition(feature_id=1, magnitude=5, enabled=True),
            FeatureCondition(feature_id=2, magnitude=3, enabled=True),
        ]
        editor = _make_editor(features)
        assert editor._build_right_panel_lines()[0].plain == "Active: 2/2"
        editor.toggle_selected()
        assert editor._build_right_panel_lines()[0].plain == "Active: 1/2"

    def test_counter_updates_after_delete(self) -> None:
        features = [
            FeatureCondition(feature_id=1, magnitude=5, enabled=True),
            FeatureCondition(feature_id=2, magnitude=3, enabled=True),
        ]
        editor = _make_editor(features)
        editor.delete_selected()
        assert editor._build_right_panel_lines()[0].plain == "Active: 1/1"


# ---------------------------------------------------------------------------
# 22. Side-by-side rendering (render_content with width)
# ---------------------------------------------------------------------------


class TestSideBySideRendering:
    def test_width_zero_returns_left_only(self) -> None:
        """Backward compatible: width=0 returns left-column only."""
        features = [FeatureCondition(feature_id=1, magnitude=5)]
        editor = _make_editor(features)
        lines = editor.render_content(height=10, width=0)
        # Should work exactly like before — no right panel
        assert any("Feature Conditioning" in line.plain for line in lines)
        # No "Active:" counter in left-only mode
        combined = " ".join(line.plain for line in lines)
        assert "Active:" not in combined

    def test_normal_width_has_right_panel(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=5)]
        editor = _make_editor(features)
        lines = editor.render_content(height=15, width=60)
        # First line should contain both header and counter
        combined_first = lines[0].plain
        assert "Feature Conditioning" in combined_first
        assert "Active:" in combined_first

    def test_narrow_width_falls_back_to_left_only(self) -> None:
        """When width is too narrow for both columns, show left only."""
        features = [FeatureCondition(feature_id=1, magnitude=5)]
        editor = _make_editor(features)
        lines = editor.render_content(height=10, width=25)
        # 25 < _RIGHT_PANEL_WIDTH + 10 = 30, so left-only
        combined = " ".join(line.plain for line in lines)
        assert "Active:" not in combined

    def test_right_panel_padded_to_width(self) -> None:
        """Each output line should be exactly `width` characters."""
        features = [FeatureCondition(feature_id=1, magnitude=5)]
        editor = _make_editor(features)
        lines = editor.render_content(height=15, width=60)
        for line in lines:
            assert len(line.plain) == 60, f"Line width {len(line.plain)} != 60: {line.plain!r}"

    def test_scrolling_applies_only_to_left(self) -> None:
        """Left column scrolls, right panel always starts from top."""
        features = [FeatureCondition(feature_id=i, magnitude=10 - i) for i in range(20)]
        editor = _make_editor(features, num_buckets=11, has_negatives=False)
        # Navigate to bottom
        for _ in range(19):
            editor.move_selection(1)
        lines = editor.render_content(height=10, width=60)
        # Right panel first line should still be the counter
        # (extract right-panel portion: last 20 chars of first line)
        right_first = lines[0].plain[-20:]
        assert "Active:" in right_first

    def test_height_respected(self) -> None:
        features = [FeatureCondition(feature_id=i, magnitude=5) for i in range(20)]
        editor = _make_editor(features)
        lines = editor.render_content(height=8, width=60)
        assert len(lines) == 8

    def test_default_width_backward_compatible(self) -> None:
        """Calling render_content(height) with no width still works."""
        features = [FeatureCondition(feature_id=1, magnitude=5)]
        editor = _make_editor(features)
        # This is the existing call signature — must not break
        lines = editor.render_content(height=10)
        assert len(lines) > 0

    def test_left_column_truncated_when_too_long(self) -> None:
        """Long feature descriptions are truncated to left column width."""
        features = [
            FeatureCondition(
                feature_id=1,
                magnitude=5,
                description="A" * 200,  # very long description
            )
        ]
        editor = _make_editor(features)
        lines = editor.render_content(height=10, width=60)
        for line in lines:
            assert len(line.plain) == 60


# ---------------------------------------------------------------------------
# 23. Integration: state mutations reflected in combined output
# ---------------------------------------------------------------------------


class TestIntegration:
    def _render_combined(
        self, editor: FeatureEditor, height: int = 25, width: int = 60
    ) -> list[Text]:
        return editor.render_content(height, width)

    def _right_text(self, lines: list[Text], width: int = 60) -> str:
        """Extract the right-panel text from combined lines."""
        rp_width = _RIGHT_PANEL_WIDTH
        return "\n".join(line.plain[width - rp_width :] for line in lines)

    def test_toggle_updates_counter_and_histogram(self) -> None:
        features = [
            FeatureCondition(feature_id=1, magnitude=5, enabled=True),
            FeatureCondition(feature_id=2, magnitude=5, enabled=True),
            FeatureCondition(feature_id=3, magnitude=3, enabled=True),
        ]
        editor = _make_editor(features, num_buckets=11, has_negatives=False)
        lines = self._render_combined(editor)
        right = self._right_text(lines)
        assert "Active: 3/3" in right

        # Toggle first feature off
        editor.toggle_selected()
        lines = self._render_combined(editor)
        right = self._right_text(lines)
        assert "Active: 2/3" in right

    def test_delete_updates_counter_and_histogram(self) -> None:
        features = [
            FeatureCondition(feature_id=1, magnitude=5, enabled=True),
            FeatureCondition(feature_id=2, magnitude=3, enabled=True),
        ]
        editor = _make_editor(features, num_buckets=11, has_negatives=False)
        lines = self._render_combined(editor)
        right = self._right_text(lines)
        assert "Active: 2/2" in right

        editor.delete_selected()
        lines = self._render_combined(editor)
        right = self._right_text(lines)
        assert "Active: 1/1" in right

    def test_magnitude_change_updates_histogram(self) -> None:
        features = [
            FeatureCondition(feature_id=1, magnitude=5, enabled=True),
            FeatureCondition(feature_id=2, magnitude=5, enabled=True),
        ]
        editor = _make_editor(features, num_buckets=11, has_negatives=False)

        # Both at magnitude 5
        lines_before = self._render_combined(editor)

        # Move first to magnitude 6
        editor.adjust_magnitude(1)
        lines_after = self._render_combined(editor)

        # The histogram should have changed (different distribution)
        right_before = self._right_text(lines_before)
        right_after = self._right_text(lines_after)
        assert right_before != right_after

    def test_digit_entry_updates_histogram(self) -> None:
        features = [
            FeatureCondition(feature_id=1, magnitude=0, enabled=True),
        ]
        editor = _make_editor(features, num_buckets=11, has_negatives=False)
        lines_before = self._render_combined(editor)

        editor.handle_digit(8)
        lines_after = self._render_combined(editor)

        right_before = self._right_text(lines_before)
        right_after = self._right_text(lines_after)
        assert right_before != right_after

    def test_left_column_still_scrolls(self) -> None:
        """Left column scrolling works in side-by-side mode."""
        features = [FeatureCondition(feature_id=i, magnitude=20 - i) for i in range(20)]
        editor = _make_editor(features, num_buckets=11, has_negatives=False)
        # Navigate to bottom
        for _ in range(19):
            editor.move_selection(1)
        lines = self._render_combined(editor, height=10)
        # Selected feature should be visible in left column
        left_text = "\n".join(line.plain[:40] for line in lines)
        assert "\u25b6" in left_text  # triangle indicator

    def test_empty_feature_list(self) -> None:
        editor = _make_editor([], num_buckets=11, has_negatives=False)
        lines = self._render_combined(editor)
        right = self._right_text(lines)
        assert "Active: 0/0" in right
        # No histogram bars
        assert "\u2588" not in right


# ---------------------------------------------------------------------------
# Selection indicator visibility based on focus
# ---------------------------------------------------------------------------


class TestSelectionIndicatorFocus:
    """Triangle indicator should only show when the editor is focused."""

    def test_indicator_visible_when_focused(self) -> None:
        features = [
            FeatureCondition(feature_id=10, magnitude=5),
            FeatureCondition(feature_id=20, magnitude=3),
        ]
        editor = _make_editor(features)
        lines = editor._build_feature_lines(focused=True)
        text = "\n".join(line.plain for line in lines)
        assert "\u25b6" in text

    def test_indicator_hidden_when_unfocused(self) -> None:
        features = [
            FeatureCondition(feature_id=10, magnitude=5),
            FeatureCondition(feature_id=20, magnitude=3),
        ]
        editor = _make_editor(features)
        lines = editor._build_feature_lines(focused=False)
        text = "\n".join(line.plain for line in lines)
        assert "\u25b6" not in text

    def test_render_content_respects_focus(self) -> None:
        features = [
            FeatureCondition(feature_id=10, magnitude=5),
        ]
        editor = _make_editor(features)
        focused_lines = editor.render_content(10, focused=True)
        unfocused_lines = editor.render_content(10, focused=False)
        focused_text = "\n".join(line.plain for line in focused_lines)
        unfocused_text = "\n".join(line.plain for line in unfocused_lines)
        assert "\u25b6" in focused_text
        assert "\u25b6" not in unfocused_text

    def test_selected_idx_preserved_when_unfocused(self) -> None:
        """Selection position is remembered even when triangle is hidden."""
        features = [
            FeatureCondition(feature_id=10, magnitude=5),
            FeatureCondition(feature_id=20, magnitude=3),
        ]
        editor = _make_editor(features)
        editor._selected_idx = 1
        # Unfocused: no triangle, but idx stays
        editor._build_feature_lines(focused=False)
        assert editor._selected_idx == 1
        # Refocused: triangle returns at same position
        lines = editor._build_feature_lines(focused=True)
        # Second line (idx 1) should have the triangle
        assert "\u25b6" in lines[1].plain
        assert "\u25b6" not in lines[0].plain


# ---------------------------------------------------------------------------
# Alt+Left/Right posts SwitchSegmentRequested
# ---------------------------------------------------------------------------


class TestSwitchSegmentMessage:
    """Alt+Left/Right should post SwitchSegmentRequested with correct delta."""

    def test_alt_left_posts_switch_minus_one(self) -> None:
        features = [FeatureCondition(feature_id=10, magnitude=5)]
        editor = _make_editor(features)
        msgs = dispatch_key(editor, "alt+left")
        assert len(msgs) == 1
        assert isinstance(msgs[0], FeatureEditor.SwitchSegmentRequested)
        assert msgs[0].delta == -1

    def test_alt_right_posts_switch_plus_one(self) -> None:
        features = [FeatureCondition(feature_id=10, magnitude=5)]
        editor = _make_editor(features)
        msgs = dispatch_key(editor, "alt+right")
        assert len(msgs) == 1
        assert isinstance(msgs[0], FeatureEditor.SwitchSegmentRequested)
        assert msgs[0].delta == 1

    # ctrl+left/right now scroll descriptions, not switch segments


# ---------------------------------------------------------------------------
# visible_feature_ids
# ---------------------------------------------------------------------------


class TestVisibleFeatureIds:
    @staticmethod
    def _features(n: int) -> list[FeatureCondition]:
        return [FeatureCondition(feature_id=i, magnitude=0) for i in range(n)]

    def test_returns_visible_ids_at_top(self) -> None:
        editor = _make_editor(self._features(10))
        editor.render_content(height=5)  # sets scroll state
        ids = editor.visible_feature_ids(3)  # 3 feature rows
        assert ids == [0, 1, 2]

    def test_returns_visible_ids_after_scroll(self) -> None:
        editor = _make_editor(self._features(10))
        for _ in range(7):
            editor.move_selection(1)
        editor.render_content(height=5)  # scroll follows selection
        ids = editor.visible_feature_ids(3)
        assert len(ids) == 3
        # Selected idx is 7; these should include 7
        assert 7 in ids

    def test_returns_all_when_list_is_short(self) -> None:
        editor = _make_editor(self._features(2))
        editor.render_content(height=10)
        ids = editor.visible_feature_ids(10)
        assert ids == [0, 1]

    def test_empty_features(self) -> None:
        editor = _make_editor([])
        assert editor.visible_feature_ids(5) == []

    def test_zero_height(self) -> None:
        editor = _make_editor(self._features(3))
        assert editor.visible_feature_ids(0) == []


# ---------------------------------------------------------------------------
# update_description
# ---------------------------------------------------------------------------


class TestUpdateDescription:
    def test_updates_matching_feature(self) -> None:
        features = [
            FeatureCondition(feature_id=10, magnitude=5),
            FeatureCondition(feature_id=20, magnitude=3),
        ]
        editor = _make_editor(features)
        editor.update_description(20, "fires on greetings")
        fc = next(f for f in editor.sorted_features if f.feature_id == 20)
        assert fc.description == "fires on greetings"

    def test_noop_for_unknown_feature(self) -> None:
        features = [FeatureCondition(feature_id=10, magnitude=5)]
        editor = _make_editor(features)
        editor.update_description(999, "should not crash")
        assert editor.sorted_features[0].description is None

    def test_description_appears_in_render(self) -> None:
        features = [FeatureCondition(feature_id=42, magnitude=7)]
        editor = _make_editor(features)
        editor.update_description(42, "test desc")
        lines = editor.render_content(height=5)
        visible_text = "\n".join(line.plain for line in lines)
        assert "test desc" in visible_text

    def test_loading_placeholder_renders_dimmed(self) -> None:
        features = [FeatureCondition(feature_id=42, magnitude=7)]
        editor = _make_editor(features)
        editor.update_description(42, DESCRIPTION_LOADING)
        lines = editor.render_content(height=5)
        # "..." should appear in the rendered text
        feature_line = lines[2]  # header + blank + first feature
        assert DESCRIPTION_LOADING in feature_line.plain
        # The "..." portion should have dim italic styling
        desc_spans = [
            s
            for s in feature_line._spans
            if feature_line.plain[s.start : s.end] == DESCRIPTION_LOADING
        ]
        assert len(desc_spans) == 1
        assert "dim" in str(desc_spans[0].style)
        assert "italic" in str(desc_spans[0].style)

    def test_no_descriptions_warning_when_unavailable(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=5)]
        seg = _make_segment(features)
        editor = FeatureEditor(
            seg, num_buckets=11, has_negatives=True, descriptions_available=False
        )
        lines = editor.render_content(height=10)
        text = "\n".join(line.plain for line in lines)
        assert "neuronpedia_model_id" in text
        assert "neuronpedia_sae_source" in text

    def test_no_warning_when_descriptions_available(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=5)]
        editor = _make_editor(features)  # descriptions_available defaults True
        lines = editor.render_content(height=10)
        text = "\n".join(line.plain for line in lines)
        assert "neuronpedia_model_id" not in text

    def test_loading_replaced_by_real_description(self) -> None:
        features = [FeatureCondition(feature_id=42, magnitude=7)]
        editor = _make_editor(features)
        editor.update_description(42, DESCRIPTION_LOADING)
        editor.update_description(42, "fires on greetings")
        fc = editor.sorted_features[0]
        assert fc.description == "fires on greetings"
        lines = editor.render_content(height=5)
        visible_text = "\n".join(line.plain for line in lines)
        assert "fires on greetings" in visible_text
        assert DESCRIPTION_LOADING not in visible_text


# ---------------------------------------------------------------------------
# Description horizontal scrolling
# ---------------------------------------------------------------------------


class TestDescriptionScroll:
    def test_scroll_right_shifts_selected_description(self) -> None:
        long_desc = "a" * 40  # longer than any reasonable column width
        features = [FeatureCondition(feature_id=1, magnitude=5, description=long_desc)]
        editor = _make_editor(features)
        # Render first to set _last_desc_width (side-by-side needs width > 32)
        editor.render_content(height=5, width=60)
        editor.scroll_description(1)
        assert _selected_desc_scroll(editor) == _DESC_SCROLL_TIERS[-1][1]
        lines = editor.render_content(height=5, width=60)
        text = "\n".join(line.plain for line in lines)
        assert text.startswith("Feature")  # header still there

    def test_scroll_left_clamps_at_zero(self) -> None:
        editor = _make_editor([FeatureCondition(feature_id=1, magnitude=5, description="abc")])
        editor.scroll_description(-1)
        assert _selected_desc_scroll(editor) == 0

    def test_scroll_only_affects_selected_feature(self) -> None:
        features = [
            FeatureCondition(feature_id=1, magnitude=5, description="a" * 40),
            FeatureCondition(feature_id=2, magnitude=3, description="xxxx_yyyy"),
        ]
        editor = _make_editor(features)
        editor.render_content(height=5, width=60)
        # Scroll feature 1 (selected)
        for _ in range(4):
            editor.scroll_description(1)
        lines = editor.render_content(height=5, width=60)
        text = "\n".join(line.plain for line in lines)
        # Feature 2 unscrolled: "xxxx" still there
        assert "xxxx" in text

    def test_scroll_persists_after_selection_change(self) -> None:
        features = [
            FeatureCondition(feature_id=1, magnitude=5, description="a" * 40),
            FeatureCondition(feature_id=2, magnitude=3, description="other"),
        ]
        editor = _make_editor(features)
        editor.render_content(height=5, width=60)
        editor.scroll_description(1)
        assert _selected_desc_scroll(editor) == _DESC_SCROLL_TIERS[-1][1]
        editor.move_selection(1)
        assert _selected_desc_scroll(editor) == 0  # feature 2 has no scroll
        editor.move_selection(-1)
        assert _selected_desc_scroll(editor) == _DESC_SCROLL_TIERS[-1][1]

    def test_cannot_scroll_past_end(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=5, description="short")]
        editor = _make_editor(features)
        # Width 60, prefix ~14 chars → desc area ~24 chars → "short" (5) fits
        editor.render_content(height=5, width=60)
        editor.scroll_description(1)
        assert _selected_desc_scroll(editor) == 0


# ---------------------------------------------------------------------------
# Column gap between left and right panels
# ---------------------------------------------------------------------------


class TestColumnGap:
    def test_gap_between_columns(self) -> None:
        features = [FeatureCondition(feature_id=1, magnitude=5)]
        editor = _make_editor(features)
        lines = editor.render_content(height=5, width=80)
        # Each line should have at least _COLUMN_GAP spaces between content
        for line in lines:
            plain = line.plain
            # Find the gap: left content, then spaces, then right content
            left_end = 80 - _RIGHT_PANEL_WIDTH - _COLUMN_GAP
            gap_region = plain[left_end : left_end + _COLUMN_GAP]
            assert gap_region == " " * _COLUMN_GAP


# ---------------------------------------------------------------------------
# Composite feature-cluster rows
# ---------------------------------------------------------------------------


class TestCompositeRows:
    @staticmethod
    def _view() -> ClusterView:
        members = (12345, 12346, 12347, 12348)
        return ClusterView(
            rep_to_members={12345: members},
            member_to_rep=dict.fromkeys(members, 12345),
        )

    @staticmethod
    def _editor(
        features: list[FeatureCondition],
        *,
        with_view: bool = True,
    ) -> FeatureEditor:
        return FeatureEditor(
            _make_segment(features),
            num_buckets=11,
            has_negatives=True,
            cluster_view=TestCompositeRows._view() if with_view else None,
        )

    def test_composite_prefix_exact_string(self) -> None:
        editor = self._editor(
            [
                FeatureCondition(
                    feature_id=12345,
                    magnitude=7,
                    description="mentions of chemical elements",
                )
            ]
        )
        assert editor._build_feature_lines()[0].plain == (
            "▶ 12345  [ 7] [4 composite features] mentions of chemical elements"
        )

    def test_prefix_not_in_condition_description(self) -> None:
        editor = self._editor(
            [
                FeatureCondition(
                    feature_id=12345,
                    magnitude=7,
                    description="mentions of chemical elements",
                )
            ]
        )
        editor._build_feature_lines()
        assert editor.sorted_features[0].description == "mentions of chemical elements"

    def test_composite_loading_description_renders_italic(self) -> None:
        editor = self._editor(
            [
                FeatureCondition(
                    feature_id=12345,
                    magnitude=7,
                    description=DESCRIPTION_LOADING,
                )
            ]
        )
        line = editor._build_feature_lines()[0]
        assert "[4 composite features] ..." in line.plain
        loading_spans = [
            span for span in line._spans if DESCRIPTION_LOADING in line.plain[span.start : span.end]
        ]
        assert len(loading_spans) == 1
        assert "italic" in str(loading_spans[0].style)

    def test_desc_scroll_over_combined_string(self) -> None:
        editor = self._editor(
            [
                FeatureCondition(
                    feature_id=12345,
                    magnitude=7,
                    description="mentions of chemical elements",
                )
            ]
        )
        editor._last_desc_width = 24
        editor.scroll_description(1)
        assert _selected_desc_scroll(editor) == 1
        assert editor._build_feature_lines()[0].plain == (
            "▶ 12345  [ 7] 4 composite features] mentions of chemical elements"
        )

    def test_enter_composite_posts_cluster_message(self) -> None:
        editor = self._editor([FeatureCondition(feature_id=12345, magnitude=7)])
        messages = dispatch_key(editor, "enter")
        assert len(messages) == 1
        assert isinstance(messages[0], FeatureEditor.OpenClusterMembers)
        assert messages[0].rep_id == 12345

    def test_enter_singleton_posts_detail(self) -> None:
        editor = self._editor([FeatureCondition(feature_id=99, magnitude=7)])
        messages = dispatch_key(editor, "enter")
        assert len(messages) == 1
        assert isinstance(messages[0], FeatureEditor.OpenFeatureDetail)
        assert messages[0].feature_id == 99

    def test_no_cluster_view_never_prefixes(self) -> None:
        editor = self._editor(
            [
                FeatureCondition(
                    feature_id=12345,
                    magnitude=7,
                    description="mentions of chemical elements",
                )
            ],
            with_view=False,
        )
        assert "composite features" not in editor._build_feature_lines()[0].plain

    def test_sort_and_histogram_unchanged_with_view(self) -> None:
        features = [
            FeatureCondition(feature_id=99, magnitude=2, enabled=True),
            FeatureCondition(feature_id=12345, magnitude=7, enabled=True),
            FeatureCondition(feature_id=50, magnitude=7, enabled=False),
        ]
        clustered = self._editor(features)
        legacy = self._editor(features, with_view=False)
        assert [f.feature_id for f in clustered.sorted_features] == [
            f.feature_id for f in legacy.sorted_features
        ]
        assert clustered._count_by_magnitude() == legacy._count_by_magnitude()
