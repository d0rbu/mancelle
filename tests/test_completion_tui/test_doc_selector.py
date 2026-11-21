"""Tests for the DocSelector widget.

Covers digit-prefix filtering, timeout/reset, Left/Right navigation,
Up/Down segment count, Enter confirmation, and edge cases.
"""

from __future__ import annotations

from scripts.completion_tui.widgets.doc_selector import DocSelector
from tests.test_completion_tui._key_dispatch import dispatch_key

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_DOC_IDS = [0, 1, 2, 3, 4, 5, 10, 40, 41, 42, 43, 100, 420]


def _make_selector(
    doc_ids: list[int] | None = None,
    total_segments: dict[int, int] | None = None,
) -> DocSelector:
    if doc_ids is None:
        doc_ids = list(_DEFAULT_DOC_IDS)
    if total_segments is None:
        total_segments = dict.fromkeys(doc_ids, 5)
    return DocSelector(available_doc_ids=doc_ids, total_segments_map=total_segments)


# ---------------------------------------------------------------------------
# 1. Typing "4" filters to docs starting with 4
# ---------------------------------------------------------------------------


class TestSingleDigitFilter:
    def test_filter_by_single_digit(self) -> None:
        sel = _make_selector()
        sel.handle_digit(4)
        assert sel.filtered_docs == [4, 40, 41, 42, 43, 420]

    def test_filter_updates_filter_text(self) -> None:
        sel = _make_selector()
        sel.handle_digit(4)
        assert sel._filter_text == "4"


# ---------------------------------------------------------------------------
# 2. Typing "42" filters to docs starting with 42
# ---------------------------------------------------------------------------


class TestMultiDigitFilter:
    def test_filter_by_two_digits(self) -> None:
        sel = _make_selector()
        sel.handle_digit(4)
        sel.handle_digit(2)
        assert sel.filtered_docs == [42, 420]

    def test_filter_text_accumulates(self) -> None:
        sel = _make_selector()
        sel.handle_digit(4)
        sel.handle_digit(2)
        assert sel._filter_text == "42"


# ---------------------------------------------------------------------------
# 3. 2s timeout resets filter
# ---------------------------------------------------------------------------


class TestFilterTimeout:
    def test_timeout_resets_filter_text(self) -> None:
        sel = _make_selector()
        sel.handle_digit(4)
        assert sel._filter_text == "4"
        sel.apply_timeout()
        assert sel._filter_text == ""

    def test_timeout_keeps_displayed_results(self) -> None:
        sel = _make_selector()
        sel.handle_digit(4)
        filtered_before = list(sel.filtered_docs)
        sel.apply_timeout()
        assert sel.filtered_docs == filtered_before

    def test_digit_after_timeout_starts_fresh_query(self) -> None:
        sel = _make_selector()
        sel.handle_digit(4)
        sel.apply_timeout()
        sel.handle_digit(1)
        # Fresh query from all docs: "1" → [1, 10, 100]
        assert sel.filtered_docs == [1, 10, 100]


# ---------------------------------------------------------------------------
# 4. Left/Right navigates filtered list
# ---------------------------------------------------------------------------


class TestNavigation:
    def test_right_moves_to_next(self) -> None:
        sel = _make_selector()
        sel.handle_digit(4)
        # Filtered: [4, 40, 41, 42, 43, 420]
        assert sel.selected_doc_idx == 4
        sel.navigate(1)
        assert sel.selected_doc_idx == 40

    def test_left_moves_to_previous(self) -> None:
        sel = _make_selector()
        sel.handle_digit(4)
        sel.navigate(1)  # → 40
        sel.navigate(-1)  # → 4
        assert sel.selected_doc_idx == 4

    def test_right_wraps_to_start(self) -> None:
        sel = _make_selector(doc_ids=[4, 42], total_segments={4: 3, 42: 5})
        sel.handle_digit(4)
        # Filtered: [4, 42]
        sel.navigate(1)  # → 42
        sel.navigate(1)  # wraps → 4
        assert sel.selected_doc_idx == 4

    def test_left_wraps_to_end(self) -> None:
        sel = _make_selector(doc_ids=[4, 42], total_segments={4: 3, 42: 5})
        sel.handle_digit(4)
        # Filtered: [4, 42], selected at 0 (doc 4)
        sel.navigate(-1)  # wraps → 42
        assert sel.selected_doc_idx == 42


# ---------------------------------------------------------------------------
# 5. Up/Down changes segment count
# ---------------------------------------------------------------------------


class TestSegmentCount:
    def test_default_is_two(self) -> None:
        sel = _make_selector(doc_ids=[4], total_segments={4: 5})
        assert sel.num_segments_for_selected == 2

    def test_default_is_one_when_total_is_one(self) -> None:
        sel = _make_selector(doc_ids=[4], total_segments={4: 1})
        assert sel.num_segments_for_selected == 1

    def test_up_decreases(self) -> None:
        sel = _make_selector(doc_ids=[4], total_segments={4: 5})
        sel.adjust_segments(-1)  # 2 → 1
        assert sel.num_segments_for_selected == 1

    def test_down_increases(self) -> None:
        sel = _make_selector(doc_ids=[4], total_segments={4: 5})
        sel.adjust_segments(1)  # 2 → 3
        assert sel.num_segments_for_selected == 3

    def test_clamps_at_one(self) -> None:
        sel = _make_selector(doc_ids=[4], total_segments={4: 2})
        sel.adjust_segments(-1)  # 2 → 1
        sel.adjust_segments(-1)  # clamp
        assert sel.num_segments_for_selected == 1

    def test_clamps_at_total(self) -> None:
        sel = _make_selector(doc_ids=[4], total_segments={4: 3})
        sel.adjust_segments(1)  # 2 → 3
        sel.adjust_segments(1)  # clamp at 3
        assert sel.num_segments_for_selected == 3

    def test_sticky_per_doc_memory(self) -> None:
        sel = _make_selector(doc_ids=[4, 42], total_segments={4: 9, 42: 5})
        sel.handle_digit(4)
        # Filtered: [4, 42], selected = 4, default = 2
        sel.adjust_segments(1)  # 2 → 3
        assert sel.num_segments_for_selected == 3
        sel.navigate(1)  # → 42, default = 2
        assert sel.num_segments_for_selected == 2
        sel.navigate(-1)  # → 4, remembered = 3
        assert sel.num_segments_for_selected == 3

    def test_reset_clears_overrides(self) -> None:
        sel = _make_selector(doc_ids=[4], total_segments={4: 5})
        sel.adjust_segments(1)  # 2 → 3
        assert sel.num_segments_for_selected == 3
        sel.reset_segment_overrides()
        assert sel.num_segments_for_selected == 2


# ---------------------------------------------------------------------------
# 6. Enter posts DocumentSelected with correct doc_idx and num_segments
# ---------------------------------------------------------------------------


class TestEnterConfirm:
    def test_enter_posts_document_selected(self) -> None:
        sel = _make_selector(doc_ids=[4, 42], total_segments={4: 3, 42: 7})
        sel.handle_digit(4)
        sel.navigate(1)  # → 42, default = 2
        sel.adjust_segments(1)  # 2 → 3
        msgs = dispatch_key(sel, "enter")
        selected_messages = [
            message for message in msgs if isinstance(message, DocSelector.DocumentSelected)
        ]
        assert len(selected_messages) == 1
        msg = selected_messages[0]
        assert isinstance(msg, DocSelector.DocumentSelected)
        assert msg.doc_idx == 42
        assert msg.num_segments == 3

    def test_enter_with_default_segments(self) -> None:
        sel = _make_selector(doc_ids=[10], total_segments={10: 8})
        msgs = dispatch_key(sel, "enter")
        selected_messages = [
            message for message in msgs if isinstance(message, DocSelector.DocumentSelected)
        ]
        assert len(selected_messages) == 1
        msg = selected_messages[0]
        assert isinstance(msg, DocSelector.DocumentSelected)
        assert msg.doc_idx == 10
        assert msg.num_segments == 2


# ---------------------------------------------------------------------------
# 7. Non-digit after filter resets query but keeps results
# ---------------------------------------------------------------------------


class TestNonDigitReset:
    def test_resets_filter_text(self) -> None:
        sel = _make_selector()
        sel.handle_digit(4)
        sel.reset_filter()
        assert sel._filter_text == ""

    def test_keeps_filtered_results(self) -> None:
        sel = _make_selector()
        sel.handle_digit(4)
        filtered_before = list(sel.filtered_docs)
        sel.reset_filter()
        assert sel.filtered_docs == filtered_before

    def test_digit_after_reset_starts_fresh(self) -> None:
        sel = _make_selector()
        sel.handle_digit(4)
        sel.reset_filter()
        sel.handle_digit(1)
        assert sel.filtered_docs == [1, 10, 100]


# ---------------------------------------------------------------------------
# 8. Edge: no matching docs for filter query
# ---------------------------------------------------------------------------


class TestNoMatchingDocs:
    def test_empty_filter_result(self) -> None:
        sel = _make_selector(doc_ids=[0, 1, 2])
        sel.handle_digit(9)
        assert sel.filtered_docs == []
        assert sel.selected_doc_idx is None

    def test_enter_with_no_matches_returns_empty(self) -> None:
        sel = _make_selector(doc_ids=[0, 1, 2])
        sel.handle_digit(9)
        msgs = dispatch_key(sel, "enter")
        assert not any(isinstance(message, DocSelector.DocumentSelected) for message in msgs)

    def test_navigate_with_no_matches_is_noop(self) -> None:
        sel = _make_selector(doc_ids=[0, 1, 2])
        sel.handle_digit(9)
        sel.navigate(1)  # should not crash
        assert sel.selected_doc_idx is None


# ---------------------------------------------------------------------------
# 9. Visible window — all docs fit (no truncation)
# ---------------------------------------------------------------------------


class TestVisibleWindowAllFit:
    """When width is large enough, _visible_window returns the full range."""

    def test_all_fit(self) -> None:
        sel = _make_selector(doc_ids=[0, 1, 2])
        assert sel._visible_window(200) == (0, 3)

    def test_exact_fit(self) -> None:
        # doc_ids=[0,1,2]: max_digits=1, slot_w=3, sep=0, prefix=4
        # Total: 4 + 3*3 = 13
        sel = _make_selector(doc_ids=[0, 1, 2])
        assert sel._visible_window(13) == (0, 3)

    def test_one_char_short_triggers_truncation(self) -> None:
        sel = _make_selector(doc_ids=[0, 1, 2])
        start, end = sel._visible_window(12)
        assert end - start < 3


# ---------------------------------------------------------------------------
# 10. Visible window — truncation
# ---------------------------------------------------------------------------


class TestVisibleWindowTruncation:
    """When width is too small, the window is centred on the selected doc."""

    def test_selected_at_start(self) -> None:
        sel = _make_selector(doc_ids=[0, 1, 2, 3, 4])
        # sep=0, slot_w=3, prefix=5: full=20, width=18 forces truncation
        start, end = sel._visible_window(18)
        assert start == 0
        assert end < 5

    def test_selected_at_end(self) -> None:
        sel = _make_selector(doc_ids=[0, 1, 2, 3, 4])
        for _ in range(4):
            sel.navigate(1)
        assert sel.selected_doc_idx == 4
        start, end = sel._visible_window(18)
        assert end == 5
        assert start > 0

    def test_selected_in_middle(self) -> None:
        sel = _make_selector(doc_ids=[0, 1, 2, 3, 4])
        sel.navigate(1)
        sel.navigate(1)  # doc 2
        start, end = sel._visible_window(18)
        assert start > 0
        assert end < 5
        assert start <= 2 < end  # selected must be visible

    def test_empty_filtered_list(self) -> None:
        sel = _make_selector(doc_ids=[0])
        sel.handle_digit(9)
        assert sel._visible_window(80) == (0, 0)

    def test_selected_always_in_window(self) -> None:
        """Critical invariant: selected_idx is always inside the window."""
        sel = _make_selector()  # 13 docs
        for width in [20, 30, 40, 60, 80]:
            for i in range(len(sel._filtered_docs)):
                sel._selected_idx = i
                start, end = sel._visible_window(width)
                assert start <= i < end, (
                    f"selected_idx={i} not in [{start}, {end}) at width={width}"
                )


# ---------------------------------------------------------------------------
# 11. Visible window — tracks navigation
# ---------------------------------------------------------------------------


class TestWindowFollowsNavigation:
    def test_window_shifts_as_selection_moves(self) -> None:
        sel = _make_selector(doc_ids=[0, 1, 2, 3, 4])
        # sep=0, slot_w=3, prefix=5: full=20, width=18 forces truncation
        start0, end0 = sel._visible_window(18)
        assert start0 == 0  # initially at doc 0

        sel.navigate(1)  # doc 1
        sel.navigate(1)  # doc 2
        start2, _ = sel._visible_window(18)
        assert start2 > 0  # window has shifted

        for _ in range(2):
            sel.navigate(1)  # doc 4
        _, end4 = sel._visible_window(18)
        assert end4 == 5  # pinned to end


# ---------------------------------------------------------------------------
# 12. render_bar — indicators
# ---------------------------------------------------------------------------


class TestRenderBarIndicators:
    def test_no_truncation_no_indicators(self) -> None:
        sel = _make_selector(doc_ids=[0, 1, 2])
        bar = sel.render_bar(200).plain
        assert "< " not in bar
        assert " >" not in bar

    def test_truncated_start_has_placeholder_left_and_indicator_right(self) -> None:
        sel = _make_selector(doc_ids=[0, 1, 2, 3, 4])
        bar = sel.render_bar(18).plain
        after_prefix = bar[len("Doc ") :]
        # Left placeholder (spaces, not arrow)
        assert after_prefix.startswith("  ")
        assert not after_prefix.startswith("< ")
        # Right indicator
        assert bar.endswith(" >")

    def test_truncated_end_has_indicator_left_and_placeholder_right(self) -> None:
        sel = _make_selector(doc_ids=[0, 1, 2, 3, 4])
        for _ in range(4):
            sel.navigate(1)
        bar = sel.render_bar(18).plain
        after_prefix = bar[len("Doc ") :]
        # Left indicator
        assert after_prefix.startswith("< ")
        # Right placeholder (spaces, not arrow)
        assert bar.endswith("  ")
        assert not bar.endswith(" >")

    def test_both_indicators(self) -> None:
        sel = _make_selector(doc_ids=[0, 1, 2, 3, 4])
        sel.navigate(1)
        sel.navigate(1)  # doc 2
        bar = sel.render_bar(18).plain
        after_prefix = bar[len("Doc ") :]
        assert after_prefix.startswith("< ")
        assert bar.endswith(" >")

    def test_bar_length_stable_across_scroll(self) -> None:
        """Positions must not jump when scrolling brings in/out indicators."""
        sel = _make_selector(doc_ids=[0, 1, 2, 3, 4])
        width = 18
        lengths = set()
        for i in range(5):
            sel._selected_idx = i
            lengths.add(len(sel.render_bar(width)))
        assert len(lengths) == 1, f"Bar lengths varied: {lengths}"


# ---------------------------------------------------------------------------
# 13. render_bar — content correctness
# ---------------------------------------------------------------------------


class TestRenderBarContent:
    def test_selected_doc_has_brackets(self) -> None:
        sel = _make_selector(doc_ids=[0, 1, 2])
        sel.navigate(1)  # select doc 1
        bar = sel.render_bar(200).plain
        assert "[1]" in bar
        assert "[0]" not in bar
        assert "[2]" not in bar

    def test_none_doc(self) -> None:
        sel = _make_selector(doc_ids=[0])
        sel.handle_digit(9)
        assert sel.render_bar(80).plain == "Doc (none)"

    def test_width_zero_shows_all(self) -> None:
        sel = _make_selector(doc_ids=[0, 1, 2, 3, 4])
        bar = sel.render_bar(0).plain
        for d in [0, 1, 2, 3, 4]:
            assert str(d) in bar
        assert "< " not in bar
        assert " >" not in bar

    def test_bar_fits_within_width(self) -> None:
        """Rendered bar must not exceed the given width."""
        sel = _make_selector()  # 13 docs, 3-digit max
        for width in [20, 30, 40, 50, 60, 80, 100, 120]:
            bar = sel.render_bar(width)
            assert len(bar) <= width, f"bar len {len(bar)} > width {width}: {bar!r}"

    def test_bar_fits_across_all_selections(self) -> None:
        """Width constraint holds regardless of which doc is selected."""
        sel = _make_selector()
        for width in [30, 50, 80]:
            for i in range(len(sel._filtered_docs)):
                sel._selected_idx = i
                bar = sel.render_bar(width)
                assert len(bar) <= width, f"selected={i}, bar len {len(bar)} > {width}: {bar!r}"


# ---------------------------------------------------------------------------
# Up/Down segment keys: up decrements, down increments
# ---------------------------------------------------------------------------


class TestUpDownSegmentDirection:
    def test_up_key_decrements_segments(self) -> None:
        sel = _make_selector(doc_ids=[4], total_segments={4: 5})
        assert sel.num_segments_for_selected == 2
        sel._key_up()
        assert sel.num_segments_for_selected == 1

    def test_down_key_increments_segments(self) -> None:
        sel = _make_selector(doc_ids=[4], total_segments={4: 5})
        assert sel.num_segments_for_selected == 2
        sel._key_down()
        assert sel.num_segments_for_selected == 3

    def test_up_clamps_at_one(self) -> None:
        sel = _make_selector(doc_ids=[4], total_segments={4: 5})
        sel._key_up()  # 2 → 1
        sel._key_up()  # clamp at 1
        assert sel.num_segments_for_selected == 1

    def test_down_clamps_at_total(self) -> None:
        sel = _make_selector(doc_ids=[4], total_segments={4: 3})
        sel._key_down()  # 2 → 3
        sel._key_down()  # clamp at 3
        assert sel.num_segments_for_selected == 3
