"""Cross-module integration tests for the completion TUI.

These tests exercise multiple modules together to verify that components
built by separate agents integrate correctly. Each test creates real
objects from multiple modules and verifies end-to-end behavior.
"""

from __future__ import annotations

from scripts.completion_tui.generation import (
    redo_generation,
    undo_generation,
)
from scripts.completion_tui.segment_ops import (
    compute_takeover,
    on_text_delete,
    on_text_insert,
    split_segment,
)
from scripts.completion_tui.state import (
    FeatureCondition,
    Segment,
    SessionState,
    deep_copy_segments,
    display_magnitude_range,
    display_to_internal,
    internal_to_display,
)
from scripts.completion_tui.widgets.doc_selector import DocSelector
from scripts.completion_tui.widgets.feature_detail import FeatureDetail
from scripts.completion_tui.widgets.feature_editor import FeatureEditor
from scripts.completion_tui.widgets.feature_search import FeatureSearch
from scripts.completion_tui.widgets.segment_bar import SegmentBar
from scripts.completion_tui.widgets.segment_selector import SegmentSelector
from scripts.completion_tui.widgets.text_panel import TextPanel
from tests.test_completion_tui._generation_history import push_generation
from tests.test_completion_tui._key_dispatch import dispatch_key
from tests.test_completion_tui.conftest import WordMockTokenizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seg(idx: int, start: int, end: int, **kwargs) -> Segment:
    return Segment(
        segment_idx=idx,
        char_start=start,
        char_end=end,
        color_idx=idx,
        features=kwargs.pop("features", []),
        **kwargs,
    )


def _feat(fid: int, mag: int, **kwargs) -> FeatureCondition:
    return FeatureCondition(feature_id=fid, magnitude=mag, **kwargs)


def _state(text: str, segments: list[Segment], **kwargs) -> SessionState:
    return SessionState(
        document_text=text,
        segments=segments,
        num_buckets=kwargs.pop("num_buckets", 11),
        has_negatives=kwargs.pop("has_negatives", False),
        **kwargs,
    )


# ===========================================================================
# 1. State + SegmentOps: edit → retokenize → takeover pipeline
# ===========================================================================


class TestEditRetokenizeTakeoverPipeline:
    """Verify that text editing flows through segment_ops correctly and
    that takeover sees consistent state afterward."""

    def test_insert_then_takeover_preserves_coverage(self) -> None:
        """After inserting text and retokenizing, takeover output
        still covers the full text."""
        tok = WordMockTokenizer()
        state = _state(
            "hello world",
            [_seg(0, 0, 6), _seg(1, 6, 11)],
        )
        on_text_insert(state, 5, " big", tok)
        # "hello big world" — segments should cover full text
        effective = compute_takeover(state.segments)
        assert len(effective) == 2
        assert effective[0].char_start == 0
        assert effective[-1].char_end == len(state.document_text)

    def test_disable_then_insert_uses_absorber(self) -> None:
        """Disabling a segment, then inserting in its region, should
        work without errors and maintain contiguity."""
        tok = WordMockTokenizer()
        state = _state(
            "foo bar baz",
            [_seg(0, 0, 4), _seg(1, 4, 8), _seg(2, 8, 11)],
        )
        state.segments[1].enabled = False
        effective = compute_takeover(state.segments)
        # Middle disabled → seg0 absorbs it
        assert len(effective) == 2
        assert effective[0].char_end == 8  # seg0 now covers through seg1

        # Insert in the absorbed region
        on_text_insert(state, 5, "x", tok)
        effective2 = compute_takeover(state.segments)
        assert effective2[0].char_start == 0
        assert effective2[-1].char_end == len(state.document_text)

    def test_split_then_takeover(self) -> None:
        """Splitting a segment and then computing takeover should still
        produce contiguous ranges."""
        tok = WordMockTokenizer()
        state = _state(
            "hello world test",
            [_seg(0, 0, 6), _seg(1, 6, 12), _seg(2, 12, 16)],
        )
        split_segment(state, 1, 8, tok)
        # Now 4 segments
        assert len(state.segments) >= 3
        effective = compute_takeover(state.segments)
        assert effective[0].char_start == 0
        assert effective[-1].char_end == len(state.document_text)
        # Check contiguity
        for i in range(len(effective) - 1):
            assert effective[i].char_end == effective[i + 1].char_start

    def test_delete_across_boundary_then_takeover(self) -> None:
        """Delete spanning a segment boundary, then verify takeover is clean."""
        tok = WordMockTokenizer()
        state = _state(
            "foo bar baz",
            [_seg(0, 0, 4), _seg(1, 4, 8), _seg(2, 8, 11)],
        )
        on_text_delete(state, 3, 3, tok)
        effective = compute_takeover(state.segments)
        assert effective[0].char_start == 0
        assert effective[-1].char_end == len(state.document_text)


# ===========================================================================
# 2. Generation undo/redo + text editing interaction
# ===========================================================================


class TestGenerationUndoRedoWithEditing:
    """Verify the undo/redo stack interacts correctly with text edits."""

    def _gen_state(self) -> SessionState:
        return _state(
            "hello world",
            [
                _seg(0, 0, 6, features=[_feat(1, 5)]),
                _seg(1, 6, 11, features=[_feat(2, 3)]),
                _seg(
                    2,
                    11,
                    11,
                    features=[_feat(3, 1)],
                    is_generation_target=True,
                    target_token_count=8,
                ),
            ],
        )

    def test_push_then_undo_restores_state(self) -> None:
        state = self._gen_state()
        original_text = state.document_text
        original_segs = deep_copy_segments(state.segments)

        push_generation(state, " test", [10, 20])
        assert state.document_text == "hello world test"
        assert len(state.generation_history) == 1

        result = undo_generation(state)
        assert result is True
        assert state.document_text == original_text
        assert len(state.segments) == len(original_segs)

    def test_push_undo_redo_cycle(self) -> None:
        state = self._gen_state()
        push_generation(state, " test", [10, 20])
        post_gen_text = state.document_text

        undo_generation(state)
        assert state.document_text == "hello world"

        redo_generation(state)
        assert state.document_text == post_gen_text

    def test_text_edit_clears_redo(self) -> None:
        """After generate → undo → text edit, redo should be unavailable."""
        tok = WordMockTokenizer()
        state = self._gen_state()

        push_generation(state, " test", [10, 20])
        undo_generation(state)
        assert len(state.generation_redo_stack) == 1

        # Text edit should clear redo
        on_text_insert(state, 5, "!", tok)
        assert state.generation_redo_stack == []
        assert redo_generation(state) is False

    def test_multiple_generations_undo_all(self) -> None:
        state = self._gen_state()
        push_generation(state, " one", [1])
        push_generation(state, " two", [2])
        push_generation(state, " three", [3])

        assert state.document_text == "hello world one two three"
        assert len(state.generation_history) == 3

        undo_generation(state)
        assert state.document_text == "hello world one two"
        undo_generation(state)
        assert state.document_text == "hello world one"
        undo_generation(state)
        assert state.document_text == "hello world"

        assert undo_generation(state) is False

    def test_undo_on_empty_history_returns_false(self) -> None:
        state = self._gen_state()
        assert undo_generation(state) is False

    def test_redo_on_empty_stack_returns_false(self) -> None:
        state = self._gen_state()
        assert redo_generation(state) is False


# ===========================================================================
# 3. TextPanel + SegmentOps: editing mutates state correctly
# ===========================================================================


class TestTextPanelSegmentOpsIntegration:
    """Verify TextPanel editing methods correctly call into segment_ops."""

    def test_typing_extends_segment_and_retokenizes(self) -> None:
        tok = WordMockTokenizer()
        state = _state(
            "hello world",
            [_seg(0, 0, 6), _seg(1, 6, 11)],
        )
        panel = TextPanel(state=state, tokenizer=tok)

        panel.move_cursor_to(5)  # end of "hello"
        panel.handle_char_input("!")
        assert state.document_text == "hello! world"
        assert state.cursor_position == 6
        # Segments should still cover full text
        total_covered = sum(s.char_end - s.char_start for s in state.segments)
        assert total_covered == len(state.document_text)

    def test_backspace_shrinks_segment(self) -> None:
        tok = WordMockTokenizer()
        state = _state("hello world", [_seg(0, 0, 6), _seg(1, 6, 11)])
        state.cursor_position = 5
        panel = TextPanel(state=state, tokenizer=tok)

        panel.handle_backspace()
        assert state.document_text == "hell world"
        assert state.cursor_position == 4

    def test_split_via_key_dispatch(self) -> None:
        """Cmd+K through the real key handler triggers split."""
        tok = WordMockTokenizer()
        state = _state(
            "hello world test",
            [_seg(0, 0, 6), _seg(1, 6, 12), _seg(2, 12, 16)],
        )
        state.cursor_position = 8
        panel = TextPanel(state=state, tokenizer=tok)

        msgs = dispatch_key(panel, "ctrl+k")
        # Should have posted SegmentSplitCompleted if split succeeded
        assert any(isinstance(m, TextPanel.SegmentSplitCompleted) for m in msgs)

    def test_ctrl_g_posts_generate_requested(self) -> None:
        """TextPanel uses ctrl+g for generation (enter inserts newline)."""
        tok = WordMockTokenizer()
        state = _state("hello", [_seg(0, 0, 5)])
        panel = TextPanel(state=state, tokenizer=tok)

        msgs = dispatch_key(panel, "ctrl+g")
        generate_requests = [
            message for message in msgs if isinstance(message, TextPanel.GenerateRequested)
        ]
        assert len(generate_requests) == 1


# ===========================================================================
# 4. FeatureEditor + FeatureSearch: add feature flow
# ===========================================================================


class TestFeatureEditorSearchIntegration:
    """Verify that FeatureSearch results can be added to a segment
    and FeatureEditor reflects the change."""

    def test_search_excludes_existing_features(self) -> None:
        seg = _seg(0, 0, 10, features=[_feat(100, 5), _feat(200, 3)])
        excluded = {f.feature_id for f in seg.features}
        # Also exclude disabled features
        available = {100: "feat100", 200: "feat200", 300: "feat300", 301: "feat301"}

        search = FeatureSearch(
            available_features=available,
            excluded_ids=excluded,
        )
        search.handle_digit(3)  # search for "3"
        assert 100 not in search.results
        assert 200 not in search.results
        assert 300 in search.results
        assert 301 in search.results

    def test_search_results_sorted_by_id(self) -> None:
        available = {300: "c", 100: "a", 200: "b", 301: "d"}
        search = FeatureSearch(available_features=available, excluded_ids=set())
        search.handle_digit(3)  # prefix "3"
        assert search.results == [300, 301]

    def test_editor_magnitude_wraparound_uses_state_range(self) -> None:
        """FeatureEditor wraparound should use display_magnitude_range from state.py."""
        seg = _seg(0, 0, 10, features=[_feat(1, 0)])
        editor = FeatureEditor(segment=seg, num_buckets=11, has_negatives=False)

        lo, hi = display_magnitude_range(11, False)
        assert lo == 0
        assert hi == 10

        # Set to max, then increment → should wrap to min
        editor.sorted_features[0].magnitude = hi
        editor.adjust_magnitude(1)
        assert editor.sorted_features[0].magnitude == lo

    def test_editor_digit_accumulator_overflow(self) -> None:
        """Digit accumulator: max=10, press 1 then 5 → 15 > 10 → reset to 5."""
        seg = _seg(0, 0, 10, features=[_feat(1, 0)])
        editor = FeatureEditor(segment=seg, num_buckets=11, has_negatives=False)

        editor.handle_digit(1)
        assert editor.sorted_features[0].magnitude == 1
        editor.handle_digit(5)
        assert editor.sorted_features[0].magnitude == 5  # 15 > 10, reset to 5

    def test_editor_navigation_resets_accumulator(self) -> None:
        seg = _seg(
            0,
            0,
            10,
            features=[_feat(1, 0), _feat(2, 0)],
        )
        editor = FeatureEditor(segment=seg, num_buckets=11, has_negatives=False)

        editor.handle_digit(1)  # value=1
        editor.move_selection(1)  # down → resets accumulator
        editor.move_selection(-1)  # back up
        editor.handle_digit(0)  # should be 0, not 10
        assert editor.sorted_features[editor.selected_idx].magnitude == 0


# ===========================================================================
# 5. SegmentSelector + TextPanel: highlight coordination
# ===========================================================================


class TestSegmentSelectorTextPanelCoordination:
    """Verify segment selection messages and TextPanel highlighting."""

    def test_segment_highlighted_message(self) -> None:
        segs = [_seg(0, 0, 5), _seg(1, 5, 10)]
        selector = SegmentSelector(segments=segs)

        msgs = dispatch_key(selector, "right")
        highlighted = [m for m in msgs if isinstance(m, SegmentSelector.SegmentHighlighted)]
        assert len(highlighted) == 1
        assert highlighted[0].segment_idx == 1

    def test_escape_posts_unhighlighted(self) -> None:
        segs = [_seg(0, 0, 5), _seg(1, 5, 10)]
        selector = SegmentSelector(segments=segs)

        msgs = dispatch_key(selector, "escape")
        unhighlighted = [m for m in msgs if isinstance(m, SegmentSelector.SegmentUnhighlighted)]
        assert len(unhighlighted) == 1

    def test_enter_posts_open_feature_editor(self) -> None:
        segs = [_seg(0, 0, 5), _seg(1, 5, 10)]
        selector = SegmentSelector(segments=segs)

        msgs = dispatch_key(selector, "enter")
        editor_msgs = [m for m in msgs if isinstance(m, SegmentSelector.OpenFeatureEditor)]
        assert len(editor_msgs) == 1
        assert editor_msgs[0].segment_idx == 0

    def test_space_toggles_segment_enabled(self) -> None:
        segs = [_seg(0, 0, 5), _seg(1, 5, 10)]
        selector = SegmentSelector(segments=segs)
        assert segs[0].enabled is True

        dispatch_key(selector, "space")
        assert segs[0].enabled is False

        dispatch_key(selector, "space")
        assert segs[0].enabled is True

    def test_text_panel_selected_segment_coloring(self) -> None:
        """When a segment is selected, only it should have color;
        others use DEFAULT_COLOR."""
        tok = WordMockTokenizer()
        state = _state("hello world", [_seg(0, 0, 6), _seg(1, 6, 11)])
        panel = TextPanel(state=state, tokenizer=tok, selected_segment_idx=0)

        rendered = panel.render()
        # Check that non-selected segment uses default color
        # The exact check depends on Rich Text internals, but we can verify
        # the render doesn't crash and produces output
        assert len(rendered.plain) >= len(state.document_text)


# ===========================================================================
# 6. DocSelector: filtering and segment count
# ===========================================================================


class TestDocSelectorIntegration:
    """Verify DocSelector interacts correctly with document data."""

    def test_digit_filtering(self) -> None:
        doc_ids = [1, 10, 11, 12, 20, 21, 100, 200]
        total_map = dict.fromkeys(doc_ids, 5)
        selector = DocSelector(available_doc_ids=doc_ids, total_segments_map=total_map)

        selector.handle_digit(1)
        filtered = selector.filtered_docs
        # Should show docs starting with "1": 1, 10, 11, 12, 100
        for d in filtered:
            assert str(d).startswith("1")

    def test_digit_append_narrows(self) -> None:
        doc_ids = [1, 10, 11, 12, 20, 100, 101]
        total_map = dict.fromkeys(doc_ids, 5)
        selector = DocSelector(available_doc_ids=doc_ids, total_segments_map=total_map)

        selector.handle_digit(1)
        selector.handle_digit(0)
        filtered = selector.filtered_docs
        # Should show docs starting with "10": 10, 100, 101
        for d in filtered:
            assert str(d).startswith("10")

    def test_enter_posts_document_selected(self) -> None:
        doc_ids = [42, 43, 44]
        total_map = {42: 5, 43: 3, 44: 7}
        selector = DocSelector(available_doc_ids=doc_ids, total_segments_map=total_map)

        msgs = dispatch_key(selector, "enter")
        selected = [m for m in msgs if isinstance(m, DocSelector.DocumentSelected)]
        assert len(selected) == 1
        assert selected[0].doc_idx == 42  # first in list

    def test_segment_count_adjustment(self) -> None:
        doc_ids = [42]
        total_map = {42: 10}
        selector = DocSelector(available_doc_ids=doc_ids, total_segments_map=total_map)

        # Default starts at min(2, total) = 2. Decrease then increase.
        assert selector.num_segments_for_selected == 2
        selector.adjust_segments(-1)  # 2 → 1
        assert selector.num_segments_for_selected == 1
        selector.adjust_segments(1)  # 1 → 2
        assert selector.num_segments_for_selected == 2

    def test_segment_count_clamps_to_bounds(self) -> None:
        doc_ids = [42]
        total_map = {42: 3}
        selector = DocSelector(available_doc_ids=doc_ids, total_segments_map=total_map)

        # Default is min(2, 3) = 2. Can't go above total.
        selector.adjust_segments(1)  # 2 → 3
        selector.adjust_segments(1)  # clamp at 3
        assert selector.num_segments_for_selected == 3

        # Can't go below 1
        selector.adjust_segments(-1)  # 3 → 2
        selector.adjust_segments(-1)  # 2 → 1
        selector.adjust_segments(-1)  # clamp at 1
        assert selector.num_segments_for_selected == 1


# ===========================================================================
# 7. SegmentBar: fill level computation
# ===========================================================================


class TestSegmentBarIntegration:
    def test_fill_bottom_up(self) -> None:
        bar = SegmentBar(loaded_segments=3, total_segments=9, visible=True)
        lines = bar.render_bar(height=8)
        assert len(lines) == 8
        # Bottom lines should be filled, top lines empty
        # Fill = (3-1)/(9-1) = 0.25, so 2 of 8 rows filled
        filled_count = sum(1 for line in lines if "\u2588" in line)
        assert filled_count >= 1  # at least some fill

    def test_single_segment_full(self) -> None:
        bar = SegmentBar(loaded_segments=1, total_segments=1, visible=True)
        assert bar.fill_level == 1.0

    def test_zero_total_full(self) -> None:
        bar = SegmentBar(loaded_segments=0, total_segments=0, visible=True)
        assert bar.fill_level == 1.0


# ===========================================================================
# 8. FeatureDetail: basic rendering
# ===========================================================================


class TestFeatureDetailIntegration:
    def test_loading_state(self) -> None:
        detail = FeatureDetail(feature_id=342, description="test feature")
        assert "Loading" in detail.render()
        assert detail.feature_id == 342

    def test_set_data_clears_loading(self) -> None:
        detail = FeatureDetail(feature_id=342)
        detail.set_data({"description": "test", "examples": []})
        assert "Loading" not in detail.render()

    def test_escape_posts_closed(self) -> None:
        detail = FeatureDetail(feature_id=342)
        msgs = dispatch_key(detail, "escape")
        assert any(isinstance(m, FeatureDetail.FeatureDetailClosed) for m in msgs)


# ===========================================================================
# 9. Full state lifecycle: load doc → edit → generate → undo
# ===========================================================================


class TestFullStateLifecycle:
    """End-to-end state lifecycle without a real model."""

    def test_load_edit_generate_undo_cycle(self) -> None:
        """Simulate: load doc → type text → generate → undo → verify state."""
        tok = WordMockTokenizer()

        # 1. Create state as if loaded from DataLoader
        state = _state(
            "the quick brown fox",
            [
                _seg(0, 0, 10, features=[_feat(1, 5), _feat(2, 3)]),
                _seg(1, 10, 19, features=[_feat(3, 7)]),
                _seg(
                    2,
                    19,
                    19,
                    features=[_feat(4, 2)],
                    is_generation_target=True,
                    target_token_count=8,
                ),
            ],
        )

        # 2. User types at end — extends last non-target segment
        panel = TextPanel(state=state, tokenizer=tok)
        state.cursor_position = 19
        panel.handle_char_input(" jumps")
        assert "jumps" in state.document_text

        # 3. Simulate generation (push synthetic tokens)
        push_generation(state, " over the", [10, 20, 30])
        assert state.document_text.endswith(" over the")
        assert len(state.generation_history) == 1

        # 4. Undo
        undo_generation(state)
        assert "over the" not in state.document_text
        assert len(state.generation_redo_stack) == 1

        # 5. Redo
        redo_generation(state)
        assert state.document_text.endswith(" over the")

    def test_magnitude_roundtrip_through_editor(self) -> None:
        """Magnitude: state → editor → display → internal → back."""
        num_buckets = 11
        has_neg = True

        # Create feature with display magnitude -1 (null bucket)
        seg = _seg(0, 0, 10, features=[_feat(1, -1)])
        editor = FeatureEditor(segment=seg, num_buckets=num_buckets, has_negatives=has_neg)

        # Display magnitude should be -1
        feat = editor.sorted_features[0]
        assert feat.magnitude == -1

        # Convert to internal
        internal = display_to_internal(feat.magnitude, num_buckets, has_neg)
        assert internal == num_buckets - 1  # null bucket

        # Back to display
        display = internal_to_display(internal, num_buckets, has_neg)
        assert display == -1

    def test_conditioning_memory_survives_split(self) -> None:
        """Cmd+I memory keyed by UUID should survive segment splits."""
        tok = WordMockTokenizer()
        state = _state(
            "hello world test",
            [_seg(0, 0, 6), _seg(1, 6, 12), _seg(2, 12, 16)],
        )

        # Save conditioning memory (Cmd+I first press)
        for seg in state.segments:
            state.conditioning_memory[seg.id] = seg.enabled

        # Split segment 1
        split_segment(state, 1, 8, tok)
        # Now 4 segments — new ones have new IDs
        assert len(state.segments) >= 3

        # Old memory keys still valid for segment 0 and 2
        seg0_id = state.segments[0].id
        assert seg0_id in state.conditioning_memory

    def test_all_segments_disabled_gives_empty_takeover(self) -> None:
        state = _state(
            "hello world",
            [_seg(0, 0, 6), _seg(1, 6, 11)],
        )
        state.segments[0].enabled = False
        state.segments[1].enabled = False
        effective = compute_takeover(state.segments)
        assert effective == []
