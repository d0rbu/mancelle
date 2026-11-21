"""Tests for scripts/completion_tui/segment_ops.py.

Exhaustive retokenization tests are the most critical — the re-snap logic
is the trickiest part of the entire TUI.
"""

from __future__ import annotations

from scripts.completion_tui.segment_ops import (
    compute_takeover,
    on_text_delete,
    on_text_insert,
    retokenize_boundaries,
    split_segment,
)
from scripts.completion_tui.state import (
    FeatureCondition,
    GenerationSnapshot,
    Segment,
    SessionState,
)
from tests.test_completion_tui.conftest import WordMockTokenizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seg(
    idx: int,
    start: int,
    end: int,
    *,
    enabled: bool = True,
    color_idx: int = 0,
    features: list[FeatureCondition] | None = None,
) -> Segment:
    """Shorthand segment constructor for tests."""
    return Segment(
        segment_idx=idx,
        char_start=start,
        char_end=end,
        color_idx=color_idx,
        features=features or [],
        enabled=enabled,
    )


def _generation_snapshot() -> GenerationSnapshot:
    return GenerationSnapshot(
        document_text="",
        segments=[],
        generated_text="",
        generated_token_ids=[],
        seed=0,
    )


def _state(
    text: str,
    segments: list[Segment],
) -> SessionState:
    """Shorthand SessionState constructor."""
    return SessionState(
        document_text=text,
        segments=segments,
        num_buckets=11,
        has_negatives=True,
    )


def _assert_contiguous(segments: list[Segment], text_len: int | None = None) -> None:
    """Assert segment char ranges are contiguous and non-overlapping."""
    if not segments:
        return
    # Sorted by char_start
    sorted_segs = sorted(segments, key=lambda s: s.char_start)
    for i in range(1, len(sorted_segs)):
        assert sorted_segs[i].char_start == sorted_segs[i - 1].char_end, (
            f"Gap or overlap between segment {i - 1} and {i}: "
            f"{sorted_segs[i - 1].char_end} != {sorted_segs[i].char_start}"
        )
    if text_len is not None:
        assert sorted_segs[0].char_start == 0
        assert sorted_segs[-1].char_end == text_len


# ===========================================================================
# retokenize_boundaries
# ===========================================================================


class TestRetokenizeBoundaries:
    """The core retokenization tests — most critical in the entire TUI."""

    def test_insert_mid_token_expands_segment(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Insert char in middle of token within segment → segment expands."""
        # "hello world" → Seg0=[0,5) Seg1=[5,11)
        # Tokens: ["hello", " ", "world"], offsets [0, 5, 6, 11]
        # Insert 'x' at pos 2 → "hexllo world"
        # Tokens: ["h", "e", "x", "l", "l", "o", " ", "world"]
        # Seg0 should expand to cover the new characters
        text = "hexllo world"
        segs = [_seg(0, 0, 6), _seg(1, 6, 12)]  # pre-shifted boundaries
        retokenize_boundaries(text, segs, word_tokenizer)
        # Token boundary: all single chars in "hexllo" then " " then "world"
        # Single chars: h=0, e=1, x=2, l=3, l=4, o=5, " "=6
        # "world" covers [7, 12)
        # Seg0 should snap to [0, 6) or [0, 7) depending on overlap
        # Seg1 should cover [6, 12) or [7, 12)
        assert segs[0].char_start == 0
        assert segs[1].char_end == len(text)
        _assert_contiguous(segs, len(text))

    def test_insert_at_segment_boundary(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Insert at exact segment boundary → correct segment absorbs."""
        # "foo bar" → tokens ["foo", " ", "bar"], offsets [0, 3, 4, 7]
        # Seg0=[0,3) Seg1=[3,7)
        # Insert "x" at pos 3 (the boundary)
        # → "foox bar", Seg0=[0,3), Seg1=[4,8) after initial shift
        text = "foox bar"
        # After shift, segments are contiguous (retokenize requires full coverage).
        segs = [_seg(0, 0, 3), _seg(1, 3, 8)]
        retokenize_boundaries(text, segs, word_tokenizer)
        # retokenize re-snaps boundaries to be contiguous
        _assert_contiguous(segs, len(text))

    def test_insert_splits_token_across_boundary(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Insert that splits a token across original boundary → re-snap via overlap."""
        # "foobar" → tokens ["foo", "bar"], offsets [0, 3, 6]
        # Seg0=[0,3) Seg1=[3,6)
        # Insert "x" at pos 3 → "fooxbar"
        # Now tokenized as single chars: f, o, o, x, b, a, r
        text = "fooxbar"
        segs = [_seg(0, 0, 3), _seg(1, 3, 7)]  # after shift (contiguous)
        retokenize_boundaries(text, segs, word_tokenizer)
        _assert_contiguous(segs, len(text))

    def test_delete_at_segment_boundary(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Delete char at segment boundary → boundary re-snaps correctly."""
        # "foo bar" → Seg0=[0,4) Seg1=[4,7)
        # Delete " " at pos 3 → "foobar"
        # Tokens: ["foo", "bar"], offsets [0, 3, 6]
        text = "foobar"
        segs = [_seg(0, 0, 3), _seg(1, 3, 6)]  # after shift
        retokenize_boundaries(text, segs, word_tokenizer)
        _assert_contiguous(segs, len(text))
        # Each segment should snap to a token boundary
        assert segs[0].char_start == 0
        assert segs[1].char_end == 6

    def test_delete_across_segment_boundary(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Delete across segment boundary → correct merge behavior."""
        # "foo bar baz" → Seg0=[0,4) Seg1=[4,8) Seg2=[8,11)
        # Delete chars at [3, 5) → "fooar baz"
        # After boundary shift, retokenize snaps to token edges
        text = "fooar baz"
        segs = [_seg(0, 0, 3), _seg(1, 3, 6), _seg(2, 6, 9)]
        retokenize_boundaries(text, segs, word_tokenizer)
        _assert_contiguous(segs, len(text))

    def test_delete_entire_segment_text(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Delete entire segment text → zero-width segment, no crash."""
        # "foo bar" → Seg0=[0,4) Seg1=[4,7)
        # Delete "foo " → "bar"
        # After shift: Seg0=[0,0) Seg1=[0,3)
        text = "bar"
        segs = [_seg(0, 0, 0), _seg(1, 0, 3)]
        retokenize_boundaries(text, segs, word_tokenizer)
        # Seg0 should be zero-width, Seg1 covers the text
        assert segs[0].char_start == segs[0].char_end  # zero-width
        assert segs[1].char_end == len(text)

    def test_multibyte_utf8_insertion(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Multi-byte UTF-8 insertion → char offsets correct."""
        # "foo bar" → insert "é" (1 Python char, multi-byte UTF-8)
        text = "fooé bar"
        segs = [_seg(0, 0, 4), _seg(1, 4, 8)]
        retokenize_boundaries(text, segs, word_tokenizer)
        _assert_contiguous(segs, len(text))
        # "é" is 1 char in Python, so len("fooé bar") == 8
        assert segs[0].char_start == 0
        assert segs[1].char_end == 8

    def test_many_sequential_edits_consistent(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Many rapid sequential edits → final state is consistent."""

        # Simulate multiple edits
        edits = [
            ("hello world!", [_seg(0, 0, 5), _seg(1, 5, 12)]),
            ("hello world!!", [_seg(0, 0, 5), _seg(1, 5, 13)]),
            ("hello world!!x", [_seg(0, 0, 5), _seg(1, 5, 14)]),
        ]
        for new_text, new_segs in edits:
            retokenize_boundaries(new_text, new_segs, word_tokenizer)
            _assert_contiguous(new_segs, len(new_text))

    def test_invariant_contiguous_nonoverlapping_full_coverage(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Edit then check invariant: boundaries contiguous, non-overlapping, cover full text."""
        text = "the quick brown fox"
        # Tokens: ["the", " ", "quick", " ", "brown", " ", "fox"]
        # offsets: [0, 3, 4, 9, 10, 15, 16, 19]
        segs = [_seg(0, 0, 4), _seg(1, 4, 10), _seg(2, 10, 19)]
        retokenize_boundaries(text, segs, word_tokenizer)
        _assert_contiguous(segs, len(text))
        # Verify each segment has non-negative width
        for seg in segs:
            assert seg.char_end >= seg.char_start

    def test_spec_hello_heltlo_worked_example(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """The spec worked example: "hello" → "heltlo".

        "hello world" with Seg0=[0,5) Seg1=[5,11)
        Insert 't' at pos 3
        After re-snap: Seg0=[0,6) Seg1=[6,12)
        """
        # Before: "hello world", Seg0=[0,5) Seg1=[5,11)
        # Insert 't' at pos 3 → "heltlo world"
        # After boundary shift: Seg0=[0,6) Seg1=[6,12)
        text = "heltlo world"
        segs = [_seg(0, 0, 6), _seg(1, 6, 12)]
        retokenize_boundaries(text, segs, word_tokenizer)

        # Tokens: ["hel", "tlo", " ", "world"]
        # offsets: [0, 3, 6, 7, 12]
        # Token assignments by max overlap:
        #   "hel" [0,3) vs Seg0 [0,6): overlap 3; vs Seg1 [6,12): overlap 0 → Seg0
        #   "tlo" [3,6) vs Seg0 [0,6): overlap 3; vs Seg1 [6,12): overlap 0 → Seg0
        #   " "   [6,7) vs Seg0 [0,6): overlap 0; vs Seg1 [6,12): overlap 1 → Seg1
        #   "world" [7,12) vs Seg0 [0,6): overlap 0; vs Seg1 [6,12): overlap 5 → Seg1
        # Assignments: [0, 0, 1, 1] — already monotonic
        # Seg0: tokens 0,1 → [offsets[0], offsets[2]) = [0, 6)
        # Seg1: tokens 2,3 → [offsets[2], offsets[4]) = [6, 12)
        assert segs[0].char_start == 0
        assert segs[0].char_end == 6
        assert segs[1].char_start == 6
        assert segs[1].char_end == 12

    def test_gap_between_segments_raises(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Segments with a gap are rejected by the underlying segmentation pass.

        Previously this test exercised cummax correction for non-monotonic
        token-to-segment assignments. The current segmentation contract
        raises ``ValueError`` when any token has zero overlap with every
        segment, so this case is now a hard error rather than a silent
        re-assignment. Callers must provide contiguous segment ranges.
        """
        import pytest

        text = "abcdef"
        segs = [_seg(0, 0, 2), _seg(1, 2, 3), _seg(2, 4, 6)]
        with pytest.raises(ValueError, match="zero overlap"):
            retokenize_boundaries(text, segs, word_tokenizer)

    def test_empty_text(self, word_tokenizer: WordMockTokenizer) -> None:
        """Empty text → all segments collapse to zero-width."""
        segs = [_seg(0, 0, 0), _seg(1, 0, 0)]
        retokenize_boundaries("", segs, word_tokenizer)
        for seg in segs:
            assert seg.char_start == 0
            assert seg.char_end == 0

    def test_single_segment_covers_all(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Single segment covering all text stays intact after retokenize."""
        text = "hello world"
        segs = [_seg(0, 0, 11)]
        retokenize_boundaries(text, segs, word_tokenizer)
        assert segs[0].char_start == 0
        assert segs[0].char_end == 11


# ===========================================================================
# compute_takeover
# ===========================================================================


class TestComputeTakeover:
    def test_all_enabled_unchanged(self) -> None:
        """All enabled → returned unchanged (but new objects)."""
        segs = [_seg(0, 0, 5), _seg(1, 5, 10), _seg(2, 10, 15)]
        result = compute_takeover(segs)
        assert len(result) == 3
        for orig, res in zip(segs, result, strict=True):
            assert res.char_start == orig.char_start
            assert res.char_end == orig.char_end
            assert res is not orig  # new objects

    def test_disable_middle_left_absorbs(self) -> None:
        """Disable middle → left absorbs."""
        segs = [
            _seg(0, 0, 5),
            _seg(1, 5, 10, enabled=False),
            _seg(2, 10, 15),
        ]
        result = compute_takeover(segs)
        assert len(result) == 2
        assert result[0].char_start == 0
        assert result[0].char_end == 10  # absorbed middle
        assert result[1].char_start == 10
        assert result[1].char_end == 15

    def test_disable_first_right_absorbs(self) -> None:
        """Disable first → right absorbs."""
        segs = [
            _seg(0, 0, 5, enabled=False),
            _seg(1, 5, 10),
            _seg(2, 10, 15),
        ]
        result = compute_takeover(segs)
        assert len(result) == 2
        assert result[0].char_start == 0  # absorbed first
        assert result[0].char_end == 10
        assert result[1].char_start == 10
        assert result[1].char_end == 15

    def test_disable_last_left_absorbs(self) -> None:
        """Disable last → left absorbs."""
        segs = [
            _seg(0, 0, 5),
            _seg(1, 5, 10),
            _seg(2, 10, 15, enabled=False),
        ]
        result = compute_takeover(segs)
        assert len(result) == 2
        assert result[0].char_start == 0
        assert result[0].char_end == 5
        assert result[1].char_start == 5
        assert result[1].char_end == 15  # absorbed last

    def test_disable_all_returns_empty(self) -> None:
        """Disable all → empty list."""
        segs = [
            _seg(0, 0, 5, enabled=False),
            _seg(1, 5, 10, enabled=False),
        ]
        result = compute_takeover(segs)
        assert result == []

    def test_disable_two_adjacent(self) -> None:
        """Disable two adjacent → proper chaining."""
        segs = [
            _seg(0, 0, 5),
            _seg(1, 5, 10, enabled=False),
            _seg(2, 10, 15, enabled=False),
            _seg(3, 15, 20),
        ]
        result = compute_takeover(segs)
        assert len(result) == 2
        # Left neighbor (seg 0) absorbs both disabled segments
        assert result[0].char_start == 0
        assert result[0].char_end == 15  # absorbed segs 1 and 2
        assert result[1].char_start == 15
        assert result[1].char_end == 20

    def test_original_segments_not_mutated(self) -> None:
        """INVARIANT: Original segments are never mutated."""
        segs = [
            _seg(0, 0, 5),
            _seg(1, 5, 10, enabled=False),
            _seg(2, 10, 15),
        ]
        originals = [(s.char_start, s.char_end, s.enabled) for s in segs]
        compute_takeover(segs)
        for seg, (start, end, en) in zip(segs, originals, strict=True):
            assert seg.char_start == start
            assert seg.char_end == end
            assert seg.enabled == en

    def test_output_contiguous_full_span(self) -> None:
        """INVARIANT: Output char ranges contiguous and cover full span."""
        segs = [
            _seg(0, 0, 5),
            _seg(1, 5, 10, enabled=False),
            _seg(2, 10, 15),
            _seg(3, 15, 20, enabled=False),
        ]
        result = compute_takeover(segs)
        assert len(result) > 0
        # Contiguous
        for i in range(1, len(result)):
            assert result[i].char_start == result[i - 1].char_end
        # Full span
        assert result[0].char_start == 0
        assert result[-1].char_end == 20

    def test_empty_input(self) -> None:
        """Empty input → empty output."""
        assert compute_takeover([]) == []

    def test_single_enabled(self) -> None:
        """Single enabled segment → returned as-is (new object)."""
        segs = [_seg(0, 0, 10)]
        result = compute_takeover(segs)
        assert len(result) == 1
        assert result[0].char_start == 0
        assert result[0].char_end == 10
        assert result[0] is not segs[0]


# ===========================================================================
# split_segment
# ===========================================================================


class TestSplitSegment:
    def test_split_mid_segment(self, word_tokenizer: WordMockTokenizer) -> None:
        """Split mid-segment → two segments, correct boundaries at token edge."""
        # "hello world" → tokens ["hello", " ", "world"]
        # offsets: [0, 5, 6, 11]
        # Seg0 covers entire text [0, 11)
        state = _state("hello world", [_seg(0, 0, 11, color_idx=0)])
        result = split_segment(state, 0, 6, word_tokenizer)  # cursor at pos 6
        assert result is True
        assert len(state.segments) == 2
        # Should split at token boundary: [0, 6) and [6, 11)
        # (nearest token boundary to cursor_pos=6 is offset 6)
        assert state.segments[0].char_end == state.segments[1].char_start
        _assert_contiguous(state.segments, 11)

    def test_split_at_segment_start_is_noop(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Split at segment start → no-op, returns False."""
        state = _state("hello world", [_seg(0, 0, 11)])
        result = split_segment(state, 0, 0, word_tokenizer)
        assert result is False
        assert len(state.segments) == 1

    def test_split_at_segment_end_is_noop(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Split at segment end → no-op, returns False."""
        state = _state("hello world", [_seg(0, 0, 11)])
        result = split_segment(state, 0, 11, word_tokenizer)
        assert result is False
        assert len(state.segments) == 1

    def test_split_deep_copies_features(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Both resulting segments have identical features (deep copied)."""
        features = [FeatureCondition(feature_id=42, magnitude=3, description="test")]
        state = _state("hello world", [_seg(0, 0, 11, features=features)])
        split_segment(state, 0, 6, word_tokenizer)
        assert len(state.segments) == 2
        # Both segments have the feature
        assert len(state.segments[0].features) == 1
        assert len(state.segments[1].features) == 1
        assert state.segments[0].features[0].feature_id == 42
        assert state.segments[1].features[0].feature_id == 42
        # Mutation independence
        state.segments[0].features[0].magnitude = 999
        assert state.segments[1].features[0].magnitude == 3

    def test_single_token_segment_is_noop(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Segment with exactly one token has no inner boundaries → returns False."""
        # "a" is a single-char token, segment covers [0,1)
        state = _state("a", [_seg(0, 0, 1)])
        result = split_segment(state, 0, 0, word_tokenizer)
        assert result is False
        assert len(state.segments) == 1

    def test_second_segment_gets_new_color(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Second segment gets a new color_idx."""
        state = _state("hello world", [_seg(0, 0, 11, color_idx=0)])
        split_segment(state, 0, 6, word_tokenizer)
        assert len(state.segments) == 2
        # Second segment should have a different color than the first
        assert state.segments[1].color_idx != state.segments[0].color_idx


# ===========================================================================
# on_text_insert
# ===========================================================================


class TestOnTextInsert:
    def test_insert_at_end(self, word_tokenizer: WordMockTokenizer) -> None:
        """Basic insert at end of text."""
        state = _state("foo", [_seg(0, 0, 3)])
        on_text_insert(state, 3, "!", word_tokenizer)
        assert state.document_text == "foo!"
        _assert_contiguous(state.segments, len(state.document_text))

    def test_insert_in_middle_of_segment(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Insert in middle of segment."""
        state = _state("foo bar", [_seg(0, 0, 4), _seg(1, 4, 7)])
        on_text_insert(state, 1, "x", word_tokenizer)
        assert state.document_text == "fxoo bar"
        _assert_contiguous(state.segments, len(state.document_text))

    def test_insert_preserves_segment_count(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Insert doesn't change number of segments."""
        state = _state("foo bar", [_seg(0, 0, 4), _seg(1, 4, 7)])
        on_text_insert(state, 4, "x", word_tokenizer)
        assert len(state.segments) == 2
        assert state.document_text == "foo xbar"

    def test_insert_multichar_string(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Insert multiple characters at once."""
        state = _state("foo", [_seg(0, 0, 3)])
        on_text_insert(state, 3, " bar", word_tokenizer)
        assert state.document_text == "foo bar"
        _assert_contiguous(state.segments, 7)

    def test_insert_clears_redo_stack(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Text insertion must clear generation_redo_stack."""
        state = _state("foo", [_seg(0, 0, 3)])
        state.generation_redo_stack.append(_generation_snapshot())
        on_text_insert(state, 3, "!", word_tokenizer)
        assert state.generation_redo_stack == []


# ===========================================================================
# on_text_delete
# ===========================================================================


class TestOnTextDelete:
    def test_delete_from_end(self, word_tokenizer: WordMockTokenizer) -> None:
        """Delete from end of text."""
        state = _state("foo!", [_seg(0, 0, 4)])
        on_text_delete(state, 3, 1, word_tokenizer)
        assert state.document_text == "foo"
        _assert_contiguous(state.segments, 3)

    def test_delete_from_middle(self, word_tokenizer: WordMockTokenizer) -> None:
        """Delete from middle of segment."""
        state = _state("fxoo bar", [_seg(0, 0, 5), _seg(1, 5, 8)])
        on_text_delete(state, 1, 1, word_tokenizer)
        assert state.document_text == "foo bar"
        _assert_contiguous(state.segments, 7)

    def test_delete_across_boundary(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Delete across segment boundary."""
        # "foo bar baz" → Seg0=[0,4), Seg1=[4,8), Seg2=[8,11)
        state = _state(
            "foo bar baz",
            [_seg(0, 0, 4), _seg(1, 4, 8), _seg(2, 8, 11)],
        )
        # Delete " ba" spanning boundary between Seg0 and Seg1
        on_text_delete(state, 3, 3, word_tokenizer)
        assert state.document_text == "foor baz"
        _assert_contiguous(state.segments, len(state.document_text))

    def test_delete_entire_text(self, word_tokenizer: WordMockTokenizer) -> None:
        """Delete entire text."""
        state = _state("foo", [_seg(0, 0, 3)])
        on_text_delete(state, 0, 3, word_tokenizer)
        assert state.document_text == ""
        # All segments should be zero-width
        for seg in state.segments:
            assert seg.char_start == seg.char_end

    def test_delete_preserves_segment_count(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Delete doesn't change number of segments."""
        state = _state("foo bar", [_seg(0, 0, 4), _seg(1, 4, 7)])
        on_text_delete(state, 3, 1, word_tokenizer)
        assert len(state.segments) == 2

    def test_delete_clears_redo_stack(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Text deletion must clear generation_redo_stack."""
        state = _state("foo!", [_seg(0, 0, 4)])
        state.generation_redo_stack.append(_generation_snapshot())
        on_text_delete(state, 3, 1, word_tokenizer)
        assert state.generation_redo_stack == []

    # --- Generation target re-enabling after prefill delete ---

    def test_delete_from_last_segment_reenables_generation_target(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Deleting from the last segment re-enables is_generation_target
        when target_token_count is set (prefill scenario)."""
        last = Segment(
            segment_idx=1, char_start=4, char_end=8,
            color_idx=1, features=[],
            is_generation_target=False, target_token_count=10,
        )
        state = _state("foo bar!", [_seg(0, 0, 4), last])
        on_text_delete(state, 7, 1, word_tokenizer)  # delete '!'
        assert state.segments[-1].is_generation_target is True
        assert state.segments[-1].target_token_count == 10  # preserved

    def test_delete_before_last_segment_does_not_reenable(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Deleting entirely before the last segment does NOT re-enable it."""
        last = Segment(
            segment_idx=1, char_start=4, char_end=8,
            color_idx=1, features=[],
            is_generation_target=False, target_token_count=10,
        )
        state = _state("foo bar!", [_seg(0, 0, 4), last])
        on_text_delete(state, 1, 1, word_tokenizer)  # delete 'o' from seg 0
        assert state.segments[-1].is_generation_target is False

    def test_delete_no_reenable_without_target_token_count(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """No re-enable when target_token_count is None (not from prefill)."""
        last = Segment(
            segment_idx=1, char_start=4, char_end=8,
            color_idx=1, features=[],
            is_generation_target=False, target_token_count=None,
        )
        state = _state("foo bar!", [_seg(0, 0, 4), last])
        on_text_delete(state, 7, 1, word_tokenizer)
        assert state.segments[-1].is_generation_target is False

    def test_delete_preserves_already_active_target(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """If is_generation_target is already True, deletion keeps it True."""
        last = Segment(
            segment_idx=1, char_start=4, char_end=8,
            color_idx=1, features=[],
            is_generation_target=True, target_token_count=10,
        )
        state = _state("foo bar!", [_seg(0, 0, 4), last])
        on_text_delete(state, 7, 1, word_tokenizer)
        assert state.segments[-1].is_generation_target is True
