"""Integration tests for BOS coordinate alignment between tokenizers.

Tests that the precompute_token_boundaries pipeline correctly aligns nanochat
token character offsets with Gemma segment character boundaries, regardless of
differences in BOS token decode length.
"""

import pytest
import torch as t

import scripts.preprocess.precompute_token_boundaries as token_boundaries
from nanochat.tokenizer import get_tokenizer
from steering.segmentation import assign_tokens_to_segments, token_to_char_offsets


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return get_tokenizer()
    except (FileNotFoundError, ValueError) as exc:
        pytest.skip(f"nanochat tokenizer cache unavailable: {exc}")


# ---------------------------------------------------------------------------
# 1. BOS decode length detection
# ---------------------------------------------------------------------------


class TestBosCharLenDetection:
    """Verify _detect_source_bos_char_len logic (without loading real shards)."""

    def test_nanochat_bos_matches_gemma(self, tokenizer):
        """With renamed special tokens, nanochat BOS should be '<bos>' (5 chars)."""
        bos_id = tokenizer.get_bos_token_id()
        decoded = tokenizer.decode([bos_id])
        assert decoded == "<bos>", f"Expected '<bos>', got {repr(decoded)}"
        assert len(decoded) == 5

    def test_bos_offset_delta_is_zero(self, tokenizer):
        """With matching BOS formats, the offset delta should be 0."""
        bos_id = tokenizer.get_bos_token_id()
        nanochat_bos_len = len(tokenizer.decode([bos_id]))
        gemma_bos_len = 5  # Gemma's <bos> is 5 chars
        assert nanochat_bos_len - gemma_bos_len == 0


# ---------------------------------------------------------------------------
# 2. Offset alignment produces correct segment assignments
# ---------------------------------------------------------------------------


class TestOffsetAlignment:
    """Verify that nanochat offsets align with Gemma segment boundaries."""

    def test_ascii_text_all_tokens_assigned(self, tokenizer):
        """Pure ASCII text: every nanochat token should be assigned to a segment."""
        text = "The quick brown fox jumps over the lazy dog. " * 10
        bos = tokenizer.get_bos_token_id()
        tokens = tokenizer.encode(text, prepend=bos)
        offsets = token_to_char_offsets(tokenizer, tokens)

        tok_starts = t.tensor(offsets[:-1], dtype=t.int64)
        tok_ends = t.tensor(offsets[1:], dtype=t.int64)

        # Create 3 segments covering the full offset range
        total = offsets[-1]
        seg_size = total // 3
        seg_starts = t.tensor([0, seg_size, 2 * seg_size], dtype=t.int64)
        seg_ends = t.tensor([seg_size, 2 * seg_size, total], dtype=t.int64)

        result = assign_tokens_to_segments(tok_starts, tok_ends, seg_starts, seg_ends)
        assert result.shape == (len(tokens),)
        assert result.min() >= 0
        assert result.max() <= 2

    def test_multibyte_text_all_tokens_assigned(self, tokenizer):
        """CJK text with byte fallback: boundaries computed via upstream expansion."""
        from scripts.preprocess.precompute_token_boundaries import _boundaries_from_offsets

        text = "Hello 世界！这是一个测试。"
        bos = tokenizer.get_bos_token_id()
        tokens = tokenizer.encode(text, prepend=bos)
        offsets = token_to_char_offsets(tokenizer, tokens)

        seg_starts = [0]
        seg_ends = [offsets[-1]]

        tok_starts, tok_ends = _boundaries_from_offsets(
            tokens, seg_starts, seg_ends, tokenizer,
            source_bos_char_len=5,
        )
        assert len(tok_starts) == 1
        assert tok_starts[0] == 0
        assert tok_ends[0] > 0

    def test_bos_token_assigned_to_first_segment(self, tokenizer):
        """BOS token should always be assigned to segment 0."""
        text = "test"
        bos = tokenizer.get_bos_token_id()
        tokens = tokenizer.encode(text, prepend=bos)
        offsets = token_to_char_offsets(tokenizer, tokens)

        tok_starts = t.tensor(offsets[:-1], dtype=t.int64)
        tok_ends = t.tensor(offsets[1:], dtype=t.int64)

        # First segment starts at 0, covers BOS
        seg_starts = t.tensor([0, offsets[-1] // 2], dtype=t.int64)
        seg_ends = t.tensor([offsets[-1] // 2, offsets[-1]], dtype=t.int64)

        result = assign_tokens_to_segments(tok_starts, tok_ends, seg_starts, seg_ends)
        assert result[0] == 0, "BOS token should be in segment 0"


# ---------------------------------------------------------------------------
# 3. Delta != 0 path (BOS length mismatch)
# ---------------------------------------------------------------------------


class TestNonZeroDelta:
    """Verify offset correction when nanochat and source BOS lengths differ."""

    def test_positive_delta_shifts_offsets_forward(self, tokenizer):
        """When source BOS is longer than nanochat, offsets shift forward."""
        from scripts.preprocess.precompute_token_boundaries import _boundaries_from_offsets

        text = "Hello world test"
        bos = tokenizer.get_bos_token_id()
        tokens = tokenizer.encode(text, prepend=bos)

        # Simulate source BOS of 10 chars (longer than nanochat's 5)
        offsets = token_to_char_offsets(tokenizer, tokens)
        total_with_source_bos = offsets[-1] + (10 - 5)  # shift by delta=+5

        seg_starts = [0]
        seg_ends = [total_with_source_bos]

        tok_starts, tok_ends = _boundaries_from_offsets(
            tokens, seg_starts, seg_ends, tokenizer,
            source_bos_char_len=10,
        )
        assert tok_starts[0] == 0
        assert tok_ends[0] > 0

    def test_delta_zero_is_identity(self, tokenizer):
        """When source BOS matches nanochat (delta=0), no shift applied."""
        from scripts.preprocess.precompute_token_boundaries import _boundaries_from_offsets

        text = "Simple test"
        bos = tokenizer.get_bos_token_id()
        tokens = tokenizer.encode(text, prepend=bos)
        offsets = token_to_char_offsets(tokenizer, tokens)

        seg_starts = [0]
        seg_ends = [offsets[-1]]

        tok_starts, tok_ends = _boundaries_from_offsets(
            tokens, seg_starts, seg_ends, tokenizer,
            source_bos_char_len=5,  # matches nanochat
        )
        assert tok_starts[0] == 0
        assert tok_ends[0] == len(tokens)


# ---------------------------------------------------------------------------
# 4. _boundaries_from_offsets integration
# ---------------------------------------------------------------------------


class TestBoundariesFromOffsets:
    """Test pre-tokenized document boundaries end-to-end."""

    def test_short_ascii_doc(self, tokenizer):
        """Short ASCII doc: boundaries should tile the token range."""
        text = "Hello world, this is a test document."
        bos = tokenizer.get_bos_token_id()
        tokens = tokenizer.encode(text, prepend=bos)
        offsets = token_to_char_offsets(tokenizer, tokens)

        # Simulate Gemma segments covering the full text
        seg_starts = [0, offsets[-1] // 2]
        seg_ends = [offsets[-1] // 2, offsets[-1]]

        tok_starts, tok_ends = token_boundaries.compute_doc_token_boundaries_from_tokens(
            tokens, seg_starts, seg_ends, tokenizer,
            source_bos_char_len=5,
        )

        assert len(tok_starts) == 2
        assert len(tok_ends) == 2
        # First segment starts at token 0
        assert tok_starts[0] == 0
        # Segments tile: end of seg 0 == start of seg 1
        assert tok_ends[0] == tok_starts[1]
        # Last segment covers remaining tokens
        assert tok_ends[1] <= len(tokens)

    def test_cjk_doc_no_crash(self, tokenizer):
        """CJK doc should not crash (byte fallback handled)."""
        text = "在新的时代，我们需要重新审视传统价值观。"
        bos = tokenizer.get_bos_token_id()
        tokens = tokenizer.encode(text, prepend=bos)
        offsets = token_to_char_offsets(tokenizer, tokens)

        seg_starts = [0]
        seg_ends = [offsets[-1]]

        tok_starts, tok_ends = token_boundaries.compute_doc_token_boundaries_from_tokens(
            tokens, seg_starts, seg_ends, tokenizer,
            source_bos_char_len=5,
        )

        assert len(tok_starts) == 1
        assert tok_starts[0] == 0
        assert tok_ends[0] > 0


# ---------------------------------------------------------------------------
# 4. Segment size validation
# ---------------------------------------------------------------------------


class TestSegmentSizeValidation:
    """Test tiered validate_segment_sizes bounds checking.

    All segments except the last are validated. The last segment is exempt
    because it is commonly a partial fill at end of document.
    Tiers: warn at 3x, error at 4x, raise at 5x. Lower bound 0.5x raises.
    """

    def test_valid_sizes_pass(self):
        from scripts.preprocess.precompute_token_boundaries import validate_segment_sizes

        # 4 segments: all non-final are 100 tokens, well within bounds
        tok_starts = [0, 100, 200, 300]
        tok_ends = [100, 200, 300, 330]
        validate_segment_sizes(tok_starts, tok_ends, source_segment_size=128)

    def test_too_small_raises(self):
        from scripts.preprocess.precompute_token_boundaries import validate_segment_sizes

        # First segment is 10 tokens, below 0.5x minimum (64)
        tok_starts = [0, 10, 110, 210]
        tok_ends = [10, 110, 210, 250]
        with pytest.raises(ValueError, match="below minimum"):
            validate_segment_sizes(tok_starts, tok_ends, source_segment_size=128)

    def test_hard_limit_raises(self):
        from scripts.preprocess.precompute_token_boundaries import validate_segment_sizes

        # First segment is 800 tokens, above 6x hard limit (768)
        tok_starts = [0, 800, 900]
        tok_ends = [800, 900, 950]
        with pytest.raises(ValueError, match="hard limit"):
            validate_segment_sizes(tok_starts, tok_ends, source_segment_size=128)

    def test_warn_tier_no_raise(self):
        from scripts.preprocess.precompute_token_boundaries import validate_segment_sizes

        # 400 tokens = 3.1x → warn tier, should not raise
        tok_starts = [0, 400, 500]
        tok_ends = [400, 500, 550]
        validate_segment_sizes(tok_starts, tok_ends, source_segment_size=128)

    def test_error_tier_no_raise(self):
        from scripts.preprocess.precompute_token_boundaries import validate_segment_sizes

        # 600 tokens = 4.7x → error tier, should not raise
        tok_starts = [0, 600, 700]
        tok_ends = [600, 700, 750]
        validate_segment_sizes(tok_starts, tok_ends, source_segment_size=128)

    def test_last_segment_exempt(self):
        from scripts.preprocess.precompute_token_boundaries import validate_segment_sizes

        # Last segment is 5 tokens — exempt (partial fill)
        tok_starts = [0, 100, 200]
        tok_ends = [100, 200, 205]
        validate_segment_sizes(tok_starts, tok_ends, source_segment_size=128)

    def test_single_segment_skipped(self):
        from scripts.preprocess.precompute_token_boundaries import validate_segment_sizes

        # Single segment is always the last → exempt
        tok_starts = [0]
        tok_ends = [500]
        validate_segment_sizes(tok_starts, tok_ends, source_segment_size=128)

    def test_empty_non_final_segments_skipped(self):
        from scripts.preprocess.precompute_token_boundaries import validate_segment_sizes

        # Empty non-final segment (beyond doc length) is fine
        tok_starts = [0, 100, 100, 200]
        tok_ends = [100, 100, 200, 300]
        validate_segment_sizes(tok_starts, tok_ends, source_segment_size=128)

    def test_max_override_allows_beyond_multiplier_limit(self):
        """max_nanochat_tokens_per_segment overrides the 6x hard limit upward."""
        from scripts.preprocess.precompute_token_boundaries import validate_segment_sizes

        # First segment is 199 tokens with source_segment_size=32:
        # default hard limit is 32*6=192 → would raise. With override 512,
        # should pass.
        tok_starts = [0, 199, 250]
        tok_ends = [199, 250, 300]
        with pytest.raises(ValueError, match="hard limit"):
            validate_segment_sizes(tok_starts, tok_ends, source_segment_size=32)
        validate_segment_sizes(
            tok_starts,
            tok_ends,
            source_segment_size=32,
            max_nanochat_tokens_per_segment=512,
        )

    def test_max_override_still_enforces_lower_bound(self):
        """Override only affects the upper hard limit, not the lower bound."""
        from scripts.preprocess.precompute_token_boundaries import validate_segment_sizes

        # First segment 10 tokens < 0.5*128 = 64 → should still raise
        tok_starts = [0, 10, 110]
        tok_ends = [10, 110, 150]
        with pytest.raises(ValueError, match="below minimum"):
            validate_segment_sizes(
                tok_starts,
                tok_ends,
                source_segment_size=128,
                max_nanochat_tokens_per_segment=4096,
            )

    def test_max_override_still_raises_above_its_own_limit(self):
        """Override sets a new hard ceiling; segments above it still raise."""
        from scripts.preprocess.precompute_token_boundaries import validate_segment_sizes

        # 600 tokens with source_segment_size=32, override=256 → above override
        tok_starts = [0, 600, 700]
        tok_ends = [600, 700, 750]
        with pytest.raises(ValueError, match="hard limit"):
            validate_segment_sizes(
                tok_starts,
                tok_ends,
                source_segment_size=32,
                max_nanochat_tokens_per_segment=256,
            )

    def test_max_override_none_preserves_default_behavior(self):
        """Passing max_nanochat_tokens_per_segment=None behaves like default."""
        from scripts.preprocess.precompute_token_boundaries import validate_segment_sizes

        # 200 tokens with source_segment_size=32 → exceeds default 192, should raise
        tok_starts = [0, 200, 250]
        tok_ends = [200, 250, 300]
        with pytest.raises(ValueError, match="hard limit"):
            validate_segment_sizes(
                tok_starts,
                tok_ends,
                source_segment_size=32,
                max_nanochat_tokens_per_segment=None,
            )

    def test_worker_failure_identifies_source_document(self, monkeypatch):
        monkeypatch.setattr(
            token_boundaries,
            "compute_doc_token_boundaries_from_tokens",
            lambda *_args, **_kwargs: (t.tensor([0, 800, 900]), t.tensor([800, 900, 950])),
        )
        monkeypatch.setattr(token_boundaries, "_worker_tokenizer", object())
        monkeypatch.setattr(token_boundaries, "_worker_source_bos_char_len", 5)
        monkeypatch.setattr(token_boundaries, "_worker_source_segment_size", 128)
        monkeypatch.setattr(
            token_boundaries, "_worker_max_nanochat_tokens_per_segment", None
        )
        monkeypatch.setattr(
            token_boundaries, "_worker_min_nanochat_tokens_per_segment", None
        )

        with pytest.raises(ValueError, match=r"doc_id=41: Segment 0/3"):
            token_boundaries._compute_doc_boundaries_from_tokens(
                (41, [1, 2, 3], [0, 1, 2], [1, 2, 3])
            )

    def test_worker_returns_four_tuple_with_doc_stats(self, monkeypatch):
        monkeypatch.setattr(
            token_boundaries,
            "compute_doc_token_boundaries_from_tokens",
            lambda *_args, **_kwargs: (t.tensor([0, 100, 200]), t.tensor([100, 200, 300])),
        )
        monkeypatch.setattr(token_boundaries, "_worker_tokenizer", object())
        monkeypatch.setattr(token_boundaries, "_worker_source_bos_char_len", 5)
        monkeypatch.setattr(token_boundaries, "_worker_source_segment_size", 128)
        monkeypatch.setattr(token_boundaries, "_worker_max_nanochat_tokens_per_segment", None)
        monkeypatch.setattr(token_boundaries, "_worker_min_nanochat_tokens_per_segment", None)

        doc_id, tok_starts, tok_ends, doc_stats = (
            token_boundaries._compute_doc_boundaries_from_tokens(
                (41, [1, 2, 3], [0, 1, 2], [1, 2, 3])
            )
        )
        assert doc_id == 41
        assert tok_starts.tolist() == [0, 100, 200]
        assert tok_ends.tolist() == [100, 200, 300]
        assert set(doc_stats["paths"]) == {"fast", "slow", "small_n"}
        assert doc_stats["compute_seconds"] >= 0.0

    def test_min_override_allows_below_default_lower_bound(self):
        """min_nanochat_tokens_per_segment lowers the 0.5x hard minimum."""
        from scripts.preprocess.precompute_token_boundaries import validate_segment_sizes

        # First segment is 1 token with source_segment_size=32:
        # default minimum is 32*0.5=16 → raises. With override 1, passes.
        tok_starts = [0, 1, 34]
        tok_ends = [1, 34, 70]
        with pytest.raises(ValueError, match="below minimum"):
            validate_segment_sizes(tok_starts, tok_ends, source_segment_size=32)
        validate_segment_sizes(
            tok_starts,
            tok_ends,
            source_segment_size=32,
            min_nanochat_tokens_per_segment=1,
        )

    def test_min_override_still_enforces_upper_limit(self):
        """Override only affects the lower bound, not the upper hard limit."""
        from scripts.preprocess.precompute_token_boundaries import validate_segment_sizes

        # 800 tokens with source_segment_size=128 → above default 768
        tok_starts = [0, 800, 900]
        tok_ends = [800, 900, 950]
        with pytest.raises(ValueError, match="hard limit"):
            validate_segment_sizes(
                tok_starts,
                tok_ends,
                source_segment_size=128,
                min_nanochat_tokens_per_segment=1,
            )

    def test_min_override_still_raises_below_its_own_floor(self):
        """Override sets a new lower floor; segments below it still raise."""
        from scripts.preprocess.precompute_token_boundaries import validate_segment_sizes

        # Floor is 2; first segment has 1 token → below floor → raise.
        tok_starts = [0, 1, 50]
        tok_ends = [1, 50, 100]
        with pytest.raises(ValueError, match="below minimum"):
            validate_segment_sizes(
                tok_starts,
                tok_ends,
                source_segment_size=32,
                min_nanochat_tokens_per_segment=2,
            )

    def test_min_override_none_preserves_default_behavior(self):
        """Passing min_nanochat_tokens_per_segment=None behaves like default."""
        from scripts.preprocess.precompute_token_boundaries import validate_segment_sizes

        # 5 tokens with source_segment_size=32 → below default 16, should raise
        tok_starts = [0, 5, 40]
        tok_ends = [5, 40, 70]
        with pytest.raises(ValueError, match="below minimum"):
            validate_segment_sizes(
                tok_starts,
                tok_ends,
                source_segment_size=32,
                min_nanochat_tokens_per_segment=None,
            )


class TestFixTrailingEmptyLastSegment:
    """Unit tests for _fix_trailing_empty_last_segment.

    The handler repairs exactly one degenerate shape: a tiny trailing source
    segment (e.g. from ``N*segment_size+k`` source tokens with small ``k``)
    where the final source segment's char width is <= a fixed absolute
    threshold and ends up empty after ``cummax``. Any other empty-segment
    pattern is left alone so ``_extract_ranges_from_monotonic`` surfaces it
    as an error.
    """

    def test_tiny_trailing_segment_1_char_is_reassigned(self):
        """Last segment of 1 char → reassign last token."""
        from scripts.preprocess.precompute_token_boundaries import (
            _fix_trailing_empty_last_segment,
        )

        assignments = t.tensor([0, 0, 1, 1, 1], dtype=t.int64)
        seg_char_starts = [0, 20, 50]
        seg_char_ends = [20, 50, 51]

        result = _fix_trailing_empty_last_segment(
            assignments.clone(), seg_char_starts, seg_char_ends
        )
        assert result.tolist() == [0, 0, 1, 1, 2]

    def test_tiny_trailing_segment_2_chars_is_reassigned(self):
        """Last segment of 2 chars → reassign (repro for shard_00307 doc 17431).

        At s32 a doc with N*32+1 Gemma tokens produces a tail source segment
        holding the last Gemma token (~2 chars). The old relative-to-
        penultimate threshold collapsed to 1 at narrow s32 segments and
        missed this case; the unified absolute
        ``CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS`` floor catches it.
        """
        from scripts.preprocess.precompute_token_boundaries import (
            _fix_trailing_empty_last_segment,
        )

        # Narrow penultimate (121 chars) mirrors the real failing doc.
        assignments = t.tensor([0, 0, 1, 1, 1], dtype=t.int64)
        seg_char_starts = [0, 20, 141]
        seg_char_ends = [20, 141, 143]

        result = _fix_trailing_empty_last_segment(
            assignments.clone(), seg_char_starts, seg_char_ends
        )
        assert result.tolist() == [0, 0, 1, 1, 2]

    def test_trailing_segment_exactly_at_threshold_is_reassigned(self):
        """Last segment of exactly threshold chars → reassign (boundary inclusive).

        Wave 3D unified the threshold to
        ``CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS`` (20). The repair is
        inclusive: widths ``<= threshold`` are reassigned.
        """
        from scripts.preprocess.precompute_token_boundaries import (
            _fix_trailing_empty_last_segment,
        )
        from steering.segmentation import CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS

        width = CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS
        assignments = t.tensor([0, 0, 1, 1, 1], dtype=t.int64)
        seg_char_starts = [0, 20, 200]
        seg_char_ends = [20, 200, 200 + width]

        result = _fix_trailing_empty_last_segment(
            assignments.clone(), seg_char_starts, seg_char_ends
        )
        assert result.tolist() == [0, 0, 1, 1, 2]

    def test_trailing_segment_just_above_threshold_is_left_alone(self):
        """Last segment one char above the unified threshold → leave alone.

        Wave 3D unified the tiny-trailing-segment floor to
        ``CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS`` (20). Widths strictly
        greater than the threshold are NOT repaired and must surface as an
        anomaly downstream.
        """
        from scripts.preprocess.precompute_token_boundaries import (
            _fix_trailing_empty_last_segment,
        )
        from steering.segmentation import CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS

        width = CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS + 1
        assignments = t.tensor([0, 0, 1, 1, 1], dtype=t.int64)
        seg_char_starts = [0, 20, 200]
        seg_char_ends = [20, 200, 200 + width]

        result = _fix_trailing_empty_last_segment(
            assignments.clone(), seg_char_starts, seg_char_ends
        )
        assert result.tolist() == [0, 0, 1, 1, 1]

    def test_non_tiny_last_segment_missing_is_left_alone(self):
        """Missing last segment with non-tiny width is NOT repaired.

        This anomaly indicates a real problem upstream (e.g. a long
        nanochat token straddling the penultimate/last boundary at small
        source segment sizes); it should bubble up to the extractor and
        raise rather than being silently papered over.
        """
        from scripts.preprocess.precompute_token_boundaries import (
            _fix_trailing_empty_last_segment,
        )

        # Last source segment is 50 chars wide (above 16-char floor) and empty.
        assignments = t.tensor([0, 0, 1, 1, 1], dtype=t.int64)
        seg_char_starts = [0, 20, 50]
        seg_char_ends = [20, 50, 100]

        result = _fix_trailing_empty_last_segment(
            assignments.clone(), seg_char_starts, seg_char_ends
        )
        assert result.tolist() == [0, 0, 1, 1, 1]

    def test_middle_missing_segment_is_left_alone(self):
        """Missing middle segment is NOT repaired — leave it to raise."""
        from scripts.preprocess.precompute_token_boundaries import (
            _fix_trailing_empty_last_segment,
        )

        # Segment 1 missing (assignments jump 0 → 2).
        assignments = t.tensor([0, 0, 2, 2, 2], dtype=t.int64)
        seg_char_starts = [0, 20, 35]
        seg_char_ends = [20, 35, 80]

        result = _fix_trailing_empty_last_segment(
            assignments.clone(), seg_char_starts, seg_char_ends
        )
        assert result.tolist() == [0, 0, 2, 2, 2]

    def test_all_segments_already_covered_is_noop(self):
        """No-op: assignments already cover all segments."""
        from scripts.preprocess.precompute_token_boundaries import (
            _fix_trailing_empty_last_segment,
        )

        assignments = t.tensor([0, 0, 1, 1, 2], dtype=t.int64)
        seg_char_starts = [0, 20, 50]
        seg_char_ends = [20, 50, 100]

        result = _fix_trailing_empty_last_segment(
            assignments.clone(), seg_char_starts, seg_char_ends
        )
        assert result.tolist() == [0, 0, 1, 1, 2]

    def test_single_segment_is_noop(self):
        """No-op: n_segs < 2."""
        from scripts.preprocess.precompute_token_boundaries import (
            _fix_trailing_empty_last_segment,
        )

        assignments = t.tensor([0, 0, 0], dtype=t.int64)
        seg_char_starts = [0]
        seg_char_ends = [30]

        result = _fix_trailing_empty_last_segment(
            assignments.clone(), seg_char_starts, seg_char_ends
        )
        assert result.tolist() == [0, 0, 0]

    def test_empty_assignments_is_noop(self):
        """No-op: empty assignments tensor."""
        from scripts.preprocess.precompute_token_boundaries import (
            _fix_trailing_empty_last_segment,
        )

        assignments = t.tensor([], dtype=t.int64)
        seg_char_starts = [0, 20]
        seg_char_ends = [20, 50]

        result = _fix_trailing_empty_last_segment(
            assignments.clone(), seg_char_starts, seg_char_ends
        )
        assert result.tolist() == []


# ---------------------------------------------------------------------------
# 5. Special token migration
# ---------------------------------------------------------------------------


class TestSpecialTokenMigration:
    """Test that old tokenizer pickles are transparently migrated."""

    def test_tokenizer_loads_with_new_format(self, tokenizer):
        """Tokenizer should have <bos> not <|bos|>."""
        bos_id = tokenizer.get_bos_token_id()
        decoded = tokenizer.decode([bos_id])
        assert decoded == "<bos>"
        assert "<|" not in decoded

    def test_special_tokens_consistent(self, tokenizer):
        """All special tokens should use the new <name> format."""
        for name in ["<bos>", "<user_start>", "<user_end>",
                      "<assistant_start>", "<assistant_end>"]:
            tid = tokenizer.encode_special(name)
            assert tid is not None, f"Special token {name} not found"
            decoded = tokenizer.decode([tid])
            assert decoded == name
