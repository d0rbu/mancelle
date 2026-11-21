"""Tests for character-level segment assignment in the encoder-decoder dataloader.

Covers:
- Property-based tests (hypothesis) for assign_tokens_to_segments
- _build_row_segments with multi-byte characters (mock tokenizer)
- Empty segment handling (cummax skips)
- Boundary conditions (equal overlap ties, single-token docs, BOS-only docs)
- BOS token assignment in char-level mode
"""

import hypothesis.strategies as st
import torch as t
from hypothesis import HealthCheck, assume, given, settings

from steering.conditional.encoder_decoder.dataloader import (
    SegFeatures,
    _build_row_segments,
    _pad_segments,
)
from steering.segmentation import assign_tokens_to_segments

# ---------------------------------------------------------------------------
# Mock tokenizer
# ---------------------------------------------------------------------------

class CharLevelMockTokenizer:
    """Each non-BOS token is one ASCII character. BOS decodes to '<bos>'."""

    BOS_ID = 2

    def get_bos_token_id(self):
        return self.BOS_ID

    def encode(self, text, prepend=None):
        tokens = [ord(c) for c in text]
        if prepend is not None:
            tokens = [prepend] + tokens
        return tokens

    def decode(self, token_ids):
        parts = []
        for tid in token_ids:
            if tid == self.BOS_ID:
                parts.append("<bos>")
            else:
                parts.append(chr(tid))
        return "".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_seg_features_char(n_segs, seg_char_size, n_features=2, char_offset=0):
    """Build SegFeatures list with explicit char ranges.

    Each segment spans seg_char_size characters starting at char_offset.
    """
    segs = []
    for i in range(n_segs):
        fids = t.arange(n_features, dtype=t.int64) + i * n_features
        buckets = t.ones(n_features, dtype=t.int64)
        pcts = t.zeros(n_features, dtype=t.float32)
        c_start = char_offset + i * seg_char_size
        c_end = char_offset + (i + 1) * seg_char_size
        segs.append(SegFeatures(
            feature_ids=fids, buckets=buckets, percentiles=pcts,
            char_start=c_start, char_end=c_end,
        ))
    return segs


def _sf(fids: t.Tensor, buckets: t.Tensor, char_start: int, char_end: int) -> SegFeatures:
    """Shorthand to build a SegFeatures with zero percentiles."""
    return SegFeatures(
        feature_ids=fids,
        buckets=buckets,
        percentiles=t.zeros_like(fids, dtype=t.float32),
        char_start=char_start,
        char_end=char_end,
    )


def _build_and_pad(row_docs, T, tokenizer):
    """Call _build_row_segments then _pad_segments to get padded tensors."""
    per_seg_fids, per_seg_bkts, per_seg_pcts, seg_map, seg_doc_id, _, _ = \
        _build_row_segments(row_docs, T, tokenizer)
    fids, mags, pcts, mask, seg_map_out = _pad_segments(
        per_seg_fids, per_seg_bkts, per_seg_pcts, seg_map,
    )
    return fids, mags, pcts, mask, seg_map_out, seg_doc_id


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

@st.composite
def _non_overlapping_ranges(draw, min_count=1, max_count=20, max_total=200):
    """Generate a list of non-overlapping, contiguous [start, end) ranges."""
    n = draw(st.integers(min_value=min_count, max_value=max_count))
    widths = draw(
        st.lists(st.integers(min_value=1, max_value=max_total // max(n, 1)),
                 min_size=n, max_size=n)
    )
    starts, ends = [], []
    pos = 0
    for w in widths:
        starts.append(pos)
        ends.append(pos + w)
        pos += w
    return t.tensor(starts, dtype=t.int64), t.tensor(ends, dtype=t.int64)


@st.composite
def _tiling_ranges(draw, total_len, min_count=1, max_count=20):
    """Generate ranges that tile exactly [0, total_len)."""
    n = draw(st.integers(min_value=min_count, max_value=min(max_count, total_len)))
    # Partition total_len into n positive integers
    cuts = sorted(draw(
        st.lists(
            st.integers(min_value=1, max_value=total_len - 1),
            min_size=n - 1, max_size=n - 1,
            unique=True,
        )
    ))
    boundaries = [0] + cuts + [total_len]
    starts = t.tensor(boundaries[:-1], dtype=t.int64)
    ends = t.tensor(boundaries[1:], dtype=t.int64)
    return starts, ends


# ---------------------------------------------------------------------------
# 1. Property-based tests for assign_tokens_to_segments
# ---------------------------------------------------------------------------

class TestAssignTokensPropertyBased:

    @given(
        tok_data=_non_overlapping_ranges(min_count=1, max_count=15),
        seg_data=_non_overlapping_ranges(min_count=1, max_count=10),
    )
    @settings(
        max_examples=200,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.differing_executors],
    )
    def test_output_length_equals_num_tokens(self, tok_data, seg_data):
        """Output has one assignment per token."""
        tok_starts, tok_ends = tok_data
        seg_starts, seg_ends = seg_data
        try:
            result = assign_tokens_to_segments(tok_starts, tok_ends, seg_starts, seg_ends)
        except ValueError:
            assume(False)  # skip examples with zero-overlap tokens
            return
        assert len(result) == len(tok_starts)

    @given(
        tok_data=_non_overlapping_ranges(min_count=1, max_count=15),
        seg_data=_non_overlapping_ranges(min_count=1, max_count=10),
    )
    @settings(
        max_examples=200,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.differing_executors],
    )
    def test_output_values_in_range(self, tok_data, seg_data):
        """All output values are valid segment indices."""
        tok_starts, tok_ends = tok_data
        seg_starts, seg_ends = seg_data
        try:
            result = assign_tokens_to_segments(tok_starts, tok_ends, seg_starts, seg_ends)
        except ValueError:
            assume(False)
            return
        n_segs = len(seg_starts)
        assert (result >= 0).all()
        assert (result < n_segs).all()

    @given(
        tok_data=_non_overlapping_ranges(min_count=1, max_count=15),
        seg_data=_non_overlapping_ranges(min_count=1, max_count=10),
    )
    @settings(
        max_examples=200,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.differing_executors],
    )
    def test_output_dtype_is_int64(self, tok_data, seg_data):
        """Output dtype is always int64."""
        tok_starts, tok_ends = tok_data
        seg_starts, seg_ends = seg_data
        try:
            result = assign_tokens_to_segments(tok_starts, tok_ends, seg_starts, seg_ends)
        except ValueError:
            assume(False)
            return
        assert result.dtype == t.int64

    def test_perfectly_aligned_tokens_segments(self):
        """When tokens and segments are identical ranges, assignment is 1:1."""
        starts = t.tensor([0, 5, 10, 15], dtype=t.int64)
        ends = t.tensor([5, 10, 15, 20], dtype=t.int64)
        result = assign_tokens_to_segments(starts, ends, starts, ends)
        assert result.tolist() == [0, 1, 2, 3]

    def test_token_entirely_within_one_segment(self):
        """A token entirely inside a segment always maps to that segment."""
        # Segments: [0,10), [10,20), [20,30)
        seg_starts = t.tensor([0, 10, 20], dtype=t.int64)
        seg_ends = t.tensor([10, 20, 30], dtype=t.int64)

        # Token [12, 18) is entirely within segment 1
        tok_starts = t.tensor([12], dtype=t.int64)
        tok_ends = t.tensor([18], dtype=t.int64)
        result = assign_tokens_to_segments(tok_starts, tok_ends, seg_starts, seg_ends)
        assert result.tolist() == [1]

    @given(
        total_len=st.integers(min_value=2, max_value=100),
        data=st.data(),
    )
    @settings(
        max_examples=200,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.differing_executors],
    )
    def test_tiling_ranges_monotonic(self, total_len, data):
        """When tokens and segments both tile [0, L), assignment is non-decreasing."""
        tok_starts, tok_ends = data.draw(_tiling_ranges(total_len, min_count=1, max_count=15))
        seg_starts, seg_ends = data.draw(_tiling_ranges(total_len, min_count=1, max_count=10))
        result = assign_tokens_to_segments(tok_starts, tok_ends, seg_starts, seg_ends)
        for i in range(1, len(result)):
            assert result[i] >= result[i - 1], (
                f"Not monotonic at {i}: {result[i - 1].item()} -> {result[i].item()}"
            )


# ---------------------------------------------------------------------------
# 2. _build_row_segments with multi-byte characters
# ---------------------------------------------------------------------------

class TestBuildRowSegmentsCharLevelMultibyte:

    def test_bos_plus_ascii_tokens(self):
        """Basic case: BOS + ASCII tokens, segments cover full char range."""
        tokenizer = CharLevelMockTokenizer()
        # Text "abc" -> tokens [BOS, ord('a'), ord('b'), ord('c')]
        # BOS decodes to "<bos>" (5 chars), 'a'->1 char, 'b'->1 char, 'c'->1 char
        # Full decode = "<bos>abc" (8 chars)
        # char offsets: [0, 5, 6, 7, 8]
        text = "abc"
        doc_tokens = tokenizer.encode(text, prepend=tokenizer.BOS_ID)
        assert doc_tokens == [2, ord('a'), ord('b'), ord('c')]

        # 2 extraction segments covering [0,7) and [7,8)
        # First segment includes BOS chars to ensure full coverage
        seg_features = [
            _sf(t.tensor([10], dtype=t.int64), t.tensor([1], dtype=t.int64), 0, 7),
            _sf(t.tensor([20], dtype=t.int64), t.tensor([1], dtype=t.int64), 7, 8),
        ]

        row_docs = [(0, seg_features, doc_tokens)]
        T = 4  # 4 tokens total
        fids, mags, _, mask, seg_map, _ = _build_and_pad(row_docs, T, tokenizer)

        assert seg_map.shape == (T,)
        # BOS at chars [0,5) -> seg 0 [0,7) overlap=5 -> seg 0
        # Token 'a' at chars [5,6) -> seg 0 [0,7) overlap=1 -> seg 0
        # Token 'b' at chars [6,7) -> seg 0 [0,7) overlap=1 -> seg 0
        # Token 'c' at chars [7,8) -> seg 1 [7,8) overlap=1 -> seg 1
        assert seg_map[0].item() == 0  # BOS
        assert seg_map[1].item() == 0  # 'a'
        assert seg_map[2].item() == 0  # 'b'
        assert seg_map[3].item() == 1  # 'c'

    def test_multibyte_emoji_token(self):
        """Token that decodes to a multi-char string via a multi-byte character."""
        # Custom tokenizer where token 999 decodes to a 2-char emoji string
        class EmojiTokenizer:
            BOS_ID = 2

            def get_bos_token_id(self):
                return self.BOS_ID

            def decode(self, token_ids):
                parts = []
                for tid in token_ids:
                    if tid == self.BOS_ID:
                        parts.append("<bos>")
                    elif tid == 999:
                        parts.append("\U0001f30d\U0001f389")  # 2 chars
                    else:
                        parts.append(chr(tid))
                return "".join(parts)

        tokenizer = EmojiTokenizer()
        # doc_tokens = [BOS, 999, ord('x')]
        # decode = "<bos>" + 2 emoji chars + "x" = 8 chars
        # char offsets: [0, 5, 7, 8]
        doc_tokens = [2, 999, ord('x')]

        # Single segment covering the full char range: [0, 8)
        seg_features = [
            _sf(t.tensor([42], dtype=t.int64), t.tensor([1], dtype=t.int64), 0, 8),
        ]

        row_docs = [(0, seg_features, doc_tokens)]
        T = 3
        fids, mags, _, mask, seg_map, _ = _build_and_pad(row_docs, T, tokenizer)

        assert seg_map.shape == (T,)
        # All tokens map to the only segment (segment 0)
        assert (seg_map == 0).all()
        assert mask[0, 0].item() is True  # feature 42 is present

    def test_nonuniform_char_widths_correct_seg_map(self):
        """Tokens with varying char widths get correct segment assignments."""
        # Tokenizer: BOS -> "<b>" (3 chars), token 65 -> "AA" (2 chars),
        # token 66 -> "B" (1 char), token 67 -> "CCC" (3 chars)
        class VarWidthTokenizer:
            BOS_ID = 2

            def get_bos_token_id(self):
                return self.BOS_ID

            def decode(self, token_ids):
                mapping = {2: "<b>", 65: "AA", 66: "B", 67: "CCC"}
                return "".join(mapping[tid] for tid in token_ids)

        tokenizer = VarWidthTokenizer()
        # doc_tokens = [BOS, 65, 66, 67]
        # decode = "<b>AABCCC" -> 9 chars
        # char offsets: [0, 3, 5, 6, 9]
        doc_tokens = [2, 65, 66, 67]

        # Two segments: [0,6) covers BOS+"AAB", [6,9) covers "CCC"
        seg_features = [
            _sf(t.tensor([100], dtype=t.int64), t.tensor([1], dtype=t.int64), 0, 6),
            _sf(t.tensor([200], dtype=t.int64), t.tensor([1], dtype=t.int64), 6, 9),
        ]

        row_docs = [(0, seg_features, doc_tokens)]
        T = 4
        fids, mags, _, mask, seg_map, _ = _build_and_pad(row_docs, T, tokenizer)

        # BOS: chars [0,3) -> no overlap with [3,6) or [6,9) -> argmax=0 -> seg 0
        # token 65 "AA": chars [3,5) -> seg0 [3,6) overlap=2 -> seg 0
        # token 66 "B": chars [5,6) -> seg0 [3,6) overlap=1 -> seg 0
        # token 67 "CCC": chars [6,9) -> seg1 [6,9) overlap=3 -> seg 1
        assert seg_map.tolist() == [0, 0, 0, 1]


# ---------------------------------------------------------------------------
# 3. Empty segment handling
# ---------------------------------------------------------------------------

class TestEmptySegmentHandling:

    def test_middle_segment_has_no_tokens(self):
        """A segment with zero assigned tokens does not crash; cummax skips it."""
        tokenizer = CharLevelMockTokenizer()
        # Text "ab" -> tokens [BOS, ord('a'), ord('b')]
        # decode = "<bos>ab" -> 7 chars
        # char offsets: [0, 5, 6, 7]
        doc_tokens = tokenizer.encode("ab", prepend=tokenizer.BOS_ID)

        # 3 extraction segments: [0,6), [6,6) (empty range!), [6,7)
        # Middle segment is impossible to overlap with.
        seg_features = [
            _sf(t.tensor([10], dtype=t.int64), t.tensor([1], dtype=t.int64), 0, 6),
            _sf(t.tensor([20], dtype=t.int64), t.tensor([1], dtype=t.int64), 6, 6),
            _sf(t.tensor([30], dtype=t.int64), t.tensor([1], dtype=t.int64), 6, 7),
        ]

        row_docs = [(0, seg_features, doc_tokens)]
        T = 3
        fids, mags, _, mask, seg_map, _ = _build_and_pad(row_docs, T, tokenizer)

        # BOS -> zero overlap -> argmax=0 -> seg 0
        # 'a' at [5,6) -> seg0 [5,6) overlap=1 -> seg 0
        # 'b' at [6,7) -> seg0 overlap=0, seg1 overlap=0, seg2 [6,7) overlap=1 -> seg 2
        # After cummax: [0, 0, 2]
        # n_segs_used = 2+1 = 3 (all three segments used in the output)
        assert seg_map.shape == (T,)
        assert seg_map[0].item() == 0
        assert seg_map[1].item() == 0
        assert seg_map[2].item() == 2

    def test_no_features_single_empty_segment(self):
        """Document with seg_features=None gets a single empty segment."""
        tokenizer = CharLevelMockTokenizer()
        doc_tokens = tokenizer.encode("abc", prepend=tokenizer.BOS_ID)

        row_docs = [(0, None, doc_tokens)]
        T = 4
        fids, mags, _, mask, seg_map, _ = _build_and_pad(row_docs, T, tokenizer)

        assert (seg_map == 0).all()
        assert not mask.any()  # no features


# ---------------------------------------------------------------------------
# 4. Boundary conditions
# ---------------------------------------------------------------------------

class TestBoundaryConditions:

    def test_equal_overlap_picks_first_segment(self):
        """Token with equal overlap to two segments -> argmax picks the first."""
        # Token [2, 6) overlaps seg0 [0,4) by 2 chars and seg1 [4,8) by 2 chars
        result = assign_tokens_to_segments(
            token_char_starts=t.tensor([2]),
            token_char_ends=t.tensor([6]),
            seg_char_starts=t.tensor([0, 4]),
            seg_char_ends=t.tensor([4, 8]),
        )
        assert result.tolist() == [0]  # tie goes to first

    def test_single_token_document(self):
        """Document with only one non-BOS token, char-level assignment."""
        tokenizer = CharLevelMockTokenizer()
        doc_tokens = tokenizer.encode("x", prepend=tokenizer.BOS_ID)
        assert doc_tokens == [2, ord('x')]

        # Single segment covering BOS + text char: [0, 6)
        seg_features = [
            _sf(t.tensor([7], dtype=t.int64), t.tensor([1], dtype=t.int64), 0, 6),
        ]

        row_docs = [(0, seg_features, doc_tokens)]
        T = 2
        fids, mags, _, mask, seg_map, _ = _build_and_pad(row_docs, T, tokenizer)

        assert seg_map.shape == (T,)
        assert seg_map.tolist() == [0, 0]

    def test_bos_only_document(self):
        """Document with only the BOS token (no text content)."""
        tokenizer = CharLevelMockTokenizer()
        doc_tokens = [tokenizer.BOS_ID]

        # One segment covering BOS + some char range
        seg_features = [
            _sf(t.tensor([99], dtype=t.int64), t.tensor([1], dtype=t.int64), 0, 10),
        ]

        row_docs = [(0, seg_features, doc_tokens)]
        T = 1
        fids, mags, _, mask, seg_map, _ = _build_and_pad(row_docs, T, tokenizer)

        assert seg_map.shape == (T,)
        assert seg_map[0].item() == 0

    def test_cropped_document_trailing_segments(self):
        """Cropped document: extraction segments extend beyond placed tokens."""
        tokenizer = CharLevelMockTokenizer()
        # Full doc would be "abcdef" but we only place first 3 text tokens
        full_tokens = tokenizer.encode("abcdef", prepend=tokenizer.BOS_ID)
        # Place only BOS + "abc" = 4 tokens
        placed_tokens = full_tokens[:4]

        # 3 extraction segments, last one covers chars beyond what's placed
        seg_features = [
            _sf(t.tensor([10], dtype=t.int64), t.tensor([1], dtype=t.int64), 0, 7),   # BOS+"ab"
            _sf(t.tensor([20], dtype=t.int64), t.tensor([1], dtype=t.int64), 7, 9),   # "cd"
            _sf(t.tensor([30], dtype=t.int64), t.tensor([1], dtype=t.int64), 9, 11),  # "ef" (beyond placed)
        ]

        row_docs = [(0, seg_features, placed_tokens)]
        T = 4
        fids, mags, _, mask, seg_map, _ = _build_and_pad(row_docs, T, tokenizer)

        assert seg_map.shape == (T,)
        # Only segments covering the placed tokens should be used
        # BOS -> seg 0, 'a' -> seg 0, 'b' -> seg 0, 'c' -> seg 1
        # 'c' at chars [7,8): seg0 [5,7) overlap=0, seg1 [7,9) overlap=1 -> seg 1
        assert seg_map.max().item() <= 1  # should not reference seg 2

    def test_two_docs_packed_char_level(self):
        """Two documents packed into one row with char-level assignment."""
        tokenizer = CharLevelMockTokenizer()

        doc1_tokens = tokenizer.encode("ab", prepend=tokenizer.BOS_ID)  # [BOS, 'a', 'b']
        doc2_tokens = tokenizer.encode("cd", prepend=tokenizer.BOS_ID)  # [BOS, 'c', 'd']

        seg_features_1 = [
            _sf(t.tensor([10], dtype=t.int64), t.tensor([1], dtype=t.int64), 0, 7),
        ]
        seg_features_2 = [
            _sf(t.tensor([20], dtype=t.int64), t.tensor([1], dtype=t.int64), 0, 7),
        ]

        row_docs = [
            (0, seg_features_1, doc1_tokens),
            (3, seg_features_2, doc2_tokens),
        ]
        T = 6
        fids, mags, _, mask, seg_map, _ = _build_and_pad(row_docs, T, tokenizer)

        assert seg_map.shape == (T,)
        # Doc 1 tokens all go to global seg 0
        assert seg_map[0].item() == 0
        assert seg_map[1].item() == 0
        assert seg_map[2].item() == 0
        # Doc 2 tokens all go to global seg 1
        assert seg_map[3].item() == 1
        assert seg_map[4].item() == 1
        assert seg_map[5].item() == 1


# ---------------------------------------------------------------------------
# 5. BOS token assignment in char-level mode
# ---------------------------------------------------------------------------

class TestBOSTokenAssignment:

    def test_bos_zero_overlap_maps_to_segment_zero(self):
        """BOS token chars are before all segments -> argmax returns 0."""
        tokenizer = CharLevelMockTokenizer()
        doc_tokens = tokenizer.encode("ab", prepend=tokenizer.BOS_ID)
        # decode = "<bos>ab" -> 7 chars
        # BOS occupies [0,5), text chars at [5,6) and [6,7)

        # First segment starts at char 0 to include BOS
        seg_features = [
            _sf(t.tensor([1], dtype=t.int64), t.tensor([1], dtype=t.int64), 0, 6),
            _sf(t.tensor([2], dtype=t.int64), t.tensor([1], dtype=t.int64), 6, 7),
        ]

        row_docs = [(0, seg_features, doc_tokens)]
        T = 3
        _, _, _, seg_map, _, _, _ = _build_row_segments(row_docs, T, tokenizer)  # ty: ignore[invalid-argument-type]

        # BOS: chars [0,5), no overlap with any seg -> argmax returns 0
        assert seg_map[0].item() == 0

    def test_bos_stays_at_zero_after_cummax(self):
        """cummax does not change BOS assignment since it is already 0."""
        tokenizer = CharLevelMockTokenizer()
        doc_tokens = tokenizer.encode("abc", prepend=tokenizer.BOS_ID)

        seg_features = [
            _sf(t.tensor([1], dtype=t.int64), t.tensor([1], dtype=t.int64), 0, 7),
            _sf(t.tensor([2], dtype=t.int64), t.tensor([1], dtype=t.int64), 7, 8),
        ]

        row_docs = [(0, seg_features, doc_tokens)]
        T = 4
        _, _, _, seg_map, _, _, _ = _build_row_segments(row_docs, T, tokenizer)  # ty: ignore[invalid-argument-type]

        # BOS at [0,5) -> seg 0, 'a' at [5,6) -> seg 0, 'b' at [6,7) -> seg 0, 'c' at [7,8) -> seg 1
        assert seg_map[0].item() == 0

    def test_cummax_fixes_regression_after_last_segment(self):
        """Tokens beyond extraction segments get assigned to last segment via cummax."""
        tokenizer = CharLevelMockTokenizer()
        # Text "abcd" -> BOS + 4 text tokens
        doc_tokens = tokenizer.encode("abcd", prepend=tokenizer.BOS_ID)

        # One extraction segment covering BOS + first 2 text chars: [0,7)
        # Single segment covering full decode range: [0,9)
        seg_features = [
            _sf(t.tensor([1], dtype=t.int64), t.tensor([1], dtype=t.int64), 0, 9),
        ]

        row_docs = [(0, seg_features, doc_tokens)]
        T = 5
        _, _, _, seg_map, _, _, _ = _build_row_segments(row_docs, T, tokenizer)  # ty: ignore[invalid-argument-type]

        # All tokens map to seg 0 since there's only one segment.
        assert (seg_map == 0).all()

    def test_cummax_prevents_decrease_with_multiple_segments(self):
        """cummax prevents assignment from decreasing when later tokens
        cross segment boundaries."""
        # Manually test assign_tokens_to_segments + cummax
        # Tokens: [0,3), [3,6), [6,9), [9,12)
        # Segs: [0,4), [4,12)
        # Token [0,3): seg0 overlap=3 -> 0
        # Token [3,6): seg0 overlap=1, seg1 overlap=2 -> 1
        # Token [6,9): seg0 overlap=0, seg1 overlap=3 -> 1
        # Token [9,12): seg0 overlap=0, seg1 overlap=3 -> 1
        tok_starts = t.tensor([0, 3, 6, 9], dtype=t.int64)
        tok_ends = t.tensor([3, 6, 9, 12], dtype=t.int64)
        seg_starts = t.tensor([0, 4], dtype=t.int64)
        seg_ends = t.tensor([4, 12], dtype=t.int64)

        raw = assign_tokens_to_segments(tok_starts, tok_ends, seg_starts, seg_ends)
        assert raw.tolist() == [0, 1, 1, 1]

        # cummax keeps the non-decreasing sequence intact
        fixed = t.cummax(raw, dim=0).values
        assert fixed.tolist() == [0, 1, 1, 1]


# ---------------------------------------------------------------------------
# 6. Padded output shape / mask correctness
# ---------------------------------------------------------------------------

class TestPaddedOutputs:

    def test_feature_ids_and_mask_shape(self):
        """Padded outputs have correct shapes and mask content."""
        tokenizer = CharLevelMockTokenizer()
        doc_tokens = tokenizer.encode("abc", prepend=tokenizer.BOS_ID)

        seg_features = [
            _sf(t.tensor([10, 11], dtype=t.int64), t.tensor([1, 2], dtype=t.int64), 0, 7),
            _sf(t.tensor([20], dtype=t.int64), t.tensor([3], dtype=t.int64), 7, 8),
        ]

        row_docs = [(0, seg_features, doc_tokens)]
        T = 4
        fids, mags, _, mask, seg_map, _ = _build_and_pad(row_docs, T, tokenizer)

        # 2 segments used; F_max = max(2, 1) = 2
        assert fids.shape == (2, 2)
        assert mags.shape == (2, 2)
        assert mask.shape == (2, 2)

        # Segment 0 has 2 features
        assert mask[0].tolist() == [True, True]
        assert fids[0].tolist() == [10, 11]
        assert mags[0].tolist() == [1, 2]

        # Segment 1 has 1 feature, padded
        assert mask[1].tolist() == [True, False]
        assert fids[1, 0].item() == 20
        assert mags[1, 0].item() == 3

    def test_seg_map_values_valid_indices(self):
        """seg_map values are valid indices into the segment dimension."""
        tokenizer = CharLevelMockTokenizer()
        doc_tokens = tokenizer.encode("abcde", prepend=tokenizer.BOS_ID)

        seg_features = [
            _sf(t.tensor([1], dtype=t.int64), t.tensor([1], dtype=t.int64), 0, 8),
            _sf(t.tensor([2], dtype=t.int64), t.tensor([1], dtype=t.int64), 8, 10),
        ]

        row_docs = [(0, seg_features, doc_tokens)]
        T = 6
        fids, mags, _, mask, seg_map, _ = _build_and_pad(row_docs, T, tokenizer)

        n_segs = fids.shape[0]
        assert (seg_map >= 0).all()
        assert (seg_map < n_segs).all()
