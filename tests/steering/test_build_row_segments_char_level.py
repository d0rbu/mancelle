"""Direct unit tests for _build_row_segments (precomputed path).

Focuses on the fast path where token_start/token_end are precomputed on each
SegFeatures, bypassing the expensive char-offset computation. Companion to
test_char_level_segments.py which covers the slow (char-level overlap) path.
"""

import pytest
import torch as t

from steering.conditional.encoder_decoder.dataloader import (
    SegFeatures,
    _build_row_segments,
    _pad_segments,
)
from tests.conftest import MockTokenizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_precomputed_seg(
    n_features: int,
    token_start: int,
    token_end: int,
    seg_idx: int = 0,
) -> SegFeatures:
    """Build a SegFeatures with precomputed token boundaries.

    Uses deterministic feature ids: seg_idx * 100 + arange(n_features).
    """
    return SegFeatures(
        feature_ids=t.arange(n_features, dtype=t.int64) + seg_idx * 100,
        buckets=t.ones(n_features, dtype=t.int64),
        percentiles=t.rand(n_features),
        char_start=token_start,   # char offsets irrelevant on precomputed path
        char_end=token_end,
        token_start=token_start,
        token_end=token_end,
    )


def _dummy_doc_tokens(n: int) -> list[int]:
    """Return a list of n dummy token ids (unused by precomputed path)."""
    return list(range(2, 2 + n))


def _build_and_pad(row_docs, T, tokenizer):
    """Call _build_row_segments then _pad_segments to get padded tensors."""
    per_seg_fids, per_seg_buckets, per_seg_pcts, seg_map, seg_doc_id, _, _ = \
        _build_row_segments(row_docs, T, tokenizer)
    fids, mags, pcts, mask, seg_map_out = _pad_segments(
        per_seg_fids, per_seg_buckets, per_seg_pcts, seg_map,
    )
    return fids, mags, pcts, mask, seg_map_out, seg_doc_id


# ---------------------------------------------------------------------------
# TestBuildRowSegments
# ---------------------------------------------------------------------------


class TestBuildRowSegments:

    def test_bucket_collection_can_be_disabled_without_bucket_attribute(self):
        """Packed callers can explicitly omit the live-only bucket surface."""

        class BucketlessSegment:
            feature_ids = t.tensor([3, 7], dtype=t.int64)
            percentiles = t.tensor([0, 255], dtype=t.uint8)
            char_start = 0
            char_end = 10
            token_start = 0
            token_end = 10
            all_feature_ids = None
            all_feature_pcts = None

        row_docs = [(0, [BucketlessSegment()], _dummy_doc_tokens(10))]
        with pytest.raises(AttributeError, match="buckets"):
            _build_row_segments(
                row_docs,  # ty: ignore[invalid-argument-type]
                10,
                MockTokenizer(),
            )

        fids, buckets, pcts, seg_map, _, _, _ = _build_row_segments(
            row_docs,  # ty: ignore[invalid-argument-type]
            10,
            MockTokenizer(),
            collect_buckets=False,
            empty_percentile_dtype=t.uint8,
        )

        assert buckets == []
        assert t.equal(fids[0], t.tensor([3, 7], dtype=t.int64))
        assert t.equal(pcts[0], t.tensor([0, 255], dtype=t.uint8))
        assert t.equal(seg_map, t.zeros(10, dtype=t.int64))

    def test_single_doc_single_segment_precomputed(self):
        """1 doc, 1 segment with precomputed token_start/end. All positions map to seg 0."""
        T = 10
        n_features = 4
        seg_features = [_make_precomputed_seg(n_features, token_start=0, token_end=10)]
        doc_tokens = _dummy_doc_tokens(T)

        row_docs = [(0, seg_features, doc_tokens)]
        fids, mags, pcts, mask, seg_map, seg_doc_id = _build_and_pad(
            row_docs, T, MockTokenizer()
        )

        # Shape checks
        assert seg_map.shape == (T,)
        assert fids.shape[0] == 1, "Expected exactly 1 segment"

        # All positions must map to segment 0
        assert (seg_map == 0).all(), f"Expected all zeros, got {seg_map.tolist()}"

        # Feature data is present
        assert mask[0].any(), "Segment 0 should have active features"
        assert t.equal(fids[0, :n_features], seg_features[0].feature_ids)
        assert t.equal(mags[0, :n_features], seg_features[0].buckets)

        # seg_doc_id: single doc -> all segments belong to doc 0
        assert seg_doc_id[0].item() == 0

    def test_single_doc_multiple_segments_precomputed(self):
        """1 doc, 3 segments at token boundaries 0-10, 10-20, 20-30. Verify seg_map."""
        T = 30
        seg_features = [
            _make_precomputed_seg(3, token_start=0,  token_end=10, seg_idx=0),
            _make_precomputed_seg(3, token_start=10, token_end=20, seg_idx=1),
            _make_precomputed_seg(3, token_start=20, token_end=30, seg_idx=2),
        ]
        doc_tokens = _dummy_doc_tokens(T)

        row_docs = [(0, seg_features, doc_tokens)]
        fids, mags, pcts, mask, seg_map, seg_doc_id = _build_and_pad(
            row_docs, T, MockTokenizer()
        )

        assert seg_map.shape == (T,)
        assert fids.shape[0] == 3, "Expected 3 segments"

        # Token positions in [0, 10) -> segment 0
        assert (seg_map[0:10] == 0).all(), f"Positions 0-9 should map to seg 0: {seg_map[0:10].tolist()}"
        # Token positions in [10, 20) -> segment 1
        assert (seg_map[10:20] == 1).all(), f"Positions 10-19 should map to seg 1: {seg_map[10:20].tolist()}"
        # Token positions in [20, 30) -> segment 2
        assert (seg_map[20:30] == 2).all(), f"Positions 20-29 should map to seg 2: {seg_map[20:30].tolist()}"

        # Feature ids match what we put in
        assert t.equal(fids[0, :3], seg_features[0].feature_ids)
        assert t.equal(fids[1, :3], seg_features[1].feature_ids)
        assert t.equal(fids[2, :3], seg_features[2].feature_ids)

        # All segments belong to doc 0
        assert (seg_doc_id == 0).all()

    def test_doc_with_no_features(self):
        """Doc with None seg_features. Should produce 1 empty segment."""
        T = 20
        doc_tokens = _dummy_doc_tokens(T)

        row_docs = [(0, None, doc_tokens)]
        fids, mags, pcts, mask, seg_map, seg_doc_id = _build_and_pad(
            row_docs, T, MockTokenizer()
        )

        assert seg_map.shape == (T,)

        # Must produce exactly 1 segment (the empty placeholder)
        assert fids.shape[0] == 1, f"Expected 1 empty segment, got {fids.shape[0]}"

        # That segment has no active features
        assert not mask[0].any(), "Empty segment should have no active features"

        # All positions map to segment 0
        assert (seg_map == 0).all()

        # seg_doc_id for the single segment is doc 0
        assert seg_doc_id[0].item() == 0

    def test_multi_doc_row_precomputed(self):
        """2 docs packed in one row. seg_doc_id tracks which doc each segment belongs to."""
        T = 20
        # Doc A starts at position 0, 10 tokens, 1 segment
        seg_a = [_make_precomputed_seg(2, token_start=0, token_end=10, seg_idx=0)]
        doc_tokens_a = _dummy_doc_tokens(10)

        # Doc B starts at position 10, 10 tokens, 1 segment
        seg_b = [_make_precomputed_seg(2, token_start=0, token_end=10, seg_idx=1)]
        doc_tokens_b = _dummy_doc_tokens(10)

        row_docs = [
            (0,  seg_a, doc_tokens_a),
            (10, seg_b, doc_tokens_b),
        ]
        fids, mags, pcts, mask, seg_map, seg_doc_id = _build_and_pad(
            row_docs, T, MockTokenizer()
        )

        assert seg_map.shape == (T,)
        assert fids.shape[0] == 2, "Expected 2 segments total (1 per doc)"

        # Doc A tokens [0, 10) map to global segment 0
        assert (seg_map[0:10] == 0).all(), f"Doc A positions should be seg 0: {seg_map[0:10].tolist()}"

        # Doc B tokens [10, 20) map to global segment 1
        assert (seg_map[10:20] == 1).all(), f"Doc B positions should be seg 1: {seg_map[10:20].tolist()}"

        # seg_doc_id: segment 0 belongs to doc 0, segment 1 belongs to doc 1
        assert seg_doc_id[0].item() == 0, "Segment 0 should belong to doc 0"
        assert seg_doc_id[1].item() == 1, "Segment 1 should belong to doc 1"

        # Feature ids are distinct per doc
        assert t.equal(fids[0, :2], seg_a[0].feature_ids)
        assert t.equal(fids[1, :2], seg_b[0].feature_ids)

    def test_seg_map_values_in_range(self):
        """min >= 0 and max < n_seg for all seg_map values."""
        T = 30
        seg_features = [
            _make_precomputed_seg(5, token_start=0,  token_end=10, seg_idx=0),
            _make_precomputed_seg(3, token_start=10, token_end=20, seg_idx=1),
            _make_precomputed_seg(2, token_start=20, token_end=30, seg_idx=2),
        ]
        doc_tokens = _dummy_doc_tokens(T)

        row_docs = [(0, seg_features, doc_tokens)]
        fids, mags, pcts, mask, seg_map, seg_doc_id = _build_and_pad(
            row_docs, T, MockTokenizer()
        )

        n_seg = fids.shape[0]

        # Lower bound: all values >= 0
        assert seg_map.min().item() >= 0, f"seg_map has negative values: {seg_map.min().item()}"

        # Upper bound: all values < n_seg
        assert seg_map.max().item() < n_seg, (
            f"seg_map max {seg_map.max().item()} >= n_seg {n_seg}"
        )

    def test_precomputed_path_does_not_call_tokenizer(self):
        """The fast precomputed path does not invoke tokenizer.encode/decode."""

        class FailingTokenizer:
            """Raises on any encode/decode call to detect unexpected invocation."""

            def encode(self, *args, **kwargs):
                raise AssertionError("encode() should not be called on precomputed path")

            def decode(self, *args, **kwargs):
                raise AssertionError("decode() should not be called on precomputed path")

            def get_bos_token_id(self) -> int:
                return 0

        T = 15
        seg_features = [
            _make_precomputed_seg(2, token_start=0, token_end=8,  seg_idx=0),
            _make_precomputed_seg(2, token_start=8, token_end=15, seg_idx=1),
        ]
        doc_tokens = _dummy_doc_tokens(T)

        row_docs = [(0, seg_features, doc_tokens)]
        # Should complete without raising from FailingTokenizer
        per_seg_fids, per_seg_buckets, per_seg_pcts, seg_map, seg_doc_id, _, _ = \
            _build_row_segments(row_docs, T, FailingTokenizer())  # ty: ignore[invalid-argument-type]

        assert seg_map.shape == (T,)
        assert len(per_seg_fids) == 2

    def test_segment_clipped_to_sequence_length(self):
        """A segment whose token_end exceeds T is clipped to T."""
        T = 8
        # Segment spans [0, 20) -- larger than the sequence
        seg_features = [_make_precomputed_seg(2, token_start=0, token_end=20, seg_idx=0)]
        doc_tokens = _dummy_doc_tokens(T)

        row_docs = [(0, seg_features, doc_tokens)]
        fids, mags, pcts, mask, seg_map, seg_doc_id = _build_and_pad(
            row_docs, T, MockTokenizer()
        )

        assert seg_map.shape == (T,)
        # All T positions should map to segment 0 (clipped to n_doc_tokens)
        assert (seg_map == 0).all()

    def test_mixed_docs_some_none_features(self):
        """Row with two docs: first has None features, second has precomputed features."""
        T = 20
        doc_tokens_a = _dummy_doc_tokens(10)
        seg_b = [_make_precomputed_seg(3, token_start=0, token_end=10, seg_idx=0)]
        doc_tokens_b = _dummy_doc_tokens(10)

        row_docs = [
            (0,  None,  doc_tokens_a),
            (10, seg_b, doc_tokens_b),
        ]
        fids, mags, pcts, mask, seg_map, seg_doc_id = _build_and_pad(
            row_docs, T, MockTokenizer()
        )

        assert fids.shape[0] == 2  # 1 empty + 1 real

        # Segment 0: empty (from None doc)
        assert not mask[0].any()
        assert (seg_map[0:10] == 0).all()

        # Segment 1: has features (from second doc)
        assert mask[1].any()
        assert (seg_map[10:20] == 1).all()

        # seg_doc_id
        assert seg_doc_id[0].item() == 0
        assert seg_doc_id[1].item() == 1
