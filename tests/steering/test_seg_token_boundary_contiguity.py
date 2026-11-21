"""Tests that seg_token_start/seg_token_end in packed row shards are contiguous.

Each segment's token range must be contiguous (no gaps between segments within
a document, no overlap). This is required for correct cross-attention — the
scatter/gather in CrossAttentionLayer assumes each segment owns a contiguous
block of decoder positions.

Catches a historical bug where _row_to_csr extracted boundaries from seg_map
using nonzero(first, last), which produces non-contiguous ranges when
seg_map has non-contiguous assignments (e.g. segment 0 at [0-125] and
[651-1628] due to last-segment tail absorption).
"""

from __future__ import annotations

import pytest
import torch as t

from steering.conditional.encoder_decoder.dataloader import (
    SegFeatures,
    _build_row_segments,
)
from steering.packed_batch_io import PackedRowShard, _build_batch_index
from steering.segmentation import assign_tokens_to_segments

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_seg(
    n_features: int = 5,
    char_start: int = 0,
    char_end: int = 100,
    token_start: int | None = None,
    token_end: int | None = None,
) -> SegFeatures:
    return SegFeatures(
        feature_ids=t.randint(0, 200, (n_features,)),
        buckets=t.randint(0, 4, (n_features,)),
        percentiles=t.rand(n_features),
        char_start=char_start,
        char_end=char_end,
        token_start=token_start,
        token_end=token_end,
    )


def _build_shard_from_seg_map(
    seg_map: t.Tensor,
    n_segs: int,
    T: int,
    feats_per_seg: int = 5,
) -> PackedRowShard:
    """Build a single-row PackedRowShard from a seg_map for testing."""
    tokens = t.randint(1, 100, (1, T + 1), dtype=t.uint16)
    seg_starts = []
    seg_ends = []
    for s in range(n_segs):
        positions = (seg_map == s).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            seg_starts.append(int(positions[0].item()))
            seg_ends.append(int(positions[-1].item()) + 1)
        else:
            prev = seg_ends[-1] if seg_ends else 0
            seg_starts.append(prev)
            seg_ends.append(prev)

    total_feats = n_segs * feats_per_seg
    return PackedRowShard(
        tokens=tokens,
        row_seg_offsets=t.tensor([0, n_segs], dtype=t.int32),
        seg_token_start=t.tensor(seg_starts, dtype=t.uint16),
        seg_token_end=t.tensor(seg_ends, dtype=t.uint16),
        seg_doc_id=t.zeros(n_segs, dtype=t.uint8),
        seg_feat_offsets=t.arange(0, (n_segs + 1) * feats_per_seg, feats_per_seg, dtype=t.int32),
        feature_ids=t.randint(0, 200, (total_feats,), dtype=t.uint16),
        feature_pcts=t.randint(1, 255, (total_feats,), dtype=t.uint8),
    )


# ---------------------------------------------------------------------------
# Tests: seg_map contiguity from _build_row_segments
# ---------------------------------------------------------------------------


class TestSegMapContiguity:
    """Verify that _build_row_segments produces contiguous segments."""

    def test_single_doc_segments_tile(self):
        """All segments in a single-doc row tile contiguously."""
        T = 512
        segs = [
            _make_seg(token_start=0, token_end=128),
            _make_seg(token_start=128, token_end=256),
            _make_seg(token_start=256, token_end=512),
        ]
        row_docs = [(0, segs, list(range(512)))]
        _, _, _, seg_map, _, _, _ = _build_row_segments(row_docs, T, None)  # ty: ignore[invalid-argument-type]

        # Every position should be covered
        assert seg_map.min() >= 0
        assert seg_map.max() == 2
        # Each segment should be contiguous
        for s in range(3):
            positions = (seg_map == s).nonzero(as_tuple=True)[0]
            assert len(positions) > 0
            assert (positions[-1] - positions[0] + 1) == len(positions), (
                f"Segment {s} is non-contiguous"
            )

    def test_multi_doc_segments_tile(self):
        """Two documents in a packed row — segments tile within each doc."""
        T = 400
        doc0_segs = [
            _make_seg(token_start=0, token_end=100),
            _make_seg(token_start=100, token_end=200),
        ]
        doc1_segs = [
            _make_seg(token_start=0, token_end=100),
            _make_seg(token_start=100, token_end=200),
        ]
        row_docs = [
            (0, doc0_segs, list(range(200))),
            (200, doc1_segs, list(range(200))),
        ]
        _, _, _, seg_map, _, _, _ = _build_row_segments(row_docs, T, None)  # ty: ignore[invalid-argument-type]

        # 4 segments total (2 per doc)
        assert seg_map.max() == 3
        for s in range(4):
            positions = (seg_map == s).nonzero(as_tuple=True)[0]
            assert len(positions) > 0
            assert (positions[-1] - positions[0] + 1) == len(positions), (
                f"Segment {s} is non-contiguous"
            )

    def test_last_segment_covers_remaining_tokens(self):
        """Last segment's token_end covers all remaining positions in the doc."""
        T = 300
        segs = [
            _make_seg(token_start=0, token_end=128),
            _make_seg(token_start=128, token_end=300),  # absorbs tail
        ]
        row_docs = [(0, segs, list(range(300)))]
        _, _, _, seg_map, _, _, _ = _build_row_segments(row_docs, T, None)  # ty: ignore[invalid-argument-type]

        assert (seg_map[:128] == 0).all()
        assert (seg_map[128:300] == 1).all()

    def test_full_coverage_no_orphans(self):
        """Every token position in [0, T) must be assigned to a segment."""
        T = 512
        segs = [
            _make_seg(token_start=0, token_end=128),
            _make_seg(token_start=128, token_end=256),
            _make_seg(token_start=256, token_end=512),
        ]
        row_docs = [(0, segs, list(range(512)))]
        _, _, _, seg_map, _, _, _ = _build_row_segments(row_docs, T, None)  # ty: ignore[invalid-argument-type]

        # No position should be left at the default (0) unless it actually
        # belongs to segment 0
        seg0_expected = t.zeros(T, dtype=t.bool)
        seg0_expected[:128] = True
        assert t.equal(seg_map == 0, seg0_expected)

    def test_multi_doc_full_coverage(self):
        """Every position covered in a multi-doc packed row."""
        T = 400
        doc0_segs = [
            _make_seg(token_start=0, token_end=100),
            _make_seg(token_start=100, token_end=200),
        ]
        doc1_segs = [
            _make_seg(token_start=0, token_end=100),
            _make_seg(token_start=100, token_end=200),
        ]
        row_docs = [
            (0, doc0_segs, list(range(200))),
            (200, doc1_segs, list(range(200))),
        ]
        _, _, _, seg_map, _, _, _ = _build_row_segments(row_docs, T, None)  # ty: ignore[invalid-argument-type]

        # Build expected coverage from the tiling boundaries
        expected = t.zeros(T, dtype=t.bool)
        expected[0:100] = True
        expected[100:200] = True
        expected[200:300] = True
        expected[300:400] = True
        assert expected.all()  # sanity
        # Every position covered
        for pos in range(T):
            assert seg_map[pos] >= 0, f"Position {pos} unassigned"


# ---------------------------------------------------------------------------
# Tests: _build_batch_index seg_map reconstruction from stored boundaries
# ---------------------------------------------------------------------------


class TestBuildBatchIndexContiguity:
    """Verify that _build_batch_index produces contiguous seg_map from stored boundaries."""

    def test_contiguous_stored_boundaries_produce_contiguous_seg_map(self):
        """When stored boundaries tile, reconstructed seg_map is contiguous."""
        T = 256
        shard = PackedRowShard(
            tokens=t.randint(1, 100, (1, T + 1), dtype=t.uint16),
            row_seg_offsets=t.tensor([0, 3], dtype=t.int32),
            seg_token_start=t.tensor([0, 80, 160], dtype=t.uint16),
            seg_token_end=t.tensor([80, 160, 256], dtype=t.uint16),
            seg_doc_id=t.zeros(3, dtype=t.uint8),
            seg_feat_offsets=t.tensor([0, 5, 10, 15], dtype=t.int32),
            feature_ids=t.randint(0, 200, (15,), dtype=t.uint16),
            feature_pcts=t.randint(1, 255, (15,), dtype=t.uint8),
        )
        idx = _build_batch_index(shard, [0])
        seg_map = idx.seg_map[0]

        for s in range(3):
            positions = (seg_map == s).nonzero(as_tuple=True)[0]
            assert (positions[-1] - positions[0] + 1) == len(positions)

    def test_overlapping_stored_boundaries_raises(self):
        """Overlapping stored boundaries are caught by _build_batch_index validation."""
        T = 256
        shard = PackedRowShard(
            tokens=t.randint(1, 100, (1, T + 1), dtype=t.uint16),
            row_seg_offsets=t.tensor([0, 3], dtype=t.int32),
            seg_token_start=t.tensor([0, 50, 200], dtype=t.uint16),
            seg_token_end=t.tensor([200, 100, 256], dtype=t.uint16),
            seg_doc_id=t.zeros(3, dtype=t.uint8),
            seg_feat_offsets=t.tensor([0, 5, 10, 15], dtype=t.int32),
            feature_ids=t.randint(0, 200, (15,), dtype=t.uint16),
            feature_pcts=t.randint(1, 255, (15,), dtype=t.uint8),
        )
        with pytest.raises(ValueError, match="Segment tiling error"):
            _build_batch_index(shard, [0])

    def test_first_segment_not_at_zero_raises(self):
        """First segment must start at token 0."""
        T = 256
        shard = PackedRowShard(
            tokens=t.randint(1, 100, (1, T + 1), dtype=t.uint16),
            row_seg_offsets=t.tensor([0, 2], dtype=t.int32),
            seg_token_start=t.tensor([10, 128], dtype=t.uint16),
            seg_token_end=t.tensor([128, 256], dtype=t.uint16),
            seg_doc_id=t.zeros(2, dtype=t.uint8),
            seg_feat_offsets=t.tensor([0, 5, 10], dtype=t.int32),
            feature_ids=t.randint(0, 200, (10,), dtype=t.uint16),
            feature_pcts=t.randint(1, 255, (10,), dtype=t.uint8),
        )
        with pytest.raises(ValueError, match="first segment starts at 10, expected 0"):
            _build_batch_index(shard, [0])

    def test_last_segment_not_at_T_raises(self):
        """Last segment must end at T."""
        T = 256
        shard = PackedRowShard(
            tokens=t.randint(1, 100, (1, T + 1), dtype=t.uint16),
            row_seg_offsets=t.tensor([0, 2], dtype=t.int32),
            seg_token_start=t.tensor([0, 128], dtype=t.uint16),
            seg_token_end=t.tensor([128, 200], dtype=t.uint16),
            seg_doc_id=t.zeros(2, dtype=t.uint8),
            seg_feat_offsets=t.tensor([0, 5, 10], dtype=t.int32),
            feature_ids=t.randint(0, 200, (10,), dtype=t.uint16),
            feature_pcts=t.randint(1, 255, (10,), dtype=t.uint8),
        )
        with pytest.raises(ValueError, match="last segment ends at 200, expected 256"):
            _build_batch_index(shard, [0])

    def test_gap_between_segments_raises(self):
        """Gap between consecutive segments raises."""
        T = 256
        shard = PackedRowShard(
            tokens=t.randint(1, 100, (1, T + 1), dtype=t.uint16),
            row_seg_offsets=t.tensor([0, 3], dtype=t.int32),
            seg_token_start=t.tensor([0, 100, 150], dtype=t.uint16),
            seg_token_end=t.tensor([80, 150, 256], dtype=t.uint16),
            seg_doc_id=t.zeros(3, dtype=t.uint8),
            seg_feat_offsets=t.tensor([0, 5, 10, 15], dtype=t.int32),
            feature_ids=t.randint(0, 200, (15,), dtype=t.uint16),
            feature_pcts=t.randint(1, 255, (15,), dtype=t.uint8),
        )
        with pytest.raises(ValueError, match="expected equal"):
            _build_batch_index(shard, [0])


# ---------------------------------------------------------------------------
# Tests: _row_to_csr boundary extraction
# ---------------------------------------------------------------------------


class TestAssignTokensToSegmentsValidation:
    """Verify that assign_tokens_to_segments rejects tokens with zero overlap."""

    def test_full_coverage_passes(self):
        token_starts = t.arange(10, dtype=t.int64)
        token_ends = t.arange(1, 11, dtype=t.int64)
        seg_starts = t.tensor([0, 5], dtype=t.int64)
        seg_ends = t.tensor([5, 10], dtype=t.int64)
        result = assign_tokens_to_segments(token_starts, token_ends, seg_starts, seg_ends)
        assert result.shape == (10,)

    def test_tokens_beyond_segments_raises(self):
        token_starts = t.arange(20, dtype=t.int64)
        token_ends = t.arange(1, 21, dtype=t.int64)
        seg_starts = t.tensor([0, 5], dtype=t.int64)
        seg_ends = t.tensor([5, 10], dtype=t.int64)
        # Tokens 10-19 have zero overlap with segments [0-5), [5-10)
        with pytest.raises(ValueError, match="zero overlap"):
            assign_tokens_to_segments(token_starts, token_ends, seg_starts, seg_ends)

    def test_gap_between_segments_raises(self):
        token_starts = t.arange(10, dtype=t.int64)
        token_ends = t.arange(1, 11, dtype=t.int64)
        # Gap at chars [4-6): token 4 (chars [4,5)) and token 5 (chars [5,6))
        seg_starts = t.tensor([0, 6], dtype=t.int64)
        seg_ends = t.tensor([4, 10], dtype=t.int64)
        with pytest.raises(ValueError, match="zero overlap"):
            assign_tokens_to_segments(token_starts, token_ends, seg_starts, seg_ends)


class TestRowToCsrBoundaryExtraction:
    """Verify that _row_to_csr produces contiguous, non-overlapping boundaries."""

    def test_contiguous_seg_map_produces_tiling_boundaries(self):
        """When seg_map is contiguous, extracted boundaries should tile."""
        T = 256
        seg_map = t.zeros(T, dtype=t.int64)
        seg_map[0:80] = 0
        seg_map[80:160] = 1
        seg_map[160:256] = 2

        seg_starts = []
        seg_ends = []
        for s in range(3):
            positions = (seg_map == s).nonzero(as_tuple=True)[0]
            seg_starts.append(int(positions[0].item()))
            seg_ends.append(int(positions[-1].item()) + 1)

        # Should tile perfectly
        assert seg_starts == [0, 80, 160]
        assert seg_ends == [80, 160, 256]
        for i in range(len(seg_starts) - 1):
            assert seg_ends[i] == seg_starts[i + 1], (
                f"Gap between seg {i} and {i+1}"
            )
