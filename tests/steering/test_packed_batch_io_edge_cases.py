"""Edge case characterization tests for _build_batch_index and reconstruct_batch.

Covers boundary conditions:
- All segments with zero features
- Single-row batches
- Variable feature counts across segments
- targets/inputs shift relationship
- seg_map value ranges
- seg_doc_id padding sentinel (-1)
- Percentile normalization range
"""

from __future__ import annotations

import torch as t

from steering.packed_batch_io import (
    PackedRowShard,
    SelectionShard,
    _build_batch_index,
    reconstruct_batch,
)


def _make_identity_selection(shard: PackedRowShard) -> SelectionShard:
    """Create a SelectionShard that selects all features from the base shard."""
    return SelectionShard(
        sel_offsets=shard.seg_feat_offsets.to(t.int32),
        sel_fids=shard.feature_ids,
        sel_buckets=t.ones(len(shard.feature_ids), dtype=t.uint8),
        sel_pcts=shard.feature_pcts,
        neg_offsets=t.zeros(shard.total_segments + 1, dtype=t.int32),
        neg_fids=t.tensor([], dtype=t.uint16),
    )


# ---------------------------------------------------------------------------
# Shard factory helpers
# ---------------------------------------------------------------------------


def _make_shard_with_empty_segments(
    n_rows: int = 2,
    context_len: int = 32,
) -> PackedRowShard:
    """Shard where every segment has zero features.

    Each row has 2 segments covering the full context, all with empty
    feat_offsets (no features stored).
    """
    T = context_len
    segs_per_row = 2
    total_segs = n_rows * segs_per_row

    tokens = t.randint(0, 32000, (n_rows, T + 1), dtype=t.uint16)

    # Each row has exactly 2 segments
    row_seg_offsets = t.tensor(
        list(range(0, (n_rows + 1) * segs_per_row, segs_per_row)),
        dtype=t.int32,
    )

    seg_token_start = t.zeros(total_segs, dtype=t.uint16)
    seg_token_end = t.zeros(total_segs, dtype=t.uint16)
    seg_doc_id = t.zeros(total_segs, dtype=t.uint8)
    half = T // 2
    for r in range(n_rows):
        base = r * segs_per_row
        seg_token_start[base] = 0
        seg_token_end[base] = half
        seg_token_start[base + 1] = half
        seg_token_end[base + 1] = T

    # All feat_offsets are zero — no features anywhere
    seg_feat_offsets = t.zeros(total_segs + 1, dtype=t.int32)
    feature_ids = t.zeros(0, dtype=t.uint16)
    feature_pcts = t.zeros(0, dtype=t.uint8)

    return PackedRowShard(
        tokens=tokens,
        row_seg_offsets=row_seg_offsets,
        seg_token_start=seg_token_start,
        seg_token_end=seg_token_end,
        seg_doc_id=seg_doc_id,
        seg_feat_offsets=seg_feat_offsets,
        feature_ids=feature_ids,
        feature_pcts=feature_pcts,
    )


def _make_single_row_shard(
    context_len: int = 32,
    n_features: int = 10,
) -> PackedRowShard:
    """Shard with exactly 1 row, 1 segment, n_features features."""
    T = context_len

    tokens = t.randint(0, 32000, (1, T + 1), dtype=t.uint16)
    row_seg_offsets = t.tensor([0, 1], dtype=t.int32)
    seg_token_start = t.tensor([0], dtype=t.uint16)
    seg_token_end = t.tensor([T], dtype=t.uint16)
    seg_doc_id = t.tensor([0], dtype=t.uint8)
    seg_feat_offsets = t.tensor([0, n_features], dtype=t.int32)
    feature_ids = t.randint(0, 16384, (n_features,), dtype=t.uint16)
    # Ensure all percentile values are >0 (valid range) for normalization test
    feature_pcts = t.randint(1, 256, (n_features,), dtype=t.uint8)

    return PackedRowShard(
        tokens=tokens,
        row_seg_offsets=row_seg_offsets,
        seg_token_start=seg_token_start,
        seg_token_end=seg_token_end,
        seg_doc_id=seg_doc_id,
        seg_feat_offsets=seg_feat_offsets,
        feature_ids=feature_ids,
        feature_pcts=feature_pcts,
    )


def _make_variable_features_shard() -> PackedRowShard:
    """1 row, 3 segments (0-16, 16-32, 32-48), with 0, 5, and 50 features."""
    T = 48  # context_len

    tokens = t.randint(0, 32000, (1, T + 1), dtype=t.uint16)
    row_seg_offsets = t.tensor([0, 3], dtype=t.int32)

    seg_token_start = t.tensor([0, 16, 32], dtype=t.uint16)
    seg_token_end = t.tensor([16, 32, 48], dtype=t.uint16)
    seg_doc_id = t.tensor([0, 0, 0], dtype=t.uint8)

    # Segment feature counts: 0, 5, 50
    seg_feat_offsets = t.tensor([0, 0, 5, 55], dtype=t.int32)
    total_feats = 55
    feature_ids = t.randint(0, 16384, (total_feats,), dtype=t.uint16)
    # Ensure non-zero pcts so normalization range test is meaningful
    feature_pcts = t.randint(1, 256, (total_feats,), dtype=t.uint8)

    return PackedRowShard(
        tokens=tokens,
        row_seg_offsets=row_seg_offsets,
        seg_token_start=seg_token_start,
        seg_token_end=seg_token_end,
        seg_doc_id=seg_doc_id,
        seg_feat_offsets=seg_feat_offsets,
        feature_ids=feature_ids,
        feature_pcts=feature_pcts,
    )


# ---------------------------------------------------------------------------
# _build_batch_index edge cases
# ---------------------------------------------------------------------------


class TestBuildBatchIndexEdgeCases:
    def test_all_segments_empty_features(self) -> None:
        """B=2, total_segs=4, seg_map has correct shape for all-empty shard."""
        shard = _make_shard_with_empty_segments(n_rows=2, context_len=32)
        idx = _build_batch_index(shard, [0, 1])

        assert idx.B == 2
        assert idx.total_segs == 4
        assert idx.seg_map.shape == (2, 32)

    def test_single_row(self) -> None:
        """B=1, batch_n_seg=1, total_segs=1."""
        shard = _make_single_row_shard(context_len=32, n_features=10)
        idx = _build_batch_index(shard, [0])

        assert idx.B == 1
        assert idx.batch_n_seg == 1
        assert idx.total_segs == 1

    def test_variable_feature_counts(self) -> None:
        """total_segs=3, batch_n_seg=3 for the 3-segment variable shard."""
        shard = _make_variable_features_shard()
        idx = _build_batch_index(shard, [0])

        assert idx.total_segs == 3
        assert idx.batch_n_seg == 3


# ---------------------------------------------------------------------------
# reconstruct_batch edge cases
# ---------------------------------------------------------------------------


class TestReconstructBatchEdgeCases:
    def test_zero_features_all_segments(self) -> None:
        """Empty-feature shard: enc_mask is all False, F_max >= 1."""
        shard = _make_shard_with_empty_segments(n_rows=2, context_len=32)
        rb = reconstruct_batch(shard, [0, 1], _make_identity_selection(shard), 10)

        assert rb.enc_mask.sum().item() == 0
        assert rb.enc_fids.shape[2] >= 1

    def test_single_row_batch(self) -> None:
        """Single-row batch: inputs shape (1, 32), enc_mask.sum()==10."""
        shard = _make_single_row_shard(context_len=32, n_features=10)
        rb = reconstruct_batch(shard, [0], _make_identity_selection(shard), 10)

        assert rb.inputs.shape == (1, 32)
        assert rb.targets.shape == (1, 32)
        assert int(rb.enc_mask.sum().item()) == 10

    def test_variable_features_per_segment(self) -> None:
        """F_max==50, seg0 has 0 feats, seg1 has 5, seg2 has 50."""
        shard = _make_variable_features_shard()
        rb = reconstruct_batch(shard, [0], _make_identity_selection(shard), 10)

        # F_max should be 50 (largest segment feature count)
        assert rb.enc_fids.shape[2] == 50

        # Segment 0: 0 features
        assert int(rb.enc_mask[0, 0].sum().item()) == 0
        # Segment 1: 5 features
        assert int(rb.enc_mask[0, 1].sum().item()) == 5
        # Segment 2: 50 features
        assert int(rb.enc_mask[0, 2].sum().item()) == 50

    def test_targets_are_shifted_inputs(self) -> None:
        """targets[i] == raw_tokens[i+1] for all valid positions."""
        shard = _make_single_row_shard(context_len=32, n_features=5)
        rb = reconstruct_batch(shard, [0], _make_identity_selection(shard), 10)

        raw = shard.tokens[0].to(t.int64)
        T = shard.context_len
        assert t.equal(rb.inputs[0], raw[:T])
        assert t.equal(rb.targets[0], raw[1 : T + 1])

    def test_seg_map_values_in_range(self) -> None:
        """seg_map values are all >= 0 and < n_seg."""
        shard = _make_variable_features_shard()
        rb = reconstruct_batch(shard, [0], _make_identity_selection(shard), 10)

        n_seg = rb.enc_fids.shape[1]
        assert int(rb.seg_map.min().item()) >= 0
        assert int(rb.seg_map.max().item()) < n_seg

    def test_seg_doc_id_minus_one_for_padding(self) -> None:
        """Real segment has doc_id=0; padding slots have -1."""
        shard = _make_single_row_shard(context_len=32, n_features=10)
        rb = reconstruct_batch(shard, [0], _make_identity_selection(shard), 10, pad_n_seg=4)

        # Segment 0 is real: doc_id == 0
        assert rb.seg_doc_id[0, 0].item() == 0
        # Segments 1..3 are padding: doc_id == -1
        for s in range(1, 4):
            assert rb.seg_doc_id[0, s].item() == -1

    def test_percentile_normalization_range(self) -> None:
        """Active (masked) percentile values are in (0, 1]."""
        shard = _make_single_row_shard(context_len=32, n_features=10)
        rb = reconstruct_batch(shard, [0], _make_identity_selection(shard), 10)

        active_pcts = rb.enc_pcts[rb.enc_mask]
        assert active_pcts.numel() > 0, "Expected at least one active feature"
        assert float(active_pcts.min().item()) > 0.0
        assert float(active_pcts.max().item()) <= 1.0
