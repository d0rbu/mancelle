"""Unit tests for build_fp_targets_from_shard in dataloader.py.

Tests the dense [B, n_seg, n_sae_features] feature prediction target builder
that scatters percentile values from CSR shard data into a dense float32 tensor.
"""

from __future__ import annotations

import pytest
import torch

from steering.conditional.encoder_decoder.dataloader import (
    _build_fp_targets_from_flat_features,
    build_fp_targets_from_shard,
)
from steering.packed_batch_io import PackedRowShard, _build_batch_index
from tests.encdec_test_helpers import N_SAE_FEATURES, _make_test_shard

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_shard():
    """Synthetic shard: 4 rows, 3 segs/row, 20 feats/seg."""
    return _make_test_shard(n_rows=4, segs_per_row=3, feats_per_seg=20)


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------


class TestOutputShape:
    """build_fp_targets_from_shard returns the correct shape and dtype."""

    def test_output_shape(self, simple_shard):
        B = 2
        n_seg = 3
        idx = _build_batch_index(simple_shard, row_indices=range(B))
        out = build_fp_targets_from_shard(simple_shard, idx, N_SAE_FEATURES)

        assert out.shape == (B, n_seg, N_SAE_FEATURES)
        assert out.dtype == torch.float32

    def test_output_shape_with_padding(self, simple_shard):
        """pad_n_seg inflates the segment dimension beyond the natural maximum."""
        B = 2
        pad_n_seg = 8  # natural max per row is 3
        idx = _build_batch_index(simple_shard, row_indices=range(B), pad_n_seg=pad_n_seg)
        out = build_fp_targets_from_shard(simple_shard, idx, N_SAE_FEATURES)

        assert out.shape == (B, pad_n_seg, N_SAE_FEATURES)
        assert out.dtype == torch.float32

    def test_single_row_batch(self):
        """B=1 batch still yields correct 3-D shape."""
        shard = _make_test_shard(n_rows=1, segs_per_row=2, feats_per_seg=5)
        idx = _build_batch_index(shard, row_indices=[0])
        out = build_fp_targets_from_shard(shard, idx, N_SAE_FEATURES)

        assert out.shape == (1, 2, N_SAE_FEATURES)
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Feature counts and value range
# ---------------------------------------------------------------------------


class TestNonzeroEntries:
    """Each segment has no more nonzero columns than the shard's feats_per_seg."""

    def test_nonzero_entries_match_feature_count(self):
        feats_per_seg = 15
        shard = _make_test_shard(n_rows=4, segs_per_row=3, feats_per_seg=feats_per_seg)
        B = 4
        idx = _build_batch_index(shard, row_indices=range(B))
        out = build_fp_targets_from_shard(shard, idx, N_SAE_FEATURES)

        # Count nonzero features per segment
        nonzero_counts = (out != 0).sum(dim=-1)  # [B, n_seg]
        assert (nonzero_counts <= feats_per_seg).all(), (
            f"Some segments have more nonzero entries than feats_per_seg={feats_per_seg}: "
            f"max={nonzero_counts.max().item()}"
        )


class TestValueRange:
    """Values in the output are normalised percentiles in (0, 1]."""

    def test_values_are_normalized_percentiles(self):
        # _make_test_shard draws feature_pcts from randint(1, 255) so min is 1
        shard = _make_test_shard(n_rows=4, segs_per_row=3, feats_per_seg=20)
        B = 4
        idx = _build_batch_index(shard, row_indices=range(B))
        out = build_fp_targets_from_shard(shard, idx, N_SAE_FEATURES)

        nonzero_mask = out != 0
        if nonzero_mask.any():
            nonzero_vals = out[nonzero_mask]
            assert (nonzero_vals > 0).all(), "Some nonzero values are <= 0"
            assert (nonzero_vals <= 1.0).all(), (
                f"Some values exceed 1.0: max={nonzero_vals.max().item()}"
            )

    def test_max_pct_255_maps_to_one(self):
        """pct_uint8=255 must map exactly to 1.0 (255/255)."""
        context_len = 8
        shard = PackedRowShard(
            tokens=torch.randint(1, 100, (1, context_len + 1), dtype=torch.uint16),
            row_seg_offsets=torch.tensor([0, 1], dtype=torch.int32),
            seg_token_start=torch.tensor([0], dtype=torch.uint16),
            seg_token_end=torch.tensor([context_len], dtype=torch.uint16),
            seg_doc_id=torch.zeros(1, dtype=torch.uint8),
            seg_feat_offsets=torch.tensor([0, 1], dtype=torch.int32),
            feature_ids=torch.tensor([7], dtype=torch.uint16),
            feature_pcts=torch.tensor([255], dtype=torch.uint8),
        )
        idx = _build_batch_index(shard, row_indices=[0])
        out = build_fp_targets_from_shard(shard, idx, N_SAE_FEATURES)

        assert out[0, 0, 7].item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Feature ID placement
# ---------------------------------------------------------------------------


class TestFeatureIdPlacement:
    """Feature IDs scatter into the correct columns with the correct values."""

    def test_flat_scatter_matches_old_fp_flat_reference(self):
        """New direct scatter is bit-identical to the old fp_flat two-step path."""
        shard = _make_test_shard(
            n_rows=3,
            segs_per_row=[2, 4, 1],
            feats_per_seg=7,
            n_features=N_SAE_FEATURES,
        )
        idx = _build_batch_index(shard, row_indices=[0, 2], pad_n_seg=5)
        feat_offsets_t = shard.seg_feat_offsets.long()
        feat_starts = feat_offsets_t[idx.global_flat]
        feat_ends = feat_offsets_t[idx.global_flat + 1]
        feat_counts = feat_ends - feat_starts
        total_feats = int(feat_counts.sum().item())
        cum_fc = torch.cat([torch.zeros(1, dtype=torch.long), feat_counts.cumsum(0)[:-1]])
        gather_offsets = torch.arange(total_feats) - torch.repeat_interleave(
            cum_fc,
            feat_counts,
        )
        gather_idx = torch.repeat_interleave(feat_starts, feat_counts) + gather_offsets
        all_fids = shard.feature_ids[gather_idx].to(torch.int64)
        all_pcts = shard.feature_pcts[gather_idx].float() * (1.0 / 255.0)

        fp_flat = torch.zeros(idx.total_segs, N_SAE_FEATURES, dtype=torch.float32)
        seg_idx = torch.repeat_interleave(torch.arange(idx.total_segs), feat_counts)
        fp_flat[seg_idx, all_fids] = all_pcts
        expected = torch.zeros(idx.B, idx.batch_n_seg, N_SAE_FEATURES, dtype=torch.float32)
        expected[idx.b_flat, idx.local_flat] = fp_flat

        actual = _build_fp_targets_from_flat_features(
            idx,
            all_fids,
            all_pcts,
            feat_counts,
            N_SAE_FEATURES,
        )

        assert torch.equal(actual, expected)

    def test_feature_ids_map_to_correct_columns(self):
        """Manual shard: features at IDs 5, 10, 15 with pcts 128, 255, 64.

        Verify exact placement (128/255, 255/255, 64/255) and all other columns zero.
        """
        context_len = 16
        shard = PackedRowShard(
            tokens=torch.randint(1, 100, (1, context_len + 1), dtype=torch.uint16),
            row_seg_offsets=torch.tensor([0, 1], dtype=torch.int32),
            seg_token_start=torch.tensor([0], dtype=torch.uint16),
            seg_token_end=torch.tensor([context_len], dtype=torch.uint16),
            seg_doc_id=torch.zeros(1, dtype=torch.uint8),
            seg_feat_offsets=torch.tensor([0, 3], dtype=torch.int32),
            feature_ids=torch.tensor([5, 10, 15], dtype=torch.uint16),
            feature_pcts=torch.tensor([128, 255, 64], dtype=torch.uint8),
        )
        idx = _build_batch_index(shard, row_indices=[0])
        out = build_fp_targets_from_shard(shard, idx, N_SAE_FEATURES)

        # Shape: (1, 1, N_SAE_FEATURES)
        row = out[0, 0]

        assert row[5].item() == pytest.approx(128 / 255.0)
        assert row[10].item() == pytest.approx(255 / 255.0)
        assert row[15].item() == pytest.approx(64 / 255.0)

        # All columns except 5, 10, 15 must be zero
        mask = torch.ones(N_SAE_FEATURES, dtype=torch.bool)
        mask[[5, 10, 15]] = False
        assert row[mask].eq(0).all(), "Unexpected nonzero columns in output row"

    def test_multiple_segments_independent(self):
        """Two segments with different features must not bleed into each other."""
        context_len = 16
        # Row has 2 segments, each with 1 feature
        shard = PackedRowShard(
            tokens=torch.randint(1, 100, (1, context_len + 1), dtype=torch.uint16),
            row_seg_offsets=torch.tensor([0, 2], dtype=torch.int32),
            seg_token_start=torch.tensor([0, 8], dtype=torch.uint16),
            seg_token_end=torch.tensor([8, 16], dtype=torch.uint16),
            seg_doc_id=torch.zeros(2, dtype=torch.uint8),
            seg_feat_offsets=torch.tensor([0, 1, 2], dtype=torch.int32),
            feature_ids=torch.tensor([3, 7], dtype=torch.uint16),
            feature_pcts=torch.tensor([100, 200], dtype=torch.uint8),
        )
        idx = _build_batch_index(shard, row_indices=[0])
        out = build_fp_targets_from_shard(shard, idx, N_SAE_FEATURES)

        # Seg 0: feature 3 = 100/255, feature 7 = 0
        assert out[0, 0, 3].item() == pytest.approx(100 / 255.0)
        assert out[0, 0, 7].item() == 0.0

        # Seg 1: feature 7 = 200/255, feature 3 = 0
        assert out[0, 1, 7].item() == pytest.approx(200 / 255.0)
        assert out[0, 1, 3].item() == 0.0


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------


class TestPaddedSegments:
    """Padding slots beyond the natural segment count are all zeros."""

    def test_padded_segments_are_zero(self):
        """Rows have 2 natural segments; pad to 5 — slots [2:5] must be zero."""
        shard = _make_test_shard(n_rows=2, segs_per_row=2, feats_per_seg=10)
        B = 2
        pad_n_seg = 5
        idx = _build_batch_index(shard, row_indices=range(B), pad_n_seg=pad_n_seg)
        out = build_fp_targets_from_shard(shard, idx, N_SAE_FEATURES)

        # Real segments: [0:2], padding: [2:5]
        padding_slots = out[:, 2:, :]
        assert padding_slots.eq(0).all(), (
            f"Padded segments are not all zeros; nonzero count: {padding_slots.ne(0).sum().item()}"
        )

    def test_no_padding_has_no_spurious_zeros_only(self):
        """Without padding, slot count equals the shard's segs_per_row."""
        segs_per_row = 3
        shard = _make_test_shard(n_rows=2, segs_per_row=segs_per_row, feats_per_seg=10)
        idx = _build_batch_index(shard, row_indices=range(2))
        out = build_fp_targets_from_shard(shard, idx, N_SAE_FEATURES)

        # Segment dimension equals segs_per_row (no extra padding added)
        assert out.shape[1] == segs_per_row


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


class TestDevice:
    """Output lands on the requested device."""

    def test_cpu_device_is_default(self, simple_shard):
        idx = _build_batch_index(simple_shard, row_indices=range(2))
        out = build_fp_targets_from_shard(simple_shard, idx, N_SAE_FEATURES)
        assert out.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self, simple_shard):
        idx = _build_batch_index(simple_shard, row_indices=range(2))
        out = build_fp_targets_from_shard(simple_shard, idx, N_SAE_FEATURES, device="cuda")
        assert out.device.type == "cuda"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Corner cases: empty shard, zero features, single feature."""

    def test_row_with_zero_segments_returns_zeros(self):
        """A row whose CSR range maps to zero segments contributes an all-zero [1, 1, N_SAE] block.

        _build_batch_index requires at least one row (empty row_indices crashes on
        seg_counts.max() for an empty tensor). The empty-feature case is already
        covered by test_segment_with_no_features_is_zero; here we verify that the
        output is entirely zero when total_segs == 0 by padding above the natural
        batch_n_seg of 1.
        """
        context_len = 8
        # Row has 1 segment in the CSR offsets but zero features.
        shard = PackedRowShard(
            tokens=torch.randint(1, 100, (1, context_len + 1), dtype=torch.uint16),
            row_seg_offsets=torch.tensor([0, 1], dtype=torch.int32),
            seg_token_start=torch.tensor([0], dtype=torch.uint16),
            seg_token_end=torch.tensor([context_len], dtype=torch.uint16),
            seg_doc_id=torch.zeros(1, dtype=torch.uint8),
            seg_feat_offsets=torch.tensor([0, 0], dtype=torch.int32),
            feature_ids=torch.zeros(0, dtype=torch.uint16),
            feature_pcts=torch.zeros(0, dtype=torch.uint8),
        )
        idx = _build_batch_index(shard, row_indices=[0])
        out = build_fp_targets_from_shard(shard, idx, N_SAE_FEATURES)

        assert out.shape == (1, 1, N_SAE_FEATURES)
        assert out.eq(0).all()

    def test_segment_with_no_features_is_zero(self):
        """A segment that has zero features in the CSR array contributes an all-zero row."""
        context_len = 8
        shard = PackedRowShard(
            tokens=torch.randint(1, 100, (1, context_len + 1), dtype=torch.uint16),
            row_seg_offsets=torch.tensor([0, 1], dtype=torch.int32),
            seg_token_start=torch.tensor([0], dtype=torch.uint16),
            seg_token_end=torch.tensor([context_len], dtype=torch.uint16),
            seg_doc_id=torch.zeros(1, dtype=torch.uint8),
            # seg_feat_offsets [0, 0] means zero features for the single segment
            seg_feat_offsets=torch.tensor([0, 0], dtype=torch.int32),
            feature_ids=torch.zeros(0, dtype=torch.uint16),
            feature_pcts=torch.zeros(0, dtype=torch.uint8),
        )
        idx = _build_batch_index(shard, row_indices=[0])
        out = build_fp_targets_from_shard(shard, idx, N_SAE_FEATURES)

        assert out.shape == (1, 1, N_SAE_FEATURES)
        assert out.eq(0).all()
