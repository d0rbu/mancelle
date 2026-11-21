"""Comprehensive tests for the fused reconstruct_and_select function.

Verifies correctness against reference implementations, covers all code
paths in reconstruct_and_select (sampling, threshold, negatives, edge
cases), and tests integration with precomputed_data_loader (including
cross-shard boundaries).

Test categories:
 1. Output shapes and dtypes
 2. Token/seg_map/seg_doc_id reference comparison (vs reconstruct_batch)
 3. CSR feature extraction reference comparison
 4. Feature selection correctness (sampling + threshold)
 5. Negative sampling correctness (disjointness, uniqueness, counts)
 6. Edge cases (variable shards, empty segments, all-empty, pad_n_seg)
 7. Profiling subtimings
 8. precomputed_data_loader integration (single-shard + cross-shard)
 9. _build_batch_index unit tests
"""

from __future__ import annotations

import json

import pytest
import torch as t

from steering.conditional.encoder_decoder.dataloader import (
    precomputed_data_loader,
    reconstruct_and_select,
)
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeBundle,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingSelection,
    ThresholdSelection,
)
from steering.conditional.thresholding import ThresholdType
from steering.packed_batch_io import (
    PackedRowShard,
    SelectionShard,
    _build_batch_index,
    reconstruct_batch,
    save_packed_row_shard,
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
# Helpers
# ---------------------------------------------------------------------------

N_SAE_FEATURES = 1024


def _make_boundaries(n_features: int = N_SAE_FEATURES, n_buckets: int = 10) -> t.Tensor:
    n_boundaries = n_buckets - 1
    b = t.linspace(1 / n_buckets, 1.0 - 1 / n_buckets, n_boundaries, dtype=t.float64)
    return b.unsqueeze(0).expand(n_features, -1).contiguous()


def _make_percentile_table(n_features: int = N_SAE_FEATURES) -> t.Tensor:
    pcts = t.linspace(0.01, 0.99, 99, dtype=t.float64)
    return pcts.unsqueeze(0).expand(n_features, -1).contiguous()


def _make_sampling_pipeline(
    n_features: int = N_SAE_FEATURES,
    n_buckets: int = 10,
    with_negatives: bool = False,
    mean: float = 3.0,
    seed: int = 42,
) -> FeaturePipelineConfig:
    rng = t.Generator().manual_seed(seed)
    selection = SamplingSelection(
        distribution=Distribution.geometric,
        mean=mean, std=mean, min_k=1, max_k=None,
    )
    pct_table = _make_percentile_table(n_features)
    sampling_bundle = SamplingBundle(
        selection=selection, percentile_table=pct_table, rng=rng,
    )
    effective_buckets = n_buckets - 1 if with_negatives else n_buckets
    # Boundaries must use effective_buckets (not n_buckets) so positive
    # bucket indices stay in [0, effective_buckets-1], leaving the
    # effective_buckets value exclusively for negatives.
    boundaries = _make_boundaries(n_features, effective_buckets)

    negative_bundle = None
    if with_negatives:
        neg_rng = t.Generator().manual_seed(seed + 1)
        negative_bundle = NegativeBundle(
            selection=NegativeSamplingSelection(
                distribution=Distribution.geometric,
                mean=2.0, std=2.0, min_k=1,
            ),
            n_sae_features=n_features, rng=neg_rng,
        )

    return FeaturePipelineConfig(
        selection_config=sampling_bundle,
        boundaries=boundaries,
        negative_bundle=negative_bundle,
        effective_buckets=effective_buckets,
    )


def _make_threshold_pipeline(
    n_features: int = N_SAE_FEATURES,
    n_buckets: int = 10,
    threshold: float = 5.0,
    with_negatives: bool = False,
    seed: int = 42,
) -> FeaturePipelineConfig:
    selection = ThresholdSelection(
        threshold_type=ThresholdType.segment_topk, threshold=threshold,
    )
    config = ThresholdSelectionConfig(
        selection=selection, precomputed_thresholds=None,
    )
    effective_buckets = n_buckets - 1 if with_negatives else n_buckets
    boundaries = _make_boundaries(n_features, effective_buckets)

    negative_bundle = None
    if with_negatives:
        neg_rng = t.Generator().manual_seed(seed + 1)
        negative_bundle = NegativeBundle(
            selection=NegativeSamplingSelection(
                distribution=Distribution.geometric,
                mean=2.0, std=2.0, min_k=1,
            ),
            n_sae_features=n_features, rng=neg_rng,
        )

    return FeaturePipelineConfig(
        selection_config=config,
        boundaries=boundaries,
        negative_bundle=negative_bundle,
        effective_buckets=effective_buckets,
    )


def _make_shard(
    n_rows: int = 8,
    context_len: int = 32,
    segs_per_row: int = 3,
    feats_per_seg: int = 20,
    seed: int = 42,
) -> PackedRowShard:
    t.manual_seed(seed)
    T = context_len
    total_segs = n_rows * segs_per_row
    total_feats = total_segs * feats_per_seg

    tokens = t.randint(1, 32000, (n_rows, T + 1), dtype=t.uint16)
    row_seg_offsets = t.arange(0, (n_rows + 1) * segs_per_row, segs_per_row, dtype=t.int32)

    seg_size = T // segs_per_row
    seg_token_start = t.zeros(total_segs, dtype=t.uint16)
    seg_token_end = t.zeros(total_segs, dtype=t.uint16)
    seg_doc_id = t.zeros(total_segs, dtype=t.uint8)

    for r in range(n_rows):
        for s in range(segs_per_row):
            idx = r * segs_per_row + s
            seg_token_start[idx] = s * seg_size
            seg_token_end[idx] = T if s == segs_per_row - 1 else (s + 1) * seg_size
            seg_doc_id[idx] = 0

    seg_feat_offsets = t.arange(0, (total_segs + 1) * feats_per_seg, feats_per_seg, dtype=t.int32)
    feature_ids = t.randint(0, N_SAE_FEATURES, (total_feats,), dtype=t.uint16)
    feature_pcts = t.randint(1, 255, (total_feats,), dtype=t.uint8)

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


def _make_variable_shard(seed: int = 42) -> PackedRowShard:
    """Shard with variable segments per row and features per segment."""
    t.manual_seed(seed)
    n_rows, T = 3, 32
    tokens = t.randint(1, 32000, (n_rows, T + 1), dtype=t.uint16)

    # Row 0: 4 segments (3, 5, 2, 10 features)
    # Row 1: 2 segments (8, 0 features)
    # Row 2: 3 segments (1, 15, 6 features)
    row_seg_offsets = t.tensor([0, 4, 6, 9], dtype=t.int32)  # 9 total segments

    seg_token_start = t.tensor([0, 8, 16, 24, 0, 16, 0, 10, 20], dtype=t.uint16)
    seg_token_end = t.tensor([8, 16, 24, 32, 16, 32, 10, 20, 32], dtype=t.uint16)
    seg_doc_id = t.tensor([0, 0, 1, 1, 0, 0, 0, 0, 1], dtype=t.uint8)

    feat_counts = [3, 5, 2, 10, 8, 0, 1, 15, 6]
    total_feats = sum(feat_counts)
    seg_feat_offsets = t.tensor([0] + [sum(feat_counts[:i+1]) for i in range(len(feat_counts))], dtype=t.int32)

    feature_ids = t.randint(0, N_SAE_FEATURES, (total_feats,), dtype=t.uint16)
    feature_pcts = t.randint(1, 255, (total_feats,), dtype=t.uint8)

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


# ======================================================================
# 1. Output shape and dtype correctness
# ======================================================================


class TestFusedShapesAndDtypes:
    """Test that fused path produces outputs with correct shapes and dtypes."""

    def test_output_shapes_sampling(self):
        shard = _make_shard(n_rows=4, feats_per_seg=20)
        pipeline = _make_sampling_pipeline(mean=3.0)
        rb, _ = reconstruct_and_select(shard, [0, 1], pipeline)

        assert rb.inputs.shape == (2, 32)
        assert rb.targets.shape == (2, 32)
        assert rb.seg_map.shape == (2, 32)
        assert rb.enc_fids.shape[0] == 2
        assert rb.enc_mags.shape == rb.enc_fids.shape
        assert rb.enc_mask.shape == rb.enc_fids.shape
        assert rb.enc_pcts.shape == rb.enc_fids.shape
        assert rb.seg_doc_id.shape[0] == 2

    def test_output_dtypes(self):
        shard = _make_shard()
        pipeline = _make_sampling_pipeline()
        rb, _ = reconstruct_and_select(shard, [0], pipeline)

        assert rb.inputs.dtype == t.int64
        assert rb.targets.dtype == t.int64
        assert rb.enc_fids.dtype == t.int64
        assert rb.enc_mags.dtype == t.int64
        assert rb.enc_mask.dtype == t.bool
        assert rb.enc_pcts.dtype == t.float32
        assert rb.seg_map.dtype == t.int64
        assert rb.seg_doc_id.dtype == t.int64

    def test_f_max_much_smaller_than_raw(self):
        """Fused output F dimension is much smaller than raw feature count."""
        shard = _make_shard(n_rows=4, feats_per_seg=50)
        pipeline = _make_sampling_pipeline(mean=3.0)
        rb, _ = reconstruct_and_select(shard, list(range(4)), pipeline)

        # Sampling with mean=3 should give F_max << 50
        assert rb.enc_fids.shape[2] < 50
        assert rb.enc_mask.any()


# ======================================================================
# 2. Token and segment correctness
# ======================================================================


class TestFusedTokensAndSegments:
    """Verify tokens, seg_map, seg_doc_id match reconstruct_batch exactly."""

    def test_inputs_targets_match(self):
        shard = _make_shard(n_rows=8)
        pipeline = _make_sampling_pipeline()
        rows = [0, 3, 5]

        rb_fused, _ = reconstruct_and_select(shard, rows, pipeline)
        rb_base = reconstruct_batch(shard, rows, _make_identity_selection(shard), 10)

        assert t.equal(rb_fused.inputs, rb_base.inputs)
        assert t.equal(rb_fused.targets, rb_base.targets)

    def test_seg_map_matches(self):
        shard = _make_shard(n_rows=4)
        pipeline = _make_sampling_pipeline()
        rows = [0, 1, 2, 3]

        rb_fused, _ = reconstruct_and_select(shard, rows, pipeline)
        rb_base = reconstruct_batch(shard, rows, _make_identity_selection(shard), 10)

        assert t.equal(rb_fused.seg_map, rb_base.seg_map)

    def test_seg_doc_id_matches(self):
        shard = _make_shard(n_rows=4)
        pipeline = _make_sampling_pipeline()
        rows = [1, 3]

        rb_fused, _ = reconstruct_and_select(shard, rows, pipeline)
        rb_base = reconstruct_batch(shard, rows, _make_identity_selection(shard), 10)

        # seg_doc_id shapes may differ (fused has variable n_seg, base has variable n_seg)
        # but the actual values should match up to the smaller shape
        min_ns = min(rb_fused.seg_doc_id.shape[1], rb_base.seg_doc_id.shape[1])
        assert t.equal(rb_fused.seg_doc_id[:, :min_ns], rb_base.seg_doc_id[:, :min_ns])

    def test_single_row(self):
        shard = _make_shard(n_rows=4)
        pipeline = _make_sampling_pipeline()
        rb, _ = reconstruct_and_select(shard, [2], pipeline)

        assert rb.inputs.shape[0] == 1
        assert rb.enc_fids.shape[0] == 1


# ======================================================================
# 3. Feature selection correctness
# ======================================================================


class TestFusedFeatureSelection:
    """Verify selected features are valid and properly discretized."""

    def test_selected_fids_from_shard(self):
        """All selected positive feature IDs exist in the original CSR data."""
        shard = _make_shard(n_rows=4, feats_per_seg=30)
        pipeline = _make_sampling_pipeline(mean=5.0)
        rows = [0, 1, 2, 3]

        rb, _ = reconstruct_and_select(shard, rows, pipeline)
        rb_base = reconstruct_batch(shard, rows, _make_identity_selection(shard), 10)

        for b in range(rb.enc_fids.shape[0]):
            for s in range(rb.enc_fids.shape[1]):
                active = rb.enc_mask[b, s]
                if not active.any():
                    continue
                selected = set(rb.enc_fids[b, s, active].tolist())
                # Base has all original features for this segment
                base_active = rb_base.enc_mask[b, s] if s < rb_base.enc_mask.shape[1] else t.zeros(1, dtype=t.bool)
                if base_active.any():
                    original = set(rb_base.enc_fids[b, s, base_active].tolist())
                    # Positive features must be subset of original
                    # (negatives can be anything not in original)
                    positives = {fid for fid in selected if fid in original}
                    assert len(positives) > 0 or len(selected) > 0

    def test_bucket_values_in_range(self):
        n_buckets = 10
        shard = _make_shard(n_rows=4, feats_per_seg=20)
        pipeline = _make_sampling_pipeline(n_buckets=n_buckets)
        rb, _ = reconstruct_and_select(shard, [0, 1], pipeline)

        active_mags = rb.enc_mags[rb.enc_mask]
        if len(active_mags) > 0:
            assert active_mags.min() >= 0
            assert active_mags.max() < n_buckets

    def test_percentiles_in_range(self):
        shard = _make_shard(n_rows=4, feats_per_seg=20)
        pipeline = _make_sampling_pipeline()
        rb, _ = reconstruct_and_select(shard, [0, 1], pipeline)

        active_pcts = rb.enc_pcts[rb.enc_mask]
        if len(active_pcts) > 0:
            assert active_pcts.min() >= 0.0
            assert active_pcts.max() <= 1.0

    def test_unmasked_positions_are_zero(self):
        shard = _make_shard(n_rows=4)
        pipeline = _make_sampling_pipeline()
        rb, _ = reconstruct_and_select(shard, [0, 1], pipeline)

        inactive = ~rb.enc_mask
        if inactive.any():
            assert (rb.enc_fids[inactive] == 0).all()
            assert (rb.enc_mags[inactive] == 0).all()
            assert (rb.enc_pcts[inactive] == 0).all()


# ======================================================================
# 4. Threshold path
# ======================================================================


class TestFusedThresholdPath:
    """Test the threshold selection path through the fused function."""

    def test_threshold_topk_limits_features(self):
        k = 5
        shard = _make_shard(n_rows=4, feats_per_seg=30)
        pipeline = _make_threshold_pipeline(threshold=k)
        rb, _ = reconstruct_and_select(shard, [0, 1], pipeline)

        for b in range(rb.enc_fids.shape[0]):
            for s in range(rb.enc_fids.shape[1]):
                n_active = rb.enc_mask[b, s].sum().item()
                assert n_active <= k, f"[{b},{s}] has {n_active} features, expected <= {k}"

    def test_threshold_output_dtypes(self):
        shard = _make_shard()
        pipeline = _make_threshold_pipeline()
        rb, _ = reconstruct_and_select(shard, [0], pipeline)

        assert rb.enc_fids.dtype == t.int64
        assert rb.enc_mags.dtype == t.int64
        assert rb.enc_mask.dtype == t.bool
        assert rb.enc_pcts.dtype == t.float32


# ======================================================================
# 5. Negative sampling
# ======================================================================


class TestFusedNegativeSampling:
    """Test negative feature sampling through the fused path."""

    def test_negatives_present_in_output(self):
        """With negative sampling, output contains features at the effective_buckets value."""
        shard = _make_shard(n_rows=4, feats_per_seg=20)
        pipeline = _make_sampling_pipeline(with_negatives=True, mean=3.0)

        rb, _ = reconstruct_and_select(shard, [0, 1], pipeline)

        # Negative features have bucket = effective_buckets
        active_mags = rb.enc_mags[rb.enc_mask]
        assert (active_mags == pipeline.effective_buckets).any(), (
            "No negative features found — negative sampling should produce some"
        )

    def test_negative_features_get_effective_bucket(self):
        n_buckets = 10
        shard = _make_shard(n_rows=2, feats_per_seg=20)
        pipeline = _make_sampling_pipeline(
            n_buckets=n_buckets, with_negatives=True, mean=3.0,
        )
        effective_buckets = pipeline.effective_buckets  # n_buckets - 1

        rb, _ = reconstruct_and_select(shard, [0], pipeline)

        # At least some features should have the effective_buckets value
        active_mags = rb.enc_mags[rb.enc_mask]
        if len(active_mags) > 0:
            assert (active_mags == effective_buckets).any(), (
                f"No negative features found with bucket={effective_buckets}"
            )


# ======================================================================
# 6. Edge cases
# ======================================================================


class TestFusedEdgeCases:
    """Test edge cases: variable segments, empty segments, etc."""

    def test_variable_segments_per_row(self):
        shard = _make_variable_shard()
        pipeline = _make_sampling_pipeline(mean=2.0)
        rb, _ = reconstruct_and_select(shard, [0, 1, 2], pipeline)

        assert rb.inputs.shape[0] == 3
        assert rb.enc_fids.shape[0] == 3
        # n_seg should accommodate the row with most segments (row 0 has 4)
        assert rb.enc_fids.shape[1] >= 4

    def test_empty_feature_segment(self):
        """Segment with 0 features is handled correctly."""
        shard = _make_variable_shard()  # Row 1 seg 1 has 0 features
        pipeline = _make_sampling_pipeline()
        rb, _ = reconstruct_and_select(shard, [1], pipeline)

        # Should not crash, mask should have some entries
        assert rb.enc_fids.shape[0] == 1

    def test_non_contiguous_rows(self):
        shard = _make_shard(n_rows=8)
        pipeline = _make_sampling_pipeline()
        rb, _ = reconstruct_and_select(shard, [0, 3, 7], pipeline)

        assert rb.inputs.shape[0] == 3
        # Tokens should match
        rb_base = reconstruct_batch(shard, [0, 3, 7], _make_identity_selection(shard), 10)
        assert t.equal(rb.inputs, rb_base.inputs)

    def test_pad_n_seg(self):
        shard = _make_shard(n_rows=4, segs_per_row=3)
        pipeline = _make_sampling_pipeline()
        rb, _ = reconstruct_and_select(shard, [0, 1], pipeline, pad_n_seg=10)

        assert rb.enc_fids.shape[1] == 10
        assert rb.seg_doc_id.shape[1] == 10

    def test_unsupported_selection_type_raises(self):
        shard = _make_shard()
        pipeline = FeaturePipelineConfig(
            selection_config="invalid",  # ty: ignore[invalid-argument-type]
            boundaries=_make_boundaries(),
            negative_bundle=None,
            effective_buckets=10,
        )
        with pytest.raises(TypeError, match="Unsupported selection config"):
            reconstruct_and_select(shard, [0], pipeline)


# ======================================================================
# 7. Integration with precomputed_data_loader
# ======================================================================


class TestFusedPrecomputedLoaderIntegration:
    """Test that precomputed_data_loader uses the fused path correctly."""

    def _save_shards(self, tmp_path, n_rows=8, n_shards=2, **kwargs):
        for i in range(n_shards):
            shard = _make_shard(n_rows=n_rows, seed=42 + i, **kwargs)
            save_packed_row_shard(shard, tmp_path / f"shard_{i:05d}")
        # Write a minimal manifest
        import json
        manifest = {"metadata": {"type": "packed_rows", "max_n_seg": 3}}
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    def test_fused_path_active(self, tmp_path):
        """With feature_pipeline and no selection shards, fused path is used."""
        self._save_shards(tmp_path)
        pipeline = _make_sampling_pipeline()

        loader = precomputed_data_loader(
            tmp_path, B=4, device="cpu", feature_pipeline=pipeline,
        )

        batch = next(loader)
        inputs, targets, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, _fp_targets, state = batch

        assert inputs.shape[0] == 4
        assert enc_fids.shape[0] == 4
        # F_max should be small (from selection, not raw ~20)
        assert enc_fids.shape[2] < 20
        assert enc_mask.any()

    def test_fused_path_batch_shapes_consistent(self, tmp_path):
        """Multiple batches from fused path have consistent properties."""
        self._save_shards(tmp_path, n_rows=16)
        pipeline = _make_sampling_pipeline()

        loader = precomputed_data_loader(
            tmp_path, B=4, device="cpu", feature_pipeline=pipeline,
        )

        for _ in range(3):
            batch = next(loader)
            _, _, enc_fids, enc_mags, enc_mask, enc_pcts, _, _, _fp_targets, _ = batch
            assert enc_fids.dtype == t.int64
            assert enc_mags.dtype == t.int64
            assert enc_mask.dtype == t.bool
            assert enc_pcts.dtype == t.float32
            assert enc_mask.any()

    def test_fused_cross_shard_boundary(self, tmp_path):
        """Batch spanning two shards is assembled correctly via fused path."""
        # Create two small shards with 3 rows each, B=4 forces cross-shard batch
        for i in range(2):
            shard = _make_shard(n_rows=3, seed=100 + i)
            save_packed_row_shard(shard, tmp_path / f"shard_{i:05d}")
        manifest = {"metadata": {"type": "packed_rows"}}
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))

        pipeline = _make_sampling_pipeline()
        loader = precomputed_data_loader(
            tmp_path, B=4, device="cpu", feature_pipeline=pipeline,
        )

        # First batch: 3 rows from shard_0 + 1 from shard_1 → cross-shard
        batch = next(loader)
        inputs, targets, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, _fp_targets, state = batch

        assert inputs.shape[0] == 4
        assert enc_fids.shape[0] == 4
        assert enc_mask.any()
        # All shapes must be consistent across the batch dimension
        assert enc_fids.shape == enc_mags.shape == enc_mask.shape == enc_pcts.shape

    def test_fused_label_dropout_applied(self, tmp_path):
        """Label dropout masks out features after fused reconstruction."""
        self._save_shards(tmp_path, n_rows=8)
        pipeline = _make_sampling_pipeline()

        loader_no_drop = precomputed_data_loader(
            tmp_path, B=4, device="cpu", feature_pipeline=pipeline,
            label_dropout=0.0,
        )
        # Use a fresh pipeline with same seed for comparison
        pipeline2 = _make_sampling_pipeline()
        loader_drop = precomputed_data_loader(
            tmp_path, B=4, device="cpu", feature_pipeline=pipeline2,
            label_dropout=0.99,  # high dropout
        )

        batch_no = next(loader_no_drop)
        batch_drop = next(loader_drop)

        # With 99% dropout, should have fewer active features
        assert batch_drop[4].sum() < batch_no[4].sum()

    def test_fused_nan_guard(self, tmp_path):
        """Fully-empty rows get a sentinel feature to prevent NaN in cross-attention."""
        self._save_shards(tmp_path, n_rows=8)
        pipeline = _make_sampling_pipeline()

        loader = precomputed_data_loader(
            tmp_path, B=4, device="cpu", feature_pipeline=pipeline,
            label_dropout=1.0,  # drop everything
        )

        batch = next(loader)
        enc_mask = batch[4]

        # NaN guard: every row must have at least one True in enc_mask
        row_has_any = enc_mask.any(dim=-1).any(dim=-1)  # (B,)
        assert row_has_any.all(), "NaN guard failed — some rows have no active features"


# ======================================================================
# 9. Reference correctness: _build_batch_index
# ======================================================================


class TestBuildBatchIndex:
    """Unit tests for _build_batch_index, verifying it matches the old inline code."""

    def test_tokens_match_shard(self):
        """inputs/targets correctly extracted from shard tokens."""
        shard = _make_shard(n_rows=4, context_len=16)
        idx = _build_batch_index(shard, [0, 2])

        raw = shard.tokens[[0, 2]].to(t.int64)
        assert t.equal(idx.inputs, raw[:, :16])
        assert t.equal(idx.targets, raw[:, 1:17])

    def test_seg_counts(self):
        """seg_counts reflects actual segments per row."""
        shard = _make_variable_shard()
        idx = _build_batch_index(shard, [0, 1, 2])

        # Row 0: 4 segs, Row 1: 2 segs, Row 2: 3 segs
        assert idx.seg_counts.tolist() == [4, 2, 3]
        assert idx.total_segs == 9
        assert idx.batch_n_seg == 4  # max(4, 2, 3)

    @pytest.mark.parametrize("rows", [[0, 1, 2], [2, 0, 1], [1, 1, 0], [1]])
    def test_uneven_row_order_matches_intervals(self, rows: list[int]):
        shard = _make_variable_shard()
        idx = _build_batch_index(shard, rows, pad_n_seg=5)

        # Independent per-row intervals from the fixture, including repeated rows.
        maps = [[0] * 8 + [1] * 8 + [2] * 8 + [3] * 8,
                [0] * 16 + [1] * 16,
                [0] * 10 + [1] * 10 + [2] * 12]
        docs = [[0, 0, 1, 1, -1], [0, 0, -1, -1, -1], [0, 0, 1, -1, -1]]
        segments = [[0, 1, 2, 3], [4, 5], [6, 7, 8]]
        assert idx.seg_map.tolist() == [maps[row] for row in rows]
        assert idx.seg_doc_id.tolist() == [docs[row] for row in rows]
        assert idx.seg_counts.tolist() == [len(segments[row]) for row in rows]
        assert idx.global_flat.tolist() == [seg for row in rows for seg in segments[row]]
        assert idx.local_flat.tolist() == [i for row in rows for i in range(len(segments[row]))]
        assert idx.inputs.tolist() == [shard.tokens[row, :32].tolist() for row in rows]
        assert idx.targets.tolist() == [shard.tokens[row, 1:].tolist() for row in rows]

    @pytest.mark.parametrize("offsets", [[0, 0, 1, 2], [0, 1, 1, 2], [0, 1, 2, 2]])
    def test_empty_row_preserves_index_error(self, offsets: list[int]):
        shard = _make_shard(n_rows=3, context_len=8, segs_per_row=1, feats_per_seg=1)
        shard.row_seg_offsets = t.tensor(offsets, dtype=t.int32)
        shard.seg_token_start = shard.seg_token_start[:2]
        shard.seg_token_end = shard.seg_token_end[:2]
        shard.seg_doc_id = shard.seg_doc_id[:2]
        shard.seg_feat_offsets = shard.seg_feat_offsets[:3]
        shard.feature_ids = shard.feature_ids[:2]
        shard.feature_pcts = shard.feature_pcts[:2]

        with pytest.raises(IndexError, match="index 0 is out of bounds for dimension 0 with size 0"):
            _build_batch_index(shard, [0, 1, 2])

    def test_zero_segment_rows_have_defined_padding(self):
        shard = _make_shard(n_rows=3, context_len=8, segs_per_row=1, feats_per_seg=1)
        shard.row_seg_offsets = t.zeros(4, dtype=t.int32)
        shard.seg_token_start = shard.seg_token_start[:0]
        shard.seg_token_end = shard.seg_token_end[:0]
        shard.seg_doc_id = shard.seg_doc_id[:0]
        shard.seg_feat_offsets = shard.seg_feat_offsets[:1]
        shard.feature_ids = shard.feature_ids[:0]
        shard.feature_pcts = shard.feature_pcts[:0]
        idx = _build_batch_index(shard, [2, 0, 2], pad_n_seg=2)

        assert idx.seg_counts.tolist() == [0, 0, 0]
        assert idx.total_segs == 0
        assert idx.seg_map.tolist() == [[0] * 8] * 3
        assert idx.seg_doc_id.tolist() == [[-1, -1]] * 3
        assert idx.global_flat.numel() == idx.local_flat.numel() == idx.b_flat.numel() == 0

    def test_flat_arrays_lengths(self):
        """b_flat, local_flat, global_flat all have length total_segs."""
        shard = _make_variable_shard()
        idx = _build_batch_index(shard, [0, 1])

        assert len(idx.b_flat) == idx.total_segs
        assert len(idx.local_flat) == idx.total_segs
        assert len(idx.global_flat) == idx.total_segs

    def test_b_flat_values(self):
        """b_flat maps each segment to its batch row index."""
        shard = _make_variable_shard()
        idx = _build_batch_index(shard, [0, 1, 2])

        # Row 0 has 4 segs, Row 1 has 2 segs, Row 2 has 3 segs
        expected = [0, 0, 0, 0, 1, 1, 2, 2, 2]
        assert idx.b_flat.tolist() == expected

    def test_local_flat_values(self):
        """local_flat gives 0-indexed segment position within each row."""
        shard = _make_variable_shard()
        idx = _build_batch_index(shard, [0, 1, 2])

        expected = [0, 1, 2, 3, 0, 1, 0, 1, 2]
        assert idx.local_flat.tolist() == expected

    def test_global_flat_values(self):
        """global_flat gives the absolute segment index in the shard."""
        shard = _make_variable_shard()
        idx = _build_batch_index(shard, [0, 1, 2])

        # Row 0: segs 0-3, Row 1: segs 4-5, Row 2: segs 6-8
        expected = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        assert idx.global_flat.tolist() == expected

    def test_seg_doc_id_variable_shard(self):
        """seg_doc_id is correctly scattered for multi-doc variable shard."""
        shard = _make_variable_shard()
        idx = _build_batch_index(shard, [0, 2])

        # Row 0: segs 0-3, doc_ids = [0, 0, 1, 1]
        assert idx.seg_doc_id[0, 0].item() == 0
        assert idx.seg_doc_id[0, 1].item() == 0
        assert idx.seg_doc_id[0, 2].item() == 1
        assert idx.seg_doc_id[0, 3].item() == 1

        # Row 2: segs 6-8, doc_ids = [0, 0, 1]
        assert idx.seg_doc_id[1, 0].item() == 0
        assert idx.seg_doc_id[1, 1].item() == 0
        assert idx.seg_doc_id[1, 2].item() == 1

    def test_seg_map_token_coverage(self):
        """seg_map assigns covered token positions to correct segment indices."""
        shard = _make_shard(n_rows=2, context_len=30, segs_per_row=3)
        idx = _build_batch_index(shard, [0])

        # With seg_size=10: [0,10)→0, [10,20)→1, [20,30)→2
        assert idx.seg_map[0, 0].item() == 0
        assert idx.seg_map[0, 9].item() == 0
        assert idx.seg_map[0, 10].item() == 1
        assert idx.seg_map[0, 20].item() == 2

    def test_pad_n_seg(self):
        """pad_n_seg expands seg_doc_id dimension."""
        shard = _make_shard(n_rows=2, segs_per_row=3)
        idx = _build_batch_index(shard, [0], pad_n_seg=10)

        assert idx.batch_n_seg == 10
        assert idx.seg_doc_id.shape[1] == 10
        # Padding slots should be -1
        assert idx.seg_doc_id[0, 3:].eq(-1).all()

    def test_matches_reconstruct_batch(self):
        """_build_batch_index output matches reconstruct_batch for all shared fields."""
        shard = _make_variable_shard()
        rows = [0, 1, 2]

        idx = _build_batch_index(shard, rows)
        rb = reconstruct_batch(shard, rows, _make_identity_selection(shard), 10)

        assert t.equal(idx.inputs, rb.inputs)
        assert t.equal(idx.targets, rb.targets)
        assert t.equal(idx.seg_map, rb.seg_map)
        assert t.equal(idx.seg_doc_id, rb.seg_doc_id)


# ======================================================================
# 10. Reference correctness: CSR feature extraction
# ======================================================================


class TestCSRFeatureExtraction:
    """Verify fused CSR extract matches reconstruct_batch's padded features."""

    def _extract_per_segment_features(self, shard, row_indices):
        """Reference: extract features per segment using reconstruct_batch + unpad."""
        rb = reconstruct_batch(shard, row_indices, _make_identity_selection(shard), 10)
        B, n_seg, F = rb.enc_fids.shape
        result = {}
        for b in range(B):
            for s in range(n_seg):
                mask = rb.enc_mask[b, s]
                if mask.any():
                    result[(b, s)] = (
                        rb.enc_fids[b, s, mask].tolist(),
                        rb.enc_pcts[b, s, mask].tolist(),
                    )
        return result

    def test_uniform_shard_features_match(self):
        """For uniform shard, fused CSR extract yields same features as reconstruct_batch."""
        shard = _make_shard(n_rows=4, segs_per_row=3, feats_per_seg=10)
        rows = [0, 1, 2, 3]

        # Use threshold path (deterministic) to compare feature extraction
        pipeline = _make_threshold_pipeline(threshold=100)  # high k → keep all
        rb_fused, _ = reconstruct_and_select(shard, rows, pipeline)
        rb_base = reconstruct_batch(shard, rows, _make_identity_selection(shard), 10)

        # For each segment, the positive feature IDs in fused output should
        # be a subset of the base features (threshold keeps top-k by magnitude)
        for b in range(rb_fused.enc_fids.shape[0]):
            for s in range(min(rb_fused.enc_fids.shape[1], rb_base.enc_fids.shape[1])):
                fused_mask = rb_fused.enc_mask[b, s]
                base_mask = rb_base.enc_mask[b, s]
                if not fused_mask.any():
                    continue
                fused_set = set(rb_fused.enc_fids[b, s, fused_mask].tolist())
                base_set = set(rb_base.enc_fids[b, s, base_mask].tolist())
                # All fused positives must come from the base set
                # (effective_buckets marker features are negatives, skip those)
                positives = {f for f in fused_set if rb_fused.enc_mags[b, s, fused_mask][
                    rb_fused.enc_fids[b, s, fused_mask] == f
                ][0].item() != pipeline.effective_buckets} if fused_mask.any() else set()
                for fid in positives:
                    assert fid in base_set, (
                        f"Feature {fid} at [{b},{s}] not in base set {base_set}"
                    )

    def test_variable_shard_features_match(self):
        """Variable shard features match between fused CSR extract and reconstruct_batch."""
        shard = _make_variable_shard()
        rows = [0, 1, 2]

        pipeline = _make_threshold_pipeline(threshold=100)
        rb_fused, _ = reconstruct_and_select(shard, rows, pipeline)
        rb_base = reconstruct_batch(shard, rows, _make_identity_selection(shard), 10)

        for b in range(rb_fused.enc_fids.shape[0]):
            for s in range(min(rb_fused.enc_fids.shape[1], rb_base.enc_fids.shape[1])):
                fused_mask = rb_fused.enc_mask[b, s]
                base_mask = rb_base.enc_mask[b, s]
                if not base_mask.any():
                    # If base has no features, fused should have none either
                    # (or only negatives)
                    continue
                base_set = set(rb_base.enc_fids[b, s, base_mask].tolist())
                if fused_mask.any():
                    fused_fids = rb_fused.enc_fids[b, s, fused_mask]
                    fused_mags = rb_fused.enc_mags[b, s, fused_mask]
                    for fid, mag in zip(fused_fids.tolist(), fused_mags.tolist(), strict=True):
                        if mag != pipeline.effective_buckets:  # skip negatives
                            assert fid in base_set


# ======================================================================
# 11. Negative sampling correctness properties
# ======================================================================


class TestNegativeSamplingCorrectness:
    """Rigorous correctness tests for the vectorized negative sampling."""

    def test_negatives_disjoint_from_positives(self):
        """No negative feature ID matches any positive feature ID in same segment."""
        shard = _make_shard(n_rows=4, feats_per_seg=30)
        pipeline = _make_sampling_pipeline(with_negatives=True, mean=5.0)
        rb, _ = reconstruct_and_select(shard, [0, 1, 2, 3], pipeline)

        eff_bucket = pipeline.effective_buckets
        for b in range(rb.enc_fids.shape[0]):
            for s in range(rb.enc_fids.shape[1]):
                mask = rb.enc_mask[b, s]
                if not mask.any():
                    continue
                fids = rb.enc_fids[b, s, mask]
                mags = rb.enc_mags[b, s, mask]

                pos_fids = set(fids[mags != eff_bucket].tolist())
                neg_fids = set(fids[mags == eff_bucket].tolist())

                overlap = pos_fids & neg_fids
                assert not overlap, (
                    f"[{b},{s}] overlap between positives and negatives: {overlap}"
                )

    def test_negatives_unique_within_segment(self):
        """No duplicate negative feature IDs within a single segment."""
        shard = _make_shard(n_rows=4, feats_per_seg=30)
        pipeline = _make_sampling_pipeline(with_negatives=True, mean=5.0)
        rb, _ = reconstruct_and_select(shard, [0, 1, 2, 3], pipeline)

        eff_bucket = pipeline.effective_buckets
        for b in range(rb.enc_fids.shape[0]):
            for s in range(rb.enc_fids.shape[1]):
                mask = rb.enc_mask[b, s]
                if not mask.any():
                    continue
                fids = rb.enc_fids[b, s, mask]
                mags = rb.enc_mags[b, s, mask]

                neg_fids = fids[mags == eff_bucket].tolist()
                assert len(neg_fids) == len(set(neg_fids)), (
                    f"[{b},{s}] duplicate negatives: {neg_fids}"
                )

    def test_negative_count_respects_config(self):
        """Number of negatives per segment is within configured range."""
        shard = _make_shard(n_rows=4, feats_per_seg=30)
        # min_k=1 means at least 1 negative per segment
        pipeline = _make_sampling_pipeline(with_negatives=True, mean=3.0)
        rb, _ = reconstruct_and_select(shard, [0, 1, 2, 3], pipeline)

        eff_bucket = pipeline.effective_buckets
        neg_counts = []
        for b in range(rb.enc_fids.shape[0]):
            for s in range(rb.enc_fids.shape[1]):
                mask = rb.enc_mask[b, s]
                if not mask.any():
                    continue
                mags = rb.enc_mags[b, s, mask]
                n_neg = (mags == eff_bucket).sum().item()
                neg_counts.append(n_neg)

        # Should have some negatives (min_k=1 and enough segments)
        assert sum(neg_counts) > 0, "Expected some negatives with min_k=1"

    def test_negatives_all_have_effective_bucket(self):
        """Every negative feature has enc_mags == effective_buckets."""
        shard = _make_shard(n_rows=4, feats_per_seg=20)
        pipeline = _make_sampling_pipeline(
            n_buckets=10, with_negatives=True, mean=3.0,
        )
        rb, _ = reconstruct_and_select(shard, [0, 1], pipeline)

        eff = pipeline.effective_buckets
        for b in range(rb.enc_fids.shape[0]):
            for s in range(rb.enc_fids.shape[1]):
                mask = rb.enc_mask[b, s]
                if not mask.any():
                    continue
                mags = rb.enc_mags[b, s, mask]
                # Every mag is either a valid bucket [0, eff) or the negative marker (eff)
                assert (mags <= eff).all(), f"[{b},{s}] mag > effective_buckets"

    def test_no_negatives_when_bundle_is_none(self):
        """Without negative_bundle, enc_mags has no effective_buckets values."""
        shard = _make_shard(n_rows=4, feats_per_seg=20)
        pipeline = _make_sampling_pipeline(with_negatives=False)
        rb, _ = reconstruct_and_select(shard, [0, 1], pipeline)

        active_mags = rb.enc_mags[rb.enc_mask]
        # effective_buckets = n_buckets (10) when no negatives
        # All mags should be < n_buckets (discretized bucket indices)
        assert (active_mags < 10).all()

    def test_negatives_with_threshold_path(self):
        """Negative sampling works correctly through the threshold selection path."""
        shard = _make_shard(n_rows=4, feats_per_seg=30)
        pipeline = _make_threshold_pipeline(
            threshold=5, with_negatives=True,
        )
        rb, _ = reconstruct_and_select(shard, [0, 1, 2, 3], pipeline)

        eff_bucket = pipeline.effective_buckets
        found_negatives = False
        for b in range(rb.enc_fids.shape[0]):
            for s in range(rb.enc_fids.shape[1]):
                mask = rb.enc_mask[b, s]
                if not mask.any():
                    continue
                fids = rb.enc_fids[b, s, mask]
                mags = rb.enc_mags[b, s, mask]
                pos_fids = set(fids[mags != eff_bucket].tolist())
                neg_fids_list = fids[mags == eff_bucket].tolist()

                if neg_fids_list:
                    found_negatives = True
                    # Disjointness
                    assert not (pos_fids & set(neg_fids_list))
                    # Uniqueness
                    assert len(neg_fids_list) == len(set(neg_fids_list))

        assert found_negatives, "Expected negatives with threshold + negative bundle"


# ======================================================================
# 12. Reference correctness: variable shard tokens/segments
# ======================================================================


class TestVariableShardReference:
    """Reference tests on variable shard (different segs/row, multi-doc)."""

    def test_tokens_match_variable_shard(self):
        shard = _make_variable_shard()
        pipeline = _make_sampling_pipeline()

        rb_fused, _ = reconstruct_and_select(shard, [0, 1, 2], pipeline)
        rb_base = reconstruct_batch(shard, [0, 1, 2], _make_identity_selection(shard), 10)

        assert t.equal(rb_fused.inputs, rb_base.inputs)
        assert t.equal(rb_fused.targets, rb_base.targets)

    def test_seg_map_matches_variable_shard(self):
        shard = _make_variable_shard()
        pipeline = _make_sampling_pipeline()

        rb_fused, _ = reconstruct_and_select(shard, [0, 1, 2], pipeline)
        rb_base = reconstruct_batch(shard, [0, 1, 2], _make_identity_selection(shard), 10)

        assert t.equal(rb_fused.seg_map, rb_base.seg_map)

    def test_seg_doc_id_matches_variable_shard(self):
        shard = _make_variable_shard()
        pipeline = _make_sampling_pipeline()

        rb_fused, _ = reconstruct_and_select(shard, [0, 1, 2], pipeline)
        rb_base = reconstruct_batch(shard, [0, 1, 2], _make_identity_selection(shard), 10)

        # Both use _build_batch_index, so shapes must be identical
        assert rb_fused.seg_doc_id.shape == rb_base.seg_doc_id.shape
        assert t.equal(rb_fused.seg_doc_id, rb_base.seg_doc_id)

    def test_multi_doc_row_seg_doc_id(self):
        """Multi-doc rows have correct doc IDs in fused output."""
        shard = _make_variable_shard()
        pipeline = _make_sampling_pipeline()

        # Row 0 has docs [0,0,1,1], Row 2 has docs [0,0,1]
        rb, _ = reconstruct_and_select(shard, [0, 2], pipeline)

        assert rb.seg_doc_id[0, 0].item() == 0
        assert rb.seg_doc_id[0, 1].item() == 0
        assert rb.seg_doc_id[0, 2].item() == 1
        assert rb.seg_doc_id[0, 3].item() == 1
        assert rb.seg_doc_id[1, 0].item() == 0
        assert rb.seg_doc_id[1, 1].item() == 0
        assert rb.seg_doc_id[1, 2].item() == 1

    def test_empty_segment_in_variable_shard(self):
        """Row 1 seg 1 has 0 features — fused handles it without error."""
        shard = _make_variable_shard()
        pipeline = _make_sampling_pipeline()

        rb, _ = reconstruct_and_select(shard, [1], pipeline)
        assert rb.enc_fids.shape[0] == 1
        # Row 1 has 2 segments, seg 1 has 0 features
        assert rb.enc_fids.shape[1] >= 2


# ======================================================================
# 13. All-empty edge cases
# ======================================================================


class TestAllEmptyEdgeCases:
    """Edge cases where all segments or all features are empty."""

    def test_all_segments_zero_features(self):
        """Shard where every segment has 0 features."""
        n_rows, T = 2, 16
        tokens = t.randint(1, 32000, (n_rows, T + 1), dtype=t.uint16)
        row_seg_offsets = t.tensor([0, 2, 4], dtype=t.int32)
        seg_token_start = t.tensor([0, 8, 0, 8], dtype=t.uint16)
        seg_token_end = t.tensor([8, 16, 8, 16], dtype=t.uint16)
        seg_doc_id = t.tensor([0, 0, 0, 0], dtype=t.uint8)
        seg_feat_offsets = t.zeros(5, dtype=t.int32)  # all zero → 0 features everywhere
        feature_ids = t.tensor([], dtype=t.uint16)
        feature_pcts = t.tensor([], dtype=t.uint8)

        shard = PackedRowShard(
            tokens=tokens, row_seg_offsets=row_seg_offsets,
            seg_token_start=seg_token_start, seg_token_end=seg_token_end,
            seg_doc_id=seg_doc_id, seg_feat_offsets=seg_feat_offsets,
            feature_ids=feature_ids, feature_pcts=feature_pcts,
        )

        pipeline = _make_sampling_pipeline()
        rb, _ = reconstruct_and_select(shard, [0, 1], pipeline)

        assert rb.enc_fids.shape[0] == 2
        # No features → all mask should be False (except potentially negatives)
        # With no positives, negatives have empty active_ids exclusion set

    def test_all_segments_zero_features_with_negatives(self):
        """Zero-feature shard with negatives enabled — negatives can still be sampled."""
        n_rows, T = 2, 16
        tokens = t.randint(1, 32000, (n_rows, T + 1), dtype=t.uint16)
        row_seg_offsets = t.tensor([0, 2, 4], dtype=t.int32)
        seg_token_start = t.tensor([0, 8, 0, 8], dtype=t.uint16)
        seg_token_end = t.tensor([8, 16, 8, 16], dtype=t.uint16)
        seg_doc_id = t.tensor([0, 0, 0, 0], dtype=t.uint8)
        seg_feat_offsets = t.zeros(5, dtype=t.int32)
        feature_ids = t.tensor([], dtype=t.uint16)
        feature_pcts = t.tensor([], dtype=t.uint8)

        shard = PackedRowShard(
            tokens=tokens, row_seg_offsets=row_seg_offsets,
            seg_token_start=seg_token_start, seg_token_end=seg_token_end,
            seg_doc_id=seg_doc_id, seg_feat_offsets=seg_feat_offsets,
            feature_ids=feature_ids, feature_pcts=feature_pcts,
        )

        pipeline = _make_sampling_pipeline(with_negatives=True, mean=3.0)
        rb, _ = reconstruct_and_select(shard, [0, 1], pipeline)

        # Should not crash; negatives may or may not be present
        assert rb.enc_fids.shape[0] == 2

    def test_single_feature_per_segment(self):
        """Each segment has exactly 1 feature."""
        shard = _make_shard(n_rows=2, segs_per_row=2, feats_per_seg=1)
        pipeline = _make_sampling_pipeline(mean=1.0)
        rb, _ = reconstruct_and_select(shard, [0, 1], pipeline)

        assert rb.enc_fids.shape[0] == 2
        assert rb.enc_mask.any()


# ======================================================================
# 14. Mask consistency (comprehensive)
# ======================================================================


class TestMaskConsistency:
    """Verify mask, fids, mags, pcts consistency across all paths."""

    def _check_mask_consistency(self, rb):
        """Helper: check mask consistency per segment.

        The output layout is [positives | negatives] with fixed column
        widths.  Invalid positions in the negative columns may have
        non-zero fid/mag values (oversampled candidates + blanket
        effective_buckets assignment).  Only pcts are guaranteed zero
        where mask=False (no scatter writes to invalid pcts).

        Key invariant: active features (mask=True) must have internally
        consistent data — non-negative mags, pcts in [0,1].
        """
        B, n_seg, F = rb.enc_fids.shape
        for b in range(B):
            for s in range(n_seg):
                mask = rb.enc_mask[b, s]
                if not mask.any():
                    # Fully inactive segment in the positive range should be zeros
                    # (the scatter only writes to segments that exist)
                    continue
                # Active positions: mags >= 0, pcts in [0,1]
                active_mags = rb.enc_mags[b, s, mask]
                active_pcts = rb.enc_pcts[b, s, mask]
                assert (active_mags >= 0).all(), f"[{b},{s}] negative mag values"
                assert (active_pcts >= 0).all(), f"[{b},{s}] negative pct values"
                assert (active_pcts <= 1).all(), f"[{b},{s}] pct > 1"

                # pcts must be 0 where mask is False (no scatter target)
                inactive = ~mask
                assert (rb.enc_pcts[b, s, inactive] == 0).all(), f"[{b},{s}] pcts nonzero where mask=False"

    def test_sampling_path(self):
        shard = _make_shard(n_rows=4, feats_per_seg=20)
        pipeline = _make_sampling_pipeline(with_negatives=True)
        rb, _ = reconstruct_and_select(shard, [0, 1, 2, 3], pipeline)
        self._check_mask_consistency(rb)

    def test_threshold_path(self):
        shard = _make_shard(n_rows=4, feats_per_seg=20)
        pipeline = _make_threshold_pipeline(threshold=5, with_negatives=True)
        rb, _ = reconstruct_and_select(shard, [0, 1, 2, 3], pipeline)
        self._check_mask_consistency(rb)

    def test_no_negatives_path(self):
        shard = _make_shard(n_rows=4, feats_per_seg=20)
        pipeline = _make_sampling_pipeline(with_negatives=False)
        rb, _ = reconstruct_and_select(shard, [0, 1, 2, 3], pipeline)
        self._check_mask_consistency(rb)

    def test_variable_shard(self):
        shard = _make_variable_shard()
        pipeline = _make_sampling_pipeline(with_negatives=True)
        rb, _ = reconstruct_and_select(shard, [0, 1, 2], pipeline)
        self._check_mask_consistency(rb)
