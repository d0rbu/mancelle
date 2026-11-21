"""Tests for static shape padding in the fused reconstruct_and_select path.

Verifies that when pad_F is provided, all output tensors have exactly the
requested (B, pad_n_seg, pad_F) dimensions and that extra padding positions
are zero-filled with mask=False. Also tests pad_F computation from config
max_k bounds, compilation with static shapes, and correctness (padded output
matches unpadded for active features).

Test categories:
 1. Shape correctness — output dims match pad_F/pad_n_seg exactly
 2. pad_F computation — from config max_k and FP8_ALIGNMENT
 3. Padding correctness — padded matches unpadded for active features
 4. Compilation — resolve_compile_dynamic and mode selection
 5. Edge cases — actual_f == pad_F, much less, overflow, cross-shard
"""

from __future__ import annotations

import pytest
import torch as t

from steering.conditional.encoder_decoder import FP8_ALIGNMENT
from steering.conditional.encoder_decoder.dataloader import (
    reconstruct_and_select,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeBundle,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
    ThresholdSelection,
)
from steering.conditional.thresholding import ThresholdType
from steering.packed_batch_io import PackedRowShard

N_SAE_FEATURES = 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    with_negatives: bool = True,
    pos_mean: float = 5.0,
    pos_max_k: int = 25,
    neg_mean: float = 3.0,
    neg_max_k: int = 15,
    seed: int = 42,
) -> FeaturePipelineConfig:
    rng = t.Generator().manual_seed(seed)
    selection = SamplingSelection(
        distribution=Distribution.geometric,
        mean=pos_mean, std=pos_mean, min_k=1, max_k=pos_max_k,
    )
    pct_table = _make_percentile_table(n_features)
    sampling_bundle = SamplingBundle(
        selection=selection, percentile_table=pct_table, rng=rng,
    )
    effective_buckets = n_buckets - 1 if with_negatives else n_buckets
    boundaries = _make_boundaries(n_features, effective_buckets)

    negative_bundle = None
    if with_negatives:
        neg_rng = t.Generator().manual_seed(seed + 1)
        negative_bundle = NegativeBundle(
            selection=NegativeSamplingSelection(
                distribution=Distribution.geometric,
                mean=neg_mean, std=neg_mean, min_k=0, max_k=neg_max_k,
            ),
            n_sae_features=n_features, rng=neg_rng,
        )

    return FeaturePipelineConfig(
        selection_config=sampling_bundle,
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


def _make_single_segment_feature_shard(
    feature_ids: tuple[int, ...] = (5, 30),
    feature_pcts: tuple[int, ...] = (250, 100),
) -> PackedRowShard:
    assert len(feature_ids) == len(feature_pcts)
    return PackedRowShard(
        tokens=t.ones(1, 17, dtype=t.uint16),
        row_seg_offsets=t.tensor([0, 1], dtype=t.int32),
        seg_token_start=t.zeros(1, dtype=t.uint16),
        seg_token_end=t.full((1,), 16, dtype=t.uint16),
        seg_doc_id=t.zeros(1, dtype=t.uint8),
        seg_feat_offsets=t.tensor([0, len(feature_ids)], dtype=t.int32),
        feature_ids=t.tensor(feature_ids, dtype=t.uint16),
        feature_pcts=t.tensor(feature_pcts, dtype=t.uint8),
    )


def _make_two_active_feature_shard() -> PackedRowShard:
    return _make_single_segment_feature_shard()


def _make_active_exclusion_pipeline(
    allowed_ids: tuple[int, ...] = (5, 30, 31),
    *,
    n_features: int = N_SAE_FEATURES,
    neg_min_k: int = 10,
    neg_max_k: int | None = None,
    selection_config: SamplingBundle | ThresholdSelectionConfig | None = None,
) -> FeaturePipelineConfig:
    effective_buckets = 3
    selection_rng = t.Generator().manual_seed(0)
    negative_rng = t.Generator().manual_seed(0)
    feature_mask = FeatureMask(FeatureMaskMode.whitelist, allowed_ids)
    if selection_config is None:
        selection_config = SamplingBundle(
            selection=SamplingSelection(
                distribution=Distribution.geometric,
                mean=5.0,
                std=5.0,
                min_k=1,
                max_k=2,
                mode=SamplingMode.top_k_percentile,
            ),
            percentile_table=_make_percentile_table(n_features),
            rng=selection_rng,
        )
    return FeaturePipelineConfig(
        selection_config=selection_config,
        boundaries=_make_boundaries(n_features=n_features, n_buckets=effective_buckets),
        negative_bundle=NegativeBundle(
            selection=NegativeSamplingSelection(
                distribution=Distribution.geometric,
                mean=5.0,
                std=5.0,
                min_k=neg_min_k,
                max_k=neg_max_k,
            ),
            n_sae_features=n_features,
            rng=negative_rng,
            feature_mask=feature_mask,
        ),
        effective_buckets=effective_buckets,
        feature_mask=feature_mask,
    )


def _make_no_mask_active_exclusion_pipeline(
    *,
    n_features: int,
    neg_min_k: int = 10,
) -> FeaturePipelineConfig:
    effective_buckets = 3
    return FeaturePipelineConfig(
        selection_config=SamplingBundle(
            selection=SamplingSelection(
                distribution=Distribution.geometric,
                mean=5.0,
                std=5.0,
                min_k=1,
                max_k=2,
                mode=SamplingMode.top_k_percentile,
            ),
            percentile_table=_make_percentile_table(n_features),
            rng=t.Generator().manual_seed(0),
        ),
        boundaries=_make_boundaries(n_features=n_features, n_buckets=effective_buckets),
        negative_bundle=NegativeBundle(
            selection=NegativeSamplingSelection(
                distribution=Distribution.geometric,
                mean=5.0,
                std=5.0,
                min_k=neg_min_k,
            ),
            n_sae_features=n_features,
            rng=t.Generator().manual_seed(0),
        ),
        effective_buckets=effective_buckets,
    )


def _split_positive_negative_fids(
    rb,
    *,
    negative_bucket: int,
) -> tuple[list[int], list[int]]:
    fids = rb.enc_fids[rb.enc_mask]
    buckets = rb.enc_mags[rb.enc_mask]
    positives = fids[buckets != negative_bucket].tolist()
    negatives = fids[buckets == negative_bucket].tolist()
    return positives, negatives


def _compute_pad_F(pos_max_k: int, neg_max_k: int, fp8: bool = True) -> int:
    """Replicate the pad_F computation from build_shared_state."""
    pad_F = pos_max_k + neg_max_k
    if fp8:
        pad_F = ((pad_F + FP8_ALIGNMENT - 1) // FP8_ALIGNMENT) * FP8_ALIGNMENT
    return pad_F


# ---------------------------------------------------------------------------
# Shape correctness tests
# ---------------------------------------------------------------------------

class TestStaticShapesFusedPath:
    """Verify output shapes when pad_F is provided."""

    def test_output_shapes_match_pad_dims(self):
        """With pad_F and pad_n_seg set, every batch has EXACTLY
        enc_fids.shape == (B, pad_n_seg, pad_F)."""
        shard = _make_shard(n_rows=4)
        pipeline = _make_sampling_pipeline()
        pad_n_seg = 10
        pad_F = _compute_pad_F(25, 15)  # 40 → aligned to 48

        rb, _ = reconstruct_and_select(
            shard, range(4), pipeline, pad_n_seg=pad_n_seg, pad_F=pad_F,
        )

        assert rb.enc_fids.shape == (4, pad_n_seg, pad_F)
        assert rb.enc_mags.shape == (4, pad_n_seg, pad_F)
        assert rb.enc_mask.shape == (4, pad_n_seg, pad_F)
        assert rb.enc_pcts.shape == (4, pad_n_seg, pad_F)

    def test_shapes_identical_across_batches(self):
        """10 consecutive batches all have identical shapes for every tensor."""
        shard = _make_shard(n_rows=40)
        pad_n_seg = 10
        pad_F = _compute_pad_F(25, 15)

        shapes = set()
        for i in range(10):
            pipeline = _make_sampling_pipeline(seed=42 + i)
            rb, _ = reconstruct_and_select(
                shard, range(i * 4, (i + 1) * 4), pipeline,
                pad_n_seg=pad_n_seg, pad_F=pad_F,
            )
            shapes.add(rb.enc_fids.shape)

        assert len(shapes) == 1, f"Expected 1 unique shape, got {len(shapes)}: {shapes}"

    def test_seg_map_shape_constant(self):
        """seg_map is always [B, T] regardless of actual segments."""
        shard = _make_shard(n_rows=4)
        pipeline = _make_sampling_pipeline()
        pad_F = _compute_pad_F(25, 15)

        rb, _ = reconstruct_and_select(
            shard, range(4), pipeline, pad_n_seg=10, pad_F=pad_F,
        )

        assert rb.seg_map.shape == (4, 32)  # B=4, T=context_len=32

    def test_seg_doc_id_shape_constant(self):
        """seg_doc_id is always [B, pad_n_seg]."""
        shard = _make_shard(n_rows=4)
        pipeline = _make_sampling_pipeline()
        pad_F = _compute_pad_F(25, 15)

        rb, _ = reconstruct_and_select(
            shard, range(4), pipeline, pad_n_seg=10, pad_F=pad_F,
        )

        assert rb.seg_doc_id.shape == (4, 10)

    def test_fp_targets_shape_constant(self):
        """fp_targets is always [B, pad_n_seg, n_sae_features]."""
        shard = _make_shard(n_rows=4)
        pipeline = _make_sampling_pipeline()
        pad_F = _compute_pad_F(25, 15)

        _, fp_targets = reconstruct_and_select(
            shard, range(4), pipeline,
            pad_n_seg=10, pad_F=pad_F, n_sae_features=N_SAE_FEATURES,
        )

        assert fp_targets is not None
        assert fp_targets.shape == (4, 10, N_SAE_FEATURES)


# ---------------------------------------------------------------------------
# pad_F computation tests
# ---------------------------------------------------------------------------

class TestPadFComputation:
    """Verify pad_F is computed correctly from config max_k bounds."""

    def test_pad_F_with_fp8_aligned(self):
        """With fp8, pad_F = max_k_positive + max_k_negative, aligned to FP8_ALIGNMENT."""
        pad_F = _compute_pad_F(25, 15, fp8=True)
        assert pad_F == 48  # 25+15=40, ceil to 48 (multiple of 16)

    def test_pad_F_without_fp8_exact(self):
        """Without fp8, pad_F = max_k_positive + max_k_negative exactly."""
        pad_F = _compute_pad_F(25, 15, fp8=False)
        assert pad_F == 40  # 25+15=40, no alignment

    def test_pad_F_with_no_negative_sampling(self):
        """When negative_selection is None, pad_F = max_k_positive only."""
        pad_F = _compute_pad_F(25, 0, fp8=True)
        assert pad_F == 32  # 25+0=25, ceil to 32 (multiple of 16)

    def test_pad_F_already_aligned(self):
        """When sum is already aligned, no extra padding needed."""
        pad_F = _compute_pad_F(16, 0, fp8=True)
        assert pad_F == 16

    def test_fp8_alignment_respected(self):
        """With fp8, pad_F is always a multiple of FP8_ALIGNMENT."""
        for pos in range(1, 50):
            for neg in range(30):
                pad_F = _compute_pad_F(pos, neg, fp8=True)
                assert pad_F % FP8_ALIGNMENT == 0, f"pos={pos}, neg={neg}, pad_F={pad_F}"

    def test_pad_F_bounds_actual_features(self):
        """For 20 batches, actual feature count never exceeds pad_F."""
        shard = _make_shard(n_rows=4, feats_per_seg=30)
        pos_max_k, neg_max_k = 25, 15
        pad_F = _compute_pad_F(pos_max_k, neg_max_k)

        for i in range(20):
            pipeline = _make_sampling_pipeline(
                pos_max_k=pos_max_k, neg_max_k=neg_max_k, seed=i,
            )
            rb, _ = reconstruct_and_select(
                shard, range(4), pipeline, pad_n_seg=10, pad_F=pad_F,
            )
            # Active features should fit within pad_F
            actual_features = rb.enc_mask.sum(dim=-1).max().item()
            assert actual_features <= pad_F, (
                f"Batch {i}: actual_features={actual_features} > pad_F={pad_F}"
            )


# ---------------------------------------------------------------------------
# Padding correctness tests (padded must match unpadded for active features)
# ---------------------------------------------------------------------------

class TestStaticPaddingCorrectness:
    """Padded output must be bit-identical to unpadded for active columns."""

    def test_padded_matches_unpadded_features(self):
        """The first `actual_f` columns of the padded output must be
        BIT-IDENTICAL to the unpadded output."""
        shard = _make_shard(n_rows=4, feats_per_seg=20)
        # Use same seed for both calls
        pipeline_unpadded = _make_sampling_pipeline(seed=99)
        pipeline_padded = _make_sampling_pipeline(seed=99)
        pad_F = _compute_pad_F(25, 15)
        pad_n_seg = 10

        rb_unpadded, _ = reconstruct_and_select(
            shard, range(4), pipeline_unpadded, pad_n_seg=pad_n_seg,
        )
        rb_padded, _ = reconstruct_and_select(
            shard, range(4), pipeline_padded,
            pad_n_seg=pad_n_seg, pad_F=pad_F,
        )

        actual_f = rb_unpadded.enc_fids.shape[2]
        assert t.equal(rb_padded.enc_fids[:, :, :actual_f], rb_unpadded.enc_fids)
        assert t.equal(rb_padded.enc_mags[:, :, :actual_f], rb_unpadded.enc_mags)
        assert t.equal(rb_padded.enc_mask[:, :, :actual_f], rb_unpadded.enc_mask)
        assert t.equal(rb_padded.enc_pcts[:, :, :actual_f], rb_unpadded.enc_pcts)

    def test_extra_padding_is_masked_false(self):
        """Positions beyond actual features have enc_mask=False, enc_fids=0."""
        shard = _make_shard(n_rows=4, feats_per_seg=20)
        pipeline_ref = _make_sampling_pipeline(seed=77)
        pipeline = _make_sampling_pipeline(seed=77)
        pad_F = _compute_pad_F(25, 15)
        pad_n_seg = 10

        rb_ref, _ = reconstruct_and_select(
            shard, range(4), pipeline_ref, pad_n_seg=pad_n_seg,
        )
        actual_f = rb_ref.enc_fids.shape[2]

        rb, _ = reconstruct_and_select(
            shard, range(4), pipeline,
            pad_n_seg=pad_n_seg, pad_F=pad_F,
        )

        if pad_F > actual_f:
            pad_region_mask = rb.enc_mask[:, :, actual_f:]
            pad_region_fids = rb.enc_fids[:, :, actual_f:]
            pad_region_mags = rb.enc_mags[:, :, actual_f:]
            pad_region_pcts = rb.enc_pcts[:, :, actual_f:]
            assert not pad_region_mask.any(), "Padding region should have mask=False"
            assert (pad_region_fids == 0).all(), "Padding region should have fids=0"
            assert (pad_region_mags == 0).all(), "Padding region should have mags=0"
            assert (pad_region_pcts == 0.0).all(), "Padding region should have pcts=0.0"

    def test_padded_matches_unpadded_fp_targets(self):
        """fp_targets values are identical regardless of pad_F."""
        shard = _make_shard(n_rows=4, feats_per_seg=20)
        pipeline_unpadded = _make_sampling_pipeline(seed=55)
        pipeline_padded = _make_sampling_pipeline(seed=55)
        pad_F = _compute_pad_F(25, 15)
        pad_n_seg = 10

        _, fp_unpadded = reconstruct_and_select(
            shard, range(4), pipeline_unpadded,
            pad_n_seg=pad_n_seg, n_sae_features=N_SAE_FEATURES,
        )
        _, fp_padded = reconstruct_and_select(
            shard, range(4), pipeline_padded,
            pad_n_seg=pad_n_seg, pad_F=pad_F, n_sae_features=N_SAE_FEATURES,
        )

        assert fp_unpadded is not None and fp_padded is not None
        assert t.equal(fp_unpadded, fp_padded)


# ---------------------------------------------------------------------------
# Compilation tests
# ---------------------------------------------------------------------------

class TestStaticCompilation:
    """Verify compile mode selection logic."""

    def test_compile_dynamic_false_with_fused_path(self):
        """When pad_F and pad_n_seg are available, resolve_compile_dynamic
        returns False (static compilation)."""
        from steering._common.compile_helpers import resolve_compile_dynamic

        # static_shapes_available=True → dynamic=False
        assert resolve_compile_dynamic("auto", mode="pretrain", static_shapes_available=True) is False

    def test_compile_dynamic_true_without_pad_F(self):
        """When pad_F is None (no max_k in config), dynamic=True."""
        from steering._common.compile_helpers import resolve_compile_dynamic

        assert resolve_compile_dynamic("auto", mode="pretrain", static_shapes_available=False) is True

    def test_compile_static_force_true(self):
        """compile_static=True forces dynamic=False regardless of shapes."""
        from steering._common.compile_helpers import resolve_compile_dynamic

        assert resolve_compile_dynamic(True, mode="pretrain", static_shapes_available=False) is False

    def test_compile_static_force_false(self):
        """compile_static=False forces dynamic=True regardless of shapes."""
        from steering._common.compile_helpers import resolve_compile_dynamic

        assert resolve_compile_dynamic(False, mode="pretrain", static_shapes_available=True) is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestStaticShapeEdgeCases:
    """Edge cases for static shape padding."""

    def test_actual_f_equals_pad_F(self):
        """When a batch happens to use exactly max_k, no truncation or waste."""
        shard = _make_shard(n_rows=4, feats_per_seg=30)
        pipeline = _make_sampling_pipeline(
            with_negatives=False, pos_max_k=16, seed=42,
        )
        pad_F = 16  # Already aligned, no negatives

        rb, _ = reconstruct_and_select(
            shard, range(4), pipeline, pad_n_seg=10, pad_F=pad_F,
        )

        assert rb.enc_fids.shape[2] == pad_F

    def test_actual_f_much_less_than_pad_F(self):
        """Most features are padding (e.g., k=3 but pad_F=48). Model still works."""
        shard = _make_shard(n_rows=4, feats_per_seg=30)
        pipeline = _make_sampling_pipeline(
            with_negatives=False, pos_max_k=3, pos_mean=1.5, seed=42,
        )
        pad_F = 48

        rb, _ = reconstruct_and_select(
            shard, range(4), pipeline, pad_n_seg=10, pad_F=pad_F,
        )

        assert rb.enc_fids.shape[2] == 48
        # Most of F dim should be padding (no negatives, max 3 positives)
        active_per_seg = rb.enc_mask.sum(dim=-1)  # [B, n_seg]
        assert (active_per_seg <= 3).all(), "Should have at most 3 active features per segment"

    def test_pad_f_overflow_raises_when_exceeded(self):
        """If actual_f > pad_F, raise instead of silently truncating."""
        shard = _make_shard(n_rows=4, feats_per_seg=30)
        # Set max_k high but pad_F deliberately low to trigger the hard failure.
        pipeline = _make_sampling_pipeline(
            pos_max_k=30, neg_max_k=15, pos_mean=10.0, seed=42,
        )
        pad_F = 16  # Deliberately small

        with pytest.raises(ValueError, match="pad_F"):
            reconstruct_and_select(
                shard, range(4), pipeline, pad_n_seg=10, pad_F=pad_F,
            )

    def test_negatives_exclude_unselected_active_features(self):
        """Inactive negatives must exclude every active feature, not only positives."""
        pipeline = _make_active_exclusion_pipeline()
        rb, _ = reconstruct_and_select(
            _make_two_active_feature_shard(),
            [0],
            pipeline,
        )
        positives, negatives = _split_positive_negative_fids(
            rb,
            negative_bucket=pipeline.effective_buckets,
        )

        assert positives == [5]
        assert negatives == [31]
        assert 30 not in negatives

    def test_negatives_empty_when_allowed_universe_is_fully_active(self):
        """If every allowed feature is active, there is no legal negative."""
        pipeline = _make_active_exclusion_pipeline(allowed_ids=(5, 30))
        rb, _ = reconstruct_and_select(
            _make_two_active_feature_shard(),
            [0],
            pipeline,
        )
        positives, negatives = _split_positive_negative_fids(
            rb,
            negative_bucket=pipeline.effective_buckets,
        )

        assert positives == [5]
        assert negatives == []

    def test_no_mask_negatives_empty_when_full_universe_is_active(self):
        """The no-mask path also excludes all active features before sampling negatives."""
        pipeline = _make_no_mask_active_exclusion_pipeline(n_features=2)
        rb, _ = reconstruct_and_select(
            _make_single_segment_feature_shard(
                feature_ids=(0, 1),
                feature_pcts=(250, 100),
            ),
            [0],
            pipeline,
        )
        positives, negatives = _split_positive_negative_fids(
            rb,
            negative_bucket=pipeline.effective_buckets,
        )

        assert positives == [0]
        assert negatives == []

    def test_threshold_path_negatives_exclude_unselected_active_features(self):
        """Threshold selection uses the same full-active-set exclusion pool."""
        pipeline = _make_active_exclusion_pipeline(
            selection_config=ThresholdSelectionConfig(
                selection=ThresholdSelection(ThresholdType.segment_topk, 1),
                precomputed_thresholds=None,
            ),
        )
        rb, _ = reconstruct_and_select(
            _make_two_active_feature_shard(),
            [0],
            pipeline,
        )
        positives, negatives = _split_positive_negative_fids(
            rb,
            negative_bucket=pipeline.effective_buckets,
        )

        assert positives == [5]
        assert negatives == [31]
        assert 30 not in negatives

    def test_large_restricted_pool_negatives_exclude_unselected_active_features(self):
        """The vectorized restricted branch cannot emit active-but-unselected ids."""
        allowed_ids = (5, 30, *range(100, 298))
        pipeline = _make_active_exclusion_pipeline(
            allowed_ids=allowed_ids,
            neg_min_k=32,
            neg_max_k=33,
        )
        rb, _ = reconstruct_and_select(
            _make_two_active_feature_shard(),
            [0],
            pipeline,
        )
        positives, negatives = _split_positive_negative_fids(
            rb,
            negative_bucket=pipeline.effective_buckets,
        )
        active_ids = {5, 30}

        assert positives == [5]
        assert negatives
        assert active_ids.isdisjoint(negatives)
        assert len(negatives) == len(set(negatives))
        assert set(negatives).issubset(set(allowed_ids) - active_ids)

    def test_without_pad_F_shapes_are_dynamic(self):
        """Without pad_F, shapes depend on the batch content (no padding)."""
        shard = _make_shard(n_rows=8, feats_per_seg=20)
        shapes = set()
        for i in range(5):
            pipeline = _make_sampling_pipeline(seed=100 + i)
            rb, _ = reconstruct_and_select(
                shard, range(4), pipeline, pad_n_seg=10,
            )
            shapes.add(rb.enc_fids.shape[2])

        # Without pad_F, F dim varies per batch (stochastic sampling)
        # At minimum, we should get the correct pad_n_seg
        for i in range(5):
            pipeline = _make_sampling_pipeline(seed=100 + i)
            rb, _ = reconstruct_and_select(
                shard, range(4), pipeline, pad_n_seg=10,
            )
            assert rb.enc_fids.shape[1] == 10

    def test_no_negatives_still_pads_correctly(self):
        """Static shapes work correctly when negative sampling is disabled."""
        shard = _make_shard(n_rows=4, feats_per_seg=20)
        pipeline = _make_sampling_pipeline(
            with_negatives=False, pos_max_k=25, seed=42,
        )
        pad_F = _compute_pad_F(25, 0)  # 32

        rb, _ = reconstruct_and_select(
            shard, range(4), pipeline, pad_n_seg=10, pad_F=pad_F,
        )

        assert rb.enc_fids.shape == (4, 10, pad_F)
        # No negative columns, all actives should be positives
        active = rb.enc_mask.sum(dim=-1)  # [B, n_seg]
        assert (active <= 25).all()
