"""Tests for the fused reconstruct_and_select path in precomputed_data_loader.

Covers the fused path (feature_pipeline set, no selection shards), verifying:
1. F_max reduction compared to raw feature count per segment
2. Correct dtypes for all output tensors
3. fp_targets shape when n_sae_features is provided
4. SamplingBundle with geometric distribution produces valid batches

Test categories:
 - TestFusedPathThreshold: threshold-based selection (segment_topk)
 - TestFusedPathSampling: stochastic sampling (SamplingBundle/geometric)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch as t

from steering.conditional.delta import DeltaMode, FeatureDeltaConfig
from steering.conditional.encoder_decoder.dataloader import precomputed_data_loader
from steering.conditional.feature_pipeline import (
    DeltaBundle,
    FeaturePipelineConfig,
)
from steering.conditional.sampling import Distribution, SamplingBundle, SamplingSelection
from steering.conditional.splice_counterfactual import packed_row_fingerprint
from steering.overlay_io import OverlayKind, write_manifest
from steering.packed_batch_io import (
    load_packed_row_shard,
    save_packed_row_shard,
    save_selection_shard,
)
from tests.encdec_test_helpers import (
    N_SAE_FEATURES,
    NUM_BUCKETS,
    SEQ_LEN,
    _make_identity_selection,
    _make_test_shard,
    build_threshold_pipeline,
)

# ---------------------------------------------------------------------------
# Shard + loader factory helpers
# ---------------------------------------------------------------------------


def _write_shard_dir(
    tmp_path: Path,
    n_rows: int = 16,
    feats_per_seg: int = 20,
    segs_per_row: int = 3,
    seed: int = 42,
) -> Path:
    """Write a single packed shard + manifest to tmp_path/packed and return path."""
    shard_dir = tmp_path / "packed"
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard = _make_test_shard(
        n_rows=n_rows,
        context_len=SEQ_LEN,
        segs_per_row=segs_per_row,
        feats_per_seg=feats_per_seg,
        n_features=N_SAE_FEATURES,
        seed=seed,
    )
    save_packed_row_shard(shard, shard_dir / "shard_00000")

    manifest_path = shard_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"metadata": {}}))

    return shard_dir


def _write_identity_selection_dir(
    packed_dir: Path,
    *,
    metadata_extra: dict | None = None,
) -> Path:
    sel_dir = packed_dir.parent / "sel"
    sel_dir.mkdir(parents=True, exist_ok=True)
    packed = load_packed_row_shard(packed_dir / "shard_00000.safetensors.zst")
    save_selection_shard(_make_identity_selection(packed), sel_dir / "shard_00000")
    metadata = {
        "packed_rows": {
            "packed_row_fingerprint": packed_row_fingerprint(packed_dir),
        },
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    write_manifest(
        sel_dir,
        "test-selection",
        OverlayKind.PACKED_ROW,
        metadata=metadata,
    )
    return sel_dir


def _make_threshold_pipeline(topk: int = 3) -> FeaturePipelineConfig:
    """Build a FeaturePipelineConfig using segment_topk threshold selection."""
    boundaries = (
        t.linspace(1 / NUM_BUCKETS, 1.0 - 1 / NUM_BUCKETS, NUM_BUCKETS - 1, dtype=t.float64)
        .unsqueeze(0)
        .expand(N_SAE_FEATURES, -1)
        .contiguous()
    )
    return build_threshold_pipeline(boundaries, num_buckets=NUM_BUCKETS, threshold=topk)


def _make_sampling_pipeline_for_test(
    mean: float = 5.0,
    min_k: int = 3,
    max_k: int = 8,
    seed: int = 42,
) -> FeaturePipelineConfig:
    """Build a FeaturePipelineConfig using SamplingBundle with geometric distribution."""
    rng = t.Generator().manual_seed(seed)

    selection = SamplingSelection(
        distribution=Distribution.geometric,
        mean=mean,
        std=mean,
        min_k=min_k,
        max_k=max_k,
    )

    # Build a simple percentile table: sorted random values in [0, 1]
    percentile_table = t.rand(N_SAE_FEATURES, 99).sort(dim=1).values

    sampling_bundle = SamplingBundle(
        selection=selection,
        percentile_table=percentile_table,
        rng=rng,
    )

    boundaries = (
        t.linspace(1 / NUM_BUCKETS, 1.0 - 1 / NUM_BUCKETS, NUM_BUCKETS - 1, dtype=t.float64)
        .unsqueeze(0)
        .expand(N_SAE_FEATURES, -1)
        .contiguous()
    )

    return FeaturePipelineConfig(
        selection_config=sampling_bundle,
        boundaries=boundaries,
        negative_bundle=None,
        effective_buckets=NUM_BUCKETS,
    )


# ---------------------------------------------------------------------------
# TestFusedPathThreshold
# ---------------------------------------------------------------------------


class TestFusedPathThreshold:
    """Fused path with threshold-based selection (segment_topk)."""

    def test_fused_batch_has_reduced_F_max(self, tmp_path):
        """F_max should be much smaller than raw 20 feats/seg.

        With topk=3, we expect enc_fids.shape[2] <= 20 (raw feats_per_seg) and
        specifically the mask should reflect sparse, selected features.
        """
        shard_dir = _write_shard_dir(tmp_path, n_rows=16, feats_per_seg=20)
        pipeline = _make_threshold_pipeline(topk=3)

        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=4,
            device="cpu",
            feature_pipeline=pipeline,
            n_sae_features=N_SAE_FEATURES,
        )

        inputs, targets, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, fp_targets, state = next(loader)

        # The fused path selects only topk=3 features, so F_max should be <= 20
        # (it won't exceed raw feats_per_seg=20, and typically will be much smaller)
        F_max = enc_fids.shape[2]
        assert F_max <= 20, f"Expected F_max <= 20 (raw feats_per_seg), got {F_max}"

        # With topk=3, each segment gets at most 3 features
        active_per_seg = enc_mask.sum(dim=-1)  # [B, n_seg]
        assert (active_per_seg <= 3).all(), (
            f"With topk=3, no segment should have more than 3 active features. "
            f"Max was {active_per_seg.max().item()}"
        )

    def test_fused_batch_dtypes(self, tmp_path):
        """Verify all output tensor dtypes are correct for the fused path.

        inputs/targets: int64, enc_fids: int64, enc_mags: int64,
        enc_mask: bool, enc_pcts: float32, seg_map: int64, seg_doc_id: int64.
        """
        shard_dir = _write_shard_dir(tmp_path, n_rows=16)
        pipeline = _make_threshold_pipeline(topk=3)

        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=4,
            device="cpu",
            feature_pipeline=pipeline,
        )

        inputs, targets, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, _fp_targets, _state = next(loader)

        assert inputs.dtype == t.int64, f"inputs dtype: {inputs.dtype}"
        assert targets.dtype == t.int64, f"targets dtype: {targets.dtype}"
        assert enc_fids.dtype == t.int64, f"enc_fids dtype: {enc_fids.dtype}"
        assert enc_mags.dtype == t.int64, f"enc_mags dtype: {enc_mags.dtype}"
        assert enc_mask.dtype == t.bool, f"enc_mask dtype: {enc_mask.dtype}"
        assert enc_pcts.dtype == t.float32, f"enc_pcts dtype: {enc_pcts.dtype}"
        assert seg_map.dtype == t.int64, f"seg_map dtype: {seg_map.dtype}"
        assert seg_doc_id.dtype == t.int64, f"seg_doc_id dtype: {seg_doc_id.dtype}"

    def test_fused_path_with_fp_targets(self, tmp_path):
        """fp_targets should be non-None with correct shape when n_sae_features is set.

        Expected: fp_targets.shape[0] == B, fp_targets.shape[2] == N_SAE_FEATURES.
        """
        B = 4
        shard_dir = _write_shard_dir(tmp_path, n_rows=16)
        pipeline = _make_threshold_pipeline(topk=3)

        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=B,
            device="cpu",
            feature_pipeline=pipeline,
            n_sae_features=N_SAE_FEATURES,
        )

        _inputs, _targets, _enc_fids, _enc_mags, _enc_mask, _enc_pcts, _seg_map, _seg_doc_id, fp_targets, _state = next(loader)

        assert fp_targets is not None, "fp_targets should not be None when n_sae_features is provided"
        assert fp_targets.shape[0] == B, (
            f"fp_targets batch dim should be {B}, got {fp_targets.shape[0]}"
        )
        assert fp_targets.shape[2] == N_SAE_FEATURES, (
            f"fp_targets last dim should be {N_SAE_FEATURES}, got {fp_targets.shape[2]}"
        )

    def test_feature_delta_fails_fast(self, tmp_path):
        """Packed rows must not silently ignore feature_delta."""
        shard_dir = _write_shard_dir(tmp_path, n_rows=16)
        pipeline = _make_threshold_pipeline(topk=3)
        pipeline.delta_bundle = DeltaBundle(
            config=FeatureDeltaConfig(mode=DeltaMode.prev_segment),
            percentile_table=t.zeros(N_SAE_FEATURES, 99, dtype=t.float64),
        )

        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=4,
            device="cpu",
            feature_pipeline=pipeline,
        )
        with pytest.raises(ValueError, match="does not support feature_delta"):
            next(loader)

    def test_selection_manifest_rejects_wrong_feature_delta(self, tmp_path):
        """A delta-free packed loader must not reuse splice-cf selection shards."""
        shard_dir = _write_shard_dir(tmp_path, n_rows=16)
        sel_dir = _write_identity_selection_dir(
            shard_dir,
            metadata_extra={
                "feature_delta": {
                    "mode": "splice_counterfactual",
                    "neg_delta_ratio": 1.0,
                },
            },
        )
        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=4,
            device="cpu",
            feature_pipeline=_make_threshold_pipeline(topk=3),
            selection_dir=sel_dir,
        )
        with pytest.raises(ValueError, match="feature_delta mismatch"):
            next(loader)

    def test_selection_manifest_rejects_stale_packed_rows(self, tmp_path):
        """Selection manifests are bound to the exact packed row shard bytes."""
        shard_dir = _write_shard_dir(tmp_path, n_rows=16)
        sel_dir = _write_identity_selection_dir(shard_dir)
        shard_path = shard_dir / "shard_00000.safetensors.zst"
        packed = load_packed_row_shard(shard_path)
        packed.tokens[0, 0] = int(packed.tokens[0, 0].item()) + 1
        save_packed_row_shard(packed, shard_dir / "shard_00000")
        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=4,
            device="cpu",
            feature_pipeline=_make_threshold_pipeline(topk=3),
            selection_dir=sel_dir,
        )
        with pytest.raises(ValueError, match="packed row fingerprint mismatch"):
            next(loader)


# ---------------------------------------------------------------------------
# TestFusedPathSampling
# ---------------------------------------------------------------------------


class TestFusedPathSampling:
    """Fused path with stochastic sampling (SamplingBundle + geometric distribution)."""

    def test_sampling_selection_produces_valid_batch(self, tmp_path):
        """SamplingBundle with geometric(mean=5, min_k=3, max_k=8) produces valid batch.

        Verifies:
        - enc_mask has at least one active feature (not all zero)
        - F_max <= max_k + 1 (sampling respects max_k upper bound; +1 accounts for
          boundary fencepost in sampling implementation)
        """
        B = 4
        max_k = 8
        min_k = 3

        shard_dir = _write_shard_dir(tmp_path, n_rows=16, feats_per_seg=20)
        pipeline = _make_sampling_pipeline_for_test(mean=5.0, min_k=min_k, max_k=max_k)

        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=B,
            device="cpu",
            feature_pipeline=pipeline,
        )

        _inputs, _targets, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, _fp_targets, _state = next(loader)

        # enc_mask must have at least one active feature across the batch
        assert enc_mask.any(), "enc_mask should have at least one active feature"

        # With max_k=8, each segment should have at most max_k active features
        # The fused path should respect the max_k bound from SamplingSelection
        active_per_seg = enc_mask.sum(dim=-1)  # [B, n_seg]
        assert (active_per_seg <= max_k).all(), (
            f"With max_k={max_k}, no segment should have more than {max_k} active features. "
            f"Max was {active_per_seg.max().item()}"
        )

        # F_max should be bounded by max_k (+1 for implementation fencepost)
        F_max = enc_fids.shape[2]
        assert F_max <= max_k + 1, (
            f"F_max={F_max} should be <= max_k+1={max_k + 1}"
        )

        # Shapes: B dim must match, seg_map must have SEQ_LEN columns
        assert enc_fids.shape[0] == B
        assert seg_map.shape == (B, SEQ_LEN)
        assert seg_doc_id.shape[0] == B
