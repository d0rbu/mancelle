"""Comprehensive tests for dataloader-built feature prediction targets.

Tests that dense feature prediction targets (fp_targets) are correctly built
from the full CSR extraction features (~4000 per segment) in the dataloader,
rather than from the ~15 selected features in the model forward pass.

Test categories:
1. Reference correctness — fp_targets matches CSR shard data exactly
2. Segment mapping — targets align with batch seg_map and seg_doc_id
3. Integration — end-to-end dataloader yield and model forward
4. Edge cases — zero features, boundary IDs, empty batches
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch as t

from steering.conditional.encoder_decoder.config import (
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.dataloader import (
    precomputed_data_loader,
    reconstruct_and_select,
)
from steering.conditional.encoder_decoder.feature_prediction import compute_feature_pred_loss
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
    _build_batch_index,
    save_packed_row_shard,
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
        mean=mean,
        std=mean,
        min_k=1,
        max_k=None,
    )
    pct_table = _make_percentile_table(n_features)
    sampling_bundle = SamplingBundle(
        selection=selection,
        percentile_table=pct_table,
        rng=rng,
    )
    effective_buckets = n_buckets - 1 if with_negatives else n_buckets
    boundaries = _make_boundaries(n_features, effective_buckets)

    negative_bundle = None
    if with_negatives:
        neg_rng = t.Generator().manual_seed(seed + 1)
        negative_bundle = NegativeBundle(
            selection=NegativeSamplingSelection(
                distribution=Distribution.geometric,
                mean=2.0,
                std=2.0,
                min_k=1,
            ),
            n_sae_features=n_features,
            rng=neg_rng,
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
        threshold_type=ThresholdType.segment_topk,
        threshold=threshold,
    )
    config = ThresholdSelectionConfig(
        selection=selection,
        precomputed_thresholds=None,
    )
    effective_buckets = n_buckets - 1 if with_negatives else n_buckets
    boundaries = _make_boundaries(n_features, effective_buckets)

    negative_bundle = None
    if with_negatives:
        neg_rng = t.Generator().manual_seed(seed + 1)
        negative_bundle = NegativeBundle(
            selection=NegativeSamplingSelection(
                distribution=Distribution.geometric,
                mean=2.0,
                std=2.0,
                min_k=1,
            ),
            n_sae_features=n_features,
            rng=neg_rng,
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
    n_features: int = N_SAE_FEATURES,
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
    seg_doc_id_arr = t.zeros(total_segs, dtype=t.uint8)

    for r in range(n_rows):
        for s in range(segs_per_row):
            idx = r * segs_per_row + s
            seg_token_start[idx] = s * seg_size
            seg_token_end[idx] = T if s == segs_per_row - 1 else (s + 1) * seg_size
            seg_doc_id_arr[idx] = 0

    seg_feat_offsets = t.arange(0, (total_segs + 1) * feats_per_seg, feats_per_seg, dtype=t.int32)
    feature_ids = t.randint(0, n_features, (total_feats,), dtype=t.uint16)
    feature_pcts = t.randint(1, 255, (total_feats,), dtype=t.uint8)

    return PackedRowShard(
        tokens=tokens,
        row_seg_offsets=row_seg_offsets,
        seg_token_start=seg_token_start,
        seg_token_end=seg_token_end,
        seg_doc_id=seg_doc_id_arr,
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
    row_seg_offsets = t.tensor([0, 4, 6, 9], dtype=t.int32)

    seg_token_start = t.tensor([0, 8, 16, 24, 0, 16, 0, 10, 20], dtype=t.uint16)
    seg_token_end = t.tensor([8, 16, 24, 32, 16, 32, 10, 20, 32], dtype=t.uint16)
    seg_doc_id_arr = t.tensor([0, 0, 1, 1, 0, 0, 0, 0, 1], dtype=t.uint8)

    feat_counts = [3, 5, 2, 10, 8, 0, 1, 15, 6]
    total_feats = sum(feat_counts)
    seg_feat_offsets = t.tensor(
        [0] + [sum(feat_counts[: i + 1]) for i in range(len(feat_counts))],
        dtype=t.int32,
    )

    feature_ids = t.randint(0, N_SAE_FEATURES, (total_feats,), dtype=t.uint16)
    feature_pcts = t.randint(1, 255, (total_feats,), dtype=t.uint8)

    return PackedRowShard(
        tokens=tokens,
        row_seg_offsets=row_seg_offsets,
        seg_token_start=seg_token_start,
        seg_token_end=seg_token_end,
        seg_doc_id=seg_doc_id_arr,
        seg_feat_offsets=seg_feat_offsets,
        feature_ids=feature_ids,
        feature_pcts=feature_pcts,
    )


def _build_fp_targets_reference(
    shard: PackedRowShard,
    row_indices: list[int] | range,
    n_sae_features: int,
    pad_n_seg: int | None = None,
) -> t.Tensor:
    """Pure-Python reference implementation: build dense targets from CSR shard data.

    Loops over each row and segment, gathers feature IDs and percentiles
    from the CSR arrays, and scatters them into a dense [B, n_seg, n_sae] tensor.
    """
    idx = _build_batch_index(shard, row_indices, pad_n_seg=pad_n_seg)
    B = idx.B
    batch_n_seg = idx.batch_n_seg

    fp_targets = t.zeros(B, batch_n_seg, n_sae_features, dtype=t.float32)

    feat_offsets = shard.seg_feat_offsets.long()

    for flat_idx in range(idx.total_segs):
        b = idx.b_flat[flat_idx].item()
        local_s = idx.local_flat[flat_idx].item()
        global_s = idx.global_flat[flat_idx].item()

        feat_start = feat_offsets[global_s].item()  # ty: ignore[invalid-argument-type]
        feat_end = feat_offsets[global_s + 1].item()  # ty: ignore[invalid-argument-type]

        for f_pos in range(feat_start, feat_end):  # ty: ignore[invalid-argument-type]
            fid = shard.feature_ids[f_pos].item()
            pct = shard.feature_pcts[f_pos].item() / 255.0
            fp_targets[b, local_s, fid] = pct  # ty: ignore[invalid-assignment]

    return fp_targets


def _write_shards_with_manifest(
    output_dir: Path,
    n_shards: int = 1,
    n_rows: int = 8,
    **shard_kwargs,
) -> list[Path]:
    """Write shards to output_dir with a manifest.json, return paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n_shards):
        shard = _make_shard(n_rows=n_rows, seed=42 + i, **shard_kwargs)
        path = save_packed_row_shard(shard, output_dir / f"shard_{i:05d}")
        paths.append(path)

    # Write minimal manifest
    manifest = {"metadata": {}}
    (output_dir / "manifest.json").write_text(json.dumps(manifest))
    return paths


# ======================================================================
# 1. Reference correctness tests
# ======================================================================


class TestDenseTargetCorrectness:
    """Verify fp_targets matches the CSR shard data exactly."""

    def test_dense_targets_match_csr_data(self):
        """Every non-zero entry in fp_targets corresponds to a real feature in the CSR shard."""
        shard = _make_shard(n_rows=4, feats_per_seg=20)
        pipeline = _make_sampling_pipeline()
        row_indices = list(range(4))

        rb, fp_targets = reconstruct_and_select(
            shard,
            row_indices,
            pipeline,
            device="cpu",
            n_sae_features=N_SAE_FEATURES,
        )
        ref = _build_fp_targets_reference(shard, row_indices, N_SAE_FEATURES)

        assert t.allclose(fp_targets, ref, atol=1e-6), (  # ty: ignore[invalid-argument-type]
            f"Max diff: {(fp_targets - ref).abs().max().item()}"  # ty: ignore[unsupported-operator]
        )

    def test_dense_targets_have_correct_percentile_values(self):
        """For each non-zero position, the value matches shard.feature_pcts / 255.0."""
        shard = _make_shard(n_rows=2, feats_per_seg=15)
        pipeline = _make_sampling_pipeline()
        row_indices = [0, 1]

        rb, fp_targets = reconstruct_and_select(
            shard,
            row_indices,
            pipeline,
            device="cpu",
            n_sae_features=N_SAE_FEATURES,
        )

        # Check specific segments manually
        idx = _build_batch_index(shard, row_indices)
        feat_offsets = shard.seg_feat_offsets.long()

        for flat_i in range(idx.total_segs):
            b = idx.b_flat[flat_i].item()
            local_s = idx.local_flat[flat_i].item()
            global_s = idx.global_flat[flat_i].item()
            f_start = feat_offsets[global_s].item()  # ty: ignore[invalid-argument-type]
            f_end = feat_offsets[global_s + 1].item()  # ty: ignore[invalid-argument-type]

            for f_pos in range(f_start, f_end):  # ty: ignore[invalid-argument-type]
                fid = shard.feature_ids[f_pos].item()
                expected_pct = shard.feature_pcts[f_pos].item() / 255.0
                actual = fp_targets[b, local_s, fid].item()  # ty: ignore[invalid-argument-type, not-subscriptable]
                assert abs(actual - expected_pct) < 1e-6, (
                    f"Segment ({b}, {local_s}), feature {fid}: "
                    f"expected {expected_pct:.6f}, got {actual:.6f}"
                )

    def test_dense_targets_have_all_extracted_features(self):
        """fp_targets has ~feats_per_seg non-zero entries per segment, NOT ~15 selected."""
        shard = _make_shard(n_rows=2, feats_per_seg=50)  # many features per segment
        pipeline = _make_sampling_pipeline(mean=3.0)  # selects ~3-4 features
        row_indices = [0, 1]

        rb, fp_targets = reconstruct_and_select(
            shard,
            row_indices,
            pipeline,
            device="cpu",
            n_sae_features=N_SAE_FEATURES,
        )

        # Count non-zeros per segment — should be close to feats_per_seg (50),
        # NOT close to the selection mean (3-4)
        idx = _build_batch_index(shard, row_indices)
        for flat_i in range(idx.total_segs):
            b = idx.b_flat[flat_i].item()
            local_s = idx.local_flat[flat_i].item()
            nnz = (fp_targets[b, local_s] > 0).sum().item()  # ty: ignore[invalid-argument-type, not-subscriptable]
            # Allow for some collisions in feature_ids but should be much more than selection
            assert nnz > 10, (
                f"Segment ({b}, {local_s}) has only {nnz} non-zeros, "
                f"expected ~50 (all extracted features)"
            )
            # Selected features are ~3-4, fp_targets should have much more
            selected_nnz = (rb.enc_mask[b, local_s]).sum().item()  # ty: ignore[invalid-argument-type]
            assert nnz > selected_nnz * 2, (
                f"fp_targets nnz ({nnz}) should be much larger than selected ({selected_nnz})"
            )

    def test_dense_targets_shape(self):
        """fp_targets shape is (B, n_seg, n_sae_features)."""
        shard = _make_shard(n_rows=4, segs_per_row=3)
        pipeline = _make_sampling_pipeline()
        row_indices = list(range(4))

        rb, fp_targets = reconstruct_and_select(
            shard,
            row_indices,
            pipeline,
            device="cpu",
            n_sae_features=N_SAE_FEATURES,
        )

        assert fp_targets.shape == (4, 3, N_SAE_FEATURES)  # ty: ignore[unresolved-attribute]

    def test_dense_targets_dtype_float32(self):
        """fp_targets dtype is float32."""
        shard = _make_shard(n_rows=2)
        pipeline = _make_sampling_pipeline()

        rb, fp_targets = reconstruct_and_select(
            shard,
            [0, 1],
            pipeline,
            device="cpu",
            n_sae_features=N_SAE_FEATURES,
        )

        assert fp_targets.dtype == t.float32  # ty: ignore[unresolved-attribute]

    def test_dense_targets_values_in_zero_one(self):
        """All values in fp_targets are in [0.0, 1.0]."""
        shard = _make_shard(n_rows=4, feats_per_seg=30)
        pipeline = _make_sampling_pipeline()

        rb, fp_targets = reconstruct_and_select(
            shard,
            list(range(4)),
            pipeline,
            device="cpu",
            n_sae_features=N_SAE_FEATURES,
        )

        assert fp_targets.min() >= 0.0  # ty: ignore[unresolved-attribute]
        assert fp_targets.max() <= 1.0  # ty: ignore[unresolved-attribute]

    def test_dense_targets_no_duplicate_features(self):
        """Each feature ID appears at most once per segment in the CSR data.

        If duplicates exist, the last one wins in scatter — verify consistency
        with the reference implementation.
        """
        shard = _make_shard(n_rows=2, feats_per_seg=20)
        pipeline = _make_sampling_pipeline()

        rb, fp_targets = reconstruct_and_select(
            shard,
            [0, 1],
            pipeline,
            device="cpu",
            n_sae_features=N_SAE_FEATURES,
        )
        ref = _build_fp_targets_reference(shard, [0, 1], N_SAE_FEATURES)

        # Even with potential duplicates, our implementation should match reference
        assert t.allclose(fp_targets, ref, atol=1e-6)  # ty: ignore[invalid-argument-type]

    def test_dense_targets_match_manual_build(self):
        """Build fp_targets manually with a Python loop and compare element-by-element."""
        shard = _make_variable_shard()
        pipeline = _make_sampling_pipeline()
        row_indices = [0, 1, 2]

        rb, fp_targets = reconstruct_and_select(
            shard,
            row_indices,
            pipeline,
            device="cpu",
            n_sae_features=N_SAE_FEATURES,
        )
        ref = _build_fp_targets_reference(shard, row_indices, N_SAE_FEATURES)

        assert fp_targets.shape == ref.shape  # ty: ignore[unresolved-attribute]
        assert t.allclose(fp_targets, ref, atol=1e-6), (  # ty: ignore[invalid-argument-type]
            f"Max diff: {(fp_targets - ref).abs().max().item()}, "  # ty: ignore[unsupported-operator]
            f"num diffs: {(fp_targets != ref).sum().item()}"  # ty: ignore[unresolved-attribute]
        )

    def test_dense_targets_with_threshold_pipeline(self):
        """fp_targets correctness with threshold selection (not just sampling)."""
        shard = _make_shard(n_rows=4, feats_per_seg=25)
        pipeline = _make_threshold_pipeline(threshold=5.0)
        row_indices = list(range(4))

        rb, fp_targets = reconstruct_and_select(
            shard,
            row_indices,
            pipeline,
            device="cpu",
            n_sae_features=N_SAE_FEATURES,
        )
        ref = _build_fp_targets_reference(shard, row_indices, N_SAE_FEATURES)

        assert t.allclose(fp_targets, ref, atol=1e-6)  # ty: ignore[invalid-argument-type]


# ======================================================================
# 2. Segment mapping tests
# ======================================================================


class TestDenseTargetSegmentMapping:
    """Verify fp_targets aligns with seg_map and batch dimensions."""

    def test_targets_aligned_with_seg_map(self):
        """fp_targets[b, s, :] contains features for the correct segment s."""
        shard = _make_shard(n_rows=4, segs_per_row=3, feats_per_seg=10)
        pipeline = _make_sampling_pipeline()

        rb, fp_targets = reconstruct_and_select(
            shard,
            list(range(4)),
            pipeline,
            device="cpu",
            n_sae_features=N_SAE_FEATURES,
        )
        ref = _build_fp_targets_reference(shard, list(range(4)), N_SAE_FEATURES)

        # For each batch element, verify segment targets match reference
        for b in range(4):
            for s in range(3):
                assert t.allclose(fp_targets[b, s], ref[b, s], atol=1e-6), (  # ty: ignore[not-subscriptable]
                    f"Mismatch at batch {b}, segment {s}"
                )

    def test_targets_for_variable_segment_counts(self):
        """Rows with different numbers of segments all get correct targets."""
        shard = _make_variable_shard()
        pipeline = _make_sampling_pipeline()

        rb, fp_targets = reconstruct_and_select(
            shard,
            [0, 1, 2],
            pipeline,
            device="cpu",
            n_sae_features=N_SAE_FEATURES,
        )
        ref = _build_fp_targets_reference(shard, [0, 1, 2], N_SAE_FEATURES)

        assert t.allclose(fp_targets, ref, atol=1e-6)  # ty: ignore[invalid-argument-type]

    def test_padded_segments_have_zero_targets(self):
        """Segments beyond the real count (padding) have all-zero targets."""
        shard = _make_variable_shard()
        pipeline = _make_sampling_pipeline()

        rb, fp_targets = reconstruct_and_select(
            shard,
            [0, 1, 2],
            pipeline,
            device="cpu",
            n_sae_features=N_SAE_FEATURES,
        )

        # Row 1 has 2 segments, batch_n_seg is 4 (from row 0)
        # Segments 2 and 3 in row 1 should be all zeros
        idx = _build_batch_index(shard, [0, 1, 2])
        seg_counts = idx.seg_counts.tolist()
        batch_n_seg = idx.batch_n_seg

        for b_idx, count in enumerate(seg_counts):
            for s in range(count, batch_n_seg):
                assert (fp_targets[b_idx, s] == 0).all(), (  # ty: ignore[not-subscriptable]
                    f"Row {b_idx}, padded segment {s} should be all zeros"
                )

    def test_non_contiguous_row_indices(self):
        """Correct targets for out-of-order row indices [0, 3, 1]."""
        shard = _make_shard(n_rows=8, feats_per_seg=15)
        pipeline = _make_sampling_pipeline()
        row_indices = [0, 3, 1]

        rb, fp_targets = reconstruct_and_select(
            shard,
            row_indices,
            pipeline,
            device="cpu",
            n_sae_features=N_SAE_FEATURES,
        )
        ref = _build_fp_targets_reference(shard, row_indices, N_SAE_FEATURES)

        assert t.allclose(fp_targets, ref, atol=1e-6)  # ty: ignore[invalid-argument-type]

    def test_with_pad_n_seg(self):
        """fp_targets respects pad_n_seg parameter for static shapes."""
        shard = _make_shard(n_rows=4, segs_per_row=3, feats_per_seg=10)
        pipeline = _make_sampling_pipeline()
        pad_n_seg = 6  # larger than actual 3

        rb, fp_targets = reconstruct_and_select(
            shard,
            list(range(4)),
            pipeline,
            device="cpu",
            n_sae_features=N_SAE_FEATURES,
            pad_n_seg=pad_n_seg,
        )

        assert fp_targets.shape == (4, pad_n_seg, N_SAE_FEATURES)  # ty: ignore[unresolved-attribute]
        # Extra segments should be all zeros
        for b in range(4):
            for s in range(3, pad_n_seg):
                assert (fp_targets[b, s] == 0).all()  # ty: ignore[not-subscriptable]


# ======================================================================
# 3. Integration tests
# ======================================================================


class TestDenseTargetIntegration:
    """End-to-end tests through the dataloader and model."""

    def test_precomputed_data_loader_yields_fp_targets(self, tmp_path):
        """The precomputed data loader yields fp_targets as the 10th element."""
        _write_shards_with_manifest(tmp_path / "packed", n_shards=1, n_rows=8)
        pipeline = _make_sampling_pipeline()
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            device="cpu",
            feature_pipeline=pipeline,
            n_sae_features=N_SAE_FEATURES,
        )
        batch = next(loader)
        assert len(batch) == 10  # 9 original + fp_targets before state_dict
        fp_targets = batch[8]  # 10th element (0-indexed: 8)
        state_dict = batch[9]

        assert isinstance(fp_targets, t.Tensor)
        assert fp_targets.dtype == t.float32
        assert fp_targets.shape[0] == 4  # B
        assert fp_targets.shape[2] == N_SAE_FEATURES
        assert isinstance(state_dict, dict)

    def test_fp_targets_on_correct_device(self, tmp_path):
        """fp_targets is on the same device as other batch tensors (CPU test)."""
        _write_shards_with_manifest(tmp_path / "packed", n_shards=1, n_rows=8)
        pipeline = _make_sampling_pipeline()
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            device="cpu",
            feature_pipeline=pipeline,
            n_sae_features=N_SAE_FEATURES,
        )
        batch = next(loader)
        inputs = batch[0]
        fp_targets = batch[8]

        assert fp_targets.device == inputs.device

    def test_fp_targets_none_without_n_sae_features(self, tmp_path):
        """When n_sae_features is not provided, fp_targets is None."""
        _write_shards_with_manifest(tmp_path / "packed", n_shards=1, n_rows=8)
        pipeline = _make_sampling_pipeline()
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            device="cpu",
            feature_pipeline=pipeline,
            # n_sae_features not passed
        )
        batch = next(loader)
        assert len(batch) == 10
        fp_targets = batch[8]
        assert fp_targets is None

    def test_fp_targets_none_without_fused_path(self, tmp_path):
        """When n_sae_features is not passed, fp_targets is None even with a pipeline."""
        _write_shards_with_manifest(tmp_path / "packed", n_shards=1, n_rows=8)
        pipeline = _make_sampling_pipeline()
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=pipeline,
            device="cpu",
            # n_sae_features NOT passed
        )
        batch = next(loader)
        assert len(batch) == 10
        fp_targets = batch[8]
        assert fp_targets is None

    def test_feature_pred_loss_with_dataloader_targets(self):
        """The feature prediction loss is finite and reasonable with dataloader-built targets."""
        shard = _make_shard(n_rows=2, segs_per_row=4, feats_per_seg=20)
        pipeline = _make_sampling_pipeline()

        rb, fp_targets = reconstruct_and_select(
            shard,
            [0, 1],
            pipeline,
            device="cpu",
            n_sae_features=N_SAE_FEATURES,
        )
        assert fp_targets is not None

        B = 2
        T = rb.inputs.shape[1]
        logits = t.randn(B, T, N_SAE_FEATURES)

        loss = compute_feature_pred_loss(
            logits,
            rb.seg_map,
            fp_targets,
            rb.seg_doc_id,
            FeaturePredLossMode.FULL_SEGMENT,
            FeaturePredLossFn.BCE,
        )
        assert t.isfinite(loss)
        assert loss.item() > 0.0

    def test_loss_uses_all_features_not_just_selected(self):
        """The fp_targets tensor has significantly more non-zeros than enc_mask."""
        shard = _make_shard(n_rows=4, segs_per_row=3, feats_per_seg=50)
        pipeline = _make_sampling_pipeline(mean=3.0)  # select ~3-4

        rb, fp_targets = reconstruct_and_select(
            shard,
            list(range(4)),
            pipeline,
            device="cpu",
            n_sae_features=N_SAE_FEATURES,
        )

        # Count non-zeros in fp_targets vs enc_mask
        fp_nnz = (fp_targets > 0).sum().item()  # ty: ignore[unsupported-operator]
        selected_nnz = rb.enc_mask.sum().item()

        # fp_targets should have ~50 features/seg * 12 segs = ~600 non-zeros
        # enc_mask should have ~3-4 features/seg * 12 segs = ~36-48 non-zeros
        assert fp_nnz > selected_nnz * 3, (
            f"fp_targets nnz ({fp_nnz}) should be much larger than selected ({selected_nnz})"
        )

    def test_backward_pass_works_with_new_targets(self):
        """Full forward + backward with dataloader-built fp_targets produces finite gradients."""
        shard = _make_shard(n_rows=2, segs_per_row=3, feats_per_seg=20)
        pipeline = _make_sampling_pipeline()

        rb, fp_targets = reconstruct_and_select(
            shard,
            [0, 1],
            pipeline,
            device="cpu",
            n_sae_features=N_SAE_FEATURES,
        )
        assert fp_targets is not None

        B = 2
        T = rb.inputs.shape[1]
        logits = t.randn(B, T, N_SAE_FEATURES, requires_grad=True)

        loss = compute_feature_pred_loss(
            logits,
            rb.seg_map,
            fp_targets,
            rb.seg_doc_id,
            FeaturePredLossMode.BEST_TOKEN,
            FeaturePredLossFn.BCE,
        )
        loss.backward()
        assert logits.grad is not None
        assert t.isfinite(logits.grad).all()
        assert logits.grad.abs().sum() > 0

    def test_precomputed_data_loader_fp_targets_match_reference(self, tmp_path):
        """fp_targets from dataloader match reference build from CSR shard data."""
        shard = _make_shard(n_rows=4, feats_per_seg=20)
        shard_dir = tmp_path / "packed"
        shard_dir.mkdir(parents=True)
        save_packed_row_shard(shard, shard_dir / "shard_00000")
        manifest = {"metadata": {}}
        (shard_dir / "manifest.json").write_text(json.dumps(manifest))

        pipeline = _make_sampling_pipeline(seed=99)
        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=4,
            device="cpu",
            feature_pipeline=pipeline,
            n_sae_features=N_SAE_FEATURES,
        )
        batch = next(loader)
        fp_targets = batch[8]

        # Build reference from the same shard/rows
        ref = _build_fp_targets_reference(shard, list(range(4)), N_SAE_FEATURES)
        assert t.allclose(fp_targets, ref, atol=1e-6)

    def test_cross_shard_fp_targets_stacking(self, tmp_path):
        """fp_targets are correctly stacked when a batch spans two shards."""
        # 2 shards with 3 rows each, B=4 forces cross-shard batching
        shard_dir = tmp_path / "packed"
        shard_dir.mkdir(parents=True)
        shard0 = _make_shard(n_rows=3, feats_per_seg=15, seed=42)
        shard1 = _make_shard(n_rows=3, feats_per_seg=15, seed=99)
        save_packed_row_shard(shard0, shard_dir / "shard_00000")
        save_packed_row_shard(shard1, shard_dir / "shard_00001")
        manifest = {"metadata": {}}
        (shard_dir / "manifest.json").write_text(json.dumps(manifest))

        pipeline = _make_sampling_pipeline(seed=123)
        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=4,
            device="cpu",
            feature_pipeline=pipeline,
            n_sae_features=N_SAE_FEATURES,
        )
        batch = next(loader)
        fp_targets = batch[8]

        assert fp_targets is not None
        assert fp_targets.shape[0] == 4  # B
        assert fp_targets.shape[2] == N_SAE_FEATURES
        assert fp_targets.dtype == t.float32
        # Each row should have non-zero features
        for b in range(4):
            assert (fp_targets[b] > 0).any(), f"Row {b} has no non-zero fp_targets"


# ======================================================================
# 4. Edge cases
# ======================================================================


class TestDenseTargetEdgeCases:
    """Edge cases and boundary conditions for fp_targets construction."""

    def test_segment_with_zero_features(self):
        """A segment with 0 extracted features gets an all-zero target row."""
        shard = _make_variable_shard()  # Row 1 segment 1 has 0 features
        pipeline = _make_sampling_pipeline()

        rb, fp_targets = reconstruct_and_select(
            shard,
            [0, 1, 2],
            pipeline,
            device="cpu",
            n_sae_features=N_SAE_FEATURES,
        )

        # Row 1 (batch index 1) has 2 segments: [8, 0] features
        # Segment 1 (local index 1) should be all zeros
        assert fp_targets is not None
        idx = _build_batch_index(shard, [0, 1, 2])

        # Find the segment with 0 features (global index 5 = row 1, seg 1)
        feat_offsets = shard.seg_feat_offsets.long()
        for flat_i in range(idx.total_segs):
            g = idx.global_flat[flat_i].item()
            n_feats = feat_offsets[g + 1].item() - feat_offsets[g].item()  # ty: ignore[invalid-argument-type]
            if n_feats == 0:
                b = idx.b_flat[flat_i].item()
                s = idx.local_flat[flat_i].item()
                assert (fp_targets[b, s] == 0).all(), (  # ty: ignore[invalid-argument-type]
                    f"Segment ({b}, {s}) with 0 features should be all zeros"
                )

    def test_single_segment_single_feature(self):
        """Minimal case: 1 segment, 1 feature."""
        tokens = t.randint(1, 1000, (1, 17), dtype=t.uint16)  # T=16
        shard = PackedRowShard(
            tokens=tokens,
            row_seg_offsets=t.tensor([0, 1], dtype=t.int32),
            seg_token_start=t.tensor([0], dtype=t.uint16),
            seg_token_end=t.tensor([16], dtype=t.uint16),
            seg_doc_id=t.tensor([0], dtype=t.uint8),
            seg_feat_offsets=t.tensor([0, 1], dtype=t.int32),
            feature_ids=t.tensor([42], dtype=t.uint16),
            feature_pcts=t.tensor([128], dtype=t.uint8),
        )
        pipeline = _make_sampling_pipeline()

        rb, fp_targets = reconstruct_and_select(
            shard,
            [0],
            pipeline,
            device="cpu",
            n_sae_features=N_SAE_FEATURES,
        )

        assert fp_targets.shape == (1, 1, N_SAE_FEATURES)  # ty: ignore[unresolved-attribute]
        assert fp_targets[0, 0, 42].item() == pytest.approx(128 / 255.0, abs=1e-6)  # ty: ignore[not-subscriptable]
        # All other entries should be 0
        fp_targets[0, 0, 42] = 0  # ty: ignore[invalid-assignment]
        assert (fp_targets == 0).all()  # ty: ignore[unresolved-attribute]

    def test_feature_id_at_boundary(self):
        """Feature ID = n_sae_features - 1 (last valid index) is correctly placed."""
        n_sae = 64
        tokens = t.randint(1, 1000, (1, 17), dtype=t.uint16)
        shard = PackedRowShard(
            tokens=tokens,
            row_seg_offsets=t.tensor([0, 1], dtype=t.int32),
            seg_token_start=t.tensor([0], dtype=t.uint16),
            seg_token_end=t.tensor([16], dtype=t.uint16),
            seg_doc_id=t.tensor([0], dtype=t.uint8),
            seg_feat_offsets=t.tensor([0, 1], dtype=t.int32),
            feature_ids=t.tensor([n_sae - 1], dtype=t.uint16),  # last valid
            feature_pcts=t.tensor([200], dtype=t.uint8),
        )
        pipeline = _make_sampling_pipeline(n_features=n_sae)

        rb, fp_targets = reconstruct_and_select(
            shard,
            [0],
            pipeline,
            device="cpu",
            n_sae_features=n_sae,
        )

        assert fp_targets[0, 0, n_sae - 1].item() == pytest.approx(200 / 255.0, abs=1e-6)  # ty: ignore[not-subscriptable]

    def test_feature_id_zero(self):
        """Feature ID = 0 (first valid index) is correctly placed."""
        n_sae = 64
        tokens = t.randint(1, 1000, (1, 17), dtype=t.uint16)
        shard = PackedRowShard(
            tokens=tokens,
            row_seg_offsets=t.tensor([0, 1], dtype=t.int32),
            seg_token_start=t.tensor([0], dtype=t.uint16),
            seg_token_end=t.tensor([16], dtype=t.uint16),
            seg_doc_id=t.tensor([0], dtype=t.uint8),
            seg_feat_offsets=t.tensor([0, 1], dtype=t.int32),
            feature_ids=t.tensor([0], dtype=t.uint16),
            feature_pcts=t.tensor([100], dtype=t.uint8),
        )
        pipeline = _make_sampling_pipeline(n_features=n_sae)

        rb, fp_targets = reconstruct_and_select(
            shard,
            [0],
            pipeline,
            device="cpu",
            n_sae_features=n_sae,
        )

        assert fp_targets[0, 0, 0].item() == pytest.approx(100 / 255.0, abs=1e-6)  # ty: ignore[not-subscriptable]

    def test_large_batch(self):
        """B=32, n_seg=18, ~20 features/seg — realistic production dimensions."""
        shard = _make_shard(n_rows=32, segs_per_row=18, feats_per_seg=20, context_len=64)
        pipeline = _make_sampling_pipeline()

        rb, fp_targets = reconstruct_and_select(
            shard,
            list(range(32)),
            pipeline,
            device="cpu",
            n_sae_features=N_SAE_FEATURES,
        )

        assert fp_targets.shape == (32, 18, N_SAE_FEATURES)  # ty: ignore[unresolved-attribute]
        assert fp_targets.dtype == t.float32  # ty: ignore[unresolved-attribute]
        assert fp_targets.min() >= 0.0  # ty: ignore[unresolved-attribute]
        assert fp_targets.max() <= 1.0  # ty: ignore[unresolved-attribute]

        # Verify at least some features are non-zero in each real segment
        for b in range(32):
            for s in range(18):
                nnz = (fp_targets[b, s] > 0).sum().item()  # ty: ignore[not-subscriptable]
                assert nnz > 0, f"Segment ({b}, {s}) has no features"

    def test_with_negatives(self):
        """fp_targets is correct when negative sampling is enabled."""
        shard = _make_shard(n_rows=4, feats_per_seg=20)
        pipeline = _make_sampling_pipeline(with_negatives=True)

        rb, fp_targets = reconstruct_and_select(
            shard,
            list(range(4)),
            pipeline,
            device="cpu",
            n_sae_features=N_SAE_FEATURES,
        )
        ref = _build_fp_targets_reference(shard, list(range(4)), N_SAE_FEATURES)

        # fp_targets should be independent of negative sampling (built before selection)
        assert t.allclose(fp_targets, ref, atol=1e-6)  # ty: ignore[invalid-argument-type]

    def test_n_sae_features_none_returns_none(self):
        """When n_sae_features is None, fp_targets is None (backwards compat)."""
        shard = _make_shard(n_rows=2)
        pipeline = _make_sampling_pipeline()

        rb, fp_targets = reconstruct_and_select(
            shard,
            [0, 1],
            pipeline,
            device="cpu",
            n_sae_features=None,
        )

        assert fp_targets is None
        # ReconstructedBatch should still be correct
        assert rb.inputs.shape[0] == 2
