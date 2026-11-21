"""Tests for pin_memory + async HtoD transfers in the encoder-decoder dataloader.

Validates that adding .pin_memory() before .to(device, non_blocking=True) does
not change output values, shapes, or dtypes.  Covers:

1. _pin_transfer helper (unit tests)
2. reconstruct_and_select output identity (CPU and CUDA)
3. build_fp_targets_from_shard output identity (CPU and CUDA)
4. Precomputed dataloader batches (shapes, devices, label dropout)
5. Edge cases (empty batch, single row, zero-feature segments)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch as t

from steering.conditional.encoder_decoder.dataloader import (
    _pin_transfer,
    build_fp_targets_from_shard,
    precomputed_data_loader,
    reconstruct_and_select,
)
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
)
from steering.conditional.sampling import ThresholdSelection
from steering.conditional.thresholding import ThresholdType
from steering.packed_batch_io import (
    PackedRowShard,
    _build_batch_index,
    save_packed_row_shard,
)

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

N_SAE_FEATURES = 256
NUM_BUCKETS = 4

requires_cuda = pytest.mark.skipif(
    not t.cuda.is_available(), reason="CUDA not available"
)


def _make_boundaries(n_features: int = N_SAE_FEATURES) -> t.Tensor:
    b = t.linspace(
        1 / NUM_BUCKETS, 1.0 - 1 / NUM_BUCKETS, NUM_BUCKETS - 1, dtype=t.float64
    )
    return b.unsqueeze(0).expand(n_features, -1).contiguous()


def _make_pipeline(n_features: int = N_SAE_FEATURES) -> FeaturePipelineConfig:
    return FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.segment_topk, 3),
            precomputed_thresholds=None,
        ),
        boundaries=_make_boundaries(n_features),
        negative_bundle=None,
        effective_buckets=NUM_BUCKETS,
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
    row_seg_offsets = t.arange(
        0, (n_rows + 1) * segs_per_row, segs_per_row, dtype=t.int32
    )

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

    if feats_per_seg > 0:
        seg_feat_offsets = t.arange(
            0, (total_segs + 1) * feats_per_seg, feats_per_seg, dtype=t.int32
        )
    else:
        seg_feat_offsets = t.zeros(total_segs + 1, dtype=t.int32)
    feature_ids = t.randint(0, N_SAE_FEATURES, (total_feats,), dtype=t.uint16) if total_feats > 0 else t.tensor([], dtype=t.uint16)
    feature_pcts = t.randint(1, 255, (total_feats,), dtype=t.uint8) if total_feats > 0 else t.tensor([], dtype=t.uint8)

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


def _write_shard_dir(tmp_path: Path, shard: PackedRowShard) -> Path:
    """Save a shard + manifest to a temp directory and return the dir."""
    shard_dir = tmp_path / "packed"
    shard_dir.mkdir(parents=True, exist_ok=True)
    save_packed_row_shard(shard, shard_dir / "shard_00000")
    (shard_dir / "manifest.json").write_text(json.dumps({"metadata": {}}))
    return shard_dir


# ======================================================================
# 1. _pin_transfer unit tests
# ======================================================================


class TestPinTransfer:
    """Unit tests for the _pin_transfer helper."""

    def test_cpu_device_returns_unchanged(self):
        x = t.tensor([1.0, 2.0, 3.0])
        result = _pin_transfer(x, "cpu")
        assert result is x  # exact same object

    @requires_cuda
    def test_transfers_to_cuda(self):
        x = t.tensor([1.0, 2.0, 3.0])
        result = _pin_transfer(x, "cuda")
        assert result.device.type == "cuda"
        t.cuda.synchronize()
        assert t.equal(result.cpu(), x)

    @requires_cuda
    def test_preserves_float32(self):
        x = t.randn(4, 4, dtype=t.float32)
        result = _pin_transfer(x, "cuda")
        t.cuda.synchronize()
        assert result.dtype == t.float32
        assert t.equal(result.cpu(), x)

    @requires_cuda
    def test_preserves_int64(self):
        x = t.randint(0, 100, (5,), dtype=t.int64)
        result = _pin_transfer(x, "cuda")
        t.cuda.synchronize()
        assert result.dtype == t.int64
        assert t.equal(result.cpu(), x)

    @requires_cuda
    def test_preserves_bool(self):
        x = t.tensor([True, False, True, False])
        result = _pin_transfer(x, "cuda")
        t.cuda.synchronize()
        assert result.dtype == t.bool
        assert t.equal(result.cpu(), x)

    @requires_cuda
    def test_preserves_float16(self):
        x = t.randn(3, 3).half()
        result = _pin_transfer(x, "cuda")
        t.cuda.synchronize()
        assert result.dtype == t.float16
        assert t.equal(result.cpu(), x)

    @requires_cuda
    def test_preserves_shape_various(self):
        for shape in [(0,), (1,), (3, 4), (2, 3, 5), (1, 1, 1, 1)]:
            x = t.randn(*shape)
            result = _pin_transfer(x, "cuda")
            t.cuda.synchronize()
            assert result.shape == x.shape, f"Shape mismatch for {shape}"

    @requires_cuda
    def test_empty_tensor(self):
        x = t.tensor([])
        result = _pin_transfer(x, "cuda")
        t.cuda.synchronize()
        assert result.shape == (0,)
        assert result.device.type == "cuda"

    @requires_cuda
    def test_data_correct_after_synchronize(self):
        """Transfer a large tensor, synchronize, verify every value."""
        x = t.randn(1000, 1000)
        result = _pin_transfer(x, "cuda")
        t.cuda.synchronize()
        assert t.equal(result.cpu(), x)

    @requires_cuda
    def test_non_blocking_no_corruption(self):
        """Transfer WITHOUT explicit sync, then sync and verify."""
        x = t.arange(10000, dtype=t.float32)
        result = _pin_transfer(x, "cuda")
        # Don't sync immediately — the copy should still complete
        # because subsequent operations on the same stream wait for it.
        t.cuda.synchronize()
        assert t.equal(result.cpu(), x)

    # --- torch.device object tests ---

    def test_torch_device_cpu_returns_unchanged(self):
        """torch.device('cpu') must trigger the early-return path."""
        x = t.tensor([1.0, 2.0, 3.0])
        result = _pin_transfer(x, t.device("cpu"))
        assert result.device.type == "cpu"
        assert t.equal(result, x)

    @requires_cuda
    def test_torch_device_cuda(self):
        """torch.device('cuda') transfers correctly."""
        x = t.tensor([1.0, 2.0, 3.0])
        result = _pin_transfer(x, t.device("cuda"))
        t.cuda.synchronize()
        assert result.device.type == "cuda"
        assert t.equal(result.cpu(), x)

    @requires_cuda
    def test_torch_device_cuda_with_index(self):
        """torch.device('cuda', 0) transfers correctly."""
        x = t.tensor([1.0, 2.0, 3.0])
        result = _pin_transfer(x, t.device("cuda", 0))
        t.cuda.synchronize()
        assert result.device.type == "cuda"
        assert t.equal(result.cpu(), x)

    @requires_cuda
    def test_string_cuda_colon_zero(self):
        """String 'cuda:0' transfers correctly."""
        x = t.tensor([1.0, 2.0, 3.0])
        result = _pin_transfer(x, "cuda:0")
        t.cuda.synchronize()
        assert result.device.type == "cuda"
        assert t.equal(result.cpu(), x)


# ======================================================================
# 2. reconstruct_and_select output identity
# ======================================================================


class TestReconstructAndSelectCPU:
    """reconstruct_and_select on CPU produces unchanged results."""

    def test_basic_output(self):
        shard = _make_shard()
        pipeline = _make_pipeline()
        rb, fp = reconstruct_and_select(
            shard, range(4), pipeline, device="cpu", n_sae_features=N_SAE_FEATURES
        )
        assert rb.inputs.device.type == "cpu"
        assert rb.enc_fids.device.type == "cpu"
        assert fp is not None
        assert fp.device.type == "cpu"

    def test_deterministic(self):
        """Two calls with the same seed produce identical results."""
        shard = _make_shard(seed=99)
        p1 = _make_pipeline()
        p2 = _make_pipeline()
        rb1, fp1 = reconstruct_and_select(
            shard, range(4), p1, device="cpu", n_sae_features=N_SAE_FEATURES
        )
        rb2, fp2 = reconstruct_and_select(
            shard, range(4), p2, device="cpu", n_sae_features=N_SAE_FEATURES
        )
        assert t.equal(rb1.inputs, rb2.inputs)
        assert t.equal(rb1.targets, rb2.targets)
        assert t.equal(rb1.enc_fids, rb2.enc_fids)
        assert t.equal(rb1.enc_mags, rb2.enc_mags)
        assert t.equal(rb1.enc_mask, rb2.enc_mask)
        assert t.equal(rb1.enc_pcts, rb2.enc_pcts)
        assert t.equal(rb1.seg_map, rb2.seg_map)
        assert t.equal(rb1.seg_doc_id, rb2.seg_doc_id)
        assert t.equal(fp1, fp2)  # ty: ignore[invalid-argument-type]


@requires_cuda
class TestReconstructAndSelectCUDA:
    """reconstruct_and_select with device='cuda' uses pinned transfers."""

    def test_output_on_cuda(self):
        shard = _make_shard()
        pipeline = _make_pipeline()
        rb, fp = reconstruct_and_select(
            shard, range(4), pipeline, device="cuda", n_sae_features=N_SAE_FEATURES
        )
        t.cuda.synchronize()
        assert rb.inputs.device.type == "cuda"
        assert rb.enc_fids.device.type == "cuda"
        assert fp is not None
        assert fp.device.type == "cuda"

    def test_bitwise_identity_vs_cpu(self):
        """CUDA results must be bitwise identical to CPU results."""
        shard = _make_shard(seed=123)
        p_cpu = _make_pipeline()
        p_cuda = _make_pipeline()

        rb_cpu, fp_cpu = reconstruct_and_select(
            shard, range(4), p_cpu, device="cpu", n_sae_features=N_SAE_FEATURES
        )
        rb_cuda, fp_cuda = reconstruct_and_select(
            shard, range(4), p_cuda, device="cuda", n_sae_features=N_SAE_FEATURES
        )
        t.cuda.synchronize()

        assert t.equal(rb_cuda.inputs.cpu(), rb_cpu.inputs)
        assert t.equal(rb_cuda.targets.cpu(), rb_cpu.targets)
        assert t.equal(rb_cuda.enc_fids.cpu(), rb_cpu.enc_fids)
        assert t.equal(rb_cuda.enc_mags.cpu(), rb_cpu.enc_mags)
        assert t.equal(rb_cuda.enc_mask.cpu(), rb_cpu.enc_mask)
        assert t.equal(rb_cuda.enc_pcts.cpu(), rb_cpu.enc_pcts)
        assert t.equal(rb_cuda.seg_map.cpu(), rb_cpu.seg_map)
        assert t.equal(rb_cuda.seg_doc_id.cpu(), rb_cpu.seg_doc_id)
        assert t.equal(fp_cuda.cpu(), fp_cpu)  # ty: ignore[invalid-argument-type, unresolved-attribute]

    def test_no_fp_targets(self):
        """n_sae_features=None produces None fp_targets on CUDA."""
        shard = _make_shard()
        pipeline = _make_pipeline()
        rb, fp = reconstruct_and_select(
            shard, range(4), pipeline, device="cuda", n_sae_features=None
        )
        t.cuda.synchronize()
        assert fp is None
        assert rb.inputs.device.type == "cuda"

    def test_with_pad_n_seg_and_pad_F(self):
        """Static shape padding works with pinned transfers."""
        shard = _make_shard()
        pipeline = _make_pipeline()
        rb, fp = reconstruct_and_select(
            shard, range(4), pipeline, device="cuda",
            n_sae_features=N_SAE_FEATURES, pad_n_seg=10, pad_F=8,
        )
        t.cuda.synchronize()
        assert rb.enc_fids.shape[1] == 10
        assert rb.enc_fids.shape[2] == 8
        assert rb.enc_fids.device.type == "cuda"

    def test_torch_device_object(self):
        """torch.device('cuda', 0) works — production passes this, not strings."""
        shard = _make_shard(seed=200)
        p_cpu = _make_pipeline()
        p_cuda = _make_pipeline()

        rb_cpu, fp_cpu = reconstruct_and_select(
            shard, range(4), p_cpu, device="cpu", n_sae_features=N_SAE_FEATURES
        )
        rb_cuda, fp_cuda = reconstruct_and_select(
            shard, range(4), p_cuda, device=t.device("cuda", 0),  # ty: ignore[invalid-argument-type]
            n_sae_features=N_SAE_FEATURES,
        )
        t.cuda.synchronize()

        assert rb_cuda.inputs.device.type == "cuda"
        assert t.equal(rb_cuda.inputs.cpu(), rb_cpu.inputs)
        assert t.equal(rb_cuda.enc_fids.cpu(), rb_cpu.enc_fids)
        assert t.equal(fp_cuda.cpu(), fp_cpu)  # ty: ignore[invalid-argument-type, unresolved-attribute]


# ======================================================================
# 3. build_fp_targets_from_shard output identity
# ======================================================================


class TestBuildFpTargetsCPU:
    """build_fp_targets_from_shard on CPU is unchanged."""

    def test_basic_shape(self):
        shard = _make_shard(n_rows=4)
        idx = _build_batch_index(shard, range(4))
        out = build_fp_targets_from_shard(shard, idx, N_SAE_FEATURES, device="cpu")
        assert out.shape == (4, 3, N_SAE_FEATURES)
        assert out.dtype == t.float32
        assert out.device.type == "cpu"


@requires_cuda
class TestBuildFpTargetsCUDA:
    """build_fp_targets_from_shard on CUDA uses pinned transfers."""

    def test_output_on_cuda(self):
        shard = _make_shard(n_rows=4)
        idx = _build_batch_index(shard, range(4))
        out = build_fp_targets_from_shard(shard, idx, N_SAE_FEATURES, device="cuda")
        t.cuda.synchronize()
        assert out.device.type == "cuda"

    def test_bitwise_identity_vs_cpu(self):
        shard = _make_shard(n_rows=4, seed=77)
        idx = _build_batch_index(shard, range(4))
        cpu_out = build_fp_targets_from_shard(shard, idx, N_SAE_FEATURES, device="cpu")
        cuda_out = build_fp_targets_from_shard(
            shard, idx, N_SAE_FEATURES, device="cuda"
        )
        t.cuda.synchronize()
        assert t.equal(cuda_out.cpu(), cpu_out)

    def test_torch_device_object(self):
        """torch.device('cuda', 0) works correctly."""
        shard = _make_shard(n_rows=4, seed=88)
        idx = _build_batch_index(shard, range(4))
        cpu_out = build_fp_targets_from_shard(shard, idx, N_SAE_FEATURES, device="cpu")
        cuda_out = build_fp_targets_from_shard(
            shard, idx, N_SAE_FEATURES, device=t.device("cuda", 0)  # ty: ignore[invalid-argument-type]
        )
        t.cuda.synchronize()
        assert cuda_out.device.type == "cuda"
        assert t.equal(cuda_out.cpu(), cpu_out)


# ======================================================================
# 4. Precomputed dataloader integration (shapes, devices, label dropout)
# ======================================================================


class TestPrecomputedDataloaderCPU:
    """Precomputed dataloader on CPU produces correct batches."""

    def test_batch_shapes_and_dtypes(self, tmp_path):
        shard = _make_shard(n_rows=4)
        shard_dir = _write_shard_dir(tmp_path, shard)
        pipeline = _make_pipeline()
        loader = precomputed_data_loader(
            packed_row_dir=shard_dir, B=2, device="cpu",
            feature_pipeline=pipeline, n_sae_features=N_SAE_FEATURES,
        )
        batch = next(loader)
        inputs, targets, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, fp_targets, state = batch
        assert inputs.shape[0] == 2
        assert inputs.dtype == t.long
        assert enc_fids.dtype == t.int64
        assert enc_mask.dtype == t.bool
        assert enc_pcts.dtype == t.float32
        assert fp_targets is not None
        assert fp_targets.dtype == t.float32
        for tensor in [inputs, targets, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, fp_targets]:
            assert tensor.device.type == "cpu"

    def test_label_dropout_masks(self, tmp_path):
        """label_dropout > 0 masks some features."""
        shard = _make_shard(n_rows=4)
        shard_dir = _write_shard_dir(tmp_path, shard)
        pipeline = _make_pipeline()
        # Very high dropout to guarantee some masking
        loader = precomputed_data_loader(
            packed_row_dir=shard_dir, B=2, device="cpu",
            feature_pipeline=pipeline, label_dropout=0.99,
        )
        batch = next(loader)
        enc_mask = batch[4]
        # With 99% dropout, almost all segments should be masked
        # (but empty-row prevention keeps at least one)
        assert enc_mask.any(), "At least one feature should be unmasked (empty-row guard)"


@requires_cuda
class TestPrecomputedDataloaderCUDA:
    """Precomputed dataloader on CUDA uses pinned transfers."""

    def test_tensors_on_cuda(self, tmp_path):
        shard = _make_shard(n_rows=4)
        shard_dir = _write_shard_dir(tmp_path, shard)
        pipeline = _make_pipeline()
        loader = precomputed_data_loader(
            packed_row_dir=shard_dir, B=2, device="cuda",
            feature_pipeline=pipeline, n_sae_features=N_SAE_FEATURES,
        )
        batch = next(loader)
        t.cuda.synchronize()
        inputs, targets, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, fp_targets, state = batch
        for tensor in [inputs, targets, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id]:
            assert tensor.device.type == "cuda", f"{tensor.shape} not on CUDA"
        assert fp_targets.device.type == "cuda"

    def test_bitwise_identity_vs_cpu(self, tmp_path):
        """CUDA batch values must be bitwise identical to CPU."""
        shard = _make_shard(n_rows=4, seed=55)
        # Write two dirs (CPU and CUDA use separate loader instances)
        cpu_dir = _write_shard_dir(tmp_path / "cpu", shard)
        cuda_dir = _write_shard_dir(tmp_path / "cuda", shard)
        pipeline_cpu = _make_pipeline()
        pipeline_cuda = _make_pipeline()

        cpu_batch = next(precomputed_data_loader(
            packed_row_dir=cpu_dir, B=2, device="cpu",
            feature_pipeline=pipeline_cpu, n_sae_features=N_SAE_FEATURES,
        ))
        cuda_batch = next(precomputed_data_loader(
            packed_row_dir=cuda_dir, B=2, device="cuda",
            feature_pipeline=pipeline_cuda, n_sae_features=N_SAE_FEATURES,
        ))
        t.cuda.synchronize()

        # Compare all tensors (index 0-8, skip state dict at index 9)
        for i in range(9):
            if cpu_batch[i] is None:
                assert cuda_batch[i] is None
                continue
            assert t.equal(cuda_batch[i].cpu(), cpu_batch[i]), (
                f"Tensor at index {i} differs: shapes {cuda_batch[i].shape} vs {cpu_batch[i].shape}"
            )


@requires_cuda
class TestPrecomputedDataloaderTorchDevice:
    """Precomputed dataloader with torch.device objects (production path)."""

    def test_torch_device_cuda_0(self, tmp_path):
        """torch.device('cuda', 0) — the actual production device type."""
        shard = _make_shard(n_rows=4, seed=66)
        cpu_dir = _write_shard_dir(tmp_path / "cpu", shard)
        cuda_dir = _write_shard_dir(tmp_path / "cuda", shard)
        pipeline_cpu = _make_pipeline()
        pipeline_cuda = _make_pipeline()

        cpu_batch = next(precomputed_data_loader(
            packed_row_dir=cpu_dir, B=2, device="cpu",
            feature_pipeline=pipeline_cpu, n_sae_features=N_SAE_FEATURES,
        ))
        cuda_batch = next(precomputed_data_loader(
            packed_row_dir=cuda_dir, B=2, device=t.device("cuda", 0),  # ty: ignore[invalid-argument-type]
            feature_pipeline=pipeline_cuda, n_sae_features=N_SAE_FEATURES,
        ))
        t.cuda.synchronize()

        for i in range(9):
            if cpu_batch[i] is None:
                assert cuda_batch[i] is None
                continue
            assert cuda_batch[i].device.type == "cuda"
            assert t.equal(cuda_batch[i].cpu(), cpu_batch[i]), (
                f"Tensor at index {i} differs with torch.device"
            )


# ======================================================================
# 5. Edge cases
# ======================================================================


class TestEdgeCases:
    """Edge cases: single row, zero features, small batches."""

    def test_single_row_batch(self):
        shard = _make_shard(n_rows=2, segs_per_row=2)
        pipeline = _make_pipeline()
        rb, fp = reconstruct_and_select(
            shard, [0], pipeline, device="cpu", n_sae_features=N_SAE_FEATURES
        )
        assert rb.inputs.shape[0] == 1
        assert fp.shape[0] == 1  # ty: ignore[unresolved-attribute]

    @requires_cuda
    def test_single_row_cuda(self):
        shard = _make_shard(n_rows=2, segs_per_row=2)
        pipeline = _make_pipeline()
        rb, fp = reconstruct_and_select(
            shard, [0], pipeline, device="cuda", n_sae_features=N_SAE_FEATURES
        )
        t.cuda.synchronize()
        assert rb.inputs.device.type == "cuda"
        assert fp.device.type == "cuda"  # ty: ignore[unresolved-attribute]

    def test_zero_feature_segments(self):
        """Shard with feats_per_seg=0 still works."""
        shard = _make_shard(n_rows=4, feats_per_seg=0)
        pipeline = _make_pipeline()
        rb, fp = reconstruct_and_select(
            shard, range(4), pipeline, device="cpu", n_sae_features=N_SAE_FEATURES
        )
        assert rb.enc_fids.shape[0] == 4
        # All masks should be False when there are no features
        # (except empty-row guard is applied by the precomputed_data_loader, not here)

    @requires_cuda
    def test_zero_feature_segments_cuda(self):
        shard = _make_shard(n_rows=4, feats_per_seg=0)
        pipeline = _make_pipeline()
        rb, fp = reconstruct_and_select(
            shard, range(4), pipeline, device="cuda", n_sae_features=N_SAE_FEATURES
        )
        t.cuda.synchronize()
        assert rb.enc_fids.device.type == "cuda"

    def test_single_feature_per_segment(self):
        shard = _make_shard(n_rows=4, feats_per_seg=1)
        pipeline = _make_pipeline()
        rb, _ = reconstruct_and_select(
            shard, range(4), pipeline, device="cpu"
        )
        assert rb.enc_fids.shape[0] == 4

    @requires_cuda
    def test_large_batch(self):
        """Larger batch to test bulk transfer efficiency."""
        shard = _make_shard(n_rows=16, segs_per_row=5, feats_per_seg=30)
        pipeline = _make_pipeline()
        rb, fp = reconstruct_and_select(
            shard, range(16), pipeline, device="cuda", n_sae_features=N_SAE_FEATURES
        )
        t.cuda.synchronize()
        assert rb.inputs.shape[0] == 16
        assert fp.shape[0] == 16  # ty: ignore[unresolved-attribute]

    def test_precomputed_loader_multi_batch(self, tmp_path):
        """Multiple batches from the loader all have correct shapes."""
        shard = _make_shard(n_rows=8)
        shard_dir = _write_shard_dir(tmp_path, shard)
        pipeline = _make_pipeline()
        loader = precomputed_data_loader(
            packed_row_dir=shard_dir, B=2, device="cpu",
            feature_pipeline=pipeline,
        )
        for _ in range(3):
            batch = next(loader)
            assert batch[0].shape[0] == 2  # inputs
            assert batch[0].dtype == t.long
