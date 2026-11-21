"""Tests that fp_targets (batch index 8) are correctly built through the full dataloader.

Verifies that when n_sae_features is set, precomputed_data_loader yields dense
float32 feature prediction targets in batch position 8, and that None is yielded
when n_sae_features is not set.
"""

import json
from pathlib import Path

import torch as t

from steering.conditional.encoder_decoder.dataloader import precomputed_data_loader
from steering.packed_batch_io import save_packed_row_shard
from tests.encdec_test_helpers import (
    N_SAE_FEATURES,
    NUM_BUCKETS,
    _make_test_shard,
    build_threshold_pipeline,
    make_precomputed_dataloader,
)


class TestFpTargetsThroughLoader:
    """fp_targets are correctly built and yielded through the full dataloader."""

    def test_fp_targets_present_when_n_sae_features_set(self, tmp_path):
        """When n_sae_features is set, fp_targets is not None and is float32."""
        loader = make_precomputed_dataloader(
            tmp_path,
            B=2,
            n_rows=8,
            segs_per_row=3,
            feats_per_seg=20,
            n_sae_features=N_SAE_FEATURES,
        )
        *_, fp_targets, _state = next(loader)

        assert fp_targets is not None, "fp_targets should not be None when n_sae_features is set"
        assert fp_targets.dtype == t.float32, (
            f"fp_targets dtype should be float32, got {fp_targets.dtype}"
        )

    def test_fp_targets_none_when_n_sae_features_not_set(self, tmp_path):
        """When n_sae_features is None, fp_targets is None."""
        shard_dir = Path(tmp_path) / "packed_no_fp"
        shard_dir.mkdir(parents=True, exist_ok=True)
        save_packed_row_shard(_make_test_shard(n_rows=4), shard_dir / "shard_00000")
        (shard_dir / "manifest.json").write_text(json.dumps({"metadata": {}}))

        boundaries = (
            t.linspace(1 / NUM_BUCKETS, 1.0 - 1 / NUM_BUCKETS, NUM_BUCKETS - 1, dtype=t.float64)
            .unsqueeze(0)
            .expand(N_SAE_FEATURES, -1)
            .contiguous()
        )
        pipeline = build_threshold_pipeline(boundaries)
        loader = precomputed_data_loader(
            shard_dir,
            B=2,
            device="cpu",
            feature_pipeline=pipeline,
            n_sae_features=None,
        )
        *_, fp_targets, _state = next(loader)

        assert fp_targets is None, (
            f"fp_targets should be None when n_sae_features is not set, got {fp_targets}"
        )

    def test_fp_targets_shape(self, tmp_path):
        """fp_targets has shape (B, n_seg, N_SAE_FEATURES)."""
        B = 2
        segs_per_row = 3
        loader = make_precomputed_dataloader(
            tmp_path,
            B=B,
            n_rows=8,
            segs_per_row=segs_per_row,
            feats_per_seg=20,
            n_sae_features=N_SAE_FEATURES,
        )
        *_, fp_targets, _state = next(loader)

        assert fp_targets is not None
        assert fp_targets.shape[0] == B, (
            f"fp_targets.shape[0] should be B={B}, got {fp_targets.shape[0]}"
        )
        assert fp_targets.shape[2] == N_SAE_FEATURES, (
            f"fp_targets.shape[2] should be N_SAE_FEATURES={N_SAE_FEATURES}, "
            f"got {fp_targets.shape[2]}"
        )

    def test_fp_targets_values_in_range(self, tmp_path):
        """fp_targets values are in [0.0, 1.0]."""
        loader = make_precomputed_dataloader(
            tmp_path,
            B=2,
            n_rows=8,
            segs_per_row=3,
            feats_per_seg=20,
            n_sae_features=N_SAE_FEATURES,
        )
        *_, fp_targets, _state = next(loader)

        assert fp_targets is not None
        assert fp_targets.min().item() >= 0.0, (
            f"fp_targets min {fp_targets.min().item()} < 0.0"
        )
        assert fp_targets.max().item() <= 1.0, (
            f"fp_targets max {fp_targets.max().item()} > 1.0"
        )

    def test_fp_targets_have_nonzero_entries(self, tmp_path):
        """With feats_per_seg=20 per segment, fp_targets has at least some nonzero entries."""
        loader = make_precomputed_dataloader(
            tmp_path,
            B=2,
            n_rows=8,
            segs_per_row=3,
            feats_per_seg=20,
            n_sae_features=N_SAE_FEATURES,
        )
        *_, fp_targets, _state = next(loader)

        assert fp_targets is not None
        nonzero_count = fp_targets.ne(0).sum().item()
        assert nonzero_count > 0, (
            "fp_targets should have at least some nonzero entries with feats_per_seg=20"
        )
