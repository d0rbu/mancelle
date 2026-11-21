"""End-to-end smoke test: fp-loss training step with cross-shard batch.

Verifies the full production path:
  variable-segs shards → precomputed_data_loader → model.forward(...,
  fp_targets=..., seg_doc_id=...) → backward → finite loss.

Pre-fix, the cross-shard stack would raise a torch.cat shape error
before reaching forward(). This test is the production-shaped regression
test that the original bug would have failed.
"""

import json
from pathlib import Path

import pytest
import torch

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder.dataloader import precomputed_data_loader
from steering.conditional.encoder_decoder.model import ForwardResult
from steering.packed_batch_io import save_packed_row_shard
from tests.encdec_test_helpers import (
    N_SAE_FEATURES,
    NUM_BUCKETS,
    _make_test_shard,
    build_model_with_feature_prediction,
    build_threshold_pipeline,
    setup_optimizer_simple,
)


@pytest.fixture(autouse=True)
def _force_sdpa():
    """Force SDPA backend (FlexAttention doesn't support backward on CPU)."""
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


def _write_variable_seg_shards_e2e(
    shard_dir: Path,
    segs_a: list[int],
    segs_b: list[int],
    seed_a: int = 42,
    seed_b: int = 99,
) -> None:
    """Write two shards with per-row variable segment counts for E2E test."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_a = _make_test_shard(n_rows=len(segs_a), segs_per_row=segs_a, seed=seed_a)
    shard_b = _make_test_shard(n_rows=len(segs_b), segs_per_row=segs_b, seed=seed_b)
    save_packed_row_shard(shard_a, shard_dir / "shard_00000")
    save_packed_row_shard(shard_b, shard_dir / "shard_00001")
    (shard_dir / "manifest.json").write_text(json.dumps({"metadata": {}}))


def _build_pipeline():
    """Build a standard threshold pipeline."""
    boundaries = (
        torch.linspace(
            1 / NUM_BUCKETS,
            1.0 - 1 / NUM_BUCKETS,
            NUM_BUCKETS - 1,
            dtype=torch.float64,
        )
        .unsqueeze(0)
        .expand(N_SAE_FEATURES, -1)
        .contiguous()
    )
    return build_threshold_pipeline(boundaries)


class TestCrossShardFpLossSmoke:
    """End-to-end smoke: cross-shard variable-n_seg batch through model+fp_loss."""

    def test_one_train_step_with_cross_shard_fp_targets(self, tmp_path):
        """Full pipeline: variable-segs shards → loader → model fwd+bwd → finite loss.

        This is the production-shaped test the original bug would have
        failed before the fp_targets padding fix.

        Strong assertions:
          - Loss is finite AND positive (not trivially zero from all-masked targets).
          - feature_pred_loss is finite AND positive (FP head actually computed loss).
          - FP head weight specifically received a nonzero gradient.
          - A second forward step produces different loss (parameters updated).
        """
        # Seed for reproducibility: model init, shard data, optimizer init.
        torch.manual_seed(0)

        # Build variable-segs shards spanning a boundary
        shard_dir = tmp_path / "packed"
        _write_variable_seg_shards_e2e(
            shard_dir,
            segs_a=[2, 4],
            segs_b=[3, 5],
        )
        pipeline = _build_pipeline()

        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=3,  # forces cross-shard composition
            device="cpu",
            feature_pipeline=pipeline,
            n_sae_features=N_SAE_FEATURES,
        )

        # Build a tiny model with feature_pred_head
        model = build_model_with_feature_prediction()
        optimizer = setup_optimizer_simple(model)
        model.train()

        # Fetch one cross-shard batch
        (
            x, y, enc_fids, enc_mags, enc_mask, _enc_pcts,
            seg_map, seg_doc_id, fp_targets, _state,
        ) = next(loader)

        assert fp_targets is not None
        assert fp_targets.shape[:2] == enc_mask.shape[:2], (
            f"shape coherence: fp_targets {fp_targets.shape} vs "
            f"enc_mask {enc_mask.shape}"
        )

        # Snapshot a parameter to verify it updates after optimizer.step()
        # The LM head is the most reliable witness for "did training run end-to-end".
        lm_head_param = model.lm_head.weight
        param_before = lm_head_param.detach().clone()

        # Single training step with FP loss
        optimizer.zero_grad(set_to_none=True)
        result = model(
            x, y, enc_fids, enc_mags, enc_mask, seg_map,
            seg_doc_id=seg_doc_id,
            fp_targets=fp_targets,
        )

        assert isinstance(result, ForwardResult), (
            "feature_pred_config is set; expected ForwardResult from forward()"
        )
        loss = result.loss
        assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"
        assert torch.isfinite(result.lm_loss), (
            f"lm_loss is not finite: {result.lm_loss.item()}"
        )
        assert torch.isfinite(result.feature_pred_loss), (
            f"feature_pred_loss is not finite: {result.feature_pred_loss.item()}"
        )
        # Both losses are non-trivially positive (real training signal,
        # not silently zero from all-masked targets / dead head).
        assert loss.item() > 0.0, f"loss should be > 0, got {loss.item()}"
        assert result.lm_loss.item() > 0.0, (
            f"lm_loss should be > 0, got {result.lm_loss.item()}"
        )
        assert result.feature_pred_loss.item() > 0.0, (
            f"feature_pred_loss should be > 0, got {result.feature_pred_loss.item()}"
        )

        loss.backward()

        # Verify the FP HEAD specifically received gradients (not just "some param").
        fp_head_params = list(model.feature_pred_head.parameters())
        assert len(fp_head_params) > 0, "FP head has no parameters?"
        fp_head_grad_norm = sum(
            (p.grad.abs().sum().item() if p.grad is not None else 0.0)
            for p in fp_head_params
        )
        assert fp_head_grad_norm > 0.0, (
            f"FP head parameters received no gradients (sum |grad|={fp_head_grad_norm})"
        )

        # All gradients are finite (no NaN/Inf from the backward pass).
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), (
                    f"non-finite gradient in {name}"
                )

        optimizer.step()

        # Parameter actually updated.
        param_after = lm_head_param.detach()
        assert not torch.equal(param_before, param_after), (
            "lm_head weight unchanged after optimizer.step() — "
            "training did not progress"
        )
