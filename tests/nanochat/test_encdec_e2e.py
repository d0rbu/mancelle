"""End-to-end loss curve correctness tests for encoder-decoder conditional pretraining.

Tests cover:
1. Conditioning effect: model learns to use encoder features
2. Feature prediction integration: auxiliary FP loss doesn't degrade text loss
3. Loss curve properties: monotonic decrease, reproducibility, grad accumulation
4. Checkpoint resume: losses after reload match original run
"""

import pytest
import torch
import torch._dynamo

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from tests.encdec_test_helpers import (
    SEQ_LEN,
    ForwardResult,
    build_model_with_feature_prediction,
    build_tiny_model,
    make_precomputed_dataloader,
    setup_optimizer_simple,
    train_steps,
)

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _force_sdpa():
    """Force SDPA attention backend so FlexAttention backward on CPU doesn't fail."""
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


@pytest.fixture(autouse=True)
def _bump_dynamo_cache():
    """Increase dynamo cache limit to avoid recompile limit during training loops."""
    old = torch._dynamo.config.cache_size_limit
    torch._dynamo.config.cache_size_limit = 32
    yield
    torch._dynamo.config.cache_size_limit = old


# ---------------------------------------------------------------------------
# TestConditioningEffect
# ---------------------------------------------------------------------------


class TestConditioningEffect:
    """Verify the model learns to exploit encoder feature conditioning."""

    def test_conditioning_sensitivity_emerges_after_training(self, tmp_path):
        """After training, matched features should differ from mismatched features.

        Trains for 30 steps then compares the normal conditioned batch against
        a batch whose encoder features are rolled across rows. Both forwards
        remain fully conditioned; the gap proves the model learned to use the
        feature conditioning signal.
        """
        torch.manual_seed(42)
        model = build_tiny_model()
        optimizer = setup_optimizer_simple(model)
        dl = make_precomputed_dataloader(tmp_path, B=4, n_rows=32)
        train_steps(model, optimizer, dl, n_steps=30)

        # Evaluation: compare matched full conditioning against mismatched
        # full conditioning from a different row.
        model.eval()
        with torch.no_grad():
            x, y, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, _fp_targets, _ = (
                next(dl)
            )

            result_cond = model(x, y, enc_fids, enc_mags, enc_mask, seg_map)
            if isinstance(result_cond, ForwardResult):
                loss_cond = result_cond.lm_loss.item()
            else:
                loss_cond = result_cond.item()

            rolled_fids = enc_fids.roll(shifts=1, dims=0)
            rolled_mags = enc_mags.roll(shifts=1, dims=0)
            rolled_mask = enc_mask.roll(shifts=1, dims=0)
            result_mismatch = model(x, y, rolled_fids, rolled_mags, rolled_mask, seg_map)
            if isinstance(result_mismatch, ForwardResult):
                loss_mismatch = result_mismatch.lm_loss.item()
            else:
                loss_mismatch = result_mismatch.item()

        assert abs(loss_cond - loss_mismatch) > 1e-4, (
            f"Expected a gap between matched ({loss_cond:.6f}) and "
            f"mismatched ({loss_mismatch:.6f}) conditioned loss after training, "
            f"but difference is {abs(loss_cond - loss_mismatch):.2e}. "
            "The model may not have learned to use encoder features."
        )


# ---------------------------------------------------------------------------
# TestFeaturePredictionIntegration
# ---------------------------------------------------------------------------


class TestFeaturePredictionIntegration:
    """Verify feature prediction auxiliary loss integrates cleanly with text loss."""

    def test_feature_prediction_trains_without_degradation(self, tmp_path):
        """Model with FP head should achieve comparable text loss to one without.

        Trains two models for 30 steps.  The FP model's text (lm) loss over
        the final 5 steps should be within 15% of the plain model's loss.
        """
        n_steps = 30
        window = 5

        # --- Model WITH feature prediction ---
        torch.manual_seed(42)
        model_fp = build_model_with_feature_prediction()
        optimizer_fp = setup_optimizer_simple(model_fp)
        dl_fp = make_precomputed_dataloader(tmp_path / "fp", B=2, n_rows=32)

        model_fp.train()
        lm_losses_fp = []
        for _ in range(n_steps):
            optimizer_fp.zero_grad(set_to_none=True)
            x, y, enc_fids, enc_mags, enc_mask, _enc_pcts, seg_map, seg_doc_id, fp_targets, _ = (
                next(dl_fp)
            )
            result = model_fp(
                x,
                y,
                enc_fids,
                enc_mags,
                enc_mask,
                seg_map,
                seg_doc_id=seg_doc_id,
                fp_targets=fp_targets,
            )
            assert isinstance(result, ForwardResult), "Expected ForwardResult from FP model"
            result.loss.backward()
            optimizer_fp.step()
            lm_losses_fp.append(result.lm_loss.item())

        # --- Model WITHOUT feature prediction ---
        torch.manual_seed(42)
        model_plain = build_tiny_model()
        optimizer_plain = setup_optimizer_simple(model_plain)
        dl_plain = make_precomputed_dataloader(tmp_path / "plain", B=2, n_rows=32)
        plain_losses = train_steps(model_plain, optimizer_plain, dl_plain, n_steps)

        # Compare final-window averages
        avg_fp = sum(lm_losses_fp[-window:]) / window
        avg_plain = sum(plain_losses[-window:]) / window

        assert avg_fp < avg_plain * 1.15, (
            f"FP model text loss ({avg_fp:.4f}) is more than 15% worse than "
            f"plain model ({avg_plain:.4f}).  Feature prediction may be "
            f"interfering with language modelling."
        )


# ---------------------------------------------------------------------------
# TestLossCurve
# ---------------------------------------------------------------------------


class TestLossCurve:
    """Verify basic loss-curve properties during training."""

    def test_loss_decreases_monotonically_early(self, tmp_path):
        """Average loss over windows (0-5) > (15-20) after 20 steps."""
        torch.manual_seed(42)
        model = build_tiny_model()
        optimizer = setup_optimizer_simple(model, lr=0.003)
        dl = make_precomputed_dataloader(tmp_path, B=2, n_rows=32)
        losses = train_steps(model, optimizer, dl, n_steps=20)

        avg_0_5 = sum(losses[0:5]) / 5
        avg_15_20 = sum(losses[15:20]) / 5

        assert avg_0_5 > avg_15_20, (
            f"Loss did not decrease from window 0-5 ({avg_0_5:.4f}) "
            f"to window 15-20 ({avg_15_20:.4f})"
        )

    def test_loss_is_reproducible(self, tmp_path):
        """Two identical runs (seed=42) produce per-step losses within 1e-5."""

        def run_once(subdir):
            torch.manual_seed(42)
            model = build_tiny_model()
            optimizer = setup_optimizer_simple(model)
            dl = make_precomputed_dataloader(tmp_path / subdir, B=2, n_rows=32)
            return train_steps(model, optimizer, dl, n_steps=10)

        losses_a = run_once("a")
        losses_b = run_once("b")

        for i, (la, lb) in enumerate(zip(losses_a, losses_b)):
            assert abs(la - lb) < 1e-5, f"Step {i} loss mismatch between runs: {la:.8f} vs {lb:.8f}"

    def test_gradient_accumulation_both_decrease(self, tmp_path):
        """B=4/accum=1 and B=2/accum=2 should both decrease loss over 10 steps.

        BOS-bestfit packing produces different batches for different B so exact
        match is not expected; we only check that both runs decrease and their
        final averages are within 50% of each other.
        """
        n_steps = 10
        window = 3

        # Run 1: B=4, grad_accum=1
        torch.manual_seed(42)
        model_1 = build_tiny_model()
        opt_1 = setup_optimizer_simple(model_1)
        dl_1 = make_precomputed_dataloader(tmp_path / "r1", B=4, n_rows=32)
        losses_1 = train_steps(model_1, opt_1, dl_1, n_steps=n_steps, grad_accum=1)

        # Run 2: B=2, grad_accum=2
        torch.manual_seed(42)
        model_2 = build_tiny_model()
        opt_2 = setup_optimizer_simple(model_2)
        dl_2 = make_precomputed_dataloader(tmp_path / "r2", B=2, n_rows=32)
        losses_2 = train_steps(model_2, opt_2, dl_2, n_steps=n_steps, grad_accum=2)

        # Both runs must show decreasing loss
        early_1 = sum(losses_1[:window]) / window
        final_1 = sum(losses_1[-window:]) / window
        assert final_1 < early_1, (
            f"Run 1 (B=4, accum=1) did not decrease: early={early_1:.4f} -> final={final_1:.4f}"
        )

        early_2 = sum(losses_2[:window]) / window
        final_2 = sum(losses_2[-window:]) / window
        assert final_2 < early_2, (
            f"Run 2 (B=2, accum=2) did not decrease: early={early_2:.4f} -> final={final_2:.4f}"
        )

        # Final averages within 50% relative difference
        rel_diff = abs(final_1 - final_2) / max(final_1, final_2)
        assert rel_diff < 0.50, (
            f"Final loss averages too far apart: run1={final_1:.4f}, "
            f"run2={final_2:.4f}, rel_diff={rel_diff:.2%}"
        )


# ---------------------------------------------------------------------------
# TestCheckpointResume
# ---------------------------------------------------------------------------


class TestCheckpointResume:
    """Verify that saving and loading a checkpoint produces identical losses."""

    def test_checkpoint_resume_matches(self, tmp_path):
        """Full 20-step run vs 10-step + save/load + 10-step should match.

        Both runs share the same dataloader generator so the data stream is
        identical after the split point.
        """
        n_total = 20
        n_first = 10

        # --- Reference run: full 20 steps ---
        torch.manual_seed(42)
        model_ref = build_tiny_model()
        opt_ref = setup_optimizer_simple(model_ref)
        dl_ref = make_precomputed_dataloader(tmp_path / "ref", B=2, n_rows=32)
        losses_ref = train_steps(model_ref, opt_ref, dl_ref, n_steps=n_total)

        # --- Resume run: 10 steps, save, reload, 10 more steps ---
        torch.manual_seed(42)
        model_resume = build_tiny_model()
        opt_resume = setup_optimizer_simple(model_resume)
        # Use the same data stream as the reference run
        dl_resume = make_precomputed_dataloader(tmp_path / "resume", B=2, n_rows=32)

        # First 10 steps (consumes the same batches as ref steps 0..9)
        train_steps(model_resume, opt_resume, dl_resume, n_steps=n_first)

        # Save checkpoint
        ckpt_path = tmp_path / "checkpoint.pt"
        torch.save(
            {
                "model": model_resume.state_dict(),
                "optimizer": opt_resume.state_dict(),
            },
            ckpt_path,
        )

        # Load checkpoint into fresh objects
        torch.manual_seed(42)
        model_loaded = build_tiny_model()
        opt_loaded = setup_optimizer_simple(model_loaded)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model_loaded.load_state_dict(ckpt["model"])
        opt_loaded.load_state_dict(ckpt["optimizer"])

        # Continue with same generator (dl_resume is at position 10)
        losses_resume_second = train_steps(
            model_loaded, opt_loaded, dl_resume, n_steps=n_total - n_first
        )

        # Steps 10-19 of the reference run should match resume steps 0-9
        for i, (l_ref, l_resume) in enumerate(zip(losses_ref[n_first:], losses_resume_second)):
            assert abs(l_ref - l_resume) < 1e-4, (
                f"Step {n_first + i} loss mismatch after checkpoint resume: "
                f"ref={l_ref:.8f} vs resumed={l_resume:.8f}"
            )


# ---------------------------------------------------------------------------
# TestNaNFreeTraining
# ---------------------------------------------------------------------------


class TestNaNFreeTraining:
    """Verify training stays NaN-free under label dropout configurations."""

    def _train_and_check_nan(self, tmp_path, n_steps, label_dropout, B=2, n_rows=32):
        """Run n_steps of training with label_dropout, asserting no NaN/Inf."""
        torch.manual_seed(42)
        model = build_tiny_model()
        optimizer = setup_optimizer_simple(model)
        dl = make_precomputed_dataloader(
            tmp_path,
            B=B,
            n_rows=n_rows,
            label_dropout=label_dropout,
        )
        model.train()
        for step in range(n_steps):
            optimizer.zero_grad(set_to_none=True)
            x, y, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, _fp_targets, _ = (
                next(dl)
            )
            result = model(x, y, enc_fids, enc_mags, enc_mask, seg_map)
            loss = result if not isinstance(result, ForwardResult) else result.loss
            assert not torch.isnan(loss), f"NaN loss at step {step}"
            assert torch.isfinite(loss), f"Inf loss at step {step}"
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name} at step {step}"
            optimizer.step()

    def test_10_steps_label_dropout_half(self, tmp_path):
        """10 steps with label_dropout=0.5 produce no NaN in loss or gradients."""
        self._train_and_check_nan(tmp_path, n_steps=10, label_dropout=0.5)

    def test_10_steps_label_dropout_full(self, tmp_path):
        """10 steps with label_dropout=1.0 produce no NaN in loss or gradients.

        Full dropout exercises the sentinel path that keeps at least one True
        in enc_mask so the encoder never receives an empty input.
        """
        self._train_and_check_nan(tmp_path, n_steps=10, label_dropout=1.0)

    def test_deeper_encoder_no_nan(self, tmp_path):
        """5 steps with the default model produce no NaN, guarding dtype leaks."""
        self._train_and_check_nan(tmp_path, n_steps=5, label_dropout=0.0)
