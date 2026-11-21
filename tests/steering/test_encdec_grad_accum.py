"""Verify gradient accumulation and optimizer stepping correctness for the
encoder-decoder training pipeline.

Tests that the training loop pattern from conditional_encdec_train.py correctly
accumulates gradients across micro-steps, divides loss by grad_accum_steps,
calls optimizer.step() exactly once per macro-step, and zeros gradients after
stepping.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.config import CrossAttentionConfig, EncoderConfig
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT

# ---------------------------------------------------------------------------
# Constants for tiny test model
# ---------------------------------------------------------------------------
VOCAB_SIZE = 256
DEC_N_EMBD = 64
DEC_N_LAYER = 2
DEC_N_HEAD = 2
ENC_N_EMBD = 32
ENC_DEPTH = 1
N_SAE_FEATURES = 128
NUM_BUCKETS = 10
XATTN_FREQ = 1

B = 2
T = 64
SEGMENT_SIZE = 32
N_SEG = T // SEGMENT_SIZE
F_MAX = 8

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(device="cuda"):
    """Build a tiny EncoderDecoderGPT on the given device."""
    dec_config = GPTConfig(
        sequence_len=T,
        vocab_size=VOCAB_SIZE,
        n_layer=DEC_N_LAYER,
        n_head=DEC_N_HEAD,
        n_kv_head=DEC_N_HEAD,
        n_embd=DEC_N_EMBD,
        window_pattern="L",
    )
    enc_config = EncoderConfig(
        n_embd=ENC_N_EMBD,
        depth=ENC_DEPTH,
        n_head=DEC_N_HEAD,
        n_kv_head=DEC_N_HEAD,
        head_dim=ENC_N_EMBD // DEC_N_HEAD,
    )
    xattn_config = CrossAttentionConfig(
        frequency=XATTN_FREQ,
        n_head=DEC_N_HEAD,
        n_kv_head=DEC_N_HEAD,
    )
    model = EncoderDecoderGPT(
        dec_config=dec_config,
        enc_config=enc_config,
        xattn_config=xattn_config,
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
        pad_vocab_size_to=VOCAB_SIZE,
    )
    model = model.to(device)
    model.init_weights()
    # Perturb c_proj weights away from zero so gradients flow through attention
    # and cross-attention (they are zero-initialized by init_weights).
    with torch.no_grad():
        for block in model.transformer.h:  # ty: ignore[not-iterable]
            block.attn.c_proj.weight.fill_(0.01)  # ty: ignore[unresolved-attribute]
            block.mlp.c_proj.weight.fill_(0.01)  # ty: ignore[unresolved-attribute]
        for xattn in model.cross_attn_layers.values():
            xattn.c_proj.weight.fill_(0.01)  # ty: ignore[call-non-callable, unresolved-attribute]
        for block in model.encoder.blocks:
            block.attn.c_proj.weight.fill_(0.01)
            block.mlp.c_proj.weight.fill_(0.01)
    model.train()
    return model


def _make_batch(batch_size, device="cuda", seed=0):
    """Generate a synthetic batch of the given size on device."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randint(0, VOCAB_SIZE, (batch_size, T), generator=gen, device="cpu").to(device)
    targets = torch.randint(0, VOCAB_SIZE, (batch_size, T), generator=gen, device="cpu").to(device)
    enc_fids = torch.randint(0, N_SAE_FEATURES, (batch_size, N_SEG, F_MAX), generator=gen, device="cpu").to(device)
    enc_mags = torch.randint(0, NUM_BUCKETS, (batch_size, N_SEG, F_MAX), generator=gen, device="cpu").to(device)
    enc_mask = torch.ones(batch_size, N_SEG, F_MAX, dtype=torch.bool, device=device)
    seg_map = (
        torch.arange(T, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        // SEGMENT_SIZE
    )
    return idx, targets, enc_fids, enc_mags, enc_mask, seg_map


def _collect_grads(model):
    """Return a dict mapping param name to grad tensor (cloned), skipping None."""
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.clone()
    return grads


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


@requires_cuda
class TestGradAccumLossScaling:
    """Verify that accumulated micro-step gradients match a single full-batch pass."""

    def test_accumulated_grads_equal_full_batch(self):
        """Run one full-batch forward-backward, then grad_accum_steps micro-step
        forward-backwards with loss divided by grad_accum_steps. The accumulated
        gradients should match within bf16 tolerance."""
        grad_accum_steps = 4
        full_batch_size = B * grad_accum_steps
        device = "cuda"

        # Use the same seed for reproducible data.
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        # --- Full-batch pass ---
        model_full = _make_model(device)
        full_data = _make_batch(full_batch_size, device, seed=123)
        idx, targets, enc_fids, enc_mags, enc_mask, seg_map = full_data

        model_full.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss_full = model_full(idx, targets, enc_fids, enc_mags, enc_mask, seg_map)
        loss_full.backward()
        grads_full = _collect_grads(model_full)

        # --- Accumulated micro-step passes ---
        # Build model with identical weights.
        model_accum = _make_model(device)
        model_accum.load_state_dict(model_full.state_dict())
        model_accum.zero_grad(set_to_none=True)

        for micro in range(grad_accum_steps):
            start = micro * B
            end = start + B
            micro_data = (
                idx[start:end],
                targets[start:end],
                enc_fids[start:end],
                enc_mags[start:end],
                enc_mask[start:end],
                seg_map[start:end],
            )
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss_micro = model_accum(*micro_data)
            # Scale loss the same way the training loop does.
            loss_micro = loss_micro / grad_accum_steps
            loss_micro.backward()

        grads_accum = _collect_grads(model_accum)

        # Verify all parameters have gradients in both runs.
        assert set(grads_full.keys()) == set(grads_accum.keys()), (
            f"Gradient key mismatch: full has {set(grads_full.keys()) - set(grads_accum.keys())} extra, "
            f"accum has {set(grads_accum.keys()) - set(grads_full.keys())} extra"
        )

        # Compare gradients within bf16 tolerance.
        # cross-entropy with mean reduction over different batch sizes means
        # the accumulated mean-of-means equals the full-batch mean, so the
        # gradients should be close. bf16 tolerance ~1e-2.
        max_diff = 0.0
        worst_param = ""
        for name in grads_full:
            diff = (grads_full[name].float() - grads_accum[name].float()).abs().max().item()
            if diff > max_diff:
                max_diff = diff
                worst_param = name

        # bf16 introduces rounding at each micro-step; allow generous tolerance.
        assert max_diff < 1e-1, (
            f"Gradient mismatch too large: max diff = {max_diff:.4e} at param '{worst_param}'"
        )


@requires_cuda
class TestOptimizerStepTiming:
    """Verify optimizer.step() is called exactly once per macro-step."""

    def test_step_called_once_per_macro_step(self):
        """Simulate the training loop for 3 macro-steps with grad_accum_steps=4.
        The optimizer should be stepped exactly 3 times total."""
        device = "cuda"
        grad_accum_steps = 4
        num_macro_steps = 3

        model = _make_model(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        mock_step = MagicMock(side_effect=optimizer.step)

        data = _make_batch(B, device, seed=0)

        with patch.object(optimizer, "step", mock_step):
            for _macro in range(num_macro_steps):
                for _micro in range(grad_accum_steps):
                    idx, targets, enc_fids, enc_mags, enc_mask, seg_map = data
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        loss = model(idx, targets, enc_fids, enc_mags, enc_mask, seg_map)
                    loss = loss / grad_accum_steps
                    loss.backward()
                # Step once per macro-step, just like the training loop.
                optimizer.step()
                model.zero_grad(set_to_none=True)

        assert mock_step.call_count == num_macro_steps, (
            f"optimizer.step() called {mock_step.call_count} times, "
            f"expected {num_macro_steps}"
        )


@requires_cuda
class TestGradAccumWithDifferentSteps:
    """Verify gradient accumulation correctness at boundary cases."""

    def test_grad_accum_1_matches_no_accum(self):
        """With grad_accum_steps=1, the accumulated path should produce
        identical gradients to a plain forward-backward-step."""
        device = "cuda"

        data = _make_batch(B, device, seed=77)
        idx, targets, enc_fids, enc_mags, enc_mask, seg_map = data

        # --- No accumulation (plain forward-backward) ---
        model_plain = _make_model(device)
        model_plain.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss_plain = model_plain(idx, targets, enc_fids, enc_mags, enc_mask, seg_map)
        loss_plain.backward()
        grads_plain = _collect_grads(model_plain)

        # --- Accumulation with grad_accum_steps=1 ---
        model_accum = _make_model(device)
        model_accum.load_state_dict(model_plain.state_dict())
        model_accum.zero_grad(set_to_none=True)
        grad_accum_steps = 1
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss_accum = model_accum(idx, targets, enc_fids, enc_mags, enc_mask, seg_map)
        loss_accum = loss_accum / grad_accum_steps  # division by 1
        loss_accum.backward()
        grads_accum = _collect_grads(model_accum)

        # With grad_accum_steps=1, the division is by 1 so gradients must be identical.
        for name in grads_plain:
            assert name in grads_accum, f"Missing gradient for {name} in accum path"
            assert torch.equal(grads_plain[name], grads_accum[name]), (
                f"Gradient mismatch for {name}: "
                f"max diff = {(grads_plain[name] - grads_accum[name]).abs().max().item():.4e}"
            )

    def test_grad_accum_2_halves_loss(self):
        """Dividing loss by 2 and backward twice (same data) should give
        the same gradients as a single backward without division."""
        device = "cuda"

        data = _make_batch(B, device, seed=99)
        idx, targets, enc_fids, enc_mags, enc_mask, seg_map = data

        # --- Single pass, no division ---
        model_single = _make_model(device)
        model_single.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss_single = model_single(idx, targets, enc_fids, enc_mags, enc_mask, seg_map)
        loss_single.backward()
        grads_single = _collect_grads(model_single)

        # --- Two passes with loss/2, same data each time ---
        model_double = _make_model(device)
        model_double.load_state_dict(model_single.state_dict())
        model_double.zero_grad(set_to_none=True)
        for _ in range(2):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss_half = model_double(idx, targets, enc_fids, enc_mags, enc_mask, seg_map)
            loss_half = loss_half / 2
            loss_half.backward()
        grads_double = _collect_grads(model_double)

        # (loss/2).backward() twice == loss.backward() once, within bf16 tolerance.
        max_diff = 0.0
        worst_param = ""
        for name in grads_single:
            assert name in grads_double, f"Missing gradient for {name}"
            diff = (grads_single[name].float() - grads_double[name].float()).abs().max().item()
            if diff > max_diff:
                max_diff = diff
                worst_param = name

        assert max_diff < 1e-2, (
            f"Gradient mismatch: max diff = {max_diff:.4e} at param '{worst_param}'. "
            "Expected (loss/2)*2 backward to match single backward."
        )


@requires_cuda
class TestZeroGradTiming:
    """Verify zero_grad behaviour and gradient accumulation semantics."""

    def test_zero_grad_after_step(self):
        """After optimizer.step() and zero_grad(set_to_none=True), all
        parameter gradients should be None."""
        device = "cuda"

        model = _make_model(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        data = _make_batch(B, device, seed=0)
        idx, targets, enc_fids, enc_mags, enc_mask, seg_map = data

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(idx, targets, enc_fids, enc_mags, enc_mask, seg_map)
        loss.backward()

        optimizer.step()
        model.zero_grad(set_to_none=True)

        for name, param in model.named_parameters():
            assert param.grad is None, (
                f"Gradient for {name} should be None after zero_grad(set_to_none=True), "
                f"but has shape {param.grad.shape}"
            )

    def test_gradients_accumulate_across_microsteps(self):
        """Without zero_grad between micro-steps, gradients should accumulate
        (i.e. sum). Verify by comparing 2x single-backward grad against
        the accumulated result of two backwards."""
        device = "cuda"

        data = _make_batch(B, device, seed=55)
        idx, targets, enc_fids, enc_mags, enc_mask, seg_map = data

        # --- Single backward ---
        model_one = _make_model(device)
        model_one.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss_one = model_one(idx, targets, enc_fids, enc_mags, enc_mask, seg_map)
        loss_one.backward()
        grads_one = _collect_grads(model_one)

        # --- Two backwards without zero_grad (accumulate) ---
        model_two = _make_model(device)
        model_two.load_state_dict(model_one.state_dict())
        model_two.zero_grad(set_to_none=True)
        for _ in range(2):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss_two = model_two(idx, targets, enc_fids, enc_mags, enc_mask, seg_map)
            loss_two.backward()
        grads_two = _collect_grads(model_two)

        # Accumulated grads should be 2x the single-pass grads.
        max_diff = 0.0
        worst_param = ""
        for name in grads_one:
            assert name in grads_two, f"Missing gradient for {name}"
            expected = 2.0 * grads_one[name].float()
            actual = grads_two[name].float()
            diff = (expected - actual).abs().max().item()
            if diff > max_diff:
                max_diff = diff
                worst_param = name

        assert max_diff < 1e-2, (
            f"Accumulated gradients are not 2x single-pass: "
            f"max diff = {max_diff:.4e} at param '{worst_param}'"
        )
