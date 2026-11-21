"""Tests for torch.compile(dynamic=False) correctness on the locked PyTorch.

Verifies that compiled and uncompiled models produce identical outputs,
gradients flow correctly through compiled models, and the compile diagnostic
in the training script correctly detects OptimizedModule.
"""

import pytest
import torch
import torch._dynamo

import nanochat.flash_attention as _fa
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPT, GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT


@pytest.fixture(autouse=True)
def _force_sdpa():
    """Tests in this file need the SDPA backend; flex_attention can fail
    on tiny tensors via torch.compile. Save and restore so the override
    doesn't leak into other test files."""
    saved = _fa._override_impl
    _fa._override_impl = FlashAttnImpl.SDPA
    yield
    _fa._override_impl = saved


def _make_gpt():
    config = GPTConfig(
        sequence_len=32, vocab_size=64, n_layer=4,
        n_head=2, n_kv_head=2, n_embd=32, window_pattern="L",
    )
    model = GPT(config, pad_vocab_size_to=64)
    model.init_weights()
    return model


def _make_encdec(n_sae=200, num_buckets=4):
    dec_config = GPTConfig(
        sequence_len=32, vocab_size=64, n_layer=4,
        n_head=2, n_kv_head=2, n_embd=32, window_pattern="L",
    )
    enc_config = EncoderConfig(n_embd=32, depth=1, n_head=2, n_kv_head=2, head_dim=16)
    xattn_config = CrossAttentionConfig(frequency=2, n_head=2, n_kv_head=2)
    model = EncoderDecoderGPT(
        dec_config, enc_config, xattn_config,
        n_sae_features=n_sae, num_buckets=num_buckets,
    )
    model.init_weights()
    return model


class TestCompileDiagnostic:
    """Verify the compile diagnostic from conditional_encdec_train.py."""

    def test_uncompiled_is_not_optimized_module(self):
        """Uncompiled model is NOT an OptimizedModule."""
        model = _make_gpt()
        assert not isinstance(model, torch._dynamo.eval_frame.OptimizedModule)

    def test_compiled_is_optimized_module(self):
        """torch.compile wraps model in OptimizedModule."""
        model = _make_gpt()
        compiled = torch.compile(model, dynamic=False)
        assert isinstance(compiled, torch._dynamo.eval_frame.OptimizedModule)

    def test_compiled_encdec_is_optimized_module(self):
        """EncoderDecoderGPT also becomes OptimizedModule when compiled."""
        model = _make_encdec()
        compiled = torch.compile(model, dynamic=False)
        assert isinstance(compiled, torch._dynamo.eval_frame.OptimizedModule)


class TestCompileCorrectnessGPT:
    """Verify compiled GPT produces same results as uncompiled."""

    def test_compiled_vs_uncompiled_logits_close(self):
        """Compiled and uncompiled GPT produce very similar logits.

        torch.compile may reorder floating-point operations, so exact match
        is not guaranteed. Tolerance allows for bf16-level rounding differences.
        """
        torch._dynamo.config.cache_size_limit = 16
        model = _make_gpt()
        model.eval()
        compiled = torch.compile(model, dynamic=False)

        idx = torch.randint(0, 64, (1, 16))
        with torch.no_grad():
            logits_orig = model(idx)
            logits_compiled = compiled(idx)

        assert torch.allclose(logits_orig, logits_compiled, atol=1e-3), (
            f"Max diff: {(logits_orig - logits_compiled).abs().max().item():.2e}"
        )

    def test_compiled_vs_uncompiled_loss_match(self):
        """Compiled and uncompiled GPT produce identical loss."""
        torch._dynamo.config.cache_size_limit = 16
        model = _make_gpt()
        model.eval()
        compiled = torch.compile(model, dynamic=False)

        idx = torch.randint(0, 64, (2, 16))
        targets = torch.randint(0, 64, (2, 16))
        with torch.no_grad():
            loss_orig = model(idx, targets)
            loss_compiled = compiled(idx, targets)

        assert torch.allclose(loss_orig, loss_compiled, atol=1e-5), (
            f"Loss diff: {abs(loss_orig.item() - loss_compiled.item()):.2e}"
        )

    def test_compiled_backward_no_nan(self):
        """Compiled GPT backward produces no NaN gradients."""
        torch._dynamo.config.cache_size_limit = 16
        model = _make_gpt()
        model.train()
        compiled = torch.compile(model, dynamic=False)

        idx = torch.randint(0, 64, (2, 16))
        targets = torch.randint(0, 64, (2, 16))
        loss = compiled(idx, targets)
        assert not torch.isnan(loss), "NaN loss from compiled GPT"
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"


class TestCompileCorrectnessEncoderDecoder:
    """Verify compiled EncoderDecoderGPT works correctly."""

    def test_compiled_forward_no_nan(self):
        """Compiled EncoderDecoderGPT forward produces no NaN."""
        torch._dynamo.config.cache_size_limit = 32
        model = _make_encdec()
        model.eval()
        compiled = torch.compile(model, dynamic=False)

        B, T, n_seg, Nf = 2, 16, 2, 8
        idx = torch.randint(0, 64, (B, T))
        targets = torch.randint(0, 64, (B, T))
        fids = torch.randint(0, 200, (B, n_seg, Nf))
        mags = torch.randint(0, 4, (B, n_seg, Nf))
        mask = torch.ones(B, n_seg, Nf, dtype=torch.bool)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[:, T // 2:] = 1

        with torch.no_grad():
            loss = compiled(idx, targets, fids, mags, mask, seg_map)
        assert not torch.isnan(loss), "NaN loss from compiled EncoderDecoderGPT"
        assert torch.isfinite(loss), "Inf loss from compiled EncoderDecoderGPT"

    def test_compiled_backward_no_nan(self):
        """Compiled EncoderDecoderGPT backward produces no NaN."""
        torch._dynamo.config.cache_size_limit = 32
        model = _make_encdec()
        model.train()
        compiled = torch.compile(model, dynamic=False)

        B, T, n_seg, Nf = 2, 16, 2, 8
        idx = torch.randint(0, 64, (B, T))
        targets = torch.randint(0, 64, (B, T))
        fids = torch.randint(0, 200, (B, n_seg, Nf))
        mags = torch.randint(0, 4, (B, n_seg, Nf))
        mask = torch.ones(B, n_seg, Nf, dtype=torch.bool)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[:, T // 2:] = 1

        loss = compiled(idx, targets, fids, mags, mask, seg_map)
        loss.backward()

        nan_params = [
            name for name, p in model.named_parameters()
            if p.grad is not None and torch.isnan(p.grad).any()
        ]
        assert len(nan_params) == 0, f"NaN grads in compiled model: {nan_params}"

    def test_compiled_vs_uncompiled_loss_close(self):
        """Compiled and uncompiled EncoderDecoderGPT produce similar loss.

        Note: torch.compile may reorder floating-point operations, so we use
        a wider tolerance than exact match.
        """
        torch._dynamo.config.cache_size_limit = 32
        model = _make_encdec()
        model.eval()
        compiled = torch.compile(model, dynamic=False)

        B, T, n_seg, Nf = 2, 16, 2, 8
        idx = torch.randint(0, 64, (B, T))
        targets = torch.randint(0, 64, (B, T))
        fids = torch.randint(0, 200, (B, n_seg, Nf))
        mags = torch.randint(0, 4, (B, n_seg, Nf))
        mask = torch.ones(B, n_seg, Nf, dtype=torch.bool)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[:, T // 2:] = 1

        with torch.no_grad():
            loss_orig = model(idx, targets, fids, mags, mask, seg_map)
            loss_compiled = compiled(idx, targets, fids, mags, mask, seg_map)

        assert torch.allclose(loss_orig, loss_compiled, atol=1e-3), (
            f"Loss diff: {abs(loss_orig.item() - loss_compiled.item()):.2e}"
        )

    def test_compiled_training_steps_stable(self):
        """Multiple training steps with compiled model are stable."""
        torch._dynamo.config.cache_size_limit = 32
        model = _make_encdec()
        model.train()
        compiled = torch.compile(model, dynamic=False)
        optimizer = model.setup_optimizer(matrix_lr=0.01, embedding_lr=0.1)

        B, T, n_seg, Nf = 2, 16, 2, 8
        for step in range(5):
            idx = torch.randint(0, 64, (B, T))
            targets = torch.randint(0, 64, (B, T))
            fids = torch.randint(0, 200, (B, n_seg, Nf))
            mags = torch.randint(0, 4, (B, n_seg, Nf))
            mask = torch.ones(B, n_seg, Nf, dtype=torch.bool)
            seg_map = torch.zeros(B, T, dtype=torch.long)
            seg_map[:, T // 2:] = 1

            loss = compiled(idx, targets, fids, mags, mask, seg_map)
            assert torch.isfinite(loss), f"Non-finite loss at step {step}"
            loss.backward()
            optimizer.step()
            model.zero_grad(set_to_none=True)
