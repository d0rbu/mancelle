"""Tests for dtype propagation through GPT forward pass (nanochat/gpt.py).

Key fix: resid_lambdas are fp32 parameters. Multiplying them with bf16 activations
promotes x to fp32. The fix adds x.to(COMPUTE_DTYPE) after the blending (line 563).
"""

import os
from unittest.mock import patch

import torch
import pytest

# Force SDPA backend for CPU testing
os.environ["NANOCHAT_ATTN_IMPL"] = "sdpa"

from nanochat.common import COMPUTE_DTYPE
from nanochat.gpt import GPT, GPTConfig, norm


@pytest.fixture(scope="module")
def tiny_gpt():
    """Tiny GPT model for dtype tests."""
    config = GPTConfig(
        sequence_len=32,
        vocab_size=64,
        n_layer=4,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
    )
    model = GPT(config, pad_vocab_size_to=64)
    model.init_weights()
    model.eval()
    return model


class TestGPTDtypePropagation:
    """Verify dtype consistency through GPT forward pass."""

    def test_rotary_embeddings_dtype(self, tiny_gpt):
        """Rotary embeddings are in COMPUTE_DTYPE after init."""
        assert tiny_gpt.cos.dtype == COMPUTE_DTYPE, (
            f"cos dtype {tiny_gpt.cos.dtype} != {COMPUTE_DTYPE}"
        )
        assert tiny_gpt.sin.dtype == COMPUTE_DTYPE, (
            f"sin dtype {tiny_gpt.sin.dtype} != {COMPUTE_DTYPE}"
        )

    def test_embedding_cast_to_compute_dtype(self, tiny_gpt):
        """Token embedding output is cast to COMPUTE_DTYPE in forward."""
        idx = torch.randint(0, 64, (1, 16))
        # Hook into the first block to check x dtype entering it
        dtypes_seen = []

        def hook(module, args, kwargs):
            x = args[0]
            dtypes_seen.append(x.dtype)

        handle = tiny_gpt.transformer.h[0].register_forward_pre_hook(hook, with_kwargs=True)
        try:
            with torch.no_grad():
                tiny_gpt(idx)
        finally:
            handle.remove()

        assert dtypes_seen[0] == COMPUTE_DTYPE, (
            f"First block received dtype {dtypes_seen[0]}, expected {COMPUTE_DTYPE}"
        )

    def test_resid_lambdas_are_fp32(self, tiny_gpt):
        """resid_lambdas and x0_lambdas are fp32 parameters (not cast)."""
        assert tiny_gpt.resid_lambdas.dtype == torch.float32
        assert tiny_gpt.x0_lambdas.dtype == torch.float32

    def test_activations_stay_compute_dtype_after_lambda_blend(self, tiny_gpt):
        """After resid_lambdas * x + x0_lambdas * x0, x is cast back to COMPUTE_DTYPE.

        Without the fix, fp32 lambdas would promote bf16 x to fp32 and it would
        propagate through Linear (which casts weight to x.dtype) all the way to
        flash attention, causing dtype mismatches.
        """
        idx = torch.randint(0, 64, (1, 16))
        dtypes_per_block = []

        def make_hook(layer_idx):
            def hook(module, args, kwargs):
                x = args[0]
                dtypes_per_block.append((layer_idx, x.dtype))
            return hook

        handles = []
        for i, block in enumerate(tiny_gpt.transformer.h):
            h = block.register_forward_pre_hook(make_hook(i), with_kwargs=True)
            handles.append(h)
        try:
            with torch.no_grad():
                tiny_gpt(idx)
        finally:
            for h in handles:
                h.remove()

        for layer_idx, dtype in dtypes_per_block:
            assert dtype == COMPUTE_DTYPE, (
                f"Block {layer_idx} received {dtype}, expected {COMPUTE_DTYPE}. "
                "resid_lambdas fp32 may have promoted activations."
            )

    def test_logits_are_fp32(self, tiny_gpt):
        """Logits are explicitly cast to fp32 for softcap and loss computation."""
        idx = torch.randint(0, 64, (1, 16))
        with torch.no_grad():
            logits = tiny_gpt(idx)
        assert logits.dtype == torch.float32, (
            f"Logits should be fp32, got {logits.dtype}"
        )

    def test_forward_no_nan(self, tiny_gpt):
        """Forward pass produces no NaN values."""
        idx = torch.randint(0, 64, (1, 16))
        with torch.no_grad():
            logits = tiny_gpt(idx)
        assert not torch.isnan(logits).any(), "NaN in GPT logits"

    def test_backward_no_nan_grads(self):
        """Backward pass produces no NaN gradients."""
        import nanochat.flash_attention as _fa
        from nanochat.flash_attention import FlashAttnImpl

        config = GPTConfig(
            sequence_len=32, vocab_size=64, n_layer=4,
            n_head=2, n_kv_head=2, n_embd=32, window_pattern="L",
        )
        model = GPT(config, pad_vocab_size_to=64)
        model.init_weights()
        model.train()
        idx = torch.randint(0, 64, (2, 16))
        targets = torch.randint(0, 64, (2, 16))
        # Force SDPA: flex_attention backward is unsupported on CPU
        old = _fa._override_impl
        _fa._override_impl = FlashAttnImpl.SDPA
        try:
            loss = model(idx, targets)
            loss.backward()
        finally:
            _fa._override_impl = old
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), (
                    f"NaN gradient in {name}"
                )
