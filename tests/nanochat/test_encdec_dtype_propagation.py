"""Tests for dtype propagation through EncoderDecoderGPT forward pass.

Verifies all three components maintain COMPUTE_DTYPE:
1. Encoder: embedding → blocks → final norm
2. Cross-attention: encoder KV (enc_dim) → decoder Q (dec_dim)
3. Decoder: embedding → blocks (with xattn injection) → lm_head
"""

import os

os.environ["NANOCHAT_ATTN_IMPL"] = "sdpa"

import pytest
import torch

from nanochat.common import COMPUTE_DTYPE
from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT


def _make_tiny_model(n_sae_features=200, num_buckets=4):
    dec_config = GPTConfig(
        sequence_len=32, vocab_size=64, n_layer=4,
        n_head=2, n_kv_head=2, n_embd=32, window_pattern="L",
    )
    enc_config = EncoderConfig(n_embd=32, depth=2, n_head=2, n_kv_head=2, head_dim=16)
    xattn_config = CrossAttentionConfig(frequency=2, n_head=2, n_kv_head=2)
    model = EncoderDecoderGPT(
        dec_config, enc_config, xattn_config,
        n_sae_features=n_sae_features, num_buckets=num_buckets,
    )
    model.init_weights()
    return model


def _make_inputs(B=2, T=16, n_seg=2, n_feat=8, n_sae=200, num_buckets=4):
    idx = torch.randint(0, 64, (B, T))
    targets = torch.randint(0, 64, (B, T))
    fids = torch.randint(0, n_sae, (B, n_seg, n_feat))
    mags = torch.randint(0, num_buckets, (B, n_seg, n_feat))
    mask = torch.ones(B, n_seg, n_feat, dtype=torch.bool)
    seg_map = torch.zeros(B, T, dtype=torch.long)
    seg_map[:, T // 2:] = 1  # second half maps to segment 1
    return idx, targets, fids, mags, mask, seg_map


@pytest.fixture(scope="module")
def model():
    return _make_tiny_model()


class TestEncoderDtypePropagation:
    """Encoder output should be in COMPUTE_DTYPE."""

    def test_encoder_output_dtype(self, model):
        """Encoder forward returns tensors in COMPUTE_DTYPE."""
        B, n_seg, n_feat = 2, 2, 8
        fids = torch.randint(0, 200, (B, n_seg, n_feat))
        mags = torch.randint(0, 4, (B, n_seg, n_feat))
        mask = torch.ones(B, n_seg, n_feat, dtype=torch.bool)
        enc_out = model.encoder(fids, mags, mask)
        assert enc_out.values.dtype == COMPUTE_DTYPE, (
            f"Encoder output dtype {enc_out.values.dtype} != {COMPUTE_DTYPE}"
        )

    def test_encoder_block_output_dtype(self, model):
        """Each encoder block outputs COMPUTE_DTYPE (the FP8 fix)."""
        block_dtypes = []

        def make_hook(i):
            def hook(module, input, output):
                block_dtypes.append((i, output.dtype))
            return hook

        handles = []
        for i, block in enumerate(model.encoder.blocks):
            handles.append(block.register_forward_hook(make_hook(i)))
        try:
            B, n_seg, n_feat = 2, 2, 8
            fids = torch.randint(0, 200, (B, n_seg, n_feat))
            mags = torch.randint(0, 4, (B, n_seg, n_feat))
            mask = torch.ones(B, n_seg, n_feat, dtype=torch.bool)
            with torch.no_grad():
                model.encoder(fids, mags, mask)
        finally:
            for h in handles:
                h.remove()

        for i, dtype in block_dtypes:
            assert dtype == COMPUTE_DTYPE, (
                f"Encoder block {i} output dtype {dtype} != {COMPUTE_DTYPE}"
            )


class TestCrossAttentionDtypePropagation:
    """Cross-attention receives correct dtypes from encoder and decoder."""

    def test_cross_attention_output_dtype(self, model):
        """Cross-attention output matches decoder activation dtype."""
        idx, targets, fids, mags, mask, seg_map = _make_inputs()
        xattn_output_dtypes = []

        def make_hook(name):
            def hook(module, input, output):
                xattn_output_dtypes.append((name, output.dtype))
            return hook

        handles = []
        for name, layer in model.cross_attn_layers.items():
            handles.append(layer.register_forward_hook(make_hook(name)))
        try:
            with torch.no_grad():
                model(idx, targets, fids, mags, mask, seg_map)
        finally:
            for h in handles:
                h.remove()

        for name, dtype in xattn_output_dtypes:
            assert dtype == COMPUTE_DTYPE, (
                f"Cross-attention layer {name} output dtype {dtype} != {COMPUTE_DTYPE}"
            )


class TestDecoderDtypePropagation:
    """Decoder activations stay in COMPUTE_DTYPE throughout."""

    def test_decoder_block_input_dtype(self, model):
        """Each decoder block receives COMPUTE_DTYPE input."""
        block_input_dtypes = []

        def make_hook(i):
            def hook(module, args, kwargs):
                block_input_dtypes.append((i, args[0].dtype))
            return hook

        handles = []
        for i, block in enumerate(model.transformer.h):
            handles.append(block.register_forward_pre_hook(make_hook(i), with_kwargs=True))
        try:
            idx, targets, fids, mags, mask, seg_map = _make_inputs()
            with torch.no_grad():
                model(idx, targets, fids, mags, mask, seg_map)
        finally:
            for h in handles:
                h.remove()

        for i, dtype in block_input_dtypes:
            assert dtype == COMPUTE_DTYPE, (
                f"Decoder block {i} input dtype {dtype} != {COMPUTE_DTYPE}. "
                "resid_lambdas fp32 may have promoted activations."
            )

    def test_rotary_embeddings_dtype(self, model):
        """Rotary embeddings are in COMPUTE_DTYPE."""
        assert model.cos.dtype == COMPUTE_DTYPE
        assert model.sin.dtype == COMPUTE_DTYPE

    def test_full_forward_no_nan(self, model):
        """Full encoder-decoder forward produces no NaN."""
        idx, targets, fids, mags, mask, seg_map = _make_inputs()
        with torch.no_grad():
            loss = model(idx, targets, fids, mags, mask, seg_map)
        assert not torch.isnan(loss), "NaN loss in EncoderDecoderGPT forward"

    def test_full_backward_no_nan(self):
        """Full encoder-decoder backward produces no NaN gradients."""
        from unittest.mock import patch

        import nanochat.flash_attention as fa_mod

        model = _make_tiny_model()
        model.train()
        idx, targets, fids, mags, mask, seg_map = _make_inputs()
        # Force SDPA path: FlexAttention does not support backward on CPU
        with patch.object(fa_mod, "_use_flex", return_value=False), \
             patch.object(fa_mod, "_use_fa3", return_value=False):
            loss = model(idx, targets, fids, mags, mask, seg_map)
            loss.backward()
        nan_params = [
            name for name, p in model.named_parameters()
            if p.grad is not None and torch.isnan(p.grad).any()
        ]
        assert len(nan_params) == 0, f"NaN gradients in: {nan_params}"
