"""Verify gradient flow through the full encoder-decoder pipeline."""

import pytest
import torch

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT


@pytest.fixture(autouse=True)
def _force_sdpa():
    """Force SDPA attention backend for all tests in this module.

    FlexAttention does not support backward on CPU. We patch the module-level
    override directly rather than using an env var, because the flash_attention
    module may already be imported by earlier tests in the suite.
    """
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


def test_gradient_flow_all_components():
    """Verify gradients flow from LM loss back through all components."""
    dec_config = GPTConfig(
        sequence_len=32,
        vocab_size=100,
        n_layer=4,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="L",
    )
    enc_config = EncoderConfig(n_embd=32, depth=2, n_head=2, n_kv_head=2, head_dim=16)
    xattn_config = CrossAttentionConfig(frequency=2, n_head=2, n_kv_head=2)

    model = EncoderDecoderGPT(
        dec_config,
        enc_config,
        xattn_config,
        n_sae_features=50,
        num_buckets=5,
    )
    model.init_weights()

    # IMPORTANT: c_proj in decoder, cross-attention, and encoder blocks are zero-initialized.
    # This means at init, attention and cross-attention contribute nothing to the residual stream,
    # so gradients can't flow through. We need to perturb c_proj weights to allow gradient flow.
    for block in model._decoder_blocks:
        block.attn.c_proj.weight.data.fill_(0.01)
        block.mlp.c_proj.weight.data.fill_(0.01)
    for xattn in model._cross_attention_layer_values():
        xattn.c_proj.weight.data.fill_(0.01)
    for block in model.encoder.blocks:
        block.attn.c_proj.weight.data.fill_(0.01)
        block.mlp.c_proj.weight.data.fill_(0.01)

    B, T = 1, 16
    n_seg = 1
    F = 3
    idx = torch.randint(0, 100, (B, T))
    targets = torch.randint(0, 100, (B, T))
    enc_fids = torch.randint(0, 50, (B, n_seg, F))
    enc_mags = torch.randint(0, 5, (B, n_seg, F))
    enc_mask = torch.ones(B, n_seg, F, dtype=torch.bool)
    seg_map = torch.zeros(B, T, dtype=torch.long)

    loss = model(idx, targets, enc_fids, enc_mags, enc_mask, seg_map)
    loss.backward()

    # Check gradients exist and are nonzero for all components
    checks = {
        "decoder wte": model._decoder_embedding.weight,
        "lm_head": model.lm_head.weight,
        "cross_attn c_q": model._cross_attention_layer_values()[0].c_q.weight,
        "cross_attn c_kv": model._cross_attention_layer_values()[0].c_kv.weight,
        "cross_attn c_proj": model._cross_attention_layer_values()[0].c_proj.weight,
        "encoder feature_embed": model.encoder.feature_embed.weight,
        "encoder magnitude_embed": model.encoder.magnitude_embed.weight,
    }

    # Check decoder layers with cross-attention get gradients
    for layer_idx in model.cross_attn_layers.keys():
        layer_idx_int = int(layer_idx)
        checks[f"decoder block {layer_idx_int} attn.c_qkv"] = model._decoder_blocks[
            layer_idx_int
        ].attn.c_qkv.weight

    if len(model.encoder.blocks) > 0:
        encoder_block = model.encoder.blocks[0]
        checks["encoder block 0 attn.c_qkv"] = encoder_block.attn.c_qkv.weight
        checks["encoder block 0 mlp.c_fc"] = encoder_block.mlp.c_fc.weight

    for name, param in checks.items():
        grad = param.grad
        assert grad is not None, f"No gradient for {name}"
        grad_norm = grad.abs().sum().item()
        assert grad_norm > 0, f"Zero gradient for {name} (grad norm = {grad_norm})"
