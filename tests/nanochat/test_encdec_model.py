"""Tests for the EncoderDecoderGPT model."""

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


@pytest.fixture
def tiny_model():
    dec_config = GPTConfig(
        sequence_len=64,
        vocab_size=100,
        n_layer=4,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="L",
    )
    enc_config = EncoderConfig(n_embd=32, depth=1, n_head=2, n_kv_head=2, head_dim=16)
    xattn_config = CrossAttentionConfig(frequency=2, n_head=2, n_kv_head=2)
    model = EncoderDecoderGPT(
        dec_config,
        enc_config,
        xattn_config,
        n_sae_features=50,
        num_buckets=5,
    )
    model.init_weights()
    return model


def _make_inputs(model, B=2, T=16, n_seg=4, F=6):
    """Create random inputs for the encoder-decoder model."""
    idx = torch.randint(0, model.dec_config.vocab_size, (B, T))
    targets = torch.randint(0, model.dec_config.vocab_size, (B, T))
    enc_feature_ids = torch.randint(0, 50, (B, n_seg, F))
    enc_magnitudes = torch.randint(0, 5, (B, n_seg, F))
    enc_mask = torch.ones(B, n_seg, F, dtype=torch.bool)
    # seg_map maps each decoder position to a segment index (must be non-decreasing;
    # the cross-attention scatter/gather assumes contiguous segments)
    segment_size = T // n_seg
    seg_map = torch.arange(T).unsqueeze(0).expand(B, -1) // segment_size
    seg_map = seg_map.clamp(max=n_seg - 1)  # handle remainder tokens
    return idx, targets, enc_feature_ids, enc_magnitudes, enc_mask, seg_map


class TestEncoderDecoderGPTForward:
    def test_loss_returned(self, tiny_model):
        """Forward with targets returns scalar loss > 0."""
        idx, targets, enc_fids, enc_mags, enc_mask, seg_map = _make_inputs(tiny_model)
        loss = tiny_model(idx, targets, enc_fids, enc_mags, enc_mask, seg_map)
        assert loss.ndim == 0, "Loss should be a scalar"
        assert loss.item() > 0, "Loss should be positive"

    def test_gradient_flows_to_encoder(self, tiny_model):
        """loss.backward() gives nonzero grad to encoder.feature_embed.weight."""
        # Perturb cross-attention c_proj away from zero so gradients flow
        # through (init_weights zeros c_proj for residual-friendly init)
        with torch.no_grad():
            for layer in tiny_model.cross_attn_layers.values():
                layer.c_proj.weight.normal_(std=0.01)
        idx, targets, enc_fids, enc_mags, enc_mask, seg_map = _make_inputs(tiny_model)
        loss = tiny_model(idx, targets, enc_fids, enc_mags, enc_mask, seg_map)
        loss.backward()
        grad = tiny_model.encoder.feature_embed.weight.grad
        assert grad is not None, "Encoder feature_embed should have gradients"
        assert grad.abs().sum() > 0, (
            "Encoder feature_embed gradients should be non-zero"
        )

    def test_cross_attention_frequency(self, tiny_model):
        """With frequency=2 and 4 layers, cross-attn at layers 1,3 (2 total)."""
        # Layers 0,1,2,3 -> (i+1) % 2 == 0 at i=1 and i=3
        assert len(tiny_model.cross_attn_layers) == 2
        assert "1" in tiny_model.cross_attn_layers
        assert "3" in tiny_model.cross_attn_layers

    def test_logits_returned_without_targets(self, tiny_model):
        """Forward with targets=None returns (B, T, vocab_size) logits."""
        B, T = 2, 16
        idx, _, enc_fids, enc_mags, enc_mask, seg_map = _make_inputs(
            tiny_model, B=B, T=T
        )
        logits = tiny_model(idx, None, enc_fids, enc_mags, enc_mask, seg_map)
        assert logits.shape == (B, T, tiny_model.dec_config.vocab_size)

    def test_optimizer_setup(self, tiny_model):
        """setup_optimizer() groups all params, total matches model.parameters()."""
        optimizer = tiny_model.setup_optimizer()
        opt_params = set()
        for group in optimizer.param_groups:
            for p in group["params"]:
                opt_params.add(id(p))
        model_params = {id(p) for p in tiny_model.parameters()}
        assert opt_params == model_params, (
            f"Optimizer covers {len(opt_params)} params but model has {len(model_params)}"
        )

    def test_optimizer_trainable_only_excludes_frozen_parameters(self, tiny_model):
        """The opt-in seam omits frozen parameters without changing defaults."""
        tiny_model.lm_head.requires_grad_(False)
        optimizer = tiny_model.setup_optimizer(trainable_only=True)
        optimized = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        expected = {
            id(parameter)
            for parameter in tiny_model.parameters()
            if parameter.requires_grad
        }
        assert optimized == expected
        assert id(tiny_model.lm_head.weight) not in optimized
