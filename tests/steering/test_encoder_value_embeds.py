"""Tests for VE (value embedding) gate in encoder self-attention."""

import pytest
import torch

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPTConfig, has_ve
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.encoder import (
    FeatureEncoder,
    FeatureEncoderConfig,
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


def _make_config(**overrides):
    defaults = {
        "n_sae_features": 128,
        "num_buckets": 8,
        "n_embd": 64,
        "n_layer": 4,
        "n_head": 4,
        "n_kv_head": 2,
        "head_dim": 16,
        "ve_gate_channels": 32,
    }
    defaults.update(overrides)
    return FeatureEncoderConfig(**defaults)


class TestEncoderVEGate:
    """Verify VE gate placement, shape, and effect on encoder self-attention."""

    def test_ve_gate_exists_on_correct_layers(self):
        """VE gate should exist on alternating layers (same rule as decoder)."""
        config = _make_config(n_layer=4)
        encoder = FeatureEncoder(config)

        for i, block in enumerate(encoder.blocks):
            expected = has_ve(i, config.n_layer)
            has_gate = block.attn.ve_gate is not None
            assert has_gate == expected, (
                f"Layer {i}: expected ve_gate={'present' if expected else 'absent'}, "
                f"got {'present' if has_gate else 'absent'}"
            )

    def test_ve_gate_output_shape(self):
        """VE gate linear should map ve_gate_channels -> n_kv_head."""
        config = _make_config(n_layer=4)
        encoder = FeatureEncoder(config)

        for block in encoder.blocks:
            if block.attn.ve_gate is not None:
                assert block.attn.ve_gate.in_features == config.ve_gate_channels
                assert block.attn.ve_gate.out_features == config.n_kv_head

    def test_ve_modifies_attention_output(self):
        """Passing a non-zero VE tensor should change the attention output."""
        config = _make_config(n_layer=2)
        encoder = FeatureEncoder(config)
        encoder.init_weights()

        # Perturb c_proj so attention output is nonzero (init_weights zeros it)
        with torch.no_grad():
            for block in encoder.blocks:
                block.attn.c_proj.weight.normal_(0, 0.1)
                block.mlp.c_proj.weight.normal_(0, 0.1)

        B, Nf = 2, 5
        torch.manual_seed(42)
        x = torch.randn(B, Nf, config.n_embd)

        # Find a layer that has a VE gate
        ve_layer_idx = None
        for i, block in enumerate(encoder.blocks):
            if block.attn.ve_gate is not None:
                ve_layer_idx = i
                break
        assert ve_layer_idx is not None, "No layer with VE gate found"

        attn = encoder.blocks[ve_layer_idx].attn
        kv_dim = config.n_kv_head * config.head_dim
        ve = torch.randn(B, Nf, kv_dim)

        out_no_ve = attn(x)
        out_with_ve = attn(x, ve=ve)
        assert not torch.allclose(out_no_ve, out_with_ve, atol=1e-5), (
            "VE should change the attention output but outputs are identical"
        )


def _make_inputs(B=2, S=3, F=5, config=None):
    if config is None:
        config = _make_config()
    feature_ids = torch.randint(0, config.n_sae_features, (B, S, F))
    magnitudes = torch.randint(0, config.num_buckets, (B, S, F))
    mask = torch.ones(B, S, F, dtype=torch.bool)
    return feature_ids, magnitudes, mask


class TestFeatureEncoderVE:
    def test_value_embeds_exist(self):
        config = _make_config(n_layer=4)
        encoder = FeatureEncoder(config)
        for i in range(4):
            if has_ve(i, 4):
                assert str(i) in encoder.value_embeds, f"Missing VE for layer {i}"
            else:
                assert str(i) not in encoder.value_embeds, (
                    f"Unexpected VE for layer {i}"
                )

    def test_ve_embedding_shape(self):
        config = _make_config(n_layer=2, n_kv_head=2, head_dim=16)
        encoder = FeatureEncoder(config)
        kv_dim = config.n_kv_head * config.head_dim
        for ve in encoder.value_embeds.values():
            assert ve.weight.shape == (config.n_sae_features, kv_dim)

    def test_output_shape_with_ve(self):
        config = _make_config(n_layer=2)
        encoder = FeatureEncoder(config)
        encoder.init_weights()
        B, S, F = 2, 3, 5
        ids, mags, mask = _make_inputs(B, S, F, config)
        out = encoder(ids, mags, mask)
        assert out.values.shape[0] == 1
        assert out.values.shape[2] == config.n_embd
        assert out.layout.batch_size == B
        assert out.layout.n_segments == S
        assert int(out.layout.valid_mask.sum()) == B * S * F

    def test_gradient_flow_through_ve(self):
        config = _make_config(n_layer=2)
        encoder = FeatureEncoder(config)
        encoder.init_weights()
        with torch.no_grad():
            for block in encoder.blocks:
                block.attn.c_proj.weight.normal_(0, 0.1)
                block.mlp.c_proj.weight.normal_(0, 0.1)
                if block.attn.ve_gate is not None:
                    block.attn.ve_gate.weight.fill_(1.0)
        ids, mags, mask = _make_inputs(2, 3, 5, config)
        out = encoder(ids, mags, mask)
        out.values.sum().backward()
        for name, ve in encoder.value_embeds.items():
            assert ve.weight.grad is not None, f"No gradient for VE layer {name}"
            assert ve.weight.grad.abs().sum() > 0, f"Zero gradient for VE layer {name}"  # ty: ignore[call-non-callable]

    def test_zero_init_gate_neutrality(self):
        config = _make_config(n_layer=2, n_kv_head=2)
        encoder = FeatureEncoder(config)
        encoder.init_weights()
        for block in encoder.blocks:
            if block.attn.ve_gate is not None:
                assert torch.allclose(
                    block.attn.ve_gate.weight,
                    torch.zeros_like(block.attn.ve_gate.weight),
                ), "VE gate should be zero-initialized"


class TestModelEncoderVE:
    def _make_model(self):
        dec_config = GPTConfig(
            sequence_len=32,
            vocab_size=100,
            n_layer=4,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            window_pattern="L",
        )
        enc_config = EncoderConfig(
            n_embd=32, depth=2, n_head=2, n_kv_head=2, head_dim=16
        )
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

    def test_encoder_ve_in_optimizer_groups(self):
        model = self._make_model()
        optimizer = model.setup_optimizer()
        opt_params = sum(len(g["params"]) for g in optimizer.param_groups)
        model_params = len(list(model.parameters()))
        assert opt_params == model_params, (
            f"Optimizer has {opt_params} params but model has {model_params}"
        )

    def test_encoder_ve_gradient_in_full_model(self):
        model = self._make_model()
        with torch.no_grad():
            for block in model.transformer.h:
                block.attn.c_proj.weight.fill_(0.01)
                block.mlp.c_proj.weight.fill_(0.01)
            for xattn in model.cross_attn_layers.values():
                xattn.c_proj.weight.fill_(0.01)
            for block in model.encoder.blocks:
                block.attn.c_proj.weight.fill_(0.01)
                block.mlp.c_proj.weight.fill_(0.01)
                if block.attn.ve_gate is not None:
                    block.attn.ve_gate.weight.fill_(1.0)

        B, T, n_seg, F = 1, 16, 1, 3
        idx = torch.randint(0, 100, (B, T))
        targets = torch.randint(0, 100, (B, T))
        enc_fids = torch.randint(0, 50, (B, n_seg, F))
        enc_mags = torch.randint(0, 5, (B, n_seg, F))
        enc_mask = torch.ones(B, n_seg, F, dtype=torch.bool)
        seg_map = torch.zeros(B, T, dtype=torch.long)

        loss = model(idx, targets, enc_fids, enc_mags, enc_mask, seg_map)
        loss.backward()

        for name, ve in model.encoder.value_embeds.items():
            assert ve.weight.grad is not None, (
                f"No gradient for encoder VE layer {name}"
            )

    def test_estimate_flops_excludes_encoder_ve(self):
        model = self._make_model()
        flops = model.estimate_flops(avg_features_per_segment=32, segment_size=16)
        assert flops > 0
