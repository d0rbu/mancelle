"""Slice 08 audit (A4 / map.md H4, B1, I10).

Pin the ``init_weights`` contract — particularly the c_proj zero-init that
remains intentional on the encoder side, in contrast to cross-attention
which switched away from zero-init due to the FP8 amax=0 issue (see
map.md §6 B1 and §8 gotcha #7).

Tests:
* attention ``c_proj`` weight is exactly zero after ``init_weights``.
* MLP ``c_proj`` weight is exactly zero after ``init_weights``.
* ``ve_gate`` weight (where present) is exactly zero (so 2*sigmoid(0)=1).
* ``c_qkv`` and ``mlp.c_fc`` are non-zero (uniform init).
* embeddings (``feature_embed``, ``magnitude_embed``, ``value_embeds``)
  are non-zero post-init.
* The encoder/cross-attention asymmetry is intentional: explicit
  documentation pin so any future change to the schema must update
  this test (and findings.md note).
"""

from __future__ import annotations

import os

os.environ["NANOCHAT_ATTN_IMPL"] = "sdpa"


from steering.conditional.encoder_decoder.encoder import (
    FeatureEncoder,
    FeatureEncoderConfig,
)


def _make_config(**overrides) -> FeatureEncoderConfig:
    defaults = {
        "n_sae_features": 64,
        "num_buckets": 4,
        "n_embd": 32,
        "n_layer": 2,
        "n_head": 4,
        "n_kv_head": 2,
        "head_dim": 8,
    }
    defaults.update(overrides)
    return FeatureEncoderConfig(**defaults)


class TestInitWeightsContract:
    """Pin the init_weights schema documented in map.md §3 I10 and §6 B1."""

    def test_attn_c_proj_is_zero(self):
        """A4: every block's attention c_proj weight is exactly zero post-init."""
        config = _make_config(n_layer=4)
        model = FeatureEncoder(config)
        model.init_weights()
        for i, block in enumerate(model.blocks):
            w = block.attn.c_proj.weight
            assert w.abs().sum().item() == 0.0, (
                f"block[{i}].attn.c_proj.weight has non-zero entries after init "
                f"(sum |w| = {w.abs().sum().item()}). The encoder c_proj zero-init "
                "is intentional per map.md §6 B1; cross-attention uses uniform init."
            )

    def test_mlp_c_proj_is_zero(self):
        """A4: every block's MLP c_proj weight is exactly zero post-init."""
        config = _make_config(n_layer=4)
        model = FeatureEncoder(config)
        model.init_weights()
        for i, block in enumerate(model.blocks):
            w = block.mlp.c_proj.weight
            assert w.abs().sum().item() == 0.0, (
                f"block[{i}].mlp.c_proj.weight has non-zero entries after init "
                f"(sum |w| = {w.abs().sum().item()})."
            )

    def test_ve_gate_is_zero_when_present(self):
        """A4: ve_gate weight is zero post-init (so 2*sigmoid(0)=1, neutral gate).

        Maps to map.md §3 I9.
        """
        config = _make_config(n_layer=4)
        model = FeatureEncoder(config)
        model.init_weights()
        any_ve_present = False
        for i, block in enumerate(model.blocks):
            if block.attn.ve_gate is None:
                continue
            any_ve_present = True
            w = block.attn.ve_gate.weight
            assert w.abs().sum().item() == 0.0, (
                f"block[{i}].attn.ve_gate.weight is not all-zero after init."
            )
        assert any_ve_present, "test config should include at least one VE layer"

    def test_c_qkv_is_non_zero(self):
        """A4: c_qkv is uniformly initialized with sqrt(3/n_embd) — non-zero with high prob."""
        config = _make_config(n_layer=2)
        model = FeatureEncoder(config)
        model.init_weights()
        for i, block in enumerate(model.blocks):
            w = block.attn.c_qkv.weight
            # Vanishing probability of all-zero uniform; assert sum > 0.
            assert w.abs().sum().item() > 0.0, (
                f"block[{i}].attn.c_qkv.weight unexpectedly all-zero post-init."
            )

    def test_mlp_c_fc_is_non_zero(self):
        """A4: c_fc is uniformly initialized — non-zero with high prob."""
        config = _make_config(n_layer=2)
        model = FeatureEncoder(config)
        model.init_weights()
        for i, block in enumerate(model.blocks):
            w = block.mlp.c_fc.weight
            assert w.abs().sum().item() > 0.0, (
                f"block[{i}].mlp.c_fc.weight unexpectedly all-zero post-init."
            )

    def test_feature_and_magnitude_embeds_non_zero(self):
        """A4: embeddings are Normal(0,1) — non-zero with high prob."""
        config = _make_config()
        model = FeatureEncoder(config)
        model.init_weights()
        assert model.feature_embed.weight.abs().sum().item() > 0.0
        assert model.magnitude_embed.weight.abs().sum().item() > 0.0

    def test_value_embeds_non_zero(self):
        """A4: value_embeds entries are Uniform(±sqrt(3/n_embd)) — non-zero."""
        config = _make_config(n_layer=4)
        model = FeatureEncoder(config)
        model.init_weights()
        assert len(model.value_embeds) > 0
        for key, ve in model.value_embeds.items():
            assert ve.weight.abs().sum().item() > 0.0, (  # ty: ignore[call-non-callable]
                f"value_embeds[{key}] is unexpectedly all-zero post-init."
            )

    def test_init_is_idempotent(self):
        """A4 / hardening: calling init_weights twice produces same c_proj zeros (deterministic for the zero parts)."""
        config = _make_config(n_layer=2)
        model = FeatureEncoder(config)
        model.init_weights()
        # The zero-init parts must remain zero after a second call.
        model.init_weights()
        for block in model.blocks:
            assert block.attn.c_proj.weight.abs().sum().item() == 0.0
            assert block.mlp.c_proj.weight.abs().sum().item() == 0.0
            if block.attn.ve_gate is not None:
                assert block.attn.ve_gate.weight.abs().sum().item() == 0.0
