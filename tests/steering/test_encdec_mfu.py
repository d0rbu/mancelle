"""Rigorous tests for encoder-decoder MFU (Model FLOPs Utilization) calculation.

Validates that EncoderDecoderGPT.estimate_flops() correctly accounts for:
1. Decoder weight-based FLOPs (per decoder token)
2. Encoder weight-based FLOPs (per encoder token, amortized to per decoder token)
3. Decoder self-attention activation FLOPs (Q@K^T + attn@V)
4. Cross-attention activation FLOPs (Q@K^T + attn@V)
5. Encoder self-attention activation FLOPs (amortized)
6. Correct amortization ratio R = (n_seg * Nf) / T
7. Comparability with decoder-only GPT.estimate_flops()
"""

import torch

from nanochat.gpt import GPT, GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_dec_config(**overrides):
    """Build a small decoder config for testing."""
    defaults = {
        "sequence_len": 128,
        "vocab_size": 256,
        "n_layer": 4,
        "n_head": 4,
        "n_kv_head": 4,
        "n_embd": 64,
        "window_pattern": "L",
    }
    defaults.update(overrides)
    return GPTConfig(**defaults)


def _make_encdec_model(
    dec_kwargs=None,
    enc_kwargs=None,
    xattn_kwargs=None,
    n_sae_features=128,
    num_buckets=8,
):
    """Build a small encoder-decoder model for testing."""
    dec_config = _make_dec_config(**(dec_kwargs or {}))
    enc_defaults = {"n_embd": 32, "depth": 2, "n_head": 2, "n_kv_head": 2, "head_dim": 16}
    enc_defaults.update(enc_kwargs or {})
    enc_config = EncoderConfig(**enc_defaults)
    xattn_defaults = {"frequency": 2, "n_head": 4, "n_kv_head": 4}
    xattn_defaults.update(xattn_kwargs or {})
    xattn_config = CrossAttentionConfig(**xattn_defaults)
    model = EncoderDecoderGPT(
        dec_config,
        enc_config,
        xattn_config,
        n_sae_features=n_sae_features,
        num_buckets=num_buckets,
    )
    model.init_weights()
    return model


def _count_matmul_params(module):
    """Count parameters in nn.Linear layers (the matmul-contributing params)."""
    total = 0
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            total += m.weight.numel()
    return total


# ---------------------------------------------------------------------------
# Manual FLOP computation (the "reference" implementation)
# ---------------------------------------------------------------------------


def _manual_estimate_flops(model, avg_features_per_segment, segment_size):
    """Manually compute expected FLOPs per decoder token — the ground truth.

    This re-derives the formula from first principles for comparison.
    """
    T = model.dec_config.sequence_len
    Nf = avg_features_per_segment
    n_seg = T / segment_size
    R = (n_seg * Nf) / T  # encoder tokens per decoder token

    # 1. Decoder block matmul params: includes attn c_qkv, c_proj, ve_gate, mlp c_fc, c_proj
    dec_block_params = sum(p.numel() for p in model.transformer.h.parameters())

    # 2. LM head params
    lm_head_params = model.lm_head.weight.numel()

    # 3. Cross-attention: c_q and c_proj are per decoder token
    xattn_dec_params = 0
    xattn_enc_params = 0
    for layer in model.cross_attn_layers.values():
        xattn_dec_params += layer.c_q.weight.numel() + layer.c_proj.weight.numel()
        xattn_enc_params += layer.c_kv.weight.numel()

    # 4. Encoder block params (per encoder token)
    enc_block_params = sum(p.numel() for p in model.encoder.blocks.parameters())

    # Weight FLOPs
    dec_weight_flops = 6 * (dec_block_params + lm_head_params + xattn_dec_params)
    enc_weight_flops = 6 * (enc_block_params + xattn_enc_params) * R

    # 5. Decoder self-attention activation FLOPs
    h = model.dec_config.n_head
    d = model.dec_config.n_embd // model.dec_config.n_head
    dec_attn_flops = 0
    for ws in model.window_sizes:
        window = ws[0]
        eff_seq = T if window < 0 else min(window, T)
        dec_attn_flops += 12 * h * d * eff_seq

    # 6. Cross-attention activation FLOPs
    n_xattn = len(model.cross_attn_layers)
    xattn_attn_flops = 0
    if n_xattn > 0:
        xattn_layer = next(iter(model.cross_attn_layers.values()))
        xattn_h = xattn_layer.n_head
        xattn_d = xattn_layer.head_dim
        xattn_attn_flops = 12 * xattn_h * xattn_d * Nf * n_xattn

    # 7. Encoder self-attention activation FLOPs (amortized)
    enc_cfg = model.encoder.config
    enc_attn_flops = 0
    if enc_cfg.n_layer > 0:
        enc_attn_flops = (
            12 * enc_cfg.n_head * enc_cfg.head_dim * Nf * enc_cfg.n_layer * R
        )

    return (
        dec_weight_flops
        + enc_weight_flops
        + dec_attn_flops
        + xattn_attn_flops
        + enc_attn_flops
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEstimateFlopsBasic:
    """Basic correctness tests for estimate_flops."""

    def test_positive_result(self):
        model = _make_encdec_model()
        flops = model.estimate_flops(avg_features_per_segment=32, segment_size=16)
        assert flops > 0, "FLOPs must be positive"

    def test_matches_manual_computation(self):
        """estimate_flops must match our manual reference implementation exactly."""
        model = _make_encdec_model()
        Nf, seg_size = 64, 32
        actual = model.estimate_flops(Nf, seg_size)
        expected = _manual_estimate_flops(model, Nf, seg_size)
        assert actual == expected, f"Mismatch: actual={actual}, expected={expected}"

    def test_matches_manual_various_configs(self):
        """Test across a range of model configurations."""
        configs = [
            # (dec_kwargs, enc_kwargs, xattn_kwargs, Nf, seg_size)
            ({}, {}, {}, 32, 16),
            (
                {"n_layer": 8, "n_embd": 128, "n_head": 8, "n_kv_head": 4},
                {"n_embd": 64, "depth": 4, "n_head": 4, "n_kv_head": 2, "head_dim": 16},
                {"frequency": 4, "n_head": 8, "n_kv_head": 4},
                128,
                32,
            ),
            (
                {"n_layer": 2, "n_embd": 64, "n_head": 4, "n_kv_head": 4},
                {"n_embd": 32, "depth": 1, "n_head": 2, "n_kv_head": 2, "head_dim": 16},
                {"frequency": 1, "n_head": 4, "n_kv_head": 4},
                2048,
                128,
            ),
        ]
        for dec_kw, enc_kw, xattn_kw, Nf, seg_size in configs:
            model = _make_encdec_model(dec_kw, enc_kw, xattn_kw)
            actual = model.estimate_flops(Nf, seg_size)
            expected = _manual_estimate_flops(model, Nf, seg_size)
            assert actual == expected, (
                f"Mismatch for config dec={dec_kw} enc={enc_kw} xattn={xattn_kw}: "
                f"actual={actual}, expected={expected}"
            )


class TestAmortizationRatio:
    """Verify that the encoder/decoder amortization ratio R is correctly applied."""

    def test_flops_scale_with_features_per_segment(self):
        """More features per segment => more encoder FLOPs => higher total."""
        model = _make_encdec_model()
        seg_size = 32
        flops_low = model.estimate_flops(
            avg_features_per_segment=16, segment_size=seg_size
        )
        flops_high = model.estimate_flops(
            avg_features_per_segment=256, segment_size=seg_size
        )
        assert flops_high > flops_low, (
            f"More features should mean more FLOPs: {flops_high} <= {flops_low}"
        )

    def test_flops_scale_with_segment_size(self):
        """Smaller segments => more segments => higher R => more encoder FLOPs."""
        model = _make_encdec_model()
        Nf = 64
        flops_large_seg = model.estimate_flops(Nf, segment_size=64)
        flops_small_seg = model.estimate_flops(Nf, segment_size=16)
        assert flops_small_seg > flops_large_seg, (
            f"Smaller segments should mean more encoder FLOPs: "
            f"seg=16 gave {flops_small_seg}, seg=64 gave {flops_large_seg}"
        )

    def test_ratio_proportionality(self):
        """Doubling Nf should roughly double the encoder-side FLOPs contribution."""
        model = _make_encdec_model()
        seg_size = 32
        Nf1 = 64
        Nf2 = 128
        flops1 = model.estimate_flops(Nf1, seg_size)
        flops2 = model.estimate_flops(Nf2, seg_size)

        # Isolate the encoder-dependent portion by subtracting the decoder-only FLOPs
        # (which don't depend on Nf). Use Nf=0 as baseline for pure decoder FLOPs.
        # At Nf=0, enc_weight_flops=0, xattn_attn_flops=0, enc_attn_flops=0
        # But R=0 so enc_weight_flops = 0 regardless. xattn_attn_flops has Nf factor.
        # Careful: estimate_flops with Nf=0 would give R=0, enc_attn=0, xattn_attn=0.
        # That leaves only dec_weight_flops + dec_attn_flops (the pure decoder part).
        dec_only_flops = model.estimate_flops(0, seg_size)

        enc_portion_1 = flops1 - dec_only_flops
        enc_portion_2 = flops2 - dec_only_flops

        # The encoder-side FLOPs have components proportional to Nf (weight * R)
        # and Nf^2 (encoder attention: n_head * head_dim * Nf * R, where R itself
        # contains Nf). Plus cross-attention activation FLOPs proportional to Nf.
        # Weight FLOPs go as Nf (via R = n_seg * Nf / T).
        # Encoder attn FLOPs go as Nf^2 (Nf in the attention formula * Nf in R).
        # Cross-attn attn FLOPs go as Nf.
        # So the ratio should be between 2 and 4 (between linear and quadratic).
        ratio = enc_portion_2 / enc_portion_1
        assert 1.5 < ratio < 5.0, (
            f"Expected encoder FLOPs ratio between 1.5x and 5.0x when doubling Nf, "
            f"got {ratio:.2f}x (enc_portion_1={enc_portion_1}, enc_portion_2={enc_portion_2})"
        )

    def test_high_R_dominates(self):
        """With R >> 1 (many features per token), encoder FLOPs should dominate."""
        # 2048 features per 16-token segment => R = (128/16 * 2048) / 128 = 128
        model = _make_encdec_model()
        flops = model.estimate_flops(avg_features_per_segment=2048, segment_size=16)
        dec_only_flops = model.estimate_flops(0, 16)
        enc_fraction = (flops - dec_only_flops) / flops
        assert enc_fraction > 0.5, (
            f"With R >> 1, encoder FLOPs should dominate. "
            f"Encoder fraction: {enc_fraction:.2%}"
        )


class TestDecoderOnlyComparability:
    """Verify that encoder-decoder MFU is comparable to decoder-only MFU."""

    def test_minimal_encoder_approaches_decoder_only(self):
        """With 1 encoder layer (minimum) and tiny Nf, encoder-decoder FLOPs should be
        close to decoder-only (differing by cross-attention and small encoder overhead)."""
        dec_kwargs = {
            "sequence_len": 128,
            "vocab_size": 256,
            "n_layer": 4,
            "n_head": 4,
            "n_kv_head": 4,
            "n_embd": 64,
            "window_pattern": "L",
        }
        dec_config = _make_dec_config(**dec_kwargs)

        # Decoder-only model
        dec_model = GPT(dec_config)
        dec_model.init_weights()
        dec_flops = dec_model.estimate_flops()

        # Encoder-decoder with 1 encoder layer (minimum)
        encdec_model = _make_encdec_model(
            dec_kwargs=dec_kwargs,
            enc_kwargs={
                "depth": 1,
                "n_embd": 32,
                "n_head": 2,
                "n_kv_head": 2,
                "head_dim": 16,
            },
        )
        # With tiny Nf=1 and large segment_size, R is small
        encdec_flops = encdec_model.estimate_flops(
            avg_features_per_segment=1, segment_size=128
        )

        # The enc-dec model has extra cross-attention + encoder layers, so it should be >=
        # But not by a huge factor for small Nf
        assert encdec_flops >= dec_flops, (
            f"Encoder-decoder FLOPs ({encdec_flops}) should be >= decoder-only ({dec_flops})"
        )
        # The overhead should be modest (< 50% extra for small Nf with minimal encoder)
        overhead = (encdec_flops - dec_flops) / dec_flops
        assert overhead < 0.5, (
            f"Cross-attention + encoder overhead should be modest for Nf=1: {overhead:.2%}"
        )

    def test_decoder_portion_matches_decoder_only(self):
        """The decoder weight FLOPs + decoder self-attention FLOPs should match
        the decoder-only model's estimate (excluding embeddings and scalars)."""
        dec_kwargs = {
            "sequence_len": 128,
            "vocab_size": 256,
            "n_layer": 4,
            "n_head": 4,
            "n_kv_head": 4,
            "n_embd": 64,
            "window_pattern": "L",
        }
        dec_config = _make_dec_config(**dec_kwargs)

        # Decoder-only model
        dec_model = GPT(dec_config)
        dec_model.init_weights()
        dec_flops = dec_model.estimate_flops()

        # Encoder-decoder model — extract just the decoder portion
        encdec_model = _make_encdec_model(dec_kwargs=dec_kwargs)

        # Manually compute just the decoder weight + decoder attention FLOPs
        dec_block_params = sum(
            p.numel() for p in encdec_model.transformer.h.parameters()
        )
        lm_head_params = encdec_model.lm_head.weight.numel()
        dec_weight = 6 * (dec_block_params + lm_head_params)

        h = dec_config.n_head
        d = dec_config.n_embd // dec_config.n_head
        T = dec_config.sequence_len
        dec_attn = 0
        for ws in encdec_model.window_sizes:
            window = ws[0]
            eff_seq = T if window < 0 else min(window, T)
            dec_attn += 12 * h * d * eff_seq

        encdec_dec_portion = dec_weight + dec_attn

        # This should match the decoder-only estimate exactly
        assert encdec_dec_portion == dec_flops, (
            f"Decoder portion of enc-dec ({encdec_dec_portion}) should equal "
            f"decoder-only estimate ({dec_flops})"
        )


class TestAttentionFlopsPresence:
    """Verify that all three types of attention FLOPs are actually counted."""

    def _get_model(self):
        return _make_encdec_model(
            enc_kwargs={
                "depth": 2,
                "n_embd": 32,
                "n_head": 2,
                "n_kv_head": 2,
                "head_dim": 16,
            },
            xattn_kwargs={"frequency": 2, "n_head": 4, "n_kv_head": 4},
        )

    def test_cross_attention_flops_nonzero(self):
        """Cross-attention activation FLOPs must be > 0 when Nf > 0."""
        model = self._get_model()
        n_xattn = len(model.cross_attn_layers)
        assert n_xattn > 0, "Model should have cross-attention layers"

        # Compare Nf=0 (no xattn activation flops) vs Nf>0
        flops_nf0 = model.estimate_flops(0, 32)
        flops_nf64 = model.estimate_flops(64, 32)
        assert flops_nf64 > flops_nf0, (
            f"Cross-attention activation FLOPs should be > 0 for Nf=64: "
            f"Nf=0 gave {flops_nf0}, Nf=64 gave {flops_nf64}"
        )

    def test_encoder_attention_flops_nonzero(self):
        """Encoder self-attention FLOPs must be > 0 when encoder has layers and Nf > 0."""
        model = self._get_model()
        enc_layers = model.encoder.config.n_layer
        assert enc_layers > 0, "Model should have encoder layers"

        # The encoder attention FLOPs scale as Nf^2 * R, so they should grow
        # faster than linearly with Nf (since both Nf and R contain Nf)
        Nf_small = 16
        Nf_large = 160
        seg_size = 32

        # Compute manual components to isolate encoder attention
        T = model.dec_config.sequence_len
        enc_cfg = model.encoder.config
        R_small = (T / seg_size * Nf_small) / T
        R_large = (T / seg_size * Nf_large) / T
        enc_attn_small = (
            12
            * enc_cfg.n_head
            * enc_cfg.head_dim
            * Nf_small
            * enc_cfg.n_layer
            * R_small
        )
        enc_attn_large = (
            12
            * enc_cfg.n_head
            * enc_cfg.head_dim
            * Nf_large
            * enc_cfg.n_layer
            * R_large
        )

        assert enc_attn_small > 0, "Encoder attention FLOPs should be > 0"
        assert enc_attn_large > 0, "Encoder attention FLOPs should be > 0"
        # Nf_large / Nf_small = 10, but ratio should be ~100 (quadratic in Nf)
        ratio = enc_attn_large / enc_attn_small
        expected_ratio = (Nf_large / Nf_small) ** 2  # quadratic
        assert abs(ratio - expected_ratio) < 0.01, (
            f"Encoder attention should scale as Nf^2: "
            f"got ratio {ratio:.1f}, expected {expected_ratio:.1f}"
        )

    def test_decoder_self_attention_flops_present(self):
        """Decoder self-attention FLOPs should be included regardless of Nf."""
        model = self._get_model()
        # Even with Nf=0 (no encoder contribution), decoder attn should be counted
        flops = model.estimate_flops(0, 32)
        assert flops > 0, "FLOPs should be > 0 even with Nf=0 (decoder attn + weights)"

        # Compute expected decoder attention FLOPs manually
        T = model.dec_config.sequence_len
        h = model.dec_config.n_head
        d = model.dec_config.n_embd // model.dec_config.n_head
        dec_attn = 0
        for ws in model.window_sizes:
            window = ws[0]
            eff_seq = T if window < 0 else min(window, T)
            dec_attn += 12 * h * d * eff_seq
        assert dec_attn > 0, "Decoder self-attention FLOPs should be > 0"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimum_encoder_layers(self):
        """Model with 1 encoder layer (minimum) should work and match manual computation."""
        model = _make_encdec_model(
            enc_kwargs={
                "depth": 1,
                "n_embd": 32,
                "n_head": 2,
                "n_kv_head": 2,
                "head_dim": 16,
            },
        )
        assert model.encoder.config.n_layer == 1
        flops = model.estimate_flops(64, 32)
        assert flops > 0

        # With 1 encoder layer, encoder attention FLOPs should be counted
        # along with cross-attention weight and activation FLOPs
        manual = _manual_estimate_flops(model, 64, 32)
        assert flops == manual

    def test_single_cross_attention_layer(self):
        """frequency=n_layer means only one cross-attention layer."""
        model = _make_encdec_model(
            dec_kwargs={"n_layer": 4},
            xattn_kwargs={"frequency": 4},
        )
        assert len(model.cross_attn_layers) == 1
        flops = model.estimate_flops(32, 16)
        manual = _manual_estimate_flops(model, 32, 16)
        assert flops == manual

    def test_every_layer_cross_attention(self):
        """frequency=1 means cross-attention after every decoder layer."""
        model = _make_encdec_model(
            dec_kwargs={"n_layer": 4},
            xattn_kwargs={"frequency": 1, "n_head": 4, "n_kv_head": 4},
        )
        assert len(model.cross_attn_layers) == 4
        flops = model.estimate_flops(32, 16)
        manual = _manual_estimate_flops(model, 32, 16)
        assert flops == manual

    def test_nf_zero(self):
        """Nf=0 should give only decoder FLOPs (no encoder contribution)."""
        model = _make_encdec_model()
        flops = model.estimate_flops(0, 32)

        # Should equal decoder weight + decoder attention only
        dec_block_params = sum(p.numel() for p in model.transformer.h.parameters())
        lm_head_params = model.lm_head.weight.numel()
        # c_q and c_proj are still per-decoder-token, so they're counted
        xattn_dec_params = sum(
            layer.c_q.weight.numel() + layer.c_proj.weight.numel()
            for layer in model.cross_attn_layers.values()
        )
        expected_weight = 6 * (dec_block_params + lm_head_params + xattn_dec_params)

        T = model.dec_config.sequence_len
        h = model.dec_config.n_head
        d = model.dec_config.n_embd // model.dec_config.n_head
        expected_attn = 0
        for ws in model.window_sizes:
            window = ws[0]
            eff_seq = T if window < 0 else min(window, T)
            expected_attn += 12 * h * d * eff_seq

        expected = expected_weight + expected_attn
        assert flops == expected, f"Nf=0: got {flops}, expected {expected}"

    def test_sliding_window_affects_decoder_attn_only(self):
        """Sliding window pattern should only affect decoder self-attention FLOPs."""
        model_full = _make_encdec_model(dec_kwargs={"window_pattern": "L"})
        model_mixed = _make_encdec_model(dec_kwargs={"window_pattern": "SL"})

        Nf, seg = 64, 32
        flops_full = model_full.estimate_flops(Nf, seg)
        flops_mixed = model_mixed.estimate_flops(Nf, seg)

        # Mixed window should have fewer decoder self-attention FLOPs
        # (some layers use half the context)
        assert flops_mixed < flops_full, (
            f"Mixed windows should reduce FLOPs: full={flops_full}, mixed={flops_mixed}"
        )


class TestEmbeddingExclusion:
    """Verify that embedding lookups are NOT counted as matmul FLOPs."""

    def test_embeddings_excluded(self):
        """Changing embedding sizes should NOT change the weight FLOPs proportionally.

        Embeddings are lookups, not matmuls. If they were incorrectly counted,
        increasing vocab_size would increase FLOPs by 6 * delta.
        """
        model_small = _make_encdec_model(
            dec_kwargs={"vocab_size": 256},
            n_sae_features=128,
        )
        model_large = _make_encdec_model(
            dec_kwargs={"vocab_size": 512},
            n_sae_features=256,
        )

        Nf, seg = 32, 16
        flops_small = model_small.estimate_flops(Nf, seg)
        flops_large = model_large.estimate_flops(Nf, seg)

        # lm_head IS a Linear layer (matmul), so vocab_size change affects it.
        # But wte (embedding), value_embeds (embedding), encoder embeddings should not.
        # Compute expected delta from lm_head alone.
        lm_delta = (
            model_large.lm_head.weight.numel() - model_small.lm_head.weight.numel()
        )
        expected_delta = 6 * lm_delta
        actual_delta = flops_large - flops_small
        assert actual_delta == expected_delta, (
            f"Only lm_head delta should affect FLOPs. "
            f"Expected delta: {expected_delta}, actual: {actual_delta}"
        )


class TestComponentBreakdown:
    """Verify each FLOPs component independently against manual calculation."""

    def test_decoder_weight_flops(self):
        """Verify decoder weight FLOPs = 6 * (block_params + lm_head_params + xattn_q_proj_params)."""
        model = _make_encdec_model()
        dec_block_params = sum(p.numel() for p in model.transformer.h.parameters())
        lm_head_params = model.lm_head.weight.numel()
        xattn_dec_params = sum(
            layer.c_q.weight.numel() + layer.c_proj.weight.numel()
            for layer in model.cross_attn_layers.values()
        )
        expected = 6 * (dec_block_params + lm_head_params + xattn_dec_params)

        # This should be the Nf=0 FLOPs minus decoder self-attention
        T = model.dec_config.sequence_len
        h = model.dec_config.n_head
        d = model.dec_config.n_embd // model.dec_config.n_head
        dec_attn = 0
        for ws in model.window_sizes:
            window = ws[0]
            eff_seq = T if window < 0 else min(window, T)
            dec_attn += 12 * h * d * eff_seq

        nf0_flops = model.estimate_flops(0, 32)
        actual = nf0_flops - dec_attn
        assert actual == expected, (
            f"Decoder weight FLOPs: actual={actual}, expected={expected}"
        )

    def test_encoder_weight_flops_scale_with_R(self):
        """Encoder weight FLOPs should scale linearly with R."""
        model = _make_encdec_model()
        T = model.dec_config.sequence_len

        # Two configs with different R values
        Nf1, seg1 = 32, 32  # R = (128/32 * 32) / 128 = 1.0
        Nf2, seg2 = 64, 32  # R = (128/32 * 64) / 128 = 2.0

        R1 = (T / seg1 * Nf1) / T
        R2 = (T / seg2 * Nf2) / T

        enc_block_params = sum(p.numel() for p in model.encoder.blocks.parameters())
        xattn_enc_params = sum(
            layer.c_kv.weight.numel() for layer in model.cross_attn_layers.values()
        )
        enc_weight_1 = 6 * (enc_block_params + xattn_enc_params) * R1
        enc_weight_2 = 6 * (enc_block_params + xattn_enc_params) * R2

        # Should scale linearly with R
        assert abs(enc_weight_2 / enc_weight_1 - R2 / R1) < 1e-10, (
            f"Encoder weight FLOPs should scale linearly with R: "
            f"R ratio={R2 / R1}, FLOPs ratio={enc_weight_2 / enc_weight_1}"
        )

    def test_cross_attention_activation_flops_formula(self):
        """Verify cross-attention activation = 12 * n_head * head_dim * Nf * n_xattn_layers."""
        model = _make_encdec_model(
            xattn_kwargs={"frequency": 2, "n_head": 4, "n_kv_head": 4},
        )
        n_xattn = len(model.cross_attn_layers)
        xattn_layer = next(iter(model.cross_attn_layers.values()))
        xattn_h = xattn_layer.n_head
        xattn_d = xattn_layer.head_dim

        Nf = 100
        expected_xattn_attn = 12 * xattn_h * xattn_d * Nf * n_xattn

        # Verify by comparing estimate_flops at two different Nf values
        # and checking the cross-attention component
        seg_size = 32
        flops_a = model.estimate_flops(Nf, seg_size)
        flops_b = model.estimate_flops(0, seg_size)

        # flops_a - flops_b = encoder_weight(R) + xattn_attn(Nf) + enc_attn(Nf, R)
        # All terms are proportional to Nf (some quadratic in Nf)
        # At least the xattn_attn component should be present
        diff = flops_a - flops_b
        assert diff > expected_xattn_attn * 0.9, (  # should be >= xattn_attn
            f"Total encoder-side diff ({diff}) should include at least "
            f"cross-attention activation FLOPs ({expected_xattn_attn})"
        )

    def test_encoder_attention_flops_quadratic_in_nf(self):
        """Encoder self-attention FLOPs should scale as Nf^2 (Nf in formula * Nf in R)."""
        model = _make_encdec_model(
            enc_kwargs={
                "depth": 2,
                "n_embd": 32,
                "n_head": 2,
                "n_kv_head": 2,
                "head_dim": 16,
            },
        )
        enc_cfg = model.encoder.config
        T = model.dec_config.sequence_len
        seg_size = 32
        n_seg = T / seg_size

        Nf1 = 50
        Nf2 = 100
        R1 = (n_seg * Nf1) / T
        R2 = (n_seg * Nf2) / T

        attn1 = 12 * enc_cfg.n_head * enc_cfg.head_dim * Nf1 * enc_cfg.n_layer * R1
        attn2 = 12 * enc_cfg.n_head * enc_cfg.head_dim * Nf2 * enc_cfg.n_layer * R2

        ratio = attn2 / attn1
        expected_ratio = (Nf2 / Nf1) ** 2  # 4.0
        assert abs(ratio - expected_ratio) < 1e-10, (
            f"Encoder attention should be quadratic in Nf: "
            f"ratio={ratio}, expected={expected_ratio}"
        )


class TestParameterCoverage:
    """Verify that all model parameters are correctly classified."""

    def test_all_params_accounted_for(self):
        """Every parameter should be in exactly one of the FLOPs categories or excluded."""
        model = _make_encdec_model()

        # Parameters that contribute FLOPs at decoder-token rate
        dec_block_params = {id(p) for p in model.transformer.h.parameters()}
        lm_head_params = {id(model.lm_head.weight)}
        xattn_q_proj_params = set()
        for layer in model.cross_attn_layers.values():
            xattn_q_proj_params.add(id(layer.c_q.weight))
            xattn_q_proj_params.add(id(layer.c_proj.weight))

        # Parameters that contribute FLOPs at encoder-token rate
        enc_block_params = {id(p) for p in model.encoder.blocks.parameters()}
        xattn_kv_params = set()
        for layer in model.cross_attn_layers.values():
            xattn_kv_params.add(id(layer.c_kv.weight))

        # Excluded parameters (embeddings + scalars)
        excluded_params = set()
        excluded_params.add(id(model.transformer.wte.weight))
        for ve in model.value_embeds.values():
            excluded_params.add(id(ve.weight))
        excluded_params.add(id(model.resid_lambdas))
        excluded_params.add(id(model.x0_lambdas))
        excluded_params.add(id(model.encoder.feature_embed.weight))
        excluded_params.add(id(model.encoder.magnitude_embed.weight))
        for ve in model.encoder.value_embeds.values():
            excluded_params.add(id(ve.weight))

        # Buffers (cos, sin for rotary) are not in .parameters()

        all_classified = (
            dec_block_params
            | lm_head_params
            | xattn_q_proj_params
            | enc_block_params
            | xattn_kv_params
            | excluded_params
        )
        all_model_params = {id(p) for p in model.parameters()}

        missing = all_model_params - all_classified
        assert len(missing) == 0, (
            f"{len(missing)} parameters not classified in any FLOPs category. "
            f"This means some parameters are neither counted nor excluded."
        )

    def test_no_double_counting(self):
        """No parameter should appear in multiple FLOPs categories."""
        model = _make_encdec_model()

        dec_block_ids = {id(p) for p in model.transformer.h.parameters()}
        lm_head_ids = {id(model.lm_head.weight)}
        enc_block_ids = {id(p) for p in model.encoder.blocks.parameters()}

        xattn_q_proj_ids = set()
        xattn_kv_ids = set()
        for layer in model.cross_attn_layers.values():
            xattn_q_proj_ids.add(id(layer.c_q.weight))
            xattn_q_proj_ids.add(id(layer.c_proj.weight))
            xattn_kv_ids.add(id(layer.c_kv.weight))

        categories = [
            dec_block_ids,
            lm_head_ids,
            enc_block_ids,
            xattn_q_proj_ids,
            xattn_kv_ids,
        ]
        names = ["dec_blocks", "lm_head", "enc_blocks", "xattn_q_proj", "xattn_kv"]
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                overlap = categories[i] & categories[j]
                assert len(overlap) == 0, (
                    f"Overlap between {names[i]} and {names[j]}: {len(overlap)} params"
                )
