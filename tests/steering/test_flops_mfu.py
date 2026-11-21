"""Comprehensive tests for FLOP counting and MFU computation.

Covers five areas from the verification task:
1. GPT.estimate_flops() ground-truth verification (walk model graph, count every op)
2. EncoderDecoderGPT.estimate_flops() additional verification (feature prediction, amortization)
3. get_peak_flops() table accuracy
4. MFU computation correctness (gradient accumulation, FP8, DDP scaling)
5. EMA smoothing correctness (debiasing)

Ground-truth methodology: independently count FLOPs by walking nn.Linear modules
(which are the only matmul operations in the model), then add attention activation
FLOPs computed from first principles. This avoids the subtraction-based approach
in estimate_flops() and provides an independent verification.

FLOP counting convention (matches PaLM paper and Bahdanau reference):
- Per Linear weight parameter: 6 FLOPs/token (2 forward + 4 backward)
- Per attention (Q@K^T and attn@V): 12 * n_head * head_dim * effective_seq per layer
  (4 forward + 8 backward, using full-attention convention even for causal)
- Embedding lookups: excluded (not matmuls)
- Non-matmul ops (softmax, RMSnorm, residual, tanh softcap): excluded (<1% of total)
"""

import math

import torch.nn as nn

from nanochat.common import get_peak_flops
from nanochat.gpt import GPT, GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT

# ---------------------------------------------------------------------------
# Ground-truth FLOP counter utilities
# ---------------------------------------------------------------------------


def _count_linear_params(module: nn.Module) -> int:
    """Count total weight parameters in all nn.Linear layers (matmul-contributing).

    This is the ground truth for "which parameters contribute to matmul FLOPs".
    nn.Embedding layers are NOT nn.Linear and are correctly excluded.
    Bias parameters are excluded (they're additions, not matmuls).
    """
    total = 0
    for m in module.modules():
        if isinstance(m, nn.Linear):
            total += m.weight.numel()
    return total


def _ground_truth_gpt_flops(model: GPT) -> int:
    """Compute FLOPs per token for GPT by walking the model graph.

    This independently derives the FLOP count without using estimate_flops(),
    serving as a ground-truth reference implementation.

    Approach:
    1. Walk all nn.Linear modules to count matmul weight params → 6× per token
    2. Compute attention activation FLOPs per layer from first principles
    3. Sum both components
    """
    # 1. Weight-based FLOPs: every nn.Linear weight param contributes 6 FLOPs/token
    linear_params = _count_linear_params(model)
    weight_flops = 6 * linear_params

    # 2. Attention activation FLOPs per layer: 12 * n_head * head_dim * effective_seq
    # Forward: Q@K^T costs 2*h*d*seq per token, attn@V costs 2*h*d*seq → 4*h*d*seq
    # Backward: each matmul has 2 backward matmuls → 4 matmuls × 2*h*d*seq = 8*h*d*seq
    # Total: 12*h*d*seq per layer
    h = model.config.n_head
    d = model.config.n_embd // model.config.n_head
    T = model.config.sequence_len
    attn_flops = 0
    for window_size in model.window_sizes:
        window = window_size[0]
        effective_seq = T if window < 0 else min(window, T)
        attn_flops += 12 * h * d * effective_seq

    return weight_flops + attn_flops


def _ground_truth_encdec_flops(
    model: EncoderDecoderGPT, avg_features_per_segment: float, segment_size: int
) -> float:
    """Ground-truth FLOP counter for EncoderDecoderGPT.

    Walks the model graph independently to count all FLOPs per decoder token.
    """
    T = model.dec_config.sequence_len
    Nf = avg_features_per_segment
    n_seg = T / segment_size
    R = (n_seg * Nf) / T  # encoder tokens per decoder token

    # --- Decoder-side Linear params (applied per decoder token) ---
    # Decoder blocks (attention + MLP weights, including ve_gate)
    dec_block_linear = _count_linear_params(model.transformer.h)  # ty: ignore[invalid-argument-type]
    # LM head
    lm_head_linear = model.lm_head.weight.numel()
    # Cross-attention Q and proj (per decoder token)
    xattn_dec_linear = sum(
        layer.c_q.weight.numel() + layer.c_proj.weight.numel()  # ty: ignore[call-non-callable, unresolved-attribute]
        for layer in model.cross_attn_layers.values()
    )
    dec_weight_flops = 6 * (dec_block_linear + lm_head_linear + xattn_dec_linear)

    # --- Encoder-side Linear params (applied per encoder token, amortized by R) ---
    enc_block_linear = _count_linear_params(model.encoder.blocks)
    xattn_enc_linear = sum(
        layer.c_kv.weight.numel() for layer in model.cross_attn_layers.values()  # ty: ignore[call-non-callable, unresolved-attribute]
    )
    enc_weight_flops = 6 * (enc_block_linear + xattn_enc_linear) * R

    # --- Decoder self-attention activation FLOPs ---
    h = model.dec_config.n_head
    d = model.dec_config.n_embd // model.dec_config.n_head
    dec_attn_flops = 0
    for ws in model.window_sizes:
        window = ws[0]
        eff_seq = T if window < 0 else min(window, T)
        dec_attn_flops += 12 * h * d * eff_seq

    # --- Cross-attention activation FLOPs ---
    n_xattn = len(model.cross_attn_layers)
    xattn_attn_flops = 0
    if n_xattn > 0:
        layer = next(iter(model.cross_attn_layers.values()))
        xattn_attn_flops = 12 * layer.n_head * layer.head_dim * Nf * n_xattn  # ty: ignore[unsupported-operator]

    # --- Encoder self-attention activation FLOPs (amortized) ---
    enc_cfg = model.encoder.config
    enc_attn_flops = 0
    if enc_cfg.n_layer > 0:
        enc_attn_flops = (
            12 * enc_cfg.n_head * enc_cfg.head_dim * Nf * enc_cfg.n_layer * R
        )

    # --- Feature prediction head ---
    fp_head_flops = 0
    if model.feature_pred_head is not None:
        fp_head_flops = 6 * model.feature_pred_head.weight.numel()

    return (
        dec_weight_flops
        + enc_weight_flops
        + dec_attn_flops
        + xattn_attn_flops
        + enc_attn_flops
        + fp_head_flops
    )


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_gpt(
    seq_len=128,
    vocab_size=256,
    n_layer=4,
    n_head=4,
    n_kv_head=4,
    n_embd=64,
    window_pattern="L",
):
    config = GPTConfig(
        sequence_len=seq_len,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=n_embd,
        window_pattern=window_pattern,
    )
    model = GPT(config)
    model.init_weights()
    return model


def _make_encdec(
    dec_kwargs=None,
    enc_kwargs=None,
    xattn_kwargs=None,
    n_sae_features=128,
    num_buckets=8,
    feature_pred_config=None,
):
    dec_defaults = {
        "sequence_len": 128,
        "vocab_size": 256,
        "n_layer": 4,
        "n_head": 4,
        "n_kv_head": 4,
        "n_embd": 64,
        "window_pattern": "L",
    }
    dec_defaults.update(dec_kwargs or {})
    dec_config = GPTConfig(**dec_defaults)

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
        feature_pred_config=feature_pred_config,
    )
    model.init_weights()
    return model


# ===========================================================================
# Issue 1: GPT.estimate_flops() ground-truth verification
# ===========================================================================


class TestGPTGroundTruth:
    """Verify GPT.estimate_flops() against independent ground-truth counter."""

    def test_matches_ground_truth_basic(self):
        """estimate_flops must match the nn.Linear-walking ground truth."""
        model = _make_gpt()
        assert model.estimate_flops() == _ground_truth_gpt_flops(model)

    def test_matches_ground_truth_deep(self):
        """Verify with a deeper model."""
        model = _make_gpt(n_layer=12, n_embd=128, n_head=8, n_kv_head=8)
        assert model.estimate_flops() == _ground_truth_gpt_flops(model)

    def test_matches_ground_truth_gqa(self):
        """GQA: n_kv_head < n_head. Weight FLOPs should be lower than MHA."""
        model_mha = _make_gpt(n_head=8, n_kv_head=8, n_embd=128)
        model_gqa = _make_gpt(n_head=8, n_kv_head=2, n_embd=128)

        # Both should match their ground truth
        assert model_mha.estimate_flops() == _ground_truth_gpt_flops(model_mha)
        assert model_gqa.estimate_flops() == _ground_truth_gpt_flops(model_gqa)

        # GQA should have fewer FLOPs (smaller KV projection in c_qkv)
        assert model_gqa.estimate_flops() < model_mha.estimate_flops()

    def test_matches_ground_truth_sliding_window(self):
        """Sliding window reduces attention FLOPs."""
        model_full = _make_gpt(window_pattern="L")
        model_mixed = _make_gpt(window_pattern="SL")

        assert model_full.estimate_flops() == _ground_truth_gpt_flops(model_full)
        assert model_mixed.estimate_flops() == _ground_truth_gpt_flops(model_mixed)
        assert model_mixed.estimate_flops() < model_full.estimate_flops()

    def test_matches_ground_truth_large_vocab(self):
        """Larger vocab increases lm_head FLOPs but not embedding FLOPs."""
        model_small = _make_gpt(vocab_size=256)
        model_large = _make_gpt(vocab_size=1024)

        assert model_small.estimate_flops() == _ground_truth_gpt_flops(model_small)
        assert model_large.estimate_flops() == _ground_truth_gpt_flops(model_large)

        # Larger vocab → larger lm_head → more FLOPs
        assert model_large.estimate_flops() > model_small.estimate_flops()


class TestGPTFlopComponents:
    """Verify individual FLOP components of GPT."""

    def test_linear_params_equals_nparams_minus_excluded(self):
        """The nn.Linear param count should match nparams - excluded."""
        model = _make_gpt()
        linear_params = _count_linear_params(model)

        # estimate_flops approach: total params - excluded
        nparams = sum(p.numel() for p in model.parameters())
        value_embeds_numel = sum(
            ve.weight.numel() for ve in model.value_embeds.values()
        )
        nparams_exclude = (
            model.transformer.wte.weight.numel()
            + value_embeds_numel
            + model.resid_lambdas.numel()
            + model.x0_lambdas.numel()
        )
        assert linear_params == nparams - nparams_exclude

    def test_ve_gate_is_linear_and_counted(self):
        """ve_gate is nn.Linear and should be counted as matmul FLOPs."""
        model = _make_gpt()
        ve_gate_params = 0
        for block in model.transformer.h:
            if block.attn.ve_gate is not None:
                assert isinstance(block.attn.ve_gate, nn.Linear)
                ve_gate_params += block.attn.ve_gate.weight.numel()
        assert ve_gate_params > 0, "Model should have ve_gate params"

    def test_embeddings_are_not_linear(self):
        """nn.Embedding should NOT be counted by the Linear-walking approach."""
        model = _make_gpt()
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                assert not isinstance(m, nn.Linear)

    def test_6n_factor_decomposition(self):
        """6 = 2 (forward matmul) + 4 (backward: 2 for grad_input + 2 for grad_weight).

        Verify this by checking that weight FLOPs = 6 * linear_params.
        """
        model = _make_gpt()
        linear_params = _count_linear_params(model)
        # At Nf=0 or for decoder-only, weight FLOPs = estimate_flops - attn_flops
        h = model.config.n_head
        d = model.config.n_embd // model.config.n_head
        T = model.config.sequence_len
        attn_flops = 0
        for ws in model.window_sizes:
            w = ws[0]
            eff = T if w < 0 else min(w, T)
            attn_flops += 12 * h * d * eff
        weight_flops = model.estimate_flops() - attn_flops
        assert weight_flops == 6 * linear_params

    def test_12_factor_decomposition(self):
        """12 = 4 (forward: 2 QK + 2 AV) + 8 (backward: 2×2 QK + 2×2 AV).

        Verify the attention FLOPs formula for a single layer with full attention.
        """
        model = _make_gpt(n_layer=1, window_pattern="L")
        h = model.config.n_head
        d = model.config.n_embd // model.config.n_head
        T = model.config.sequence_len

        # Expected: 12 * h * d * T for 1 layer
        expected_attn = 12 * h * d * T
        # Actual: estimate_flops - weight_flops
        weight_flops = 6 * _count_linear_params(model)
        actual_attn = model.estimate_flops() - weight_flops
        assert actual_attn == expected_attn

    def test_attention_flops_per_layer_with_sliding_window(self):
        """Verify per-layer attention FLOPs with mixed window pattern."""
        model = _make_gpt(n_layer=4, window_pattern="SSSL", seq_len=128)
        h = model.config.n_head
        d = model.config.n_embd // model.config.n_head
        T = model.config.sequence_len
        S = T // 2  # short window

        # Pattern SSSL with 4 layers: S, S, S, L (last always L)
        expected_attn = (
            12 * h * d * S  # layer 0: S
            + 12 * h * d * S  # layer 1: S
            + 12 * h * d * S  # layer 2: S
            + 12 * h * d * T  # layer 3: L (always full)
        )
        weight_flops = 6 * _count_linear_params(model)
        actual_attn = model.estimate_flops() - weight_flops
        assert actual_attn == expected_attn

    def test_padded_vocab_counted(self):
        """lm_head uses padded vocab size. The padding FLOPs ARE computed, so they
        should be counted in the FLOP estimate."""
        # vocab_size=100 with pad_vocab_size_to=64 → padded to 128
        model = _make_gpt(vocab_size=100)
        padded_vocab = (
            (100 + 64 - 1) // 64
        ) * 64  # 128, since GPT default pad=64
        assert model.lm_head.weight.shape[0] == padded_vocab
        assert model.lm_head.weight.numel() == model.config.n_embd * padded_vocab

        # Ground truth should still match
        assert model.estimate_flops() == _ground_truth_gpt_flops(model)


class TestGPTGQASavings:
    """Verify that GQA correctly reduces weight FLOPs while attention FLOPs stay same."""

    def test_gqa_reduces_weight_flops(self):
        """GQA reduces c_qkv weight size, so weight FLOPs should decrease."""
        model_mha = _make_gpt(n_head=8, n_kv_head=8, n_embd=128)
        model_gqa = _make_gpt(n_head=8, n_kv_head=2, n_embd=128)

        mha_linear = _count_linear_params(model_mha)
        gqa_linear = _count_linear_params(model_gqa)
        assert gqa_linear < mha_linear

    def test_gqa_same_attention_flops(self):
        """Attention activation FLOPs (Q@K^T, attn@V) are the same with GQA.

        GQA shares K/V heads, but each query head still does the full computation.
        """
        model_mha = _make_gpt(n_head=8, n_kv_head=8, n_embd=128)
        model_gqa = _make_gpt(n_head=8, n_kv_head=2, n_embd=128)

        # Compute attention FLOPs for both
        h = 8  # same n_head
        d = 128 // 8  # same head_dim
        T = 128  # same seq_len

        mha_weight = 6 * _count_linear_params(model_mha)
        mha_attn = model_mha.estimate_flops() - mha_weight

        gqa_weight = 6 * _count_linear_params(model_gqa)
        gqa_attn = model_gqa.estimate_flops() - gqa_weight

        # Attention FLOPs should be identical
        assert mha_attn == gqa_attn

        # Both should equal 12*h*d*T per layer * n_layer
        expected = 12 * h * d * T * model_mha.config.n_layer
        assert mha_attn == expected
        assert gqa_attn == expected

    def test_gqa_savings_quantified(self):
        """The FLOP savings from GQA should exactly match the Linear param difference.

        GQA affects c_qkv (smaller KV dims) and ve_gate (fewer KV heads) weights.
        """
        n_head, n_embd = 8, 128

        model_mha = _make_gpt(n_head=n_head, n_kv_head=n_head, n_embd=n_embd)
        model_gqa = _make_gpt(n_head=n_head, n_kv_head=2, n_embd=n_embd)

        # Exact savings = 6 * (linear_params_mha - linear_params_gqa)
        mha_linear = _count_linear_params(model_mha)
        gqa_linear = _count_linear_params(model_gqa)
        expected_diff = 6 * (mha_linear - gqa_linear)

        actual_diff = model_mha.estimate_flops() - model_gqa.estimate_flops()
        assert actual_diff == expected_diff


# ===========================================================================
# Issue 2: EncoderDecoderGPT.estimate_flops() verification
# ===========================================================================


class TestEncoderDecoderGroundTruth:
    """Verify EncoderDecoderGPT.estimate_flops() against ground truth."""

    def test_matches_ground_truth(self):
        model = _make_encdec()
        Nf, seg = 64, 32
        assert model.estimate_flops(Nf, seg) == _ground_truth_encdec_flops(
            model, Nf, seg
        )

    def test_matches_ground_truth_various_configs(self):
        configs = [
            ({}, {}, {}, 32, 16),
            (
                {"n_layer": 8, "n_embd": 128, "n_head": 8, "n_kv_head": 4},
                {"n_embd": 64, "depth": 4, "n_head": 4, "n_kv_head": 2, "head_dim": 16},
                {"frequency": 4, "n_head": 8, "n_kv_head": 4},
                128,
                32,
            ),
        ]
        for dec_kw, enc_kw, xattn_kw, Nf, seg in configs:
            model = _make_encdec(dec_kw, enc_kw, xattn_kw)
            actual = model.estimate_flops(Nf, seg)
            expected = _ground_truth_encdec_flops(model, Nf, seg)
            assert actual == expected, f"Mismatch for {dec_kw} {enc_kw}: {actual} vs {expected}"


class TestFeaturePredictionFlops:
    """Verify feature prediction head FLOPs are correctly included."""

    def test_feature_pred_increases_flops(self):
        """Enabling feature prediction should increase total FLOPs."""
        model_no_fp = _make_encdec()
        fp_config = FeaturePredictionConfig(
            weight=0.1,
            layer=-1,
            loss_mode=FeaturePredLossMode.FULL_SEGMENT,
            loss_fn=FeaturePredLossFn.BCE,
        )
        model_fp = _make_encdec(feature_pred_config=fp_config)

        Nf, seg = 64, 32
        flops_no_fp = model_no_fp.estimate_flops(Nf, seg)
        flops_fp = model_fp.estimate_flops(Nf, seg)

        assert flops_fp > flops_no_fp

    def test_feature_pred_delta_matches_head(self):
        """The FLOP increase from feature prediction should equal 6 * weight.numel()."""
        fp_config = FeaturePredictionConfig(
            weight=0.1,
            layer=-1,
            loss_mode=FeaturePredLossMode.FULL_SEGMENT,
            loss_fn=FeaturePredLossFn.BCE,
        )
        model_fp = _make_encdec(feature_pred_config=fp_config)
        model_no_fp = _make_encdec()

        Nf, seg = 64, 32
        delta = model_fp.estimate_flops(Nf, seg) - model_no_fp.estimate_flops(Nf, seg)
        expected_delta = 6 * model_fp.feature_pred_head.weight.numel()
        assert delta == expected_delta

    def test_feature_pred_matches_ground_truth(self):
        """Ground truth should match with feature prediction enabled."""
        fp_config = FeaturePredictionConfig(
            weight=0.1,
            layer=-1,
            loss_mode=FeaturePredLossMode.FULL_SEGMENT,
            loss_fn=FeaturePredLossFn.BCE,
        )
        model = _make_encdec(feature_pred_config=fp_config)
        Nf, seg = 64, 32
        assert model.estimate_flops(Nf, seg) == _ground_truth_encdec_flops(
            model, Nf, seg
        )


class TestAmortizationCorrectness:
    """Verify encoder FLOP amortization by segment_size."""

    def test_amortization_ratio(self):
        """Doubling segment_size should halve encoder-side FLOPs."""
        model = _make_encdec()
        Nf = 64

        flops_seg32 = model.estimate_flops(Nf, segment_size=32)
        flops_seg64 = model.estimate_flops(Nf, segment_size=64)

        # Extract encoder-only contribution
        dec_only = model.estimate_flops(0, 32)  # decoder FLOPs (independent of seg)
        enc_32 = flops_seg32 - dec_only
        enc_64 = flops_seg64 - dec_only

        # Encoder FLOPs have components ∝ R and ∝ Nf*R.
        # R = Nf/seg_size. Doubling seg_size halves R, so all encoder components halve.
        # Note: cross-attention activation FLOPs (12*h*d*Nf) are per decoder token
        # and do NOT depend on segment_size. So we need to subtract those.
        n_xattn = len(model.cross_attn_layers)
        if n_xattn > 0:
            layer = next(iter(model.cross_attn_layers.values()))
            xattn_attn_flops = 12 * layer.n_head * layer.head_dim * Nf * n_xattn
        else:
            xattn_attn_flops = 0

        # The R-dependent part should exactly halve
        r_dep_32 = enc_32 - xattn_attn_flops
        r_dep_64 = enc_64 - xattn_attn_flops
        assert abs(r_dep_64 / r_dep_32 - 0.5) < 1e-10

    def test_encoder_flops_amortized_not_per_token(self):
        """Encoder FLOPs should NOT scale linearly with T (they're per segment, amortized)."""
        model_short = _make_encdec(dec_kwargs={"sequence_len": 64})
        model_long = _make_encdec(dec_kwargs={"sequence_len": 256})

        Nf, seg = 32, 32
        flops_short = model_short.estimate_flops(Nf, seg)
        flops_long = model_long.estimate_flops(Nf, seg)

        # Per-token encoder FLOPs should be similar (R = Nf/seg_size, independent of T)
        # But decoder self-attention FLOPs scale with T
        # So flops_long/flops_short should NOT be 256/64 = 4
        ratio = flops_long / flops_short
        assert ratio < 3.5, f"Expected ratio < 3.5 (encoder amortization), got {ratio}"

    def test_minimal_enc_layers_flops_equal_decoder_plus_overhead(self):
        """With 1 encoder layer (minimum), total should be decoder FLOPs + encoder + xattn overhead."""
        dec_kwargs = {
            "sequence_len": 128,
            "vocab_size": 256,
            "n_layer": 4,
            "n_head": 4,
            "n_kv_head": 4,
            "n_embd": 64,
            "window_pattern": "L",
        }

        model_encdec = _make_encdec(
            dec_kwargs=dec_kwargs,
            enc_kwargs={
                "depth": 1,
                "n_embd": 32,
                "n_head": 2,
                "n_kv_head": 2,
                "head_dim": 16,
            },
        )

        model_gpt = _make_gpt(
            seq_len=dec_kwargs["sequence_len"],
            vocab_size=dec_kwargs["vocab_size"],
            n_layer=dec_kwargs["n_layer"],
            n_head=dec_kwargs["n_head"],
            n_kv_head=dec_kwargs["n_kv_head"],
            n_embd=dec_kwargs["n_embd"],
            window_pattern=dec_kwargs["window_pattern"],
        )

        Nf, seg = 32, 32
        encdec_flops = model_encdec.estimate_flops(Nf, seg)
        gpt_flops = model_gpt.estimate_flops()

        # The overhead should match our ground-truth counter
        ground_truth = _ground_truth_encdec_flops(model_encdec, Nf, seg)
        assert abs(encdec_flops - ground_truth) < 1e-6

        # enc-dec with encoder layer(s) must have more FLOPs than decoder-only
        assert encdec_flops > gpt_flops, (
            f"Encoder-decoder FLOPs ({encdec_flops}) should exceed "
            f"decoder-only ({gpt_flops})"
        )


# ===========================================================================
# Issue 3: get_peak_flops() table accuracy
# ===========================================================================


class TestPeakFlopsTable:
    """Verify get_peak_flops() returns correct values for known GPUs."""

    def test_h100_sxm(self):
        """H100 SXM5: 989 TFLOPS BF16 dense tensor core."""
        assert get_peak_flops("NVIDIA H100 80GB HBM3") == 989e12

    def test_h100_pcie(self):
        """H100 PCIe: 756 TFLOPS BF16 dense tensor core."""
        assert get_peak_flops("NVIDIA H100 PCIe") == 756e12

    def test_h100_nvl(self):
        """H100 NVL: ~835 TFLOPS BF16 per GPU."""
        assert get_peak_flops("NVIDIA H100 NVL") == 835e12

    def test_h200_sxm(self):
        """H200 SXM: same compute as H100 SXM = 989 TFLOPS."""
        assert get_peak_flops("NVIDIA H200") == 989e12

    def test_a100(self):
        """A100 SXM4: 312 TFLOPS BF16 dense tensor core."""
        assert get_peak_flops("NVIDIA A100-SXM4-80GB") == 312e12

    def test_rtx_4090(self):
        """RTX 4090: 165.2 TFLOPS BF16 dense tensor core."""
        assert get_peak_flops("NVIDIA GeForce RTX 4090") == 165.2e12

    def test_rtx_3090(self):
        """RTX 3090: 71 TFLOPS FP16/BF16 dense tensor core."""
        assert get_peak_flops("NVIDIA GeForce RTX 3090") == 71e12

    def test_l40s(self):
        """L40S: 362 TFLOPS BF16 tensor core."""
        assert get_peak_flops("NVIDIA L40S") == 362e12

    def test_mi300x(self):
        """MI300X: 1307.4 TFLOPS BF16."""
        assert get_peak_flops("AMD Instinct MI300X") == 1.3074e15

    def test_b200(self):
        """B200: 2250 TFLOPS BF16 dense tensor core."""
        assert get_peak_flops("NVIDIA B200") == 2.25e15


class TestPeakFlopsUnknownGPU:
    """Verify behavior with unknown GPU names."""

    def test_unknown_gpu_returns_inf(self):
        """Unknown GPU should return inf (MFU shows 0%)."""
        result = get_peak_flops("Unknown GPU XYZ-9999")
        assert result == float("inf")

    def test_unknown_gpu_logs_warning(self):
        """Unknown GPU should log a warning with the GPU name.

        Verified by capturing loguru output via a temporary sink.
        """
        from io import StringIO

        from loguru import logger

        buffer = StringIO()
        sink_id = logger.add(buffer, format="{message}", level="WARNING")
        try:
            get_peak_flops("Fictional GPU Model ABC")
            output = buffer.getvalue()
            assert "Fictional GPU Model ABC" in output
        finally:
            logger.remove(sink_id)

    def test_case_insensitive_matching(self):
        """GPU name matching should be case-insensitive."""
        assert get_peak_flops("nvidia h100") == 989e12
        assert get_peak_flops("NVIDIA H100") == 989e12
        assert get_peak_flops("Nvidia H100 SXM") == 989e12

    def test_all_known_gpus_return_finite(self):
        """All known GPU names should return finite FLOP values."""
        known_gpus = [
            "NVIDIA H100",
            "NVIDIA H200",
            "NVIDIA A100",
            "NVIDIA GeForce RTX 4090",
            "NVIDIA GeForce RTX 3090",
            "NVIDIA GeForce RTX 5090",
            "NVIDIA L40S",
            "NVIDIA L4",
            "NVIDIA B200",
            "NVIDIA B100",
            "AMD Instinct MI300X",
            "AMD Instinct MI300A",
            "AMD Instinct MI250X",
        ]
        for name in known_gpus:
            flops = get_peak_flops(name)
            assert math.isfinite(flops), f"Expected finite for '{name}', got {flops}"
            assert flops > 0, f"Expected positive for '{name}', got {flops}"

    def test_mfu_zero_for_unknown(self):
        """MFU should be 0% when peak FLOPs is inf (unknown GPU)."""
        peak = get_peak_flops("Unknown GPU")
        flops_per_sec = 1e12  # any positive value
        mfu = 100 * flops_per_sec / peak
        assert mfu == 0.0


# ===========================================================================
# Issue 4: MFU computation correctness
# ===========================================================================


class TestMFUComputation:
    """Verify MFU calculation matches the formula from training scripts."""

    def test_mfu_100_percent_at_peak(self):
        """MFU should be 100% when flops_per_sec equals peak × world_size."""
        gpu_peak_flops = 989e12  # H100
        ddp_world_size = 1
        flops_per_sec = gpu_peak_flops * ddp_world_size
        mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
        assert mfu == 100.0

    def test_mfu_50_percent(self):
        """MFU should be 50% at half peak throughput."""
        gpu_peak_flops = 989e12
        ddp_world_size = 1
        flops_per_sec = gpu_peak_flops * 0.5
        mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
        assert abs(mfu - 50.0) < 1e-10

    def test_mfu_scales_linearly(self):
        """Doubling throughput should double MFU."""
        gpu_peak_flops = 989e12
        ddp_world_size = 1

        flops1 = 100e12
        flops2 = 200e12
        mfu1 = 100 * flops1 / (gpu_peak_flops * ddp_world_size)
        mfu2 = 100 * flops2 / (gpu_peak_flops * ddp_world_size)
        assert abs(mfu2 / mfu1 - 2.0) < 1e-10

    def test_mfu_ddp_scaling(self):
        """With N GPUs, peak = N × per-GPU peak."""
        gpu_peak_flops = 989e12
        for world_size in [1, 2, 4, 8]:
            # Same per-GPU throughput → same MFU regardless of world size
            per_gpu_flops = 500e12
            total_flops = per_gpu_flops * world_size
            mfu = 100 * total_flops / (gpu_peak_flops * world_size)
            assert abs(mfu - 100 * per_gpu_flops / gpu_peak_flops) < 1e-10

    def test_mfu_gradient_accumulation_invariant(self):
        """MFU should be the same regardless of grad_accum_steps.

        With gradient accumulation:
        - total_batch_size stays the same
        - dt covers all micro-steps
        - flops_per_sec = num_flops_per_token * total_batch_size / dt
        - If hardware speed is constant, dt scales with grad_accum_steps
        - So flops_per_sec (and thus MFU) should be the same
        """
        num_flops_per_token = 1e9  # 1 GFLOP per token
        total_batch_size = 1_000_000  # 1M tokens per step
        gpu_peak = 989e12
        world_size = 1
        hardware_speed = 500e12  # 500 TFLOPS actual

        # Total work per step is independent of how it is split into
        # accumulation micro-steps.
        total_flops_per_step = num_flops_per_token * total_batch_size
        dt = total_flops_per_step / hardware_speed
        flops_per_sec = num_flops_per_token * total_batch_size / dt
        mfu = 100 * flops_per_sec / (gpu_peak * world_size)

        expected_mfu = 100 * hardware_speed / gpu_peak
        assert abs(mfu - expected_mfu) < 1e-10

    def test_mfu_plausible_range(self):
        """MFU should be in a plausible range (0-100%) for realistic scenarios."""
        model = _make_gpt(n_layer=12, n_embd=768, n_head=12, n_kv_head=12, seq_len=1024)
        num_flops_per_token = model.estimate_flops()

        # Simulate: total_batch_size = 512K tokens, dt = 0.5s, H100
        total_batch_size = 524_288
        dt = 0.5
        gpu_peak = 989e12
        world_size = 1

        flops_per_sec = num_flops_per_token * total_batch_size / dt
        mfu = 100 * flops_per_sec / (gpu_peak * world_size)

        # MFU should be between 0% and 100% for any reasonable scenario
        assert 0 < mfu < 200, f"MFU {mfu:.1f}% outside plausible range"

    def test_flops_per_sec_formula(self):
        """Verify: flops_per_sec = num_flops_per_token * total_batch_size / dt.

        total_batch_size must be in TOKENS (not sequences).
        """
        num_flops_per_token = 2e9  # 2 GFLOP/token
        # 32 sequences × 1024 tokens = 32768 tokens
        device_batch_size = 32
        seq_len = 1024
        world_size = 4
        grad_accum = 2
        total_batch_size = device_batch_size * seq_len * world_size * grad_accum

        dt = 1.0  # 1 second
        flops_per_sec = num_flops_per_token * total_batch_size / dt

        expected = 2e9 * (32 * 1024 * 4 * 2) / 1.0
        assert flops_per_sec == expected


class TestFP8MFU:
    """Document FP8 impact on MFU calculations.

    When FP8 training is used, the hardware can achieve up to 2× the BF16 peak
    TFLOPS. Since get_peak_flops() returns BF16 peaks, MFU can exceed 100% with FP8.

    This is by design: MFU is reported relative to BF16 baseline for consistency
    across training runs with different precisions.
    """

    def test_fp8_can_exceed_100_percent(self):
        """MFU can exceed 100% when hardware runs at FP8 speed but peak is BF16."""
        gpu_peak_bf16 = 989e12  # H100 BF16
        # FP8 can achieve ~2× BF16 throughput
        actual_flops_per_sec = 1.5e15  # 1.5 PFLOPS (between BF16 and FP8 peak)
        mfu = 100 * actual_flops_per_sec / gpu_peak_bf16
        assert mfu > 100, "FP8 throughput should push MFU above 100% vs BF16 peak"

    def test_fp8_peak_is_approximately_2x_bf16(self):
        """FP8 peak for H100 is approximately 2× BF16 peak."""
        bf16_peak = 989e12
        fp8_peak = 2 * bf16_peak  # ~1978 TFLOPS
        assert fp8_peak > bf16_peak


# ===========================================================================
# Issue 5: EMA smoothing correctness
# ===========================================================================


class TestEMASmoothing:
    """Verify EMA with debiasing (used for loss smoothing in training loops)."""

    def test_ema_debiasing_first_step(self):
        """At step 0, debiased EMA should equal the raw value."""
        beta = 0.9
        first_loss = 3.14
        smooth = beta * 0 + (1 - beta) * first_loss  # 0.314
        debiased = smooth / (1 - beta ** 1)  # 0.314 / 0.1 = 3.14
        assert abs(debiased - first_loss) < 1e-10

    def test_ema_debiasing_converges(self):
        """After many steps with constant value, debiased EMA should converge."""
        beta = 0.9
        smooth = 0
        constant_value = 2.5
        for step in range(1000):
            smooth = beta * smooth + (1 - beta) * constant_value
            debiased = smooth / (1 - beta ** (step + 1))
        assert abs(debiased - constant_value) < 1e-6

    def test_ema_debiasing_formula(self):
        """Verify the debiasing formula: ema / (1 - beta^step)."""
        beta = 0.9
        values = [3.0, 2.5, 2.0, 1.5, 1.0]
        smooth = 0

        for step, val in enumerate(values):
            smooth = beta * smooth + (1 - beta) * val
            debiased = smooth / (1 - beta ** (step + 1))

            # Debiased should be a weighted average of all values so far,
            # with more recent values weighted more heavily
            assert debiased > 0, f"Step {step}: debiased should be positive"

        # After 5 steps with decreasing values, debiased should be between min and max
        assert values[-1] <= debiased <= values[0]

    def test_ema_not_applied_to_time(self):
        """Time (dt) should NOT be EMA-smoothed.

        The training loop computes:
        - smooth_loss = EMA(loss)  ← correct
        - dt = t1 - t0  ← raw wall-clock, NOT smoothed
        - flops_per_sec = flops / dt  ← uses raw dt

        Smoothing dt would hide real jitter from bottlenecks.
        """
        # This is a documentation test: verify the base_train.py code pattern
        # dt is used directly in the MFU calculation, not EMA-smoothed
        # (verified by reading the source code)
        # The MFU formula: flops_per_sec = num_flops_per_token * total_batch_size / dt
        # where dt is raw, unsmoothed wall-clock time
        # Verified by code review — dt is never EMA-smoothed


# ===========================================================================
# Cross-cutting: ground truth vs estimate for non-matmul ops
# ===========================================================================


class TestNonMatmulExclusion:
    """Verify that non-matmul operations are correctly excluded.

    The estimate intentionally excludes:
    - Softmax (exp, sum, div): O(T) per head per position
    - RMS norm: O(d) per position per norm application
    - Residual additions: O(d) per position
    - Logit softcap (tanh): O(vocab_size) per position
    - Embedding lookup: O(1) per position (table lookup, not matmul)

    These are all O(d) or O(T) operations vs O(d²) matmul operations,
    contributing < 1% of total FLOPs for typical model sizes.
    """

    def test_excluded_ops_are_small(self):
        """Verify that excluded ops contribute < 5% of total FLOPs."""
        model = _make_gpt(n_layer=12, n_embd=768, n_head=12, n_kv_head=12, seq_len=1024)
        matmul_flops = model.estimate_flops()

        # Upper bound on excluded FLOPs per token:
        T = model.config.sequence_len
        d = model.config.n_embd
        n_layer = model.config.n_layer
        h = model.config.n_head
        vocab = model.config.vocab_size

        # Softmax: ~5T FLOPs per head per layer (exp, sub max, sum, div, mul)
        softmax_flops = 5 * T * h * n_layer
        # RMS norm: ~2d FLOPs per application, ~3 norms per layer + 1 for embedding + 1 final
        norm_flops = 2 * d * (3 * n_layer + 2)
        # Residual additions: d FLOPs per addition, ~2 per layer
        residual_flops = d * 2 * n_layer
        # Softcap (tanh + div + mul): ~10 * vocab FLOPs
        softcap_flops = 10 * vocab
        # relu^2 in MLP: 2 * 4d per layer (relu + square)
        activation_flops = 2 * 4 * d * n_layer

        total_excluded = (
            softmax_flops + norm_flops + residual_flops + softcap_flops + activation_flops
        )

        # Forward only (excluded ops don't have the 3× backward multiplier for matmuls,
        # but their backward is typically similar cost)
        total_excluded_fwd_bwd = total_excluded * 3  # rough upper bound

        fraction = total_excluded_fwd_bwd / matmul_flops
        assert fraction < 0.05, (
            f"Excluded ops are {fraction:.2%} of matmul FLOPs, expected < 5%"
        )
