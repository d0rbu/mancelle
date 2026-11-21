from typing import cast

import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.config import (
    resolve_encoder_depth as _resolve_encoder_depth,
)
from steering.conditional.encoder_decoder.config import (
    resolve_encoder_dim as _resolve_encoder_dim,
)
from steering.conditional.encoder_decoder.config import (
    resolve_encoder_heads as _resolve_encoder_heads,
)
from steering.refined_types import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)


def resolve_encoder_depth(depth: int | float, decoder_depth: int) -> NonNegativeInt:
    """Pass raw test values through the production beartype boundary."""
    return _resolve_encoder_depth(
        cast(NonNegativeInt | NonNegativeFloat, depth),
        cast(PositiveInt, decoder_depth),
    )


def resolve_encoder_dim(n_embd: int | float, decoder_dim: int) -> PositiveInt:
    """Pass raw test values through the production beartype boundary."""
    return _resolve_encoder_dim(
        cast(PositiveInt | PositiveFloat, n_embd),
        cast(PositiveInt, decoder_dim),
    )


def resolve_encoder_heads(
    n_head: int | None,
    n_kv_head: int | None,
    enc_dim: int,
    head_dim: int,
) -> tuple[PositiveInt, PositiveInt]:
    """Pass raw test values through the production beartype boundary."""
    return _resolve_encoder_heads(
        cast(PositiveInt | None, n_head),
        cast(PositiveInt | None, n_kv_head),
        cast(PositiveInt, enc_dim),
        cast(PositiveInt, head_dim),
    )

# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------


class TestEncoderConfigDefaults:
    def test_default_n_embd(self):
        cfg = EncoderConfig()
        assert cfg.n_embd == 0.5

    def test_default_depth(self):
        cfg = EncoderConfig()
        assert cfg.depth == 0.5

    def test_default_n_head(self):
        cfg = EncoderConfig()
        assert cfg.n_head is None

    def test_default_n_kv_head(self):
        cfg = EncoderConfig()
        assert cfg.n_kv_head is None

    def test_default_head_dim(self):
        cfg = EncoderConfig()
        assert cfg.head_dim == 64


class TestCrossAttentionConfigDefaults:
    def test_default_frequency(self):
        cfg = CrossAttentionConfig()
        assert cfg.frequency == 4

    def test_default_n_head(self):
        cfg = CrossAttentionConfig()
        assert cfg.n_head is None

    def test_default_n_kv_head(self):
        cfg = CrossAttentionConfig()
        assert cfg.n_kv_head is None


# ---------------------------------------------------------------------------
# resolve_encoder_depth
# ---------------------------------------------------------------------------


class TestResolveEncoderDepth:
    # --- int (absolute) ---

    def test_int_absolute(self):
        assert resolve_encoder_depth(3, decoder_depth=12) == 3

    def test_int_one_is_absolute(self):
        """1 (int) means exactly 1 layer, NOT a ratio."""
        assert resolve_encoder_depth(1, decoder_depth=12) == 1

    def test_int_zero_allowed(self):
        """0 depth is valid: embed + project only, no self-attention."""
        assert resolve_encoder_depth(0, decoder_depth=12) == 0

    def test_int_max_boundary(self):
        """Exactly 2x decoder_depth is allowed."""
        assert resolve_encoder_depth(24, decoder_depth=12) == 24

    def test_int_exceeds_2x_decoder_raises(self):
        with pytest.raises(ValueError):
            resolve_encoder_depth(25, decoder_depth=12)

    def test_int_negative_raises(self):
        with pytest.raises(BeartypeCallHintParamViolation, match="depth"):
            resolve_encoder_depth(-1, decoder_depth=12)

    # --- float (ratio) ---

    def test_float_ratio(self):
        """0.5 * 12 = 6."""
        assert resolve_encoder_depth(0.5, decoder_depth=12) == 6

    def test_float_one_is_ratio(self):
        """1.0 (float) means ratio: 1.0 * 12 = 12."""
        assert resolve_encoder_depth(1.0, decoder_depth=12) == 12

    def test_float_rounding(self):
        """0.3 * 12 = 3.6, rounds to 4."""
        assert resolve_encoder_depth(0.3, decoder_depth=12) == 4

    def test_float_rounding_down(self):
        """0.1 * 12 = 1.2, rounds to 1."""
        assert resolve_encoder_depth(0.1, decoder_depth=12) == 1

    def test_float_zero_ratio(self):
        """0.0 * 12 = 0."""
        assert resolve_encoder_depth(0.0, decoder_depth=12) == 0

    def test_float_max_boundary(self):
        """2.0 ratio is allowed: 2.0 * 12 = 24."""
        assert resolve_encoder_depth(2.0, decoder_depth=12) == 24

    def test_float_exceeds_2_raises(self):
        with pytest.raises(ValueError):
            resolve_encoder_depth(2.1, decoder_depth=12)

    def test_float_negative_raises(self):
        with pytest.raises(BeartypeCallHintParamViolation, match="depth"):
            resolve_encoder_depth(-0.5, decoder_depth=12)


# ---------------------------------------------------------------------------
# resolve_encoder_dim
# ---------------------------------------------------------------------------


class TestResolveEncoderDim:
    # --- int (absolute) ---

    def test_int_absolute(self):
        assert resolve_encoder_dim(256, decoder_dim=768) == 256

    def test_int_positive_required(self):
        with pytest.raises(BeartypeCallHintParamViolation, match="n_embd"):
            resolve_encoder_dim(0, decoder_dim=768)

    def test_int_negative_raises(self):
        with pytest.raises(BeartypeCallHintParamViolation, match="n_embd"):
            resolve_encoder_dim(-128, decoder_dim=768)

    # --- float (ratio) ---

    def test_float_ratio(self):
        """0.5 * 768 = 384, nearest 128 = 384 (already a multiple)."""
        assert resolve_encoder_dim(0.5, decoder_dim=768) == 384

    def test_float_ratio_rounding_to_128(self):
        """0.3 * 768 = 230.4, nearest 128 = 256."""
        assert resolve_encoder_dim(0.3, decoder_dim=768) == 256

    def test_float_ratio_rounding_down_to_128(self):
        """0.15 * 1024 = 153.6, nearest 128 = 128."""
        assert resolve_encoder_dim(0.15, decoder_dim=1024) == 128

    def test_float_minimum_128(self):
        """Even a tiny ratio yields at least 128."""
        assert resolve_encoder_dim(0.01, decoder_dim=768) == 128

    def test_float_one_is_ratio(self):
        """1.0 (float) means ratio: 1.0 * 768 = 768, nearest 128 = 768."""
        assert resolve_encoder_dim(1.0, decoder_dim=768) == 768

    def test_float_negative_raises(self):
        with pytest.raises(BeartypeCallHintParamViolation, match="n_embd"):
            resolve_encoder_dim(-0.5, decoder_dim=768)

    def test_float_zero_raises(self):
        """0.0 ratio results in 0 raw dim, which is non-positive."""
        with pytest.raises(BeartypeCallHintParamViolation, match="n_embd"):
            resolve_encoder_dim(0.0, decoder_dim=768)

    def test_float_large_ratio(self):
        """2.0 * 768 = 1536, nearest 128 = 1536."""
        assert resolve_encoder_dim(2.0, decoder_dim=768) == 1536


# ---------------------------------------------------------------------------
# resolve_encoder_dim — additional edge cases
# ---------------------------------------------------------------------------

class TestResolveEncoderDimEdgeCases:
    """Edge cases for rounding, minimum clamping, and non-standard decoder dims."""

    def test_small_decoder_dim_ratio_clamps_to_min(self):
        """decoder_dim=128, ratio=0.5 -> raw=64, but min 128 -> 128."""
        assert resolve_encoder_dim(0.5, decoder_dim=128) == 128

    def test_exact_multiple_of_128(self):
        """decoder_dim=1024, ratio=0.5 -> 512 (already a multiple of 128)."""
        assert resolve_encoder_dim(0.5, decoder_dim=1024) == 512

    def test_rounding_down_near_128_boundary(self):
        """0.4 * 768 = 307.2 -> round(307.2/128)*128 = round(2.4)*128 = 2*128 = 256."""
        assert resolve_encoder_dim(0.4, decoder_dim=768) == 256

    def test_rounding_up_near_128_boundary(self):
        """0.45 * 768 = 345.6 -> round(345.6/128)*128 = round(2.7)*128 = 3*128 = 384."""
        assert resolve_encoder_dim(0.45, decoder_dim=768) == 384

    def test_very_large_float_ratio(self):
        """3.0 * 768 = 2304, nearest 128 = 2304 (no upper bound for float ratios)."""
        assert resolve_encoder_dim(3.0, decoder_dim=768) == 2304

    def test_non_standard_decoder_dim(self):
        """decoder_dim=100 (not a multiple of 128): 0.5 * 100 = 50 -> min 128."""
        assert resolve_encoder_dim(0.5, decoder_dim=100) == 128

    def test_non_standard_decoder_dim_large_ratio(self):
        """decoder_dim=100, ratio=2.0 -> raw=200 -> round(200/128)*128 = round(1.5625)*128 = 2*128 = 256."""
        assert resolve_encoder_dim(2.0, decoder_dim=100) == 256

    def test_int_1_is_absolute_not_ratio(self):
        """int 1 should be treated as absolute dim=1 (no rounding to 128)."""
        assert resolve_encoder_dim(1, decoder_dim=768) == 1

    def test_int_not_multiple_of_128(self):
        """int values are passed through verbatim, no rounding."""
        assert resolve_encoder_dim(200, decoder_dim=768) == 200

    def test_rounding_exactly_at_half(self):
        """Exactly at .5 boundary: 0.5 * 256 = 128.0 -> round(1.0)*128 = 128."""
        assert resolve_encoder_dim(0.5, decoder_dim=256) == 128


# ---------------------------------------------------------------------------
# resolve_encoder_heads
# ---------------------------------------------------------------------------

class TestResolveEncoderHeads:
    """Tests for resolve_encoder_heads derivation and validation."""

    def test_none_derives_from_dim_and_head_dim(self):
        """n_head=None -> enc_dim // head_dim."""
        n_head, n_kv_head = resolve_encoder_heads(None, None, enc_dim=256, head_dim=64)
        assert n_head == 4
        assert n_kv_head == 4  # None -> matches n_head (MHA)

    def test_none_n_kv_head_defaults_to_n_head(self):
        """n_kv_head=None -> same as n_head (MHA)."""
        n_head, n_kv_head = resolve_encoder_heads(8, None, enc_dim=512, head_dim=64)
        assert n_head == 8
        assert n_kv_head == 8

    def test_explicit_both(self):
        """Both explicitly provided, no derivation needed."""
        n_head, n_kv_head = resolve_encoder_heads(8, 4, enc_dim=512, head_dim=64)
        assert n_head == 8
        assert n_kv_head == 4

    def test_minimal_valid_case(self):
        """n_head=1, n_kv_head=1: minimal valid config."""
        n_head, n_kv_head = resolve_encoder_heads(1, 1, enc_dim=64, head_dim=64)
        assert n_head == 1
        assert n_kv_head == 1

    def test_enc_dim_not_divisible_by_head_dim_derives_zero(self):
        """enc_dim=100, head_dim=64: 100//64=1, so n_head=1. Does not raise
        (integer division truncates), but enc_dim % n_head != 0 check triggers
        only if enc_dim % derived_n_head != 0. Here 100 % 1 == 0, so it passes."""
        n_head, n_kv_head = resolve_encoder_heads(None, None, enc_dim=100, head_dim=64)
        assert n_head == 1
        assert n_kv_head == 1

    def test_enc_dim_not_divisible_by_explicit_n_head_raises(self):
        """enc_dim % n_head != 0 should raise ValueError."""
        with pytest.raises(ValueError, match="divisible by n_head"):
            resolve_encoder_heads(3, 3, enc_dim=256, head_dim=64)

    def test_n_head_not_divisible_by_n_kv_head_raises(self):
        """n_head % n_kv_head != 0 should raise ValueError."""
        with pytest.raises(ValueError, match="divisible by n_kv_head"):
            resolve_encoder_heads(6, 4, enc_dim=384, head_dim=64)

    def test_n_head_zero_from_small_dim_raises(self):
        """enc_dim=32 cannot provide even one complete attention head."""
        with pytest.raises(ValueError, match="at least head_dim"):
            resolve_encoder_heads(None, None, enc_dim=32, head_dim=64)

    def test_explicit_zero_n_head_raises(self):
        """Explicitly passing n_head=0 should raise."""
        with pytest.raises(BeartypeCallHintParamViolation, match="n_head"):
            resolve_encoder_heads(0, 1, enc_dim=256, head_dim=64)

    def test_gqa_valid(self):
        """GQA: n_head=8, n_kv_head=2, enc_dim=512, head_dim=64."""
        n_head, n_kv_head = resolve_encoder_heads(8, 2, enc_dim=512, head_dim=64)
        assert n_head == 8
        assert n_kv_head == 2


# ---------------------------------------------------------------------------
# Cross-attention frequency — layer placement
# ---------------------------------------------------------------------------

class TestCrossAttentionFrequencyPlacement:
    """Verify which decoder layers receive cross-attention based on (i+1) % freq == 0."""

    @staticmethod
    def _xattn_layer_indices(freq, n_layer):
        """Replicate the placement logic from model.py line 122-123."""
        return [i for i in range(n_layer) if (i + 1) % freq == 0]

    def test_freq4_nlayer12(self):
        """freq=4, n_layer=12 -> layers 3, 7, 11 (3 total)."""
        indices = self._xattn_layer_indices(4, 12)
        assert indices == [3, 7, 11]

    def test_freq1_nlayer4(self):
        """freq=1, n_layer=4 -> all layers 0, 1, 2, 3 (4 total)."""
        indices = self._xattn_layer_indices(1, 4)
        assert indices == [0, 1, 2, 3]

    def test_freq6_nlayer12(self):
        """freq=6, n_layer=12 -> layers 5, 11 (2 total)."""
        indices = self._xattn_layer_indices(6, 12)
        assert indices == [5, 11]

    def test_freq_exceeds_nlayer(self):
        """freq=20, n_layer=12 -> 0 cross-attention layers (valid but vacuous)."""
        indices = self._xattn_layer_indices(20, 12)
        assert indices == []

    def test_freq_equals_nlayer(self):
        """freq=12, n_layer=12 -> only layer 11."""
        indices = self._xattn_layer_indices(12, 12)
        assert indices == [11]

    def test_freq0_would_cause_zerodiv(self):
        """frequency=0 is not validated by CrossAttentionConfig; it would cause
        ZeroDivisionError at model construction time."""
        with pytest.raises(ZeroDivisionError):
            # Replicate the modulo operation from model.py
            _freq = 0
            _ = [i for i in range(12) if (i + 1) % _freq == 0]


# ---------------------------------------------------------------------------
# Model-level tests (EncoderDecoderGPT construction + forward)
# ---------------------------------------------------------------------------

class TestEncoderDecoderGPTConstruction:
    """Model construction with various config edge cases.
    Uses torch.no_grad() throughout to avoid flex_attention backward issues on CPU."""

    @staticmethod
    def _make_model(
        dec_n_embd=128,
        dec_n_layer=2,
        dec_n_head=2,
        dec_n_kv_head=2,
        enc_n_embd=0.5,
        enc_depth=0.5,
        enc_n_head=None,
        enc_n_kv_head=None,
        enc_head_dim=64,
        xattn_freq=1,
        xattn_n_head=None,
        xattn_n_kv_head=None,
        n_sae_features=256,
        num_buckets=16,
        seq_len=64,
    ):
        from steering.conditional.encoder_decoder.model import EncoderDecoderGPT

        dec_config = GPTConfig(
            n_embd=dec_n_embd,
            n_layer=dec_n_layer,
            n_head=dec_n_head,
            n_kv_head=dec_n_kv_head,
            sequence_len=seq_len,
            vocab_size=256,
        )
        enc_config = EncoderConfig(
            n_embd=enc_n_embd,
            depth=enc_depth,
            n_head=enc_n_head,
            n_kv_head=enc_n_kv_head,
            head_dim=enc_head_dim,
        )
        xattn_config = CrossAttentionConfig(
            frequency=xattn_freq,
            n_head=xattn_n_head,
            n_kv_head=xattn_n_kv_head,
        )
        return EncoderDecoderGPT(
            dec_config=dec_config,
            enc_config=enc_config,
            xattn_config=xattn_config,
            n_sae_features=n_sae_features,
            num_buckets=num_buckets,
        )

    def test_minimal_model_constructs(self):
        """Minimal viable model: dec n_embd=64, n_layer=1, enc ratio -> 128 min, depth=1."""
        model = self._make_model(
            dec_n_embd=64,
            dec_n_layer=1,
            dec_n_head=1,
            dec_n_kv_head=1,
            enc_n_embd=0.5,   # 0.5 * 64 = 32 -> min 128
            enc_depth=1.0,    # 1.0 * 1 = 1
            enc_head_dim=64,
            xattn_freq=1,
        )
        assert model.enc_dim == 128

    def test_enc_dim_clamps_to_128_for_small_decoder(self):
        """When dec_n_embd is small, encoder dim should clamp to 128."""
        model = self._make_model(
            dec_n_embd=64,
            dec_n_layer=2,
            dec_n_head=1,
            dec_n_kv_head=1,
            enc_n_embd=0.5,  # 0.5 * 64 = 32 -> min 128
            enc_depth=1,
            enc_head_dim=64,
            xattn_freq=1,
        )
        assert model.enc_dim == 128

    def test_xattn_layers_placed_correctly(self):
        """Verify cross-attention layers exist at the right indices."""
        model = self._make_model(
            dec_n_embd=128,
            dec_n_layer=4,
            dec_n_head=2,
            dec_n_kv_head=2,
            enc_n_embd=128,
            enc_depth=1,
            enc_head_dim=64,
            xattn_freq=2,
        )
        xattn_indices = sorted(int(k) for k in model.cross_attn_layers)
        assert xattn_indices == [1, 3]

    def test_freq_exceeds_nlayer_no_xattn_layers(self):
        """freq > n_layer: no cross-attention layers created (vacuous)."""
        model = self._make_model(
            dec_n_embd=128,
            dec_n_layer=2,
            dec_n_head=2,
            dec_n_kv_head=2,
            enc_n_embd=128,
            enc_depth=1,
            enc_head_dim=64,
            xattn_freq=10,
        )
        assert len(model.cross_attn_layers) == 0

    def test_xattn_n_head_defaults_to_decoder(self):
        """When xattn n_head is None, it should default to dec_config.n_head."""
        model = self._make_model(
            dec_n_embd=128,
            dec_n_layer=2,
            dec_n_head=2,
            dec_n_kv_head=2,
            enc_n_embd=128,
            enc_depth=1,
            enc_head_dim=64,
            xattn_freq=1,
            xattn_n_head=None,
        )
        # Each cross-attention layer should have n_head == dec's n_head
        for layer in model.cross_attn_layers.values():
            assert layer.n_head == 2

    def test_xattn_head_dim_derived_from_dec(self):
        """xattn_head_dim = dec_n_embd // xattn_n_head."""
        model = self._make_model(
            dec_n_embd=128,
            dec_n_layer=2,
            dec_n_head=2,
            dec_n_kv_head=2,
            enc_n_embd=128,
            enc_depth=1,
            enc_head_dim=64,
            xattn_freq=1,
            xattn_n_head=None,  # -> 2 from decoder
        )
        # head_dim should be 128 // 2 = 64
        for layer in model.cross_attn_layers.values():
            assert layer.head_dim == 64

    def test_enc_depth_zero_raises(self):
        """Encoder depth=0 is rejected: FeatureEncoder requires at least 1 layer."""
        with pytest.raises(ValueError, match="at least 1 layer"):
            self._make_model(
                dec_n_embd=128,
                dec_n_layer=2,
                dec_n_head=2,
                dec_n_kv_head=2,
                enc_n_embd=128,
                enc_depth=0,
                enc_head_dim=64,
                xattn_freq=1,
            )

    def test_enc_depth_one_has_one_block(self):
        """Encoder depth=1 is the minimum: exactly 1 self-attention block."""
        model = self._make_model(
            dec_n_embd=128,
            dec_n_layer=2,
            dec_n_head=2,
            dec_n_kv_head=2,
            enc_n_embd=128,
            enc_depth=1,
            enc_head_dim=64,
            xattn_freq=1,
        )
        assert len(model.encoder.blocks) == 1


class TestWeightSharing:
    """Verify encoder and decoder do not accidentally share weight tensors."""

    @staticmethod
    def _make_model():
        from steering.conditional.encoder_decoder.model import EncoderDecoderGPT

        dec_config = GPTConfig(
            n_embd=128, n_layer=2, n_head=2, n_kv_head=2,
            sequence_len=64, vocab_size=256,
        )
        enc_config = EncoderConfig(n_embd=128, depth=1, head_dim=64)
        xattn_config = CrossAttentionConfig(frequency=1)
        return EncoderDecoderGPT(
            dec_config=dec_config, enc_config=enc_config,
            xattn_config=xattn_config, n_sae_features=256, num_buckets=16,
        )

    def test_encoder_feature_embed_distinct_from_decoder_wte(self):
        """feature_embed and wte must not share the same data pointer."""
        model = self._make_model()
        enc_ptr = model.encoder.feature_embed.weight.data_ptr()
        dec_ptr = model.transformer.wte.weight.data_ptr()
        assert enc_ptr != dec_ptr

    def test_encoder_blocks_distinct_from_decoder_blocks(self):
        """Encoder and decoder transformer blocks must be independent."""
        model = self._make_model()
        enc_params = {p.data_ptr() for p in model.encoder.blocks.parameters()}
        dec_params = {p.data_ptr() for p in model.transformer.h.parameters()}
        overlap = enc_params & dec_params
        assert len(overlap) == 0, f"Shared parameter data_ptrs: {overlap}"

    def test_cross_attn_distinct_from_decoder_blocks(self):
        """Cross-attention parameters must not overlap with decoder blocks."""
        model = self._make_model()
        xattn_params = {p.data_ptr() for p in model.cross_attn_layers.parameters()}
        dec_params = {p.data_ptr() for p in model.transformer.h.parameters()}
        overlap = xattn_params & dec_params
        assert len(overlap) == 0, f"Shared parameter data_ptrs: {overlap}"


class TestEncoderDecoderGPTForward:
    """Forward pass tests. Uses torch.no_grad() to avoid flex_attention backward issues on CPU."""

    @staticmethod
    def _make_model_and_inputs(
        dec_n_embd=128,
        dec_n_layer=2,
        dec_n_head=2,
        dec_n_kv_head=2,
        enc_n_embd=128,
        enc_depth=1,
        enc_head_dim=64,
        xattn_freq=1,
        n_sae_features=256,
        num_buckets=16,
        seq_len=64,
        batch_size=2,
        n_seg=2,
        n_features=4,
    ):
        from steering.conditional.encoder_decoder.model import EncoderDecoderGPT

        dec_config = GPTConfig(
            n_embd=dec_n_embd, n_layer=dec_n_layer,
            n_head=dec_n_head, n_kv_head=dec_n_kv_head,
            sequence_len=seq_len, vocab_size=256,
        )
        enc_config = EncoderConfig(
            n_embd=enc_n_embd, depth=enc_depth, head_dim=enc_head_dim,
        )
        xattn_config = CrossAttentionConfig(frequency=xattn_freq)
        model = EncoderDecoderGPT(
            dec_config=dec_config, enc_config=enc_config,
            xattn_config=xattn_config, n_sae_features=n_sae_features,
            num_buckets=num_buckets,
        )
        model.init_weights()

        B, T = batch_size, seq_len
        idx = torch.randint(0, 256, (B, T))
        targets = torch.randint(0, 256, (B, T))
        feature_ids = torch.randint(0, n_sae_features, (B, n_seg, n_features))
        magnitudes = torch.randint(0, num_buckets, (B, n_seg, n_features))
        mask = torch.ones(B, n_seg, n_features, dtype=torch.bool)
        seg_size = T // n_seg
        seg_map = torch.arange(T).unsqueeze(0).expand(B, -1) // seg_size
        seg_map = seg_map.clamp(max=n_seg - 1)

        return model, idx, targets, feature_ids, magnitudes, mask, seg_map

    @torch.no_grad()
    def test_forward_returns_scalar_loss(self):
        """Forward with targets should return a scalar loss."""
        model, idx, targets, fids, mags, mask, seg_map = self._make_model_and_inputs()
        loss = model(idx, targets, fids, mags, mask, seg_map)
        assert loss.ndim == 0
        assert loss.item() > 0

    @torch.no_grad()
    def test_forward_without_targets_returns_logits(self):
        """Forward without targets should return logits of shape (B, T, vocab)."""
        model, idx, targets, fids, mags, mask, seg_map = self._make_model_and_inputs()
        logits = model(idx, None, fids, mags, mask, seg_map)
        assert logits.shape == (2, 64, 256)

    @torch.no_grad()
    def test_forward_no_xattn_layers(self):
        """freq > n_layer: model still runs, just without cross-attention."""
        model, idx, targets, fids, mags, mask, seg_map = self._make_model_and_inputs(
            xattn_freq=100,
        )
        loss = model(idx, targets, fids, mags, mask, seg_map)
        assert loss.ndim == 0

    @torch.no_grad()
    def test_forward_enc_depth_one(self):
        """Encoder with depth=1 (minimum): still produces output."""
        model, idx, targets, fids, mags, mask, seg_map = self._make_model_and_inputs(
            enc_depth=1,
        )
        loss = model(idx, targets, fids, mags, mask, seg_map)
        assert loss.ndim == 0

    @torch.no_grad()
    def test_forward_minimal_model(self):
        """Minimal model: dec n_embd=64, 1 layer, encoder dim clamps to 128."""
        model, idx, targets, fids, mags, mask, seg_map = self._make_model_and_inputs(
            dec_n_embd=64,
            dec_n_layer=1,
            dec_n_head=1,
            dec_n_kv_head=1,
            enc_n_embd=128,
            enc_depth=1,
            enc_head_dim=64,
            xattn_freq=1,
            seq_len=32,
        )
        loss = model(idx, targets, fids, mags, mask, seg_map)
        assert loss.ndim == 0

    @torch.no_grad()
    def test_forward_with_precomputed_encoder_out(self):
        """Forward with precomputed encoder_out skips encoder."""
        model, idx, targets, fids, mags, mask, seg_map = self._make_model_and_inputs()
        # Run encoder separately to get encoder_out
        encoder_out = model.encoder(fids, mags, mask)
        # Forward with precomputed encoder_out
        loss = model(
            idx, targets,
            enc_feature_ids=None, enc_magnitudes=None,
            seg_map=seg_map,
            encoder_out=encoder_out,
        )
        assert loss.ndim == 0

    @torch.no_grad()
    def test_forward_loss_reduction_none(self):
        """loss_reduction='none' returns per-token losses."""
        model, idx, targets, fids, mags, mask, seg_map = self._make_model_and_inputs()
        loss = model(idx, targets, fids, mags, mask, seg_map, loss_reduction="none")
        assert loss.shape == (2 * 64,)  # B*T

    @torch.no_grad()
    def test_all_masked_encoder_features(self):
        """All encoder features masked: cross-attention should produce zeros (via nan_to_num)."""
        model, idx, targets, fids, mags, mask, seg_map = self._make_model_and_inputs()
        mask_all_false = torch.zeros_like(mask)
        loss = model(idx, targets, fids, mags, mask_all_false, seg_map)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# resolve_encoder_depth — additional edge cases
# ---------------------------------------------------------------------------

class TestResolveEncoderDepthEdgeCases:

    def test_float_rounds_half_to_even(self):
        """Python's round() uses banker's rounding. 0.5 * 5 = 2.5 -> round(2.5) = 2."""
        assert resolve_encoder_depth(0.5, decoder_depth=5) == 2

    def test_float_rounds_half_up_for_odd(self):
        """0.5 * 3 = 1.5 -> round(1.5) = 2 (banker's: rounds to even)."""
        assert resolve_encoder_depth(0.5, decoder_depth=3) == 2

    def test_float_very_small_nonzero(self):
        """0.01 * 12 = 0.12 -> rounds to 0."""
        assert resolve_encoder_depth(0.01, decoder_depth=12) == 0

    def test_int_exactly_2x(self):
        """Exactly 2x decoder depth boundary: 2 * 6 = 12."""
        assert resolve_encoder_depth(12, decoder_depth=6) == 12

    def test_int_exceeds_2x_by_one_raises(self):
        """13 > 2*6 = 12, should raise."""
        with pytest.raises(ValueError):
            resolve_encoder_depth(13, decoder_depth=6)


# ---------------------------------------------------------------------------
# Config round-trip: EncoderConfig + resolve functions compose correctly
# ---------------------------------------------------------------------------

class TestConfigRoundTrip:
    """Integration tests: EncoderConfig defaults compose with resolve functions."""

    def test_default_config_with_typical_decoder(self):
        """Default EncoderConfig + typical decoder (768, 12) should resolve cleanly."""
        cfg = EncoderConfig()
        dim = resolve_encoder_dim(cfg.n_embd, decoder_dim=768)
        depth = resolve_encoder_depth(cfg.depth, decoder_depth=12)
        n_head, n_kv_head = resolve_encoder_heads(cfg.n_head, cfg.n_kv_head, dim, cfg.head_dim)
        assert dim == 384
        assert depth == 6
        assert n_head == 6   # 384 // 64
        assert n_kv_head == 6

    def test_default_config_with_small_decoder(self):
        """Default ratios with a small decoder: n_embd=128, n_layer=4."""
        cfg = EncoderConfig()
        dim = resolve_encoder_dim(cfg.n_embd, decoder_dim=128)
        depth = resolve_encoder_depth(cfg.depth, decoder_depth=4)
        assert dim == 128  # 0.5 * 128 = 64 -> min 128
        assert depth == 2  # 0.5 * 4 = 2.0

    def test_custom_head_dim_changes_head_count(self):
        """head_dim=32 instead of 64: 384 // 32 = 12 heads."""
        cfg = EncoderConfig(head_dim=32)
        dim = resolve_encoder_dim(cfg.n_embd, decoder_dim=768)
        n_head, n_kv_head = resolve_encoder_heads(cfg.n_head, cfg.n_kv_head, dim, cfg.head_dim)
        assert dim == 384
        assert n_head == 12  # 384 // 32
        assert n_kv_head == 12
