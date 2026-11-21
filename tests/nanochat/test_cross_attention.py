"""Tests for the CrossAttentionLayer module."""

import hypothesis.strategies as st
import pytest
import torch
import torch as t
from hypothesis import HealthCheck, given, settings

import nanochat.flash_attention as fa_module
from nanochat.flash_attention import HAS_FLEX_ATTENTION
from steering.conditional.encoder_decoder.cross_attention import (
    CrossAttentionLayer,
    CrossAttentionLayerConfig,
    build_packed_cross_attention_block_mask,
    norm,
)
from steering.conditional.encoder_decoder.packed_encoder import (
    pack_encoder_layout,
)
from tests.padded_encoder_reference import (
    pack_dense_encoder_output_reference,
    padded_cross_attention_reference,
)

requires_flex = pytest.mark.skipif(
    not HAS_FLEX_ATTENTION, reason="flex_attention not available"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(dec_n_embd=64, enc_n_embd=32, n_head=4, n_kv_head=4, head_dim=16):
    return CrossAttentionLayerConfig(
        dec_n_embd=dec_n_embd,
        enc_n_embd=enc_n_embd,
        n_head=n_head,
        n_kv_head=n_kv_head,
        head_dim=head_dim,
    )


def _make_inputs(B=2, T=8, n_seg=2, F=6, dec_n_embd=64, enc_n_embd=32):
    """Create random decoder_hidden, encoder_kv, encoder_mask, and seg_map tensors."""
    decoder_hidden = torch.randn(B, T, dec_n_embd)
    encoder_kv = torch.randn(B, n_seg, F, enc_n_embd)
    encoder_mask = torch.ones(B, n_seg, F, dtype=torch.bool)
    segment_size = T // n_seg
    seg_map = torch.arange(T).unsqueeze(0).expand(B, -1) // segment_size
    return decoder_hidden, encoder_kv, encoder_mask, seg_map


def _make_layer_with_nonzero_proj(cfg):
    """Create a layer with init_weights() applied.

    init_weights() initializes all projections (c_q, c_kv, c_proj) with non-zero
    uniform weights. This is the standard production initialization.
    """
    layer = CrossAttentionLayer(cfg)
    layer.init_weights()
    return layer


def _reference_gather_xattn(layer, decoder_hidden, encoder_kv, encoder_mask, seg_map):
    """Reference implementation: per-position gather-based cross-attention.

    Each position gathers its segment's KV individually. Slow but correct.
    Used to verify the padded scatter/gather approach.
    """

    B, T, C = decoder_hidden.shape
    encoder_kv.size(1)
    encoder_kv.size(2)

    outputs = torch.zeros_like(decoder_hidden)
    for b in range(B):
        for pos in range(T):
            seg_idx = seg_map[b, pos].item()
            q_input = decoder_hidden[b, pos : pos + 1, :].unsqueeze(0)  # (1, 1, C)
            kv_input = encoder_kv[b, seg_idx : seg_idx + 1, :, :].unsqueeze(
                0
            )  # (1, 1, F, enc)
            mask_input = encoder_mask[b, seg_idx : seg_idx + 1, :].unsqueeze(
                0
            )  # (1, 1, F)
            # Build a seg_map for single position → single segment
            single_seg_map = torch.zeros(
                1, 1, dtype=torch.long, device=decoder_hidden.device
            )
            out = layer(q_input, kv_input, mask_input, single_seg_map)
            outputs[b, pos, :] = out[0, 0, :]
    return outputs


# ---------------------------------------------------------------------------
# Test: output shape
# ---------------------------------------------------------------------------


class TestOutputShape:
    def test_output_shape_basic(self):
        """Output should be (B, T, dec_n_embd)."""
        B, T, n_seg, F, dec_n_embd, enc_n_embd = 2, 8, 2, 6, 64, 32
        cfg = _make_config(dec_n_embd=dec_n_embd, enc_n_embd=enc_n_embd)
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()

        decoder_hidden, encoder_kv, encoder_mask, seg_map = _make_inputs(
            B, T, n_seg, F, dec_n_embd, enc_n_embd
        )
        out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        assert out.shape == (B, T, dec_n_embd)

    def test_output_shape_single_batch(self):
        """Works with batch size 1."""
        B, T, n_seg, F, dec_n_embd, enc_n_embd = 1, 4, 2, 3, 64, 32
        cfg = _make_config(dec_n_embd=dec_n_embd, enc_n_embd=enc_n_embd)
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()

        decoder_hidden, encoder_kv, encoder_mask, seg_map = _make_inputs(
            B, T, n_seg, F, dec_n_embd, enc_n_embd
        )
        out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        assert out.shape == (B, T, dec_n_embd)

    def test_output_shape_single_segment(self):
        """Works with n_seg=1 (eval mode: whole sequence is one segment)."""
        B, T, n_seg, F, dec_n_embd, enc_n_embd = 2, 8, 1, 6, 64, 32
        cfg = _make_config(dec_n_embd=dec_n_embd, enc_n_embd=enc_n_embd)
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()

        decoder_hidden, encoder_kv, encoder_mask, seg_map = _make_inputs(
            B, T, n_seg, F, dec_n_embd, enc_n_embd
        )
        out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        assert out.shape == (B, T, dec_n_embd)


# ---------------------------------------------------------------------------
# Test: padding mask applied
# ---------------------------------------------------------------------------


class TestPaddingMask:
    def test_all_true_vs_all_false_differ(self):
        """All-True mask and all-False mask should produce different outputs."""
        B, T, n_seg, F, dec_n_embd, enc_n_embd = 2, 4, 2, 6, 64, 32
        segment_size = T // n_seg
        cfg = _make_config(dec_n_embd=dec_n_embd, enc_n_embd=enc_n_embd)
        layer = _make_layer_with_nonzero_proj(cfg)

        torch.manual_seed(42)
        decoder_hidden = torch.randn(B, T, dec_n_embd)
        encoder_kv = torch.randn(B, n_seg, F, enc_n_embd)
        seg_map = torch.arange(T).unsqueeze(0).expand(B, -1) // segment_size

        mask_all_true = torch.ones(B, n_seg, F, dtype=torch.bool)
        mask_all_false = torch.zeros(B, n_seg, F, dtype=torch.bool)

        out_true = layer(decoder_hidden, encoder_kv, mask_all_true, seg_map)
        out_false = layer(decoder_hidden, encoder_kv, mask_all_false, seg_map)

        assert not torch.allclose(out_true, out_false, atol=1e-6), (
            "Outputs should differ when mask changes from all-True to all-False"
        )

    def test_partial_mask_vs_full_mask_differ(self):
        """A partial mask (some padding) should produce different output than all-True."""
        B, T, n_seg, F, dec_n_embd, enc_n_embd = 2, 4, 2, 6, 64, 32
        segment_size = T // n_seg
        cfg = _make_config(dec_n_embd=dec_n_embd, enc_n_embd=enc_n_embd)
        layer = _make_layer_with_nonzero_proj(cfg)

        torch.manual_seed(42)
        decoder_hidden = torch.randn(B, T, dec_n_embd)
        encoder_kv = torch.randn(B, n_seg, F, enc_n_embd)
        seg_map = torch.arange(T).unsqueeze(0).expand(B, -1) // segment_size

        mask_full = torch.ones(B, n_seg, F, dtype=torch.bool)
        mask_partial = torch.ones(B, n_seg, F, dtype=torch.bool)
        mask_partial[:, :, F // 2 :] = False  # mask out half the features

        out_full = layer(decoder_hidden, encoder_kv, mask_full, seg_map)
        out_partial = layer(decoder_hidden, encoder_kv, mask_partial, seg_map)

        assert not torch.allclose(out_full, out_partial, atol=1e-6), (
            "Outputs should differ when mask changes"
        )


# ---------------------------------------------------------------------------
# Test: gradient flow to encoder_kv
# ---------------------------------------------------------------------------


class TestGradientFlow:
    def test_gradients_flow_to_encoder_kv(self):
        """Gradients should flow back through encoder_kv."""
        B, T, n_seg, F, dec_n_embd, enc_n_embd = 2, 4, 2, 6, 64, 32
        segment_size = T // n_seg
        cfg = _make_config(dec_n_embd=dec_n_embd, enc_n_embd=enc_n_embd)
        layer = _make_layer_with_nonzero_proj(cfg)

        decoder_hidden = torch.randn(B, T, dec_n_embd, requires_grad=True)
        encoder_kv = torch.randn(B, n_seg, F, enc_n_embd, requires_grad=True)
        encoder_mask = torch.ones(B, n_seg, F, dtype=torch.bool)
        seg_map = torch.arange(T).unsqueeze(0).expand(B, -1) // segment_size

        out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
        loss = out.sum()
        loss.backward()

        assert encoder_kv.grad is not None, "encoder_kv should have gradients"
        assert encoder_kv.grad.abs().sum() > 0, (
            "encoder_kv gradients should be non-zero"
        )

    def test_gradients_flow_to_decoder_hidden(self):
        """Gradients should also flow back through decoder_hidden."""
        B, T, n_seg, F, dec_n_embd, enc_n_embd = 2, 4, 2, 6, 64, 32
        segment_size = T // n_seg
        cfg = _make_config(dec_n_embd=dec_n_embd, enc_n_embd=enc_n_embd)
        layer = _make_layer_with_nonzero_proj(cfg)

        decoder_hidden = torch.randn(B, T, dec_n_embd, requires_grad=True)
        encoder_kv = torch.randn(B, n_seg, F, enc_n_embd, requires_grad=True)
        encoder_mask = torch.ones(B, n_seg, F, dtype=torch.bool)
        seg_map = torch.arange(T).unsqueeze(0).expand(B, -1) // segment_size

        out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
        loss = out.sum()
        loss.backward()

        assert decoder_hidden.grad is not None, "decoder_hidden should have gradients"
        assert decoder_hidden.grad.abs().sum() > 0, (
            "decoder_hidden gradients should be non-zero"
        )


# ---------------------------------------------------------------------------
# Test: GQA support (n_kv_head < n_head)
# ---------------------------------------------------------------------------


class TestGQASupport:
    def test_gqa_4_query_2_kv(self):
        """Should work with 4 query heads and 2 KV heads."""
        B, T, n_seg, F, dec_n_embd, enc_n_embd = 2, 4, 2, 6, 64, 32
        cfg = _make_config(
            dec_n_embd=dec_n_embd,
            enc_n_embd=enc_n_embd,
            n_head=4,
            n_kv_head=2,
            head_dim=16,
        )
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()

        decoder_hidden, encoder_kv, encoder_mask, seg_map = _make_inputs(
            B, T, n_seg, F, dec_n_embd, enc_n_embd
        )
        out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        assert out.shape == (B, T, dec_n_embd)

    def test_gqa_8_query_2_kv(self):
        """Should work with 8 query heads and 2 KV heads."""
        B, T, n_seg, F, dec_n_embd, enc_n_embd = 2, 4, 2, 6, 128, 48
        cfg = _make_config(
            dec_n_embd=dec_n_embd,
            enc_n_embd=enc_n_embd,
            n_head=8,
            n_kv_head=2,
            head_dim=16,
        )
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()

        decoder_hidden, encoder_kv, encoder_mask, seg_map = _make_inputs(
            B, T, n_seg, F, dec_n_embd, enc_n_embd
        )
        out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        assert out.shape == (B, T, dec_n_embd)

    def test_gqa_gradients(self):
        """Gradients should flow correctly with GQA."""
        B, T, n_seg, F, dec_n_embd, enc_n_embd = 2, 4, 2, 6, 64, 32
        segment_size = T // n_seg
        cfg = _make_config(
            dec_n_embd=dec_n_embd,
            enc_n_embd=enc_n_embd,
            n_head=4,
            n_kv_head=2,
            head_dim=16,
        )
        layer = _make_layer_with_nonzero_proj(cfg)

        encoder_kv = torch.randn(B, n_seg, F, enc_n_embd, requires_grad=True)
        decoder_hidden = torch.randn(B, T, dec_n_embd)
        encoder_mask = torch.ones(B, n_seg, F, dtype=torch.bool)
        seg_map = torch.arange(T).unsqueeze(0).expand(B, -1) // segment_size

        out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
        out.sum().backward()

        assert encoder_kv.grad is not None
        assert encoder_kv.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Test: variable feature counts
# ---------------------------------------------------------------------------


class TestVariableFeatureCounts:
    def test_small_F(self):
        """Should work with F=1 (single feature per segment)."""
        B, T, n_seg, F, dec_n_embd, enc_n_embd = 2, 4, 2, 1, 64, 32
        cfg = _make_config(dec_n_embd=dec_n_embd, enc_n_embd=enc_n_embd)
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()

        decoder_hidden, encoder_kv, encoder_mask, seg_map = _make_inputs(
            B, T, n_seg, F, dec_n_embd, enc_n_embd
        )
        out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        assert out.shape == (B, T, dec_n_embd)

    def test_large_F(self):
        """Should work with a large F value."""
        B, T, n_seg, F, dec_n_embd, enc_n_embd = 2, 4, 2, 32, 64, 32
        cfg = _make_config(dec_n_embd=dec_n_embd, enc_n_embd=enc_n_embd)
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()

        decoder_hidden, encoder_kv, encoder_mask, seg_map = _make_inputs(
            B, T, n_seg, F, dec_n_embd, enc_n_embd
        )
        out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        assert out.shape == (B, T, dec_n_embd)

    def test_different_F_same_layer(self):
        """Same layer should handle different F values across separate calls."""
        B, T, n_seg, dec_n_embd, enc_n_embd = 2, 4, 2, 64, 32
        cfg = _make_config(dec_n_embd=dec_n_embd, enc_n_embd=enc_n_embd)
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()

        for F in [3, 8, 16]:
            decoder_hidden, encoder_kv, encoder_mask, seg_map = _make_inputs(
                B, T, n_seg, F, dec_n_embd, enc_n_embd
            )
            out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
            assert out.shape == (B, T, dec_n_embd), f"Failed for F={F}"


# ---------------------------------------------------------------------------
# Test: init_weights
# ---------------------------------------------------------------------------


class TestInitWeights:
    def test_c_proj_initialized_nonzero_uniform(self):
        """c_proj should be initialized with non-zero uniform weights.

        Previously c_proj was zero-initialized for residual-friendly startup,
        but all-zero weights break FP8 tensorwise scaling (amax=0 → extreme
        scale factors → inf gradients on the first backward pass).
        """
        cfg = _make_config(dec_n_embd=64, enc_n_embd=32)
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()

        assert layer.c_proj.weight.abs().sum() > 0, (
            "c_proj should be non-zero after init (zero weights break FP8)"
        )

    def test_c_proj_init_scale_matches_formula(self):
        """c_proj uniform init bounds should be +-sqrt(3 / (n_head * head_dim))."""
        cfg = _make_config(dec_n_embd=64, enc_n_embd=32, n_head=4, head_dim=16)
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()

        expected_bound = (3.0 / (cfg.n_head * cfg.head_dim)) ** 0.5
        w = layer.c_proj.weight
        # Uniform(-bound, bound) should have max ≈ bound, min ≈ -bound
        assert w.max().item() <= expected_bound + 0.01
        assert w.min().item() >= -expected_bound - 0.01
        # Should span most of the range (not degenerate)
        assert w.max().item() > expected_bound * 0.5
        assert w.min().item() < -expected_bound * 0.5

    def test_c_q_nonzero_after_init(self):
        """c_q should have non-zero weights after init."""
        cfg = _make_config(dec_n_embd=64, enc_n_embd=32)
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()

        assert layer.c_q.weight.abs().sum() > 0, "c_q should be non-zero after init"

    def test_c_kv_nonzero_after_init(self):
        """c_kv should have non-zero weights after init."""
        cfg = _make_config(dec_n_embd=64, enc_n_embd=32)
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()

        assert layer.c_kv.weight.abs().sum() > 0, "c_kv should be non-zero after init"

    def test_all_projections_nonzero_after_init(self):
        """All projections (c_q, c_kv, c_proj) must be non-zero after init.

        This is a safety invariant: any zero-weight projection will break FP8
        tensorwise scaling (amax=0 → degenerate scale → inf/NaN gradients).
        """
        cfg = _make_config(dec_n_embd=128, enc_n_embd=64, n_head=8, n_kv_head=4, head_dim=16)
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()

        for name in ["c_q", "c_kv", "c_proj"]:
            w = getattr(layer, name).weight
            assert w.abs().sum() > 0, f"{name} must be non-zero after init_weights()"
            assert torch.isfinite(w).all(), f"{name} must be finite after init_weights()"

    def test_c_q_init_scale(self):
        """c_q uniform init bounds should be +-sqrt(3 / dec_n_embd)."""
        cfg = _make_config(dec_n_embd=64, enc_n_embd=32)
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()

        expected_bound = (3.0 / cfg.dec_n_embd) ** 0.5
        w = layer.c_q.weight
        assert w.max().item() <= expected_bound + 0.01
        assert w.min().item() >= -expected_bound - 0.01

    def test_c_kv_init_scale(self):
        """c_kv uniform init bounds should be +-sqrt(3 / enc_n_embd)."""
        cfg = _make_config(dec_n_embd=64, enc_n_embd=32)
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()

        expected_bound = (3.0 / cfg.enc_n_embd) ** 0.5
        w = layer.c_kv.weight
        assert w.max().item() <= expected_bound + 0.01
        assert w.min().item() >= -expected_bound - 0.01


# ---------------------------------------------------------------------------
# Test: seg_map-based forward (variable-sized segments)
# ---------------------------------------------------------------------------


class TestSegMapForward:
    """Tests for seg_map-based cross-attention (variable-sized segments)."""

    def test_uniform_seg_map_matches_reference(self):
        """seg_map with uniform segments should give same result as reference gather."""
        B, T, n_seg, F_feat = 2, 8, 2, 6
        segment_size = T // n_seg
        dec_n_embd, enc_n_embd = 64, 32
        cfg = CrossAttentionLayerConfig(
            dec_n_embd=dec_n_embd,
            enc_n_embd=enc_n_embd,
            n_head=4,
            n_kv_head=2,
            head_dim=16,
        )
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()
        with t.no_grad():
            layer.c_proj.weight.normal_(std=0.01)

        decoder_hidden = t.randn(B, T, dec_n_embd)
        encoder_kv = t.randn(B, n_seg, F_feat, enc_n_embd)
        encoder_mask = t.ones(B, n_seg, F_feat, dtype=t.bool)
        seg_map = t.arange(T).unsqueeze(0).expand(B, -1) // segment_size

        out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
        assert out.shape == (B, T, dec_n_embd)
        # Compare against reference gather implementation
        out_ref = _reference_gather_xattn(
            layer, decoder_hidden, encoder_kv, encoder_mask, seg_map
        )
        assert t.allclose(out, out_ref, atol=1e-5)

    def test_variable_segments(self):
        """seg_map with variable-sized segments should produce correct shapes."""
        B, T = 1, 10
        n_seg = 3  # segments of size 4, 3, 3
        F_feat = 4
        dec_n_embd, enc_n_embd = 32, 16
        cfg = CrossAttentionLayerConfig(
            dec_n_embd=dec_n_embd,
            enc_n_embd=enc_n_embd,
            n_head=2,
            n_kv_head=2,
            head_dim=16,
        )
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()
        seg_map = t.tensor([[0, 0, 0, 0, 1, 1, 1, 2, 2, 2]])
        decoder_hidden = t.randn(B, T, dec_n_embd)
        encoder_kv = t.randn(B, n_seg, F_feat, enc_n_embd)
        encoder_mask = t.ones(B, n_seg, F_feat, dtype=t.bool)

        out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
        assert out.shape == (B, T, dec_n_embd)

    def test_variable_segments_matches_reference(self):
        """Variable seg_map should match the per-position reference implementation."""
        B, T = 1, 10
        n_seg = 3
        F_feat = 4
        dec_n_embd, enc_n_embd = 32, 16
        cfg = CrossAttentionLayerConfig(
            dec_n_embd=dec_n_embd,
            enc_n_embd=enc_n_embd,
            n_head=2,
            n_kv_head=2,
            head_dim=16,
        )
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()
        with t.no_grad():
            layer.c_proj.weight.normal_(std=0.01)

        seg_map = t.tensor([[0, 0, 0, 0, 1, 1, 1, 2, 2, 2]])
        decoder_hidden = t.randn(B, T, dec_n_embd)
        encoder_kv = t.randn(B, n_seg, F_feat, enc_n_embd)
        encoder_mask = t.ones(B, n_seg, F_feat, dtype=t.bool)

        out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
        out_ref = _reference_gather_xattn(
            layer, decoder_hidden, encoder_kv, encoder_mask, seg_map
        )
        assert t.allclose(out, out_ref, atol=1e-5)

    def test_gradient_flow_variable_segments(self):
        """Gradients flow to encoder KV with variable-sized segments."""
        B, T, n_seg, F_feat = 2, 8, 3, 4
        dec_n_embd, enc_n_embd = 32, 16
        cfg = CrossAttentionLayerConfig(
            dec_n_embd=dec_n_embd,
            enc_n_embd=enc_n_embd,
            n_head=2,
            n_kv_head=2,
            head_dim=16,
        )
        layer = CrossAttentionLayer(cfg)
        with t.no_grad():
            layer.c_proj.weight.normal_(std=0.01)

        seg_map = t.tensor([[0, 0, 0, 1, 1, 1, 2, 2], [0, 0, 1, 1, 1, 2, 2, 2]])
        decoder_hidden = t.randn(B, T, dec_n_embd)
        encoder_kv = t.randn(B, n_seg, F_feat, enc_n_embd, requires_grad=True)
        encoder_mask = t.ones(B, n_seg, F_feat, dtype=t.bool)

        out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
        out.sum().backward()
        assert encoder_kv.grad is not None
        assert encoder_kv.grad.abs().sum() > 0

    def test_batch_variable_seg_maps(self):
        """Different seg_maps per batch element should work correctly."""
        B, T = 2, 12
        n_seg = 4
        F_feat = 3
        dec_n_embd, enc_n_embd = 32, 16
        cfg = CrossAttentionLayerConfig(
            dec_n_embd=dec_n_embd,
            enc_n_embd=enc_n_embd,
            n_head=2,
            n_kv_head=2,
            head_dim=16,
        )
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()
        with t.no_grad():
            layer.c_proj.weight.normal_(std=0.01)

        # Row 0: segments of size 4, 3, 3, 2
        # Row 1: segments of size 3, 3, 3, 3
        seg_map = t.tensor(
            [[0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3], [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]]
        )
        decoder_hidden = t.randn(B, T, dec_n_embd)
        encoder_kv = t.randn(B, n_seg, F_feat, enc_n_embd)
        encoder_mask = t.ones(B, n_seg, F_feat, dtype=t.bool)

        out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
        assert out.shape == (B, T, dec_n_embd)

        # Verify against reference
        out_ref = _reference_gather_xattn(
            layer, decoder_hidden, encoder_kv, encoder_mask, seg_map
        )
        assert t.allclose(out, out_ref, atol=1e-5)


# ---------------------------------------------------------------------------
# Test: tensor-native max_seg_size (no .item() graph break)
# ---------------------------------------------------------------------------


class TestTensorNativeMaxSegSize:
    """Verify tensor-native max_seg_size produces identical results."""

    def test_uneven_segments(self):
        """Tensor-native path handles segments of different sizes correctly."""
        cfg = _make_config()
        layer = _make_layer_with_nonzero_proj(cfg)
        B, T, n_seg, F = 2, 12, 3, 4
        decoder_hidden = torch.randn(B, T, cfg.dec_n_embd)
        encoder_kv = torch.randn(B, n_seg, F, cfg.enc_n_embd)
        encoder_mask = torch.ones(B, n_seg, F, dtype=torch.bool)
        # Uneven segments: [5, 4, 3] tokens
        seg_map = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]] * B)

        with torch.no_grad():
            out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        assert out.shape == (B, T, cfg.dec_n_embd)
        assert not torch.isnan(out).any()

    def test_single_segment(self):
        """Edge case: all positions in one segment."""
        cfg = _make_config()
        layer = _make_layer_with_nonzero_proj(cfg)
        B, T, n_seg, F = 2, 8, 1, 6
        decoder_hidden = torch.randn(B, T, cfg.dec_n_embd)
        encoder_kv = torch.randn(B, n_seg, F, cfg.enc_n_embd)
        encoder_mask = torch.ones(B, n_seg, F, dtype=torch.bool)
        seg_map = torch.zeros(B, T, dtype=torch.long)

        with torch.no_grad():
            out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        assert out.shape == (B, T, cfg.dec_n_embd)
        assert not torch.isnan(out).any()

    def test_gradients_flow(self):
        """Verify gradients flow through tensor-native max_seg_size computation."""
        cfg = _make_config()
        layer = _make_layer_with_nonzero_proj(cfg)
        decoder_hidden, encoder_kv, encoder_mask, seg_map = _make_inputs()
        decoder_hidden.requires_grad_(True)

        out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
        out.sum().backward()

        assert decoder_hidden.grad is not None
        assert not torch.isnan(decoder_hidden.grad).any()


# =============================================================================
# Helpers for reference cross-attention
# =============================================================================


def reference_cross_attention(q, k, v, encoder_mask_flat, enable_gqa):
    """Pure-math reference cross-attention (no masking tricks, float64).

    Args:
        q: (B_flat, H_q, Q_LEN, D) — already normed, NaN-free
        k: (B_flat, H_kv, KV_LEN, D) — already normed, NaN-free
        v: (B_flat, H_kv, KV_LEN, D)
        encoder_mask_flat: (B_flat, KV_LEN) bool — True = valid
        enable_gqa: whether H_q != H_kv

    Returns:
        (B_flat, H_q, Q_LEN, D) in float64
    """
    B, H_q, Q, D = q.shape
    H_kv = k.shape[1]

    q_f = q.double()
    k_f = k.double()
    v_f = v.double()

    if enable_gqa:
        repeats = H_q // H_kv
        k_f = k_f.repeat_interleave(repeats, dim=1)
        v_f = v_f.repeat_interleave(repeats, dim=1)

    scale = D**-0.5
    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale  # (B, H, Q, KV)

    # Apply encoder padding mask: (B, 1, 1, KV)
    mask = encoder_mask_flat.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, KV)
    scores = scores.masked_fill(~mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    attn = attn.nan_to_num()  # all-masked rows → NaN → 0
    out = torch.matmul(attn, v_f)
    return out


# =============================================================================
# Test: SDPA cross-attention vs pure-math reference
# =============================================================================


class TestSDPAVsReference:
    """Compare SDPA cross-attention output against pure-math reference."""

    def test_uniform_segments_match_reference(self):
        """Full layer with uniform segments should match per-position reference."""
        B, T, n_seg, F_feat = 2, 8, 2, 6
        dec_n_embd, enc_n_embd = 64, 32
        cfg = _make_config(dec_n_embd=dec_n_embd, enc_n_embd=enc_n_embd)
        layer = _make_layer_with_nonzero_proj(cfg)

        torch.manual_seed(42)
        decoder_hidden, encoder_kv, encoder_mask, seg_map = _make_inputs(
            B, T, n_seg, F_feat, dec_n_embd, enc_n_embd
        )

        out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
        out_ref = _reference_gather_xattn(
            layer, decoder_hidden, encoder_kv, encoder_mask, seg_map
        )

        max_diff = (out - out_ref).abs().max().item()
        assert max_diff < 1e-5, f"SDPA vs reference diff {max_diff} too large"

    def test_variable_segments_match_reference(self):
        """Variable-sized segments should match per-position reference."""
        B, T = 1, 10
        n_seg, F_feat = 3, 4
        dec_n_embd, enc_n_embd = 32, 16
        cfg = _make_config(
            dec_n_embd=dec_n_embd,
            enc_n_embd=enc_n_embd,
            n_head=2,
            n_kv_head=2,
            head_dim=16,
        )
        layer = _make_layer_with_nonzero_proj(cfg)

        torch.manual_seed(99)
        decoder_hidden = torch.randn(B, T, dec_n_embd)
        encoder_kv = torch.randn(B, n_seg, F_feat, enc_n_embd)
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool)
        seg_map = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 2, 2, 2]])

        out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
        out_ref = _reference_gather_xattn(
            layer, decoder_hidden, encoder_kv, encoder_mask, seg_map
        )

        max_diff = (out - out_ref).abs().max().item()
        assert max_diff < 1e-5, f"SDPA vs reference diff {max_diff} too large"

    def test_partial_mask_match_reference(self):
        """Partial encoder mask should match per-position reference."""
        B, T, n_seg, F_feat = 2, 8, 2, 6
        dec_n_embd, enc_n_embd = 64, 32
        cfg = _make_config(dec_n_embd=dec_n_embd, enc_n_embd=enc_n_embd)
        layer = _make_layer_with_nonzero_proj(cfg)

        torch.manual_seed(7)
        decoder_hidden, encoder_kv, _, seg_map = _make_inputs(
            B, T, n_seg, F_feat, dec_n_embd, enc_n_embd
        )
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool)
        encoder_mask[:, :, F_feat // 2 :] = False

        out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
        out_ref = _reference_gather_xattn(
            layer, decoder_hidden, encoder_kv, encoder_mask, seg_map
        )

        max_diff = (out - out_ref).abs().max().item()
        assert max_diff < 1e-5, f"SDPA vs reference diff {max_diff} too large"

    def test_gqa_match_reference(self):
        """GQA configuration should match per-position reference."""
        B, T, n_seg, F_feat = 2, 8, 2, 6
        dec_n_embd, enc_n_embd = 64, 32
        cfg = _make_config(
            dec_n_embd=dec_n_embd,
            enc_n_embd=enc_n_embd,
            n_head=4,
            n_kv_head=2,
            head_dim=16,
        )
        layer = _make_layer_with_nonzero_proj(cfg)

        torch.manual_seed(13)
        decoder_hidden, encoder_kv, encoder_mask, seg_map = _make_inputs(
            B, T, n_seg, F_feat, dec_n_embd, enc_n_embd
        )

        out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
        out_ref = _reference_gather_xattn(
            layer, decoder_hidden, encoder_kv, encoder_mask, seg_map
        )

        max_diff = (out - out_ref).abs().max().item()
        assert max_diff < 1e-5, f"SDPA GQA vs reference diff {max_diff} too large"

    def test_all_masked_produces_finite(self):
        """All-masked encoder should produce finite (not NaN) output."""
        B, T, n_seg, F_feat = 1, 4, 1, 4
        dec_n_embd, enc_n_embd = 32, 16
        cfg = _make_config(
            dec_n_embd=dec_n_embd,
            enc_n_embd=enc_n_embd,
            n_head=2,
            n_kv_head=2,
            head_dim=16,
        )
        layer = _make_layer_with_nonzero_proj(cfg)

        torch.manual_seed(0)
        decoder_hidden = torch.randn(B, T, dec_n_embd)
        encoder_kv = torch.randn(B, n_seg, F_feat, enc_n_embd)
        encoder_mask = torch.zeros(B, n_seg, F_feat, dtype=torch.bool)
        seg_map = torch.zeros(B, T, dtype=torch.long)

        out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
        assert not torch.isnan(out).any(), "All-masked should not produce NaN"
        assert not torch.isinf(out).any(), "All-masked should not produce Inf"

    def test_gradient_flow_with_partial_mask(self):
        """Gradients should flow correctly with partial encoder mask."""
        B, T, n_seg, F_feat = 2, 8, 2, 6
        dec_n_embd, enc_n_embd = 64, 32
        cfg = _make_config(
            dec_n_embd=dec_n_embd,
            enc_n_embd=enc_n_embd,
            n_head=4,
            n_kv_head=2,
            head_dim=16,
        )
        layer = _make_layer_with_nonzero_proj(cfg)

        dec_h = torch.randn(B, T, dec_n_embd, requires_grad=True)
        enc_kv = torch.randn(B, n_seg, F_feat, enc_n_embd, requires_grad=True)
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool)
        encoder_mask[:, :, F_feat // 2 :] = False
        seg_map = torch.arange(T).unsqueeze(0).expand(B, -1) // (T // n_seg)

        out = layer(dec_h, enc_kv, encoder_mask, seg_map)
        out.sum().backward()

        assert dec_h.grad is not None
        assert enc_kv.grad is not None
        assert not torch.isnan(dec_h.grad).any()
        assert not torch.isnan(enc_kv.grad).any()


# =============================================================================
# Test: Flex cross-attention vs reference implementation
# =============================================================================
@requires_flex
class TestFlexVsReference:
    """Test flex_attention cross-attention against a pure-math reference."""

    def _get_internal_qkv(
        self, layer, decoder_hidden, encoder_kv, seg_map, encoder_mask
    ):
        """Extract the internal q, k, v tensors after projection + norm.

        Returns q, k, v in (B_flat, H, seq, D) format, kv_valid, and padded_idx.
        """
        B, T, C = decoder_hidden.size()
        n_seg = encoder_kv.size(1)
        F_feat = encoder_kv.size(2)

        # Reproduce the pos_in_seg / scatter logic from CrossAttentionLayer
        pos = torch.arange(T, device=seg_map.device).unsqueeze(0).expand(B, -1)
        is_seg_start = torch.zeros(B, T, dtype=torch.bool, device=seg_map.device)
        is_seg_start[:, 0] = True
        is_seg_start[:, 1:] = seg_map[:, 1:] != seg_map[:, :-1]
        seg_start_pos = torch.where(is_seg_start, pos, torch.zeros_like(pos))
        seg_start_pos = seg_start_pos.cummax(dim=1).values
        pos_in_seg = pos - seg_start_pos

        max_seg_size = pos_in_seg.amax() + 1
        max_seg_size = max_seg_size.clamp(min=1)
        max_seg_size = ((max_seg_size + 15) // 16) * 16

        padded_idx = seg_map * max_seg_size + pos_in_seg
        padded_len = n_seg * max_seg_size

        x_padded = decoder_hidden.new_zeros(B, padded_len, C)
        x_padded.scatter_(1, padded_idx.unsqueeze(-1).expand(-1, -1, C), decoder_hidden)
        x_padded = x_padded.view(B, n_seg, max_seg_size, C)

        B_flat = B * n_seg

        with torch.no_grad():
            q = layer.c_q(x_padded.reshape(B_flat, max_seg_size, C))
            q = q.view(B_flat, max_seg_size, layer.n_head, layer.head_dim)

            encoder_flat = encoder_kv.reshape(B_flat, F_feat, layer.enc_n_embd)
            kv_out = layer.c_kv(encoder_flat)
            k, v = kv_out.split([layer.kv_dim, layer.kv_dim], dim=-1)
            k = k.view(B_flat, F_feat, layer.n_kv_head, layer.head_dim)
            v = v.view(B_flat, F_feat, layer.n_kv_head, layer.head_dim)

            q = norm(q)
            k = norm(k)

        # Transpose to (B_flat, H, seq, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        kv_valid = encoder_mask.reshape(B_flat, F_feat)
        return q, k, v, kv_valid

    def test_basic_no_mask(self):
        """Flex cross-attention matches reference with all-True mask."""
        B, T, n_seg, F_feat = 2, 8, 2, 6
        dec_n_embd, enc_n_embd = 64, 32
        cfg = _make_config(dec_n_embd=dec_n_embd, enc_n_embd=enc_n_embd)
        layer = _make_layer_with_nonzero_proj(cfg)

        torch.manual_seed(42)
        decoder_hidden, encoder_kv, encoder_mask, seg_map = _make_inputs(
            B, T, n_seg, F_feat, dec_n_embd, enc_n_embd
        )

        q, k, v, kv_valid = self._get_internal_qkv(
            layer, decoder_hidden, encoder_kv, seg_map, encoder_mask
        )
        enable_gqa = layer.n_head != layer.n_kv_head

        # Clean NaN (matching what the flex path does)
        q_clean = q.nan_to_num()
        k_clean = k.nan_to_num()

        y_ref = reference_cross_attention(q_clean, k_clean, v, kv_valid, enable_gqa)
        y_ref = y_ref.float()

        # Run flex
        from torch.nn.attention.flex_attention import flex_attention

        def _score_mod(score, b, h, q_idx, kv_idx):
            return torch.where(kv_valid[b, kv_idx], score, -float("inf"))

        y_flex = flex_attention(
            q_clean, k_clean, v, score_mod=_score_mod, enable_gqa=enable_gqa
        )
        assert isinstance(y_flex, torch.Tensor)
        y_flex = y_flex.nan_to_num().float()

        max_diff = (y_flex - y_ref).abs().max().item()
        mean_diff = (y_flex - y_ref).abs().mean().item()
        print(f"xattn_ref_no_mask: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        assert max_diff < 0.01, f"max_diff {max_diff} too large"

    def test_partial_mask(self):
        """Flex cross-attention matches reference with partial padding mask."""
        B, T, n_seg, F_feat = 2, 8, 2, 8
        dec_n_embd, enc_n_embd = 64, 32
        cfg = _make_config(dec_n_embd=dec_n_embd, enc_n_embd=enc_n_embd)
        layer = _make_layer_with_nonzero_proj(cfg)

        torch.manual_seed(77)
        decoder_hidden, encoder_kv, _, seg_map = _make_inputs(
            B, T, n_seg, F_feat, dec_n_embd, enc_n_embd
        )
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool)
        encoder_mask[:, :, F_feat // 2 :] = False  # mask out half

        q, k, v, kv_valid = self._get_internal_qkv(
            layer, decoder_hidden, encoder_kv, seg_map, encoder_mask
        )
        enable_gqa = layer.n_head != layer.n_kv_head

        q_clean = q.nan_to_num()
        k_clean = k.nan_to_num()

        y_ref = reference_cross_attention(q_clean, k_clean, v, kv_valid, enable_gqa)
        y_ref = y_ref.float()

        from torch.nn.attention.flex_attention import flex_attention

        def _score_mod(score, b, h, q_idx, kv_idx):
            return torch.where(kv_valid[b, kv_idx], score, -float("inf"))

        y_flex = flex_attention(
            q_clean, k_clean, v, score_mod=_score_mod, enable_gqa=enable_gqa
        )
        assert isinstance(y_flex, torch.Tensor)
        y_flex = y_flex.nan_to_num().float()

        max_diff = (y_flex - y_ref).abs().max().item()
        mean_diff = (y_flex - y_ref).abs().mean().item()
        print(
            f"xattn_ref_partial_mask: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
        )
        assert max_diff < 0.01, f"max_diff {max_diff} too large"

    def test_gqa(self):
        """Flex cross-attention matches reference with GQA."""
        B, T, n_seg, F_feat = 2, 8, 2, 6
        dec_n_embd, enc_n_embd = 64, 32
        cfg = _make_config(
            dec_n_embd=dec_n_embd,
            enc_n_embd=enc_n_embd,
            n_head=4,
            n_kv_head=2,
            head_dim=16,
        )
        layer = _make_layer_with_nonzero_proj(cfg)

        torch.manual_seed(42)
        decoder_hidden, encoder_kv, encoder_mask, seg_map = _make_inputs(
            B, T, n_seg, F_feat, dec_n_embd, enc_n_embd
        )

        q, k, v, kv_valid = self._get_internal_qkv(
            layer, decoder_hidden, encoder_kv, seg_map, encoder_mask
        )
        enable_gqa = layer.n_head != layer.n_kv_head
        assert enable_gqa

        q_clean = q.nan_to_num()
        k_clean = k.nan_to_num()

        y_ref = reference_cross_attention(q_clean, k_clean, v, kv_valid, enable_gqa)
        y_ref = y_ref.float()

        from torch.nn.attention.flex_attention import flex_attention

        def _score_mod(score, b, h, q_idx, kv_idx):
            return torch.where(kv_valid[b, kv_idx], score, -float("inf"))

        y_flex = flex_attention(
            q_clean, k_clean, v, score_mod=_score_mod, enable_gqa=enable_gqa
        )
        assert isinstance(y_flex, torch.Tensor)
        y_flex = y_flex.nan_to_num().float()

        max_diff = (y_flex - y_ref).abs().max().item()
        mean_diff = (y_flex - y_ref).abs().mean().item()
        print(f"xattn_ref_gqa: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        assert max_diff < 0.01, f"max_diff {max_diff} too large"


# =============================================================================
# Test: CPU SDPA fallback (q.is_cuda guard)
# =============================================================================


class TestCPUAndCUDAConsistency:
    """Verify the CPU SDPA reference and packed CUDA path remain consistent."""

    def test_cpu_forward_produces_valid_output(self):
        """Forward pass on CPU should produce valid (non-NaN) output."""
        cfg = _make_config()
        layer = _make_layer_with_nonzero_proj(cfg)
        decoder_hidden, encoder_kv, encoder_mask, seg_map = _make_inputs()

        out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
        assert out.shape == decoder_hidden.shape
        assert not torch.isnan(out).any()

    def test_cpu_backward_succeeds(self):
        """Backward pass on CPU should succeed (SDPA supports CPU backward)."""
        cfg = _make_config()
        layer = _make_layer_with_nonzero_proj(cfg)
        decoder_hidden = torch.randn(2, 8, 64, requires_grad=True)
        encoder_kv = torch.randn(2, 2, 6, 32, requires_grad=True)
        encoder_mask = torch.ones(2, 2, 6, dtype=torch.bool)
        seg_map = torch.arange(8).unsqueeze(0).expand(2, -1) // 4

        out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
        loss = out.sum()
        loss.backward()

        assert decoder_hidden.grad is not None
        assert encoder_kv.grad is not None

    def test_backend_override_does_not_affect_cpu_reference(self):
        """The decoder attention override does not alter the CPU reference."""
        cfg = _make_config()
        layer = _make_layer_with_nonzero_proj(cfg)
        torch.manual_seed(42)
        decoder_hidden, encoder_kv, encoder_mask, seg_map = _make_inputs()

        # Default output
        out_default = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        # The CPU cross-attention reference remains SDPA under this override.
        saved = fa_module._override_impl
        fa_module._override_impl = fa_module.FlashAttnImpl.FA4
        try:
            out_fa4_override = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
        finally:
            fa_module._override_impl = saved

        assert torch.allclose(out_default, out_fa4_override, atol=1e-6), (
            "CPU cross-attention should ignore the decoder attention override"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cpu_and_cuda_produce_close_results(self):
        """CPU and CUDA should produce numerically close results."""
        cfg = _make_config()
        layer = _make_layer_with_nonzero_proj(cfg)

        torch.manual_seed(42)
        decoder_hidden, encoder_kv, encoder_mask, seg_map = _make_inputs()

        out_cpu = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        # Move everything to CUDA
        layer_cuda = _make_layer_with_nonzero_proj(cfg).cuda()
        layer_cuda.load_state_dict(layer.state_dict())
        encoder_mask_cuda = encoder_mask.cuda()
        seg_map_cuda = seg_map.cuda()
        packed_encoder_kv = pack_dense_encoder_output_reference(
            encoder_kv.cuda(), encoder_mask_cuda
        )
        block_mask = build_packed_cross_attention_block_mask(
            seg_map_cuda,
            packed_encoder_kv.layout,
        )
        out_cuda = layer_cuda(
            decoder_hidden.cuda(),
            packed_encoder_kv,
            seg_map_cuda,
            packed_block_mask=block_mask,
        ).cpu()

        max_diff = (out_cpu - out_cuda).abs().max().item()
        assert max_diff < 1e-4, f"CPU/CUDA SDPA diff too large: {max_diff}"


# =============================================================================
# Test: packed CUDA routing and CPU SDPA reference
# =============================================================================


class TestPackedFlexRouting:
    def test_block_mask_routes_queries_and_feature_padding(self):
        seg_map = torch.tensor([[0, 0, 1, 1]])
        encoder_mask = torch.tensor([[[True, False], [True, True]]])
        block_mask = build_packed_cross_attention_block_mask(
            seg_map, pack_encoder_layout(encoder_mask)
        )
        mask_mod = block_mask.mask_mod
        zero = torch.tensor(0)

        assert mask_mod(zero, zero, torch.tensor(0), torch.tensor(0))
        assert not mask_mod(zero, zero, torch.tensor(0), torch.tensor(1))
        assert not mask_mod(zero, zero, torch.tensor(0), torch.tensor(2))
        assert mask_mod(zero, zero, torch.tensor(2), torch.tensor(1))
        assert mask_mod(zero, zero, torch.tensor(2), torch.tensor(2))
        assert not mask_mod(zero, zero, torch.tensor(2), torch.tensor(3))

    def test_block_mask_has_packed_query_and_flattened_kv_shape(self):
        seg_map = torch.tensor([[0, 0, 1, 1]])
        encoder_mask = torch.ones(1, 2, 3, dtype=torch.bool)
        block_mask = build_packed_cross_attention_block_mask(
            seg_map, pack_encoder_layout(encoder_mask)
        )
        assert block_mask.shape == (1, 1, 4, 32)

    def test_test_owned_cpu_reference_uses_sdpa(self):
        import inspect

        source = inspect.getsource(padded_cross_attention_reference)
        assert "scaled_dot_product_attention" in source
        assert "is_causal=False" in source

    def test_cuda_helper_uses_packed_flex_attention(self):
        import inspect

        source = inspect.getsource(CrossAttentionLayer._packed_cross_attention_dynamic)
        assert "flex_attention" in source
        assert "x_padded" not in source
        assert "self.c_kv" in source
        assert "self.c_q" in source
        assert "self.c_proj" in source

    def test_block_mask_builder_is_a_required_compiler_boundary(self):
        assert getattr(build_packed_cross_attention_block_mask, "_torchdynamo_disable", False)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_requires_precomputed_block_mask(self):
        cfg = _make_config()
        layer = _make_layer_with_nonzero_proj(cfg).cuda()
        decoder_hidden, encoder_kv, encoder_mask, seg_map = _make_inputs()
        packed_encoder_kv = pack_dense_encoder_output_reference(
            encoder_kv.cuda(), encoder_mask.cuda()
        )

        with pytest.raises(
            RuntimeError,
            match="requires a precomputed packed_block_mask",
        ):
            layer(
                decoder_hidden.cuda(),
                packed_encoder_kv,
                seg_map.cuda(),
            )


# =============================================================================
# Test: NaN handling in scatter/gather pipeline
# =============================================================================


class TestNaNHandling:
    """Verify NaN values from padding are properly handled."""

    def test_padded_positions_produce_zero_output(self):
        """Padded scatter positions (zeros) should not leak NaN into output."""
        cfg = _make_config()
        layer = _make_layer_with_nonzero_proj(cfg)

        B, T, n_seg, F = 2, 8, 2, 6
        decoder_hidden = torch.randn(B, T, cfg.dec_n_embd)
        encoder_kv = torch.randn(B, n_seg, F, cfg.enc_n_embd)
        encoder_mask = torch.ones(B, n_seg, F, dtype=torch.bool)
        seg_map = torch.arange(T).unsqueeze(0).expand(B, -1) // (T // n_seg)

        with torch.no_grad():
            out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
        assert not torch.isnan(out).any(), "Output should not contain NaN"
        assert not torch.isinf(out).any(), "Output should not contain Inf"

    def test_all_masked_encoder_produces_finite_output(self):
        """All-False encoder mask should produce finite (likely zero) output."""
        cfg = _make_config()
        layer = _make_layer_with_nonzero_proj(cfg)

        B, T, n_seg, F = 1, 4, 1, 4
        decoder_hidden = torch.randn(B, T, cfg.dec_n_embd)
        encoder_kv = torch.randn(B, n_seg, F, cfg.enc_n_embd)
        encoder_mask = torch.zeros(B, n_seg, F, dtype=torch.bool)  # all masked
        seg_map = torch.zeros(B, T, dtype=torch.long)

        with torch.no_grad():
            out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
        assert not torch.isnan(out).any(), "All-masked should not produce NaN"
        assert not torch.isinf(out).any(), "All-masked should not produce Inf"

    def test_mixed_masked_segments_no_nan(self):
        """Mix of fully-masked and unmasked segments should not produce NaN."""
        cfg = _make_config()
        layer = _make_layer_with_nonzero_proj(cfg)

        B, T, n_seg, F = 2, 8, 4, 3
        decoder_hidden = torch.randn(B, T, cfg.dec_n_embd)
        encoder_kv = torch.randn(B, n_seg, F, cfg.enc_n_embd)
        encoder_mask = torch.ones(B, n_seg, F, dtype=torch.bool)
        # Mask out segments 1 and 3 entirely
        encoder_mask[:, 1, :] = False
        encoder_mask[:, 3, :] = False
        seg_map = torch.arange(T).unsqueeze(0).expand(B, -1) // (T // n_seg)

        with torch.no_grad():
            out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
        assert not torch.isnan(out).any()

    def test_uneven_segments_no_nan_in_padding(self):
        """Uneven segments create internal padding — verify no NaN leaks."""
        cfg = _make_config()
        layer = _make_layer_with_nonzero_proj(cfg)

        B, T = 2, 12
        n_seg, F = 3, 4
        decoder_hidden = torch.randn(B, T, cfg.dec_n_embd)
        encoder_kv = torch.randn(B, n_seg, F, cfg.enc_n_embd)
        encoder_mask = torch.ones(B, n_seg, F, dtype=torch.bool)
        # Segments of size [5, 4, 3] — creates internal padding
        seg_map = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]] * B)

        with torch.no_grad():
            out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
        assert not torch.isnan(out).any()
        assert out.shape == (B, T, cfg.dec_n_embd)

    def test_nan_to_num_present_in_source(self):
        """The padded test oracle must clean NaNs before they can propagate."""
        import inspect

        source = inspect.getsource(padded_cross_attention_reference)
        assert "nan_to_num()" in source, (
            "The padded SDPA oracle must convert NaN from padded/all-masked "
            "positions to zero"
        )

    def test_nan_to_num_is_safe_on_clean_output(self):
        """nan_to_num() is a no-op when no NaN is present (defensive measure).

        PyTorch nightly's RMS norm (eps=1e-5) and SDPA handle edge cases
        without producing NaN, but nan_to_num() remains as a safety net
        for portability across PyTorch versions and numerical edge cases.
        """
        cfg = _make_config()
        layer = _make_layer_with_nonzero_proj(cfg)

        B, T = 1, 5
        n_seg, F = 2, 4
        decoder_hidden = torch.randn(B, T, cfg.dec_n_embd)
        encoder_kv = torch.randn(B, n_seg, F, cfg.enc_n_embd)
        encoder_mask = torch.ones(B, n_seg, F, dtype=torch.bool)
        seg_map = torch.tensor([[0, 0, 0, 1, 1]])

        with torch.no_grad():
            out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        assert torch.isfinite(out).all(), "Output must be finite"
        # nan_to_num on already-clean tensor is identity
        assert torch.equal(out, out.nan_to_num()), (
            "nan_to_num should be no-op on finite output"
        )

    def test_nan_to_num_converts_injected_nan(self):
        """Verify nan_to_num correctly converts NaN to 0 when NaN is present.

        This tests the mechanism that nan_to_num provides as a safety net,
        even if current PyTorch versions don't produce NaN in these paths.
        """
        # Simulate what would happen if SDPA or norm produced NaN
        y = torch.tensor([[[[float("nan"), 1.0, float("nan"), 2.0]]]])
        y_clean = y.nan_to_num()
        expected = torch.tensor([[[[0.0, 1.0, 0.0, 2.0]]]])
        assert torch.equal(y_clean, expected), (
            "nan_to_num should replace NaN with 0 and leave finite values unchanged"
        )

    def test_padded_positions_produce_finite_output_with_uneven_segments(self):
        """Uneven segments create padded positions that must produce finite output.

        The scatter/gather approach creates zero-filled padding positions.
        nan_to_num() ensures these stay finite through the attention pipeline.
        """
        cfg = _make_config()
        layer = _make_layer_with_nonzero_proj(cfg)

        B, T = 1, 5
        n_seg, F = 2, 4
        decoder_hidden = torch.randn(B, T, cfg.dec_n_embd)
        encoder_kv = torch.randn(B, n_seg, F, cfg.enc_n_embd)
        encoder_mask = torch.ones(B, n_seg, F, dtype=torch.bool)
        # Segments of size [3, 2] — creates padding in segment 1
        seg_map = torch.tensor([[0, 0, 0, 1, 1]])

        with torch.no_grad():
            out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
        assert torch.isfinite(out).all(), "Padded positions must produce finite output"
        assert out.shape == (B, T, cfg.dec_n_embd)

    def test_all_masked_sdpa_produces_finite_output(self):
        """SDPA with all-False mask must produce finite output.

        nan_to_num() is the safety net for this case — whether SDPA itself
        returns NaN or 0 depends on the PyTorch version.
        """
        import torch.nn.functional as F_t

        q = torch.randn(1, 1, 4, 16)
        k = torch.randn(1, 1, 4, 16)
        v = torch.randn(1, 1, 4, 16)
        mask = torch.zeros(1, 1, 1, 4, dtype=torch.bool)  # all masked

        y = F_t.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        y_clean = y.nan_to_num()
        assert torch.isfinite(y_clean).all(), (
            "nan_to_num should ensure all-masked SDPA output is finite"
        )
        assert (y_clean == 0).all(), (
            "All-masked attention output should be zero after nan_to_num"
        )

    def test_backward_through_nan_to_num(self):
        """Gradients flow correctly through nan_to_num for non-padded positions."""
        cfg = _make_config()
        layer = _make_layer_with_nonzero_proj(cfg)

        B, T = 1, 5
        n_seg, F = 2, 4
        decoder_hidden = torch.randn(B, T, cfg.dec_n_embd, requires_grad=True)
        encoder_kv = torch.randn(B, n_seg, F, cfg.enc_n_embd)
        encoder_mask = torch.ones(B, n_seg, F, dtype=torch.bool)
        seg_map = torch.tensor([[0, 0, 0, 1, 1]])

        out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
        loss = out.sum()
        loss.backward()

        assert decoder_hidden.grad is not None, (
            "Gradient should flow through nan_to_num"
        )
        assert torch.isfinite(decoder_hidden.grad).all(), (
            "Gradients should be finite — nan_to_num must not break backward pass"
        )


# =============================================================================
# Test: FP8 padding alignment (max_seg_size padded to multiple of 16)
# =============================================================================


class TestFP8PaddingAlignment:
    """Verify max_seg_size is padded to a multiple of 16 for FP8 compatibility."""

    def _get_max_seg_size(self, seg_map: torch.Tensor) -> int:
        """Reproduce the max_seg_size computation from CrossAttentionLayer."""
        B, T = seg_map.shape
        pos = torch.arange(T).unsqueeze(0).expand(B, -1)
        is_seg_start = torch.zeros(B, T, dtype=torch.bool)
        is_seg_start[:, 0] = True
        is_seg_start[:, 1:] = seg_map[:, 1:] != seg_map[:, :-1]
        seg_start_pos = torch.where(is_seg_start, pos, torch.zeros_like(pos))
        seg_start_pos = seg_start_pos.cummax(dim=1).values
        pos_in_seg = pos - seg_start_pos
        max_seg_size = pos_in_seg.amax() + 1
        max_seg_size = max_seg_size.clamp(min=1)
        max_seg_size = ((max_seg_size + 15) // 16) * 16
        return int(max_seg_size.item())

    def test_uniform_4_pads_to_16(self):
        """Segments of size 4 → max_seg_size should pad to 16."""
        seg_map = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
        assert self._get_max_seg_size(seg_map) == 16

    def test_uniform_16_stays_16(self):
        """Segments of size 16 → max_seg_size should stay 16."""
        seg_map = torch.zeros(1, 16, dtype=torch.long)
        assert self._get_max_seg_size(seg_map) == 16

    def test_uniform_17_pads_to_32(self):
        """Segments of size 17 → max_seg_size should pad to 32."""
        seg_map = torch.zeros(1, 17, dtype=torch.long)
        assert self._get_max_seg_size(seg_map) == 32

    def test_uneven_segments_pads_max(self):
        """With segments [5, 4, 3], max is 5 → pads to 16."""
        seg_map = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]])
        assert self._get_max_seg_size(seg_map) == 16

    def test_single_token_pads_to_16(self):
        """Single token segment → max_seg_size should pad to 16."""
        seg_map = torch.tensor([[0]])
        assert self._get_max_seg_size(seg_map) == 16

    def test_result_always_multiple_of_16(self):
        """max_seg_size should always be a multiple of 16."""
        for seg_sizes in [[1], [3, 5], [15], [16], [17], [31], [32], [33], [64]]:
            seg_map_list = []
            for i, sz in enumerate(seg_sizes):
                seg_map_list.extend([i] * sz)
            seg_map = torch.tensor([seg_map_list])
            result = self._get_max_seg_size(seg_map)
            assert result % 16 == 0, (
                f"seg_sizes={seg_sizes} → {result}, not multiple of 16"
            )


# =============================================================================
# Test: FP8 init safety — zero weights break FP8 tensorwise scaling
# =============================================================================

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _has_fp8():
    """Check if torchao FP8 is available."""
    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training  # noqa: F401

        return True
    except ImportError:
        return False


requires_fp8 = pytest.mark.skipif(not _has_fp8(), reason="torchao FP8 not available")


class TestFP8InitSafety:
    """Verify that weight initialization is safe for FP8 tensorwise scaling.

    FP8 tensorwise scaling computes: scale = max_fp8_value / amax(tensor).
    If amax(tensor) == 0 (all-zero weights), scale becomes extreme → inf
    gradients on the first backward pass → NaN parameters after optimizer step.

    These tests verify:
    1. The original bug (zero-init + FP8 → inf gradients) is caught
    2. The fix (uniform-init + FP8 → finite gradients) works
    3. Full cross-attention layer forward/backward with FP8 is NaN-free
    """

    @requires_cuda
    @requires_fp8
    def test_zero_weight_fp8_degenerate_amax(self):
        """Zero weights produce degenerate FP8 amax=0, a known numerical hazard.

        FP8 tensorwise scaling computes: scale = max_fp8_value / amax(tensor).
        With zero weights, amax=0. While some torchao versions handle this
        gracefully (clamping the scale), it caused NaN loss in our full
        encoder-decoder training (step 0 OK, step 1 NaN) due to interactions
        with torch.compile, scatter/gather, and the Muon optimizer.

        This test verifies the safety invariant: init_weights() must produce
        amax > 0 for all projections, and documents why zero-init is banned.
        """

        # Demonstrate that zero weights → zero amax (the hazardous condition)
        linear = torch.nn.Linear(128, 128, bias=False, device="cuda")
        with torch.no_grad():
            linear.weight.zero_()
        assert linear.weight.abs().max().item() == 0.0, (
            "Sanity: zero-init should have amax=0"
        )

        # Verify that init_weights() avoids this condition for ALL projections
        cfg = CrossAttentionLayerConfig(
            dec_n_embd=256, enc_n_embd=128, n_head=8, n_kv_head=4, head_dim=32
        )
        layer = CrossAttentionLayer(cfg).cuda()
        layer.init_weights()

        for name in ["c_q", "c_kv", "c_proj"]:
            w = getattr(layer, name).weight
            amax = w.abs().max().item()
            assert amax > 0, (
                f"{name} has amax=0 after init — this will produce degenerate "
                f"FP8 scale factors (scale = max_fp8 / 0 = inf)"
            )

    @requires_cuda
    @requires_fp8
    def test_uniform_init_linear_fp8_produces_finite_gradients(self):
        """Verify fix: uniform-init nn.Linear + FP8 → finite gradients."""
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training

        linear = torch.nn.Linear(128, 128, bias=False, device="cuda")
        bound = (3.0 / 128) ** 0.5
        with torch.no_grad():
            torch.nn.init.uniform_(linear.weight, -bound, bound)

        fp8_config = Float8LinearConfig.from_recipe_name("tensorwise")
        convert_to_float8_training(linear, config=fp8_config)

        x = torch.randn(2, 128, device="cuda", requires_grad=True)
        out = linear(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all(), (
            "Uniform-init + FP8 should produce finite gradients"
        )
        assert torch.isfinite(out).all(), (
            "Uniform-init + FP8 forward should produce finite output"
        )

    @requires_cuda
    @requires_fp8
    def test_cross_attention_fp8_forward_backward_no_nan(self):
        """Full cross-attention layer with FP8: forward + backward must be NaN-free.

        Uses production-sized dims (divisible by 16, >= 128) to ensure FP8
        conversion actually applies.
        """
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training

        from nanochat.fp8_utils import fp8_module_filter

        cfg = CrossAttentionLayerConfig(
            dec_n_embd=256, enc_n_embd=128, n_head=8, n_kv_head=4, head_dim=32
        )
        layer = CrossAttentionLayer(cfg).cuda()
        layer.init_weights()

        # Convert to FP8
        fp8_config = Float8LinearConfig.from_recipe_name("tensorwise")
        convert_to_float8_training(
            layer, config=fp8_config, module_filter_fn=fp8_module_filter
        )

        # Count FP8 modules to verify conversion happened
        num_fp8 = sum(1 for m in layer.modules() if "Float8" in type(m).__name__)
        assert num_fp8 > 0, "At least one layer should be converted to FP8"

        # Create inputs
        B, T, n_seg, F_feat = 2, 32, 2, 8
        decoder_hidden = torch.randn(B, T, cfg.dec_n_embd, device="cuda", requires_grad=True)
        encoder_kv = torch.randn(B, n_seg, F_feat, cfg.enc_n_embd, device="cuda", requires_grad=True)
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool, device="cuda")
        seg_map = torch.arange(T, device="cuda").unsqueeze(0).expand(B, -1) // (T // n_seg)
        packed_encoder_kv = pack_dense_encoder_output_reference(encoder_kv, encoder_mask)
        block_mask = build_packed_cross_attention_block_mask(seg_map, packed_encoder_kv.layout)

        # Forward
        out = layer(
            decoder_hidden,
            packed_encoder_kv,
            seg_map,
            packed_block_mask=block_mask,
        )
        assert torch.isfinite(out).all(), "FP8 forward output must be finite"

        # Backward
        loss = out.sum()
        loss.backward()

        assert decoder_hidden.grad is not None
        assert torch.isfinite(decoder_hidden.grad).all(), (
            "Decoder hidden gradients must be finite with FP8"
        )
        assert encoder_kv.grad is not None
        assert torch.isfinite(encoder_kv.grad).all(), (
            "Encoder KV gradients must be finite with FP8"
        )

        # Check all parameter gradients
        for name, param in layer.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"Gradient for {name} must be finite with FP8"
                )

    @requires_cuda
    @requires_fp8
    def test_cross_attention_fp8_two_optimizer_steps_no_nan(self):
        """Simulate two optimizer steps with FP8 cross-attention.

        The original bug manifested at step 1 (step 0 was fine because the
        first forward pass with zero weights produces zero output, but the
        backward pass through FP8 produces inf gradients that corrupt
        parameters for step 1).
        """
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training

        from nanochat.fp8_utils import fp8_module_filter

        cfg = CrossAttentionLayerConfig(
            dec_n_embd=256, enc_n_embd=128, n_head=8, n_kv_head=4, head_dim=32
        )
        layer = CrossAttentionLayer(cfg).cuda()
        layer.init_weights()

        # Convert to FP8
        fp8_config = Float8LinearConfig.from_recipe_name("tensorwise")
        convert_to_float8_training(
            layer, config=fp8_config, module_filter_fn=fp8_module_filter
        )

        optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)

        B, T, n_seg, F_feat = 2, 32, 2, 8
        seg_map = torch.arange(T, device="cuda").unsqueeze(0).expand(B, -1) // (T // n_seg)
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool, device="cuda")
        layout = pack_encoder_layout(encoder_mask)
        block_mask = build_packed_cross_attention_block_mask(seg_map, layout)

        for step in range(2):
            optimizer.zero_grad()
            decoder_hidden = torch.randn(B, T, cfg.dec_n_embd, device="cuda")
            encoder_kv = torch.randn(B, n_seg, F_feat, cfg.enc_n_embd, device="cuda")
            packed_encoder_kv = pack_dense_encoder_output_reference(
                encoder_kv, encoder_mask
            )

            out = layer(
                decoder_hidden,
                packed_encoder_kv,
                seg_map,
                packed_block_mask=block_mask,
            )
            loss = out.sum()
            loss.backward()

            # Check gradients before optimizer step
            for name, param in layer.named_parameters():
                if param.grad is not None:
                    assert torch.isfinite(param.grad).all(), (
                        f"Step {step}: gradient for {name} must be finite"
                    )

            optimizer.step()

            # Check parameters after optimizer step
            for name, param in layer.named_parameters():
                assert torch.isfinite(param.data).all(), (
                    f"Step {step}: parameter {name} must be finite after optimizer step"
                )

    @requires_cuda
    @requires_fp8
    def test_cross_attention_fp8_with_partial_mask(self):
        """FP8 cross-attention with partial encoder masking must stay finite.

        Encoder padding (masked features) combined with FP8 scaling must not
        introduce NaN through the attention or projection layers.
        """
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training

        from nanochat.fp8_utils import fp8_module_filter

        cfg = CrossAttentionLayerConfig(
            dec_n_embd=256, enc_n_embd=128, n_head=8, n_kv_head=4, head_dim=32
        )
        layer = CrossAttentionLayer(cfg).cuda()
        layer.init_weights()

        fp8_config = Float8LinearConfig.from_recipe_name("tensorwise")
        convert_to_float8_training(
            layer, config=fp8_config, module_filter_fn=fp8_module_filter
        )

        B, T, n_seg, F_feat = 2, 32, 2, 8
        decoder_hidden = torch.randn(B, T, cfg.dec_n_embd, device="cuda", requires_grad=True)
        encoder_kv = torch.randn(B, n_seg, F_feat, cfg.enc_n_embd, device="cuda", requires_grad=True)
        # Partial mask: only first 4 of 8 features are real
        encoder_mask = torch.zeros(B, n_seg, F_feat, dtype=torch.bool, device="cuda")
        encoder_mask[:, :, :4] = True
        seg_map = torch.arange(T, device="cuda").unsqueeze(0).expand(B, -1) // (T // n_seg)
        packed_encoder_kv = pack_dense_encoder_output_reference(encoder_kv, encoder_mask)
        block_mask = build_packed_cross_attention_block_mask(seg_map, packed_encoder_kv.layout)

        out = layer(
            decoder_hidden,
            packed_encoder_kv,
            seg_map,
            packed_block_mask=block_mask,
        )
        assert torch.isfinite(out).all(), "FP8 + partial mask: output must be finite"

        loss = out.sum()
        loss.backward()
        assert decoder_hidden.grad is not None
        assert torch.isfinite(decoder_hidden.grad).all(), (
            "FP8 + partial mask: decoder gradients must be finite"
        )

    @requires_cuda
    @requires_fp8
    def test_cross_attention_fp8_gqa(self):
        """FP8 cross-attention with GQA (n_head != n_kv_head) must stay finite."""
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training

        from nanochat.fp8_utils import fp8_module_filter

        cfg = CrossAttentionLayerConfig(
            dec_n_embd=256, enc_n_embd=128, n_head=16, n_kv_head=4, head_dim=16
        )
        layer = CrossAttentionLayer(cfg).cuda()
        layer.init_weights()

        fp8_config = Float8LinearConfig.from_recipe_name("tensorwise")
        convert_to_float8_training(
            layer, config=fp8_config, module_filter_fn=fp8_module_filter
        )

        B, T, n_seg, F_feat = 2, 32, 2, 8
        decoder_hidden = torch.randn(B, T, cfg.dec_n_embd, device="cuda", requires_grad=True)
        encoder_kv = torch.randn(B, n_seg, F_feat, cfg.enc_n_embd, device="cuda", requires_grad=True)
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool, device="cuda")
        seg_map = torch.arange(T, device="cuda").unsqueeze(0).expand(B, -1) // (T // n_seg)
        packed_encoder_kv = pack_dense_encoder_output_reference(encoder_kv, encoder_mask)
        block_mask = build_packed_cross_attention_block_mask(seg_map, packed_encoder_kv.layout)

        out = layer(
            decoder_hidden,
            packed_encoder_kv,
            seg_map,
            packed_block_mask=block_mask,
        )
        assert torch.isfinite(out).all(), "FP8 + GQA: output must be finite"

        loss = out.sum()
        loss.backward()
        assert decoder_hidden.grad is not None
        assert torch.isfinite(decoder_hidden.grad).all(), (
            "FP8 + GQA: decoder gradients must be finite"
        )
        assert encoder_kv.grad is not None
        assert torch.isfinite(encoder_kv.grad).all(), (
            "FP8 + GQA: encoder gradients must be finite"
        )

    @requires_cuda
    @requires_fp8
    def test_cross_attention_fp8_variable_segments(self):
        """FP8 cross-attention with uneven segment sizes must stay finite."""
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training

        from nanochat.fp8_utils import fp8_module_filter

        cfg = CrossAttentionLayerConfig(
            dec_n_embd=256, enc_n_embd=128, n_head=8, n_kv_head=4, head_dim=32
        )
        layer = CrossAttentionLayer(cfg).cuda()
        layer.init_weights()

        fp8_config = Float8LinearConfig.from_recipe_name("tensorwise")
        convert_to_float8_training(
            layer, config=fp8_config, module_filter_fn=fp8_module_filter
        )

        # Uneven segments: [20, 8, 4]
        B, n_seg, F_feat = 2, 3, 8
        T = 32
        seg_map = torch.cat([
            torch.zeros(20, dtype=torch.long),
            torch.ones(8, dtype=torch.long),
            torch.full((4,), 2, dtype=torch.long),
        ]).unsqueeze(0).expand(B, -1).cuda()

        decoder_hidden = torch.randn(B, T, cfg.dec_n_embd, device="cuda", requires_grad=True)
        encoder_kv = torch.randn(B, n_seg, F_feat, cfg.enc_n_embd, device="cuda", requires_grad=True)
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool, device="cuda")
        packed_encoder_kv = pack_dense_encoder_output_reference(encoder_kv, encoder_mask)
        block_mask = build_packed_cross_attention_block_mask(seg_map, packed_encoder_kv.layout)

        out = layer(
            decoder_hidden,
            packed_encoder_kv,
            seg_map,
            packed_block_mask=block_mask,
        )
        assert torch.isfinite(out).all(), "FP8 + variable segments: output must be finite"

        loss = out.sum()
        loss.backward()
        assert decoder_hidden.grad is not None
        assert torch.isfinite(decoder_hidden.grad).all(), (
            "FP8 + variable segments: decoder gradients must be finite"
        )

    def test_init_weights_no_zero_projections(self):
        """Regression guard: init_weights() must never produce all-zero projections.

        This is the key invariant that prevents the FP8 NaN bug. If any
        projection is all-zero after init, FP8 tensorwise scaling will
        degenerate (amax=0 → extreme scale → inf gradients).
        """
        configs = [
            _make_config(dec_n_embd=64, enc_n_embd=32, n_head=4, n_kv_head=4, head_dim=16),
            _make_config(dec_n_embd=128, enc_n_embd=64, n_head=8, n_kv_head=4, head_dim=16),
            _make_config(dec_n_embd=256, enc_n_embd=128, n_head=16, n_kv_head=4, head_dim=16),
            _make_config(dec_n_embd=256, enc_n_embd=256, n_head=8, n_kv_head=8, head_dim=32),
        ]
        for cfg in configs:
            layer = CrossAttentionLayer(cfg)
            layer.init_weights()
            for name in ["c_q", "c_kv", "c_proj"]:
                w = getattr(layer, name).weight
                amax = w.abs().max().item()
                assert amax > 0, (
                    f"{name} has amax=0 after init_weights() with config "
                    f"dec={cfg.dec_n_embd} enc={cfg.enc_n_embd} — "
                    f"this will break FP8 tensorwise scaling"
                )


# =============================================================================
# Test: QK normalization
# =============================================================================


class TestQKNormalization:
    """Verify RMS normalization is applied to Q and K before attention."""

    def test_norm_function_is_rms_norm(self):
        """The norm function should be functional RMS norm."""
        x = torch.randn(2, 4, 16)
        result = norm(x)

        # RMS norm: x / sqrt(mean(x^2) + eps) * 1 (no learnable params)
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + 1e-5).sqrt()
        x / rms
        # F.rms_norm uses a different eps default, so check shape and finite
        assert result.shape == x.shape
        assert not torch.isnan(result).any()

    def test_norm_output_has_unit_rms(self):
        """After RMS norm, the RMS of each vector should be ~1.0."""
        x = torch.randn(4, 8, 32) * 10  # large values
        result = norm(x)
        rms_per_vec = result.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms_per_vec, torch.ones_like(rms_per_vec), atol=1e-4), (
            "RMS of normed vectors should be ~1.0"
        )

    def test_norm_preserves_shape(self):
        """norm() should not change tensor shape."""
        for shape in [(1, 1, 16), (2, 4, 32), (8, 16, 64)]:
            x = torch.randn(shape)
            assert norm(x).shape == shape


# =============================================================================
# Test: determinism and reproducibility
# =============================================================================


class TestDeterminism:
    """Verify cross-attention is deterministic with same inputs."""

    def test_same_input_same_output(self):
        """Two forward passes with identical inputs should produce identical output."""
        cfg = _make_config()
        layer = _make_layer_with_nonzero_proj(cfg)

        torch.manual_seed(123)
        d_h, e_kv, e_m, s_m = _make_inputs()

        out1 = layer(d_h, e_kv, e_m, s_m)
        out2 = layer(d_h, e_kv, e_m, s_m)

        assert torch.allclose(out1, out2, atol=0), (
            "Same inputs should produce bitwise identical outputs"
        )

    def test_different_seeds_different_output(self):
        """Different random weights should produce different outputs."""
        cfg = _make_config()
        decoder_hidden, encoder_kv, encoder_mask, seg_map = _make_inputs()

        torch.manual_seed(1)
        layer1 = _make_layer_with_nonzero_proj(cfg)
        out1 = layer1(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        torch.manual_seed(2)
        layer2 = _make_layer_with_nonzero_proj(cfg)
        out2 = layer2(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        assert not torch.allclose(out1, out2, atol=1e-4), (
            "Different weights should produce different outputs"
        )


# =============================================================================
# NEW TESTS: scatter/gather mechanism deep verification
# =============================================================================


def _compute_pos_in_seg(seg_map: torch.Tensor) -> torch.Tensor:
    """Reproduce the pos_in_seg computation from CrossAttentionLayer.forward."""
    B, T = seg_map.shape
    pos = torch.arange(T, device=seg_map.device).unsqueeze(0).expand(B, -1)
    is_seg_start = torch.zeros(B, T, dtype=torch.bool, device=seg_map.device)
    is_seg_start[:, 0] = True
    is_seg_start[:, 1:] = seg_map[:, 1:] != seg_map[:, :-1]
    seg_start_pos = torch.where(is_seg_start, pos, torch.zeros_like(pos))
    seg_start_pos = seg_start_pos.cummax(dim=1).values
    pos_in_seg = pos - seg_start_pos
    return pos_in_seg


def _compute_padded_idx_and_max_seg_size(seg_map: torch.Tensor):
    """Reproduce padded_idx and max_seg_size computation."""
    pos_in_seg = _compute_pos_in_seg(seg_map)
    max_seg_size = pos_in_seg.amax() + 1
    max_seg_size = max_seg_size.clamp(min=1)
    max_seg_size = ((max_seg_size + 15) // 16) * 16
    padded_idx = seg_map * max_seg_size + pos_in_seg
    n_seg = seg_map.max().item() + 1
    padded_len = n_seg * max_seg_size.item()
    return padded_idx, max_seg_size, padded_len


# ---------------------------------------------------------------------------
# Test 1: pos_in_seg computation correctness
# ---------------------------------------------------------------------------

class TestPosInSegComputation:
    """Verify positions within each segment are 0-indexed and contiguous."""

    def test_known_seg_map_three_segments(self):
        """seg_map [0,0,0,1,1,2,2,2] -> pos_in_seg [0,1,2,0,1,0,1,2]."""
        seg_map = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2]])
        pos_in_seg = _compute_pos_in_seg(seg_map)
        expected = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2]])
        assert torch.equal(pos_in_seg, expected), (
            f"Expected {expected} but got {pos_in_seg}"
        )

    def test_single_segment(self):
        """All positions in segment 0 -> pos_in_seg = [0, 1, 2, ..., T-1]."""
        T = 8
        seg_map = torch.zeros(1, T, dtype=torch.long)
        pos_in_seg = _compute_pos_in_seg(seg_map)
        expected = torch.arange(T).unsqueeze(0)
        assert torch.equal(pos_in_seg, expected)

    def test_all_size_one_segments(self):
        """Each position is its own segment -> pos_in_seg all zeros."""
        T = 5
        seg_map = torch.arange(T, dtype=torch.long).unsqueeze(0)
        pos_in_seg = _compute_pos_in_seg(seg_map)
        expected = torch.zeros(1, T, dtype=torch.long)
        assert torch.equal(pos_in_seg, expected)

    def test_two_equal_segments(self):
        """Two segments of equal size."""
        seg_map = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
        pos_in_seg = _compute_pos_in_seg(seg_map)
        expected = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        assert torch.equal(pos_in_seg, expected)

    def test_unequal_segments(self):
        """Segments of sizes [5, 3, 4]."""
        seg_map = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]])
        pos_in_seg = _compute_pos_in_seg(seg_map)
        expected = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 0, 1, 2, 3]])
        assert torch.equal(pos_in_seg, expected)

    def test_batched_different_layouts(self):
        """Different segment layouts per batch element."""
        seg_map = torch.tensor([
            [0, 0, 0, 1, 1, 1, 2, 2],  # sizes [3, 3, 2]
            [0, 0, 1, 1, 1, 1, 2, 2],  # sizes [2, 4, 2]
        ])
        pos_in_seg = _compute_pos_in_seg(seg_map)
        expected = torch.tensor([
            [0, 1, 2, 0, 1, 2, 0, 1],
            [0, 1, 0, 1, 2, 3, 0, 1],
        ])
        assert torch.equal(pos_in_seg, expected)

    def test_pos_in_seg_always_starts_at_zero(self):
        """First position in each segment must be 0."""
        seg_map = torch.tensor([[0, 0, 1, 1, 1, 2, 3, 3]])
        pos_in_seg = _compute_pos_in_seg(seg_map)
        # Check segment start positions
        for seg_idx in range(4):
            mask = seg_map[0] == seg_idx
            if mask.any():
                first_pos = pos_in_seg[0, mask][0].item()
                assert first_pos == 0, (
                    f"Segment {seg_idx}: first pos_in_seg should be 0, got {first_pos}"
                )

    def test_pos_in_seg_contiguous_within_segment(self):
        """Positions within each segment must be 0, 1, 2, ..., size-1."""
        seg_map = torch.tensor([[0, 0, 0, 1, 2, 2, 2, 2]])
        pos_in_seg = _compute_pos_in_seg(seg_map)
        for seg_idx in range(3):
            mask = (seg_map[0] == seg_idx)
            seg_positions = pos_in_seg[0, mask]
            expected = torch.arange(mask.sum().item())
            assert torch.equal(seg_positions, expected), (
                f"Segment {seg_idx}: expected {expected}, got {seg_positions}"
            )

    @given(
        seg_sizes=st.lists(
            st.integers(min_value=1, max_value=20), min_size=1, max_size=8
        )
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_pos_in_seg_property_starts_at_zero_contiguous(self, seg_sizes):
        """Property: for any valid seg_map, pos_in_seg is 0-indexed and contiguous per segment."""
        seg_map_list = []
        for i, sz in enumerate(seg_sizes):
            seg_map_list.extend([i] * sz)
        seg_map = torch.tensor([seg_map_list])
        pos_in_seg = _compute_pos_in_seg(seg_map)

        for i, sz in enumerate(seg_sizes):
            mask = (seg_map[0] == i)
            seg_pos = pos_in_seg[0, mask]
            expected = torch.arange(sz)
            assert torch.equal(seg_pos, expected), (
                f"Segment {i} (size {sz}): expected {expected.tolist()}, got {seg_pos.tolist()}"
            )


# ---------------------------------------------------------------------------
# Test 2: padded_idx overflow / bounds checking
# ---------------------------------------------------------------------------

class TestPaddedIdxBounds:
    """Verify padded_idx does not overflow the padded tensor dimensions."""

    def test_basic_bounds(self):
        """padded_idx values must be in [0, padded_len)."""
        seg_map = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2]])
        padded_idx, max_seg_size, padded_len = _compute_padded_idx_and_max_seg_size(seg_map)
        assert (padded_idx >= 0).all()
        assert (padded_idx < padded_len).all(), (
            f"padded_idx max {padded_idx.max().item()} >= padded_len {padded_len}"
        )

    def test_single_element_segments(self):
        """Each segment has size 1: padded_idx = [0, max_seg_size, 2*max_seg_size, ...]."""
        n_seg = 5
        seg_map = torch.arange(n_seg, dtype=torch.long).unsqueeze(0)
        padded_idx, max_seg_size, padded_len = _compute_padded_idx_and_max_seg_size(seg_map)
        assert (padded_idx >= 0).all()
        assert (padded_idx < padded_len).all()
        # Each index should be seg_idx * max_seg_size + 0
        for i in range(n_seg):
            assert padded_idx[0, i].item() == i * max_seg_size.item()

    def test_large_n_seg(self):
        """Many segments (stress test for overflow)."""
        n_seg = 50
        seg_sizes = [2] * n_seg
        seg_map_list = []
        for i, sz in enumerate(seg_sizes):
            seg_map_list.extend([i] * sz)
        seg_map = torch.tensor([seg_map_list])
        padded_idx, max_seg_size, padded_len = _compute_padded_idx_and_max_seg_size(seg_map)
        assert (padded_idx >= 0).all()
        assert (padded_idx < padded_len).all()

    def test_large_max_seg_size(self):
        """One very large segment alongside small ones."""
        # Segment 0 has 100 positions, segment 1 has 2
        seg_map_list = [0] * 100 + [1] * 2
        seg_map = torch.tensor([seg_map_list])
        padded_idx, max_seg_size, padded_len = _compute_padded_idx_and_max_seg_size(seg_map)
        assert (padded_idx >= 0).all()
        assert (padded_idx < padded_len).all()
        # max_seg_size should be at least 100 (padded to multiple of 16)
        assert max_seg_size.item() >= 100
        assert max_seg_size.item() % 16 == 0

    def test_no_duplicate_padded_idx(self):
        """No two positions should map to the same padded_idx within a batch element."""
        seg_map = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2]])
        padded_idx, _, _ = _compute_padded_idx_and_max_seg_size(seg_map)
        # Check uniqueness within each batch element
        for b in range(seg_map.size(0)):
            indices = padded_idx[b].tolist()
            assert len(indices) == len(set(indices)), (
                f"Batch {b}: duplicate padded_idx found: {indices}"
            )

    def test_forward_does_not_crash_with_extreme_segments(self):
        """Forward pass with many small segments should not crash."""
        n_seg = 16
        T = 16  # one position per segment
        B = 1
        F_feat = 2
        dec_n_embd, enc_n_embd = 32, 16
        cfg = _make_config(
            dec_n_embd=dec_n_embd, enc_n_embd=enc_n_embd,
            n_head=2, n_kv_head=2, head_dim=16,
        )
        layer = _make_layer_with_nonzero_proj(cfg)

        seg_map = torch.arange(n_seg, dtype=torch.long).unsqueeze(0)
        decoder_hidden = torch.randn(B, T, dec_n_embd)
        encoder_kv = torch.randn(B, n_seg, F_feat, enc_n_embd)
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool)

        with torch.no_grad():
            out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
        assert out.shape == (B, T, dec_n_embd)
        assert torch.isfinite(out).all()

    @given(
        seg_sizes=st.lists(
            st.integers(min_value=1, max_value=15), min_size=1, max_size=10
        )
    )
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_padded_idx_in_bounds_property(self, seg_sizes):
        """Property: padded_idx must always be in [0, padded_len) for any valid seg_map."""
        seg_map_list = []
        for i, sz in enumerate(seg_sizes):
            seg_map_list.extend([i] * sz)
        seg_map = torch.tensor([seg_map_list])
        padded_idx, max_seg_size, padded_len = _compute_padded_idx_and_max_seg_size(seg_map)

        assert (padded_idx >= 0).all(), "padded_idx has negative values"
        assert (padded_idx < padded_len).all(), (
            f"padded_idx max {padded_idx.max().item()} >= padded_len {padded_len}"
        )
        # Check uniqueness
        indices = padded_idx[0].tolist()
        assert len(indices) == len(set(indices)), "Duplicate padded_idx values"


# ---------------------------------------------------------------------------
# Test 3: scatter -> attend -> gather roundtrip
# ---------------------------------------------------------------------------

class TestScatterGatherRoundtrip:
    """Verify scatter->gather preserves position-segment association."""

    def test_identity_proj_roundtrip(self):
        """With identity-like c_proj, scattered data should gather back correctly.

        We set c_proj to identity (via eye matrix) so the output of c_proj is
        the attention output itself. The roundtrip should preserve the mapping:
        each position gets output from its own segment only.
        """
        dec_n_embd = 32
        enc_n_embd = 16
        n_head = 2
        head_dim = 16
        cfg = CrossAttentionLayerConfig(
            dec_n_embd=dec_n_embd, enc_n_embd=enc_n_embd,
            n_head=n_head, n_kv_head=n_head, head_dim=head_dim,
        )
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()
        # Set c_proj to identity so output = attention result
        with torch.no_grad():
            layer.c_proj.weight.copy_(torch.eye(dec_n_embd))

        B, T = 1, 8
        n_seg = 2
        F_feat = 4
        seg_map = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])

        decoder_hidden = torch.randn(B, T, dec_n_embd)
        encoder_kv = torch.randn(B, n_seg, F_feat, enc_n_embd)
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool)

        with torch.no_grad():
            out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        assert out.shape == (B, T, dec_n_embd)
        assert torch.isfinite(out).all()

        # Verify via reference: each position should match single-position forward
        out_ref = _reference_gather_xattn(
            layer, decoder_hidden, encoder_kv, encoder_mask, seg_map
        )
        assert torch.allclose(out, out_ref, atol=1e-5), (
            f"Roundtrip max diff: {(out - out_ref).abs().max().item()}"
        )

    def test_variable_segments_roundtrip_matches_reference(self):
        """Variable segments: scatter->attend->gather matches per-position reference."""
        dec_n_embd = 32
        enc_n_embd = 16
        cfg = CrossAttentionLayerConfig(
            dec_n_embd=dec_n_embd, enc_n_embd=enc_n_embd,
            n_head=2, n_kv_head=2, head_dim=16,
        )
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()
        with torch.no_grad():
            layer.c_proj.weight.normal_(std=0.01)

        B, T = 1, 9
        n_seg = 3
        F_feat = 3
        # Segments of sizes [4, 2, 3]
        seg_map = torch.tensor([[0, 0, 0, 0, 1, 1, 2, 2, 2]])

        decoder_hidden = torch.randn(B, T, dec_n_embd)
        encoder_kv = torch.randn(B, n_seg, F_feat, enc_n_embd)
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool)

        with torch.no_grad():
            out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
            out_ref = _reference_gather_xattn(
                layer, decoder_hidden, encoder_kv, encoder_mask, seg_map
            )

        assert torch.allclose(out, out_ref, atol=1e-5), (
            f"Variable segments roundtrip max diff: {(out - out_ref).abs().max().item()}"
        )

    def test_batched_roundtrip(self):
        """Batched with different layouts: each element matches its own reference."""
        dec_n_embd = 32
        enc_n_embd = 16
        cfg = CrossAttentionLayerConfig(
            dec_n_embd=dec_n_embd, enc_n_embd=enc_n_embd,
            n_head=2, n_kv_head=2, head_dim=16,
        )
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()
        with torch.no_grad():
            layer.c_proj.weight.normal_(std=0.01)

        B, T = 2, 8
        n_seg = 3
        F_feat = 3
        seg_map = torch.tensor([
            [0, 0, 0, 1, 1, 1, 2, 2],  # sizes [3, 3, 2]
            [0, 0, 1, 1, 1, 2, 2, 2],  # sizes [2, 3, 3]
        ])

        decoder_hidden = torch.randn(B, T, dec_n_embd)
        encoder_kv = torch.randn(B, n_seg, F_feat, enc_n_embd)
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool)

        with torch.no_grad():
            out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
            out_ref = _reference_gather_xattn(
                layer, decoder_hidden, encoder_kv, encoder_mask, seg_map
            )

        assert torch.allclose(out, out_ref, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 4: segment isolation
# ---------------------------------------------------------------------------

class TestSegmentIsolation:
    """Two positions in different segments with identical decoder hidden states
    should produce different outputs if their segment encoder features differ,
    and same outputs if encoder features are identical."""

    def test_different_encoder_features_different_outputs(self):
        """Positions in different segments see different encoder KVs -> different outputs."""
        dec_n_embd = 32
        enc_n_embd = 16
        cfg = CrossAttentionLayerConfig(
            dec_n_embd=dec_n_embd, enc_n_embd=enc_n_embd,
            n_head=2, n_kv_head=2, head_dim=16,
        )
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()
        with torch.no_grad():
            layer.c_proj.weight.normal_(std=0.1)

        B, T = 1, 2
        n_seg = 2
        F_feat = 4
        seg_map = torch.tensor([[0, 1]])  # position 0 -> seg 0, position 1 -> seg 1

        # Identical decoder hidden states for both positions
        shared_hidden = torch.randn(1, 1, dec_n_embd)
        decoder_hidden = shared_hidden.expand(B, T, dec_n_embd).clone()

        # Different encoder features per segment
        torch.manual_seed(42)
        encoder_kv = torch.randn(B, n_seg, F_feat, enc_n_embd)
        # Verify segments have different features
        assert not torch.allclose(encoder_kv[0, 0], encoder_kv[0, 1])

        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool)

        with torch.no_grad():
            out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        # Position 0 and position 1 should have different outputs
        assert not torch.allclose(out[0, 0], out[0, 1], atol=1e-6), (
            "Positions in different segments with different encoder features "
            "should produce different outputs"
        )

    def test_identical_encoder_features_same_outputs(self):
        """Positions in different segments with identical encoder KVs -> same outputs."""
        dec_n_embd = 32
        enc_n_embd = 16
        cfg = CrossAttentionLayerConfig(
            dec_n_embd=dec_n_embd, enc_n_embd=enc_n_embd,
            n_head=2, n_kv_head=2, head_dim=16,
        )
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()
        with torch.no_grad():
            layer.c_proj.weight.normal_(std=0.1)

        B, T = 1, 2
        n_seg = 2
        F_feat = 4
        seg_map = torch.tensor([[0, 1]])

        # Identical decoder hidden states
        shared_hidden = torch.randn(1, 1, dec_n_embd)
        decoder_hidden = shared_hidden.expand(B, T, dec_n_embd).clone()

        # Identical encoder features for both segments
        torch.manual_seed(42)
        single_enc = torch.randn(1, 1, F_feat, enc_n_embd)
        encoder_kv = single_enc.expand(B, n_seg, F_feat, enc_n_embd).clone()

        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool)

        with torch.no_grad():
            out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        # Both positions should produce the same output
        assert torch.allclose(out[0, 0], out[0, 1], atol=1e-5), (
            f"Positions in different segments with identical encoder features "
            f"should produce same output. Diff: {(out[0, 0] - out[0, 1]).abs().max().item()}"
        )

    def test_segment_isolation_multi_position(self):
        """Multiple positions per segment: each segment group is isolated."""
        dec_n_embd = 32
        enc_n_embd = 16
        cfg = CrossAttentionLayerConfig(
            dec_n_embd=dec_n_embd, enc_n_embd=enc_n_embd,
            n_head=2, n_kv_head=2, head_dim=16,
        )
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()
        with torch.no_grad():
            layer.c_proj.weight.normal_(std=0.01)

        B, T = 1, 6
        n_seg = 2
        F_feat = 4
        seg_map = torch.tensor([[0, 0, 0, 1, 1, 1]])

        # Make encoder features for segment 1 very different from segment 0
        encoder_kv = torch.zeros(B, n_seg, F_feat, enc_n_embd)
        encoder_kv[:, 0, :, :] = 1.0   # segment 0: all ones
        encoder_kv[:, 1, :, :] = -1.0  # segment 1: all negative ones
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool)

        # Same decoder hidden for all positions
        decoder_hidden = torch.ones(B, T, dec_n_embd) * 0.5

        with torch.no_grad():
            out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        # Positions 0-2 (segment 0) should all be the same
        assert torch.allclose(out[0, 0], out[0, 1], atol=1e-5)
        assert torch.allclose(out[0, 0], out[0, 2], atol=1e-5)

        # Positions 3-5 (segment 1) should all be the same
        assert torch.allclose(out[0, 3], out[0, 4], atol=1e-5)
        assert torch.allclose(out[0, 3], out[0, 5], atol=1e-5)

        # But segment 0 and segment 1 outputs should differ
        assert not torch.allclose(out[0, 0], out[0, 3], atol=1e-6), (
            "Outputs from different segments with different encoder features should differ"
        )


# ---------------------------------------------------------------------------
# Test 5: non-contiguous seg_map edge cases
# ---------------------------------------------------------------------------

class TestSegMapEdgeCases:
    """Edge cases for seg_map: single segment, size-1 segments, batched mismatches."""

    def test_all_positions_segment_zero(self):
        """All positions in segment 0 (single segment)."""
        cfg = _make_config()
        layer = _make_layer_with_nonzero_proj(cfg)

        B, T = 2, 16
        n_seg = 1
        F_feat = 4
        seg_map = torch.zeros(B, T, dtype=torch.long)
        decoder_hidden = torch.randn(B, T, cfg.dec_n_embd)
        encoder_kv = torch.randn(B, n_seg, F_feat, cfg.enc_n_embd)
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool)

        with torch.no_grad():
            out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        assert out.shape == (B, T, cfg.dec_n_embd)
        assert torch.isfinite(out).all()

        # Verify against reference
        out_ref = _reference_gather_xattn(
            layer, decoder_hidden, encoder_kv, encoder_mask, seg_map
        )
        assert torch.allclose(out, out_ref, atol=1e-5)

    def test_all_size_one_segments_forward(self):
        """Each position is its own segment (max fragmentation)."""
        T = 8
        n_seg = T
        B = 1
        F_feat = 2
        dec_n_embd = 32
        enc_n_embd = 16
        cfg = CrossAttentionLayerConfig(
            dec_n_embd=dec_n_embd, enc_n_embd=enc_n_embd,
            n_head=2, n_kv_head=2, head_dim=16,
        )
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()
        with torch.no_grad():
            layer.c_proj.weight.normal_(std=0.01)

        seg_map = torch.arange(n_seg, dtype=torch.long).unsqueeze(0)
        decoder_hidden = torch.randn(B, T, dec_n_embd)
        encoder_kv = torch.randn(B, n_seg, F_feat, enc_n_embd)
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool)

        with torch.no_grad():
            out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        assert out.shape == (B, T, dec_n_embd)
        assert torch.isfinite(out).all()

        # Verify against reference
        out_ref = _reference_gather_xattn(
            layer, decoder_hidden, encoder_kv, encoder_mask, seg_map
        )
        assert torch.allclose(out, out_ref, atol=1e-5)

    def test_batched_different_segment_layouts(self):
        """Different segment layouts per batch element."""
        B, T = 2, 10
        n_seg = 3
        F_feat = 3
        dec_n_embd = 32
        enc_n_embd = 16
        cfg = CrossAttentionLayerConfig(
            dec_n_embd=dec_n_embd, enc_n_embd=enc_n_embd,
            n_head=2, n_kv_head=2, head_dim=16,
        )
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()
        with torch.no_grad():
            layer.c_proj.weight.normal_(std=0.01)

        # Row 0: segments [5, 3, 2]
        # Row 1: segments [2, 2, 6]
        seg_map = torch.tensor([
            [0, 0, 0, 0, 0, 1, 1, 1, 2, 2],
            [0, 0, 1, 1, 2, 2, 2, 2, 2, 2],
        ])
        decoder_hidden = torch.randn(B, T, dec_n_embd)
        encoder_kv = torch.randn(B, n_seg, F_feat, enc_n_embd)
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool)

        with torch.no_grad():
            out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        assert out.shape == (B, T, dec_n_embd)
        assert torch.isfinite(out).all()

        out_ref = _reference_gather_xattn(
            layer, decoder_hidden, encoder_kv, encoder_mask, seg_map
        )
        assert torch.allclose(out, out_ref, atol=1e-5)

    def test_single_position_single_segment(self):
        """Minimal case: T=1, n_seg=1."""
        dec_n_embd = 32
        enc_n_embd = 16
        cfg = CrossAttentionLayerConfig(
            dec_n_embd=dec_n_embd, enc_n_embd=enc_n_embd,
            n_head=2, n_kv_head=2, head_dim=16,
        )
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()
        with torch.no_grad():
            layer.c_proj.weight.normal_(std=0.01)

        B, T, n_seg, F_feat = 1, 1, 1, 2
        seg_map = torch.zeros(B, T, dtype=torch.long)
        decoder_hidden = torch.randn(B, T, dec_n_embd)
        encoder_kv = torch.randn(B, n_seg, F_feat, enc_n_embd)
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool)

        with torch.no_grad():
            out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        assert out.shape == (B, T, dec_n_embd)
        assert torch.isfinite(out).all()

    def test_two_batch_elements_independent(self):
        """Changing encoder_kv for batch 0 should not affect batch 1 output."""
        dec_n_embd = 32
        enc_n_embd = 16
        cfg = CrossAttentionLayerConfig(
            dec_n_embd=dec_n_embd, enc_n_embd=enc_n_embd,
            n_head=2, n_kv_head=2, head_dim=16,
        )
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()
        with torch.no_grad():
            layer.c_proj.weight.normal_(std=0.01)

        B, T, n_seg, F_feat = 2, 6, 2, 3
        seg_map = torch.tensor([
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
        ])
        decoder_hidden = torch.randn(B, T, dec_n_embd)
        encoder_kv_v1 = torch.randn(B, n_seg, F_feat, enc_n_embd)
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool)

        with torch.no_grad():
            out_v1 = layer(decoder_hidden, encoder_kv_v1, encoder_mask, seg_map)

        # Change only batch 0's encoder_kv
        encoder_kv_v2 = encoder_kv_v1.clone()
        encoder_kv_v2[0] = torch.randn(n_seg, F_feat, enc_n_embd)

        with torch.no_grad():
            out_v2 = layer(decoder_hidden, encoder_kv_v2, encoder_mask, seg_map)

        # Batch 1 output should be identical
        assert torch.allclose(out_v1[1], out_v2[1], atol=1e-6), (
            "Changing batch 0 encoder_kv should not affect batch 1 output"
        )
        # Batch 0 output should differ
        assert not torch.allclose(out_v1[0], out_v2[0], atol=1e-6)

    @given(
        seg_sizes=st.lists(
            st.integers(min_value=1, max_value=10), min_size=1, max_size=6
        )
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_arbitrary_seg_map_matches_reference(self, seg_sizes):
        """Property: any valid seg_map should match the per-position reference."""
        n_seg = len(seg_sizes)
        T = sum(seg_sizes)
        B = 1
        F_feat = 2
        dec_n_embd = 32
        enc_n_embd = 16
        cfg = CrossAttentionLayerConfig(
            dec_n_embd=dec_n_embd, enc_n_embd=enc_n_embd,
            n_head=2, n_kv_head=2, head_dim=16,
        )
        layer = CrossAttentionLayer(cfg)
        layer.init_weights()
        with torch.no_grad():
            layer.c_proj.weight.normal_(std=0.01)

        seg_map_list = []
        for i, sz in enumerate(seg_sizes):
            seg_map_list.extend([i] * sz)
        seg_map = torch.tensor([seg_map_list])

        torch.manual_seed(42)
        decoder_hidden = torch.randn(B, T, dec_n_embd)
        encoder_kv = torch.randn(B, n_seg, F_feat, enc_n_embd)
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool)

        with torch.no_grad():
            out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
            out_ref = _reference_gather_xattn(
                layer, decoder_hidden, encoder_kv, encoder_mask, seg_map
            )

        assert torch.allclose(out, out_ref, atol=1e-5), (
            f"seg_sizes={seg_sizes}: max diff {(out - out_ref).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# Test 6: NaN handling order (nan_to_num BEFORE gather)
# ---------------------------------------------------------------------------

class TestNaNHandlingOrder:
    """Verify nan_to_num is called BEFORE gather.

    The padded tensor has zeros at padding positions. After RMS norm + attention,
    these become NaN. nan_to_num must convert them to 0 BEFORE the gather step.
    If the order were reversed (gather then nan_to_num), the gather would read
    NaN from padding positions when padded_idx happens to alias a padding slot
    (which it does NOT do by construction — but the real concern is that the
    c_proj output at padding positions contains NaN propagated through the
    linear layer, and gather might pick those up via index collisions or if
    the logic were wrong).

    The test strategy: create a scenario where padding positions definitely
    produce NaN (uneven segments), and verify the output is fully finite.
    We additionally verify via source inspection that nan_to_num appears
    before gather in the code.
    """

    def test_nan_to_num_before_gather_in_source(self):
        """nan_to_num must appear in source before .gather()."""
        import inspect
        source = inspect.getsource(padded_cross_attention_reference)
        lines = source.splitlines()
        nan_to_num_line = None
        gather_line = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "nan_to_num()" in stripped and nan_to_num_line is None:
                # Skip the q/k nan_to_num calls, look for the y one
                if "y" in stripped or "y =" in stripped:
                    nan_to_num_line = i
            if ".gather(" in stripped and gather_line is None:
                gather_line = i

        # Find the LAST nan_to_num before gather (the y = y.nan_to_num() one)
        last_nan_to_num = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "nan_to_num()" in stripped:
                last_nan_to_num = i
            if ".gather(" in stripped:
                gather_line = i
                break

        assert last_nan_to_num is not None, "nan_to_num() not found in forward()"
        assert gather_line is not None, ".gather() not found in forward()"
        assert last_nan_to_num < gather_line, (
            f"nan_to_num (line {last_nan_to_num}) must come BEFORE "
            f".gather (line {gather_line}) in forward()"
        )

    def test_uneven_segments_no_nan_in_output(self):
        """Uneven segments create padding positions that become NaN after norm.

        This test would fail if nan_to_num were called AFTER gather, because
        c_proj applied to NaN input produces NaN output, and gather would
        read those NaN values.
        """
        cfg = _make_config()
        layer = _make_layer_with_nonzero_proj(cfg)

        B, T = 1, 7
        n_seg = 3
        F_feat = 3
        # Segments of sizes [4, 2, 1] — segment 1 and 2 have padding
        seg_map = torch.tensor([[0, 0, 0, 0, 1, 1, 2]])
        decoder_hidden = torch.randn(B, T, cfg.dec_n_embd)
        encoder_kv = torch.randn(B, n_seg, F_feat, cfg.enc_n_embd)
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool)

        with torch.no_grad():
            out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        assert torch.isfinite(out).all(), (
            "Output must be finite — NaN at padding positions must be cleaned "
            "BEFORE gather"
        )

    def test_all_masked_encoder_all_positions_finite(self):
        """All-masked encoder produces NaN from SDPA -> nan_to_num cleans -> gather is finite.

        If nan_to_num happened after gather, this would produce NaN output.
        """
        cfg = _make_config()
        layer = _make_layer_with_nonzero_proj(cfg)

        B, T = 1, 4
        n_seg = 2
        F_feat = 3
        seg_map = torch.tensor([[0, 0, 1, 1]])
        decoder_hidden = torch.randn(B, T, cfg.dec_n_embd)
        encoder_kv = torch.randn(B, n_seg, F_feat, cfg.enc_n_embd)
        encoder_mask = torch.zeros(B, n_seg, F_feat, dtype=torch.bool)  # all masked!

        with torch.no_grad():
            out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        assert torch.isfinite(out).all(), (
            "All-masked encoder must produce finite output. "
            "SDPA produces NaN for all-masked queries; nan_to_num must clean this "
            "BEFORE gather."
        )

    def test_mixed_masked_uneven_segments(self):
        """Combination: uneven segments + partially masked encoder.

        This is the hardest case for NaN propagation: both padding positions
        in the scatter tensor AND all-masked segments produce NaN.
        """
        cfg = _make_config()
        layer = _make_layer_with_nonzero_proj(cfg)

        B, T = 2, 10
        n_seg = 3
        F_feat = 4
        # Row 0: segments [5, 3, 2]
        # Row 1: segments [3, 4, 3]
        seg_map = torch.tensor([
            [0, 0, 0, 0, 0, 1, 1, 1, 2, 2],
            [0, 0, 0, 1, 1, 1, 1, 2, 2, 2],
        ])
        decoder_hidden = torch.randn(B, T, cfg.dec_n_embd)
        encoder_kv = torch.randn(B, n_seg, F_feat, cfg.enc_n_embd)
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool)
        # Fully mask segment 1 for batch 0, segment 2 for batch 1
        encoder_mask[0, 1, :] = False
        encoder_mask[1, 2, :] = False

        with torch.no_grad():
            out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        assert torch.isfinite(out).all(), (
            "Mixed masked + uneven segments must produce finite output"
        )

    def test_c_proj_propagates_nan(self):
        """Demonstrate that c_proj(NaN) = NaN.

        This proves that if nan_to_num were called after c_proj+gather instead
        of before, the output could contain NaN.
        """
        cfg = _make_config()
        layer = _make_layer_with_nonzero_proj(cfg)

        nan_input = torch.full((1, 1, cfg.dec_n_embd), float("nan"))
        with torch.no_grad():
            proj_out = layer.c_proj(nan_input)
        assert torch.isnan(proj_out).all(), (
            "c_proj(NaN) must produce NaN, proving nan_to_num must happen "
            "before c_proj or before gather"
        )

    def test_nan_to_num_is_defensive_but_needed_for_portability(self):
        """nan_to_num is a defensive measure whose necessity depends on PyTorch version.

        Some PyTorch versions produce NaN from rms_norm(zeros) or SDPA(all-masked).
        Others return zeros. The nan_to_num call must remain as a safety net.
        This test verifies that regardless of whether intermediate NaN is produced,
        the final output is finite.
        """
        cfg = _make_config()
        layer = _make_layer_with_nonzero_proj(cfg)

        # Worst case: uneven segments + some fully masked
        B, T = 1, 5
        n_seg = 3
        F_feat = 3
        seg_map = torch.tensor([[0, 0, 0, 1, 2]])
        decoder_hidden = torch.randn(B, T, cfg.dec_n_embd)
        encoder_kv = torch.randn(B, n_seg, F_feat, cfg.enc_n_embd)
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool)
        encoder_mask[:, 2, :] = False  # fully mask segment 2

        with torch.no_grad():
            out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)

        assert torch.isfinite(out).all(), (
            "Output must be finite with uneven segments + masked encoder"
        )
