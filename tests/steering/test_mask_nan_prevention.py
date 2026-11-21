"""Tests for NaN prevention in encoder SDPA and cross-attention.

Three NaN sources addressed:
1. Encoder: all-masked segments → softmax(all -inf) → NaN → nan_to_num (encoder.py:116)
2. Cross-attention: padding zeros → NaN after RMS norm → nan_to_num on q,k (cross_attention.py:194-195)
3. Cross-attention: all-masked encoder segments → NaN output → nan_to_num (cross_attention.py:209)
"""

import os

os.environ["NANOCHAT_ATTN_IMPL"] = "sdpa"

import pytest
import torch

from steering.conditional.encoder_decoder.cross_attention import (
    CrossAttentionLayer,
    CrossAttentionLayerConfig,
)
from steering.conditional.encoder_decoder.encoder import (
    EncoderBlock,
    EncoderSelfAttention,
    FeatureEncoderConfig,
)


@pytest.fixture
def enc_config():
    return FeatureEncoderConfig(
        n_sae_features=100, num_buckets=4, n_embd=32,
        n_layer=2, n_head=2, n_kv_head=2, head_dim=16,
    )


@pytest.fixture
def xattn_config():
    return CrossAttentionLayerConfig(
        dec_n_embd=32, enc_n_embd=32, n_head=2, n_kv_head=2, head_dim=16,
    )


class TestEncoderSDPANanPrevention:
    """Encoder SDPA handles all-masked inputs without NaN."""

    def test_all_masked_segment_no_nan(self, enc_config):
        """Segment with mask=all-False produces zero output, not NaN."""
        attn = EncoderSelfAttention(enc_config, layer_idx=0)
        B, Nf, C = 2, 8, 32
        x = torch.randn(B, Nf, C)
        mask = torch.zeros(B, Nf, dtype=torch.bool)  # ALL False
        out = attn(x, ve=None, mask=mask)
        assert not torch.isnan(out).any(), (
            "NaN in encoder attention output with all-False mask. "
            "nan_to_num missing after SDPA?"
        )

    def test_partially_masked_no_nan(self, enc_config):
        """Segment with some positions masked produces no NaN."""
        attn = EncoderSelfAttention(enc_config, layer_idx=0)
        B, Nf, C = 2, 8, 32
        x = torch.randn(B, Nf, C)
        mask = torch.ones(B, Nf, dtype=torch.bool)
        mask[:, 4:] = False  # Last 4 positions masked
        out = attn(x, ve=None, mask=mask)
        assert not torch.isnan(out).any()

    def test_mixed_batch_some_all_masked(self, enc_config):
        """Batch where some rows are all-masked and others are not."""
        attn = EncoderSelfAttention(enc_config, layer_idx=0)
        B, Nf, C = 4, 8, 32
        x = torch.randn(B, Nf, C)
        mask = torch.ones(B, Nf, dtype=torch.bool)
        mask[0] = False  # Row 0: all masked
        mask[2] = False  # Row 2: all masked
        out = attn(x, ve=None, mask=mask)
        assert not torch.isnan(out).any(), "NaN in mixed-mask batch"

    def test_encoder_block_all_masked_no_nan(self, enc_config):
        """Full EncoderBlock with all-False mask produces no NaN."""
        block = EncoderBlock(enc_config, layer_idx=0)
        B, Nf, C = 2, 8, 32
        x = torch.randn(B, Nf, C)
        mask = torch.zeros(B, Nf, dtype=torch.bool)
        out = block(x, ve=None, mask=mask)
        assert not torch.isnan(out).any()


class TestCrossAttentionNanPrevention:
    """Cross-attention handles NaN from padding and all-masked encoder."""

    def test_all_masked_encoder_no_nan(self, xattn_config):
        """All-False encoder mask → no NaN in cross-attention output."""
        layer = CrossAttentionLayer(xattn_config)
        layer.init_weights()
        B, T, n_seg, F_feat = 2, 16, 2, 8
        dec_hidden = torch.randn(B, T, 32)
        enc_kv = torch.randn(B, n_seg, F_feat, 32)
        enc_mask = torch.zeros(B, n_seg, F_feat, dtype=torch.bool)  # ALL False
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[:, T // 2:] = 1
        out = layer(dec_hidden, enc_kv, enc_mask, seg_map)
        assert not torch.isnan(out).any(), (
            "NaN in cross-attention with all-False encoder mask"
        )

    def test_padding_zeros_no_nan(self, xattn_config):
        """Decoder padding (zeros) → NaN after RMS norm → cleaned by nan_to_num."""
        layer = CrossAttentionLayer(xattn_config)
        layer.init_weights()
        B, T, n_seg, F_feat = 2, 16, 2, 8
        # Some decoder positions are zeros (padding from scatter)
        dec_hidden = torch.randn(B, T, 32)
        enc_kv = torch.randn(B, n_seg, F_feat, 32)
        enc_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[:, T // 2:] = 1
        out = layer(dec_hidden, enc_kv, enc_mask, seg_map)
        assert not torch.isnan(out).any()

    def test_mixed_mask_no_nan(self, xattn_config):
        """Some segments masked, some not — no NaN anywhere."""
        layer = CrossAttentionLayer(xattn_config)
        layer.init_weights()
        B, T, n_seg, F_feat = 2, 16, 4, 8
        dec_hidden = torch.randn(B, T, 32)
        enc_kv = torch.randn(B, n_seg, F_feat, 32)
        enc_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool)
        enc_mask[:, 2:] = False  # Last 2 segments all-masked
        seg_map = torch.arange(T).unsqueeze(0).expand(B, -1) // (T // n_seg)
        seg_map = seg_map.clamp(max=n_seg - 1)
        out = layer(dec_hidden, enc_kv, enc_mask, seg_map)
        assert not torch.isnan(out).any()

    def test_gradient_no_nan_with_all_masked(self, xattn_config):
        """Backward through all-masked cross-attention produces no NaN grads."""
        layer = CrossAttentionLayer(xattn_config)
        layer.init_weights()
        B, T, n_seg, F_feat = 2, 16, 2, 8
        dec_hidden = torch.randn(B, T, 32, requires_grad=True)
        enc_kv = torch.randn(B, n_seg, F_feat, 32, requires_grad=True)
        # One segment real, one all-masked
        enc_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool)
        enc_mask[:, 1] = False
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[:, T // 2:] = 1
        out = layer(dec_hidden, enc_kv, enc_mask, seg_map)
        out.sum().backward()
        assert not torch.isnan(dec_hidden.grad).any(), "NaN in decoder grad"  # ty: ignore[invalid-argument-type]
        assert not torch.isnan(enc_kv.grad).any(), "NaN in encoder grad"  # ty: ignore[invalid-argument-type]
        for name, p in layer.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"


class TestCrossAttentionInitWeights:
    """Cross-attention c_proj init is non-zero (FP8 amax safety)."""

    def test_c_proj_not_zero_after_init(self, xattn_config):
        """c_proj weights are NOT zero after init_weights (was zero, now uniform)."""
        layer = CrossAttentionLayer(xattn_config)
        layer.init_weights()
        assert layer.c_proj.weight.abs().sum() > 0, (
            "c_proj weights are all zero — FP8 tensorwise scaling will degenerate"
        )

    def test_all_projections_nonzero_after_init(self, xattn_config):
        """All three projections (c_q, c_kv, c_proj) are non-zero after init."""
        layer = CrossAttentionLayer(xattn_config)
        layer.init_weights()
        for name in ["c_q", "c_kv", "c_proj"]:
            w = getattr(layer, name).weight
            assert w.abs().sum() > 0, f"{name} weights are all zero"


class TestCrossAttentionGQA:
    """NaN safety with GQA (n_head != n_kv_head) in cross-attention."""

    def test_gqa_all_masked_no_nan(self):
        """GQA cross-attention with all-False encoder mask produces no NaN."""
        gqa_config = CrossAttentionLayerConfig(
            dec_n_embd=32, enc_n_embd=32, n_head=4, n_kv_head=2, head_dim=8,
        )
        layer = CrossAttentionLayer(gqa_config)
        layer.init_weights()
        B, T, n_seg, F_feat = 2, 16, 2, 8
        dec_hidden = torch.randn(B, T, 32)
        enc_kv = torch.randn(B, n_seg, F_feat, 32)
        enc_mask = torch.zeros(B, n_seg, F_feat, dtype=torch.bool)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[:, T // 2:] = 1
        out = layer(dec_hidden, enc_kv, enc_mask, seg_map)
        assert not torch.isnan(out).any(), "NaN in GQA cross-attention with all-False mask"

    def test_gqa_gradient_no_nan(self):
        """GQA cross-attention backward with mixed masks produces no NaN."""
        gqa_config = CrossAttentionLayerConfig(
            dec_n_embd=32, enc_n_embd=32, n_head=4, n_kv_head=2, head_dim=8,
        )
        layer = CrossAttentionLayer(gqa_config)
        layer.init_weights()
        B, T, n_seg, F_feat = 2, 16, 2, 8
        dec_hidden = torch.randn(B, T, 32, requires_grad=True)
        enc_kv = torch.randn(B, n_seg, F_feat, 32, requires_grad=True)
        enc_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool)
        enc_mask[:, 1] = False  # Second segment all-masked
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[:, T // 2:] = 1
        out = layer(dec_hidden, enc_kv, enc_mask, seg_map)
        out.sum().backward()
        assert not torch.isnan(dec_hidden.grad).any(), "NaN in GQA decoder grad"  # ty: ignore[invalid-argument-type]
        assert not torch.isnan(enc_kv.grad).any(), "NaN in GQA encoder grad"  # ty: ignore[invalid-argument-type]
