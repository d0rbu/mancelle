"""Slice 08 audit (A9 / map.md H10).

CPU-only encoder→cross-attention integration test:

The existing ``tests/steering/test_fp8_alignment.py`` covers this on CUDA
with FP8 backward but is skipped on CPU. This test pins the CPU path:

* Construct ``FeatureEncoder`` on CPU; forward; capture
  ``PackedEncoderOutput``.
* Construct ``CrossAttentionLayer`` matching enc_n_embd; forward with
  a synthetic ``(B, T, dec_n_embd)`` decoder hidden + a deterministic
  ``seg_map``.
* Assert: output shape ``(B, T, dec_n_embd)``, no NaN/Inf, gradient
  flows back to encoder embeddings.
* Vary ``S`` (n_seg) and ``Nf`` to ensure the contract holds for the
  cross-product.

This is the audit's CPU-only "encoder produces consumable KV" check —
deliberately separate from slice-9's cross-attention internal tests.
"""

from __future__ import annotations

import os

os.environ["NANOCHAT_ATTN_IMPL"] = "sdpa"

import pytest
import torch

from nanochat.common import COMPUTE_DTYPE
from steering.conditional.encoder_decoder.cross_attention import (
    CrossAttentionLayer,
    CrossAttentionLayerConfig,
)
from steering.conditional.encoder_decoder.encoder import (
    FeatureEncoder,
    FeatureEncoderConfig,
)


def _make_encoder_config(n_embd=64, n_layer=1, n_head=4, n_kv_head=2, head_dim=16):
    return FeatureEncoderConfig(
        n_sae_features=64,
        num_buckets=4,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_kv_head,
        head_dim=head_dim,
    )


def _make_xattn_config(enc_n_embd: int, dec_n_embd: int = 64, n_head: int = 4, n_kv_head: int = 2, head_dim: int = 16):
    return CrossAttentionLayerConfig(
        dec_n_embd=dec_n_embd,
        enc_n_embd=enc_n_embd,
        n_head=n_head,
        n_kv_head=n_kv_head,
        head_dim=head_dim,
    )


def _build_decoder_hidden(B: int, T: int, dec_n_embd: int) -> torch.Tensor:
    """Random decoder hidden states with non-zero variance (so attention has signal).

    Cast to ``COMPUTE_DTYPE`` to match the model's contract: ``model.py:346,353``
    casts decoder ``x`` to ``COMPUTE_DTYPE`` before any block / cross-attention
    call. Mismatched dtype (fp32 query vs bf16 encoder K/V) raises in SDPA.
    """
    return (torch.randn(B, T, dec_n_embd, requires_grad=False) * 0.1).to(COMPUTE_DTYPE)


def _build_seg_map(B: int, T: int, n_seg: int) -> torch.Tensor:
    """Build a deterministic ``seg_map`` (B, T) with ``n_seg`` distinct segments.

    Each segment gets ``T // n_seg`` positions; remainder goes to the last.
    """
    assert n_seg <= T, "n_seg must not exceed T"
    seg_size = T // n_seg
    seg_map = torch.zeros(B, T, dtype=torch.int64)
    for s in range(n_seg):
        start = s * seg_size
        end = start + seg_size if s < n_seg - 1 else T
        seg_map[:, start:end] = s
    return seg_map


class TestEncoderCrossAttentionIntegration:
    """End-to-end CPU forward through encoder + cross-attention."""

    @pytest.mark.parametrize("S,Nf", [(1, 1), (1, 5), (2, 5), (3, 9)])
    def test_forward_no_nan(self, S, Nf):
        """A9 / H10: encoder→cross-attention forward produces a finite tensor."""
        torch.manual_seed(7)
        enc_config = _make_encoder_config()
        encoder = FeatureEncoder(enc_config)
        encoder.init_weights()

        xattn_config = _make_xattn_config(enc_n_embd=enc_config.n_embd)
        xattn = CrossAttentionLayer(xattn_config)
        xattn.init_weights()

        B = 1
        ids = torch.randint(0, enc_config.n_sae_features, (B, S, Nf))
        mags = torch.randint(0, enc_config.num_buckets, (B, S, Nf))
        mask = torch.ones(B, S, Nf, dtype=torch.bool)
        encoder_out = encoder(ids, mags, mask)

        # Decoder side: T = S * 4 (so each segment has 4 decoder positions).
        T = S * 4
        decoder_hidden = _build_decoder_hidden(B, T, xattn_config.dec_n_embd)
        seg_map = _build_seg_map(B, T, n_seg=S)

        out = xattn(decoder_hidden, encoder_out, seg_map)
        assert out.shape == (B, T, xattn_config.dec_n_embd)
        assert not torch.isnan(out).any(), "Cross-attention output contains NaN"
        assert not torch.isinf(out).any(), "Cross-attention output contains Inf"

    def test_gradient_flows_to_encoder_embeddings(self):
        """A9 / H10: backward from cross-attention output reaches encoder feature_embed.weight."""
        torch.manual_seed(13)
        enc_config = _make_encoder_config(n_layer=2)
        encoder = FeatureEncoder(enc_config)
        encoder.init_weights()
        # Perturb encoder c_proj so the forward signal is non-trivial through the
        # blocks (init zeros them).
        with torch.no_grad():
            for block in encoder.blocks:
                torch.nn.init.normal_(block.attn.c_proj.weight, std=0.02)
                torch.nn.init.normal_(block.mlp.c_proj.weight, std=0.02)

        xattn_config = _make_xattn_config(enc_n_embd=enc_config.n_embd)
        xattn = CrossAttentionLayer(xattn_config)
        xattn.init_weights()

        B, S, Nf, T = 1, 2, 5, 6
        ids = torch.randint(0, enc_config.n_sae_features, (B, S, Nf))
        mags = torch.randint(0, enc_config.num_buckets, (B, S, Nf))
        mask = torch.ones(B, S, Nf, dtype=torch.bool)
        encoder_out = encoder(ids, mags, mask)

        decoder_hidden = _build_decoder_hidden(B, T, xattn_config.dec_n_embd)
        seg_map = _build_seg_map(B, T, n_seg=S)

        out = xattn(decoder_hidden, encoder_out, seg_map)
        loss = out.sum()
        loss.backward()

        # Encoder embeddings receive gradient from the cross-attention path.
        # (Cast to fp32 so bf16 abs.sum doesn't underflow to zero.)
        g = encoder.feature_embed.weight.grad
        assert g is not None, "feature_embed.weight has no gradient"
        assert g.float().abs().sum().item() > 0.0, (
            "Gradient did not propagate from CrossAttentionLayer back to "
            "FeatureEncoder.feature_embed — integration is silently broken."
        )

    def test_partial_mask_does_not_produce_nan(self):
        """A9 / H10: partially masked encoder features still yield a finite output."""
        torch.manual_seed(21)
        enc_config = _make_encoder_config()
        encoder = FeatureEncoder(enc_config)
        encoder.init_weights()
        xattn_config = _make_xattn_config(enc_n_embd=enc_config.n_embd)
        xattn = CrossAttentionLayer(xattn_config)
        xattn.init_weights()

        B, S, Nf, T = 1, 2, 7, 4
        ids = torch.randint(0, enc_config.n_sae_features, (B, S, Nf))
        mags = torch.randint(0, enc_config.num_buckets, (B, S, Nf))
        mask = torch.ones(B, S, Nf, dtype=torch.bool)
        # Partially mask each segment.
        mask[0, 0, 4:] = False
        mask[0, 1, 3:] = False

        encoder_out = encoder(ids, mags, mask)
        decoder_hidden = _build_decoder_hidden(B, T, xattn_config.dec_n_embd)
        seg_map = _build_seg_map(B, T, n_seg=S)

        out = xattn(decoder_hidden, encoder_out, seg_map)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_all_masked_segment_does_not_produce_nan(self):
        """A9 / H10: even an all-masked encoder segment leaves the cross-attention output finite.

        Encoder ``nan_to_num`` plus cross-attention's own ``nan_to_num`` jointly
        scrub NaN. This is the headline guard that justifies why slice 8 must
        match slice 9 on this contract.
        """
        torch.manual_seed(33)
        enc_config = _make_encoder_config(n_layer=2)
        encoder = FeatureEncoder(enc_config)
        encoder.init_weights()
        with torch.no_grad():
            for block in encoder.blocks:
                torch.nn.init.normal_(block.attn.c_proj.weight, std=0.02)
                torch.nn.init.normal_(block.mlp.c_proj.weight, std=0.02)
        xattn_config = _make_xattn_config(enc_n_embd=enc_config.n_embd)
        xattn = CrossAttentionLayer(xattn_config)
        xattn.init_weights()

        B, S, Nf, T = 1, 2, 4, 4
        ids = torch.randint(0, enc_config.n_sae_features, (B, S, Nf))
        mags = torch.randint(0, enc_config.num_buckets, (B, S, Nf))
        mask = torch.ones(B, S, Nf, dtype=torch.bool)
        # Fully mask segment 0.
        mask[0, 0, :] = False

        encoder_out = encoder(ids, mags, mask)
        decoder_hidden = _build_decoder_hidden(B, T, xattn_config.dec_n_embd)
        seg_map = _build_seg_map(B, T, n_seg=S)

        out = xattn(decoder_hidden, encoder_out, seg_map)
        assert not torch.isnan(out).any(), (
            "All-masked encoder segment produced NaN downstream — encoder "
            "nan_to_num + cross-attn nan_to_num joint guarantee broken."
        )
