"""Tests for EncoderBlock COMPUTE_DTYPE cast after residual (encoder.py:150).

Root cause: Float8Linear outputs float32. Without the cast, float32 propagates
through encoder residual connections into cross-attention, causing dtype mismatches
(float32 keys/values vs bf16 queries) that produce NaN in SDPA.
"""

import os

os.environ["NANOCHAT_ATTN_IMPL"] = "sdpa"

import pytest
import torch

from nanochat.common import COMPUTE_DTYPE
from steering.conditional.encoder_decoder.encoder import (
    EncoderBlock,
    FeatureEncoder,
    FeatureEncoderConfig,
)
from steering.conditional.encoder_decoder.packed_encoder import PACKED_ENCODER_ALIGNMENT


@pytest.fixture
def enc_config():
    return FeatureEncoderConfig(
        n_sae_features=100, num_buckets=4, n_embd=32,
        n_layer=2, n_head=2, n_kv_head=2, head_dim=16,
    )


class TestEncoderBlockDtypeCast:
    """Verify EncoderBlock casts output to COMPUTE_DTYPE."""

    def test_block_output_is_compute_dtype(self, enc_config):
        """EncoderBlock.forward() returns COMPUTE_DTYPE even with fp32 input."""
        block = EncoderBlock(enc_config, layer_idx=0)
        # Simulate fp32 input (what happens when Float8Linear outputs fp32)
        x = torch.randn(4, 8, 32, dtype=torch.float32)
        mask = torch.ones(4, 8, dtype=torch.bool)
        out = block(x, ve=None, mask=mask)
        assert out.dtype == COMPUTE_DTYPE, (
            f"EncoderBlock output dtype {out.dtype} != {COMPUTE_DTYPE}. "
            "Missing x.to(COMPUTE_DTYPE) after residual."
        )

    def test_block_output_is_compute_dtype_with_bf16_input(self, enc_config):
        """EncoderBlock with bf16 input also returns COMPUTE_DTYPE."""
        block = EncoderBlock(enc_config, layer_idx=0)
        x = torch.randn(4, 8, 32, dtype=COMPUTE_DTYPE)
        mask = torch.ones(4, 8, dtype=torch.bool)
        out = block(x, ve=None, mask=mask)
        assert out.dtype == COMPUTE_DTYPE

    def test_block_output_with_value_embeddings(self, enc_config):
        """EncoderBlock with VE produces COMPUTE_DTYPE output."""
        block = EncoderBlock(enc_config, layer_idx=1)
        # has_ve(1, 2) = (1 % 2 == (2-1) % 2) = (1 == 1) = True
        assert block.attn.ve_gate is not None, "layer_idx=1 should have VE"
        if block.attn.ve_gate is not None:
            kv_dim = enc_config.n_kv_head * enc_config.head_dim
            ve = torch.randn(4, 8, kv_dim, dtype=torch.float32)
            x = torch.randn(4, 8, 32, dtype=torch.float32)
            mask = torch.ones(4, 8, dtype=torch.bool)
            out = block(x, ve=ve, mask=mask)
            assert out.dtype == COMPUTE_DTYPE

    def test_multiple_blocks_preserve_dtype(self, enc_config):
        """Chaining multiple blocks preserves COMPUTE_DTYPE."""
        blocks = [EncoderBlock(enc_config, i) for i in range(2)]
        x = torch.randn(4, 8, 32, dtype=torch.float32)  # Start with fp32
        mask = torch.ones(4, 8, dtype=torch.bool)
        for block in blocks:
            x = block(x, ve=None, mask=mask)
            assert x.dtype == COMPUTE_DTYPE, (
                f"Block output dtype {x.dtype} after chaining"
            )


class TestEncoderBlockGradientWithDtypeCast:
    """Verify gradients flow correctly through the dtype cast."""

    def test_gradient_flows_through_cast(self, enc_config):
        """Gradients propagate correctly despite the dtype cast."""
        block = EncoderBlock(enc_config, layer_idx=0)
        x = torch.randn(4, 8, 32, dtype=torch.float32, requires_grad=True)
        mask = torch.ones(4, 8, dtype=torch.bool)
        out = block(x, ve=None, mask=mask)
        out.sum().backward()
        assert x.grad is not None, "No gradient for input"
        assert not torch.isnan(x.grad).any(), "NaN in input gradient"
        for name, p in block.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"


class TestFeatureEncoderFP8Padding:
    """Verify FP8 padding to multiple of 16 in FeatureEncoder."""

    def test_nf_padded_to_multiple_of_16(self, enc_config):
        """Feature dimension is padded to multiple of 16 for FP8 compatibility."""
        encoder = FeatureEncoder(enc_config)
        encoder.init_weights()
        B, S, Nf = 2, 3, 5  # Nf=5 → padded to 16
        fids = torch.randint(0, 100, (B, S, Nf))
        mags = torch.randint(0, 4, (B, S, Nf))
        mask = torch.ones(B, S, Nf, dtype=torch.bool)
        out = encoder(fids, mags, mask)
        assert out.layout.packed_count % PACKED_ENCODER_ALIGNMENT == 0
        assert int(out.layout.valid_mask.sum()) == B * S * Nf
        assert out.layout.packed_count - B * S * Nf < PACKED_ENCODER_ALIGNMENT

    def test_padding_mask_false_for_padded_positions(self, enc_config):
        """Padded positions have mask=False."""
        encoder = FeatureEncoder(enc_config)
        encoder.init_weights()
        B, S, Nf = 2, 3, 5
        fids = torch.randint(0, 100, (B, S, Nf))
        mags = torch.randint(0, 4, (B, S, Nf))
        mask = torch.ones(B, S, Nf, dtype=torch.bool)
        output = encoder(fids, mags, mask)
        real_count = B * S * Nf
        assert output.layout.valid_mask[:real_count].all()
        assert not output.layout.valid_mask[real_count:].any()

    def test_already_aligned_no_extra_padding(self, enc_config):
        """Nf already a multiple of 16 → no additional padding."""
        encoder = FeatureEncoder(enc_config)
        encoder.init_weights()
        B, S, Nf = 2, 3, 16
        fids = torch.randint(0, 100, (B, S, Nf))
        mags = torch.randint(0, 4, (B, S, Nf))
        mask = torch.ones(B, S, Nf, dtype=torch.bool)
        out = encoder(fids, mags, mask)
        assert out.layout.packed_count == B * S * Nf
