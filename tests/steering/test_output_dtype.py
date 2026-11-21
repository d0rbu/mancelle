"""Slice 08 audit (A1, A2 / map.md H1, H2, B4, I12).

Pin the output dtype contract of ``FeatureEncoder.forward``:

* The encoded tensor returned by ``forward(...)[0]`` must be the
  ``COMPUTE_DTYPE`` set in ``nanochat.common`` — the trailing
  ``EncoderBlock.forward: x.to(COMPUTE_DTYPE)`` cast plus the no-op
  ``norm`` and ``reshape`` are supposed to preserve this dtype.
* The mask returned by ``forward(...)[1]`` must be ``torch.bool`` —
  cross-attention's ``attn_mask`` is interpreted as bool.

Both invariants are documented in map.md §3 (I12) and §7.2 (H1, H2).
Pinning them here so any silent dtype drift caused by future refactors
is caught at test time, not in the cross-attention NaN surface.
"""

from __future__ import annotations

import os

# Force SDPA backend before importing torch / encoder so flex_attention is not
# selected for CPU tests (matches the convention used in the existing
# tests/nanochat/test_encoder_block_dtype.py file).
os.environ["NANOCHAT_ATTN_IMPL"] = "sdpa"

import pytest
import torch

from nanochat.common import COMPUTE_DTYPE
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


def _make_inputs(B: int, S: int, F: int, config: FeatureEncoderConfig):
    feature_ids = torch.randint(0, config.n_sae_features, (B, S, F))
    magnitudes = torch.randint(0, config.num_buckets, (B, S, F))
    mask = torch.ones(B, S, F, dtype=torch.bool)
    return feature_ids, magnitudes, mask


class TestEncoderOutputDtype:
    """Output dtype contract: encoder_out is COMPUTE_DTYPE; mask_out is bool."""

    def test_output_dtype_equals_compute_dtype(self):
        """A1 / I12: ``forward(...)[0].dtype == COMPUTE_DTYPE`` for the canonical config."""
        config = _make_config()
        model = FeatureEncoder(config)
        model.init_weights()
        ids, mags, mask = _make_inputs(2, 3, 5, config)
        out = model(ids, mags, mask)
        assert out.values.dtype == COMPUTE_DTYPE, (
            f"FeatureEncoder.forward output dtype {out.values.dtype} does not match "
            f"COMPUTE_DTYPE {COMPUTE_DTYPE}. The trailing EncoderBlock cast "
            "plus norm/reshape must preserve compute dtype for cross-attention."
        )

    def test_output_dtype_with_n_layer_1(self):
        """A1: minimum n_layer=1 still yields COMPUTE_DTYPE."""
        config = _make_config(n_layer=1)
        model = FeatureEncoder(config)
        model.init_weights()
        ids, mags, mask = _make_inputs(1, 1, 4, config)
        out = model(ids, mags, mask)
        assert out.values.dtype == COMPUTE_DTYPE

    def test_output_dtype_with_deep_encoder(self):
        """A1: deeper stacks still yield COMPUTE_DTYPE."""
        config = _make_config(n_layer=4)
        model = FeatureEncoder(config)
        model.init_weights()
        ids, mags, mask = _make_inputs(1, 2, 8, config)
        out = model(ids, mags, mask)
        assert out.values.dtype == COMPUTE_DTYPE

    def test_mask_out_is_bool(self):
        """A2 / H2: packed validity metadata remains boolean."""
        config = _make_config()
        model = FeatureEncoder(config)
        model.init_weights()
        ids, mags, mask = _make_inputs(2, 3, 5, config)
        out = model(ids, mags, mask)
        assert out.layout.valid_mask.dtype == torch.bool

    def test_mask_out_is_bool_with_partial_mask(self):
        """A2: partial input mask -> bool output mask."""
        config = _make_config()
        model = FeatureEncoder(config)
        model.init_weights()
        ids, mags, mask = _make_inputs(1, 1, 5, config)
        mask[0, 0, 3:] = False
        out = model(ids, mags, mask)
        assert out.layout.valid_mask.dtype == torch.bool
        assert out.layout.valid_mask.sum().item() == 3

    @pytest.mark.parametrize("input_id_dtype", [torch.int32, torch.int64])
    def test_output_dtype_invariant_to_input_int_dtype(self, input_id_dtype):
        """A1: encoder output dtype does not depend on the input integer dtype.

        Both ``torch.int32`` and ``torch.int64`` IDs must yield COMPUTE_DTYPE
        outputs (the embeddings cast to fp32 / bf16 internally).
        """
        config = _make_config()
        model = FeatureEncoder(config)
        model.init_weights()
        feature_ids = torch.randint(0, config.n_sae_features, (1, 1, 4), dtype=input_id_dtype)
        magnitudes = torch.randint(0, config.num_buckets, (1, 1, 4), dtype=input_id_dtype)
        mask = torch.ones(1, 1, 4, dtype=torch.bool)
        out = model(feature_ids, magnitudes, mask)
        assert out.values.dtype == COMPUTE_DTYPE
        assert out.layout.valid_mask.dtype == torch.bool
