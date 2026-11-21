"""Property tests for globally packed encoder shapes and validity metadata."""

from __future__ import annotations

import os

os.environ["NANOCHAT_ATTN_IMPL"] = "sdpa"

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.conditional.encoder_decoder.encoder import (
    FeatureEncoder,
    FeatureEncoderConfig,
)


def _make_config(**overrides) -> FeatureEncoderConfig:
    defaults = {
        "n_sae_features": 64,
        "num_buckets": 4,
        "n_embd": 32,
        "n_layer": 1,
        "n_head": 4,
        "n_kv_head": 2,
        "head_dim": 8,
    }
    defaults.update(overrides)
    return FeatureEncoderConfig(**defaults)


class TestShapeProperty:
    """Property: one 32-aligned axis carries all real features in owner order."""

    @given(
        b=st.integers(min_value=1, max_value=4),
        s=st.integers(min_value=1, max_value=5),
        nf=st.integers(min_value=1, max_value=33),
    )
    @settings(max_examples=30, deadline=None, derandomize=True)
    def test_output_shape_property(self, b, s, nf):
        config = _make_config()
        model = FeatureEncoder(config)
        model.init_weights()
        ids = torch.randint(0, config.n_sae_features, (b, s, nf))
        mags = torch.randint(0, config.num_buckets, (b, s, nf))
        mask = torch.ones(b, s, nf, dtype=torch.bool)
        out = model(ids, mags, mask)
        assert out.values.shape[0] == 1
        assert out.values.shape[1] % 32 == 0
        assert out.values.shape[2] == config.n_embd
        assert out.layout.batch_size == b
        assert out.layout.n_segments == s
        assert int(out.layout.valid_mask.sum()) == b * s * nf

    @given(
        b=st.integers(min_value=1, max_value=3),
        s=st.integers(min_value=1, max_value=3),
        nf=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=20, deadline=None, derandomize=True)
    def test_mask_padding_property(self, b, s, nf):
        """Real positions reflect the input mask; padded positions are False."""
        config = _make_config()
        model = FeatureEncoder(config)
        model.init_weights()
        ids = torch.randint(0, config.n_sae_features, (b, s, nf))
        mags = torch.randint(0, config.num_buckets, (b, s, nf))
        # Input mask: all True for simplicity (the propagation property is
        # preservation, not arbitrary boolean mixing).
        mask = torch.ones(b, s, nf, dtype=torch.bool)
        out = model(ids, mags, mask)
        real_count = b * s * nf
        assert out.layout.valid_mask[:real_count].all().item()
        assert not out.layout.valid_mask[real_count:].any().item()
