"""Slice 08 audit (A5 / map.md H5).

Pin that ``FeatureEncoder.forward`` is deterministic:

The encoder has no dropout, no RNG-dependent ops (no sampling, no random
masking). A regression that introduced dropout would not be caught by
the existing unit tests (they use specific input/seed combinations).
This test pins the property explicitly: same input → same output, bit
identical.
"""

from __future__ import annotations

import os

os.environ["NANOCHAT_ATTN_IMPL"] = "sdpa"

import torch

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


class TestEncoderDeterminism:
    """Forward is deterministic: same input → identical output."""

    def test_forward_deterministic_eval_mode(self):
        config = _make_config()
        model = FeatureEncoder(config)
        model.init_weights()
        model.eval()
        torch.manual_seed(0)
        ids = torch.randint(0, config.n_sae_features, (2, 3, 5))
        mags = torch.randint(0, config.num_buckets, (2, 3, 5))
        mask = torch.ones(2, 3, 5, dtype=torch.bool)
        out1 = model(ids, mags, mask)
        out2 = model(ids, mags, mask)
        assert torch.equal(out1.values, out2.values), (
            "FeatureEncoder.forward is not deterministic in eval mode "
            "(same input → different output). This indicates hidden RNG "
            "(e.g. dropout) was introduced — the slice 8 spec requires no RNG."
        )
        assert torch.equal(out1.layout.valid_mask, out2.layout.valid_mask)
        assert torch.equal(out1.layout.flat_indices, out2.layout.flat_indices)
        assert torch.equal(out1.layout.batch_indices, out2.layout.batch_indices)
        assert torch.equal(out1.layout.segment_indices, out2.layout.segment_indices)

    def test_forward_deterministic_train_mode(self):
        """A5: even in train mode (no dropout), forward is deterministic."""
        config = _make_config()
        model = FeatureEncoder(config)
        model.init_weights()
        model.train()
        ids = torch.randint(0, config.n_sae_features, (1, 2, 4))
        mags = torch.randint(0, config.num_buckets, (1, 2, 4))
        mask = torch.ones(1, 2, 4, dtype=torch.bool)
        out1 = model(ids, mags, mask)
        out2 = model(ids, mags, mask)
        assert torch.equal(out1.values, out2.values)
        assert torch.equal(out1.layout.valid_mask, out2.layout.valid_mask)

    def test_no_dropout_modules(self):
        """A5: explicit structural check — no nn.Dropout instances anywhere."""
        config = _make_config(n_layer=4)
        model = FeatureEncoder(config)
        for name, m in model.named_modules():
            assert not isinstance(m, torch.nn.Dropout), (
                f"FeatureEncoder unexpectedly contains nn.Dropout at {name!r}; "
                "slice 8 spec disallows hidden RNG."
            )
