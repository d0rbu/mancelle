"""Pin the packed encoder contract at its minimum feature dimension."""

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
        "n_layer": 1,
        "n_head": 4,
        "n_kv_head": 2,
        "head_dim": 8,
    }
    defaults.update(overrides)
    return FeatureEncoderConfig(**defaults)


class TestNfZeroEdgeCase:
    """Nf=0 fails cleanly; Nf=1 produces one real token plus one global tail."""

    def test_nf_zero_does_not_silently_corrupt(self):
        """A3 / H3: ``Nf == 0`` must fail without returning corrupt state."""
        config = _make_config()
        model = FeatureEncoder(config)
        model.init_weights()
        B, S, F = 1, 1, 0
        feature_ids = torch.zeros(B, S, F, dtype=torch.int64)
        magnitudes = torch.zeros(B, S, F, dtype=torch.int64)
        mask = torch.zeros(B, S, F, dtype=torch.bool)

        try:
            model(feature_ids, magnitudes, mask)
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError("Nf=0 must fail cleanly")

    def test_nf_one_uses_one_global_alignment_block(self):
        """A3 control: ``Nf == 1`` produces one valid token in a 32-row state."""
        config = _make_config()
        model = FeatureEncoder(config)
        model.init_weights()
        B, S, F = 1, 1, 1
        ids = torch.zeros(B, S, F, dtype=torch.int64)
        mags = torch.zeros(B, S, F, dtype=torch.int64)
        mask = torch.ones(B, S, F, dtype=torch.bool)
        out = model(ids, mags, mask)
        assert out.values.shape == (1, 32, config.n_embd)
        assert bool(out.layout.valid_mask[0])
        assert not out.layout.valid_mask[1:].any()
