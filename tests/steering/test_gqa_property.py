"""Slice 08 audit (A7 / map.md H7).

Hypothesis property tests over GQA configs ``(n_head, n_kv_head)``:

The existing ``tests/steering/test_feature_encoder.py::TestGQAConfig`` only
covers three explicit configs (MQA, MHA, half-GQA). A property-based
search over the full valid space catches off-by-one issues in the
``repeat_interleave(n_rep, dim=1)`` expansion and any constraint drift.

Constraints (per map.md §3 I6):
* ``1 <= n_kv_head <= n_head``
* ``n_head % n_kv_head == 0``
"""

from __future__ import annotations

import os

os.environ["NANOCHAT_ATTN_IMPL"] = "sdpa"

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.conditional.encoder_decoder.encoder import (
    FeatureEncoder,
    FeatureEncoderConfig,
)

# Choose head pairs that satisfy n_head % n_kv_head == 0 and 1 <= n_kv_head <= n_head.
_VALID_HEAD_PAIRS = [
    (n_head, n_kv_head)
    for n_head in (1, 2, 4, 8, 16)
    for n_kv_head in (1, 2, 4, 8, 16)
    if 1 <= n_kv_head <= n_head and n_head % n_kv_head == 0
]


def _make_config(n_head: int, n_kv_head: int, head_dim: int = 8, n_layer: int = 1) -> FeatureEncoderConfig:
    # ``n_embd = n_head * head_dim`` must be at least ``ve_gate_channels=32`` because
    # ``EncoderSelfAttention.forward`` slices ``x[..., :ve_gate_channels]`` before
    # feeding into ``ve_gate`` Linear (encoder.py:104). Pin head_dim so n_embd≥32
    # for every (n_head, n_kv_head) explored.
    n_embd = max(n_head * head_dim, 32)
    return FeatureEncoderConfig(
        n_sae_features=64,
        num_buckets=4,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_kv_head,
        head_dim=head_dim,
    )


class TestGQAProperty:
    """Property: any valid (n_head, n_kv_head) pair → forward succeeds with correct shape."""

    @given(pair=st.sampled_from(_VALID_HEAD_PAIRS))
    @settings(max_examples=len(_VALID_HEAD_PAIRS), deadline=None, derandomize=True)
    def test_forward_succeeds_for_valid_gqa(self, pair):
        n_head, n_kv_head = pair
        config = _make_config(n_head=n_head, n_kv_head=n_kv_head)
        model = FeatureEncoder(config)
        model.init_weights()
        ids = torch.randint(0, config.n_sae_features, (1, 1, 4))
        mags = torch.randint(0, config.num_buckets, (1, 1, 4))
        mask = torch.ones(1, 1, 4, dtype=torch.bool)
        out = model(ids, mags, mask)
        assert out.values.shape == (1, 32, config.n_embd), (
            f"GQA pair (n_head={n_head}, n_kv_head={n_kv_head}): "
            f"unexpected output shape {tuple(out.values.shape)} "
            f"(expected n_embd={config.n_embd})"
        )
        assert int(out.layout.valid_mask.sum()) == 4

    @given(pair=st.sampled_from(_VALID_HEAD_PAIRS))
    @settings(max_examples=len(_VALID_HEAD_PAIRS), deadline=None, derandomize=True)
    def test_gradient_flows_for_valid_gqa(self, pair):
        """A7: backward pass works for every valid GQA config."""
        n_head, n_kv_head = pair
        config = _make_config(n_head=n_head, n_kv_head=n_kv_head, n_layer=2)
        model = FeatureEncoder(config)
        model.init_weights()
        # Replace c_proj with a non-zero weight so attention has a non-trivial gradient.
        with torch.no_grad():
            for block in model.blocks:
                torch.nn.init.normal_(block.attn.c_proj.weight, std=0.02)
                torch.nn.init.normal_(block.mlp.c_proj.weight, std=0.02)
        ids = torch.randint(0, config.n_sae_features, (1, 1, 4))
        mags = torch.randint(0, config.num_buckets, (1, 1, 4))
        mask = torch.ones(1, 1, 4, dtype=torch.bool)
        out = model(ids, mags, mask)
        loss = out.values.sum()
        loss.backward()
        # feature_embed.weight must accumulate gradient (cast to fp32 first since
        # COMPUTE_DTYPE bf16 has lossy abs.sum)
        g = model.feature_embed.weight.grad
        assert g is not None, "feature_embed.weight has no grad"
        assert g.float().abs().sum().item() > 0.0, (
            f"feature_embed.weight grad is all-zero for GQA pair "
            f"(n_head={n_head}, n_kv_head={n_kv_head}): silent gradient-flow break."
        )

    @pytest.mark.parametrize("invalid_pair", [(4, 3), (8, 5), (4, 6), (4, 0)])
    def test_invalid_gqa_raises(self, invalid_pair):
        """A7 control: invalid (n_head, n_kv_head) pairs are rejected.

        Encoder asserts ``n_kv_head <= n_head`` and ``n_head % n_kv_head == 0``.
        Note: ``EncoderSelfAttention.__init__`` validates this (encoder.py:53);
        the assert fires on FeatureEncoder construction (which builds blocks).
        """
        n_head, n_kv_head = invalid_pair
        config = _make_config(n_head=n_head, n_kv_head=n_kv_head)
        with pytest.raises((AssertionError, ZeroDivisionError)):
            FeatureEncoder(config)
