"""Regression: `_get_activations_nnterp` must honor its `-> t.Tensor` contract.

The gemma-scope-2 attention SAE path routes to
``nnterp_model.model.layers[L].self_attn.output``, which HF Gemma3
attention returns as a ``(hidden_states, attn_weights)`` tuple — not a
Tensor. Two ``extract_features`` consumers unwrapped it manually, but
``steering.eval.probe_activations.collect``-style callers did not, so a
v2 attention feature via the probe path got a 2-tuple where the contract
promises ``(batch, seq, d_model)``.

This test exercises the real attention ``hook_attn_out`` branch through
``_get_activations_nnterp`` and asserts a Tensor is returned (centralized
unwrap). Uses the same tiny randomly-initialised ``Gemma3ForCausalLM``
fixture pattern as ``test_nnterp_gemma3_compat.py`` — no HF download, no
GPU, ~1s. Runs in the default suite.
"""

from __future__ import annotations

import pytest
import torch as t
from nnterp import StandardizedTransformer
from transformers import Gemma3ForCausalLM
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

from steering.extract_features import _get_activations_nnterp
from steering.sae_resolver import ActivationType

_TINY_VOCAB = 128
_TINY_HIDDEN = 32
_TINY_LAYERS = 4
_TINY_SEQ = 5
_LAYER = _TINY_LAYERS // 2


@pytest.fixture(scope="module")
def tiny_nnterp() -> StandardizedTransformer:
    cfg = Gemma3TextConfig(
        vocab_size=_TINY_VOCAB,
        hidden_size=_TINY_HIDDEN,
        intermediate_size=64,
        num_hidden_layers=_TINY_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=128,
    )
    model = Gemma3ForCausalLM(cfg).eval()
    return StandardizedTransformer(model)


def test_attention_hook_attn_out_returns_tensor(
    tiny_nnterp: StandardizedTransformer,
) -> None:
    """gemma-scope-2 attention path: self_attn.output is a tuple upstream;
    the function must centralize the unwrap and return a 3D Tensor.
    """
    ids = t.randint(0, _TINY_VOCAB, (1, _TINY_SEQ))
    result = _get_activations_nnterp(
        nnterp_model=tiny_nnterp,
        tokens=ids,
        layer=_LAYER,
        activation_type=ActivationType.attention,
        hook_name=f"blocks.{_LAYER}.hook_attn_out",
    )
    assert isinstance(result, t.Tensor), (
        f"expected Tensor per `-> t.Tensor` contract, got {type(result)}"
    )
    assert result.dim() == 3, f"expected 3D [batch, seq, d_model], got {result.dim()}D"
    assert result.shape[-1] == _TINY_HIDDEN, (
        f"expected last dim == hidden_size {_TINY_HIDDEN}, got {result.shape[-1]}"
    )


def test_residual_path_still_returns_tensor(
    tiny_nnterp: StandardizedTransformer,
) -> None:
    """No regression: the residual branch already returned a Tensor."""
    ids = t.randint(0, _TINY_VOCAB, (1, _TINY_SEQ))
    result = _get_activations_nnterp(
        nnterp_model=tiny_nnterp,
        tokens=ids,
        layer=_LAYER,
        activation_type=ActivationType.residual,
        hook_name=f"blocks.{_LAYER}.hook_resid_post",
    )
    assert isinstance(result, t.Tensor)
    assert result.dim() == 3
    assert result.shape[-1] == _TINY_HIDDEN
