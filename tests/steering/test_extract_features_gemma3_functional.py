"""Non-network functional routing test for the Gemma-3 / gemma-scope-2
extraction path.

Builds a tiny random ``Gemma3ForCausalLM`` and monkeypatches the two HF
download sites in ``steering.extract_features`` so the REAL production
dispatch runs end to end with NO network:

  build_nnterp_model(model_name="gemma-3-4b-it", ...)
      -> _build_nnterp_gemma3 (real)
          -> load_gemma3_text_decoder (stubbed → tiny model)
          -> resolve_tokenizer        (stubbed → stub tokenizer)
  _get_activations_nnterp(..., attention, hook_name="...hook_attn_out")
      -> post-projection self_attn.output, centrally tuple-unwrapped

This guards the P0-2/3 attention tuple-unwrap fix through the *real*
production routing that the mocked test_ensure_chat_template.py tests skip,
and confirms attention + residual both yield a finite (B,S,d_model)
**Tensor** (not a tuple).
"""

from __future__ import annotations

import pytest
import torch as t
from nnterp import StandardizedTransformer
from transformers import Gemma3ForCausalLM
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

import steering.extract_features as ef
from steering.extract_features import _get_activations_nnterp, build_nnterp_model
from steering.sae_resolver import ActivationType

_TINY_VOCAB = 128
_TINY_HIDDEN = 32
_TINY_LAYERS = 4
_TINY_SEQ = 5


@pytest.fixture(scope="module")
def tiny_gemma3() -> Gemma3ForCausalLM:
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
    return Gemma3ForCausalLM(cfg).eval()


class _StubTokenizer:
    """Minimal stand-in; resolve_tokenizer's return value is unused by the
    nnterp build path beyond the gated-repo early-failure check."""

    pad_token_id = 0


@pytest.fixture()
def patched_build(monkeypatch, tiny_gemma3: Gemma3ForCausalLM):
    def fake_loader(hf_id: str, *, dtype, attn_implementation: str = "sdpa"):
        return tiny_gemma3

    def fake_resolve_tokenizer(_model: str):
        return _StubTokenizer()

    monkeypatch.setattr(ef, "load_gemma3_text_decoder", fake_loader)
    monkeypatch.setattr(ef, "resolve_tokenizer", fake_resolve_tokenizer)


class TestGemma3FunctionalRouting:
    def test_build_then_attention_and_residual_capture(self, patched_build):
        # REAL production dispatch — no network.
        model = build_nnterp_model(
            model_name="gemma-3-4b-it",
            tl_model=None,
            device=t.device("cpu"),
            model_dtype=t.float32,
        )
        assert isinstance(model, StandardizedTransformer)

        ids = t.randint(0, _TINY_VOCAB, (1, _TINY_SEQ))

        # Attention via the post-projection hook_attn_out route. This is the
        # path that returns an (hidden, attn_weights) tuple from the HF
        # self_attn module; the centralized unwrap must yield a Tensor.
        attn = _get_activations_nnterp(
            model,
            ids,
            layer=2,
            activation_type=ActivationType.attention,
            hook_name="blocks.2.hook_attn_out",
        )
        assert isinstance(attn, t.Tensor), (
            f"expected Tensor, got {type(attn)!r} (P0-2/3 unwrap regression)"
        )
        assert tuple(attn.shape) == (1, _TINY_SEQ, _TINY_HIDDEN)
        assert bool(attn.isfinite().all())

        # Residual path through the same production function.
        resid = _get_activations_nnterp(
            model,
            ids,
            layer=2,
            activation_type=ActivationType.residual,
            hook_name="blocks.2.hook_resid_post",
        )
        assert isinstance(resid, t.Tensor)
        assert tuple(resid.shape) == (1, _TINY_SEQ, _TINY_HIDDEN)
        assert bool(resid.isfinite().all())
