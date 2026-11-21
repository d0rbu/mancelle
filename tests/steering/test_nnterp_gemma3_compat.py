"""Structural compatibility probe: nnterp.StandardizedTransformer + Gemma3ForCausalLM.

nnterp does name-based renaming (`model.layers`, `model.embed_tokens`, …)
rather than HF-class-based standardisation, so Gemma 3 should "just work"
without architecture-specific code in nnterp. This test verifies that
assumption with a tiny randomly-initialised `Gemma3TextConfig` model — no
HF download, no GPU, ~1s runtime.

Complements the planned T32 live integration test (`gemma-3-270m` real
load + `_verify_logits_match`) by catching structural breakage —
nnterp version bumps, transformers' Gemma 3 module-name churn, etc. —
earlier and cheaper in the CI matrix.

Key signal: the substring routing in
`steering.extract_features._get_activations_nnterp` returns
`nnterp_model.layers_output[layer]` for `hook_resid_post`. This test
confirms that accessor produces finite tensors of shape
`(batch, seq, hidden_size)` for a Gemma 3 decoder.
"""

from __future__ import annotations

import pytest
import torch as t
from nnterp import StandardizedTransformer
from transformers import Gemma3ForCausalLM
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer

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


@pytest.fixture(scope="module")
def tiny_nnterp(tiny_gemma3: Gemma3ForCausalLM) -> StandardizedTransformer:
    return StandardizedTransformer(tiny_gemma3)


class TestNnterpGemma3Compat:
    def test_decoder_layer_class(self, tiny_gemma3: Gemma3ForCausalLM) -> None:
        # nnterp's name-based renaming relies on the standard `.model.layers`
        # path. We additionally pin the class identity to catch upstream
        # rename events (Gemma3DecoderLayer → something else).
        assert isinstance(tiny_gemma3.model.layers[0], Gemma3DecoderLayer)
        assert len(tiny_gemma3.model.layers) == _TINY_LAYERS

    def test_standardized_transformer_wraps_cleanly(
        self, tiny_nnterp: StandardizedTransformer
    ) -> None:
        # No architecture-specific code in nnterp for Gemma 3 — the wrap
        # succeeds iff the rename dictionary recognises the standard
        # module-name layout.
        assert tiny_nnterp is not None

    def test_layers_output_residual_post_shape_and_finite(
        self, tiny_nnterp: StandardizedTransformer
    ) -> None:
        # Mirrors `_get_activations_nnterp`'s residual-post branch:
        # `acts = nnterp_model.layers_output[layer].save()`.
        ids = t.randint(0, _TINY_VOCAB, (1, _TINY_SEQ))
        with tiny_nnterp.trace(ids) as tracer:
            mid = tiny_nnterp.layers_output[_TINY_LAYERS // 2].save()
            tracer.stop()
        assert tuple(mid.shape) == (1, _TINY_SEQ, _TINY_HIDDEN)
        assert bool(mid.isfinite().all())

    def test_layers_output_last_layer_finite(
        self, tiny_nnterp: StandardizedTransformer
    ) -> None:
        # The actual extraction targets layer 29 of `gemma-3-4b-it`,
        # which is its final residual position; confirm the equivalent
        # last-layer index on this tiny model is reachable.
        ids = t.randint(0, _TINY_VOCAB, (1, _TINY_SEQ))
        with tiny_nnterp.trace(ids) as tracer:
            last = tiny_nnterp.layers_output[_TINY_LAYERS - 1].save()
            tracer.stop()
        assert tuple(last.shape) == (1, _TINY_SEQ, _TINY_HIDDEN)
        assert bool(last.isfinite().all())
