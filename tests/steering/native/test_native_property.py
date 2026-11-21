"""Property test: truncated residual capture == HF reference layer output.

Re-pointed for Phase-2 PR-1: a FRESH truncated collector is built per
``layer`` example via ``build_native_model(..., layer)`` (the collector is
structurally truncated for ONE fixed layer, so the swept ``layer`` selects
the truncation depth) and compared against an independent, NON-truncated HF
twin's ``output_hidden_states``. The removed ``NativeCollector(hf)`` + hook
path is gone; this asserts the production truncated mechanism instead, with
the same property the Phase-1 test asserted (capture == HF
``hidden_states[L+1]`` for non-final layers; determinism across re-runs).
"""

from __future__ import annotations

import torch as t
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.native_collector import NativeCollector
from steering.sae_resolver import ActivationType
from tests.steering.native.conftest import (  # seeded-twin factory (reused)
    TINY_LAYERS,
    TINY_VOCAB,
    _build_tiny_gemma2,
)


@settings(max_examples=50, deadline=None)
@given(
    seq_len=st.integers(min_value=1, max_value=16),
    layer=st.integers(min_value=0, max_value=TINY_LAYERS - 1),
    batch=st.integers(min_value=1, max_value=3),
)
def test_native_residual_matches_hf_layer_output(seq_len, layer, batch):
    """Truncated residual capture == HF reference forward layer output.

    A fresh seeded twin is truncated at ``layer`` and driven through the
    production ``capture()``; an independent fresh twin (identical weights,
    NOT truncated) provides the reference. ``resid_post(L)`` is the layer-L
    decoder output BEFORE the final norm, i.e. HF ``hidden_states[L+1]`` for
    every NON-final layer (for the final layer the reference applies the
    final norm to the last entry, so we instead assert determinism — two
    captures byte-identical)."""
    col = NativeCollector.truncated(
        _build_tiny_gemma2(), model_name="gemma-2-2b", layer=layer,
    )
    assert col is not None
    ids = t.randint(0, TINY_VOCAB, (batch, seq_len))
    hook = f"blocks.{layer}.hook_resid_post"

    out1 = col.capture(ids, layer, hook_name=hook,
                       activation_type=ActivationType.residual)
    out2 = col.capture(ids, layer, hook_name=hook,
                       activation_type=ActivationType.residual)
    assert t.equal(out1, out2), "truncated capture must be deterministic"

    if layer < TINY_LAYERS - 1:
        ref_model = _build_tiny_gemma2()  # identical weights, NOT truncated
        with t.no_grad():
            ref = ref_model(input_ids=ids, output_hidden_states=True)
        assert t.allclose(out1, ref.hidden_states[layer + 1], atol=1e-5), (
            f"truncated layer {layer} != HF hidden_states[{layer + 1}]"
        )
