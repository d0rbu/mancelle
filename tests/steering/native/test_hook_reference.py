"""Tests for the TEST-ONLY Phase-1 hook residual reference helper.

These prove `hook_reference_residual` faithfully reproduces the Phase-1
`NativeCollector.capture` residual forward-hook mechanism that PR-1 removes
from production (spec §3.2). Gates §4.1/§4.6 assert the Phase-2 truncated
collector is byte-identical to this reference, so the reference itself must
be verified to BE the hook mechanism — not merely "run".
"""

from __future__ import annotations

import torch as t

from tests.steering.native._hook_reference import hook_reference_residual

_LAYER = 2


def test_hook_reference_shape_dtype_grad(tiny_gemma2):
    """Shape == (B, S, hidden_size); dtype == model param dtype; no grad."""
    ids = t.randint(0, 64, (3, 7))
    out = hook_reference_residual(tiny_gemma2, ids, layer=_LAYER)
    param_dtype = next(tiny_gemma2.parameters()).dtype
    hidden_size = tiny_gemma2.config.hidden_size
    assert out.shape == (3, 7, hidden_size)
    assert out.dtype == param_dtype
    assert out.requires_grad is False


def test_hook_reference_equals_direct_inline_forward_hook(tiny_gemma2):
    """KEY: helper == an independent, inline forward-hook capture.

    This registers a forward hook on `model.model.layers[2]` directly in the
    test (the verbatim Phase-1 mechanism: full forward under no_grad, maskless,
    tuple/list `[0]` unwrap, detach().clone()) and asserts byte-identical
    equality with the helper. If the helper ever drifts from the real hook
    mechanism, this fails — which is what every later gate trusts.
    """
    ids = t.randint(0, 64, (2, 6))

    captured: dict[str, t.Tensor] = {}

    def _inline_hook(_mod, _inp, out):
        tensor = out[0] if isinstance(out, (tuple, list)) else out
        captured["v"] = tensor.detach().clone()

    handle = tiny_gemma2.model.layers[_LAYER].register_forward_hook(_inline_hook)
    try:
        with t.no_grad():
            tiny_gemma2(input_ids=ids, attention_mask=None)
    finally:
        handle.remove()
    expected = captured["v"]

    out = hook_reference_residual(tiny_gemma2, ids, layer=_LAYER)

    assert out.shape == expected.shape
    assert out.dtype == expected.dtype
    assert t.equal(out, expected), "helper diverged from the inline hook capture"


def test_hook_reference_does_not_leak_hooks(tiny_gemma2):
    """The reference must remove its hook (finally:), like Phase-1's capture."""
    ids = t.randint(0, 64, (1, 5))
    hook_reference_residual(tiny_gemma2, ids, layer=_LAYER)
    for lyr in tiny_gemma2.model.layers:
        assert len(lyr._forward_hooks) == 0
