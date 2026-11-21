"""Unit contracts for the production truncated collector + the preserved
Phase-1 hook reference.

Re-pointed for Phase-2 PR-1 (spec §6, Risk #7):

* The detached/deterministic residual contract is now exercised through the
  TRUNCATED ``build_native_model(..., layer)`` + ``capture()`` path (the
  removed ``NativeCollector(hf)`` + forward-hook path is gone).
* ``test_hook_fires_exactly_once_and_is_removed`` genuinely tested the
  *Phase-1 hook mechanism itself* (a registered hook fires once and is
  removed in ``finally``) — that behaviour now lives in the preserved
  verbatim reference ``tests/steering/native/_hook_reference.py``, so the
  test is RETARGETED there (its home moved; the assertion is unchanged).
* ``mlp``/``hook_z`` were the removed hook-capture paths; the truncated
  collector is residual-only by design, so the old "mlp on a model without
  ``.mlp`` raises" / "hook_z preproj reshape" tests are obsolete (they tested
  a removed production mechanism, not preserved in the residual-only
  reference). The replacement is the typed ``residual-only`` ExtractionError,
  asserted here and in ``test_collector_functional.py`` /
  ``test_collector_branch_unit.py``.
"""

from __future__ import annotations

import pytest
import torch as t

from errors import ExtractionError
from steering.native_collector import NativeCollector
from steering.sae_resolver import ActivationType
from tests.steering.native._hook_reference import hook_reference_residual
from tests.steering.native.conftest import (
    TINY_D_MODEL,
    _build_tiny_gemma2,
)

_LAYER = 1  # 0 <= 1 < TINY_LAYERS (4)
_RESID_HOOK = f"blocks.{_LAYER}.hook_resid_post"


def _built():
    col = NativeCollector.truncated(
        _build_tiny_gemma2(), model_name="gemma-2-2b", layer=_LAYER,
    )
    assert col is not None
    return col


def test_capture_returns_detached_clone():
    """``capture(residual)`` returns a DETACHED tensor whose mutation cannot
    corrupt a subsequent capture (no shared storage / no autograd graph)."""
    col = _built()
    ids = t.randint(0, 64, (2, 5))
    out = col.capture(ids, _LAYER, hook_name=_RESID_HOOK,
                      activation_type=ActivationType.residual)
    assert out.shape == (2, 5, TINY_D_MODEL)
    assert out.requires_grad is False
    before = out.clone()
    out.add_(999.0)  # mutate the returned tensor
    out2 = col.capture(ids, _LAYER, hook_name=_RESID_HOOK,
                       activation_type=ActivationType.residual)
    assert t.allclose(out2, before), (
        "mutating a returned capture corrupted a later capture — the result "
        "is not a detached/independent tensor"
    )


def test_hook_reference_fires_exactly_once_and_is_removed():
    """RETARGETED to the preserved Phase-1 hook reference (its new home).

    The reference registers a forward hook on the target decoder layer, runs
    one maskless forward, and removes the handle in ``finally``. A probe hook
    on the same layer must fire EXACTLY once, and NO forward hook may remain
    on ANY layer afterwards (the reference cleans up after itself)."""
    hf = _build_tiny_gemma2()
    target = hf.model.layers[_LAYER]
    calls: list[int] = []
    probe = target.register_forward_hook(lambda *a: calls.append(1))
    try:
        out = hook_reference_residual(hf, t.randint(0, 64, (1, 4)), _LAYER)
    finally:
        probe.remove()
    assert calls == [1], (
        f"target layer forward should fire exactly once, got {calls}"
    )
    assert out.shape == (1, 4, TINY_D_MODEL)
    # the reference's own hook handle must be gone (only our probe remained,
    # which we just removed) — no leaked hooks on any layer.
    for lyr in hf.model.layers:
        assert len(lyr._forward_hooks) == 0


@pytest.mark.parametrize(
    ("atype", "hook"),
    [
        (ActivationType.mlp, f"blocks.{_LAYER}.hook_mlp_out"),
        (ActivationType.attention, f"blocks.{_LAYER}.attn.hook_z"),
    ],
)
def test_non_residual_capture_raises_typed_extraction_error(atype, hook):
    """The removed mlp/hook_z hook-capture paths are now a single typed
    residual-only ExtractionError (spec §3.2) — mlp/attention are
    unsupported by the truncated collector and must fail loudly (never an
    AttributeError, never a silently-wrong tensor)."""
    col = _built()
    with pytest.raises(ExtractionError, match="residual-only"):
        col.capture(t.randint(0, 64, (1, 4)), _LAYER, hook_name=hook,
                    activation_type=atype)
