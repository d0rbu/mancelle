"""Functional contract of the production truncated ``capture()``.

Re-pointed for Phase-2 PR-1: the collector is built via
``build_native_model(..., layer)`` (structural truncation) — NOT the removed
``NativeCollector(hf)`` + hook path — using a fresh seeded gemma-2 twin
(truncation mutates in place; the shared conftest factory is reused, not
copy-pasted). Residual is the ONLY supported site; mlp/attention now raise a
typed ``ExtractionError`` ('residual-only') by design (spec §3.2), which this
file asserts as the NEW correct behaviour rather than the removed per-site
capture.
"""

from __future__ import annotations

import pytest
import torch as t

from errors import ExtractionError
from steering.native_collector import NativeCollector
from steering.sae_resolver import ActivationType
from tests.steering.native.conftest import TINY_D_MODEL, _build_tiny_gemma2

_LAYER = 2  # 0 <= 2 < TINY_LAYERS (4)
_RESID_HOOK = f"blocks.{_LAYER}.hook_resid_post"

_DTYPES_FUNCTIONAL = [
    pytest.param(t.float32, id="fp32"),
    pytest.param(t.bfloat16, id="bf16"),
]


def _built(dtype: t.dtype = t.float32):
    col = NativeCollector.truncated(
        # HF `.to()` overload stubs don't model `nn.Module.to(dtype)`; production
        # `native_collector` uses the same idiom and ty-checks clean there.
        _build_tiny_gemma2().to(dtype),  # ty: ignore[invalid-argument-type]
        model_name="gemma-2-2b",
        layer=_LAYER,
    )
    assert col is not None
    return col


def test_residual_shape_dtype_device():
    """residual capture: [B, S, d_model], fp32, no autograd graph.

    (The Phase-1 per-site parametrize over mlp/hook_z/hook_attn_out is gone —
    those are non-residual and now raise; see
    ``test_non_residual_sites_raise_typed`` below.)"""
    col = _built()
    ids = t.randint(0, 64, (3, 7))
    out = col.capture(ids, _LAYER, hook_name=_RESID_HOOK,
                      activation_type=ActivationType.residual)
    assert out.ndim == 3
    assert out.shape == (3, 7, TINY_D_MODEL)
    assert out.dtype == t.float32
    assert out.requires_grad is False
    assert out.device == t.device("cpu")


# G5 — explicit contiguity invariant.
#
# The consumer's `activations.view(-1, d)` (extract_features.py:1229) requires
# the tensor returned by `capture` / `_truncated_forward` to be contiguous. No
# existing test asserts this. This test pins the invariant explicitly at fp32
# AND bf16 (the production dtype). Non-vacuity: if the returned tensor is
# wrapped in a non-contiguous view (e.g. `.permute(0,2,1)`) before returning,
# `is_contiguous()` is False and this assertion FAILS.

@pytest.mark.parametrize("dtype", _DTYPES_FUNCTIONAL)
def test_residual_capture_output_is_contiguous(dtype):
    """capture() output is contiguous — required by consumer's .view(-1, d).

    G5 gap: no existing test asserts contiguity of the capture output. The
    consumer ``_get_sae_activations_batched`` calls ``activations.view(-1, d)``
    (NOT ``.reshape``) immediately after capture; if the tensor is not
    contiguous this raises a RuntimeError in production. ``is_contiguous()``
    must be True at fp32 AND bf16.
    """
    col = _built(dtype)
    ids = t.randint(0, 64, (3, 7))
    out = col.capture(ids, _LAYER, hook_name=_RESID_HOOK,
                      activation_type=ActivationType.residual)
    # The contiguity assertion — the key invariant for the consumer's .view(-1,d)
    assert out.is_contiguous(), (
        f"dtype={dtype}: capture() returned a non-contiguous tensor — "
        "consumer's .view(-1, d) will raise; _truncated_forward must return "
        "a contiguous tensor (e.g. via .detach() on last_hidden_state which "
        "HF models return as a standard contiguous tensor)"
    )
    # Confirm .view(-1, d) actually works (not just that is_contiguous() is True)
    d = out.shape[-1]
    flat = out.view(-1, d)
    assert flat.shape == (3 * 7, d), (
        f"dtype={dtype}: .view(-1, {d}) after capture gave {flat.shape}, "
        f"expected ({3*7}, {d})"
    )
    assert flat.dtype == dtype


@pytest.mark.parametrize(
    ("atype", "hook"),
    [
        (ActivationType.mlp, f"blocks.{_LAYER}.hook_mlp_out"),
        (ActivationType.attention, f"blocks.{_LAYER}.attn.hook_z"),
        (ActivationType.attention, f"blocks.{_LAYER}.hook_attn_out"),
    ],
)
def test_non_residual_sites_raise_typed(atype, hook):
    """The truncated collector is residual-only (spec §3.2): every
    non-residual site raises a typed ExtractionError so the dispatch fallback
    chain degrades exactly as for any other ExtractionError (never a silently
    wrong residual tensor)."""
    col = _built()
    ids = t.randint(0, 64, (3, 7))
    with pytest.raises(ExtractionError, match="residual-only"):
        col.capture(ids, _LAYER, hook_name=hook, activation_type=atype)


def test_batched_equals_sequential_residual():
    """Right-padded batched capture == per-doc unpadded capture (fp32 tight).

    Pins the documented maskless right-pad invariance: a ragged batch
    captured maskless then sliced per true length must match each doc
    captured alone. (fp32 batched-vs-sequential matmul reduction order in the
    real HF Gemma-2 forward is nondeterministic at ~1e-6 — the documented
    tolerance, identical to ``test_dispatch_e2e.py``.)"""
    col = _built()
    docs = [[3, 9, 1, 7, 2], [5, 5, 8], [2, 1, 4, 6]]
    lengths = [len(d) for d in docs]
    max_s = max(lengths)
    padded = t.zeros(len(docs), max_s, dtype=t.long)
    for i, d in enumerate(docs):
        padded[i, : len(d)] = t.tensor(d)
    batched = col.capture(
        padded, _LAYER, hook_name=_RESID_HOOK,
        activation_type=ActivationType.residual,
    )
    for i, d in enumerate(docs):
        single = col.capture(
            t.tensor(d).unsqueeze(0), _LAYER,
            hook_name=_RESID_HOOK,
            activation_type=ActivationType.residual,
        )
        assert t.allclose(
            batched[i, : lengths[i]], single[0], atol=1e-5, rtol=1e-4
        ), f"doc {i} batched != sequential"
