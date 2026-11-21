"""Fast in-process e2e test of the REAL ``_get_sae_activations_batched``
native dispatch path.

Closes a robustness gap independently flagged by two audits: the production
native chain
``_get_sae_activations_batched`` → ``_get_activations_native`` →
``NativeCollector.capture`` is otherwise exercised in-process ONLY via
mocks (``test_dispatch_unit.py`` / ``test_side_channel_unit.py`` monkeypatch
``_get_activations_native``). The only REAL end-to-end native dispatch test
is the slow GPU-gated Layer-4 test, which does not run in default CI.

This test runs the *real* dispatch chain with real tensors on a tiny CPU
Gemma-2 model and a tiny real (deterministic linear+ReLU) SAE — nothing on
the native path is mocked. It is non-vacuous: it actually runs
``NativeCollector`` capture through the production dispatch function and
verifies the per-doc results equal a direct per-doc capture, that
right-padding is stripped per true length, and that the native branch
genuinely produced real (non-zero) activations rather than silently
short-circuiting.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch as t
from sae_lens import SAE

import steering.extract_features as ef
from steering.native_collector import NativeCollector
from steering.sae_resolver import ActivationType
from tests.steering.native.conftest import TINY_D_MODEL, _build_tiny_gemma2

_N_FEATURES = 24
_HOOK = "blocks.2.hook_resid_post"
_LAYER = 2


class _TinySAE(t.nn.Module):
    """Minimal real SAE matching the surface ``_get_sae_activations_batched``
    touches: ``next(sae.parameters()).device`` and
    ``sae.encode(flat_acts)`` with ``flat_acts`` shaped ``[N, d_model]`` →
    ``[N, n_features]``.

    ``encode`` is a real deterministic ``Linear(d_model, n_features)``
    followed by ReLU so the output is non-negative (matching the real SAE
    post-ReLU encode contract) and the cross-path comparison is meaningful
    (not a constant). The dispatch function does NOT call ``.to()``,
    ``.cfg``, or any other attribute on the SAE, so no further surface is
    added.
    """

    def __init__(self, d_model: int, n_features: int) -> None:
        super().__init__()
        with t.random.fork_rng():
            t.manual_seed(20260517)
            self.enc = t.nn.Linear(d_model, n_features, bias=True)
        self.eval()

    def encode(self, x: t.Tensor) -> t.Tensor:
        with t.no_grad():
            return t.relu(self.enc(x))


def _make_sae() -> _TinySAE:
    return _TinySAE(TINY_D_MODEL, _N_FEATURES).eval()


_DOCS = [[3, 9, 1, 7, 2], [5, 5, 8], [2, 1, 4, 6]]
_LENGTHS = [len(d) for d in _DOCS]


def test_get_sae_activations_batched_native_e2e():
    """Real ``_get_sae_activations_batched`` native path, end-to-end.

    No mocking of ``_get_activations_native`` / ``NativeCollector.capture``.
    The collector is the PRODUCTION truncated build (``build_native_model``)
    from a fresh seeded gemma-2 twin (truncation mutates in place; shared
    conftest factory reused). ``model=`` is an unused placeholder on the
    native path (``_get_sae_activations_batched`` routes straight to
    ``_get_activations_native(native_model, ...)`` when ``native_model`` is
    given — verified at extract_features.py:1159) — pass the truncated inner
    model itself so nothing extraneous is constructed.
    """
    col = NativeCollector.truncated(
        _build_tiny_gemma2(), model_name="gemma-2-2b", layer=_LAYER,
    )
    assert col is not None
    sae = _make_sae()
    # _get_sae_activations_batched is annotated `sae: SAE` but on the native
    # path it only ever touches `sae.parameters()` and `sae.encode(...)`,
    # which _TinySAE provides for real. cast() bridges the duck-typed test
    # double to the nominal annotation without changing runtime behavior.
    sae_arg = cast(SAE, sae)

    # ``model`` is annotated HookedTransformer but is UNUSED on the native
    # path (native_model given → routed straight to _get_activations_native,
    # verified extract_features.py:1159); the truncated inner model is a
    # harmless placeholder. cast(Any, ...) bridges the duck-typed double to
    # the nominal annotation without runtime effect (an honest ``Any`` — the
    # value is never touched on this path, so no real HookedTransformer
    # surface is required and importing TL into this fast test is avoided).
    model_arg = cast(Any, col.hf_model)

    out = ef._get_sae_activations_batched(
        model=model_arg,
        sae=sae_arg,
        token_id_lists=_DOCS,
        layer=_LAYER,
        native_model=col,
        hook_name=_HOOK,
        activation_type=ActivationType.residual,
    )

    # --- Assertion 1: production dispatch correctness (non-vacuous) ---
    assert isinstance(out, list)
    assert len(out) == len(_DOCS)
    for i, length in enumerate(_LENGTHS):
        assert out[i].shape == (length, _N_FEATURES), (
            f"doc {i}: padding-strip wrong — got {tuple(out[i].shape)}, "
            f"expected ({length}, {_N_FEATURES})"
        )
        assert out[i].dtype == t.float32
        assert out[i].requires_grad is False
        assert bool((out[i] >= 0).all()), f"doc {i}: SAE encode not post-ReLU"

    # --- Assertion 3: native actually ran (no silent all-zero / short-circuit) ---
    # tiny_gemma2 has no TL/nnterp support; an all-zero result would mean the
    # native branch never produced real activations through the dispatch.
    assert any(o.abs().sum() > 0 for o in out), (
        "native dispatch produced an all-zero result — native branch did not "
        "run real activations through _get_sae_activations_batched"
    )

    # --- Assertion 2: adapter path == direct per-doc capture ---
    # The native branch calls capture(padded_batch, ...) maskless then strips
    # padding to per-true-length. A direct per-doc capture (single doc, no
    # padding, no mask) must reproduce each stripped row: same model, same
    # SAE, the batched dispatch must yield identical per-doc results. This
    # pins that _get_sae_activations_batched's native branch + padding-strip
    # is correct (not just mocked).
    #
    # Exact (torch.equal) was tried first but FAILS for the right-padded
    # shorter docs: batched-vs-sequential fp32 matmul reduction order in the
    # real HF Gemma-2 forward is nondeterministic (max abs diff ~2e-6;
    # test_collector_functional.py needs the same tolerance for the identical
    # tiny_gemma2 batched-vs-sequential comparison). Fall back to a TIGHT
    # tolerance — far below any real activation magnitude.
    for i, doc in enumerate(_DOCS):
        direct_acts = col.capture(
            t.tensor(doc).unsqueeze(0),
            _LAYER,
            hook_name=_HOOK,
            activation_type=ActivationType.residual,
        )[0]
        direct = sae.encode(direct_acts)
        assert t.allclose(direct, out[i], atol=1e-5, rtol=1e-4), (
            f"doc {i}: batched native dispatch != direct per-doc capture "
            f"beyond tight fp32 tolerance "
            f"(max abs diff {float((direct - out[i]).abs().max()):.3e})"
        )


# G1 — bf16 consumer-contract: the production dtype path through the real dispatch chain.
#
# The existing test above exercises fp32 only. Production extractions run bf16. This
# test adds a fast in-process bf16 regression guard: builds a bf16 collector + bf16
# SAE, runs the real _get_sae_activations_batched dispatch chain, and verifies the
# same invariants (shape, padding-strip, non-zero presence, batched≈direct-per-doc).
# bf16-vs-bf16 comparison uses a bf16-appropriate tolerance (NOT vs fp32 reference).
#
# The SAE encoder weights are also cast to bf16 (the dispatch chain receives bf16
# activations from the bf16 model and passes them to sae.encode — the SAE's Linear
# must be bf16 too or torch matmul silently upcasts in a way that masks the test).

class _TinySAEDtype(t.nn.Module):
    """Like _TinySAE but the linear weights are cast to `dtype` at construction.

    Required for the bf16 e2e: the dispatch chain feeds bf16 activations directly
    to sae.encode (the model is bf16, capture returns bf16, view→encode passes it
    through). The Linear must match so the output is bf16 (production dtype).
    """

    def __init__(self, d_model: int, n_features: int, dtype: t.dtype) -> None:
        super().__init__()
        with t.random.fork_rng():
            t.manual_seed(20260517)
            self.enc = t.nn.Linear(d_model, n_features, bias=True)
        self.enc = self.enc.to(dtype)
        self.eval()

    def encode(self, x: t.Tensor) -> t.Tensor:
        with t.no_grad():
            return t.relu(self.enc(x))


_E2E_DTYPES = [
    pytest.param(t.float32, id="fp32"),
    pytest.param(t.bfloat16, id="bf16"),
]


@pytest.mark.parametrize("dtype", _E2E_DTYPES)
def test_get_sae_activations_batched_native_e2e_dtype(dtype):
    """Real ``_get_sae_activations_batched`` native dispatch, fp32 AND bf16.

    G1 gap: the existing ``test_get_sae_activations_batched_native_e2e`` tests
    only fp32; production extractions run bf16. This parametrized sibling runs
    the same REAL dispatch chain (no mocking) at both dtypes so the bf16
    consumer path (capture → .view(-1,d) → sae.encode → per-length strip) has
    a fast in-process regression guard on CPU.

    For bf16, comparison is bf16-native-vs-bf16-direct (NOT bf16-vs-fp32) with
    a bf16-appropriate tolerance (~1e-2 rtol). Shape, padding-strip, non-zero
    presence, and positive-only (post-ReLU) invariants hold at both dtypes.
    """
    col = NativeCollector.truncated(
        _build_tiny_gemma2().to(dtype), model_name="gemma-2-2b", layer=_LAYER,
    )
    assert col is not None
    sae = _TinySAEDtype(TINY_D_MODEL, _N_FEATURES, dtype).eval()
    sae_arg = cast(SAE, sae)
    model_arg = cast(Any, col.hf_model)

    out = ef._get_sae_activations_batched(
        model=model_arg,
        sae=sae_arg,
        token_id_lists=_DOCS,
        layer=_LAYER,
        native_model=col,
        hook_name=_HOOK,
        activation_type=ActivationType.residual,
    )

    # --- shape, dtype, padding-strip ---
    assert isinstance(out, list)
    assert len(out) == len(_DOCS)
    for i, length in enumerate(_LENGTHS):
        assert out[i].shape == (length, _N_FEATURES), (
            f"dtype={dtype} doc {i}: padding-strip wrong — got {tuple(out[i].shape)}, "
            f"expected ({length}, {_N_FEATURES})"
        )
        assert out[i].dtype == dtype, (
            f"dtype={dtype} doc {i}: output dtype {out[i].dtype} != {dtype}"
        )
        assert out[i].requires_grad is False
        assert bool((out[i] >= 0).all()), (
            f"dtype={dtype} doc {i}: SAE encode output not non-negative (post-ReLU)"
        )

    # --- non-zero: native actually ran ---
    assert any(o.abs().sum() > 0 for o in out), (
        f"dtype={dtype}: native dispatch produced all-zero — branch did not fire"
    )

    # --- batched ≈ direct per-doc (same dtype, same model, same SAE) ---
    # bf16-appropriate tolerance: bf16 accumulation adds ~0.5% relative error per op.
    # Use rtol=2e-2, atol=5e-3 — far below any real activation magnitude (~0.1–10).
    _atol: float = 5e-3 if dtype == t.bfloat16 else 1e-5
    _rtol: float = 2e-2 if dtype == t.bfloat16 else 1e-4
    for i, doc in enumerate(_DOCS):
        direct_acts = col.capture(
            t.tensor(doc).unsqueeze(0),
            _LAYER,
            hook_name=_HOOK,
            activation_type=ActivationType.residual,
        )[0]
        direct = sae.encode(direct_acts)
        assert t.allclose(direct, out[i], atol=_atol, rtol=_rtol), (
            f"dtype={dtype} doc {i}: batched dispatch != direct per-doc capture "
            f"(max abs diff {float((direct - out[i]).abs().max()):.3e})"
        )
