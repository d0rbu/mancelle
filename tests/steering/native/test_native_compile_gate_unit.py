"""Unit tests for the ideal-math-relative gated torch.compile path.

Two groups:

* ``test_compile_gate_error_hierarchy`` — the typed-error contract.
* ``_native_compile_gate`` behaviour — the runtime self-verifying compile
  gate. These use the conftest tiny gemma twins (CPU, no download), a tiny
  linear+ReLU ``_SAE`` (same shape contract as
  ``tests/steering/native/test_dispatch_unit.py``'s), and control the
  compiled-vs-eager divergence by monkeypatching ``native_model._compiled_base``
  to a behaviour-preserving wrapper that perturbs the truncated-forward output
  by a CONTROLLABLE epsilon — so the resulting SAE-feature delta can be made to
  land deterministically INSIDE or OUTSIDE the ideal-math-relative gate
  (real-model correctness is the slow gate in test_native_integration.py).

The gate's pass/fail decision MUST be the IDENTICAL predicate
``compiled_passes_ideal_math_gate`` (the production gate calls this and
nothing else). The unit tests pin both the predicate semantics directly AND
the gate's three-feature-set wiring (fp32 ref, bf16 baseline, compiled).
"""
from __future__ import annotations

import ast
import gc as _gc
import weakref
from pathlib import Path

import loguru
import pytest
import torch as t
import torch._inductor.config as _inductor_config

import steering.extract_features as ef
from errors import CompileGateError, ExtractionError, NanoGPTError

# The compile gate uses the production predicate below. The separate
# feature-equivalence acceptance oracle is test-only.
from steering.extract_features import (
    _build_compile_probe_ids,
    _format_compile_gate_envelope,
    _gate_sparse_features,
    _native_compile_gate,
)
from steering.feature_equivalence import (
    BASELINE_RELATIVE_SLACK,
    compiled_passes_ideal_math_gate,
)
from steering.sae_resolver import ActivationType
from tests.steering.native._feature_equivalence_oracles import (
    feature_equivalence_passed,
)
from tests.steering.native._parity_thresholds import (
    BF16_NATIVE_VS_TL_MAG_ATOL,
    BF16_NATIVE_VS_TL_MAG_RTOL,
    FP32_NATIVE_VS_TL_MAG_ATOL,
    FP32_NATIVE_VS_TL_MAG_RTOL,
)
from tests.steering.native.conftest import _build_tiny_gemma2
from tests.steering.native.conftest_parity import assert_feature_equivalent


def _extract_features_source() -> str:
    assert ef.__file__ is not None
    return Path(ef.__file__).read_text()


def _extract_features_func_def(name: str) -> ast.FunctionDef:
    tree = ast.parse(_extract_features_source())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found in steering.extract_features")


def test_compile_gate_error_hierarchy():
    assert issubclass(CompileGateError, ExtractionError)
    assert issubclass(CompileGateError, NanoGPTError)
    assert not issubclass(ExtractionError, CompileGateError)


# --------------------------------------------------------------------------- #
# Shared tiny harness for the _native_compile_gate behaviour tests.            #
# --------------------------------------------------------------------------- #

# 0 <= 2 < TINY_LAYERS (4): a real interior truncation (mirrors the other
# native compile/collector tests' `_LAYER`).
_LAYER = 2
_RESID_HOOK = f"blocks.{_LAYER}.hook_resid_post"

# A ragged right-padded probe (row 1 has 2 trailing pads) so the gate's probe
# exercises the documented maskless / right-pad / per-length-strip contract on
# a variable-length batch (the unit-level analogue of the production probe).
_PROBE_IDS = t.tensor([[3, 9, 1, 7, 2], [5, 5, 8, 0, 0]], dtype=t.long)


class _SAE:
    """Tiny deterministic linear+ReLU SAE (frozen, fp32 or bf16).

    Same shape contract as test_dispatch_unit.py's ``_SAE`` (``parameters`` +
    ``encode``); ``encode`` maps ``[*, d] -> [*, f]`` so the gate's
    ``_get_sae_activations_batched``-mirrored flatten/encode/strip/max-pool
    transform produces real, non-vacuous per-segment (ids, mags). Deterministic
    seeded weights ⇒ a fixed truncated-forward perturbation maps to a fixed,
    reproducible SAE-feature delta (so a threshold can be straddled exactly).
    """

    def __init__(self, d: int = 32, f: int = 48) -> None:
        with t.random.fork_rng():
            t.manual_seed(1234)
            self._w = t.nn.Parameter(t.randn(d, f), requires_grad=False)
            self._b = t.nn.Parameter(t.randn(f), requires_grad=False)
        self.d, self.f = d, f

    def parameters(self):
        yield self._w
        yield self._b

    def to(self, dtype):
        self._w = t.nn.Parameter(self._w.to(dtype), requires_grad=False)
        self._b = t.nn.Parameter(self._b.to(dtype), requires_grad=False)
        return self

    def encode(self, x: t.Tensor) -> t.Tensor:
        return t.relu(x.float().matmul(self._w.float()) + self._b.float())


class _CastRecordingSAE(_SAE):
    """An ``_SAE`` that records every dtype passed to ``.to`` (so a test can
    pin the gate's fp32-then-back cast contract without monkeypatching the
    method — keeps ty clean: no attribute-reassignment of a bound method)."""

    def __init__(self, d: int = 32, f: int = 48) -> None:
        super().__init__(d, f)
        self.cast_history: list[t.dtype] = []

    def to(self, dtype):
        self.cast_history.append(dtype)
        return super().to(dtype)


class _RaiseOnFp32SAE(_SAE):
    """An ``_SAE`` whose ``.to(torch.float32)`` raises (every other dtype
    works) — pins the gate's "SAE cannot fp32-cast" failure mode without
    monkeypatching a bound method (keeps ty clean)."""

    def to(self, dtype):
        if dtype == t.float32:
            raise AttributeError("test: SAE class does not support .to()")
        return super().to(dtype)


class _PerturbCompiled:
    """Behaviour-preserving stand-in for the compiled base, +epsilon.

    Delegates to the wrapped eager base (so the "compiled" output is the real
    eager output) then ADDS ``eps`` to ``last_hidden_state``. eps=0 ⇒ compiled
    ≡ eager (gate must PASS); a large eps ⇒ the SAE features diverge from BOTH
    the ideal-math ref AND the bf16-eager baseline past the slack (gate must
    FAIL). The gate never compiles in unit tests — this deterministic CPU twin
    replaces ``native_model._compiled_base`` so the pass/fail boundary is
    exercised exactly without real inductor.
    """

    def __init__(self, base, eps: float) -> None:
        self._base = base
        self._eps = eps
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        out = self._base(*args, **kwargs)
        out.last_hidden_state = out.last_hidden_state + self._eps
        return out


class _ProbeThrew(RuntimeError):
    """A unique sentinel raised by a swapped base so the test can prove the
    EXACT failure propagated out of ``_native_compile_gate`` (not some other
    incidental error). ``capture()`` wraps non-``ExtractionError`` exceptions in
    an ``ExtractionError`` whose message embeds ``type(e).__name__: {e}``
    (native_collector.py:339-343), so a raised ``_ProbeThrew("<tag>")`` surfaces
    as an ``ExtractionError`` whose ``str`` contains ``_ProbeThrew: <tag>`` —
    the unambiguous propagation marker the throw-path tests match on.
    """


class _ThrowingCompiled:
    """A base stand-in that runs the real eager base except it raises
    ``_ProbeThrew`` on the call whose 1-based index is in ``throw_on``.

    The gate calls ``capture()`` multiple times (once on the fp32 ref, once on
    the bf16 baseline, once on the compiled candidate). ``throw_on=(1,)`` ⇒
    ONLY the first call through this object raises and every later call
    delegates normally — so a test can make EXACTLY the targeted call throw
    with the OTHERS served faithfully (isolating which path's failure is
    asserted).
    """

    def __init__(self, base, *, throw_on: tuple[int, ...] = (1,)) -> None:
        self._base = base
        self._throw_on = set(throw_on)
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls in self._throw_on:
            raise _ProbeThrew(f"compiled probe boom #{self.calls}")
        return self._base(*args, **kwargs)

    def modules(self):
        # Faithful delegation to the WRAPPED base (a real nn.Module): any
        # caller that walks submodules sees the real model's structure, not
        # this throwing wrapper's empty one.
        return self._base.modules()

    def parameters(self):
        # Faithful delegation: the gate's ``native_model.device`` property
        # calls ``next(self._base.parameters()).device``. When _base is a
        # wrapper, the call must still see the real model's parameters or
        # the device probe fails with AttributeError before the intended
        # throw fires.
        return self._base.parameters()


def _fresh_collector():
    """A fresh truncated gemma-2 collector (mutating build ⇒ fresh twin)."""
    nc = ef.NativeCollector.truncated(
        _build_tiny_gemma2(), model_name="gemma-2-2b", layer=_LAYER
    )
    assert nc is not None
    return nc


def _patch_build_ref(monkeypatch) -> list[dict]:
    """Patch ``ef.build_native_model`` (which the gate calls for the fp32
    reference build) to return a fresh tiny gemma-2 truncated collector — no
    HF download, deterministic, byte-identical weights to the test's
    `_fresh_collector` baseline. Records each call's args so tests can pin the
    contract (``(model_name, device, model_dtype=t.float32, layer)``).

    The tiny twin is built in fp32 (it always is in this conftest), so this
    faithfully stands in for the production fp32-eager reference build.
    Returns the per-call recorder list.
    """
    calls: list[dict] = []

    def _fake_build(model_name, device, model_dtype, layer):
        calls.append({
            "model_name": model_name,
            "device": device,
            "model_dtype": model_dtype,
            "layer": layer,
        })
        # Tiny twins are always fp32 — that's the correct ref dtype regardless
        # of model_dtype the gate requests.
        return ef.NativeCollector.truncated(
            _build_tiny_gemma2(), model_name=model_name, layer=layer
        )

    monkeypatch.setattr(ef, "build_native_model", _fake_build)
    return calls


def _arm_with_eps(nc, eps: float, monkeypatch) -> _PerturbCompiled:
    """Arm the gate the way ``_native_compile_gate`` will (so its
    ``enable_compiled()`` is idempotent), swap ``_compiled_base`` for the
    deterministic +eps twin, AND patch ``ef.build_native_model`` to return a
    fresh tiny gemma-2 twin (the fp32 reference build). Returns the +eps twin
    so a test can inspect ``calls``.
    """
    _patch_build_ref(monkeypatch)
    nc.enable_compiled()
    twin = _PerturbCompiled(nc._base, eps)
    nc._compiled_base = twin
    return twin


def _capture_eager(nc, ids):
    return nc.capture(
        ids, _LAYER, hook_name=_RESID_HOOK,
        activation_type=ActivationType.residual,
    )


# --------------------------------------------------------------------------- #
# Test 1: pass (compiled == eager) ⇒ compiled ACTIVATED + info logged.         #
# Under the ideal-math-relative gate, eps=0 ⇒ compiled byte-identical to       #
# baseline ⇒ same jaccard vs ref ⇒ margin=slack>=0 ⇒ PASS.                     #
# --------------------------------------------------------------------------- #


def test_gate_pass_activates_compiled(monkeypatch, caplog):
    """eps=0 ⇒ compiled ≡ baseline ⇒ j_compiled_vs_ref == j_baseline_vs_ref
    ⇒ margin = slack >= 0 ⇒ PASS ⇒ the gate ACTIVATES the compiled path
    (``_use_compiled is True``) and emits a single ``[compile-gate]`` INFO
    line carrying the four documented measurements (j_baseline_vs_ref,
    j_compiled_vs_ref, slack, margin).

    Mutation-probe target: inverting the pass branch (activate on FAIL, leave
    eager on PASS) flips ``_use_compiled`` to False here ⇒ this test fails.
    """
    sae = _SAE()
    nc = _fresh_collector()
    twin = _arm_with_eps(nc, eps=0.0, monkeypatch=monkeypatch)

    sink: list[str] = []
    handler_id = loguru.logger.add(sink.append, level="INFO")
    try:
        _native_compile_gate(
            nc, sae, probe_input_ids=_PROBE_IDS,  # ty: ignore[invalid-argument-type]
            model_dtype=t.float32,
        )
    finally:
        loguru.logger.remove(handler_id)

    assert nc._use_compiled is True, (
        "compiled ≡ baseline (eps=0) ⇒ margin >= 0 ⇒ the gate MUST PASS and "
        "ACTIVATE the compiled path (_use_compiled True)"
    )
    assert twin.calls >= 1, (
        "the gate must have probed the compiled base at least once"
    )
    text = "".join(sink)
    assert "[compile-gate]" in text, (
        f"pass must emit a tagged [compile-gate] line; got: {text!r}"
    )
    # All four documented measurements MUST be present in the log line.
    for key in (
        "j_baseline_vs_ref", "j_compiled_vs_ref", "slack", "margin"
    ):
        assert key in text, (
            f"the [compile-gate] line must carry {key!r}; got {text!r}"
        )


# --------------------------------------------------------------------------- #
# Fail path: gate-fail ⇒ CompileGateError, collector eager after the raise.    #
# There is no opt-out (no `force` param, no eager-fallback) — every fail       #
# raises.                                                                      #
# --------------------------------------------------------------------------- #


def test_gate_fail_raises_compile_gate_error(monkeypatch):
    """A large eps diverges the compiled features past the ideal-math
    reference further than baseline does — by more than SLACK — ⇒ the gate
    raises ``CompileGateError`` carrying the four measurements
    (j_baseline_vs_ref, j_compiled_vs_ref, slack, margin), AND after the
    raise ``_use_compiled is False`` (the collector is NEVER left
    compiled-but-unverified, even on the abort path).

    Mutation-probe target: removing the step-4 finally restore (leaving it
    active) flips the post-raise ``_use_compiled is False`` assertion.
    """
    sae = _SAE()
    nc = _fresh_collector()
    _arm_with_eps(nc, eps=50.0, monkeypatch=monkeypatch)

    with pytest.raises(CompileGateError) as excinfo:
        _native_compile_gate(
            nc, sae, probe_input_ids=_PROBE_IDS,  # ty: ignore[invalid-argument-type]
            model_dtype=t.float32,
        )

    msg = str(excinfo.value)
    for key in (
        "j_baseline_vs_ref", "j_compiled_vs_ref", "slack", "margin"
    ):
        assert key in msg, (
            f"CompileGateError must carry {key!r}; got {msg!r}"
        )
    assert nc._use_compiled is False, (
        "even on the abort path the collector must NOT be left "
        "compiled-active (no compiled-but-unverified state)"
    )


# --------------------------------------------------------------------------- #
# The "no compiled-but-unverified state" invariant: after a gate-fail-raise,   #
# the collector is eager AND a subsequent capture() is byte-identical to the   #
# eager truncated forward (NOT the +eps compiled twin).                        #
# --------------------------------------------------------------------------- #


def test_gate_never_leaves_compiled_active_on_fail(monkeypatch):
    """After a gate-fail raise, ``_use_compiled is False`` AND a subsequent
    ``capture()`` is byte-identical to the eager truncated forward (NOT the
    +eps twin) — the gate never leaves the collector compiled-but-unverified.
    """
    sae2 = _SAE()
    nc2 = _fresh_collector()
    twin2 = _arm_with_eps(nc2, eps=50.0, monkeypatch=monkeypatch)
    with pytest.raises(CompileGateError):
        _native_compile_gate(
            nc2, sae2, probe_input_ids=_PROBE_IDS,  # ty: ignore[invalid-argument-type]
            model_dtype=t.float32,
        )
    assert nc2._use_compiled is False
    calls_after_gate2 = twin2.calls
    eager_ref2 = nc2._base(input_ids=_PROBE_IDS, attention_mask=None) \
        .last_hidden_state.detach()
    got2 = _capture_eager(nc2, _PROBE_IDS)
    assert twin2.calls == calls_after_gate2, (
        "post-(forced-fail) capture() routed through the compiled twin — "
        "left compiled-but-unverified after a forced abort"
    )
    assert t.equal(got2, eager_ref2), (
        "post-(forced-fail) capture() is NOT the eager truncated forward"
    )


# --------------------------------------------------------------------------- #
# The gate's three-feature-set wiring: build_native_model is called once with  #
# t.float32 for the ideal-math reference, and the resulting features drive the #
# predicate. This is the dtype-aware portion of the new gate (dtype-awareness  #
# moved out of magnitude tols into the ref-construction).                      #
# --------------------------------------------------------------------------- #


def test_gate_builds_fp32_reference_exactly_once_with_correct_args(monkeypatch):
    """The gate calls ``build_native_model`` EXACTLY once for the ideal-math
    reference with ``model_dtype=t.float32`` (regardless of the production
    dtype passed to the gate). Pins the dtype-awareness of the new gate: the
    reference is ALWAYS fp32 (ideal-math) even when the production cell is
    bf16, because the entire premise of the gate is to anchor against the
    ideal-math reference.

    Mutation-probe target: in ``_native_compile_gate``, skipping the fp32
    reference build (using baseline as ref) makes ``calls`` empty here ⇒
    this test fails.
    """
    sae = _SAE()
    nc = _fresh_collector()
    calls = _patch_build_ref(monkeypatch)
    nc.enable_compiled()
    nc._compiled_base = _PerturbCompiled(nc._base, eps=0.0)

    _native_compile_gate(
        nc, sae, probe_input_ids=_PROBE_IDS,  # ty: ignore[invalid-argument-type]
        model_dtype=t.bfloat16,  # production dtype is bf16 but ref must be fp32
    )

    assert len(calls) == 1, (
        f"the gate must build the fp32 reference EXACTLY once, got "
        f"{len(calls)} calls: {calls}"
    )
    call = calls[0]
    assert call["model_dtype"] is t.float32, (
        f"the ideal-math reference MUST be fp32 regardless of the production "
        f"dtype; got model_dtype={call['model_dtype']}"
    )
    assert call["model_name"] == nc.model_name, (
        f"reference build must use the same model_name as the collector "
        f"({nc.model_name}); got {call['model_name']}"
    )
    assert call["layer"] == nc.truncated_layer, (
        f"reference build must use the same truncated_layer; "
        f"got layer={call['layer']} vs nc.truncated_layer={nc.truncated_layer}"
    )


def test_gate_casts_sae_to_fp32_for_reference_capture(monkeypatch):
    """The gate must cast the SAE to ``torch.float32`` for the ideal-math
    reference capture, then cast it BACK to the production ``model_dtype``
    before the baseline+compiled probes. Pins the SAE-dtype contract of the
    new gate (the production SAE arrives in production dtype and the gate
    temporarily borrows it for the ref capture).
    """
    _patch_build_ref(monkeypatch)

    sae = _CastRecordingSAE()
    sae.to(t.bfloat16)  # start in bf16 (production dtype)
    sae.cast_history.clear()  # ignore the setup cast; record only gate casts
    nc = _fresh_collector()
    nc.enable_compiled()
    nc._compiled_base = _PerturbCompiled(nc._base, eps=0.0)

    _native_compile_gate(
        nc, sae, probe_input_ids=_PROBE_IDS,  # ty: ignore[invalid-argument-type]
        model_dtype=t.bfloat16,
    )

    assert sae.cast_history == [t.float32, t.bfloat16], (
        f"the SAE must be cast to fp32 for the ref capture then back to the "
        f"production dtype (bfloat16) before the baseline/compiled probes; "
        f"got cast history {sae.cast_history}"
    )


# --------------------------------------------------------------------------- #
# Failure modes — fp32 reference build / SAE .to / empty ref features.         #
# --------------------------------------------------------------------------- #


def test_gate_fp32_ref_build_failure_raises_compile_gate_error(monkeypatch):
    """If ``build_native_model`` raises (e.g. OOM at load time), the gate
    must convert it to ``CompileGateError`` with a clear message; collector
    stays EAGER (no compiled-but-unverified)."""
    def _raise(*a, **k):
        raise RuntimeError("CUDA out of memory: tried to allocate 4 GB")

    monkeypatch.setattr(ef, "build_native_model", _raise)
    sae = _SAE()
    nc = _fresh_collector()
    nc.enable_compiled()
    nc._compiled_base = _PerturbCompiled(nc._base, eps=0.0)

    with pytest.raises(CompileGateError, match="ideal-math gate FAILED to build fp32 reference"):
        _native_compile_gate(
            nc, sae, probe_input_ids=_PROBE_IDS,  # ty: ignore[invalid-argument-type]
            model_dtype=t.bfloat16,
        )
    assert nc._use_compiled is False


def test_gate_fp32_ref_returns_none_raises_compile_gate_error(monkeypatch):
    """If ``build_native_model`` returns None (out-of-scope architecture),
    the gate must raise ``CompileGateError`` with a clear message; collector
    stays EAGER."""
    monkeypatch.setattr(ef, "build_native_model", lambda *a, **k: None)
    sae = _SAE()
    nc = _fresh_collector()
    nc.enable_compiled()
    nc._compiled_base = _PerturbCompiled(nc._base, eps=0.0)

    with pytest.raises(CompileGateError, match="build_native_model returned None"):
        _native_compile_gate(
            nc, sae, probe_input_ids=_PROBE_IDS,  # ty: ignore[invalid-argument-type]
            model_dtype=t.bfloat16,
        )
    assert nc._use_compiled is False


def test_gate_sae_fp32_cast_failure_raises_compile_gate_error(monkeypatch):
    """If the SAE class does not support ``.to(torch.float32)`` (e.g. its
    ``.to`` raises), the gate must raise ``CompileGateError`` with a clear
    message; collector stays EAGER."""
    _patch_build_ref(monkeypatch)
    sae = _RaiseOnFp32SAE()
    nc = _fresh_collector()
    nc.enable_compiled()
    nc._compiled_base = _PerturbCompiled(nc._base, eps=0.0)

    with pytest.raises(
        CompileGateError,
        match="SAE class .* does not support .to.torch.float32.",
    ):
        _native_compile_gate(
            nc, sae, probe_input_ids=_PROBE_IDS,  # ty: ignore[invalid-argument-type]
            model_dtype=t.bfloat16,
        )
    assert nc._use_compiled is False


def test_gate_restores_sae_dtype_when_ref_capture_throws(monkeypatch):
    """If the fp32-reference ``capture()`` raises (OOM is the docstring-
    anticipated failure mode for the gate's most memory-heavy op), the
    production SAE — cast IN-PLACE to fp32 at gate start (``sae.to`` is
    in-place; ``ref_sae`` IS ``sae``) — MUST be restored to ``model_dtype``
    before the exception propagates. Otherwise a caller that catches the
    gate's raise would reuse a SILENTLY-fp32 SAE (the latent bug the
    adversarial review found).

    Mutation-probe: deleting the ``finally: sae.to(model_dtype)`` in
    ``_native_compile_gate``'s step-1 ref capture leaves ``cast_history ==
    [float32]`` (no restore) ⇒ this test FAILS on the last-cast assertion.
    """
    class _RefCaptureThrows:
        """Real tiny collector for everything EXCEPT ``capture()``, which
        raises — simulates the fp32-ref OOM after the SAE was cast to fp32."""

        def __init__(self, inner) -> None:
            object.__setattr__(self, "_inner", inner)

        def capture(self, *a, **k):
            raise RuntimeError(
                "CUDA out of memory (simulated fp32 ref capture OOM)"
            )

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_inner"), name)

    def _fake_build(model_name, device, model_dtype, layer):
        inner = ef.NativeCollector.truncated(
            _build_tiny_gemma2(), model_name=model_name, layer=layer
        )
        return _RefCaptureThrows(inner)

    monkeypatch.setattr(ef, "build_native_model", _fake_build)
    sae = _CastRecordingSAE()
    nc = _fresh_collector()
    nc.enable_compiled()
    nc._compiled_base = _PerturbCompiled(nc._base, eps=0.0)

    # The raw OOM propagates (the capture failure is NOT wrapped — only the
    # build/SAE-cast failures are); the point of THIS test is the dtype
    # restoration on that propagation, not the exception type.
    with pytest.raises(RuntimeError, match="out of memory"):
        _native_compile_gate(
            nc, sae, probe_input_ids=_PROBE_IDS,  # ty: ignore[invalid-argument-type]
            model_dtype=t.bfloat16,
        )

    assert sae.cast_history and sae.cast_history[-1] == t.bfloat16, (
        f"production SAE dtype was NOT restored after the fp32-ref capture "
        f"threw: cast_history={sae.cast_history} (the last cast MUST be "
        f"model_dtype=bfloat16 — the in-place fp32 cast leaked otherwise)"
    )
    assert nc._use_compiled is False


class _EmptyRefSAE:
    """An SAE that emits a never-positive feature vector ⇒ ``_gate_sparse_features``
    yields EMPTY (ids, mags) per segment ⇒ the gate's non-vacuity guard rejects.

    Used to certify the "reference produced no features → cannot decide" path
    raises a clear CompileGateError instead of falling through to a garbage
    predicate decision.
    """
    d, f = 32, 48

    def parameters(self):
        return iter([])

    def to(self, dtype):
        return self

    def encode(self, x: t.Tensor) -> t.Tensor:
        return t.full((x.shape[0], self.f), -1.0)


def test_gate_empty_ref_features_raises_compile_gate_error(monkeypatch):
    """If the fp32-eager reference produces NO features (every segment empty),
    the gate must raise ``CompileGateError`` (cannot decide without a
    non-empty reference) — never silently fall through to a vacuous
    predicate decision."""
    _patch_build_ref(monkeypatch)
    sae = _EmptyRefSAE()
    nc = _fresh_collector()
    nc.enable_compiled()
    nc._compiled_base = _PerturbCompiled(nc._base, eps=0.0)

    with pytest.raises(
        CompileGateError,
        match="fp32-eager reference produced no features",
    ):
        _native_compile_gate(
            nc, sae, probe_input_ids=_PROBE_IDS,  # ty: ignore[invalid-argument-type]
            model_dtype=t.float32,
        )
    assert nc._use_compiled is False


# --------------------------------------------------------------------------- #
# Memory safety: the fp32 reference is freed (del + gc.collect +               #
# empty_cache) BEFORE the bf16 baseline/compiled probes run, so the transient  #
# fp32 stack does not co-resident the production bf16 stack longer than the    #
# ref capture. Robustness requirement 3 of the spec.                           #
# --------------------------------------------------------------------------- #


def test_gate_frees_fp32_reference_before_baseline_probes(monkeypatch):
    """The gate must explicitly free the fp32 reference (del + gc.collect +
    empty_cache) AFTER the ref capture and BEFORE the bf16 baseline/compiled
    probes — so the transient fp32 stack does not co-resident the production
    bf16 stack while the (memory-heavy) compiled graph is being built.

    Pins the EARLINESS (CPU-runnable, no GPU): at the moment the gate computes
    the bf16 BASELINE sparse features (step-3, which runs AFTER the step-1 ref
    teardown), the fp32 reference collector must ALREADY be dead. We track a
    ``weakref`` to the ref collector the gate builds, hook
    ``_gate_sparse_features`` to record weakref-liveness at each call, and
    assert the reference is dead by the time the SECOND call (the baseline)
    runs. Mutation-probe: removing the ``del ref_native, ref_sae, ref_acts`` +
    ``gc.collect()`` line keeps the captured ``ref_acts`` (and through it the
    fp32 model output graph) alive past the baseline capture ⇒ the recorded
    liveness at the baseline call is ALIVE ⇒ this test fails.

    The real GPU-watermark check is the slow arm
    ``test_compile_gate_frees_fp32_reference_memory``; this is its
    CPU-runnable earliness complement.
    """
    ref_weakrefs: list[weakref.ref] = []

    def _tracking_build(model_name, device, model_dtype, layer):
        built = ef.NativeCollector.truncated(
            _build_tiny_gemma2(), model_name=model_name, layer=layer
        )
        ref_weakrefs.append(weakref.ref(built))
        return built

    monkeypatch.setattr(ef, "build_native_model", _tracking_build)

    # Record, at each _gate_sparse_features call, whether the fp32 reference
    # collector is still alive. Call 1 = ref capture (alive — it just built
    # it). Call 2 = baseline (MUST be dead — step-1 freed it). Call 3 =
    # compiled.
    liveness_at_call: list[bool] = []
    real_sparse = ef._gate_sparse_features

    def _tracking_sparse(sae, acts, lengths):
        # collect first so a still-collectable ref is reclaimed before we read
        # liveness (the production `del` + gc.collect already ran for the
        # baseline call; this defeats only the test-frame's incidental pins).
        _gc.collect()
        alive = bool(ref_weakrefs and ref_weakrefs[0]() is not None)
        liveness_at_call.append(alive)
        return real_sparse(sae, acts, lengths)

    monkeypatch.setattr(ef, "_gate_sparse_features", _tracking_sparse)

    sae = _SAE()
    nc = _fresh_collector()
    nc.enable_compiled()
    nc._compiled_base = _PerturbCompiled(nc._base, eps=0.0)

    _native_compile_gate(
        nc, sae, probe_input_ids=_PROBE_IDS,  # ty: ignore[invalid-argument-type]
        model_dtype=t.float32,
    )
    assert nc._use_compiled is True  # the gate ran to completion (eps=0 PASS)
    assert len(ref_weakrefs) == 1, (
        f"the gate must have built exactly one fp32 reference; got "
        f"{len(ref_weakrefs)}"
    )
    # Three sparse-feature computations: ref (call 1), baseline (call 2),
    # compiled (call 3).
    assert len(liveness_at_call) == 3, (
        f"expected 3 _gate_sparse_features calls (ref, baseline, compiled); "
        f"got {len(liveness_at_call)}: {liveness_at_call}"
    )
    # By the BASELINE call (index 1) the fp32 reference must ALREADY be freed.
    assert liveness_at_call[1] is False, (
        "the fp32 reference collector was STILL ALIVE when the gate computed "
        "the bf16 BASELINE features — the step-1 teardown (`del ref_native, "
        "ref_sae, ref_acts` + gc.collect()) did NOT free it before the "
        "memory-heavy baseline/compiled probes ran; a gated gemma-3-1b "
        "extraction would co-resident the ~4 GB fp32 stack with the compiled "
        "graph build (the exact memory regression this pins)"
    )


# --------------------------------------------------------------------------- #
# Single-source proof: feature_equivalence_passed(...) is byte-equivalent to   #
# (assert_feature_equivalent does-not-raise) — over a representative battery   #
# incl. the bf16-ULP envelope and just-inside / just-outside straddles for     #
# jaccard, count, AND magnitude. This is what makes "three copies → one        #
# predicate" provably true for the slow Layer-4 ORACLE (the gate now uses a    #
# different predicate; this still keeps the asserter contract).                #
# --------------------------------------------------------------------------- #


def _asserter_passes(a_ids, a_mags, b_ids, b_mags, **tols) -> bool:
    """True iff conftest_parity.assert_feature_equivalent does NOT raise."""
    try:
        assert_feature_equivalent(a_ids, a_mags, b_ids, b_mags, **tols)
        return True
    except AssertionError:
        return False


# (a_ids, a_mags, b_ids, b_mags, tol-kwargs, human label). Each case is a
# concrete arithmetic pin; the straddles bracket every gate at ±1 unit.
_PREDICATE_BATTERY = [
    # --- exact match: trivially passes both ---
    (
        [t.tensor([1, 2, 3])], [t.tensor([1.0, 2.0, 3.0])],
        [t.tensor([1, 2, 3])], [t.tensor([1.0, 2.0, 3.0])],
        {"mag_rtol": 1e-2, "mag_atol": 1e-5,
             "jaccard_threshold": 0.99, "count_rtol": 0.01},
        "exact-match",
    ),
    # --- bf16-ULP envelope: huge feat |Δ|=16 + small feat |Δ|=3, jaccard 1.0.
    #     FAILS the tight fp32 mag tol, PASSES the loosened bf16 mag tol. ---
    (
        [t.tensor([7, 42])], [t.tensor([1312.0, 7.0])],
        [t.tensor([7, 42])], [t.tensor([1328.0, 4.0])],
        {"mag_rtol": FP32_NATIVE_VS_TL_MAG_RTOL,
             "mag_atol": FP32_NATIVE_VS_TL_MAG_ATOL,
             "jaccard_threshold": 0.99, "count_rtol": 0.01},
        "bf16-envelope @ fp32-tol (must FAIL both)",
    ),
    (
        [t.tensor([7, 42])], [t.tensor([1312.0, 7.0])],
        [t.tensor([7, 42])], [t.tensor([1328.0, 4.0])],
        {"mag_rtol": BF16_NATIVE_VS_TL_MAG_RTOL,
             "mag_atol": BF16_NATIVE_VS_TL_MAG_ATOL,
             "jaccard_threshold": 0.99, "count_rtol": 0.01},
        "bf16-envelope @ bf16-tol (must PASS both)",
    ),
    # --- jaccard straddle: union 201, share 199 ⇒ jaccard=199/201≈0.99005.
    #     threshold just-below (0.99 ⇒ PASS) vs just-above (0.991 ⇒ FAIL). ---
    (
        [t.arange(0, 200)], [t.ones(200)],
        [t.arange(1, 201)], [t.ones(200)],
        {"mag_rtol": 1e-2, "mag_atol": 1e-5,
             "jaccard_threshold": 0.99, "count_rtol": 0.01},
        "jaccard=0.99005 just-INSIDE 0.99 (PASS)",
    ),
    (
        [t.arange(0, 200)], [t.ones(200)],
        [t.arange(1, 201)], [t.ones(200)],
        {"mag_rtol": 1e-2, "mag_atol": 1e-5,
             "jaccard_threshold": 0.991, "count_rtol": 0.01},
        "jaccard=0.99005 just-OUTSIDE 0.991 (FAIL)",
    ),
    # --- count straddle: 100 vs 99 ⇒ rdiff=1/100=0.01. rtol just-at 0.01
    #     (<=, PASS) vs just-below 0.009 (FAIL). ---
    (
        [t.arange(100)], [t.ones(100)],
        [t.arange(99)], [t.ones(99)],
        {"mag_rtol": 1e-2, "mag_atol": 1e-5,
             "jaccard_threshold": 0.0, "count_rtol": 0.01},
        "count_rdiff=0.01 == rtol 0.01 (PASS, <=)",
    ),
    (
        [t.arange(100)], [t.ones(100)],
        [t.arange(99)], [t.ones(99)],
        {"mag_rtol": 1e-2, "mag_atol": 1e-5,
             "jaccard_threshold": 0.0, "count_rtol": 0.009},
        "count_rdiff=0.01 > rtol 0.009 (FAIL)",
    ),
    # --- magnitude straddle: one shared feature |Δ|=0.5 on |b|=1.0; with
    #     rtol=0 the gate is atol-only. atol just-above 0.5 (PASS) vs
    #     just-below 0.5 (FAIL). ---
    (
        [t.tensor([1, 2])], [t.tensor([1.0, 2.0])],
        [t.tensor([1, 2])], [t.tensor([1.5, 2.0])],
        {"mag_rtol": 0.0, "mag_atol": 0.5001,
             "jaccard_threshold": 0.99, "count_rtol": 0.01},
        "mag |Δ|=0.5 <= atol 0.5001 (PASS)",
    ),
    (
        [t.tensor([1, 2])], [t.tensor([1.0, 2.0])],
        [t.tensor([1, 2])], [t.tensor([1.5, 2.0])],
        {"mag_rtol": 0.0, "mag_atol": 0.4999,
             "jaccard_threshold": 0.99, "count_rtol": 0.01},
        "mag |Δ|=0.5 > atol 0.4999 (FAIL)",
    ),
    # --- empty-common short-circuit: disjoint ids but equal COUNT ⇒
    #     count_rdiff=0 yet jaccard=0; with jaccard_threshold=0 the
    #     structural gates pass and magnitude is SKIPPED (no common) ⇒ PASS
    #     even though magnitudes are wildly different. Pins the short-circuit. -
    (
        [t.tensor([1, 2])], [t.tensor([9.0, 9.0])],
        [t.tensor([7, 8])], [t.tensor([1.0, 1.0])],
        {"mag_rtol": 0.0, "mag_atol": 0.0,
             "jaccard_threshold": 0.0, "count_rtol": 0.01},
        "disjoint, equal-count, jth=0 ⇒ magnitude SKIPPED (PASS)",
    ),
    # --- segment-count mismatch ⇒ predicate False / asserter raises. ---
    (
        [t.tensor([1])], [t.tensor([1.0])],
        [t.tensor([1]), t.tensor([2])], [t.tensor([1.0]), t.tensor([2.0])],
        {"mag_rtol": 1e-2, "mag_atol": 1e-5,
             "jaccard_threshold": 0.99, "count_rtol": 0.01},
        "segment-count mismatch (predicate False / asserter raises)",
    ),
]


@pytest.mark.parametrize(
    "a_ids,a_mags,b_ids,b_mags,tols,label",
    _PREDICATE_BATTERY,
    ids=[c[-1] for c in _PREDICATE_BATTERY],
)
def test_feature_equivalence_passed_equals_asserter(
    a_ids, a_mags, b_ids, b_mags, tols, label
):
    """``feature_equivalence_passed(...) == (assert_feature_equivalent does
    NOT raise)`` for every case in the battery.

    This is THE single-source proof for the Layer-4 native-vs-TL oracle: it
    pins that the slow asserter's boolean and the legacy
    ``feature_equivalence_passed`` predicate are the SAME function (not two
    hand-duplicated copies that can drift). The runtime COMPILE gate now uses
    a different predicate (``compiled_passes_ideal_math_gate``); see the new
    predicate tests below.
    """
    predicate = feature_equivalence_passed(
        a_ids, a_mags, b_ids, b_mags, **tols
    )
    asserter = _asserter_passes(a_ids, a_mags, b_ids, b_mags, **tols)
    assert predicate == asserter, (
        f"[{label}] feature_equivalence_passed={predicate} but "
        f"assert_feature_equivalent-does-not-raise={asserter} — the "
        f"single-source predicate and the asserter have DRIFTED"
    )


def test_feature_equivalence_passed_used_by_both_gate_and_asserter():
    """Belt-and-braces: the SAME (ids, mags) inputs, run through the legacy
    predicate and through the asserter, agree — using a case that exercises
    the magnitude gate (the most drift-prone branch). The runtime compile
    gate's predicate is exercised separately below."""
    a_ids = [t.tensor([1, 2, 3])]
    a_mags = [t.tensor([1.0, 2.0, 3.0])]
    b_ids = [t.tensor([1, 2, 3])]
    b_mags = [t.tensor([1.0, 2.0, 3.6])]  # |Δ|=0.6 on feat 3
    tight = {"mag_rtol": 1e-2, "mag_atol": 1e-5,
                 "jaccard_threshold": 0.99, "count_rtol": 0.01}
    loose = {"mag_rtol": 0.0, "mag_atol": 1.0,
                 "jaccard_threshold": 0.99, "count_rtol": 0.01}
    # tight ⇒ both FAIL; loose ⇒ both PASS.
    assert feature_equivalence_passed(a_ids, a_mags, b_ids, b_mags, **tight) \
        is False
    assert _asserter_passes(a_ids, a_mags, b_ids, b_mags, **tight) is False
    assert feature_equivalence_passed(a_ids, a_mags, b_ids, b_mags, **loose) \
        is True
    assert _asserter_passes(a_ids, a_mags, b_ids, b_mags, **loose) is True


# --------------------------------------------------------------------------- #
# NEW predicate tests for compiled_passes_ideal_math_gate.                     #
# These exercise the gate's pure decision logic without any model/SAE.         #
# Each test pins one mutation target on the predicate (see the docstrings).    #
# --------------------------------------------------------------------------- #


def _ids_with_jaccard(j: float) -> list[t.Tensor]:
    """Build a single-segment id list whose feature set, when compared against
    a reference of ``{0..99}``, gives an EXACT min-jaccard of approximately
    ``j``. Uses the union/intersection formula so the predicate sees the
    target jaccard verbatim (within rounding).

    We need ``|A ∩ R| / |A ∪ R| = j`` where R = {0..99} (100 features).
    For the simplest construction, take A = {0..n-1} for some n. Then
    |A ∩ R| = min(n, 100) = n (n<=100), |A ∪ R| = max(n, 100) = 100. So
    jaccard = n/100. For arbitrary jaccard, scale: choose n such that
    n/100 ≈ j.
    """
    n = round(j * 100)
    return [t.arange(0, n, dtype=t.int32)]


def _ref_ids() -> list[t.Tensor]:
    return [t.arange(0, 100, dtype=t.int32)]


def test_predicate_compiled_better_than_baseline_passes():
    """j_compiled > j_baseline ⇒ compiled is CLOSER to ref than baseline.
    margin > slack ⇒ PASS regardless of absolute jaccard.

    Mutation-probe target: changing the predicate to use the structural
    feature_equivalence_passed (absolute threshold) would reject this case
    when both jaccards are below 0.99 ⇒ test fails. Test concretely uses
    j_baseline=0.50, j_compiled=0.80 (both below the legacy 0.99 threshold,
    but compiled is much closer to ref).
    """
    ref_ids = _ref_ids()
    baseline_ids = _ids_with_jaccard(0.50)  # |A|=50, jaccard=50/100=0.50
    compiled_ids = _ids_with_jaccard(0.80)  # |A|=80, jaccard=80/100=0.80

    passed, m = compiled_passes_ideal_math_gate(
        ref_ids=ref_ids,
        baseline_ids=baseline_ids,
        compiled_ids=compiled_ids,
    )
    assert m["j_baseline_vs_ref"] == 0.50
    assert m["j_compiled_vs_ref"] == 0.80
    assert m["margin"] == 0.80 - (0.50 - BASELINE_RELATIVE_SLACK)
    assert m["margin"] > 0
    assert passed is True, (
        f"compiled jaccard 0.80 > baseline 0.50 ⇒ margin "
        f"{m['margin']:.6f} > 0 ⇒ MUST PASS regardless of absolute level"
    )


def test_predicate_compiled_tied_with_baseline_passes():
    """j_compiled == j_baseline ⇒ margin = slack >= 0 ⇒ PASS.

    Mutation-probe target: changing ``>=`` to ``>`` in the predicate would
    fail this case only when slack==0; with the default slack>0 the
    inequality direction is still pinned by the boundary test below.
    """
    ref_ids = _ref_ids()
    same = _ids_with_jaccard(0.85)

    passed, m = compiled_passes_ideal_math_gate(
        ref_ids=ref_ids,
        baseline_ids=same,
        compiled_ids=same,
    )
    assert m["j_baseline_vs_ref"] == m["j_compiled_vs_ref"] == 0.85
    assert m["margin"] == pytest.approx(BASELINE_RELATIVE_SLACK)
    assert passed is True


def test_predicate_compiled_slightly_worse_within_slack_passes():
    """j_compiled = j_baseline - slack/2 ⇒ margin = slack/2 > 0 ⇒ PASS.

    Mutation-probe target: changing the predicate's slack default to 0
    (effectively removing the SLACK band) makes ANY worse-than-baseline
    compiled config fail this case ⇒ test fails.
    """
    # Use a 1/10000 grid so baseline - slack/2 is exactly representable.
    n_ref = 10000
    ref_ids = [t.arange(0, n_ref, dtype=t.int32)]
    baseline_ids = [t.arange(0, 9000, dtype=t.int32)]
    # compiled = baseline - slack/2 = 0.9000 - 0.0025 = 0.8975 → 8975 ids
    compiled_ids = [t.arange(0, 8975, dtype=t.int32)]

    # Explicit slack=0.005 (this test constructs compiled at baseline-slack/2
    # for THAT slack); pinning it keeps this predicate-logic test decoupled
    # from the production BASELINE_RELATIVE_SLACK constant, which is exercised
    # separately by the constant-using tests.
    passed, m = compiled_passes_ideal_math_gate(
        ref_ids=ref_ids,
        baseline_ids=baseline_ids,
        compiled_ids=compiled_ids,
        slack=0.005,
    )
    assert m["j_baseline_vs_ref"] == pytest.approx(0.9, abs=1e-6)
    assert m["j_compiled_vs_ref"] == pytest.approx(0.8975, abs=1e-6)
    # margin = j_compiled - (j_baseline - slack) = 0.8975 - 0.895 = 0.0025
    assert m["margin"] == pytest.approx(0.0025, abs=1e-6)
    assert m["margin"] > 0
    assert passed is True


def test_predicate_compiled_at_slack_boundary():
    """j_compiled == j_baseline - slack ⇒ margin == 0 ⇒ PASS (>=, not >).

    Mutation-probe target: changing ``>=`` to ``>`` in the predicate makes
    this exact-boundary case FAIL ⇒ test fails. This is the key test that
    pins the inequality direction.
    """
    # Use a 1/1000 grid for an exact slack=0.005 boundary.
    n_ref = 1000
    ref_ids = [t.arange(0, n_ref, dtype=t.int32)]
    # baseline j = 0.900; compiled = baseline - 0.005 = 0.895 → 895 ids
    baseline_ids = [t.arange(0, 900, dtype=t.int32)]
    compiled_ids = [t.arange(0, 895, dtype=t.int32)]

    passed, m = compiled_passes_ideal_math_gate(
        ref_ids=ref_ids,
        baseline_ids=baseline_ids,
        compiled_ids=compiled_ids,
        slack=0.005,
    )
    assert m["j_baseline_vs_ref"] == pytest.approx(0.9, abs=1e-9)
    assert m["j_compiled_vs_ref"] == pytest.approx(0.895, abs=1e-9)
    # margin = 0.895 - (0.9 - 0.005) = 0 exactly
    assert m["margin"] == pytest.approx(0.0, abs=1e-9)
    assert passed is True, (
        "j_compiled == j_baseline - slack ⇒ margin == 0 ⇒ MUST PASS "
        "(predicate is >=, not >). Mutation: change >= to > and this "
        "test fails."
    )


def test_predicate_compiled_just_below_slack_fails():
    """j_compiled == j_baseline - slack - 1e-6 ⇒ margin < 0 ⇒ FAIL.

    Symmetric counterpart to the boundary test above: just barely below the
    slack admits a FAIL. Pins the inequality direction from the OTHER side.
    """
    # Use a 1/10000 grid so we can have a fine 1e-4 step below the boundary.
    n_ref = 10000
    ref_ids = [t.arange(0, n_ref, dtype=t.int32)]
    # baseline j = 9000/10000 = 0.9000
    baseline_ids = [t.arange(0, 9000, dtype=t.int32)]
    # compiled = baseline - slack - 0.0001 = 0.9 - 0.005 - 0.0001 = 0.8949
    # → 8949 ids
    compiled_ids = [t.arange(0, 8949, dtype=t.int32)]

    passed, m = compiled_passes_ideal_math_gate(
        ref_ids=ref_ids,
        baseline_ids=baseline_ids,
        compiled_ids=compiled_ids,
        slack=0.005,
    )
    assert m["j_baseline_vs_ref"] == pytest.approx(0.9, abs=1e-9)
    assert m["j_compiled_vs_ref"] == pytest.approx(0.8949, abs=1e-9)
    # margin = 0.8949 - 0.895 = -0.0001 < 0
    assert m["margin"] == pytest.approx(-0.0001, abs=1e-9)
    assert m["margin"] < 0
    assert passed is False


def test_predicate_returns_measurements_dict():
    """The predicate returns a (bool, dict) tuple; dict carries exactly the
    four documented keys with the documented arithmetic. Pins the dict
    contract the gate's log + error message rely on."""
    ref_ids = _ref_ids()
    baseline_ids = _ids_with_jaccard(0.5)
    compiled_ids = _ids_with_jaccard(0.8)

    passed, m = compiled_passes_ideal_math_gate(
        ref_ids=ref_ids,
        baseline_ids=baseline_ids,
        compiled_ids=compiled_ids,
    )
    assert isinstance(passed, bool)
    assert isinstance(m, dict)
    assert set(m.keys()) == {
        "j_baseline_vs_ref", "j_compiled_vs_ref", "slack", "margin",
    }
    # All values are floats.
    for k, v in m.items():
        assert isinstance(v, float), f"measurements[{k!r}] is not float: {v!r}"
    # Arithmetic is the documented formula.
    assert m["slack"] == BASELINE_RELATIVE_SLACK
    assert m["margin"] == (
        m["j_compiled_vs_ref"] - (m["j_baseline_vs_ref"] - m["slack"])
    )


def test_predicate_slack_parameterizable():
    """Passing a custom ``slack`` changes the pass/fail boundary EXACTLY as
    arithmetic predicts. Pins that the slack argument is honored (not just
    a no-op default fall-through).

    Concretely: j_baseline=0.5, j_compiled=0.4. The default slack
    (BASELINE_RELATIVE_SLACK=0.010) ⇒ margin = 0.4 - (0.5 - 0.010) = -0.09
    ⇒ FAIL. A custom slack=0.2 ⇒ margin = 0.4 - (0.5 - 0.2) = +0.1 ⇒ PASS.
    Same inputs, different slack ⇒ different decision.
    """
    ref_ids = _ref_ids()
    baseline_ids = _ids_with_jaccard(0.5)
    compiled_ids = _ids_with_jaccard(0.4)

    # Default slack: FAIL (compiled much worse than baseline).
    passed_default, m_default = compiled_passes_ideal_math_gate(
        ref_ids=ref_ids,
        baseline_ids=baseline_ids,
        compiled_ids=compiled_ids,
    )
    assert passed_default is False
    assert m_default["slack"] == BASELINE_RELATIVE_SLACK

    # Custom large slack: PASS.
    passed_loose, m_loose = compiled_passes_ideal_math_gate(
        ref_ids=ref_ids,
        baseline_ids=baseline_ids,
        compiled_ids=compiled_ids,
        slack=0.2,
    )
    assert passed_loose is True
    assert m_loose["slack"] == 0.2
    assert m_loose["margin"] == pytest.approx(0.1, abs=1e-9)


def test_predicate_baseline_perfect_ref_compiled_must_match_exactly():
    """Edge case: j_baseline_vs_ref = 1.0 (baseline IS the reference, which
    happens for fp32 production cells where baseline == ref). Then compiled
    must also be at 1.0 - slack to pass — tight but valid. Pins the special
    case the spec calls out for the fp32 production cells.

    With slack=0.005, j_compiled >= 0.995 ⇒ PASS; j_compiled = 0.99 ⇒ FAIL.
    """
    ref_ids = _ref_ids()
    # baseline == ref exactly (the fp32-production case).
    baseline_ids = _ref_ids()

    # Compiled at j=1.0 exactly (matches ref) — passes at any slack. Explicit
    # slack=0.005 keeps this edge-case-logic test decoupled from the
    # production BASELINE_RELATIVE_SLACK constant (the constant's value is
    # pinned separately by the constant-using tests); here we exercise the
    # baseline==ref EDGE behaviour at a fixed, readable slack.
    passed_match, _ = compiled_passes_ideal_math_gate(
        ref_ids=ref_ids,
        baseline_ids=baseline_ids,
        compiled_ids=_ref_ids(),
        slack=0.005,
    )
    assert passed_match is True

    # Compiled at j=0.99 (1 unit off the 1/100 grid). margin = 0.99 - 0.995
    # = -0.005 < 0 ⇒ FAIL. This is the "must match exactly (within slack)"
    # bound the spec calls out (at the fixed slack=0.005 used here).
    passed_drift, m_drift = compiled_passes_ideal_math_gate(
        ref_ids=ref_ids,
        baseline_ids=baseline_ids,
        compiled_ids=[t.arange(0, 99, dtype=t.int32)],
        slack=0.005,
    )
    assert m_drift["j_baseline_vs_ref"] == 1.0
    assert m_drift["j_compiled_vs_ref"] == pytest.approx(0.99, abs=1e-9)
    assert m_drift["margin"] == pytest.approx(-0.005, abs=1e-9)
    assert passed_drift is False


# --------------------------------------------------------------------------- #
# _format_compile_gate_envelope: pure formatter for the new measurements dict. #
# --------------------------------------------------------------------------- #


def test_format_compile_gate_envelope_substrings():
    """The decomposed envelope formatter returns ``(measured, gate)`` carrying
    the four documented measurements (j_baseline_vs_ref, j_compiled_vs_ref,
    slack, margin) and the dtype tag. Pins the substrings the
    CompileGateError / INFO both reuse."""
    measurements = {
        "j_baseline_vs_ref": 0.988895,
        "j_compiled_vs_ref": 0.990183,
        "slack": 0.005,
        "margin": 0.990183 - (0.988895 - 0.005),
    }
    measured, gate = _format_compile_gate_envelope(
        measurements, is_bf16=True,
    )
    assert "j_baseline_vs_ref=0.988895" in measured, measured
    assert "j_compiled_vs_ref=0.990183" in measured, measured
    assert "slack=0.005000" in measured, measured
    assert "margin=" in measured, measured
    assert "[bf16]" in gate, gate

    # fp32 path tags [fp32].
    _, gate_fp32 = _format_compile_gate_envelope(
        measurements, is_bf16=False,
    )
    assert "[fp32]" in gate_fp32, gate_fp32


def test_compile_gate_envelope_reused_by_log_and_error(monkeypatch):
    """The fail-raise CompileGateError must carry the SAME envelope substrings
    produced by ``_format_compile_gate_envelope`` (proves the f-string was
    decomposed, not re-duplicated). Verifies the message contains the
    documented measurement keys AND the dtype tag."""
    sae2 = _SAE()
    nc2 = _fresh_collector()
    _arm_with_eps(nc2, eps=50.0, monkeypatch=monkeypatch)
    with pytest.raises(CompileGateError) as ei:
        _native_compile_gate(
            nc2, sae2, probe_input_ids=_PROBE_IDS,  # ty: ignore[invalid-argument-type]
            model_dtype=t.float32,
        )
    msg = str(ei.value)
    # All four measurement keys present, plus the dtype tag.
    for key in (
        "j_baseline_vs_ref", "j_compiled_vs_ref", "slack", "margin", "[fp32]"
    ):
        assert key in msg, f"CompileGateError missing {key!r}; got {msg!r}"


# --------------------------------------------------------------------------- #
# Robustness: (a) re-probe when ALREADY compiled-active from a prior pass    #
# with a now-diverging twin ⇒ flips back to eager; (b) enable_compiled()       #
# raising ⇒ propagates, collector never left active.                           #
# --------------------------------------------------------------------------- #


def test_gate_reprobes_when_already_compiled_active_and_now_diverging(monkeypatch):
    """The collector enters ALREADY ``_use_compiled=True`` (a prior pass
    activated it) but its compiled twin now DIVERGES past the slack. The gate
    must re-probe on the EAGER baseline (step-3 forces inactive first), detect
    the divergence, and RAISE ``CompileGateError`` — never trusting a stale
    activation; never silently falling back. Post-raise, ``_use_compiled is
    False`` (the gate left it eager before the raise).

    Mutation-probe target: dropping the step-3 ``set_compiled_active(False)``
    (so the 'baseline' probe is taken through the still-active compiled twin)
    makes the baseline and compiled probes identical ⇒ the gate would wrongly
    PASS and this ``pytest.raises`` stops triggering.
    """
    sae = _SAE()
    nc = _fresh_collector()
    twin = _arm_with_eps(nc, eps=50.0, monkeypatch=monkeypatch)
    # Simulate a prior pass having activated the compiled path.
    nc.set_compiled_active(True)
    assert nc._use_compiled is True

    with pytest.raises(CompileGateError):
        _native_compile_gate(
            nc, sae, probe_input_ids=_PROBE_IDS,  # ty: ignore[invalid-argument-type]
            model_dtype=t.float32,
        )

    assert nc._use_compiled is False, (
        "entering compiled-active with a now-diverging twin must RE-PROBE, "
        "detect the divergence, and end with the collector EAGER before the "
        "raise (the gate never trusts a stale activation)"
    )
    assert twin.calls >= 1, "the gate must have re-probed the compiled twin"
    # And a subsequent capture really routes through eager.
    eager_ref = nc._base(input_ids=_PROBE_IDS, attention_mask=None) \
        .last_hidden_state.detach()
    assert t.equal(_capture_eager(nc, _PROBE_IDS), eager_ref)


def test_gate_propagates_enable_compiled_failure_collector_never_active(monkeypatch):
    """If ``enable_compiled()`` itself raises (e.g. an inductor/availability
    error), the exception must PROPAGATE out of ``_native_compile_gate`` and
    the collector must be left ``_use_compiled=False`` (the gate's enable
    step must never end up compiled-but-unverified because arming blew up).

    Mutation-probe: wrapping ``enable_compiled()`` in a bare
    ``try/except: pass`` (swallowing the arm failure) makes the gate proceed
    and this ``pytest.raises`` stops triggering.
    """
    _patch_build_ref(monkeypatch)
    sae = _SAE()
    nc = _fresh_collector()
    assert nc._use_compiled is False

    class _Boom(RuntimeError):
        pass

    def _raise():
        raise _Boom("inductor unavailable")

    nc.enable_compiled = _raise  # type: ignore[method-assign]

    with pytest.raises(_Boom, match="inductor unavailable"):
        _native_compile_gate(
            nc, sae, probe_input_ids=_PROBE_IDS,  # ty: ignore[invalid-argument-type]
            model_dtype=t.float32,
        )
    assert nc._use_compiled is False, (
        "enable_compiled() raising must leave the collector EAGER "
        "(never compiled-but-unverified due to an arm failure)"
    )


# --------------------------------------------------------------------------- #
# Non-vacuity: the COMPILED (step-4) probe THROWING mid-gate must (a)         #
# propagate OUT of _native_compile_gate, (b) leave _use_compiled=False (the    #
# step-4 try/FINALLY restored eager — collector NEVER left compiled-but-       #
# unverified), (c) a subsequent capture() is byte-identical to the EAGER       #
# truncated forward.                                                          #
# --------------------------------------------------------------------------- #


def test_gate_compiled_probe_throw_restores_eager_and_propagates(monkeypatch):
    """The step-4 *compiled* ``capture()`` raises mid-``_native_compile_gate``.

    Asserts:
      (a) the failure PROPAGATES out of ``_native_compile_gate``
          (``capture()`` wraps it as ``ExtractionError`` carrying the
          ``_ProbeThrew`` sentinel — pinned via ``match=``);
      (b) ``native_model._use_compiled is False`` afterward — the step-4
          ``try/finally`` restored eager even though ``capture()`` threw AFTER
          ``set_compiled_active(True)`` (the collector is NEVER left
          compiled-but-unverified);
      (c) a subsequent ``capture()`` is byte-identical to the EAGER truncated
          forward.

    Mutation-probe (the proven non-vacuity hole from the prior gate, preserved):
    delete the step-4 ``finally: native_model.set_compiled_active(False)`` in
    ``_native_compile_gate``. The other tests stay GREEN; THIS test FAILS at
    assertion (b) — ``_use_compiled`` stays ``True`` because the throw skipped
    the (now-deleted) restore.
    """
    _patch_build_ref(monkeypatch)
    sae = _SAE()
    nc = _fresh_collector()
    # Arm the gate's way (idempotent enable_compiled) so step-4's
    # set_compiled_active(True) actually routes through _compiled_base, then
    # swap in a base that throws on its FIRST (== the step-4 compiled) call.
    nc.enable_compiled()
    twin = _ThrowingCompiled(nc._base, throw_on=(1,))
    nc._compiled_base = twin
    assert nc._use_compiled is False  # gate must arm-then-probe, not pre-active

    with pytest.raises(ExtractionError, match="_ProbeThrew: compiled probe"):
        _native_compile_gate(
            nc, sae, probe_input_ids=_PROBE_IDS,  # ty: ignore[invalid-argument-type]
            model_dtype=t.float32,
        )

    assert twin.calls == 1, (
        "the step-4 compiled probe must have been attempted exactly once "
        f"(got {twin.calls}) — the throw is the compiled capture(), not eager"
    )
    assert nc._use_compiled is False, (
        "the step-4 try/FINALLY must restore eager even when the compiled "
        "capture() THROWS after set_compiled_active(True) — the collector is "
        "NEVER left compiled-but-unverified (this is the proven hole: "
        "delete that `finally` and only THIS assertion catches it)"
    )
    # And a real subsequent capture() routes through EAGER, byte-identical to
    # the truncated forward (NOT the throwing compiled twin).
    eager_ref = nc._base(input_ids=_PROBE_IDS, attention_mask=None) \
        .last_hidden_state.detach()
    got = _capture_eager(nc, _PROBE_IDS)
    assert twin.calls == 1, (
        "post-throw capture() routed through the compiled twin — the "
        "collector was left compiled-but-unverified after the step-4 throw"
    )
    assert t.equal(got, eager_ref), (
        "post-(compiled-throw) capture() is NOT the eager truncated forward"
    )


# --------------------------------------------------------------------------- #
# Non-vacuity (sibling): the step-3 BASELINE probe THROWING must propagate and #
# leave the collector eager (the gate is never armed-active — the only         #
# set_compiled_active(True) is the pass branch, never reached).                #
# --------------------------------------------------------------------------- #


def test_gate_eager_probe_throw_propagates_collector_eager(monkeypatch):
    """The step-3 *baseline* ``capture()`` raises mid-``_native_compile_gate``.

    The baseline probe runs through ``native_model._base`` (step-3 forces
    ``set_compiled_active(False)`` first), so to make ONLY the baseline probe
    throw we wrap ``_base`` itself. Asserts the failure PROPAGATES out and
    ``_use_compiled is False`` (the gate never reached its pass branch — the
    sole ``set_compiled_active(True)`` — so the collector is left eager).

    Mutation-probe: wrap the step-3 baseline ``capture()`` in a bare
    ``try/except: pass`` (swallowing a baseline-probe failure) ⇒ the gate
    proceeds with garbage baseline features and this ``pytest.raises`` stops
    triggering.
    """
    _patch_build_ref(monkeypatch)
    sae = _SAE()
    nc = _fresh_collector()
    # Make the EAGER base throw on its first call (the step-3 baseline probe).
    # Do NOT arm a compiled twin: step-4 is never reached.
    # throw_on=(1,): ONLY the step-3 baseline probe (the first _base call)
    # raises; any later _base call succeeds — so propagation is the SOLE
    # observable that distinguishes clean (raises) from a swallow (returns).
    nc._base = _ThrowingCompiled(  # type: ignore[assignment]
        nc._base, throw_on=(1,)
    )
    assert nc._use_compiled is False

    # The PRIMARY non-vacuity observable: the baseline-probe failure must
    # PROPAGATE (gate raised) — a swallow mutation leaves ``raised`` None.
    raised: BaseException | None = None
    try:
        _native_compile_gate(
            nc, sae, probe_input_ids=_PROBE_IDS,  # ty: ignore[invalid-argument-type]
            model_dtype=t.float32,
        )
    except BaseException as exc:  # noqa: BLE001 — assert on the exact type next
        raised = exc

    assert raised is not None, (
        "the step-3 baseline-probe failure was SWALLOWED — _native_compile_"
        "gate returned instead of propagating (a try/except:pass around the "
        "baseline capture() makes the gate proceed on garbage; this assertion "
        "is the non-vacuity pin)"
    )
    assert isinstance(raised, ExtractionError) and (
        "_ProbeThrew: compiled probe" in str(raised)
    ), (
        f"propagated exception must be the baseline probe's wrapped "
        f"_ProbeThrew (ExtractionError carrying it); got "
        f"{type(raised).__name__}: {raised}"
    )
    assert nc._base.calls == 1, (
        "the step-3 baseline probe must have been attempted exactly once "
        f"(got {nc._base.calls})"
    )
    assert nc._use_compiled is False, (
        "a baseline-probe throw must propagate with the collector still "
        "EAGER — the gate's only set_compiled_active(True) is its pass "
        "branch, never reached when step-3 throws"
    )


# --------------------------------------------------------------------------- #
# Test (AST/inspect, mirrors test_require_cuda_for_native_unit.py Part-2):    #
# After the behaviour-preserving extraction, main() WIRES the gate by calling  #
# _run_compile_gate exactly once, AFTER _build_accelerated_backend, with       #
# (cfg, native_model, sae, device, model_dtype).                               #
# --------------------------------------------------------------------------- #


def _line_of_first_call(func_def: ast.AST, name: str) -> int:
    for node in ast.walk(func_def):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        ):
            return node.lineno
    return -1


def test_main_wires_gate_under_native_compile():
    """Non-vacuous wiring pin: main() must call ``_run_compile_gate`` exactly
    once, AFTER ``_build_accelerated_backend``, passing
    ``(cfg, native_model, sae, device, model_dtype)`` — and must NOT inline
    ``_native_compile_gate`` itself anymore (the guarded block was extracted
    verbatim into the helper)."""
    func_def = _extract_features_func_def("main")

    # main() WIRES via _run_compile_gate exactly once.
    helper_calls = [
        node
        for node in ast.walk(func_def)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_run_compile_gate"
    ]
    assert len(helper_calls) == 1, (
        f"main() must call _run_compile_gate exactly once, got "
        f"{len(helper_calls)}"
    )
    helper_call = helper_calls[0]

    # main() must NOT inline _native_compile_gate.
    inlined = [
        node
        for node in ast.walk(func_def)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_native_compile_gate"
    ]
    assert inlined == [], (
        "main() must NOT call _native_compile_gate directly anymore — it is "
        "wired exclusively through _run_compile_gate"
    )

    # It is AFTER _build_accelerated_backend.
    bab_line = _line_of_first_call(func_def, "_build_accelerated_backend")
    assert bab_line > 0, "main() must call _build_accelerated_backend"
    assert helper_call.lineno > bab_line, (
        "the compile gate must be wired AFTER _build_accelerated_backend "
        f"(helper@{helper_call.lineno} vs build@{bab_line})"
    )

    # It is called with positional (cfg, native_model, sae, device,
    # model_dtype) — the exact in-scope vars the inline block used.
    arg_names = [
        a.id if isinstance(a, ast.Name) else ast.dump(a)
        for a in helper_call.args
    ]
    assert arg_names == [
        "cfg", "native_model", "sae", "device", "model_dtype"
    ], (
        "main() must call _run_compile_gate(cfg, native_model, sae, "
        f"device, model_dtype) — got positional args {arg_names}"
    )


def test_run_compile_gate_calls_native_compile_gate_exactly_once():
    """``_run_compile_gate`` calls ``_native_compile_gate`` EXACTLY once
    (with no ``force=`` kwarg — that parameter was removed when the opt-out
    eager-fallback was deleted) and the call is guarded only by
    ``native_model is not None``."""
    func_def = _extract_features_func_def("_run_compile_gate")

    gate_calls = [
        node
        for node in ast.walk(func_def)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_native_compile_gate"
    ]
    assert len(gate_calls) == 1, (
        f"_run_compile_gate must call _native_compile_gate exactly "
        f"once, got {len(gate_calls)}"
    )
    gate_call = gate_calls[0]

    # No `force=` kwarg.
    force_kw = next(
        (k for k in gate_call.keywords if k.arg == "force"), None
    )
    assert force_kw is None, (
        "_native_compile_gate must NOT be called with a force= kwarg"
    )

    # Guarded by an `if` whose test references native_model (native-only).
    guarded = False
    for node in ast.walk(func_def):
        if not isinstance(node, ast.If):
            continue
        contains = any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Name)
            and c.func.id == "_native_compile_gate"
            for c in ast.walk(node)
        )
        if not contains:
            continue
        test_dump = ast.dump(node.test)
        if (
            "native_model" in test_dump
            and "native_compile" not in test_dump
        ):
            guarded = True
            break
    assert guarded, (
        "the _native_compile_gate call must be guarded by an `if` whose "
        "condition references native_model (NOT native_compile)"
    )


# --------------------------------------------------------------------------- #
# _build_compile_probe_ids — pinning the probe builder.                        #
# --------------------------------------------------------------------------- #


_PROBE_LONG_IDS = list(range(2, 2 + 37))   # 37 ids, none == 0
_PROBE_SHORT_IDS = [11, 22, 33, 44, 55]    # 5 ids, none == 0


class _FakeTokenizer:
    """A deterministic stand-in for the gated-HF tokenizer."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, text, *, padding, add_special_tokens):
        self.calls.append(
            {"padding": padding, "add_special_tokens": add_special_tokens}
        )
        ids = _PROBE_LONG_IDS if len(self.calls) == 1 else _PROBE_SHORT_IDS
        return {"input_ids": list(ids)}


def _patch_fake_tokenizer(monkeypatch) -> _FakeTokenizer:
    fake = _FakeTokenizer()
    monkeypatch.setattr(ef, "resolve_tokenizer", lambda _m: fake)
    return fake


def test_build_compile_probe_ids_shape_dtype_and_right_pad_id_0(monkeypatch):
    """``_build_compile_probe_ids`` returns ``[2, max_len]`` int64 on the
    requested device, the long row fully populated with its true ids, and the
    short row RIGHT-padded with EXACTLY id 0 past its true length."""
    fake = _patch_fake_tokenizer(monkeypatch)

    out = _build_compile_probe_ids("gemma-2-2b", "cpu")

    max_len = max(len(_PROBE_LONG_IDS), len(_PROBE_SHORT_IDS))
    assert out.shape == (2, max_len), out.shape
    assert out.dtype == t.long, out.dtype
    assert str(out.device) == "cpu"

    assert fake.calls == [
        {"padding": False, "add_special_tokens": True},
        {"padding": False, "add_special_tokens": True},
    ], fake.calls

    assert out[0].tolist() == _PROBE_LONG_IDS
    assert int((out[0] != 0).sum()) == len(_PROBE_LONG_IDS)

    assert out[1, : len(_PROBE_SHORT_IDS)].tolist() == _PROBE_SHORT_IDS
    assert int((out[1] != 0).sum()) == len(_PROBE_SHORT_IDS)
    assert t.equal(
        out[1, len(_PROBE_SHORT_IDS):],
        t.zeros(max_len - len(_PROBE_SHORT_IDS), dtype=t.long),
    )


def test_build_compile_probe_ids_is_deterministic(monkeypatch):
    """Two calls produce byte-identical tensors."""
    _patch_fake_tokenizer(monkeypatch)
    a = _build_compile_probe_ids("gemma-2-2b", "cpu")
    _patch_fake_tokenizer(monkeypatch)
    b = _build_compile_probe_ids("gemma-2-2b", "cpu")
    assert t.equal(a, b)


def test_build_compile_probe_ids_honors_device_arg(monkeypatch):
    """The ``device`` argument is honored end-to-end."""
    _patch_fake_tokenizer(monkeypatch)
    dev = t.device("cpu")
    out = _build_compile_probe_ids("gemma-2-2b", dev)
    assert out.device == dev


# --------------------------------------------------------------------------- #
# _run_compile_gate executed directly.                                         #
# --------------------------------------------------------------------------- #


class _Cfg:
    """Minimal cfg stand-in supporting cfg.model."""

    def __init__(self, **kw) -> None:
        self.__dict__.update(kw)


def _spy_gate(monkeypatch) -> list[dict]:
    """Spy ef._native_compile_gate."""
    calls: list[dict] = []

    def _rec(native_model, sae, *, probe_input_ids, model_dtype):
        calls.append({
            "native_model": native_model,
            "sae": sae,
            "probe_input_ids": probe_input_ids,
            "model_dtype": model_dtype,
        })

    monkeypatch.setattr(ef, "_native_compile_gate", _rec)
    return calls


def test_run_compile_gate_no_native_model_does_not_call(monkeypatch):
    """``native_model=None`` ⇒ ``_native_compile_gate`` NOT called and the
    probe is NOT built."""
    calls = _spy_gate(monkeypatch)
    built: list[int] = []
    monkeypatch.setattr(
        ef, "_build_compile_probe_ids",
        lambda *a, **k: built.append(1) or t.zeros(1, 1, dtype=t.long),
    )

    ef._run_compile_gate(
        _Cfg(model="gemma-2-2b"),  # ty: ignore[invalid-argument-type]
        None, _SAE(), "cpu", t.float32,  # ty: ignore[invalid-argument-type]
    )

    assert calls == [], "native_model=None ⇒ the gate must not be called"
    assert built == [], "native_model=None ⇒ the probe must not be built"


def test_run_compile_gate_calls_once_with_wired_args(monkeypatch):
    """native_model is not None ⇒ ``_native_compile_gate`` called EXACTLY
    once with probe and dtype wired correctly."""
    calls = _spy_gate(monkeypatch)
    sentinel = t.tensor([[1, 2, 0]], dtype=t.long)
    seen: list[tuple] = []

    def _fake_build(model_name, device):
        seen.append((model_name, device))
        return sentinel

    monkeypatch.setattr(ef, "_build_compile_probe_ids", _fake_build)
    nc = _fresh_collector()
    sae = _SAE()

    ef._run_compile_gate(
        _Cfg(model="gemma-3-1b-pt"),  # ty: ignore[invalid-argument-type]
        nc, sae, "cuda:3", t.bfloat16,  # ty: ignore[invalid-argument-type]
    )

    assert len(calls) == 1, f"native ⇒ exactly one gate call, {calls}"
    c = calls[0]
    assert seen == [("gemma-3-1b-pt", "cuda:3")]
    assert c["probe_input_ids"] is sentinel
    assert c["native_model"] is nc and c["sae"] is sae
    assert c["model_dtype"] is t.bfloat16
    assert "force" not in c


# --------------------------------------------------------------------------- #
# _gate_sparse_features degenerate edges (unchanged across the gate redesign). #
# --------------------------------------------------------------------------- #


class _AllNonPositiveSAE:
    """An SAE whose ``encode`` maps every input to an all-(-1.0) feature
    vector, so the gate's ``(pooled > 0)`` selector yields the EMPTY set."""

    f = 48

    def encode(self, x: t.Tensor) -> t.Tensor:
        return t.full((x.shape[0], self.f), -1.0)


def test_gate_sparse_features_all_zero_row_yields_empty_seg():
    """A doc whose pooled features are all ≤ 0 ⇒ that segment's ids/mags are
    EMPTY (no crash), still int32 / float32 on CPU."""
    sae = _AllNonPositiveSAE()
    acts = t.zeros(1, 4, 32)
    ids, mags = _gate_sparse_features(sae, acts, [4])  # ty: ignore[invalid-argument-type]
    assert len(ids) == 1 and len(mags) == 1
    assert ids[0].numel() == 0 and mags[0].numel() == 0
    assert ids[0].dtype == t.int32 and mags[0].dtype == t.float32
    assert ids[0].device.type == "cpu" and mags[0].device.type == "cpu"


def test_gate_sparse_features_length_1_doc():
    """A length-1 doc returns well-formed int32 ids / float32 mags."""
    t.manual_seed(0)
    sae = _SAE()
    acts = t.randn(1, 5, sae.d)
    ids, mags = _gate_sparse_features(sae, acts, [1])  # ty: ignore[invalid-argument-type]
    # Reference via the SAME batched flatten+encode path the gate uses
    # (acts.view(-1, d), not a single-row acts[0,0] matmul). A single-row
    # matmul has different fp accumulation than the batched one and can flip a
    # near-threshold feature's (> 0) membership, making `t.equal` RNG-flaky;
    # matching the gate's matmul shape makes the comparison byte-exact.
    feats = t.relu(
        acts.view(-1, sae.d).float().matmul(sae._w.float()) + sae._b.float()
    ).view(1, acts.shape[1], -1)
    pooled = feats[0, :1].max(dim=0).values  # strip to length 1, max-pool
    nz = (pooled > 0).nonzero().flatten()
    assert ids[0].dtype == t.int32 and mags[0].dtype == t.float32
    assert ids[0].device.type == "cpu" and mags[0].device.type == "cpu"
    assert t.equal(ids[0], nz.to(t.int32))
    assert t.allclose(mags[0], pooled[nz].float())


def test_gate_sparse_features_length_equals_full_seq_no_strip():
    """When ``lengths[i] == S`` the result equals the max-pool over the FULL
    sequence — the strip ``feats[i, :lengths[i]]`` is an exact no-op."""
    sae = _SAE()
    S = 6
    acts = t.randn(1, S, sae.d)
    ids_full, mags_full = _gate_sparse_features(sae, acts, [S])  # ty: ignore[invalid-argument-type]

    feats = t.relu(
        acts.view(-1, sae.d).float().matmul(sae._w.float()) + sae._b.float()
    ).view(1, S, -1)
    pooled = feats[0, :S].max(dim=0).values
    nz = (pooled > 0).nonzero().flatten()
    assert t.equal(ids_full[0], nz.to(t.int32))
    assert t.allclose(mags_full[0], pooled[nz].float())
    assert ids_full[0].dtype == t.int32 and mags_full[0].dtype == t.float32


# --------------------------------------------------------------------------- #
# emulate_precision_casts is a DELIBERATE process-global no-restore. The new   #
# gate sets it to FALSE (the inductor DEFAULT, empirically closer to ideal-    #
# math), and DELIBERATELY never restores it.                                   #
# --------------------------------------------------------------------------- #


def test_enable_compiled_sets_emulate_precision_casts_to_false_and_never_restores():
    """``enable_compiled()`` sets ``torch._inductor.config.
    emulate_precision_casts = False`` (the inductor DEFAULT, empirically the
    closer-to-ideal-math choice) and DELIBERATELY never restores it; the
    flag stays False after enable_compiled() returns AND after a subsequent
    unrelated collector op (an eager ``capture()``). Non-vacuous: if a future
    edit wrapped the flag in a try/finally-restore (or used a context manager),
    the post-op assertion flips. This test owns the cleanup (restores the prior
    value at the end) so the intentional no-restore can't leak elsewhere."""
    prior = _inductor_config.emulate_precision_casts
    try:
        # Start from the opposite value so "stays False" is a real observation,
        # not a coincidental default.
        _inductor_config.emulate_precision_casts = True
        nc = _fresh_collector()
        nc.enable_compiled()
        assert _inductor_config.emulate_precision_casts is False, (
            "enable_compiled() must set emulate_precision_casts = False "
            "(empirical winner against ideal-math)"
        )

        # A subsequent unrelated collector op must NOT restore/clear it.
        _ = _capture_eager(nc, _PROBE_IDS)
        assert _inductor_config.emulate_precision_casts is False, (
            "emulate_precision_casts must REMAIN False after a subsequent "
            "collector op — it is a deliberate, never-restored process-global"
        )
    finally:
        _inductor_config.emulate_precision_casts = prior
