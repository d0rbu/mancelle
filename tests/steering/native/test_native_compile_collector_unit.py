"""S1 Task 3 — opt-in compiled MECHANISM on `NativeCollector` (unit, CPU).

This sub-project adds an opt-in `torch.compile` path to the home-grown native
collector's structurally-truncated forward, behind a runtime gate (the gate
itself is a LATER task; THIS test pins only the additive compiled MECHANISM and,
crucially, that the EAGER DEFAULT PATH remains byte-identical to PR-1).

The whole sub-project's safety rests on the eager path being unchanged: it is
the trusted anchor (`tests/steering/native/test_truncated_collector.py` gates it
against the verbatim Phase-1 hook reference). So the LOAD-BEARING test here
(test 1) proves that with compile NOT enabled, `torch.compile` is never even
called and `capture()` is `torch.equal` to `hook_reference_residual` for BOTH
the gemma-2 and gemma-3 tiny twins.

`torch.compile` is STUBBED in every test (real inductor compilation is the slow
4090 gate exercised in a later task) — these tests deterministically verify, on
CPU: (1) byte-identical-eager-by-default, (2) `enable_compiled()` ARMS but does
NOT activate, (3) `set_compiled_active` routes through the compiled base both
ways, (4) `enable_compiled()` is idempotent (no recompile).

The transform `NativeCollector.truncated` MUTATES `.layers`/`.norm`/`config` in
place, so every collector is built from a FRESH seeded twin via the conftest
factories (the session fixtures must not be mutated).
"""

from __future__ import annotations

import torch as t
import torch._inductor.config as _ind_cfg

from steering.native_collector import NativeCollector
from steering.sae_resolver import ActivationType
from tests.steering.native._hook_reference import hook_reference_residual
from tests.steering.native.conftest import (
    _build_tiny_gemma2,
    _build_tiny_gemma3,
)

# 0 <= 2 < TINY_LAYERS (4): a real interior truncation (mirrors the
# truncated-collector gate's `_LAYER`).
_LAYER = 2

# Ragged right-padded batch (row 1 has 2 trailing pads) + a single doc —
# exercises the documented maskless contract on a variable-length batch, same
# inputs the truncated-collector byte-identity gate uses.
_IDS_BATCH = t.tensor([[3, 9, 1, 7, 2], [5, 5, 8, 0, 0]], dtype=t.long)
_IDS_SINGLE = t.tensor([[3, 9, 1, 7, 2]], dtype=t.long)

_RESID_HOOK = f"blocks.{_LAYER}.hook_resid_post"

# (model_name, fresh-twin factory) for the two tiny architectures. The
# factories build an independent mutable byte-identical twin per call (required
# — the truncation build mutates in place).
_G2 = ("gemma-2-2b", _build_tiny_gemma2)
_G3 = ("gemma-3-1b-pt", _build_tiny_gemma3)


def _build_truncated(model_name: str, fresh_factory) -> NativeCollector:
    nc = NativeCollector.truncated(
        fresh_factory(), model_name=model_name, layer=_LAYER
    )
    assert nc is not None, f"{model_name}: build returned None"
    return nc


def _capture(nc: NativeCollector, ids: t.Tensor) -> t.Tensor:
    return nc.capture(
        ids,
        _LAYER,
        hook_name=_RESID_HOOK,
        activation_type=ActivationType.residual,
    )


class _SpyCompile:
    """Records every `torch.compile` call WITHOUT compiling.

    Behaviour-preserving on CPU: returns a `_Recorder` wrapping the module so
    a routed forward still produces the correct eager output (the unit tests
    test ROUTING + byte-identical eager, not real inductor compilation — that
    is the slow 4090 gate).
    """

    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, module, *args, **kwargs):
        self.calls.append((args, kwargs))
        return _Recorder(module)


class _Recorder:
    """Behaviour-preserving stand-in for a compiled module.

    `__call__` records that the (pretend-)compiled path was used and then
    delegates to the wrapped eager module so the output stays correct on CPU.
    """

    def __init__(self, module) -> None:
        self._module = module
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self._module(*args, **kwargs)


# --------------------------------------------------------------------------- #
# Test 1 (LOAD-BEARING): default path is eager AND byte-identical to PR-1.     #
#                                                                              #
# With compile NOT enabled: `torch.compile` MUST never be called and          #
# `capture()` MUST be `torch.equal` to `hook_reference_residual` (the trusted  #
# Phase-1 anchor). The whole sub-project's safety rests on this.               #
# --------------------------------------------------------------------------- #


def _assert_default_eager_byte_identical(monkeypatch, model_name, fresh_factory):
    spy = _SpyCompile()
    # Spy BEFORE building/capturing — catches an unconditional compile in
    # __init__/truncated/_truncated_forward as well as in capture().
    monkeypatch.setattr("torch.compile", spy)

    nc = _build_truncated(model_name, fresh_factory)
    # A freshly built collector must default to the inert eager state.
    assert nc._use_compiled is False, (
        f"{model_name}: a freshly built collector must default "
        f"_use_compiled=False (compile is opt-in)"
    )
    assert nc._compiled_base is None, (
        f"{model_name}: a freshly built collector must default "
        f"_compiled_base=None (compile is not armed)"
    )

    for ids in (_IDS_BATCH, _IDS_SINGLE):
        got = _capture(nc, ids)
        expected = hook_reference_residual(fresh_factory(), ids, _LAYER)
        assert t.equal(got, expected), (
            f"{model_name} ids{tuple(ids.shape)}: default capture() != "
            f"hook_reference (eager byte-identity BROKEN; max_abs="
            f"{(got - expected).abs().max().item()!r})"
        )

    assert spy.calls == [], (
        f"{model_name}: torch.compile was called {len(spy.calls)} time(s) "
        f"with compile NOT enabled (default path must be pure eager)"
    )


def test_default_is_eager_byte_identical_no_compile_g2(monkeypatch):
    """gemma-2 tiny twin: default (compile-off) capture == Phase-1 hook, and
    `torch.compile` is never called. The load-bearing safety test."""
    _assert_default_eager_byte_identical(monkeypatch, *_G2)


def test_default_is_eager_byte_identical_no_compile_g3(monkeypatch):
    """gemma-3 tiny twin: default (compile-off) capture == Phase-1 hook, and
    `torch.compile` is never called. The load-bearing safety test."""
    _assert_default_eager_byte_identical(monkeypatch, *_G3)


# --------------------------------------------------------------------------- #
# Test 2: enable_compiled() ARMS compile but does NOT activate it.            #
# --------------------------------------------------------------------------- #


def test_enable_compiled_arms_but_does_not_activate(monkeypatch):
    """After `enable_compiled()`: `_compiled_base` is set, the inductor
    `emulate_precision_casts` config is False (the inductor DEFAULT — the
    empirical winner against the fp32-eager ideal-math reference), but
    `_use_compiled` stays False and `capture()` STILL uses eager (the
    recorder is never called) and is byte-identical to the Phase-1 hook
    reference.

    `enable_compiled()` only arms; the runtime gate decides activation.
    Arming alone must not change a single emitted activation.
    """
    spy = _SpyCompile()
    monkeypatch.setattr("torch.compile", spy)
    # Restore the global inductor flag after the test regardless of its prior
    # value (enable_compiled mutates process-global inductor config).
    monkeypatch.setattr(
        _ind_cfg, "emulate_precision_casts",
        getattr(_ind_cfg, "emulate_precision_casts", False),
    )

    model_name, fresh_factory = _G2
    nc = _build_truncated(model_name, fresh_factory)

    nc.enable_compiled()

    # Armed: compiled base built (via the stub), inductor flag set.
    assert len(spy.calls) == 1, (
        f"enable_compiled() must call torch.compile exactly once to ARM "
        f"(got {len(spy.calls)})"
    )
    assert nc._compiled_base is not None, (
        "enable_compiled() must set _compiled_base (arm the compiled path)"
    )
    # C2: the arm MUST request a shape-polymorphic inductor graph. Extraction
    # feeds ragged/right-padded batches of varying (batch, seq); without
    # dynamic=True torch.compile specializes the first probed shape and a later
    # production shape silently recompiles past the one-shot gate. Pin both the
    # backend and dynamic kwargs on the recorded torch.compile call.
    (_compile_args, _compile_kwargs) = spy.calls[0]
    assert _compile_kwargs.get("backend") == "inductor", (
        f"enable_compiled() must compile with backend='inductor' "
        f"(recorded kwargs={_compile_kwargs!r})"
    )
    assert _compile_kwargs.get("dynamic") is True, (
        "enable_compiled() must compile with dynamic=True (one shape-"
        "polymorphic graph the single gate probe fully validates; without "
        "it a later production shape silently recompiles past the gate) — "
        f"recorded kwargs={_compile_kwargs!r}"
    )
    assert isinstance(nc._compiled_base, _Recorder)
    assert _ind_cfg.emulate_precision_casts is False, (
        "enable_compiled() must set "
        "torch._inductor.config.emulate_precision_casts = False (the inductor "
        "DEFAULT — empirical winner against fp32-eager / ideal-math reference; "
        "emF dyn j=0.990 > emT dyn j=0.989 on gemma-3-1b-pt @7 bf16, 4090, "
        "FineWeb-Edu shard 00388)"
    )
    # NOT activated: arming must not flip the runtime switch.
    assert nc._use_compiled is False, (
        "enable_compiled() must NOT activate (it only arms; the gate "
        "decides activation) — _use_compiled must stay False"
    )

    # capture() STILL eager + byte-identical (recorder never invoked).
    recorder = nc._compiled_base
    for ids in (_IDS_BATCH, _IDS_SINGLE):
        got = _capture(nc, ids)
        expected = hook_reference_residual(fresh_factory(), ids, _LAYER)
        assert t.equal(got, expected), (
            f"armed-but-inactive ids{tuple(ids.shape)}: capture() != "
            f"hook_reference (eager byte-identity BROKEN by mere arming; "
            f"max_abs={(got - expected).abs().max().item()!r})"
        )
    assert recorder.calls == 0, (
        f"armed-but-inactive: the compiled base was invoked {recorder.calls} "
        f"time(s); capture() must STILL route through eager until activated"
    )


# --------------------------------------------------------------------------- #
# Test 3: set_compiled_active(True) routes through the compiled base; False    #
# routes back to eager. Stub is behaviour-preserving so output stays correct.  #
# --------------------------------------------------------------------------- #


def test_set_compiled_active_routes_through_compiled(monkeypatch):
    """After `enable_compiled()` + `set_compiled_active(True)`, `capture()`
    routes through the (stubbed) compiled base (recorder called >= 1) and the
    output is STILL `torch.equal` to the Phase-1 hook reference (the stub is
    behaviour-preserving). Then `set_compiled_active(False)` routes back to
    eager (the recorder is not called again)."""
    spy = _SpyCompile()
    monkeypatch.setattr("torch.compile", spy)
    monkeypatch.setattr(
        _ind_cfg, "emulate_precision_casts",
        getattr(_ind_cfg, "emulate_precision_casts", False),
    )

    model_name, fresh_factory = _G2
    nc = _build_truncated(model_name, fresh_factory)
    nc.enable_compiled()
    recorder = nc._compiled_base
    assert isinstance(recorder, _Recorder)

    # Activate → routes through the compiled base.
    nc.set_compiled_active(True)
    assert nc._use_compiled is True

    got_b = _capture(nc, _IDS_BATCH)
    assert recorder.calls >= 1, (
        "set_compiled_active(True): capture() must route through the "
        f"compiled base (recorder calls={recorder.calls}, expected >= 1)"
    )
    expected_b = hook_reference_residual(fresh_factory(), _IDS_BATCH, _LAYER)
    assert t.equal(got_b, expected_b), (
        "active-compiled (behaviour-preserving stub): capture() != "
        f"hook_reference (max_abs={(got_b - expected_b).abs().max().item()!r})"
    )
    calls_after_active = recorder.calls

    # Deactivate → back to eager (recorder NOT called again), still correct.
    nc.set_compiled_active(False)
    assert nc._use_compiled is False

    got_s = _capture(nc, _IDS_SINGLE)
    assert recorder.calls == calls_after_active, (
        "set_compiled_active(False): capture() must route back to EAGER — "
        f"the compiled base was invoked again (calls {calls_after_active} -> "
        f"{recorder.calls})"
    )
    expected_s = hook_reference_residual(fresh_factory(), _IDS_SINGLE, _LAYER)
    assert t.equal(got_s, expected_s), (
        "deactivated (back to eager) ids(1, 5): capture() != hook_reference "
        f"(max_abs={(got_s - expected_s).abs().max().item()!r})"
    )


# --------------------------------------------------------------------------- #
# Test 4: enable_compiled() is idempotent — no recompile, identity stable.     #
# --------------------------------------------------------------------------- #


def test_enable_compiled_idempotent(monkeypatch):
    """Two `enable_compiled()` calls invoke the `torch.compile` stub EXACTLY
    once (the second call must be a no-op — never recompile) and
    `_compiled_base` keeps the same object identity across both calls."""
    spy = _SpyCompile()
    monkeypatch.setattr("torch.compile", spy)
    monkeypatch.setattr(
        _ind_cfg, "emulate_precision_casts",
        getattr(_ind_cfg, "emulate_precision_casts", False),
    )

    model_name, fresh_factory = _G2
    nc = _build_truncated(model_name, fresh_factory)

    nc.enable_compiled()
    first = nc._compiled_base
    assert first is not None
    assert len(spy.calls) == 1

    nc.enable_compiled()  # second call: must be a no-op (no recompile)
    assert len(spy.calls) == 1, (
        f"enable_compiled() recompiled on the 2nd call (torch.compile called "
        f"{len(spy.calls)} times; must be idempotent — exactly once)"
    )
    assert nc._compiled_base is first, (
        "enable_compiled() replaced _compiled_base on the 2nd call; the "
        "armed compiled base identity must be stable (idempotent)"
    )


# --------------------------------------------------------------------------- #
# Test 5: active-but-NOT-armed must fall to verbatim eager.                     #
#                                                                              #
# Pins the `_compiled_base is not None` half of the `_truncated_forward` guard. #
# `set_compiled_active(True)` WITHOUT `enable_compiled()` leaves the runtime    #
# switch on but the compiled base unarmed; the guard's two-clause AND must keep #
# capture() on the verbatim eager path. A guard-weakening mutation that trusts  #
# `_use_compiled` alone (drops the `and self._compiled_base is not None`) would #
# crash on `None(...)` here — so this test is non-vacuous.                      #
# --------------------------------------------------------------------------- #


def test_active_but_not_armed_falls_to_eager(monkeypatch):
    """set_compiled_active(True) WITHOUT enable_compiled(): the guard's
    `_compiled_base is not None` half must keep capture() on the verbatim
    eager path (byte-identical to the Phase-1 hook)."""
    spy = _SpyCompile()
    monkeypatch.setattr("torch.compile", spy)
    model_name, fresh_factory = _G2
    nc = _build_truncated(model_name, fresh_factory)
    nc.set_compiled_active(True)  # activate WITHOUT arming
    assert nc._use_compiled is True
    assert nc._compiled_base is None
    for ids in (_IDS_BATCH, _IDS_SINGLE):
        got = _capture(nc, ids)
        expected = hook_reference_residual(fresh_factory(), ids, _LAYER)
        assert t.equal(got, expected), (
            f"active-but-unarmed ids{tuple(ids.shape)}: must fall to EAGER"
        )
    assert spy.calls == [], "unarmed path must never call torch.compile"
