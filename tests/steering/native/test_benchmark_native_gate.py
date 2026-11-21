"""Phase-2 PR-1 carry-over #2 — benchmark pre-timing correctness gate (smoke).

Proves, on tiny CPU gemma models (built the SAME way as the rest of the
native suite — same conftest constants/seed/fork_rng), that the wired
truncated-native benchmark method and its **pre-timing correctness gate**
behave correctly WITHOUT a GPU (real perf numbers are T9 on the 4090):

* the truncated native method (`run_method_native`) and the Phase-1
  full-forward-hook reference (`hook_reference_residual`) run and agree;
* the pre-timing gate (`_assert_native_matches_reference`) PASSES when the
  truncated output is byte-identical to the hook reference (`max_abs == 0`);
* the gate ABORTS (`BenchmarkGateError`) when fed a deliberately-WRONG
  reference — a NON-VACUOUS fail-closed check: the gate must actually fail,
  never silently let a numerically-wrong method be timed/reported;
* the `--smoke` entrypoint (`run_gemma_model_benchmark` on a tiny twin) runs
  the gate end-to-end and returns timed results (gate passed first).

The benchmark's `hook_reference_residual` is asserted byte-identical to the
canonical test-only `tests/steering/native/_hook_reference.py` so the gate's
baseline IS the verified Phase-1 mechanism.
"""

from __future__ import annotations

import pytest
import torch as t
from omegaconf import OmegaConf

from errors import ExtractionError
from scripts.benchmark.benchmark_activation_methods import (
    BenchmarkGateError,
    _assert_native_matches_reference,
    hook_reference_residual,
    run_gemma_model_benchmark,
    run_method_native,
)
from steering.native_collector import NativeCollector
from steering.sae_resolver import ActivationType
from tests.steering.native._hook_reference import (
    hook_reference_residual as canonical_hook_reference,
)
from tests.steering.native.conftest import (
    _build_tiny_gemma2,
    _build_tiny_gemma3,
)

_LAYER = 2  # 0 <= 2 < TINY_LAYERS (4): a real interior truncation.
_IDS = t.tensor([[3, 9, 1, 7, 2], [5, 5, 8, 0, 0]], dtype=t.long)
_RESID_HOOK = f"blocks.{_LAYER}.hook_resid_post"

_MODELS = [
    ("gemma-2-2b", _build_tiny_gemma2),
    ("gemma-3-1b-pt", _build_tiny_gemma3),
]


@pytest.mark.parametrize(("model_name", "factory"), _MODELS)
def test_benchmark_hook_reference_is_canonical_phase1_mechanism(
    model_name, factory
):
    """The in-script hook reference == the canonical test-only Phase-1 hook.

    The pre-timing gate trusts `hook_reference_residual` to BE the Phase-1
    full-forward-hook mechanism. Assert it is byte-identical to the verified
    `tests/steering/native/_hook_reference.py` on the same seeded weights so
    the gate's baseline is not a re-derived approximation.
    """
    got = hook_reference_residual(factory(), _IDS, _LAYER)
    expected = canonical_hook_reference(factory(), _IDS, _LAYER)
    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    assert t.equal(got, expected), (
        f"{model_name}: benchmark hook reference diverged from the canonical "
        f"Phase-1 _hook_reference (max_abs="
        f"{(got - expected).abs().max().item()!r})"
    )


@pytest.mark.parametrize(("model_name", "factory"), _MODELS)
def test_pre_timing_gate_passes_on_correct_truncated(model_name, factory):
    """Gate PASSES: truncated native byte-identical to the hook reference.

    Runs the wired `run_method_native` (production `capture(residual)`) on a
    truncated tiny twin and the Phase-1 hook reference on an independent
    twin (same weights); the pre-timing gate must accept it (max_abs == 0,
    no exception).
    """
    nc = NativeCollector.truncated(
        factory(), model_name=model_name, layer=_LAYER
    )
    assert nc is not None
    native_out = run_method_native(nc, None, _IDS, _LAYER)
    reference = hook_reference_residual(factory(), _IDS, _LAYER)

    # Independent confirmation it really is byte-identical on tiny.
    assert native_out.shape == reference.shape
    assert t.equal(native_out.cpu(), reference.cpu()), (
        f"{model_name}: tiny truncated != hook reference "
        f"(max_abs={(native_out - reference).abs().max().item()!r})"
    )

    # The gate itself must not raise on the correct pair.
    _assert_native_matches_reference(native_out, reference, model_name)


@pytest.mark.parametrize(("model_name", "factory"), _MODELS)
def test_pre_timing_gate_aborts_on_wrong_reference(model_name, factory):
    """NON-VACUOUS fail-closed: gate ABORTS on a deliberately-wrong reference.

    Feed the gate a reference that is DELIBERATELY corrupted (a clone with one
    element perturbed — same shape/dtype so only the value check can catch it).
    The gate MUST raise `BenchmarkGateError` (it must actually fail-closed —
    a numerically-wrong method must never be timed/reported). If the gate did
    not raise here the whole "no fast-but-wrong" guarantee would be vacuous.
    """
    nc = NativeCollector.truncated(
        factory(), model_name=model_name, layer=_LAYER
    )
    assert nc is not None
    native_out = run_method_native(nc, None, _IDS, _LAYER)

    wrong_reference = native_out.clone()
    wrong_reference[0, 0, 0] += 1.0  # one-element perturbation

    # Sanity: it really differs (otherwise the test would be vacuous).
    assert not t.equal(native_out.cpu(), wrong_reference.cpu())

    with pytest.raises(BenchmarkGateError, match="PRE-TIMING GATE FAILED"):
        _assert_native_matches_reference(
            native_out, wrong_reference, model_name
        )


@pytest.mark.parametrize(("model_name", "factory"), _MODELS)
def test_pre_timing_gate_aborts_on_shape_mismatch(model_name, factory):
    """Fail-closed on a shape mismatch too (defense in depth).

    A wrong-shape reference (e.g. a single-doc capture vs a 2-row batch) must
    also abort with `BenchmarkGateError` — the gate must never compare across
    incompatible shapes and silently pass.
    """
    nc = NativeCollector.truncated(
        factory(), model_name=model_name, layer=_LAYER
    )
    assert nc is not None
    native_out = run_method_native(nc, None, _IDS, _LAYER)
    wrong_shape_ref = native_out[:1]  # [1, S, d] vs [2, S, d]

    with pytest.raises(BenchmarkGateError, match="PRE-TIMING GATE FAILED"):
        _assert_native_matches_reference(
            native_out, wrong_shape_ref, model_name
        )


@pytest.mark.parametrize(("model_name", "factory"), _MODELS)
def test_run_method_native_residual_only_typed_error(model_name, factory):
    """`run_method_native` surfaces the residual-only typed error.

    The wired benchmark method requests `ActivationType.residual`; if a
    caller mutates it to mlp/attention the underlying `capture` must raise
    (the benchmark must never silently benchmark a wrong activation type).
    """
    nc = NativeCollector.truncated(
        factory(), model_name=model_name, layer=_LAYER
    )
    assert nc is not None
    with pytest.raises(ExtractionError, match="residual-only"):
        nc.capture(
            _IDS,
            _LAYER,
            hook_name=_RESID_HOOK,
            activation_type=ActivationType.mlp,
        )


@pytest.mark.parametrize(("model_name", "factory"), _MODELS)
def test_smoke_entrypoint_runs_gate_then_times(model_name, factory):
    """`run_gemma_model_benchmark` (smoke) gates BEFORE timing, then returns.

    End-to-end smoke: the tiny-twin path builds the reference, runs the
    pre-timing gate (which PASSES on tiny), then times the truncated native
    method and returns a result dict whose `methods` includes
    `truncated_native`. Proves the gate is wired in front of the timed loop
    in the real entrypoint, not just in isolation.
    """
    cfg = OmegaConf.create(
        {
            "n_warmup": 1,
            "n_iters": 1,
            "skip_nnterp": True,
            "model_dtype": "float32",
            "batch_size": 2,
            "seq_len": 5,
        }
    )
    res = run_gemma_model_benchmark(
        model_name, _LAYER, cfg, "cpu", tiny_factory=factory
    )
    assert res is not None
    assert res["model"] == model_name
    assert res["layer"] == _LAYER
    labels = [m["label"] for m in res["methods"]]
    assert "truncated_native" in labels
    nat = next(m for m in res["methods"] if m["label"] == "truncated_native")
    assert nat["median_ms"] >= 0.0
    assert nat["n_iters"] == 1
