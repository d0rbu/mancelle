"""S2 Task C — production wiring of always-on SDPA row-packing.

Pins the caller + run-gate integration (CPU tiny twins, no GPU/download):
  1. `_get_sae_activations_batched` packed branch (native_model._rowpack_verified
     True) produces per-doc SAE features IDENTICAL to the unpacked padded path
     (verified False) — the C2 wiring (capture_packed -> flat-encode -> split)
     equals (capture padded -> flat-encode -> strip).
  2. When `_rowpack_verified` is False, `capture_packed` is NOT called (the
     unpacked path is used) — pins the gate flag as the routing switch.
  3. `_run_rowpack_gate` is a no-op when `native_model is None`, and otherwise
     builds a production-shaped `RowpackGateProbe` from the resolved runtime
     values and calls `_native_rowpack_gate` with the run's pooling/dtype.

Row-packing is ALWAYS-ON when verified (no per-batch pad-fraction guard); the
per-run `_native_rowpack_gate` is the only gate (its own correctness is pinned
in test_native_rowpack_gate_unit.py). FlexAttention was dropped — SDPA only.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch as t

import steering.extract_features as ef
from errors import AcceleratedBackendError, ExtractionError
from steering.feature_paths import PoolingMode
from steering.native_collector import NativeCollector
from steering.sae_resolver import ActivationType
from tests.steering.native.conftest import TINY_D_MODEL, _build_tiny_gemma2

_LAYER = 2
_RESID_HOOK = f"blocks.{_LAYER}.hook_resid_post"
_DOCS: list[list[int]] = [[3, 9, 1, 7, 2], [5, 8], [1, 4, 6, 0, 5, 2]]


class _SAE:
    """Tiny deterministic linear+ReLU SAE — position-wise `encode([*, d]) ->
    [*, f]`, so packed (flat-encode of stripped acts) and unpacked (flat-encode
    of padded acts, then strip) must agree wherever the underlying residuals
    agree (A1 proved packed==padded per-doc)."""

    def __init__(self, d: int = TINY_D_MODEL, f: int = 48) -> None:
        with t.random.fork_rng():
            t.manual_seed(1234)
            self._w = t.nn.Parameter(t.randn(d, f), requires_grad=False)
            self._b = t.nn.Parameter(t.randn(f), requires_grad=False)

    def parameters(self):
        yield self._w
        yield self._b

    def to(self, dtype):
        self._w = t.nn.Parameter(self._w.to(dtype), requires_grad=False)
        self._b = t.nn.Parameter(self._b.to(dtype), requires_grad=False)
        return self

    def encode(self, x: t.Tensor) -> t.Tensor:
        return t.relu(x.float().matmul(self._w.float()) + self._b.float())


def _truncated() -> NativeCollector:
    nc = NativeCollector.truncated(
        _build_tiny_gemma2(), model_name="gemma-2-2b", layer=_LAYER
    )
    assert nc is not None
    return nc


def _batched(nc: NativeCollector, sae: _SAE) -> list[t.Tensor]:
    # model=None: the native path never touches `model` (only the absent
    # TransformerLens fallback would). sae provides the device + encode.
    return ef._get_sae_activations_batched(
        None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        sae,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        _DOCS,
        _LAYER,
        native_model=nc,
        hook_name=_RESID_HOOK,
        activation_type=ActivationType.residual,
    )


# --------------------------------------------------------------------------- #
# 1. packed branch == unpacked padded path (the C2 wiring)                     #
# --------------------------------------------------------------------------- #


def test_packed_route_matches_unpacked_route() -> None:
    sae = _SAE()
    nc = _truncated()

    nc.set_rowpack_verified(False)
    unpacked = _batched(nc, sae)
    nc.set_rowpack_verified(True)
    packed = _batched(nc, sae)

    assert len(packed) == len(unpacked) == len(_DOCS)
    for i, (p, u) in enumerate(zip(packed, unpacked, strict=True)):
        assert p.shape == u.shape, f"doc {i}: {p.shape} != {u.shape}"
        max_abs = (p - u).abs().max().item()
        assert t.allclose(p, u, atol=1e-5, rtol=1e-5), (
            f"doc {i}: packed-route SAE features != unpacked-route "
            f"(max_abs={max_abs!r})"
        )


# --------------------------------------------------------------------------- #
# 2. the verified flag is the routing switch (non-vacuous)                     #
# --------------------------------------------------------------------------- #


def test_unverified_does_not_call_capture_packed(monkeypatch) -> None:
    sae = _SAE()
    nc = _truncated()
    nc.set_rowpack_verified(False)

    calls: list[int] = []
    real = nc.capture_packed
    monkeypatch.setattr(
        nc, "capture_packed",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1],
    )
    _batched(nc, sae)
    assert calls == [], "capture_packed must NOT run when _rowpack_verified is False"


def test_verified_calls_capture_packed(monkeypatch) -> None:
    sae = _SAE()
    nc = _truncated()
    nc.set_rowpack_verified(True)

    calls: list[int] = []
    real = nc.capture_packed
    monkeypatch.setattr(
        nc, "capture_packed",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1],
    )
    _batched(nc, sae)
    assert calls == [1], "capture_packed must run exactly once when verified"


# --------------------------------------------------------------------------- #
# 3. _run_rowpack_gate routing                                                 #
# --------------------------------------------------------------------------- #


def test_run_rowpack_gate_noop_when_native_none(monkeypatch) -> None:
    called: list[int] = []
    monkeypatch.setattr(ef, "_native_rowpack_gate",
                        lambda *a, **k: called.append(1))
    ef._run_rowpack_gate(
        SimpleNamespace(model="gemma-2-2b"), None, _SAE(), t.float32,  # ty: ignore[invalid-argument-type]
        pooling=PoolingMode.max_pool,
        segment_size=128,
        context_independent_features=False,
        ignore_bos_features=True,
    )
    assert called == [], "gate must not run when native_model is None"


def test_run_rowpack_gate_builds_production_shaped_probe(monkeypatch) -> None:
    seen: dict = {}

    probe = ef.RowpackGateProbe(
        token_id_lists=[[1, 2, 3]],
        segments_by_sequence=[[(0, 3)]],
        keep_masks_by_sequence=[(True, True, True)],
        context_independent_features=True,
    )

    def _fake_build_probe(
        model_name,
        *,
        segment_size,
        context_independent_features,
        ignore_bos_features,
    ):
        seen["model_name"] = model_name
        seen["segment_size"] = segment_size
        seen["context_independent_features"] = context_independent_features
        seen["ignore_bos_features"] = ignore_bos_features
        return probe

    monkeypatch.setattr(ef, "_build_rowpack_gate_probe", _fake_build_probe)

    def _spy(native_model, sae, *, probe, pooling, model_dtype):
        seen["probe"] = probe
        seen["pooling"] = pooling
        seen["dtype"] = model_dtype

    monkeypatch.setattr(ef, "_native_rowpack_gate", _spy)
    nc = _truncated()
    ef._run_rowpack_gate(
        SimpleNamespace(model="gemma-2-2b"), nc, _SAE(), t.bfloat16,  # ty: ignore[invalid-argument-type]
        pooling=PoolingMode.mean_pool,
        segment_size=64,
        context_independent_features=True,
        ignore_bos_features=True,
    )
    assert seen.get("model_name") == "gemma-2-2b"
    assert seen.get("segment_size") == 64
    assert seen.get("context_independent_features") is True
    assert seen.get("ignore_bos_features") is True
    assert seen.get("probe") is probe
    assert seen.get("pooling") is PoolingMode.mean_pool
    assert seen.get("dtype") is t.bfloat16


# --------------------------------------------------------------------------- #
# 4. packed-branch fail-fast: capture_packed error -> AcceleratedBackendError  #
# --------------------------------------------------------------------------- #


def test_packed_branch_capture_packed_failure_is_accelerated_error(monkeypatch) -> None:
    """In the packed branch, a `capture_packed` ExtractionError must surface as
    AcceleratedBackendError (no silent TransformerLens fallback) — the same
    fail-fast contract as the unpacked native path. Pins the packed branch's
    `except ExtractionError` codepath, which the unpacked-path fail-fast test
    does NOT reach (it runs unverified)."""
    sae = _SAE()
    nc = _truncated()
    nc.set_rowpack_verified(True)

    def _boom(*a, **k):
        raise ExtractionError("simulated packed forward failure")

    monkeypatch.setattr(nc, "capture_packed", _boom)
    with pytest.raises(AcceleratedBackendError):
        _batched(nc, sae)
