"""Accelerated-backend fail-fast: no silent TransformerLens degradation.

Spec: docs/superpowers/specs/2026-05-18-accelerated-backend-fail-fast-design.md
"""

from __future__ import annotations

import pytest
import torch as t

import steering.extract_features as ef
from errors import AcceleratedBackendError, ExtractionError, NanoGPTError


def test_accelerated_backend_error_type():
    assert issubclass(AcceleratedBackendError, NanoGPTError)
    assert not issubclass(AcceleratedBackendError, ExtractionError)
    assert not issubclass(ExtractionError, AcceleratedBackendError)


def test_accelerated_backend_error_message_and_chaining():
    cause = ExtractionError("native boom")
    err = ef._accelerated_backend_error("native", "forward", cause)
    assert isinstance(err, AcceleratedBackendError)
    msg = str(err)
    assert "native" in msg
    assert "forward" in msg
    assert "backend=transformerlens" in msg  # the explicit remedy
    # raised `from cause` by callers; the factory attaches it as __cause__
    assert err.__cause__ is cause


class _SAE:
    def __init__(self, d=8, f=16):
        self._p = t.nn.Parameter(t.zeros(1))
        self.d, self.f = d, f

    def parameters(self):
        yield self._p

    def encode(self, x):
        return t.zeros(x.shape[0], self.f)


def _model():
    return t.nn.Linear(2, 2)


def test_batched_native_failure_raises_not_silent_tl(monkeypatch):
    def boom(*a, **k):
        raise ExtractionError("native boom")

    tl_called = []
    monkeypatch.setattr(ef, "_get_activations_native", boom)
    monkeypatch.setattr(
        ef, "_get_activations_tl",
        lambda *a, **k: tl_called.append(1) or t.zeros(1, 3, 8),
    )
    with pytest.raises(AcceleratedBackendError, match="native"):
        ef._get_sae_activations_batched(
            _model(), _SAE(), [[1, 2, 3]], layer=0,  # ty: ignore[invalid-argument-type]
            native_model=object(),  # ty: ignore[invalid-argument-type]
            hook_name="blocks.0.hook_resid_post",
            activation_type=ef.ActivationType.residual,
        )
    assert tl_called == [], "must NOT silently fall back to TransformerLens"


def test_batched_nnterp_failure_raises_not_silent_tl(monkeypatch):
    def boom(*a, **k):
        raise ExtractionError("nnterp boom")

    tl_called = []
    monkeypatch.setattr(ef, "_get_activations_nnterp", boom)
    monkeypatch.setattr(
        ef, "_get_activations_tl",
        lambda *a, **k: tl_called.append(1) or t.zeros(1, 3, 8),
    )
    with pytest.raises(AcceleratedBackendError, match="nnterp"):
        ef._get_sae_activations_batched(
            _model(), _SAE(), [[1, 2, 3]], layer=0,  # ty: ignore[invalid-argument-type]
            nnterp_model=object(),  # ty: ignore[invalid-argument-type]
            hook_name="blocks.0.hook_resid_post",
            activation_type=ef.ActivationType.residual,
        )
    assert tl_called == []


def test_batched_explicit_tl_path_still_runs(monkeypatch):
    """No accelerated backend configured (explicit transformerlens):
    both models absent -> TL runs, no AcceleratedBackendError."""
    monkeypatch.setattr(
        ef, "_get_activations_tl", lambda *a, **k: t.zeros(1, 3, 8)
    )
    out = ef._get_sae_activations_batched(
        _model(), _SAE(), [[1, 2, 3]], layer=0,  # ty: ignore[invalid-argument-type]
        nnterp_model=None, native_model=None,
        hook_name="blocks.0.hook_resid_post",
        activation_type=ef.ActivationType.residual,
    )
    assert out is not None


def _hooked_model(native=None, nnterp=None):
    m = t.nn.Linear(2, 2)
    if native is not None:
        m.native_model = native
    if nnterp is not None:
        m.nnterp_model = nnterp
    return m


def test_singledoc_native_failure_raises_not_silent_tl(monkeypatch):
    def boom(*a, **k):
        raise ExtractionError("native boom")

    tl_called = []
    monkeypatch.setattr(ef, "_get_activations_native", boom)
    monkeypatch.setattr(
        ef, "_get_activations_tl",
        lambda *a, **k: tl_called.append(1) or t.zeros(1, 3, 8),
    )
    with pytest.raises(AcceleratedBackendError, match="native"):
        ef._get_sae_activations(
            _hooked_model(native=object()), _SAE(),  # ty: ignore[invalid-argument-type]
            t.tensor([1, 2, 3]), layer=0,
            hook_name="blocks.0.hook_resid_post",
        )
    assert tl_called == []


def test_singledoc_nnterp_failure_raises_not_silent_tl(monkeypatch):
    def boom(*a, **k):
        raise ExtractionError("nnterp boom")

    tl_called = []
    monkeypatch.setattr(ef, "_get_activations_nnterp", boom)
    monkeypatch.setattr(
        ef, "_get_activations_tl",
        lambda *a, **k: tl_called.append(1) or t.zeros(1, 3, 8),
    )
    with pytest.raises(AcceleratedBackendError, match="nnterp"):
        ef._get_sae_activations(
            _hooked_model(nnterp=object()), _SAE(),  # ty: ignore[invalid-argument-type]
            t.tensor([1, 2, 3]), layer=0,
            hook_name="blocks.0.hook_resid_post",
        )
    assert tl_called == []


def test_singledoc_explicit_tl_path_still_runs(monkeypatch):
    monkeypatch.setattr(
        ef, "_get_activations_tl", lambda *a, **k: t.zeros(1, 3, 8)
    )
    out = ef._get_sae_activations(
        _hooked_model(), _SAE(), t.tensor([1, 2, 3]), layer=0,  # ty: ignore[invalid-argument-type]
        hook_name="blocks.0.hook_resid_post",
    )
    assert out is not None


class _Cfg:
    def __init__(self, model="gemma-2-2b", layer=6):
        self.model = model
        self.layer = layer


def test_build_accel_native_noncuda_raises():
    with pytest.raises(AcceleratedBackendError, match="native"):
        ef._build_accelerated_backend(
            ef.Backend.native, _model(), "cpu", t.float32, _Cfg()
        )


def test_build_accel_native_build_none_raises(monkeypatch):
    monkeypatch.setattr(ef, "build_native_model", lambda *a, **k: None)
    with pytest.raises(AcceleratedBackendError, match="native"):
        ef._build_accelerated_backend(
            ef.Backend.native, _model(), "cuda:0", t.float32, _Cfg()
        )


def test_build_accel_explicit_nnterp_build_none_raises(monkeypatch):
    monkeypatch.setattr(ef, "build_nnterp_model", lambda *a, **k: None)
    with pytest.raises(AcceleratedBackendError, match="nnterp"):
        ef._build_accelerated_backend(
            ef.Backend.nnterp, _model(), "cuda:0", t.float32, _Cfg()
        )


def test_build_accel_auto_build_none_raises(monkeypatch):
    """The vestigial-fallback fix: auto with nnterp build None must now ALSO
    hard-raise (was: silent GPU-TL)."""
    monkeypatch.setattr(ef, "build_nnterp_model", lambda *a, **k: None)
    with pytest.raises(AcceleratedBackendError, match="nnterp"):
        ef._build_accelerated_backend(
            ef.Backend.auto, _model(), "cuda:0", t.float32, _Cfg()
        )


def test_build_accel_success_returns_models_no_raise(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(ef, "build_nnterp_model", lambda *a, **k: sentinel)
    nn_model, native_model = ef._build_accelerated_backend(
        ef.Backend.auto, _model(), "cuda:0", t.float32, _Cfg()
    )
    assert nn_model is sentinel and native_model is None


def test_build_accel_transformerlens_returns_none_none(monkeypatch):
    """Explicit transformerlens: no accelerated build, no raise."""
    monkeypatch.setattr(
        ef, "build_native_model",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not build")),
    )
    monkeypatch.setattr(
        ef, "build_nnterp_model",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not build")),
    )
    nn_model, native_model = ef._build_accelerated_backend(
        ef.Backend.transformerlens, _model(), "cuda:0", t.float32, _Cfg()
    )
    assert nn_model is None and native_model is None


def test_build_accel_noncuda_nnterp_raises():
    """Non-CUDA explicit nnterp can't build an accelerated backend -> hard
    abort (closes the silent CPU-TL gap)."""
    with pytest.raises(AcceleratedBackendError, match="nnterp"):
        ef._build_accelerated_backend(
            ef.Backend.nnterp, _model(), "cpu", t.float32, _Cfg()
        )


def test_build_accel_noncuda_auto_raises():
    """Non-CUDA auto likewise hard-aborts (no silent CPU-TL)."""
    with pytest.raises(AcceleratedBackendError, match="auto"):
        ef._build_accelerated_backend(
            ef.Backend.auto, _model(), "cpu", t.float32, _Cfg()
        )


def test_build_accel_transformerlens_noncuda_returns_none_none():
    """transformerlens stays the sole intentional (None, None)/TL route,
    incl. on a non-CUDA device."""
    nn_model, native_model = ef._build_accelerated_backend(
        ef.Backend.transformerlens, _model(), "cpu", t.float32, _Cfg()
    )
    assert nn_model is None and native_model is None
