import pytest
import torch as t

import steering.extract_features as ef
from errors import AcceleratedBackendError, ExtractionError


class _SAE:
    def __init__(self, d=8, f=16):
        self._p = t.nn.Parameter(t.zeros(1))
        self.d, self.f = d, f

    def parameters(self):
        yield self._p

    def encode(self, x):
        return t.zeros(x.shape[0], self.f)


def _model_on(device="cpu"):
    m = t.nn.Linear(2, 2)
    return m


def test_native_preferred_over_nnterp(monkeypatch):
    sae = _SAE()
    calls = []

    def fake_native(native_model, tokens, layer, activation_type, hook_name):
        calls.append("native")
        return t.zeros(1, len(tokens[0]), 8)

    def fake_nnterp(*a, **k):
        calls.append("nnterp")
        return t.zeros(1, 1, 8)

    monkeypatch.setattr(ef, "_get_activations_native", fake_native)
    monkeypatch.setattr(ef, "_get_activations_nnterp", fake_nnterp)
    ef._get_sae_activations_batched(
        _model_on(), sae, [[1, 2, 3]], layer=0,  # ty: ignore[invalid-argument-type]
        nnterp_model=object(), native_model=object(),  # ty: ignore[invalid-argument-type]
        hook_name="blocks.0.hook_resid_post",
        activation_type=ef.ActivationType.residual,
    )
    assert calls == ["native"], f"native must win, got {calls}"


def test_native_failure_raises_accelerated_backend_error(monkeypatch):
    """Site-1a dispatch: a mid-run native ``ExtractionError`` is NO LONGER
    silently swallowed into an nnterp (or TL) fallback. Sub-project #4
    deliberately inverted this contract: native failure now hard-raises
    ``AcceleratedBackendError`` (chained from the injected ``ExtractionError``)
    and the nnterp adapter is NEVER reached, even though an ``nnterp_model``
    is configured. Pins native-failure dispatch behavior (was: silent
    fallback ``calls == ["native", "nnterp"]``; now: raise, nnterp uncalled).
    """
    sae = _SAE()
    calls = []
    injected = ExtractionError("native boom")

    def boom(*a, **k):
        calls.append("native")
        raise injected

    def fake_nnterp(*a, **k):
        calls.append("nnterp")
        return t.zeros(1, 3, 8)

    monkeypatch.setattr(ef, "_get_activations_native", boom)
    monkeypatch.setattr(ef, "_get_activations_nnterp", fake_nnterp)
    with pytest.raises(AcceleratedBackendError, match="native") as excinfo:
        ef._get_sae_activations_batched(
            _model_on(), sae, [[1, 2, 3]], layer=0,  # ty: ignore[invalid-argument-type]
            nnterp_model=object(), native_model=object(),  # ty: ignore[invalid-argument-type]
            hook_name="blocks.0.hook_resid_post",
            activation_type=ef.ActivationType.residual,
        )
    # nnterp (and therefore TL) must NOT be reached: no silent degradation.
    assert calls == ["native"], (
        f"native failure must NOT fall back to nnterp/TL, got {calls}"
    )
    # The original ExtractionError is preserved as the chained cause.
    assert excinfo.value.__cause__ is injected


def test_native_none_goes_straight_to_nnterp(monkeypatch):
    sae = _SAE()
    calls = []

    def fake_native(*a, **k):
        calls.append("native")
        return t.zeros(1, 3, 8)

    def fake_nnterp(*a, **k):
        calls.append("nnterp")
        return t.zeros(1, 3, 8)

    monkeypatch.setattr(ef, "_get_activations_native", fake_native)
    monkeypatch.setattr(ef, "_get_activations_nnterp", fake_nnterp)
    ef._get_sae_activations_batched(
        _model_on(), sae, [[1, 2, 3]], layer=0,  # ty: ignore[invalid-argument-type]
        nnterp_model=object(), native_model=None,  # ty: ignore[invalid-argument-type]
        hook_name="blocks.0.hook_resid_post",
        activation_type=ef.ActivationType.residual,
    )
    assert calls == ["nnterp"], f"native=None must skip native, got {calls}"


def test_both_none_falls_back_to_tl(monkeypatch):
    sae = _SAE()
    calls = []

    def fake_native(*a, **k):
        calls.append("native")
        return t.zeros(1, 3, 8)

    def fake_nnterp(*a, **k):
        calls.append("nnterp")
        return t.zeros(1, 3, 8)

    def fake_tl(model, tokens, layer, hook_name=None, attention_mask=None):
        calls.append("tl")
        # [batch, seq, d_model] matching _SAE.encode's expected last dim
        return t.zeros(tokens.shape[0], tokens.shape[1], 8)

    monkeypatch.setattr(ef, "_get_activations_native", fake_native)
    monkeypatch.setattr(ef, "_get_activations_nnterp", fake_nnterp)
    monkeypatch.setattr(ef, "_get_activations_tl", fake_tl)
    ef._get_sae_activations_batched(
        _model_on(), sae, [[1, 2, 3]], layer=0,  # ty: ignore[invalid-argument-type]
        nnterp_model=None, native_model=None,
        hook_name="blocks.0.hook_resid_post",
        activation_type=ef.ActivationType.residual,
    )
    assert calls == ["tl"], f"both None must use TL only, got {calls}"
