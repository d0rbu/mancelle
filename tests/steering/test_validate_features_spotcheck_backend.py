"""Spot-check backend parity (#1633): the validator recomputes through the
backend extraction used, not unconditionally through TransformerLens."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
import torch as t
import yaml

from steering import validate_features as vf
from steering.extract_features import Backend, _build_accelerated_backend

REPO = Path(__file__).resolve().parents[2]


class _FakeTLModel:
    """Minimal stand-in for a HookedTransformer: records device/dtype moves."""

    def __init__(self) -> None:
        self.moves: list[object] = []

    def to(self, target):  # noqa: ANN001 - mirrors torch.nn.Module.to
        self.moves.append(target)
        return self


def _install_fake_builder(monkeypatch, *, nn_model):
    calls: list[tuple] = []

    def fake_build(backend, model, device, model_dtype, cfg):
        calls.append((backend, model, device, model_dtype, cfg))
        return nn_model, None

    monkeypatch.setattr(vf, "_build_accelerated_backend", fake_build)
    return calls


@pytest.mark.parametrize("backend", [Backend.auto, Backend.nnterp])
def test_accelerated_backend_is_attached_and_tl_model_offloaded(monkeypatch, backend):
    sentinel = object()
    calls = _install_fake_builder(monkeypatch, nn_model=sentinel)
    monkeypatch.setattr(vf.t.cuda, "empty_cache", lambda: None)
    model = _FakeTLModel()

    label = vf.attach_spot_check_backend(
        model,
        backend=backend,
        device="cuda:0",
        model_dtype=t.bfloat16,
    )

    assert label == "nnterp"
    assert model.nnterp_model is sentinel
    assert model.moves == ["cpu"], "TL weights are offloaded once the nnterp backend owns the forward"
    assert len(calls) == 1
    got_backend, got_model, got_device, got_dtype, got_cfg = calls[0]
    assert got_backend is backend
    assert got_model is model
    assert got_device == "cuda:0"
    assert got_dtype is t.bfloat16
    # The cfg argument is only read on the (rejected) native branch, so the
    # validator passes an inert one rather than pretending to configure it.
    assert (got_cfg.model, got_cfg.layer) == (None, None)


def test_transformerlens_backend_attaches_nothing_and_converts_dtype(monkeypatch):
    calls = _install_fake_builder(monkeypatch, nn_model=None)
    model = _FakeTLModel()

    label = vf.attach_spot_check_backend(
        model,
        backend=Backend.transformerlens,
        device="cuda:0",
        model_dtype=t.bfloat16,
    )

    assert label == "transformerlens"
    assert not hasattr(model, "nnterp_model")
    assert not hasattr(model, "native_model")
    assert model.moves == [t.bfloat16], "the TL forward itself must run in the extraction dtype"
    assert calls[0][0] is Backend.transformerlens


def test_transformerlens_float32_keeps_model_untouched(monkeypatch):
    _install_fake_builder(monkeypatch, nn_model=None)
    model = _FakeTLModel()
    vf.attach_spot_check_backend(
        model,
        backend=Backend.transformerlens,
        device="cpu",
        model_dtype=t.float32,
    )
    assert model.moves == []


def test_native_backend_is_rejected_loudly(monkeypatch):
    calls = _install_fake_builder(monkeypatch, nn_model=None)
    with pytest.raises(ValueError, match=r"native collector is layer-bound"):
        vf.attach_spot_check_backend(
            _FakeTLModel(),
            backend=Backend.native,
            device="cuda:0",
            model_dtype=t.bfloat16,
        )
    assert calls == [], "rejected before any backend build"


def test_builder_failure_propagates(monkeypatch):
    def failing_build(*_args, **_kwargs):
        raise RuntimeError("nnterp build exploded")

    monkeypatch.setattr(vf, "_build_accelerated_backend", failing_build)
    with pytest.raises(RuntimeError, match="nnterp build exploded"):
        vf.attach_spot_check_backend(
            _FakeTLModel(),
            backend=Backend.auto,
            device="cuda:0",
            model_dtype=t.bfloat16,
        )


def test_offload_leaves_a_real_module_backend_where_it_was(monkeypatch):
    """`model.to("cpu")` must not drag the attached backend along.

    That holds today because the nnterp wrapper is not an ``nn.Module``; if it
    ever becomes one, ``nn.Module.__setattr__`` would register it as a submodule
    and the offload would silently move the recompute to CPU. A real module as
    the TL stand-in and a real ``Linear`` as the backend pin the property.
    """
    backend_module = t.nn.Linear(2, 2).to(t.bfloat16)
    _install_fake_builder(monkeypatch, nn_model=backend_module)
    monkeypatch.setattr(vf.t.cuda, "empty_cache", lambda: None)
    tl_model = t.nn.Linear(3, 3)
    before = {name: p.device for name, p in backend_module.named_parameters()}

    vf.attach_spot_check_backend(
        tl_model, backend=Backend.auto, device="cpu", model_dtype=t.bfloat16
    )

    assert tl_model.nnterp_model is backend_module
    assert {name: p.device for name, p in backend_module.named_parameters()} == before
    assert backend_module.weight.dtype is t.bfloat16, "offloading TL must not touch the backend dtype"
    assert "nnterp_model" not in dict(tl_model.named_children()), (
        "the backend is attached as a plain attribute, not registered as a submodule"
    )


def test_builder_cfg_contract_is_native_only():
    """The inert cfg is valid only while `_build_accelerated_backend` reads
    `cfg.*` solely on its native branch — pin that against the real source."""
    src = inspect.getsource(_build_accelerated_backend)
    reads = set(re.findall(r"\bcfg\.(\w+)", src))
    assert reads == {"model", "layer"}
    native_block = src.split("backend == Backend.native")[1].split("backend in (Backend.nnterp")[0]
    assert "cfg.model" in native_block and "cfg.layer" in native_block
    nnterp_onwards = src.split("backend in (Backend.nnterp")[1]
    assert "cfg." not in nnterp_onwards


def test_validate_features_config_defaults_to_metadata_with_legacy_auto_fallback():
    validate_cfg = yaml.safe_load((REPO / "configs/extraction/validate_features.yaml").read_text())
    extract_cfg = yaml.safe_load((REPO / "configs/extraction/extract_features.yaml").read_text())
    assert validate_cfg["backend"] is None
    provenance = vf.ExtractionBackendProvenance(None, 2, 2)
    selection = vf.resolve_spot_check_backend(validate_cfg["backend"], provenance)
    assert selection.backend.value == extract_cfg["backend"] == "auto"
    assert selection.needs_warning


def test_spot_check_backend_appears_in_the_spot_check_banner():
    """The banner names the backend so a mismatch is visible next to any [FAIL]."""
    src = (REPO / "steering/validate_features.py").read_text()
    assert "backend={spot_check_backend}" in src
    assert "attach_spot_check_backend(" in src.split("def main(")[1], (
        "main() must route the loaded model through attach_spot_check_backend"
    )
