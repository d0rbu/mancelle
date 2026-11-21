"""Functional tests for apply_intervention context manager.

Test strategy:
- hook-free branches (ours, prompting) tested with torch.nn.Linear.
- diffmean/linear_probe + lora branches use monkeypatched axbench-base
  primitives so we don't need full nanochat model structure in tests.
- Real-model integration is exercised by Phase 12 LoRA lifecycle tests.
"""
from __future__ import annotations

from typing import Any

import pytest
import torch

from steering.eval.axbench_base.config import LoraConfig
from steering.eval.axbench_bpb import intervention as intervention_module
from steering.eval.axbench_bpb.intervention import apply_intervention
from steering.eval.axbench_bpb.types import (
    AxBenchBpbError,
    Intervention,
    InterventionKind,
    LoraParams,
)


class _FakeHandle:
    def __init__(self):
        self.removed = 0

    def remove(self):
        self.removed += 1


# ---------------------------------------------------------------------------
# Hook-free branches
# ---------------------------------------------------------------------------


def test_apply_empty_prompting_intervention_is_hook_free():
    model = torch.nn.Linear(4, 4)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    iv = Intervention.prompting(target_model="encdec", prompt_prefix_tokens=[])
    with apply_intervention(model, iv):
        pass
    after = model.state_dict()
    for k, before_value in before.items():
        assert torch.equal(before_value, after[k]), f"state_dict key {k!r} changed"


def test_apply_none_intervention_is_hook_free():
    model = torch.nn.Linear(4, 4)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    iv = Intervention.none()
    with apply_intervention(model, iv):
        pass
    after = model.state_dict()
    for k, before_value in before.items():
        assert torch.equal(before_value, after[k]), f"state_dict key {k!r} changed"


def test_apply_ours_intervention_is_hook_free():
    model = torch.nn.Linear(4, 4)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    iv = Intervention.ours(feature_conditions={10: 3})
    with apply_intervention(model, iv):
        pass
    after = model.state_dict()
    for k, before_value in before.items():
        assert torch.equal(before_value, after[k]), f"state_dict key {k!r} changed"


def test_apply_prompting_intervention_is_hook_free():
    model = torch.nn.Linear(4, 4)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    iv = Intervention.prompting(target_model="encdec", prompt_prefix_tokens=[1, 2, 3])
    with apply_intervention(model, iv):
        pass
    after = model.state_dict()
    for k, before_value in before.items():
        assert torch.equal(before_value, after[k]), f"state_dict key {k!r} changed"


# ---------------------------------------------------------------------------
# Unknown kind
# ---------------------------------------------------------------------------


def test_apply_unknown_kind_raises():
    model = torch.nn.Linear(4, 4)
    # Corrupt the frozen enum-backed field after construction to exercise the
    # defensive apply_intervention branch. Normal construction rejects this.
    iv = Intervention.none()
    object.__setattr__(iv, "kind", "bogus")
    with pytest.raises(AxBenchBpbError, match="Unknown intervention kind"):
        with apply_intervention(model, iv):
            pass


def test_apply_intervention_accepts_enum_kind():
    model = torch.nn.Linear(4, 4)
    iv = Intervention(kind=InterventionKind.NONE, target_model="decoder_only")
    with apply_intervention(model, iv):
        pass


# ---------------------------------------------------------------------------
# Direction branches (diffmean / linear_probe)
# ---------------------------------------------------------------------------


def test_apply_direction_intervention_registers_and_removes_hook(monkeypatch):
    model = torch.nn.Linear(4, 4)
    direction = torch.randn(4)
    fake_handle = _FakeHandle()
    calls: list[tuple[Any, ...]] = []

    def fake_register(m, layer, d, coeff):
        calls.append((m, layer, d, coeff))
        return fake_handle

    monkeypatch.setattr(intervention_module, "register_residual_hook", fake_register)

    iv = Intervention.from_direction(
        kind="diffmean",
        target_model="decoder_only",
        direction=direction,
        direction_layer=2,
        coefficient=1.5,
    )
    with apply_intervention(model, iv):
        assert len(calls) == 1, "hook should be registered once on enter"
        assert fake_handle.removed == 0, "handle.remove() should not be called inside block"

    # Verify exact args passed to register_residual_hook.
    m_arg, layer_arg, dir_arg, coeff_arg = calls[0]
    assert m_arg is model
    assert layer_arg == 2
    assert torch.equal(dir_arg, direction)
    assert coeff_arg == 1.5
    assert fake_handle.removed == 1, "handle.remove() should be called once on exit"


def test_apply_direction_intervention_removes_hook_on_exception(monkeypatch):
    model = torch.nn.Linear(4, 4)
    direction = torch.randn(4)
    fake_handle = _FakeHandle()

    monkeypatch.setattr(
        intervention_module, "register_residual_hook",
        lambda m, layer, d, coeff: fake_handle,
    )

    iv = Intervention.from_direction(
        kind="diffmean",
        target_model="decoder_only",
        direction=direction,
        direction_layer=2,
        coefficient=1.0,
    )
    with pytest.raises(RuntimeError, match="inner error"):
        with apply_intervention(model, iv):
            raise RuntimeError("inner error")

    assert fake_handle.removed == 1, "handle.remove() must be called even on exception"


def test_apply_linear_probe_removes_hook_on_exception(monkeypatch):
    """Parallel of test_apply_direction_intervention_removes_hook_on_exception for
    kind='linear_probe'. linear_probe and diffmean share a branch today; this
    test catches a future refactor that accidentally splits them."""
    fake_handle = _FakeHandle()

    def _fake_register(model, layer, direction, coefficient):
        return fake_handle

    monkeypatch.setattr(intervention_module, "register_residual_hook", _fake_register)

    iv = Intervention.from_direction(
        kind="linear_probe", target_model="encdec",
        direction=torch.randn(4), direction_layer=3, coefficient=0.5,
    )
    model = torch.nn.Linear(4, 4)

    with pytest.raises(RuntimeError, match="synthetic"):
        with apply_intervention(model, iv):
            raise RuntimeError("synthetic")

    assert fake_handle.removed == 1


def test_apply_direction_missing_direction_raises():
    model = torch.nn.Linear(4, 4)
    # Construct directly to bypass from_direction factory.
    iv = Intervention(
        kind="diffmean", target_model="decoder_only",
        direction=None, direction_layer=2, direction_coefficient=1.0,
    )
    with pytest.raises(AxBenchBpbError, match="missing direction tensor"):
        with apply_intervention(model, iv):
            pass


# ---------------------------------------------------------------------------
# LoRA branch
# ---------------------------------------------------------------------------


def test_apply_lora_intervention_attaches_and_detaches(monkeypatch, tmp_path):
    model = torch.nn.Linear(4, 4)
    adapter_path = tmp_path / "fake.safetensors"
    # File doesn't need to exist — we monkeypatch the loader.
    fake_adapters = object()
    detach_calls: list[int] = []

    def fake_detach():
        detach_calls.append(1)

    attach_calls: list[dict[str, Any]] = []

    def fake_attach_lora(model, adapter_path, lora_cfg, *, layer):
        attach_calls.append({"model": model, "adapter_path": adapter_path, "lora_cfg": lora_cfg, "layer": layer})
        return (fake_adapters, fake_detach)

    monkeypatch.setattr(intervention_module, "attach_lora_from_disk", fake_attach_lora)

    lora_cfg = LoraConfig(rank=4, alpha=8.0)
    iv = Intervention.lora(adapter_path=adapter_path)

    with apply_intervention(model, iv, lora=LoraParams(cfg=lora_cfg, layer=6)):
        assert len(attach_calls) == 1, "attach should be called once on enter"
        assert len(detach_calls) == 0, "detach should not be called inside block"

    assert attach_calls[0]["model"] is model
    assert attach_calls[0]["adapter_path"] == adapter_path
    assert attach_calls[0]["lora_cfg"] is lora_cfg
    assert attach_calls[0]["layer"] == 6
    assert len(detach_calls) == 1, "detach_fn() should be called once on exit"


def test_apply_lora_missing_cfg_raises(tmp_path):
    model = torch.nn.Linear(4, 4)
    adapter_path = tmp_path / "fake.safetensors"
    iv = Intervention.lora(adapter_path=adapter_path)

    with pytest.raises(AxBenchBpbError, match="requires the lora kwarg"):
        with apply_intervention(model, iv):
            pass


def test_apply_lora_detaches_on_exception(monkeypatch, tmp_path):
    model = torch.nn.Linear(4, 4)
    adapter_path = tmp_path / "fake.safetensors"
    detach_calls: list[int] = []

    def fake_detach():
        detach_calls.append(1)

    monkeypatch.setattr(
        intervention_module, "attach_lora_from_disk",
        lambda model, adapter_path, lora_cfg, *, layer: (object(), fake_detach),
    )

    iv = Intervention.lora(adapter_path=adapter_path)

    with pytest.raises(RuntimeError, match="lora inner error"):
        with apply_intervention(
            model, iv, lora=LoraParams(cfg=LoraConfig(rank=4, alpha=8.0), layer=6),
        ):
            raise RuntimeError("lora inner error")

    assert len(detach_calls) == 1, "detach_fn() must be called even on exception"
