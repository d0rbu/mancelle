"""Independent state oracle for the shared GCG model guard."""

from __future__ import annotations

import random
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
import torch as t

from scripts.eval.axbench_adv.gcg_runtime_state import preserve_gcg_model_state


class _RecordingLinear(t.nn.Linear):
    def __init__(self, events: list[str]) -> None:
        super().__init__(2, 2)
        self.events = events

    def train(self, mode: bool = True) -> _RecordingLinear:
        self.events.append("shared")
        return super().train(mode)


class _RecordingSequential(t.nn.Sequential):
    def __init__(self, label: str, events: list[str], *modules: t.nn.Module) -> None:
        super().__init__(*modules)
        self.label = label
        self.events = events

    def train(self, mode: bool = True) -> _RecordingSequential:
        self.events.append(self.label)
        return super().train(mode)


def _numpy_state_equal(left: object, right: object) -> bool:
    assert isinstance(left, tuple) and isinstance(right, tuple)
    assert isinstance(left[1], np.ndarray) and isinstance(right[1], np.ndarray)
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]


def test_shared_guard_restores_aliased_roots_and_every_state_after_failure() -> None:
    train_events: list[str] = []
    shared = _RecordingLinear(train_events)
    first = _RecordingSequential("first", train_events, shared, t.nn.ReLU())
    second = _RecordingSequential("second", train_events, shared, t.nn.Sigmoid())
    first.training = True
    second.training = False
    shared.training = True
    first[1].training = False
    second[1].training = True
    train_events.clear()
    modules = (first, shared, first[1], second, second[1])
    expected_modes = tuple(module.training for module in modules)

    shared.bias.requires_grad_(False)
    parameters = tuple(dict.fromkeys((*first.parameters(), *second.parameters())))
    expected_requires_grad = tuple(parameter.requires_grad for parameter in parameters)
    sentinel = t.full_like(shared.weight, 0.125)
    expected_gradient = sentinel.clone()
    shared.weight.grad = sentinel

    calls: list[None] = []
    handle = shared.register_forward_hook(
        lambda _module, _inputs, _output: calls.append(None), always_call=True
    )
    original_hooks = shared._forward_hooks
    expected_hooks = tuple(original_hooks.items())
    padding = SimpleNamespace(padding_side="left")
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = t.random.get_rng_state()

    try:
        with pytest.raises(RuntimeError, match="guard failure"):
            with preserve_gcg_model_state((first, second), padding_owner=padding):
                assert train_events == ["first", "shared", "second", "shared"]
                assert not any(module.training for module in modules)
                assert not any(parameter.requires_grad for parameter in parameters)
                sentinel.add_(4.0)
                shared._forward_hooks = OrderedDict()
                shared._is_full_backward_hook = True
                padding.padding_side = "right"
                random.random()
                np.random.random()
                t.rand(1)
                raise RuntimeError("guard failure")

        assert tuple(module.training for module in modules) == expected_modes
        assert tuple(parameter.requires_grad for parameter in parameters) == expected_requires_grad
        assert shared.weight.grad is sentinel
        assert t.equal(sentinel, expected_gradient)
        assert shared._forward_hooks is original_hooks
        assert tuple(original_hooks.items()) == expected_hooks
        assert shared._is_full_backward_hook is None
        assert padding.padding_side == "left"
        assert random.getstate() == python_rng
        assert _numpy_state_equal(np.random.get_state(), numpy_rng)
        assert t.equal(t.random.get_rng_state(), torch_rng)
        shared(t.ones(1, 2))
        assert calls == [None]
    finally:
        handle.remove()


@pytest.mark.parametrize("replace_registries", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_guard_removes_new_hooks_from_initially_empty_registries(
    replace_registries: bool, fail: bool
) -> None:
    model = t.nn.Linear(2, 2).eval()
    attributes = (
        "_forward_hooks",
        "_forward_hooks_with_kwargs",
        "_forward_hooks_always_called",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_backward_hooks",
        "_backward_pre_hooks",
    )
    originals = {name: getattr(model, name) for name in attributes}
    assert all(not registry for registry in originals.values())

    def hook(*_args: object) -> None:
        pass

    def mutate() -> None:
        with preserve_gcg_model_state((model,)):
            model.register_forward_hook(hook, with_kwargs=True, always_call=True)
            model.register_forward_pre_hook(hook, with_kwargs=True)
            model.register_full_backward_hook(hook)
            model.register_full_backward_pre_hook(hook)
            assert all(getattr(model, name) for name in attributes)
            if replace_registries:
                for name in attributes:
                    setattr(model, name, OrderedDict(getattr(model, name)))
            model.train()
            if fail:
                raise RuntimeError("new hooks before failure")

    if fail:
        with pytest.raises(RuntimeError, match="new hooks before failure"):
            mutate()
    else:
        mutate()
    assert model.training is False
    assert model._is_full_backward_hook is None
    for name, registry in originals.items():
        assert getattr(model, name) is registry
        assert not registry


def test_guard_restores_nonempty_hook_order_and_live_handles() -> None:
    model = t.nn.Linear(2, 2).eval()
    calls: list[int] = []
    first = model.register_forward_hook(lambda *_args: calls.append(1))
    second = model.register_forward_hook(lambda *_args: calls.append(2), prepend=True)
    original = model._forward_hooks
    expected = tuple(original.items())
    try:
        with preserve_gcg_model_state((model,)):
            original.clear()
            model._forward_hooks = OrderedDict(reversed(expected))
        assert model._forward_hooks is original
        assert tuple(original.items()) == expected
        model(t.ones(1, 2))
        assert calls == [2, 1]
    finally:
        first.remove()
        second.remove()
    assert not original


@pytest.mark.slow
def test_guard_restores_cuda_rng_and_existing_gradient_after_failure() -> None:
    if not t.cuda.is_available():
        pytest.skip("CUDA is required to exercise CUDA state restoration")
    model = t.nn.Linear(2, 2, device="cuda").train()
    gradient = t.full_like(model.weight, 0.25)
    model.weight.grad = gradient
    expected_gradient = gradient.clone()
    states = t.cuda.get_rng_state_all()
    with pytest.raises(RuntimeError, match="CUDA state failure"):
        with preserve_gcg_model_state((model,)):
            assert not model.training and not model.weight.requires_grad
            gradient.add_(7)
            model.weight.grad = None
            for device in range(t.cuda.device_count()):
                t.rand(8, device=f"cuda:{device}")
            raise RuntimeError("CUDA state failure")
    assert model.training and model.weight.requires_grad
    assert model.weight.grad is gradient
    assert t.equal(gradient, expected_gradient)
    for expected, actual in zip(states, t.cuda.get_rng_state_all(), strict=True):
        assert t.equal(expected, actual)
