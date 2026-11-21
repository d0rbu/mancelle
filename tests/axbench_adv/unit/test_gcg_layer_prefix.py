"""Activation captures preserve the full-depth target and registered state."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch as t

from tests.axbench_adv.gcg_layer_reference import full_depth_capture
from tests.axbench_adv.unit.test_gcg_vendor_worker import GCGSession

_WORKER = sys.modules[GCGSession.__module__]


class _Model(t.nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type=kind, num_hidden_layers=3)
        self._decoder_name = "gpt_neox" if kind == "gpt_neox" else "model"
        decoder = t.nn.Module()
        decoder.layers = t.nn.ModuleList([t.nn.Linear(3, 3, bias=False) for _ in range(3)])
        setattr(self, self._decoder_name, decoder)
        self.lm_head = t.nn.Linear(3, 5, bias=False)

    @property
    def layers(self) -> t.nn.ModuleList:
        return getattr(self, self._decoder_name).layers

    def forward(
        self,
        *,
        inputs_embeds: t.Tensor,
        logits_to_keep: int | t.Tensor = 0,
        **_kwargs: Any,
    ) -> SimpleNamespace:
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden).tanh()
        indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        return SimpleNamespace(logits=self.lm_head(hidden[:, indices, :]))


def _session(kind: str, layer: int) -> tuple[Any, _Model]:
    t.manual_seed(763)
    model = _Model(kind)
    runtime = SimpleNamespace(
        benchmark_model=SimpleNamespace(model=model),
        method="PromptSteering",
        layer=layer,
        tokenizer=SimpleNamespace(padding_side="left"),
    )
    return GCGSession(runtime), model


@pytest.mark.parametrize("kind", ["llama", "qwen2", "smollm3", "gpt_neox"])
@pytest.mark.parametrize("target", [0, 1, 2])
def test_prefix_preserves_exact_target_gradient_and_registered_model(
    kind: str,
    target: int,
) -> None:
    session, model = _session(kind, target)
    layers = model.layers
    modules = tuple(model.modules())
    parameters = tuple(model.parameters())
    state = {key: value.clone() for key, value in model.state_dict().items()}
    model.train()
    layers[0].eval()
    modes = tuple(module.training for module in modules)
    inputs = t.randn(2, 4, 3, requires_grad=True)
    attention = t.ones(2, 4, dtype=t.long)
    ordinary_output = model(inputs_embeds=inputs).logits.detach().clone()
    visited: list[int] = []
    handles = [
        block.register_forward_hook(lambda _m, _i, _o, index=index: visited.append(index))
        for index, block in enumerate(layers)
    ]
    registries = [block._forward_hooks for block in layers]
    entries = [tuple(registry.items()) for registry in registries]
    try:
        with _WORKER._preserve_runtime(session.runtime), t.enable_grad():
            with patch.object(model.lm_head, "forward", wraps=model.lm_head.forward) as head:
                actual = session._capture_hidden(inputs, attention)
                assert head.call_count == 1
                assert head.call_args.args[0].shape == (2, 0, 3)
            assert model.layers is layers
            actual_gradient = t.autograd.grad(actual.square().sum(), inputs)[0]
        assert visited == list(range(target + 1))
        visited.clear()
        with _WORKER._preserve_runtime(session.runtime), t.enable_grad():
            reference = full_depth_capture(_WORKER, session, inputs, attention)
            reference_gradient = t.autograd.grad(reference.square().sum(), inputs)[0]
        assert visited == [0, 1, 2]
        assert t.equal(actual, reference)
        assert t.equal(actual_gradient, reference_gradient)
        assert model.layers is layers and model.config.num_hidden_layers == 3
        assert tuple(model.modules()) == modules
        assert tuple(model.parameters()) == parameters
        assert tuple(module.training for module in modules) == modes
        assert state.keys() == model.state_dict().keys()
        assert all(t.equal(value, model.state_dict()[key]) for key, value in state.items())
        for block, registry, expected in zip(layers, registries, entries, strict=True):
            assert block._forward_hooks is registry
            assert tuple(registry.items()) == expected
        visited.clear()
        assert t.equal(model(inputs_embeds=inputs).logits, ordinary_output)
        assert visited == [0, 1, 2]
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("failure", ["target", "head"])
def test_prefix_restores_original_container_and_hooks_on_forward_failure(failure: str) -> None:
    session, model = _session("qwen2", 1)
    layers = model.layers
    target = layers[1] if failure == "target" else model.lm_head
    original = RuntimeError("original forward failure")
    with patch.object(target, "forward", side_effect=original):
        with pytest.raises(RuntimeError, match="original forward failure") as raised:
            session._capture_hidden(t.ones(1, 2, 3), t.ones(1, 2, dtype=t.long))
    assert raised.value is original
    assert model.layers is layers and len(model.layers) == 3
    assert not layers[1]._forward_hooks


@pytest.mark.parametrize("kind", ["unsupported", "llama"])
def test_prefix_fallback_keeps_unqualified_model_structure(kind: str) -> None:
    model = _Model(kind)
    if kind == "llama":
        # A supported config alone does not prove the expected backbone exists.
        del model.model
        model.layers_elsewhere = t.nn.ModuleList([t.nn.Identity()])
    modules = tuple(model.modules())
    with _WORKER._decoder_layer_prefix(model, 0):
        assert tuple(model.modules()) == modules
    assert tuple(model.modules()) == modules


def test_prefix_does_not_replace_a_non_module_list() -> None:
    model = _Model("qwen2")
    backbone: Any = model.model
    del backbone.layers
    backbone.layers = [t.nn.Identity() for _ in range(3)]
    layers = backbone.layers
    with _WORKER._decoder_layer_prefix(model, 0):
        assert backbone.layers is layers
    assert backbone.layers is layers


@pytest.mark.parametrize("target", [-1, 2, 3])
def test_prefix_keeps_container_when_no_proper_prefix_applies(target: int) -> None:
    model = _Model("llama")
    original = model.layers
    with _WORKER._decoder_layer_prefix(model, target):
        assert model.layers is original
    assert model.layers is original


@pytest.mark.parametrize("kind", ["unsupported", "absent"])
def test_unqualified_capture_retains_one_logit_position(kind: str) -> None:
    session, model = _session(kind, 1)
    if kind == "absent":
        del model.config.model_type
    inputs = t.randn(2, 4, 3, requires_grad=True)
    attention = t.ones(2, 4, dtype=t.long)
    with patch.object(model.lm_head, "forward", wraps=model.lm_head.forward) as head:
        actual = session._capture_hidden(inputs, attention)
        assert head.call_count == 1
        assert head.call_args.args[0].shape == (2, 1, 3)
    reference = full_depth_capture(_WORKER, session, inputs, attention)
    assert t.equal(actual, reference)
    assert t.equal(
        t.autograd.grad(actual.square().sum(), inputs)[0],
        t.autograd.grad(reference.square().sum(), inputs)[0],
    )


def test_inference_capture_permits_a_later_normal_suffix_gradient() -> None:
    session, _model = _session("llama", 1)
    inputs = t.randn(2, 4, 3, requires_grad=True)
    attention = t.ones(2, 4, dtype=t.long)
    with t.inference_mode():
        inferred = session._capture_hidden(inputs, attention)
    assert not t.is_inference_mode_enabled()
    actual = session._capture_hidden(inputs, attention)
    reference = full_depth_capture(_WORKER, session, inputs, attention)
    assert t.equal(inferred, actual) and t.equal(actual, reference)
    assert t.equal(
        t.autograd.grad(actual.square().sum(), inputs)[0],
        t.autograd.grad(reference.square().sum(), inputs)[0],
    )


def test_nested_prefix_restores_each_callers_container() -> None:
    model = _Model("smollm3")
    original = model.layers
    with _WORKER._decoder_layer_prefix(model, 1):
        outer = model.layers
        assert len(outer) == 2
        with _WORKER._decoder_layer_prefix(model, 0):
            assert len(model.layers) == 1 and model.layers[0] is original[0]
        assert model.layers is outer
    assert model.layers is original
