"""Independent project-model GCG routing, root, and state-preservation oracles."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch as t
import torch.nn.functional as F

import nanochat.flash_attention as attention
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder import encoder as encoder_module
from steering.conditional.encoder_decoder import eval_residual as eval_residual_module
from steering.conditional.encoder_decoder import model as model_module
from steering.conditional.encoder_decoder.eval_residual import (
    post_cross_attention_residual_counter,
)
from steering.eval.axbench_adv import self_conditioning as self_conditioning_module
from steering.eval.axbench_adv.method_grid import ADV_FULL_FACTORS
from steering.eval.axbench_adv.self_conditioning import _value_embedding_modules
from tests.axbench_adv.unit.test_gcg_project_adapter import _cell


@pytest.fixture(autouse=True)
def _force_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attention, "_override_impl", FlashAttnImpl.SDPA)


@pytest.mark.parametrize("fail_after_backward", [False, True])
def test_project_guard_restores_existing_gradient_contents(fail_after_backward: bool) -> None:
    first_layer = t.nn.Linear(3, 3)
    model = t.nn.Sequential(first_layer, t.nn.Tanh(), t.nn.Linear(3, 1))
    parameter = first_layer.weight
    sentinel = t.full_like(parameter, 0.125)
    expected = sentinel.clone()
    parameter.grad = sentinel

    def mutate_existing_gradient(
        _module: object, _grad_input: object, _grad_output: object
    ) -> None:
        with t.no_grad():
            assert parameter.grad is sentinel
            parameter.grad.add_(4.0)

    handle = model[-1].register_full_backward_hook(mutate_existing_gradient)

    def run() -> None:
        inputs = t.randn(2, 3, requires_grad=True)
        with self_conditioning_module._preserve_project_model(model):
            output = model(inputs).sum()
            t.autograd.grad(output, inputs)
            assert not t.equal(sentinel, expected)
            if fail_after_backward:
                raise RuntimeError("failure after backward")

    try:
        if fail_after_backward:
            with pytest.raises(RuntimeError, match="failure after backward"):
                run()
        else:
            run()
        assert parameter.grad is sentinel
        assert t.equal(parameter.grad, expected)
    finally:
        handle.remove()


def test_project_gradient_restores_backward_hook_registry_and_handle() -> None:
    """A search backward must not consume a caller-owned self-removing hook."""

    cell = _cell(activation=False)
    embedding = cell._runtime._target_model.transformer["wte"]
    registry = embedding._backward_hooks
    calls: list[None] = []
    handle_box: dict[str, t.utils.hooks.RemovableHandle] = {}

    def remove_self(_module: object, _grad_input: object, _grad_output: object) -> None:
        calls.append(None)
        handle_box["handle"].remove()

    handle = embedding.register_full_backward_hook(remove_self)
    handle_box["handle"] = handle
    try:
        cell.gcg_gradient((12,), [0])
        assert calls
        assert embedding._backward_hooks is registry
        assert handle.id in registry
    finally:
        handle.remove()
    assert handle.id not in registry


def test_project_response_bpb_uses_exact_routing_and_all_response_positions() -> None:
    """Compare the adapter with direct logits and an independent BPB reduction."""

    cell = _cell(activation=False)
    model = cell._runtime._target_model
    tokenizer = cell._runtime._target_tokenizer
    row = cell._gcg_pairs[0].positive
    suffix = (12, 13)
    prompt = [*row.prompt_prefix, *suffix, *row.prompt_tail]
    sequence = [*prompt, *row.response]
    expected_seg_map = t.tensor(
        [
            [
                *row.prompt_prefix_seg,
                *([row.source_slot] * len(suffix)),
                *([row.source_slot] * (len(row.prompt_tail) - 1)),
                row.target_slot,
                *([row.target_slot] * (len(row.response) - 1)),
            ]
        ],
        dtype=t.long,
    )
    observed_seg_maps: list[t.Tensor] = []

    def capture_seg_map(
        _module: object, _args: tuple[object, ...], kwargs: dict[str, object]
    ) -> None:
        seg_map = kwargs["seg_map"]
        assert isinstance(seg_map, t.Tensor)
        observed_seg_maps.append(seg_map.detach().cpu().clone())

    handle = model.register_forward_pre_hook(capture_seg_map, with_kwargs=True)
    try:
        [(actual, _gradient)] = cell._project_batch_values([(row, suffix)], gradient=False)
    finally:
        handle.remove()
    assert len(observed_seg_maps) == 1
    assert t.equal(observed_seg_maps[0], expected_seg_map)

    dec_input = t.tensor([sequence[:-1]], dtype=t.long)
    with t.no_grad():
        logits = model(
            dec_input,
            None,
            seg_map=expected_seg_map,
            encoder_out=row.encoder_out,
            encoder_strict=True,
        )
    response_logits = logits[0, len(prompt) - 1 : len(sequence) - 1]
    response_targets = t.tensor(row.response, dtype=t.long)
    losses = F.cross_entropy(response_logits, response_targets, reduction="none")
    weights = t.tensor(
        [len(tokenizer.decode([token_id]).encode("utf-8")) for token_id in row.response],
        dtype=t.float32,
    )
    expected = (losses.float() * weights).sum() / (t.log(t.tensor(2.0)) * weights.sum())
    t.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


def test_project_activation_vjp_includes_value_embedding_at_capture_layer() -> None:
    """The production-like same-layer VE root is upstream of post-cross capture."""

    cell = _cell(activation=True)
    model = cell._runtime._target_model
    direction = cell._gcg_direction
    assert direction is not None
    cell._gcg_direction = replace(direction, provenance={"layer": 1})
    suffix = (12,)
    cell.gcg_score([suffix], [0], candidate_eval_device_batch_size=4)
    analytic = cell.gcg_gradient(suffix, [0])[0][0]

    value_modules = _value_embedding_modules(model)
    assert [layer for layer, _module in value_modules] == [1]
    wte_width = model.transformer["wte"].weight.shape[1]
    value_gradient = analytic[wte_width:]
    assert bool(value_gradient.ne(0).any())

    generator = t.Generator().manual_seed(901)
    perturbation = t.randn(value_gradient.shape, generator=generator)
    perturbation /= perturbation.norm()
    epsilon = 0.25
    suffix_start = len(cell._gcg_pairs[0].positive.prompt_prefix)

    def perturbed_score(sign: float) -> float:
        def inject(_module: object, _inputs: tuple[object, ...], output: t.Tensor) -> t.Tensor:
            changed = output.clone()
            changed[:, suffix_start, :] += (sign * epsilon * perturbation).to(output)
            return changed

        handle = value_modules[0][1].register_forward_hook(inject)
        try:
            return cell.gcg_score([suffix], [0], candidate_eval_device_batch_size=4)[0]
        finally:
            handle.remove()

    finite_difference = (perturbed_score(1.0) - perturbed_score(-1.0)) / (2 * epsilon)
    predicted = float(value_gradient.dot(perturbation))
    assert finite_difference == pytest.approx(predicted, rel=0.1, abs=2e-3)


@pytest.mark.parametrize("factor", ADV_FULL_FACTORS)
@pytest.mark.parametrize("dtype", [t.bfloat16, t.float32])
def test_project_activation_offset_matches_project_hook_scalar_order(
    monkeypatch: pytest.MonkeyPatch,
    factor: float,
    dtype: t.dtype,
) -> None:
    """Match the hook offset and return the unscaled mean of both row losses."""

    monkeypatch.setattr(encoder_module, "COMPUTE_DTYPE", dtype)
    monkeypatch.setattr(eval_residual_module, "COMPUTE_DTYPE", dtype)
    monkeypatch.setattr(model_module, "COMPUTE_DTYPE", dtype)
    cell = _cell(activation=True, factor=factor)
    model = cell._runtime._target_model
    direction = cell._gcg_direction
    assert direction is not None
    direction = replace(direction, max_activation=0.1)
    cell._gcg_direction = direction
    model.eval()
    residual = t.zeros(1, 1, model.dec_config.n_embd, dtype=dtype)
    with post_cross_attention_residual_counter(
        model,
        layer=int(direction.provenance["layer"]),
        direction=direction.direction,
        magnitude=direction.max_activation * factor,
    ):
        callback = model._post_cross_attention_eval_callback
        if factor == 0.0:
            assert callback is None
            hook_offset = t.zeros_like(residual)
        else:
            assert callback is not None
            hook_offset = -callback(int(direction.provenance["layer"]), residual)
    hook_offset = hook_offset[0, 0]

    observed: list[t.Tensor] = []
    expected_losses: list[float] = []
    real_activation_l2 = self_conditioning_module.activation_l2_row

    def capture_offset(
        attacked_hidden: t.Tensor,
        clean_hidden: t.Tensor,
        direction_offset: t.Tensor,
    ) -> t.Tensor:
        observed.append(direction_offset.detach().cpu().clone())
        delta = attacked_hidden.float() + hook_offset.float() - clean_hidden.detach().float()
        expected_losses.append(float(delta.square().sum(dim=-1).sqrt().mean()))
        return real_activation_l2(attacked_hidden, clean_hidden, direction_offset)

    monkeypatch.setattr(self_conditioning_module, "activation_l2_row", capture_offset)
    [score] = cell.gcg_score([(12,)], [0], candidate_eval_device_batch_size=4)
    assert len(observed) == 2
    for offset in observed:
        assert offset.dtype is dtype
        assert t.equal(offset, hook_offset)
    assert score == pytest.approx(sum(expected_losses) / 2, rel=1e-6, abs=1e-7)
