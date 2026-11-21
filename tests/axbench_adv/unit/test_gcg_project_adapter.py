"""CPU functional checks for project-model GCG batching and VJPs."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch as t

import nanochat.flash_attention as attention
from nanochat.flash_attention import FlashAttnImpl
from steering.eval.axbench_adv.counter import GcgActivationDirection
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.self_conditioning import (
    _BoundSelfConditioningCell,
    _project_token_bytes,
    _ProjectGcgPair,
    _ProjectGcgRow,
    _validate_project_gcg_context_budget,
    _value_embedding_modules,
)
from tests.axbench_adv.gcg_gradient_reference import full_sequence_logits
from tests.conftest import TINY_N_EMBD, MockTokenizer
from tests.steering.test_self_conditioned_wrapper import _build_fp_encdec


@pytest.fixture(autouse=True)
def _force_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attention, "_override_impl", FlashAttnImpl.SDPA)


def _cell(*, activation: bool, factor: float = 1.0) -> _BoundSelfConditioningCell:
    with t.random.fork_rng(), t.no_grad():
        t.manual_seed(90210)
        model = _build_fp_encdec()
        generator = t.Generator().manual_seed(90210)
        for parameter in model.parameters():
            if parameter.ndim >= 2 and not bool(parameter.ne(0).any()):
                parameter.copy_(
                    t.randn(parameter.shape, generator=generator, dtype=parameter.dtype) * 0.02
                )
        feature_ids = t.tensor([[[0], [1]]], dtype=t.long)
        magnitudes = t.ones_like(feature_ids)
        encoder_out = model.encoder.forward_packed(
            feature_ids,
            magnitudes,
            t.ones_like(feature_ids, dtype=t.bool),
        )
    first = _ProjectGcgRow(
        raw_prompt="first",
        prompt_prefix=(1, 2, 3, 4),
        prompt_prefix_seg=(0, 0, 0, 0),
        prompt_tail=(5, 6),
        clean_prompt=(1, 2, 3, 5, 6),
        response=(7, 8),
        response_byte_weights=(
            None if activation else _project_token_bytes(MockTokenizer(), (7, 8))
        ),
        clean_seg_map=t.tensor([[0, 0, 0, 0, 1]]),
        encoder_out=encoder_out,
        target_slot=1,
        source_slot=0,
        clean_l2=None,
    )
    second = replace(
        first,
        raw_prompt="second",
        response=(9, 10, 11),
        response_byte_weights=(
            None if activation else _project_token_bytes(MockTokenizer(), (9, 10, 11))
        ),
    )
    cell = object.__new__(_BoundSelfConditioningCell)
    cell._factor = factor
    cell._runtime = cast(
        Any, SimpleNamespace(_target_model=model, _target_tokenizer=MockTokenizer())
    )
    cell._gcg_pairs = (_ProjectGcgPair(first, second),)
    cell._gcg_alpha = 0.01
    cell._gcg_device_batch_size = 4
    cell._gcg_direction = (
        GcgActivationDirection(
            direction=t.linspace(-0.3, 0.4, TINY_N_EMBD, dtype=t.bfloat16),
            max_activation=1.25,
            provenance={"layer": 0},
        )
        if activation
        else None
    )
    return cell


def test_project_response_margin_batches_unequal_responses_and_preserves_state() -> None:
    cell = _cell(activation=False)
    model = cell._runtime._target_model
    model.eval()
    model.transformer["h"][0].train()
    modes = tuple(module.training for module in model.modules())
    requires_grad = tuple(parameter.requires_grad for parameter in model.parameters())
    cpu_rng = t.random.get_rng_state().clone()

    unbatched = cell.gcg_score([(12,), (13,)], [0], candidate_eval_device_batch_size=1)
    batched = cell.gcg_score([(12,), (13,)], [0], candidate_eval_device_batch_size=8)
    t.testing.assert_close(t.tensor(batched), t.tensor(unbatched), rtol=1e-5, atol=1e-5)
    [gradient] = cell.gcg_gradient((12,), [0])
    assert gradient.shape[0] == 1
    assert bool(t.isfinite(gradient).all()) and bool(gradient.ne(0).any())
    assert tuple(module.training for module in model.modules()) == modes
    assert tuple(parameter.requires_grad for parameter in model.parameters()) == requires_grad
    assert t.equal(t.random.get_rng_state(), cpu_rng)


def test_project_activation_l2_batches_clean_cache_and_repeated_vjp() -> None:
    cell = _cell(activation=True)
    unbatched = cell.gcg_score([(12,), (13,)], [0], candidate_eval_device_batch_size=1)
    batched = cell.gcg_score([(12,), (13,)], [0], candidate_eval_device_batch_size=8)
    t.testing.assert_close(t.tensor(batched), t.tensor(unbatched), rtol=1e-5, atol=1e-5)
    first = cell.gcg_gradient((12,), [0])[0]
    second = cell.gcg_gradient((12,), [0])[0]
    assert t.equal(first, second)
    assert bool(t.isfinite(first).all()) and bool(first.ne(0).any())
    for pair in cell._gcg_pairs:
        for row in (pair.positive, pair.negative):
            assert row.clean_l2 is not None
            assert row.clean_l2.device.type == "cpu"
            assert row.clean_l2.shape[0] == len(row.prompt_tail) + len(row.response)


@pytest.mark.parametrize("activation", [False, True])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_project_reduced_logits_preserve_clean_states_scores_and_gradients(
    activation: bool, batch_size: int
) -> None:
    cell = _cell(activation=activation)
    cell._gcg_device_batch_size = batch_size
    model = cell._runtime._target_model
    with patch.object(model.lm_head, "forward", wraps=model.lm_head.forward) as head:
        actual_scores = cell.gcg_score(
            [(12,), (13,)], [0], candidate_eval_device_batch_size=batch_size
        )
        actual_gradient = cell.gcg_gradient((12,), [0])[0]
    assert head.call_count > 0
    extents = [call.args[0].shape[1] for call in head.call_args_list]
    assert all((extent == 1) if activation else (extent > 1) for extent in extents)
    actual_clean = [
        row.clean_l2 for pair in cell._gcg_pairs for row in (pair.positive, pair.negative)
    ]
    cell._gcg_pairs = tuple(
        _ProjectGcgPair(
            replace(pair.positive, clean_l2=None), replace(pair.negative, clean_l2=None)
        )
        for pair in cell._gcg_pairs
    )
    with full_sequence_logits(model):
        expected_scores = cell.gcg_score(
            [(12,), (13,)], [0], candidate_eval_device_batch_size=batch_size
        )
        expected_gradient = cell.gcg_gradient((12,), [0])[0]
    assert actual_scores == expected_scores
    assert t.equal(actual_gradient, expected_gradient)
    expected_clean = [
        row.clean_l2 for pair in cell._gcg_pairs for row in (pair.positive, pair.negative)
    ]
    for actual, expected in zip(actual_clean, expected_clean, strict=True):
        if activation:
            assert actual is not None and expected is not None
            assert t.equal(actual, expected)
        else:
            assert actual is expected is None


@pytest.mark.parametrize("activation", [False, True])
def test_project_vjp_matches_all_embedding_tables_finite_difference(
    activation: bool,
) -> None:
    cell = _cell(activation=activation)
    model = cell._runtime._target_model
    token_ids = (12,)
    if activation:
        # The finite difference holds the detached clean reference fixed, as the
        # search objective does.  Populate it before perturbing attacked roots.
        cell.gcg_score([token_ids], [0], candidate_eval_device_batch_size=4)
    analytic = cell.gcg_gradient(token_ids, [0])[0][0]
    generator = t.Generator().manual_seed(443)
    perturbation = t.randn(analytic.shape, generator=generator)
    perturbation /= perturbation.norm()
    epsilon = 0.5

    modules = [
        model.transformer["wte"],
        *(module for _layer, module in _value_embedding_modules(model)),
    ]

    def perturbed_score(sign: float) -> float:
        handles = []
        offset = 0
        for module in modules:
            width = module.weight.shape[1]
            delta = perturbation[offset : offset + width]
            offset += width

            def inject(
                _module: object,
                _inputs: tuple[object, ...],
                output: t.Tensor,
                *,
                delta: t.Tensor = delta,
            ) -> t.Tensor:
                changed = output.clone()
                changed[:, 4, :] += (sign * epsilon * delta).to(output)
                return changed

            handles.append(module.register_forward_hook(inject))
        assert offset == perturbation.numel()
        try:
            return cell.gcg_score([token_ids], [0], candidate_eval_device_batch_size=4)[0]
        finally:
            for handle in handles:
                handle.remove()

    finite_difference = (perturbed_score(1.0) - perturbed_score(-1.0)) / (2 * epsilon)
    predicted = float(analytic.dot(perturbation))
    tolerance = 5e-3 if activation else 2e-3
    assert finite_difference == pytest.approx(predicted, rel=0.15, abs=tolerance)


def test_project_failure_restores_exact_hooks_grads_modes_and_rng() -> None:
    cell = _cell(activation=False)
    model = cell._runtime._target_model
    parameter = next(model.parameters())
    sentinel = t.full_like(parameter, 0.125)
    parameter.grad = sentinel
    model.eval()
    model.transformer["h"][0].train()
    modes = tuple(module.training for module in model.modules())
    cpu_rng = t.random.get_rng_state().clone()
    registry = model._forward_pre_hooks

    def fail(_module: object, _inputs: tuple[object, ...]) -> None:
        raise RuntimeError("injected project forward failure")

    handle = model.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="injected project forward failure"):
            cell.gcg_gradient((12,), [0])
        assert model._forward_pre_hooks is registry
        assert handle.id in registry
        assert parameter.grad is sentinel
        assert t.equal(parameter.grad, t.full_like(parameter, 0.125))
        assert tuple(module.training for module in model.modules()) == modes
        assert t.equal(t.random.get_rng_state(), cpu_rng)
    finally:
        handle.remove()


def test_project_rejects_nonfinite_scores_and_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = _cell(activation=False)

    def nonfinite_values(
        items: list[tuple[_ProjectGcgRow, tuple[int, ...] | None]],
        *,
        gradient: bool,
        batch_size: int,
        embeddings: t.Tensor | None = None,
    ) -> list[tuple[t.Tensor, t.Tensor | None]]:
        del batch_size, embeddings
        bad_gradient = t.full((1, TINY_N_EMBD), float("nan")) if gradient else None
        return [(t.tensor(float("nan")), bad_gradient) for _item in items]

    monkeypatch.setattr(cell, "_batched_project_values", nonfinite_values)
    with pytest.raises(AxBenchAdvError, match="score is non-finite"):
        cell.gcg_score([(12,)], [0], candidate_eval_device_batch_size=1)
    with pytest.raises(AxBenchAdvError, match="gradient is non-finite"):
        cell.gcg_gradient((12,), [0])


def test_project_cell_close_releases_gcg_state_even_when_judge_close_fails() -> None:
    cell = _cell(activation=True)

    class FailingJudge:
        def close(self) -> None:
            raise RuntimeError("judge close failed")

    cell._judges = (cast(Any, FailingJudge()),)
    cell._runtime._reuse_judge_workers = False
    cell._soft_prepared = True
    cell._cache = cast(Any, {(0, "suffix"): object()})
    cell._closed = False
    cell._runtime._active_cell = cell
    with pytest.raises(RuntimeError, match="judge close failed"):
        cell.close()
    assert cell._closed is True
    assert cell._runtime._active_cell is None
    assert cell._gcg_pairs == ()
    assert cell._gcg_seed == ()
    assert cell._gcg_allowed == frozenset()
    assert cell._gcg_direction is None
    assert cell._soft_prepared is False


def test_project_seed_prompt_budget_fails_before_search() -> None:
    _validate_project_gcg_context_budget(
        prompt_tokens=8,
        response_tokens=2,
        model_window=10,
        prompt_budget=8,
    )
    with pytest.raises(AxBenchAdvError, match="seed exceeds the decoder prompt budget"):
        _validate_project_gcg_context_budget(
            prompt_tokens=9,
            response_tokens=1,
            model_window=10,
            prompt_budget=8,
        )
