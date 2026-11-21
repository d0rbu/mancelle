"""Exact seed parity and independent continuous-coordinate checks for project suffixes."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch as t

import nanochat.flash_attention as attention
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder import encoder as encoder_module
from steering.conditional.encoder_decoder import model as model_module
from steering.eval.axbench_adv import self_conditioning
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.self_conditioning import _project_embedding_weight, _ProjectGcgPair
from tests.axbench_adv.unit.test_gcg_project_adapter import _cell
from tests.axbench_adv.unit.test_gcg_project_response_weights import _prepare_cell
from tests.conftest import MockTokenizer


@pytest.fixture(autouse=True)
def _fp32_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attention, "_override_impl", FlashAttnImpl.SDPA)
    monkeypatch.setattr(model_module, "COMPUTE_DTYPE", t.float32)
    monkeypatch.setattr(encoder_module, "COMPUTE_DTYPE", t.float32)
    monkeypatch.setattr(self_conditioning, "COMPUTE_DTYPE", t.float32)


def _soft_cell(*, activation: bool, count: int = 2, factor: float = 1.0) -> tuple[Any, t.Tensor]:
    cell = _cell(activation=activation, factor=factor)
    pair = cell._gcg_pairs[0]
    cell._gcg_pairs = (
        pair,
        _ProjectGcgPair(
            replace(pair.negative, prompt_prefix=(1, 2, 3, 4, 3), prompt_prefix_seg=(0,) * 5),
            pair.positive,
        ),
    )
    cell._gcg_seed = tuple(range(12, 12 + count))
    cell._closed = False
    cell._soft_prepared = True
    cell._runtime._active_cell = cell
    if activation:
        assert cell._gcg_direction is not None
        cell._gcg_direction = replace(cell._gcg_direction, provenance={"layer": 1})
    embeddings = _project_embedding_weight(cell._runtime._target_model)[list(cell._gcg_seed)]
    return cell, embeddings


@pytest.mark.parametrize("activation", [False, True])
@pytest.mark.parametrize("factor", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_soft_initialization_is_exact_gcg_score_and_vjp(
    activation: bool,
    factor: float,
    count: int,
    batch_size: int,
) -> None:
    cell, embeddings = _soft_cell(activation=activation, count=count, factor=factor)
    cell._gcg_device_batch_size = batch_size
    indices = [1, 0]
    expected_scores = cell.gcg_score(
        [cell._gcg_seed], indices, candidate_eval_device_batch_size=batch_size
    )
    expected_gradients = cell.gcg_gradient(cell._gcg_seed, indices)
    assert cell.soft_score(embeddings, indices) == expected_scores
    actual_gradients = cell.soft_gradient(embeddings, indices)
    for expected, actual in zip(expected_gradients, actual_gradients, strict=True):
        assert t.equal(expected, actual)
        assert actual.device.type == "cpu" and actual.dtype is t.float32
        assert not actual.requires_grad


@pytest.mark.parametrize("activation", [False, True])
@pytest.mark.parametrize("lookup_dtype", [t.float32, t.bfloat16])
def test_bf16_compute_seed_parity_including_native_lookup_cast(
    activation: bool, lookup_dtype: t.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_module, "COMPUTE_DTYPE", t.bfloat16)
    monkeypatch.setattr(encoder_module, "COMPUTE_DTYPE", t.bfloat16)
    monkeypatch.setattr(self_conditioning, "COMPUTE_DTYPE", t.bfloat16)
    cell, _embeddings = _soft_cell(activation=activation)
    for _name, module in cell._soft_embedding_modules():
        module.to(dtype=lookup_dtype)
    embeddings = _project_embedding_weight(cell._runtime._target_model)[list(cell._gcg_seed)]
    assert cell.soft_score(embeddings, [0]) == cell.gcg_score(
        [cell._gcg_seed], [0], candidate_eval_device_batch_size=cell._gcg_device_batch_size
    )
    assert t.equal(
        cell.soft_gradient(embeddings, [0])[0], cell.gcg_gradient(cell._gcg_seed, [0])[0]
    )


@pytest.mark.parametrize("alpha", [0.0, 0.01, 1.0])
def test_soft_pair_arithmetic_retains_each_class_weight(alpha: float) -> None:
    cell, embeddings = _soft_cell(activation=False)
    cell._gcg_alpha = alpha
    assert cell.soft_score(embeddings, [0]) == cell.gcg_score(
        [cell._gcg_seed],
        [0],
        candidate_eval_device_batch_size=cell._gcg_device_batch_size,
    )
    assert t.equal(
        cell.soft_gradient(embeddings, [0])[0], cell.gcg_gradient(cell._gcg_seed, [0])[0]
    )


@pytest.mark.parametrize("activation", [False, True])
def test_nonseed_full_tuple_matches_independent_injection_and_finite_differences(
    activation: bool,
) -> None:
    cell, seed = _soft_cell(activation=activation)
    cell._gcg_pairs = cell._gcg_pairs[:1]
    cell._gcg_device_batch_size = 4
    if activation:
        cell._ensure_clean_l2()
    generator = t.Generator().manual_seed(917)
    embeddings = seed + t.randn(seed.shape, generator=generator) * 0.1
    actual = cell.soft_score(embeddings, [0])[0]
    analytic = cell.soft_gradient(embeddings, [0])[0]
    modules = cell._soft_embedding_modules()
    handles = []
    offset = 0
    try:
        for _name, module in modules:
            width = module.weight.shape[1]
            values = embeddings[:, offset : offset + width].clone()
            offset += width

            def oracle(
                _module: Any, _args: Any, output: t.Tensor, *, values: t.Tensor = values
            ) -> t.Tensor:
                with t.no_grad():
                    result = output.detach().clone()
                    result[:, 4 : 4 + len(cell._gcg_seed)] = values.to(result)
                return result

            handles.append(module.register_forward_hook(oracle))
        expected = cell.gcg_score([cell._gcg_seed], [0], candidate_eval_device_batch_size=4)[0]
        expected_gradient = cell.gcg_gradient(cell._gcg_seed, [0])[0]
    finally:
        for handle in handles:
            handle.remove()
    assert actual == expected
    assert t.equal(analytic, expected_gradient)
    assert actual != cell.soft_score(seed, [0])[0]
    # Probe each independently optimized lookup coordinate, including value tables.
    offset = 0
    for _name, module in modules:
        width = module.weight.shape[1]
        direction = t.zeros_like(embeddings)
        part = t.randn((embeddings.shape[0], width), generator=generator)
        part /= part.norm()
        direction[:, offset : offset + width] = part
        offset += width
        epsilon = 0.001
        plus = cell.soft_score(embeddings + epsilon * direction, [0])[0]
        minus = cell.soft_score(embeddings - epsilon * direction, [0])[0]
        expected_slope = float((analytic * direction).sum())
        assert expected_slope != 0.0
        assert (plus - minus) / (2 * epsilon) == pytest.approx(expected_slope, rel=0.03, abs=0.002)


def test_activation_l2_keeps_later_value_coordinates_zero() -> None:
    cell, embeddings = _soft_cell(activation=True)
    cell._gcg_direction = replace(cell._gcg_direction, provenance={"layer": 0})
    gradient = cell.soft_gradient(embeddings + 0.13, [0])[0]
    width = cell._runtime._target_model.transformer["wte"].weight.shape[1]
    assert bool(gradient[:, :width].ne(0).any())
    assert t.equal(gradient[:, width:], t.zeros_like(gradient[:, width:]))


@pytest.mark.parametrize("bad", ["shape", "dtype", "nan", "inf", "compute_overflow"])
def test_soft_rejects_invalid_master_and_native_casts(
    bad: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell, embeddings = _soft_cell(activation=False)
    if bad == "shape":
        embeddings = embeddings[:, :-1]
    elif bad == "dtype":
        embeddings = embeddings.double()
    else:
        embeddings[0, 0] = {
            "nan": float("nan"),
            "inf": float("inf"),
            "compute_overflow": t.finfo(t.float32).max,
        }[bad]
    if bad == "compute_overflow":
        monkeypatch.setattr(model_module, "COMPUTE_DTYPE", t.bfloat16)
        assert bool(t.isfinite(embeddings).all())
        assert cell._runtime._target_model.transformer["wte"].weight.dtype is t.float32
    for operation in (cell.soft_score, cell.soft_gradient):
        with pytest.raises(AxBenchAdvError, match="finite FP32|overflows"):
            operation(embeddings, [0])


@pytest.mark.parametrize("indices", [[], [True], [-1], [2], [0, 0]])
def test_soft_rejects_malformed_member_indices(indices: list[int]) -> None:
    cell, embeddings = _soft_cell(activation=False)
    with pytest.raises(AxBenchAdvError, match="member indices"):
        cell.soft_score(embeddings, indices)
    with pytest.raises(AxBenchAdvError, match="member indices"):
        cell.soft_gradient(embeddings, indices)


def test_soft_prepare_identity_and_failed_reprepare_invalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = _prepare_cell(monkeypatch, activation=False)
    cell._closed = False
    cell._runtime._active_cell = cell
    cell._runtime._target_tokenizer = MockTokenizer()
    cell._runtime._gcg_activation_directions = {}
    cell._concept_id = 0
    cell._method = "SelfConditioning"
    arguments = {
        "objective": "response_margin",
        "alpha": 0.01,
        "suffix_token_count": 1,
        "device_batch_size": 2,
        "separator": "\n\n",
        "rows": [
            {
                "pair_id": 0,
                "positive": {"sample_id": 1, "prompt": "x", "response": "ab"},
                "negative": {"sample_id": 2, "prompt": "x", "response": "bc"},
            }
        ],
    }
    legacy = cell.prepare_gcg(**arguments)
    prepared = cell.prepare_soft_prompt(**arguments)
    expected_layout = [
        {"name": name, "width": module.weight.shape[1], "dtype": str(module.weight.dtype)}
        for name, module in cell._soft_embedding_modules()
    ]
    assert prepared.identity == {
        **legacy.identity,
        "embedding_layout": expected_layout,
        "generation_conditioning": "clean_prompt_S0_dynamic_decode",
        "compute_dtype": "torch.float32",
    }
    assert t.equal(prepared.embedding_weight, legacy.embedding_weight)
    embeddings = prepared.embedding_weight[list(prepared.seed_token_ids)]
    assert len(cell.soft_score(embeddings, [0])) == 1
    with pytest.raises(AxBenchAdvError, match="unsupported separator/objective"):
        cell.prepare_soft_prompt(**{**arguments, "objective": "wrong"})
    for operation in (cell.soft_score, cell.soft_gradient, cell.evaluate_soft):
        with pytest.raises(AxBenchAdvError, match="prepare_soft_prompt must succeed"):
            operation(embeddings, [0])
    cell.prepare_soft_prompt(**arguments)
    assert len(cell.soft_gradient(embeddings, [0])) == 1
    cell.prepare_gcg(**arguments)
    with pytest.raises(AxBenchAdvError, match="prepare_soft_prompt must succeed"):
        cell.soft_score(embeddings, [0])


@pytest.mark.parametrize("fail", [False, True])
def test_soft_backward_preserves_caller_state_and_hooks(
    fail: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell, embeddings = _soft_cell(activation=False)
    model = cell._runtime._target_model
    model.train()
    model.transformer["h"][0].eval()
    parameter = next(model.parameters())
    parameter.grad = t.full_like(parameter, 0.5)
    original_grad = parameter.grad
    values = [p.detach().clone() for p in model.parameters()]
    flags = [p.requires_grad for p in model.parameters()]
    modes = [module.training for module in model.modules()]
    rng = t.random.get_rng_state().clone()
    hooks = [(module._forward_hooks, module._forward_hooks.copy()) for module in model.modules()]
    if fail:

        def failed_backward(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("injected backward failure")

        monkeypatch.setattr(t.autograd, "grad", failed_backward)
        with pytest.raises(RuntimeError, match="injected backward"):
            cell.soft_gradient(embeddings, [0])
    else:
        with t.inference_mode():
            assert len(cell.soft_gradient(embeddings, [0])) == 1
            assert t.is_inference_mode_enabled()
    assert [module.training for module in model.modules()] == modes
    assert [p.requires_grad for p in model.parameters()] == flags
    assert parameter.grad is original_grad and bool(parameter.grad.eq(0.5).all())
    assert t.equal(t.random.get_rng_state(), rng)
    for value, current in zip(values, model.parameters(), strict=True):
        assert t.equal(value, current)
    for (registry, contents), module in zip(hooks, model.modules(), strict=True):
        assert module._forward_hooks is registry and registry == contents
