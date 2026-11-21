"""Frozen-response byte weights are prepared once without changing the objective."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch as t

import nanochat.flash_attention as attention
from nanochat.flash_attention import FlashAttnImpl
from steering.eval.axbench_adv import self_conditioning
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.self_conditioning import _ProjectGcgPair
from tests.axbench_adv.unit.test_gcg_project_adapter import _cell
from tests.conftest import MockTokenizer


class _ResponseTokenizer(MockTokenizer):
    def __init__(self, *, empty: bool = False) -> None:
        self.decoded: list[list[int]] = []
        self.empty = empty

    def decode(self, ids: list[int]) -> str:
        self.decoded.append(ids.copy())
        if self.empty:
            return ""
        return "".join({99: "a", 100: "é", 101: "🙂"}[token] for token in ids)


def _prepare_cell(monkeypatch: pytest.MonkeyPatch, *, activation: bool) -> Any:
    monkeypatch.setattr(attention, "_override_impl", FlashAttnImpl.SDPA)
    cell = _cell(activation=activation)
    template = cell._gcg_pairs[0].positive
    runtime = cast(Any, cell._runtime)
    runtime._nanochat_tokenizer = MockTokenizer()
    runtime._target_tokenizer = _ResponseTokenizer()
    runtime._max_prompt_tokens = 64
    runtime._prompt_rng = t.Generator().manual_seed(10)
    runtime._setup = SimpleNamespace()
    runtime._extraction_model = runtime._sae = None
    monkeypatch.setattr(cell, "_policy", lambda: None)
    monkeypatch.setattr(
        self_conditioning,
        "prepare_prompt_context",
        lambda *_args, **kwargs: SimpleNamespace(
            token_ids=kwargs["token_ids"],
            segments=(),
            segment_mode="context_independent",
            token_assignments=None,
        ),
    )
    monkeypatch.setattr(self_conditioning, "SingleEntryContextProvider", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        self_conditioning, "build_self_conditioning_config", lambda *_a, **_kw: None
    )
    monkeypatch.setattr(
        self_conditioning,
        "TargetAwareEncoderDecoderEvalWrapper",
        lambda **_kwargs: SimpleNamespace(
            _build_conditioning=lambda _ids, **_kw: (
                template.encoder_out,
                template.clean_seg_map,
            ),
            _self_cond=object(),
        ),
    )
    monkeypatch.setattr(
        self_conditioning,
        "seed_self_conditioning_from_prompt",
        lambda **_kwargs: SimpleNamespace(
            state=SimpleNamespace(encoder_out=template.encoder_out, target_slot_idx=1)
        ),
    )
    return cell


@pytest.mark.parametrize("activation", [False, True])
@pytest.mark.parametrize("inference_mode", [False, True])
def test_prepared_response_weights_follow_target_tokenizer_and_are_reused(
    monkeypatch: pytest.MonkeyPatch, activation: bool, inference_mode: bool
) -> None:
    cell = _prepare_cell(monkeypatch, activation=activation)
    tokenizer = cell._runtime._target_tokenizer
    rng = cell._runtime._prompt_rng.get_state().clone()
    direction = cell._gcg_direction
    # Public preparation sets the new direction after constructing the rows.
    cell._gcg_direction = None if activation else _cell(activation=True)._gcg_direction
    with t.inference_mode(inference_mode):
        prepared = [
            cell._prepare_gcg_row(
                prompt="x",
                response=response,
                sample_id=31,
                seed_text="!",
                objective="activation_l2" if activation else "response_margin",
            )[0]
            for response in ("abc", "cab")
        ]
        assert t.is_inference_mode_enabled() is inference_mode
        assert t.is_grad_enabled() is not inference_mode
    assert t.equal(cell._runtime._prompt_rng.get_state(), rng)
    cell._gcg_direction = direction
    assert [row.response for row in prepared] == [(99, 100, 101), (101, 99, 100)]
    assert tokenizer.decoded == ([] if activation else [[99], [100], [101], [101], [99], [100]])
    for row, expected in zip(prepared, ([1, 2, 4], [4, 1, 2]), strict=True):
        weights = row.response_byte_weights
        if activation:
            assert weights is None
        else:
            assert weights is not None
            assert weights.device.type == "cpu" and weights.dtype is t.float32
            assert not weights.requires_grad
            assert not weights.is_inference()
            assert t.equal(weights, t.tensor(expected, dtype=t.float32))

    def unexpected_decode(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("search must reuse the frozen response weights")

    monkeypatch.setattr(self_conditioning, "_project_token_bytes", unexpected_decode)
    cell._gcg_pairs = (_ProjectGcgPair(*prepared),)
    reference_pairs = (
        _ProjectGcgPair(
            replace(prepared[0], response_byte_weights=t.tensor([1, 2, 4], dtype=t.float32)),
            replace(prepared[1], response_byte_weights=t.tensor([4, 1, 2], dtype=t.float32)),
        ),
    )
    before = [row.response_byte_weights for row in prepared]
    for _ in range(2):
        scores = cell.gcg_score([(35,), (36,)], [0], candidate_eval_device_batch_size=4)
        assert len(scores) == 2
        [gradient] = cell.gcg_gradient((35,), [0])
        assert bool(t.isfinite(gradient).all()) and bool(gradient.ne(0).any())
        if not activation:
            with patch.object(cell, "_gcg_pairs", reference_pairs):
                assert scores == cell.gcg_score(
                    [(35,), (36,)], [0], candidate_eval_device_batch_size=4
                )
                [reference_gradient] = cell.gcg_gradient((35,), [0])
            assert t.equal(gradient, reference_gradient)
    after = cell._gcg_pairs[0]
    assert after.positive.response_byte_weights is before[0]
    assert after.negative.response_byte_weights is before[1]


@pytest.mark.slow
def test_inference_preparation_cuda_transfer_keeps_weights_usable_by_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not t.cuda.is_available():
        pytest.skip("CUDA is required to check the preparation device transfer")
    cell = _prepare_cell(monkeypatch, activation=False)
    cell._runtime._target_model.to("cuda")
    with t.inference_mode():
        row, _seed = cell._prepare_gcg_row(
            prompt="x",
            response="abc",
            sample_id=31,
            seed_text="!",
            objective="response_margin",
        )
        assert t.is_inference_mode_enabled() and not t.is_grad_enabled()
    weights = row.response_byte_weights
    assert weights is not None and weights.is_cuda and not weights.is_inference()
    losses = t.tensor([0.5, 1.5, 2.5], device="cuda", requires_grad=True)
    reference = losses.detach().clone().requires_grad_()
    expected_weights = t.tensor([1, 2, 4], device="cuda", dtype=t.float32)
    actual = self_conditioning.weighted_bpb(losses, weights, tensor_ln2=True)
    expected = self_conditioning.weighted_bpb(reference, expected_weights, tensor_ln2=True)
    assert t.equal(actual, expected)
    assert t.equal(t.autograd.grad(actual, losses)[0], t.autograd.grad(expected, reference)[0])


@pytest.mark.parametrize("objective", ["response_margin", "activation_l2"])
@pytest.mark.parametrize("inference_mode", [False, True])
def test_zero_byte_response_is_validated_only_for_response_margin(
    monkeypatch: pytest.MonkeyPatch, objective: str, inference_mode: bool
) -> None:
    cell = _prepare_cell(monkeypatch, activation=objective == "activation_l2")
    cell._runtime._target_tokenizer = _ResponseTokenizer(empty=True)
    arguments = {
        "prompt": "x",
        "response": "abc",
        "sample_id": 31,
        "seed_text": "!",
        "objective": objective,
    }
    with t.inference_mode(inference_mode):
        if objective == "response_margin":
            with pytest.raises(AxBenchAdvError, match="response tokens decode to zero bytes"):
                cell._prepare_gcg_row(**arguments)
        else:
            row, _seed = cell._prepare_gcg_row(**arguments)
            assert row.response_byte_weights is None
            assert cell._runtime._target_tokenizer.decoded == []
        assert t.is_inference_mode_enabled() is inference_mode
        assert t.is_grad_enabled() is not inference_mode
