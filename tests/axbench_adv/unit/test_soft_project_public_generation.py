"""Public PROJECT soft generation against an independently sampled tiny decoder."""

from __future__ import annotations

import copy
from threading import Event
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t

import nanochat.flash_attention as attention
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder import encoder as encoder_module
from steering.conditional.encoder_decoder import model as model_module
from steering.conditional.encoder_decoder.eval_adapter import TargetAwareEncoderDecoderEvalWrapper
from steering.eval.axbench_adv import project_soft_prompt, self_conditioning
from steering.eval.axbench_adv.gepa_optimization import OptimizationExample, RawEvaluation
from steering.eval.axbench_adv.self_conditioning import _BoundSelfConditioningCell
from steering.eval.axbench_upstream.methods import PreparedPromptContext
from steering.eval.axbench_upstream.run_methods import derive_generation_seed, derive_selection_seed
from steering.eval.target_conditioning import SelfConditionedTarget
from tests.axbench_adv.unit.test_soft_project_generation import _fixture, _reference_prefill
from tests.steering.test_self_conditioned_wrapper import _new_kv


@pytest.fixture(autouse=True)
def _fp32_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attention, "_override_impl", FlashAttnImpl.SDPA)
    monkeypatch.setattr(model_module, "COMPUTE_DTYPE", t.float32)
    monkeypatch.setattr(encoder_module, "COMPUTE_DTYPE", t.float32)


def _public_cell(
    monkeypatch: pytest.MonkeyPatch, *, turn: bool, inflight: int, reuse_mask: bool
) -> tuple[Any, dict[str, Any], list[Any]]:
    fixture = _fixture(turn, 3 if turn else 64)
    config, provider = fixture["config"], fixture["common"]["context_provider"]
    prepared = PreparedPromptContext(
        tuple(fixture["clean"]), fixture["tokenizer"].encode_special("<assistant_end>"),
        provider.get(fixture["clean"]), config.segment_mode, provider.token_assignments,
    )
    runtime = SimpleNamespace(
        _concepts={0: "topic"},
        _originals={
            (0, index): SimpleNamespace(raw_prompt="q", prepared=prepared) for index in range(10)
        },
        _target_model=fixture["model"], _target_tokenizer=fixture["tokenizer"],
        _nanochat_tokenizer=fixture["tokenizer"], _master_seed=42,
        _max_new_tokens=8, _max_prompt_tokens=56, _judge_inflight=inflight,
        _reuse_single_token_mask=reuse_mask,
        _dense_diffmeans={0: config.score_policy.target_scores},
        _setup=SimpleNamespace(
            output_negative_selection=config.score_policy.negative_selection,
            output_selection=config.selection.selection,
            percentile_table=config.selection.percentile_table,
            n_sae_features=config.n_sae_features, feature_mask=None, consolidation=None,
            segment_mode=config.segment_mode, turn_self_cond_segment_size_rustbpe=3,
        ),
    )
    cell = _BoundSelfConditioningCell(
        cast(Any, runtime), method="SelfConditioning", concept_id=0, concept="topic", factor=3.0,
        examples=[OptimizationExample(index, "q") for index in range(10)],
    )
    runtime._active_cell = cell
    cell._soft_prepared = True
    cell._gcg_seed = tuple(fixture["tokenizer"].encode("!"))
    wrappers: list[Any] = []
    wrapper_class = project_soft_prompt.ProjectSoftPromptWrapper

    def capture_wrapper(**kwargs: Any) -> Any:
        wrapper = wrapper_class(**kwargs)
        wrappers.append(wrapper)
        return wrapper

    def unexpected_extraction(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("public soft evaluation must reuse the prepared clean features")

    monkeypatch.setattr(project_soft_prompt, "ProjectSoftPromptWrapper", capture_wrapper)
    monkeypatch.setattr(self_conditioning, "prepare_prompt_context", unexpected_extraction)
    return cell, fixture, wrappers


def _expected_text(fixture: dict[str, Any], input_id: int) -> str:
    """Manual attacked prefill plus top-50 sampling, without the soft generation entry."""
    config = copy.deepcopy(fixture["config"])
    config.selection.rng.manual_seed(derive_selection_seed(42, 0, input_id))
    reference = TargetAwareEncoderDecoderEvalWrapper(
        **fixture["common"], target=SelfConditionedTarget(), self_cond_config=config,
    )
    reference_fixture = {**fixture, "baseline": reference}
    rng = t.Generator().manual_seed(derive_generation_seed(42, "SelfConditioning", 0, input_id, 3.0))
    cache = _new_kv(fixture["model"])
    stops = {0, 67} | set(reference.generation_stop_token_ids)
    generated: list[int] = []
    with t.inference_mode():
        logits, _seed = _reference_prefill(reference_fixture, cache, attack=True)
        for position in range(8):
            values, ids = t.topk(logits[:, -1, :], 50, dim=-1)
            choice = t.multinomial(t.softmax(values, dim=-1), 1, generator=rng)
            token = ids.gather(1, choice)
            if int(token.item()) in stops:
                break
            generated.append(int(token.item()))
            if position < 7:
                logits = reference(token, kv_cache=cache)
    assert len(generated) >= 3, "the oracle must exercise multiple decoder inputs"
    return fixture["tokenizer"].decode(generated)


def _assert_restored(model: Any, wrappers: list[Any], hooks: list[Any]) -> None:
    assert model._single_token_mask_cache is None
    assert model._single_token_mask_cache_enabled is False
    assert all(wrapper._active_context_token_ids is None for wrapper in wrappers)
    for module, (registry, contents) in zip(model.modules(), hooks, strict=True):
        assert module._forward_hooks is registry and registry == contents


@pytest.mark.parametrize(
    ("turn", "inflight", "reuse_mask"), [(False, 1, False), (True, 1, True), (True, 2, False)]
)
def test_public_soft_evaluation_replays_sampling_and_order_without_reextracting(
    turn: bool, inflight: int, reuse_mask: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell, fixture, wrappers = _public_cell(
        monkeypatch, turn=turn, inflight=inflight, reuse_mask=reuse_mask
    )
    expected = [_expected_text(fixture, index) for index in (8, 3)]
    assert expected[0] != expected[1]
    model = fixture["model"]
    values = [parameter.detach().clone() for parameter in model.parameters()]
    modes = [module.training for module in model.modules()]
    flags = [parameter.requires_grad for parameter in model.parameters()]
    hooks = [(module._forward_hooks, module._forward_hooks.copy()) for module in model.modules()]
    embeddings = fixture["embeddings"].clone()
    cache = cell._cache.copy()
    completed: list[int] = []
    first_started, second_completed = Event(), Event()

    class Judge:
        def evaluate(self, *, generation: str, input_id: int) -> RawEvaluation:
            if inflight == 2:
                if input_id == 8:
                    first_started.set()
                    assert second_completed.wait(10)
                else:
                    assert first_started.wait(10)
            completed.append(input_id)
            if input_id == 3:
                second_completed.set()
            return RawEvaluation(generation, 1.0, 1.5, str(input_id), 0)

    cell._judges = cast(Any, (Judge(), Judge()) if inflight == 2 else (Judge(),))
    for _ in range(2):
        completed.clear()
        first_started.clear()
        second_completed.clear()
        results = cell.evaluate_soft(fixture["embeddings"], [8, 3])
        assert [result.generation for result in results] == expected
        assert [result.judge_rationale for result in results] == ["8", "3"]
        assert completed == ([3, 8] if inflight == 2 else [8, 3])
        _assert_restored(model, wrappers, hooks)
    assert cell._cache == cache
    assert [module.training for module in model.modules()] == modes
    assert [parameter.requires_grad for parameter in model.parameters()] == flags
    assert t.equal(embeddings, fixture["embeddings"])
    for before, after in zip(values, model.parameters(), strict=True):
        assert t.equal(before, after)


@pytest.mark.parametrize("failure_site", ["embedding", "judge"])
def test_public_soft_evaluation_failure_restores_state_and_allows_fresh_replay(
    failure_site: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell, fixture, wrappers = _public_cell(monkeypatch, turn=True, inflight=1, reuse_mask=True)
    expected = _expected_text(fixture, 8)
    model = fixture["model"]
    hooks = [(module._forward_hooks, module._forward_hooks.copy()) for module in model.modules()]
    fail = True

    class Judge:
        def evaluate(self, *, generation: str, input_id: int) -> RawEvaluation:
            if fail and failure_site == "judge":
                raise RuntimeError("public judge failure")
            return RawEvaluation(generation, 1.0, 1.5, str(input_id), 0)

    cell._judges = cast(Any, (Judge(),))
    module = fixture["modules"][-1][1]
    original = module.forward

    def forward(ids: t.Tensor) -> t.Tensor:
        if fail and failure_site == "embedding" and ids.shape[1] == len(fixture["attacked"]) - 1:
            raise RuntimeError("public embedding failure")
        return original(ids)

    monkeypatch.setattr(module, "forward", forward)
    with pytest.raises(RuntimeError, match=f"public {failure_site} failure"):
        cell.evaluate_soft(fixture["embeddings"], [8])
    _assert_restored(model, wrappers, hooks)
    assert cell._soft_prepared and cell._runtime._active_cell is cell
    fail = False
    assert cell.evaluate_soft(fixture["embeddings"], [8])[0].generation == expected
    _assert_restored(model, wrappers, hooks)
