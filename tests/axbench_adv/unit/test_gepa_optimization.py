"""Root GEPA call-shape, reflection, accounting, and fail-closed tests."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any

import pytest
from gepa.core.result import GEPAResult

from steering.eval.axbench_adv import gepa_optimization
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.fitness import Adequacy
from steering.eval.axbench_adv.gepa_optimization import (
    REFLECTION_CACHE_TAG,
    SIDE_INFO_EXCERPT_CHARS,
    OpenRouterReflectionLM,
    OptimizationExample,
    RawEvaluation,
    build_openrouter_reflection_lm,
    optimize_suffix,
)
from steering.eval.axbench_base.config import LLMClientConfig
from steering.eval.axbench_base.llm_judge_client import OPENROUTER_API_KEY_ENV
from steering.eval.axbench_base.synth.llm_response_cache import make_cache_key
from steering.eval.axbench_base.types import AxBenchBaseError


class MemoryCache:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.gets: list[str] = []
        self.sets: list[tuple[str, Any]] = []

    def get(self, cache_key: str) -> str | None:
        self.gets.append(cache_key)
        return self.values.get(cache_key)

    def set(self, cache_key: str, response: str) -> None:
        self.sets.append((cache_key, response))
        self.values[cache_key] = response


class FakeClient:
    def __init__(self, response: Any = "```better```") -> None:
        self.response = response
        self.messages: list[list[dict[str, str]]] = []

    def complete(
        self, messages: list[dict[str, str]], *, cache_key: str | None = None
    ) -> str:
        assert cache_key is None
        self.messages.append(messages)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class StaticReflection:
    def __init__(self, response: Any = "```better```", paid: int = 0) -> None:
        self.response = response
        self._paid = paid
        self.prompts: list[object] = []

    @property
    def paid_cache_misses(self) -> int:
        return self._paid

    def __call__(self, prompt: object) -> str:
        self.prompts.append(prompt)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _examples(count: int = 4) -> list[OptimizationExample]:
    return [OptimizationExample(i, f"user message {i}") for i in range(count)]


def _raw(
    *,
    generation: Any = "A sufficiently long useful response.",
    concept: Any = 1.0,
    fluency: Any = 2.0,
    rationale: Any = "judge rationale",
    paid: Any = 0,
) -> RawEvaluation:
    return RawEvaluation(generation, concept, fluency, rationale, paid)


def _result(
    *,
    candidate: Any = None,
    score: float = 0.5,
    nominal: Any = 4,
    val_aggregate_subscores: Any = None,
    per_objective_best_candidates: Any = None,
    objective_pareto_front: Any = None,
) -> GEPAResult[Any, int]:
    if candidate is None:
        candidate = {"suffix": "  selected  "}
    return GEPAResult(
        candidates=[candidate],
        parents=[[None]],
        val_aggregate_scores=[score],
        val_subscores=[dict.fromkeys(range(4), score)],
        per_val_instance_best_candidates={i: {0} for i in range(4)},
        discovery_eval_counts=[0],
        val_aggregate_subscores=val_aggregate_subscores,
        per_objective_best_candidates=per_objective_best_candidates,
        objective_pareto_front=objective_pareto_front,
        total_metric_calls=nominal,
    )


def _base_kwargs(tmp_path: Path) -> dict[str, Any]:
    return {
        "examples": _examples(),
        "runtime": lambda _suffix, _input_id: _raw(),
        "reflection_lm": StaticReflection(),
        "objective": "Improve the suffix.",
        "run_dir": tmp_path / "gepa",
        "proposal_count": 1,
        "gepa_seed": 17,
    }


def test_openrouter_reflection_payload_cache_key_and_paid_counter() -> None:
    client = FakeClient("```improved suffix```")
    cache = MemoryCache()
    reflection = OpenRouterReflectionLM(client=client, model="provider/model", cache=cache)
    prompt = "exact GEPA prompt"
    expected_key = make_cache_key(
        model="provider/model", prompt=prompt, tag=REFLECTION_CACHE_TAG
    )

    assert reflection(prompt) == "```improved suffix```"
    assert reflection(prompt) == "```improved suffix```"
    assert client.messages == [[{"role": "user", "content": prompt}]]
    assert cache.gets == [expected_key, expected_key]
    assert cache.sets == [(expected_key, "```improved suffix```")]
    assert reflection.paid_cache_misses == 1


def test_openrouter_reflection_uses_valid_preexisting_cache_entry() -> None:
    cache = MemoryCache()
    prompt = "cached"
    key = make_cache_key(model="model", prompt=prompt, tag=REFLECTION_CACHE_TAG)
    cache.values[key] = "```cached response```"
    client = FakeClient(AssertionError("network path must be unreachable"))
    reflection = OpenRouterReflectionLM(client=client, model="model", cache=cache)
    assert reflection(prompt) == "```cached response```"
    assert reflection.paid_cache_misses == 0


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("client_blank", "OpenRouter reflection returned"),
        ("client_nonstring", "OpenRouter reflection returned"),
        ("cache_blank", "reflection cache contains"),
        ("cache_nonstring", "reflection cache contains"),
    ],
)
def test_openrouter_reflection_rejects_malformed_responses(source: str, expected: str) -> None:
    cache = MemoryCache()
    prompt = "prompt"
    key = make_cache_key(model="model", prompt=prompt, tag=REFLECTION_CACHE_TAG)
    response: Any = "" if source.endswith("blank") else 7
    if source.startswith("cache"):
        cache.values[key] = response
        client = FakeClient()
    else:
        client = FakeClient(response)
    reflection = OpenRouterReflectionLM(client=client, model="model", cache=cache)
    with pytest.raises(AxBenchAdvError, match=expected):
        reflection(prompt)
    assert reflection.paid_cache_misses == int(source.startswith("client"))
    assert cache.sets == []


def test_openrouter_reflection_rejects_nontext_prompt_and_propagates_client_error() -> None:
    cache = MemoryCache()
    with pytest.raises(AxBenchAdvError, match="textual GEPA prompt"):
        OpenRouterReflectionLM(client=FakeClient(), model="model", cache=cache)([])
    reflection = OpenRouterReflectionLM(
        client=FakeClient(RuntimeError("transport failed")), model="model", cache=cache
    )
    with pytest.raises(RuntimeError, match="transport failed"):
        reflection("prompt")
    assert reflection.paid_cache_misses == 1


def test_reflection_constructor_and_factory_validate_before_provider_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = MemoryCache()
    with pytest.raises(AxBenchAdvError, match="non-blank"):
        OpenRouterReflectionLM(client=FakeClient(), model=" ", cache=cache)
    with pytest.raises(AxBenchAdvError, match="provider must be openrouter"):
        build_openrouter_reflection_lm(
            cfg=LLMClientConfig(provider="other", model="model"), cache=cache
        )
    with pytest.raises(AxBenchAdvError, match="non-blank"):
        build_openrouter_reflection_lm(cfg=LLMClientConfig(model=" "), cache=cache)
    for retries in (True, -1):
        with pytest.raises(AxBenchAdvError, match="non-negative integer"):
            build_openrouter_reflection_lm(
                cfg=LLMClientConfig(model="model"), cache=cache, max_retries=retries
            )

    captured: dict[str, object] = {}

    def fake_client(cfg: LLMClientConfig, *, max_retries: int) -> FakeClient:
        captured.update(cfg=cfg, max_retries=max_retries)
        return FakeClient()

    monkeypatch.setattr(gepa_optimization, "OpenRouterLLMClient", fake_client)
    cfg = LLMClientConfig(model="provider/model")
    built = build_openrouter_reflection_lm(cfg=cfg, cache=cache, max_retries=5)
    assert isinstance(built, OpenRouterReflectionLM)
    assert captured == {"cfg": cfg, "max_retries": 5}


def test_reflection_factory_missing_key_fails_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(OPENROUTER_API_KEY_ENV, raising=False)
    with pytest.raises(AxBenchBaseError, match="not set in environment"):
        build_openrouter_reflection_lm(
            cfg=LLMClientConfig(model="provider/model"), cache=MemoryCache()
        )


def test_optimizer_call_shape_side_info_final_emissions_and_accounting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}
    side_infos: list[dict[str, Any]] = []
    generation = "g" * (SIDE_INFO_EXCERPT_CHARS + 5)
    rationale = "r" * (SIDE_INFO_EXCERPT_CHARS + 7)

    def runtime(_suffix: str, _input_id: int) -> RawEvaluation:
        return _raw(generation=generation, rationale=rationale, paid=1)

    def fake_optimize(
        *, seed_candidate: dict[str, str], evaluator: Callable[..., Any], dataset: list[Any],
        objective: str, config: Any
    ) -> GEPAResult[Any, int]:
        captured.update(
            seed_candidate=seed_candidate,
            dataset=dataset,
            objective=objective,
            config=config,
        )
        for example in dataset:
            score, side_info = evaluator({"suffix": "  selected  "}, example)
            assert score == 0.5
            side_infos.append(side_info)
        return _result()

    monkeypatch.setattr(gepa_optimization, "optimize_anything", fake_optimize)
    reflection = StaticReflection(paid=3)
    observed = optimize_suffix(
        examples=_examples(),
        runtime=runtime,
        reflection_lm=reflection,
        objective="Improve the suffix.",
        run_dir=tmp_path / "fresh",
        proposal_count=1,
        gepa_seed=17,
        seed_suffix="  seed  ",
    )

    assert captured["seed_candidate"] == {"suffix": "seed"}
    assert captured["dataset"] == _examples()
    assert captured["objective"] == "Improve the suffix."
    config = captured["config"]
    assert config.engine.run_dir == str(tmp_path / "fresh")
    assert config.engine.seed == 17
    assert config.engine.max_metric_calls is None
    assert config.engine.max_candidate_proposals == 1
    assert config.engine.val_evaluation_policy == "full_eval"
    assert config.engine.candidate_selection_strategy == "current_best"
    assert config.engine.frontier_type == "instance"
    assert config.engine.parallel is False
    assert config.engine.cache_evaluation is False
    assert config.engine.display_progress_bar is False
    assert config.engine.raise_on_exception is True
    assert callable(config.reflection.reflection_lm)
    assert not isinstance(config.reflection.reflection_lm, str)
    assert config.reflection.reflection_minibatch_size == 3
    assert callable(config.stop_callbacks)

    assert observed.suffix == "selected"
    assert observed.best_fitness == 0.5
    assert [value.example for value in observed.evaluations] == _examples()
    assert all(value.raw.generation == generation for value in observed.evaluations)
    assert all(value.fitness.adequacy is Adequacy.ADEQUATE for value in observed.evaluations)
    assert observed.accounting == gepa_optimization.CallAccounting(4, 4, 4, 4, 0)
    assert len(side_infos) == 4
    for side_info in side_infos:
        assert set(side_info) == {"Fitness", "Details"}
        assert "scores" not in side_info
        assert side_info["Details"]["suffix"] == "selected"
        assert side_info["Details"]["model_output"] == generation[:SIDE_INFO_EXCERPT_CHARS]
        assert side_info["Details"]["judge_rationale"] == rationale[:SIDE_INFO_EXCERPT_CHARS]
        assert len(side_info["Details"]["Feedback"]) < 2 * SIDE_INFO_EXCERPT_CHARS + 200


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("empty_examples", "must not be empty"),
        ("bad_example", "OptimizationExample"),
        ("duplicate", "duplicate GEPA input_id"),
        ("message_surrogate", "user_message must be valid UTF-8"),
        ("runtime", "runtime must be callable"),
        ("reflection", "reflection_lm must be callable"),
        ("objective_type", "objective must be a non-blank string"),
        ("objective_blank", "objective must be a non-blank string"),
        ("proposal_bool", "proposal_count must be a non-negative integer"),
        ("proposal_grid", "proposal_count must be one of"),
        ("seed_negative", "gepa_seed must be a non-negative integer"),
        ("seed_large", "gepa_seed must not exceed"),
        ("minibatch_bool", "reflection_minibatch_size must be a non-negative integer"),
        ("minibatch_zero", "reflection_minibatch_size must be in"),
        ("minibatch_large", "reflection_minibatch_size must be in"),
        ("run_dir_type", "run_dir must be a pathlib.Path"),
        ("run_dir_exists", "already exists"),
        ("seed_empty", "seed suffix must not be empty"),
        ("seed_unsafe", "control tokens"),
    ],
)
def test_optimizer_rejects_invalid_ingress_before_gepa(
    mutation: str, match: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    kwargs = _base_kwargs(tmp_path)
    if mutation == "empty_examples":
        kwargs["examples"] = []
    elif mutation == "bad_example":
        kwargs["examples"] = [object()]
    elif mutation == "duplicate":
        kwargs["examples"] = [OptimizationExample(0, "a"), OptimizationExample(0, "b")]
    elif mutation == "message_surrogate":
        kwargs["examples"] = [OptimizationExample(0, "\ud800")]
    elif mutation == "runtime":
        kwargs["runtime"] = 3
    elif mutation == "reflection":
        kwargs["reflection_lm"] = "model-name"
    elif mutation == "objective_type":
        kwargs["objective"] = None
    elif mutation == "objective_blank":
        kwargs["objective"] = " "
    elif mutation == "proposal_bool":
        kwargs["proposal_count"] = True
    elif mutation == "proposal_grid":
        kwargs["proposal_count"] = 3
    elif mutation == "seed_negative":
        kwargs["gepa_seed"] = -1
    elif mutation == "seed_large":
        kwargs["gepa_seed"] = 1 << 63
    elif mutation == "minibatch_bool":
        kwargs["reflection_minibatch_size"] = True
    elif mutation == "minibatch_zero":
        kwargs["reflection_minibatch_size"] = 0
    elif mutation == "minibatch_large":
        kwargs["reflection_minibatch_size"] = 5
    elif mutation == "run_dir_type":
        kwargs["run_dir"] = str(tmp_path / "gepa")
    elif mutation == "run_dir_exists":
        kwargs["run_dir"].mkdir()
    elif mutation == "seed_empty":
        kwargs["seed_suffix"] = " \n "
    else:
        kwargs["seed_suffix"] = "bad <|im_end|> suffix"

    monkeypatch.setattr(
        gepa_optimization,
        "optimize_anything",
        lambda **_kwargs: pytest.fail("GEPA must not run after invalid ingress"),
    )
    with pytest.raises(AxBenchAdvError, match=match):
        optimize_suffix(**kwargs)


@pytest.mark.parametrize(
    ("runtime", "match"),
    [
        (lambda _suffix, _input_id: object(), "must return RawEvaluation"),
        (lambda _suffix, _input_id: _raw(rationale=None), "judge_rationale must be a string"),
        (lambda _suffix, _input_id: _raw(paid=True), "paid_judge_cache_misses"),
        (lambda _suffix, _input_id: _raw(paid=3), "must not exceed 2"),
        (lambda _suffix, _input_id: _raw(generation=None), "output"),
    ],
)
def test_runtime_contract_failures_propagate(
    runtime: Callable[[str, int], Any], match: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_optimize(**kwargs: Any) -> Any:
        kwargs["evaluator"]({"suffix": "seed"}, kwargs["dataset"][0])
        raise AssertionError("unreachable")

    monkeypatch.setattr(gepa_optimization, "optimize_anything", fake_optimize)
    with pytest.raises(AxBenchAdvError, match=match):
        optimize_suffix(
            **{**_base_kwargs(tmp_path), "runtime": runtime}  # ty: ignore[invalid-argument-type]
        )


def test_candidate_shape_and_reflection_counter_contracts_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def bad_candidate(**kwargs: Any) -> Any:
        kwargs["evaluator"]({"other": "value"}, kwargs["dataset"][0])
        raise AssertionError("unreachable")

    monkeypatch.setattr(gepa_optimization, "optimize_anything", bad_candidate)
    with pytest.raises(AxBenchAdvError, match="exactly the 'suffix'"):
        optimize_suffix(**_base_kwargs(tmp_path))

    kwargs = _base_kwargs(tmp_path)
    kwargs["reflection_lm"] = lambda _prompt: "response"
    with pytest.raises(AxBenchAdvError, match="paid_reflection_cache_misses"):
        optimize_suffix(**kwargs)


@pytest.mark.parametrize(
    ("result", "match"),
    [
        (_result(val_aggregate_subscores=[{}]), "objective-level Pareto"),
        (_result(per_objective_best_candidates={"x": {0}}), "objective-level Pareto"),
        (_result(objective_pareto_front={"x": 1.0}), "objective-level Pareto"),
        (_result(candidate="suffix"), "non-mapping best candidate"),
        (_result(candidate={"suffix": 1}), "non-string mapping best candidate"),
        (_result(score=0.6), "independent best-fitness recomputation mismatch"),
        (_result(candidate={"other": "x"}), "exactly the 'suffix'"),
        (_result(score=float("nan")), "selected aggregate must be finite"),
        (_result(score=1.1), "selected aggregate must be finite"),
        (_result(nominal=None), "GEPA nominal metric calls"),
        (_result(nominal=3), "accounting diverged"),
    ],
)
def test_gepa_result_contract_failures_are_detected(
    result: GEPAResult[Any, int], match: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_optimize(**kwargs: Any) -> Any:
        for example in kwargs["dataset"]:
            kwargs["evaluator"]({"suffix": "selected"}, example)
        return result

    monkeypatch.setattr(gepa_optimization, "optimize_anything", fake_optimize)
    with pytest.raises(AxBenchAdvError, match=match):
        optimize_suffix(**_base_kwargs(tmp_path))


def test_repeated_gepa_evaluation_reuses_root_scientific_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = 0

    def runtime(_suffix: str, _input_id: int) -> RawEvaluation:
        nonlocal calls
        calls += 1
        return _raw(concept=1.0)

    def fake_optimize(**kwargs: Any) -> Any:
        for example in kwargs["dataset"]:
            kwargs["evaluator"]({"suffix": "selected"}, example)
        kwargs["evaluator"]({"suffix": "selected"}, kwargs["dataset"][0])
        return _result(score=0.5, nominal=5)

    monkeypatch.setattr(gepa_optimization, "optimize_anything", fake_optimize)
    result = optimize_suffix(
        **{**_base_kwargs(tmp_path), "runtime": runtime}  # ty: ignore[invalid-argument-type]
    )

    assert calls == 4
    assert result.accounting == gepa_optimization.CallAccounting(5, 4, 4, 0, 0)


def test_selected_candidate_must_have_retained_full_validation_emissions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_optimize(**kwargs: Any) -> Any:
        for example in kwargs["dataset"][:-1]:
            kwargs["evaluator"]({"suffix": "selected"}, example)
        return _result(score=0.5, nominal=3)

    monkeypatch.setattr(gepa_optimization, "optimize_anything", fake_optimize)
    with pytest.raises(AxBenchAdvError, match="did not fully evaluate"):
        optimize_suffix(**_base_kwargs(tmp_path))


def test_decreasing_reflection_counter_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class DecreasingReflection(StaticReflection):
        @property
        def paid_cache_misses(self) -> int:
            self._paid -= 1
            return self._paid

    def fake_optimize(**kwargs: Any) -> Any:
        for example in kwargs["dataset"]:
            kwargs["evaluator"]({"suffix": "selected"}, example)
        return _result()

    monkeypatch.setattr(gepa_optimization, "optimize_anything", fake_optimize)
    kwargs = _base_kwargs(tmp_path)
    kwargs["reflection_lm"] = DecreasingReflection(paid=3)
    with pytest.raises(AxBenchAdvError, match="counter decreased"):
        optimize_suffix(**kwargs)


def test_unexpected_parallel_duplicate_evaluations_fail_accounting_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    duplicate = Barrier(2)

    def runtime(_suffix: str, input_id: int) -> RawEvaluation:
        if input_id == 0:
            duplicate.wait(timeout=5)
        return _raw()

    def parallel_dependency(**kwargs: Any) -> Any:
        # Production requests parallel=False. If GEPA violates that request,
        # both calls reach the runtime before either can populate the cache.
        assert kwargs["config"].engine.parallel is False
        evaluate, dataset = kwargs["evaluator"], kwargs["dataset"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            calls = [pool.submit(evaluate, {"suffix": "selected"}, dataset[0]) for _ in range(2)]
            for call in calls:
                call.result(timeout=10)
        for example in dataset[1:]:
            evaluate({"suffix": "selected"}, example)
        return _result(nominal=5)

    monkeypatch.setattr(gepa_optimization, "optimize_anything", parallel_dependency)
    kwargs = _base_kwargs(tmp_path)
    kwargs["runtime"] = runtime
    with pytest.raises(AxBenchAdvError, match="cache did not eliminate a repeated evaluator call"):
        optimize_suffix(**kwargs)
