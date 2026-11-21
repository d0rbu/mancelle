"""Vendored-Python tests for the shared generation worker and judge/cache parity."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import io
import math
import os
import random
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import openai
import pandas as pd
import pytest
import torch as t
from axbench.evaluators.judge_cache import JudgeCache
from axbench.evaluators.lm_judge import LMJudgeEvaluator
from axbench.models import language_models as vendor_language_models
from axbench.models.language_models import LanguageModel
from axbench.models.lora import LoRA
from axbench.models.prompt import PromptSteering
from axbench.scripts import inference
from axbench.utils import model_utils
from transformers import set_seed

from scripts.eval.axbench_adv.vendored import worker
from scripts.eval.axbench_adv.vendored.protocol import (
    PROTOCOL_VERSION,
    decode_frame,
    encode_frame,
)


def _draws() -> tuple[float, float, list[float]]:
    return random.random(), float(np.random.random()), t.rand(4).tolist()


def test_row_seed_matches_transformers_for_supported_seed_and_accepts_high_seed() -> None:
    set_seed(42)
    expected = _draws()
    worker.set_generation_seed(42)
    assert _draws() == expected

    high = (1 << 62) + 123
    worker.set_generation_seed(high)
    first = _draws()
    worker.set_generation_seed(high)
    assert _draws() == first
    with pytest.raises(ValueError, match="Seed must be between"):
        set_seed(1 << 32)
    worker.set_generation_seed(1 << 32)


@pytest.mark.parametrize("seed", [True, -1, 1 << 63])
def test_row_seed_rejects_values_outside_signed_63_bit(seed: Any) -> None:
    with pytest.raises(ValueError, match="row_seed"):
        worker.set_generation_seed(seed)


class FakeTokenizer:
    unk_token = "<unk>"
    pad_token = "<pad>"
    model_max_length = 8
    padding_side = "right"

    def __init__(self, events: list[tuple[str, Any]] | None = None) -> None:
        self.events = [] if events is None else events

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        self.events.append(("tokenize", (text, kwargs)))
        return {"input_ids": list(range(len(text)))}

    def add_special_tokens(self, value: dict[str, str]) -> None:
        self.events.append(("add_special_tokens", value))

    def __len__(self) -> int:
        return 17


@pytest.mark.parametrize(
    ("method", "expected_prompt", "positions"),
    [
        ("DiffMean", "plain", "all_prompt"),
        ("LinearProbe", "plain", "all_prompt"),
        ("LoRA", "plain", None),
        ("PromptSteering", "steer", None),
    ],
)
def test_evaluate_preflights_then_seeds_then_calls_shared_generation(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    expected_prompt: str,
    positions: str | None,
) -> None:
    events: list[tuple[str, Any]] = []
    runtime = object.__new__(worker.WorkerRuntime)
    runtime.method = method
    runtime.concept_id = 2
    runtime.factor = 1.5
    runtime.temperature = 1.0
    runtime.output_length = 32
    runtime.worker_max_prompt_tokens = 8
    runtime.prefix_length = 3
    runtime.tokenizer = FakeTokenizer(events)
    runtime.benchmark_model = object()
    runtime._judge = lambda generation: (0.25, 1.75, "rationale", 1)
    monkeypatch.setattr(worker, "set_generation_seed", lambda seed: events.append(("seed", seed)))

    def fake_generate(model: object, frame: pd.DataFrame, **kwargs: Any) -> dict[str, list[str]]:
        assert t.is_inference_mode_enabled()
        events.append(("generate", (model, frame.copy(deep=True), kwargs)))
        return {"steered_generation": ["exact generation"]}

    monkeypatch.setattr(worker, "run_steering_generation", fake_generate)
    result = runtime.evaluate(
        {"input": "plain", "input_id": 4, "row_seed": 99, "steered_input": "steer"}
    )
    assert not t.is_inference_mode_enabled()
    assert [event[0] for event in events] == ["tokenize", "seed", "generate"]
    assert events[0][1][0] == expected_prompt
    _, frame, kwargs = events[2][1]
    assert frame.to_dict("records") == [
        {
            "input": "plain",
            "steered_input": "steer",
            "factor": 1.5,
            "concept_id": 2,
            "sae_id": 2,
        }
    ]
    assert kwargs == {
        "steering_batch_size": 1,
        "steering_output_length": 32,
        "temperature": 1.0,
        "prefix_length": 3,
        "positions": positions,
        "disable_neuronpedia_max_act": False,
        "intervene_on_prompt": True,
        "compute_perplexity": False,
    }
    assert result == {
        "concept_rating": 0.25,
        "fluency_rating": 1.75,
        "generation": "exact generation",
        "input_id": 4,
        "judge_rationale": "rationale",
        "paid_judge_cache_misses": 1,
    }


def test_overlength_prompt_raises_before_seed_or_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(worker.WorkerRuntime)
    runtime.method = "LoRA"
    runtime.tokenizer = FakeTokenizer()
    runtime.worker_max_prompt_tokens = 8
    monkeypatch.setattr(
        worker,
        "set_generation_seed",
        lambda _seed: pytest.fail("seed must follow successful prompt preflight"),
    )
    monkeypatch.setattr(
        worker,
        "run_steering_generation",
        lambda *_args, **_kwargs: pytest.fail("generation must not start"),
    )
    with pytest.raises(ValueError, match="configured prompt-token limit 8"):
        runtime.evaluate(
            {
                "input": "ninechars",
                "input_id": 0,
                "row_seed": 1,
                "steered_input": "short",
            }
        )


def test_evaluate_rejects_row_seed_above_signed_63_bit() -> None:
    runtime = object.__new__(worker.WorkerRuntime)
    runtime.method = "LoRA"
    with pytest.raises(ValueError, match="must not exceed"):
        runtime.evaluate(
            {
                "input": "plain",
                "input_id": 0,
                "row_seed": 1 << 63,
                "steered_input": "steered",
            }
        )


def test_evaluate_rejects_bad_tokenizer_or_generation_contract() -> None:
    runtime = object.__new__(worker.WorkerRuntime)
    runtime.method = "LoRA"
    runtime.concept_id = 0
    runtime.factor = 1.0
    runtime.temperature = 1.0
    runtime.output_length = 1
    runtime.worker_max_prompt_tokens = 8
    runtime.prefix_length = 1
    runtime.benchmark_model = object()
    runtime._judge = lambda _generation: (0.0, 0.0, "", 0)

    class BadTokenizer(FakeTokenizer):
        def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
            del text, kwargs
            return {"input_ids": [True]}

    runtime.tokenizer = BadTokenizer()
    with pytest.raises(ValueError, match="invalid input_ids"):
        runtime.evaluate({"input": "p", "input_id": 0, "row_seed": 1, "steered_input": "s"})

    runtime.tokenizer = FakeTokenizer()
    original_generate = worker.run_steering_generation
    try:
        for invalid in (None, {}, {"steered_generation": []}, {"steered_generation": [1]}):
            worker.run_steering_generation = lambda *_args, _invalid=invalid, **_kwargs: _invalid
            with pytest.raises(ValueError, match="exactly one text"):
                runtime.evaluate({"input": "p", "input_id": 0, "row_seed": 1, "steered_input": "s"})
    finally:
        worker.run_steering_generation = original_generate


@pytest.mark.slow
def test_cuda_generation_is_repeatable_and_sensitive_to_seed_and_factor(tmp_path: Path) -> None:
    source_value = os.environ.get("AXBENCH_ADV_CUDA_SOURCE_RUN")
    prepared_value = os.environ.get("AXBENCH_ADV_CUDA_PREPARED_DIR")
    if source_value is None or prepared_value is None:
        pytest.skip("set AXBENCH_ADV_CUDA_SOURCE_RUN and AXBENCH_ADV_CUDA_PREPARED_DIR")
    if not t.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    source_run = Path(source_value).resolve()
    prepared_dir = Path(prepared_value).resolve()
    frame = pd.read_parquet(source_run / "main/inference/steering_data.parquet")
    selected = frame.loc[
        frame["concept_id"].eq(0) & frame["input_id"].eq(0) & frame["factor"].eq(1.0)
    ]
    assert len(selected) == 1
    row = selected.iloc[0]
    judge_cache = tmp_path / "judge.sqlite3"
    JudgeCache(judge_cache).close()
    runtime = worker.WorkerRuntime(
        {
            "api_base_url": "https://openrouter.ai/api/v1",
            "aux_dump_dir": str(source_run / "aux"),
            "cache_only": True,
            "concept": str(row["input_concept"]),
            "concept_id": 0,
            "device": "cuda:0",
            "factor": 1.0,
            "intervene_on_prompt": True,
            "intervention_type": "addition",
            "judge_cache": str(judge_cache),
            "judge_model": "openai/gpt-5.6-luna",
            "layer": 28,
            "main_dump_dir": str(source_run / "main"),
            "max_concepts": 2,
            "method": "DiffMean",
            "model_name": "Qwen/Qwen2.5-3B-Instruct",
            "model_revision": None,
            "output_length": 128,
            "prepared_dir": str(prepared_dir),
            "steering_layers": [28],
            "temperature": 1.0,
            "use_bf16": True,
            "worker_max_prompt_tokens": 1024,
        }
    )

    def fake_judge(_generation: str) -> tuple[float, float, str, int]:
        return 0.0, 2.0, "{}", 0

    runtime._judge = fake_judge
    request = {
        "input": str(row["input"]),
        "input_id": 0,
        "steered_input": str(row["steered_input"]),
    }
    try:
        first = runtime.evaluate(request | {"row_seed": 42})["generation"]
        repeated = runtime.evaluate(request | {"row_seed": 42})["generation"]
        wrong_seed = runtime.evaluate(request | {"row_seed": 43})["generation"]
        runtime.factor = 1.5
        wrong_factor = runtime.evaluate(request | {"row_seed": 42})["generation"]
    finally:
        runtime.close()

    assert first == repeated
    assert first != wrong_seed
    assert first != wrong_factor


def test_worker_validation_helpers_reject_python_type_traps() -> None:
    for value in (object(), {"wrong": 1}):
        with pytest.raises(ValueError, match="fields must be exactly"):
            worker._exact_keys(value, {"right"}, label="value")
    for value in (None, " ", "\ud800"):
        with pytest.raises(ValueError, match="must be"):
            worker._text(value, name="text")
    for value in (True, 1.5, -1):
        with pytest.raises(ValueError, match="integer"):
            worker._integer(value, name="integer")
    for value in (True, "1", math.inf, -0.5):
        with pytest.raises(ValueError, match="numeric|finite"):
            worker._finite_float(value, name="float", minimum=0.0)


class FakeStats:
    total_call = 0
    total_cache_hit = 0


class FakeInnerEvaluator:
    def __init__(self, stats: FakeStats, *, bad_accounting: bool = False) -> None:
        self.stats = stats
        self.bad_accounting = bad_accounting
        self.calls: list[tuple[list[str], str]] = []

    def _get_ratings_from_prompts(
        self, prompts: list[str], api_name: str
    ) -> tuple[list[float], list[str]]:
        self.calls.append((prompts, api_name))
        if not self.bad_accounting:
            self.stats.total_call += 1
        if api_name.endswith("_concept"):
            return [0.5], ["concept reason Rating: [[0.5]]"]
        return [2.0], ["fluency reason Rating: [[2]]"]


def test_inner_judge_uses_exact_two_templates_and_canonical_rationale() -> None:
    runtime = object.__new__(worker.WorkerRuntime)
    runtime.method = "LoRA"
    runtime.concept = "target concept"
    runtime.language_model = SimpleNamespace(stats=FakeStats())
    runtime.evaluator = FakeInnerEvaluator(runtime.language_model.stats)
    result = runtime._judge("generated answer")
    assert result == (
        0.5,
        2.0,
        '{"concept":"concept reason Rating: [[0.5]]","fluency":"fluency reason Rating: [[2]]"}',
        2,
    )
    assert [api_name for _, api_name in runtime.evaluator.calls] == [
        "LoRA_concept",
        "LoRA_fluency",
    ]
    assert "[Concept Start]\ntarget concept\n[Concept End]" in runtime.evaluator.calls[0][0][0]
    assert (
        "[Sentence Fragment Start]\ngenerated answer\n[Sentence Fragment End]"
        in (runtime.evaluator.calls[1][0][0])
    )
    assert all("instruction" not in api_name for _, api_name in runtime.evaluator.calls)


class FlakyInnerEvaluator(FakeInnerEvaluator):
    """Fails the fluency call with a connection error ``failures`` times, caching the concept call."""

    def __init__(self, stats: FakeStats, *, failures: int, error: BaseException) -> None:
        super().__init__(stats)
        self.failures = failures
        self.error = error
        self.concept_cached = False

    def _get_ratings_from_prompts(
        self, prompts: list[str], api_name: str
    ) -> tuple[list[float], list[str]]:
        if api_name.endswith("_concept") and self.concept_cached:
            self.calls.append((prompts, api_name))
            self.stats.total_call += 1
            self.stats.total_cache_hit += 1
            return [0.5], ["concept reason Rating: [[0.5]]"]
        result = super()._get_ratings_from_prompts(prompts, api_name)
        if api_name.endswith("_concept"):
            self.concept_cached = True
        elif self.failures > 0:
            self.failures -= 1
            raise self.error
        return result


def _inner_judge_with(evaluator_factory: Any) -> worker._InnerJudge:
    judge = object.__new__(worker._InnerJudge)
    judge.method = "LoRA"
    judge.concept = "concept"
    judge.language_model = SimpleNamespace(stats=FakeStats())
    judge.evaluator = evaluator_factory(judge.language_model.stats)
    return judge


def _connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx.Request("POST", "https://openrouter.ai/api/v1"))


class PairCompletions:
    def __init__(
        self,
        *,
        fail_fluency_once: bool = False,
        slow_concept: bool = False,
        bad_fluency: bool = False,
    ) -> None:
        self.active = 0
        self.peak = 0
        self.calls: list[str] = []
        self.fail_fluency_once = fail_fluency_once
        self.slow_concept = slow_concept
        self.bad_fluency = bad_fluency

    async def create(self, *, messages: list[dict[str, str]], **_kwargs: Any) -> Any:
        prompt = messages[0]["content"]
        is_concept = "[Concept Start]" in prompt
        self.calls.append("concept" if is_concept else "fluency")
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.03 if is_concept and self.slow_concept else 0.01)
            if not is_concept and self.fail_fluency_once:
                self.fail_fluency_once = False
                raise _connection_error()
            content = (
                "concept reason Rating: [[0.5]]"
                if is_concept
                else "invalid completion" if self.bad_fluency else "fluency reason Rating: [[2]]"
            )
            return SimpleNamespace(
                to_dict=lambda: {
                    "choices": [{"message": {"content": content}}],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 4},
                }
            )
        finally:
            self.active -= 1


class PairClient:
    def __init__(self, completions: PairCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_project_judge_pair_preserves_results_cache_and_bounds_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    results = []
    for pair_concurrency in (1, 2):
        completions = PairCompletions(slow_concept=pair_concurrency == 2)
        client = PairClient(completions)
        monkeypatch.setattr(worker, "create_openrouter_client", lambda _purpose, c=client: c)
        judge = worker._InnerJudge(
            method="SelfConditioning",
            concept_id=0,
            concept="cats",
            judge_model="openai/gpt-5.6-luna",
            judge_cache=str(tmp_path / f"pair-{pair_concurrency}.sqlite"),
            cache_only=False,
            api_base_url="https://openrouter.ai/api/v1",
            pair_concurrency=pair_concurrency,
        )
        try:
            first = judge.judge("exact generation")
            replay = judge.judge("exact generation")
            results.append((first, replay))
            assert judge.language_model.stats.total_call == 4
            assert judge.language_model.stats.total_cache_hit == 2
            assert completions.calls == ["concept", "fluency"]
            assert completions.peak == pair_concurrency
        finally:
            judge.close()
        assert client.closed
    assert results[0] == results[1]
    assert results[0][0][:2] == (0.5, 2.0)
    assert results[0][0][3] == 2
    assert results[0][1][3] == 0
    cache_only = worker._InnerJudge(
        method="SelfConditioning",
        concept_id=0,
        concept="cats",
        judge_model="openai/gpt-5.6-luna",
        judge_cache=str(tmp_path / "pair-1.sqlite"),
        cache_only=True,
        api_base_url="https://openrouter.ai/api/v1",
        pair_concurrency=2,
    )
    try:
        assert cache_only.judge("exact generation") == results[0][1]
    finally:
        cache_only.close()


def test_project_pair_waits_for_both_on_failure_and_retries_cached_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(vendor_language_models, "MAX_TOTAL_ATTEMPTS", 1)
    monkeypatch.setattr(worker.time, "sleep", lambda _seconds: None)
    completions = PairCompletions(fail_fluency_once=True)
    client = PairClient(completions)
    monkeypatch.setattr(worker, "create_openrouter_client", lambda _purpose: client)
    judge = worker._InnerJudge(
        method="SelfConditioning",
        concept_id=0,
        concept="cats",
        judge_model="openai/gpt-5.6-luna",
        judge_cache=str(tmp_path / "pair-retry.sqlite"),
        cache_only=False,
        api_base_url="https://openrouter.ai/api/v1",
        pair_concurrency=2,
    )
    try:
        concept, fluency, _rationale, paid_misses = judge.judge("exact generation")
        assert (concept, fluency, paid_misses) == (0.5, 2.0, 1)
        assert completions.calls == ["concept", "fluency", "fluency"]
        assert judge.language_model.stats.total_call == 3
        assert judge.language_model.stats.total_cache_hit == 1
    finally:
        judge.close()
    assert client.closed


def test_project_pair_rejects_bad_rating_after_other_request_finishes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(vendor_language_models, "MAX_TOTAL_ATTEMPTS", 1)
    completions = PairCompletions(bad_fluency=True)
    client = PairClient(completions)
    monkeypatch.setattr(worker, "create_openrouter_client", lambda _purpose: client)
    judge = worker._InnerJudge(
        method="SelfConditioning",
        concept_id=0,
        concept="cats",
        judge_model="openai/gpt-5.6-luna",
        judge_cache=str(tmp_path / "pair-invalid.sqlite"),
        cache_only=False,
        api_base_url="https://openrouter.ai/api/v1",
        pair_concurrency=2,
    )
    try:
        with pytest.raises(ValueError, match="exactly one Rating"):
            judge.judge("exact generation")
        assert completions.calls == ["concept", "fluency"]
        assert judge.language_model.stats.total_call == 1
    finally:
        judge.close()
    assert client.closed


def test_inner_judge_retries_connection_failures_with_backoff_and_exact_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(worker.time, "sleep", sleeps.append)
    judge = _inner_judge_with(
        lambda stats: FlakyInnerEvaluator(stats, failures=2, error=_connection_error())
    )

    concept, fluency, rationale, paid_misses = judge.judge("answer")

    assert (concept, fluency) == (0.5, 2.0)
    assert '"fluency":"fluency reason Rating: [[2]]"' in rationale
    # The retry's concept call is a cache hit, so only the fluency call is a paid miss.
    assert paid_misses == 1
    assert sleeps == [
        worker.JUDGE_CONNECTION_BASE_WAIT_SECONDS,
        2 * worker.JUDGE_CONNECTION_BASE_WAIT_SECONDS,
    ]
    assert [api_name for _, api_name in judge.evaluator.calls] == [
        "LoRA_concept",
        "LoRA_fluency",
        "LoRA_concept",
        "LoRA_fluency",
        "LoRA_concept",
        "LoRA_fluency",
    ]


def test_inner_judge_backoff_is_capped_and_gives_up_after_the_attempt_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(worker.time, "sleep", sleeps.append)
    judge = _inner_judge_with(
        lambda stats: FlakyInnerEvaluator(
            stats, failures=worker.JUDGE_CONNECTION_ATTEMPTS + 5, error=_connection_error()
        )
    )

    with pytest.raises(openai.APIConnectionError):
        judge.judge("answer")

    assert len(sleeps) == worker.JUDGE_CONNECTION_ATTEMPTS - 1
    assert max(sleeps) == worker.JUDGE_CONNECTION_MAX_WAIT_SECONDS
    assert sleeps == sorted(sleeps)
    assert len(judge.evaluator.calls) == 2 * worker.JUDGE_CONNECTION_ATTEMPTS


def test_inner_judge_does_not_retry_non_connection_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(worker.time, "sleep", sleeps.append)
    judge = _inner_judge_with(
        lambda stats: FlakyInnerEvaluator(stats, failures=1, error=ValueError("bad rating"))
    )

    with pytest.raises(ValueError, match="bad rating"):
        judge.judge("answer")

    assert sleeps == []
    assert len(judge.evaluator.calls) == 2


def test_inner_judge_rejects_counter_drift() -> None:
    runtime = object.__new__(worker.WorkerRuntime)
    runtime.method = "LoRA"
    runtime.concept = "concept"
    runtime.language_model = SimpleNamespace(stats=FakeStats())
    runtime.evaluator = FakeInnerEvaluator(runtime.language_model.stats, bad_accounting=True)
    with pytest.raises(RuntimeError, match="must be two calls"):
        runtime._judge("answer")


def test_worker_judge_delegates_to_initialized_inner_judge() -> None:
    runtime = object.__new__(worker.WorkerRuntime)
    runtime.inner_judge = SimpleNamespace(
        judge=lambda generation: (0.5, 1.5, f"rationale:{generation}", 1)
    )

    assert runtime._judge("exact") == (0.5, 1.5, "rationale:exact", 1)


def _judge_init_payload(tmp_path: Path, **updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "api_base_url": "https://openrouter.ai/api/v1",
        "cache_only": True,
        "concept": "target concept",
        "concept_id": 2,
        "factor": 6.0,
        "judge_cache": str(tmp_path / "judge.sqlite"),
        "judge_model": "openai/gpt-5.6-luna",
        "method": "SelfConditioning",
        "runtime_kind": "judge",
    }
    payload.update(updates)
    return payload


class FakeJudgeOnlyInner:
    instances: list[FakeJudgeOnlyInner] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[str] = []
        self.closed = False
        self.instances.append(self)

    def judge(self, generation: str) -> tuple[float, float, str, int]:
        self.calls.append(generation)
        return 0.5, 2.0, '{"concept":"c","fluency":"f"}', 2

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("method", ["SelfConditioning", "SelfConditioningMasked", "NLSteering"])
def test_judge_only_runtime_has_exact_identity_payload_and_two_field_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, method: str
) -> None:
    FakeJudgeOnlyInner.instances.clear()
    monkeypatch.setattr(worker, "_InnerJudge", FakeJudgeOnlyInner)
    payload = _judge_init_payload(tmp_path, method=method)
    runtime = worker.JudgeRuntime(payload)
    inner = FakeJudgeOnlyInner.instances[-1]
    assert inner.kwargs == {
        "api_base_url": payload["api_base_url"],
        "cache_only": payload["cache_only"],
        "concept": payload["concept"],
        "concept_id": payload["concept_id"],
        "judge_cache": payload["judge_cache"],
        "judge_model": payload["judge_model"],
        "method": method,
    }
    result = runtime.evaluate({"generation": "exact output café", "input_id": 7})
    assert result == {
        "concept_rating": 0.5,
        "fluency_rating": 2.0,
        "generation": "exact output café",
        "input_id": 7,
        "judge_rationale": '{"concept":"c","fluency":"f"}',
        "paid_judge_cache_misses": 2,
    }
    assert inner.calls == ["exact output café"]
    runtime.close()
    assert inner.closed


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"extra": 1}, "fields"),
        ({"runtime_kind": "worker"}, "runtime_kind"),
        ({"method": "LoRA"}, "only SelfConditioning"),
        ({"concept_id": True}, "concept_id"),
        ({"factor": True}, "factor must be numeric"),
    ],
)
def test_judge_only_runtime_rejects_every_identity_extension(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    updates: dict[str, Any],
    match: str,
) -> None:
    monkeypatch.setattr(worker, "_InnerJudge", FakeJudgeOnlyInner)
    with pytest.raises(ValueError, match=match):
        worker.JudgeRuntime(_judge_init_payload(tmp_path, **updates))


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"generation": "ok", "input_id": 0, "instruction": "forbidden"}, "fields"),
        ({"generation": 1, "input_id": 0}, "generation must be a string"),
        ({"generation": "\ud800", "input_id": 0}, "valid UTF-8"),
        ({"generation": "ok", "input_id": True}, "input_id"),
    ],
)
def test_judge_only_runtime_rejects_instruction_and_malformed_evaluations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: dict[str, Any],
    match: str,
) -> None:
    monkeypatch.setattr(worker, "_InnerJudge", FakeJudgeOnlyInner)
    runtime = worker.JudgeRuntime(_judge_init_payload(tmp_path))
    with pytest.raises(ValueError, match=match):
        runtime.evaluate(payload)


def test_runtime_factory_selects_judge_only_on_explicit_kind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        worker,
        "JudgeRuntime",
        lambda payload: seen.append(("judge", dict(payload))) or "judge-runtime",
    )
    monkeypatch.setattr(
        worker,
        "WorkerRuntime",
        lambda payload: seen.append(("worker", dict(payload))) or "worker-runtime",
    )
    payload = _judge_init_payload(tmp_path)
    assert worker._runtime_factory(payload) == "judge-runtime"
    assert worker._runtime_factory({"method": "LoRA"}) == "worker-runtime"
    assert [kind for kind, _payload in seen] == ["judge", "worker"]


def _init_payload(tmp_path: Path, method: str, **updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "api_base_url": "https://openrouter.ai/api/v1",
        "aux_dump_dir": str(tmp_path / "aux"),
        "cache_only": True,
        "concept": "concept zero",
        "concept_id": 0,
        "device": "cpu",
        "factor": 1.0,
        "intervene_on_prompt": True,
        "intervention_type": "addition",
        "judge_cache": str(tmp_path / "judge.sqlite"),
        "judge_model": "openai/gpt-5.6-luna",
        "layer": 28,
        "main_dump_dir": str(tmp_path / "main"),
        "max_concepts": 2,
        "method": method,
        "model_name": "fake-model",
        "model_revision": None,
        "output_length": 128,
        "prepared_dir": str(tmp_path / "prepared"),
        "steering_layers": [28],
        "temperature": 1.0,
        "use_bf16": True,
        "worker_max_prompt_tokens": 1024,
    }
    payload.update(updates)
    return payload


class FakeBaseModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(model_type="qwen2")
        self.resized_to: int | None = None

    def eval(self) -> FakeBaseModel:
        return self

    def resize_token_embeddings(self, size: int) -> None:
        self.resized_to = size


class FakeBenchmark:
    instances: list[FakeBenchmark] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init = (args, kwargs)
        self.loads: list[dict[str, Any]] = []
        self.precompute: list[Path] = []
        self.max_activations: dict[int, float] = {}
        self.instances.append(self)

    def load(self, **kwargs: Any) -> None:
        self.loads.append(kwargs)

    def to(self, device: str) -> FakeBenchmark:
        self.device = device
        return self

    def pre_compute_mean_activations(self, path: Path) -> None:
        self.precompute.append(path)
        self.max_activations = {0: 2.0}


class FakeLanguageModel:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init = (args, kwargs)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _patch_runtime_loads(
    monkeypatch: pytest.MonkeyPatch,
    *,
    method: str,
    metadata: list[dict[str, Any]] | None = None,
    tokenizer: FakeTokenizer | None = None,
    benchmark_class: type[FakeBenchmark] = FakeBenchmark,
) -> FakeBaseModel:
    rows = metadata or [
        {"concept_id": 0, "concept": "concept zero", "ref": "x/0"},
        {"concept_id": 1, "concept": "concept one", "ref": "x/1"},
    ]
    monkeypatch.setattr(worker, "load_metadata_flatten", lambda _path, _limit: rows)
    tokenizer = tokenizer or FakeTokenizer()
    monkeypatch.setattr(
        worker.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: tokenizer
    )
    base_model = FakeBaseModel()
    monkeypatch.setattr(
        worker.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: base_model,
    )
    monkeypatch.setattr(worker.axbench, method, benchmark_class)
    monkeypatch.setattr(worker, "LanguageModel", FakeLanguageModel)
    return base_model


@pytest.mark.parametrize("method", ["DiffMean", "LinearProbe", "LoRA", "PromptSteering"])
def test_runtime_init_loads_one_method_without_training_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, method: str
) -> None:
    FakeBenchmark.instances.clear()
    _patch_runtime_loads(monkeypatch, method=method)
    runtime = worker.WorkerRuntime(_init_payload(tmp_path, method))
    benchmark = FakeBenchmark.instances[-1]
    assert benchmark.init[1]["training_args"] is None
    assert benchmark.init[1]["steering_layers"] == [28]
    expected_dump = tmp_path / ("aux/train" if method == "LoRA" else "main/train")
    assert benchmark.loads == [
        {
            "dump_dir": expected_dump,
            "sae_path": "x/0",
            "mode": "steering",
            "priority_mode": "compute_priority",
            "intervention_type": "addition",
            "concept_id": 0,
            "low_rank_dimension": 1,
        }
    ]
    assert bool(benchmark.precompute) is (method in {"DiffMean", "LinearProbe"})
    assert runtime.prefix_length == 1
    runtime.close()
    assert runtime.language_model.closed is True


@pytest.mark.parametrize("revision", [None, "it-soup-APO"])
def test_runtime_loads_tokenizer_and_model_at_the_source_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, revision: str | None
) -> None:
    _patch_runtime_loads(monkeypatch, method="DiffMean")
    calls: dict[str, dict[str, Any]] = {}

    def load_tokenizer(*_args: Any, **kwargs: Any) -> FakeTokenizer:
        calls["tokenizer"] = kwargs
        return FakeTokenizer()

    def load_model(*_args: Any, **kwargs: Any) -> FakeBaseModel:
        calls["model"] = kwargs
        return FakeBaseModel()

    monkeypatch.setattr(worker.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(worker.AutoModelForCausalLM, "from_pretrained", load_model)
    payload = _init_payload(tmp_path, "DiffMean")
    payload["model_revision"] = revision
    payload["worker_max_prompt_tokens"] = 2048
    runtime = worker.WorkerRuntime(payload)
    try:
        assert calls["tokenizer"]["revision"] == revision
        assert calls["tokenizer"]["model_max_length"] == 2048
        assert runtime.worker_max_prompt_tokens == 2048
        assert calls["model"]["revision"] == revision
    finally:
        runtime.close()


def test_runtime_init_exercises_padding_ax_chat_and_live_client_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)

    class FakeAx:
        def eval(self) -> None:
            self.evaluated = True

        def to(self, dtype: t.dtype) -> None:
            self.dtype = dtype

    class BenchmarkWithAx(FakeBenchmark):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.ax = FakeAx()

    tokenizer = FakeTokenizer()
    tokenizer.unk_token = None
    tokenizer.pad_token = None
    base_model = _patch_runtime_loads(
        monkeypatch,
        method="LoRA",
        tokenizer=tokenizer,
        benchmark_class=BenchmarkWithAx,
    )
    monkeypatch.setattr(worker, "is_chat_model_name", lambda name: name == "fake-model")
    monkeypatch.setattr(worker, "get_prefix_length", lambda _tokenizer: 5)
    live_client = object()
    monkeypatch.setattr(worker, "create_openrouter_client", lambda _reason: live_client)
    runtime = worker.WorkerRuntime(_init_payload(tmp_path, "LoRA", cache_only=False))
    assert base_model.resized_to == 17
    assert runtime.prefix_length == 5
    assert runtime.benchmark_model.ax.evaluated is True
    assert runtime.benchmark_model.ax.dtype == t.bfloat16
    assert runtime.language_model.init[0][1] is live_client


def test_runtime_init_rejects_metadata_and_missing_max_activation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cases = [
        ([{"concept_id": 0, "concept": "concept zero", "ref": "x/0"}], {}, "contiguous"),
        (
            [
                {"concept_id": 0, "concept": "concept zero", "ref": "x/0"},
                {"concept_id": 1, "concept": "concept one", "ref": "x/1"},
            ],
            {"concept_id": 2},
            "absent",
        ),
        (
            [
                {"concept_id": 0, "concept": "different", "ref": "x/0"},
                {"concept_id": 1, "concept": "concept one", "ref": "x/1"},
            ],
            {},
            "does not match",
        ),
    ]
    for metadata, updates, match in cases:
        _patch_runtime_loads(monkeypatch, method="LoRA", metadata=metadata)
        with pytest.raises(ValueError, match=match):
            worker.WorkerRuntime(_init_payload(tmp_path, "LoRA", **updates))

    class MissingMax(FakeBenchmark):
        def pre_compute_mean_activations(self, path: Path) -> None:
            self.precompute.append(path)

    _patch_runtime_loads(monkeypatch, method="DiffMean", benchmark_class=MissingMax)
    with pytest.raises(ValueError, match="no max activation"):
        worker.WorkerRuntime(_init_payload(tmp_path, "DiffMean"))


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"method": "bad"}, "unsupported method"),
        ({"temperature": 0.0}, "must be positive"),
        ({"use_bf16": False}, "use_bf16"),
        ({"intervention_type": "clamping"}, "addition"),
        ({"intervene_on_prompt": False}, "intervene_on_prompt"),
        ({"steering_layers": [7]}, "steering_layers"),
        ({"cache_only": 1}, "cache_only"),
        ({"judge_model": "other"}, "judge_model"),
        ({"device": "cuda"}, "CUDA worker requested"),
        ({"cache_only": False, "api_base_url": "https://wrong.invalid/v1"}, "must match"),
    ],
)
def test_runtime_init_representative_contract_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    update: dict[str, Any],
    match: str,
) -> None:
    monkeypatch.setattr(worker.t.cuda, "is_available", lambda: False)
    payload = _init_payload(tmp_path, "LoRA")
    payload.update(update)
    with pytest.raises(ValueError, match=match):
        worker.WorkerRuntime(payload)


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content

    def to_dict(self) -> dict[str, Any]:
        return {
            "choices": [{"message": {"content": self.content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }


class FakeCompletions:
    def __init__(self, outcomes: list[str]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        return FakeResponse(self.outcomes.pop(0))


class FakeClient:
    def __init__(self, outcomes: list[str]) -> None:
        self.completions = FakeCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_inner_and_final_judge_share_exact_cache_keys_and_instruction_is_only_new_call(
    tmp_path: Path,
) -> None:
    client = FakeClient(
        [
            "concept explanation Rating: [[0.5]]",
            "fluency explanation Rating: [[2]]",
            "instruction explanation Rating: [[1]]",
        ]
    )
    language_model = LanguageModel(
        "openai/gpt-5.6-luna",
        client,
        use_cache=False,
        judge_cache=tmp_path / "judge.sqlite",
        api_base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
    )
    runtime = object.__new__(worker.WorkerRuntime)
    runtime.method = "PromptSteering"
    runtime.concept = "target concept"
    runtime.language_model = language_model
    runtime.evaluator = LMJudgeEvaluator("PromptSteering", concept_id=0, lm_model=language_model)
    assert runtime._judge("the exact generation")[:2] == (0.5, 2.0)
    assert language_model.stats.get_report()["total_calls"] == 2
    assert language_model.stats.get_report()["total_cache_hits"] == 0

    final = LMJudgeEvaluator("PromptSteering", concept_id=0, lm_model=language_model)
    metrics = final.compute_metrics(
        pd.DataFrame(
            {
                "dataset_name": ["AlpacaEval"],
                "factor": [1.0],
                "input_concept": ["target concept"],
                "original_prompt": ["original instruction"],
                "PromptSteering_steered_generation": ["the exact generation"],
            }
        )
    )
    assert metrics["raw_relevance_concept_ratings"] == [0.5]
    assert metrics["raw_relevance_instruction_ratings"] == [1.0]
    assert metrics["raw_fluency_ratings"] == [2.0]
    assert len(client.completions.calls) == 3
    report = language_model.stats.get_report()
    assert report["total_calls"] == 5
    assert report["total_cache_hits"] == 2
    with sqlite3.connect(tmp_path / "judge.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM judge_responses").fetchone()[0] == 3
    asyncio.run(language_model.close())
    assert client.closed is True


class FakeServeRuntime:
    method = "LoRA"
    concept_id = 0
    factor = 1.0

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.closed = False

    def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"echo": payload}

    def close(self) -> None:
        self.closed = True


def _request(request_id: int, operation: str, payload: dict[str, Any]) -> bytes:
    return encode_frame(
        {
            "id": request_id,
            "op": operation,
            "payload": payload,
            "version": PROTOCOL_VERSION,
        }
    )


def test_worker_server_supports_exact_init_evaluate_close_lifecycle() -> None:
    reader = io.BytesIO(
        _request(1, "init", {"x": 1})
        + _request(2, "evaluate", {"row": 7})
        + _request(3, "close", {})
    )
    writer = io.BytesIO()
    assert worker.serve(reader, writer, runtime_factory=FakeServeRuntime) == 0
    writer.seek(0)
    assert decode_frame(writer) == {
        "id": 1,
        "op": "init",
        "result": {"concept_id": 0, "factor": 1.0, "method": "LoRA"},
        "status": "ok",
        "version": 1,
    }
    assert decode_frame(writer) == {
        "id": 2,
        "op": "evaluate",
        "result": {"echo": {"row": 7}},
        "status": "ok",
        "version": 1,
    }
    assert decode_frame(writer)["op"] == "close"


def test_worker_request_parser_rejects_version_payload_and_numeric_id_traps() -> None:
    base = {"id": 1, "op": "init", "payload": {}, "version": 1}
    for mutation, match in (
        ({"version": 2}, "protocol version"),
        ({"version": True}, "protocol version"),
        ({"id": True}, "request id"),
        ({"op": ""}, "operation"),
        ({"payload": []}, "payload"),
    ):
        with pytest.raises(ValueError, match=match):
            worker._request(base | mutation)


def test_worker_server_rejects_second_init_malformed_frame_and_broken_error_writer() -> None:
    writer = io.BytesIO()
    reader = io.BytesIO(_request(1, "init", {}) + _request(2, "init", {}))
    assert worker.serve(reader, writer, runtime_factory=FakeServeRuntime) == 1
    writer.seek(0)
    assert decode_frame(writer)["status"] == "ok"
    assert decode_frame(writer)["status"] == "error"

    assert worker.serve(io.BytesIO(), io.BytesIO()) == 1

    class BrokenWriter(io.BytesIO):
        def write(self, value: bytes) -> int:
            del value
            raise OSError("broken writer")

    assert worker.serve(io.BytesIO(_request(1, "unknown", {})), BrokenWriter()) == 1


@pytest.mark.parametrize("operation", ["unknown", "evaluate", "close"])
def test_worker_server_fails_cell_on_unknown_or_preinit_operation(operation: str) -> None:
    writer = io.BytesIO()
    assert worker.serve(io.BytesIO(_request(1, operation, {})), writer) == 1
    writer.seek(0)
    response = decode_frame(writer)
    assert response["status"] == "error"
    assert response["id"] == 1


def test_worker_source_has_no_steering_import() -> None:
    tree = ast.parse(Path(worker.__file__).read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
    assert not any(name == "steering" or name.startswith("steering.") for name in imports)


def test_worker_main_redirects_stdout_before_serving(monkeypatch: pytest.MonkeyPatch) -> None:
    reader, writer, stderr = io.BytesIO(), io.BytesIO(), object()
    fake_sys = SimpleNamespace(
        stdin=SimpleNamespace(buffer=reader),
        stdout=SimpleNamespace(buffer=writer),
        stderr=stderr,
    )
    monkeypatch.setattr(worker, "sys", fake_sys)
    monkeypatch.setattr(
        worker,
        "serve",
        lambda observed_reader, observed_writer: (
            17
            if (observed_reader, observed_writer, fake_sys.stdout) == (reader, writer, stderr)
            else pytest.fail("stdout was not redirected before serve")
        ),
    )
    assert worker.main() == 17


def test_shared_generation_wrapper_preserves_existing_kwargs_and_perplexity_default() -> None:
    class Benchmark:
        def predict_steer(self, frame: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
            self.frame = frame
            self.kwargs = kwargs
            return {"steered_generation": ["answer"]}

    benchmark = Benchmark()
    frame = pd.DataFrame({"concept_id": [4], "input": ["prompt"]})
    assert inference.run_steering_generation(
        benchmark,
        frame,
        steering_batch_size="1",
        steering_output_length="16",
        temperature="1.0",
        prefix_length=2,
        positions="all_prompt",
        disable_neuronpedia_max_act=False,
        intervene_on_prompt=True,
    ) == {"steered_generation": ["answer"]}
    assert benchmark.frame is frame
    assert benchmark.kwargs == {
        "concept_id": 4,
        "sae_link": None,
        "sae_id": None,
        "batch_size": 1,
        "eval_output_length": 16,
        "temperature": 1.0,
        "prefix_length": 2,
        "positions": "all_prompt",
        "use_synergy": False,
        "disable_neuronpedia_max_act": False,
        "intervene_on_prompt": True,
        "return_vector": False,
        "compute_perplexity": True,
    }


class _PredictionBatch(dict[str, t.Tensor]):
    @property
    def input_ids(self) -> t.Tensor:
        return self["input_ids"]

    def to(self, _device: str) -> _PredictionBatch:
        return self


class _PredictionTokenizer:
    pad_token_id = 0
    padding_side = "right"

    def __call__(self, _texts: list[str], **_kwargs: Any) -> _PredictionBatch:
        return _PredictionBatch(input_ids=t.tensor([[1, 2]], dtype=t.long))

    def decode(self, _tokens: t.Tensor, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return "exact generation"


class _PredictionModel:
    def __init__(self) -> None:
        self.forward_calls = 0

    def eval(self) -> None:
        return None

    def generate(self, **_kwargs: Any) -> t.Tensor:
        return t.tensor([[1, 2, 3]], dtype=t.long)

    def __call__(self, **_kwargs: Any) -> Any:
        self.forward_calls += 1
        return SimpleNamespace(logits=t.zeros((1, 2, 3)))


@pytest.mark.parametrize("model_type", [LoRA, PromptSteering])
@pytest.mark.parametrize(
    ("compute_perplexity", "expected_perplexity", "expected_forward_calls"),
    [(False, [], 0), (None, [3.0], 1)],
)
def test_vendored_prediction_can_skip_perplexity_without_changing_default(
    model_type: type[LoRA] | type[PromptSteering],
    compute_perplexity: bool | None,
    expected_perplexity: list[float],
    expected_forward_calls: int,
) -> None:
    benchmark = object.__new__(model_type)
    fake_model = _PredictionModel()
    benchmark.model = fake_model
    if model_type is LoRA:
        benchmark.ax_model = fake_model
    benchmark.tokenizer = _PredictionTokenizer()
    benchmark.device = "cpu"
    kwargs = {} if compute_perplexity is None else {"compute_perplexity": compute_perplexity}
    result = benchmark.predict_steer(
        pd.DataFrame({"input": ["plain"], "steered_input": ["steered"]}),
        batch_size=1,
        eval_output_length=1,
        **kwargs,
    )
    assert result["steered_generation"] == ["exact generation"]
    assert result["perplexity"] == pytest.approx(expected_perplexity)
    assert fake_model.forward_calls == expected_forward_calls


def test_distributed_rank_helper_preserves_ddp_and_supports_plain_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_utils.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(model_utils.torch.distributed, "is_initialized", lambda: False)
    assert model_utils.distributed_rank_or_zero() == 0
    monkeypatch.setattr(model_utils.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(model_utils.torch.distributed, "get_rank", lambda: 7)
    assert model_utils.distributed_rank_or_zero() == 7


@pytest.mark.parametrize("method", ["DiffMean", "LinearProbe", "LoRA", "PromptSteering"])
@pytest.mark.parametrize(
    "direction_method,pressure",
    [("DiffMean", "diffmean_counter"), ("LinearProbe", "linear_probe_counter")],
)
def test_worker_counter_composes_with_generation_intervention(
    tmp_path, monkeypatch, method, direction_method, pressure
):
    blocks = t.nn.ModuleList([t.nn.Identity(), t.nn.Identity(), t.nn.Identity()])

    class CounterModel(t.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(model_type="llama")
            self.model = SimpleNamespace(layers=blocks)

        def forward(self, value: t.Tensor, layer: int) -> t.Tensor:
            return self.model.layers[layer](value)

    model = CounterModel()
    runtime = object.__new__(worker.WorkerRuntime)
    runtime.method = method
    runtime.concept_id = 0
    runtime.layer = 1
    runtime.factor = 1.5
    runtime.temperature = 1.0
    runtime.output_length = 32
    runtime.worker_max_prompt_tokens = 8
    runtime.prefix_length = 1
    runtime.tokenizer = FakeTokenizer()
    runtime.benchmark_model = SimpleNamespace(model=model)
    runtime._judge = lambda generation: (0.25, 1.75, "rationale", 0)
    path = tmp_path / f"{direction_method}_weight.pt"
    direction = t.tensor([2.0, -4.0])
    t.save(direction.unsqueeze(0), path)
    observed = []

    def generate(_model, _frame, **_kwargs):
        value = t.ones(1, 2, 2)
        # The baseline's own additive pressure remains present.
        own = 3 * direction
        output = model(value + own, 1)
        observed.append(output)
        t.testing.assert_close(model(value, 0), value, rtol=0, atol=0)
        t.testing.assert_close(model(value, 2), value, rtol=0, atol=0)
        return {"steered_generation": ["generation"]}

    monkeypatch.setattr(worker, "run_steering_generation", generate)
    payload = {"input": "plain", "input_id": 0, "row_seed": 99, "steered_input": "steer"}
    runtime.evaluate(payload)
    runtime.evaluate(
        {
            **payload,
            "counter": {
                "pressure": pressure,
                "method": method,
                "concept_id": 0,
                "layer": 1,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "magnitude": 0.5,
            },
        }
    )
    t.testing.assert_close(observed[1], observed[0] - 0.5 * direction, rtol=0, atol=0)
    assert not blocks[1]._forward_hooks
    runtime.evaluate(payload)
    t.testing.assert_close(observed[2], observed[0], rtol=0, atol=0)
