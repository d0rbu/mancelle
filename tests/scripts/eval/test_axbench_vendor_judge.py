"""Focused regressions for the vendored AxBench judge path."""

from __future__ import annotations

import ast
import asyncio
import importlib
import json
import logging
import os
import subprocess
import sys
import traceback
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

_VENDOR_PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "vendor" / "axbench" / "axbench"
_TEST_PACKAGE = "test_vendored_axbench"
_DEFAULT_USAGE = object()
_OMIT_USAGE = object()


class _APIConnectionError(Exception):
    pass


class _APITimeoutError(Exception):
    pass


class _OpenAIStub(types.ModuleType):
    APIConnectionError: type[Exception]
    APITimeoutError: type[Exception]


_openai_stub = _OpenAIStub("openai")
_openai_stub.APIConnectionError = _APIConnectionError
_openai_stub.APITimeoutError = _APITimeoutError

_package = types.ModuleType(_TEST_PACKAGE)
_package.__path__ = [str(_VENDOR_PACKAGE_ROOT)]
sys.modules[_TEST_PACKAGE] = _package
_previous_openai = sys.modules.get("openai")
sys.modules["openai"] = _openai_stub
try:
    lm_judge = importlib.import_module(f"{_TEST_PACKAGE}.evaluators.lm_judge")
    language_models = importlib.import_module(f"{_TEST_PACKAGE}.models.language_models")
    judge_cache_module = importlib.import_module(f"{_TEST_PACKAGE}.evaluators.judge_cache")
    eval_args_module = importlib.import_module(f"{_TEST_PACKAGE}.scripts.args.eval_args")
finally:
    if _previous_openai is None:
        del sys.modules["openai"]
    else:
        sys.modules["openai"] = _previous_openai


class _Response:
    def __init__(
        self,
        content: Any,
        *,
        prompt_tokens: int = 3,
        completion_tokens: int = 2,
        usage: Any = _DEFAULT_USAGE,
    ):
        self.content = content
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.usage = usage

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "choices": [{"message": {"content": self.content}}],
        }
        if self.usage is _DEFAULT_USAGE:
            payload["usage"] = {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            }
        elif self.usage is not _OMIT_USAGE:
            payload["usage"] = self.usage
        return payload


class _StatusError(Exception):
    def __init__(self, status_code: int, retry_after: str | None = None):
        super().__init__(f"HTTP {status_code}")
        headers = {} if retry_after is None else {"Retry-After": retry_after}
        self.status_code = status_code
        self.response = SimpleNamespace(status_code=status_code, headers=headers)


class _Completions:
    def __init__(self, outcomes: list[Any]):
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _Client:
    def __init__(self, outcomes: list[Any]):
        self.completions = _Completions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _language_model(
    tmp_path: Path,
    client: _Client | None,
    *,
    cache_only: bool = False,
) -> Any:
    return language_models.LanguageModel(
        "openai/gpt-5.6-luna",
        client,
        use_cache=False,
        judge_cache=tmp_path / "judge.sqlite",
        cache_only=cache_only,
        api_base_url=" https://openrouter.ai/api/v1/// ",
        temperature=0.0,
    )


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("Explanation. Rating: [[0]]", 0.0),
        ("Rating: [[ 1.25 ]]", 1.25),
        ("Reasoning\nRating:[[2e0]]", 2.0),
    ],
)
def test_strict_rating_parser_accepts_one_finite_in_range_frame(
    completion: str, expected: float
) -> None:
    assert lm_judge.parse_judge_rating(completion) == expected


@pytest.mark.parametrize(
    "completion",
    [
        "No rating",
        "Rating: 1",
        "Rating: [[nan]]",
        "Rating: [[inf]]",
        "Rating: [[-0.1]]",
        "Rating: [[2.1]]",
        "Rating: [[1]] and Rating: [[2]]",
        "Rating: [[1]] and a stray Rating:",
    ],
)
def test_strict_rating_parser_never_silently_substitutes_zero(completion: str) -> None:
    with pytest.raises(lm_judge.JudgeRatingParseError):
        lm_judge.parse_judge_rating(completion)


def test_lm_judge_calls_only_declared_cells_and_expands_nulls() -> None:
    class FakeLanguageModel:
        def __init__(self) -> None:
            self.calls: list[tuple[str, list[str]]] = []

        async def chat_completions(
            self,
            api_name: str,
            prompts: list[str],
            *,
            batch_size: int,
            validator: Any,
        ) -> list[str]:
            self.calls.append((api_name, prompts))
            if "concept" in api_name:
                response = "Concept explanation. Rating: [[2]]"
            elif "instruction" in api_name:
                response = "Instruction explanation. Rating: [[1]]"
            else:
                response = "Fluency explanation. Rating: [[2]]"
            validator(response)
            return [response] * len(prompts)

    fake = FakeLanguageModel()
    evaluator = lm_judge.LMJudgeEvaluator("PromptSteering", concept_id=0, lm_model=fake)
    data = pd.DataFrame(
        {
            "dataset_name": ["AlpacaEval"] * 3,
            "input_concept": ["concept"] * 3,
            "original_prompt": ["prompt"] * 3,
            "factor": [0.0, 1.0, 2.0],
            "PromptSteering_steered_generation": [None, "", np.nan],
        }
    )

    metrics = evaluator.compute_metrics(data)

    assert len(fake.calls) == 3
    assert all(len(prompts) == 1 for _, prompts in fake.calls)
    assert metrics["factor"] == [1.0]
    assert metrics["relevance_concept_ratings"] == [2.0]
    assert metrics["relevance_instruction_ratings"] == [1.0]
    assert metrics["fluency_ratings"] == [2.0]
    assert metrics["raw_relevance_concept_ratings"] == [None, 2.0, None]
    assert metrics["raw_relevance_instruction_ratings"] == [None, 1.0, None]
    assert metrics["raw_fluency_ratings"] == [None, 2.0, None]
    assert metrics["raw_aggregated_ratings"] == [None, 1.5, None]
    assert metrics["relevance_concept_completions"][0] is None
    assert metrics["relevance_concept_completions"][2] is None
    assert "[Sentence Fragment Start]\n\n[Sentence Fragment End]" in fake.calls[0][1][0]
    json.dumps(metrics, allow_nan=False)


def test_live_cache_then_cache_only_hit_uses_exact_endpoint(tmp_path: Path) -> None:
    client = _Client([_Response("Explanation. Rating: [[1.5]]")])
    model = _language_model(tmp_path, client)

    live = asyncio.run(
        model.chat_completions(
            "judge",
            ["prompt"],
            validator=lm_judge.parse_judge_rating,
        )
    )
    request = model._judge_request("prompt")
    assert request == {
        "version": 1,
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "model": "openai/gpt-5.6-luna",
        "messages": [{"role": "user", "content": "prompt"}],
        "temperature": 0.0,
    }
    assert client.completions.calls == [
        {
            "messages": request["messages"],
            "model": "openai/gpt-5.6-luna",
            "temperature": 0.0,
        }
    ]
    assert live == ["Explanation. Rating: [[1.5]]"]
    asyncio.run(model.close())

    cached = _language_model(tmp_path, None, cache_only=True)
    assert asyncio.run(
        cached.chat_completions(
            "judge",
            ["prompt"],
            validator=lm_judge.parse_judge_rating,
        )
    ) == ["Explanation. Rating: [[1.5]]"]
    assert cached.stats.total_cache_hit == 1
    with pytest.raises(language_models.JudgeCacheOnlyMissError, match="cache-only judge miss"):
        asyncio.run(
            cached.chat_completions(
                "judge",
                ["different prompt"],
                validator=lm_judge.parse_judge_rating,
            )
        )
    asyncio.run(cached.close())


def test_cache_hits_are_strictly_validated(tmp_path: Path) -> None:
    request = {
        "version": 1,
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "model": "openai/gpt-5.6-luna",
        "messages": [{"role": "user", "content": "prompt"}],
        "temperature": 0.0,
    }
    with judge_cache_module.JudgeCache(tmp_path / "judge.sqlite") as cache:
        cache.put_if_absent(request, "malformed cached response")

    model = _language_model(tmp_path, None, cache_only=True)
    with pytest.raises(lm_judge.JudgeRatingParseError):
        asyncio.run(
            model.chat_completions("judge", ["prompt"], validator=lm_judge.parse_judge_rating)
        )
    asyncio.run(model.close())


def test_retry_policy_is_one_layer_bounded_and_validates_before_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client(
        [
            _StatusError(429, retry_after="0.25"),
            _Response(None),
            _Response("malformed"),
            _Response("Explanation. Rating: [[2]]"),
        ]
    )
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(language_models.asyncio, "sleep", fake_sleep)
    model = _language_model(tmp_path, client)
    assert asyncio.run(
        model.chat_completions("judge", ["prompt"], validator=lm_judge.parse_judge_rating)
    ) == ["Explanation. Rating: [[2]]"]
    assert len(client.completions.calls) == 4
    request = model._judge_request("prompt")
    assert sleeps == [
        0.25,
        language_models._jittered_retry_delay(2.0, 1, request),
        language_models._jittered_retry_delay(4.0, 2, request),
    ]
    assert model.stats.total_call == 1
    assert model.stats.total_cache_hit == 0
    asyncio.run(model.close())

    with judge_cache_module.JudgeCache(tmp_path / "judge.sqlite") as cache:
        assert cache.get(model._judge_request("prompt")) == "Explanation. Rating: [[2]]"


# --- Retry-After ruling conformance (issue #1313) --------------------------
#
# Owner ruling, verbatim: honor the server-sent `Retry-After` but
# `logger.warning` if it's over 60 seconds and `logger.error` if it's over
# 300 seconds. `_retry_after_seconds` must never clamp a header value -- only
# the headerless exponential-backoff fallback keeps its 60s cap.


def _retry_after_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Only records logged from inside ``_retry_after_seconds`` itself --
    never the retry loop's own generic per-attempt bookkeeping
    ``logger.warning`` (same logger, different call site, fires on every
    retry regardless of band), so that unrelated log can't hide a missing
    band log or masquerade as an extra one."""
    return [
        r
        for r in caplog.records
        if r.name == language_models.logger.name and r.funcName == "_retry_after_seconds"
    ]


@pytest.mark.parametrize(
    ("retry_after", "expected_levels"),
    [
        (30.0, []),
        (60.0, []),
        (60.5, [logging.WARNING]),
        (61.0, [logging.WARNING]),
        (300.0, [logging.WARNING]),
        (301.0, [logging.ERROR]),
        (3600.0, [logging.ERROR]),
    ],
    ids=[
        "30-silent",
        "exactly-60-silent",
        "60.5-warns",
        "61-warns",
        "exactly-300-warns",
        "301-errors",
        "3600-errors",
    ],
)
def test_retry_after_seconds_honors_header_verbatim_and_logs_by_band(
    caplog: pytest.LogCaptureFixture,
    retry_after: float,
    expected_levels: list[int],
) -> None:
    """The returned delay is the server's number unchanged in every band
    (including 3600s, which the old 60s clamp would have silently
    shortened 60-fold), and the band LEVELS are pinned as an exact list so
    an "error INSTEAD OF warning" regression (both logged at once) fails
    here rather than merely "something was logged"."""
    exc = _StatusError(429, retry_after=str(retry_after))

    with caplog.at_level(logging.DEBUG, logger=language_models.logger.name):
        delay = language_models._retry_after_seconds(exc, 0, "openai/gpt-5.6-luna")

    assert delay == retry_after
    assert [r.levelno for r in _retry_after_records(caplog)] == expected_levels


@pytest.mark.parametrize(
    ("retry_after", "level"),
    [(61.0, logging.WARNING), (301.0, logging.ERROR)],
    ids=["warning-band", "error-band"],
)
def test_retry_after_seconds_log_message_names_model_delay_and_server(
    caplog: pytest.LogCaptureFixture,
    retry_after: float,
    level: int,
) -> None:
    """Message CONTENT is pinned, not merely its level: an operator reading
    the log during a paid run must be able to tell which model is parked,
    for how long, and that the SERVER (not our backoff formula) asked for
    it."""
    exc = _StatusError(429, retry_after=str(retry_after))

    with caplog.at_level(logging.DEBUG, logger=language_models.logger.name):
        language_models._retry_after_seconds(exc, 0, "openai/gpt-5.6-luna")

    records = _retry_after_records(caplog)
    assert len(records) == 1
    assert records[0].levelno == level
    message = records[0].getMessage()
    assert "openai/gpt-5.6-luna" in message
    assert str(retry_after) in message
    assert "Retry-After" in message
    assert "server" in message.lower()


@pytest.mark.parametrize(
    "exc",
    [
        _StatusError(429, retry_after="soon"),
        _StatusError(429, retry_after="-5"),
        _StatusError(429, retry_after="inf"),
        _StatusError(429, retry_after="nan"),
        _StatusError(429),
    ],
    ids=["malformed", "negative", "infinite", "nan", "missing"],
)
def test_retry_after_seconds_falls_back_to_capped_backoff_without_band_logs(
    caplog: pytest.LogCaptureFixture,
    exc: Exception,
) -> None:
    """A missing or unusable header must never itself trigger a band log --
    only a header value that the server actually sent and that we actually
    honor does."""
    with caplog.at_level(logging.DEBUG, logger=language_models.logger.name):
        delay = language_models._retry_after_seconds(exc, 10, "openai/gpt-5.6-luna")

    assert delay == language_models.MAX_RETRY_DELAY_SECONDS
    assert _retry_after_records(caplog) == []


def test_retry_loop_sleeps_the_verbatim_retry_after_and_logs_the_error_band(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """INTEGRATION, through the real retry loop: a 429 carrying a
    Retry-After above the error band must make the actual sleep receive the
    server's exact number (no cap, no rounding) and emit exactly one ERROR
    record."""
    client = _Client(
        [
            _StatusError(429, retry_after="301"),
            _Response("Explanation. Rating: [[1]]"),
        ]
    )
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(language_models.asyncio, "sleep", fake_sleep)
    model = _language_model(tmp_path, client)

    with caplog.at_level(logging.DEBUG, logger=language_models.logger.name):
        result = asyncio.run(
            model.chat_completions("judge", ["prompt"], validator=lm_judge.parse_judge_rating)
        )

    assert result == ["Explanation. Rating: [[1]]"]
    assert sleeps == [301.0]
    assert [r.levelno for r in _retry_after_records(caplog)] == [logging.ERROR]
    asyncio.run(model.close())


def test_retry_loop_without_a_header_sleeps_the_jittered_fallback_and_never_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A plain 429 uses request-derived fallback jitter without a band log."""
    client = _Client([_StatusError(429), _Response("Explanation. Rating: [[1]]")])
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(language_models.asyncio, "sleep", fake_sleep)
    model = _language_model(tmp_path, client)

    with caplog.at_level(logging.DEBUG, logger=language_models.logger.name):
        result = asyncio.run(
            model.chat_completions("judge", ["prompt"], validator=lm_judge.parse_judge_rating)
        )

    assert result == ["Explanation. Rating: [[1]]"]
    assert sleeps == [
        language_models._jittered_retry_delay(
            1.0,
            0,
            model._judge_request("prompt"),
        )
    ]
    assert _retry_after_records(caplog) == []
    asyncio.run(model.close())


def test_exhausted_malformed_responses_raise_named_parser_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client([_Response("bad")] * 4)
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(language_models.asyncio, "sleep", fake_sleep)
    model = _language_model(tmp_path, client)
    with pytest.raises(lm_judge.JudgeRatingParseError):
        asyncio.run(
            model.chat_completions("judge", ["prompt"], validator=lm_judge.parse_judge_rating)
        )
    assert len(client.completions.calls) == 4
    request = model._judge_request("prompt")
    assert sleeps == [
        language_models._jittered_retry_delay(1.0, 0, request),
        language_models._jittered_retry_delay(2.0, 1, request),
        language_models._jittered_retry_delay(4.0, 2, request),
    ]
    asyncio.run(model.close())


@pytest.mark.parametrize("error", [_StatusError(400), RuntimeError("unexpected")])
def test_nonretryable_errors_propagate_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    client = _Client([error])

    async def forbidden_sleep(_delay: float) -> None:
        pytest.fail("nonretryable failure slept")

    monkeypatch.setattr(language_models.asyncio, "sleep", forbidden_sleep)
    model = _language_model(tmp_path, client)
    with pytest.raises(type(error), match=str(error)):
        asyncio.run(model.chat_completions("judge", ["prompt"]))
    assert len(client.completions.calls) == 1
    asyncio.run(model.close())


def test_cache_only_rejects_a_live_client_and_unknown_price_is_unavailable(
    tmp_path: Path,
) -> None:
    with judge_cache_module.JudgeCache(tmp_path / "judge.sqlite"):
        pass
    with pytest.raises(ValueError, match="must not receive a live client"):
        _language_model(tmp_path, _Client([]), cache_only=True)

    stats = language_models.LanguageModelStats("openai/gpt-5.6-luna")
    stats.record(
        "judge",
        {"prompt_tokens": 2, "completion_tokens": 3},
    )
    assert stats.get_total_price() is None
    assert stats.get_report()["total_price"] is None

    priced = language_models.LanguageModelStats("gpt-4o")
    priced.record(
        "judge",
        {"prompt_tokens": 2, "completion_tokens": 3},
    )
    assert priced.get_total_price() == pytest.approx((2 / 1_000_000) * 5.0 + (3 / 1_000_000) * 15.0)


def test_eval_args_parse_explicit_cache_endpoint_and_boolean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "evaluate.yaml"
    config.write_text(
        "evaluate:\n"
        "  judge_cache: from-config.sqlite\n"
        "  cache_only: false\n"
        "  api_base_url: https://configured.example/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--config",
            str(config),
            "--judge_cache",
            "from-cli.sqlite",
            "--cache_only",
            "true",
            "--api_base_url",
            "https://openrouter.ai/api/v1",
        ],
    )

    args = eval_args_module.EvalArgs(section="evaluate")

    assert args.judge_cache == "from-cli.sqlite"
    assert args.cache_only is True
    assert args.api_base_url == "https://openrouter.ai/api/v1"


def test_evaluate_client_uses_openrouter_key_timeout_and_no_sdk_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (_VENDOR_PACKAGE_ROOT / "scripts" / "evaluate.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_create_judge_client"
    )
    namespace: dict[str, Any] = {
        "os": os,
        "httpx": SimpleNamespace(
            Limits=lambda **kwargs: ("limits", kwargs),
            AsyncClient=lambda **kwargs: ("http-client", kwargs),
        ),
    }
    created: list[dict[str, Any]] = []

    def fake_async_openai(**kwargs: Any) -> object:
        created.append(kwargs)
        return object()

    namespace["AsyncOpenAI"] = fake_async_openai
    exec(compile(ast.Module(body=[function], type_ignores=[]), "evaluate.py", "exec"), namespace)
    create_client = namespace["_create_judge_client"]

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert create_client("https://openrouter.ai/api/v1", True) is None
    assert created == []
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        create_client("https://openrouter.ai/api/v1", False)

    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    create_client(" https://openrouter.ai/api/v1/// ", False)
    assert created[0]["api_key"] == "secret"
    assert created[0]["base_url"] == "https://openrouter.ai/api/v1"
    assert created[0]["timeout"] == 60.0
    assert created[0]["max_retries"] == 0


def _steering_iterator_namespace(
    task: Any,
    executor: type[Any],
) -> dict[str, Any]:
    source = (_VENDOR_PACKAGE_ROOT / "scripts" / "evaluate.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_eval_steering_process_task", "_steering_result_iterator"}
    ]
    namespace: dict[str, Any] = {
        "contextmanager": contextmanager,
        "eval_steering_single_task": task,
        "ProcessPoolExecutor": executor,
        "traceback": traceback,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), "evaluate.py", "exec"), namespace)
    return namespace


def test_single_worker_evaluation_propagates_provider_error_in_process() -> None:
    sentinel = _StatusError(401)

    def fail_task(_task: object) -> object:
        raise sentinel

    class ForbiddenExecutor:
        def __init__(self, **_kwargs: Any) -> None:
            pytest.fail("single-worker evaluation constructed a process pool")

    namespace = _steering_iterator_namespace(fail_task, ForbiddenExecutor)

    with pytest.raises(_StatusError) as captured:
        with namespace["_steering_result_iterator"]([object()], 1) as results:
            list(results)
    assert captured.value is sentinel


def test_multiworker_evaluation_is_ordered_exactly_once_and_closes_executor() -> None:
    events: list[object] = []

    def task(value: tuple[object, ...]) -> int:
        concept_id = value[0]
        assert isinstance(concept_id, int)
        events.append(("task", concept_id))
        return concept_id * 10

    class Executor:
        def __init__(self, *, max_workers: int) -> None:
            events.append(("init", max_workers))

        def __enter__(self) -> Executor:
            events.append("enter")
            return self

        def map(self, function: Any, values: list[tuple[object, ...]]) -> Any:
            events.append("map")
            return map(function, values)

        def __exit__(self, *exc: object) -> None:
            events.append(("exit", exc[0]))

    namespace = _steering_iterator_namespace(task, Executor)
    tasks = [
        (3, object(), "Evaluator3", "Model3"),
        (1, object(), "Evaluator1", "Model1"),
        (2, object(), "Evaluator2", "Model2"),
    ]
    with namespace["_steering_result_iterator"](tasks, 2) as results:
        assert list(results) == [30, 10, 20]

    assert events == [
        ("init", 2),
        "enter",
        "map",
        ("task", 3),
        ("task", 1),
        ("task", 2),
        ("exit", None),
    ]


@pytest.mark.parametrize("workers", [True, False, 0, -1, 1.5, "1", None])
def test_steering_iterator_rejects_invalid_worker_counts(workers: object) -> None:
    namespace = _steering_iterator_namespace(lambda value: value, object)

    with pytest.raises(ValueError, match="positive integer"):
        with namespace["_steering_result_iterator"]([], workers):
            pytest.fail("invalid worker count yielded an iterator")


def test_multiworker_wraps_unpicklable_error_with_task_identity_and_traceback() -> None:
    sentinel = _StatusError(401)

    def fail_task(_task: object) -> object:
        raise sentinel

    class Executor:
        def __init__(self, *, max_workers: int) -> None:
            assert max_workers == 2

        def __enter__(self) -> Executor:
            return self

        def map(self, function: Any, values: list[tuple[object, ...]]) -> Any:
            return map(function, values)

        def __exit__(self, *exc: object) -> None:
            pass

    namespace = _steering_iterator_namespace(fail_task, Executor)
    task = (7, object(), "LMJudgeEvaluator", "SelfConditioning")
    with pytest.raises(RuntimeError) as captured:
        with namespace["_steering_result_iterator"]([task], 2) as results:
            list(results)

    message = str(captured.value)
    assert "concept_id=7" in message
    assert "model=SelfConditioning" in message
    assert "evaluator=LMJudgeEvaluator" in message
    assert "_StatusError" in message
    assert "original traceback" in message
    assert "fail_task" in message


def test_single_task_preserves_primary_error_when_cleanup_also_fails() -> None:
    source = (_VENDOR_PACKAGE_ROOT / "scripts" / "evaluate.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "eval_steering_single_task"
    )
    primary = _StatusError(401)
    close_calls: list[None] = []

    class Model:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def close(self) -> None:
            close_calls.append(None)
            raise RuntimeError("close failed")

    class Evaluator:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def compute_metrics(self, _frame: pd.DataFrame) -> object:
            raise primary

    namespace: dict[str, Any] = {
        "asyncio": asyncio,
        "_create_judge_client": lambda *_args: object(),
        "LanguageModel": Model,
        "axbench": SimpleNamespace(LMJudgeEvaluator=Evaluator),
        "NEED_STANZA": set(),
        "CONCEPT_TO_RULE": {},
        "_get_stanza_pipeline": lambda: None,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "evaluate.py", "exec"), namespace)
    task = (
        7,
        pd.DataFrame({"input_concept": ["concept"]}),
        "LMJudgeEvaluator",
        "SelfConditioning",
        "dump",
        "judge-model",
        None,
        "concept",
        "cache.sqlite3",
        False,
        "https://openrouter.ai/api/v1",
    )

    with pytest.raises(_StatusError) as captured:
        namespace["eval_steering_single_task"](task)

    assert captured.value is primary
    assert close_calls == [None]
    assert any("cleanup also failed" in note for note in primary.__notes__)


@pytest.mark.slow
def test_real_vendor_openai_error_is_preserved_or_serialized_without_pickle_masking() -> None:
    root = Path(__file__).resolve().parents[3]
    program = r'''
import pickle

import httpx
from openai import APIStatusError
from axbench.scripts import evaluate

request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
response = httpx.Response(401, request=request)
sentinel = APIStatusError("expired credential", response=response, body={"error": "expired"})
try:
    pickle.loads(pickle.dumps(sentinel))
except TypeError:
    pass
else:
    raise AssertionError("installed APIStatusError unexpectedly became pickle-safe")

def fail(_task):
    raise sentinel

evaluate.eval_steering_single_task = fail
with evaluate._steering_result_iterator([(7, None, "LMJudgeEvaluator", "SelfConditioning")], 1) as results:
    try:
        list(results)
    except APIStatusError as exc:
        assert exc is sentinel
        assert exc.status_code == 401
    else:
        raise AssertionError("single-worker path swallowed the provider error")

try:
    evaluate._eval_steering_process_task((7, None, "LMJudgeEvaluator", "SelfConditioning"))
except RuntimeError as exc:
    message = str(exc)
    assert "APIStatusError" in message
    assert "401" in message
    assert "concept_id=7" in message
    assert "original traceback" in message
    pickle.loads(pickle.dumps(exc))
else:
    raise AssertionError("multi-worker adapter swallowed the provider error")
'''
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "vendor" / "axbench")
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(root / "vendor" / "axbench"),
            "--frozen",
            "--offline",
            "python",
            "-c",
            program,
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "retry_after",
    [None, "soon", "nan", "-1"],
    ids=["missing", "malformed", "nonfinite", "negative"],
)
def test_fallback_jitter_is_request_derived_stable_and_decorrelated(
    retry_after: str | None,
) -> None:
    error = _StatusError(429, retry_after=retry_after)
    assert not language_models._has_usable_retry_after(error)
    base_delay = language_models._retry_after_seconds(
        error,
        2,
        "openai/gpt-5.6-luna",
    )
    requests = [
        {
            "version": 1,
            "endpoint": "https://openrouter.ai/api/v1/chat/completions",
            "model": "openai/gpt-5.6-luna",
            "messages": [{"role": "user", "content": f"prompt-{index}"}],
            "temperature": 0.0,
        }
        for index in range(32)
    ]

    first = [language_models._jittered_retry_delay(base_delay, 2, request) for request in requests]
    second = [language_models._jittered_retry_delay(base_delay, 2, request) for request in requests]

    assert first == second
    assert len(set(first)) == len(first)
    assert all(base_delay <= delay < base_delay + 1.0 for delay in first)
    assert all(
        language_models._jittered_retry_delay(
            language_models.MAX_RETRY_DELAY_SECONDS,
            10,
            request,
        )
        == language_models.MAX_RETRY_DELAY_SECONDS
        for request in requests
    )


@pytest.mark.parametrize(
    "usage",
    [None, _OMIT_USAGE],
    ids=["null", "omitted"],
)
def test_optional_live_usage_is_cached_without_becoming_a_cache_hit(
    tmp_path: Path,
    usage: object,
) -> None:
    response = "Explanation. Rating: [[1.5]]"
    client = _Client([_Response(response, usage=usage)])
    model = _language_model(tmp_path, client)

    assert asyncio.run(
        model.chat_completions(
            "judge",
            ["prompt"],
            validator=lm_judge.parse_judge_rating,
        )
    ) == [response]
    assert model.stats.total_call == 1
    assert model.stats.total_cache_hit == 0
    assert model.stats.total_usage_unavailable == 1
    assert model.stats.get_total_tokens() == (0, 0)
    asyncio.run(model.close())

    replay = _language_model(tmp_path, None, cache_only=True)
    assert asyncio.run(
        replay.chat_completions(
            "judge",
            ["prompt"],
            validator=lm_judge.parse_judge_rating,
        )
    ) == [response]
    assert replay.stats.total_call == 1
    assert replay.stats.total_cache_hit == 1
    assert replay.stats.total_usage_unavailable == 0
    asyncio.run(replay.close())


def test_missing_usage_makes_known_price_unavailable_without_counting_a_hit() -> None:
    stats = language_models.LanguageModelStats("gpt-4o")

    stats.record("live", None, cache_hit=False)
    stats.record("cached", None, cache_hit=True)

    assert stats.total_call == 2
    assert stats.total_cache_hit == 1
    assert stats.total_usage_unavailable == 1
    assert stats.get_total_price() is None
