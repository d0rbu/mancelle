"""Property tests for concurrent Stage 5 judge scoring."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.eval.axbench_base import judge as judge_module
from steering.eval.axbench_base.config import AxBenchBaseConfig, JudgeConfig
from steering.eval.axbench_base.generation import GENERATIONS_SCHEMA
from steering.eval.axbench_base.judge import (
    _JUDGE_SCORES_SCHEMA,
    AxBenchJudgeClient,
    judge_all,
)
from steering.eval.axbench_base.parquet_writer import (
    ParquetAppendWriter,
    load_existing,
)
from steering.eval.axbench_base.types import Concept
from tests.axbench_base import paths as test_paths


class _Response:
    def __init__(self, score: int) -> None:
        self.status_code = 200
        self.text = ""
        self._score = score

    def json(self) -> dict[str, Any]:
        return {
            "choices": [
                {"message": {"content": f"Explanation. Rating: [[{self._score}]]"}},
            ],
        }


class _FakeSession:
    def __init__(
        self,
        post: Callable[..., _Response],
    ) -> None:
        self._post = post

    def post(self, *args: Any, **kwargs: Any) -> _Response:
        return self._post(*args, **kwargs)


class _BatchClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.closed = False

    def score(self, prompt: str) -> tuple[int | None, int]:
        self.prompts.append(prompt)
        return 1, 1

    def score_many(self, prompts: list[str]) -> list[tuple[int | None, int]]:
        self.prompts.extend(prompts)
        return [(1, 1)] * len(prompts)

    def __enter__(self) -> _BatchClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.closed = True


def _as_client(fake: _BatchClient) -> AxBenchJudgeClient:
    return cast(AxBenchJudgeClient, fake)


def _cfg(tmp_path: Path, *, run_tag: str, judge: JudgeConfig) -> AxBenchBaseConfig:
    return AxBenchBaseConfig(
        feature_dir="out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base",
        feature_statistics_path="stats.safetensors",
        decoder_only={"source": "base", "model_tag": "d13", "step": 10000},
        encdec={"source": "encdec", "model_tag": "d13", "step": 10000},
        device="cpu",
        output_root=None,
        run_tag=run_tag,
        judge=judge,
    )


def _concept() -> Concept:
    return Concept(
        concept_id=0,
        feature_index=0,
        layer=8,
        sae_source="stub",
        description="property concept",
        corpus_count=100,
        description_source="stub",
    )


def _generation_row(index: int) -> dict[str, Any]:
    return {
        "generation_key": f"prop_key_{index}",
        "concept_id": 0,
        "method_name": "diffmean_decoder_only",
        "mode": "positive",
        "eval_seed": index,
        "coefficient": "1.0",
        "generation_text": f"property generation {index}",
        "target_model": "decoder_only",
        "prefix_text": "prefix",
        "prompt_token_key": "tokens_sha256:" + "0" * 64,
        "prompt_token_count": 4,
        "seed_pool_fingerprint": "f" * 64,
        "token_count": 8,
        "stop_reason": "bos_token",
    }


def _write_generations(run_tag: str, rows: list[dict[str, Any]]) -> None:
    with ParquetAppendWriter(
        out_dir=test_paths.default_paths(run_tag).generations_dir(),
        schema=GENERATIONS_SCHEMA,
        batch_size=100,
    ) as writer:
        for row in rows:
            writer.append(row)


def _patch_openrouter_post(
    monkeypatch: pytest.MonkeyPatch,
    post: Callable[..., _Response],
) -> None:
    session = _FakeSession(post)
    monkeypatch.setattr(judge_module, "get_openrouter_session", lambda: session)


@given(st.lists(st.integers(min_value=0, max_value=2), min_size=1, max_size=20))
@settings(max_examples=20)
def test_score_many_deterministic_across_max_concurrent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scores: list[int],
) -> None:
    """max_concurrent=1 and max_concurrent=20 return identical ordered results."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-xyz")

    def _post(
        url: str,
        *,
        json: Any,
        headers: dict[str, str],
        timeout: float,
    ) -> _Response:
        del url, headers, timeout
        prompt = json["messages"][0]["content"]
        return _Response(int(prompt.rsplit("-", 1)[1]))

    _patch_openrouter_post(monkeypatch, _post)
    prompts = [f"prompt-{i}-{score}" for i, score in enumerate(scores)]

    serial = AxBenchJudgeClient(
        JudgeConfig(requests_per_minute=0, max_concurrent=1),
        cache_path=tmp_path / f"serial-{uuid.uuid4().hex}.sqlite",
    )
    concurrent = AxBenchJudgeClient(
        JudgeConfig(requests_per_minute=0, max_concurrent=20),
        cache_path=tmp_path / f"concurrent-{uuid.uuid4().hex}.sqlite",
    )
    try:
        assert concurrent.score_many(prompts) == serial.score_many(prompts)
    finally:
        serial.close()
        concurrent.close()


@given(st.lists(st.integers(min_value=0, max_value=2), min_size=1, max_size=20))
@settings(max_examples=20)
def test_score_many_cache_monotonicity_on_back_to_back_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scores: list[int],
) -> None:
    """A warm second run performs no extra HTTP calls and preserves values."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-xyz")
    n_calls = 0
    n_calls_lock = threading.Lock()

    def _post(
        url: str,
        *,
        json: Any,
        headers: dict[str, str],
        timeout: float,
    ) -> _Response:
        del url, headers, timeout
        nonlocal n_calls
        with n_calls_lock:
            n_calls += 1
        prompt = json["messages"][0]["content"]
        return _Response(int(prompt.rsplit("-", 1)[1]))

    _patch_openrouter_post(monkeypatch, _post)
    prompts = [f"cache-{i}-{score}" for i, score in enumerate(scores)]

    client = AxBenchJudgeClient(
        JudgeConfig(requests_per_minute=0, max_concurrent=8),
        cache_path=tmp_path / f"cache-{uuid.uuid4().hex}.sqlite",
    )
    try:
        first = client.score_many(prompts)
        first_call_count = n_calls
        second = client.score_many(prompts)
    finally:
        client.close()

    assert second == [(score, 1) for score, _ in first]
    assert n_calls == first_call_count


@given(st.integers(min_value=1, max_value=45))
@settings(max_examples=15)
def test_judge_all_has_no_lost_or_duplicated_generation_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    n_rows: int,
) -> None:
    """Stage 5 output has exactly one row for each input generation key."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = f"prop_keys_{uuid.uuid4().hex}"
    cfg = _cfg(
        tmp_path,
        run_tag=run_tag,
        judge=JudgeConfig(requests_per_minute=0, max_concurrent=3),
    )
    rows = [_generation_row(i) for i in range(n_rows)]
    _write_generations(run_tag, rows)

    judge_all(cfg, concepts=[_concept()], client=_as_client(_BatchClient()))

    table = load_existing(
        test_paths.default_paths(run_tag).judge_scores_dir(),
        schema=_JUDGE_SCORES_SCHEMA,
    )
    output_keys = [row["generation_key"] for row in table.to_pylist()]
    input_keys = [row["generation_key"] for row in rows]
    assert output_keys == input_keys
    assert len(output_keys) == len(set(output_keys))
