"""Tests for async judge methods on CompletionJudgeClient."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
import yaml

from errors import ParseError
from steering.eval.completion_judge import (
    CompletionJudgeClient,
    CompletionJudgeItem,
    JudgementScore,
    build_judge_prompt,
    save_scores_yaml,
)


@pytest.fixture
def tmp_cache(tmp_path: Path) -> str:
    """Return a temporary cache path for test clients."""
    return str(tmp_path / "test_judge.db")


@pytest.fixture
def client(tmp_cache: str) -> CompletionJudgeClient:
    """Create a CompletionJudgeClient with a temp cache."""
    return CompletionJudgeClient(
        model="test-model",
        base_url="https://test.example.com/v1",
        api_key="test-key",
        cache_path=tmp_cache,
        concurrency=5,
    )


def _make_mock_session(response_text: str, delay: float = 0.0) -> AsyncMock:
    """Create a mock aiohttp session that returns the given response text."""
    mock_resp = AsyncMock()
    mock_resp.status = 200

    async def mock_json() -> dict:
        return {"choices": [{"message": {"content": response_text}}]}

    mock_resp.json = mock_json
    mock_resp.raise_for_status = MagicMock()

    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_ctx = AsyncMock()

    async def mock_aenter(*_args: object) -> AsyncMock:
        if delay > 0:
            await asyncio.sleep(delay)
        return mock_resp

    mock_ctx.__aenter__ = mock_aenter
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_session.post = MagicMock(return_value=mock_ctx)

    return mock_session


class TestCallApiAsync:
    """Tests for _call_api_async."""

    def test_call_api_async_returns_content(self, client: CompletionJudgeClient) -> None:
        """_call_api_async sends correct payload and returns response content."""
        mock_response_data = {
            "choices": [{"message": {"content": "coherence: 4\nfeature_alignment: 3"}}]
        }

        async def run() -> str:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response_data)
            mock_resp.raise_for_status = MagicMock()

            mock_session = AsyncMock(spec=aiohttp.ClientSession)
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_session.post = MagicMock(return_value=mock_ctx)

            messages = [
                {"role": "system", "content": "You are a judge."},
                {"role": "user", "content": "Rate this."},
            ]
            result = await client._call_api_async(mock_session, messages)

            mock_session.post.assert_called_once()
            call_kwargs = mock_session.post.call_args
            assert call_kwargs[1]["json"]["model"] == "test-model"
            assert call_kwargs[1]["json"]["messages"] == messages

            return result

        result = asyncio.run(run())
        assert result == "coherence: 4\nfeature_alignment: 3"


class TestJudgeAsync:
    """Tests for judge_async method."""

    def test_judge_async_returns_score(self, client: CompletionJudgeClient) -> None:
        async def run() -> JudgementScore | None:
            session = _make_mock_session("coherence: 4\nfeature_alignment: 3")
            sem = asyncio.Semaphore(5)
            return await client.judge_async(
                session=session,
                semaphore=sem,
                context="The cat sat on the",
                completion="mat and purred softly.",
                feature_description="cat-related content",
                ground_truth="mat.",
            )

        result = asyncio.run(run())
        assert result is not None
        assert result.coherence == 4
        assert result.feature_alignment == 3

    def test_judge_async_cache_hit_skips_api(self, client: CompletionJudgeClient) -> None:
        async def run() -> tuple[JudgementScore | None, int]:
            session = _make_mock_session("coherence: 5\nfeature_alignment: 5")
            sem = asyncio.Semaphore(5)
            kwargs = {
                "context": "Hello",
                "completion": "World",
                "feature_description": "greeting",
                "ground_truth": "World!",
            }
            await client.judge_async(session=session, semaphore=sem, **kwargs)  # ty: ignore[invalid-argument-type]
            session.post.reset_mock()
            result2 = await client.judge_async(session=session, semaphore=sem, **kwargs)  # ty: ignore[invalid-argument-type]
            return result2, session.post.call_count

        result, api_calls = asyncio.run(run())
        assert result is not None
        assert result.coherence == 5
        assert api_calls == 0

    def test_judge_async_retries_on_error(self, client: CompletionJudgeClient) -> None:
        call_count = 0

        async def run() -> JudgementScore | None:
            nonlocal call_count
            mock_resp = AsyncMock()
            mock_resp.status = 500
            mock_resp.raise_for_status = MagicMock(
                side_effect=aiohttp.ClientResponseError(
                    request_info=MagicMock(),
                    history=(),
                    status=500,
                    message="Internal Server Error",
                )
            )
            mock_session = AsyncMock(spec=aiohttp.ClientSession)
            mock_ctx = AsyncMock()

            async def counting_aenter(*_args: object) -> AsyncMock:
                nonlocal call_count
                call_count += 1
                return mock_resp

            mock_ctx.__aenter__ = counting_aenter
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_session.post = MagicMock(return_value=mock_ctx)

            sem = asyncio.Semaphore(5)
            return await client.judge_async(
                session=mock_session,
                semaphore=sem,
                context="test",
                completion="test",
                feature_description="test",
                ground_truth="test",
            )

        with pytest.raises(ParseError, match="Failed to judge"):
            asyncio.run(run())
        assert call_count == 3


class TestJudgeBatchAsync:
    """Tests for judge_batch_async method."""

    def test_batch_async_returns_ordered_results(self, client: CompletionJudgeClient) -> None:
        items = [
            CompletionJudgeItem(
                context=f"ctx_{i}",
                completion=f"comp_{i}",
                feature_description="test feature",
                ground_truth=f"gt_{i}",
            )
            for i in range(5)
        ]

        async def run() -> list[JudgementScore]:
            session = _make_mock_session("coherence: 3\nfeature_alignment: 4")
            return await client.judge_batch_async(items, session=session)

        results = asyncio.run(run())
        assert len(results) == 5
        assert all(r is not None for r in results)
        assert all(r.coherence == 3 for r in results if r)

    def test_batch_async_concurrency_speedup(self, client: CompletionJudgeClient) -> None:
        items = [
            CompletionJudgeItem(
                context=f"ctx_{i}",
                completion=f"comp_{i}",
                feature_description="test feature",
                ground_truth=f"gt_{i}",
            )
            for i in range(5)
        ]

        async def run() -> float:
            session = _make_mock_session("coherence: 3\nfeature_alignment: 4", delay=0.2)
            t0 = time.monotonic()
            await client.judge_batch_async(items, concurrency=5, session=session)
            return time.monotonic() - t0

        elapsed = asyncio.run(run())
        assert elapsed < 0.8, f"Expected <0.8s, got {elapsed:.2f}s — not running concurrently"

    def test_semaphore_caps_concurrency(self, client: CompletionJudgeClient) -> None:
        peak_concurrent = 0
        current_concurrent = 0
        concurrency_limit = 3
        items = [
            CompletionJudgeItem(
                context=f"ctx_{i}",
                completion=f"comp_{i}",
                feature_description="test feature",
                ground_truth=f"gt_{i}",
            )
            for i in range(10)
        ]

        async def run() -> int:
            nonlocal peak_concurrent, current_concurrent
            original_call = client._call_api_async

            async def tracking_call(
                session: aiohttp.ClientSession,
                messages: list[dict[str, str]],
            ) -> str:
                nonlocal peak_concurrent, current_concurrent
                current_concurrent += 1
                if current_concurrent > peak_concurrent:
                    peak_concurrent = current_concurrent
                try:
                    return await original_call(session, messages)
                finally:
                    current_concurrent -= 1

            client._call_api_async = tracking_call  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
            session = _make_mock_session("coherence: 3\nfeature_alignment: 4", delay=0.1)
            await client.judge_batch_async(items, concurrency=concurrency_limit, session=session)
            return peak_concurrent

        peak = asyncio.run(run())
        assert peak <= concurrency_limit, f"Peak ({peak}) exceeded limit ({concurrency_limit})"
        assert peak >= 2, f"Peak ({peak}) suggests no actual concurrency"

    def test_batch_async_with_optional_session(self, client: CompletionJudgeClient) -> None:
        items = [
            CompletionJudgeItem(
                context="ctx",
                completion="comp",
                feature_description="test",
                ground_truth="gt",
            )
        ]

        async def run() -> list[JudgementScore]:
            inner_session = _make_mock_session("coherence: 4\nfeature_alignment: 4")
            inner_session.close = AsyncMock()
            with patch(
                "steering.eval.completion_judge.aiohttp.ClientSession",
                return_value=inner_session,
            ):
                return await client.judge_batch_async(items)

        results = asyncio.run(run())
        assert len(results) == 1
        assert results[0] is not None


class TestIncrementalScoreSaving:
    """Tests for incremental YAML score saving."""

    def test_save_scores_yaml_atomic_write(self, tmp_path: Path) -> None:

        scores_path = tmp_path / "scores.yaml"
        scores = {1: {"real": [{"coherence": 4, "feature_alignment": 3}]}}
        save_scores_yaml(scores, scores_path)

        assert scores_path.exists()
        with open(scores_path) as f:
            loaded = yaml.safe_load(f)
        assert loaded == scores

    def test_incremental_saves_grow_per_feature(self, tmp_path: Path) -> None:

        scores_path = tmp_path / "scores.yaml"
        all_scores: dict = {}

        all_scores[1] = {"real": [{"coherence": 4, "feature_alignment": 3}]}
        save_scores_yaml(all_scores, scores_path)
        with open(scores_path) as f:
            loaded1 = yaml.safe_load(f)
        assert len(loaded1) == 1

        all_scores[2] = {"ablated": [{"coherence": 5, "feature_alignment": 5}]}
        save_scores_yaml(all_scores, scores_path)
        with open(scores_path) as f:
            loaded2 = yaml.safe_load(f)
        assert len(loaded2) == 2
        assert 1 in loaded2
        assert 2 in loaded2


class TestCacheAsyncLock:
    """Tests for asyncio.Lock protecting cache in concurrent access."""

    def test_concurrent_cache_access_no_corruption(self, client: CompletionJudgeClient) -> None:
        """Multiple concurrent judge_async calls with same input produce valid results.

        The asyncio.Lock prevents corrupt reads/writes but does NOT deduplicate
        API calls for identical inputs (the lock is released between cache-check
        and API call). This test verifies data integrity, not deduplication.
        """

        async def run() -> list[JudgementScore | None]:
            session = _make_mock_session("coherence: 4\nfeature_alignment: 3", delay=0.05)
            sem = asyncio.Semaphore(10)
            kwargs = {
                "context": "same context",
                "completion": "same completion",
                "feature_description": "same feature",
                "ground_truth": "same gt",
            }
            tasks = [
                client.judge_async(session=session, semaphore=sem, **kwargs)  # ty: ignore[invalid-argument-type]
                for _ in range(10)
            ]
            return list(await asyncio.gather(*tasks))

        results = asyncio.run(run())

        assert len(results) == 10
        assert all(r is not None for r in results)
        assert all(r.coherence == 4 for r in results if r)
        assert all(r.feature_alignment == 3 for r in results if r)

        # Verify cache was populated
        sys_p, usr_p = build_judge_prompt(
            "same context", "same completion", "same feature", "same gt"
        )
        key = client._cache_key(f"{sys_p}\n---\n{usr_p}")
        assert client._cache_get(key) is not None, "Cache should contain the result"
