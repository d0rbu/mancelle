"""Threading / asyncio regression tests for CompletionJudgeClient sqlite cache.

Audit Wave 3B task 7 (P0): the persistent ``sqlite3.Connection`` stored on
``CompletionJudgeClient`` was opened with ``check_same_thread=True`` (the
sqlite3 default). Any call to ``judge()`` from a ``ThreadPoolExecutor``
worker or from a coroutine scheduled on a non-creating thread raises
``sqlite3.ProgrammingError: SQLite objects created in a thread can only be
used in that same thread``. ``asyncio.Lock`` does not cross threads either.

The tests below use real ``ThreadPoolExecutor`` dispatch, real
``asyncio.gather`` dispatch, and a real on-disk sqlite file under
``tmp_path``. API calls are mocked at the module level so no network
traffic is emitted, but everything else runs genuinely.

Cache-row counts are verified by opening an independent ``sqlite3``
connection from the main test thread — this never shares state with the
client under test.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from steering.eval.completion_judge import (
    CompletionJudgeClient,
    JudgementScore,
    build_judge_prompt,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_cache_rows(cache_path: str) -> int:
    """Open an independent sqlite connection and count cache rows."""
    conn = sqlite3.connect(cache_path)
    try:
        (count,) = conn.execute("SELECT COUNT(*) FROM cache").fetchone()
        return int(count)
    finally:
        conn.close()


def _fetch_all_cache_keys(cache_path: str) -> list[str]:
    """Return all cache keys via an independent connection."""
    conn = sqlite3.connect(cache_path)
    try:
        rows = conn.execute("SELECT key FROM cache").fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def _make_sync_mock_response(content: str) -> MagicMock:
    """requests.post return value with the given choices[0].message.content."""
    resp = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    resp.raise_for_status = MagicMock()
    return resp


def _make_async_mock_session(content: str) -> AsyncMock:
    """aiohttp.ClientSession-like mock whose ``post`` returns ``content``."""
    mock_resp = AsyncMock()
    mock_resp.status = 200

    async def mock_json() -> dict:
        return {"choices": [{"message": {"content": content}}]}

    mock_resp.json = mock_json
    mock_resp.raise_for_status = MagicMock()

    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_ctx = AsyncMock()

    async def mock_aenter(*_args: object) -> AsyncMock:
        return mock_resp

    mock_ctx.__aenter__ = mock_aenter
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_session.post = MagicMock(return_value=mock_ctx)
    return mock_session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cache_path(tmp_path: Path) -> str:
    """Provide an on-disk sqlite cache path under tmp_path."""
    return str(tmp_path / "threading_judge_cache.db")


@pytest.fixture
def client(cache_path: str) -> CompletionJudgeClient:
    """CompletionJudgeClient wired to a tmp cache, no rate limit."""
    return CompletionJudgeClient(
        model="test-model",
        base_url="https://test.example.com/v1",
        api_key="test-key",
        cache_path=cache_path,
        requests_per_minute=0,
        concurrency=8,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompletionJudgeThreading:
    """Direct regression coverage for the shared-sqlite threading bug."""

    def test_completion_judge_concurrent_threaded_dispatch(
        self, client: CompletionJudgeClient, cache_path: str
    ) -> None:
        """64 threads, identical payload: no ProgrammingError, one cached row.

        The first API call populates the cache. All subsequent calls MUST
        hit the cache (same key). We count cache rows via an independent
        sqlite connection from the main test thread, so there is no risk
        of reusing the client's own connection.
        """
        api_content = "coherence: 4\nfeature_alignment: 3"
        errors: list[BaseException] = []

        def _run_one() -> JudgementScore | BaseException:
            try:
                return client.judge(
                    context="The cat sat on the mat.",
                    completion="Then the dog arrived.",
                    feature_description="animal references",
                    ground_truth="Then the dog came in.",
                )
            except BaseException as exc:  # capture cross-thread errors
                errors.append(exc)
                return exc

        mock_resp = _make_sync_mock_response(api_content)
        with (
            patch(
                "steering.eval.completion_judge._requests.post",
                return_value=mock_resp,
            ),
            ThreadPoolExecutor(max_workers=8) as pool,
        ):
            futures = [pool.submit(_run_one) for _ in range(64)]
            results = [f.result() for f in as_completed(futures)]

        # No sqlite3.ProgrammingError from any worker.
        programming_errors = [
            e for e in errors if isinstance(e, sqlite3.ProgrammingError)
        ]
        assert not programming_errors, (
            f"Got {len(programming_errors)} sqlite3.ProgrammingError: "
            f"{programming_errors[:3]}"
        )
        assert not errors, f"Unexpected worker errors: {errors[:3]}"

        # All 64 calls produced a valid JudgementScore, and they all agree.
        assert len(results) == 64
        assert all(isinstance(r, JudgementScore) for r in results)
        first = results[0]
        assert isinstance(first, JudgementScore)
        assert all(r == first for r in results if isinstance(r, JudgementScore))
        assert first.coherence == 4
        assert first.feature_alignment == 3

        # Cache holds exactly one row — same key for identical input.
        assert _count_cache_rows(cache_path) == 1

    def test_completion_judge_concurrent_cache_writes(
        self, client: CompletionJudgeClient, cache_path: str
    ) -> None:
        """16 unique payloads concurrently → 16 cache rows, no crashes."""
        api_content = "coherence: 3\nfeature_alignment: 5"
        mock_resp = _make_sync_mock_response(api_content)
        errors: list[BaseException] = []

        items: list[dict[str, str]] = [
            {
                "context": f"ctx_{i}",
                "completion": f"comp_{i}",
                "feature_description": f"desc_{i}",
                "ground_truth": f"gt_{i}",
            }
            for i in range(16)
        ]

        def _run_one(item: dict[str, str]) -> JudgementScore | BaseException:
            try:
                return client.judge(
                    context=item["context"],
                    completion=item["completion"],
                    feature_description=item["feature_description"],
                    ground_truth=item["ground_truth"],
                )
            except BaseException as exc:
                errors.append(exc)
                return exc

        with (
            patch(
                "steering.eval.completion_judge._requests.post",
                return_value=mock_resp,
            ),
            ThreadPoolExecutor(max_workers=16) as pool,
        ):
            futures = [pool.submit(_run_one, item) for item in items]
            results = [f.result() for f in as_completed(futures)]

        programming_errors = [
            e for e in errors if isinstance(e, sqlite3.ProgrammingError)
        ]
        assert not programming_errors, (
            f"Got {len(programming_errors)} sqlite3.ProgrammingError: "
            f"{programming_errors[:3]}"
        )
        assert not errors, f"Unexpected worker errors: {errors[:3]}"
        assert len(results) == 16
        assert all(isinstance(r, JudgementScore) for r in results)

        # All 16 unique payloads produced 16 distinct cache keys.
        assert _count_cache_rows(cache_path) == 16
        keys = _fetch_all_cache_keys(cache_path)
        assert len(set(keys)) == 16

    def test_completion_judge_async_dispatch(
        self, client: CompletionJudgeClient, cache_path: str
    ) -> None:
        """asyncio.gather over 16 unique payloads — no errors, 16 rows."""
        api_content = "coherence: 5\nfeature_alignment: 2"
        items: list[dict[str, str]] = [
            {
                "context": f"async_ctx_{i}",
                "completion": f"async_comp_{i}",
                "feature_description": f"async_desc_{i}",
                "ground_truth": f"async_gt_{i}",
            }
            for i in range(16)
        ]

        async def _run() -> list[JudgementScore]:
            session = _make_async_mock_session(api_content)
            sem = asyncio.Semaphore(8)
            tasks = [
                client.judge_async(
                    session=session,
                    semaphore=sem,
                    context=item["context"],
                    completion=item["completion"],
                    feature_description=item["feature_description"],
                    ground_truth=item["ground_truth"],
                )
                for item in items
            ]
            return list(await asyncio.gather(*tasks))

        results = asyncio.run(_run())
        assert len(results) == 16
        assert all(isinstance(r, JudgementScore) for r in results)
        assert all(r.coherence == 5 for r in results)
        assert all(r.feature_alignment == 2 for r in results)

        assert _count_cache_rows(cache_path) == 16

    def test_completion_judge_mixed_sync_async(
        self, client: CompletionJudgeClient, cache_path: str
    ) -> None:
        """Threaded judge() + asyncio judge_async() over disjoint payloads.

        We build 8 unique sync payloads and 8 unique async payloads (16
        total distinct keys). The sync half runs in a ThreadPoolExecutor,
        the async half via asyncio.gather. Both are dispatched at the
        same time — the asyncio loop itself runs on a freshly-created
        event-loop thread so we exercise the "event loop on non-creating
        thread" scenario that also broke with check_same_thread=True.
        """
        sync_items: list[dict[str, str]] = [
            {
                "context": f"sync_ctx_{i}",
                "completion": f"sync_comp_{i}",
                "feature_description": f"sync_desc_{i}",
                "ground_truth": f"sync_gt_{i}",
            }
            for i in range(8)
        ]
        async_items: list[dict[str, str]] = [
            {
                "context": f"async_ctx_{i}",
                "completion": f"async_comp_{i}",
                "feature_description": f"async_desc_{i}",
                "ground_truth": f"async_gt_{i}",
            }
            for i in range(8)
        ]

        api_content = "coherence: 2\nfeature_alignment: 4"
        sync_mock = _make_sync_mock_response(api_content)

        sync_errors: list[BaseException] = []
        async_errors: list[BaseException] = []
        async_results: list[JudgementScore] = []
        sync_results: list[JudgementScore] = []

        def _sync_worker(item: dict[str, str]) -> None:
            try:
                sync_results.append(
                    client.judge(
                        context=item["context"],
                        completion=item["completion"],
                        feature_description=item["feature_description"],
                        ground_truth=item["ground_truth"],
                    )
                )
            except BaseException as exc:
                sync_errors.append(exc)

        async def _async_all() -> None:
            session = _make_async_mock_session(api_content)
            sem = asyncio.Semaphore(8)
            try:
                tasks = [
                    client.judge_async(
                        session=session,
                        semaphore=sem,
                        context=item["context"],
                        completion=item["completion"],
                        feature_description=item["feature_description"],
                        ground_truth=item["ground_truth"],
                    )
                    for item in async_items
                ]
                out = await asyncio.gather(*tasks)
                async_results.extend(out)
            except BaseException as exc:
                async_errors.append(exc)

        def _async_thread() -> None:
            asyncio.run(_async_all())

        with patch(
            "steering.eval.completion_judge._requests.post",
            return_value=sync_mock,
        ):
            async_t = threading.Thread(target=_async_thread, daemon=True)
            async_t.start()
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(_sync_worker, item) for item in sync_items]
                for f in as_completed(futures):
                    f.result()
            async_t.join(timeout=30)

        prog_errs_sync = [
            e for e in sync_errors if isinstance(e, sqlite3.ProgrammingError)
        ]
        prog_errs_async = [
            e for e in async_errors if isinstance(e, sqlite3.ProgrammingError)
        ]
        assert not prog_errs_sync, f"sync ProgrammingError: {prog_errs_sync[:3]}"
        assert not prog_errs_async, f"async ProgrammingError: {prog_errs_async[:3]}"
        assert not sync_errors, f"sync errors: {sync_errors[:3]}"
        assert not async_errors, f"async errors: {async_errors[:3]}"

        assert len(sync_results) == 8
        assert len(async_results) == 8

        # 8 unique sync + 8 unique async = 16 distinct cache keys.
        assert _count_cache_rows(cache_path) == 16
        keys = set(_fetch_all_cache_keys(cache_path))
        # Every expected payload should have produced a key.
        expected_keys: set[str] = set()

        def _collect(items_iter: Iterable[dict[str, str]]) -> None:
            for item in items_iter:
                sys_p, usr_p = build_judge_prompt(
                    item["context"],
                    item["completion"],
                    item["feature_description"],
                    item["ground_truth"],
                )
                expected_keys.add(client._cache_key(f"{sys_p}\n---\n{usr_p}"))

        _collect(sync_items)
        _collect(async_items)
        assert expected_keys == keys
