"""Concurrency, retry, Retry-After, and cache-migration tests for
:class:`NeuronpediaClient`.

Split from ``test_neuronpedia_client.py`` so the per-call mocks stay
in one place and the multi-thread/multi-namespace contracts stay in
another. Both files patch the client's ``requests.get`` dependency.
"""

from __future__ import annotations

import multiprocessing
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from errors import DataLoadError
from steering.eval.neuronpedia_client import (
    NEURONPEDIA_API_KEY_ENV,
    NeuronpediaAuthError,
    NeuronpediaClient,
    NeuronpediaFetchError,
    NeuronpediaRateLimitError,
)
from steering.eval.neuronpedia_config import NeuronpediaConfig

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        json_data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.headers = headers or {}
        self.text = text or ""

    def json(self) -> dict[str, Any]:
        if self._json_data is None:
            raise ValueError("no json data")
        return self._json_data


def _ok(description: str) -> _FakeResponse:
    return _FakeResponse(
        200,
        json_data={
            "modelId": "gemma-2-2b",
            "layer": "8-gemmascope-res-16k",
            "index": "0",
            "explanations": [{"description": description}],
        },
    )


def _concurrent_cache_writer(
    cache_path: str,
    feature_id: int,
    description: str,
    barrier: Any,
    result_queue: Any,
) -> None:
    """Process target: start stale, synchronize, then save one entry."""
    try:
        client = NeuronpediaClient("model-a", "src-a", Path(cache_path))
        with client._cache_lock:
            client._cache[feature_id] = description
        barrier.wait(timeout=10)
        client._save_cache()
        result_queue.put(("ok", feature_id))
    except BaseException as exc:  # pragma: no cover - reported in parent
        result_queue.put(("error", repr(exc)))
        raise


def _conflicting_cache_writer(
    cache_path: str,
    description: str,
    barrier: Any,
    result_queue: Any,
) -> None:
    """Process target: race one feature and report the typed outcome."""
    try:
        client = NeuronpediaClient("model-a", "src-a", Path(cache_path))
        barrier.wait(timeout=10)
        client._save_cache({7: description})
        result_queue.put(("ok", description))
    except DataLoadError as exc:
        result_queue.put(("conflict", description, str(exc)))
    except BaseException as exc:  # pragma: no cover - reported in parent
        result_queue.put(("error", description, repr(exc)))
        raise


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(NEURONPEDIA_API_KEY_ENV, "np-test-key")


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the backoff sleep so tests run in milliseconds.

    Tests that need to inspect sleep durations (Retry-After tests)
    re-monkeypatch ``time.sleep`` inside the test body to capture
    instead of skip. The autouse fixture provides a safe default for
    every other test."""
    import steering.eval.neuronpedia_client as mod
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_get_descriptions_dispatches_concurrently(
        self, tmp_path: Path,
    ) -> None:
        """Verify multiple HTTP calls are in flight simultaneously."""
        max_in_flight = [0]
        in_flight = [0]
        lock = threading.Lock()
        cv = threading.Condition(lock)
        n_started = [0]

        def slow_get(url: str, *, headers: Any, timeout: Any) -> Any:
            with cv:
                in_flight[0] += 1
                max_in_flight[0] = max(max_in_flight[0], in_flight[0])
                n_started[0] += 1
                cv.notify_all()
                cv.wait_for(lambda: n_started[0] >= 3, timeout=2.0)
            fid = url.rsplit("/", 1)[-1]
            with lock:
                in_flight[0] -= 1
            return _ok(f"desc-{fid}")

        with patch(
            "steering.eval.neuronpedia_client.requests.get",
            side_effect=slow_get,
        ):
            client = NeuronpediaClient(
                "gemma-2-2b", "8-gemmascope-res-16k",
                tmp_path / "cache.yaml",
                config=NeuronpediaConfig(max_concurrent=5),
            )
            descs = client.get_descriptions([1, 2, 3, 4, 5])

        assert len(descs) == 5
        # The slow_get callback uses cv.wait_for(n_started >= 3), so by
        # construction the test guarantees >= 3 concurrent in-flight calls
        # if dispatch is parallel. A stronger lower bound than the original
        # >= 2 catches any future regression that drops concurrency to 2.
        assert max_in_flight[0] >= 3, (
            f"Expected at least 3 concurrent fetches "
            f"(slow_get cv.wait_for n_started >= 3), "
            f"saw max_in_flight={max_in_flight[0]}"
        )
        assert max_in_flight[0] <= 5, (
            f"Expected no more than 5 concurrent fetches "
            f"(max_workers=5), saw max_in_flight={max_in_flight[0]}"
        )

    def test_max_concurrent_respects_config(self, tmp_path: Path) -> None:
        """The concurrency cap must equal ``config.max_concurrent``."""
        max_in_flight = [0]
        in_flight = [0]
        lock = threading.Lock()

        def slow_get(url: str, *, headers: Any, timeout: Any) -> Any:
            with lock:
                in_flight[0] += 1
                max_in_flight[0] = max(max_in_flight[0], in_flight[0])
            time.sleep(0.005)  # bypass autouse stub by calling stdlib time
            with lock:
                in_flight[0] -= 1
            fid = url.rsplit("/", 1)[-1]
            return _ok(f"d-{fid}")

        with patch(
            "steering.eval.neuronpedia_client.requests.get",
            side_effect=slow_get,
        ):
            client = NeuronpediaClient(
                "gemma-2-2b", "8-gemmascope-res-16k",
                tmp_path / "cache.yaml",
                config=NeuronpediaConfig(max_concurrent=2),
            )
            client.get_descriptions(list(range(20)))

        assert max_in_flight[0] <= 2

    def test_save_cache_called_once_per_batch(self, tmp_path: Path) -> None:
        """``_save_cache`` is called exactly once per :meth:`get_descriptions`,
        regardless of how many features were fetched."""
        with patch("steering.eval.neuronpedia_client.requests.get") as mget:
            mget.return_value = _ok("desc")
            client = NeuronpediaClient(
                "gemma-2-2b", "8-gemmascope-res-16k",
                tmp_path / "cache.yaml",
                config=NeuronpediaConfig(max_concurrent=4),
            )
            with patch.object(
                client, "_save_cache", wraps=client._save_cache,
            ) as save_spy:
                client.get_descriptions(list(range(50)))
            assert save_spy.call_count == 1

    def test_save_cache_skipped_if_no_fetches(self, tmp_path: Path) -> None:
        """All-cached path makes ZERO save calls."""
        with patch("steering.eval.neuronpedia_client.requests.get") as mget:
            mget.return_value = _ok("desc")
            client = NeuronpediaClient(
                "gemma-2-2b", "8-gemmascope-res-16k",
                tmp_path / "cache.yaml",
            )
            client.get_descriptions([1, 2, 3])
        with patch("steering.eval.neuronpedia_client.requests.get") as mget2:
            with patch.object(
                client, "_save_cache", wraps=client._save_cache,
            ) as save_spy:
                client.get_descriptions([1, 2, 3])
            assert mget2.call_count == 0
            assert save_spy.call_count == 0

    def test_save_cache_called_once_in_partial_cache_path(
        self, tmp_path: Path,
    ) -> None:
        """Mixed-cache batch (some cached, some fetched) saves exactly once.

        The all-fresh path is covered by ``test_save_cache_called_once_per_batch``;
        the all-cached path is covered by ``test_save_cache_skipped_if_no_fetches``;
        this test covers the intermediate case where only NEW fetches trigger
        a save, even though the result includes both old and new entries."""
        with patch("steering.eval.neuronpedia_client.requests.get") as mget:
            mget.return_value = _ok("desc")
            client = NeuronpediaClient(
                "gemma-2-2b", "8-gemmascope-res-16k",
                tmp_path / "cache.yaml",
                config=NeuronpediaConfig(max_concurrent=4),
            )
            client.get_descriptions([1, 2, 3])  # warm 1-3
            with patch.object(
                client, "_save_cache", wraps=client._save_cache,
            ) as save_spy:
                result = client.get_descriptions([1, 2, 3, 4, 5])
            assert save_spy.call_count == 1
            assert set(result) == {1, 2, 3, 4, 5}

    def test_partial_cache_only_fetches_missing(self, tmp_path: Path) -> None:
        """When some features are cached, only the uncached ones hit the wire."""
        with patch("steering.eval.neuronpedia_client.requests.get") as mget:
            mget.return_value = _ok("desc")
            client = NeuronpediaClient(
                "gemma-2-2b", "8-gemmascope-res-16k",
                tmp_path / "cache.yaml",
            )
            client.get_descriptions([1, 2, 3])
            assert mget.call_count == 3
            mget.reset_mock()
            client.get_descriptions([2, 3, 4, 5])
            assert mget.call_count == 2  # only 4 and 5

    def test_duplicate_feature_ids_issue_one_fetch_and_one_result(
        self,
        tmp_path: Path,
    ) -> None:
        with patch("steering.eval.neuronpedia_client.requests.get") as mget:
            mget.return_value = _ok("deduplicated")
            client = NeuronpediaClient(
                "gemma-2-2b",
                "8-gemmascope-res-16k",
                tmp_path / "cache.yaml",
            )

            result = client.get_descriptions([7, 7, 7])

        assert result == {7: "deduplicated"}
        assert mget.call_count == 1

    def test_per_feature_failure_does_not_fail_batch(
        self, tmp_path: Path,
    ) -> None:
        """A failure for one feature must not abort the whole batch
        (preserves the historical concept_list.py contract)."""
        def maybe_500(url: str, *, headers: Any, timeout: Any) -> Any:
            fid = url.rsplit("/", 1)[-1]
            if fid == "2":
                return _FakeResponse(500, text="oops")
            return _ok(f"d-{fid}")

        with patch(
            "steering.eval.neuronpedia_client.requests.get",
            side_effect=maybe_500,
        ):
            client = NeuronpediaClient(
                "gemma-2-2b", "8-gemmascope-res-16k",
                tmp_path / "cache.yaml",
                config=NeuronpediaConfig(max_retries=1, max_concurrent=2),
            )
            descs = client.get_descriptions([1, 2, 3])
        assert set(descs) == {1, 3}


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


class TestRetry:
    def test_retries_on_connection_error_then_succeeds(
        self, tmp_path: Path,
    ) -> None:
        import requests as _rq
        attempts = [0]

        def flaky(url: str, *, headers: Any, timeout: Any) -> Any:
            attempts[0] += 1
            if attempts[0] <= 2:
                raise _rq.ConnectionError("kapow")
            return _ok("eventually-succeeded")

        with patch(
            "steering.eval.neuronpedia_client.requests.get",
            side_effect=flaky,
        ):
            client = NeuronpediaClient(
                "gemma-2-2b", "8-gemmascope-res-16k",
                tmp_path / "cache.yaml",
                config=NeuronpediaConfig(max_retries=3),
            )
            assert client.get_description(42) == "eventually-succeeded"
        assert attempts[0] == 3

    def test_retries_on_timeout(self, tmp_path: Path) -> None:
        import requests as _rq
        attempts = [0]

        def flaky(url: str, *, headers: Any, timeout: Any) -> Any:
            attempts[0] += 1
            if attempts[0] == 1:
                raise _rq.Timeout("slow")
            return _ok("ok")

        with patch(
            "steering.eval.neuronpedia_client.requests.get",
            side_effect=flaky,
        ):
            client = NeuronpediaClient(
                "gemma-2-2b", "8-gemmascope-res-16k",
                tmp_path / "cache.yaml",
                config=NeuronpediaConfig(max_retries=3),
            )
            assert client.get_description(42) == "ok"

    def test_retries_on_chunked_encoding_error(self, tmp_path: Path) -> None:
        """Real-world flaky-network failure mode: connection drops mid-response.
        ChunkedEncodingError is a RequestException but NOT a Timeout or
        ConnectionError, so a narrow retry clause would let it kill the batch.
        Verify it's caught and retried."""
        import requests as _rq
        attempts = [0]

        def flaky(url: str, *, headers: Any, timeout: Any) -> Any:
            attempts[0] += 1
            if attempts[0] == 1:
                raise _rq.exceptions.ChunkedEncodingError("connection broken")
            return _ok("recovered")

        with patch(
            "steering.eval.neuronpedia_client.requests.get",
            side_effect=flaky,
        ):
            client = NeuronpediaClient(
                "gemma-2-2b", "8-gemmascope-res-16k",
                tmp_path / "cache.yaml",
                config=NeuronpediaConfig(max_retries=3),
            )
            assert client.get_description(42) == "recovered"
        assert attempts[0] == 2

    def test_retries_on_ssl_error(self, tmp_path: Path) -> None:
        """SSL handshake failure (subclass of ConnectionError, but verify
        explicitly that it's retried, since SSL flakes are common)."""
        import requests as _rq
        attempts = [0]

        def flaky(url: str, *, headers: Any, timeout: Any) -> Any:
            attempts[0] += 1
            if attempts[0] == 1:
                raise _rq.exceptions.SSLError("handshake failed")
            return _ok("recovered")

        with patch(
            "steering.eval.neuronpedia_client.requests.get",
            side_effect=flaky,
        ):
            client = NeuronpediaClient(
                "gemma-2-2b", "8-gemmascope-res-16k",
                tmp_path / "cache.yaml",
                config=NeuronpediaConfig(max_retries=3),
            )
            assert client.get_description(42) == "recovered"

    def test_chunked_encoding_error_in_batch_drops_only_that_feature(
        self, tmp_path: Path,
    ) -> None:
        """A persistent ChunkedEncodingError on one feature drops just
        that feature, not the whole batch."""
        import requests as _rq

        def maybe_chunked(url: str, *, headers: Any, timeout: Any) -> Any:
            fid = url.rsplit("/", 1)[-1]
            if fid == "2":
                raise _rq.exceptions.ChunkedEncodingError("always broken")
            return _ok(f"d-{fid}")

        with patch(
            "steering.eval.neuronpedia_client.requests.get",
            side_effect=maybe_chunked,
        ):
            client = NeuronpediaClient(
                "gemma-2-2b", "8-gemmascope-res-16k",
                tmp_path / "cache.yaml",
                config=NeuronpediaConfig(max_retries=1, max_concurrent=2),
            )
            descs = client.get_descriptions([1, 2, 3])
        assert set(descs) == {1, 3}

    def test_retries_on_5xx_then_succeeds(self, tmp_path: Path) -> None:
        attempts = [0]

        def flaky(url: str, *, headers: Any, timeout: Any) -> Any:
            attempts[0] += 1
            if attempts[0] == 1:
                return _FakeResponse(503, text="bad gateway")
            if attempts[0] == 2:
                return _FakeResponse(502, text="bad gateway")
            return _ok("ok")

        with patch(
            "steering.eval.neuronpedia_client.requests.get",
            side_effect=flaky,
        ):
            client = NeuronpediaClient(
                "gemma-2-2b", "8-gemmascope-res-16k",
                tmp_path / "cache.yaml",
                config=NeuronpediaConfig(max_retries=3),
            )
            assert client.get_description(42) == "ok"
        assert attempts[0] == 3

    def test_retry_exhaustion_drops_feature_in_batch(
        self, tmp_path: Path,
    ) -> None:
        """Per-feature retry exhaustion must not fail the whole batch."""
        def always_500(url: str, *, headers: Any, timeout: Any) -> Any:
            fid = url.rsplit("/", 1)[-1]
            if fid == "2":
                return _FakeResponse(500, text="oops")
            return _ok(f"d-{fid}")

        with patch(
            "steering.eval.neuronpedia_client.requests.get",
            side_effect=always_500,
        ):
            client = NeuronpediaClient(
                "gemma-2-2b", "8-gemmascope-res-16k",
                tmp_path / "cache.yaml",
                config=NeuronpediaConfig(max_retries=2),
            )
            descs = client.get_descriptions([1, 2, 3])
        # Feature 2 is dropped; 1 and 3 succeed.
        assert set(descs) == {1, 3}

    def test_retry_exhaustion_singular_raises(self, tmp_path: Path) -> None:
        """:meth:`get_description` (singular) propagates the exception."""
        with patch(
            "steering.eval.neuronpedia_client.requests.get",
        ) as mget:
            mget.return_value = _FakeResponse(500, text="oops")
            client = NeuronpediaClient(
                "gemma-2-2b", "8-gemmascope-res-16k",
                tmp_path / "cache.yaml",
                config=NeuronpediaConfig(max_retries=2),
            )
            with pytest.raises(NeuronpediaFetchError, match="500"):
                client.get_description(42)

    def test_non_retryable_4xx_does_not_retry(self, tmp_path: Path) -> None:
        attempts = [0]

        def fail_400(url: str, *, headers: Any, timeout: Any) -> Any:
            attempts[0] += 1
            return _FakeResponse(400, text="bad request")

        with patch(
            "steering.eval.neuronpedia_client.requests.get",
            side_effect=fail_400,
        ):
            client = NeuronpediaClient(
                "gemma-2-2b", "8-gemmascope-res-16k",
                tmp_path / "cache.yaml",
                config=NeuronpediaConfig(max_retries=3),
            )
            with pytest.raises(NeuronpediaFetchError, match="400"):
                client.get_description(42)
        assert attempts[0] == 1


# ---------------------------------------------------------------------------
# Retry-After
# ---------------------------------------------------------------------------


class TestRetryAfter:
    def test_429_with_retry_after_honored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sleeps: list[float] = []

        import steering.eval.neuronpedia_client as mod
        monkeypatch.setattr(mod.time, "sleep", lambda s: sleeps.append(s))

        attempts = [0]

        def flaky_429(url: str, *, headers: Any, timeout: Any) -> Any:
            attempts[0] += 1
            if attempts[0] == 1:
                return _FakeResponse(
                    429, headers={"Retry-After": "5"}, text="too fast",
                )
            return _ok("ok")

        with patch(
            "steering.eval.neuronpedia_client.requests.get",
            side_effect=flaky_429,
        ):
            client = NeuronpediaClient(
                "gemma-2-2b", "8-gemmascope-res-16k",
                tmp_path / "cache.yaml",
                config=NeuronpediaConfig(
                    max_retries=3, backoff_max_seconds=60,
                ),
            )
            assert client.get_description(42) == "ok"
        # First (and only) sleep should be ~5s (the Retry-After value).
        assert sleeps, "expected at least one sleep"
        assert 4.5 <= sleeps[0] <= 5.5, sleeps

    def test_429_retry_after_capped_at_max(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A pathological ``Retry-After: 99999`` is capped at
        ``backoff_max_seconds`` so a server bug can't induce hour-long waits."""
        sleeps: list[float] = []

        import steering.eval.neuronpedia_client as mod
        monkeypatch.setattr(mod.time, "sleep", lambda s: sleeps.append(s))

        attempts = [0]

        def flaky(url: str, *, headers: Any, timeout: Any) -> Any:
            attempts[0] += 1
            if attempts[0] == 1:
                return _FakeResponse(429, headers={"Retry-After": "99999"})
            return _ok("ok")

        with patch(
            "steering.eval.neuronpedia_client.requests.get",
            side_effect=flaky,
        ):
            client = NeuronpediaClient(
                "gemma-2-2b", "8-gemmascope-res-16k",
                tmp_path / "cache.yaml",
                config=NeuronpediaConfig(
                    max_retries=3,
                    backoff_base_seconds=1.0,
                    backoff_max_seconds=10.0,
                ),
            )
            client.get_description(42)
        assert sleeps[0] == pytest.approx(10.0)

    def test_429_without_retry_after_uses_backoff(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sleeps: list[float] = []

        import steering.eval.neuronpedia_client as mod
        monkeypatch.setattr(mod.time, "sleep", lambda s: sleeps.append(s))

        attempts = [0]

        def flaky(url: str, *, headers: Any, timeout: Any) -> Any:
            attempts[0] += 1
            if attempts[0] == 1:
                return _FakeResponse(429, headers={})
            return _ok("ok")

        with patch(
            "steering.eval.neuronpedia_client.requests.get",
            side_effect=flaky,
        ):
            client = NeuronpediaClient(
                "gemma-2-2b", "8-gemmascope-res-16k",
                tmp_path / "cache.yaml",
                config=NeuronpediaConfig(
                    max_retries=3,
                    backoff_base_seconds=1.0,
                    backoff_max_seconds=10.0,
                ),
            )
            client.get_description(42)
        # Backoff base=1, jitter [0, 1), so first sleep in [1, 2).
        assert 1.0 <= sleeps[0] < 2.5, sleeps

    def test_429_nan_retry_after_uses_backoff(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A NaN Retry-After must NOT propagate to time.sleep (which would
        raise ValueError). Defensive guard tested explicitly."""
        sleeps: list[float] = []
        import steering.eval.neuronpedia_client as mod
        monkeypatch.setattr(mod.time, "sleep", lambda s: sleeps.append(s))

        attempts = [0]

        def flaky(url: str, *, headers: Any, timeout: Any) -> Any:
            attempts[0] += 1
            if attempts[0] == 1:
                return _FakeResponse(429, headers={"Retry-After": "nan"})
            return _ok("ok")

        with patch(
            "steering.eval.neuronpedia_client.requests.get",
            side_effect=flaky,
        ):
            client = NeuronpediaClient(
                "gemma-2-2b", "8-gemmascope-res-16k",
                tmp_path / "cache.yaml",
                config=NeuronpediaConfig(
                    max_retries=3,
                    backoff_base_seconds=1.0,
                    backoff_max_seconds=10.0,
                ),
            )
            client.get_description(42)
        # Falls back to backoff: jitter [0, 1), so first sleep in [1, 2.5).
        assert 1.0 <= sleeps[0] < 2.5

    def test_429_inf_retry_after_uses_backoff(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An infinite Retry-After is also rejected as 'no header' rather
        than capped, since it strongly signals a server bug."""
        sleeps: list[float] = []
        import steering.eval.neuronpedia_client as mod
        monkeypatch.setattr(mod.time, "sleep", lambda s: sleeps.append(s))

        attempts = [0]

        def flaky(url: str, *, headers: Any, timeout: Any) -> Any:
            attempts[0] += 1
            if attempts[0] == 1:
                return _FakeResponse(429, headers={"Retry-After": "inf"})
            return _ok("ok")

        with patch(
            "steering.eval.neuronpedia_client.requests.get",
            side_effect=flaky,
        ):
            client = NeuronpediaClient(
                "gemma-2-2b", "8-gemmascope-res-16k",
                tmp_path / "cache.yaml",
                config=NeuronpediaConfig(
                    max_retries=3,
                    backoff_base_seconds=1.0,
                    backoff_max_seconds=10.0,
                ),
            )
            client.get_description(42)
        assert 1.0 <= sleeps[0] < 2.5

    def test_429_invalid_retry_after_uses_backoff(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An ill-formed Retry-After (e.g. an HTTP-date) falls back to
        exponential backoff -- the date form is intentionally not honored."""
        sleeps: list[float] = []
        import steering.eval.neuronpedia_client as mod
        monkeypatch.setattr(mod.time, "sleep", lambda s: sleeps.append(s))

        attempts = [0]

        def flaky(url: str, *, headers: Any, timeout: Any) -> Any:
            attempts[0] += 1
            if attempts[0] == 1:
                return _FakeResponse(
                    429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"},
                )
            return _ok("ok")

        with patch(
            "steering.eval.neuronpedia_client.requests.get",
            side_effect=flaky,
        ):
            client = NeuronpediaClient(
                "gemma-2-2b", "8-gemmascope-res-16k",
                tmp_path / "cache.yaml",
                config=NeuronpediaConfig(
                    max_retries=3,
                    backoff_base_seconds=1.0,
                    backoff_max_seconds=10.0,
                ),
            )
            client.get_description(42)
        assert 1.0 <= sleeps[0] < 2.5

    def test_429_after_retries_raises_rate_limit_error(
        self, tmp_path: Path,
    ) -> None:
        with patch("steering.eval.neuronpedia_client.requests.get") as mget:
            mget.return_value = _FakeResponse(
                429, headers={"Retry-After": "2"}, text="rate limited",
            )
            client = NeuronpediaClient(
                "gemma-2-2b", "8-gemmascope-res-16k",
                tmp_path / "cache.yaml",
                config=NeuronpediaConfig(max_retries=2),
            )
            with pytest.raises(NeuronpediaRateLimitError) as exc_info:
                client.get_description(42)
        assert exc_info.value.retry_after_seconds == 2.0


# ---------------------------------------------------------------------------
# Cache key safety
# ---------------------------------------------------------------------------


class TestCacheKeyNamespacing:
    def test_two_clients_different_pairs_different_files_isolated(
        self, tmp_path: Path,
    ) -> None:
        """Two clients with different ``(model_id, sae_source)`` pointing
        at different cache files do not interfere."""
        cache_a = tmp_path / "a.yaml"
        cache_b = tmp_path / "b.yaml"

        with patch("steering.eval.neuronpedia_client.requests.get") as mget:
            mget.return_value = _ok("from-a")
            client_a = NeuronpediaClient("model-a", "src-a", cache_a)
            client_a.get_description(42)

            mget.return_value = _ok("from-b")
            client_b = NeuronpediaClient("model-b", "src-b", cache_b)
            client_b.get_description(42)

        assert client_a.get_cached(42) == "from-a"
        assert client_b.get_cached(42) == "from-b"

    def test_reusing_cache_for_different_pair_raises(
        self, tmp_path: Path,
    ) -> None:
        cache = tmp_path / "shared.yaml"
        with patch("steering.eval.neuronpedia_client.requests.get") as mget:
            mget.return_value = _ok("from-a")
            client_a = NeuronpediaClient("model-a", "src-a", cache)
            client_a.get_description(42)
        with pytest.raises(DataLoadError) as exc_info:
            NeuronpediaClient("model-b", "src-b", cache)
        msg = str(exc_info.value)
        assert "model-a" in msg
        assert "model-b" in msg
        assert "src-a" in msg
        assert "src-b" in msg

    def test_cache_serializes_namespace(self, tmp_path: Path) -> None:
        cache = tmp_path / "ns.yaml"
        with patch("steering.eval.neuronpedia_client.requests.get") as mget:
            mget.return_value = _ok("desc-1")
            client = NeuronpediaClient("model-a", "src-a", cache)
            client.get_description(1)
        with open(cache) as f:
            doc = yaml.safe_load(f)
        assert doc["model_id"] == "model-a"
        assert doc["sae_source"] == "src-a"
        assert doc["descriptions"] == {1: "desc-1"}

    def test_same_pair_reused_works(self, tmp_path: Path) -> None:
        """Same ``(model_id, sae_source)`` opening the same cache file
        is the warm-cache case and must work."""
        cache = tmp_path / "warm.yaml"
        with patch("steering.eval.neuronpedia_client.requests.get") as mget:
            mget.return_value = _ok("desc-x")
            c1 = NeuronpediaClient("model-a", "src-a", cache)
            c1.get_description(7)
        c2 = NeuronpediaClient("model-a", "src-a", cache)
        assert c2.get_cached(7) == "desc-x"

    def test_unicode_descriptions_round_trip_human_readable(
        self, tmp_path: Path,
    ) -> None:
        """Unicode descriptions are written verbatim (allow_unicode=True),
        so the cache file is human-readable, AND round-trip via load
        preserves the exact string."""
        cache = tmp_path / "ru.yaml"
        unicode_desc = "Descripción con árboles 🌳 y кириллица"
        with patch("steering.eval.neuronpedia_client.requests.get") as mget:
            mget.return_value = _ok(unicode_desc)
            c1 = NeuronpediaClient("model-a", "src-a", cache)
            c1.get_description(1)
        # File on disk must contain the literal Unicode characters,
        # not \uXXXX escapes.
        raw_text = cache.read_text(encoding="utf-8")
        assert "árboles" in raw_text
        assert "🌳" in raw_text
        assert "кириллица" in raw_text
        # And the round trip must preserve the description.
        c2 = NeuronpediaClient("model-a", "src-a", cache)
        assert c2.get_cached(1) == unicode_desc


# ---------------------------------------------------------------------------
# Process-safe cache persistence
# ---------------------------------------------------------------------------


class TestProcessSafeCachePersistence:
    @staticmethod
    def _put(
        client: NeuronpediaClient,
        feature_id: int,
        description: str,
    ) -> None:
        with client._cache_lock:
            client._cache[feature_id] = description

    def test_stale_clients_merge_disjoint_updates_without_loss(
        self,
        tmp_path: Path,
    ) -> None:
        """The second writer must reread disk after taking the file lock."""
        cache = tmp_path / "shared.yaml"
        first = NeuronpediaClient("model-a", "src-a", cache)
        second = NeuronpediaClient("model-a", "src-a", cache)
        assert first._cache == {}
        assert second._cache == {}

        self._put(first, 7, "description-seven")
        self._put(second, 42, "description-forty-two")
        first._save_cache()
        second._save_cache()

        reloaded = NeuronpediaClient("model-a", "src-a", cache)
        assert reloaded._cache == {
            7: "description-seven",
            42: "description-forty-two",
        }
        assert second._cache == reloaded._cache
        document = yaml.safe_load(cache.read_text(encoding="utf-8"))
        assert document == {
            "model_id": "model-a",
            "sae_source": "src-a",
            "descriptions": {
                7: "description-seven",
                42: "description-forty-two",
            },
        }

    def test_stale_clients_accept_identical_overlap_and_keep_union(
        self,
        tmp_path: Path,
    ) -> None:
        cache = tmp_path / "shared.yaml"
        first = NeuronpediaClient("model-a", "src-a", cache)
        second = NeuronpediaClient("model-a", "src-a", cache)

        self._put(first, 7, "stable-description")
        self._put(second, 7, "stable-description")
        self._put(second, 42, "second-only")
        first._save_cache()
        second._save_cache()

        assert NeuronpediaClient("model-a", "src-a", cache)._cache == {
            7: "stable-description",
            42: "second-only",
        }

    def test_stale_clients_reject_conflicting_overlap_without_corruption(
        self,
        tmp_path: Path,
    ) -> None:
        cache = tmp_path / "shared.yaml"
        first = NeuronpediaClient("model-a", "src-a", cache)
        second = NeuronpediaClient("model-a", "src-a", cache)
        self._put(first, 7, "accepted-first")
        self._put(second, 7, "conflicting-second")
        first._save_cache()
        bytes_before_conflict = cache.read_bytes()

        with pytest.raises(
            DataLoadError,
            match="different descriptions for the same feature IDs",
        ) as exc_info:
            second._save_cache()

        assert "accepted-first" in str(exc_info.value)
        assert "conflicting-second" in str(exc_info.value)
        assert cache.read_bytes() == bytes_before_conflict
        assert not cache.with_suffix(".yaml.tmp").exists()
        assert NeuronpediaClient("model-a", "src-a", cache)._cache == {7: "accepted-first"}

    def test_two_real_processes_preserve_simultaneous_disjoint_updates(
        self,
        tmp_path: Path,
    ) -> None:
        """Exercise the OS advisory lock, not merely two in-process clients."""
        cache = tmp_path / "process-shared.yaml"
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(2)
        result_queue = context.Queue()
        processes = [
            context.Process(
                target=_concurrent_cache_writer,
                args=(str(cache), 7, "from-process-seven", barrier, result_queue),
            ),
            context.Process(
                target=_concurrent_cache_writer,
                args=(str(cache), 42, "from-process-forty-two", barrier, result_queue),
            ),
        ]
        try:
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=15)
            for process in processes:
                assert not process.is_alive(), (
                    f"cache writer process {process.pid} failed to terminate"
                )
                assert process.exitcode == 0
            assert sorted(result_queue.get(timeout=2) for _ in processes) == [
                ("ok", 7),
                ("ok", 42),
            ]
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            result_queue.close()
            result_queue.join_thread()

        assert NeuronpediaClient("model-a", "src-a", cache)._cache == {
            7: "from-process-seven",
            42: "from-process-forty-two",
        }
        assert cache.with_suffix(".yaml.lock").is_file()
        assert not cache.with_suffix(".yaml.tmp").exists()

    def test_same_client_threads_reject_divergent_same_feature(
        self,
        tmp_path: Path,
    ) -> None:
        barrier = threading.Barrier(2)

        def divergent_get(
            url: str,
            *,
            headers: Any,
            timeout: Any,
        ) -> _FakeResponse:
            del url, headers, timeout
            description = f"from-{threading.current_thread().name}"
            barrier.wait(timeout=5)
            return _ok(description)

        client = NeuronpediaClient("model-a", "src-a", tmp_path / "threaded.yaml")
        outcomes: list[tuple[str, str]] = []
        outcomes_lock = threading.Lock()

        def worker() -> None:
            try:
                outcome = ("ok", client.get_description(7))
            except DataLoadError as exc:
                outcome = ("conflict", str(exc))
            with outcomes_lock:
                outcomes.append(outcome)

        with patch(
            "steering.eval.neuronpedia_client.requests.get",
            side_effect=divergent_get,
        ):
            threads = [
                threading.Thread(target=worker, name="writer-a"),
                threading.Thread(target=worker, name="writer-b"),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
                assert not thread.is_alive()

        assert [kind for kind, _value in outcomes].count("ok") == 1
        assert [kind for kind, _value in outcomes].count("conflict") == 1
        winner = next(value for kind, value in outcomes if kind == "ok")
        assert winner in {"from-writer-a", "from-writer-b"}
        assert client.get_cached(7) == winner
        assert NeuronpediaClient(
            "model-a",
            "src-a",
            tmp_path / "threaded.yaml",
        ).get_cached(7) == winner

    def test_two_real_processes_reject_simultaneous_conflicting_update(
        self,
        tmp_path: Path,
    ) -> None:
        cache = tmp_path / "process-conflict.yaml"
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(2)
        result_queue = context.Queue()
        descriptions = ("from-process-a", "from-process-b")
        processes = [
            context.Process(
                target=_conflicting_cache_writer,
                args=(str(cache), description, barrier, result_queue),
            )
            for description in descriptions
        ]
        try:
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=15)
                assert not process.is_alive()
                assert process.exitcode == 0
            outcomes = [result_queue.get(timeout=2) for _ in processes]
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            result_queue.close()
            result_queue.join_thread()

        assert [outcome[0] for outcome in outcomes].count("ok") == 1
        assert [outcome[0] for outcome in outcomes].count("conflict") == 1
        winner = next(outcome[1] for outcome in outcomes if outcome[0] == "ok")
        conflict = next(outcome for outcome in outcomes if outcome[0] == "conflict")
        assert "different descriptions for the same feature IDs" in conflict[2]
        assert NeuronpediaClient("model-a", "src-a", cache).get_cached(7) == winner
        assert not list(tmp_path.glob(".process-conflict.yaml.tmp.*"))

    @pytest.mark.parametrize("failure_point", ["dump", "file_fsync", "replace"])
    def test_persistence_failure_is_transactional_and_retryable(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        failure_point: str,
    ) -> None:
        import steering.eval.neuronpedia_client as mod

        cache = tmp_path / "transactional.yaml"
        client = NeuronpediaClient("model-a", "src-a", cache)
        client._save_cache({1: "committed-before-failure"})
        bytes_before = cache.read_bytes()

        with monkeypatch.context() as failure_patch:
            if failure_point == "dump":
                failure_patch.setattr(
                    mod.yaml,
                    "safe_dump",
                    lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        OSError("injected dump failure")
                    ),
                )
            elif failure_point == "file_fsync":
                failure_patch.setattr(
                    mod.os,
                    "fsync",
                    lambda _fd: (_ for _ in ()).throw(
                        OSError("injected file fsync failure")
                    ),
                )
            else:
                failure_patch.setattr(
                    mod.os,
                    "replace",
                    lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        OSError("injected replace failure")
                    ),
                )

            with pytest.raises(OSError, match="injected"):
                client._save_cache({2: "pending-must-not-publish"})

        assert cache.read_bytes() == bytes_before
        assert client.get_cached(1) == "committed-before-failure"
        assert client.get_cached(2) is None
        assert not list(tmp_path.glob(".transactional.yaml.tmp.*"))

        client._save_cache({2: "pending-must-not-publish"})
        assert client.get_cached(2) == "pending-must-not-publish"
        assert NeuronpediaClient("model-a", "src-a", cache)._cache == {
            1: "committed-before-failure",
            2: "pending-must-not-publish",
        }

    def test_parent_directory_fsync_failure_withholds_memory_until_retry(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A replaced-but-not-durably-linked file is not published in memory."""
        import steering.eval.neuronpedia_client as mod

        cache = tmp_path / "directory-fsync-failure.yaml"
        client = NeuronpediaClient("model-a", "src-a", cache)
        client._save_cache({1: "committed-before-failure"})
        real_fsync = mod.os.fsync
        fsync_calls = 0

        def fail_directory_fsync(fd: int) -> None:
            nonlocal fsync_calls
            fsync_calls += 1
            if fsync_calls == 2:
                raise OSError("injected directory fsync failure")
            real_fsync(fd)

        with monkeypatch.context() as failure_patch:
            failure_patch.setattr(mod.os, "fsync", fail_directory_fsync)
            with pytest.raises(
                OSError,
                match="injected directory fsync failure",
            ):
                client._save_cache({2: "replaced-but-not-published"})

        assert fsync_calls == 2
        assert client.get_cached(1) == "committed-before-failure"
        assert client.get_cached(2) is None
        assert NeuronpediaClient("model-a", "src-a", cache)._cache == {
            1: "committed-before-failure",
            2: "replaced-but-not-published",
        }

        client._save_cache({2: "replaced-but-not-published"})
        assert client.get_cached(2) == "replaced-but-not-published"
        assert not list(tmp_path.glob(".directory-fsync-failure.yaml.tmp.*"))

    def test_singular_fetch_never_exposes_a_failed_cache_commit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Exercise the transaction boundary through the public singular API."""
        import steering.eval.neuronpedia_client as mod

        cache = tmp_path / "singular-commit.yaml"
        client = NeuronpediaClient("model-a", "src-a", cache)
        responses = iter((_ok("first-fetch"), _ok("retry-fetch")))
        monkeypatch.setattr(
            mod.requests,
            "get",
            lambda *_args, **_kwargs: next(responses),
        )
        real_replace = mod.os.replace
        replace_attempts = 0

        def fail_first_replace(source: Path, destination: Path) -> None:
            nonlocal replace_attempts
            replace_attempts += 1
            if replace_attempts == 1:
                raise OSError("injected singular commit failure")
            real_replace(source, destination)

        monkeypatch.setattr(mod.os, "replace", fail_first_replace)

        with pytest.raises(OSError, match="injected singular commit failure"):
            client.get_description(7)

        assert client.get_cached(7) is None
        assert not cache.exists()
        assert not list(tmp_path.glob(".singular-commit.yaml.tmp.*"))

        assert client.get_description(7) == "retry-fetch"
        assert client.get_cached(7) == "retry-fetch"
        assert NeuronpediaClient("model-a", "src-a", cache).get_cached(7) == (
            "retry-fetch"
        )
        assert replace_attempts == 2

    def test_original_persistence_error_survives_temp_cleanup_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cleanup is best-effort and must not mask the causal write error."""
        import steering.eval.neuronpedia_client as mod

        cache = tmp_path / "cleanup-failure.yaml"
        client = NeuronpediaClient("model-a", "src-a", cache)
        monkeypatch.setattr(
            mod.yaml,
            "safe_dump",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("causal dump failure")
            ),
        )
        monkeypatch.setattr(
            Path,
            "unlink",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("secondary cleanup failure")
            ),
        )

        with pytest.raises(RuntimeError, match="causal dump failure"):
            client._save_cache({7: "never committed"})

        assert client.get_cached(7) is None
        assert not cache.exists()
        assert len(list(tmp_path.glob(".cleanup-failure.yaml.tmp.*"))) == 1

    def test_conflicting_loser_reconciles_memory_to_committed_disk(
        self,
        tmp_path: Path,
    ) -> None:
        cache = tmp_path / "reconcile.yaml"
        winner = NeuronpediaClient("model-a", "src-a", cache)
        loser = NeuronpediaClient("model-a", "src-a", cache)
        winner._save_cache({7: "disk-winner"})

        with pytest.raises(DataLoadError, match="different descriptions"):
            loser._save_cache({7: "rejected-loser"})

        assert loser.get_cached(7) == "disk-winner"
        assert loser._cache == NeuronpediaClient("model-a", "src-a", cache)._cache

    def test_file_replace_and_parent_directory_are_fsynced_in_order(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import steering.eval.neuronpedia_client as mod

        real_fsync = mod.os.fsync
        real_open = mod.os.open
        real_replace = mod.os.replace
        events: list[str] = []
        replaced = False

        def recording_fsync(fd: int) -> None:
            events.append("directory_fsync" if replaced else "file_fsync")
            real_fsync(fd)

        def recording_replace(src: Any, dst: Any) -> None:
            nonlocal replaced
            events.append("replace")
            real_replace(src, dst)
            replaced = True

        def recording_open(path: Any, flags: int, *args: Any) -> int:
            if flags & mod.os.O_DIRECTORY:
                events.append("directory_open")
            return real_open(path, flags, *args)

        monkeypatch.setattr(mod.os, "fsync", recording_fsync)
        monkeypatch.setattr(mod.os, "replace", recording_replace)
        monkeypatch.setattr(mod.os, "open", recording_open)

        client = NeuronpediaClient("model-a", "src-a", tmp_path / "durable.yaml")
        client._save_cache({7: "durable-description"})

        assert events == [
            "file_fsync",
            "replace",
            "directory_open",
            "directory_fsync",
        ]


# ---------------------------------------------------------------------------
# Legacy cache migration
# ---------------------------------------------------------------------------


class TestLegacyCacheMigration:
    def test_legacy_flat_cache_ignored_not_adopted(self, tmp_path: Path) -> None:
        # A legacy (non-namespaced) cache carries no model_id/sae_source, so it
        # cannot be verified against the current (model, source). It is
        # discarded (not adopted) -- so descriptions are re-fetched under the
        # current source and the source-mismatch guard still runs (issue #657)
        # -- rather than silently reused or hard-crashing the client.
        cache = tmp_path / "legacy.yaml"
        cache.write_text("42: 'legacy desc'\n100: 'another'\n")
        client = NeuronpediaClient("model-a", "src-a", cache)
        assert client.get_cached(42) is None
        assert client.get_cached(100) is None

    def test_legacy_cache_ignore_warns_and_names_file(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        from loguru import logger as _logger

        handler_id = _logger.add(caplog.handler, format="{message}")
        try:
            cache = tmp_path / "legacy.yaml"
            cache.write_text("1: 'a'\n")
            NeuronpediaClient("model-a", "src-a", cache)
        finally:
            _logger.remove(handler_id)
        text = " ".join(rec.message for rec in caplog.records)
        assert str(cache) in text
        assert "legacy" in text.lower()

    def test_corrupted_yaml_raises(self, tmp_path: Path) -> None:
        cache = tmp_path / "bad.yaml"
        cache.write_text("not: valid: yaml: at: all: ::: ::")
        with pytest.raises(DataLoadError):
            NeuronpediaClient("model-a", "src-a", cache)

    def test_unexpected_top_level_type_raises(self, tmp_path: Path) -> None:
        cache = tmp_path / "list.yaml"
        cache.write_text("- 1\n- 2\n")
        with pytest.raises(DataLoadError, match="dict"):
            NeuronpediaClient("model-a", "src-a", cache)

    def test_unexpected_descriptions_type_raises(
        self, tmp_path: Path,
    ) -> None:
        """A new-format cache with the right top-level keys but a
        non-dict ``descriptions`` value is rejected with a clear error."""
        cache = tmp_path / "bad_descriptions.yaml"
        cache.write_text(
            "model_id: model-a\n"
            "sae_source: src-a\n"
            "descriptions:\n"
            "  - one\n"
            "  - two\n"
        )
        with pytest.raises(DataLoadError, match="descriptions"):
            NeuronpediaClient("model-a", "src-a", cache)

    @pytest.mark.parametrize(
        "descriptions",
        [
            {"7": "string feature ID"},
            {2.5: "float feature ID"},
            {True: "boolean feature ID"},
            {-1: "negative feature ID"},
            {7: 42},
            {7: ["list description"]},
            {7: None},
            {7: ""},
            {7: "   "},
        ],
    )
    def test_invalid_description_entries_fail_without_coercion(
        self,
        tmp_path: Path,
        descriptions: dict[Any, Any],
    ) -> None:
        cache = tmp_path / "invalid-entry.yaml"
        cache.write_text(
            yaml.safe_dump(
                {
                    "model_id": "model-a",
                    "sae_source": "src-a",
                    "descriptions": descriptions,
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(DataLoadError, match="invalid"):
            NeuronpediaClient("model-a", "src-a", cache)

    def test_falsey_non_mapping_descriptions_is_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        cache = tmp_path / "empty-list.yaml"
        cache.write_text(
            "model_id: model-a\nsae_source: src-a\ndescriptions: []\n",
            encoding="utf-8",
        )

        with pytest.raises(DataLoadError, match="descriptions.*list"):
            NeuronpediaClient("model-a", "src-a", cache)

    def test_namespaced_document_requires_descriptions_key(
        self,
        tmp_path: Path,
    ) -> None:
        cache = tmp_path / "missing-descriptions.yaml"
        cache.write_text(
            "model_id: model-a\nsae_source: src-a\n",
            encoding="utf-8",
        )

        with pytest.raises(DataLoadError, match="missing required.*descriptions"):
            NeuronpediaClient("model-a", "src-a", cache)

    def test_symlinked_cache_path_is_rejected_without_mutating_target(
        self,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "target.yaml"
        target.write_text(
            "model_id: model-a\n"
            "sae_source: src-a\n"
            "descriptions:\n"
            "  7: target-description\n",
            encoding="utf-8",
        )
        bytes_before = target.read_bytes()
        link = tmp_path / "canonical.yaml"
        link.symlink_to(target)

        with pytest.raises(DataLoadError, match="symlinked"):
            NeuronpediaClient("model-a", "src-a", link)

        assert link.is_symlink()
        assert target.read_bytes() == bytes_before

    def test_empty_cache_file_loads_empty(self, tmp_path: Path) -> None:
        cache = tmp_path / "empty.yaml"
        cache.write_text("")
        client = NeuronpediaClient("model-a", "src-a", cache)
        assert client.get_cached(42) is None

    def test_missing_cache_file_loads_empty(self, tmp_path: Path) -> None:
        client = NeuronpediaClient(
            "model-a", "src-a", tmp_path / "absent.yaml",
        )
        assert client.get_cached(42) is None

    def test_unreadable_cache_file_raises_data_load_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If ``open()`` raises OSError (e.g. perms changed between is_file
        check and read), a DataLoadError is raised with a clear message
        rather than the raw OSError."""
        cache = tmp_path / "x.yaml"
        cache.write_text("a: b\n")

        def fake_open(*args: Any, **kwargs: Any) -> Any:
            raise PermissionError("simulated denied")

        monkeypatch.setattr(
            "steering.eval.neuronpedia_client.open", fake_open, raising=False,
        )
        # is_file() succeeds (file exists), then open() raises.
        with pytest.raises(DataLoadError, match="Could not read"):
            NeuronpediaClient("model-a", "src-a", cache)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class TestAuth:
    def test_401_raises_auth_error_no_retry(
        self, tmp_path: Path,
    ) -> None:
        attempts = [0]

        def fail_401(url: str, *, headers: Any, timeout: Any) -> Any:
            attempts[0] += 1
            return _FakeResponse(401, text="bad key")

        with patch(
            "steering.eval.neuronpedia_client.requests.get",
            side_effect=fail_401,
        ):
            client = NeuronpediaClient(
                "gemma-2-2b", "8-gemmascope-res-16k",
                tmp_path / "cache.yaml",
                config=NeuronpediaConfig(max_retries=3),
            )
            with pytest.raises(NeuronpediaAuthError):
                client.get_description(42)
        assert attempts[0] == 1


# ---------------------------------------------------------------------------
# Logging cadence
# ---------------------------------------------------------------------------


class TestLoggingCadence:
    def test_does_not_log_per_fetch_when_batch_large(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Old code logged per-fetch; new code logs every ~progress_step.

        A 200-feature batch should produce <= ~25 progress lines, not 200.
        """
        from loguru import logger as _logger
        handler_id = _logger.add(
            caplog.handler, level="INFO", format="{message}",
        )
        try:
            with patch("steering.eval.neuronpedia_client.requests.get") as mget:
                mget.return_value = _ok("desc")
                client = NeuronpediaClient(
                    "gemma-2-2b", "8-gemmascope-res-16k",
                    tmp_path / "cache.yaml",
                    config=NeuronpediaConfig(max_concurrent=4),
                )
                client.get_descriptions(list(range(200)))
        finally:
            _logger.remove(handler_id)
        n_fetched = sum(
            1 for rec in caplog.records
            if "fetched" in rec.message and "/" in rec.message
        )
        assert n_fetched <= 25, f"too noisy: {n_fetched} progress lines"
