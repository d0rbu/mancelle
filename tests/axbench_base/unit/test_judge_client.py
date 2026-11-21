"""Unit tests for ``AxBenchJudgeClient`` (Task 33).

Covers cache hit/miss, parse-retry loop, API backoff/retries, rate-limiting,
error mapping, unicode handling, NULL-cache-sentinel round-trip, and
lifecycle (close + context manager).

All network access is mocked via ``monkeypatch`` on
``requests.Session.post`` so the tests never talk to a real OpenRouter
endpoint and do not require ``OPENROUTER_API_KEY``.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
import requests

from steering.eval.axbench_base import judge as judge_module
from steering.eval.axbench_base.config import JudgeConfig
from steering.eval.axbench_base.judge import (
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_CHAT_COMPLETIONS_URL,
    AxBenchJudgeClient,
    RateLimiter,
    _JudgeCacheEntry,
    _make_cache_key,
)
from steering.eval.axbench_base.profiler import AxBenchProfiler
from steering.eval.axbench_base.types import AxBenchBaseError
from tests.runtime_attribute_helpers import set_runtime_attribute

# ---------------------------------------------------------------------------
# Test helpers: fake Response + monkeypatch stubs
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for ``requests.Response``.

    Mirrors the methods the client touches: ``status_code``, ``json()``,
    ``raise_for_status()``, and ``text``.
    """

    def __init__(
        self,
        status_code: int,
        json_data: Any | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self) -> Any:
        if self._json_data is None:
            raise ValueError("no json")
        return self._json_data

    def raise_for_status(self) -> None:
        if 400 <= self.status_code < 600:
            err = requests.HTTPError(
                f"{self.status_code} error", response=self,
            )
            raise err


class _PostStub:
    """Configurable stub for ``requests.Session.post``.

    Each call pops the next item from ``responses``. If the item is an
    Exception instance, it is raised instead of returned.
    """

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        url: str,
        *,
        json: Any,
        headers: dict[str, str],
        timeout: Any,
    ) -> _FakeResponse:
        self.calls.append(
            {"url": url, "json": json, "headers": headers, "timeout": timeout},
        )
        if not self.responses:
            raise AssertionError(
                "_PostStub: unexpected extra call (responses exhausted)",
            )
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _rating_response(score: int) -> _FakeResponse:
    return _FakeResponse(
        200,
        json_data={
            "choices": [
                {"message": {"content": f"Explanation. Rating: [[{score}]]"}},
            ],
        },
    )


def _bad_format_response(text: str = "no rating here") -> _FakeResponse:
    return _FakeResponse(
        200,
        json_data={"choices": [{"message": {"content": text}}]},
    )


def _null_content_response() -> _FakeResponse:
    return _FakeResponse(
        200,
        json_data={"choices": [{"message": {"content": None}}]},
    )


def _stub_post(monkeypatch: pytest.MonkeyPatch, stub: _PostStub) -> None:
    monkeypatch.setattr(requests.Session, "post", stub)


def _noop_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(judge_module.time, "sleep", lambda _: None)


def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "sk-test-xyz")


def _mk_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    cfg_kwargs: dict[str, Any] | None = None,
    cache_name: str = "j.sqlite",
) -> AxBenchJudgeClient:
    _set_api_key(monkeypatch)
    _noop_sleep(monkeypatch)
    kwargs: dict[str, Any] = {}
    if cfg_kwargs:
        kwargs.update(cfg_kwargs)
    cfg = JudgeConfig(**kwargs)
    return AxBenchJudgeClient(cfg, cache_path=tmp_path / cache_name)


# ===========================================================================
# Cache key helper
# ===========================================================================


def test_make_cache_key_is_deterministic_sha256() -> None:
    """_make_cache_key returns a 64-char hex SHA256 digest, deterministic."""
    k1 = _make_cache_key("m1", "prompt A")
    k2 = _make_cache_key("m1", "prompt A")
    assert k1 == k2
    assert len(k1) == 64
    assert all(c in "0123456789abcdef" for c in k1)


def test_make_cache_key_sensitive_to_model_and_prompt() -> None:
    """Different model or prompt → different key."""
    k_model_a = _make_cache_key("model-a", "same prompt")
    k_model_b = _make_cache_key("model-b", "same prompt")
    k_prompt_x = _make_cache_key("model-a", "other prompt")
    assert k_model_a != k_model_b
    assert k_model_a != k_prompt_x


def test_cache_key_distinct_for_different_models() -> None:
    """Same prompt, two JudgeConfig with different models → different keys."""
    key_a = _make_cache_key("openai/gpt-5-nano", "prompt")
    key_b = _make_cache_key("openai/gpt-4o-mini", "prompt")
    assert key_a != key_b


def test_two_clients_different_models_isolated_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Two clients sharing a cache file but with different models don't
    collide: each writes to its own row and reads back its own value."""
    _set_api_key(monkeypatch)
    _noop_sleep(monkeypatch)
    stub = _PostStub([_rating_response(1), _rating_response(2)])
    _stub_post(monkeypatch, stub)
    cache_path = tmp_path / "shared.sqlite"
    client_a = AxBenchJudgeClient(
        JudgeConfig(model="model-a"), cache_path=cache_path,
    )
    client_b = AxBenchJudgeClient(
        JudgeConfig(model="model-b"), cache_path=cache_path,
    )
    try:
        score_a, _ = client_a.score("common prompt")
        score_b, _ = client_b.score("common prompt")
        assert score_a == 1
        assert score_b == 2
        # Both rows exist; no collision.
        row_a = client_a._cache_get(_make_cache_key("model-a", "common prompt"))
        row_b = client_b._cache_get(_make_cache_key("model-b", "common prompt"))
        assert row_a is not None and row_a.score == 1
        assert row_b is not None and row_b.score == 2
    finally:
        client_a.close()
        client_b.close()


# ===========================================================================
# Construction
# ===========================================================================


def test_construction_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing OPENROUTER_API_KEY → construction raises AxBenchBaseError."""
    monkeypatch.delenv(OPENROUTER_API_KEY_ENV, raising=False)
    with pytest.raises(AxBenchBaseError, match=OPENROUTER_API_KEY_ENV):
        AxBenchJudgeClient(JudgeConfig(), cache_path=tmp_path / "j.sqlite")


def test_cache_path_parent_created(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The cache's parent directory is created if it does not exist."""
    _set_api_key(monkeypatch)
    nested = tmp_path / "nested" / "deeper" / "j.sqlite"
    client = AxBenchJudgeClient(JudgeConfig(), cache_path=nested)
    try:
        assert nested.parent.exists()
        assert nested.exists()  # SQLite creates the file on connect
    finally:
        client.close()


def test_corrupt_cache_file_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A file containing garbage bytes must raise AxBenchBaseError on open."""
    _set_api_key(monkeypatch)
    path = tmp_path / "corrupt.sqlite"
    path.write_bytes(b"this is not a valid sqlite db file " * 64)
    with pytest.raises(AxBenchBaseError):
        AxBenchJudgeClient(JudgeConfig(), cache_path=path)


# ===========================================================================
# Cache hit path
# ===========================================================================


def test_cache_hit_returns_cached_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A populated cache returns the stored score with zero API calls."""
    client = _mk_client(monkeypatch, tmp_path)
    try:
        prompt = "judge this thing please"
        # Pre-populate the cache: model matches cfg.model default
        key = _make_cache_key(client._cfg.model, prompt)
        client._cache_put(
            key=key, score=2, raw_response="Rating: [[2]]", attempts=1,
        )
        # Any call to requests.post would blow up the test.
        stub = _PostStub([])
        _stub_post(monkeypatch, stub)
        score, attempts = client.score(prompt)
        assert score == 2
        assert attempts == 1
        assert stub.calls == []  # API never hit
    finally:
        client.close()


def test_score_returns_none_for_cached_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A prior exhausted-retries run stored as NULL score → (None, 1), no API."""
    client = _mk_client(monkeypatch, tmp_path)
    try:
        prompt = "previously-exhausted prompt"
        key = _make_cache_key(client._cfg.model, prompt)
        client._cache_put(
            key=key,
            score=None,  # NULL sentinel: "we tried and gave up"
            raw_response="garbage from last run",
            attempts=2,
        )
        stub = _PostStub([])
        _stub_post(monkeypatch, stub)
        score, attempts = client.score(prompt)
        assert score is None
        assert attempts == 1
        assert stub.calls == []
    finally:
        client.close()


def test_score_zero_round_trips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A cached score of 0 must NOT collapse to None anywhere in the path."""
    client = _mk_client(monkeypatch, tmp_path)
    try:
        prompt = "zero-scoring judge case"
        stub = _PostStub([_rating_response(0)])
        _stub_post(monkeypatch, stub)
        score, attempts = client.score(prompt)
        assert score == 0
        assert attempts == 1

        # Second call hits the cache without touching the network.
        score2, attempts2 = client.score(prompt)
        assert score2 == 0
        assert attempts2 == 1
        assert len(stub.calls) == 1  # first call only
    finally:
        client.close()


# ===========================================================================
# Cache miss → single successful API call
# ===========================================================================


def test_cache_miss_calls_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A cache miss hits the API exactly once and caches the parsed result."""
    client = _mk_client(monkeypatch, tmp_path)
    try:
        prompt = "first call prompt"
        stub = _PostStub([_rating_response(1)])
        _stub_post(monkeypatch, stub)
        score, attempts = client.score(prompt)
        assert score == 1
        assert attempts == 1
        assert len(stub.calls) == 1
        call = stub.calls[0]
        assert call["url"] == OPENROUTER_CHAT_COMPLETIONS_URL
        assert call["headers"]["Authorization"] == "Bearer sk-test-xyz"
        assert call["headers"]["Content-Type"] == "application/json"
        assert call["timeout"] == (10, 60)
        assert call["json"]["model"] == client._cfg.model
        assert call["json"]["messages"] == [
            {"role": "user", "content": prompt},
        ]
        # Confirm the row was written under the expected key.
        key = _make_cache_key(client._cfg.model, prompt)
        row = client._cache_get(key)
        assert row is not None
        assert row.score == 1
        assert row.raw_response == "Explanation. Rating: [[1]]"
        assert row.attempts == 1
    finally:
        client.close()


def test_cache_get_returns_typed_entry_with_dataclass_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """_cache_get returns a frozen _JudgeCacheEntry dataclass, not a dict.

    Exercises every accessor on the dataclass (``score``, ``raw_response``,
    ``attempts``, ``timestamp``) so the type replaces the old
    ``dict[str, Any]`` return type cleanly. Also confirms immutability —
    attempting to set a field must raise ``dataclasses.FrozenInstanceError``.
    """
    client = _mk_client(monkeypatch, tmp_path)
    try:
        key = _make_cache_key(client._cfg.model, "typed-entry-prompt")
        before = time.time()
        client._cache_put(
            key=key, score=3, raw_response="Rating: [[3]]", attempts=2,
        )
        after = time.time()
        entry = client._cache_get(key)
        assert entry is not None
        assert isinstance(entry, _JudgeCacheEntry)
        # Typed field access (not dict-style).
        assert entry.score == 3
        assert entry.raw_response == "Rating: [[3]]"
        assert entry.attempts == 2
        assert before <= entry.timestamp <= after
        # Frozen dataclass: mutation must fail. Use the runtime mutation seam to bypass
        # the static read-only check while still exercising the runtime
        # FrozenInstanceError on the frozen dataclass.
        with pytest.raises(dataclasses.FrozenInstanceError):
            set_runtime_attribute(entry, "score", 4)
    finally:
        client.close()


def test_cache_get_returns_none_on_miss(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """_cache_get returns ``None`` (not an empty entry) on an unknown key."""
    client = _mk_client(monkeypatch, tmp_path)
    try:
        result = client._cache_get(_make_cache_key(client._cfg.model, "never-seen"))
        assert result is None
    finally:
        client.close()


def test_cache_get_preserves_null_score_in_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A persisted NULL-score row round-trips as entry.score is None.

    The dataclass must NOT coerce ``None`` to a sentinel int or collapse
    the row into ``None`` — it is the dataclass being ``None`` that
    signals "no row", while ``entry.score is None`` signals "row existed
    but the judge gave up".
    """
    client = _mk_client(monkeypatch, tmp_path)
    try:
        key = _make_cache_key(client._cfg.model, "exhausted-prompt")
        client._cache_put(
            key=key, score=None, raw_response="garbage", attempts=3,
        )
        entry = client._cache_get(key)
        assert entry is not None
        assert isinstance(entry, _JudgeCacheEntry)
        assert entry.score is None
        assert entry.raw_response == "garbage"
        assert entry.attempts == 3
    finally:
        client.close()


# ===========================================================================
# Parse-retry loop
# ===========================================================================


def test_parse_retry_on_format_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """parse_max_retries=1: first call unparseable, second parses → (1, 2)."""
    client = _mk_client(
        monkeypatch, tmp_path, cfg_kwargs={"parse_max_retries": 1},
    )
    try:
        stub = _PostStub(
            [_bad_format_response("nothing here"), _rating_response(1)],
        )
        _stub_post(monkeypatch, stub)
        score, attempts = client.score("retryable prompt")
        assert score == 1
        assert attempts == 2
        assert len(stub.calls) == 2
    finally:
        client.close()


def test_invalid_flag_after_parse_exhaustion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """All parse attempts fail → (None, parse_max_retries + 1); raw stored."""
    client = _mk_client(
        monkeypatch, tmp_path, cfg_kwargs={"parse_max_retries": 2},
    )
    try:
        stub = _PostStub(
            [
                _bad_format_response("garbage 1"),
                _bad_format_response("garbage 2"),
                _bad_format_response("garbage 3"),
            ],
        )
        _stub_post(monkeypatch, stub)
        score, attempts = client.score("doomed prompt")
        assert score is None
        assert attempts == 3  # parse_max_retries + 1
        assert len(stub.calls) == 3

        # Verify the NULL-sentinel row was persisted with the LAST raw response.
        key = _make_cache_key(client._cfg.model, "doomed prompt")
        row = client._cache_get(key)
        assert row is not None
        assert row.score is None
        assert row.raw_response == "garbage 3"
        assert row.attempts == 3
    finally:
        client.close()


def test_parse_max_retries_zero_means_single_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """parse_max_retries=0 → exactly 1 parse attempt; failure → (None, 1)."""
    client = _mk_client(
        monkeypatch, tmp_path, cfg_kwargs={"parse_max_retries": 0},
    )
    try:
        stub = _PostStub([_bad_format_response("no rating")])
        _stub_post(monkeypatch, stub)
        score, attempts = client.score("single-shot prompt")
        assert score is None
        assert attempts == 1
        assert len(stub.calls) == 1
    finally:
        client.close()


# ===========================================================================
# Rate limiting
# ===========================================================================


def test_rate_limiter_unlimited_returns_without_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """requests_per_minute <= 0 disables throttling entirely."""
    sleep_calls: list[float] = []
    monkeypatch.setattr(judge_module.time, "sleep", lambda s: sleep_calls.append(s))

    limiter = RateLimiter(requests_per_minute=0)
    for _ in range(100):
        limiter.acquire()

    assert sleep_calls == []


def test_rate_limiter_sliding_window_blocks_after_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The limiter allows N calls per 60-second window, then sleeps."""
    t = {"now": 1000.0}

    def _monotonic() -> float:
        return t["now"]

    sleep_durations: list[float] = []

    def _sleep(seconds: float) -> None:
        sleep_durations.append(seconds)
        t["now"] += seconds

    monkeypatch.setattr(judge_module.time, "monotonic", _monotonic)
    monkeypatch.setattr(judge_module.time, "sleep", _sleep)

    limiter = RateLimiter(requests_per_minute=2)
    limiter.acquire()
    limiter.acquire()
    limiter.acquire()

    assert sleep_durations == [pytest.approx(60.0)]


def test_rate_limiter_concurrent_bounded_mode_has_no_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent callers cannot reserve more than the window capacity."""
    t = {"now": 500.0}
    clock_lock = threading.Lock()

    def _monotonic() -> float:
        with clock_lock:
            return t["now"]

    sleep_durations: list[float] = []

    def _sleep(seconds: float) -> None:
        with clock_lock:
            sleep_durations.append(seconds)
            t["now"] += seconds

    monkeypatch.setattr(judge_module.time, "monotonic", _monotonic)
    monkeypatch.setattr(judge_module.time, "sleep", _sleep)

    limiter = RateLimiter(requests_per_minute=1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: limiter.acquire(), range(2)))

    assert sleep_durations == [pytest.approx(60.0)]


def test_rate_limit_zero_rpm_skips_sleep(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """requests_per_minute <= 0 → no rate-limit sleep, no div-by-zero."""
    _set_api_key(monkeypatch)
    sleep_calls: list[float] = []
    monkeypatch.setattr(judge_module.time, "sleep", lambda s: sleep_calls.append(s))
    stub = _PostStub([_rating_response(1), _rating_response(2)])
    _stub_post(monkeypatch, stub)
    cfg = JudgeConfig(requests_per_minute=0)
    client = AxBenchJudgeClient(cfg, cache_path=tmp_path / "rl0.sqlite")
    try:
        client.score("a")
        client.score("b")
        assert sleep_calls == []  # never invoked
    finally:
        client.close()


# ===========================================================================
# API error handling
# ===========================================================================


def test_malformed_api_response_consumes_parse_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Response JSON with no 'choices' key → NULL score after retries."""
    client = _mk_client(
        monkeypatch, tmp_path, cfg_kwargs={"parse_max_retries": 0},
    )
    try:
        stub = _PostStub([_FakeResponse(200, json_data={"foo": "bar"})])
        _stub_post(monkeypatch, stub)
        score, attempts = client.score("nogo")
        assert score is None
        assert attempts == 1
    finally:
        client.close()


def test_null_content_consumes_parse_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """content is None → NULL score after retries."""
    client = _mk_client(
        monkeypatch, tmp_path, cfg_kwargs={"parse_max_retries": 0},
    )
    try:
        stub = _PostStub(
            [
                _FakeResponse(
                    200,
                    json_data={
                        "choices": [{"message": {"content": None}}],
                    },
                ),
            ],
        )
        _stub_post(monkeypatch, stub)
        score, attempts = client.score("nullc")
        assert score is None
        assert attempts == 1
    finally:
        client.close()


def test_score_null_content_retries_then_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Direct score() and score_many() share transient malformed-body handling."""
    client = _mk_client(
        monkeypatch, tmp_path, cfg_kwargs={"parse_max_retries": 1},
    )
    try:
        stub = _PostStub([_null_content_response(), _rating_response(2)])
        _stub_post(monkeypatch, stub)
        score, attempts = client.score("transient-null-direct")
        assert score == 2
        assert attempts == 2
        assert len(stub.calls) == 2
    finally:
        client.close()


def test_non_string_content_consumes_parse_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """content is a non-str (e.g. int) → NULL score after retries."""
    client = _mk_client(
        monkeypatch, tmp_path, cfg_kwargs={"parse_max_retries": 0},
    )
    try:
        stub = _PostStub(
            [
                _FakeResponse(
                    200,
                    json_data={
                        "choices": [{"message": {"content": 42}}],
                    },
                ),
            ],
        )
        _stub_post(monkeypatch, stub)
        score, attempts = client.score("intcontent")
        assert score is None
        assert attempts == 1
    finally:
        client.close()


def test_api_error_with_backoff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """500 twice then 200 → 3 calls, returns parsed result."""
    client = _mk_client(monkeypatch, tmp_path)
    try:
        stub = _PostStub(
            [
                _FakeResponse(500, text="oops"),
                _FakeResponse(500, text="again"),
                _rating_response(2),
            ],
        )
        _stub_post(monkeypatch, stub)
        score, attempts = client.score("backoff prompt")
        assert score == 2
        # API retries are transparent to the "attempts" counter: that counter
        # only records parse attempts. So attempts == 1 for one parse success.
        assert attempts == 1
        assert len(stub.calls) == 3
    finally:
        client.close()


def test_429_triggers_backoff_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """429 is retried with backoff, same as 5xx."""
    client = _mk_client(monkeypatch, tmp_path)
    try:
        stub = _PostStub(
            [
                _FakeResponse(429, text="rate limit"),
                _rating_response(1),
            ],
        )
        _stub_post(monkeypatch, stub)
        score, attempts = client.score("ratelimited")
        assert score == 1
        assert attempts == 1
        assert len(stub.calls) == 2
    finally:
        client.close()


def test_exhausts_retries_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """All 5 transport retries return 500 → AxBenchBaseError."""
    client = _mk_client(monkeypatch, tmp_path)
    try:
        stub = _PostStub(
            [_FakeResponse(500, text="boom") for _ in range(6)],
        )
        _stub_post(monkeypatch, stub)
        with pytest.raises(AxBenchBaseError, match="500"):
            client.score("hopeless")
        # 5 retries + original = 6 total calls
        assert len(stub.calls) == 6
    finally:
        client.close()


def test_4xx_does_not_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """400 on the first call → immediate raise, no retry."""
    client = _mk_client(monkeypatch, tmp_path)
    try:
        stub = _PostStub([_FakeResponse(400, text="bad request: badprompt")])
        _stub_post(monkeypatch, stub)
        with pytest.raises(AxBenchBaseError, match="400") as excinfo:
            client.score("badprompt")
        assert len(stub.calls) == 1  # no retry
        assert "badprompt" not in str(excinfo.value)
        assert "bad request" not in str(excinfo.value)
    finally:
        client.close()


def test_network_error_retried_then_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Transient RequestException then success → retried and succeeds."""
    client = _mk_client(monkeypatch, tmp_path)
    try:
        stub = _PostStub(
            [
                requests.ConnectionError("dns failure"),
                _rating_response(1),
            ],
        )
        _stub_post(monkeypatch, stub)
        score, attempts = client.score("flaky")
        assert score == 1
        assert attempts == 1
        assert len(stub.calls) == 2
    finally:
        client.close()


def test_network_error_exhausts_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Network errors exceed retry budget → AxBenchBaseError wraps the cause."""
    client = _mk_client(monkeypatch, tmp_path)
    try:
        stub = _PostStub(
            [requests.ConnectionError(f"net err {i}") for i in range(6)],
        )
        _stub_post(monkeypatch, stub)
        with pytest.raises(AxBenchBaseError, match="OpenRouter request failed"):
            client.score("always-down")
        assert len(stub.calls) == 6
    finally:
        client.close()


def test_backoff_capped_at_60s(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Exponential backoff saturates at 60 seconds."""
    _set_api_key(monkeypatch)
    sleep_calls: list[float] = []
    monkeypatch.setattr(judge_module.time, "sleep", lambda s: sleep_calls.append(s))
    client = AxBenchJudgeClient(
        JudgeConfig(requests_per_minute=0),  # no rate-limit sleeps to pollute
        cache_path=tmp_path / "cap.sqlite",
    )
    try:
        stub = _PostStub(
            [_FakeResponse(500, text="err") for _ in range(6)],
        )
        _stub_post(monkeypatch, stub)
        with pytest.raises(AxBenchBaseError):
            client.score("cap")
        # 5 backoff sleeps: 1, 2, 4, 8, 16 (all <= 60 → no cap hit here).
        # Exercise the cap explicitly:
        sleep_calls.clear()
        client._backoff(10)  # 2**10 = 1024, capped → 60
        assert sleep_calls == [60.0]
    finally:
        client.close()


# ===========================================================================
# Lifecycle: close + context manager
# ===========================================================================


def test_close_releases_connection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """After close(), cache operations raise sqlite3.ProgrammingError."""
    _set_api_key(monkeypatch)
    client = AxBenchJudgeClient(
        JudgeConfig(), cache_path=tmp_path / "c.sqlite",
    )
    client.close()
    with pytest.raises(sqlite3.ProgrammingError):
        client._cache_get("whatever")


def test_context_manager_closes_on_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """with AxBenchJudgeClient(...) as c: ... closes the conn at exit."""
    _set_api_key(monkeypatch)
    _noop_sleep(monkeypatch)
    stub = _PostStub([_rating_response(2)])
    _stub_post(monkeypatch, stub)
    with AxBenchJudgeClient(
        JudgeConfig(), cache_path=tmp_path / "ctx.sqlite",
    ) as client:
        score, attempts = client.score("ctx-prompt")
        assert score == 2
        assert attempts == 1
    # Post-exit, the connection should be closed.
    with pytest.raises(sqlite3.ProgrammingError):
        client._cache_get("anything")


# ===========================================================================
# Unicode round-trip
# ===========================================================================


@pytest.mark.parametrize(
    ("label", "prompt"),
    [
        ("cafe", "rate this café"),
        ("emoji", "\U0001f525 fire completion"),
        ("japanese", "こんにちは 世界"),
        ("multiline", "line1\nline2\nline3"),
    ],
)
def test_unicode_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    label: str,
    prompt: str,
) -> None:
    """Non-ASCII prompts round-trip through the cache + key derivation."""
    client = _mk_client(
        monkeypatch, tmp_path, cache_name=f"u-{label}.sqlite",
    )
    try:
        stub = _PostStub([_rating_response(2)])
        _stub_post(monkeypatch, stub)
        score, attempts = client.score(prompt)
        assert score == 2
        assert attempts == 1
        # Second call hits the cache, proving the key/lookup path handles
        # non-ASCII correctly on both write and read.
        score2, attempts2 = client.score(prompt)
        assert score2 == 2
        assert attempts2 == 1
        assert len(stub.calls) == 1
    finally:
        client.close()


# ===========================================================================
# Mixed cache / retry / API path: exhaustion persists, second run short-circuits
# ===========================================================================


def test_exhausted_result_cached_and_short_circuits_second_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """After parse exhaustion, re-calling score() for the same prompt uses
    the cached NULL sentinel with zero API calls."""
    client = _mk_client(
        monkeypatch, tmp_path, cfg_kwargs={"parse_max_retries": 1},
    )
    try:
        stub = _PostStub(
            [
                _bad_format_response("try-1"),
                _bad_format_response("try-2"),
            ],
        )
        _stub_post(monkeypatch, stub)
        score, attempts = client.score("irrecoverable")
        assert score is None
        assert attempts == 2

        # Second call: must short-circuit via the NULL row.
        stub.responses = []  # any post() would blow up
        score2, attempts2 = client.score("irrecoverable")
        assert score2 is None
        assert attempts2 == 1
        assert len(stub.calls) == 2  # only the first run's calls
    finally:
        client.close()


# ===========================================================================
# Batched / concurrent scoring
# ===========================================================================


def test_score_many_preserves_input_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Variable-latency workers still return results in prompt order."""
    client = _mk_client(
        monkeypatch,
        tmp_path,
        cfg_kwargs={"requests_per_minute": 0, "max_concurrent": 3},
        cache_name="many-order.sqlite",
    )
    try:
        def _post(
            _self: Any,
            url: str,
            *,
            json: Any,
            headers: dict[str, str],
            timeout: Any,
        ) -> _FakeResponse:
            del _self, url, headers, timeout
            prompt = json["messages"][0]["content"]
            score = int(prompt[-1])
            # Make lower scores slower so completion order differs from
            # input order. Event.wait is unaffected by the monkeypatched
            # time.sleep used elsewhere in this file.
            threading.Event().wait(0.02 * (2 - score))
            return _rating_response(score)

        monkeypatch.setattr(requests.Session, "post", _post)

        results = client.score_many(["prompt-2", "prompt-0", "prompt-1"])

        assert results == [(2, 1), (0, 1), (1, 1)]
    finally:
        client.close()


def test_score_many_concurrent_cache_hits_make_zero_api_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Batched cache hits never touch OpenRouter."""
    client = _mk_client(
        monkeypatch,
        tmp_path,
        cfg_kwargs={"requests_per_minute": 0, "max_concurrent": 8},
        cache_name="many-cache.sqlite",
    )
    try:
        prompts = [f"cached-{i}" for i in range(6)]
        for i, prompt in enumerate(prompts):
            client._cache_put(
                key=_make_cache_key(client._cfg.model, prompt),
                score=i % 3,
                raw_response=f"Rating: [[{i % 3}]]",
                attempts=1,
            )
        stub = _PostStub([])
        _stub_post(monkeypatch, stub)

        results = client.score_many(prompts)

        assert results == [(i % 3, 1) for i in range(6)]
        assert stub.calls == []
    finally:
        client.close()


def test_score_many_duplicate_uncached_prompts_use_one_api_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The first duplicate owns parse retries; later duplicates look cached."""
    client = _mk_client(
        monkeypatch,
        tmp_path,
        cfg_kwargs={
            "parse_max_retries": 1,
            "requests_per_minute": 0,
            "max_concurrent": 8,
        },
        cache_name="many-dupe.sqlite",
    )
    try:
        stub = _PostStub(
            [_bad_format_response("not parseable"), _rating_response(2)],
        )
        _stub_post(monkeypatch, stub)

        results = client.score_many(["same prompt"] * 4)

        assert results == [(2, 2), (2, 1), (2, 1), (2, 1)]
        assert len(stub.calls) == 2
    finally:
        client.close()


def test_concurrent_score_calls_dedupe_same_key_miss(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Direct concurrent score() calls share the same per-key miss lock."""
    client = _mk_client(
        monkeypatch,
        tmp_path,
        cfg_kwargs={"requests_per_minute": 0},
        cache_name="score-dupe.sqlite",
    )
    try:
        stub = _PostStub([_rating_response(1)])
        _stub_post(monkeypatch, stub)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: client.score("shared"), range(8)))

        assert results == [(1, 1)] * 8
        assert len(stub.calls) == 1
    finally:
        client.close()


def test_score_many_distinct_prompts_runs_in_parallel_with_peak_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Distinct uncached prompts use multiple workers but respect the cap."""
    client = _mk_client(
        monkeypatch,
        tmp_path,
        cfg_kwargs={
            "requests_per_minute": 0,
            "max_concurrent": 3,
        },
        cache_name="many-peak.sqlite",
    )
    in_flight = 0
    peak = 0
    lock = threading.Lock()
    barrier = threading.Barrier(3, timeout=5.0)
    try:
        def _post(
            _self: Any,
            url: str,
            *,
            json: Any,
            headers: dict[str, str],
            timeout: Any,
        ) -> _FakeResponse:
            del _self, url, headers, timeout
            nonlocal in_flight, peak
            prompt = json["messages"][0]["content"]
            with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            barrier.wait()
            with lock:
                in_flight -= 1
            return _rating_response(int(prompt[-1]) % 3)

        monkeypatch.setattr(requests.Session, "post", _post)

        results = client.score_many([f"parallel-{i}" for i in range(6)])

        assert [score for score, _ in results] == [0, 1, 2, 0, 1, 2]
        assert peak == 3
    finally:
        client.close()


def test_score_many_propagates_profiler_context_to_worker_threads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    client = _mk_client(
        monkeypatch,
        tmp_path,
        cfg_kwargs={
            "requests_per_minute": 0,
            "max_concurrent": 2,
        },
        cache_name="many-profile.sqlite",
    )
    try:
        def _post(
            _self: Any,
            url: str,
            *,
            json: Any,
            headers: dict[str, str],
            timeout: Any,
        ) -> _FakeResponse:
            del _self, url, headers, timeout
            prompt = json["messages"][0]["content"]
            return _rating_response(int(prompt[-1]) % 3)

        monkeypatch.setattr(requests.Session, "post", _post)
        profile_dir = tmp_path / "profile"

        with AxBenchProfiler(profile_dir, synchronize_cuda=False):
            results = client.score_many([f"profile-{i}" for i in range(4)])

        assert [score for score, _attempts in results] == [0, 1, 2, 0]
        events = [
            json.loads(line)
            for line in (profile_dir / "events.jsonl").read_text().splitlines()
        ]
        api_attempts = [
            event for event in events
            if event["type"] == "counter"
            and event["name"] == "api.openrouter.attempts"
        ]
        assert len(api_attempts) == 4
        assert any(
            event["type"] == "region"
            and event["name"] == "api.openrouter.chat_completion"
            for event in events
        )
    finally:
        client.close()


def test_cache_put_null_does_not_overwrite_existing_valid_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _set_api_key(monkeypatch)
    _noop_sleep(monkeypatch)
    cache_path = tmp_path / "shared-null-race.sqlite"
    client_valid = AxBenchJudgeClient(JudgeConfig(), cache_path=cache_path)
    client_null = AxBenchJudgeClient(JudgeConfig(), cache_path=cache_path)
    try:
        key = _make_cache_key(client_valid._cfg.model, "shared prompt")
        client_valid._cache_put(
            key=key,
            score=2,
            raw_response="Rating: [[2]]",
            attempts=1,
        )
        client_null._cache_put(
            key=key,
            score=None,
            raw_response="not parseable",
            attempts=4,
        )

        entry = client_valid._cache_get(key)
        assert entry is not None
        assert entry.score == 2
        assert entry.raw_response == "Rating: [[2]]"
        assert entry.attempts == 1
    finally:
        client_valid.close()
        client_null.close()


def test_score_many_parse_exhaustion_persists_null_under_original_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Parse exhaustion writes one NULL sentinel under the canonical key."""
    client = _mk_client(
        monkeypatch,
        tmp_path,
        cfg_kwargs={
            "parse_max_retries": 1,
            "requests_per_minute": 0,
            "max_concurrent": 2,
        },
        cache_name="many-null.sqlite",
    )
    try:
        stub = _PostStub(
            [_bad_format_response("bad-1"), _bad_format_response("bad-2")],
        )
        _stub_post(monkeypatch, stub)

        results = client.score_many(["doomed"])

        assert results == [(None, 2)]
        key = _make_cache_key(client._cfg.model, "doomed")
        row = client._cache_get(key)
        assert row is not None
        assert row.score is None
        assert row.raw_response == "bad-2"
        assert row.attempts == 2
        with client._conn_lock:
            count = client._conn.execute(
                "SELECT COUNT(*) FROM judge_cache",
            ).fetchone()[0]
        assert count == 1
    finally:
        client.close()


def test_score_many_null_content_consumes_parse_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A transient null content response should not abort the judge batch."""
    client = _mk_client(
        monkeypatch,
        tmp_path,
        cfg_kwargs={
            "parse_max_retries": 1,
            "requests_per_minute": 0,
            "max_concurrent": 1,
        },
        cache_name="many-null-content.sqlite",
    )
    try:
        stub = _PostStub([_null_content_response(), _rating_response(2)])
        _stub_post(monkeypatch, stub)

        results = client.score_many(["transient-null"])

        assert results == [(2, 2)]
        assert len(stub.calls) == 2
        key = _make_cache_key(client._cfg.model, "transient-null")
        row = client._cache_get(key)
        assert row is not None
        assert row.score == 2
        assert row.raw_response == "Explanation. Rating: [[2]]"
        assert row.attempts == 2
    finally:
        client.close()


def test_score_many_transport_retries_are_per_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Each worker gets its own transport retry loop."""
    client = _mk_client(
        monkeypatch,
        tmp_path,
        cfg_kwargs={
            "requests_per_minute": 0,
            "max_concurrent": 2,
        },
        cache_name="many-transport.sqlite",
    )
    calls_by_prompt: dict[str, int] = {}
    lock = threading.Lock()
    try:
        def _post(
            _self: Any,
            url: str,
            *,
            json: Any,
            headers: dict[str, str],
            timeout: Any,
        ) -> _FakeResponse:
            del _self, url, headers, timeout
            prompt = json["messages"][0]["content"]
            with lock:
                calls_by_prompt[prompt] = calls_by_prompt.get(prompt, 0) + 1
                call_number = calls_by_prompt[prompt]
            if call_number == 1:
                return _FakeResponse(500, text="transient")
            return _rating_response(1 if prompt == "alpha" else 2)

        monkeypatch.setattr(requests.Session, "post", _post)

        results = client.score_many(["alpha", "beta"])

        assert results == [(1, 1), (2, 1)]
        assert calls_by_prompt == {"alpha": 2, "beta": 2}
    finally:
        client.close()


def test_score_many_non_retryable_4xx_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A 4xx from any worker propagates as AxBenchBaseError."""
    client = _mk_client(
        monkeypatch,
        tmp_path,
        cfg_kwargs={"requests_per_minute": 0, "max_concurrent": 1},
        cache_name="many-4xx.sqlite",
    )
    try:
        stub = _PostStub([_FakeResponse(400, text="bad request")])
        _stub_post(monkeypatch, stub)

        with pytest.raises(AxBenchBaseError, match="400"):
            client.score_many(["bad", "pending"])
    finally:
        client.close()


def test_score_many_concurrent_4xx_cancels_pending_without_deadlock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Failure while another worker is running shuts down cleanly."""
    client = _mk_client(
        monkeypatch,
        tmp_path,
        cfg_kwargs={"requests_per_minute": 0, "max_concurrent": 2},
        cache_name="many-4xx-concurrent.sqlite",
    )
    slow_started = threading.Event()
    failure_started = threading.Event()
    release_slow = threading.Event()
    calls: list[str] = []
    calls_lock = threading.Lock()

    def _release_slow_after_failure() -> None:
        assert failure_started.wait(timeout=5.0)
        # Keep the slow worker occupying one thread while score_many
        # observes the bad worker's failure and cancels queued work.
        threading.Event().wait(0.5)
        release_slow.set()

    releaser = threading.Thread(target=_release_slow_after_failure)
    releaser.start()
    try:
        def _post(
            _self: Any,
            url: str,
            *,
            json: Any,
            headers: dict[str, str],
            timeout: Any,
        ) -> _FakeResponse:
            del _self, url, headers, timeout
            prompt = json["messages"][0]["content"]
            with calls_lock:
                calls.append(prompt)
            if prompt == "slow":
                slow_started.set()
                assert release_slow.wait(timeout=5.0)
                return _rating_response(1)
            if prompt == "bad":
                assert slow_started.wait(timeout=5.0)
                failure_started.set()
                return _FakeResponse(400, text="bad request")

            return _rating_response(2)

        monkeypatch.setattr(requests.Session, "post", _post)

        with pytest.raises(AxBenchBaseError, match="400"):
            client.score_many(["slow", "bad", "pending"])

        assert "slow" in calls
        assert "bad" in calls
        # ThreadPoolExecutor may start the next queued task in the same
        # worker that just produced the 4xx before the main thread observes
        # that failing future. The contract we need here is clean failure
        # propagation with no deadlock or retry storm.
        assert set(calls).issubset({"slow", "bad", "pending"})
        assert len(calls) <= 3
    finally:
        release_slow.set()
        releaser.join(timeout=5.0)
        client.close()


# ---------- W2: judge.py uses session helper + (10, 60) timeout ----------

def test_axbench_judge_client_uses_openrouter_session(monkeypatch, tmp_path):
    """AxBenchJudgeClient routes through get_openrouter_session and
    passes the (10, 60) timeout tuple."""
    import requests

    from steering.eval.axbench_base import judge
    from steering.eval.axbench_base.config import JudgeConfig

    captured = {"session_used": None, "headers": None, "timeout": None}

    def fake_session_post(self, url, *, json, headers, timeout):
        captured["session_used"] = self
        captured["headers"] = headers
        captured["timeout"] = timeout
        class _Resp:
            status_code = 200
            text = '{"choices": [{"message": {"content": "Rating: 1"}}]}'
            def json(self):
                return {"choices": [{"message": {"content": "Rating: 1"}}]}
        return _Resp()

    monkeypatch.setattr(requests.Session, "post", fake_session_post)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    cli = judge.AxBenchJudgeClient(cfg=JudgeConfig(), cache_path=tmp_path / "j.sqlite")
    # Call the lowest-level HTTP method (_call_api). Skip the higher-level
    # score_* methods — they'd require building up rubric inputs.
    out = cli._call_api("test prompt")
    session_used = captured["session_used"]
    headers = captured["headers"]
    assert isinstance(session_used, requests.Session)
    assert isinstance(headers, dict)
    # The session itself supplies Connection: close — either the per-call
    # headers dict or the session-level headers MUST carry it.
    assert (
        headers.get("Connection") == "close"
        or session_used.headers.get("Connection") == "close"
    )
    assert captured["timeout"] == (10, 60)
    assert out == "Rating: 1"


def test_axbench_judge_client_retries_on_invalid_json_then_succeeds(monkeypatch, tmp_path):
    """First response decodes to ValueError on .json(); second succeeds;
    judge must retry once and return the second body."""
    import requests

    from steering.eval.axbench_base import judge
    from steering.eval.axbench_base.config import JudgeConfig

    attempts = {"n": 0}

    def fake_session_post(self, url, *, json, headers, timeout):
        attempts["n"] += 1
        class _Resp:
            status_code = 200
            text = "non-json" if attempts["n"] == 1 else (
                '{"choices": [{"message": {"content": "Rating: 2"}}]}'
            )
            def json(self):
                if attempts["n"] == 1:
                    raise ValueError("not json")
                return {"choices": [{"message": {"content": "Rating: 2"}}]}
        return _Resp()

    monkeypatch.setattr(requests.Session, "post", fake_session_post)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(judge.AxBenchJudgeClient, "_backoff", lambda self, attempt: None)
    cli = judge.AxBenchJudgeClient(cfg=JudgeConfig(), cache_path=tmp_path / "j.sqlite")
    out = cli._call_api("test prompt")
    assert out == "Rating: 2"
    assert attempts["n"] == 2
