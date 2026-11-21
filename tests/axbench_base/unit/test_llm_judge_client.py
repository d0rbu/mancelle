"""Unit tests for ``llm_judge_client`` (review-fix F + 7)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import requests
from beartype.roar import BeartypeCallHintParamViolation

from steering.eval.axbench_base import llm_judge_client
from steering.eval.axbench_base.config import LLMClientConfig
from steering.eval.axbench_base.llm_judge_client import (
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_CHAT_COMPLETIONS_URL,
    LLMJudgeClient,
    OpenRouterLLMClient,
)
from steering.eval.axbench_base.synth.llm_response_cache import SynthResponseCache
from steering.eval.axbench_base.types import AxBenchBaseError


class _FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, status_code, json_data=None, text="fake-text"):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        return self._json_data


class _PostStub:
    """Configurable stub for ``requests.post``.

    Each call pops the next item from ``responses``. If the item is an
    Exception instance, it is raised instead of returned.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, url, *, json, headers, timeout):
        self.calls.append(
            {"url": url, "json": json, "headers": headers, "timeout": timeout}
        )
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _stub_post(monkeypatch, stub):
    # Production code routes through get_openrouter_session().post(...) so we
    # patch the Session method. The unbound method receives the Session instance
    # as its first argument; we drop it to preserve the _PostStub.__call__ sig.
    monkeypatch.setattr(requests.Session, "post", lambda _self, *args, **kwargs: stub(*args, **kwargs))


def _noop_sleep(monkeypatch):
    """Replace ``time.sleep`` inside ``llm_judge_client`` with a no-op."""
    monkeypatch.setattr(llm_judge_client.time, "sleep", lambda seconds: None)


def _set_api_key(monkeypatch):
    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "sk-test-123")


def _cache_row_count(cache: SynthResponseCache) -> int:
    with cache._lock:
        row = cache._conn.execute("SELECT COUNT(*) FROM cache").fetchone()
    return int(row[0])


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


def test_llm_judge_client_base_raises_not_implemented():
    """Calling the base class directly should hit NotImplementedError."""
    client = LLMJudgeClient()
    with pytest.raises(NotImplementedError):
        client.complete([{"role": "user", "content": "hi"}])


def test_llm_judge_client_base_complete_accepts_cache_key_kwarg():
    """The base class's ``complete`` signature should accept ``cache_key`` kwarg.

    This is required so subclasses (and callers thereof) can pass
    ``cache_key`` through the uniform LLMJudgeClient interface. The base
    class still raises NotImplementedError — the kwarg just needs to be
    an accepted parameter.
    """
    client = LLMJudgeClient()
    with pytest.raises(NotImplementedError):
        client.complete(
            [{"role": "user", "content": "hi"}], cache_key="some-key",
        )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_openrouter_client_requires_api_key(monkeypatch):
    """Clearing the env var → construction raises AxBenchBaseError."""
    monkeypatch.delenv(OPENROUTER_API_KEY_ENV, raising=False)
    with pytest.raises(AxBenchBaseError, match=OPENROUTER_API_KEY_ENV):
        OpenRouterLLMClient(LLMClientConfig())


def test_openrouter_client_stores_api_key(monkeypatch):
    _set_api_key(monkeypatch)
    client = OpenRouterLLMClient(LLMClientConfig(requests_per_minute=30))
    assert client._api_key == "sk-test-123"


@pytest.mark.parametrize("provider,key_env,other_key", [
    ("openai", "OPENAI_API_KEY", "OPENROUTER_API_KEY"),
    ("openrouter", "OPENROUTER_API_KEY", "OPENAI_API_KEY"),
])
def test_provider_requires_its_own_key_without_fallback(monkeypatch, provider, key_env, other_key):
    monkeypatch.delenv(key_env, raising=False)
    monkeypatch.setenv(other_key, "wrong-provider-key")
    with pytest.raises(AxBenchBaseError, match=key_env):
        OpenRouterLLMClient(LLMClientConfig(provider=provider))


def test_unknown_provider_is_rejected_before_any_credential_lookup(monkeypatch):
    environ = SimpleNamespace(get=lambda *_: pytest.fail("must validate route before reading credentials"))
    monkeypatch.setattr(llm_judge_client, "os", SimpleNamespace(environ=environ))
    with pytest.raises(AxBenchBaseError, match="unsupported chat provider"):
        OpenRouterLLMClient(LLMClientConfig(provider="unknown"))


@pytest.mark.parametrize("provider,endpoint,key", [
    ("openrouter", OPENROUTER_CHAT_COMPLETIONS_URL, "router-key"),
    ("openai", "https://api.openai.com/v1/chat/completions", "direct-key"),
])
def test_provider_selects_exact_endpoint_auth_and_metric_namespace(monkeypatch, provider, endpoint, key):
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    monkeypatch.setenv("OPENAI_API_KEY", "direct-key")
    counters = []
    monkeypatch.setattr(llm_judge_client, "record_counter", lambda name, *_a, **_kw: counters.append(name))
    stub = _PostStub([_FakeResponse(200, {"choices": [{"message": {"content": "reply"}}]})])
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(LLMClientConfig(provider=provider, model="requested", requests_per_minute=0),
                                 max_tokens=17)
    messages = [{"role": "system", "content": " exact system\n"}, {"role": "user", "content": "Ω request "}]
    assert client.complete(messages) == "reply"
    assert stub.calls == [{"url": endpoint, "json": {"model": "requested", "messages": messages, "max_tokens": 17},
                           "headers": {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                           "timeout": (10, 60)}]
    assert counters == [f"api.{provider}.rate_limit_bypass", f"api.{provider}.attempts"]


def test_openai_transport_errors_use_actual_provider_and_do_not_leak_body(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "direct-key")
    stub = _PostStub([_FakeResponse(401, text="private request body")])
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(LLMClientConfig(provider="openai", requests_per_minute=0))
    with pytest.raises(AxBenchBaseError, match="OpenAI returned 401") as error:
        client.complete([{"role": "user", "content": "private request body"}])
    assert "private request body" not in str(error.value)
    assert len(stub.calls) == 1


def test_providers_do_not_share_text_or_structured_cache_records(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    monkeypatch.setenv("OPENAI_API_KEY", "direct-key")
    responses = [{"choices": [{"finish_reason": "stop", "message": {"content": text}}]}
                 for text in ("router-text", "direct-text", "router-raw", "direct-raw")]
    stub = _PostStub([_FakeResponse(200, response) for response in responses])
    _stub_post(monkeypatch, stub)
    messages = [{"role": "user", "content": "original"}]
    with SynthResponseCache(tmp_path / "cache.sqlite") as cache:
        clients = [OpenRouterLLMClient(LLMClientConfig(provider=provider, model="same", requests_per_minute=0),
                                      response_cache=cache) for provider in ("openrouter", "openai")]
        for _ in range(2):
            assert clients[0].complete(messages, cache_key="same-key") == "router-text"
            assert clients[1].complete(messages, cache_key="same-key") == "direct-text"
        for _ in range(2):
            for index, client in enumerate(clients):
                assert client.complete_response(messages, sampling={"max_tokens": 32},
                                                cache_key="same-key") == responses[index + 2]
        assert cache.get("same-key") == "router-text"  # Exact legacy text namespace retained.
        assert _cache_row_count(cache) == 4
    assert len(stub.calls) == 4


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_openrouter_client_sends_expected_payload(monkeypatch):
    _set_api_key(monkeypatch)
    _noop_sleep(monkeypatch)
    stub = _PostStub(
        [
            _FakeResponse(
                200,
                json_data={
                    "choices": [
                        {"message": {"content": "hello from the judge"}},
                    ],
                },
            ),
        ],
    )
    _stub_post(monkeypatch, stub)
    cfg = LLMClientConfig(model="openai/gpt-4o-mini", requests_per_minute=120)
    client = OpenRouterLLMClient(cfg)
    messages = [{"role": "user", "content": "say hi"}]
    result = client.complete(messages)
    assert result == "hello from the judge"
    assert len(stub.calls) == 1
    call = stub.calls[0]
    assert call["url"] == OPENROUTER_CHAT_COMPLETIONS_URL
    assert call["json"] == {"model": "openai/gpt-4o-mini", "messages": messages}
    assert call["headers"]["Authorization"] == "Bearer sk-test-123"
    assert call["headers"]["Content-Type"] == "application/json"
    assert call["timeout"] == (10, 60)


def test_openrouter_client_includes_explicit_max_tokens(monkeypatch):
    """A caller-provided cap is sent without changing the default payload."""
    _set_api_key(monkeypatch)
    _noop_sleep(monkeypatch)
    stub = _PostStub(
        [
            _FakeResponse(
                200,
                json_data={"choices": [{"message": {"content": "ok"}}]},
            ),
        ],
    )
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(
        LLMClientConfig(model="provider/model", requests_per_minute=120),
        max_tokens=256,
    )

    assert client.complete([{"role": "user", "content": "continue"}]) == "ok"
    assert stub.calls[0]["json"] == {
        "model": "provider/model",
        "messages": [{"role": "user", "content": "continue"}],
        "max_tokens": 256,
    }


@pytest.mark.parametrize("max_tokens", [True, False, 0, -1, 1.5, "256"])
def test_openrouter_client_rejects_invalid_max_tokens(monkeypatch, max_tokens):
    _set_api_key(monkeypatch)
    with pytest.raises(BeartypeCallHintParamViolation, match="max_tokens"):
        OpenRouterLLMClient(LLMClientConfig(), max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------------


def test_openrouter_client_retries_on_429(monkeypatch):
    """429 twice, then 200 → expect 3 calls and a success."""
    _set_api_key(monkeypatch)
    _noop_sleep(monkeypatch)
    stub = _PostStub(
        [
            _FakeResponse(429, text="rate limit"),
            _FakeResponse(429, text="rate limit"),
            _FakeResponse(
                200,
                json_data={"choices": [{"message": {"content": "ok"}}]},
            ),
        ],
    )
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(LLMClientConfig(), max_retries=3)
    assert client.complete([{"role": "user", "content": "x"}]) == "ok"
    assert len(stub.calls) == 3


def test_openrouter_client_raises_after_max_retries(monkeypatch):
    """Persistent 500 → AxBenchBaseError after max_retries + 1 attempts."""
    _set_api_key(monkeypatch)
    _noop_sleep(monkeypatch)
    stub = _PostStub(
        [_FakeResponse(500, text="server err") for _ in range(4)],
    )
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(LLMClientConfig(), max_retries=3)
    with pytest.raises(AxBenchBaseError, match="500"):
        client.complete([{"role": "user", "content": "x"}])
    assert len(stub.calls) == 4


def test_openrouter_client_retries_on_network_error(monkeypatch):
    """Transient ConnectionError twice then 200 → success."""
    _set_api_key(monkeypatch)
    _noop_sleep(monkeypatch)
    stub = _PostStub(
        [
            requests.ConnectionError("dns failure"),
            requests.ConnectionError("connection reset"),
            _FakeResponse(
                200,
                json_data={"choices": [{"message": {"content": "eventual win"}}]},
            ),
        ],
    )
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(LLMClientConfig(), max_retries=3)
    result = client.complete([{"role": "user", "content": "x"}])
    assert result == "eventual win"
    assert len(stub.calls) == 3


def test_openrouter_client_retries_on_embedded_retryable_error(monkeypatch):
    """OpenRouter can return HTTP 200 with an embedded retryable error body."""
    _set_api_key(monkeypatch)
    _noop_sleep(monkeypatch)
    stub = _PostStub(
        [
            _FakeResponse(
                200,
                json_data={
                    "error": {
                        "message": "JSON error injected into SSE stream",
                        "code": 502,
                    },
                },
            ),
            _FakeResponse(
                200,
                json_data={"choices": [{"message": {"content": "ok"}}]},
            ),
        ],
    )
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(LLMClientConfig(), max_retries=3)
    assert client.complete([{"role": "user", "content": "x"}]) == "ok"
    assert len(stub.calls) == 2


def test_openrouter_client_raises_after_embedded_retryable_error_retries(monkeypatch):
    _set_api_key(monkeypatch)
    _noop_sleep(monkeypatch)
    stub = _PostStub(
        [
            _FakeResponse(
                200,
                json_data={"error": {"message": "transient", "code": "502"}},
            )
            for _ in range(3)
        ],
    )
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(LLMClientConfig(), max_retries=2)
    with pytest.raises(
        AxBenchBaseError,
        match="embedded retryable error 502 after 3 attempts",
    ):
        client.complete([{"role": "user", "content": "x"}])
    assert len(stub.calls) == 3


def test_openrouter_client_raises_on_persistent_network_error(monkeypatch):
    """Network errors continue → AxBenchBaseError after max_retries + 1 attempts."""
    _set_api_key(monkeypatch)
    _noop_sleep(monkeypatch)
    stub = _PostStub(
        [requests.ConnectionError(f"err {i}") for i in range(4)],
    )
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(LLMClientConfig(), max_retries=3)
    with pytest.raises(AxBenchBaseError, match="OpenRouter request failed after 4 attempts"):
        client.complete([{"role": "user", "content": "x"}])
    assert len(stub.calls) == 4


# ---------------------------------------------------------------------------
# Non-retryable failures
# ---------------------------------------------------------------------------


def test_openrouter_client_raises_on_bad_shape(monkeypatch):
    """200 + unexpected JSON shape → AxBenchBaseError."""
    _set_api_key(monkeypatch)
    _noop_sleep(monkeypatch)
    stub = _PostStub(
        [_FakeResponse(200, json_data={"foo": "bar"})],
    )
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(LLMClientConfig())
    with pytest.raises(AxBenchBaseError, match="Unexpected OpenRouter response shape"):
        client.complete([{"role": "user", "content": "x"}])


def test_openrouter_client_raises_on_null_content(monkeypatch):
    """200 + ``content`` is ``None`` → AxBenchBaseError mentioning 'null content'."""
    _set_api_key(monkeypatch)
    _noop_sleep(monkeypatch)
    stub = _PostStub(
        [
            _FakeResponse(
                200,
                json_data={"choices": [{"message": {"content": None}}]},
            ),
        ],
    )
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(LLMClientConfig())
    with pytest.raises(AxBenchBaseError, match="null content"):
        client.complete([{"role": "user", "content": "x"}])


def test_openrouter_client_raises_on_non_string_content(monkeypatch):
    """200 + ``content`` is an int → AxBenchBaseError mentioning 'non-string content'."""
    _set_api_key(monkeypatch)
    _noop_sleep(monkeypatch)
    stub = _PostStub(
        [
            _FakeResponse(
                200,
                json_data={"choices": [{"message": {"content": 42}}]},
            ),
        ],
    )
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(LLMClientConfig())
    with pytest.raises(AxBenchBaseError, match="non-string content"):
        client.complete([{"role": "user", "content": "x"}])


def test_openrouter_client_raises_on_non_retryable_4xx(monkeypatch):
    """400 is a user error: should raise immediately with no retry."""
    _set_api_key(monkeypatch)
    _noop_sleep(monkeypatch)
    stub = _PostStub(
        [_FakeResponse(400, text="bad request")],
    )
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(LLMClientConfig(), max_retries=3)
    with pytest.raises(AxBenchBaseError, match="400"):
        client.complete([{"role": "user", "content": "x"}])
    assert len(stub.calls) == 1  # no retries


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_openrouter_client_rate_limits_after_window_capacity(monkeypatch):
    """The LLM client allows N calls per minute, then sleeps for the window."""
    _set_api_key(monkeypatch)
    stub = _PostStub(
        [
            _FakeResponse(
                200,
                json_data={"choices": [{"message": {"content": "1"}}]},
            ),
            _FakeResponse(
                200,
                json_data={"choices": [{"message": {"content": "2"}}]},
            ),
            _FakeResponse(
                200,
                json_data={"choices": [{"message": {"content": "3"}}]},
            ),
        ],
    )
    _stub_post(monkeypatch, stub)

    t = {"now": 1000.0}

    def _monotonic():
        return t["now"]

    sleep_durations: list[float] = []

    def _sleep(seconds):
        sleep_durations.append(seconds)
        t["now"] += seconds

    monkeypatch.setattr(llm_judge_client.time, "monotonic", _monotonic)
    monkeypatch.setattr(llm_judge_client.time, "sleep", _sleep)

    cfg = LLMClientConfig(requests_per_minute=2)
    client = OpenRouterLLMClient(cfg)
    client.complete([{"role": "user", "content": "a"}])
    client.complete([{"role": "user", "content": "b"}])
    client.complete([{"role": "user", "content": "c"}])
    assert sleep_durations == [pytest.approx(60.0)]


def test_openrouter_client_rate_limit_skips_sleep_when_under_capacity(monkeypatch):
    """Calls under the per-minute capacity do not sleep."""
    _set_api_key(monkeypatch)
    stub = _PostStub(
        [
            _FakeResponse(
                200,
                json_data={"choices": [{"message": {"content": "1"}}]},
            ),
            _FakeResponse(
                200,
                json_data={"choices": [{"message": {"content": "2"}}]},
            ),
        ],
    )
    _stub_post(monkeypatch, stub)

    t = {"now": 100.0}

    def _monotonic():
        return t["now"]

    sleep_durations: list[float] = []

    def _sleep(seconds):
        sleep_durations.append(seconds)
        t["now"] += seconds

    monkeypatch.setattr(llm_judge_client.time, "monotonic", _monotonic)
    monkeypatch.setattr(llm_judge_client.time, "sleep", _sleep)

    cfg = LLMClientConfig(requests_per_minute=2)
    client = OpenRouterLLMClient(cfg)
    client.complete([{"role": "user", "content": "a"}])
    client.complete([{"role": "user", "content": "b"}])
    assert sleep_durations == []


# ---------------------------------------------------------------------------
# Backoff helper
# ---------------------------------------------------------------------------


def test_openrouter_client_backoff_sleeps_exponentially(monkeypatch):
    """``_backoff`` grows as 2**attempt, jittered into [0.5, 1.0] of that.

    Jitter is deliberate: without it, every worker that entered the same
    upstream outage resumes on the same tick and re-exhausts the capacity it
    was waiting on. The exponential shape is still pinned per attempt, and
    ``_backoff`` returns what it slept so the caller can bill the budget.
    """
    _set_api_key(monkeypatch)
    sleep_durations: list[float] = []

    def _sleep(seconds):
        sleep_durations.append(seconds)

    monkeypatch.setattr(llm_judge_client.time, "sleep", _sleep)
    client = OpenRouterLLMClient(LLMClientConfig())
    returned = [client._backoff(0), client._backoff(1), client._backoff(2)]
    assert returned == sleep_durations
    for attempt, (base, delay) in enumerate(
        zip([1.0, 2.0, 4.0], sleep_durations, strict=True)
    ):
        assert 0.5 * base <= delay <= base, f"attempt {attempt}: {delay} vs {base}"


# ---------------------------------------------------------------------------
# Response cache integration (review-fix 7)
# ---------------------------------------------------------------------------


def test_openrouter_client_cache_hit_skips_api_call(monkeypatch, tmp_path):
    """Populated cache + matching key → no HTTP call, return stored value."""
    _set_api_key(monkeypatch)
    _noop_sleep(monkeypatch)
    cache = SynthResponseCache(tmp_path / "c.sqlite")
    try:
        cache.set("fixed-key", "previously-stored-response")
        stub = _PostStub([])  # empty — any request.post would blow up
        _stub_post(monkeypatch, stub)
        client = OpenRouterLLMClient(LLMClientConfig(), response_cache=cache)
        result = client.complete(
            [{"role": "user", "content": "ignored"}],
            cache_key="fixed-key",
        )
        assert result == "previously-stored-response"
        assert stub.calls == []  # requests.post never called
    finally:
        cache.close()


def test_openrouter_client_cache_miss_writes_after_success(monkeypatch, tmp_path):
    """Empty cache + success → response stored under the requested key."""
    _set_api_key(monkeypatch)
    _noop_sleep(monkeypatch)
    cache = SynthResponseCache(tmp_path / "c.sqlite")
    try:
        stub = _PostStub(
            [
                _FakeResponse(
                    200,
                    json_data={
                        "choices": [{"message": {"content": "fresh response"}}],
                    },
                ),
            ],
        )
        _stub_post(monkeypatch, stub)
        client = OpenRouterLLMClient(LLMClientConfig(), response_cache=cache)
        assert cache.get("key-1") is None
        result = client.complete(
            [{"role": "user", "content": "q"}], cache_key="key-1",
        )
        assert result == "fresh response"
        assert cache.get("key-1") == "fresh response"
        # A second call with the same key returns the cached value.
        stub.responses = []  # would blow up if hit
        again = client.complete(
            [{"role": "user", "content": "q"}], cache_key="key-1",
        )
        assert again == "fresh response"
    finally:
        cache.close()


def test_openrouter_client_cache_key_none_skips_cache(monkeypatch, tmp_path):
    """cache_key=None → no read, no write, even with a cache attached."""
    _set_api_key(monkeypatch)
    _noop_sleep(monkeypatch)
    cache = SynthResponseCache(tmp_path / "c.sqlite")
    try:
        stub = _PostStub(
            [
                _FakeResponse(
                    200,
                    json_data={
                        "choices": [{"message": {"content": "uncached"}}],
                    },
                ),
            ],
        )
        _stub_post(monkeypatch, stub)
        client = OpenRouterLLMClient(LLMClientConfig(), response_cache=cache)
        result = client.complete(
            [{"role": "user", "content": "q"}], cache_key=None,
        )
        assert result == "uncached"
        assert _cache_row_count(cache) == 0
    finally:
        cache.close()


def test_openrouter_client_cache_is_optional(monkeypatch):
    """A client with no cache should silently ignore cache_key args."""
    _set_api_key(monkeypatch)
    _noop_sleep(monkeypatch)
    stub = _PostStub(
        [
            _FakeResponse(
                200,
                json_data={
                    "choices": [{"message": {"content": "nocacheclient"}}],
                },
            ),
        ],
    )
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(LLMClientConfig())  # response_cache default None
    result = client.complete(
        [{"role": "user", "content": "q"}], cache_key="some-key",
    )
    assert result == "nocacheclient"
    assert len(stub.calls) == 1


def test_openrouter_client_failed_call_does_not_cache(monkeypatch, tmp_path):
    """500-then-200 retry pattern: only the successful response is cached."""
    _set_api_key(monkeypatch)
    _noop_sleep(monkeypatch)
    cache = SynthResponseCache(tmp_path / "c.sqlite")
    try:
        stub = _PostStub(
            [
                _FakeResponse(500, text="server err"),
                _FakeResponse(
                    200,
                    json_data={
                        "choices": [{"message": {"content": "second try"}}],
                    },
                ),
            ],
        )
        _stub_post(monkeypatch, stub)
        client = OpenRouterLLMClient(
            LLMClientConfig(), max_retries=3, response_cache=cache,
        )
        result = client.complete(
            [{"role": "user", "content": "q"}], cache_key="retried",
        )
        assert result == "second try"
        # Cache should have the final successful value, not the intermediate
        # 500 that was retried.
        assert cache.get("retried") == "second try"
        assert _cache_row_count(cache) == 1
    finally:
        cache.close()


# ---------- W2: session usage + JSON-parse retry ----------

def test_llm_judge_client_uses_openrouter_session(monkeypatch):
    """Verify every retry attempt routes through get_openrouter_session
    rather than the bare ``requests.post`` path."""
    from steering.eval.axbench_base import llm_judge_client
    from steering.eval.axbench_base.config import LLMClientConfig

    captured: dict[str, Any] = {"used_sessions": []}

    def fake_session_post(self, url, *, json, headers, timeout):
        captured["used_sessions"].append(self)
        captured["headers"] = headers
        captured["timeout"] = timeout
        class _Resp:
            status_code = 200
            text = '{"choices": [{"message": {"content": "ok"}}]}'
            def json(self):
                return {"choices": [{"message": {"content": "ok"}}]}
        return _Resp()

    monkeypatch.setattr(requests.Session, "post", fake_session_post)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    cli = llm_judge_client.OpenRouterLLMClient(
        cfg=LLMClientConfig(),
        max_retries=0,
    )
    out = cli.complete([{"role": "user", "content": "hi"}])
    assert out == "ok"
    assert captured["timeout"] == (10, 60)
    # The session itself supplies Connection: close — either the
    # per-call headers dict or the session-level headers MUST carry it.
    assert (
        captured["headers"].get("Connection") == "close"
        or any(
            s.headers.get("Connection") == "close"
            for s in captured["used_sessions"]
        )
    )


def test_llm_judge_client_retries_on_invalid_json_then_succeeds(monkeypatch):
    """First response decodes to ValueError on .json(); second response
    is valid. The client should retry once and return the second body."""
    from steering.eval.axbench_base import llm_judge_client
    from steering.eval.axbench_base.config import LLMClientConfig

    attempts = {"n": 0}

    def fake_session_post(self, url, *, json, headers, timeout):
        attempts["n"] += 1
        class _Resp:
            status_code = 200
            text = "non-json bytes" if attempts["n"] == 1 else (
                '{"choices": [{"message": {"content": "after-retry"}}]}'
            )
            def json(self):
                if attempts["n"] == 1:
                    raise ValueError("not json")
                return {"choices": [{"message": {"content": "after-retry"}}]}
        return _Resp()

    monkeypatch.setattr(requests.Session, "post", fake_session_post)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(
        llm_judge_client.OpenRouterLLMClient,
        "_backoff",
        # ``_backoff`` returns the seconds it slept so the caller can bill
        # the wall-clock retry budget; a stub returning None breaks that sum.
        lambda self, attempt, retry_after=None: 0.0,
    )
    cli = llm_judge_client.OpenRouterLLMClient(
        cfg=LLMClientConfig(),
        max_retries=2,
    )
    out = cli.complete([{"role": "user", "content": "hi"}])
    assert out == "after-retry"
    assert attempts["n"] == 2


def test_llm_judge_client_raises_after_max_retries_on_invalid_json(monkeypatch):
    """Every response is invalid JSON; client must raise an
    AxBenchBaseError naming the failure mode."""
    from steering.eval.axbench_base import llm_judge_client
    from steering.eval.axbench_base.config import LLMClientConfig
    from steering.eval.axbench_base.types import AxBenchBaseError

    def fake_session_post(self, url, *, json, headers, timeout):
        class _Resp:
            status_code = 200
            text = "still not json"
            def json(self):
                raise ValueError("not json")
        return _Resp()

    monkeypatch.setattr(requests.Session, "post", fake_session_post)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(
        llm_judge_client.OpenRouterLLMClient,
        "_backoff",
        # ``_backoff`` returns the seconds it slept so the caller can bill
        # the wall-clock retry budget; a stub returning None breaks that sum.
        lambda self, attempt, retry_after=None: 0.0,
    )
    cli = llm_judge_client.OpenRouterLLMClient(
        cfg=LLMClientConfig(),
        max_retries=1,
    )
    with pytest.raises(AxBenchBaseError) as exc_info:
        cli.complete([{"role": "user", "content": "hi"}])
    assert "JSON" in str(exc_info.value) or "json" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Wall-clock transport retry budget
#
# The default synth model is served by a single upstream provider that emits
# transient 429s at a ~20% rate even at concurrency 1, clearing within
# seconds. Attempt-count-only retries tolerate 1+2+4 = 7s of backoff, so a
# long stage reliably loses an in-flight call to one burst and aborts the
# whole run. These tests pin the wall-clock budget that absorbs the bursts,
# and — just as importantly — pin that it still gives up loudly.
# ---------------------------------------------------------------------------


def _raise_value_error(*_args, **_kwargs):
    """Stand in for ``Response.json()`` on a truncated body."""
    raise ValueError("Expecting value: line 1013 column 1 (char 5566)")


class _RecordingSleep:
    """Capture backoff durations without actually sleeping."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def _record_sleep(monkeypatch) -> _RecordingSleep:
    rec = _RecordingSleep()
    monkeypatch.setattr(llm_judge_client.time, "sleep", rec)
    return rec


def _budget_cfg(seconds: float) -> LLMClientConfig:
    return LLMClientConfig(transport_retry_budget_seconds=seconds)


def test_retry_budget_survives_burst_that_exceeds_attempt_count(monkeypatch):
    """The regression: 8 consecutive 429s with max_retries=3.

    Under attempt-count-only semantics this raises on the 4th response. The
    wall-clock budget must carry it through to the eventual 200.
    """
    _set_api_key(monkeypatch)
    _record_sleep(monkeypatch)
    stub = _PostStub(
        [_FakeResponse(429, text="rate limit") for _ in range(8)]
        + [_FakeResponse(200, json_data={"choices": [{"message": {"content": "ok"}}]})],
    )
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(_budget_cfg(300.0), max_retries=3)
    assert client.complete([{"role": "user", "content": "x"}]) == "ok"
    assert len(stub.calls) == 9


def test_retry_budget_zero_preserves_attempt_count_semantics(monkeypatch):
    """Budget 0 (the base default) must not change existing behavior."""
    _set_api_key(monkeypatch)
    _record_sleep(monkeypatch)
    stub = _PostStub([_FakeResponse(429, text="rate limit") for _ in range(10)])
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(_budget_cfg(0.0), max_retries=3)
    with pytest.raises(AxBenchBaseError, match="OpenRouter returned 429 after 4 attempts"):
        client.complete([{"role": "user", "content": "x"}])
    assert len(stub.calls) == 4


def test_retry_budget_exhaustion_raises_loudly(monkeypatch):
    """A genuinely down provider must still fail, reporting attempts + backoff."""
    _set_api_key(monkeypatch)
    rec = _record_sleep(monkeypatch)
    stub = _PostStub([_FakeResponse(429, text="rate limit") for _ in range(500)])
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(_budget_cfg(120.0), max_retries=3)
    with pytest.raises(AxBenchBaseError, match=r"OpenRouter returned 429 after \d+ attempts and \d+\.\ds of backoff"):
        client.complete([{"role": "user", "content": "x"}])
    # It gave up because the wall clock ran out, not the attempt counter.
    assert len(stub.calls) > 4
    assert sum(rec.delays) >= 120.0


def test_retry_budget_applies_to_embedded_retryable_error(monkeypatch):
    """A 200 body carrying an embedded 429 draws on the same budget."""
    _set_api_key(monkeypatch)
    _record_sleep(monkeypatch)
    embedded = _FakeResponse(200, json_data={"error": {"code": 429}})
    stub = _PostStub(
        [embedded for _ in range(8)]
        + [_FakeResponse(200, json_data={"choices": [{"message": {"content": "ok"}}]})],
    )
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(_budget_cfg(300.0), max_retries=3)
    assert client.complete([{"role": "user", "content": "x"}]) == "ok"
    assert len(stub.calls) == 9


def test_retry_budget_covers_truncated_json_body(monkeypatch):
    """The second regression: an HTTP 200 whose body will not parse.

    Production hit `Expecting value: line 1013 column 1` — a keepalive
    preamble from a connection held open against a throttled upstream — and
    it killed a 77-minute run that the 429 handling had just rescued. A
    truncated body is transport corruption, not a verdict on the request, so
    it draws on the same budget.
    """
    _set_api_key(monkeypatch)
    _record_sleep(monkeypatch)
    truncated = _FakeResponse(200, json_data=None, text="")
    truncated.json = _raise_value_error  # ty: ignore[invalid-assignment]
    stub = _PostStub(
        [truncated for _ in range(8)]
        + [_FakeResponse(200, json_data={"choices": [{"message": {"content": "ok"}}]})],
    )
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(_budget_cfg(300.0), max_retries=3)
    assert client.complete([{"role": "user", "content": "x"}]) == "ok"
    assert len(stub.calls) == 9


def test_truncated_json_body_still_raises_when_budget_runs_out(monkeypatch):
    """A body that never parses must still fail loudly, not spin forever."""
    _set_api_key(monkeypatch)
    rec = _record_sleep(monkeypatch)
    broken = _FakeResponse(200, json_data=None, text="")
    broken.json = _raise_value_error  # ty: ignore[invalid-assignment]
    stub = _PostStub([broken for _ in range(500)])
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(_budget_cfg(120.0), max_retries=3)
    with pytest.raises(
        AxBenchBaseError,
        match=r"invalid JSON body after \d+ attempts and \d+\.\ds of backoff",
    ):
        client.complete([{"role": "user", "content": "x"}])
    assert len(stub.calls) > 4
    assert sum(rec.delays) >= 120.0


def test_retry_budget_covers_transport_exceptions(monkeypatch):
    """Connection resets cluster with upstream congestion, so they get it too.

    Scoping the budget narrowly is what let the truncated-body path kill a
    run; a broken endpoint still fails, just after the budget rather than
    after 7s.
    """
    _set_api_key(monkeypatch)
    _record_sleep(monkeypatch)
    stub = _PostStub(
        [requests.ConnectionError("reset by peer") for _ in range(8)]
        + [_FakeResponse(200, json_data={"choices": [{"message": {"content": "ok"}}]})],
    )
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(_budget_cfg(300.0), max_retries=2)
    assert client.complete([{"role": "user", "content": "x"}]) == "ok"
    assert len(stub.calls) == 9


def test_transport_exception_still_raises_when_budget_runs_out(monkeypatch):
    """A genuinely dead endpoint fails loudly once the wall clock is spent."""
    _set_api_key(monkeypatch)
    rec = _record_sleep(monkeypatch)
    stub = _PostStub([requests.ConnectionError("dns failure") for _ in range(500)])
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(_budget_cfg(120.0), max_retries=2)
    with pytest.raises(
        AxBenchBaseError,
        match=r"request failed after \d+ attempts and \d+\.\ds of backoff",
    ):
        client.complete([{"role": "user", "content": "x"}])
    assert len(stub.calls) > 3
    assert sum(rec.delays) >= 120.0


def test_non_retryable_4xx_bypasses_the_budget_entirely(monkeypatch):
    """The one exclusion: a server verdict on the request itself.

    No amount of waiting makes a 400 valid, so it must fail on the first
    response rather than burning five minutes first.
    """
    _set_api_key(monkeypatch)
    rec = _record_sleep(monkeypatch)
    stub = _PostStub([_FakeResponse(400, text="bad request") for _ in range(5)])
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(_budget_cfg(300.0), max_retries=3)
    with pytest.raises(AxBenchBaseError, match="OpenRouter returned 400"):
        client.complete([{"role": "user", "content": "x"}])
    assert len(stub.calls) == 1
    assert rec.delays == []


def test_backoff_is_capped_and_jittered(monkeypatch):
    """Delays stay in [0.5, 1.0] x min(2**attempt, cap) — never unbounded."""
    _set_api_key(monkeypatch)
    rec = _record_sleep(monkeypatch)
    stub = _PostStub([_FakeResponse(429, text="rate limit") for _ in range(500)])
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(_budget_cfg(600.0), max_retries=3)
    with pytest.raises(AxBenchBaseError):
        client.complete([{"role": "user", "content": "x"}])
    assert rec.delays, "expected at least one backoff"
    for attempt, delay in enumerate(rec.delays):
        base = min(float(2 ** attempt), llm_judge_client._BACKOFF_CAP_SECONDS)
        assert 0.5 * base <= delay <= base, f"attempt {attempt}: {delay} vs base {base}"
    assert max(rec.delays) <= llm_judge_client._BACKOFF_CAP_SECONDS
    # Jitter must actually vary, or concurrent workers resume in lockstep.
    tail = rec.delays[-8:]
    assert len(set(tail)) > 1


class _FakeResponseWithHeaders(_FakeResponse):
    def __init__(self, status_code, headers, json_data=None, text="fake-text"):
        super().__init__(status_code, json_data=json_data, text=text)
        self.headers = headers


def test_retry_after_header_is_honored_exactly(monkeypatch):
    """A sane Retry-After overrides exponential backoff, unjittered."""
    _set_api_key(monkeypatch)
    rec = _record_sleep(monkeypatch)
    stub = _PostStub(
        [
            _FakeResponseWithHeaders(429, {"Retry-After": "17"}),
            _FakeResponse(200, json_data={"choices": [{"message": {"content": "ok"}}]}),
        ],
    )
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(_budget_cfg(300.0), max_retries=3)
    assert client.complete([{"role": "user", "content": "x"}]) == "ok"
    assert rec.delays == [17.0]


def test_retry_after_is_clamped(monkeypatch):
    """A hostile Retry-After must not park the worker indefinitely."""
    _set_api_key(monkeypatch)
    rec = _record_sleep(monkeypatch)
    stub = _PostStub(
        [
            _FakeResponseWithHeaders(429, {"Retry-After": "99999"}),
            _FakeResponse(200, json_data={"choices": [{"message": {"content": "ok"}}]}),
        ],
    )
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(_budget_cfg(300.0), max_retries=3)
    assert client.complete([{"role": "user", "content": "x"}]) == "ok"
    assert rec.delays == [llm_judge_client._MAX_RETRY_AFTER_SECONDS]


@pytest.mark.parametrize("raw", ["not-a-number", "", "-5", "nan", "Wed, 21 Oct 2015 07:28:00 GMT"])
def test_unusable_retry_after_falls_back_to_exponential(monkeypatch, raw):
    """Garbage, negative, NaN, and HTTP-date forms all fall back safely."""
    _set_api_key(monkeypatch)
    rec = _record_sleep(monkeypatch)
    stub = _PostStub(
        [
            _FakeResponseWithHeaders(429, {"Retry-After": raw}),
            _FakeResponse(200, json_data={"choices": [{"message": {"content": "ok"}}]}),
        ],
    )
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(_budget_cfg(300.0), max_retries=3)
    assert client.complete([{"role": "user", "content": "x"}]) == "ok"
    assert len(rec.delays) == 1
    assert 0.5 <= rec.delays[0] <= 1.0


@pytest.mark.parametrize("structured", [False, True])
def test_mixed_embedded_retryable_error_keeps_structured_response_type_and_legacy_text(monkeypatch, structured):
    _set_api_key(monkeypatch)
    _noop_sleep(monkeypatch)
    mixed = {"error": {"code": 502}, "choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": "partial"}}]}
    complete = {"choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": "finished"}}]}
    stub = _PostStub([_FakeResponse(200, mixed), _FakeResponse(200, complete)])
    _stub_post(monkeypatch, stub)
    client = OpenRouterLLMClient(LLMClientConfig(requests_per_minute=0,
                                               transport_retry_budget_seconds=0), max_retries=1)
    messages = [{"role": "user", "content": "original request"}]
    if structured:
        sampling = {"temperature": 1., "top_p": 1., "max_tokens": 20, "seed": 5}
        assert client.complete_response(messages, sampling=sampling) == complete
        assert len(stub.calls) == 2 and stub.calls[0]["json"] == stub.calls[1]["json"]
    else:
        assert client.complete(messages) == "partial"
        assert len(stub.calls) == 1
