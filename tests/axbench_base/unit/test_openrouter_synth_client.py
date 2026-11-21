"""Unit tests for :class:`OpenRouterSynthClient` (Task 14 + review-fix 7)."""

from __future__ import annotations

import inspect

import pytest
import requests

from steering.eval.axbench_base import llm_judge_client
from steering.eval.axbench_base.config import LLMClientConfig
from steering.eval.axbench_base.synth import openrouter_client
from steering.eval.axbench_base.synth.llm_response_cache import SynthResponseCache
from steering.eval.axbench_base.synth.openrouter_client import (
    OpenRouterSynthClient,
)
from steering.eval.axbench_base.types import AxBenchBaseError


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-synth-456")


class _FakeResponse:
    """Minimal stand-in for ``requests.Response`` (end-to-end test)."""

    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json_data = json_data
        self.text = ""

    def json(self):
        return self._json_data


def test_synth_client_constructs_wrapped_llm_client(api_key, monkeypatch):
    """OpenRouterSynthClient constructs an underlying OpenRouterLLMClient."""
    captured: dict = {}

    class _SentinelLLMClient:
        def __init__(self, cfg, max_retries, response_cache=None, max_tokens=None):
            captured["cfg"] = cfg
            captured["max_retries"] = max_retries
            captured["response_cache"] = response_cache
            captured["max_tokens"] = max_tokens

        def complete(self, messages, *, cache_key=None):
            return "dummy"

    monkeypatch.setattr(
        openrouter_client,
        "OpenRouterLLMClient",
        _SentinelLLMClient,
    )
    cfg = LLMClientConfig(model="openai/gpt-5-nano", requests_per_minute=60)
    OpenRouterSynthClient(cfg, max_retries=5)
    assert captured["cfg"] is cfg
    assert captured["max_retries"] == 5
    assert captured["response_cache"] is None
    assert captured["max_tokens"] is None


def test_synth_client_complete_wraps_prompt_as_user_message(api_key, monkeypatch):
    """complete(prompt) forwards [{role:user, content:prompt}] to the client."""
    captured_messages: list = []

    class _RecordingLLMClient:
        def __init__(self, cfg, max_retries, response_cache=None, max_tokens=None):
            pass

        def complete(self, messages, *, cache_key=None):
            captured_messages.append(messages)
            return "the response"

    monkeypatch.setattr(
        openrouter_client,
        "OpenRouterLLMClient",
        _RecordingLLMClient,
    )
    client = OpenRouterSynthClient(LLMClientConfig())
    result = client.complete("hello world")
    assert result == "the response"
    assert captured_messages == [
        [{"role": "user", "content": "hello world"}],
    ]


def test_synth_client_complete_places_optional_system_message_first(
    api_key,
    monkeypatch,
):
    captured_messages: list = []

    class _RecordingLLMClient:
        def __init__(self, cfg, max_retries, response_cache=None, max_tokens=None):
            pass

        def complete(self, messages, *, cache_key=None):
            captured_messages.append(messages)
            return "the response"

    monkeypatch.setattr(
        openrouter_client,
        "OpenRouterLLMClient",
        _RecordingLLMClient,
    )
    client = OpenRouterSynthClient(LLMClientConfig())

    result = client.complete(
        "user contract",
        system_prompt="system contract",
    )

    assert result == "the response"
    assert captured_messages == [
        [
            {"role": "system", "content": "system contract"},
            {"role": "user", "content": "user contract"},
        ],
    ]


def test_synth_client_propagates_errors(api_key, monkeypatch):
    """AxBenchBaseError from the underlying client must propagate."""

    class _FailingLLMClient:
        def __init__(self, cfg, max_retries, response_cache=None, max_tokens=None):
            pass

        def complete(self, messages, *, cache_key=None):
            raise AxBenchBaseError("openrouter down")

    monkeypatch.setattr(
        openrouter_client,
        "OpenRouterLLMClient",
        _FailingLLMClient,
    )
    client = OpenRouterSynthClient(LLMClientConfig())
    with pytest.raises(AxBenchBaseError, match="openrouter down"):
        client.complete("hi")


def test_synth_client_returns_content_end_to_end(api_key, monkeypatch):
    """Full path with real OpenRouterLLMClient but stubbed ``requests.post``."""

    def _fake_post(url, *, json, headers, timeout):
        return _FakeResponse(
            200,
            {"choices": [{"message": {"content": "synth completion output"}}]},
        )

    monkeypatch.setattr(
        requests.Session,
        "post",
        lambda _self, *args, **kwargs: _fake_post(*args, **kwargs),
    )
    monkeypatch.setattr(llm_judge_client.time, "sleep", lambda s: None)

    client = OpenRouterSynthClient(LLMClientConfig(requests_per_minute=60))
    assert client.complete("make a continuation") == "synth completion output"


def test_synth_client_default_max_retries_is_three():
    """The default transport retry budget is three additional HTTP attempts."""
    sig = inspect.signature(OpenRouterSynthClient.__init__)
    assert sig.parameters["max_retries"].default == 3


def test_transport_failure_stops_after_three_retries_with_backoff(
    api_key,
    monkeypatch,
):
    """Persistent 503s make 4 HTTP calls and sleep only 1s, 2s, and 4s."""
    post_calls: list[dict] = []
    sleeps: list[float] = []

    def _post(_self, url, *, json, headers, timeout):
        post_calls.append(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return _FakeResponse(503, {"error": {"message": "busy"}})

    monkeypatch.setattr(requests.Session, "post", _post)
    monkeypatch.setattr(llm_judge_client.time, "sleep", sleeps.append)

    client = OpenRouterSynthClient(
        LLMClientConfig(requests_per_minute=0),
        max_retries=3,
    )
    with pytest.raises(AxBenchBaseError, match="503 after 4 attempts"):
        client.complete("make a continuation")

    assert len(post_calls) == 4
    # Exponential shape per attempt, jittered into [0.5, 1.0] of the base so
    # concurrent workers do not resume in lockstep. ``LLMClientConfig`` leaves
    # the wall-clock retry budget at 0, so attempt-count semantics still
    # decide when to give up — only ``SynthConfig`` opts into the longer wait.
    assert len(sleeps) == 3
    for base, delay in zip([1.0, 2.0, 4.0], sleeps, strict=True):
        assert 0.5 * base <= delay <= base


def test_synth_client_forwards_provider_max_tokens(api_key, monkeypatch):
    captured: dict = {}

    class _SentinelLLMClient:
        def __init__(self, cfg, max_retries, response_cache=None, max_tokens=None):
            captured["max_tokens"] = max_tokens

        def complete(self, messages, *, cache_key=None):
            return "dummy"

    monkeypatch.setattr(
        openrouter_client,
        "OpenRouterLLMClient",
        _SentinelLLMClient,
    )
    OpenRouterSynthClient(LLMClientConfig(), max_tokens=384)
    assert captured["max_tokens"] == 384


# ---------------------------------------------------------------------------
# Response cache forwarding (review-fix 7)
# ---------------------------------------------------------------------------


def test_synth_client_forwards_cache_to_underlying_client(
    api_key,
    monkeypatch,
    tmp_path,
):
    """The SynthClient must forward the response_cache to its underlying client."""
    captured: dict = {}

    class _SentinelLLMClient:
        def __init__(self, cfg, max_retries, response_cache=None, max_tokens=None):
            captured["response_cache"] = response_cache

        def complete(self, messages, *, cache_key=None):
            return "dummy"

    monkeypatch.setattr(
        openrouter_client,
        "OpenRouterLLMClient",
        _SentinelLLMClient,
    )
    cache = SynthResponseCache(tmp_path / "c.sqlite")
    try:
        OpenRouterSynthClient(LLMClientConfig(), response_cache=cache)
        assert captured["response_cache"] is cache
    finally:
        cache.close()


def test_synth_client_complete_forwards_cache_key(api_key, monkeypatch):
    """complete(prompt, cache_key=...) must forward cache_key to the underlying call."""
    captured: dict = {}

    class _RecordingLLMClient:
        def __init__(self, cfg, max_retries, response_cache=None, max_tokens=None):
            pass

        def complete(self, messages, *, cache_key=None):
            captured["cache_key"] = cache_key
            captured["messages"] = messages
            return "ok"

    monkeypatch.setattr(
        openrouter_client,
        "OpenRouterLLMClient",
        _RecordingLLMClient,
    )
    client = OpenRouterSynthClient(LLMClientConfig())
    result = client.complete("hello", cache_key="forwarded-key")
    assert result == "ok"
    assert captured["cache_key"] == "forwarded-key"
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
