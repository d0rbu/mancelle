"""Focused safety and protocol tests for the shared offline OpenRouter wire."""

from __future__ import annotations

import pytest
import requests

from steering.eval.axbench_base import openrouter
from steering.eval.axbench_base.config import LLMClientConfig
from steering.eval.axbench_base.llm_judge_client import OpenRouterLLMClient
from steering.eval.axbench_base.openrouter import get_openrouter_session
from tests.axbench_base.fixtures.offline_openrouter import (
    OfflineOpenRouterError,
    OfflineOpenRouterTransport,
    OpenRouterRequest,
    classify_cache_key,
)


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        (
            "[[AXBENCH_DEFINITION_BEGIN_A]]definition[[AXBENCH_DEFINITION_END_A]]",
            "stage0c_definition",
        ),
        (
            "[[AXBENCH_CONTINUATION_BEGIN_A]]suffix[[AXBENCH_CONTINUATION_END_A]]",
            "stage0c_continuation",
        ),
        ("[[AXBENCH_STAGE1B_RICH_BEGIN_A]]", "stage1_synth_rich"),
        ("[[AXBENCH_STAGE1B_AVOID_BEGIN_A]]", "stage1_synth_avoid"),
        ("[[AXBENCH_STAGE1C_AVOID_BEGIN_A]]", "stage1_synth_avoid"),
        ("Complete the content by adding elements that are related to 'x'.", "stage1_synth_rich"),
        ("gracefully transitions away from 'x'.", "stage1_synth_avoid"),
        ("Return a rating: [[1]]", "stage5_judge"),
    ],
)
def test_request_classification_uses_stable_production_signals(
    prompt: str,
    expected: str,
) -> None:
    assert classify_cache_key(None, prompt) == expected


def test_install_clears_cached_session_before_and_after_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = get_openrouter_session()
    transport = OfflineOpenRouterTransport(lambda _request: "ok")
    transport.install(monkeypatch)
    assert old is not get_openrouter_session()
    transport.close()
    assert not hasattr(openrouter._thread_local, "session")


def test_session_post_reaches_recorded_production_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-offline")
    transport = OfflineOpenRouterTransport(lambda _request: "offline answer")
    transport.install(monkeypatch)
    transport.bind_production_clients(monkeypatch)

    client = OpenRouterLLMClient(
        LLMClientConfig(requests_per_minute=0),
        max_retries=0,
    )
    assert client.complete(
        [
            {
                "role": "user",
                "content": "Complete the content by adding elements related to x.",
            }
        ]
    ) == "offline answer"
    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert request.url == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer sk-or-test-offline"
    assert request.timeout == (10, 60)
    assert request.message_roles == ("user",)
    assert request.response_hash is not None
    assert request.handled
    assert transport.post_mock.call_count == 1
    assert transport.top_level_calls == []
    assert transport.transport_tripwire_calls == []
    transport.close()


def test_module_and_lower_level_boundaries_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = OfflineOpenRouterTransport(lambda _request: "unused")
    transport.install(monkeypatch)
    with pytest.raises(OfflineOpenRouterError, match="requests.post"):
        requests.post("https://openrouter.ai/api/v1/chat/completions")
    with pytest.raises(OfflineOpenRouterError, match="Session.get"):
        requests.Session().get("https://openrouter.ai/api/v1/chat/completions")
    with pytest.raises(OfflineOpenRouterError, match="Session.request"):
        requests.Session().request("GET", "https://openrouter.ai/api/v1/chat/completions")
    with pytest.raises(OfflineOpenRouterError, match="Session.send"):
        requests.Session().send(
            requests.Request(
                "GET", "https://openrouter.ai/api/v1/chat/completions"
            ).prepare()
        )
    assert len(transport.top_level_calls) == 1
    assert len(transport.transport_tripwire_calls) == 3
    transport.close()


def test_unknown_request_class_fails_with_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_unknown(request: OpenRouterRequest) -> str:
        if request.request_class is None:
            raise OfflineOpenRouterError(
                f"unknown request class ordinal={request.ordinal} "
                f"url={request.url} model={request.model} "
                f"cache_key_prefix={request.cache_key_prefix} "
                f"message_roles={request.message_roles} "
                f"prompt_hash={request.prompt_hash[:16]}"
            )
        return "ok"

    transport = OfflineOpenRouterTransport(reject_unknown)
    transport.install(monkeypatch)
    session = requests.Session()
    with pytest.raises(OfflineOpenRouterError, match="unknown request class"):
        session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "x"}]},
            headers={"Authorization": "Bearer dummy"},
            timeout=(10, 60),
        )
    assert len(transport.requests) == 1
    assert transport.requests[0].handled is False
    transport.close()


def test_response_has_exact_client_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = OfflineOpenRouterTransport(lambda _request: "Rating: [[1]]")
    transport.install(monkeypatch)
    response = requests.Session().post(
        "https://openrouter.ai/api/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Return rating: [[1]]"}],
        },
        headers={},
        timeout=(10, 60),
    )
    assert response.status_code == 200
    assert response.json() == {"choices": [{"message": {"content": "Rating: [[1]]"}}]}
    assert response.text == "Rating: [[1]]"
    response.raise_for_status()
    transport.close()


def test_cleanup_runs_when_client_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-offline")
    transport = OfflineOpenRouterTransport(
        lambda _request: (_ for _ in ()).throw(OfflineOpenRouterError("boom"))
    )
    transport.install(monkeypatch)
    client = OpenRouterLLMClient(LLMClientConfig(requests_per_minute=0), max_retries=0)
    with pytest.raises(OfflineOpenRouterError, match="boom"):
        client.complete(
            [
                {
                    "role": "user",
                    "content": "Complete the content by adding elements related to x.",
                }
            ]
        )
    transport.close()
    assert not hasattr(openrouter._thread_local, "session")
