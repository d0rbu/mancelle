"""Structured transport remains additive, typed, and metadata preserving."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import requests

from steering.eval.axbench_base import llm_judge_client
from steering.eval.axbench_base.config import LLMClientConfig
from steering.eval.axbench_base.llm_judge_client import OpenRouterLLMClient
from steering.eval.axbench_base.synth.llm_response_cache import SynthResponseCache
from steering.eval.axbench_negative import api as api_module
from steering.eval.axbench_negative.api import (
    ApiResponse,
    ChatApiClient,
    ConversationQualityJudge,
    QualityAssessment,
    SamplingConfig,
    parse_api_response,
)


def test_api_and_quality_identity_snapshot_semantic_version_changes(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-fixture")
    client = OpenRouterLLMClient(LLMClientConfig(model="requested", requests_per_minute=0),
                                 max_retries=0)
    first = ChatApiClient(client, SamplingConfig())
    initial = first.identity
    initial_quality = ConversationQualityJudge(first).identity
    assert initial["format"] == api_module.API_FORMAT
    assert initial["model"] == "requested"
    assert initial["sampling"] == SamplingConfig().to_dict()
    escaped = first.identity
    escaped["sampling"]["temperature"] = -1
    assert first.identity == initial
    monkeypatch.setattr(api_module, "API_FORMAT", "axbench-original-messages-api-v2")
    second = ChatApiClient(client, SamplingConfig())
    assert first.identity == initial  # Already-loaded clients retain their snapshot.
    assert ConversationQualityJudge(first).identity == initial_quality
    assert second.identity == {**initial, "format": api_module.API_FORMAT}
    assert ConversationQualityJudge(second).identity != initial_quality


@pytest.mark.parametrize("changes", [
    {"status": "invalid"}, {"content": 1}, {"response_sha256": "0" * 64},
    {"returned_model": "different"}, {"refusal": 0}, {"raw": []},
])
def test_persisted_response_reconstructs_derived_fields_from_raw(changes):
    response = parse_api_response({"model": "returned", "choices": [{
        "finish_reason": "stop", "message": {"role": "assistant", "content": "answer"},
    }]}, "requested")
    record = response.to_dict()
    assert ApiResponse.from_dict(record) == response
    with pytest.raises(ValueError):
        ApiResponse.from_dict({**record, **changes})


@pytest.mark.parametrize("provider", ["openrouter", "openai"])
def test_structured_transport_retries_exact_body_and_cache_does_not_alias_legacy(monkeypatch, tmp_path, provider):
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-fixture")
    monkeypatch.setenv("OPENAI_API_KEY", "offline-openai-fixture")
    monkeypatch.setattr(llm_judge_client.time, "sleep", lambda _seconds: None)
    raw = {"id": "real-response-id", "model": "actual-model", "provider": "actual-provider",
           "choices": [{"finish_reason": "length", "message": {"content": "unchanged response"}}]}
    replies = [SimpleNamespace(status_code=429, headers={}),
               SimpleNamespace(status_code=200, json=lambda: raw),
               SimpleNamespace(status_code=200, json=lambda: {
                   "choices": [{"message": {"content": "legacy"}}]})]
    calls = []

    def post(_session, _url, **kwargs):
        calls.append(copy.deepcopy(kwargs["json"]))
        return replies.pop(0)

    monkeypatch.setattr(requests.Session, "post", post)
    messages = [{"role": "system", "content": " System Ω \n"},
                {"role": "user", "content": "exact request\n"}]
    original = copy.deepcopy(messages)
    with SynthResponseCache(tmp_path / "cache.sqlite") as cache:
        cfg = LLMClientConfig(provider=provider, model="requested", requests_per_minute=0,
                              transport_retry_budget_seconds=0)
        llm = OpenRouterLLMClient(cfg, max_retries=1, response_cache=cache)
        api = ChatApiClient(llm, SamplingConfig(temperature=.75, top_p=.9, max_tokens=123, seed=15))
        first = api.complete(messages, cache_key="slot-zero")
        assert first.status == "truncated" and first.raw == raw
        assert first.returned_model == "actual-model" and first.response_id == "real-response-id"
        assert api.complete(messages, cache_key="slot-zero") == first
        assert calls[0] == calls[1]
        assert calls[0] == {"model": "requested", "messages": original,
                            **api.requested_sampling("slot-zero")}
        assert messages == original
        # The shared old API retains its exact payload and independently typed cache.
        assert llm.complete(messages, cache_key="slot-zero") == "legacy"
        assert calls[2] == {"model": "requested", "messages": original}
        assert llm.complete(messages, cache_key="slot-zero") == "legacy"
    assert len(calls) == 3


def test_provider_and_endpoint_bind_api_and_quality_identity(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-router")
    monkeypatch.setenv("OPENAI_API_KEY", "offline-direct")
    clients = [ChatApiClient(OpenRouterLLMClient(LLMClientConfig(
        provider=provider, model="same-requested-model", requests_per_minute=0)), SamplingConfig())
        for provider in ("openrouter", "openai")]
    router, direct = clients
    assert router.identity["provider"] == "openrouter"
    assert router.identity["endpoint"] == "https://openrouter.ai/api/v1/chat/completions"
    assert direct.identity["provider"] == "openai"
    assert direct.identity["endpoint"] == "https://api.openai.com/v1/chat/completions"
    assert router.identity != direct.identity
    assert ConversationQualityJudge(router).identity != ConversationQualityJudge(direct).identity
    assert "offline-router" not in json.dumps(router.identity)
    assert "offline-direct" not in json.dumps(direct.identity)


@pytest.mark.parametrize("raw", [
    {"choices": [{"finish_reason": "stop", "message": {"content": ["multipart"]}}]},
    {"choices": [{"message": {"content": "missing finish reason"}}]},
    {"choices": [{"finish_reason": "stop", "message": {"role": "user", "content": "wrong role"}}]},
    {"choices": []}, {"error": {"code": 400}},
])
def test_malformed_provider_responses_are_persisted_as_unknown_observations(monkeypatch, tmp_path, raw):
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-fixture")
    calls = []

    def post(_session, _url, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status_code=200, json=lambda: raw)

    monkeypatch.setattr(requests.Session, "post", post)
    with SynthResponseCache(tmp_path / "cache.sqlite") as cache:
        llm = OpenRouterLLMClient(LLMClientConfig(model="requested", requests_per_minute=0),
                                 max_retries=0, response_cache=cache)
        api = ChatApiClient(llm, SamplingConfig())
        result = api.complete([{"role": "user", "content": "original"}], cache_key="slot")
        assert result.status == "malformed" and result.raw == raw
        assert api.complete([{"role": "user", "content": "original"}], cache_key="slot") == result
    assert len(calls) == 1


@pytest.mark.parametrize("content", [
    '{"useful":true,"useful":false}',
    json.dumps({"useful": True, "refusal": True, "coherent": True, "substantive": True,
                "task_fulfilled": True, "reason": "contradiction"}),
    json.dumps({"useful": True, "refusal": False, "coherent": True, "substantive": True,
                "task_fulfilled": True, "reason": "extra", "concept": "leaked"}),
    '{"useful": NaN}',
])
def test_quality_never_promotes_malformed_or_contradictory_judgments(content):
    class FixtureApi(ChatApiClient):
        def __init__(self):
            self.sampling = SamplingConfig()

        @property
        def identity(self):
            return {"fixture": True}

        def complete(self, messages, *, cache_key):
            assert "concept" not in json.loads(messages[-1]["content"])
            return parse_api_response({"choices": [{"finish_reason": "stop",
                                                    "message": {"content": content}}]}, "quality")

    assessment = ConversationQualityJudge(FixtureApi()).assess(
        [{"role": "user", "content": "Original task"}], "Original response")
    assert not assessment.useful and assessment.status == "unavailable"
    assert assessment.api_response is not None
    assert assessment.api_response["content"] == content


@pytest.mark.parametrize(("finish", "refusal"), [("content_filter", None), ("stop", "provider refusal")])
def test_quality_provider_refusal_is_unavailable_even_with_valid_rubric_json(finish, refusal):
    content = json.dumps({"useful": True, "refusal": False, "coherent": True,
                          "substantive": True, "task_fulfilled": True, "reason": "valid rubric"})

    class RefusingApi(ChatApiClient):
        def __init__(self):
            self.sampling = SamplingConfig()

        @property
        def identity(self):
            return {"fixture": True}

        def complete(self, messages, *, cache_key):
            return parse_api_response({"choices": [{"finish_reason": finish,
                "message": {"role": "assistant", "content": content, "refusal": refusal}}]}, "quality")

    assessment = ConversationQualityJudge(RefusingApi()).assess(
        [{"role": "user", "content": "Original task"}], "Original response")
    assert assessment.status == "unavailable" and assessment.useful is False
    assert assessment.api_response is not None
    assert assessment.api_response["content"] == content
    assert assessment.api_response["refusal"] is True


def test_persisted_quality_cannot_promote_a_negative_raw_judgment():
    rubric = {"useful": False, "refusal": False, "coherent": True,
              "substantive": False, "task_fulfilled": False, "reason": "generic filler"}
    response = parse_api_response({"choices": [{"finish_reason": "stop",
        "message": {"role": "assistant", "content": json.dumps(rubric)}}]}, "quality")
    record = {**rubric, "api_response": response.to_dict(),
              "requested_sampling": SamplingConfig().to_dict()}
    assert QualityAssessment.from_dict(record).useful is False
    with pytest.raises(ValueError, match="contradict the preserved"):
        QualityAssessment.from_dict({**record, "useful": True, "substantive": True,
                                     "task_fulfilled": True})
    override = {**record, "refusal": True, "reason": "provider marked refusal; generic filler"}
    assert QualityAssessment.from_dict(override).useful is False


def test_persisted_quality_requires_explicit_valid_sampling():
    response = parse_api_response({"transport_error": "fixture timeout"}, "quality")
    value = QualityAssessment(False, False, False, False, False, "timeout", "unavailable",
                              response.to_dict(), {"max_tokens": 100}).to_dict()
    with pytest.raises(ValueError, match="explicit sampling"):
        QualityAssessment.from_dict(value)


def test_unavailable_quality_without_provider_observation_remains_replayable():
    assessment = QualityAssessment(False, False, False, False, False,
                                   "upstream call failed", "unavailable")
    assert QualityAssessment.from_dict(assessment.to_dict()) == assessment


@pytest.mark.parametrize("max_retries", [0, 1])
def test_exhausted_mixed_embedded_error_becomes_unknown_transport_observation(monkeypatch, max_retries):
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-fixture")
    monkeypatch.setattr(llm_judge_client.time, "sleep", lambda _seconds: None)
    raw = {"error": {"code": 502}, "choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": "partial"}}]}
    calls = []

    def post(_session, _url, **kwargs):
        calls.append(copy.deepcopy(kwargs["json"]))
        return SimpleNamespace(status_code=200, headers={}, json=lambda: raw)

    monkeypatch.setattr(requests.Session, "post", post)
    cfg = LLMClientConfig(model="requested", requests_per_minute=0, transport_retry_budget_seconds=0)
    api = ChatApiClient(OpenRouterLLMClient(cfg, max_retries=max_retries),
                        SamplingConfig(temperature=1, top_p=1, max_tokens=20))
    result = api.complete([{"role": "user", "content": "original request"}], cache_key="slot")
    assert result.status == "transport_error" and result.content is None
    assert "embedded retryable error 502" in result.raw["transport_error"]
    assert ApiResponse.from_dict(result.to_dict()) == result
    assert len(calls) == max_retries + 1 and all(payload == calls[0] for payload in calls)
