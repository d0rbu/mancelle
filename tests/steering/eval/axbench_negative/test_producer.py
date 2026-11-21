"""Offline request-through-publication tests using real source identities and cache."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import requests

from steering.eval.axbench_base import llm_judge_client
from steering.eval.axbench_base.config import LLMClientConfig
from steering.eval.axbench_base.llm_judge_client import OpenRouterLLMClient
from steering.eval.axbench_base.synth.llm_response_cache import SynthResponseCache
from steering.eval.axbench_negative import api as negative_api
from steering.eval.axbench_negative import producer
from steering.eval.axbench_negative.api import (
    AVOIDANCE_INSTRUCTION,
    QUALITY_RUBRIC,
    ChatApiClient,
    ConversationQualityJudge,
    SamplingConfig,
    canonical_json,
    parse_api_response,
)
from steering.eval.axbench_negative.artifacts import load_negative_source, read_record
from steering.eval.axbench_negative.producer import (
    ProducerConfig,
    build_negative_source,
    neutral_sample_id,
)
from steering.eval.axbench_negative.qc import ChatAssessment
from steering.eval.axbench_negative.source import (
    FeatureTarget,
    iter_assistant_prefixes,
    value_sha256,
)
from steering.immutable_io import ImmutableArtifactError


def _raw(text: Any, finish: str = "stop", *, refusal: str | None = None) -> dict[str, Any]:
    return {"id": "response-id", "model": "returned-model", "provider": "fixture-provider",
            "choices": [{"finish_reason": finish,
                         "message": {"role": "assistant", "content": text, "refusal": refusal}}]}


class FakeScorer:
    identity = {"extractor": "offline-exact-span-fixture", "threshold": 0}

    def __init__(self) -> None:
        self.prefix_calls = 0
        self.completion_calls = 0

    @staticmethod
    def result(active: bool) -> ChatAssessment:
        return ChatAssessment("active" if active else "inactive", 1.0 if active else 0.0,
                              int(active), 2, 4, "fixed-prepared-digest", 50.0 if active else None)

    def assess_prefix(self, candidate, targets):
        self.prefix_calls += 1
        return {target.feature_id: self.result(True) for target in targets}

    def assess_completion(self, candidate, response, target):
        self.completion_calls += 1
        if response == "ERROR":
            raise ValueError("fixture extraction failure")
        return self.result("HIT" in response)


class FakeHttp:
    def __init__(self, neutral, avoiding=None):
        self.neutral = list(neutral)
        self.avoiding = list(avoiding or [_raw("SAFE detailed answer")])
        self.calls = []

    def __call__(self, _session, _url, *, json: dict[str, Any], **_kwargs):
        self.calls.append(json)
        model = json["model"]
        if model == "neutral":
            raw = self.neutral.pop(0)
        elif model == "avoiding":
            raw = self.avoiding.pop(0)
        else:
            assert model == "quality"
            assert json["messages"][0]["content"] == QUALITY_RUBRIC
            data = json_load(json["messages"][1]["content"])
            assert set(data) == {"conversation", "response"}
            good = "REFUSE" not in data["response"] and "FILLER" not in data["response"]
            raw = _raw(canonical_json({"useful": good, "refusal": not good, "coherent": True,
                                      "substantive": good, "task_fulfilled": good, "reason": "fixture"}))
        if isinstance(raw, Exception):
            raise raw
        return type("Response", (), {"status_code": 200, "json": lambda _self: raw})()


json_load = json.loads


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-fixture-key")
    monkeypatch.setattr(llm_judge_client.time, "sleep", lambda _seconds: None)
    candidate = list(iter_assistant_prefixes({
        "source_id": "real-source-0", "group_id": "family-0", "split": "test", "source_row": 0,
        "messages": [{"role": "system", "content": "  Original résumé system.\n"},
                     {"role": "user", "content": "First request"},
                     {"role": "assistant", "content": "Earlier supplied answer"},
                     {"role": "user", "content": "Original final request? \n"}],
    }))[-1]
    cache = SynthResponseCache(tmp_path / "responses.sqlite")

    def make(neutral, *, avoiding=None, targets=None, scorer=None, config=None):
        http = FakeHttp(neutral, avoiding)
        monkeypatch.setattr(requests.Session, "post", lambda session, *a, **kw: http(session, *a, **kw))

        def api(model):
            client = OpenRouterLLMClient(LLMClientConfig(
                model=model, requests_per_minute=0, transport_retry_budget_seconds=0),
                max_retries=0, response_cache=cache)
            return ChatApiClient(client, SamplingConfig(seed=123, max_tokens=100))

        kwargs = {"candidates": [candidate], "targets": targets or [FeatureTarget(3, (3, 4), "private concept")],
                  "scorer": scorer or FakeScorer(), "neutral_api": api("neutral"),
                  "avoiding_api": api("avoiding"), "quality_judge": ConversationQualityJudge(api("quality")),
                  "config": config or ProducerConfig(n_verification_seeds=3, min_consistency_passes=2,
                                                       max_concurrent=1, avoid_max_attempts=1),
                  "output_root": tmp_path / "bank", "semantic_identity": {"source": "offline-fixture"}}
        return kwargs, http

    yield make
    cache.close()


def test_original_messages_cache_refusal_and_strict_references(setup):
    kwargs, http = setup([_raw("HIT useful answer"), _raw("HIT REFUSE", refusal="refused"), _raw("neutral")])
    bank = build_negative_source(**kwargs)
    assert len(bank.groups) == 1
    assert bank.groups[0].rich_responses == ("HIT useful answer",)
    assert bank.groups[0].avoiding_response == "SAFE detailed answer"
    assert bank.groups[0].member_ids == (3, 4)
    assert bank.counts["neutral_admitted_pairs"] == 1
    quality_evidence = [json.loads(path.read_text())["quality"] for path in (bank.root / "quality").glob("*.json")]
    assert all(value["api_response"]["requested_model"] == "quality" for value in quality_evidence)
    assert all(value["api_response"]["returned_model"] == "returned-model" for value in quality_evidence)
    assert all(value["requested_sampling"]["max_tokens"] == 100 for value in quality_evidence)
    candidate = kwargs["candidates"][0]
    neutral = [call for call in http.calls if call["model"] == "neutral"]
    assert len(neutral) == 3
    assert all(call["messages"] == list(candidate.messages) for call in neutral)
    assert len({call["seed"] for call in neutral}) == 3
    assert all(set(call) == {"model", "messages", "temperature", "top_p", "seed", "max_tokens"} for call in neutral)
    avoiding = next(call for call in http.calls if call["model"] == "avoiding")
    assert avoiding["messages"][0]["content"] == AVOIDANCE_INSTRUCTION
    assert json.loads(avoiding["messages"][1]["content"])["conversation"] == list(candidate.messages)
    assert load_negative_source(bank.root).manifest_sha256 == bank.manifest_sha256
    previous_calls = len(http.calls)
    prefix_calls = kwargs["scorer"].prefix_calls
    completion_calls = kwargs["scorer"].completion_calls
    replay = build_negative_source(**kwargs)
    assert replay == bank
    assert len(http.calls) == previous_calls
    assert kwargs["scorer"].prefix_calls == prefix_calls
    assert kwargs["scorer"].completion_calls == completion_calls


@pytest.mark.parametrize("third", [_raw(""), _raw("HIT shortened", "length")])
def test_invalid_draw_is_fixed_nonhit_and_does_not_resample(setup, third):
    kwargs, http = setup([_raw("HIT answer one"), _raw("HIT answer two"), third])
    bank = build_negative_source(**kwargs)
    assert len(bank.groups) == 1
    assert sum(call["model"] == "neutral" for call in http.calls) == 3
    decisions = [read_record(bank.root, p.relative_to(bank.root).as_posix())
                 for p in (bank.root / "decisions").glob("*.json")]
    assert decisions[0] is not None
    assert decisions[0]["hits"] == 2 and decisions[0]["unknown"] == 0
    assert len(bank.groups[0].rich_responses) == 2
    outcome = read_record(bank.root, decisions[0]["assessment_paths"][2])
    assert outcome is not None and outcome["hit"] is False
    expected = "active" if third["choices"][0]["finish_reason"] == "length" else "inactive"
    assert outcome["diagnostic_assessment"]["status"] == expected


@pytest.mark.parametrize("third", [
    {"choices": [{"message": {"content": "HIT"}}]},
    {"choices": [{"finish_reason": "length", "message": {}}]},
    _raw([{"text": "HIT"}]), _raw("ERROR"), requests.ConnectionError("fixture failure"),
])
def test_unknown_blocks_admission_and_is_immutable_on_replay(setup, third):
    kwargs, http = setup([_raw("HIT one"), _raw("HIT two"), third])
    bank = build_negative_source(**kwargs)
    assert bank.groups == () and bank.counts["unknown_pairs"] == 1
    assert len(http.calls) == 3
    assert build_negative_source(**kwargs) == bank
    assert len(http.calls) == 3


def test_all_refusals_count_raw_hits_but_never_trigger_avoidance(setup):
    kwargs, http = setup([_raw("HIT REFUSE") for _ in range(3)])
    bank = build_negative_source(**kwargs)
    assert bank.groups == ()
    assert bank.counts["neutral_admitted_pairs"] == 1
    assert bank.counts["rich_quality_rejected_pairs"] == 1
    assert not any(call["model"] == "avoiding" for call in http.calls)


@pytest.mark.parametrize(("k", "m", "hits", "admitted"), [(1, 1, 1, True), (3, 3, 2, False), (3, 3, 3, True)])
def test_exact_admission_boundaries_keep_duplicate_draws(setup, k, m, hits, admitted):
    kwargs, http = setup([_raw("HIT repeated answer")] * hits + [_raw("ordinary")] * (k - hits),
                         config=ProducerConfig(k, m, 1, 1))
    bank = build_negative_source(**kwargs)
    assert bool(bank.groups) == admitted
    assert sum(call["model"] == "neutral" for call in http.calls) == k
    if admitted:
        assert len(bank.groups[0].rich_sample_ids) == hits
        assert len(set(bank.groups[0].rich_sample_ids)) == hits
        assert len(set(bank.groups[0].rich_responses)) == 1


def test_feature_unions_share_neutral_draws_and_target_reordering(setup):
    targets = [FeatureTarget(9, (9,), "unrelated other concept"), FeatureTarget(3, (3, 4), "private concept")]
    kwargs, http = setup([_raw("HIT answer")] * 3, targets=targets,
                         avoiding=[_raw("SAFE answer")] * 2)
    bank = build_negative_source(**kwargs)
    assert len(bank.groups) == 2
    assert kwargs["scorer"].prefix_calls == 1
    assert sum(call["model"] == "neutral" for call in http.calls) == 3
    assert bank.groups[0].rich_sample_ids == bank.groups[1].rich_sample_ids
    kwargs["targets"] = targets[::-1]
    assert build_negative_source(**kwargs) == bank


def test_prefix_outcomes_preserve_order_and_cap_without_sampling_ineligible_rows(setup):
    class PrefixScorer(FakeScorer):
        def assess_prefix(self, candidate, targets):
            self.prefix_calls += 1
            if candidate.source_row == 0:
                raise ValueError("prefix extraction failed")
            assessment = (ChatAssessment("no_tokens", 0.0, 0, 0, 4, "prefix-no-tokens")
                          if candidate.source_row == 2 else self.result(candidate.source_row >= 3))
            return {target.feature_id: assessment for target in targets}

    kwargs, http = setup([_raw("HIT first"), _raw("HIT second")],
                         avoiding=[_raw("SAFE first"), _raw("SAFE second")],
                         scorer=PrefixScorer(), config=ProducerConfig(1, 1, 1, 1, 2))
    kwargs["candidates"] = [next(iter_assistant_prefixes({
        "source_id": f"source-{index}", "group_id": f"family-{index}", "split": "test",
        "source_row": index, "messages": [{"role": "user", "content": f"Original task {index}"}],
    })) for index in range(6)]
    bank = build_negative_source(**kwargs)
    assert [group.source_row for group in bank.groups] == [3, 4]
    assert bank.counts == {
        "candidates": 6, "targets": 1, "prefix_unknown": 1, "screened_pairs": 2,
        "neutral_admitted_pairs": 2, "unknown_pairs": 0, "rich_quality_rejected_pairs": 0,
        "avoidance_rejected_pairs": 0, "neutral_samples": 2, "final_groups": 2,
    }
    assert kwargs["scorer"].prefix_calls == 6
    neutral = [call for call in http.calls if call["model"] == "neutral"]
    assert [call["messages"] for call in neutral] == [
        list(kwargs["candidates"][index].messages) for index in (3, 4)
    ]
    assert load_negative_source(bank.root) == bank
    before = (len(http.calls), kwargs["scorer"].prefix_calls, kwargs["scorer"].completion_calls)
    assert build_negative_source(**kwargs) == bank
    assert (len(http.calls), kwargs["scorer"].prefix_calls, kwargs["scorer"].completion_calls) == before


def test_identical_messages_from_distinct_rows_keep_separate_sample_associations(setup):
    targets = [FeatureTarget(3, (3, 4), "private concept"), FeatureTarget(9, (9,), "another concept")]
    kwargs, http = setup([_raw("HIT one"), _raw("HIT two"), _raw("ordinary")] * 2,
                         targets=targets, avoiding=[_raw("SAFE answer")] * 4)
    first = kwargs["candidates"][0]
    second = list(iter_assistant_prefixes({
        "source_id": "real-source-1", "group_id": "family-1", "split": first.split,
        "source_row": 1, "messages": list(first.messages),
    }))[-1]
    assert first.messages_sha256 == second.messages_sha256 and first.candidate_id != second.candidate_id
    kwargs["candidates"].append(second)
    bank = build_negative_source(**kwargs)
    assert bank.counts["candidates"] == 2 and bank.counts["screened_pairs"] == 4
    assert bank.counts["neutral_samples"] == 6 and bank.counts["final_groups"] == 4
    neutral = [call for call in http.calls if call["model"] == "neutral"]
    assert len(neutral) == 6 and all(call["messages"] == list(first.messages) for call in neutral)
    all_sample_ids = set()
    for candidate in (first, second):
        ids = {neutral_sample_id(candidate, kwargs["neutral_api"], slot) for slot in range(3)}
        assert not ids & all_sample_ids
        all_sample_ids.update(ids)
        for sid in ids:
            record = read_record(bank.root, f"samples/{sid[:2]}/{sid}.json")
            assert record is not None and record["candidate_id"] == candidate.candidate_id
        groups = [group for group in bank.groups if group.candidate_id == candidate.candidate_id]
        assert len(groups) == 2 and groups[0].rich_sample_ids == groups[1].rich_sample_ids
        assert set(groups[0].rich_sample_ids) <= ids
    assert len(list((bank.root / "samples").glob("*/*.json"))) == len(all_sample_ids) == 6
    assert load_negative_source(bank.root) == bank
    calls, prefixes, completions = len(http.calls), kwargs["scorer"].prefix_calls, kwargs["scorer"].completion_calls
    assert build_negative_source(**kwargs) == bank
    assert (len(http.calls), kwargs["scorer"].prefix_calls, kwargs["scorer"].completion_calls) == (calls, prefixes, completions)


def test_partial_replay_reuses_all_committed_slots(setup, monkeypatch):
    kwargs, http = setup([_raw("HIT one"), _raw("HIT two"), _raw("none")])
    publish = producer.publish_negative_source
    monkeypatch.setattr(producer, "publish_negative_source", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError, match="crash"):
        build_negative_source(**kwargs)
    assert not (kwargs["output_root"] / "manifest.json").exists()
    calls = len(http.calls)
    completion_calls = kwargs["scorer"].completion_calls
    monkeypatch.setattr(producer, "publish_negative_source", publish)
    assert len(build_negative_source(**kwargs).groups) == 1
    assert len(http.calls) == calls
    assert kwargs["scorer"].completion_calls == completion_calls


def test_changed_contract_rejected_before_calls(setup):
    kwargs, http = setup([_raw("HIT one"), _raw("HIT two"), _raw("none")])
    build_negative_source(**kwargs)
    calls = len(http.calls)
    kwargs["config"] = replace(kwargs["config"], min_consistency_passes=1)
    with pytest.raises(ImmutableArtifactError, match="refusing to overwrite"):
        build_negative_source(**kwargs)
    assert len(http.calls) == calls


@pytest.mark.parametrize("change", ["protocol", "draw_count", "sampling", "description", "quality_rubric"])
@pytest.mark.parametrize("completed", [False, True])
def test_changed_source_semantics_reject_partial_and_complete_resume(setup, monkeypatch, change, completed):
    kwargs, http = setup([_raw("HIT one"), _raw("HIT two"), _raw("none")])
    bank = build_negative_source(**kwargs)
    if not completed:
        (bank.root / "manifest.json").unlink()
    before = (len(http.calls), kwargs["scorer"].prefix_calls, kwargs["scorer"].completion_calls)
    if change == "protocol":
        monkeypatch.setattr(producer, "SOURCE_FORMAT", "changed-source-protocol")
    elif change == "draw_count":
        kwargs["config"] = replace(kwargs["config"], n_verification_seeds=4)
    elif change == "sampling":
        api = kwargs["neutral_api"]
        kwargs["neutral_api"] = ChatApiClient(api._client, replace(api.sampling, temperature=0.5))
    elif change == "description":
        kwargs["targets"] = [replace(kwargs["targets"][0], description="different concept")]
    else:
        monkeypatch.setattr(negative_api, "QUALITY_RUBRIC", QUALITY_RUBRIC + " Require factual correctness.")
    with pytest.raises(ImmutableArtifactError, match="refusing to overwrite"):
        build_negative_source(**kwargs)
    assert (len(http.calls), kwargs["scorer"].prefix_calls, kwargs["scorer"].completion_calls) == before


def test_caller_cannot_override_source_population_identity(setup):
    kwargs, http = setup([])
    kwargs["semantic_identity"] = {"source_population_sha256": "caller-value"}
    with pytest.raises(ValueError, match="producer-owned"):
        build_negative_source(**kwargs)
    assert http.calls == []


def test_mutated_nested_original_message_is_rejected_before_provider_call(setup):
    class MutatingScorer(FakeScorer):
        def assess_prefix(self, candidate, targets):
            result = super().assess_prefix(candidate, targets)
            candidate.messages[-1]["content"] = "changed after source identity was frozen"
            return result

    kwargs, http = setup([_raw("HIT answer")] * 3, scorer=MutatingScorer())
    with pytest.raises(ValueError, match="messages differ from their source identity"):
        build_negative_source(**kwargs)
    assert http.calls == []


def _corrupt_with_updated_hash(root: Path, relative: str, mutate) -> None:
    path = root / relative
    value = json.loads(path.read_text())
    mutate(value)
    payload = (canonical_json(value) + "\n").encode()
    path.write_bytes(payload)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][relative] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    manifest_path.write_text(canonical_json(manifest) + "\n")


@pytest.mark.parametrize("field,value", [
    ("source_id", "another source"), ("group_id", "another family"), ("split", "validation"),
    ("source_row", False), ("cut_message_count", 3), ("description", "another concept"),
    ("member_ids", [3, 5]), ("avoiding_response", "Unobserved avoiding response"),
    ("candidate_id", []), ("feature_id", True), ("member_ids", []), ("member_ids", [3, 3]),
    ("member_ids", [False, 3]), ("member_ids", [4]), ("rich_sample_ids", None),
    ("rich_sample_ids", []), ("rich_responses", []), ("avoiding_response", ""),
    ("messages_sha256", "0" * 64),
])
def test_loader_binds_reference_metadata_and_avoiding_text_to_evidence(setup, field, value):
    kwargs, _http = setup([_raw("HIT one"), _raw("HIT two"), _raw("none")])
    bank = build_negative_source(**kwargs)
    _corrupt_with_updated_hash(bank.root, "references.json", lambda record: record["groups"][0].update({field: value}))
    with pytest.raises(ValueError, match="reference"):
        load_negative_source(bank.root)


@pytest.mark.parametrize("change", ["history", "rich_text", "rich_subset", "duplicate_rich_id", "duplicate_group"])
def test_loader_rejects_changed_reference_population_even_with_updated_hash(setup, change):
    kwargs, _http = setup([_raw("HIT one"), _raw("HIT two"), _raw("none")])
    bank = build_negative_source(**kwargs)

    def mutate(record):
        group = record["groups"][0]
        if change == "history":
            group["messages"][0]["content"] = "Changed original system instruction"
            group["messages_sha256"] = value_sha256(group["messages"])
        elif change == "rich_text":
            group["rich_responses"][0] = "Unobserved rich response"
        elif change == "rich_subset":
            group["rich_sample_ids"].pop()
            group["rich_responses"].pop()
        elif change == "duplicate_rich_id":
            group["rich_sample_ids"][-1] = group["rich_sample_ids"][0]
        else:
            record["groups"].append(dict(group))

    _corrupt_with_updated_hash(bank.root, "references.json", mutate)
    with pytest.raises(ValueError, match="reference"):
        load_negative_source(bank.root)


@pytest.mark.parametrize("stage,field,value", [
    ("decisions", "k", True), ("decisions", "m", True), ("avoidance", "accepted_attempt", False),
])
def test_loader_rejects_boolean_decision_counts_and_avoidance_index(setup, stage, field, value):
    kwargs, _http = setup([_raw("HIT one")], config=ProducerConfig(1, 1, 1, 1))
    bank = build_negative_source(**kwargs)
    relative = next((bank.root / stage).glob("*.json")).relative_to(bank.root).as_posix()
    _corrupt_with_updated_hash(bank.root, relative, lambda record: record.update({field: value}))
    with pytest.raises(ValueError, match="fixed-k|index"):
        load_negative_source(bank.root)


def test_loader_checks_rejected_population_not_only_published_groups(setup):
    kwargs, _http = setup([_raw("none")] * 3)
    bank = build_negative_source(**kwargs)
    assert not bank.groups
    relative = next((bank.root / "decisions").glob("*.json")).relative_to(bank.root).as_posix()
    _corrupt_with_updated_hash(bank.root, relative, lambda v: v.update(hits=2, status="admitted"))
    with pytest.raises(ValueError, match="fixed-k"):
        load_negative_source(bank.root)


def test_loader_rejects_provider_metadata_relabeling_even_with_updated_hash(setup):
    kwargs, _http = setup([_raw("HIT one"), _raw("HIT two"), _raw("", "length")])
    bank = build_negative_source(**kwargs)
    sid = neutral_sample_id(kwargs["candidates"][0], kwargs["neutral_api"], 2)
    _corrupt_with_updated_hash(bank.root, f"samples/{sid[:2]}/{sid}.json",
                              lambda v: v["response"].update(status="complete", content="HIT fake"))
    with pytest.raises(ValueError, match="contradict"):
        load_negative_source(bank.root)


@pytest.mark.parametrize("stage", ["rich", "avoidance"])
@pytest.mark.parametrize("remove", [False, True])
def test_loader_requires_preserved_provider_evidence_for_assessed_quality(setup, stage, remove):
    kwargs, _http = setup([_raw("HIT one"), _raw("HIT two"), _raw("ordinary")])
    bank = build_negative_source(**kwargs)

    def strip_evidence(quality):
        if remove:
            quality.pop("api_response")
            quality.pop("requested_sampling")
        else:
            quality.update(api_response=None, requested_sampling=None)

    if stage == "rich":
        relative = next((bank.root / "quality").glob("*.json")).relative_to(bank.root).as_posix()
        _corrupt_with_updated_hash(bank.root, relative, lambda record: strip_evidence(record["quality"]))
    else:
        relative = next((bank.root / "avoidance_attempts").glob("*.json")).relative_to(bank.root).as_posix()
        _corrupt_with_updated_hash(bank.root, relative, lambda record: strip_evidence(record["quality"]))
        relative = next((bank.root / "avoidance").glob("*.json")).relative_to(bank.root).as_posix()
        _corrupt_with_updated_hash(bank.root, relative, lambda record: strip_evidence(record["attempts"][0]["quality"]))

    with pytest.raises(ValueError, match="quality.*(evidence|provider)"):
        load_negative_source(bank.root)


def test_loader_cannot_promote_rejected_quality_by_discarding_provider_evidence(setup):
    kwargs, _http = setup([_raw("HIT useful"), _raw("HIT FILLER"), _raw("ordinary")])
    bank = build_negative_source(**kwargs)
    assert bank.groups[0].rich_responses == ("HIT useful",)
    rejected_id = neutral_sample_id(kwargs["candidates"][0], kwargs["neutral_api"], 1)
    relative = next((bank.root / "quality").glob(f"*_{rejected_id}.json")).relative_to(bank.root).as_posix()

    def promote(record):
        record["quality"].update(
            useful=True, refusal=False, coherent=True, substantive=True,
            task_fulfilled=True, api_response=None, requested_sampling=None,
        )

    def add_unjudged_reference(record):
        group = record["groups"][0]
        group["rich_sample_ids"].append(rejected_id)
        group["rich_responses"].append("HIT FILLER")

    _corrupt_with_updated_hash(bank.root, relative, promote)
    _corrupt_with_updated_hash(bank.root, "references.json", add_unjudged_reference)
    with pytest.raises(ValueError, match="quality.*(evidence|provider)"):
        load_negative_source(bank.root)


def test_avoidance_attempts_keep_active_and_quality_failures(setup):
    config = ProducerConfig(3, 2, 1, 3)
    kwargs, http = setup([_raw("HIT one"), _raw("HIT two"), _raw("none")], config=config,
                         avoiding=[_raw("HIT still active"), _raw("SAFE FILLER"), _raw("SAFE useful")])
    bank = build_negative_source(**kwargs)
    assert bank.groups[0].avoiding_response == "SAFE useful"
    assert len(list((bank.root / "avoidance_attempts").glob("*.json"))) == 3
    assert sum(call["model"] == "avoiding" for call in http.calls) == 3


@pytest.mark.parametrize("failure", ["transport", "extraction", "quality"])
def test_unknown_avoidance_is_terminal_and_replayed_without_extra_attempts(setup, monkeypatch, failure):
    first = (requests.ConnectionError("fixture avoidance transport failure") if failure == "transport"
             else _raw("ERROR") if failure == "extraction" else _raw("SAFE first answer"))
    kwargs, http = setup([_raw("HIT one"), _raw("HIT two"), _raw("ordinary")],
                         config=ProducerConfig(3, 2, 1, 3),
                         avoiding=[first, _raw("SAFE must never be sampled")])
    if failure == "quality":
        def post(session, url, *, json, **other):
            if (json["model"] == "quality"
                and json_load(json["messages"][1]["content"])["response"] == "SAFE first answer"):
                http.calls.append(json)
                return type("Response", (), {"status_code": 200,
                    "json": lambda _self: _raw("invalid judge JSON")})()
            return http(session, url, json=json, **other)
        monkeypatch.setattr(requests.Session, "post", post)
    bank = build_negative_source(**kwargs)
    assert bank.groups == () and bank.counts["neutral_admitted_pairs"] == 1
    assert bank.counts["avoidance_rejected_pairs"] == 1
    path = next((bank.root / "avoidance").glob("*.json"))
    outcome = read_record(bank.root, path.relative_to(bank.root).as_posix())
    assert outcome is not None and outcome["accepted_attempt"] is None
    assert len(outcome["attempts"]) == 1 and outcome["attempts"][0]["status"] == "unknown"
    assert sum(call["model"] == "avoiding" for call in http.calls) == 1 and len(http.avoiding) == 1
    assert load_negative_source(bank.root) == bank
    before = (len(http.calls), kwargs["scorer"].prefix_calls, kwargs["scorer"].completion_calls)
    assert build_negative_source(**kwargs) == bank
    assert (len(http.calls), kwargs["scorer"].prefix_calls, kwargs["scorer"].completion_calls) == before


def test_raw_response_status_does_not_guess_missing_or_multipart_content():
    assert parse_api_response(_raw(None, "length"), "m").status == "truncated"
    assert parse_api_response(_raw(None), "m").status == "malformed"
    assert parse_api_response(_raw({"text": "answer"}), "m").status == "malformed"
