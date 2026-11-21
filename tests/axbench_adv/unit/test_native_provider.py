"""Offline provider handoff, interruption recovery, and durable replay."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from steering.eval.axbench_adv import native_mailbox, native_provider
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.gepa_optimization import RawEvaluation
from steering.eval.axbench_adv.native_mailbox import (
    ArtifactMailbox,
    MailboxReflection,
    checked_raw,
    digest_json,
    read_json,
    write_json,
)


@pytest.fixture
def provider_config(tmp_path):
    return OmegaConf.create({"mailbox_dir": str(tmp_path), "judge_cache": str(tmp_path / "judge.sqlite"),
                            "reflection_cache": str(tmp_path / "reflection.sqlite"),
                            "reflection_model": native_provider.JUDGE_MODEL, "requests_per_minute": 60,
                            "idle_timeout_seconds": 10.0, "poll_seconds": 0.01, "once": False})


@pytest.mark.parametrize("response_timing", ["publication", "wait"])
def test_reflection_counts_only_new_mailbox_responses(tmp_path, monkeypatch, response_timing):
    payload = {"model": "approved", "prompt": "reflect"}
    request_id = digest_json({"schema_version": 1, "operation": "reflection", "payload": payload})
    result = {"response": "```avoid cats```", "paid_cache_misses": 1}

    def respond(*_args):
        write_json(tmp_path / "responses" / f"{request_id}.json",
                   {"request_id": request_id, "result": result, "result_sha256": digest_json(result)})

    if response_timing == "publication":
        def publish(path, value):
            write_json(path, value)
            respond()

        monkeypatch.setattr(native_mailbox, "write_json", publish)
    else:
        monkeypatch.setattr(native_mailbox.time, "sleep", respond)
    reflection = MailboxReflection(ArtifactMailbox(tmp_path), "approved")
    assert reflection("reflect") == result["response"]
    assert reflection.paid_cache_misses == 1
    response_path = tmp_path / "responses" / f"{request_id}.json"
    response_bytes = response_path.read_bytes()
    assert reflection("reflect") == result["response"]
    assert reflection.paid_cache_misses == 1
    resumed = MailboxReflection(ArtifactMailbox(tmp_path), "approved")
    assert resumed("reflect") == result["response"]
    assert resumed.paid_cache_misses == 0
    assert response_path.read_bytes() == response_bytes


@pytest.mark.parametrize("interrupted", [False, True])
def test_provider_drains_and_reuses_durable_responses(tmp_path, monkeypatch, provider_config, interrupted):
    clients, judged, reflected = [], [], []
    fail = interrupted

    class Reflection:
        paid_cache_misses = 0

        def __call__(self, prompt):
            reflected.append(prompt)
            self.paid_cache_misses += 1
            return "```avoid cats```"

    class Client:
        def __init__(self, *, init_payload, timeout_seconds):
            assert not any(not client.closed for client in clients)
            self.concept_id = init_payload["concept_id"]
            self.closed = False
            clients.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def close(self):
            self.closed = True

        def evaluate(self, *, generation, input_id):
            nonlocal fail
            assert not self.closed
            if fail and len(judged) == 1:
                fail = False
                raise OSError("interrupted judging")
            judged.append((self.concept_id, input_id))
            return RawEvaluation(generation, 1.0, 2.0, '{"concept":"ok","fluency":"ok"}', 2)

    monkeypatch.setattr(native_provider, "SynthResponseCache", lambda _path: nullcontext())
    monkeypatch.setattr(native_provider, "build_openrouter_reflection_lm", lambda **kwargs: Reflection())
    monkeypatch.setattr(native_provider, "AxBenchAdvJudgeClient", Client)
    requests = [("judge", {"concept_id": c, "concept": f"concept {c}", "generation": "A fluent answer.",
                            "input_id": i, "model": native_provider.JUDGE_MODEL}) for c, i in ((0, 0), (1, 0), (0, 1), (0, 2))]
    requests.append(("reflection", {"model": native_provider.JUDGE_MODEL, "prompt": "reflect"}))
    for operation, payload in requests:
        body = {"schema_version": 1, "operation": operation, "payload": payload}
        request_id = digest_json(body)
        write_json(tmp_path / "requests" / f"{request_id}.json", {"request_id": request_id, **body})
    (tmp_path / "STOP").touch()
    cfg = provider_config
    if interrupted:
        with pytest.raises(OSError, match="interrupted judging"):
            native_provider.run_provider(cfg)
        assert all(client.closed for client in clients)
    native_provider.run_provider(cfg)
    assert all(client.closed for client in clients)
    assert sorted(judged) == [(0, 0), (0, 1), (0, 2), (1, 0)]
    assert reflected == ["reflect"]
    before = {path: path.read_bytes() for path in (tmp_path / "responses").glob("*.json")}
    native_provider.run_provider(cfg)
    assert sorted(judged) == [(0, 0), (0, 1), (0, 2), (1, 0)]
    assert reflected == ["reflect"]
    assert all(path.read_bytes() == content for path, content in before.items())
    for operation, payload in requests:
        replay = ArtifactMailbox(tmp_path).request(operation, payload)
        field = "paid_cache_misses" if operation == "reflection" else "paid_judge_cache_misses"
        assert replay[field] == 0


@pytest.mark.parametrize("mutation", ["config_model", "fields", "id", "version", "reflection", "judge", "operation"])
def test_provider_rejects_bad_requests_without_calls(tmp_path, monkeypatch, provider_config, mutation):
    reflection = Mock(paid_cache_misses=0, side_effect=AssertionError("unexpected provider call"))
    judge = Mock(side_effect=AssertionError("unexpected judge construction"))
    monkeypatch.setattr(native_provider, "SynthResponseCache", lambda _path: nullcontext())
    monkeypatch.setattr(native_provider, "build_openrouter_reflection_lm", lambda **kwargs: reflection)
    monkeypatch.setattr(native_provider, "AxBenchAdvJudgeClient", judge)
    body = {"schema_version": 1, "operation": "reflection",
            "payload": {"model": native_provider.JUDGE_MODEL, "prompt": "reflect"}}
    if mutation == "config_model":
        provider_config.reflection_model = "wrong"
    elif mutation == "version":
        body["schema_version"] = 2
    elif mutation == "reflection":
        body["payload"]["model"] = "wrong"
    elif mutation in {"judge", "operation"}:
        body["operation"] = "judge" if mutation == "judge" else "unknown"
    request_id = digest_json(body)
    request = {"request_id": request_id, **body}
    if mutation == "fields":
        request["extra"] = True
    elif mutation == "id":
        request["request_id"] = "wrong"
    write_json(tmp_path / "requests" / f"{request_id}.json", request)
    with pytest.raises(AxBenchAdvError):
        native_provider.run_provider(provider_config)
    reflection.assert_not_called()
    judge.assert_not_called()
    assert not (tmp_path / "responses").exists()


@pytest.mark.parametrize("once", [True, False])
def test_empty_provider_exits_once_or_after_idle_timeout(monkeypatch, provider_config, once):
    provider_config.once = once
    monkeypatch.setattr(native_provider, "SynthResponseCache", lambda _path: nullcontext())
    reflection = Mock(side_effect=AssertionError("unexpected provider call"))
    monkeypatch.setattr(native_provider, "build_openrouter_reflection_lm", lambda **kwargs: reflection)
    clock = iter([0.0, 0.0, 11.0])
    monkeypatch.setattr(native_provider.time, "monotonic", lambda: next(clock))
    sleep = Mock()
    monkeypatch.setattr(native_provider.time, "sleep", sleep)
    native_provider.run_provider(provider_config)
    assert sleep.call_count == int(not once)
    reflection.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("extra", 1), ("generation", "changed"), ("judge_rationale", None),
    ("paid_judge_cache_misses", True), ("paid_judge_cache_misses", -1), ("paid_judge_cache_misses", 3),
    ("generation_calls", True), ("generation_calls", -1), ("generation_calls", 2),
    ("concept_rating", float("nan")), ("fluency_rating", 3.0),
])
def test_bad_judge_packets_fail_closed(field, value):
    packet = asdict(RawEvaluation("A fluent response.", 1.0, 2.0, '{"concept":"ok","fluency":"ok"}', 2))
    packet[field] = value
    with pytest.raises(AxBenchAdvError):
        checked_raw(packet, "A fluent response.")


@pytest.mark.parametrize("payload", ["[]", '{"x":1,"x":2}', '{"x":NaN}'])
def test_mailbox_json_fails_closed(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(payload)
    with pytest.raises((ValueError, AxBenchAdvError)):
        read_json(path)


def test_mailbox_timeout_leaves_request_for_recovery(tmp_path, monkeypatch):
    clock = iter([0.0, 2.0])
    monkeypatch.setattr(native_mailbox.time, "monotonic", lambda: next(clock))
    with pytest.raises(AxBenchAdvError, match="timed out"):
        ArtifactMailbox(tmp_path, timeout_seconds=1).request("reflection", {"prompt": "reflect"})
    assert len(list((tmp_path / "requests").glob("*.json"))) == 1
    assert not (tmp_path / "responses").exists()


@pytest.mark.parametrize("mutation", ["request_collision", "response_fields", "result_type"])
def test_mailbox_rejects_malformed_durable_handoff(tmp_path, mutation):
    payload = {"model": "approved", "prompt": "reflect"}
    body = {"schema_version": 1, "operation": "reflection", "payload": payload}
    request_id = digest_json(body)
    result = [] if mutation == "result_type" else {"response": "answer", "paid_cache_misses": 1}
    response = {"request_id": request_id, "result": result, "result_sha256": digest_json(result)}
    if mutation == "request_collision":
        write_json(tmp_path / "requests" / f"{request_id}.json", {"wrong": True})
    elif mutation == "response_fields":
        response["extra"] = True
    write_json(tmp_path / "responses" / f"{request_id}.json", response)
    with pytest.raises(AxBenchAdvError):
        ArtifactMailbox(tmp_path).request("reflection", payload)


@pytest.mark.parametrize("prompt,result", [
    ([], {}), ("prompt", {"extra": True}),
    ("prompt", {"response": "answer", "paid_cache_misses": True}),
])
def test_reflection_rejects_invalid_input_or_response(tmp_path, monkeypatch, prompt, result):
    monkeypatch.setattr(ArtifactMailbox, "request", lambda *_args: result)
    reflection = MailboxReflection(ArtifactMailbox(tmp_path), "approved")
    with pytest.raises(AxBenchAdvError):
        reflection(prompt)
    assert reflection.paid_cache_misses == 0


def test_mailbox_rejects_invalid_timeout_and_operation(tmp_path):
    with pytest.raises(ValueError):
        ArtifactMailbox(tmp_path, timeout_seconds=0)
    with pytest.raises(AxBenchAdvError):
        ArtifactMailbox(tmp_path).request("unknown", {})
    assert not (tmp_path / "requests").exists()
