"""CPU-only Campaign 2 judging with a mocked HTTP provider; paid gate stays off."""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import Mock

import pytest
import requests
from omegaconf import OmegaConf

from errors import ParseError
from scripts.eval import campaign2 as runner
from scripts.utils.campaign2 import install_json_once
from steering.eval import campaign2_judge as resume
from steering.eval import harmfulness_artifact as artifact
from steering.eval import harmfulness_judge as judge
from steering.eval.harmfulness_artifact import build_harmfulness_payload

MODEL = "mock/campaign2-judge"
FALLBACK_MODEL = "mock/campaign2-fallback"
RAW_HARMFUL = 'categories: [Violence]\nscore: 4\nreasoning: "harmful"'
RAW_SAFE = 'categories: []\nscore: 1\nreasoning: "safe"'
RAW_COMPLIANCE = 'helpfulness_score: 3\nis_refusal: false\nreasoning: "helpful"'
ROWS = [
    {"prompt": "Q0", "response": "R0", "pool": "harmful", "category": "Violence"},
    {"prompt": "Q1", "response": "R1", "pool": "benign", "category": "benign"},
    {"prompt": "Q2", "response": "R2", "pool": "harmful", "category": "Violence"},
    {"prompt": "Q3", "response": "R3", "pool": "benign", "category": "benign"},
]


def response(raw: str) -> Mock:
    return Mock(
        status_code=200,
        json=lambda: {"choices": [{"message": {"content": raw}}]},
    )


def request_key(payload: dict) -> tuple[str, int]:
    messages = payload["messages"]
    kind = (
        resume.JudgmentKind.HARMFULNESS
        if messages[0]["content"] == judge.JUDGE_SYSTEM_PROMPT
        else resume.JudgmentKind.COMPLIANCE
    )
    index = next(i for i in range(len(ROWS)) if f"Q{i}" in messages[1]["content"])
    return kind.value, index


def raw_for(kind: str, index: int) -> str:
    if kind == resume.JudgmentKind.COMPLIANCE:
        return RAW_COMPLIANCE
    return RAW_HARMFUL if index % 2 == 0 else RAW_SAFE


def successful_provider(url: str, *, headers: dict, json: dict, timeout: int) -> Mock:
    assert json["model"] == MODEL
    assert headers["Authorization"] == "Bearer offline-test"
    return response(raw_for(*request_key(json)))


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(judge.requests, "post", Mock(side_effect=AssertionError("unexpected HTTP")))


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[judge.JudgeClient, Path, list]:
    output = tmp_path / "cell"
    install_json_once(
        output / "identity.json", {"stage": "instruct", "variant": "pythia_prompt"}
    )
    client = judge.JudgeClient(
        model=MODEL,
        api_key="offline-test",
        cache_path=str(tmp_path / "judge.sqlite"),
        requests_per_minute=0,
    )
    sleeps = []
    monkeypatch.setattr(judge.time, "sleep", sleeps.append)
    monkeypatch.setattr(resume, "JUDGE_MAX_CONCURRENT", 1)
    return client, output, sleeps


def record_path(output: Path, kind: str = resume.JudgmentKind.HARMFULNESS, index: int = 0) -> Path:
    return output / resume.JUDGE_RECORDS_DIR / kind / f"{index:06d}.json"


def expected_payload(rows: list[dict]) -> dict:
    harmfulness = [
        judge.parse_judge_yaml(raw_for(resume.JudgmentKind.HARMFULNESS, int(row["prompt"][1:])))
        for row in rows
    ]
    benign = [row for row in rows if row["pool"] == resume.BENIGN_POOL]
    compliance = [judge.parse_compliance_yaml(RAW_COMPLIANCE) for _ in benign]
    payload = build_harmfulness_payload(
        metrics=judge.compute_metrics(rows, harmfulness),
        compliance_metrics=judge.compute_compliance_metrics(benign, compliance),
        model=MODEL,
        pairs=rows,
    )
    payload["judge_model_assignments"] = [
        {
            "prompt_index": index,
            "harmfulness": MODEL,
            "compliance": MODEL if row["pool"] == resume.BENIGN_POOL else None,
        }
        for index, row in enumerate(rows)
    ]
    payload["fallback_scored_pair_count"] = 0
    return payload


def test_fallback_envelope_persists_campaign_assignment_and_count(
    setup: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, output, _ = setup
    client.refusal_fallback_model = FALLBACK_MODEL
    def score(requests: list, *, max_concurrent: int) -> list[tuple[int, judge.JudgeScoreEnvelope]]:
        assert max_concurrent == 1
        envelopes = []
        for request_index, (instruction, _response, kind) in enumerate(requests):
            index = int(instruction[1:])
            model = (
                FALLBACK_MODEL
                if kind is resume.JudgmentKind.HARMFULNESS and index == 0
                else MODEL
            )
            raw = raw_for(kind, index)
            envelope = judge.JudgeScoreEnvelope(
                raw=raw,
                result=client._parse_score(kind, raw, model),
                model=model,
                attempts=2 if model == FALLBACK_MODEL else 1,
                error=None,
            )
            envelopes.append((request_index, envelope))
        return envelopes

    monkeypatch.setattr(client, "score_envelopes", score)
    payload = resume.judge_records(client, ROWS[:2], output)

    assert payload["judge_model_assignments"] == [
        {
            "prompt_index": 0,
            "harmfulness": FALLBACK_MODEL,
            "compliance": None,
        },
        {
            "prompt_index": 1,
            "harmfulness": MODEL,
            "compliance": MODEL,
        },
    ]
    assert payload["fallback_scored_pair_count"] == 1
    saved = json.loads(record_path(output).read_text())
    assert saved["judge_model"] == FALLBACK_MODEL
    assert saved["attempts"] == 2


@pytest.mark.parametrize(
    "failure",
    [
        requests.Timeout("timeout"),
        ParseError("parse"),
        KeyError("key"),
        IndexError("index"),
        TypeError("type"),
        ValueError("value"),
        OverflowError("overflow"),
    ],
)
def test_campaign_retry_policy_retries_only_declared_failures(
    setup: tuple,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    client, output, sleeps = setup
    call = Mock(side_effect=[failure, failure, RAW_HARMFUL])
    monkeypatch.setattr(client, "_call_api", call)

    assert resume.judge_records(client, ROWS[:1], output) == expected_payload(ROWS[:1])
    assert call.call_count == 3
    assert sleeps == [1.0, 2.0]
    assert json.loads(record_path(output).read_text())["attempts"] == 3


@pytest.mark.parametrize("failure", [RuntimeError("runtime"), AssertionError("assertion")])
def test_campaign_retry_policy_propagates_undeclared_failures(
    setup: tuple,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    client, output, sleeps = setup
    call = Mock(side_effect=failure)
    monkeypatch.setattr(client, "_call_api", call)

    with pytest.raises(type(failure), match=str(failure)):
        resume.judge_records(client, ROWS[:1], output)
    assert call.call_count == 1
    assert not sleeps
    assert not record_path(output).exists()


def test_partial_resume_matches_one_pass_and_reuses_both_judges(
    setup: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, output, sleeps = setup
    calls = Counter()
    failures = {
        (resume.JudgmentKind.HARMFULNESS.value, 2),
        (resume.JudgmentKind.COMPLIANCE.value, 3),
    }

    def provider(url: str, **kwargs: dict) -> Mock:
        key = request_key(kwargs["json"])
        calls[key] += 1
        if key in failures:
            return Mock(json=lambda: {"error": {"message": "provider unavailable", "code": 503}})
        return successful_provider(url, **kwargs)

    monkeypatch.setattr(judge.requests, "post", provider)
    with pytest.raises(ValueError) as exc:
        resume.judge_records(client, ROWS, output)
    assert str(exc.value) == (
        "harmfulness judge returned incomplete scores; remaining prompt indexes: [2]; "
        "benign compliance judge returned incomplete scores; remaining prompt indexes: [3]"
    )
    assert not (output / "results.json").exists()
    assert len(list((output / resume.JUDGE_RECORDS_DIR).glob("*/*.json"))) == 6
    for kind, index in failures:
        saved = json.loads(record_path(output, kind, index).read_text())
        assert saved["raw"] is None and "provider unavailable" in saved["error"]
        assert saved["attempts"] == judge.JUDGE_MAX_ATTEMPTS
    successful_bytes = {
        p: p.read_bytes()
        for p in (output / resume.JUDGE_RECORDS_DIR).glob("*/*.json")
        if json.loads(p.read_text())["error"] is None
    }
    failures.clear()
    calls.clear()
    payload = resume.judge_records(client, ROWS, output)
    assert calls == {
        (resume.JudgmentKind.HARMFULNESS.value, 2): 1,
        (resume.JudgmentKind.COMPLIANCE.value, 3): 1,
    }
    assert all(p.read_bytes() == content for p, content in successful_bytes.items())
    assert payload == expected_payload(ROWS)
    one_pass = output.parent / "one-pass"
    install_json_once(
        one_pass / "identity.json", {"stage": "instruct", "variant": "pythia_prompt"}
    )
    assert resume.judge_records(client, ROWS, one_pass) == payload
    install_json_once(output / "results.json", payload)
    result_bytes = (output / "results.json").read_bytes()
    calls.clear()
    install_json_once(output / "results.json", resume.judge_records(client, ROWS, output))
    assert not calls
    assert (output / "results.json").read_bytes() == result_bytes


@pytest.mark.parametrize(
    "kind,index", [(resume.JudgmentKind.HARMFULNESS, 0), (resume.JudgmentKind.COMPLIANCE, 1)]
)
@pytest.mark.parametrize(
    "damage",
    [
        "truncated",
        "non_mapping",
        "unicode",
        "checksum",
        "schema",
        "model",
        "prompt_version",
        "cell",
        "index",
        "kind",
        "response",
        "invalid_score",
        "invalid_raw_type",
        "missing_field",
        "missing_judge_model",
        "attempts",
    ],
)
def test_corrupt_or_mismatched_records_are_rejudged(
    setup: tuple, monkeypatch: pytest.MonkeyPatch, kind: str, index: int, damage: str
) -> None:
    client, output, _ = setup
    provider = Mock(side_effect=successful_provider)
    monkeypatch.setattr(judge.requests, "post", provider)
    resume.judge_records(client, ROWS[:2], output)
    path = record_path(output, kind, index)
    saved = json.loads(path.read_text())
    if damage == "truncated":
        path.write_text('{"raw":')
    elif damage == "non_mapping":
        path.write_text("[]")
    elif damage == "unicode":
        path.write_bytes(b"\xff")
    else:
        identity_keys = {
            "schema": "schema_version",
            "model": "model",
            "prompt_version": "harmfulness_prompt_sha256",
            "cell": "cell_sha256",
            "index": "prompt_index",
            "kind": "kind",
            "response": "ordered_pairs_sha256",
        }
        if damage in identity_keys:
            saved["identity"][identity_keys[damage]] = "mismatched"
        elif damage == "missing_field":
            del saved["raw"]
        elif damage == "missing_judge_model":
            del saved["judge_model"]
        elif damage == "attempts":
            saved["attempts"] = True
        elif damage == "invalid_raw_type":
            saved["raw"] = []
        else:
            saved["raw"] = 'categories: []\nscore: 99\nreasoning: "invalid"'
        if damage != "checksum":
            saved["sha256"] = resume._digest(
                {key: value for key, value in saved.items() if key != "sha256"}
            )
        path.write_text(json.dumps(saved))
    provider.reset_mock()
    assert resume.judge_records(client, ROWS[:2], output) == expected_payload(ROWS[:2])
    assert provider.call_count == 1
    assert request_key(provider.call_args.kwargs["json"]) == (kind, index)


@pytest.mark.parametrize("change", ["response_bytes", "prompt", "cell", "model", "prompt_version"])
def test_changed_inputs_invalidate_cache(
    setup: tuple, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    client, output, _ = setup
    provider = Mock(return_value=response(RAW_HARMFUL))
    monkeypatch.setattr(judge.requests, "post", provider)
    rows = [dict(ROWS[0])]
    original = resume.judge_records(client, rows, output)
    provider.reset_mock()
    if change == "response_bytes":
        rows[0]["response"] += " \né"
    elif change == "prompt":
        rows[0]["prompt"] += " changed"
    elif change == "cell":
        (output / "identity.json").write_text('{"stage": "safety"}')
    elif change == "model":
        client.model = "mock/new-judge"
    else:
        monkeypatch.setattr(artifact, "harmfulness_prompt_sha256", lambda: "changed-prompt-version")
    payload = resume.judge_records(client, rows, output)
    assert provider.call_count == 1
    assert payload["overall"] == original["overall"]


def test_success_persisted_while_other_request_is_pending(
    setup: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, output, _ = setup
    started, release = Event(), Event()
    monkeypatch.setattr(resume, "JUDGE_MAX_CONCURRENT", 2)
    original_write = resume.atomic_json_dump
    persisted = Event()

    def write(record: dict, path: Path) -> None:
        original_write(record, path)
        if path == record_path(output):
            persisted.set()

    def provider(url: str, **kwargs: dict) -> Mock:
        kind, index = request_key(kwargs["json"])
        if index == 0:
            assert started.wait(5)
        else:
            started.set()
            assert release.wait(5)
        return response(raw_for(kind, index))

    monkeypatch.setattr(resume, "atomic_json_dump", write)
    monkeypatch.setattr(judge.requests, "post", provider)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(resume.judge_records, client, [ROWS[0], ROWS[2]], output)
        try:
            assert persisted.wait(5)
            assert not future.done()
            assert json.loads(record_path(output).read_text())["raw"] == RAW_HARMFUL
        finally:
            release.set()
        assert future.result() == expected_payload([ROWS[0], ROWS[2]])


def test_interrupt_preserves_completed_work(setup: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    client, output, _ = setup
    provider = Mock(side_effect=[response(RAW_HARMFUL), KeyboardInterrupt()])
    monkeypatch.setattr(judge.requests, "post", provider)
    with pytest.raises(KeyboardInterrupt):
        resume.judge_records(client, [ROWS[0], ROWS[2]], output)
    assert record_path(output).exists()
    provider = Mock(return_value=response(RAW_HARMFUL))
    monkeypatch.setattr(judge.requests, "post", provider)
    assert resume.judge_records(client, [ROWS[0], ROWS[2]], output) == expected_payload(
        [ROWS[0], ROWS[2]]
    )
    assert provider.call_count == 1


def test_storage_failure_is_fatal_without_provider_retry(
    setup: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, output, sleeps = setup
    provider = Mock(return_value=response(RAW_HARMFUL))
    monkeypatch.setattr(judge.requests, "post", provider)
    monkeypatch.setattr(resume, "atomic_json_dump", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        resume.judge_records(client, ROWS[:1], output)
    assert provider.call_count == 1 and not sleeps


def test_cell_identity_required_before_provider_call(setup: tuple) -> None:
    client, output, _ = setup
    (output / "identity.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        resume.judge_records(client, ROWS[:1], output)


@pytest.mark.parametrize("api_key", [None, "offline-test"])
def test_runner_paid_gate_stays_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_key: str | None
) -> None:
    if api_key is not None:
        monkeypatch.setenv("OPENROUTER_API_KEY", api_key)
    cfg = OmegaConf.create({"allow_paid_judge": False})
    monkeypatch.setattr(
        runner, "JudgeClient", Mock(side_effect=AssertionError("paid gate bypassed"))
    )
    with pytest.raises(ValueError, match="allow_paid_judge=true and OPENROUTER_API_KEY"):
        runner.judge(cfg, ROWS, tmp_path)
    assert cfg.allow_paid_judge is False


def test_legacy_sqlite_cache_cannot_bypass_cell_identity(
    setup: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, output, _ = setup
    client._cache_put(
        client._cache_key(ROWS[0]["prompt"], ROWS[0]["response"]), judge.parse_judge_yaml(RAW_SAFE)
    )
    provider = Mock(return_value=response(RAW_HARMFUL))
    monkeypatch.setattr(judge.requests, "post", provider)
    assert resume.judge_records(client, ROWS[:1], output) == expected_payload(ROWS[:1])
    assert provider.call_count == 1


def test_cache_io_failure_is_loud_without_rejudging(
    setup: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, output, _ = setup
    provider = Mock(return_value=response(RAW_HARMFUL))
    monkeypatch.setattr(judge.requests, "post", provider)
    resume.judge_records(client, ROWS[:1], output)
    original_read = Path.read_text

    def read(path: Path, *args: tuple, **kwargs: dict) -> str:
        if path == record_path(output):
            raise PermissionError("unreadable cache")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    provider.reset_mock()
    with pytest.raises(PermissionError, match="unreadable cache"):
        resume.judge_records(client, ROWS[:1], output)
    provider.assert_not_called()
