"""Method-aware extension of the native provider mailbox protocol."""

from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from steering.eval.axbench_adv import native_provider
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.gepa_optimization import RawEvaluation
from steering.eval.axbench_adv.native_mailbox import digest_json, read_json, write_json


@pytest.fixture
def provider_config(tmp_path):
    return OmegaConf.create(
        {
            "mailbox_dir": str(tmp_path),
            "judge_cache": str(tmp_path / "judge.sqlite"),
            "reflection_cache": str(tmp_path / "reflection.sqlite"),
            "reflection_model": native_provider.JUDGE_MODEL,
            "requests_per_minute": 60,
            "idle_timeout_seconds": 10.0,
            "poll_seconds": 0.01,
            "once": True,
        }
    )


def _publish_request(root, payload):
    body = {"schema_version": 1, "operation": "judge", "payload": payload}
    request_id = digest_json(body)
    write_json(root / "requests" / f"{request_id}.json", {"request_id": request_id, **body})
    return request_id


def test_provider_routes_all_native_standalone_methods(tmp_path, monkeypatch, provider_config):
    init_payloads, evaluated = [], []

    class Client:
        def __init__(self, *, init_payload, timeout_seconds):
            init_payloads.append(init_payload)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def evaluate(self, *, generation, input_id):
            evaluated.append((generation, input_id))
            return RawEvaluation(generation, 1.0, 2.0, '{"concept":"ok","fluency":"ok"}', 0)

    monkeypatch.setattr(native_provider, "SynthResponseCache", lambda _path: nullcontext())
    monkeypatch.setattr(
        native_provider,
        "build_openrouter_reflection_lm",
        lambda **_kwargs: Mock(paid_cache_misses=0),
    )
    monkeypatch.setattr(native_provider, "AxBenchAdvJudgeClient", Client)
    payloads = [
        {
            "concept_id": 0,
            "concept": "concept zero",
            "generation": f"{method} output",
            "input_id": index,
            "model": native_provider.JUDGE_MODEL,
            "method": method,
            "factor": factor,
        }
        for index, (method, factor) in enumerate(
            (("DiffMean", 0.2), ("LinearProbe", 0.4), ("LoRA", 1.0))
        )
    ]
    payloads[-1]["run_identity_sha256"] = "d" * 64
    request_ids = [_publish_request(tmp_path, payload) for payload in payloads]
    native_provider.run_provider(provider_config)
    assert sorted(evaluated, key=lambda item: item[1]) == [
        (payload["generation"], payload["input_id"]) for payload in payloads
    ]
    assert len(init_payloads) == 1
    assert init_payloads[0]["method"] == native_provider.NATIVE_METHOD
    for request_id, payload in zip(request_ids, payloads, strict=True):
        response = read_json(tmp_path / "responses" / f"{request_id}.json")
        assert response["result"]["generation"] == payload["generation"]


@pytest.mark.parametrize(
    "method,factor",
    [("DiffMean", 0.3), ("LinearProbe", True), ("LoRA", 0.2), ("PromptSteering", 1.0)],
)
def test_provider_rejects_bad_method_coordinates_before_judging(
    tmp_path, monkeypatch, provider_config, method, factor
):
    monkeypatch.setattr(native_provider, "SynthResponseCache", lambda _path: nullcontext())
    monkeypatch.setattr(
        native_provider,
        "build_openrouter_reflection_lm",
        lambda **_kwargs: Mock(paid_cache_misses=0),
    )
    judge = Mock(side_effect=AssertionError("unexpected judge construction"))
    monkeypatch.setattr(native_provider, "AxBenchAdvJudgeClient", judge)
    _publish_request(
        tmp_path,
        {
            "concept_id": 0,
            "concept": "concept zero",
            "generation": "output",
            "input_id": 0,
            "model": native_provider.JUDGE_MODEL,
            "method": method,
            "factor": factor,
        },
    )
    with pytest.raises(AxBenchAdvError):
        native_provider.run_provider(provider_config)
    judge.assert_not_called()


def test_provider_rejects_invalid_run_identity_before_judging(
    tmp_path, monkeypatch, provider_config
):
    monkeypatch.setattr(native_provider, "SynthResponseCache", lambda _path: nullcontext())
    monkeypatch.setattr(
        native_provider,
        "build_openrouter_reflection_lm",
        lambda **_kwargs: Mock(paid_cache_misses=0),
    )
    judge = Mock(side_effect=AssertionError("unexpected judge construction"))
    monkeypatch.setattr(native_provider, "AxBenchAdvJudgeClient", judge)
    _publish_request(
        tmp_path,
        {
            "concept_id": 0,
            "concept": "concept zero",
            "generation": "output",
            "input_id": 0,
            "model": native_provider.JUDGE_MODEL,
            "method": "DiffMean",
            "factor": 0.2,
            "run_identity_sha256": "not-a-sha",
        },
    )
    with pytest.raises(AxBenchAdvError, match="run identity"):
        native_provider.run_provider(provider_config)
    judge.assert_not_called()
