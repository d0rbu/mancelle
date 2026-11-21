"""Independent launch-preflight regressions for the bounded paid pilot."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from scripts.eval.axbench_adv.pilot_budget import BUDGET_ENV, create_pilot_budget
from steering.eval.axbench_adv import pipeline
from steering.eval.axbench_adv.errors import AxBenchAdvError


def _ledger(path: Path, *, model: str, endpoint: str) -> Path:
    create_pilot_budget(
        path,
        limit_usd="20",
        request_bound_usd="1.25",
        model=model,
        endpoint=endpoint,
        max_output_tokens=128_000,
        pricing_evidence="offline launch-preflight fixture",
    )
    return path


def _config(
    *, model: str, api_base_url: str, output_dir: Path | None = None
) -> pipeline.AxBenchAdvRunConfig:
    return cast(
        pipeline.AxBenchAdvRunConfig,
        SimpleNamespace(
            reflection=SimpleNamespace(model=model),
            api_base_url=api_base_url,
            output_dir=output_dir,
        ),
    )


def test_unset_budget_does_not_inspect_run_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(BUDGET_ENV, raising=False)

    class UnusableConfig:
        def __getattribute__(self, _name: str) -> object:
            raise AssertionError("unset pilot must preserve the default path")

    pipeline._preflight_pilot_budget(cast(pipeline.AxBenchAdvRunConfig, UnusableConfig()))


def test_local_endpoint_passes_only_when_both_actual_clients_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_base_url = "http://127.0.0.1:32109/v1"
    endpoint = f"{api_base_url}/chat/completions"
    path = _ledger(tmp_path / "pilot.sqlite3", model=pipeline.JUDGE_MODEL, endpoint=endpoint)
    monkeypatch.setenv(BUDGET_ENV, str(path))
    monkeypatch.setenv("OPENROUTER_BASE_URL", api_base_url)
    monkeypatch.setattr(
        pipeline.llm_judge_client,
        "OPENROUTER_CHAT_COMPLETIONS_URL",
        endpoint,
    )
    pipeline._preflight_pilot_budget(_config(model=pipeline.JUDGE_MODEL, api_base_url=api_base_url))


@pytest.mark.parametrize(
    ("reflection_model", "api_base_url", "message"),
    [
        ("different/model", "https://openrouter.ai/api/v1", "models"),
        (pipeline.JUDGE_MODEL, "https://other.invalid/v1", "endpoints"),
    ],
)
def test_run_pipeline_rejects_client_mismatch_before_creating_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reflection_model: str,
    api_base_url: str,
    message: str,
) -> None:
    endpoint = pipeline.llm_judge_client.OPENROUTER_CHAT_COMPLETIONS_URL
    path = _ledger(tmp_path / "pilot.sqlite3", model=pipeline.JUDGE_MODEL, endpoint=endpoint)
    monkeypatch.setenv(BUDGET_ENV, str(path))
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    output = tmp_path / "must-not-exist"
    config = _config(
        model=reflection_model,
        api_base_url=api_base_url,
        output_dir=output,
    )
    with pytest.raises(AxBenchAdvError, match=message):
        pipeline.run_pipeline(config)
    assert not output.exists()


@pytest.mark.parametrize("ledger_field", ["model", "endpoint"])
def test_run_pipeline_rejects_ledger_identity_before_creating_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ledger_field: str,
) -> None:
    endpoint = pipeline.llm_judge_client.OPENROUTER_CHAT_COMPLETIONS_URL
    path = _ledger(
        tmp_path / "pilot.sqlite3",
        model="different/model" if ledger_field == "model" else pipeline.JUDGE_MODEL,
        endpoint="https://other.invalid/v1/chat/completions"
        if ledger_field == "endpoint"
        else endpoint,
    )
    monkeypatch.setenv(BUDGET_ENV, str(path))
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    output = tmp_path / "must-not-exist"
    config = _config(
        model=pipeline.JUDGE_MODEL,
        api_base_url="https://openrouter.ai/api/v1",
        output_dir=output,
    )
    with pytest.raises(AxBenchAdvError, match="priced ledger"):
        pipeline.run_pipeline(config)
    assert not output.exists()
