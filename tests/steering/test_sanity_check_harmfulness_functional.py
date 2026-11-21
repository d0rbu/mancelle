"""Functional orchestration proof for the standalone harmfulness sanity check."""

from __future__ import annotations

from typing import Any

import pytest

from steering.eval.harmfulness_judge import ComplianceJudgeResult, JudgeResult
from steering.eval.sanity_check import harmfulness


def test_main_runs_harmfulness_and_benign_compliance_with_luna(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(harmfulness, "load_project_dotenv", lambda: None)
    monkeypatch.setattr(harmfulness, "autodetect_device_type", lambda: "cpu")
    monkeypatch.setattr(harmfulness, "get_base_dir", lambda: "/model-root")
    monkeypatch.setattr(harmfulness, "find_largest_model", lambda _path: "d12")
    monkeypatch.setattr(harmfulness, "find_last_step", lambda _path: 123)
    monkeypatch.setattr(
        harmfulness,
        "build_model",
        lambda *_args: (object(), object(), {"step": 123}),
    )
    monkeypatch.setattr(
        harmfulness,
        "load_harmful_prompts",
        lambda: [{"prompt": "harm", "category": "Violence"}],
    )
    monkeypatch.setattr(
        harmfulness,
        "load_benign_prompts",
        lambda *, n, seed: [{"prompt": "benign", "category": "benign"}],
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    class _Engine:
        def __init__(self, *_args: Any) -> None:
            pass

        @staticmethod
        def generate(prompt: str, **_kwargs: Any) -> str:
            return f"response-for-{prompt}"

    class _Judge:
        def __init__(self, *, model: str, requests_per_minute: int) -> None:
            assert model == "openai/gpt-5.6-luna"
            assert requests_per_minute == 60

        @staticmethod
        def score_all(pairs: list[dict[str, str]]) -> list[JudgeResult]:
            assert [pair["pool"] for pair in pairs] == ["harmful", "benign"]
            return [
                JudgeResult(["Violence"], 4, "Harmful."),
                JudgeResult([], 1, "Safe."),
            ]

        @staticmethod
        def score_compliance_all(
            pairs: list[dict[str, str]],
        ) -> list[ComplianceJudgeResult]:
            assert [pair["prompt"] for pair in pairs] == ["benign"]
            return [ComplianceJudgeResult(5, False, "Complete answer.")]

    monkeypatch.setattr(harmfulness, "InferenceEngine", _Engine)
    monkeypatch.setattr(harmfulness, "JudgeClient", _Judge)

    harmfulness.main()

    output = capsys.readouterr().out
    assert "Loading model: tag=d12, step=123" in output
    assert "SCORING WITH JUDGE (openai/gpt-5.6-luna)" in output
    assert "'compliance_rate': 1.0" in output
    assert "HARMFULNESS JUDGE SMOKE TEST COMPLETE" in output
