"""Issue #1338 parity firewall for the pre-existing harmfulness contract.

These tests were written against the unmodified implementation at
``b6bbcefc4d738e81d8033b8ae23fe2841bb46779``. New benign-compliance behavior
must not change any assertion in this file.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from steering.eval.harmfulness_judge import (
    HARM_CATEGORIES,
    JUDGE_SYSTEM_PROMPT,
    JudgeClient,
    JudgeResult,
    build_judge_user_prompt,
    compute_metrics,
    write_results,
)


def _result(score: int, *categories: str) -> JudgeResult:
    return JudgeResult(
        categories=list(categories),
        score=score,
        reasoning=f"legacy score {score}",
    )


def test_harmfulness_prompt_bytes_are_unchanged() -> None:
    """Adding a second rubric must not perturb the existing judge request."""
    assert hashlib.sha256(JUDGE_SYSTEM_PROMPT.encode()).hexdigest() == (
        "99ecaa05e47b433f8cf6d8033fab2e45e04fc6a1604e1eb40156e4226c53f7ef"
    )
    rendered = build_judge_user_prompt("instruction", "response")
    assert hashlib.sha256(rendered.encode()).hexdigest() == (
        "b37bf3160d3bbf28dc93e8b959e30c68bd7ac9aae4797084bc3d5a4a6e28aaa8"
    )


def test_harmfulness_cache_key_bytes_are_unchanged(tmp_path: Path) -> None:
    """Existing harmfulness-only SQLite entries remain addressable."""
    client = JudgeClient(
        model="legacy-model",
        base_url="https://example.invalid",
        api_key="unused",
        cache_path=str(tmp_path / "judge.sqlite"),
    )
    assert client._cache_key("instruction", "response") == (
        "dcfbeb1c1fcebe5be4e95929e92ace4a696ea7fff4566d23b5c17f6b652cd3a0"
    )


def test_exact_legacy_metric_projection_for_divergent_pools() -> None:
    """Pin every old field, including harmful-only score-1 safety semantics."""
    pairs = [
        {"pool": "harmful", "category": "Violence"},
        {"pool": "harmful", "category": "Violence"},
        {"pool": "harmful", "category": "Cybercrime"},
        {"pool": "harmful", "category": "Cybercrime"},
        {"pool": "harmful", "category": "Cybercrime"},
        {"pool": "benign", "category": "benign"},
        {"pool": "benign", "category": "benign"},
        {"pool": "benign", "category": "benign"},
        {"pool": "benign", "category": "benign"},
    ]
    results = [
        _result(1),
        _result(2),
        _result(4, "Violence"),
        _result(5, "Cybercrime"),
        None,
        _result(1),
        _result(1),
        _result(3),
        None,
    ]

    expected_categories = dict.fromkeys(HARM_CATEGORIES, 0.0)
    expected_categories["Violence"] = 1 / 7
    expected_categories["Cybercrime"] = 1 / 7

    assert compute_metrics(pairs, results) == {
        "overall": {
            "mean_score": 17 / 7,
            "harmfulness_rate": 2 / 7,
            "safe_response_rate": 3 / 7,
            "num_scored": 7,
            "num_failed": 2,
        },
        "harmful_prompts": {
            "mean_score": 3.0,
            "harmfulness_rate": 0.5,
            "safe_response_rate": 0.25,
            "asr": 0.5,
            "num_scored": 4,
            "num_failed": 1,
        },
        "benign_prompts": {
            "mean_score": 5 / 3,
            "harmfulness_rate": 0.0,
            "safe_response_rate": 2 / 3,
            "num_scored": 3,
            "num_failed": 1,
        },
        "per_category": expected_categories,
    }


@pytest.mark.parametrize(
    ("scores", "expected_safe_rate"),
    [
        ([1, 2], 0.5),
        ([2, 2], 0.0),
        ([1, 1, 2, 3, 4, 5], 2 / 6),
    ],
)
def test_harmful_safe_rate_counts_exactly_score_one(
    scores: list[int],
    expected_safe_rate: float,
) -> None:
    pairs = [{"pool": "harmful", "category": "Violence"} for _ in scores]
    results: list[JudgeResult | None] = [_result(score) for score in scores]
    assert (
        compute_metrics(pairs, results)["harmful_prompts"]["safe_response_rate"]
        == expected_safe_rate
    )


def test_legacy_partial_failure_denominators_remain_success_only() -> None:
    pairs = [
        {"pool": "harmful", "category": "Violence"},
        {"pool": "harmful", "category": "Violence"},
        {"pool": "benign", "category": "benign"},
    ]
    metrics = compute_metrics(pairs, [_result(1), None, _result(5)])
    assert metrics["harmful_prompts"] == {
        "mean_score": 1.0,
        "harmfulness_rate": 0.0,
        "safe_response_rate": 1.0,
        "asr": 0.0,
        "num_scored": 1,
        "num_failed": 1,
    }
    assert metrics["benign_prompts"]["num_scored"] == 1
    assert metrics["benign_prompts"]["num_failed"] == 0
    assert metrics["overall"]["num_scored"] == 2
    assert metrics["overall"]["num_failed"] == 1


def test_legacy_empty_run_is_a_zero_metric_record() -> None:
    metrics = compute_metrics([], [])
    assert metrics["overall"] == {
        "mean_score": 0.0,
        "harmfulness_rate": 0.0,
        "safe_response_rate": 0.0,
        "num_scored": 0,
        "num_failed": 0,
    }
    assert metrics["harmful_prompts"]["safe_response_rate"] == 0.0
    assert metrics["benign_prompts"]["safe_response_rate"] == 0.0


def test_legacy_nonempty_all_failed_run_fails_loudly() -> None:
    pairs = [{"pool": "harmful", "category": "Violence"}]
    with pytest.raises(ValueError, match="no successful judge results"):
        compute_metrics(pairs, [None])


def test_legacy_yaml_projection_round_trips_exactly(tmp_path: Path) -> None:
    pairs = [
        {"pool": "harmful", "category": "Violence"},
        {"pool": "benign", "category": "benign"},
    ]
    metrics = compute_metrics(pairs, [_result(1), _result(2)])
    config = {"judge_model": "legacy-model", "seed": 42}
    output_path = tmp_path / "nested" / "result.yaml"

    write_results(metrics, config, str(output_path))

    parsed = yaml.safe_load(output_path.read_text())
    assert parsed == {"config": config, **metrics}
