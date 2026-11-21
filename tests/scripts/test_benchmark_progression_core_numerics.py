"""Tests for cumulative progression CORE numerical comparison."""

import torch as t

from scripts.benchmark.benchmark_progression_core_numerics import compare_payloads


def _payload(logit_delta: float = 0.0) -> dict:
    return {
        "git_sha": "0" * 40,
        "evals": {"core": {"metric": 0.5}},
        "records": [
            {
                "key": "row",
                "key_payload": {"positions": [1]},
                "logits": t.arange(100, dtype=t.bfloat16).reshape(1, 100) + logit_delta,
                "losses": t.tensor([0.25 + logit_delta]),
                "predictions": t.tensor([99]),
            }
        ],
    }


def test_identical_payload_passes_all_numerical_gates() -> None:
    result = compare_payloads(_payload(), _payload())

    assert result["passed"] is True
    assert all(result["gates"].values())
    assert result["metrics"]["top1_agreement"] == 1.0


def test_out_of_tolerance_payload_fails() -> None:
    result = compare_payloads(_payload(), _payload(logit_delta=1.0))

    assert result["passed"] is False
    assert result["gates"]["logits_max_abs"] is False
    assert result["gates"]["losses_max_abs"] is False
