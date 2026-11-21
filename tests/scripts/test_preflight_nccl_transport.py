from __future__ import annotations

import pytest
import torch as t

from scripts.benchmark.preflight_nccl_transport import (
    _aggregate_rank_results,
    _elements_for_mib,
    _positive_int,
    _resolve_dtype,
)


def test_elements_for_mib_matches_dtype_width() -> None:
    assert _elements_for_mib(1, t.bfloat16) == 2**19
    assert _elements_for_mib(1, t.float32) == 2**18


def test_transport_config_validation_is_strict() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        _positive_int(True, name="value")
    with pytest.raises(ValueError, match="must be one of"):
        _resolve_dtype("float16")


def test_rank_aggregation_uses_critical_rank_each_step() -> None:
    rank_results = [
        {
            "runs": [
                {
                    "operation": "reduce_scatter",
                    "shard_mib": 32,
                    "rank_samples_ms": [3.0, 8.0, 4.0],
                }
            ]
        },
        {
            "runs": [
                {
                    "operation": "reduce_scatter",
                    "shard_mib": 32,
                    "rank_samples_ms": [5.0, 2.0, 7.0],
                }
            ]
        },
    ]

    result = _aggregate_rank_results(rank_results)

    assert result[0]["critical_samples_ms"] == [5.0, 8.0, 7.0]
    assert result[0]["critical"]["median_ms"] == 7.0
