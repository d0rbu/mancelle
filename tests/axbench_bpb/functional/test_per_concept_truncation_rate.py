"""Per-concept truncation_rate must NOT inflate with the coefficient count K.

Regression coverage for a P0 numerical bug in
``steering/eval/axbench_bpb/bpb_aggregation.py::compute_per_concept_rollup``.
The previous implementation summed ``n_truncated_*`` across the K coefficient
rows but divided by the per-coefficient (single-row) sample count, producing
a rate K-times the true rate. With the default ``invalid_drop_threshold=0.05``
this silently dropped any method with even mild (~0.6%) per-call truncation.

The correct semantics are:
    per_coef_rate_i = (n_truncated_rich_i + n_truncated_avoid_i)
                       / (n_rich_i + n_avoid_i)
    truncation_rate = mean over the K coefficient rows of per_coef_rate_i
"""

from __future__ import annotations

import random

import pyarrow as pa
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from steering.eval.axbench_bpb.bpb_aggregation import compute_per_concept_rollup


def _make_per_tuple_row(
    method_name: str = "m",
    concept_id: int = 1,
    mode: str = "positive",
    target: str = "encdec",
    coefficient_str: str = "1",
    mean_bpb_intervened_rich: float = 1.0,
    mean_bpb_intervened_avoid: float = 2.0,
    n_rich: int = 10,
    n_avoid: int = 10,
    n_truncated_rich: int = 0,
    n_truncated_avoid: int = 0,
    margin_intervened: float = 1.0,
    primary_margin: float = 1.0,
    primary_bpb: float = 1.0,
) -> dict:
    """Construct one row matching the per-tuple rollup schema."""
    return {
        "method_name": method_name,
        "concept_id": concept_id,
        "mode": mode,
        "coefficient_str": coefficient_str,
        "target": target,
        "mean_bpb_intervened_rich": mean_bpb_intervened_rich,
        "n_rich": n_rich,
        "n_truncated_rich": n_truncated_rich,
        "mean_bpb_intervened_avoid": mean_bpb_intervened_avoid,
        "n_avoid": n_avoid,
        "n_truncated_avoid": n_truncated_avoid,
        "margin_intervened": margin_intervened,
        "primary_margin": primary_margin,
        "primary_bpb": primary_bpb,
    }


def _with_feature_diffmean_target(rows: list[dict]) -> list[dict]:
    return [
        _make_per_tuple_row(
            method_name="feature_diffmean_encdec",
            coefficient_str="None",
            target="encdec",
            primary_margin=1.0,
            margin_intervened=1.0,
        ),
        *rows,
    ]


def _method_under_test_row(concept: pa.Table) -> dict:
    return next(row for row in concept.to_pylist() if row["method_name"] == "m")


def test_truncation_rate_does_not_scale_with_coefficient_count():
    """K=5 dose-coefficients each with per-call rate 0.10 → concept rate 0.10.

    The buggy implementation reports 5 * (2/20) = 0.50 because it sums
    truncated counts across K rows but divides by a single row's sample count.
    """
    per_tuple = pa.Table.from_pylist(
        _with_feature_diffmean_target(
            [
                _make_per_tuple_row(
                    coefficient_str=str(c),
                    n_rich=10,
                    n_avoid=10,
                    n_truncated_rich=1,
                    n_truncated_avoid=1,
                )
                for c in range(5)
            ]
        )
    )
    concept = compute_per_concept_rollup(per_tuple, min_rows_per_side=1)
    row = _method_under_test_row(concept)
    assert row["truncation_rate"] == pytest.approx(0.10)


def test_truncation_rate_zero_when_no_truncation():
    """K=5 coefficients, zero truncation everywhere → rate 0.0."""
    per_tuple = pa.Table.from_pylist(
        _with_feature_diffmean_target(
            [
                _make_per_tuple_row(
                    coefficient_str=str(c),
                    n_rich=10,
                    n_avoid=10,
                    n_truncated_rich=0,
                    n_truncated_avoid=0,
                )
                for c in range(5)
            ]
        )
    )
    concept = compute_per_concept_rollup(per_tuple, min_rows_per_side=1)
    row = _method_under_test_row(concept)
    assert row["truncation_rate"] == pytest.approx(0.0)


def test_truncation_rate_one_when_fully_truncated():
    """K=5 coefficients, every sample truncated on both sides → rate 1.0."""
    per_tuple = pa.Table.from_pylist(
        _with_feature_diffmean_target(
            [
                _make_per_tuple_row(
                    coefficient_str=str(c),
                    n_rich=10,
                    n_avoid=10,
                    n_truncated_rich=10,
                    n_truncated_avoid=10,
                )
                for c in range(5)
            ]
        )
    )
    concept = compute_per_concept_rollup(per_tuple, min_rows_per_side=1)
    row = _method_under_test_row(concept)
    assert row["truncation_rate"] == pytest.approx(1.0)


def test_truncation_rate_handles_uneven_per_coefficient_counts():
    """Three coefficients with truncation 0/20, 5/20, 10/20 → mean 0.25."""
    per_tuple = pa.Table.from_pylist(
        _with_feature_diffmean_target(
            [
                _make_per_tuple_row(
                    coefficient_str="0",
                    n_rich=10,
                    n_avoid=10,
                    n_truncated_rich=0,
                    n_truncated_avoid=0,
                ),
                _make_per_tuple_row(
                    coefficient_str="1",
                    n_rich=10,
                    n_avoid=10,
                    n_truncated_rich=2,
                    n_truncated_avoid=3,
                ),
                _make_per_tuple_row(
                    coefficient_str="2",
                    n_rich=10,
                    n_avoid=10,
                    n_truncated_rich=5,
                    n_truncated_avoid=5,
                ),
            ]
        )
    )
    concept = compute_per_concept_rollup(per_tuple, min_rows_per_side=1)
    row = _method_under_test_row(concept)
    # (0/20 + 5/20 + 10/20) / 3 = (0 + 0.25 + 0.5) / 3 = 0.25
    assert row["truncation_rate"] == pytest.approx(0.25)


@given(
    n_rich=st.integers(min_value=1, max_value=200),
    n_avoid=st.integers(min_value=1, max_value=200),
    seed=st.integers(min_value=0, max_value=10_000),
    k=st.integers(min_value=1, max_value=10),
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_truncation_rate_in_unit_interval(n_rich: int, n_avoid: int, seed: int, k: int) -> None:
    """For any valid per-tuple table, the rate must lie in [0, 1].

    n_rich and n_avoid are constant per (method, concept, mode, target)
    group — they're a function of the underlying samples, not the
    coefficient. Truncation counts vary per coefficient row but cannot
    exceed the side's sample count.
    """
    rng = random.Random(seed)
    rows = [
        _make_per_tuple_row(
            coefficient_str=str(c),
            n_rich=n_rich,
            n_avoid=n_avoid,
            n_truncated_rich=rng.randint(0, n_rich),
            n_truncated_avoid=rng.randint(0, n_avoid),
        )
        for c in range(k)
    ]
    per_tuple = pa.Table.from_pylist(_with_feature_diffmean_target(rows))
    concept = compute_per_concept_rollup(per_tuple, min_rows_per_side=1)
    row = _method_under_test_row(concept)
    assert 0.0 <= row["truncation_rate"] <= 1.0
