"""Exact unit-contract tests for AxBench percentile QC thresholds."""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from steering.eval.axbench_base.percentile_units import (
    PERCENTILE_POINTS_PER_FRACTION,
    percentile_delta_meets_or_exceeds,
    percentile_fraction_to_points,
)
from steering.eval.axbench_base.types import AxBenchBaseError


def test_percentile_points_per_fraction_is_exactly_one_hundred() -> None:
    assert PERCENTILE_POINTS_PER_FRACTION == 100.0


@pytest.mark.parametrize(
    ("fraction", "expected_points"),
    [
        (0, 0.0),
        (0.0, 0.0),
        (0.001, 0.1),
        (0.3, 30.0),
        (0.5, 50.0),
        (0.7, 70.0),
        (0.999, 99.9),
        (1, 100.0),
        (1.0, 100.0),
        (Fraction(3, 10), 30.0),
    ],
)
def test_percentile_fraction_to_points_exact_examples(
    fraction: object,
    expected_points: float,
) -> None:
    assert percentile_fraction_to_points(
        fraction,
        field_name="threshold",
    ) == expected_points


@pytest.mark.parametrize(
    "bad",
    [
        True,
        False,
        None,
        "0.3",
        b"0.3",
        [],
        {},
        object(),
        -math.inf,
        math.inf,
        math.nan,
        -1.0,
        -1e-12,
        1.0 + 1e-12,
        2.0,
    ],
)
def test_percentile_fraction_to_points_rejects_invalid_values(
    bad: object,
) -> None:
    with pytest.raises(
        AxBenchBaseError,
        match=r"unit_test_threshold.*finite real percentile fraction.*\[0, 1\]",
    ):
        percentile_fraction_to_points(
            bad,
            field_name="unit_test_threshold",
        )


def test_percentile_fraction_error_preserves_value_and_runtime_type() -> None:
    with pytest.raises(AxBenchBaseError) as caught:
        percentile_fraction_to_points(
            "0.3",
            field_name="accept_threshold",
        )
    message = str(caught.value)
    assert "accept_threshold" in message
    assert "'0.3'" in message
    assert "str" in message


@pytest.mark.parametrize(
    ("delta", "threshold", "expected"),
    [
        (-30.0, 30.0, False),
        (-0.0, 0.0, True),
        (0.0, 0.0, True),
        (0.5, 30.0, False),
        (29.999999999, 30.0, False),
        (30.0, 30.0, True),
        (30.000000001, 30.0, True),
        (99.5, 100.0, False),
        (100.0, 100.0, True),
        (Fraction(3, 10), Fraction(3, 10), True),
    ],
)
def test_percentile_delta_meets_or_exceeds_is_inclusive(
    delta: object,
    threshold: object,
    expected: bool,
) -> None:
    assert percentile_delta_meets_or_exceeds(delta, threshold) is expected


@pytest.mark.parametrize(
    ("delta", "threshold", "bad_field"),
    [
        (math.nan, 30.0, "delta_points"),
        (math.inf, 30.0, "delta_points"),
        (-math.inf, 30.0, "delta_points"),
        (True, 30.0, "delta_points"),
        ("30", 30.0, "delta_points"),
        (30.0, math.nan, "threshold_points"),
        (30.0, math.inf, "threshold_points"),
        (30.0, False, "threshold_points"),
        (30.0, None, "threshold_points"),
    ],
)
def test_percentile_delta_meets_or_exceeds_fails_loudly_on_nonfinite_or_nonreal(
    delta: object,
    threshold: object,
    bad_field: str,
) -> None:
    with pytest.raises(
        AxBenchBaseError,
        match=rf"{bad_field} must be a finite real number",
    ):
        percentile_delta_meets_or_exceeds(delta, threshold)


def test_issue_1048_minimum_rejection_and_acceptance_contract() -> None:
    threshold_points = percentile_fraction_to_points(
        0.3,
        field_name="accept_threshold",
    )
    assert threshold_points == 30.0
    assert not percentile_delta_meets_or_exceeds(0.5, threshold_points)
    assert percentile_delta_meets_or_exceeds(30.0, threshold_points)


def test_verification_and_accept_thresholds_share_the_same_conversion() -> None:
    accept_points = percentile_fraction_to_points(
        0.3,
        field_name="accept_threshold",
    )
    verification_points = percentile_fraction_to_points(
        0.7,
        field_name="verification_percentile_threshold",
    )
    assert accept_points == 30.0
    assert verification_points == 70.0
    assert verification_points / accept_points == pytest.approx(7 / 3)
