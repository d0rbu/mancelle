"""Property tests for the shared AxBench percentile-unit boundary."""

from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st

from steering.eval.axbench_base.percentile_units import (
    percentile_delta_meets_or_exceeds,
    percentile_fraction_to_points,
)

_FRACTIONS = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)
_DELTAS = st.floats(
    min_value=-100.0,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)


@given(fraction=_FRACTIONS)
def test_fraction_to_points_matches_the_mathematical_reference(
    fraction: float,
) -> None:
    points = percentile_fraction_to_points(
        fraction,
        field_name="property_threshold",
    )
    assert points == fraction * 100.0
    assert math.isfinite(points)
    assert 0.0 <= points <= 100.0


@given(first=_FRACTIONS, second=_FRACTIONS)
def test_fraction_to_points_is_monotone(first: float, second: float) -> None:
    first_points = percentile_fraction_to_points(
        first,
        field_name="first",
    )
    second_points = percentile_fraction_to_points(
        second,
        field_name="second",
    )
    assert (first_points <= second_points) == (first <= second)


@given(delta=_DELTAS, fraction=_FRACTIONS)
def test_stage1b_decision_matches_normalized_reference(
    delta: float,
    fraction: float,
) -> None:
    threshold_points = percentile_fraction_to_points(
        fraction,
        field_name="accept_threshold",
    )
    actual = percentile_delta_meets_or_exceeds(delta, threshold_points)
    assert actual == (delta >= 100.0 * fraction)


@given(delta=_DELTAS, lower=_FRACTIONS, upper=_FRACTIONS)
def test_raising_threshold_never_turns_rejection_into_acceptance(
    delta: float,
    lower: float,
    upper: float,
) -> None:
    lower, upper = sorted((lower, upper))
    passes_lower = percentile_delta_meets_or_exceeds(
        delta,
        percentile_fraction_to_points(lower, field_name="lower"),
    )
    passes_upper = percentile_delta_meets_or_exceeds(
        delta,
        percentile_fraction_to_points(upper, field_name="upper"),
    )
    assert not passes_upper or passes_lower


@given(fraction=_FRACTIONS)
def test_exact_resolved_boundary_is_always_accepted(fraction: float) -> None:
    threshold_points = percentile_fraction_to_points(
        fraction,
        field_name="accept_threshold",
    )
    assert percentile_delta_meets_or_exceeds(
        threshold_points,
        threshold_points,
    )
