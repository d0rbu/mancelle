"""Property-based invariants for compute_mean_stderr.

Hypothesis-driven invariants:

- mean ∈ [min(clean), max(clean)] for any non-empty input.
- stderr >= 0 for any non-empty input.
- The NaN-filter is monotonic in ``len(clean)`` — adding NaN to the input
  does not change the result; adding finite values does.
"""

from __future__ import annotations

import math

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from steering.eval._stats_helpers import (
    compute_mean_stderr,
    compute_mean_stderr_or_nan,
    compute_mean_stderr_or_zero,
)

_FINITE_FLOAT = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)


@settings(
    derandomize=True,
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(st.lists(_FINITE_FLOAT, min_size=1, max_size=50))
def test_mean_in_range(vals: list[float]) -> None:
    """mean ∈ [min, max] for any all-finite input."""
    m, _ = compute_mean_stderr(vals)
    lo = min(vals)
    hi = max(vals)
    # Float roundoff: allow tiny epsilon
    eps = 1e-9 * max(1.0, abs(lo), abs(hi))
    assert lo - eps <= m <= hi + eps


@settings(
    derandomize=True,
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(st.lists(_FINITE_FLOAT, min_size=1, max_size=50))
def test_stderr_non_negative(vals: list[float]) -> None:
    _, se = compute_mean_stderr(vals)
    assert se >= 0.0


@settings(
    derandomize=True,
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(st.lists(_FINITE_FLOAT, min_size=1, max_size=50))
def test_nan_filter_idempotent(vals: list[float]) -> None:
    """Adding NaNs to the input does not change the result."""
    m1, se1 = compute_mean_stderr(vals)
    m2, se2 = compute_mean_stderr(vals + [math.nan, math.nan])
    assert m1 == m2
    assert se1 == se2


@settings(
    derandomize=True,
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(st.lists(_FINITE_FLOAT, min_size=1, max_size=20))
def test_or_nan_wrapper_matches_canonical_when_clean(vals: list[float]) -> None:
    """``compute_mean_stderr_or_nan`` matches ``compute_mean_stderr`` on
    all-finite inputs (the wrapper only diverges on empty/all-NaN)."""
    m1, se1 = compute_mean_stderr(vals)
    m2, se2 = compute_mean_stderr_or_nan(vals)
    assert m1 == m2
    assert se1 == se2


@settings(
    derandomize=True,
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(st.lists(_FINITE_FLOAT, min_size=1, max_size=20))
def test_or_zero_wrapper_matches_canonical_when_clean(vals: list[float]) -> None:
    m1, se1 = compute_mean_stderr(vals)
    m2, se2 = compute_mean_stderr_or_zero(vals)
    assert m1 == m2
    assert se1 == se2


@settings(
    derandomize=True,
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(st.lists(_FINITE_FLOAT, min_size=2, max_size=20))
def test_constant_input_yields_zero_stderr(vals_seed: list[float]) -> None:
    """All identical values → stderr is ~0 (sample variance is theoretically 0,
    in practice limited by float roundoff to a tiny relative epsilon)."""
    if not vals_seed:
        return
    constant = vals_seed[0]
    n = len(vals_seed)
    m, se = compute_mean_stderr([constant] * n)
    # Float roundoff: mean and stderr can drift by ulp-scale errors when
    # ``constant`` is large. Use relative tolerance keyed to magnitude.
    mag = max(1.0, abs(constant))
    eps_mean = 1e-12 * mag
    eps_se = 1e-9 * mag
    assert abs(m - constant) <= eps_mean
    assert abs(se) <= eps_se
