"""Pin the three semantic seams of compute_mean_stderr after consolidation.

Per manifest §1.4 the canonical implementation lives in
:mod:`steering.eval._stats_helpers`. The consolidated API has one strict
helper and two runtime wrappers with intentionally different empty-input
handling:

- ``compute_mean_stderr`` — raises ``ValidationError``.
- ``shuffled_conditioning_eval._mean_stderr`` — returns ``(NaN, NaN)``.
- ``scripts/eval/completion_steering_eval._compute_mean_stderr`` — returns
  ``(0.0, 0.0)``.

This file pins the remaining runtime seams via the canonical helper trio.
"""

from __future__ import annotations

import math

import pytest

from errors import ValidationError
from steering.eval._stats_helpers import (
    compute_mean_stderr,
    compute_mean_stderr_or_nan,
    compute_mean_stderr_or_zero,
)

# ---------------------------------------------------------------------------
# Canonical helper — raises on empty/all-NaN
# ---------------------------------------------------------------------------


class TestCanonicalComputeMeanStderr:
    def test_simple_three_values(self) -> None:
        m, se = compute_mean_stderr([1.0, 2.0, 3.0])
        assert m == pytest.approx(2.0)
        # variance = 1.0 (Bessel), se = sqrt(1/3)
        assert se == pytest.approx(math.sqrt(1.0 / 3.0))

    def test_single_value_returns_zero_stderr(self) -> None:
        m, se = compute_mean_stderr([42.0])
        assert m == 42.0
        assert se == 0.0

    def test_filters_nan(self) -> None:
        m, se = compute_mean_stderr([1.0, math.nan, 3.0])
        assert m == pytest.approx(2.0)
        # Two clean values, variance = 2.0, se = sqrt(2.0/2) = 1.0
        assert se == pytest.approx(1.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValidationError, match="all values are NaN or list is empty"):
            compute_mean_stderr([])

    def test_all_nan_raises(self) -> None:
        with pytest.raises(ValidationError, match="all values are NaN or list is empty"):
            compute_mean_stderr([math.nan, math.nan, math.nan])


# ---------------------------------------------------------------------------
# NaN-returning wrapper (shuffled_conditioning_eval seam)
# ---------------------------------------------------------------------------


class TestComputeMeanStderrOrNan:
    def test_normal_input_matches_canonical(self) -> None:
        m, se = compute_mean_stderr_or_nan([1.0, 2.0, 3.0])
        assert m == pytest.approx(2.0)
        assert se == pytest.approx(math.sqrt(1.0 / 3.0))

    def test_empty_returns_nan_pair(self) -> None:
        m, se = compute_mean_stderr_or_nan([])
        assert math.isnan(m)
        assert math.isnan(se)

    def test_all_nan_returns_nan_pair(self) -> None:
        m, se = compute_mean_stderr_or_nan([math.nan, math.nan])
        assert math.isnan(m)
        assert math.isnan(se)

    def test_partial_nan_filtered(self) -> None:
        m, se = compute_mean_stderr_or_nan([math.nan, 5.0])
        assert m == 5.0
        assert se == 0.0


# ---------------------------------------------------------------------------
# Zero-returning wrapper (scripts/eval seam — plotting)
# ---------------------------------------------------------------------------


class TestComputeMeanStderrOrZero:
    def test_normal_input_matches_canonical(self) -> None:
        m, se = compute_mean_stderr_or_zero([1.0, 2.0, 3.0])
        assert m == pytest.approx(2.0)
        assert se == pytest.approx(math.sqrt(1.0 / 3.0))

    def test_empty_returns_zero_pair(self) -> None:
        m, se = compute_mean_stderr_or_zero([])
        assert m == 0.0
        assert se == 0.0

    def test_all_nan_returns_zero_pair(self) -> None:
        m, se = compute_mean_stderr_or_zero([math.nan, math.nan])
        assert m == 0.0
        assert se == 0.0

    def test_partial_nan_filtered(self) -> None:
        m, se = compute_mean_stderr_or_zero([math.nan, 4.0, 6.0])
        assert m == 5.0
        assert se == pytest.approx(math.sqrt(2.0 / 2.0))
