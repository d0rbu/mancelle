"""Behavioral tests for the isotonic-regression helpers in steering_eval.py.

``isotonic_regression_pava`` (steering_eval.py:362) is a real Pool Adjacent
Violators Algorithm implementation and ``compute_isotonic_r2`` (line 393) is its
goodness-of-fit summary.  Both are still live but their tests were dropped in the
encdec-full-conditioning-cleanup PR.  We assert the exact numeric outputs of the
algorithm on small hand-checked inputs, not merely that it runs.
"""

from __future__ import annotations

import itertools
import math

from steering.eval.steering_eval import (
    compute_isotonic_r2,
    isotonic_regression_pava,
)

# ---------------------------------------------------------------------------
# isotonic_regression_pava — exact numeric output
# ---------------------------------------------------------------------------


class TestIsotonicRegressionPava:
    def test_already_monotonic_passthrough(self) -> None:
        # A non-decreasing input is its own isotonic fit.
        values = [0.1, 0.3, 0.5, 0.7, 0.9]
        result = isotonic_regression_pava(values)
        assert len(result) == len(values)
        for got, want in zip(result, values, strict=True):
            assert math.isclose(got, want, abs_tol=1e-12)

    def test_single_violation_is_pooled(self) -> None:
        # [0.5, 0.3, 0.7]: the first two violate monotonicity and pool to their
        # mean (0.4, 0.4); the trailing 0.7 already exceeds 0.4 and is untouched.
        values = [0.5, 0.3, 0.7]
        result = isotonic_regression_pava(values)
        assert math.isclose(result[0], 0.4, abs_tol=1e-12)
        assert math.isclose(result[1], 0.4, abs_tol=1e-12)
        assert math.isclose(result[2], 0.7, abs_tol=1e-12)

    def test_all_decreasing_pools_to_global_mean(self) -> None:
        # A strictly decreasing input collapses into one block at the mean.
        values = [4.0, 3.0, 2.0, 1.0]
        result = isotonic_regression_pava(values)
        mean = sum(values) / len(values)  # 2.5
        assert all(math.isclose(v, mean, abs_tol=1e-12) for v in result)

    def test_constant_input_unchanged(self) -> None:
        values = [0.5, 0.5, 0.5]
        result = isotonic_regression_pava(values)
        assert all(math.isclose(v, 0.5, abs_tol=1e-12) for v in result)

    def test_output_is_non_decreasing(self) -> None:
        # The defining property of the fit: monotone non-decreasing.
        values = [0.2, 0.9, 0.1, 0.8, 0.3, 0.7]
        result = isotonic_regression_pava(values)
        for a, b in itertools.pairwise(result):
            assert a <= b + 1e-12

    def test_pava_preserves_total_sum(self) -> None:
        # PAVA only pools by averaging, so the sum is conserved.
        values = [3.0, 1.0, 2.0, 0.0, 5.0]
        result = isotonic_regression_pava(values)
        assert math.isclose(sum(result), sum(values), abs_tol=1e-9)

    def test_middle_block_pools_three(self) -> None:
        # [1, 4, 2, 1, 5]: the run 4,2,1 violates and pools to mean 7/3;
        # leading 1 < 7/3 stays, trailing 5 > 7/3 stays.
        values = [1.0, 4.0, 2.0, 1.0, 5.0]
        result = isotonic_regression_pava(values)
        third = 7.0 / 3.0
        assert math.isclose(result[0], 1.0, abs_tol=1e-12)
        assert math.isclose(result[1], third, abs_tol=1e-12)
        assert math.isclose(result[2], third, abs_tol=1e-12)
        assert math.isclose(result[3], third, abs_tol=1e-12)
        assert math.isclose(result[4], 5.0, abs_tol=1e-12)

    def test_singleton(self) -> None:
        assert isotonic_regression_pava([0.42]) == [0.42]


# ---------------------------------------------------------------------------
# compute_isotonic_r2 — R² of the isotonic fit
# ---------------------------------------------------------------------------


class TestComputeIsotonicR2:
    def test_perfect_fit_returns_one(self) -> None:
        # Already-monotonic data fits perfectly => R² == 1.
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        r2 = compute_isotonic_r2(values)
        assert math.isclose(r2, 1.0, abs_tol=1e-12)

    def test_constant_input_returns_one(self) -> None:
        # ss_tot ~ 0 path: the function returns 1.0 by definition.
        r2 = compute_isotonic_r2([0.5, 0.5, 0.5, 0.5])
        assert r2 == 1.0

    def test_noisy_fit_in_unit_interval(self) -> None:
        # A noisy-but-trending sequence yields a partial fit strictly inside (0, 1).
        values = [0.1, 0.3, 0.2, 0.5, 0.4, 0.6, 0.55, 0.8, 0.7, 0.9]
        r2 = compute_isotonic_r2(values)
        assert 0.0 < r2 < 1.0

    def test_r2_matches_manual_formula(self) -> None:
        # Hand-compute against the fitted block means for [0.5, 0.3, 0.7].
        values = [0.5, 0.3, 0.7]
        fitted = [0.4, 0.4, 0.7]
        mean = sum(values) / len(values)
        ss_res = sum((v - f) ** 2 for v, f in zip(values, fitted, strict=True))
        ss_tot = sum((v - mean) ** 2 for v in values)
        expected = 1.0 - ss_res / ss_tot
        assert math.isclose(compute_isotonic_r2(values), expected, rel_tol=1e-9)

    def test_anti_monotonic_low_r2(self) -> None:
        # A decreasing sequence pools to a flat line (the mean) => ss_res == ss_tot,
        # so R² collapses to 0.
        values = [0.9, 0.7, 0.5, 0.3, 0.1]
        r2 = compute_isotonic_r2(values)
        assert math.isclose(r2, 0.0, abs_tol=1e-12)
