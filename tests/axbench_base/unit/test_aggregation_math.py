"""Unit tests for aggregation math helpers (Task 39)."""

from __future__ import annotations

import pytest

from steering.eval.axbench_base.aggregation import (
    best_of_grid,
    bootstrap_ci,
    hmean,
)
from steering.eval.axbench_base.judge_parser import JUDGE_MAX_SCORE
from steering.eval.axbench_base.types import AxBenchBaseError

# ---------------------------------------------------------------------------
# hmean
# ---------------------------------------------------------------------------


def test_hmean_zero_bottom() -> None:
    assert hmean(0, 2) == 0.0
    assert hmean(2, 0) == 0.0
    assert hmean(0, 0) == 0.0


def test_hmean_both_two() -> None:
    assert hmean(2, 2) == 2.0


def test_hmean_both_one() -> None:
    assert hmean(1, 1) == 1.0


def test_hmean_symmetric() -> None:
    for a in range(3):
        for b in range(3):
            assert hmean(a, b) == hmean(b, a)


def test_hmean_bounded_0_to_judge_max_score() -> None:
    """``hmean`` never exceeds the AxBench judge scale's upper bound.

    The 0/1/2 scale is surfaced as
    :data:`~steering.eval.axbench_base.judge_parser.JUDGE_MAX_SCORE`
    so the test doesn't hardcode the sentinel ``2``.
    """
    for a in range(JUDGE_MAX_SCORE + 1):
        for b in range(JUDGE_MAX_SCORE + 1):
            result = hmean(a, b)
            assert 0 <= result <= JUDGE_MAX_SCORE


def test_hmean_one_and_two() -> None:
    assert hmean(1, 2) == pytest.approx(4 / 3)


# ---------------------------------------------------------------------------
# best_of_grid
# ---------------------------------------------------------------------------


def test_best_of_grid_max() -> None:
    assert best_of_grid([1.0, 3.5, 2.0]) == 3.5


def test_best_of_grid_single_element() -> None:
    assert best_of_grid([7.0]) == 7.0


def test_best_of_grid_negative_values() -> None:
    assert best_of_grid([-3.0, -1.0, -2.0]) == -1.0


def test_best_of_grid_empty_raises() -> None:
    with pytest.raises(AxBenchBaseError):
        best_of_grid([])


def test_best_of_grid_all_same_value() -> None:
    assert best_of_grid([4.0, 4.0, 4.0]) == 4.0


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


def test_bootstrap_ci_deterministic() -> None:
    values = [0.1, 0.5, 0.9, 0.3, 0.7]
    ci1 = bootstrap_ci(values, seed=42)
    ci2 = bootstrap_ci(values, seed=42)
    assert ci1 == ci2


def test_bootstrap_ci_different_seeds_differ() -> None:
    values = [float(i) for i in range(20)]
    ci_42 = bootstrap_ci(values, seed=42)
    ci_99 = bootstrap_ci(values, seed=99)
    # Different seeds should (almost certainly) produce different CIs for
    # a non-trivial distribution.
    assert ci_42 != ci_99


def test_bootstrap_ci_single_value() -> None:
    lower, upper = bootstrap_ci([0.5])
    assert lower == 0.5
    assert upper == 0.5


def test_bootstrap_ci_empty_raises() -> None:
    with pytest.raises(AxBenchBaseError):
        bootstrap_ci([])


def test_bootstrap_ci_constant_values() -> None:
    lower, upper = bootstrap_ci([1.0, 1.0, 1.0])
    assert lower == 1.0
    assert upper == 1.0


def test_bootstrap_ci_alpha_covered() -> None:
    """95% CI for the mean of uniform[0,1] should contain 0.5 at n=1000."""
    import random
    rng = random.Random(2024)
    values = [rng.random() for _ in range(1000)]
    lower, upper = bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=7)
    assert lower <= 0.5 <= upper


def test_bootstrap_ci_default_params() -> None:
    values = [0.2, 0.4, 0.6, 0.8]
    lower, upper = bootstrap_ci(values)
    # Just verifies default n_boot=1000, alpha=0.05, seed=42 run without error.
    assert lower <= upper

