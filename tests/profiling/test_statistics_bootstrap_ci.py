"""Unit tests for steering.profiling.statistics.bootstrap_ci."""

from __future__ import annotations

import random
import statistics

import pytest

from steering.profiling.statistics import bootstrap_ci


def test_constant_sample_zero_width_ci():
    lo, hi, stat = bootstrap_ci([5.0, 5.0, 5.0, 5.0, 5.0])
    assert lo == 5.0
    assert hi == 5.0
    assert stat == 5.0


def test_single_value_returns_point_triple():
    lo, hi, stat = bootstrap_ci([3.7])
    assert (lo, hi, stat) == (3.7, 3.7, 3.7)


def test_empty_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_seed_determinism():
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    a = bootstrap_ci(v, rng=rng1)
    b = bootstrap_ci(v, rng=rng2)
    assert a == b


def test_mean_vs_median_statistic():
    # Skewed data: mean > median
    v = [1.0, 2.0, 3.0, 4.0, 100.0]
    _, _, med = bootstrap_ci(v, statistic=statistics.median)
    _, _, avg = bootstrap_ci(v, statistic=statistics.fmean)
    assert med < avg


def test_increasing_variance_widens_ci():
    narrow = bootstrap_ci([5.0, 5.1, 4.9, 5.0, 5.05], rng=random.Random(7))
    wide = bootstrap_ci([5.0, 2.0, 8.0, 3.0, 7.0], rng=random.Random(7))
    narrow_width = narrow[1] - narrow[0]
    wide_width = wide[1] - wide[0]
    assert wide_width > narrow_width


def test_normal_sample_contains_true_mean_most_of_the_time():
    rng = random.Random(2026)
    covered = 0
    trials = 100
    for _ in range(trials):
        sample = [rng.gauss(0.0, 1.0) for _ in range(100)]
        lo, hi, _stat = bootstrap_ci(
            sample,
            statistic=statistics.fmean,
            rng=random.Random(rng.randint(0, 2**31)),
        )
        if lo <= 0.0 <= hi:
            covered += 1
    # 95% CI should cover the true mean >= 94/100 of the time (slightly lenient).
    assert covered >= 94
