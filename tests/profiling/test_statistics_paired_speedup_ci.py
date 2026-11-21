"""Unit tests for steering.profiling.statistics.paired_speedup_ci.

A paired-bootstrap CI of the speedup RATIO (baseline_stat / optimized_stat). A
CI that excludes 1.0 is the significance signal PR2 recommended for the PR3
before/after claim (the codebase has no Mann-Whitney helper).
"""

from __future__ import annotations

import random
import statistics

import pytest

from steering.profiling.statistics import paired_speedup_ci


def test_point_ratio_is_stat_ratio():
    # baseline median 10, optimized median 5 → 2x speedup point estimate.
    baseline = [10.0, 10.0, 10.0, 10.0, 10.0]
    optimized = [5.0, 5.0, 5.0, 5.0, 5.0]
    lo, hi, ratio = paired_speedup_ci(baseline, optimized)
    assert ratio == pytest.approx(2.0)
    # constant samples ⇒ zero-width CI at the point estimate.
    assert lo == pytest.approx(2.0)
    assert hi == pytest.approx(2.0)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        paired_speedup_ci([1.0, 2.0], [1.0])


def test_empty_raises():
    with pytest.raises(ValueError, match="at least one"):
        paired_speedup_ci([], [])


def test_speedup_ci_excludes_one_for_clear_win():
    # Optimized is consistently ~2x faster with small noise → CI well above 1.0.
    rng = random.Random(0)
    baseline = [rng.gauss(10.0, 0.3) for _ in range(60)]
    optimized = [rng.gauss(5.0, 0.15) for _ in range(60)]
    lo, hi, ratio = paired_speedup_ci(baseline, optimized, rng=random.Random(1))
    assert ratio > 1.5
    assert lo > 1.0  # significant speedup: CI excludes 1.0


def test_no_difference_ci_brackets_one():
    # Same distribution → ratio ~1.0, CI brackets 1.0 (not significant).
    rng = random.Random(3)
    baseline = [rng.gauss(7.0, 0.5) for _ in range(80)]
    optimized = [rng.gauss(7.0, 0.5) for _ in range(80)]
    lo, hi, ratio = paired_speedup_ci(baseline, optimized, rng=random.Random(4))
    assert lo <= 1.0 <= hi


def test_seed_determinism():
    b = [float(i) + 1.0 for i in range(20)]
    o = [0.5 * (float(i) + 1.0) for i in range(20)]
    a1 = paired_speedup_ci(b, o, rng=random.Random(42))
    a2 = paired_speedup_ci(b, o, rng=random.Random(42))
    assert a1 == a2


def test_statistic_override_mean():
    # With the mean statistic the point ratio is mean(baseline)/mean(optimized).
    baseline = [2.0, 4.0, 6.0]  # mean 4
    optimized = [1.0, 1.0, 1.0]  # mean 1
    _lo, _hi, ratio = paired_speedup_ci(
        baseline, optimized, statistic=statistics.fmean
    )
    assert ratio == pytest.approx(4.0)


def test_ci_lo_le_point_le_ci_hi():
    rng = random.Random(11)
    baseline = [rng.gauss(9.0, 1.0) for _ in range(50)]
    optimized = [rng.gauss(4.0, 0.5) for _ in range(50)]
    lo, hi, ratio = paired_speedup_ci(baseline, optimized, rng=random.Random(12))
    assert lo <= ratio <= hi


# ---------------------------------------------------------------------------
# Branch coverage for the zero-statistic guards + the single-sample fast path.
# These pin already-correct behavior; each raise/continue/fallback is
# mutation-proven (see the inline notes) so the assertions are non-vacuous.
# ---------------------------------------------------------------------------


def test_zero_optimized_statistic_raises():
    # opt_point == 0.0 (median of all-zeros) ⇒ the ratio is undefined and the
    # function raises BEFORE the bootstrap loop (statistics.py:80-83).
    # Mutation-proof: removing the `if opt_point == 0.0: raise` makes this a
    # ZeroDivisionError (point = base/0), not the contracted ValueError.
    baseline = [10.0, 10.0, 10.0]
    optimized = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="optimized statistic is 0"):
        paired_speedup_ci(baseline, optimized)


def test_degenerate_resample_is_skipped_ci_still_sane():
    # opt_point (median) is non-zero (5.0), but MANY zeros mean SOME bootstrap
    # resamples have median 0.0 → those are `continue`-skipped (statistics.py:
    # 97-98) while the rest still yield ratios, so the call returns a finite,
    # ordered, positive CI. Mutation-proof: replacing `continue` with appending
    # `0.0`/inf would push lo to 0.0 (or NaN-poison the sort), breaking the
    # `lo > 0` + ordering asserts below.
    baseline = [9.0, 10.0, 11.0, 10.0, 10.0]
    optimized = [0.0, 0.0, 5.0, 5.0, 5.0]
    lo, hi, point = paired_speedup_ci(
        baseline, optimized, rng=random.Random(0)
    )
    assert point == pytest.approx(10.0 / 5.0)  # median(base)/median(opt) = 2.0
    assert 0.0 < lo <= point <= hi
    assert lo == lo and hi == hi  # finite (not NaN from a 0-division leak)


def test_all_resamples_degenerate_falls_back_to_point():
    # Force the ONLY bootstrap resample to be degenerate (median 0.0): with
    # n_resamples=1 and seed 4, the single optimized resample drawn from
    # [0,0,5,5] is all-but-one zeros (median 0.0) → skipped → `ratios` empty →
    # the fallback `if not ratios: return (point, point, point)`
    # (statistics.py:100-101). opt_point (median=2.5) is non-zero so the
    # earlier raise does not fire. Mutation-proof: deleting the fallback makes
    # this raise IndexError on `ratios[lo_idx]` (empty list).
    baseline = [1.0, 2.0, 3.0, 4.0]
    optimized = [0.0, 0.0, 5.0, 5.0]
    lo, hi, point = paired_speedup_ci(
        baseline, optimized, n_resamples=1, rng=random.Random(4)
    )
    # median(base)=2.5, median(opt)=2.5 ⇒ point ratio 1.0; CI collapses to it.
    assert point == pytest.approx(1.0)
    assert lo == pytest.approx(point)
    assert hi == pytest.approx(point)


def test_single_sample_zero_width_ci():
    # n == 1 ⇒ both inputs are "constant" ⇒ the zero-width fast path
    # (statistics.py:88-89) returns (point, point, point) without resampling.
    lo, hi, point = paired_speedup_ci([10.0], [5.0])
    assert (lo, hi, point) == pytest.approx((2.0, 2.0, 2.0))
