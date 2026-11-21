"""Small statistics helpers for profiling reports."""

from __future__ import annotations

import random
import statistics
from collections.abc import Callable, Sequence


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] | None = None,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    rng: random.Random | None = None,
) -> tuple[float, float, float]:
    """Return ``(ci_lower, ci_upper, statistic_value)`` for ``values``."""
    n = len(values)
    if n == 0:
        raise ValueError("bootstrap_ci requires at least one sample")
    stat_fn = statistic or statistics.median
    values_list = [float(v) for v in values]
    point = float(stat_fn(values_list))
    if n == 1 or all(v == values_list[0] for v in values_list):
        return point, point, point

    rng = rng if rng is not None else random.Random(0)
    samples: list[float] = []
    for _ in range(n_resamples):
        resample = [values_list[rng.randrange(n)] for _ in range(n)]
        samples.append(float(stat_fn(resample)))
    samples.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_idx = int(alpha * n_resamples)
    hi_idx = min(n_resamples - 1, int((1.0 - alpha) * n_resamples))
    return samples[lo_idx], samples[hi_idx], point


def paired_speedup_ci(
    baseline: Sequence[float],
    optimized: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] | None = None,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    rng: random.Random | None = None,
) -> tuple[float, float, float]:
    """Paired-bootstrap CI of the speedup RATIO ``stat(baseline) / stat(optimized)``.

    The two sequences are unpaired per-iteration timing samples (same number of
    iterations) of the SAME workload measured under the baseline vs. optimized
    implementation. Each bootstrap iteration resamples ``baseline`` and
    ``optimized`` independently (with replacement) and records the ratio of their
    statistics; the returned ``(ci_lower, ci_upper, point_ratio)`` is the
    percentile CI of that ratio plus the point estimate
    ``stat(baseline) / stat(optimized)``.

    A CI that EXCLUDES 1.0 is the significance signal recommended in PR2's
    profiling report (the codebase has no Mann-Whitney test): ``ci_lower > 1.0``
    ⇒ a statistically clear speedup at ``confidence`` level. ``statistic``
    defaults to the median (robust to timing outliers).

    Constant or single-sample inputs collapse to a zero-width CI at the point
    ratio (mirrors :func:`bootstrap_ci`). Raises ``ValueError`` on empty or
    unequal-length inputs, or when the optimized statistic is 0 (undefined ratio).
    """
    if len(baseline) != len(optimized):
        raise ValueError(
            "paired_speedup_ci requires baseline and optimized of the same length"
        )
    n = len(baseline)
    if n == 0:
        raise ValueError("paired_speedup_ci requires at least one paired sample")
    stat_fn = statistic or statistics.median
    base_list = [float(v) for v in baseline]
    opt_list = [float(v) for v in optimized]
    base_point = float(stat_fn(base_list))
    opt_point = float(stat_fn(opt_list))
    if opt_point == 0.0:
        raise ValueError(
            "paired_speedup_ci: optimized statistic is 0 (speedup ratio undefined)"
        )
    point = base_point / opt_point

    base_const = n == 1 or all(v == base_list[0] for v in base_list)
    opt_const = n == 1 or all(v == opt_list[0] for v in opt_list)
    if base_const and opt_const:
        return point, point, point

    rng = rng if rng is not None else random.Random(0)
    ratios: list[float] = []
    for _ in range(n_resamples):
        b_res = [base_list[rng.randrange(n)] for _ in range(n)]
        o_res = [opt_list[rng.randrange(n)] for _ in range(n)]
        o_stat = float(stat_fn(o_res))
        if o_stat == 0.0:
            continue  # degenerate resample; skip (undefined ratio)
        ratios.append(float(stat_fn(b_res)) / o_stat)
    if not ratios:
        return point, point, point
    ratios.sort()
    m = len(ratios)
    alpha = (1.0 - confidence) / 2.0
    lo_idx = int(alpha * m)
    hi_idx = min(m - 1, int((1.0 - alpha) * m))
    return ratios[lo_idx], ratios[hi_idx], point
