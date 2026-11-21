"""Shared timing helpers for benchmark harnesses."""

from __future__ import annotations

import gc
import statistics
import time
from collections.abc import Callable
from typing import Any

import torch as t

from steering.profiling import bootstrap_ci


def _cuda_sync(device: str) -> None:
    if device == "cuda" and t.cuda.is_available():
        t.cuda.synchronize()


def _free() -> None:
    gc.collect()
    if t.cuda.is_available():
        t.cuda.empty_cache()


def _p90(values: list[float]) -> float:
    """90th percentile of *values* (0.0 if empty). Shared by the packed-rows + token-boundary
    benchmark harnesses (issue #540)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))
    return ordered[idx]


def _median(values: list[float]) -> float:
    """Median of *values* (0.0 if empty). Shared by the packed-rows + token-boundary harnesses."""
    return statistics.median(values) if values else 0.0


def benchmark_fn(
    fn: Callable[[], Any], n_warmup: int, n_iters: int, device: str
) -> dict[str, Any]:
    """Warmup (discarded) then N device-synced timed iterations of ``fn``.

    Returns median/mean/std/min/max in ms + the 95% bootstrap CI of the median
    (ms). The benchmarks group per-region breakdowns via ``timed_region``; this
    measures the whole ``fn``.
    """
    for _ in range(n_warmup):
        fn()
    _cuda_sync(device)

    times_s: list[float] = []
    for _ in range(n_iters):
        _cuda_sync(device)
        start = time.perf_counter()
        fn()
        _cuda_sync(device)
        times_s.append(time.perf_counter() - start)

    tt = t.tensor(times_s)
    ms = [s * 1000.0 for s in times_s]
    ci_lo, ci_hi, ci_med = bootstrap_ci(ms)
    return {
        "median_ms": float(tt.median() * 1000),
        "mean_ms": float(tt.mean() * 1000),
        "std_ms": float(tt.std() * 1000),
        "min_ms": float(tt.min() * 1000),
        "max_ms": float(tt.max() * 1000),
        "ci_lo_ms": ci_lo,
        "ci_hi_ms": ci_hi,
        "ci_med_ms": ci_med,
        "n_iters": n_iters,
    }
