"""Paired CUDA timings with independent snapshots of every GCG result."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch as t


def measure_paired(
    operation: Callable[[], Any],
    repeats: int,
    *,
    reference: Callable[[], AbstractContextManager[Any]],
    numerics: Callable[[Any], t.Tensor],
    warmups: int,
) -> tuple[Any, dict[str, Any]]:
    """Alternate order; exclude patch setup and snapshots from measured time."""
    samples: dict[str, list[float]] = {"reference": [], "candidate": []}
    peaks: dict[str, list[int]] = {"reference": [], "candidate": []}
    for pair in range(-warmups, repeats):
        order = (
            ("reference", "candidate") if pair < 0 or pair % 2 == 0 else ("candidate", "reference")
        )
        values, snapshots = {}, {}
        for arm in order:
            with reference() if arm == "reference" else nullcontext():
                t.cuda.synchronize()
                t.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                values[arm] = operation()
                t.cuda.synchronize()
                elapsed = time.perf_counter() - start
                peak = t.cuda.max_memory_allocated()
                # A worker can overwrite its sidecar on the next operation.
                snapshots[arm] = numerics(values[arm]).detach().clone()
                if pair >= 0:
                    samples[arm].append(elapsed)
                    peaks[arm].append(peak)
        t.testing.assert_close(snapshots["candidate"], snapshots["reference"], rtol=0, atol=0)
    ratios = [
        old / new for old, new in zip(samples["reference"], samples["candidate"], strict=True)
    ]
    return values["candidate"], {
        "seconds": samples["candidate"],
        "median_seconds": statistics.median(samples["candidate"]),
        "peak_allocated": max(peaks["candidate"]),
        "reference_seconds": samples["reference"],
        "reference_median_seconds": statistics.median(samples["reference"]),
        "reference_peak_allocated": max(peaks["reference"]),
        "paired_speedups": ratios,
        "median_paired_speedup": statistics.median(ratios),
        "faster_pairs": sum(ratio > 1 for ratio in ratios),
        "exact_pairs": repeats,
    }
