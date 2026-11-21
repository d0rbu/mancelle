"""Extraction profiling: GPU-idle (primary metric), region timings, shards/hour.

Free-when-off: extraction_profiler(enabled=False) yields None and installs
nothing, so timed_region() in the hot loop pays nothing in production. When
enabled it installs a TimingRecorder (region timings) and, on local-rank 0, a
GPUUtilizationMonitor; on exit it writes a CSV (rows) + JSON (summary) sidecar.
The headline metric is gpu_idle_fraction — the share of GPU samples below
idle_threshold — per issue #564's true goal of minimizing GPU idle time.
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from steering.profiling.gpu_utilization import GPUUtilizationMonitor
from steering.profiling.markers import TimingRecorder, use_recorder


def _idle_fraction(samples: list[float], threshold: float) -> float:
    if not samples:
        return 0.0
    return sum(1 for s in samples if s < threshold) / len(samples)


def _longest_idle_run_s(samples: list[float], threshold: float, poll_interval_s: float) -> float:
    longest = run = 0
    for s in samples:
        run = run + 1 if s < threshold else 0
        longest = max(longest, run)
    return longest * poll_interval_s


@dataclass
class ExtractionProfile:
    recorder: TimingRecorder
    _monitor: GPUUtilizationMonitor | None
    _poll_interval_s: float
    _idle_threshold: float
    shards_completed: int = 0

    def mark_shard_done(self) -> None:
        self.shards_completed += 1

    def summary(self, wall_seconds: float) -> dict:
        samples = self._monitor.samples if self._monitor is not None else []
        mean = (sum(samples) / len(samples)) if samples else 0.0
        return {
            "gpu_idle_fraction": _idle_fraction(samples, self._idle_threshold),
            "gpu_longest_idle_s": _longest_idle_run_s(samples, self._idle_threshold, self._poll_interval_s),
            "gpu_util_mean": mean,
            "gpu_sample_count": len(samples),
            "shards_completed": self.shards_completed,
            "shards_per_hour": (self.shards_completed / wall_seconds * 3600.0) if wall_seconds > 0 else 0.0,
            "wall_seconds": wall_seconds,
        }


def _write_sidecar(profile: ExtractionProfile, sidecar_path: Path, wall_seconds: float, rank: int) -> None:
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    # APPEND the extension by string — never Path.with_suffix, which replaces
    # everything after the last dot and would strip the per-rank suffix from a
    # dotted stem (e.g. "extraction.profile_rank0" -> "extraction.csv"), collapsing
    # every rank onto one file.
    profile.recorder.to_csv(Path(f"{sidecar_path}.csv"))
    summary = profile.summary(wall_seconds)
    summary["rank"] = rank
    Path(f"{sidecar_path}.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "extraction profile (rank {}): gpu_idle_fraction={:.3f} longest_idle={:.1f}s shards/hr={:.1f}",
        rank, summary["gpu_idle_fraction"], summary["gpu_longest_idle_s"], summary["shards_per_hour"],
    )


@contextmanager
def extraction_profiler(
    *,
    enabled: bool,
    rank: int,
    local_rank: int,
    sidecar_path: str | Path,
    poll_interval_ms: int = 50,
    idle_threshold: float = 5.0,
) -> Iterator[ExtractionProfile | None]:
    if not enabled:
        yield None
        return
    recorder = TimingRecorder()
    monitor = (
        GPUUtilizationMonitor(poll_interval_ms=poll_interval_ms, device_index=local_rank)
        if local_rank == 0
        else None
    )
    profile = ExtractionProfile(recorder, monitor, poll_interval_ms / 1000.0, idle_threshold)
    start = time.perf_counter()
    sidecar = Path(sidecar_path)
    # ExitStack unwinds LIFO: register the sidecar callback FIRST so it runs LAST —
    # after the monitor thread has stopped and samples are final.
    with ExitStack() as stack:
        stack.callback(lambda: _write_sidecar(profile, sidecar, time.perf_counter() - start, rank))
        stack.enter_context(use_recorder(recorder))
        if monitor is not None:
            stack.enter_context(monitor)
        yield profile
