"""Opt-in timing sidecar for feature-statistics computation."""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nanochat.atomic_io import atomic_json_dump

STATISTICS_PROFILE_SCHEMA_VERSION = 2
STATISTICS_SHARD_PHASES: tuple[str, ...] = (
    "load",
    "validate",
    "exact_counts",
    "moments",
    "segment_stats",
    "bookkeeping",
    "checkpoint",
)


def _empty_phase_seconds() -> dict[str, float]:
    return dict.fromkeys(STATISTICS_SHARD_PHASES, 0.0)


@dataclass
class StatisticsShardProfile:
    """Timing and fixed workload counters for one shard."""

    shard_index: int
    shard_name: str
    input_bytes: int = 0
    n_activations: int = 0
    n_segments: int = 0
    phase_seconds: dict[str, float] = field(default_factory=_empty_phase_seconds)

    @staticmethod
    def start() -> int:
        return time.perf_counter_ns()

    def finish(self, phase: str, started_ns: int) -> None:
        if phase not in STATISTICS_SHARD_PHASES:
            raise ValueError(f"unknown statistics shard profile phase {phase!r}")
        elapsed = (time.perf_counter_ns() - started_ns) / 1_000_000_000
        self.phase_seconds[phase] += elapsed

    def record_workload(
        self,
        *,
        n_activations: int,
        n_segments: int,
        input_bytes: int | None = None,
    ) -> None:
        self.n_activations = n_activations
        self.n_segments = n_segments
        if input_bytes is not None:
            self.input_bytes = input_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_index": self.shard_index,
            "shard_name": self.shard_name,
            "input_bytes": self.input_bytes,
            "n_activations": self.n_activations,
            "n_segments": self.n_segments,
            "phase_seconds": dict(self.phase_seconds),
        }


@dataclass
class StatisticsProfile:
    """Small in-memory timing sink; instantiated only for profiled runs."""

    config: dict[str, Any]
    wall_seconds: float = 0.0
    run_regions_seconds: dict[str, float] = field(default_factory=dict)
    per_shard: dict[int, StatisticsShardProfile] = field(default_factory=dict)
    remote_transport: dict[str, Any] | None = None

    @staticmethod
    def start() -> int:
        return time.perf_counter_ns()

    def finish_run_region(self, name: str, started_ns: int) -> None:
        elapsed = (time.perf_counter_ns() - started_ns) / 1_000_000_000
        self.run_regions_seconds[name] = self.run_regions_seconds.get(name, 0.0) + elapsed

    def shard(self, shard_index: int, shard_name: str) -> StatisticsShardProfile:
        shard = self.per_shard.get(shard_index)
        if shard is None:
            shard = StatisticsShardProfile(shard_index, shard_name)
            self.per_shard[shard_index] = shard
        elif shard.shard_name != shard_name:
            raise ValueError(f"shard index {shard_index} has conflicting profile names")
        return shard

    def to_dict(self) -> dict[str, Any]:
        ordered_profiles = [self.per_shard[index] for index in sorted(self.per_shard)]
        ordered_shards = [shard.to_dict() for shard in ordered_profiles]
        phase_totals = {
            phase: sum(shard["phase_seconds"][phase] for shard in ordered_shards)
            for phase in STATISTICS_SHARD_PHASES
        }
        input_bytes = sum(int(shard["input_bytes"]) for shard in ordered_shards)
        n_activations = sum(int(shard["n_activations"]) for shard in ordered_shards)
        wall = self.wall_seconds
        payload = {
            "schema_version": STATISTICS_PROFILE_SCHEMA_VERSION,
            "config": dict(self.config),
            "run_regions_seconds": dict(sorted(self.run_regions_seconds.items())),
            "shard_phase_seconds": phase_totals,
            "per_shard": ordered_shards,
            "totals": {
                "wall_seconds": wall,
                "n_shards_profiled": len(ordered_shards),
                "input_bytes": input_bytes,
                "n_activations": n_activations,
                "shards_per_hour": len(ordered_shards) / wall * 3600.0 if wall else 0.0,
                "input_gib_per_second": input_bytes / (1024**3) / wall if wall else 0.0,
                "million_activations_per_second": (
                    n_activations / 1_000_000 / wall if wall else 0.0
                ),
            },
        }
        if self.remote_transport is not None:
            payload["remote_transport"] = dict(self.remote_transport)
        return payload

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(self.to_dict(), path)


@contextmanager
def statistics_profiler(
    *,
    output_path: str | Path | None,
    config: Mapping[str, Any],
) -> Iterator[StatisticsProfile | None]:
    """Activate statistics timing only when an output sidecar is requested."""
    if output_path is None:
        yield None
        return

    profile = StatisticsProfile(config=dict(config))
    start = time.perf_counter_ns()
    try:
        yield profile
    except BaseException as exc:
        profile.wall_seconds = (time.perf_counter_ns() - start) / 1_000_000_000
        if profile.remote_transport is not None:
            profile.remote_transport["status"] = "failed"
            profile.remote_transport.setdefault(
                "terminal_error",
                {
                    "stage": "statistics",
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "index": None,
                    "path": None,
                },
            )
            try:
                profile.write(Path(output_path))
            except BaseException as profile_exc:
                exc.add_note(
                    "remote statistics failure profile also failed to write: "
                    f"{profile_exc!r}"
                )
        raise
    else:
        profile.wall_seconds = (time.perf_counter_ns() - start) / 1_000_000_000
        profile.write(Path(output_path))


__all__ = [
    "STATISTICS_PROFILE_SCHEMA_VERSION",
    "STATISTICS_SHARD_PHASES",
    "StatisticsProfile",
    "StatisticsShardProfile",
    "statistics_profiler",
]
