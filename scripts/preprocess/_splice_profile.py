"""Profiling sidecar data holders for feature-shard splicing."""

from __future__ import annotations

import os
import resource
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from nanochat.atomic_io import atomic_json_dump

PHASE_KEYS: tuple[str, ...] = (
    "load_features",
    "read_parquet",
    "load_source_segments",
    "recombine",
    "build_derived_shard",
    "build_provenance",
    "write_parquet",
    "write_provenance",
    "save_features",
    "total",
)


DETAIL_KEYS: tuple[str, ...] = (
    "source.truncate_tokenize",
    "source.align_attach",
    "source.batch_roundtrip",
    "source.trim_unstable",
    "recombine.prepare",
    "recombine.select_cuts",
    "recombine.draw_derangement",
    "recombine.opposite_pairing",
    "recombine.prefilter_seams",
    "recombine.read_graph",
    "recombine.resolve_loops",
    "recombine.split_overflow",
    "recombine.full_doc_oracle",
    "derived.prepare",
    "derived.piece_boundaries",
    "derived.char_text",
    "derived.feature_refs",
    "derived.tensor_assembly",
    "provenance.placement",
    "provenance.docs",
    "save.sort_segments",
    "save.delta_encode",
    "save.assemble_payload",
    "save.safetensors_serialize",
    "save.zstd_compress",
    "save.atomic_write",
)


def _phase_seconds() -> dict[str, float]:
    return dict.fromkeys(PHASE_KEYS, 0.0)


def _detail_seconds() -> dict[str, float]:
    return dict.fromkeys(DETAIL_KEYS, 0.0)


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _current_rss_mb() -> float | None:
    statm = Path("/proc/self/statm")
    if not statm.exists():
        return None
    try:
        resident_pages = int(statm.read_text(encoding="utf-8").split()[1])
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        return None
    return resident_pages * page_size / (1024 * 1024)


def _hwm_mb() -> float | None:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
    except (OSError, ValueError):
        return None
    # Linux reports KiB, macOS reports bytes.
    scale = 1024 * 1024 if sys.platform == "darwin" else 1024
    return float(usage.ru_maxrss) / scale


@dataclass
class SpliceShardProfile:
    """One shard's timing, counters, sizes, memory, CPU, and terminal status."""

    shard_stem: str
    shard_index: int
    status: str = "pending"
    n_docs: int = 0
    n_switches: int = 0
    n_segments: int = 0
    n_features: int = 0
    input_feature_bytes: int = 0
    input_parquet_bytes: int = 0
    output_feature_bytes: int = 0
    output_parquet_bytes: int = 0
    output_provenance_bytes: int = 0
    phase_seconds: dict[str, float] = field(default_factory=_phase_seconds)
    detail_seconds: dict[str, float] = field(default_factory=_detail_seconds)
    rss_mb: float | None = None
    hwm_mb: float | None = None
    cpu_seconds: float = 0.0
    error_type: str | None = None
    error_message: str | None = None
    _total_start: float | None = field(default=None, init=False, repr=False)
    _cpu_start: float | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_paths(
        cls,
        *,
        shard_stem: str,
        shard_index: int,
        base_shard_path: Path,
        parquet_path: Path,
    ) -> SpliceShardProfile:
        return cls(
            shard_stem=shard_stem,
            shard_index=shard_index,
            input_feature_bytes=_file_size(base_shard_path),
            input_parquet_bytes=_file_size(parquet_path),
        )

    @contextmanager
    def time_phase(self, key: str) -> Iterator[None]:
        if key not in PHASE_KEYS:
            raise KeyError(f"unknown splice profile phase {key!r}")
        start = time.perf_counter()
        try:
            yield
        finally:
            self.phase_seconds[key] += time.perf_counter() - start

    @contextmanager
    def time_detail(self, key: str) -> Iterator[None]:
        if key not in DETAIL_KEYS:
            raise KeyError(f"unknown splice profile detail {key!r}")
        start = time.perf_counter()
        try:
            yield
        finally:
            self.detail_seconds[key] += time.perf_counter() - start

    def start_total(self, *, track_process_cpu: bool = True) -> bool:
        """Start the total timer if it is not already owned by an outer caller."""
        if self._total_start is not None:
            return False
        self._total_start = time.perf_counter()
        self._cpu_start = time.process_time() if track_process_cpu else None
        return True

    def add_cpu_seconds(self, seconds: float) -> None:
        self.cpu_seconds += seconds

    def finish_total(self) -> None:
        if self._total_start is not None:
            self.phase_seconds["total"] += time.perf_counter() - self._total_start
            self._total_start = None
        if self._cpu_start is not None:
            self.cpu_seconds += time.process_time() - self._cpu_start
            self._cpu_start = None
        self.rss_mb = _current_rss_mb()
        self.hwm_mb = _hwm_mb()

    def record_derived(self, derived: dict, n_switches: int) -> None:
        self.n_docs = int(derived["n_docs"])
        self.n_segments = int(derived["n_segments"])
        self.n_features = int(derived["feature_ids"].numel())
        self.n_switches = int(n_switches)

    def record_outputs(
        self,
        *,
        feature_path: Path | None,
        parquet_path: Path | None,
        provenance_path: Path | None,
    ) -> None:
        if feature_path is not None:
            self.output_feature_bytes = _file_size(feature_path)
        if parquet_path is not None:
            self.output_parquet_bytes = _file_size(parquet_path)
        if provenance_path is not None:
            self.output_provenance_bytes = _file_size(provenance_path)

    def finish(
        self,
        *,
        status: str,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        self.status = status
        self.error_type = error_type
        self.error_message = error_message
        self.finish_total()

    def to_dict(self) -> dict:
        return {
            "shard_stem": self.shard_stem,
            "shard_index": self.shard_index,
            "status": self.status,
            "n_docs": self.n_docs,
            "n_switches": self.n_switches,
            "n_segments": self.n_segments,
            "n_features": self.n_features,
            "input_feature_bytes": self.input_feature_bytes,
            "input_parquet_bytes": self.input_parquet_bytes,
            "output_feature_bytes": self.output_feature_bytes,
            "output_parquet_bytes": self.output_parquet_bytes,
            "output_provenance_bytes": self.output_provenance_bytes,
            "phase_seconds": {key: self.phase_seconds.get(key, 0.0) for key in PHASE_KEYS},
            "detail_seconds": {
                key: self.detail_seconds.get(key, 0.0) for key in DETAIL_KEYS
            },
            "rss_mb": self.rss_mb,
            "hwm_mb": self.hwm_mb,
            "cpu_seconds": self.cpu_seconds,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass
class SpliceRunProfile:
    """Whole-run splice profile with incremental atomic JSON writes."""

    config: dict
    rank: int
    world_size: int
    shard_parallelism: int
    n_skipped: int = 0
    runtime_totals: dict = field(default_factory=dict)
    per_shard: list[dict] = field(default_factory=list)
    totals: dict = field(default_factory=dict)
    _wall_start: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.config = dict(self.config)
        self.config.update(
            {
                "rank": self.rank,
                "world_size": self.world_size,
                "shard_parallelism": self.shard_parallelism,
            }
        )
        self._recompute_totals()

    def start(self) -> None:
        self._wall_start = time.perf_counter()

    def update_wall(self) -> None:
        if self._wall_start is not None:
            self.totals["wall_seconds"] = time.perf_counter() - self._wall_start

    def add_shard(self, record: SpliceShardProfile | dict) -> None:
        if isinstance(record, SpliceShardProfile):
            record = record.to_dict()
        self.per_shard.append(record)
        self._recompute_totals()

    def finish(self, wall_seconds: float | None = None) -> None:
        if wall_seconds is not None:
            self.totals["wall_seconds"] = wall_seconds
            self._wall_start = None
        else:
            self.update_wall()
        self._recompute_totals(preserve_wall=True)

    def _recompute_totals(self, *, preserve_wall: bool = False) -> None:
        wall_seconds = self.totals.get("wall_seconds", 0.0) if preserve_wall else 0.0
        n_ok = sum(1 for r in self.per_shard if r.get("status") == "ok")
        n_failed = sum(1 for r in self.per_shard if r.get("status") == "failed")
        phase_totals = {
            key: sum(float(r.get("phase_seconds", {}).get(key, 0.0)) for r in self.per_shard)
            for key in PHASE_KEYS
        }
        detail_totals = {
            key: sum(float(r.get("detail_seconds", {}).get(key, 0.0)) for r in self.per_shard)
            for key in DETAIL_KEYS
        }
        hwm_values = [
            float(r["hwm_mb"])
            for r in self.per_shard
            if r.get("hwm_mb") is not None
        ]
        self.totals = {
            "wall_seconds": wall_seconds,
            "n_ok": n_ok,
            "n_failed": n_failed,
            "n_skipped": self.n_skipped,
            "n_docs": sum(int(r.get("n_docs", 0)) for r in self.per_shard),
            "n_switches": sum(int(r.get("n_switches", 0)) for r in self.per_shard),
            "n_segments": sum(int(r.get("n_segments", 0)) for r in self.per_shard),
            "n_features": sum(int(r.get("n_features", 0)) for r in self.per_shard),
            "input_bytes": sum(
                int(r.get("input_feature_bytes", 0)) + int(r.get("input_parquet_bytes", 0))
                for r in self.per_shard
            ),
            "output_bytes": sum(
                int(r.get("output_feature_bytes", 0))
                + int(r.get("output_parquet_bytes", 0))
                + int(r.get("output_provenance_bytes", 0))
                for r in self.per_shard
            ),
            "cpu_seconds": sum(float(r.get("cpu_seconds", 0.0)) for r in self.per_shard),
            "max_hwm_mb": max(hwm_values) if hwm_values else None,
            "phase_seconds": phase_totals,
            "detail_seconds": detail_totals,
        }
        self.totals.update(self.runtime_totals)

    def update_runtime_totals(self, **values: object) -> None:
        """Attach mode-specific totals without extending the fixed phase schema."""
        self.runtime_totals.update(values)
        self._recompute_totals(preserve_wall=True)

    def to_dict(self) -> dict:
        self.update_wall()
        return {
            "config": self.config,
            "totals": self.totals,
            "per_shard": sorted(self.per_shard, key=lambda r: int(r["shard_index"])),
        }

    def write(self, path: Path) -> None:
        atomic_json_dump(self.to_dict(), path)
