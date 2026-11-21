"""Lightweight AxBench-Base profiling instrumentation.

The profiler is intentionally opt-in.  Production stage code can call
``profile_region(...)`` and ``record_counter(...)`` freely; when no
``AxBenchProfiler`` is active those calls are no-ops.  The profiling CLI
activates a profiler around the normal AxBench dispatcher and writes:

* ``profile/events.jsonl`` -- one JSON object per region / counter event
* ``profile/summary.yaml`` -- aggregate region and counter statistics
* ``profile/summary.csv`` -- flat aggregate table for quick spreadsheet use
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Any

import torch as t
import yaml
from beartype import beartype

from steering.profiling.gpu_utilization import GPUUtilizationMonitor
from steering.refined_types import (
    NonBlankStrLike,
    NonNegativeFiniteNumberLike,
    PositiveIntLike,
)

_CURRENT_PROFILER: ContextVar[AxBenchProfiler | None] = ContextVar(
    "axbench_base_profiler",
    default=None,
)

try:
    _resource = importlib.import_module("resource")
except ImportError:
    _resource = None


def _jsonable(value: Any) -> Any:
    """Convert common Python / torch objects to JSON-serialisable values.

    :class:`enum.Enum` members (including :class:`enum.StrEnum`) are
    unwrapped to their ``value`` attribute BEFORE the ``str``/``int``
    fast path so that PyYAML's default representer (which only knows
    about plain Python primitives) can round-trip them. Without this
    pre-unwrap, a :class:`StrEnum` member would short-circuit on
    ``isinstance(value, str)`` and reach :func:`yaml.safe_dump`
    untouched, raising ``RepresenterError``.
    """
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "value"):
        return _jsonable(value.value)
    if isinstance(value, t.Tensor):
        if value.numel() == 1:
            return _jsonable(value.item())
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    return repr(value)


def _cuda_snapshot() -> dict[str, int] | None:
    """Return CUDA memory counters for the current device, if available."""
    if not t.cuda.is_available():
        return None
    try:
        return {
            "memory_allocated_bytes": int(t.cuda.memory_allocated()),
            "memory_reserved_bytes": int(t.cuda.memory_reserved()),
            "max_memory_allocated_bytes": int(t.cuda.max_memory_allocated()),
            "max_memory_reserved_bytes": int(t.cuda.max_memory_reserved()),
        }
    except Exception:  # noqa: BLE001 - profiling must never break the run
        return None


def _cpu_rss_kb() -> int | None:
    """Return process peak RSS in KB, if the platform exposes it."""
    if _resource is None:
        return None
    try:
        ru = _resource.getrusage(_resource.RUSAGE_SELF)
        rss_kb = int(ru.ru_maxrss)
        if sys.platform == "darwin":
            rss_kb //= 1024
        return rss_kb
    except (OSError, AttributeError):
        return None


@dataclass(frozen=True)
class ProcessMemorySnapshot:
    """Best-effort current and peak process-memory counters in KiB.

    Linux exposes the complete Stage 0c diagnostic surface directly through
    ``/proc/self/status``. Other platforms, restricted procfs mounts, and
    malformed procfs data may return only the fields available from psutil and
    ``resource.getrusage``. Missing fields stay ``None`` rather than being
    guessed.
    """

    rss_kb: int | None
    rss_anon_kb: int | None
    swap_kb: int | None
    anonymous_total_kb: int | None
    peak_rss_kb: int | None


_LINUX_STATUS_MEMORY_FIELDS = ("VmRSS", "RssAnon", "VmSwap", "VmHWM")


def _parse_linux_process_status(text: str) -> ProcessMemorySnapshot | None:
    """Parse the required non-negative ``kB`` fields from proc status text."""
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, separator, raw_value = line.partition(":")
        if not separator or key not in _LINUX_STATUS_MEMORY_FIELDS:
            continue
        parts = raw_value.split()
        if len(parts) != 2 or not parts[0].isdigit() or parts[1] != "kB" or key in values:
            return None
        value = int(parts[0])
        values[key] = value
    if set(values) != set(_LINUX_STATUS_MEMORY_FIELDS):
        return None
    return ProcessMemorySnapshot(
        rss_kb=values["VmRSS"],
        rss_anon_kb=values["RssAnon"],
        swap_kb=values["VmSwap"],
        anonymous_total_kb=values["RssAnon"] + values["VmSwap"],
        peak_rss_kb=values["VmHWM"],
    )


def _read_linux_process_status() -> str:
    """Read the inexpensive Linux process-status summary."""
    return Path("/proc/self/status").read_text(encoding="utf-8")


def _fallback_process_memory_snapshot() -> ProcessMemorySnapshot | None:
    """Return the subset available from psutil/resource without throwing."""
    rss_kb: int | None = None
    swap_kb: int | None = None
    try:
        psutil = importlib.import_module("psutil")
        process = psutil.Process()
        memory_info = process.memory_info()
        rss_bytes = int(memory_info.rss)
        if rss_bytes >= 0:
            rss_kb = rss_bytes // 1024
        try:
            memory_full_info = process.memory_full_info()
            swap_bytes = getattr(memory_full_info, "swap", None)
            if swap_bytes is not None and int(swap_bytes) >= 0:
                swap_kb = int(swap_bytes) // 1024
        except Exception:  # noqa: BLE001 - optional fallback detail
            swap_kb = None
    except Exception:  # noqa: BLE001 - profiling must never break the run
        rss_kb = None
        swap_kb = None

    peak_rss_kb = _cpu_rss_kb()
    if rss_kb is None and swap_kb is None and peak_rss_kb is None:
        return None
    return ProcessMemorySnapshot(
        rss_kb=rss_kb,
        rss_anon_kb=None,
        swap_kb=swap_kb,
        anonymous_total_kb=None,
        peak_rss_kb=peak_rss_kb,
    )


def process_memory_snapshot() -> ProcessMemorySnapshot | None:
    """Return a no-throw process-memory snapshot.

    Linux procfs is authoritative for current RSS, anonymous RSS, swap, and
    high-water RSS. If it is unavailable or malformed, gracefully fall back to
    psutil/resource and report only counters those facilities expose.
    """
    try:
        if sys.platform.startswith("linux"):
            try:
                parsed = _parse_linux_process_status(_read_linux_process_status())
            except Exception:  # noqa: BLE001 - procfs is observational only
                parsed = None
            if parsed is not None:
                return parsed
        return _fallback_process_memory_snapshot()
    except Exception:  # noqa: BLE001 - profiling must never break the run
        return None


def _synchronize_cuda_if_available() -> None:
    if not t.cuda.is_available():
        return
    try:
        t.cuda.synchronize()
    except Exception:  # noqa: BLE001 - profiling must never break the run
        return


def _gpu_summary_from_values(
    samples: list[float],
    *,
    idle_threshold: float,
) -> dict[str, float | int]:
    if not samples:
        return {
            "gpu_util_mean": 0.0,
            "gpu_util_min": 0.0,
            "gpu_util_max": 0.0,
            "gpu_sample_count": 0,
            "gpu_idle_fraction": 0.0,
        }
    return {
        "gpu_util_mean": sum(samples) / len(samples),
        "gpu_util_min": min(samples),
        "gpu_util_max": max(samples),
        "gpu_sample_count": len(samples),
        "gpu_idle_fraction": sum(1 for value in samples if value < idle_threshold) / len(samples),
    }


def resolve_profile_gpu_device_index(
    device: str,
    *,
    ddp_local_rank: int,
    ddp_world_size: int,
) -> int:
    """Resolve the CUDA device whose utilization the profiler should sample.

    In torchrun/DDP, ``compute_init`` maps each process to ``ddp_local_rank``.
    In single-process runs, operators may pin the eval to an explicit device
    such as ``device=cuda:1``; there is no local rank in that case, so the
    profiler must honor the explicit CUDA index instead of sampling GPU 0.
    """
    if int(ddp_world_size) > 1:
        return int(ddp_local_rank)
    try:
        parsed = t.device(str(device).strip().lower())
    except (TypeError, RuntimeError):
        return int(ddp_local_rank)
    if parsed.type == "cuda" and parsed.index is not None:
        return int(parsed.index)
    return int(ddp_local_rank)


@beartype
@dataclass(frozen=True)
class ProfileConfig:
    """Shared opt-in profiling config for AxBench eval pipelines."""

    enabled: bool = False
    output_dir: NonBlankStrLike | None = None
    synchronize_cuda: bool = True
    gpu_utilization: bool = True
    gpu_poll_interval_ms: PositiveIntLike = 50
    gpu_idle_threshold: NonNegativeFiniteNumberLike = 5.0


@dataclass
class _RegionAggregate:
    count: int = 0
    total_s: float = 0.0
    min_s: float | None = None
    max_s: float = 0.0
    rows: float = 0.0
    peak_cuda_allocated_bytes: int = 0
    peak_cuda_reserved_bytes: int = 0
    peak_cpu_rss_kb: int = 0
    gpu_util_sample_count: int = 0
    gpu_util_total: float = 0.0
    gpu_util_min: float | None = None
    gpu_util_max: float = 0.0
    gpu_idle_sample_count: int = 0
    first_duration_seconds: float | None = None

    def add(self, event: dict[str, Any]) -> None:
        duration_s = float(event.get("duration_s", 0.0))
        errored = bool(event.get("errored", False))
        self.count += 1
        self.total_s += duration_s
        self.min_s = duration_s if self.min_s is None else min(self.min_s, duration_s)
        self.max_s = max(self.max_s, duration_s)
        # Errored regions surface a partial timing that does not represent
        # the cost of "running this region successfully" — they would
        # mislead readers of the first-verified-candidate breakdown into
        # thinking, e.g., "checkpoint validation took 5ms" when in reality
        # it never finished. Aggregate count / total_s / min_s / max_s
        # still update so JSONL consumers see the partial timing.
        if not errored and self.first_duration_seconds is None:
            self.first_duration_seconds = duration_s
        attrs = event.get("attrs") or {}
        for key in ("rows", "row_count", "n_rows"):
            value = attrs.get(key)
            if isinstance(value, (int, float)):
                self.rows += float(value)
        cuda_end = event.get("cuda_end") or {}
        self.peak_cuda_allocated_bytes = max(
            self.peak_cuda_allocated_bytes,
            int(cuda_end.get("max_memory_allocated_bytes", 0) or 0),
        )
        self.peak_cuda_reserved_bytes = max(
            self.peak_cuda_reserved_bytes,
            int(cuda_end.get("max_memory_reserved_bytes", 0) or 0),
        )
        cpu_end_rss_kb = event.get("cpu_end_rss_kb")
        if isinstance(cpu_end_rss_kb, int):
            self.peak_cpu_rss_kb = max(self.peak_cpu_rss_kb, cpu_end_rss_kb)
        gpu = event.get("gpu_utilization") or {}
        sample_count = int(gpu.get("gpu_sample_count", 0) or 0)
        if sample_count > 0:
            mean = float(gpu.get("gpu_util_mean", 0.0) or 0.0)
            self.gpu_util_sample_count += sample_count
            self.gpu_util_total += mean * sample_count
            sample_min = float(gpu.get("gpu_util_min", 0.0) or 0.0)
            self.gpu_util_min = (
                sample_min if self.gpu_util_min is None else min(self.gpu_util_min, sample_min)
            )
            self.gpu_util_max = max(
                self.gpu_util_max,
                float(gpu.get("gpu_util_max", 0.0) or 0.0),
            )
            self.gpu_idle_sample_count += round(
                float(gpu.get("gpu_idle_fraction", 0.0) or 0.0) * sample_count
            )


_BREAKDOWN_PHASE_TO_REGION: dict[str, str] = {
    "model_load": "stage0c.load_feature_extraction_pair",
    "percentile_table": "stage0c.build_percentile_table",
    "checkpoint_validate": "stage0c.try_load_partial_checkpoint",
    "first_shard_load": "stage0c.candidate_index_load_shard",
    "first_qc_call": "stage0c.examples_qc_filter_one",
    "first_verify_call": "stage0c.verify_concept",
}


def _build_first_candidate_breakdown(
    region_aggregates: dict[str, _RegionAggregate],
) -> list[dict[str, Any]]:
    """Build the 'first verified candidate timing breakdown' derived rows.

    Always emits exactly six rows in canonical order. Phases whose region
    never fired (or only fired with errors) get ``elapsed_s=None`` so
    downstream readers can rely on a fixed schema rather than guessing
    which rows are present.
    """
    rows: list[dict[str, Any]] = []
    for phase, region_name in _BREAKDOWN_PHASE_TO_REGION.items():
        agg = region_aggregates.get(region_name)
        elapsed_s = agg.first_duration_seconds if agg is not None else None
        rows.append(
            {
                "section": "first_verified_candidate_breakdown",
                "phase": phase,
                "region": region_name,
                "elapsed_s": elapsed_s,
            }
        )
    return rows


@dataclass
class _CounterAggregate:
    count: int = 0
    total: float = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value


@dataclass(frozen=True)
class _DeferredCudaRegionTiming:
    name: str
    start_event: Any
    end_event: Any
    errored: bool


@dataclass
class AxBenchProfiler:
    """Streaming JSONL profiler with aggregate summaries.

    Instances are used as context managers.  While active, calls to
    :func:`profile_region` and :func:`record_counter` emit events and update
    in-memory aggregates.  After the profiled run finishes,
    :meth:`write_reports` writes the YAML and CSV summaries.
    """

    output_dir: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    synchronize_cuda: bool = True
    gpu_utilization: bool = False
    gpu_poll_interval_ms: int = 50
    gpu_idle_threshold: float = 5.0
    gpu_device_index: int = 0
    cuda_event_regions: Sequence[str] = ()

    _events_fp: Any = field(default=None, init=False, repr=False)
    _token: Any = field(default=None, init=False, repr=False)
    _gpu_monitor: GPUUtilizationMonitor | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _started_at_wall: float = field(default=0.0, init=False)
    _started_at_perf: float = field(default=0.0, init=False)
    _region_aggregates: dict[str, _RegionAggregate] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _counter_aggregates: dict[str, _CounterAggregate] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _deferred_cuda_timings: list[_DeferredCudaRegionTiming] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _cuda_event_summary: dict[str, Any] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if isinstance(self.cuda_event_regions, (str, bytes)):
            raise ValueError("cuda_event_regions must be a sequence of exact region names")
        region_names = tuple(self.cuda_event_regions)
        for name in region_names:
            if not isinstance(name, str) or not name:
                raise ValueError("cuda_event_regions entries must be nonempty strings")
            if name != name.strip():
                raise ValueError(
                    "cuda_event_regions entries must not contain surrounding whitespace"
                )
        if len(set(region_names)) != len(region_names):
            raise ValueError("cuda_event_regions entries must be unique")
        if region_names and (type(self.gpu_device_index) is not int or self.gpu_device_index < 0):
            raise ValueError("gpu_device_index must be a nonnegative integer for CUDA event timing")
        self.cuda_event_regions = region_names

    def __enter__(self) -> AxBenchProfiler:
        if self.cuda_event_regions:
            if self.synchronize_cuda:
                raise ValueError("deferred CUDA event timing requires synchronize_cuda=False")
            if not t.cuda.is_available():
                raise RuntimeError("deferred CUDA event timing requires CUDA")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._events_fp = (self.output_dir / "events.jsonl").open(
            "w",
            encoding="utf-8",
        )
        self._started_at_wall = time.time()
        self._started_at_perf = time.perf_counter()
        if self.gpu_utilization:
            self._gpu_monitor = GPUUtilizationMonitor(
                poll_interval_ms=self.gpu_poll_interval_ms,
                device_index=self.gpu_device_index,
            )
            self._gpu_monitor.__enter__()
        self._token = _CURRENT_PROFILER.set(self)
        self._write_event(
            {
                "type": "profiler_start",
                "time": self._started_at_wall,
                "attrs": _jsonable(self.metadata),
            },
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        try:
            self.finish(status="error" if exc_type is not None else "ok")
        except BaseException as finish_error:
            if exc is None:
                raise
            exc.add_note(
                "AxBench profiler finalization also failed: "
                f"{type(finish_error).__name__}: {finish_error}"
            )

    def finish(self, *, status: str = "ok") -> None:
        """Deactivate the profiler, write summaries, and close the JSONL file."""
        if self._events_fp is None:
            return
        try:
            try:
                self._cuda_event_summary = (
                    self._finalize_deferred_cuda_timings() if self.cuda_event_regions else None
                )
                elapsed_s = time.perf_counter() - self._started_at_perf
                gpu_summary = self._gpu_summary_between(
                    self._started_at_perf,
                    time.perf_counter(),
                )
                self._write_event(
                    {
                        "type": "profiler_stop",
                        "time": time.time(),
                        "duration_s": elapsed_s,
                        "attrs": {
                            "status": status,
                            "gpu_utilization": gpu_summary,
                        },
                    },
                )
            finally:
                _CURRENT_PROFILER.reset(self._token)
                self._token = None
                if self._gpu_monitor is not None:
                    self._gpu_monitor.__exit__(None, None, None)
            self.write_reports(status=status, elapsed_s=elapsed_s)
        finally:
            self._events_fp.close()
            self._events_fp = None
            self._gpu_monitor = None

    def _begin_deferred_cuda_timing(self, name: str) -> tuple[Any, Any, Any] | None:
        if name not in self.cuda_event_regions:
            return None
        stream = t.cuda.current_stream(self.gpu_device_index)
        start_event = t.cuda.Event(enable_timing=True)
        end_event = t.cuda.Event(enable_timing=True)
        start_event.record(stream)
        return start_event, end_event, stream

    def _complete_deferred_cuda_timing(
        self,
        name: str,
        timing: tuple[Any, Any, Any] | None,
        *,
        errored: bool,
    ) -> _DeferredCudaRegionTiming | None:
        if timing is None:
            return None
        start_event, end_event, stream = timing
        end_event.record(stream)
        return _DeferredCudaRegionTiming(
            name=name,
            start_event=start_event,
            end_event=end_event,
            errored=errored,
        )

    def require_cuda_event_regions_complete(self) -> None:
        """Require one deferred event pair for every selected region event."""

        timing_counts: dict[str, int] = {}
        for timing in self._deferred_cuda_timings:
            timing_counts[timing.name] = timing_counts.get(timing.name, 0) + 1
        failures = []
        for name in self.cuda_event_regions:
            region_count = self._region_aggregates.get(name, _RegionAggregate()).count
            timing_count = timing_counts.get(name, 0)
            if region_count == 0:
                failures.append(f"{name!r} was not observed")
            elif timing_count != region_count:
                failures.append(
                    f"{name!r} has {timing_count} CUDA event pairs for {region_count} region events"
                )
        if failures:
            raise RuntimeError(
                "requested deferred CUDA event regions are incomplete: " + "; ".join(failures)
            )

    def _finalize_deferred_cuda_timings(self) -> dict[str, Any]:
        if self._deferred_cuda_timings:
            t.cuda.synchronize(self.gpu_device_index)
        durations_by_name: dict[str, list[float]] = {name: [] for name in self.cuda_event_regions}
        errored_counts = dict.fromkeys(self.cuda_event_regions, 0)
        for timing in self._deferred_cuda_timings:
            duration_s = float(timing.start_event.elapsed_time(timing.end_event)) / 1000.0
            if not math.isfinite(duration_s) or duration_s < 0.0:
                raise RuntimeError(
                    "deferred CUDA event timing produced a nonfinite or negative duration"
                )
            durations_by_name[timing.name].append(duration_s)
            errored_counts[timing.name] += int(timing.errored)
        region_rows = []
        for name in self.cuda_event_regions:
            durations = durations_by_name[name]
            if not durations:
                continue
            region_rows.append(
                {
                    "name": name,
                    "count": len(durations),
                    "errored_count": errored_counts[name],
                    "total_s": sum(durations),
                    "mean_s": sum(durations) / len(durations),
                    "min_s": min(durations),
                    "max_s": max(durations),
                }
            )
        summary = {
            "schema_version": 1,
            "device_index": self.gpu_device_index,
            "configured_regions": list(self.cuda_event_regions),
            "missing_regions": [
                name
                for name in self.cuda_event_regions
                if name not in {row["name"] for row in region_rows}
            ],
            "event_count": len(self._deferred_cuda_timings),
            "regions": region_rows,
        }
        self._write_event(
            {
                "type": "cuda_event_timing",
                "time": time.time(),
                **summary,
            }
        )
        return summary

    def _emit_region(
        self,
        name: str,
        *,
        started_at_perf: float,
        started_at_wall: float,
        duration_s: float,
        attrs: dict[str, Any],
        cuda_start: dict[str, int] | None,
        cuda_end: dict[str, int] | None,
        cpu_start_rss_kb: int | None = None,
        cpu_end_rss_kb: int | None = None,
        gpu_utilization: dict[str, float | int] | None = None,
        errored: bool = False,
        deferred_cuda_timing: _DeferredCudaRegionTiming | None = None,
    ) -> None:
        event = {
            "type": "region",
            "name": name,
            "time": started_at_wall,
            "duration_s": duration_s,
            "perf_start_s": started_at_perf,
            "attrs": _jsonable(attrs),
            "cuda_start": cuda_start,
            "cuda_end": cuda_end,
            "cpu_start_rss_kb": cpu_start_rss_kb,
            "cpu_end_rss_kb": cpu_end_rss_kb,
            "gpu_utilization": gpu_utilization,
            "errored": errored,
        }
        with self._lock:
            self._write_event_unlocked(event)
            self._region_aggregates.setdefault(name, _RegionAggregate()).add(event)
            if deferred_cuda_timing is not None:
                self._deferred_cuda_timings.append(deferred_cuda_timing)

    def record_counter(
        self,
        name: str,
        value: int | float = 1,
        **attrs: Any,
    ) -> None:
        numeric_value = float(value)
        event = {
            "type": "counter",
            "name": name,
            "time": time.time(),
            "value": numeric_value,
            "attrs": _jsonable(attrs),
        }
        with self._lock:
            self._write_event_unlocked(event)
            self._counter_aggregates.setdefault(name, _CounterAggregate()).add(
                numeric_value,
            )

    def write_reports(
        self,
        *,
        status: str = "ok",
        elapsed_s: float | None = None,
    ) -> None:
        """Write ``summary.yaml`` and ``summary.csv`` under ``output_dir``."""
        elapsed = (
            float(elapsed_s)
            if elapsed_s is not None
            else time.perf_counter() - self._started_at_perf
        )
        regions = []
        for name, agg in sorted(
            self._region_aggregates.items(),
            key=lambda item: item[1].total_s,
            reverse=True,
        ):
            regions.append(
                {
                    "name": name,
                    "count": agg.count,
                    "total_s": agg.total_s,
                    "mean_s": agg.total_s / agg.count if agg.count else 0.0,
                    "min_s": agg.min_s if agg.min_s is not None else 0.0,
                    "max_s": agg.max_s,
                    "rows": agg.rows,
                    "peak_cuda_allocated_bytes": agg.peak_cuda_allocated_bytes,
                    "peak_cuda_reserved_bytes": agg.peak_cuda_reserved_bytes,
                    "peak_cpu_rss_kb": agg.peak_cpu_rss_kb,
                    "gpu_util_mean": (
                        agg.gpu_util_total / agg.gpu_util_sample_count
                        if agg.gpu_util_sample_count
                        else 0.0
                    ),
                    "gpu_util_min": (agg.gpu_util_min if agg.gpu_util_min is not None else 0.0),
                    "gpu_util_max": agg.gpu_util_max,
                    "gpu_sample_count": agg.gpu_util_sample_count,
                    "gpu_idle_fraction": (
                        agg.gpu_idle_sample_count / agg.gpu_util_sample_count
                        if agg.gpu_util_sample_count
                        else 0.0
                    ),
                }
            )
        counters = [
            {"name": name, "count": agg.count, "total": agg.total}
            for name, agg in sorted(self._counter_aggregates.items())
        ]
        breakdown_rows = _build_first_candidate_breakdown(self._region_aggregates)
        summary = {
            "metadata": _jsonable(self.metadata),
            "status": status,
            "elapsed_s": elapsed,
            "gpu_utilization": self._gpu_summary_between(
                self._started_at_perf,
                time.perf_counter(),
            ),
            "regions": regions,
            "counters": counters,
            "first_verified_candidate_breakdown": breakdown_rows,
        }
        if self._cuda_event_summary is not None:
            summary["cuda_event_timing"] = self._cuda_event_summary
        (self.output_dir / "summary.yaml").write_text(
            yaml.safe_dump(summary, sort_keys=False),
            encoding="utf-8",
        )

        with (self.output_dir / "summary.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            fieldnames = [
                "kind",
                "name",
                "count",
                "total_s",
                "mean_s",
                "min_s",
                "max_s",
                "rows",
                "total_value",
                "peak_cuda_allocated_bytes",
                "peak_cuda_reserved_bytes",
                "peak_cpu_rss_kb",
                "gpu_util_mean",
                "gpu_util_min",
                "gpu_util_max",
                "gpu_sample_count",
                "gpu_idle_fraction",
                "section",
                "phase",
                "region",
                "elapsed_s",
            ]

            def _blank_row() -> dict[str, Any]:
                """Return a row dict with every fieldname set to an empty string.

                Used by the writer below so each row carries the full schema.
                Combined with extrasaction="raise" this catches typos in the
                writer (e.g. a row that emits a field not in fieldnames).
                """
                return dict.fromkeys(fieldnames, "")

            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            for row in regions:
                out = _blank_row()
                out.update(
                    {
                        "kind": "region",
                        "name": row["name"],
                        "count": row["count"],
                        "total_s": row["total_s"],
                        "mean_s": row["mean_s"],
                        "min_s": row["min_s"],
                        "max_s": row["max_s"],
                        "rows": row["rows"],
                        "peak_cuda_allocated_bytes": row["peak_cuda_allocated_bytes"],
                        "peak_cuda_reserved_bytes": row["peak_cuda_reserved_bytes"],
                        "peak_cpu_rss_kb": row["peak_cpu_rss_kb"],
                        "gpu_util_mean": row["gpu_util_mean"],
                        "gpu_util_min": row["gpu_util_min"],
                        "gpu_util_max": row["gpu_util_max"],
                        "gpu_sample_count": row["gpu_sample_count"],
                        "gpu_idle_fraction": row["gpu_idle_fraction"],
                    }
                )
                writer.writerow(out)
            if self._cuda_event_summary is not None:
                for row in self._cuda_event_summary["regions"]:
                    out = _blank_row()
                    out.update(
                        {
                            "kind": "cuda_event_region",
                            "name": row["name"],
                            "count": row["count"],
                            "total_s": row["total_s"],
                            "mean_s": row["mean_s"],
                            "min_s": row["min_s"],
                            "max_s": row["max_s"],
                        }
                    )
                    writer.writerow(out)
            for row in counters:
                out = _blank_row()
                out.update(
                    {
                        "kind": "counter",
                        "name": row["name"],
                        "count": row["count"],
                        "total_value": row["total"],
                    }
                )
                writer.writerow(out)
            for row in breakdown_rows:
                out = _blank_row()
                # Phases whose region never fired surface as ``elapsed_s=None``
                # in the YAML payload; the CSV equivalent is the empty
                # string so the file remains a flat numeric / string table.
                elapsed_s = row["elapsed_s"]
                out.update(
                    {
                        "section": row["section"],
                        "phase": row["phase"],
                        "region": row["region"],
                        "elapsed_s": "" if elapsed_s is None else elapsed_s,
                    }
                )
                writer.writerow(out)

    def _write_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._write_event_unlocked(event)

    def _write_event_unlocked(self, event: dict[str, Any]) -> None:
        if self._events_fp is None:
            return
        self._events_fp.write(json.dumps(_jsonable(event), sort_keys=True) + "\n")
        self._events_fp.flush()

    def _gpu_summary_between(
        self,
        started_at_perf: float,
        ended_at_perf: float,
    ) -> dict[str, float | int]:
        if self._gpu_monitor is None:
            return _gpu_summary_from_values([], idle_threshold=self.gpu_idle_threshold)
        values = [
            value
            for sample_time, value in self._gpu_monitor.timestamped_samples
            if started_at_perf <= sample_time <= ended_at_perf
        ]
        return _gpu_summary_from_values(values, idle_threshold=self.gpu_idle_threshold)


def get_current_profiler() -> AxBenchProfiler | None:
    """Return the active profiler for the current context, if any."""
    return _CURRENT_PROFILER.get()


@contextmanager
def profile_region(name: str, **attrs: Any) -> Iterator[None]:
    """Profile a region when an :class:`AxBenchProfiler` is active.

    When disabled, the function is a no-op context manager.
    """
    profiler = _CURRENT_PROFILER.get()
    if profiler is None:
        yield
        return

    deferred_cuda_timing = (
        profiler._begin_deferred_cuda_timing(name) if profiler.cuda_event_regions else None
    )
    if profiler.synchronize_cuda:
        _synchronize_cuda_if_available()
    cuda_start = _cuda_snapshot()
    cpu_start_rss_kb = _cpu_rss_kb()
    started_at_perf = time.perf_counter()
    started_at_wall = time.time()
    errored = False
    body_error: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        errored = True
        body_error = exc
        raise
    finally:
        try:
            if profiler.synchronize_cuda:
                _synchronize_cuda_if_available()
            duration_s = time.perf_counter() - started_at_perf
            ended_at_perf = time.perf_counter()
            completed_cuda_timing = profiler._complete_deferred_cuda_timing(
                name,
                deferred_cuda_timing,
                errored=errored,
            )
            cuda_end = _cuda_snapshot()
            cpu_end_rss_kb = _cpu_rss_kb()
            profiler._emit_region(
                name,
                started_at_perf=started_at_perf,
                started_at_wall=started_at_wall,
                duration_s=duration_s,
                attrs=attrs,
                cuda_start=cuda_start,
                cuda_end=cuda_end,
                cpu_start_rss_kb=cpu_start_rss_kb,
                cpu_end_rss_kb=cpu_end_rss_kb,
                gpu_utilization=profiler._gpu_summary_between(
                    started_at_perf,
                    ended_at_perf,
                ),
                errored=errored,
                deferred_cuda_timing=completed_cuda_timing,
            )
        except BaseException as finalization_error:
            if body_error is None:
                raise
            body_error.add_note(
                f"AxBench region {name!r} profiling finalization also failed: "
                f"{type(finalization_error).__name__}: {finalization_error}"
            )


def record_counter(
    name: str,
    value: int | float = 1,
    **attrs: Any,
) -> None:
    """Record a numeric counter on the active profiler, or no-op."""
    profiler = _CURRENT_PROFILER.get()
    if profiler is None:
        return
    profiler.record_counter(name, value, **attrs)
