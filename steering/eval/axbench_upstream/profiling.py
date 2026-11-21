"""Opt-in exact-cell torch-profiler traces for AxBench-Upstream."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch as t
from torch.profiler import ProfilerActivity, profile, record_function

from steering.eval.axbench_upstream.work_grid import CellKey

_PHASE_EVENT_PREFIXES = (
    "axbench_upstream.",
    "encdec_",
    "generation/",
    "selfcond/",
)


@dataclass(frozen=True)
class CellTraceSelector:
    """A placement-free selector for one unique logical generation cell."""

    driver: str
    method: str
    concept_id: int
    input_id: int
    factor_hex: str

    def matches(self, cell: CellKey) -> bool:
        return (
            cell.driver == self.driver
            and cell.method == self.method
            and cell.concept_id == self.concept_id
            and cell.input_id == self.input_id
            and cell.factor_hex == self.factor_hex
        )

    def to_dict(self) -> dict[str, int | str]:
        return {
            "driver": self.driver,
            "method": self.method,
            "concept_id": self.concept_id,
            "input_id": self.input_id,
            "factor_hex": self.factor_hex,
        }


@dataclass
class CellTraceController:
    """Capture CPU/CUDA stacks for each requested cell at most once."""

    output_dir: Path
    selectors: Sequence[CellTraceSelector]
    row_limit: int = 50
    _captured: set[CellTraceSelector] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        self.selectors = tuple(self.selectors)
        if not self.selectors:
            raise ValueError("cell trace controller requires at least one selector")
        if len(set(self.selectors)) != len(self.selectors):
            raise ValueError("cell trace selectors must be unique")
        if type(self.row_limit) is not int or self.row_limit <= 0:
            raise ValueError("cell trace row_limit must be a positive integer")

    def _selector_for(self, cell: CellKey) -> CellTraceSelector | None:
        matches = [selector for selector in self.selectors if selector.matches(cell)]
        if len(matches) > 1:
            raise RuntimeError(f"multiple torch-profiler selectors match one cell: {cell}")
        if not matches or matches[0] in self._captured:
            return None
        return matches[0]

    def will_trace(self, cell: CellKey) -> bool:
        """Return whether the next execution of ``cell`` needs its own trace."""
        return self._selector_for(cell) is not None

    @staticmethod
    def _event_value(event: Any, name: str) -> int | float | str | list[int] | None:
        value = getattr(event, name, None)
        if isinstance(value, (int, float, str)) or value is None:
            return value
        if isinstance(value, tuple):
            return [int(item) for item in value]
        return str(value)

    @classmethod
    def _event_row(cls, event: Any) -> dict[str, Any]:
        return {
            "key": str(event.key),
            "count": int(event.count),
            "input_shapes": event.input_shapes,
            "self_cpu_time_total_us": cls._event_value(event, "self_cpu_time_total"),
            "cpu_time_total_us": cls._event_value(event, "cpu_time_total"),
            "self_cuda_time_total_us": cls._event_value(event, "self_cuda_time_total"),
            "cuda_time_total_us": cls._event_value(event, "cuda_time_total"),
            "cpu_memory_usage_bytes": cls._event_value(event, "cpu_memory_usage"),
            "cuda_memory_usage_bytes": cls._event_value(event, "cuda_memory_usage"),
        }

    def _write_trace(self, profiler: Any, selector: CellTraceSelector, cell: CellKey) -> None:
        stem = (
            f"{selector.driver}-{selector.method}-c{selector.concept_id:05d}-"
            f"i{selector.input_id:05d}-{selector.factor_hex.replace('.', '_')}"
        )
        trace_dir = self.output_dir / stem
        trace_dir.mkdir(parents=True, exist_ok=False)
        profiler.export_chrome_trace(str(trace_dir / "trace.json"))
        for metric in ("self_cuda_time_total", "self_cpu_time_total"):
            try:
                profiler.export_stacks(str(trace_dir / f"stacks-{metric}.txt"), metric)
            except (AssertionError, RuntimeError):
                # Some CPU-only builds do not expose a CUDA stack metric. The
                # trace and key averages remain authoritative in that case.
                if metric != "self_cuda_time_total":
                    raise
        events = profiler.key_averages(group_by_input_shape=True)
        ranked = sorted(
            events,
            key=lambda event: (
                float(getattr(event, "self_cuda_time_total", 0.0) or 0.0),
                float(getattr(event, "self_cpu_time_total", 0.0) or 0.0),
            ),
            reverse=True,
        )[: self.row_limit]
        rows = [self._event_row(event) for event in ranked]
        phase_rows = [
            self._event_row(event)
            for event in sorted(events, key=lambda item: str(item.key))
            if str(event.key).startswith(_PHASE_EVENT_PREFIXES)
        ]
        payload = {
            "schema_version": 2,
            "selector": selector.to_dict(),
            "cell": cell.to_dict(),
            "activities": [activity.name for activity in _profiler_activities()],
            "phase_events": phase_rows,
            "top_events": rows,
        }
        (trace_dir / "key_averages.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @contextmanager
    def trace(self, cell: CellKey) -> Iterator[None]:
        selector = self._selector_for(cell)
        if selector is None:
            yield
            return
        with profile(
            activities=_profiler_activities(),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        ) as profiler:
            with record_function("axbench_upstream.selected_cell"):
                yield
        self._write_trace(profiler, selector, cell)
        self._captured.add(selector)

    def require_complete(self) -> None:
        missing = [selector for selector in self.selectors if selector not in self._captured]
        if missing:
            raise RuntimeError(f"requested torch-profiler cells were not observed: {missing}")


def _profiler_activities() -> list[ProfilerActivity]:
    activities = [ProfilerActivity.CPU]
    if t.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    return activities


_CURRENT_CELL_TRACE: ContextVar[CellTraceController | None] = ContextVar(
    "axbench_upstream_cell_trace",
    default=None,
)


@contextmanager
def activate_cell_tracing(controller: CellTraceController) -> Iterator[None]:
    """Bind one trace controller around an otherwise-normal producer run."""

    token = _CURRENT_CELL_TRACE.set(controller)
    try:
        yield
    finally:
        _CURRENT_CELL_TRACE.reset(token)


@contextmanager
def trace_cell(cell: CellKey) -> Iterator[None]:
    """Trace one cell only when an active controller selected it."""

    controller = _CURRENT_CELL_TRACE.get()
    if controller is None:
        yield
        return
    with controller.trace(cell):
        yield


def cell_trace_requested(cell: CellKey) -> bool:
    """Return whether batching this cell would hide a requested exact-cell trace."""
    controller = _CURRENT_CELL_TRACE.get()
    return controller is not None and controller.will_trace(cell)


__all__ = (
    "CellTraceController",
    "CellTraceSelector",
    "activate_cell_tracing",
    "cell_trace_requested",
    "trace_cell",
)
