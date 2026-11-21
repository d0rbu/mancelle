"""Lightweight timing markers for profiling runs.

The public API is intentionally small:

* ``TimingRecorder`` collects append-only timing rows.
* ``use_recorder`` activates a recorder in the current context.
* ``timing_tags`` adds inherited row metadata for nested regions.
* ``timed_region`` emits a ``torch.profiler.record_function`` marker and,
  when a recorder is active, appends a wall-clock timing row.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import torch
from torch.profiler import record_function


class TimingRecorder:
    """Append-only ``(name, elapsed_ns, **extra)`` row collector."""

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def append(
        self,
        name: object,
        elapsed_ns: int | float,
        extra: Mapping[Any, Any] | None = None,
    ) -> None:
        row: dict[str, Any] = {"name": str(name), "elapsed_ns": int(elapsed_ns)}
        if extra:
            for key, value in extra.items():
                row[str(key)] = value
        self._rows.append(row)

    def __len__(self) -> int:
        """Return the number of rows recorded so far."""
        return len(self._rows)

    def tag_rows_since(
        self,
        start_index: int,
        tags: Mapping[str, object],
    ) -> int:
        """Set default tags on rows appended since ``start_index``.

        Returns the new append position so callers can use it as the next
        ``start_index``. Existing row values win over the supplied defaults.
        """
        if not 0 <= start_index <= len(self._rows):
            raise IndexError(
                f"start_index must be in [0, {len(self._rows)}], got {start_index}"
            )
        for row in self._rows[start_index:]:
            for key, value in tags.items():
                row.setdefault(key, value)
        return len(self._rows)

    def rows(self) -> list[dict[str, Any]]:
        """Return a shallow independent copy of accumulated rows."""
        return [dict(row) for row in self._rows]

    def to_csv(self, path: Path) -> None:
        """Write accumulated rows to ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._rows:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["name", "elapsed_ns"])
            return

        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in self._rows:
            for key in row:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(self._rows)


_ACTIVE_RECORDER: ContextVar[TimingRecorder | None] = ContextVar(
    "_ACTIVE_RECORDER",
    default=None,
)
_ACTIVE_TIMING_TAGS: ContextVar[dict[str, Any] | None] = ContextVar(
    "_ACTIVE_TIMING_TAGS",
    default=None,
)
_FORCE_RECORD_FUNCTION_MARKERS: ContextVar[bool] = ContextVar(
    "_FORCE_RECORD_FUNCTION_MARKERS",
    default=False,
)


def _current_tags_raw() -> dict[str, Any]:
    return _ACTIVE_TIMING_TAGS.get() or {}


@contextmanager
def use_recorder(recorder: TimingRecorder) -> Iterator[TimingRecorder]:
    """Set ``recorder`` as the active timing sink for this scope."""
    token = _ACTIVE_RECORDER.set(recorder)
    try:
        yield recorder
    finally:
        _ACTIVE_RECORDER.reset(token)


@contextmanager
def use_profiler_markers() -> Iterator[None]:
    """Emit markers for a known-active profiler whose private flag is false.

    PyTorch's ``profile_all_threads=True`` profiler does not set
    ``torch.autograd._profiler_enabled()`` on the entering thread. Callers that
    own that profiler may activate this scope so ``timed_region`` emits
    ``record_function`` labels without constructing a timing recorder or
    synchronizing CUDA.
    """

    token = _FORCE_RECORD_FUNCTION_MARKERS.set(True)
    try:
        yield
    finally:
        _FORCE_RECORD_FUNCTION_MARKERS.reset(token)


def recorder_active() -> bool:
    """True when a ``TimingRecorder`` is active OR any torch profiler is attached.

    When ``False``, :func:`timed_region` skips ``record_function`` and all timing —
    the production default (no recorder, no profiler) pays nothing. ``record_function``
    is still emitted under a profiler so trace markers are preserved (``torch``'s
    ``_profiler_enabled`` is ``True`` under both ``torch.profiler.profile()`` /
    kineto and the legacy profiler, and ``False`` when nothing is attached).
    """
    return (
        _ACTIVE_RECORDER.get() is not None
        or _FORCE_RECORD_FUNCTION_MARKERS.get()
        or torch.autograd._profiler_enabled()
    )


@contextmanager
def timing_tags(**tags: Any) -> Iterator[dict[str, Any]]:
    """Apply inherited metadata tags to nested ``timed_region`` rows.

    Nested calls merge with outer tags. Inner tags override outer tags on key
    collisions, and explicit ``timed_region(extra=...)`` values override both.
    """
    merged = dict(_current_tags_raw())
    merged.update(tags)
    token = _ACTIVE_TIMING_TAGS.set(merged)
    try:
        yield dict(merged)
    finally:
        _ACTIVE_TIMING_TAGS.reset(token)


def current_timing_tags() -> dict[str, Any]:
    """Return a copy of the currently active timing tags."""
    return dict(_current_tags_raw())


@contextmanager
def timed_region(
    name: str,
    *,
    sync_cuda: bool = True,
    extra: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    """Record wall time for a named region when something is listening.

    FREE WHEN OFF: when neither a recorder nor a torch profiler is active (the
    production default), this yields immediately — NO ``record_function`` marker,
    NO timing, NO ``cuda.synchronize`` — so hot call sites pay
    nothing. The ``torch.profiler.record_function`` marker is still emitted under
    a profiler so the same region names appear in profiler traces even when CSV
    timing is disabled, and a wall-clock row is appended only when a recorder is
    active.
    """
    active = _ACTIVE_RECORDER.get()
    profiler_is_listening = (
        _FORCE_RECORD_FUNCTION_MARKERS.get()
        or torch.autograd._profiler_enabled()
    )
    if active is None and not profiler_is_listening:
        # Nothing is listening — skip record_function and all timing entirely.
        yield
        return

    with record_function(name):
        if active is None:
            # A profiler is attached but no recorder: emit the marker, no timing.
            yield
            return

        merged_extra = dict(_current_tags_raw())
        if extra is not None:
            merged_extra.update(dict(extra))

        cuda_is_available = torch.cuda.is_available()
        if sync_cuda and cuda_is_available:
            torch.cuda.synchronize()
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            if sync_cuda and cuda_is_available:
                torch.cuda.synchronize()
            active.append(name, time.perf_counter_ns() - start, merged_extra)
