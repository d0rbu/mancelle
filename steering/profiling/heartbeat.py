"""Periodic liveness lines for long silent phases.

During the 2026-08-28 hstor013 storage incidents, startup phases that are
normally seconds (selection-body scans, checkpoint reads) ran for 30-60
minutes with no log output, so distinguishing "slowly progressing" from
"hung" required attaching to compute nodes and reading ``/proc``.  A
``Heartbeat`` wraps such a phase and, from a daemon thread, logs one line per
interval while the phase is still running — including the latest progress
detail the phase reported via :meth:`tick`.  A phase that finishes inside the
first interval logs nothing.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from loguru import logger


class Heartbeat:
    """Context manager emitting ``<label>: still running after Ns`` lines."""

    def __init__(
        self,
        label: str,
        *,
        interval_s: float = 60.0,
        emit: Callable[[str], object] = logger.info,
    ) -> None:
        if interval_s <= 0:
            raise ValueError(f"interval_s must be positive, got {interval_s}")
        self._label = label
        self._interval_s = interval_s
        self._emit = emit
        self._done = threading.Event()
        self._detail_lock = threading.Lock()
        self._detail: str | None = None
        self._thread: threading.Thread | None = None
        self._started_at = 0.0

    def tick(self, detail: str) -> None:
        """Record the latest progress detail (shown in subsequent beats)."""
        with self._detail_lock:
            self._detail = detail

    def _run(self) -> None:
        while not self._done.wait(self._interval_s):
            elapsed = time.monotonic() - self._started_at
            with self._detail_lock:
                detail = self._detail
            suffix = f" ({detail})" if detail is not None else ""
            self._emit(f"{self._label}: still running after {elapsed:.0f}s{suffix}")

    def __enter__(self) -> Heartbeat:
        self._started_at = time.monotonic()
        self._thread = threading.Thread(
            target=self._run,
            name=f"heartbeat:{self._label}"[:60],
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._done.set()
        assert self._thread is not None
        self._thread.join(timeout=5.0)
