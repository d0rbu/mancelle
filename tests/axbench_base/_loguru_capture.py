"""Shared loguru capture helper for axbench_base tests.

Loguru does not propagate to pytest's caplog. Use this helper to capture
INFO-level (or higher) log messages emitted during a block. Mirrors the
existing `_capture_loguru_warnings()` pattern in test_runtime_state.py.
"""

from __future__ import annotations

import io
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from loguru import logger as loguru_logger


class _ThreadSafeStringIO(io.StringIO):
    """StringIO subclass that serialises writes through an explicit lock.

    Loguru already serialises calls to a single handler internally so
    in practice the underlying StringIO never sees a true concurrent
    write today, but wrapping the buffer makes the thread-safety
    contract visible at the seam: a future helper that adds threads on
    the test side (e.g. multi-thread emission tests) inherits the lock
    without needing to add one at every call-site.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
        with self._lock:
            return super().write(s)

    def getvalue(self) -> str:
        with self._lock:
            return super().getvalue()


@contextmanager
def capture_loguru_info() -> Iterator[list[str]]:
    """Capture loguru INFO+ messages.

    Returns a list that is filled when the context exits. The captured
    messages are the lines of the logged messages (no metadata).
    """
    sink = _ThreadSafeStringIO()
    handler_id = loguru_logger.add(sink, level="INFO", format="{message}")
    captured: list[str] = []
    try:
        yield captured
    finally:
        loguru_logger.remove(handler_id)
        captured.extend(sink.getvalue().splitlines())


__all__ = ["capture_loguru_info"]
