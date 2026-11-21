"""Depth-bounded FIFO background executor — the single primitive behind both
StagedSaver (fire-and-forget) and StagedPrefetcher (submit/get pipeline).

Generic over the job callable. submit() runs fn on a bounded ThreadPoolExecutor;
when `depth` are in flight, submit() drains the OLDEST future first (FIFO — surfaces
exceptions in submission order, bounds resident memory). get() pops and returns the
oldest pending result (consumer-driven pipelines). drain() waits for all pending
without closing the pool; __exit__ drains all pending FIFO then shuts the pool down.
Exceptions reach the next drainer / get() / drain() / __exit__.
"""
from __future__ import annotations

import collections
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from types import TracebackType
from typing import Any, Self

from beartype import beartype
from loguru import logger

from steering.refined_types import PositiveInt


class BoundedFifoExecutor:
    @beartype
    def __init__(
        self,
        *,
        depth: PositiveInt,
        thread_name_prefix: str = "staging",
    ) -> None:
        self._depth = depth
        self._executor = ThreadPoolExecutor(max_workers=depth, thread_name_prefix=thread_name_prefix)
        self._pending: collections.deque[Future] = collections.deque()
        self._closed = False
        self._lock = threading.Lock()

    @property
    def in_flight(self) -> int:
        with self._lock:
            return len(self._pending)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        drain_exc: BaseException | None = None
        with self._lock:
            self._closed = True
            pending = list(self._pending)
            self._pending.clear()
        for fut in pending:
            try:
                fut.result()
            except BaseException as e:  # noqa: BLE001
                if drain_exc is None:
                    drain_exc = e
                else:
                    logger.warning("BoundedFifoExecutor: additional drain exception suppressed: {!r}", e)
        self._executor.shutdown(wait=True)
        if exc_type is None and drain_exc is not None:
            raise drain_exc
        if drain_exc is not None and exc_type is not None:
            logger.warning("BoundedFifoExecutor: drain exception suppressed by enclosing exception: {!r}", drain_exc)
        return False

    def drain(self) -> None:
        """Block until all in-flight jobs finish (FIFO), surfacing the first exception.

        Unlike __exit__, this does NOT close the pool — submit() still works after.
        Used to flush the pipeline inside a measured region so the GPU-idle save tail
        is counted there rather than after the profiler stops.
        """
        with self._lock:
            pending = list(self._pending)
            self._pending.clear()
        drain_exc: BaseException | None = None
        for fut in pending:
            try:
                fut.result()
            except BaseException as e:  # noqa: BLE001
                if drain_exc is None:
                    drain_exc = e
                else:
                    logger.warning("BoundedFifoExecutor: additional drain exception suppressed: {!r}", e)
        if drain_exc is not None:
            raise drain_exc

    def get(self) -> Any:
        """Pop and return the OLDEST pending result (FIFO), surfacing the job's
        exception. For consumer-driven pipelines (StagedPrefetcher) that pull each
        result in submission order. Raises if nothing is pending.
        """
        with self._lock:
            if not self._pending:
                raise RuntimeError("BoundedFifoExecutor: get() with nothing pending")
            fut = self._pending.popleft()
        return fut.result()

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future:
        with self._lock:
            if self._closed:
                raise RuntimeError("BoundedFifoExecutor: submit() after __exit__")
            while len(self._pending) >= self._depth:
                oldest = self._pending.popleft()
                self._lock.release()
                try:
                    oldest.result()
                finally:
                    self._lock.acquire()
            if self._closed:
                raise RuntimeError("BoundedFifoExecutor: closed during submit() wait")
            future = self._executor.submit(fn, *args, **kwargs)
            self._pending.append(future)
            return future
