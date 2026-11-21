"""Boolean code grading with nanochat's isolated execution worker."""

from __future__ import annotations

import multiprocessing
import time
from collections.abc import Sequence
from contextlib import ExitStack
from ctypes import c_bool
from dataclasses import dataclass
from multiprocessing.connection import wait
from multiprocessing.process import BaseProcess
from threading import current_thread, main_thread

from nanochat.execution import _unsafe_execute


@dataclass
class _SuccessFlag:
    value: c_bool

    def update(self, result: dict[str, object]) -> None:
        # Publish at the same point as the old manager dict, including when
        # execution succeeds but the worker subsequently fails during cleanup.
        if "success" in result:
            self.value.value = result["success"] is True


def python_program_passes(
    code: str,
    timeout: float = 5.0,
    maximum_memory_bytes: int | None = 256 * 1024 * 1024,
) -> bool:
    """Return the existing worker's verdict without a result-manager process.

    HumanEval consumes only success. A shared boolean avoids starting a second
    process and serializing the worker's captured output. Execution still uses
    nanochat's temporary directory, reliability guard, memory limit, and alarm.
    The parent retains the same timeout grace and reaps the worker on every exit.
    """
    context = multiprocessing.get_context()
    success = context.RawValue(c_bool, False)
    process = context.Process(
        target=_unsafe_execute,
        args=(code, timeout, maximum_memory_bytes, _SuccessFlag(success)),
    )
    try:
        process.start()
        process.join(timeout=timeout + 1)
        return not process.is_alive() and bool(success.value)
    finally:
        if process.is_alive():
            process.kill()
            process.join()
        process.close()


@dataclass
class _ProgramWorker:
    index: int
    process: BaseProcess
    success: c_bool
    deadline: float = 0.0
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        try:
            if self.process.is_alive():
                self.process.kill()
            if self.process.pid is not None:
                self.process.join()
        finally:
            self.process.close()
            self.closed = True


def python_programs_pass(
    codes: Sequence[str],
    *,
    max_workers: int,
    timeout: float = 5.0,
    maximum_memory_bytes: int | None = 256 * 1024 * 1024,
) -> list[bool]:
    """Grade ordered programs using bounded, fresh execution processes.

    Launching and sentinel supervision stay on the calling process's main
    thread. Every child retains the scalar grader's worker, memory limit,
    execution alarm, and separate timeout-plus-one-second parent grace. Queue
    time does not consume a child's grace; its deadline starts after start()
    returns. Results retain submission order, including duplicate programs.

    The supervisor checks all live children before every launch. As with the
    scalar grader, OS scheduling can delay deadline observation; another slow
    process launch can also delay this single-threaded supervisor.
    """
    if not isinstance(max_workers, int) or isinstance(max_workers, bool):
        raise TypeError("max_workers must be an int")
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if isinstance(codes, (str, bytes)) or any(not isinstance(code, str) for code in codes):
        raise TypeError("codes must be a sequence of program strings")
    if not codes:
        return []
    if current_thread() is not main_thread():
        raise RuntimeError("batch program grading must run on the main thread")

    context = multiprocessing.get_context()
    results = [False] * len(codes)
    active: list[_ProgramWorker] = []
    next_index = 0
    with ExitStack() as cleanup:
        while next_index < len(codes) or active:
            for worker in tuple(active):
                if not worker.process.is_alive():
                    results[worker.index] = bool(worker.success.value)
                    worker.close()
                    active.remove(worker)
                elif time.monotonic() >= worker.deadline:
                    worker.close()
                    active.remove(worker)

            if next_index < len(codes) and len(active) < max_workers:
                success = context.RawValue(c_bool, False)
                process = context.Process(
                    target=_unsafe_execute,
                    args=(codes[next_index], timeout, maximum_memory_bytes, _SuccessFlag(success)),
                )
                worker = _ProgramWorker(next_index, process, success)
                # Register before start() so a startup error also cleans up
                # earlier live children and any partially started new child.
                cleanup.callback(worker.close)
                process.start()
                worker.deadline = time.monotonic() + timeout + 1
                active.append(worker)
                next_index += 1
                continue

            if active:
                remaining = max(0.0, min(worker.deadline for worker in active) - time.monotonic())
                wait([worker.process.sentinel for worker in active], timeout=remaining)
    return results
