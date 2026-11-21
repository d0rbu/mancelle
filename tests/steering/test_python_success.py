"""Compare boolean grading with the unchanged nanochat execution contract."""

from __future__ import annotations

import multiprocessing
import os
import sys
import threading
import time
from ctypes import c_bool
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nanochat.execution import execute_code
from steering.eval import python_success


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("assert sum(range(10)) == 45", True),
        ("print('hello'); import sys; print('error stream', file=sys.stderr)", True),
        ("print('x' * 200_000)", True),
        ("assert 1 == 2", False),
        ("def broken(", False),
        ("raise SystemExit(0)", False),
        ("raise MemoryError('probe')", False),
        ("while True: pass", False),
        ("import os; os.system('true')", False),
        ("input()", False),
        ("import os; os._exit(7)", False),
        (
            "import tempfile\n"
            "def failed_cleanup(self): raise RuntimeError('cleanup failed')\n"
            "tempfile.TemporaryDirectory.cleanup = failed_cleanup",
            True,
        ),
    ],
)
def test_success_matches_reference(code: str, expected: bool) -> None:
    reference = execute_code(code, timeout=0.1, maximum_memory_bytes=None)
    actual = python_success.python_program_passes(code, timeout=0.1, maximum_memory_bytes=None)
    assert actual is reference.success is expected


@pytest.mark.skipif(sys.platform == "darwin", reason="nanochat disables memory limits on macOS")
def test_memory_limit_matches_reference() -> None:
    code = "data = bytearray(512 * 1024 * 1024)"
    reference = execute_code(code, timeout=1)
    assert reference.memory_exceeded
    assert python_success.python_program_passes(code, timeout=1) is reference.success is False


@pytest.mark.parametrize("method", multiprocessing.get_all_start_methods())
def test_start_methods_and_no_manager(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    context = multiprocessing.get_context(method)

    def forbidden_manager():
        pytest.fail("boolean grading must not start a manager")

    monkeypatch.setattr(python_success.multiprocessing, "get_context", lambda: context)
    monkeypatch.setattr(context, "Manager", forbidden_manager)
    assert python_success.python_program_passes("assert 2 + 2 == 4", maximum_memory_bytes=None)
    assert not python_success.python_program_passes("assert False", maximum_memory_bytes=None)


def test_guard_and_temporary_directory_are_isolated(tmp_path: Path) -> None:
    marker = tmp_path / "execution_directory.txt"
    code = (
        "import os\n"
        "from pathlib import Path\n"
        "assert os.system is None and os.fork is None\n"
        f"Path({str(marker)!r}).write_text(os.readlink('/proc/self/cwd'))\n"
        "Path('local_file').write_text('removed with the execution directory')\n"
    )
    # The guard disables getcwd; /proc reports the worker cwd.
    if not Path("/proc/self/cwd").exists():
        pytest.skip("temporary-directory observation requires procfs")
    assert python_success.python_program_passes(code, maximum_memory_bytes=None)
    assert not Path(marker.read_text()).exists()
    assert callable(os.getcwd)
    assert callable(os.fork)


def test_parent_deadline_kills_and_reaps_worker() -> None:
    before = {child.pid for child in multiprocessing.active_children()}
    # The execution alarm is disabled by the program; the parent must enforce
    # its separate timeout + one-second grace and reap the child.
    code = "import signal; signal.setitimer(signal.ITIMER_REAL, 0);\nwhile True: pass"
    start = time.monotonic()
    assert not python_success.python_program_passes(code, timeout=0.1, maximum_memory_bytes=None)
    assert time.monotonic() - start < 4
    assert {child.pid for child in multiprocessing.active_children()} == before


@pytest.mark.parametrize("max_workers", [1, 2])
def test_batch_matches_scalar_verdicts_in_input_order(max_workers: int) -> None:
    codes = [
        "assert sum(range(10)) == 45",
        "assert False",
        "def broken(",
        "print('x' * 200_000)",
        "raise SystemExit(0)",
        "raise MemoryError('probe')",
        "import os; os._exit(7)",
        "input()",
        "import os; os.system('true')",
        "while True: pass",
        "import tempfile\n"
        "def failed_cleanup(self): raise RuntimeError('cleanup failed')\n"
        "tempfile.TemporaryDirectory.cleanup = failed_cleanup",
        "assert sum(range(10)) == 45",
    ]
    expected = [True, False, False, True, False, False, False, False, False, False, True, True]
    before = {child.pid for child in multiprocessing.active_children()}
    reference = [
        python_success.python_program_passes(code, timeout=0.1, maximum_memory_bytes=None)
        for code in codes
    ]

    actual = python_success.python_programs_pass(
        codes,
        max_workers=max_workers,
        timeout=0.1,
        maximum_memory_bytes=None,
    )

    assert actual == reference == expected
    assert {child.pid for child in multiprocessing.active_children()} == before


@pytest.mark.parametrize("method", multiprocessing.get_all_start_methods())
def test_batch_respects_context_and_does_not_start_manager(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    context = multiprocessing.get_context(method)

    def forbidden_manager():
        pytest.fail("batch grading must not start a manager")

    monkeypatch.setattr(python_success.multiprocessing, "get_context", lambda: context)
    monkeypatch.setattr(context, "Manager", forbidden_manager)

    assert python_success.python_programs_pass(
        ["assert True", "assert False", "assert True"],
        max_workers=2,
        maximum_memory_bytes=None,
    ) == [True, False, True]


@pytest.mark.skipif(sys.platform == "darwin", reason="nanochat disables memory limits on macOS")
def test_batch_retains_per_worker_memory_limit() -> None:
    code = "data = bytearray(512 * 1024 * 1024)"
    expected = python_success.python_program_passes(code, timeout=1)

    assert (
        python_success.python_programs_pass([code, code], max_workers=2, timeout=1)
        == [
            expected,
            expected,
        ]
        == [False, False]
    )


def test_batch_overlaps_fresh_guarded_workers_and_removes_tempdirs(tmp_path: Path) -> None:
    if not Path("/proc/self/cwd").exists():
        pytest.skip("temporary-directory observation requires procfs")
    markers = [tmp_path / f"worker-{index}.txt" for index in range(2)]
    codes = [
        "import builtins, os, time\n"
        "from pathlib import Path\n"
        "assert os.system is None and os.fork is None\n"
        "assert not hasattr(builtins, '_batch_grader_marker')\n"
        "builtins._batch_grader_marker = True\n"
        f"Path({str(marker)!r}).write_text(str(os.getpid()) + '\\n' + os.readlink('/proc/self/cwd'))\n"
        f"while not Path({str(markers[1 - index])!r}).exists(): time.sleep(0.005)\n"
        "Path('worker_local_file').write_text('temporary')\n"
        for index, marker in enumerate(markers)
    ]
    before = {child.pid for child in multiprocessing.active_children()}

    assert python_success.python_programs_pass(
        codes,
        max_workers=2,
        timeout=2,
        maximum_memory_bytes=None,
    ) == [True, True]

    records = [marker.read_text().splitlines() for marker in markers]
    assert records[0][0] != records[1][0]
    assert records[0][1] != records[1][1]
    assert all(not Path(record[1]).exists() for record in records)
    assert {child.pid for child in multiprocessing.active_children()} == before
    assert callable(os.getcwd)
    assert callable(os.fork)


def test_batch_parent_deadlines_reap_workers_with_disabled_alarms() -> None:
    before = {child.pid for child in multiprocessing.active_children()}
    hanging = "import signal; signal.setitimer(signal.ITIMER_REAL, 0);\nwhile True: pass"

    assert python_success.python_programs_pass(
        [hanging, "assert True", hanging],
        max_workers=2,
        timeout=0.1,
        maximum_memory_bytes=None,
    ) == [False, True, False]

    assert {child.pid for child in multiprocessing.active_children()} == before


@dataclass
class _ProgramPlan:
    runtime: float
    success: bool = True
    start_delay: float = 0.0
    start_error: str | None = None
    publish_after: float | None = None


class _SimulatedProcess:
    def __init__(self, context: _SimulatedContext, plan: _ProgramPlan, args: tuple[Any, ...]):
        self.context = context
        self.plan = plan
        self.args = args
        self.pid: int | None = None
        self.sentinel = len(context.processes)
        self.started_at = 0.0
        self.finished_at = float("inf")
        self.killed_at: float | None = None
        self.joined = False
        self.closed = False

    def start(self) -> None:
        if self.plan.start_error == "before":
            raise RuntimeError("start failed before creating a child")
        other_live = sum(
            process.is_alive()
            for process in self.context.processes
            if process is not self and not process.closed
        )
        self.context.peak_live = max(self.context.peak_live, other_live + 1)
        self.pid = 1000 + self.sentinel
        self.started_at = self.context.now
        self.finished_at = self.started_at + self.plan.runtime
        self.context.now += self.plan.start_delay
        if self.plan.start_error == "after":
            raise RuntimeError("start failed after creating a child")

    def is_alive(self) -> bool:
        assert not self.closed
        if self.pid is None or self.killed_at is not None:
            return False
        publication = (
            self.finished_at
            if self.plan.publish_after is None
            else self.started_at + self.plan.publish_after
        )
        if self.context.now >= publication:
            self.args[-1].update({"success": self.plan.success})
        return self.context.now < self.finished_at

    def kill(self) -> None:
        assert self.is_alive()
        self.killed_at = self.context.now

    def join(self) -> None:
        assert self.pid is not None
        assert not self.is_alive()
        self.joined = True

    def close(self) -> None:
        assert not self.is_alive()
        assert self.pid is None or self.joined
        self.closed = True


class _SimulatedContext:
    def __init__(self, plans: dict[str, _ProgramPlan]):
        self.plans = plans
        self.now = 0.0
        self.peak_live = 0
        self.processes: list[_SimulatedProcess] = []

    def RawValue(self, _ctype: object, value: bool) -> c_bool:  # noqa: N802
        return c_bool(value)

    def Process(self, *, target: object, args: tuple[Any, ...]) -> _SimulatedProcess:  # noqa: N802
        assert target is python_success._unsafe_execute
        process = _SimulatedProcess(self, self.plans[args[0]], args)
        self.processes.append(process)
        return process

    def wait(self, sentinels: list[int], timeout: float) -> list[int]:
        processes = [self.processes[sentinel] for sentinel in sentinels]
        self.now = min(self.now + timeout, min(process.finished_at for process in processes))
        return [process.sentinel for process in processes if not process.is_alive()]


def _install_simulated_context(monkeypatch: pytest.MonkeyPatch, context: _SimulatedContext) -> None:
    monkeypatch.setattr(python_success.multiprocessing, "get_context", lambda: context)
    monkeypatch.setattr(python_success, "time", SimpleNamespace(monotonic=lambda: context.now))
    monkeypatch.setattr(python_success, "wait", context.wait)


def test_batch_bound_order_and_refill_do_not_wait_for_first_program(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _SimulatedContext(
        {
            "slow": _ProgramPlan(2.0),
            "fail": _ProgramPlan(0.2, success=False),
            "pass": _ProgramPlan(0.3),
        }
    )
    _install_simulated_context(monkeypatch, context)

    assert python_success.python_programs_pass(
        ["slow", "fail", "pass", "pass"],
        max_workers=2,
        timeout=5,
        maximum_memory_bytes=12345,
    ) == [True, False, True, True]

    assert context.peak_live == 2
    assert context.processes[-1].started_at == pytest.approx(0.5)
    assert [process.args[:3] for process in context.processes] == [
        (code, 5, 12345) for code in ("slow", "fail", "pass", "pass")
    ]
    assert all(process.joined and process.closed for process in context.processes)


def test_batch_independent_deadlines_start_after_launch_and_exclude_queue_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _SimulatedContext(
        {
            "first": _ProgramPlan(float("inf"), start_delay=0.4, publish_after=0.1),
            "second": _ProgramPlan(float("inf"), start_delay=0.2),
            "queued": _ProgramPlan(0.5),
        }
    )
    _install_simulated_context(monkeypatch, context)

    assert python_success.python_programs_pass(
        ["first", "second", "queued"],
        max_workers=2,
        timeout=0.1,
    ) == [False, False, True]

    first, second, queued = context.processes
    assert first.killed_at == pytest.approx(1.5)
    assert second.killed_at == pytest.approx(1.7)
    assert queued.started_at == pytest.approx(1.5)
    assert context.now == pytest.approx(2.0)
    assert all(process.joined and process.closed for process in context.processes)


@pytest.mark.parametrize("first_runtime", [0.5, 1.5, float("inf")])
def test_batch_observes_deadlines_after_a_blocked_start(
    monkeypatch: pytest.MonkeyPatch,
    first_runtime: float,
) -> None:
    context = _SimulatedContext(
        {
            "first": _ProgramPlan(first_runtime),
            "blocked_start": _ProgramPlan(3.0, start_delay=2.0),
        }
    )
    _install_simulated_context(monkeypatch, context)

    result = python_success.python_programs_pass(
        ["first", "blocked_start"],
        max_workers=2,
        timeout=0.1,
    )

    # The blocked start returns after the first deadline (1.1). Preserve the
    # scalar grader's observed is_alive/flag rule: an already exited worker
    # can pass; a still-live one is killed immediately on resumption. A ready
    # sentinel cannot reveal whether an earlier exit was before the deadline.
    assert result == [first_runtime != float("inf"), True]
    assert context.processes[0].killed_at == (2.0 if first_runtime == float("inf") else None)
    assert all(process.joined and process.closed for process in context.processes)


@pytest.mark.parametrize("failure", ["before", "after"])
def test_batch_startup_error_cleans_up_previous_and_partially_started_workers(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    context = _SimulatedContext(
        {
            "first": _ProgramPlan(float("inf")),
            "failure": _ProgramPlan(float("inf"), start_error=failure),
            "never_started": _ProgramPlan(0),
        }
    )
    _install_simulated_context(monkeypatch, context)

    with pytest.raises(RuntimeError, match="start failed"):
        python_success.python_programs_pass(["first", "failure", "never_started"], max_workers=2)

    assert len(context.processes) == 2
    assert all(process.closed for process in context.processes)
    assert context.processes[0].killed_at == 0
    assert context.processes[0].joined
    assert context.processes[1].joined is (failure == "after")


def test_batch_parent_interruption_cleans_up_all_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    context = _SimulatedContext({"hang": _ProgramPlan(float("inf"))})
    _install_simulated_context(monkeypatch, context)

    def interrupt_wait(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(python_success, "wait", interrupt_wait)
    with pytest.raises(KeyboardInterrupt):
        python_success.python_programs_pass(["hang", "hang"], max_workers=2)

    assert len(context.processes) == 2
    assert all(
        process.killed_at == 0 and process.joined and process.closed
        for process in context.processes
    )


@pytest.mark.parametrize(
    ("max_workers", "error"),
    [(0, ValueError), (-1, ValueError), (True, TypeError), (1.0, TypeError)],
)
def test_batch_rejects_invalid_worker_bound(max_workers: Any, error: type[Exception]) -> None:
    with pytest.raises(error):
        python_success.python_programs_pass([], max_workers=max_workers)


def test_batch_empty_and_singleton_inputs() -> None:
    assert python_success.python_programs_pass([], max_workers=2) == []
    assert python_success.python_programs_pass(
        ["assert True"], max_workers=2, maximum_memory_bytes=None
    ) == [True]


def test_batch_rejects_launching_from_background_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    errors: list[BaseException] = []

    def forbidden_context():
        pytest.fail("thread rejection must happen before creating workers")

    def invoke():
        try:
            python_success.python_programs_pass(["assert True"], max_workers=2)
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(python_success.multiprocessing, "get_context", forbidden_context)
    thread = threading.Thread(target=invoke)
    thread.start()
    thread.join()

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "main thread" in str(errors[0])
