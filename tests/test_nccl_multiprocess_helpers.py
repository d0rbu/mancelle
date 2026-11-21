"""Unit contracts for the opt-in single-node NCCL spawn harness."""

from __future__ import annotations

import os
import queue
from collections.abc import Callable
from typing import Any, cast

import pytest

import tests.nccl_multiprocess_helpers as harness


@pytest.fixture(autouse=True)
def _restore_launcher_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct worker-unit calls must not leak child-only coordinates."""

    for name in ("MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.setenv(name, os.environ.get(name, "<unit-test-unset>"))


@pytest.mark.parametrize("world_size", [True, 0, -1, 1.0, "2"])
def test_run_nccl_multiprocess_rejects_nonpositive_or_nonstrict_world_size(
    world_size: Any,
) -> None:
    with pytest.raises(ValueError, match="world_size must be >= 1"):
        harness.run_nccl_multiprocess(world_size, lambda *_args: None)


def test_run_nccl_multiprocess_rejects_noncallable_target() -> None:
    target = cast(Callable[..., Any], None)
    with pytest.raises(TypeError, match="target must be callable"):
        harness.run_nccl_multiprocess(1, target)


def test_run_nccl_multiprocess_reports_cuda_and_nccl_preconditions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(harness.t.cuda, "is_available", lambda: False)
    monkeypatch.setattr(harness.t.cuda, "device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="requires 2 local CUDA devices, found 0"):
        harness.run_nccl_multiprocess(2, lambda *_args: None)

    monkeypatch.setattr(harness.t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(harness.t.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(harness.dist, "is_nccl_available", lambda: False)
    with pytest.raises(RuntimeError, match="NCCL backend is unavailable"):
        harness.run_nccl_multiprocess(2, lambda *_args: None)


def test_worker_main_sets_coordinates_returns_payload_and_tears_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Any, ...]] = []
    result_queue: queue.Queue[Any] = queue.Queue()
    monkeypatch.setattr(harness.t.cuda, "set_device", lambda rank: calls.append(("cuda", rank)))
    monkeypatch.setattr(
        harness.dist,
        "init_process_group",
        lambda backend, rank, world_size: calls.append(
            ("init", backend, rank, world_size)
        ),
    )
    monkeypatch.setattr(harness.dist, "is_available", lambda: True)
    monkeypatch.setattr(harness.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        harness.dist,
        "destroy_process_group",
        lambda: calls.append(("destroy",)),
    )

    def target(rank: int, world_size: int, suffix: str) -> str:
        assert harness.os.environ["MASTER_ADDR"] == "127.0.0.1"
        assert harness.os.environ["MASTER_PORT"] == "12345"
        assert harness.os.environ["RANK"] == "1"
        assert harness.os.environ["LOCAL_RANK"] == "1"
        assert harness.os.environ["WORLD_SIZE"] == "2"
        return f"{rank}/{world_size}/{suffix}"

    harness._worker_main(1, 2, 12345, target, ("ok",), result_queue)

    assert result_queue.get_nowait() == (1, ("ok", "1/2/ok"))
    assert calls == [
        ("cuda", 1),
        ("init", "nccl", 1, 2),
        ("destroy",),
    ]


def test_worker_main_preserves_full_child_traceback_and_still_tears_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destroyed = False
    result_queue: queue.Queue[Any] = queue.Queue()
    monkeypatch.setattr(harness.t.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(harness.dist, "init_process_group", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(harness.dist, "is_available", lambda: True)
    monkeypatch.setattr(harness.dist, "is_initialized", lambda: True)

    def destroy() -> None:
        nonlocal destroyed
        destroyed = True

    monkeypatch.setattr(harness.dist, "destroy_process_group", destroy)

    def target(_rank: int, _world_size: int) -> None:
        raise LookupError("child sentinel")

    harness._worker_main(0, 1, 12346, target, (), result_queue)
    rank, (status, child_traceback) = result_queue.get_nowait()
    assert rank == 0
    assert status == "error"
    assert "LookupError: child sentinel" in child_traceback
    assert "in target" in child_traceback
    assert destroyed


class _FakeProcess:
    def __init__(self, target: Callable[..., Any], args: tuple[Any, ...]) -> None:
        self._target = target
        self._args = args
        self.exitcode: int | None = None
        self.started = False
        self.terminated = False
        self.killed = False

    def start(self) -> None:
        self.started = True
        self._target(*self._args)
        self.exitcode = 0

    def join(self, timeout: int) -> None:
        assert timeout in (30, 5)

    def is_alive(self) -> bool:
        return False

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class _FakeContext:
    def __init__(self) -> None:
        self.result_queue: queue.Queue[Any] = queue.Queue()
        self.processes: list[_FakeProcess] = []

    def Queue(self) -> queue.Queue[Any]:  # noqa: N802 - multiprocessing API
        return self.result_queue

    def Process(self, *, target: Callable[..., Any], args: tuple[Any, ...]) -> _FakeProcess:  # noqa: N802 - multiprocessing API
        process = _FakeProcess(target, args)
        self.processes.append(process)
        return process


def _install_fake_parent_runtime(
    monkeypatch: pytest.MonkeyPatch,
    worker: Callable[..., None],
) -> _FakeContext:
    context = _FakeContext()
    monkeypatch.setattr(harness.t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(harness.t.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(harness.dist, "is_nccl_available", lambda: True)
    monkeypatch.setattr(harness.mp, "get_context", lambda _method: context)
    monkeypatch.setattr(harness, "_find_free_port", lambda: 23456)
    monkeypatch.setattr(harness, "_worker_main", worker)
    return context


def test_worker_deadline_covers_cold_h200_inductor_compilation() -> None:
    assert harness._RESULT_TIMEOUT_S == 900


def test_cleanup_escalates_to_kill_when_worker_ignores_terminate() -> None:
    class StubbornProcess:
        def __init__(self) -> None:
            self.join_timeouts: list[int] = []
            self.terminated = False
            self.killed = False

        def join(self, timeout: int) -> None:
            self.join_timeouts.append(timeout)

        def is_alive(self) -> bool:
            return not self.killed

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            assert self.terminated
            self.killed = True

    process = StubbornProcess()
    harness._join_or_kill_process(process)

    assert process.join_timeouts == [30, 5, 5]
    assert process.terminated
    assert process.killed


def test_parent_returns_payloads_in_rank_order_and_checks_exit_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def worker(
        rank: int,
        world_size: int,
        master_port: int,
        target: Callable[..., Any],
        args: tuple[Any, ...],
        result_queue: queue.Queue[Any],
    ) -> None:
        assert world_size == 3
        assert master_port == 23456
        result_queue.put((rank, ("ok", target(rank, world_size, *args))))

    context = _install_fake_parent_runtime(monkeypatch, worker)
    results = harness.run_nccl_multiprocess(
        3,
        lambda rank, world_size, offset: rank + world_size + offset,
        10,
    )
    assert results == [13, 14, 15]
    assert len(context.processes) == 3
    assert all(process.started and process.exitcode == 0 for process in context.processes)


@pytest.mark.parametrize(
    ("second_rank", "second_status", "message"),
    [
        (0, "ok", "rank 0 reported more than once"),
        (1, "maybe", "rank 1 reported invalid status 'maybe'"),
        (3, "ok", "reported invalid rank 3"),
    ],
)
def test_parent_rejects_duplicate_invalid_status_and_out_of_range_reports(
    monkeypatch: pytest.MonkeyPatch,
    second_rank: int,
    second_status: str,
    message: str,
) -> None:
    def worker(
        rank: int,
        _world_size: int,
        _master_port: int,
        _target: Callable[..., Any],
        _args: tuple[Any, ...],
        result_queue: queue.Queue[Any],
    ) -> None:
        if rank == 0:
            result_queue.put((0, ("ok", "first")))
        else:
            result_queue.put((second_rank, (second_status, "second")))

    _install_fake_parent_runtime(monkeypatch, worker)
    with pytest.raises(RuntimeError, match=message):
        harness.run_nccl_multiprocess(2, lambda *_args: None)
