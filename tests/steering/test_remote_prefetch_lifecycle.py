from __future__ import annotations

import fcntl
import json
import os
import threading
from pathlib import Path
from typing import Any, cast

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

import steering.conditional.encoder_decoder.dataloader as dataloader
from scripts.train.conditional_encdec_train import _close_training_background_workers
from steering.conditional.encoder_decoder.dataloader import ClosableBatchPrefetcher
from steering.refined_types import PositiveInt
from steering.remote_body_prefetch import (
    REMOTE_PREFETCH_GRACEFUL_JOIN_SECONDS,
    REMOTE_PREFETCH_HARD_JOIN_SECONDS,
    RemoteBodyPrefetcher,
)
from steering.remote_training import RemoteTrainingError

_PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True)
_PREFETCH_BATCHES = PositiveInt.parse(1)


class _OuterProducer:
    def __init__(
        self,
        calls: list[str],
        *,
        terminal_after_close: bool,
        close_error: BaseException | None = None,
    ) -> None:
        self._calls = calls
        self._terminal_after_close = terminal_after_close
        self._close_error = close_error
        self._terminal = False

    def close(self) -> None:
        self._calls.append("outer.close")
        self._terminal = self._terminal_after_close
        if self._close_error is not None:
            raise self._close_error

    def is_terminal(self) -> bool:
        self._calls.append("outer.is_terminal")
        return self._terminal


class _BodyController:
    def __init__(self, calls: list[str], *, terminal: bool) -> None:
        self._calls = calls
        self._terminal = terminal

    def stop_worker(self) -> None:
        self._calls.append("body.stop_worker")

    def worker_is_terminal(self) -> bool:
        self._calls.append("body.worker_is_terminal")
        return self._terminal

    def release_session(self) -> None:
        self._calls.append("body.release_session")


class _RecordingPipe:
    def __init__(self) -> None:
        self.sent: list[object] = []
        self.closed = False

    def send(self, message: object) -> None:
        self.sent.append(message)

    def poll(self) -> bool:
        return False

    def recv(self) -> object:
        raise AssertionError("an empty status pipe must not be read")

    def close(self) -> None:
        self.closed = True


class _HardTerminatedProcess:
    def __init__(self) -> None:
        self.pid = 4242
        self._alive = True
        self.join_calls: list[float | None] = []
        self.terminate_calls = 0

    @property
    def exitcode(self) -> int | None:
        return None if self._alive else -15

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._alive = False


def test_trainer_teardown_retains_session_for_nonterminal_outer_producer() -> None:
    calls: list[str] = []
    outer = _OuterProducer(calls, terminal_after_close=False)
    body = _BodyController(calls, terminal=True)

    with pytest.raises(RemoteTrainingError, match="session lock was retained"):
        _close_training_background_workers(
            batch_prefetch_loader=cast(Any, outer),
            cuda_staging_loader=None,
            remote_body_prefetcher=cast(Any, body),
            checkpoint_publisher=None,
            active_error=None,
        )

    assert calls == [
        "outer.close",
        "outer.is_terminal",
        "body.stop_worker",
        "body.worker_is_terminal",
    ]
    assert "body.release_session" not in calls


def test_trainer_teardown_releases_after_terminal_close_error() -> None:
    calls: list[str] = []
    close_error = OSError("outer close failed after becoming terminal")
    outer = _OuterProducer(
        calls,
        terminal_after_close=True,
        close_error=close_error,
    )
    body = _BodyController(calls, terminal=True)

    with pytest.raises(OSError) as exc_info:
        _close_training_background_workers(
            batch_prefetch_loader=cast(Any, outer),
            cuda_staging_loader=None,
            remote_body_prefetcher=cast(Any, body),
            checkpoint_publisher=None,
            active_error=None,
        )

    assert exc_info.value is close_error
    assert calls == [
        "outer.close",
        "outer.is_terminal",
        "body.stop_worker",
        "body.worker_is_terminal",
        "body.release_session",
    ]


def test_hard_termination_records_terminal_status_before_session_release(
    tmp_path: Path,
) -> None:
    control_root = tmp_path / "control"
    (control_root / "status").mkdir(parents=True)
    lock_path = tmp_path / "remote-prefetch.session.lock"
    session_lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    probe_fd = os.open(lock_path, os.O_RDWR)
    fcntl.flock(session_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    process = _HardTerminatedProcess()
    command_pipe = _RecordingPipe()
    status_pipe = _RecordingPipe()
    controller = cast(Any, object.__new__(RemoteBodyPrefetcher))
    controller.nonce = "ab" * 16
    controller.rank = 0
    controller.world_size = 1
    controller.control_root = control_root
    controller.process = process
    controller.command_pipe = command_pipe
    controller.status_pipe = status_pipe
    controller.session_lock_fd = session_lock_fd
    controller.cancel_event = threading.Event()
    controller._stopping = False
    controller._worker_stopped = False
    controller._session_released = False
    controller._failure = None
    controller._local_failure_lock = threading.RLock()
    controller._last_successful_local_failure_check_monotonic = None
    controller._peer_status_sweep_lock = threading.Lock()
    controller._last_successful_peer_status_sweep_monotonic = None

    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(probe_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

        controller.stop_worker()

        assert process.join_calls == [
            REMOTE_PREFETCH_GRACEFUL_JOIN_SECONDS,
            REMOTE_PREFETCH_HARD_JOIN_SECONDS,
        ]
        assert process.terminate_calls == 1
        assert command_pipe.sent == [{"op": "cancel"}]
        assert command_pipe.closed
        assert status_pipe.closed
        assert controller.worker_is_terminal()

        status_path = control_root / "status" / "rank_00000.json"
        status = json.loads(status_path.read_text())
        assert status["state"] == "stopped"
        assert status["error_type"] == "HardTerminated"
        assert status["rank"] == 0
        assert status["pid"] == process.pid

        controller.release_session()

        assert controller._session_released
        assert controller.session_lock_fd is None
        fcntl.flock(probe_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(probe_fd, fcntl.LOCK_UN)
    finally:
        if controller.session_lock_fd is not None:
            fcntl.flock(controller.session_lock_fd, fcntl.LOCK_UN)
            os.close(controller.session_lock_fd)
        os.close(probe_fd)


def test_controller_close_preserves_stop_error_over_release_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    stop_error = RuntimeError("worker stop is primary")
    release_error = OSError("session release is secondary")
    controller = cast(Any, object.__new__(RemoteBodyPrefetcher))

    def stop_worker() -> None:
        calls.append("stop_worker")
        raise stop_error

    def worker_is_terminal() -> bool:
        calls.append("worker_is_terminal")
        return True

    def release_session() -> None:
        calls.append("release_session")
        raise release_error

    monkeypatch.setattr(controller, "stop_worker", stop_worker)
    monkeypatch.setattr(controller, "worker_is_terminal", worker_is_terminal)
    monkeypatch.setattr(controller, "release_session", release_session)

    with pytest.raises(RuntimeError) as exc_info:
        controller.close()

    assert exc_info.value is stop_error
    assert calls == ["stop_worker", "worker_is_terminal", "release_session"]
    assert stop_error.__notes__ == [
        "remote prefetch session release also failed: OSError: session release is secondary"
    ]


def test_closable_prefetcher_callback_failure_still_joins_and_closes_inner() -> None:
    started = threading.Event()
    finalized = threading.Event()
    callback_error = OSError("cancel callback failed")
    callback_calls = 0

    def source():
        try:
            value = 0
            while True:
                started.set()
                yield value
                value += 1
        finally:
            finalized.set()

    def failing_callback() -> None:
        nonlocal callback_calls
        callback_calls += 1
        raise callback_error

    prefetcher = ClosableBatchPrefetcher(
        source(),
        prefetch_batches=_PREFETCH_BATCHES,
        stop_callback=failing_callback,
        failure_check=lambda: None,
    )
    assert started.wait(timeout=1)

    with pytest.raises(OSError) as exc_info:
        prefetcher.close()

    assert exc_info.value is callback_error
    assert finalized.wait(timeout=1)
    assert prefetcher.is_terminal()
    prefetcher.close()
    assert callback_calls == 1


def test_closable_prefetcher_preserves_inner_close_error_over_producer_error() -> None:
    started = threading.Event()
    producer_error = RuntimeError("producer failed")
    close_error = OSError("inner close failed")
    close_calls = 0

    class BrokenIterator:
        def __iter__(self) -> BrokenIterator:
            return self

        def __next__(self) -> object:
            started.set()
            raise producer_error

        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1
            raise close_error

    prefetcher = ClosableBatchPrefetcher(
        BrokenIterator(),
        prefetch_batches=_PREFETCH_BATCHES,
        stop_callback=lambda: None,
        failure_check=lambda: None,
    )
    assert started.wait(timeout=1)
    prefetcher._thread.join(timeout=1)

    with pytest.raises(OSError) as exc_info:
        prefetcher.close()

    assert exc_info.value is close_error
    assert prefetcher.is_terminal()
    assert close_calls == 1
    assert close_error.__notes__ == [
        "remote batch prefetch cleanup also failed: RuntimeError: producer failed"
    ]


def test_closable_prefetcher_retries_join_after_bounded_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    callback_calls = 0

    def source():
        started.set()
        assert release.wait(timeout=1)
        if False:
            yield None

    def callback() -> None:
        nonlocal callback_calls
        callback_calls += 1

    monkeypatch.setattr(dataloader, "CLOSABLE_BATCH_PREFETCH_JOIN_SECONDS", 0.001)
    prefetcher = ClosableBatchPrefetcher(
        source(),
        prefetch_batches=_PREFETCH_BATCHES,
        stop_callback=callback,
        failure_check=lambda: None,
    )
    assert started.wait(timeout=1)

    try:
        with pytest.raises(RuntimeError, match="did not stop"):
            prefetcher.close()
        assert not prefetcher.is_terminal()
    finally:
        release.set()
        prefetcher._thread.join(timeout=1)
    prefetcher.close()

    assert prefetcher.is_terminal()
    assert callback_calls == 1


@_PROPERTY_SETTINGS
@given(
    st.one_of(
        st.integers(max_value=0),
        st.booleans(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.text(),
        st.none(),
    )
)
def test_closable_prefetcher_rejects_every_invalid_depth(invalid_depth: object) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="prefetch_batches"):
        ClosableBatchPrefetcher(
            iter(()),
            prefetch_batches=cast(Any, invalid_depth),
            stop_callback=lambda: None,
            failure_check=lambda: None,
        )


def test_closable_prefetcher_rejects_invalid_depth_before_thread_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread_calls = 0

    def forbidden_thread(*_args: Any, **_kwargs: Any) -> None:
        nonlocal thread_calls
        thread_calls += 1
        raise AssertionError("invalid depth must fail before producer-thread creation")

    monkeypatch.setattr(dataloader.threading, "Thread", forbidden_thread)

    with pytest.raises(BeartypeCallHintParamViolation, match="prefetch_batches"):
        ClosableBatchPrefetcher(
            iter(()),
            prefetch_batches=cast(Any, 0),
            stop_callback=lambda: None,
            failure_check=lambda: None,
        )

    assert thread_calls == 0
