"""Ownership, lease, janitor, signal, and PID-reuse tests for shared stores."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import cast

import pytest

from steering.conditional import shared_store_lifecycle as lifecycle


def _new_store(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "shm"
    root.mkdir()
    store = Path(tempfile.mkdtemp(prefix=lifecycle.STORE_PREFIX, dir=root))
    return root, store


def _fake_proc_stat(pid: int, starttime: int, comm: str = "worker ) with space") -> str:
    tail = [
        "S",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        str(starttime),
    ]
    return f"{pid} ({comm}) {' '.join(tail)}\n"


def _fake_identity_tree(tmp_path: Path, *, pid: int, starttime: int) -> tuple[Path, Path]:
    proc_root = tmp_path / "proc"
    (proc_root / str(pid)).mkdir(parents=True)
    (proc_root / str(pid) / "stat").write_text(_fake_proc_stat(pid, starttime))
    boot_id_path = tmp_path / "boot_id"
    boot_id_path.write_text("boot-test-123\n")
    return proc_root, boot_id_path


def _wait_for_path(path: Path, *, exists: bool, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() == exists:
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for exists={exists}: {path}")


def _wait_for_identity_dead(identity: lifecycle.ProcessIdentity, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not identity.is_alive():
            return
        time.sleep(0.02)
    raise AssertionError(f"process identity remained alive: {identity}")


def _dead_owner_for(owner: lifecycle.StoreOwner, *, pid: int = 2_000_000_000) -> lifecycle.StoreOwner:
    return lifecycle.StoreOwner(
        owner=lifecycle.ProcessIdentity(
            boot_id=owner.owner.boot_id,
            pid=pid,
            starttime_ticks=1,
        ),
        store_dir=owner.store_dir,
        store_root=owner.store_root,
        uid=owner.uid,
        created_time_ns=owner.created_time_ns,
    )


def test_read_process_starttime_uses_field_22_after_final_comm_parenthesis(
    tmp_path: Path,
) -> None:
    proc_root, _ = _fake_identity_tree(tmp_path, pid=123, starttime=987654321)
    assert lifecycle._read_process_starttime(123, proc_root) == 987654321


@pytest.mark.parametrize(
    "raw",
    ["", "123 no-parentheses", "123 (x)", "123 (x) S 1 2"],
)
def test_read_process_starttime_rejects_malformed_proc_stat(tmp_path: Path, raw: str) -> None:
    proc_root = tmp_path / "proc"
    (proc_root / "123").mkdir(parents=True)
    (proc_root / "123" / "stat").write_text(raw)
    with pytest.raises((lifecycle.SharedStoreLifecycleError, ProcessLookupError)):
        lifecycle._read_process_starttime(123, proc_root)


def test_process_identity_capture_and_liveness_use_boot_pid_and_starttime(
    tmp_path: Path,
) -> None:
    proc_root, boot_id_path = _fake_identity_tree(tmp_path, pid=123, starttime=456)
    identity = lifecycle.ProcessIdentity.capture(
        123,
        proc_root=proc_root,
        boot_id_path=boot_id_path,
    )
    assert identity == lifecycle.ProcessIdentity("boot-test-123", 123, 456)
    assert identity.is_alive(proc_root=proc_root, boot_id_path=boot_id_path)

    (proc_root / "123" / "stat").write_text(_fake_proc_stat(123, 457))
    assert not identity.is_alive(proc_root=proc_root, boot_id_path=boot_id_path)
    (proc_root / "123" / "stat").unlink()
    assert not identity.is_alive(proc_root=proc_root, boot_id_path=boot_id_path)


def test_process_identity_treats_zombie_as_dead_for_mapping_and_lease_safety(
    tmp_path: Path,
) -> None:
    proc_root, boot_id_path = _fake_identity_tree(tmp_path, pid=123, starttime=456)
    identity = lifecycle.ProcessIdentity.capture(
        123,
        proc_root=proc_root,
        boot_id_path=boot_id_path,
    )
    zombie = _fake_proc_stat(123, 456).replace(") S ", ") Z ", 1)
    (proc_root / "123" / "stat").write_text(zombie)
    assert not identity.is_alive(proc_root=proc_root, boot_id_path=boot_id_path)


def test_process_identity_permission_error_is_unknown_not_dead(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc_root, boot_id_path = _fake_identity_tree(tmp_path, pid=123, starttime=456)
    identity = lifecycle.ProcessIdentity.capture(
        123,
        proc_root=proc_root,
        boot_id_path=boot_id_path,
    )
    stat_path = proc_root / "123" / "stat"
    real_read_text = Path.read_text

    def deny_stat(path: Path, *args, **kwargs):
        if path == stat_path:
            raise PermissionError("synthetic hidden /proc entry")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", deny_stat)
    with pytest.raises(lifecycle.SharedStoreLifecycleError, match="cannot prove"):
        identity.is_alive(proc_root=proc_root, boot_id_path=boot_id_path)


def test_exact_worker_signal_uses_pidfd_without_pid_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(300)"])
    try:
        identity = lifecycle.ProcessIdentity.capture(worker.pid)

        def forbidden_pid_kill(*_args, **_kwargs):
            raise AssertionError("exact worker signaling must never call kill(pid)")

        monkeypatch.setattr(lifecycle.os, "kill", forbidden_pid_kill)
        assert lifecycle._signal_matching_worker(identity, signal.SIGTERM) is True
        assert worker.wait(timeout=5) == -signal.SIGTERM
    finally:
        if worker.poll() is None:
            worker.kill()
        worker.wait(timeout=5)


def test_exact_worker_signal_revalidates_identity_after_binding_pidfd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = lifecycle.ProcessIdentity.capture(os.getpid())
    opened: list[int] = []

    def fake_pidfd_open(_pid: int) -> int:
        fd = os.open("/dev/null", os.O_RDONLY)
        opened.append(fd)
        return fd

    monkeypatch.setattr(lifecycle, "_pidfd_open", fake_pidfd_open)
    monkeypatch.setattr(
        lifecycle.ProcessIdentity,
        "is_alive",
        lambda self: False,
    )

    def forbidden_send(*_args, **_kwargs):
        raise AssertionError("reused PID identity must not be signaled")

    monkeypatch.setattr(lifecycle, "_pidfd_send_signal", forbidden_send)
    assert lifecycle._signal_matching_worker(identity, signal.SIGTERM) is False
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])


def test_exact_worker_signal_fails_closed_without_pidfd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = lifecycle.ProcessIdentity.capture(os.getpid())

    def unsupported(_pid: int) -> int:
        raise NotImplementedError("synthetic unsupported architecture")

    monkeypatch.setattr(lifecycle, "_pidfd_open", unsupported)
    with pytest.raises(lifecycle.SharedStoreLifecycleError, match="PID-racy fallback"):
        lifecycle._signal_matching_worker(identity, signal.SIGTERM)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "1"])
def test_process_identity_capture_rejects_nonpositive_or_noninteger_pid(bad: object) -> None:
    with pytest.raises(ValueError, match="pid must"):
        lifecycle.ProcessIdentity.capture(cast(int, bad))


def test_process_identity_json_roundtrip_is_exact() -> None:
    identity = lifecycle.ProcessIdentity("boot", 12, 34)
    assert lifecycle.ProcessIdentity.from_dict(identity.as_dict(), surface="test") == identity


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        {"boot_id": "b", "pid": 1, "starttime_ticks": 2, "extra": 3},
        {"boot_id": "", "pid": 1, "starttime_ticks": 2},
        {"boot_id": 4, "pid": 1, "starttime_ticks": 2},
        {"boot_id": "b", "pid": 0, "starttime_ticks": 2},
        {"boot_id": "b", "pid": True, "starttime_ticks": 2},
        {"boot_id": "b", "pid": 1, "starttime_ticks": -1},
        {"boot_id": "b", "pid": 1, "starttime_ticks": False},
    ],
)
def test_process_identity_rejects_ambiguous_json(bad: object) -> None:
    with pytest.raises(lifecycle.SharedStoreLifecycleError):
        lifecycle.ProcessIdentity.from_dict(bad, surface="test")


def test_store_owner_json_roundtrip_is_exact(tmp_path: Path) -> None:
    root, store = _new_store(tmp_path)
    owner = lifecycle.initialize_store_ownership(store, root)
    assert lifecycle.StoreOwner.from_dict(owner.as_dict(), surface="test") == owner
    assert lifecycle.read_store_owner(store) == owner
    assert owner.store_dir == str(store.resolve())
    assert owner.store_root == str(root.resolve())
    assert owner.uid == os.getuid()


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 99),
        ("store_dir", "relative"),
        ("store_root", "relative"),
        ("uid", -1),
        ("uid", True),
        ("created_time_ns", -1),
        ("created_time_ns", False),
    ],
)
def test_store_owner_rejects_bad_schema_and_boundaries(
    tmp_path: Path, field: str, value: object
) -> None:
    root, store = _new_store(tmp_path)
    owner = lifecycle.initialize_store_ownership(store, root).as_dict()
    owner[field] = value
    with pytest.raises(lifecycle.SharedStoreLifecycleError):
        lifecycle.StoreOwner.from_dict(owner, surface="test")


def test_initialize_store_publishes_exact_files_and_private_permissions(tmp_path: Path) -> None:
    root, store = _new_store(tmp_path)
    lifecycle.initialize_store_ownership(store, root)
    assert sorted(path.name for path in store.iterdir()) == sorted(
        [
            lifecycle.LEASE_FILENAME,
            lifecycle.OWNER_FILENAME,
            lifecycle.WORKERS_DIRNAME,
        ]
    )
    assert (store / lifecycle.LEASE_FILENAME).stat().st_mode & 0o777 == 0o600
    assert (store / lifecycle.WORKERS_DIRNAME).stat().st_mode & 0o777 == 0o700
    assert json.loads((store / lifecycle.OWNER_FILENAME).read_text())["schema_version"] == 1


def test_initialize_store_rejects_reinitialization(tmp_path: Path) -> None:
    root, store = _new_store(tmp_path)
    lifecycle.initialize_store_ownership(store, root)
    with pytest.raises(lifecycle.SharedStoreLifecycleError, match="already contains"):
        lifecycle.initialize_store_ownership(store, root)


def test_store_boundary_rejects_nested_directory(tmp_path: Path) -> None:
    root = tmp_path / "shm"
    nested = root / "nested" / f"{lifecycle.STORE_PREFIX}x"
    nested.mkdir(parents=True)
    with pytest.raises(lifecycle.SharedStoreLifecycleError, match="not a direct child"):
        lifecycle.initialize_store_ownership(nested, root)


def test_store_boundary_rejects_symlink_even_when_target_is_a_directory(tmp_path: Path) -> None:
    root = tmp_path / "shm"
    root.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    link = root / f"{lifecycle.STORE_PREFIX}link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(lifecycle.SharedStoreLifecycleError, match="real directory"):
        lifecycle.initialize_store_ownership(link, root)


def test_store_boundary_rejects_unmarked_directory(tmp_path: Path) -> None:
    root = tmp_path / "shm"
    root.mkdir()
    store = root / "unmarked"
    store.mkdir()
    with pytest.raises(lifecycle.SharedStoreLifecycleError, match="required prefix"):
        lifecycle.initialize_store_ownership(store, root)


def test_worker_registration_holds_shared_lease_and_unlinks_on_close(tmp_path: Path) -> None:
    root, store = _new_store(tmp_path)
    lifecycle.initialize_store_ownership(store, root)
    lease = lifecycle.register_worker_store_lease(store)
    assert not lease.closed
    assert lease.identity.pid == os.getpid()
    assert lease.registration.is_file()
    registration = json.loads(lease.registration.read_text())
    assert registration == {
        "schema_version": 1,
        "worker": lease.identity.as_dict(),
    }
    with pytest.raises(lifecycle.SharedStoreLifecycleError, match="exclusive store lease"):
        lifecycle._acquire_exclusive_lease(store, 0.0)
    lease.close()
    assert lease.closed
    assert not lease.registration.exists()
    lease.close()
    fd = lifecycle._acquire_exclusive_lease(store, 0.0)
    os.close(fd)


def test_worker_registration_rejects_closing_store_without_registry_side_effect(
    tmp_path: Path,
) -> None:
    root, store = _new_store(tmp_path)
    lifecycle.initialize_store_ownership(store, root)
    lifecycle._mark_closing(store, "test")
    with pytest.raises(lifecycle.SharedStoreLifecycleError, match="already closing"):
        lifecycle.register_worker_store_lease(store)
    assert list((store / lifecycle.WORKERS_DIRNAME).iterdir()) == []


def test_worker_registration_rejects_dead_exact_parent_identity(tmp_path: Path) -> None:
    root, store = _new_store(tmp_path)
    owner = lifecycle.initialize_store_ownership(store, root)
    dead = _dead_owner_for(owner)
    lifecycle._atomic_write_json(store / lifecycle.OWNER_FILENAME, dead.as_dict())
    with pytest.raises(lifecycle.SharedStoreLifecycleError, match="no longer"):
        lifecycle.register_worker_store_lease(store)
    assert list((store / lifecycle.WORKERS_DIRNAME).iterdir()) == []


def test_worker_registration_race_cannot_resurrect_a_cleaned_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, store = _new_store(tmp_path)
    lifecycle.initialize_store_ownership(store, root)
    real_atomic_write = lifecycle._atomic_write_json

    def remove_store_before_registration(path: Path, value: dict) -> None:
        assert path.parent.name == lifecycle.WORKERS_DIRNAME
        # Models the janitor acquiring its exclusive lease and completing
        # rmtree after the worker's initial owner/closing checks but before its
        # registration tempfile is opened.
        shutil.rmtree(store)
        real_atomic_write(path, value)

    monkeypatch.setattr(lifecycle, "_atomic_write_json", remove_store_before_registration)
    with pytest.raises(FileNotFoundError):
        lifecycle.register_worker_store_lease(store)
    assert not store.exists()


def test_cleanup_store_normal_requires_all_worker_leases_closed(tmp_path: Path) -> None:
    root, store = _new_store(tmp_path)
    lifecycle.initialize_store_ownership(store, root)
    lease = lifecycle.register_worker_store_lease(store)
    with pytest.raises(lifecycle.SharedStoreLifecycleError, match="exclusive store lease"):
        lifecycle.cleanup_store(store, owner_died=False, timeout_seconds=0.0)
    assert store.is_dir()
    lease.close()
    lifecycle.cleanup_store(store, owner_died=False, timeout_seconds=1.0)
    assert not store.exists()


def test_parent_lifecycle_starts_separate_session_and_cleans_after_closed_worker(
    tmp_path: Path,
) -> None:
    root, store = _new_store(tmp_path)
    parent = lifecycle.ParentStoreLifecycle.start(store, root)
    try:
        assert parent.process.poll() is None
        assert os.getsid(parent.process.pid) == parent.process.pid
        assert not os.get_inheritable(parent._control_fd)
        assert not os.get_inheritable(parent._status_fd)
        lease = lifecycle.register_worker_store_lease(store)
        lease.close()
        parent.cleanup_after_workers(timeout_seconds=5.0)
        assert parent.process.returncode == 0
        assert not store.exists()
    finally:
        parent.abandon()


def test_parent_lifecycle_rejects_double_cleanup(tmp_path: Path) -> None:
    root, store = _new_store(tmp_path)
    parent = lifecycle.ParentStoreLifecycle.start(store, root)
    parent.cleanup_after_workers(timeout_seconds=5.0)
    with pytest.raises(lifecycle.SharedStoreLifecycleError, match="already closed"):
        parent.cleanup_after_workers(timeout_seconds=1.0)


def test_parent_cleanup_timeout_is_one_shot_and_releases_parent_fds(
    tmp_path: Path,
) -> None:
    root, store = _new_store(tmp_path)
    parent = lifecycle.ParentStoreLifecycle.start(
        store,
        root,
        cleanup_timeout_seconds=5.0,
    )
    lease = lifecycle.register_worker_store_lease(store)
    try:
        with pytest.raises(
            lifecycle.SharedStoreLifecycleError,
            match="shared-store janitor cleanup failed",
        ) as error:
            parent.cleanup_after_workers(timeout_seconds=0.01)
        assert "status read failed" in str(error.value)
        assert "janitor wait failed" in str(error.value)
        assert parent.closed is True
        assert parent._control_fd == -1
        assert parent._status_fd == -1
        assert parent.process.poll() is None
        assert store.is_dir()
        assert (store / lifecycle.CLOSING_FILENAME).is_file()

        with pytest.raises(lifecycle.SharedStoreLifecycleError, match="already closed"):
            parent.cleanup_after_workers(timeout_seconds=1.0)

        # The one valid command remains in force after the caller's deadline.
        # Once the conforming worker releases its shared lease, the janitor can
        # finish without a second command or any parent-side descriptors.
        lease.close()
        assert parent.process.wait(timeout=5.0) == 0
        assert not store.exists()
    finally:
        lease.close()
        parent.abandon()
        if parent.process.poll() is None:
            parent.process.kill()
        parent.process.wait(timeout=5.0)


def test_negative_parent_cleanup_timeout_does_not_consume_lifecycle(
    tmp_path: Path,
) -> None:
    root, store = _new_store(tmp_path)
    parent = lifecycle.ParentStoreLifecycle.start(store, root)
    try:
        with pytest.raises(ValueError, match="timeout_seconds must be >= 0"):
            parent.cleanup_after_workers(timeout_seconds=-0.01)
        assert parent.closed is False
        assert parent._control_fd >= 0
        assert parent._status_fd >= 0

        parent.cleanup_after_workers(timeout_seconds=5.0)
        assert parent.closed is True
        assert parent.process.returncode == 0
        assert not store.exists()
    finally:
        parent.abandon()


def test_parent_lifecycle_abort_terminates_registered_worker_only(
    tmp_path: Path,
) -> None:
    root, store = _new_store(tmp_path)
    parent = lifecycle.ParentStoreLifecycle.start(
        store,
        root,
        worker_term_grace_seconds=0.5,
    )
    ready = tmp_path / "registered-worker.json"
    worker_code = (
        "import json,time;"
        "from pathlib import Path;"
        "from steering.conditional.shared_store_lifecycle import "
        "register_worker_store_lease;"
        f"lease=register_worker_store_lease(Path({str(store)!r}));"
        f"Path({str(ready)!r}).write_text(json.dumps(lease.identity.as_dict()));"
        "time.sleep(300)"
    )
    worker = subprocess.Popen([sys.executable, "-c", worker_code])
    unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(300)"])
    try:
        _wait_for_path(ready, exists=True)
        identity = lifecycle.ProcessIdentity.from_dict(
            json.loads(ready.read_text()),
            surface="registered worker",
        )
        started = time.monotonic()
        parent.abort_after_worker_failure(timeout_seconds=5.0)
        elapsed = time.monotonic() - started

        _wait_for_identity_dead(identity)
        assert elapsed < 5.0
        assert parent.closed is True
        assert parent.process.returncode == 0
        assert not store.exists()
        assert unrelated.poll() is None
        with pytest.raises(lifecycle.SharedStoreLifecycleError, match="already closed"):
            parent.abort_after_worker_failure(timeout_seconds=1.0)
    finally:
        parent.abandon()
        for process in (worker, unrelated):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


def test_owner_death_cleanup_terminates_only_registered_exact_worker(tmp_path: Path) -> None:
    root, store = _new_store(tmp_path)
    lifecycle.initialize_store_ownership(store, root)
    ready = tmp_path / "worker-ready.json"
    code = (
        "import json,os,time;"
        "from pathlib import Path;"
        "from steering.conditional.shared_store_lifecycle import "
        "ProcessIdentity,register_worker_store_lease;"
        f"lease=register_worker_store_lease(Path({str(store)!r}));"
        f"Path({str(ready)!r}).write_text(json.dumps(lease.identity.as_dict()));"
        "time.sleep(300)"
    )
    worker = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    try:
        _wait_for_path(ready, exists=True)
        identity = lifecycle.ProcessIdentity.from_dict(
            json.loads(ready.read_text()), surface="test worker"
        )
        lifecycle.cleanup_store(
            store,
            owner_died=True,
            timeout_seconds=5.0,
            worker_term_grace_seconds=1.0,
        )
        _wait_for_identity_dead(identity)
        assert not store.exists()
        assert identity.pid == worker.pid
    finally:
        if worker.poll() is None:
            worker.kill()
        worker.wait(timeout=5)


def _launch_parent_with_worker(
    root: Path,
    store: Path,
    state_path: Path,
) -> subprocess.Popen[bytes]:
    worker_ready = state_path.with_suffix(".worker-ready")
    worker_code = (
        "import time;import torch as t;"
        "from pathlib import Path;"
        "from steering.conditional.shared_store_lifecycle import register_worker_store_lease;"
        "from steering.conditional.shared_stem_store import SharedStemStore;"
        f"lease=register_worker_store_lease(Path({str(store)!r}));"
        f"shared=SharedStemStore(Path({str(store)!r}),capacity=1);"
        "source=t.arange(262144,dtype=t.int32);"
        "assert shared.put('mapped',{'ids':source},{});"
        "held=shared.get('mapped')['ids'];"
        "assert t.equal(held,source);"
        f"Path({str(worker_ready)!r}).write_text('ready');"
        "time.sleep(300)"
    )
    parent_code = (
        "import json,subprocess,sys,time;"
        "from pathlib import Path;"
        "from steering.conditional.shared_store_lifecycle import "
        "ParentStoreLifecycle,ProcessIdentity;"
        f"manager=ParentStoreLifecycle.start(Path({str(store)!r}),Path({str(root)!r}));"
        f"worker=subprocess.Popen([sys.executable,'-c',{worker_code!r}]);"
        f"ready=Path({str(worker_ready)!r});"
        "deadline=time.monotonic()+10;"
        "\nwhile not ready.exists():\n"
        "  assert time.monotonic()<deadline\n"
        "  time.sleep(0.01)\n"
        f"Path({str(state_path)!r}).write_text(json.dumps({{"
        "'worker':ProcessIdentity.capture(worker.pid).as_dict(),"
        "'janitor':ProcessIdentity.capture(manager.process.pid).as_dict()}));"
        "time.sleep(300)"
    )
    return subprocess.Popen(
        [sys.executable, "-c", parent_code],
        start_new_session=True,
    )


@pytest.mark.parametrize(
    ("scope", "signum"),
    [
        ("parent", signal.SIGINT),
        ("parent", signal.SIGTERM),
        ("parent", signal.SIGKILL),
        ("group", signal.SIGINT),
        ("group", signal.SIGTERM),
        ("group", signal.SIGKILL),
    ],
)
def test_janitor_reclaims_after_parent_or_process_group_signal(
    tmp_path: Path,
    scope: str,
    signum: int,
) -> None:
    root, store = _new_store(tmp_path)
    state = tmp_path / "state.json"
    parent = _launch_parent_with_worker(root, store, state)
    identity: lifecycle.ProcessIdentity | None = None
    janitor_identity: lifecycle.ProcessIdentity | None = None
    try:
        _wait_for_path(state, exists=True, timeout=20.0)
        payload = json.loads(state.read_text())
        identity = lifecycle.ProcessIdentity.from_dict(payload["worker"], surface="state")
        janitor_identity = lifecycle.ProcessIdentity.from_dict(
            payload["janitor"], surface="janitor state"
        )
        assert identity.is_alive()
        assert os.getsid(janitor_identity.pid) == janitor_identity.pid
        if scope == "group":
            os.killpg(parent.pid, signum)
        else:
            os.kill(parent.pid, signum)
        parent.wait(timeout=10)
        _wait_for_path(store, exists=False, timeout=20.0)
        _wait_for_identity_dead(identity, timeout=20.0)
        _wait_for_identity_dead(janitor_identity, timeout=20.0)
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=5)
        if identity is not None and identity.is_alive():
            os.kill(identity.pid, signal.SIGKILL)
        if janitor_identity is not None and janitor_identity.is_alive():
            os.kill(janitor_identity.pid, signal.SIGKILL)


def test_sweeper_removes_exact_dead_owner_store(tmp_path: Path) -> None:
    root, store = _new_store(tmp_path)
    owner = lifecycle.initialize_store_ownership(store, root)
    dead = _dead_owner_for(owner)
    lifecycle._atomic_write_json(store / lifecycle.OWNER_FILENAME, dead.as_dict())

    result = lifecycle.sweep_orphaned_stores(root)

    assert result.removed == (str(store),)
    assert result.skipped_live == ()
    assert result.skipped_unsafe == ()
    assert not store.exists()


def test_sweeper_does_not_delete_live_owner(tmp_path: Path) -> None:
    root, store = _new_store(tmp_path)
    lifecycle.initialize_store_ownership(store, root)
    result = lifecycle.sweep_orphaned_stores(root)
    assert result.removed == ()
    assert result.skipped_live == (str(store),)
    assert result.skipped_unsafe == ()
    assert store.is_dir()


def test_sweeper_treats_pid_reuse_as_dead_when_starttime_differs(tmp_path: Path) -> None:
    root, store = _new_store(tmp_path)
    owner = lifecycle.initialize_store_ownership(store, root)
    reused = lifecycle.StoreOwner(
        owner=lifecycle.ProcessIdentity(
            boot_id=owner.owner.boot_id,
            pid=os.getpid(),
            starttime_ticks=owner.owner.starttime_ticks + 1,
        ),
        store_dir=owner.store_dir,
        store_root=owner.store_root,
        uid=owner.uid,
        created_time_ns=owner.created_time_ns,
    )
    lifecycle._atomic_write_json(store / lifecycle.OWNER_FILENAME, reused.as_dict())
    result = lifecycle.sweep_orphaned_stores(root)
    assert result.removed == (str(store),)
    assert not store.exists()


def test_sweeper_refuses_dead_owner_while_any_shared_lease_is_held(tmp_path: Path) -> None:
    root, store = _new_store(tmp_path)
    owner = lifecycle.initialize_store_ownership(store, root)
    lease = lifecycle.register_worker_store_lease(store)
    dead = _dead_owner_for(owner)
    lifecycle._atomic_write_json(store / lifecycle.OWNER_FILENAME, dead.as_dict())
    try:
        result = lifecycle.sweep_orphaned_stores(root)
        assert result.removed == ()
        assert result.skipped_unsafe == (str(store),)
        assert store.is_dir()
    finally:
        lease.close()
    result_after_close = lifecycle.sweep_orphaned_stores(root)
    assert result_after_close.removed == (str(store),)


def test_sweeper_refuses_legacy_pid_only_store_without_owner_schema(tmp_path: Path) -> None:
    root, store = _new_store(tmp_path)
    (store / "payload.bin").write_bytes(b"do not delete without proof")
    result = lifecycle.sweep_orphaned_stores(root)
    assert result.removed == ()
    assert result.skipped_unsafe == (str(store),)
    assert (store / "payload.bin").read_bytes() == b"do not delete without proof"


def test_sweeper_refuses_symlink_and_does_not_touch_target(tmp_path: Path) -> None:
    root = tmp_path / "shm"
    root.mkdir()
    target = tmp_path / "outside"
    target.mkdir()
    marker = target / "marker"
    marker.write_text("keep")
    link = root / f"{lifecycle.STORE_PREFIX}link"
    link.symlink_to(target, target_is_directory=True)
    result = lifecycle.sweep_orphaned_stores(root)
    assert result.removed == ()
    assert result.skipped_unsafe == (str(link),)
    assert link.is_symlink()
    assert marker.read_text() == "keep"


def test_malformed_worker_registry_prevents_owner_death_cleanup(tmp_path: Path) -> None:
    root, store = _new_store(tmp_path)
    lifecycle.initialize_store_ownership(store, root)
    bad = store / lifecycle.WORKERS_DIRNAME / "worker-bad.json"
    bad.write_text("not json")
    with pytest.raises(lifecycle.SharedStoreLifecycleError, match="cannot read"):
        lifecycle.cleanup_store(store, owner_died=True, timeout_seconds=0.1)
    assert store.is_dir()


def test_worker_registry_symlink_prevents_cleanup_without_following_target(tmp_path: Path) -> None:
    root, store = _new_store(tmp_path)
    lifecycle.initialize_store_ownership(store, root)
    outside = tmp_path / "outside.json"
    outside.write_text('{"important":true}')
    link = store / lifecycle.WORKERS_DIRNAME / "worker-link.json"
    link.symlink_to(outside)
    with pytest.raises(lifecycle.SharedStoreLifecycleError, match="unsafe non-file"):
        lifecycle.cleanup_store(store, owner_died=True, timeout_seconds=0.1)
    assert outside.read_text() == '{"important":true}'
    assert store.is_dir()
