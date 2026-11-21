"""Crash-safe ownership and cleanup for splice-preselection shared stores.

The parent process cannot rely on ``finally`` for SIGTERM, SIGKILL, or abrupt
death.  A janitor in a separate session owns cleanup authority.  Every worker
registers its exact Linux process identity and holds a shared ``flock`` before
using the store; cleanup first marks the store closing, then obtains the
exclusive lease.  Consequently a directory is never removed while a
conforming live worker can still create or retain a store mapping.

PID alone is not identity because Linux reuses PIDs.  Identity is the tuple
``(boot_id, pid, /proc/PID/stat starttime)``.  The janitor additionally watches
a pidfd where available and a control pipe whose only writer is the parent.
If every coordinator is SIGKILLed simultaneously, no user-space cleanup code
can run; the next invocation's lease-aware sweeper is the recovery path.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import json
import os
import select
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OWNER_SCHEMA_VERSION = 1
OWNER_FILENAME = "owner.json"
LEASE_FILENAME = "users.lease"
WORKERS_DIRNAME = "workers"
CLOSING_FILENAME = "closing"
STORE_PREFIX = "nanochat-sel-"
DEFAULT_READY_TIMEOUT_SECONDS = 10.0
DEFAULT_CLEANUP_TIMEOUT_SECONDS = 30.0
DEFAULT_WORKER_TERM_GRACE_SECONDS = 5.0

# Linux assigns these two modern process syscalls the same numbers on the
# architectures supported here. CPython exposes wrappers only when its build
# headers did, so an older userspace on a newer kernel (Ubuntu 22.04 in the
# production environment) needs the raw path. Unknown architectures fail
# closed rather than falling back to a PID-racy ``kill(2)``.
_PIDFD_SYSCALLS_BY_MACHINE: dict[str, tuple[int, int]] = {
    "aarch64": (434, 424),
    "arm64": (434, 424),
    "i386": (434, 424),
    "i686": (434, 424),
    "ppc64": (434, 424),
    "ppc64le": (434, 424),
    "riscv64": (434, 424),
    "s390x": (434, 424),
    "x86_64": (434, 424),
}
_LIBC = ctypes.CDLL(None, use_errno=True)


class SharedStoreLifecycleError(RuntimeError):
    """The ownership/lease protocol could not prove cleanup was safe."""


def _raw_linux_syscall(number: int, *args: object) -> int:
    ctypes.set_errno(0)
    result = _LIBC.syscall(ctypes.c_long(number), *args)
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(result)


def _pidfd_syscall_numbers() -> tuple[int, int]:
    machine = os.uname().machine.lower()
    try:
        return _PIDFD_SYSCALLS_BY_MACHINE[machine]
    except KeyError as exc:
        raise NotImplementedError(
            f"pidfd raw syscalls are not registered for architecture {machine!r}"
        ) from exc


def _pidfd_open(pid: int) -> int:
    native = getattr(os, "pidfd_open", None)
    if native is not None:
        return int(native(pid, 0))
    open_number, _send_number = _pidfd_syscall_numbers()
    return _raw_linux_syscall(
        open_number,
        ctypes.c_int(pid),
        ctypes.c_uint(0),
    )


def _pidfd_send_signal(pidfd: int, signum: int) -> None:
    native = getattr(signal, "pidfd_send_signal", None)
    if native is not None:
        native(pidfd, signum, None, 0)
        return
    _open_number, send_number = _pidfd_syscall_numbers()
    _raw_linux_syscall(
        send_number,
        ctypes.c_int(pidfd),
        ctypes.c_int(signum),
        ctypes.c_void_p(),
        ctypes.c_uint(0),
    )


def _read_boot_id(path: Path = Path("/proc/sys/kernel/random/boot_id")) -> str:
    try:
        value = path.read_text().strip()
    except OSError as exc:
        raise SharedStoreLifecycleError(f"cannot read Linux boot ID at {path}: {exc}") from exc
    if not value:
        raise SharedStoreLifecycleError(f"Linux boot ID at {path} is empty")
    return value


def _read_process_state_and_starttime(
    pid: int, proc_root: Path = Path("/proc")
) -> tuple[str, int]:
    path = proc_root / str(pid) / "stat"
    try:
        raw = path.read_text()
    except (FileNotFoundError, ProcessLookupError) as exc:
        raise ProcessLookupError(f"cannot read process identity from {path}: {exc}") from exc
    except OSError as exc:
        raise SharedStoreLifecycleError(
            f"cannot prove process identity from {path}: {exc}"
        ) from exc
    close_idx = raw.rfind(")")
    if close_idx < 0 or close_idx + 2 > len(raw):
        raise SharedStoreLifecycleError(f"malformed Linux process stat at {path}")
    tail = raw[close_idx + 2 :].split()
    if len(tail) < 20:
        raise SharedStoreLifecycleError(
            f"truncated Linux process stat at {path}: {len(tail)} fields after comm"
        )
    try:
        return tail[0], int(tail[19])
    except ValueError as exc:
        raise SharedStoreLifecycleError(
            f"non-integer process starttime in {path}: {tail[19]!r}"
        ) from exc


def _read_process_starttime(pid: int, proc_root: Path = Path("/proc")) -> int:
    return _read_process_state_and_starttime(pid, proc_root)[1]


@dataclass(frozen=True)
class ProcessIdentity:
    """One exact process instance within one Linux boot."""

    boot_id: str
    pid: int
    starttime_ticks: int

    @classmethod
    def capture(
        cls,
        pid: int,
        *,
        proc_root: Path = Path("/proc"),
        boot_id_path: Path = Path("/proc/sys/kernel/random/boot_id"),
    ) -> ProcessIdentity:
        if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
            raise ValueError(f"pid must be an int >= 1, got {pid!r}")
        return cls(
            boot_id=_read_boot_id(boot_id_path),
            pid=pid,
            starttime_ticks=_read_process_starttime(pid, proc_root),
        )

    @classmethod
    def from_dict(cls, value: Any, *, surface: str) -> ProcessIdentity:
        if not isinstance(value, dict):
            raise SharedStoreLifecycleError(f"{surface}: process identity must be an object")
        if set(value) != {"boot_id", "pid", "starttime_ticks"}:
            raise SharedStoreLifecycleError(
                f"{surface}: process identity keys are {sorted(value)}, expected "
                "['boot_id', 'pid', 'starttime_ticks']"
            )
        boot_id = value["boot_id"]
        pid = value["pid"]
        starttime = value["starttime_ticks"]
        if not isinstance(boot_id, str) or not boot_id:
            raise SharedStoreLifecycleError(f"{surface}: boot_id must be a non-empty string")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
            raise SharedStoreLifecycleError(f"{surface}: pid must be an int >= 1")
        if isinstance(starttime, bool) or not isinstance(starttime, int) or starttime < 0:
            raise SharedStoreLifecycleError(
                f"{surface}: starttime_ticks must be an int >= 0"
            )
        return cls(boot_id=boot_id, pid=pid, starttime_ticks=starttime)

    def as_dict(self) -> dict[str, str | int]:
        return {
            "boot_id": self.boot_id,
            "pid": self.pid,
            "starttime_ticks": self.starttime_ticks,
        }

    def is_alive(
        self,
        *,
        proc_root: Path = Path("/proc"),
        boot_id_path: Path = Path("/proc/sys/kernel/random/boot_id"),
    ) -> bool:
        try:
            boot_id = _read_boot_id(boot_id_path)
            state, starttime = _read_process_state_and_starttime(self.pid, proc_root)
        except ProcessLookupError:
            return False
        # A zombie has already released file descriptors, mappings, and flock
        # leases; waiting for its parent to reap /proc state would incorrectly
        # block cleanup even though it can no longer touch the store.
        if state in {"Z", "X", "x"}:
            return False
        return boot_id == self.boot_id and starttime == self.starttime_ticks


@dataclass(frozen=True)
class StoreOwner:
    """Versioned identity and path boundary for one mkdtemp store."""

    owner: ProcessIdentity
    store_dir: str
    store_root: str
    uid: int
    created_time_ns: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": OWNER_SCHEMA_VERSION,
            "owner": self.owner.as_dict(),
            "store_dir": self.store_dir,
            "store_root": self.store_root,
            "uid": self.uid,
            "created_time_ns": self.created_time_ns,
        }

    @classmethod
    def from_dict(cls, value: Any, *, surface: str) -> StoreOwner:
        if not isinstance(value, dict):
            raise SharedStoreLifecycleError(f"{surface}: owner metadata must be an object")
        expected = {
            "schema_version",
            "owner",
            "store_dir",
            "store_root",
            "uid",
            "created_time_ns",
        }
        if set(value) != expected:
            raise SharedStoreLifecycleError(
                f"{surface}: owner metadata keys are {sorted(value)}, expected {sorted(expected)}"
            )
        if value["schema_version"] != OWNER_SCHEMA_VERSION:
            raise SharedStoreLifecycleError(
                f"{surface}: schema_version={value['schema_version']!r}, "
                f"expected {OWNER_SCHEMA_VERSION}"
            )
        store_dir = value["store_dir"]
        store_root = value["store_root"]
        uid = value["uid"]
        created = value["created_time_ns"]
        if not isinstance(store_dir, str) or not os.path.isabs(store_dir):
            raise SharedStoreLifecycleError(f"{surface}: store_dir must be absolute")
        if not isinstance(store_root, str) or not os.path.isabs(store_root):
            raise SharedStoreLifecycleError(f"{surface}: store_root must be absolute")
        if isinstance(uid, bool) or not isinstance(uid, int) or uid < 0:
            raise SharedStoreLifecycleError(f"{surface}: uid must be an int >= 0")
        if isinstance(created, bool) or not isinstance(created, int) or created < 0:
            raise SharedStoreLifecycleError(
                f"{surface}: created_time_ns must be an int >= 0"
            )
        return cls(
            owner=ProcessIdentity.from_dict(value["owner"], surface=f"{surface}.owner"),
            store_dir=store_dir,
            store_root=store_root,
            uid=uid,
            created_time_ns=created,
        )


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    # Every lifecycle parent is created explicitly before this call. Never
    # recreate it here: cleanup may remove a store between a worker's initial
    # owner check and registration, and a recursive mkdir in that race would
    # resurrect an unowned store after the janitor had safely deleted it.
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w") as output:
            json.dump(value, output, sort_keys=True, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        _fsync_dir(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _read_json(path: Path, *, surface: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SharedStoreLifecycleError(f"{surface}: cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SharedStoreLifecycleError(f"{surface}: {path} must contain a JSON object")
    return value


def _validate_store_boundary(store_dir: Path, store_root: Path) -> tuple[Path, Path]:
    try:
        root = store_root.resolve(strict=True)
        store_lstat = store_dir.lstat()
        store = store_dir.resolve(strict=True)
    except OSError as exc:
        raise SharedStoreLifecycleError(
            f"cannot resolve shared-store boundary {store_dir} under {store_root}: {exc}"
        ) from exc
    if not stat.S_ISDIR(store_lstat.st_mode) or stat.S_ISLNK(store_lstat.st_mode):
        raise SharedStoreLifecycleError(f"shared store must be a real directory, got {store_dir}")
    if store.parent != root:
        raise SharedStoreLifecycleError(
            f"shared store {store} is not a direct child of configured root {root}"
        )
    if not store.name.startswith(STORE_PREFIX):
        raise SharedStoreLifecycleError(
            f"shared store name {store.name!r} lacks required prefix {STORE_PREFIX!r}"
        )
    return store, root


def initialize_store_ownership(
    store_dir: Path,
    store_root: Path,
) -> StoreOwner:
    """Publish owner metadata and an empty lease/registry in a fresh store."""
    store, root = _validate_store_boundary(store_dir, store_root)
    owner_path = store / OWNER_FILENAME
    lease_path = store / LEASE_FILENAME
    workers = store / WORKERS_DIRNAME
    if owner_path.exists() or lease_path.exists() or workers.exists():
        raise SharedStoreLifecycleError(
            f"fresh shared store {store} already contains lifecycle metadata"
        )
    workers.mkdir(mode=0o700)
    lease_fd = os.open(lease_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    os.close(lease_fd)
    owner = StoreOwner(
        owner=ProcessIdentity.capture(os.getpid()),
        store_dir=str(store),
        store_root=str(root),
        uid=os.getuid(),
        created_time_ns=time.time_ns(),
    )
    _atomic_write_json(owner_path, owner.as_dict())
    _fsync_dir(store)
    return owner


def read_store_owner(store_dir: Path) -> StoreOwner:
    owner_path = store_dir / OWNER_FILENAME
    return StoreOwner.from_dict(
        _read_json(owner_path, surface="shared-store owner"),
        surface=str(owner_path),
    )


def _assert_owner_matches_boundary(owner: StoreOwner, store_dir: Path, store_root: Path) -> None:
    store, root = _validate_store_boundary(store_dir, store_root)
    if owner.store_dir != str(store) or owner.store_root != str(root):
        raise SharedStoreLifecycleError(
            "shared-store owner path mismatch: "
            f"metadata=({owner.store_dir}, {owner.store_root}), actual=({store}, {root})"
        )
    if owner.uid != os.getuid():
        raise SharedStoreLifecycleError(
            f"shared-store owner uid={owner.uid}, current uid={os.getuid()}"
        )


class WorkerStoreLease:
    """A worker's registration and shared lock, held for process lifetime."""

    def __init__(self, store_dir: Path, identity: ProcessIdentity, fd: int, registration: Path):
        self.store_dir = store_dir
        self.identity = identity
        self._fd = fd
        self.registration = registration

    @property
    def closed(self) -> bool:
        return self._fd < 0

    def close(self) -> None:
        if self._fd < 0:
            return
        fd = self._fd
        self._fd = -1
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
        self.registration.unlink(missing_ok=True)

    def __enter__(self) -> WorkerStoreLease:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def register_worker_store_lease(store_dir: Path) -> WorkerStoreLease:
    """Register this worker and acquire a shared lease before any store use."""
    owner = read_store_owner(store_dir)
    root = Path(owner.store_root)
    _assert_owner_matches_boundary(owner, store_dir, root)
    if not owner.owner.is_alive():
        raise SharedStoreLifecycleError(
            f"shared-store parent {owner.owner.pid} is no longer the recorded process"
        )
    closing = store_dir / CLOSING_FILENAME
    if closing.exists():
        raise SharedStoreLifecycleError(f"shared store {store_dir} is already closing")

    identity = ProcessIdentity.capture(os.getpid())
    registration = (
        store_dir
        / WORKERS_DIRNAME
        / f"worker-{identity.pid}-{identity.starttime_ticks}.json"
    )
    _atomic_write_json(registration, {"schema_version": 1, "worker": identity.as_dict()})
    try:
        fd = os.open(store_dir / LEASE_FILENAME, os.O_RDONLY)
    except BaseException:
        registration.unlink(missing_ok=True)
        raise
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        owner_after_lock = read_store_owner(store_dir)
        if owner_after_lock != owner or closing.exists() or not owner.owner.is_alive():
            raise SharedStoreLifecycleError(
                f"shared store {store_dir} began closing or changed owner during worker registration"
            )
    except BaseException:
        os.close(fd)
        registration.unlink(missing_ok=True)
        raise
    return WorkerStoreLease(store_dir, identity, fd, registration)


def _mark_closing(store_dir: Path, reason: str) -> None:
    path = store_dir / CLOSING_FILENAME
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return
    with os.fdopen(fd, "w") as output:
        output.write(reason + "\n")
        output.flush()
        os.fsync(output.fileno())
    _fsync_dir(store_dir)


def _read_registered_workers(store_dir: Path) -> list[ProcessIdentity]:
    workers_dir = store_dir / WORKERS_DIRNAME
    identities: list[ProcessIdentity] = []
    try:
        entries = sorted(workers_dir.iterdir())
    except OSError as exc:
        raise SharedStoreLifecycleError(f"cannot enumerate worker registry {workers_dir}: {exc}") from exc
    for path in entries:
        if path.is_symlink() or not path.is_file():
            raise SharedStoreLifecycleError(f"unsafe non-file worker registration {path}")
        value = _read_json(path, surface="worker registration")
        if set(value) != {"schema_version", "worker"} or value["schema_version"] != 1:
            raise SharedStoreLifecycleError(f"invalid worker registration schema in {path}")
        identities.append(ProcessIdentity.from_dict(value["worker"], surface=str(path)))
    return identities


def _signal_matching_worker(identity: ProcessIdentity, signum: int) -> bool:
    """Signal only the process instance bound by an exact pidfd.

    Opening the pidfd before revalidating boot ID/start time closes the classic
    check-then-``kill(pid)`` reuse race: if the PID was already reused, the
    identity check fails; if it exits afterward, the pidfd remains bound to the
    dead original and cannot target the replacement.
    """
    try:
        pidfd = _pidfd_open(identity.pid)
    except ProcessLookupError:
        return False
    except NotImplementedError as exc:
        raise SharedStoreLifecycleError(
            "exact worker signaling requires pidfd support; refusing a "
            f"PID-racy fallback for worker {identity.pid}"
        ) from exc
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        raise SharedStoreLifecycleError(
            f"cannot open pidfd for exact worker identity {identity.pid}: {exc}"
        ) from exc
    try:
        if not identity.is_alive():
            return False
        try:
            _pidfd_send_signal(pidfd, signum)
        except ProcessLookupError:
            return False
        return True
    except NotImplementedError as exc:
        raise SharedStoreLifecycleError(
            "exact worker signaling requires pidfd support; refusing a "
            f"PID-racy fallback for worker {identity.pid}"
        ) from exc
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        raise SharedStoreLifecycleError(
            f"cannot signal exact worker identity {identity.pid} through pidfd: {exc}"
        ) from exc
    finally:
        os.close(pidfd)


def _terminate_registered_workers(store_dir: Path, grace_seconds: float) -> None:
    identities = _read_registered_workers(store_dir)
    for identity in identities:
        _signal_matching_worker(identity, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not any(identity.is_alive() for identity in identities):
            return
        time.sleep(0.05)
    for identity in identities:
        _signal_matching_worker(identity, signal.SIGKILL)
    kill_deadline = time.monotonic() + grace_seconds
    while time.monotonic() < kill_deadline:
        if not any(identity.is_alive() for identity in identities):
            return
        time.sleep(0.05)
    remaining = [identity.pid for identity in identities if identity.is_alive()]
    if remaining:
        raise SharedStoreLifecycleError(
            f"registered workers did not exit after SIGKILL: {remaining}"
        )


def _acquire_exclusive_lease(store_dir: Path, timeout_seconds: float) -> int:
    fd = os.open(store_dir / LEASE_FILENAME, os.O_RDONLY)
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            if time.monotonic() >= deadline:
                os.close(fd)
                raise SharedStoreLifecycleError(
                    f"timed out waiting for exclusive store lease at {store_dir}"
                ) from None
            time.sleep(0.05)


def cleanup_store(
    store_dir: Path,
    *,
    owner_died: bool,
    timeout_seconds: float = DEFAULT_CLEANUP_TIMEOUT_SECONDS,
    worker_term_grace_seconds: float = DEFAULT_WORKER_TERM_GRACE_SECONDS,
) -> None:
    """Close admission, stop orphan workers if needed, lease, and remove."""
    owner = read_store_owner(store_dir)
    root = Path(owner.store_root)
    _assert_owner_matches_boundary(owner, store_dir, root)
    _mark_closing(store_dir, "parent-death" if owner_died else "normal-cleanup")
    if owner_died:
        _terminate_registered_workers(store_dir, worker_term_grace_seconds)
    lease_fd = _acquire_exclusive_lease(store_dir, timeout_seconds)
    try:
        shutil.rmtree(store_dir)
    finally:
        fcntl.flock(lease_fd, fcntl.LOCK_UN)
        os.close(lease_fd)
    if store_dir.exists() or store_dir.is_symlink():
        raise SharedStoreLifecycleError(f"shared store still exists after cleanup: {store_dir}")


def _status_write(fd: int, message: str) -> None:
    try:
        os.write(fd, (message + "\n").encode())
    except BrokenPipeError:
        pass


def _wait_for_parent_event(control_fd: int, parent: ProcessIdentity) -> bool:
    """Return whether cleanup must terminate registered workers.

    Parent death/EOF and an explicit ``ABORT`` mean the worker pool cannot be
    trusted to exit on its own. ``CLEANUP`` is sent only after a normal pool
    shutdown, so it takes the lease without signaling workers.
    """
    poller = select.poll()
    poller.register(control_fd, select.POLLIN | select.POLLHUP | select.POLLERR)
    pidfd = -1
    try:
        pidfd = _pidfd_open(parent.pid)
    except (OSError, NotImplementedError):
        # Observation has a safe exact fallback: poll the full boot/PID/start
        # identity below. Signaling deliberately has no analogous PID fallback.
        pidfd = -1
    if pidfd >= 0:
        if not parent.is_alive():
            os.close(pidfd)
            return True
        poller.register(pidfd, select.POLLIN | select.POLLHUP | select.POLLERR)
    try:
        while True:
            for fd, event in poller.poll(1000):
                if fd == pidfd and event:
                    return True
                if fd != control_fd:
                    continue
                data = os.read(control_fd, 4096)
                if not data:
                    return True
                if data.strip() == b"CLEANUP":
                    return False
                if data.strip() == b"ABORT":
                    return True
                raise SharedStoreLifecycleError(
                    f"janitor received invalid control command {data!r}"
                )
            if not parent.is_alive():
                return True
    finally:
        if pidfd >= 0:
            os.close(pidfd)


def run_janitor(
    store_dir: Path,
    control_fd: int,
    status_fd: int,
    *,
    cleanup_timeout_seconds: float = DEFAULT_CLEANUP_TIMEOUT_SECONDS,
    worker_term_grace_seconds: float = DEFAULT_WORKER_TERM_GRACE_SECONDS,
) -> int:
    """Janitor subprocess entry point; returns a process exit code."""
    try:
        owner = read_store_owner(store_dir)
        _assert_owner_matches_boundary(owner, store_dir, Path(owner.store_root))
        _status_write(status_fd, "READY")
        owner_died = _wait_for_parent_event(control_fd, owner.owner)
        cleanup_store(
            store_dir,
            owner_died=owner_died,
            timeout_seconds=cleanup_timeout_seconds,
            worker_term_grace_seconds=worker_term_grace_seconds,
        )
        _status_write(status_fd, "CLEANED")
        return 0
    except BaseException as exc:
        _status_write(status_fd, f"ERROR {type(exc).__name__}: {exc}")
        return 2
    finally:
        os.close(control_fd)
        os.close(status_fd)


def _read_status_line(fd: int, timeout_seconds: float) -> str:
    ready, _, _ = select.select([fd], [], [], timeout_seconds)
    if not ready:
        raise SharedStoreLifecycleError(
            f"timed out after {timeout_seconds}s waiting for shared-store janitor"
        )
    data = os.read(fd, 4096)
    if not data:
        raise SharedStoreLifecycleError("shared-store janitor closed status pipe without a result")
    return data.decode().strip()


class ParentStoreLifecycle:
    """Parent-side janitor control. Start before creating the worker pool."""

    def __init__(
        self,
        store_dir: Path,
        process: subprocess.Popen[bytes],
        control_fd: int,
        status_fd: int,
    ) -> None:
        self.store_dir = store_dir
        self.process = process
        self._control_fd = control_fd
        self._status_fd = status_fd
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @classmethod
    def start(
        cls,
        store_dir: Path,
        store_root: Path,
        *,
        ready_timeout_seconds: float = DEFAULT_READY_TIMEOUT_SECONDS,
        cleanup_timeout_seconds: float = DEFAULT_CLEANUP_TIMEOUT_SECONDS,
        worker_term_grace_seconds: float = DEFAULT_WORKER_TERM_GRACE_SECONDS,
    ) -> ParentStoreLifecycle:
        initialize_store_ownership(store_dir, store_root)
        control_read, control_write = os.pipe2(os.O_CLOEXEC)
        status_read, status_write = os.pipe2(os.O_CLOEXEC)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "janitor",
            "--store-dir",
            str(store_dir),
            "--control-fd",
            str(control_read),
            "--status-fd",
            str(status_write),
            "--cleanup-timeout-seconds",
            str(cleanup_timeout_seconds),
            "--worker-term-grace-seconds",
            str(worker_term_grace_seconds),
        ]
        try:
            process = subprocess.Popen(
                command,
                pass_fds=(control_read, status_write),
                start_new_session=True,
            )
        except BaseException:
            for fd in (control_read, control_write, status_read, status_write):
                os.close(fd)
            raise
        os.close(control_read)
        os.close(status_write)
        lifecycle = cls(store_dir, process, control_write, status_read)
        try:
            status = _read_status_line(status_read, ready_timeout_seconds)
            if status != "READY":
                raise SharedStoreLifecycleError(
                    f"shared-store janitor failed startup: {status}"
                )
        except BaseException:
            lifecycle.abandon()
            raise
        return lifecycle

    def cleanup_after_workers(
        self,
        *,
        timeout_seconds: float = DEFAULT_CLEANUP_TIMEOUT_SECONDS,
    ) -> None:
        """Remove the store after a normally closed worker pool."""
        self._request_cleanup(b"CLEANUP\n", timeout_seconds=timeout_seconds)

    def abort_after_worker_failure(
        self,
        *,
        timeout_seconds: float = DEFAULT_CLEANUP_TIMEOUT_SECONDS,
    ) -> None:
        """Terminate exact registered workers, lease, and remove the store."""
        self._request_cleanup(b"ABORT\n", timeout_seconds=timeout_seconds)

    def _request_cleanup(
        self,
        command: bytes,
        *,
        timeout_seconds: float,
    ) -> None:
        if self._closed:
            raise SharedStoreLifecycleError("shared-store lifecycle is already closed")
        if timeout_seconds < 0:
            raise ValueError(
                f"cleanup timeout_seconds must be >= 0, got {timeout_seconds}"
            )
        # A cleanup command is one-shot. Mark the controller closed before any
        # I/O so a timeout or broken pipe cannot provoke an invalid second
        # command through already-closed descriptors in an outer ``finally``.
        self._closed = True
        deadline = time.monotonic() + timeout_seconds
        failures: list[str] = []
        try:
            os.write(self._control_fd, command)
        except BrokenPipeError:
            pass
        except OSError as exc:
            failures.append(f"control write failed: {type(exc).__name__}: {exc}")
        finally:
            if self._control_fd >= 0:
                os.close(self._control_fd)
                self._control_fd = -1

        status: str | None = None
        try:
            status = _read_status_line(
                self._status_fd,
                max(0.0, deadline - time.monotonic()),
            )
        except Exception as exc:
            failures.append(f"status read failed: {type(exc).__name__}: {exc}")
        finally:
            if self._status_fd >= 0:
                os.close(self._status_fd)
                self._status_fd = -1

        returncode: int | None = None
        try:
            returncode = self.process.wait(
                timeout=max(0.0, deadline - time.monotonic())
            )
        except Exception as exc:
            failures.append(f"janitor wait failed: {type(exc).__name__}: {exc}")
        if failures or status != "CLEANED" or returncode != 0:
            raise SharedStoreLifecycleError(
                "shared-store janitor cleanup failed: "
                f"status={status!r}, returncode={returncode}, failures={failures}"
            )

    def abandon(self) -> None:
        """Close parent pipe ends, allowing the janitor to observe parent-side EOF."""
        if self._closed:
            return
        for attribute in ("_control_fd", "_status_fd"):
            fd = getattr(self, attribute)
            if fd >= 0:
                os.close(fd)
                setattr(self, attribute, -1)
        self._closed = True


@dataclass(frozen=True)
class SweepResult:
    removed: tuple[str, ...]
    skipped_live: tuple[str, ...]
    skipped_unsafe: tuple[str, ...]


def sweep_orphaned_stores(store_root: Path) -> SweepResult:
    """Lease-aware next-run recovery; never reaps legacy/PID-only stores."""
    root = store_root.resolve(strict=True)
    removed: list[str] = []
    skipped_live: list[str] = []
    skipped_unsafe: list[str] = []
    for path in sorted(root.iterdir()):
        if not path.name.startswith(STORE_PREFIX):
            continue
        try:
            owner = read_store_owner(path)
            _assert_owner_matches_boundary(owner, path, root)
        except SharedStoreLifecycleError:
            skipped_unsafe.append(str(path))
            continue
        if owner.owner.is_alive():
            skipped_live.append(str(path))
            continue
        _mark_closing(path, "startup-sweep")
        try:
            lease_fd = _acquire_exclusive_lease(path, 0.0)
        except SharedStoreLifecycleError:
            skipped_unsafe.append(str(path))
            continue
        try:
            shutil.rmtree(path)
        finally:
            fcntl.flock(lease_fd, fcntl.LOCK_UN)
            os.close(lease_fd)
        removed.append(str(path))
    return SweepResult(
        removed=tuple(removed),
        skipped_live=tuple(skipped_live),
        skipped_unsafe=tuple(skipped_unsafe),
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    janitor = subparsers.add_parser("janitor")
    janitor.add_argument("--store-dir", type=Path, required=True)
    janitor.add_argument("--control-fd", type=int, required=True)
    janitor.add_argument("--status-fd", type=int, required=True)
    janitor.add_argument("--cleanup-timeout-seconds", type=float, required=True)
    janitor.add_argument("--worker-term-grace-seconds", type=float, required=True)
    args = parser.parse_args()
    if args.command == "janitor":
        return run_janitor(
            args.store_dir,
            args.control_fd,
            args.status_fd,
            cleanup_timeout_seconds=args.cleanup_timeout_seconds,
            worker_term_grace_seconds=args.worker_term_grace_seconds,
        )
    raise AssertionError(f"unreachable command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(_main())
