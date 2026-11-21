from __future__ import annotations

import fcntl
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from beartype.roar import BeartypeCallHintParamViolation

import steering.remote_body_prefetch as prefetch
from steering.conditional.encoder_decoder.dataloader import ClosableBatchPrefetcher
from steering.refined_types import PositiveInt
from steering.remote_body_prefetch import (
    RemoteBodyBinding,
    RemoteBodyEntry,
    RemoteBodyFamily,
    RemoteBodyPrefetcher,
    _WorkerConfig,
)
from steering.remote_training import (
    RemoteMaterializationCancelled,
    RemotePackedRowMaterializer,
    RemoteSelectionMaterializer,
    RemoteTrainingConfig,
    RemoteTrainingError,
)


def _payload(kind: str, index: int) -> bytes:
    return f"{kind}-validation-body-{index}".encode()


def _entry(kind: str, index: int) -> RemoteBodyEntry:
    name = f"shard_{index:05d}.safetensors.zst"
    payload = _payload(kind, index)
    fingerprint = None
    if kind == "packed":
        digest = hashlib.sha256()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(payload)
        fingerprint = digest.hexdigest()
    return RemoteBodyEntry(
        name=name,
        byte_length=len(payload),
        content_sha256=hashlib.sha256(payload).hexdigest(),
        file_fingerprint=fingerprint,
    )


def _reader_entry(entry: RemoteBodyEntry) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": entry.name,
        "byte_length": entry.byte_length,
        "content_sha256": entry.content_sha256,
    }
    if entry.file_fingerprint is not None:
        result["file_fingerprint"] = entry.file_fingerprint
    return result


def _family(
    root: Path,
    *,
    kind: str = "selection",
    n_entries: int = 2,
    family_char: str = "1",
    aggregate_char: str = "a",
    schedule_id: str = "c" * 64,
    train_schedule: bool = True,
) -> RemoteBodyFamily:
    local_dir = root / f"{kind}-{family_char}"
    local_dir.mkdir(parents=True, exist_ok=True)
    return RemoteBodyFamily(
        family_id=family_char * 64,
        family_aggregate=aggregate_char * 64,
        kind=cast(Any, kind),
        local_dir=str(local_dir),
        remote_dir=f"features/{kind}-{family_char}",
        entries=tuple(_entry(kind, index) for index in range(n_entries)),
        preexisting_descriptors=(),
        schedule_id=schedule_id,
        train_schedule=train_schedule,
    )


def _cfg(
    cache_root: Path,
    families: tuple[RemoteBodyFamily, ...],
    *,
    rank: int = 0,
    world_size: int = 1,
    nonce: str = "d" * 32,
) -> _WorkerConfig:
    return _WorkerConfig(
        bucket_id="owner/bucket",
        cache_root=str(cache_root),
        nonce=nonce,
        rank=rank,
        world_size=world_size,
        lookahead_shards_per_rank=2,
        cache_shards=2,
        timeout_seconds=1,
        retry_interval_seconds=0.001,
        families=families,
    )


def _selection_materializer(
    local_dir: Path,
    *,
    aggregate: str = "a" * 64,
    remote_dir: str = "features/selection",
    entries: tuple[RemoteBodyEntry, ...] | None = None,
) -> RemoteSelectionMaterializer:
    signed = entries if entries is not None else (_entry("selection", 0),)
    return RemoteSelectionMaterializer(
        local_dir=local_dir,
        remote_dir=remote_dir,
        inventory={entry.name: _reader_entry(entry) for entry in signed},
        loader_certificate={"selection_inventory_aggregate": aggregate},
        identity_contract={},
        timeout_seconds=1,
        retry_interval_seconds=0.001,
        trust_existing_cache=False,
    )


def _packed_materializer(
    local_dir: Path,
    *,
    entries: tuple[RemoteBodyEntry, ...],
) -> RemotePackedRowMaterializer:
    return RemotePackedRowMaterializer(
        local_dir=local_dir,
        remote_dir="features/packed",
        inventory={
            "aggregate_sha256": "b" * 64,
            "packed_manifest_file_fingerprint": "c" * 64,
            "packed_row_fingerprint": "e" * 64,
            "provenance_fingerprint": "f" * 64,
            "shards": [
                {
                    **_reader_entry(entry),
                    "total_segments": 1,
                }
                for entry in entries
            ],
        },
        cache_shards=2,
        base_shard_stems=[],
        provenance_shard_paths=[],
        timeout_seconds=1,
        retry_interval_seconds=0.001,
        trust_existing_cache=False,
    )


class _Process:
    def __init__(
        self,
        *,
        alive: bool = False,
        exitcode: int | None = None,
        survive_terminate: bool = False,
    ) -> None:
        self.pid = 4242
        self._alive = alive
        self._exitcode = exitcode
        self.survive_terminate = survive_terminate
        self.join_calls: list[float | None] = []
        self.terminate_calls = 0

    @property
    def exitcode(self) -> int | None:
        return None if self._alive else self._exitcode

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.terminate_calls += 1
        if not self.survive_terminate:
            self._alive = False
            self._exitcode = -15


class _Pipe:
    def __init__(
        self,
        messages: tuple[object, ...] = (),
        *,
        send_error: BaseException | None = None,
    ) -> None:
        self.messages = deque(messages)
        self.send_error = send_error
        self.sent: list[object] = []
        self.closed = False

    def poll(self, timeout: float | None = None) -> bool:
        del timeout
        return bool(self.messages)

    def recv(self) -> object:
        return self.messages.popleft()

    def send(self, value: object) -> None:
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(value)

    def close(self) -> None:
        self.closed = True


class _StartPipe(_Pipe):
    def __init__(
        self,
        messages: tuple[object, ...] = (),
        *,
        send_error: BaseException | None = None,
        poll_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        super().__init__(messages, send_error=send_error)
        self.poll_error = poll_error
        self.close_error = close_error

    def poll(self, timeout: float | None = None) -> bool:
        if self.poll_error is not None:
            raise self.poll_error
        return super().poll(timeout)

    def close(self) -> None:
        super().close()
        if self.close_error is not None:
            raise self.close_error


class _StartProcess(_Process):
    def __init__(
        self,
        *,
        alive: bool,
        exitcode: int | None = None,
        survive_terminate: bool = False,
        join_error: BaseException | None = None,
        on_start: Callable[[_StartProcess], None] | None = None,
    ) -> None:
        super().__init__(
            alive=alive,
            exitcode=exitcode,
            survive_terminate=survive_terminate,
        )
        self.join_error = join_error
        self.on_start = on_start
        self.args: tuple[Any, ...] = ()
        self.started = False

    def start(self) -> None:
        self.started = True
        if self.on_start is not None:
            self.on_start(self)

    def join(self, timeout: float | None = None) -> None:
        super().join(timeout)
        if self.join_error is not None:
            raise self.join_error


class _StartContext:
    def __init__(
        self,
        *,
        process: _StartProcess,
        child_command: _StartPipe | None = None,
        parent_command: _StartPipe | None = None,
        parent_status: _StartPipe | None = None,
        child_status: _StartPipe | None = None,
    ) -> None:
        self.process = process
        self.child_command = child_command or _StartPipe()
        self.parent_command = parent_command or _StartPipe()
        self.parent_status = parent_status or _StartPipe()
        self.child_status = child_status or _StartPipe()
        self._pipe_calls = 0

    def Pipe(self, *, duplex: bool) -> tuple[_StartPipe, _StartPipe]:
        assert duplex is False
        self._pipe_calls += 1
        if self._pipe_calls == 1:
            return self.child_command, self.parent_command
        if self._pipe_calls == 2:
            return self.parent_status, self.child_status
        raise AssertionError("unexpected third startup pipe")

    def Process(self, **kwargs: Any) -> _StartProcess:
        assert kwargs["daemon"] is False
        self.process.args = tuple(kwargs["args"])
        return self.process


def _remote_config(*, wait_timeout: float = 0.02) -> RemoteTrainingConfig:
    remote = RemoteTrainingConfig.parse(
        {
            "bucket_id": "owner/bucket",
            "body_prefetch": {"lookahead_shards_per_rank": 2},
            "selection_prefix": "features",
            "checkpoint_prefix": "checkpoints",
            "selection_wait_timeout_seconds": wait_timeout,
            "checkpoint_retry_timeout_seconds": 1,
            "retry_interval_seconds": 0.001,
            "resume_latest": False,
        }
    )
    assert remote is not None
    return remote


def _controller(
    cache_root: Path,
    *,
    families: tuple[RemoteBodyFamily, ...] | None = None,
    process: _Process | None = None,
    command_pipe: _Pipe | None = None,
    status_pipe: _Pipe | None = None,
    packed_binding: RemoteBodyBinding | None = None,
    remote: RemoteTrainingConfig | None = None,
    session_lock_fd: int | None = None,
    world_size: int = 1,
) -> tuple[RemoteBodyPrefetcher, RemoteBodyBinding]:
    configured_families = families if families is not None else (_family(cache_root, n_entries=2),)
    train_family = next(family for family in configured_families if family.train_schedule)
    train_binding = RemoteBodyBinding(
        family_id=train_family.family_id,
        schedule_id=train_family.schedule_id,
        kind=train_family.kind,
        index_by_name=tuple(
            (entry.name, index) for index, entry in enumerate(train_family.entries)
        ),
        paired=packed_binding is not None,
        train_schedule=True,
    )
    bindings = {
        str(Path(family.local_dir).resolve()): RemoteBodyBinding(
            family_id=family.family_id,
            schedule_id=family.schedule_id,
            kind=family.kind,
            index_by_name=tuple((entry.name, index) for index, entry in enumerate(family.entries)),
            paired=(
                family.schedule_id == train_binding.schedule_id
                and len(prefetch._schedule_map(configured_families)[family.schedule_id]) == 2
            ),
            train_schedule=family.train_schedule,
        )
        for family in configured_families
        if family.kind == "selection"
    }
    nonce = "d" * 32
    control = prefetch._create_control_tree(
        cache_root=cache_root,
        nonce=nonce,
        world_size=world_size,
        families=configured_families,
    )
    controller = RemoteBodyPrefetcher(
        remote=remote or _remote_config(),
        cache_root=cache_root,
        nonce=nonce,
        rank=0,
        world_size=world_size,
        families=configured_families,
        bindings=bindings,
        packed_binding=packed_binding,
        train_binding=train_binding,
        control_root=control,
        process=cast(Any, process or _Process()),
        command_pipe=cast(Any, command_pipe or _Pipe()),
        status_pipe=cast(Any, status_pipe or _Pipe()),
        session_lock_fd=session_lock_fd,
        admission=prefetch.calculate_remote_prefetch_disk_admission(
            cache_root=cache_root,
            families=configured_families,
            world_size=world_size,
            cache_shards=1,
            lookahead_shards_per_rank=2,
            safety_reserve_bytes=0,
        ),
    )
    return controller, train_binding


def _write_controller_status(
    controller: RemoteBodyPrefetcher,
    *,
    owner_rank: int,
    state: str = "idle",
    error: BaseException | None = None,
) -> None:
    family = controller.families[0]
    prefetch._write_worker_status(
        replace(controller._worker_cfg, rank=owner_rank),
        state=state,
        family_id=family.family_id if error is not None else None,
        inventory_index=0 if error is not None else None,
        body_name=family.entries[0].name if error is not None else None,
        error=error,
    )


class _FakeMonotonicClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class _TimeProxy:
    """Override this module's clock without mutating process-global `time`."""

    def __init__(self, monotonic: Callable[[], float]) -> None:
        self.monotonic = monotonic

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


def _patch_prefetch_monotonic(
    monkeypatch: pytest.MonkeyPatch,
    monotonic: Callable[[], float],
) -> None:
    monkeypatch.setattr(prefetch, "time", _TimeProxy(monotonic))


@pytest.mark.parametrize(
    ("operation", "error_type"),
    [
        pytest.param(
            lambda family, binding: family.entry(False),
            BeartypeCallHintParamViolation,
            id="family-index-bool",
        ),
        pytest.param(
            lambda family, binding: binding.inventory_index("missing"),
            RemoteTrainingError,
            id="binding-name",
        ),
        pytest.param(
            lambda family, binding: prefetch.owner_for_inventory_index(False, 1),
            BeartypeCallHintParamViolation,
            id="owner-index-bool",
        ),
        pytest.param(
            lambda family, binding: prefetch.owner_for_inventory_index(0, False),
            BeartypeCallHintParamViolation,
            id="owner-world-bool",
        ),
        pytest.param(
            lambda family, binding: prefetch._require_sha256(
                "not-a-digest",
                surface="test digest",
            ),
            RemoteTrainingError,
            id="sha256",
        ),
        pytest.param(
            lambda family, binding: prefetch._descriptor_dict_to_tuple(
                {"device": True},
            ),
            RemoteTrainingError,
            id="descriptor-dict",
        ),
        pytest.param(
            lambda family, binding: prefetch._descriptor_tuple_to_dict(
                (1, 2, 3, 4, 5, 6, True),
            ),
            RemoteTrainingError,
            id="descriptor-tuple",
        ),
    ],
)
def test_public_identity_helpers_reject_non_strict_values(
    tmp_path: Path,
    operation: Callable[[RemoteBodyFamily, RemoteBodyBinding], object],
    error_type: type[Exception],
) -> None:
    family = _family(tmp_path)
    binding = RemoteBodyBinding(
        family_id=family.family_id,
        schedule_id=family.schedule_id,
        kind="selection",
        index_by_name=((family.entries[0].name, 0),),
        paired=False,
        train_schedule=True,
    )

    with pytest.raises(error_type):
        operation(family, binding)


def test_direct_directory_and_file_guards_reject_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(RemoteTrainingError, match="directory is unavailable"):
        prefetch._ensure_direct_directory(missing)

    direct = tmp_path / "direct"
    direct.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(direct, target_is_directory=True)
    with pytest.raises(RemoteTrainingError, match="direct directory"):
        prefetch._ensure_direct_directory(alias)

    with pytest.raises(RemoteTrainingError, match="cannot open"):
        prefetch._direct_file_stat(missing)

    regular = tmp_path / "regular"
    regular.write_bytes(b"body")
    hardlink = tmp_path / "hardlink"
    os.link(regular, hardlink)
    with pytest.raises(RemoteTrainingError, match="exactly one hard link"):
        prefetch._direct_file_stat(regular)

    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "fifo"
        os.mkfifo(fifo)
        with pytest.raises(RemoteTrainingError, match="direct regular file"):
            prefetch._direct_file_stat(fifo)

    with monkeypatch.context() as scoped:
        scoped.delattr(prefetch.os, "O_NOFOLLOW")
        with pytest.raises(RemoteTrainingError, match="O_NOFOLLOW"):
            prefetch._direct_file_stat(regular)
        with pytest.raises(RemoteTrainingError, match="O_NOFOLLOW"):
            prefetch._read_direct_bytes(regular, max_bytes=10)


def test_direct_file_stat_detects_replacement_during_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "record"
    replacement = tmp_path / "replacement"
    path.write_bytes(b"old")
    replacement.write_bytes(b"new")
    original_open = os.open

    def replacing_open(target: Any, *args: Any, **kwargs: Any) -> int:
        descriptor = original_open(target, *args, **kwargs)
        if Path(target) == path:
            os.replace(replacement, path)
        return descriptor

    monkeypatch.setattr(prefetch.os, "open", replacing_open)
    with pytest.raises(RemoteTrainingError, match="changed while opening"):
        prefetch._direct_file_stat(path)


def test_safe_tree_removal_is_exact(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    body = cache / "body"
    body.write_bytes(b"signed")

    stale = tmp_path / "stale"
    nested = stale / "nested"
    nested.mkdir(parents=True)
    (stale / "record").write_bytes(b"one")
    (nested / "record").write_bytes(b"two")
    prefetch._safe_remove_tree(stale)
    assert not stale.exists()

    unsafe = tmp_path / "unsafe"
    unsafe.mkdir()
    (unsafe / "alias").symlink_to(body)
    with pytest.raises(RemoteTrainingError, match="unsafe stale"):
        prefetch._safe_remove_tree(unsafe)


def test_direct_record_reader_rejects_size_mutation_and_io_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "record"
    path.write_bytes(b"0123456789")
    with pytest.raises(RemoteTrainingError, match="cannot open"):
        prefetch._read_direct_bytes(tmp_path / "missing", max_bytes=5)
    with pytest.raises(RemoteTrainingError, match="exceeds 5 bytes"):
        prefetch._read_direct_bytes(path, max_bytes=5)

    original_fstat = os.fstat
    fstat_calls = 0

    def mutate_before_second_fstat(descriptor: int):
        nonlocal fstat_calls
        fstat_calls += 1
        if fstat_calls == 2:
            path.write_bytes(b"abcdefghij")
        return original_fstat(descriptor)

    with monkeypatch.context() as scoped:
        scoped.setattr(prefetch.os, "fstat", mutate_before_second_fstat)
        with pytest.raises(RemoteTrainingError, match="changed while reading"):
            prefetch._read_direct_bytes(path, max_bytes=20)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            prefetch.os,
            "fdopen",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("fdopen failed")),
        )
        with pytest.raises(RemoteTrainingError, match="cannot read"):
            prefetch._read_direct_bytes(path, max_bytes=20)


def test_direct_record_reader_reaches_opened_link_identity_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "record"
    anchor = tmp_path / "anchor"
    replacement = tmp_path / "replacement"
    path.write_bytes(b"old")
    os.link(path, anchor)
    replacement.write_bytes(b"new")
    original_open = os.open

    def replacing_open(target: Any, *args: Any, **kwargs: Any) -> int:
        descriptor = original_open(target, *args, **kwargs)
        if Path(target) == path:
            os.replace(replacement, path)
        return descriptor

    monkeypatch.setattr(prefetch.os, "open", replacing_open)
    with pytest.raises(RemoteTrainingError, match="changed while opening"):
        prefetch._read_direct_bytes(path, max_bytes=10)


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        pytest.param(b"\xff", "valid UTF-8 JSON", id="utf8"),
        pytest.param(b"{", "valid UTF-8 JSON", id="json"),
        pytest.param(b"[]", "contain exactly", id="non-object"),
        pytest.param(b'{"schema_version":1,"extra":0}', "contain exactly", id="keys"),
        pytest.param(b'{"schema_version":true}', "schema_version", id="bool-schema"),
    ],
)
def test_json_record_schema_is_fail_loud(
    tmp_path: Path,
    raw: bytes,
    match: str,
) -> None:
    path = tmp_path / "record.json"
    path.write_bytes(raw)
    with pytest.raises(RemoteTrainingError, match=match):
        prefetch._load_json_record(
            path,
            expected_keys=frozenset({"schema_version"}),
            surface="test record",
        )


@pytest.mark.parametrize(
    "replacement_point",
    [
        pytest.param("after-open", id="after-open"),
        pytest.param("after-first-stat", id="after-first-stat"),
        pytest.param("after-read", id="after-read"),
    ],
)
def test_status_reader_accepts_coherent_snapshot_during_atomic_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_point: str,
) -> None:
    path = tmp_path / "rank_00000.json"
    nonce = "a" * 32
    old_status = {
        "schema_version": prefetch.REMOTE_PREFETCH_SCHEMA_VERSION,
        "nonce": nonce,
        "rank": 0,
        "pid": os.getpid(),
        "state": "starting",
        "family_id": None,
        "inventory_index": None,
        "body_name": None,
        "updated_ns": 1,
        "error_type": None,
        "error_message": None,
        "traceback": None,
    }
    new_status = dict(old_status, state="idle", updated_ns=2)
    prefetch._atomic_record(old_status, path)

    original_open = os.open
    original_fstat = os.fstat
    replaced = False
    fstat_calls = 0

    def replace_after_open(
        target: Any,
        *args: Any,
        **kwargs: Any,
    ) -> int:
        nonlocal replaced
        descriptor = original_open(target, *args, **kwargs)
        if Path(target) == path and not replaced:
            replaced = True
            prefetch._atomic_record(new_status, path)
        return descriptor

    def replace_while_descriptor_is_pinned(descriptor: int) -> os.stat_result:
        nonlocal fstat_calls, replaced
        fstat_calls += 1
        opened = original_fstat(descriptor)
        expected_call = 1 if replacement_point == "after-first-stat" else 2
        if fstat_calls == expected_call:
            replaced = True
            prefetch._atomic_record(new_status, path)
        return opened

    with monkeypatch.context() as scoped:
        if replacement_point == "after-open":
            scoped.setattr(prefetch.os, "open", replace_after_open)
        else:
            scoped.setattr(prefetch.os, "fstat", replace_while_descriptor_is_pinned)
        observed = prefetch._load_status(path, nonce=nonce, world_size=1)

    # The opened old inode is one complete atomic generation, even though the
    # pathname names the new inode by the time the descriptor is read.
    assert replaced
    assert observed == old_status
    assert prefetch._load_status(path, nonce=nonce, world_size=1) == new_status


def test_atomic_record_snapshot_still_rejects_in_place_mutation_and_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "record.json"
    path.write_bytes(b'{"schema_version":1}')

    original_fstat = os.fstat
    fstat_calls = 0

    def mutate_before_second_fstat(descriptor: int) -> os.stat_result:
        nonlocal fstat_calls
        fstat_calls += 1
        if fstat_calls == 2:
            path.write_bytes(b'{"schema_version":2}')
        return original_fstat(descriptor)

    with monkeypatch.context() as scoped:
        scoped.setattr(prefetch.os, "fstat", mutate_before_second_fstat)
        with pytest.raises(RemoteTrainingError, match="changed while reading"):
            prefetch._load_json_record(
                path,
                expected_keys=frozenset({"schema_version"}),
                surface="test record",
            )

    path.write_bytes(b'{"schema_version":1}')
    anchor = tmp_path / "anchor.json"
    os.link(path, anchor)
    with pytest.raises(RemoteTrainingError, match="direct single-link regular file"):
        prefetch._load_json_record(
            path,
            expected_keys=frozenset({"schema_version"}),
            surface="test record",
        )

    path.unlink()
    alias = tmp_path / "alias.json"
    alias.symlink_to(anchor)
    with pytest.raises(RemoteTrainingError, match="cannot open"):
        prefetch._load_json_record(
            alias,
            expected_keys=frozenset({"schema_version"}),
            surface="test record",
        )


def test_body_plan_rejects_missing_duplicate_and_misaligned_inputs(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    train = _selection_materializer(train_dir)

    with pytest.raises(RemoteTrainingError, match="requires selection"):
        prefetch.build_remote_body_plan(
            train_selection_dir=train_dir,
            selection_materializers=[],
            packed_materializer=None,
        )

    with pytest.raises(RemoteTrainingError, match="duplicate selection"):
        prefetch.build_remote_body_plan(
            train_selection_dir=train_dir,
            selection_materializers=[train, train],
            packed_materializer=None,
        )

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other = _selection_materializer(other_dir)
    with pytest.raises(RemoteTrainingError, match="no train selection"):
        prefetch.build_remote_body_plan(
            train_selection_dir=train_dir,
            selection_materializers=[other],
            packed_materializer=None,
        )

    empty = _selection_materializer(train_dir, entries=())
    with pytest.raises(RemoteTrainingError, match="train inventory must not be empty"):
        prefetch.build_remote_body_plan(
            train_selection_dir=train_dir,
            selection_materializers=[empty],
            packed_materializer=None,
        )

    packed_dir = tmp_path / "packed"
    packed_dir.mkdir()
    packed = _packed_materializer(
        packed_dir,
        entries=(_entry("packed", 1),),
    )
    with pytest.raises(RemoteTrainingError, match="identical name-sorted"):
        prefetch.build_remote_body_plan(
            train_selection_dir=train_dir,
            selection_materializers=[train],
            packed_materializer=packed,
        )


def test_body_plan_builds_validation_family_and_rejects_collisions(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    empty_dir = tmp_path / "empty"
    for directory in (train_dir, val_dir, empty_dir):
        directory.mkdir()
    train = _selection_materializer(train_dir)
    validation = _selection_materializer(
        val_dir,
        aggregate="b" * 64,
        remote_dir="features/validation",
    )

    families, bindings, packed = prefetch.build_remote_body_plan(
        train_selection_dir=train_dir,
        selection_materializers=[train, validation],
        packed_materializer=None,
    )
    assert len(families) == 2
    assert packed is None
    assert bindings[str(val_dir.resolve())].train_schedule is False

    empty_validation = _selection_materializer(
        empty_dir,
        aggregate="e" * 64,
        remote_dir="features/empty",
        entries=(),
    )
    with pytest.raises(RemoteTrainingError, match="validation inventory must not be empty"):
        prefetch.build_remote_body_plan(
            train_selection_dir=train_dir,
            selection_materializers=[train, empty_validation],
            packed_materializer=None,
        )

    collision = _selection_materializer(
        val_dir,
        aggregate="a" * 64,
        remote_dir=train.remote_dir,
    )
    with pytest.raises(RemoteTrainingError, match="family identities collide"):
        prefetch.build_remote_body_plan(
            train_selection_dir=train_dir,
            selection_materializers=[train, collision],
            packed_materializer=None,
        )


def test_schedule_map_rejects_misalignment_components_and_train_count(
    tmp_path: Path,
) -> None:
    selection = _family(tmp_path, n_entries=2)
    packed_short = _family(
        tmp_path,
        kind="packed",
        n_entries=1,
        family_char="2",
        aggregate_char="b",
    )
    with pytest.raises(RemoteTrainingError, match="aligned inventories"):
        prefetch._schedule_map((selection, packed_short))

    second_selection = _family(
        tmp_path,
        family_char="3",
        aggregate_char="d",
    )
    with pytest.raises(RemoteTrainingError, match="invalid components"):
        prefetch._schedule_map((selection, second_selection))

    validation_only = _family(
        tmp_path,
        family_char="4",
        aggregate_char="e",
        train_schedule=False,
    )
    with pytest.raises(RemoteTrainingError, match="exactly one train"):
        prefetch._schedule_map((validation_only,))

    second_train = _family(
        tmp_path,
        family_char="5",
        aggregate_char="f",
        schedule_id="9" * 64,
    )
    with pytest.raises(RemoteTrainingError, match="exactly one train"):
        prefetch._schedule_map((selection, second_train))


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"world_size": False}, id="world"),
        pytest.param({"cache_shards": False}, id="cache-shards"),
        pytest.param({"lookahead_shards_per_rank": False}, id="lookahead"),
        pytest.param({"safety_reserve_bytes": False}, id="reserve"),
    ],
)
def test_disk_admission_rejects_non_strict_parameters(
    tmp_path: Path,
    kwargs: dict[str, Any],
) -> None:
    family = _family(tmp_path)
    arguments = {
        "cache_root": tmp_path,
        "families": (family,),
        "world_size": 1,
        "cache_shards": 1,
        "lookahead_shards_per_rank": 1,
        "safety_reserve_bytes": 0,
    }
    arguments.update(kwargs)
    with pytest.raises(BeartypeCallHintParamViolation):
        prefetch.calculate_remote_prefetch_disk_admission(**arguments)


def test_control_record_loaders_reject_semantic_drift(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    family = _family(cache_root, n_entries=2)
    families = (family,)
    cfg = _cfg(cache_root, families)
    control = prefetch._create_control_tree(
        cache_root=cache_root,
        nonce=cfg.nonce,
        world_size=1,
        families=families,
    )
    cursor_path = prefetch._cursor_path(control, 0)
    progress_path = prefetch._progress_path(control, 0)
    demand_path = prefetch._demand_path(
        control,
        requester_rank=0,
        schedule_id=family.schedule_id,
        inventory_index=0,
    )
    status_path = prefetch._status_path(control, 0)
    cursor = {
        "schema_version": prefetch.REMOTE_PREFETCH_SCHEMA_VERSION,
        "nonce": cfg.nonce,
        "rank": 0,
        "schedule_id": family.schedule_id,
        "epoch": 1,
        "inventory_index": 0,
        "generation": 0,
    }
    progress = {
        "schema_version": prefetch.REMOTE_PREFETCH_SCHEMA_VERSION,
        "nonce": cfg.nonce,
        "rank": 0,
        "schedule_id": family.schedule_id,
        "generation": -1,
    }
    demand = {
        "schema_version": prefetch.REMOTE_PREFETCH_SCHEMA_VERSION,
        "nonce": cfg.nonce,
        "requester_rank": 0,
        "schedule_id": family.schedule_id,
        "epoch": 1,
        "inventory_index": 0,
        "generation": 0,
        "created_ns": 1,
    }
    prefetch._atomic_record(cursor, cursor_path)
    prefetch._atomic_record(progress, progress_path)
    prefetch._atomic_record(demand, demand_path)
    prefetch._write_worker_status(cfg, state="idle")
    status = json.loads(status_path.read_text(encoding="utf-8"))

    malformed = dict(status, nonce="stale")
    prefetch._atomic_record(malformed, status_path)
    with pytest.raises(RemoteTrainingError, match="stale nonce"):
        prefetch._load_status(status_path, nonce=cfg.nonce, world_size=1)
    malformed = dict(status, state="unknown")
    prefetch._atomic_record(malformed, status_path)
    with pytest.raises(RemoteTrainingError, match="invalid state"):
        prefetch._load_status(status_path, nonce=cfg.nonce, world_size=1)
    malformed = dict(status, error_type=False)
    prefetch._atomic_record(malformed, status_path)
    with pytest.raises(RemoteTrainingError, match="string or null"):
        prefetch._load_status(status_path, nonce=cfg.nonce, world_size=1)
    malformed = dict(status, state="error", error_type=None, error_message=None)
    prefetch._atomic_record(malformed, status_path)
    with pytest.raises(RemoteTrainingError, match="lacks typed diagnostics"):
        prefetch._load_status(status_path, nonce=cfg.nonce, world_size=1)

    malformed = dict(cursor, nonce="stale")
    prefetch._atomic_record(malformed, cursor_path)
    with pytest.raises(RemoteTrainingError, match="stale run identity"):
        prefetch._load_cursor(
            cursor_path,
            cfg=cfg,
            train_schedule_id=family.schedule_id,
            inventory_length=2,
        )

    malformed = dict(progress, schedule_id="9" * 64)
    prefetch._atomic_record(malformed, progress_path)
    with pytest.raises(RemoteTrainingError, match="stale run identity"):
        prefetch._load_progress(
            progress_path,
            cfg=cfg,
            train_schedule_id=family.schedule_id,
        )
    malformed = dict(progress, generation=-2)
    prefetch._atomic_record(malformed, progress_path)
    with pytest.raises(RemoteTrainingError, match="strict int"):
        prefetch._load_progress(
            progress_path,
            cfg=cfg,
            train_schedule_id=family.schedule_id,
        )

    schedules = prefetch._schedule_map(families)
    malformed = dict(demand, nonce="stale")
    prefetch._atomic_record(malformed, demand_path)
    with pytest.raises(RemoteTrainingError, match="stale nonce"):
        prefetch._load_demand(
            demand_path,
            cfg=cfg,
            schedules=schedules,
            requester_rank=0,
        )
    malformed = dict(demand, requester_rank=0)
    prefetch._atomic_record(malformed, demand_path)
    with pytest.raises(RemoteTrainingError, match="requester-specific"):
        prefetch._load_demand(
            demand_path,
            cfg=cfg,
            schedules=schedules,
            requester_rank=1,
        )
    malformed = dict(demand, schedule_id="9" * 64)
    prefetch._atomic_record(malformed, demand_path)
    with pytest.raises(RemoteTrainingError, match="unknown schedule"):
        prefetch._load_demand(
            demand_path,
            cfg=cfg,
            schedules=schedules,
            requester_rank=0,
        )
    malformed = dict(demand, created_ns=False)
    prefetch._atomic_record(malformed, demand_path)
    with pytest.raises(RemoteTrainingError, match="created_ns"):
        prefetch._load_demand(
            demand_path,
            cfg=cfg,
            schedules=schedules,
            requester_rank=0,
        )
    wrong_name = demand_path.with_name(f"{family.schedule_id}.00000001.json")
    prefetch._atomic_record(demand, wrong_name)
    with pytest.raises(RemoteTrainingError, match="filename"):
        prefetch._load_demand(
            wrong_name,
            cfg=cfg,
            schedules=schedules,
            requester_rank=0,
        )


@pytest.mark.parametrize(
    "record",
    [
        pytest.param(
            {"epoch": False, "inventory_index": 0, "generation": 0},
            id="epoch",
        ),
        pytest.param(
            {"epoch": 1, "inventory_index": False, "generation": 0},
            id="index",
        ),
        pytest.param(
            {"epoch": 1, "inventory_index": 0, "generation": False},
            id="generation",
        ),
    ],
)
def test_epoch_index_generation_rejects_boolean_fields(record: dict[str, Any]) -> None:
    with pytest.raises(RemoteTrainingError):
        prefetch._validate_epoch_index_generation(
            record,
            inventory_length=1,
            surface="test generation",
        )


@pytest.mark.parametrize(
    "surface",
    ["control", "demands", "receipts", "ready"],
)
def test_control_tree_rejects_exact_schema_drift(
    tmp_path: Path,
    surface: str,
) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    family = _family(cache_root, n_entries=1)
    nonce = "d" * 32
    control = prefetch._create_control_tree(
        cache_root=cache_root,
        nonce=nonce,
        world_size=1,
        families=(family,),
    )
    target = {
        "control": control,
        "demands": control / "demands",
        "receipts": control / "receipts",
        "ready": control / "ready",
    }[surface]
    (target / "unexpected").mkdir()

    with pytest.raises(RemoteTrainingError, match="invalid"):
        prefetch._validate_control_tree(
            cache_root=cache_root,
            nonce=nonce,
            world_size=1,
            families=(family,),
        )


def test_control_tree_rejects_nonce_reuse_and_sweeps_stale_nonce(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    family = _family(cache_root, n_entries=1)
    stale = cache_root / ".remote-prefetch" / "control" / ("a" * 32)
    stale.mkdir(parents=True)
    (stale / "record").write_bytes(b"stale")
    nonce = "d" * 32

    prefetch._create_control_tree(
        cache_root=cache_root,
        nonce=nonce,
        world_size=1,
        families=(family,),
    )
    assert not stale.exists()
    with pytest.raises(RemoteTrainingError, match="nonce already exists"):
        prefetch._create_control_tree(
            cache_root=cache_root,
            nonce=nonce,
            world_size=1,
            families=(family,),
        )


def test_poll_worker_command_rejects_unknown_command() -> None:
    class Pipe:
        def __init__(self) -> None:
            self.polled = False

        def poll(self) -> bool:
            if self.polled:
                return False
            self.polled = True
            return True

        def recv(self) -> object:
            return {"op": "take-over"}

    with pytest.raises(RemoteTrainingError, match="invalid command"):
        prefetch._poll_worker_command(cast(Any, Pipe()))


def test_initial_resume_demand_validates_and_deduplicates_rows() -> None:
    binding = RemoteBodyBinding(
        family_id="1" * 64,
        schedule_id="c" * 64,
        kind="selection",
        index_by_name=(
            ("shard_00000.safetensors.zst", 0),
            ("shard_00001.safetensors.zst", 1),
        ),
        paired=False,
        train_schedule=True,
    )
    state = {
        "epoch": 2,
        "shard": "shard_00001.safetensors.zst",
        "batch_rows": [
            [1, "shard_00000.safetensors.zst", 7],
            [1, "shard_00000.safetensors.zst", 9],
            [2, "shard_00001.safetensors.zst", 0],
        ],
    }

    assert prefetch._initial_resume_demand(
        resume_state_dict=state,
        train_binding=binding,
    ) == (2, 1, [(1, 0), (2, 1)])


@pytest.mark.parametrize(
    "state",
    [
        pytest.param(
            {"epoch": False, "shard": "shard_00000.safetensors.zst"},
            id="epoch",
        ),
        pytest.param({"epoch": 1, "shard": "missing"}, id="shard"),
        pytest.param(
            {
                "epoch": 1,
                "shard": "shard_00000.safetensors.zst",
                "batch_rows": [],
            },
            id="empty-rows",
        ),
        pytest.param(
            {
                "epoch": 1,
                "shard": "shard_00000.safetensors.zst",
                "batch_rows": [["malformed"]],
            },
            id="row-shape",
        ),
        pytest.param(
            {
                "epoch": 1,
                "shard": "shard_00000.safetensors.zst",
                "batch_rows": [[False, "shard_00000.safetensors.zst", 0]],
            },
            id="row-identity",
        ),
    ],
)
def test_initial_resume_demand_rejects_malformed_state(
    state: dict[str, Any],
) -> None:
    binding = RemoteBodyBinding(
        family_id="1" * 64,
        schedule_id="c" * 64,
        kind="selection",
        index_by_name=(("shard_00000.safetensors.zst", 0),),
        paired=False,
        train_schedule=True,
    )
    with pytest.raises(RemoteTrainingError):
        prefetch._initial_resume_demand(
            resume_state_dict=state,
            train_binding=binding,
        )


def test_controller_sticky_failure_replaces_malformed_status_and_is_sticky(
    tmp_path: Path,
) -> None:
    controller, _binding = _controller(tmp_path / "cache")
    status_path = prefetch._status_path(controller.control_root, 0)
    status_path.write_text("not-json", encoding="utf-8")

    first = controller._sticky_failure(
        "worker failed",
        family_id=controller.families[0].family_id,
        inventory_index=0,
        body_name=controller.families[0].entries[0].name,
    )
    second = controller._sticky_failure("different failure")

    assert second is first
    status = prefetch._load_status(
        status_path,
        nonce=controller.nonce,
        world_size=1,
    )
    assert status["state"] == "error"
    assert status["error_message"] == "worker failed"


def test_controller_sticky_failure_preserves_status_persistence_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _binding = _controller(tmp_path / "cache")

    def fail_record(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("status fsync failed")

    monkeypatch.setattr(prefetch, "_atomic_record", fail_record)
    failure = controller._sticky_failure("worker failed")

    assert failure.__notes__ == [
        "also failed to persist sticky worker state: OSError: status fsync failed"
    ]
    with pytest.raises(RemoteTrainingError) as exc_info:
        controller.check_worker_errors()
    assert exc_info.value is failure


def test_controller_detects_pipe_exit_and_owner_status_failures(tmp_path: Path) -> None:
    pipe_message = {
        "state": "error",
        "rank": 0,
        "family_id": "1" * 64,
        "inventory_index": 0,
        "body_name": "shard_00000.safetensors.zst",
        "error_type": "RuntimeError",
        "error_message": "pipe failure",
    }
    controller, _binding = _controller(
        tmp_path / "pipe",
        status_pipe=_Pipe((pipe_message,)),
    )
    with pytest.raises(RemoteTrainingError, match="pipe failure"):
        controller.check_worker_errors()

    exited, _binding = _controller(
        tmp_path / "exit",
        process=_Process(exitcode=17),
    )
    with pytest.raises(RemoteTrainingError, match="exited unexpectedly"):
        exited.check_worker_errors()

    owner, _binding = _controller(tmp_path / "owner")
    family = owner.families[0]
    prefetch._write_worker_status(
        owner._worker_cfg,
        state="error",
        family_id=family.family_id,
        inventory_index=0,
        body_name=family.entries[0].name,
        error=RuntimeError("owner failed"),
    )
    with pytest.raises(RemoteTrainingError, match="owner reported terminal failure"):
        owner.check_worker_errors()


def test_peer_status_sweep_uses_fixed_ten_second_schedule_at_world_size_eight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _binding = _controller(tmp_path / "cache", world_size=8)
    for owner_rank in range(controller.world_size):
        _write_controller_status(controller, owner_rank=owner_rank)

    clock = _FakeMonotonicClock()
    _patch_prefetch_monotonic(monkeypatch, clock)
    original_load_status = prefetch._load_status
    load_times: list[float] = []

    def recording_load_status(
        path: Path,
        *,
        nonce: str,
        world_size: int,
    ) -> dict[str, Any]:
        load_times.append(clock.now)
        return original_load_status(path, nonce=nonce, world_size=world_size)

    monkeypatch.setattr(prefetch, "_load_status", recording_load_status)
    for tick in range(1_200):
        clock.now = tick / 20
        controller.check_worker_errors()

    expected_sweep_times = [
        float(second) for second in range(0, 60, 10) for _owner_rank in range(controller.world_size)
    ]
    assert load_times == expected_sweep_times
    assert len(load_times) == 48


def test_peer_status_sweep_boundary_suppresses_all_peer_path_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _binding = _controller(tmp_path / "cache")
    clock = _FakeMonotonicClock()
    _patch_prefetch_monotonic(monkeypatch, clock)
    original_status_path = prefetch._status_path
    path_calls: list[int] = []

    def recording_status_path(control_root: Path, rank: int) -> Path:
        path_calls.append(rank)
        return original_status_path(control_root, rank)

    monkeypatch.setattr(prefetch, "_status_path", recording_status_path)
    controller.check_worker_errors()
    assert path_calls == [0]

    path_calls.clear()
    clock.now = prefetch.REMOTE_PREFETCH_PEER_STATUS_SWEEP_INTERVAL_SECONDS - 1e-9
    controller.check_worker_errors()
    assert path_calls == []

    clock.now = prefetch.REMOTE_PREFETCH_PEER_STATUS_SWEEP_INTERVAL_SECONDS
    controller.check_worker_errors()
    assert path_calls == [0]


def test_peer_status_sweep_interval_starts_at_successful_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _binding = _controller(tmp_path / "cache")
    _write_controller_status(controller, owner_rank=0)
    clock = _FakeMonotonicClock()
    _patch_prefetch_monotonic(monkeypatch, clock)
    original_load_status = prefetch._load_status
    load_started_at: list[float] = []

    def slow_load_status(
        path: Path,
        *,
        nonce: str,
        world_size: int,
    ) -> dict[str, Any]:
        load_started_at.append(clock.now)
        result = original_load_status(path, nonce=nonce, world_size=world_size)
        clock.now += 3.0
        return result

    monkeypatch.setattr(prefetch, "_load_status", slow_load_status)

    controller.check_worker_errors()
    assert load_started_at == [0.0]
    assert controller._last_successful_peer_status_sweep_monotonic == 3.0

    clock.now = 13.0 - 1e-9
    controller.check_worker_errors()
    assert load_started_at == [0.0]

    clock.now = 13.0
    controller.check_worker_errors()
    assert load_started_at == [0.0, 13.0]
    assert controller._last_successful_peer_status_sweep_monotonic == 16.0


def test_peer_error_is_delayed_until_due_then_sticky(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _binding = _controller(tmp_path / "cache")
    _write_controller_status(controller, owner_rank=0)
    clock = _FakeMonotonicClock()
    _patch_prefetch_monotonic(monkeypatch, clock)

    controller.check_worker_errors()
    _write_controller_status(
        controller,
        owner_rank=0,
        state="error",
        error=RuntimeError("peer failed after sweep"),
    )

    clock.now = prefetch.REMOTE_PREFETCH_PEER_STATUS_SWEEP_INTERVAL_SECONDS - 1e-9
    controller.check_worker_errors()

    clock.now = prefetch.REMOTE_PREFETCH_PEER_STATUS_SWEEP_INTERVAL_SECONDS
    with pytest.raises(RemoteTrainingError, match="peer failed after sweep") as exc_info:
        controller.check_worker_errors()
    failure = exc_info.value

    clock.now += 0.001
    with pytest.raises(RemoteTrainingError) as sticky_info:
        controller.check_worker_errors()
    assert sticky_info.value is failure


def test_peer_error_sweep_preserves_rank_order_and_includes_local_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _binding = _controller(tmp_path / "cache", world_size=4)
    for owner_rank in range(controller.world_size):
        _write_controller_status(
            controller,
            owner_rank=owner_rank,
            state="error" if owner_rank == 3 else "idle",
            error=RuntimeError("later-rank peer failure") if owner_rank == 3 else None,
        )
    original_load_status = prefetch._load_status
    loaded_ranks: list[int] = []

    def recording_load_status(
        path: Path,
        *,
        nonce: str,
        world_size: int,
    ) -> dict[str, Any]:
        loaded_ranks.append(int(path.stem.removeprefix("rank_")))
        return original_load_status(path, nonce=nonce, world_size=world_size)

    monkeypatch.setattr(prefetch, "_load_status", recording_load_status)

    with pytest.raises(RemoteTrainingError, match="later-rank peer failure") as exc_info:
        controller.check_worker_errors()

    assert loaded_ranks[:4] == [0, 1, 2, 3]
    assert loaded_ranks[4:] == [0]
    assert controller._last_successful_peer_status_sweep_monotonic is None
    failure = exc_info.value

    with pytest.raises(RemoteTrainingError) as sticky_info:
        controller.check_worker_errors()
    assert sticky_info.value is failure
    assert loaded_ranks == [0, 1, 2, 3, 0]


def test_local_failure_sampling_uses_fixed_one_second_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CountingPipe(_Pipe):
        def __init__(self) -> None:
            super().__init__()
            self.poll_calls = 0

        def poll(self, timeout: float | None = None) -> bool:
            self.poll_calls += 1
            return super().poll(timeout)

    class CountingProcess(_Process):
        def __init__(self) -> None:
            super().__init__()
            self.exitcode_calls = 0

        @property
        def exitcode(self) -> int | None:
            self.exitcode_calls += 1
            return None if self._alive else self._exitcode

    status_pipe = CountingPipe()
    process = CountingProcess()
    controller, _binding = _controller(
        tmp_path / "cache",
        process=process,
        status_pipe=status_pipe,
    )
    clock = _FakeMonotonicClock()
    _patch_prefetch_monotonic(monkeypatch, clock)

    for tick in range(1_200):
        clock.now = tick / 20
        controller.check_worker_errors()

    assert status_pipe.poll_calls == 60
    assert process.exitcode_calls == 60
    assert controller._last_successful_local_failure_check_monotonic == 59.0


def test_new_local_failures_wait_for_boundary_but_sticky_failure_is_immediate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeMonotonicClock()
    _patch_prefetch_monotonic(monkeypatch, clock)

    sticky, _binding = _controller(tmp_path / "sticky")
    sticky.check_worker_errors()
    sticky_failure = sticky._sticky_failure("already sticky")
    clock.now = 0.1
    with pytest.raises(RemoteTrainingError) as sticky_info:
        sticky.check_worker_errors()
    assert sticky_info.value is sticky_failure

    pipe = _Pipe()
    piped, _binding = _controller(tmp_path / "pipe", status_pipe=pipe)
    clock.now = 10.0
    piped.check_worker_errors()
    pipe.messages.append(
        {
            "state": "error",
            "rank": 0,
            "family_id": "1" * 64,
            "inventory_index": 0,
            "body_name": "shard_00000.safetensors.zst",
            "error_type": "RuntimeError",
            "error_message": "new pipe failure",
        }
    )
    clock.now = 10.0 + prefetch.REMOTE_PREFETCH_LOCAL_FAILURE_CHECK_INTERVAL_SECONDS - 1e-9
    piped.check_worker_errors()
    clock.now = 10.0 + prefetch.REMOTE_PREFETCH_LOCAL_FAILURE_CHECK_INTERVAL_SECONDS
    with pytest.raises(RemoteTrainingError, match="new pipe failure") as pipe_info:
        piped.check_worker_errors()
    clock.now += 0.001
    with pytest.raises(RemoteTrainingError) as pipe_sticky_info:
        piped.check_worker_errors()
    assert pipe_sticky_info.value is pipe_info.value

    process = _Process()
    exited, _binding = _controller(tmp_path / "exit", process=process)
    clock.now = 20.0
    exited.check_worker_errors()
    process._exitcode = 23
    clock.now = 20.0 + prefetch.REMOTE_PREFETCH_LOCAL_FAILURE_CHECK_INTERVAL_SECONDS - 1e-9
    exited.check_worker_errors()
    clock.now = 20.0 + prefetch.REMOTE_PREFETCH_LOCAL_FAILURE_CHECK_INTERVAL_SECONDS
    with pytest.raises(RemoteTrainingError, match="exitcode=23"):
        exited.check_worker_errors()


def test_local_failure_interval_starts_at_successful_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeMonotonicClock()

    class SlowPipe(_Pipe):
        def __init__(self) -> None:
            super().__init__()
            self.poll_started_at: list[float] = []

        def poll(self, timeout: float | None = None) -> bool:
            self.poll_started_at.append(clock.now)
            clock.now += 0.25
            return super().poll(timeout)

    status_pipe = SlowPipe()
    controller, _binding = _controller(
        tmp_path / "cache",
        status_pipe=status_pipe,
    )
    _patch_prefetch_monotonic(monkeypatch, clock)

    controller.check_worker_errors()
    assert status_pipe.poll_started_at == [0.0]
    assert controller._last_successful_local_failure_check_monotonic == 0.25

    clock.now = 1.25 - 1e-9
    controller.check_worker_errors()
    assert status_pipe.poll_started_at == [0.0]

    clock.now = 1.25
    controller.check_worker_errors()
    assert status_pipe.poll_started_at == [0.0, 1.25]
    assert controller._last_successful_local_failure_check_monotonic == 1.5


def test_failed_local_failure_check_is_retried_without_delay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailsOncePipe(_Pipe):
        def __init__(self) -> None:
            super().__init__()
            self.poll_calls = 0

        def poll(self, timeout: float | None = None) -> bool:
            self.poll_calls += 1
            if self.poll_calls == 1:
                raise OSError("local pipe poll failed")
            return super().poll(timeout)

    status_pipe = FailsOncePipe()
    controller, _binding = _controller(
        tmp_path / "cache",
        status_pipe=status_pipe,
    )
    clock = _FakeMonotonicClock()
    _patch_prefetch_monotonic(monkeypatch, clock)

    with pytest.raises(OSError, match="local pipe poll failed"):
        controller.check_worker_errors()
    assert controller._last_successful_local_failure_check_monotonic is None

    controller.check_worker_errors()
    assert status_pipe.poll_calls == 2
    assert controller._last_successful_local_failure_check_monotonic == 0.0


@pytest.mark.parametrize(
    "corruption",
    [
        "malformed",
        "schema",
        "stale",
        "symlink",
        "dangling-symlink",
        "self-symlink",
    ],
)
def test_failed_due_peer_sweep_is_fatal_and_retried_without_delay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    controller, _binding = _controller(tmp_path / corruption)
    _write_controller_status(controller, owner_rank=0)
    status_path = prefetch._status_path(controller.control_root, 0)
    clock = _FakeMonotonicClock()
    _patch_prefetch_monotonic(monkeypatch, clock)
    controller.check_worker_errors()

    clock.now = prefetch.REMOTE_PREFETCH_PEER_STATUS_SWEEP_INTERVAL_SECONDS
    if corruption == "malformed":
        status_path.write_text("not-json", encoding="utf-8")
    elif corruption in {"schema", "stale"}:
        record = json.loads(status_path.read_text(encoding="utf-8"))
        if corruption == "schema":
            record["unexpected"] = True
        else:
            record["nonce"] = "e" * 32
        status_path.write_text(json.dumps(record), encoding="utf-8")
    elif corruption == "symlink":
        target = tmp_path / "status-target.json"
        target.write_bytes(status_path.read_bytes())
        status_path.unlink()
        status_path.symlink_to(target)
    elif corruption == "dangling-symlink":
        status_path.unlink()
        status_path.symlink_to(tmp_path / "missing-status-target.json")
    else:
        status_path.unlink()
        status_path.symlink_to(status_path.name)

    with pytest.raises(RemoteTrainingError):
        controller.check_worker_errors()
    assert controller._last_successful_peer_status_sweep_monotonic == 0.0

    status_path.unlink()
    _write_controller_status(controller, owner_rank=0)
    controller.check_worker_errors()
    assert (
        controller._last_successful_peer_status_sweep_monotonic
        == prefetch.REMOTE_PREFETCH_PEER_STATUS_SWEEP_INTERVAL_SECONDS
    )


def test_failed_later_rank_sweep_retries_from_rank_zero_without_delay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _binding = _controller(tmp_path / "cache", world_size=4)
    for owner_rank in range(controller.world_size):
        _write_controller_status(controller, owner_rank=owner_rank)
    clock = _FakeMonotonicClock()
    _patch_prefetch_monotonic(monkeypatch, clock)
    controller.check_worker_errors()

    status_path = prefetch._status_path(controller.control_root, 3)
    status_path.write_text("not-json", encoding="utf-8")
    clock.now = prefetch.REMOTE_PREFETCH_PEER_STATUS_SWEEP_INTERVAL_SECONDS
    original_load_status = prefetch._load_status
    loaded_ranks: list[int] = []

    def recording_load_status(
        path: Path,
        *,
        nonce: str,
        world_size: int,
    ) -> dict[str, Any]:
        loaded_ranks.append(int(path.stem.removeprefix("rank_")))
        return original_load_status(path, nonce=nonce, world_size=world_size)

    monkeypatch.setattr(prefetch, "_load_status", recording_load_status)
    with pytest.raises(RemoteTrainingError):
        controller.check_worker_errors()
    assert loaded_ranks == [0, 1, 2, 3]
    assert controller._last_successful_peer_status_sweep_monotonic == 0.0

    _write_controller_status(controller, owner_rank=3)
    controller.check_worker_errors()
    assert loaded_ranks == [0, 1, 2, 3, 0, 1, 2, 3]
    assert (
        controller._last_successful_peer_status_sweep_monotonic
        == prefetch.REMOTE_PREFETCH_PEER_STATUS_SWEEP_INTERVAL_SECONDS
    )


def test_permission_failure_does_not_advance_peer_sweep_timer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _binding = _controller(tmp_path / "cache")
    _write_controller_status(controller, owner_rank=0)
    status_path = prefetch._status_path(controller.control_root, 0)
    clock = _FakeMonotonicClock()
    _patch_prefetch_monotonic(monkeypatch, clock)
    controller.check_worker_errors()
    clock.now = prefetch.REMOTE_PREFETCH_PEER_STATUS_SWEEP_INTERVAL_SECONDS

    original_open = prefetch.os.open
    denied = True

    def deny_status_once(path: Any, *args: Any, **kwargs: Any) -> int:
        nonlocal denied
        if denied and Path(path) == status_path:
            denied = False
            raise PermissionError("status permission denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(prefetch.os, "open", deny_status_once)
    with pytest.raises(RemoteTrainingError, match="cannot open remote prefetch record"):
        controller.check_worker_errors()
    assert controller._last_successful_peer_status_sweep_monotonic == 0.0

    controller.check_worker_errors()
    assert (
        controller._last_successful_peer_status_sweep_monotonic
        == prefetch.REMOTE_PREFETCH_PEER_STATUS_SWEEP_INTERVAL_SECONDS
    )


def test_missing_peer_status_records_are_successful_sweep_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _binding = _controller(tmp_path / "cache", world_size=8)
    clock = _FakeMonotonicClock(7.0)
    _patch_prefetch_monotonic(monkeypatch, clock)

    controller.check_worker_errors()

    assert controller._last_successful_peer_status_sweep_monotonic == 7.0


def test_peer_status_sweep_defers_atomic_snapshot_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _binding = _controller(tmp_path / "cache")
    _write_controller_status(controller, owner_rank=0)
    clock = _FakeMonotonicClock()
    _patch_prefetch_monotonic(monkeypatch, clock)
    original_load_status = prefetch._load_status
    calls = 0

    def race_once(
        path: Path,
        *,
        nonce: str,
        world_size: int,
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RemoteTrainingError(
                f"remote prefetch record changed while reading: {path}"
            )
        return original_load_status(path, nonce=nonce, world_size=world_size)

    monkeypatch.setattr(prefetch, "_load_status", race_once)
    controller.check_worker_errors()
    assert controller._last_successful_peer_status_sweep_monotonic == 0.0

    clock.now = prefetch.REMOTE_PREFETCH_PEER_STATUS_SWEEP_INTERVAL_SECONDS
    controller.check_worker_errors()
    assert calls == 2
    assert (
        controller._last_successful_peer_status_sweep_monotonic
        == prefetch.REMOTE_PREFETCH_PEER_STATUS_SWEEP_INTERVAL_SECONDS
    )


def test_due_peer_sweep_is_nonblocking_and_single_threaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _binding = _controller(tmp_path / "cache")
    _write_controller_status(controller, owner_rank=0)
    clock = _FakeMonotonicClock()
    _patch_prefetch_monotonic(monkeypatch, clock)
    controller.check_worker_errors()
    clock.now = prefetch.REMOTE_PREFETCH_PEER_STATUS_SWEEP_INTERVAL_SECONDS

    entered_load = threading.Event()
    release_load = threading.Event()
    original_load_status = prefetch._load_status
    loads = 0

    def blocking_load_status(
        path: Path,
        *,
        nonce: str,
        world_size: int,
    ) -> dict[str, Any]:
        nonlocal loads
        loads += 1
        entered_load.set()
        assert release_load.wait(timeout=2)
        return original_load_status(path, nonce=nonce, world_size=world_size)

    monkeypatch.setattr(prefetch, "_load_status", blocking_load_status)
    original_status_path = prefetch._status_path
    status_path_threads: list[str] = []

    def recording_status_path(control_root: Path, rank: int) -> Path:
        status_path_threads.append(threading.current_thread().name)
        return original_status_path(control_root, rank)

    monkeypatch.setattr(prefetch, "_status_path", recording_status_path)
    original_drain = controller._drain_status_pipe
    local_check_threads: list[str] = []

    def recording_drain() -> None:
        local_check_threads.append(threading.current_thread().name)
        original_drain()

    monkeypatch.setattr(controller, "_drain_status_pipe", recording_drain)
    failures: list[BaseException] = []

    def invoke() -> None:
        try:
            controller.check_worker_errors()
        except BaseException as exc:
            failures.append(exc)

    sweeping = threading.Thread(target=invoke, name="sweeping")
    suppressed = threading.Thread(target=invoke, name="suppressed")
    sweeping.start()
    assert entered_load.wait(timeout=1)
    suppressed.start()
    suppressed.join(timeout=0.5)
    assert not suppressed.is_alive()
    assert loads == 1
    assert local_check_threads == ["sweeping"]

    release_load.set()
    sweeping.join(timeout=1)
    assert not sweeping.is_alive()
    assert failures == []
    assert loads == 1
    assert status_path_threads == ["sweeping"]


def test_failed_peer_lock_rechecks_an_already_detected_sticky_failure(
    tmp_path: Path,
) -> None:
    controller, _binding = _controller(tmp_path / "cache")
    failure = RemoteTrainingError("detected during concurrent sweep")

    class ConcurrentFailureLock:
        def acquire(self, *, blocking: bool) -> bool:
            assert blocking is False
            with controller._local_failure_lock:
                controller._failure = failure
            return False

    controller._peer_status_sweep_lock = cast(Any, ConcurrentFailureLock())
    with pytest.raises(RemoteTrainingError) as exc_info:
        controller.check_worker_errors()
    assert exc_info.value is failure


def test_unavailable_peer_lock_without_failure_returns_without_peer_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _binding = _controller(tmp_path / "cache")

    class UnavailableLock:
        def acquire(self, *, blocking: bool) -> bool:
            assert blocking is False
            return False

    def forbidden_status_path(_control_root: Path, _rank: int) -> Path:
        raise AssertionError("peer-lock contention must suppress peer I/O")

    controller._peer_status_sweep_lock = cast(Any, UnavailableLock())
    monkeypatch.setattr(prefetch, "_status_path", forbidden_status_path)

    controller.check_worker_errors()


def test_acquired_peer_lock_rechecks_sticky_failure_before_peer_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _binding = _controller(tmp_path / "cache")
    failure = RemoteTrainingError("detected immediately before peer sweep")
    released = False

    class ConcurrentFailureLock:
        def acquire(self, *, blocking: bool) -> bool:
            assert blocking is False
            with controller._local_failure_lock:
                controller._failure = failure
            return True

        def release(self) -> None:
            nonlocal released
            released = True

    def forbidden_status_path(_control_root: Path, _rank: int) -> Path:
        raise AssertionError("sticky failure must suppress peer path computation")

    controller._peer_status_sweep_lock = cast(Any, ConcurrentFailureLock())
    monkeypatch.setattr(prefetch, "_status_path", forbidden_status_path)

    with pytest.raises(RemoteTrainingError) as exc_info:
        controller.check_worker_errors()

    assert exc_info.value is failure
    assert released


def test_throttled_peer_sweep_rechecks_sticky_failure_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _binding = _controller(tmp_path / "cache")
    clock = _FakeMonotonicClock()
    _patch_prefetch_monotonic(monkeypatch, clock)
    controller.check_worker_errors()
    failure = RemoteTrainingError("detected while checking the throttle")

    def install_failure_during_clock_read() -> float:
        with controller._local_failure_lock:
            controller._failure = failure
        return 0.1

    _patch_prefetch_monotonic(monkeypatch, install_failure_during_clock_read)

    with pytest.raises(RemoteTrainingError) as exc_info:
        controller.check_worker_errors()

    assert exc_info.value is failure


def test_status_pipe_eof_and_non_error_messages_do_not_create_failures(
    tmp_path: Path,
) -> None:
    class EofPipe(_Pipe):
        def poll(self, timeout: float | None = None) -> bool:
            del timeout
            return True

        def recv(self) -> object:
            raise EOFError

    eof, _binding = _controller(tmp_path / "eof", status_pipe=EofPipe())
    eof._drain_status_pipe()
    assert eof._failure is None

    informational, _binding = _controller(
        tmp_path / "informational",
        status_pipe=_Pipe((None, {"state": "idle"}, "unknown")),
    )
    informational._drain_status_pipe()
    assert informational._failure is None


def test_due_local_failure_check_is_serialized_and_coalesced_across_callers(
    tmp_path: Path,
) -> None:
    class BlockingPollPipe(_Pipe):
        def __init__(self) -> None:
            super().__init__()
            self.first_entered = threading.Event()
            self.release_first = threading.Event()
            self.metrics_lock = threading.Lock()
            self.active = 0
            self.max_active = 0
            self.poll_calls = 0

        def poll(self, timeout: float | None = None) -> bool:
            del timeout
            with self.metrics_lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                self.poll_calls += 1
                call = self.poll_calls
            try:
                if call == 1:
                    self.first_entered.set()
                    assert self.release_first.wait(timeout=2)
                return False
            finally:
                with self.metrics_lock:
                    self.active -= 1

    status_pipe = BlockingPollPipe()
    controller, _binding = _controller(
        tmp_path / "cache",
        status_pipe=status_pipe,
    )
    second_attempting = threading.Event()

    class RecordingRLock:
        def __init__(self) -> None:
            self._inner = threading.RLock()

        def __enter__(self) -> None:
            if threading.current_thread().name == "second":
                second_attempting.set()
            self._inner.acquire()

        def __exit__(self, *_args: Any) -> None:
            self._inner.release()

    controller._local_failure_lock = cast(Any, RecordingRLock())
    failures: list[BaseException] = []

    def invoke() -> None:
        try:
            controller.check_worker_errors()
        except BaseException as exc:
            failures.append(exc)

    first = threading.Thread(target=invoke, name="first")
    second = threading.Thread(target=invoke, name="second")
    first.start()
    try:
        assert status_pipe.first_entered.wait(timeout=1)
        second.start()
        assert second_attempting.wait(timeout=1)
        with status_pipe.metrics_lock:
            assert status_pipe.poll_calls == 1
            assert status_pipe.max_active == 1
    finally:
        status_pipe.release_first.set()
        first.join(timeout=2)
        second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert failures == []
    assert status_pipe.poll_calls == 1
    assert status_pipe.max_active == 1


def test_real_pipe_error_is_consumed_once_and_shared_across_callers(
    tmp_path: Path,
) -> None:
    context = mp.get_context("spawn")
    parent_status, child_status = context.Pipe(duplex=False)
    controller, _binding = _controller(
        tmp_path / "cache",
        status_pipe=cast(Any, parent_status),
    )
    child_status.send(
        {
            "state": "error",
            "rank": 0,
            "family_id": "1" * 64,
            "inventory_index": 0,
            "body_name": "shard_00000.safetensors.zst",
            "error_type": "RuntimeError",
            "error_message": "one real pipe failure",
        }
    )
    barrier = threading.Barrier(6)
    failures: list[RemoteTrainingError] = []

    def invoke() -> None:
        barrier.wait()
        try:
            controller.check_worker_errors()
        except RemoteTrainingError as exc:
            failures.append(exc)

    callers = [threading.Thread(target=invoke, name=f"caller-{index}") for index in range(6)]
    try:
        for caller in callers:
            caller.start()
        for caller in callers:
            caller.join(timeout=2)
        assert all(not caller.is_alive() for caller in callers)
        assert len(failures) == len(callers)
        assert len({id(failure) for failure in failures}) == 1
        assert "one real pipe failure" in str(failures[0])
        assert not parent_status.poll()
    finally:
        parent_status.close()
        child_status.close()


@pytest.mark.parametrize("peer_load_fails", [False, True])
def test_pipe_failure_during_blocked_peer_load_is_shared_by_every_caller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    peer_load_fails: bool,
) -> None:
    context = mp.get_context("spawn")
    parent_status, child_status = context.Pipe(duplex=False)
    controller, _binding = _controller(
        tmp_path / "cache",
        status_pipe=cast(Any, parent_status),
    )
    _write_controller_status(controller, owner_rank=0)
    controller.check_worker_errors()
    controller._last_successful_peer_status_sweep_monotonic = (
        time.monotonic() - prefetch.REMOTE_PREFETCH_PEER_STATUS_SWEEP_INTERVAL_SECONDS
    )

    peer_snapshot_loaded = threading.Event()
    release_peer_load = threading.Event()
    original_load_status = prefetch._load_status
    blocked_once = False

    def blocking_load_status(
        path: Path,
        *,
        nonce: str,
        world_size: int,
    ) -> dict[str, Any]:
        nonlocal blocked_once
        result = original_load_status(path, nonce=nonce, world_size=world_size)
        if threading.current_thread().name == "sweeping" and not blocked_once:
            blocked_once = True
            peer_snapshot_loaded.set()
            assert release_peer_load.wait(timeout=2)
            if peer_load_fails:
                raise RemoteTrainingError("concurrent peer record failure")
        return result

    monkeypatch.setattr(prefetch, "_load_status", blocking_load_status)
    failures: list[RemoteTrainingError] = []
    failure_lock = threading.Lock()

    def invoke() -> None:
        try:
            controller.check_worker_errors()
        except RemoteTrainingError as exc:
            with failure_lock:
                failures.append(exc)

    sweeping = threading.Thread(target=invoke, name="sweeping")
    followers = [threading.Thread(target=invoke, name=f"follower-{index}") for index in range(4)]
    try:
        sweeping.start()
        assert peer_snapshot_loaded.wait(timeout=1)
        child_status.send(
            {
                "state": "error",
                "rank": 0,
                "family_id": "1" * 64,
                "inventory_index": 0,
                "body_name": "shard_00000.safetensors.zst",
                "error_type": "RuntimeError",
                "error_message": "pipe failure during peer load",
            }
        )
        with controller._local_failure_lock:
            controller._last_successful_local_failure_check_monotonic = (
                time.monotonic() - prefetch.REMOTE_PREFETCH_LOCAL_FAILURE_CHECK_INTERVAL_SECONDS
            )
        for follower in followers:
            follower.start()
        for follower in followers:
            follower.join(timeout=2)
        assert all(not follower.is_alive() for follower in followers)
        assert len(failures) == len(followers)

        release_peer_load.set()
        sweeping.join(timeout=2)
        assert not sweeping.is_alive()
        assert len(failures) == len(followers) + 1
        assert len({id(failure) for failure in failures}) == 1
        assert "pipe failure during peer load" in str(failures[0])
        assert not parent_status.poll()
    finally:
        release_peer_load.set()
        sweeping.join(timeout=2)
        for follower in followers:
            follower.join(timeout=2)
        parent_status.close()
        child_status.close()


def test_closable_prefetch_owner_wait_and_consumer_share_peer_sweep_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _binding = _controller(tmp_path / "cache")
    _write_controller_status(controller, owner_rank=0)
    real_monotonic = time.monotonic
    started_at = real_monotonic()

    def scaled_monotonic() -> float:
        return (real_monotonic() - started_at) * 20.0

    _patch_prefetch_monotonic(monkeypatch, scaled_monotonic)
    original_load_status = prefetch._load_status
    load_times: list[float] = []
    count_lock = threading.Lock()
    check_threads: set[str] = set()

    def recording_load_status(
        path: Path,
        *,
        nonce: str,
        world_size: int,
    ) -> dict[str, Any]:
        with count_lock:
            load_times.append(scaled_monotonic())
        return original_load_status(path, nonce=nonce, world_size=world_size)

    monkeypatch.setattr(prefetch, "_load_status", recording_load_status)
    check_calls = 0

    def check() -> None:
        nonlocal check_calls
        with count_lock:
            check_calls += 1
            check_threads.add(threading.current_thread().name)
        controller.check_worker_errors()

    producer_stop = threading.Event()

    def owner_wait_source():
        while not producer_stop.wait(prefetch.REMOTE_PREFETCH_POLL_SECONDS):
            check()
        if False:
            yield None

    loader = ClosableBatchPrefetcher(
        owner_wait_source(),
        prefetch_batches=PositiveInt.parse(1),
        stop_callback=producer_stop.set,
        failure_check=check,
    )
    outcome: list[str] = []

    def consume_empty_queue() -> None:
        try:
            next(loader)
        except StopIteration:
            outcome.append("stopped")

    consumer = threading.Thread(target=consume_empty_queue, name="consumer")
    consumer.start()
    deadline = real_monotonic() + 2.0
    while real_monotonic() < deadline:
        with count_lock:
            if (
                len(load_times) >= 2
                and check_calls > len(load_times)
                and {"consumer", "remote-batch-prefetch"} <= check_threads
            ):
                break
        time.sleep(0.01)
    loader.close()
    consumer.join(timeout=1)

    assert not consumer.is_alive()
    assert outcome == ["stopped"]
    with count_lock:
        observed_load_times = list(load_times)
        observed_check_calls = check_calls
        observed_check_threads = set(check_threads)
    assert len(observed_load_times) >= 2
    assert observed_check_calls > len(observed_load_times)
    assert {"consumer", "remote-batch-prefetch"} <= observed_check_threads
    assert all(
        later - earlier >= prefetch.REMOTE_PREFETCH_PEER_STATUS_SWEEP_INTERVAL_SECONDS - 0.05
        for earlier, later in itertools.pairwise(observed_load_times)
    )


def test_controller_family_cursor_progress_and_window_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "cache"
    train = _family(cache_root, n_entries=2)
    validation = _family(
        cache_root,
        n_entries=2,
        family_char="2",
        aggregate_char="b",
        schedule_id="2" * 64,
        train_schedule=False,
    )
    controller, train_binding = _controller(
        cache_root,
        families=(train, validation),
    )
    validation_binding = controller.bindings[str(Path(validation.local_dir).resolve())]
    unknown = replace(train_binding, family_id="9" * 64)
    with pytest.raises(RemoteTrainingError, match="unknown family"):
        controller._family(unknown)

    controller.publish_cursor(
        binding=validation_binding,
        epoch=1,
        inventory_index=0,
    )
    controller.mark_loaded(
        binding=validation_binding,
        epoch=1,
        inventory_index=0,
    )
    controller.request_window(binding=train_binding)

    for epoch, index in ((False, 0), (1, False), (1, 2)):
        error_type = (
            BeartypeCallHintParamViolation
            if isinstance(epoch, bool) or isinstance(index, bool)
            else RemoteTrainingError
        )
        with pytest.raises(error_type):
            controller.publish_cursor(
                binding=train_binding,
                epoch=epoch,
                inventory_index=index,
            )
        with pytest.raises(error_type):
            controller.mark_loaded(
                binding=train_binding,
                epoch=epoch,
                inventory_index=index,
            )

    controller.publish_cursor(
        binding=train_binding,
        epoch=2,
        inventory_index=1,
    )
    controller.mark_loaded(
        binding=train_binding,
        epoch=2,
        inventory_index=1,
    )
    cursor = prefetch._load_cursor(
        prefetch._cursor_path(controller.control_root, 0),
        cfg=controller._worker_cfg,
        train_schedule_id=train.schedule_id,
        inventory_length=2,
    )
    progress = prefetch._load_progress(
        prefetch._progress_path(controller.control_root, 0),
        cfg=controller._worker_cfg,
        train_schedule_id=train.schedule_id,
    )
    assert cursor["generation"] == 3
    assert progress["generation"] == 3

    with pytest.raises(BeartypeCallHintParamViolation):
        controller.request_window(
            binding=validation_binding,
            start_index=False,
        )
    published: list[int] = []

    def record_demand(
        *,
        binding: RemoteBodyBinding,
        epoch: int,
        inventory_index: int,
    ) -> int:
        assert binding is validation_binding
        assert epoch == 1
        published.append(inventory_index)
        return inventory_index

    monkeypatch.setattr(controller, "_publish_demand", record_demand)
    controller.request_window(binding=validation_binding, start_index=1)
    assert published == [1, 0]


def test_publish_and_remove_demand_handle_existing_and_failed_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, binding = _controller(tmp_path / "cache")
    with pytest.raises(BeartypeCallHintParamViolation):
        controller._publish_demand(
            binding=binding,
            epoch=False,
            inventory_index=0,
        )
    with pytest.raises(BeartypeCallHintParamViolation):
        controller._publish_demand(
            binding=binding,
            epoch=1,
            inventory_index=False,
        )

    assert (
        controller._publish_demand(
            binding=binding,
            epoch=2,
            inventory_index=0,
        )
        == 2
    )
    demand_path = prefetch._demand_path(
        controller.control_root,
        requester_rank=0,
        schedule_id=binding.schedule_id,
        inventory_index=0,
    )
    original = demand_path.read_bytes()
    assert (
        controller._publish_demand(
            binding=binding,
            epoch=1,
            inventory_index=0,
        )
        == 0
    )
    assert demand_path.read_bytes() == original
    controller._remove_satisfied_demand(
        binding=binding,
        inventory_index=0,
        generation=0,
    )
    assert demand_path.exists()
    controller._remove_satisfied_demand(
        binding=binding,
        inventory_index=0,
        generation=2,
    )
    assert not demand_path.exists()
    controller._remove_satisfied_demand(
        binding=binding,
        inventory_index=0,
        generation=2,
    )

    original_atomic = prefetch._atomic_record

    def fail_after_record(value: dict[str, Any], path: Path) -> None:
        original_atomic(value, path)
        raise OSError("demand directory fsync failed")

    monkeypatch.setattr(prefetch, "_atomic_record", fail_after_record)
    with pytest.raises(OSError, match="demand directory fsync failed"):
        controller._publish_demand(
            binding=binding,
            epoch=1,
            inventory_index=1,
        )
    failed_path = prefetch._demand_path(
        controller.control_root,
        requester_rank=0,
        schedule_id=binding.schedule_id,
        inventory_index=1,
    )
    assert not failed_path.exists()


def test_publish_demand_preserves_primary_when_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, binding = _controller(tmp_path / "cache")
    primary = OSError("record failed")
    cleanup = OSError("cleanup failed")

    def fail_record(value: dict[str, Any], path: Path) -> None:
        prefetch.atomic_json_dump(dict(value), path)
        raise primary

    def fail_fsync(_path: Path) -> None:
        raise cleanup

    monkeypatch.setattr(prefetch, "_atomic_record", fail_record)
    monkeypatch.setattr(prefetch, "_fsync_directory", fail_fsync)
    with pytest.raises(OSError) as exc_info:
        controller._publish_demand(
            binding=binding,
            epoch=1,
            inventory_index=0,
        )

    assert exc_info.value is primary
    assert primary.__notes__ == [
        "also failed to roll back requester demand: OSError: cleanup failed"
    ]


def test_wait_ready_cancellation_and_timeout_include_owner_status(
    tmp_path: Path,
) -> None:
    controller, binding = _controller(
        tmp_path / "cancel",
        remote=_remote_config(wait_timeout=0.01),
    )
    controller.cancel_event.set()
    with pytest.raises(RemoteMaterializationCancelled, match="owner=0"):
        controller._wait_ready(
            binding=binding,
            inventory_index=0,
            generation=0,
            cancel_event=None,
        )

    timed_out, timed_binding = _controller(
        tmp_path / "timeout",
        remote=_remote_config(wait_timeout=0.001),
    )
    prefetch._write_worker_status(timed_out._worker_cfg, state="idle")
    with pytest.raises(RemoteTrainingError, match="worker_status=.*idle"):
        timed_out._wait_ready(
            binding=timed_binding,
            inventory_index=0,
            generation=0,
            cancel_event=None,
        )


def test_read_body_rejects_entry_drift_and_missing_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, binding = _controller(tmp_path / "cache")
    family = controller.families[0]
    entry = family.entries[0]
    path = Path(family.local_dir) / entry.name
    with pytest.raises(RemoteTrainingError, match="entry drift"):
        controller.read_body(
            binding=binding,
            path=path,
            entry={**_reader_entry(entry), "byte_length": entry.byte_length + 1},
            trust_existing_cache=False,
            cancel_event=None,
            epoch=1,
        )

    monkeypatch.setattr(controller, "_publish_demand", lambda **_kwargs: 0)
    monkeypatch.setattr(controller, "_wait_ready", lambda **_kwargs: None)
    with pytest.raises(RemoteTrainingError, match="missing its nonce-scoped"):
        controller.read_body(
            binding=binding,
            path=path,
            entry=_reader_entry(entry),
            trust_existing_cache=False,
            cancel_event=None,
            epoch=1,
        )


def test_read_body_preserves_primary_when_demand_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, binding = _controller(tmp_path / "cache")
    family = controller.families[0]
    entry = family.entries[0]
    path = Path(family.local_dir) / entry.name
    primary = RemoteMaterializationCancelled("cancelled")
    cleanup = OSError("demand cleanup failed")

    monkeypatch.setattr(controller, "_publish_demand", lambda **_kwargs: 0)

    def fail_wait(**_kwargs: Any) -> None:
        raise primary

    def fail_remove(**_kwargs: Any) -> None:
        raise cleanup

    monkeypatch.setattr(controller, "_wait_ready", fail_wait)
    monkeypatch.setattr(controller, "_remove_satisfied_demand", fail_remove)
    with pytest.raises(RemoteMaterializationCancelled) as exc_info:
        controller.read_body(
            binding=binding,
            path=path,
            entry=_reader_entry(entry),
            trust_existing_cache=False,
            cancel_event=threading.Event(),
            epoch=1,
        )

    assert exc_info.value is primary
    assert primary.__notes__ == [
        "also failed to remove requester demand: OSError: demand cleanup failed"
    ]


def test_stop_release_and_close_lifecycle_defenses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped, _binding = _controller(
        tmp_path / "stopped",
        command_pipe=_Pipe(send_error=BrokenPipeError()),
    )
    stopped.stop_worker()
    assert stopped.worker_is_terminal()
    stopped.stop_worker()

    survivor, _binding = _controller(
        tmp_path / "survivor",
        process=_Process(alive=True, survive_terminate=True),
    )
    with pytest.raises(RemoteTrainingError, match="survived bounded hard termination"):
        survivor.stop_worker()

    before_stop, _binding = _controller(tmp_path / "before-stop")
    with pytest.raises(RemoteTrainingError, match="before worker shutdown"):
        before_stop.release_session()

    released, _binding = _controller(tmp_path / "released")
    released._worker_stopped = True
    released._session_released = True
    released.release_session()

    missing_status, _binding = _controller(tmp_path / "missing-status")
    lock_path = missing_status.cache_root / "session.lock"
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    missing_status.session_lock_fd = lock_fd
    missing_status._worker_stopped = True
    with monkeypatch.context() as scoped:
        scoped.setattr(prefetch, "REMOTE_PREFETCH_GRACEFUL_JOIN_SECONDS", 0.01)
        with pytest.raises(RemoteTrainingError, match="another rank worker"):
            missing_status.release_session()
    os.close(lock_fd)
    missing_status.session_lock_fd = None

    release_error, _binding = _controller(tmp_path / "release-error")
    monkeypatch.setattr(release_error, "stop_worker", lambda: None)
    monkeypatch.setattr(release_error, "worker_is_terminal", lambda: True)
    monkeypatch.setattr(
        release_error,
        "release_session",
        lambda: (_ for _ in ()).throw(OSError("unlock failed")),
    )
    with pytest.raises(OSError, match="unlock failed"):
        release_error.close()


def test_worker_pipe_failures_do_not_mask_stop_or_transport_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "stopped"
    cache_root.mkdir()
    family = _family(cache_root, n_entries=1)
    cfg = _cfg(cache_root, (family,))
    control = prefetch._create_control_tree(
        cache_root=cache_root,
        nonce=cfg.nonce,
        world_size=1,
        families=(family,),
    )

    class FailSecondSend(_Pipe):
        def send(self, value: object) -> None:
            if self.sent:
                raise BrokenPipeError
            self.sent.append(value)

    stopped_status = FailSecondSend()
    prefetch._worker_main(
        cfg,
        cast(Any, _Pipe(({"op": "cancel"},))),
        cast(Any, stopped_status),
        lambda _bucket_id: object(),
    )
    first_status = cast(dict[str, Any], stopped_status.sent[0])
    assert first_status["state"] == "ready"
    stopped = prefetch._load_status(
        prefetch._status_path(control, 0),
        nonce=cfg.nonce,
        world_size=1,
    )
    assert stopped["state"] == "stopped"

    cache_root_2 = tmp_path / "error"
    cache_root_2.mkdir()
    family_2 = _family(cache_root_2, n_entries=1)
    cfg_2 = _cfg(cache_root_2, (family_2,))
    control_2 = prefetch._create_control_tree(
        cache_root=cache_root_2,
        nonce=cfg_2.nonce,
        world_size=1,
        families=(family_2,),
    )
    primary = RuntimeError("transport failed")

    def corrupt_then_fail(**_kwargs: Any):
        prefetch._status_path(control_2, 0).write_text("malformed", encoding="utf-8")
        raise primary

    prefetch._atomic_record(
        {
            "schema_version": prefetch.REMOTE_PREFETCH_SCHEMA_VERSION,
            "nonce": cfg_2.nonce,
            "rank": 0,
            "schedule_id": family_2.schedule_id,
            "epoch": 1,
            "inventory_index": 0,
            "generation": 0,
        },
        prefetch._cursor_path(control_2, 0),
    )
    prefetch._atomic_record(
        {
            "schema_version": prefetch.REMOTE_PREFETCH_SCHEMA_VERSION,
            "nonce": cfg_2.nonce,
            "requester_rank": 0,
            "schedule_id": family_2.schedule_id,
            "epoch": 1,
            "inventory_index": 0,
            "generation": 0,
            "created_ns": 1,
        },
        prefetch._demand_path(
            control_2,
            requester_rank=0,
            schedule_id=family_2.schedule_id,
            inventory_index=0,
        ),
    )
    monkeypatch.setattr(prefetch, "_publish_schedule", corrupt_then_fail)
    error_status = FailSecondSend()
    with pytest.raises(RuntimeError) as caught:
        prefetch._worker_main(
            cfg_2,
            cast(Any, _Pipe()),
            cast(Any, error_status),
            lambda _bucket_id: object(),
        )
    assert caught.value is primary
    assert any("recover active worker identity" in note for note in primary.__notes__)


def _start_materializer(cache_root: Path) -> RemoteSelectionMaterializer:
    selection_dir = cache_root / "selection"
    selection_dir.mkdir(parents=True)
    return _selection_materializer(selection_dir)


def _assert_session_lock_released(cache_root: Path) -> None:
    path = cache_root / ".remote-prefetch.session.lock"
    if not path.exists():
        return
    descriptor = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    "failure_point",
    [
        "session-lock",
        "nonce",
        "sentinel-publish",
        "sentinel-observe",
        "sentinel-remove",
        "canary-publish",
        "canary-remove",
        "control-create",
        "control-validate",
        "disk-admission",
        "resume",
        "rank-record",
        "spawn",
    ],
)
def test_startup_phase_failures_release_session_before_worker_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    materializer = _start_materializer(cache_root)
    remote = _remote_config()
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("WORLD_SIZE", "1")
    resume_state: dict[str, Any] | None = None
    expected = failure_point

    if failure_point == "session-lock":
        monkeypatch.setattr(
            prefetch,
            "_open_cache_lock",
            lambda _path: (_ for _ in ()).throw(OSError("session open failed")),
        )
    elif failure_point == "nonce":
        monkeypatch.setattr(prefetch, "_broadcast_object", lambda *_args, **_kwargs: "bad")
        expected = "run nonce"
    elif failure_point == "sentinel-publish":
        monkeypatch.setattr(
            prefetch,
            "atomic_write_bytes",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("sentinel write failed")),
        )
    elif failure_point == "sentinel-observe":
        monkeypatch.setattr(
            prefetch,
            "_read_direct_bytes",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("sentinel read failed")),
        )
    elif failure_point == "sentinel-remove":
        original_unlink = Path.unlink

        def fail_sentinel_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
            if ".remote-prefetch-sentinel." in path.name:
                raise OSError("sentinel remove failed")
            original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_sentinel_unlink)
    elif failure_point == "canary-publish":
        original_open = prefetch._open_cache_lock
        opens = 0

        def fail_second_lock(path: Path) -> int:
            nonlocal opens
            opens += 1
            if opens == 2:
                raise OSError("canary open failed")
            return original_open(path)

        monkeypatch.setattr(prefetch, "_open_cache_lock", fail_second_lock)
    elif failure_point == "canary-remove":
        original_unlink = Path.unlink

        def fail_canary_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
            if ".remote-prefetch-flock-canary." in path.name:
                raise OSError("canary remove failed")
            original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_canary_unlink)
    elif failure_point == "control-create":
        monkeypatch.setattr(
            prefetch,
            "_create_control_tree",
            lambda **_kwargs: (_ for _ in ()).throw(OSError("control create failed")),
        )
    elif failure_point == "control-validate":
        original_validate = prefetch._validate_control_tree
        validates = 0

        def fail_second_validate(**kwargs: Any) -> Path:
            nonlocal validates
            validates += 1
            if validates == 2:
                raise OSError("control validate failed")
            return original_validate(**kwargs)

        monkeypatch.setattr(prefetch, "_validate_control_tree", fail_second_validate)
    elif failure_point == "disk-admission":
        monkeypatch.setattr(
            prefetch,
            "calculate_remote_prefetch_disk_admission",
            lambda **_kwargs: prefetch.RemotePrefetchDiskAdmission(
                peak_bytes=1,
                current_managed_bytes=0,
                additional_required_bytes=1,
                available_bytes=0,
                safety_reserve_bytes=0,
                selection_final_bytes=1,
                packed_final_bytes=0,
                active_temporary_bytes=0,
                receipt_and_control_bytes=0,
            ),
        )
    elif failure_point == "resume":
        resume_state = {
            "epoch": False,
            "shard": materializer.local_dir.name,
        }
        expected = "resume-demand-parse"
    elif failure_point == "rank-record":
        original_atomic = prefetch._atomic_record

        def fail_cursor(value: dict[str, Any], path: Path) -> None:
            if path.parent.name == "cursors":
                raise OSError("cursor write failed")
            original_atomic(value, path)

        monkeypatch.setattr(prefetch, "_atomic_record", fail_cursor)
        expected = "rank-control-publish"
    elif failure_point == "spawn":
        monkeypatch.setattr(
            prefetch.mp,
            "get_context",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("spawn failed")),
        )
        expected = "worker-spawn"
    else:
        raise AssertionError(f"unhandled failure point {failure_point}")

    with pytest.raises((RemoteTrainingError, OSError), match=expected):
        RemoteBodyPrefetcher.start(
            remote=remote,
            cache_root=cache_root,
            rank=0,
            world_size=1,
            train_selection_dir=materializer.local_dir,
            selection_materializers=[materializer],
            packed_materializer=None,
            resume_state_dict=resume_state,
        )

    _assert_session_lock_released(cache_root)


def test_start_rejects_local_plan_and_cross_rank_plan_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("WORLD_SIZE", "1")
    remote = _remote_config()
    missing = tmp_path / "missing"
    with pytest.raises(RemoteTrainingError, match="local-plan"):
        RemoteBodyPrefetcher.start(
            remote=remote,
            cache_root=missing,
            rank=0,
            world_size=1,
            train_selection_dir=missing,
            selection_materializers=[],
            packed_materializer=None,
            resume_state_dict=None,
        )

    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    materializer = _start_materializer(cache_root)
    monkeypatch.setattr(
        prefetch,
        "_all_gather_object",
        lambda _value, **_kwargs: ["different-plan"],
    )
    with pytest.raises(RemoteTrainingError, match="plans differ"):
        RemoteBodyPrefetcher.start(
            remote=remote,
            cache_root=cache_root,
            rank=0,
            world_size=1,
            train_selection_dir=materializer.local_dir,
            selection_materializers=[materializer],
            packed_materializer=None,
            resume_state_dict=None,
        )


def test_start_rejects_nonshared_flock_namespace_from_nonzero_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    materializer = _start_materializer(cache_root)
    sentinel = b"s" * 32
    broadcasts = iter(("d" * 32, sentinel.hex()))
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(
        prefetch,
        "_broadcast_object",
        lambda *_args, **_kwargs: next(broadcasts),
    )
    monkeypatch.setattr(prefetch, "_read_direct_bytes", lambda *_args, **_kwargs: sentinel)
    monkeypatch.setattr(
        prefetch,
        "_all_gather_object",
        lambda value, **_kwargs: [value, value],
    )

    def raise_local_startup_error(
        *,
        phase: str,
        local_error: BaseException | None,
        world_size: int,
    ) -> None:
        del phase, world_size
        if local_error is not None:
            raise local_error

    monkeypatch.setattr(
        prefetch,
        "_raise_if_startup_phase_failed",
        raise_local_startup_error,
    )

    with pytest.raises(RemoteTrainingError, match="do not share one flock namespace"):
        RemoteBodyPrefetcher.start(
            remote=_remote_config(),
            cache_root=cache_root,
            rank=1,
            world_size=2,
            train_selection_dir=materializer.local_dir,
            selection_materializers=[materializer],
            packed_materializer=None,
            resume_state_dict=None,
        )


def test_start_propagates_nonzero_rank_flock_probe_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    materializer = _start_materializer(cache_root)
    sentinel = b"s" * 32
    broadcasts = iter(("d" * 32, sentinel.hex()))
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(
        prefetch,
        "_broadcast_object",
        lambda *_args, **_kwargs: next(broadcasts),
    )
    monkeypatch.setattr(prefetch, "_read_direct_bytes", lambda *_args, **_kwargs: sentinel)
    monkeypatch.setattr(
        prefetch,
        "_all_gather_object",
        lambda value, **_kwargs: [value, value],
    )
    original_open = prefetch._open_cache_lock

    def fail_canary_probe(path: Path) -> int:
        if ".remote-prefetch-flock-canary." in Path(path).name:
            raise OSError("probe open failed")
        return original_open(path)

    monkeypatch.setattr(prefetch, "_open_cache_lock", fail_canary_probe)

    def raise_local_startup_error(
        *,
        phase: str,
        local_error: BaseException | None,
        world_size: int,
    ) -> None:
        del phase, world_size
        if local_error is not None:
            raise local_error

    monkeypatch.setattr(
        prefetch,
        "_raise_if_startup_phase_failed",
        raise_local_startup_error,
    )

    with pytest.raises(OSError, match="probe open failed"):
        RemoteBodyPrefetcher.start(
            remote=_remote_config(),
            cache_root=cache_root,
            rank=1,
            world_size=2,
            train_selection_dir=materializer.local_dir,
            selection_materializers=[materializer],
            packed_materializer=None,
            resume_state_dict=None,
        )


def test_start_rejects_malformed_resume_demand_gather(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    materializer = _start_materializer(cache_root)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("WORLD_SIZE", "1")

    def malformed_resume_gather(value: Any, *, world_size: int) -> list[Any]:
        assert world_size == 1
        if isinstance(value, dict) and set(value) == {
            "rank",
            "epoch",
            "inventory_index",
            "demands",
        }:
            return []
        return [value]

    monkeypatch.setattr(prefetch, "_all_gather_object", malformed_resume_gather)
    with pytest.raises(RemoteTrainingError, match="resume-demand-gather"):
        RemoteBodyPrefetcher.start(
            remote=_remote_config(),
            cache_root=cache_root,
            rank=0,
            world_size=1,
            train_selection_dir=materializer.local_dir,
            selection_materializers=[materializer],
            packed_materializer=None,
            resume_state_dict=None,
        )

    _assert_session_lock_released(cache_root)


@pytest.mark.parametrize(
    "failure_mode",
    ["worker-exit", "status-poll", "malformed-ready"],
)
def test_start_worker_readiness_failures_are_symmetric_and_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    materializer = _start_materializer(cache_root)
    if failure_mode == "worker-exit":
        process = _StartProcess(alive=False, exitcode=17)
        parent_status = _StartPipe()
        parent_command = _StartPipe()
    elif failure_mode == "status-poll":
        process = _StartProcess(alive=True)
        parent_status = _StartPipe(poll_error=OSError("status poll failed"))
        parent_command = _StartPipe(send_error=BrokenPipeError())
    else:
        process = _StartProcess(alive=True)
        parent_status = _StartPipe(({"state": "ready", "rank": 99},))
        parent_command = _StartPipe()
    context = _StartContext(
        process=process,
        parent_command=parent_command,
        parent_status=parent_status,
    )
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(prefetch.mp, "get_context", lambda *_args: context)

    expected = "worker-ready" if failure_mode == "status-poll" else "did not start"
    with pytest.raises(RemoteTrainingError, match=expected):
        RemoteBodyPrefetcher.start(
            remote=_remote_config(),
            cache_root=cache_root,
            rank=0,
            world_size=1,
            train_selection_dir=materializer.local_dir,
            selection_materializers=[materializer],
            packed_materializer=None,
            resume_state_dict=None,
        )

    assert process.started
    assert not process.is_alive()
    assert parent_command.closed
    assert parent_status.closed
    _assert_session_lock_released(cache_root)


def test_start_reader_attach_failure_stops_worker_and_releases_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    materializer = _start_materializer(cache_root)
    materializer.attach_body_prefetch(object(), object())
    process = _StartProcess(alive=True)
    context = _StartContext(
        process=process,
        parent_status=_StartPipe(({"state": "ready", "rank": 0},)),
    )
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(prefetch.mp, "get_context", lambda *_args: context)

    with pytest.raises(RemoteTrainingError, match="reader-attach"):
        RemoteBodyPrefetcher.start(
            remote=_remote_config(),
            cache_root=cache_root,
            rank=0,
            world_size=1,
            train_selection_dir=materializer.local_dir,
            selection_materializers=[materializer],
            packed_materializer=None,
            resume_state_dict=None,
        )

    assert process.terminate_calls == 1
    assert not process.is_alive()
    _assert_session_lock_released(cache_root)


@pytest.mark.parametrize("status_kind", ["terminal", "malformed"])
def test_start_cleanup_handles_preexisting_worker_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status_kind: str,
) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    materializer = _start_materializer(cache_root)

    def publish_status(process: _StartProcess) -> None:
        cfg = cast(_WorkerConfig, process.args[0])
        if status_kind == "terminal":
            prefetch._write_worker_status(
                cfg,
                state="error",
                error=RuntimeError("child failed before ready"),
            )
        else:
            path = prefetch._status_path(
                prefetch._control_root(Path(cfg.cache_root), cfg.nonce),
                cfg.rank,
            )
            path.write_text("malformed", encoding="utf-8")

    process = _StartProcess(
        alive=False,
        exitcode=19,
        on_start=publish_status,
    )
    context = _StartContext(process=process)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(prefetch.mp, "get_context", lambda *_args: context)

    with pytest.raises(RemoteTrainingError, match="did not start"):
        RemoteBodyPrefetcher.start(
            remote=_remote_config(),
            cache_root=cache_root,
            rank=0,
            world_size=1,
            train_selection_dir=materializer.local_dir,
            selection_materializers=[materializer],
            packed_materializer=None,
            resume_state_dict=None,
        )

    _assert_session_lock_released(cache_root)


def test_start_cleanup_preserves_status_pipe_and_missing_status_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    materializer = _start_materializer(cache_root)
    process = _StartProcess(alive=False, exitcode=23)
    parent_command = _StartPipe(close_error=OSError("command close failed"))
    parent_status = _StartPipe(close_error=OSError("status close failed"))
    context = _StartContext(
        process=process,
        parent_command=parent_command,
        parent_status=parent_status,
    )
    original_atomic = prefetch._atomic_record
    original_open = prefetch._open_cache_lock
    session_fds: list[int] = []

    def fail_status_record(value: dict[str, Any], path: Path) -> None:
        if path.parent.name == "status":
            raise OSError("status persistence failed")
        original_atomic(value, path)

    def track_session(path: Path) -> int:
        descriptor = original_open(path)
        if Path(path).name == ".remote-prefetch.session.lock":
            session_fds.append(descriptor)
        return descriptor

    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(prefetch.mp, "get_context", lambda *_args: context)
    monkeypatch.setattr(prefetch, "_atomic_record", fail_status_record)
    monkeypatch.setattr(prefetch, "_open_cache_lock", track_session)
    monkeypatch.setattr(prefetch, "REMOTE_PREFETCH_GRACEFUL_JOIN_SECONDS", 0.005)
    monkeypatch.setattr(prefetch, "REMOTE_PREFETCH_POLL_SECONDS", 0.001)

    with pytest.raises(RemoteTrainingError, match="did not start") as caught:
        RemoteBodyPrefetcher.start(
            remote=_remote_config(),
            cache_root=cache_root,
            rank=0,
            world_size=1,
            train_selection_dir=materializer.local_dir,
            selection_materializers=[materializer],
            packed_materializer=None,
            resume_state_dict=None,
        )

    notes = caught.value.__notes__
    assert any("status persistence failed" in note for note in notes)
    assert any("command close failed" in note for note in notes)
    assert any("status close failed" in note for note in notes)
    assert any("session lock retained" in note for note in notes)
    assert len(session_fds) == 1
    fcntl.flock(session_fds[0], fcntl.LOCK_UN)
    os.close(session_fds[0])
    _assert_session_lock_released(cache_root)


def test_start_cleanup_retains_session_for_nonterminal_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    materializer = _start_materializer(cache_root)

    def publish_idle(process: _StartProcess) -> None:
        prefetch._write_worker_status(
            cast(_WorkerConfig, process.args[0]),
            state="idle",
        )

    process = _StartProcess(
        alive=False,
        exitcode=29,
        on_start=publish_idle,
    )
    context = _StartContext(process=process)
    original_atomic = prefetch._atomic_record
    original_open = prefetch._open_cache_lock
    session_fds: list[int] = []

    def preserve_idle(value: dict[str, Any], path: Path) -> None:
        if path.parent.name != "status" or not path.exists():
            original_atomic(value, path)

    def track_session(path: Path) -> int:
        descriptor = original_open(path)
        if Path(path).name == ".remote-prefetch.session.lock":
            session_fds.append(descriptor)
        return descriptor

    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(prefetch.mp, "get_context", lambda *_args: context)
    monkeypatch.setattr(prefetch, "_atomic_record", preserve_idle)
    monkeypatch.setattr(prefetch, "_open_cache_lock", track_session)
    monkeypatch.setattr(prefetch, "REMOTE_PREFETCH_GRACEFUL_JOIN_SECONDS", 0.05)
    monkeypatch.setattr(prefetch, "REMOTE_PREFETCH_POLL_SECONDS", 0.001)

    with pytest.raises(RemoteTrainingError, match="did not start") as caught:
        RemoteBodyPrefetcher.start(
            remote=_remote_config(),
            cache_root=cache_root,
            rank=0,
            world_size=1,
            train_selection_dir=materializer.local_dir,
            selection_materializers=[materializer],
            packed_materializer=None,
            resume_state_dict=None,
        )

    assert any("session lock retained" in note for note in caught.value.__notes__)
    assert len(session_fds) == 1
    fcntl.flock(session_fds[0], fcntl.LOCK_UN)
    os.close(session_fds[0])
    _assert_session_lock_released(cache_root)


@pytest.mark.parametrize("cleanup_mode", ["survivor", "join-error"])
def test_start_cleanup_retains_session_when_worker_is_not_proven_dead(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_mode: str,
) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    materializer = _start_materializer(cache_root)
    process = _StartProcess(
        alive=True,
        survive_terminate=cleanup_mode == "survivor",
        join_error=OSError("join failed") if cleanup_mode == "join-error" else None,
    )
    context = _StartContext(
        process=process,
        parent_status=_StartPipe(({"state": "not-ready"},)),
    )
    original_open = prefetch._open_cache_lock
    session_fds: list[int] = []

    def track_session(path: Path) -> int:
        descriptor = original_open(path)
        if Path(path).name == ".remote-prefetch.session.lock":
            session_fds.append(descriptor)
        return descriptor

    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(prefetch.mp, "get_context", lambda *_args: context)
    monkeypatch.setattr(prefetch, "_open_cache_lock", track_session)
    monkeypatch.setattr(prefetch, "REMOTE_PREFETCH_GRACEFUL_JOIN_SECONDS", 0.0)

    with pytest.raises(RemoteTrainingError, match="did not start") as caught:
        RemoteBodyPrefetcher.start(
            remote=_remote_config(),
            cache_root=cache_root,
            rank=0,
            world_size=1,
            train_selection_dir=materializer.local_dir,
            selection_materializers=[materializer],
            packed_materializer=None,
            resume_state_dict=None,
        )

    if cleanup_mode == "survivor":
        assert any("survived bounded termination" in note for note in caught.value.__notes__)
    else:
        assert any("join failed" in note for note in caught.value.__notes__)
    assert any("session lock retained" in note for note in caught.value.__notes__)
    assert len(session_fds) == 1
    fcntl.flock(session_fds[0], fcntl.LOCK_UN)
    os.close(session_fds[0])
    _assert_session_lock_released(cache_root)


def test_start_cleanup_preserves_canary_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    materializer = _start_materializer(cache_root)
    original_open = prefetch._open_cache_lock
    original_close = os.close
    canary_fds: list[int] = []

    def track_canary(path: Path) -> int:
        descriptor = original_open(path)
        if ".remote-prefetch-flock-canary." in Path(path).name:
            canary_fds.append(descriptor)
        return descriptor

    def fail_canary_close(descriptor: int) -> None:
        if canary_fds and descriptor == canary_fds[0]:
            raise OSError("canary close failed")
        original_close(descriptor)

    def reject_flock_gather(value: Any, *, world_size: int) -> list[Any]:
        assert world_size == 1
        if type(value) is bool:
            return [False]
        return [value]

    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(prefetch, "_open_cache_lock", track_canary)
    monkeypatch.setattr(prefetch.os, "close", fail_canary_close)
    monkeypatch.setattr(prefetch, "_all_gather_object", reject_flock_gather)

    with pytest.raises(RemoteTrainingError, match="do not share one flock namespace") as caught:
        RemoteBodyPrefetcher.start(
            remote=_remote_config(),
            cache_root=cache_root,
            rank=0,
            world_size=1,
            train_selection_dir=materializer.local_dir,
            selection_materializers=[materializer],
            packed_materializer=None,
            resume_state_dict=None,
        )

    assert any("canary close failed" in note for note in caught.value.__notes__)
    assert len(canary_fds) == 1
    fcntl.flock(canary_fds[0], fcntl.LOCK_UN)
    original_close(canary_fds[0])
    canary_fds.clear()
    _assert_session_lock_released(cache_root)


def test_start_cleanup_preserves_session_unlock_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    materializer = _start_materializer(cache_root)
    process = _StartProcess(alive=False, exitcode=31)
    context = _StartContext(process=process)
    original_open = prefetch._open_cache_lock
    original_flock = fcntl.flock
    session_fds: list[int] = []

    def track_session(path: Path) -> int:
        descriptor = original_open(path)
        if Path(path).name == ".remote-prefetch.session.lock":
            session_fds.append(descriptor)
        return descriptor

    def fail_session_unlock(descriptor: int, operation: int) -> None:
        if session_fds and descriptor == session_fds[0] and operation == fcntl.LOCK_UN:
            raise OSError("session unlock failed")
        original_flock(descriptor, operation)

    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(prefetch.mp, "get_context", lambda *_args: context)
    monkeypatch.setattr(prefetch, "_open_cache_lock", track_session)
    monkeypatch.setattr(prefetch.fcntl, "flock", fail_session_unlock)

    with pytest.raises(RemoteTrainingError, match="did not start") as caught:
        RemoteBodyPrefetcher.start(
            remote=_remote_config(),
            cache_root=cache_root,
            rank=0,
            world_size=1,
            train_selection_dir=materializer.local_dir,
            selection_materializers=[materializer],
            packed_materializer=None,
            resume_state_dict=None,
        )

    assert any("session unlock failed" in note for note in caught.value.__notes__)
    assert len(session_fds) == 1
    original_flock(session_fds[0], fcntl.LOCK_UN)
    os.close(session_fds[0])
    session_fds.clear()
    _assert_session_lock_released(cache_root)


def test_read_body_cancellation_interrupts_descriptor_lock_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    family = _family(cache_root, n_entries=1)
    entry = family.entries[0]
    final = Path(family.local_dir) / entry.name
    final.write_bytes(_payload("selection", 0))
    descriptor = prefetch._descriptor_dict(prefetch._direct_file_stat(final))
    family = replace(
        family,
        preexisting_descriptors=(
            (
                entry.name,
                prefetch._descriptor_dict_to_tuple(descriptor),
            ),
        ),
    )
    controller, binding = _controller(cache_root, families=(family,))
    lock_fd = prefetch._open_cache_lock(Path(family.local_dir) / f".{entry.name}.lock")
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    blocked = threading.Event()
    cancel = threading.Event()
    result: list[BaseException] = []
    original_sleep = prefetch.time.sleep

    def observe_lock_wait(seconds: float) -> None:
        blocked.set()
        original_sleep(min(seconds, 0.001))

    def read() -> None:
        try:
            controller.read_body(
                binding=binding,
                path=final,
                entry=_reader_entry(entry),
                trust_existing_cache=False,
                cancel_event=cancel,
                epoch=1,
            )
        except BaseException as exc:
            result.append(exc)

    monkeypatch.setattr(prefetch.time, "sleep", observe_lock_wait)
    thread = threading.Thread(target=read)
    thread.start()
    try:
        assert blocked.wait(timeout=1)
        cancel.set()
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert len(result) == 1
        assert isinstance(result[0], RemoteMaterializationCancelled)
        assert "before descriptor read" in str(result[0])
    finally:
        cancel.set()
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
        thread.join(timeout=1)


def test_release_session_refuses_nonterminal_status_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "cache"
    lock_path = cache_root / "session.lock"
    cache_root.mkdir()
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    controller, _binding = _controller(
        cache_root,
        session_lock_fd=lock_fd,
    )
    controller._worker_stopped = True
    prefetch._write_worker_status(controller._worker_cfg, state="idle")
    monkeypatch.setattr(prefetch, "REMOTE_PREFETCH_GRACEFUL_JOIN_SECONDS", 0.003)
    monkeypatch.setattr(prefetch, "REMOTE_PREFETCH_POLL_SECONDS", 0.001)

    with pytest.raises(RemoteTrainingError, match="another rank worker"):
        controller.release_session()

    assert controller.session_lock_fd == lock_fd
    os.close(lock_fd)
    controller.session_lock_fd = None


def test_direct_remote_config_requires_prefetch() -> None:
    remote = RemoteTrainingConfig.parse(
        {
            "bucket_id": "owner/bucket",
            "body_prefetch": {"lookahead_shards_per_rank": 2},
            "selection_prefix": "features",
            "checkpoint_prefix": "checkpoints",
            "selection_wait_timeout_seconds": 1,
            "checkpoint_retry_timeout_seconds": 1,
            "retry_interval_seconds": 0.001,
            "resume_latest": False,
        }
    )
    assert remote is not None
    with pytest.raises(BeartypeCallHintParamViolation):
        replace(remote, body_prefetch=None)
