from __future__ import annotations

import fcntl
import hashlib
import multiprocessing as mp
import os
import threading
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import hf_xet
import pytest

import steering.remote_body_prefetch as prefetch
from steering.remote_body_prefetch import (
    RemoteBodyEntry,
    RemoteBodyFamily,
    _create_control_tree,
    _publish_schedule,
    _read_direct_bytes,
    _ready_path,
    _receipt_path,
    _WorkerConfig,
)
from steering.remote_training import RemoteObjectInfo, RemoteTrainingError


def _payload(kind: str) -> bytes:
    return f"{kind}-signed-body".encode()


def _entry(kind: str) -> RemoteBodyEntry:
    name = "shard_00000.safetensors.zst"
    payload = _payload(kind)
    fingerprint: str | None = None
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


def _families(
    tmp_path: Path,
    *,
    paired: bool,
) -> tuple[Path, tuple[RemoteBodyFamily, ...]]:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    schedule_id = "c" * 64
    selection_dir = cache_root / "selection"
    selection_dir.mkdir()
    selection = RemoteBodyFamily(
        family_id="1" * 64,
        family_aggregate="a" * 64,
        kind="selection",
        local_dir=str(selection_dir),
        remote_dir="features/selection",
        entries=(_entry("selection"),),
        preexisting_descriptors=(),
        schedule_id=schedule_id,
        train_schedule=True,
    )
    families = [selection]
    if paired:
        packed_dir = cache_root / "packed"
        packed_dir.mkdir()
        families.append(
            RemoteBodyFamily(
                family_id="2" * 64,
                family_aggregate="b" * 64,
                kind="packed",
                local_dir=str(packed_dir),
                remote_dir="features/packed",
                entries=(_entry("packed"),),
                preexisting_descriptors=(),
                schedule_id=schedule_id,
                train_schedule=True,
            )
        )
    return cache_root, tuple(families)


def _worker_config(
    *,
    cache_root: Path,
    families: tuple[RemoteBodyFamily, ...],
    nonce: str,
) -> _WorkerConfig:
    return _WorkerConfig(
        bucket_id="owner/bucket",
        cache_root=str(cache_root),
        nonce=nonce,
        rank=0,
        world_size=1,
        lookahead_shards_per_rank=2,
        cache_shards=2,
        timeout_seconds=1,
        retry_interval_seconds=0.001,
        families=families,
    )


class _StageOnlyBackend:
    def __init__(self, remote_root: Path) -> None:
        self.remote_root = remote_root

    def stat(self, remote_path: str) -> RemoteObjectInfo | None:
        path = self.remote_root / remote_path
        if not path.is_file():
            return None
        xet_hash = str(cast(Any, hf_xet).hash_files([str(path)])[0].hash)
        return RemoteObjectInfo(
            path=remote_path,
            byte_length=path.stat().st_size,
            xet_hash=xet_hash,
        )

    def download(self, remote_path: str, local_path: Path) -> None:
        local_path.write_bytes((self.remote_root / remote_path).read_bytes())


def _setup_publish(
    tmp_path: Path,
    *,
    paired: bool,
) -> tuple[
    Path,
    Path,
    tuple[RemoteBodyFamily, ...],
    _WorkerConfig,
    _StageOnlyBackend,
]:
    cache_root, families = _families(tmp_path, paired=paired)
    nonce = "d" * 32
    control = _create_control_tree(
        cache_root=cache_root,
        nonce=nonce,
        world_size=1,
        families=families,
    )
    remote_root = tmp_path / "remote"
    for family in families:
        entry = family.entries[0]
        remote_path = remote_root / family.remote_dir / entry.name
        remote_path.parent.mkdir(parents=True, exist_ok=True)
        remote_path.write_bytes(_payload(family.kind))
    return (
        cache_root,
        control,
        families,
        _worker_config(cache_root=cache_root, families=families, nonce=nonce),
        _StageOnlyBackend(remote_root),
    )


def _assert_no_final_receipt_or_ready(
    *,
    control: Path,
    families: Sequence[RemoteBodyFamily],
) -> None:
    for family in families:
        entry = family.entries[0]
        assert not (Path(family.local_dir) / entry.name).exists()
        assert not _receipt_path(
            control,
            family_id=family.family_id,
            inventory_index=0,
        ).exists()
    assert not _ready_path(
        control,
        schedule_id=families[0].schedule_id,
        inventory_index=0,
    ).exists()


def _corrupt_same_length(data: bytes) -> bytes:
    assert data
    return bytes([data[0] ^ 0xFF]) + data[1:]


def test_current_ready_schedule_is_not_republished(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cache_root, control, families, cfg, backend = _setup_publish(
        tmp_path,
        paired=True,
    )
    child_command, parent_command = mp.get_context("spawn").Pipe(duplex=False)
    try:
        _publish_schedule(
            cfg=cfg,
            backend=cast(Any, backend),
            schedules=prefetch._schedule_map(families),
            schedule_id=families[0].schedule_id,
            inventory_index=0,
            generation=0,
            command_pipe=child_command,
        )
        ready = _ready_path(
            control,
            schedule_id=families[0].schedule_id,
            inventory_index=0,
        )
        ready_inode = ready.stat().st_ino

        def fail_republication(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("a current ready schedule must not be republished")

        monkeypatch.setattr(prefetch, "_publish_component", fail_republication)
        _publish_schedule(
            cfg=cfg,
            backend=cast(Any, backend),
            schedules=prefetch._schedule_map(families),
            schedule_id=families[0].schedule_id,
            inventory_index=0,
            generation=0,
            command_pipe=child_command,
        )
    finally:
        child_command.close()
        parent_command.close()

    assert ready.stat().st_ino == ready_inode


def test_completed_download_is_published_without_reopening_or_verifying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cache_root, control, families, cfg, backend = _setup_publish(
        tmp_path,
        paired=False,
    )
    family = families[0]

    def download_without_contract_check(_remote_path: str, local_path: Path) -> None:
        local_path.write_bytes(b"intentionally not the signed payload")

    def fail_verification(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("completed remote bodies must not be reopened for verification")

    monkeypatch.setattr(backend, "download", download_without_contract_check)
    monkeypatch.setattr(
        "steering.remote_training._stable_file_identity",
        fail_verification,
    )
    child_command, parent_command = mp.get_context("spawn").Pipe(duplex=False)
    try:
        publication = prefetch._publish_component(
            cfg=cfg,
            backend=cast(Any, backend),
            family=family,
            inventory_index=0,
            generation=0,
            command_pipe=child_command,
        )
    finally:
        child_command.close()
        parent_command.close()

    assert publication is not None
    final = Path(family.local_dir) / family.entries[0].name
    assert final.read_bytes() == b"intentionally not the signed payload"
    assert _receipt_path(
        control,
        family_id=family.family_id,
        inventory_index=0,
    ).is_file()


@pytest.mark.parametrize("target_kind", ["selection", "packed"])
def test_final_directory_fsync_failure_rolls_back_entire_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    _cache_root, control, families, cfg, backend = _setup_publish(
        tmp_path,
        paired=True,
    )
    target = next(family for family in families if family.kind == target_kind)
    target_dir = Path(target.local_dir)
    target_final = target_dir / target.entries[0].name
    original_fsync_directory = prefetch._fsync_directory
    failures = 0

    def fail_once_after_final_rename(path: Path) -> None:
        nonlocal failures
        if Path(path) == target_dir and target_final.exists() and failures == 0:
            failures += 1
            raise OSError("injected final-directory fsync failure")
        original_fsync_directory(path)

    monkeypatch.setattr(prefetch, "_fsync_directory", fail_once_after_final_rename)
    child_command, parent_command = mp.get_context("spawn").Pipe(duplex=False)
    try:
        with pytest.raises(OSError, match="injected final-directory fsync failure"):
            _publish_schedule(
                cfg=cfg,
                backend=cast(Any, backend),
                schedules=prefetch._schedule_map(families),
                schedule_id=families[0].schedule_id,
                inventory_index=0,
                generation=0,
                command_pipe=child_command,
            )
    finally:
        child_command.close()
        parent_command.close()

    assert failures == 1
    _assert_no_final_receipt_or_ready(control=control, families=families)


def test_ready_cleanup_fsync_failure_preserves_primary_and_rolls_back_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cache_root, control, families, cfg, backend = _setup_publish(
        tmp_path,
        paired=True,
    )
    ready = _ready_path(
        control,
        schedule_id=families[0].schedule_id,
        inventory_index=0,
    )
    original_atomic_record = prefetch._atomic_record
    original_fsync_directory = prefetch._fsync_directory
    ready_written = False
    cleanup_fsync_failures = 0

    def fail_after_ready_write(value: dict[str, Any], path: Path) -> None:
        nonlocal ready_written
        original_atomic_record(value, path)
        if Path(path) == ready:
            ready_written = True
            raise RuntimeError("injected primary ready-publication failure")

    def fail_ready_cleanup_fsync_once(path: Path) -> None:
        nonlocal cleanup_fsync_failures
        if Path(path) == ready.parent and ready_written and cleanup_fsync_failures == 0:
            cleanup_fsync_failures += 1
            raise OSError("injected ready-cleanup fsync failure")
        original_fsync_directory(path)

    monkeypatch.setattr(prefetch, "_atomic_record", fail_after_ready_write)
    monkeypatch.setattr(prefetch, "_fsync_directory", fail_ready_cleanup_fsync_once)
    child_command, parent_command = mp.get_context("spawn").Pipe(duplex=False)
    try:
        with pytest.raises(
            RuntimeError,
            match="injected primary ready-publication failure",
        ) as caught:
            _publish_schedule(
                cfg=cfg,
                backend=cast(Any, backend),
                schedules=prefetch._schedule_map(families),
                schedule_id=families[0].schedule_id,
                inventory_index=0,
                generation=0,
                command_pipe=child_command,
            )
    finally:
        child_command.close()
        parent_command.close()

    assert ready_written
    assert cleanup_fsync_failures == 1
    notes = getattr(caught.value, "__notes__", ())
    assert any("injected ready-cleanup fsync failure" in note for note in notes)
    _assert_no_final_receipt_or_ready(control=control, families=families)


def test_cancellation_during_receipt_atomic_write_rolls_back_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cache_root, control, families, cfg, backend = _setup_publish(
        tmp_path,
        paired=False,
    )
    receipt = _receipt_path(
        control,
        family_id=families[0].family_id,
        inventory_index=0,
    )
    original_atomic_record = prefetch._atomic_record
    child_command, parent_command = mp.get_context("spawn").Pipe(duplex=False)
    cancellations = 0

    def cancel_after_receipt_write(value: dict[str, Any], path: Path) -> None:
        nonlocal cancellations
        original_atomic_record(value, path)
        if Path(path) == receipt and cancellations == 0:
            parent_command.send({"op": "cancel"})
            cancellations += 1

    monkeypatch.setattr(prefetch, "_atomic_record", cancel_after_receipt_write)
    try:
        with pytest.raises(prefetch._WorkerCancelled):
            _publish_schedule(
                cfg=cfg,
                backend=cast(Any, backend),
                schedules=prefetch._schedule_map(families),
                schedule_id=families[0].schedule_id,
                inventory_index=0,
                generation=0,
                command_pipe=child_command,
            )
    finally:
        child_command.close()
        parent_command.close()

    assert cancellations == 1
    _assert_no_final_receipt_or_ready(control=control, families=families)


def test_cancellation_during_ready_atomic_write_rolls_back_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cache_root, control, families, cfg, backend = _setup_publish(
        tmp_path,
        paired=True,
    )
    ready = _ready_path(
        control,
        schedule_id=families[0].schedule_id,
        inventory_index=0,
    )
    original_atomic_record = prefetch._atomic_record
    child_command, parent_command = mp.get_context("spawn").Pipe(duplex=False)
    cancellations = 0

    def cancel_after_ready_write(value: dict[str, Any], path: Path) -> None:
        nonlocal cancellations
        original_atomic_record(value, path)
        if Path(path) == ready and cancellations == 0:
            parent_command.send({"op": "cancel"})
            cancellations += 1

    monkeypatch.setattr(prefetch, "_atomic_record", cancel_after_ready_write)
    try:
        with pytest.raises(prefetch._WorkerCancelled):
            _publish_schedule(
                cfg=cfg,
                backend=cast(Any, backend),
                schedules=prefetch._schedule_map(families),
                schedule_id=families[0].schedule_id,
                inventory_index=0,
                generation=0,
                command_pipe=child_command,
            )
    finally:
        child_command.close()
        parent_command.close()

    assert cancellations == 1
    _assert_no_final_receipt_or_ready(control=control, families=families)


def test_cancellation_during_receipt_generation_refresh_restores_prior_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cache_root, control, families, cfg, backend = _setup_publish(
        tmp_path,
        paired=False,
    )
    receipt = _receipt_path(
        control,
        family_id=families[0].family_id,
        inventory_index=0,
    )
    initial_child, initial_parent = mp.get_context("spawn").Pipe(duplex=False)
    try:
        _publish_schedule(
            cfg=cfg,
            backend=cast(Any, backend),
            schedules=prefetch._schedule_map(families),
            schedule_id=families[0].schedule_id,
            inventory_index=0,
            generation=0,
            command_pipe=initial_child,
        )
    finally:
        initial_child.close()
        initial_parent.close()
    prior_receipt = receipt.read_bytes()

    original_atomic_record = prefetch._atomic_record
    child_command, parent_command = mp.get_context("spawn").Pipe(duplex=False)
    cancellations = 0

    def cancel_after_refresh(value: dict[str, Any], path: Path) -> None:
        nonlocal cancellations
        original_atomic_record(value, path)
        if Path(path) == receipt and value.get("generation") == 1 and cancellations == 0:
            parent_command.send({"op": "cancel"})
            cancellations += 1

    monkeypatch.setattr(prefetch, "_atomic_record", cancel_after_refresh)
    try:
        with pytest.raises(prefetch._WorkerCancelled):
            _publish_schedule(
                cfg=cfg,
                backend=cast(Any, backend),
                schedules=prefetch._schedule_map(families),
                schedule_id=families[0].schedule_id,
                inventory_index=0,
                generation=1,
                command_pipe=child_command,
            )
    finally:
        child_command.close()
        parent_command.close()

    assert cancellations == 1
    assert receipt.read_bytes() == prior_receipt
    final = Path(families[0].local_dir) / families[0].entries[0].name
    assert final.read_bytes() == _payload("selection")


def test_worker_preserves_transport_error_when_terminal_status_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cache_root, control, families, cfg, backend = _setup_publish(
        tmp_path,
        paired=False,
    )
    family = families[0]
    prefetch._atomic_record(
        {
            "schema_version": prefetch.REMOTE_PREFETCH_SCHEMA_VERSION,
            "nonce": cfg.nonce,
            "rank": 0,
            "schedule_id": family.schedule_id,
            "epoch": 1,
            "inventory_index": 0,
            "generation": 0,
        },
        prefetch._cursor_path(control, 0),
    )
    prefetch._atomic_record(
        {
            "schema_version": prefetch.REMOTE_PREFETCH_SCHEMA_VERSION,
            "nonce": cfg.nonce,
            "rank": 0,
            "schedule_id": family.schedule_id,
            "generation": -1,
        },
        prefetch._progress_path(control, 0),
    )
    prefetch._atomic_record(
        {
            "schema_version": prefetch.REMOTE_PREFETCH_SCHEMA_VERSION,
            "nonce": cfg.nonce,
            "requester_rank": 0,
            "schedule_id": family.schedule_id,
            "epoch": 1,
            "inventory_index": 0,
            "generation": 0,
            "created_ns": 1,
        },
        prefetch._demand_path(
            control,
            requester_rank=0,
            schedule_id=family.schedule_id,
            inventory_index=0,
        ),
    )
    primary = RuntimeError("owner transport failed")
    status_error = OSError("terminal status fsync failed")
    original_write_status = prefetch._write_worker_status

    def fail_schedule(**_kwargs: Any) -> None:
        original_write_status(
            cfg,
            state="staging",
            family_id=family.family_id,
            inventory_index=0,
            body_name=family.entries[0].name,
        )
        raise primary

    def fail_terminal_status(*args: Any, **kwargs: Any) -> None:
        if kwargs.get("state") == "error":
            raise status_error
        original_write_status(*args, **kwargs)

    monkeypatch.setattr(prefetch, "_publish_schedule", fail_schedule)
    monkeypatch.setattr(prefetch, "_write_worker_status", fail_terminal_status)
    child_command, parent_command = mp.get_context("spawn").Pipe(duplex=False)
    parent_status, child_status = mp.get_context("spawn").Pipe(duplex=False)
    try:
        with pytest.raises(RuntimeError) as exc_info:
            prefetch._worker_main(
                cfg,
                child_command,
                child_status,
                lambda _bucket_id: backend,
            )
        assert exc_info.value is primary
        assert primary.__notes__ == [
            "also failed to persist terminal worker status: OSError: terminal status fsync failed"
        ]
        assert parent_status.recv()["state"] == "ready"
        terminal_message = parent_status.recv()
        assert terminal_message["state"] == "error"
        assert terminal_message["family_id"] == family.family_id
        assert terminal_message["inventory_index"] == 0
        assert terminal_message["body_name"] == family.entries[0].name
        assert terminal_message["error_message"] == "owner transport failed"
    finally:
        child_command.close()
        parent_command.close()
        parent_status.close()
        child_status.close()


def test_worker_error_text_is_strictly_bounded() -> None:
    text = prefetch._bounded_error_text(RuntimeError("x" * 3_000))
    assert text == "x" * 2_000 + "... <1000 error characters omitted>"


def test_rollback_unknown_publication_family_is_rejected(tmp_path: Path) -> None:
    cache_root, _control, families, cfg, _backend = _setup_publish(
        tmp_path,
        paired=False,
    )
    publication = prefetch._NewPublication(
        family_id="9" * 64,
        inventory_index=0,
    )
    with pytest.raises(RemoteTrainingError, match="unknown remote family"):
        prefetch._rollback_new_publication(
            cfg=cfg,
            publication=publication,
        )
    assert cache_root.exists()
    assert families


def test_component_rejects_nonowner_and_unreceipted_final(
    tmp_path: Path,
) -> None:
    _cache_root, _control, families, cfg, backend = _setup_publish(
        tmp_path,
        paired=False,
    )
    family = families[0]
    nonowner_cfg = replace(cfg, rank=1, world_size=2)
    child_command, parent_command = mp.get_context("spawn").Pipe(duplex=False)
    try:
        with pytest.raises(RemoteTrainingError, match="non-owned"):
            prefetch._publish_component(
                cfg=nonowner_cfg,
                backend=cast(Any, backend),
                family=family,
                inventory_index=0,
                generation=0,
                command_pipe=child_command,
            )

        final = Path(family.local_dir) / family.entries[0].name
        final.write_bytes(_payload(family.kind))
        with pytest.raises(RemoteTrainingError, match="without its exact"):
            prefetch._publish_component(
                cfg=cfg,
                backend=cast(Any, backend),
                family=family,
                inventory_index=0,
                generation=0,
                command_pipe=child_command,
            )
    finally:
        child_command.close()
        parent_command.close()


def test_receipt_refresh_reports_restore_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cache_root, control, families, cfg, backend = _setup_publish(
        tmp_path,
        paired=False,
    )
    receipt = _receipt_path(
        control,
        family_id=families[0].family_id,
        inventory_index=0,
    )
    initial_child, initial_parent = mp.get_context("spawn").Pipe(duplex=False)
    try:
        _publish_schedule(
            cfg=cfg,
            backend=cast(Any, backend),
            schedules=prefetch._schedule_map(families),
            schedule_id=families[0].schedule_id,
            inventory_index=0,
            generation=0,
            command_pipe=initial_child,
        )
    finally:
        initial_child.close()
        initial_parent.close()

    original_atomic = prefetch._atomic_record
    child_command, parent_command = mp.get_context("spawn").Pipe(duplex=False)
    writes = 0

    def cancel_then_fail_restore(value: dict[str, Any], path: Path) -> None:
        nonlocal writes
        if Path(path) == receipt:
            writes += 1
            if writes == 1:
                original_atomic(value, path)
                parent_command.send({"op": "cancel"})
                return
            raise OSError("receipt restore failed")
        original_atomic(value, path)

    monkeypatch.setattr(prefetch, "_atomic_record", cancel_then_fail_restore)
    try:
        with pytest.raises(prefetch._WorkerCancelled) as caught:
            _publish_schedule(
                cfg=cfg,
                backend=cast(Any, backend),
                schedules=prefetch._schedule_map(families),
                schedule_id=families[0].schedule_id,
                inventory_index=0,
                generation=1,
                command_pipe=child_command,
            )
    finally:
        child_command.close()
        parent_command.close()
    assert writes == 2
    assert caught.value.__notes__ == [
        "also failed to restore the prior publication receipt: OSError: receipt restore failed"
    ]


def test_component_cleanup_failures_are_notes_on_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cache_root, control, families, cfg, backend = _setup_publish(
        tmp_path,
        paired=False,
    )
    family = families[0]
    receipt = _receipt_path(
        control,
        family_id=family.family_id,
        inventory_index=0,
    )
    primary = OSError("receipt publication failed")
    rollback = OSError("final rollback failed")
    original_atomic = prefetch._atomic_record

    def fail_receipt(value: dict[str, Any], path: Path) -> None:
        if Path(path) == receipt:
            raise primary
        original_atomic(value, path)

    monkeypatch.setattr(prefetch, "_atomic_record", fail_receipt)
    monkeypatch.setattr(
        prefetch,
        "_rollback_final_under_lock",
        lambda **_kwargs: (_ for _ in ()).throw(rollback),
    )
    child_command, parent_command = mp.get_context("spawn").Pipe(duplex=False)
    try:
        with pytest.raises(OSError) as caught:
            prefetch._publish_component(
                cfg=cfg,
                backend=cast(Any, backend),
                family=family,
                inventory_index=0,
                generation=0,
                command_pipe=child_command,
            )
    finally:
        child_command.close()
        parent_command.close()
    assert caught.value is primary
    assert primary.__notes__ == [
        "also failed to roll back component publication: OSError: final rollback failed"
    ]


def test_owner_waits_behind_reader_lock_without_bypassing_ownership(
    tmp_path: Path,
) -> None:
    _cache_root, _control, families, cfg, backend = _setup_publish(
        tmp_path,
        paired=False,
    )
    family = families[0]
    entry = family.entries[0]
    lock_fd = prefetch._open_cache_lock(Path(family.local_dir) / f".{entry.name}.lock")
    blocked = threading.Event()
    completed: list[BaseException | None] = []

    class CommandPipe:
        def __init__(self) -> None:
            self.timed_polls = 0

        def poll(self, timeout: float | None = None) -> bool:
            if timeout is None:
                return False
            self.timed_polls += 1
            blocked.set()
            return self.timed_polls == 1

    command_pipe = CommandPipe()

    def publish() -> None:
        try:
            prefetch._publish_component(
                cfg=cfg,
                backend=cast(Any, backend),
                family=family,
                inventory_index=0,
                generation=0,
                command_pipe=cast(Any, command_pipe),
            )
        except BaseException as exc:
            completed.append(exc)
        else:
            completed.append(None)

    fcntl.flock(lock_fd, fcntl.LOCK_SH)
    thread = threading.Thread(target=publish)
    thread.start()
    try:
        assert blocked.wait(timeout=1)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert completed == [None]
    assert command_pipe.timed_polls >= 1


def test_schedule_cleanup_preserves_ready_and_component_rollback_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cache_root, control, families, cfg, backend = _setup_publish(
        tmp_path,
        paired=False,
    )
    ready = _ready_path(
        control,
        schedule_id=families[0].schedule_id,
        inventory_index=0,
    )
    primary = RuntimeError("component failed")
    cleanup = OSError("ready unlink failed")
    original_unlink = Path.unlink

    def fail_component(**_kwargs: Any) -> None:
        raise primary

    def fail_ready_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        if Path(path) == ready:
            raise cleanup
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(prefetch, "_publish_component", fail_component)
    monkeypatch.setattr(Path, "unlink", fail_ready_unlink)
    child_command, parent_command = mp.get_context("spawn").Pipe(duplex=False)
    try:
        with pytest.raises(RuntimeError) as caught:
            _publish_schedule(
                cfg=cfg,
                backend=cast(Any, backend),
                schedules=prefetch._schedule_map(families),
                schedule_id=families[0].schedule_id,
                inventory_index=0,
                generation=0,
                command_pipe=child_command,
            )
    finally:
        child_command.close()
        parent_command.close()
    assert caught.value is primary
    assert primary.__notes__ == [
        "also failed to roll back owner publication: OSError: ready unlink failed"
    ]

    publication = prefetch._NewPublication(
        family_id=families[0].family_id,
        inventory_index=0,
    )
    primary_2 = RuntimeError("ready write failed")
    rollback_2 = OSError("publication rollback failed")
    monkeypatch.setattr(
        prefetch,
        "_publish_component",
        lambda **_kwargs: publication,
    )
    monkeypatch.setattr(
        prefetch,
        "_atomic_record",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(primary_2),
    )
    monkeypatch.setattr(
        prefetch,
        "_rollback_new_publication",
        lambda **_kwargs: (_ for _ in ()).throw(rollback_2),
    )
    monkeypatch.setattr(Path, "unlink", original_unlink)
    child_command_2, parent_command_2 = mp.get_context("spawn").Pipe(duplex=False)
    try:
        with pytest.raises(RuntimeError) as caught_2:
            _publish_schedule(
                cfg=cfg,
                backend=cast(Any, backend),
                schedules=prefetch._schedule_map(families),
                schedule_id=families[0].schedule_id,
                inventory_index=0,
                generation=0,
                command_pipe=child_command_2,
            )
    finally:
        child_command_2.close()
        parent_command_2.close()
    assert caught_2.value is primary_2
    assert primary_2.__notes__ == [
        "also failed to roll back owner publication: OSError: publication rollback failed"
    ]


def test_read_direct_bytes_detects_path_replacement_using_one_open_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = tmp_path / "record.json"
    replacement = tmp_path / "replacement.json"
    record.write_bytes(b'{"value":"old"}')
    replacement.write_bytes(b'{"value":"new"}')
    original_open = os.open
    target_opens = 0

    def replacing_open(path: Any, *args: Any, **kwargs: Any) -> int:
        nonlocal target_opens
        fd = original_open(path, *args, **kwargs)
        if Path(path) == record:
            target_opens += 1
            if target_opens == 1:
                os.replace(replacement, record)
        return fd

    monkeypatch.setattr(prefetch.os, "open", replacing_open)
    with pytest.raises(RemoteTrainingError, match="direct single-link|changed while opening"):
        _read_direct_bytes(record, max_bytes=1024)
    assert target_opens == 1


def test_immutable_direct_reader_rejects_hardlinked_replacement_after_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = tmp_path / "record.json"
    old_anchor = tmp_path / "old-anchor.json"
    replacement = tmp_path / "replacement.json"
    replacement_anchor = tmp_path / "replacement-anchor.json"
    record.write_bytes(b'{"value":"old"}')
    os.link(record, old_anchor)
    replacement.write_bytes(b'{"value":"new"}')
    os.link(replacement, replacement_anchor)
    original_open = os.open
    replaced = False

    def replacing_open(path: Any, *args: Any, **kwargs: Any) -> int:
        nonlocal replaced
        descriptor = original_open(path, *args, **kwargs)
        if Path(path) == record and not replaced:
            replaced = True
            os.replace(replacement, record)
        return descriptor

    monkeypatch.setattr(prefetch.os, "open", replacing_open)
    with pytest.raises(RemoteTrainingError, match="direct single-link regular file"):
        _read_direct_bytes(record, max_bytes=1024)
    assert replaced


def test_immutable_direct_reader_rejects_replacement_after_descriptor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = tmp_path / "record.json"
    replacement = tmp_path / "replacement.json"
    record.write_bytes(b'{"value":"old"}')
    replacement.write_bytes(b'{"value":"new"}')
    original_fstat = os.fstat
    fstat_calls = 0

    def replace_after_second_fstat(descriptor: int) -> os.stat_result:
        nonlocal fstat_calls
        fstat_calls += 1
        observed = original_fstat(descriptor)
        if fstat_calls == 2:
            os.replace(replacement, record)
        return observed

    monkeypatch.setattr(prefetch.os, "fstat", replace_after_second_fstat)
    with pytest.raises(RemoteTrainingError, match="changed while reading"):
        _read_direct_bytes(record, max_bytes=1024)
    assert fstat_calls == 2
    assert record.read_bytes() == b'{"value":"new"}'


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO support is required")
def test_read_direct_bytes_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "record.fifo"
    os.mkfifo(fifo)

    started = time.monotonic()
    with pytest.raises(RemoteTrainingError, match="direct single-link regular file"):
        _read_direct_bytes(fifo, max_bytes=1024)
    assert time.monotonic() - started < 1
