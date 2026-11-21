from __future__ import annotations

import fcntl
import hashlib
import inspect
import json
import multiprocessing as mp
import os
import shutil
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, cast

import hf_xet
import pytest
import torch.distributed as dist

import steering.remote_body_prefetch as prefetch
import steering.remote_training as remote_training
from steering.conditional.encoder_decoder.dataloader import ClosableBatchPrefetcher
from steering.refined_types import PositiveInt
from steering.remote_body_prefetch import (
    REMOTE_PREFETCH_SCHEMA_VERSION,
    RemoteBodyBinding,
    RemoteBodyEntry,
    RemoteBodyFamily,
    RemoteBodyPrefetcher,
    _atomic_record,
    _control_root,
    _create_control_tree,
    _cursor_path,
    _demand_path,
    _parse_local_world_size,
    _progress_path,
    _ready_path,
    _receipt_path,
    _status_path,
    _worker_main,
    _WorkerConfig,
    build_remote_body_plan,
    calculate_remote_prefetch_disk_admission,
    owner_for_inventory_index,
)
from steering.remote_training import (
    RemoteObjectInfo,
    RemotePackedRowMaterializer,
    RemoteSelectionMaterializer,
    RemoteTrainingConfig,
    RemoteTrainingError,
)


def _payload(kind: str, index: int) -> bytes:
    return f"{kind}-signed-body-{index}".encode()


def _entry(kind: str, index: int) -> RemoteBodyEntry:
    name = f"shard_{index:05d}.safetensors.zst"
    payload = _payload(kind, index)
    content_sha256 = hashlib.sha256(payload).hexdigest()
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
        content_sha256=content_sha256,
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


def _owned_inventory_indices(
    inventory_length: int,
    *,
    world_size: int,
    rank: int,
) -> tuple[int, ...]:
    return tuple(
        index
        for index in range(inventory_length)
        if owner_for_inventory_index(index, world_size) == rank
    )


def _families(
    tmp_path: Path,
    *,
    n_entries: int,
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
        entries=tuple(_entry("selection", index) for index in range(n_entries)),
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
                entries=tuple(_entry("packed", index) for index in range(n_entries)),
                preexisting_descriptors=(),
                schedule_id=schedule_id,
                train_schedule=True,
            )
        )
    return cache_root, tuple(families)


def _remote_config(*, selection_wait_timeout_seconds: float = 2) -> RemoteTrainingConfig:
    remote = RemoteTrainingConfig.parse(
        {
            "bucket_id": "owner/bucket",
            "body_prefetch": {"lookahead_shards_per_rank": 2},
            "selection_prefix": "features",
            "checkpoint_prefix": "checkpoints",
            "selection_wait_timeout_seconds": selection_wait_timeout_seconds,
            "checkpoint_retry_timeout_seconds": 2,
            "retry_interval_seconds": 0.001,
            "resume_latest": False,
        }
    )
    assert remote is not None
    return remote


def _session_lock_failure_target(
    rank: int,
    world_size: int,
    cache_root: Path,
) -> str:
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    selection_dir = cache_root / "selection"
    entry = _entry("selection", 0)
    materializer = RemoteSelectionMaterializer(
        local_dir=selection_dir,
        remote_dir="features/selection",
        inventory={entry.name: _reader_entry(entry)},
        loader_certificate={"selection_inventory_aggregate": "a" * 64},
        identity_contract={},
        timeout_seconds=2,
        retry_interval_seconds=0.001,
        trust_existing_cache=False,
    )
    try:
        controller = RemoteBodyPrefetcher.start(
            remote=_remote_config(),
            cache_root=cache_root,
            rank=rank,
            world_size=world_size,
            train_selection_dir=selection_dir,
            selection_materializers=[materializer],
            packed_materializer=None,
            resume_state_dict=None,
        )
    except RemoteTrainingError as exc:
        return str(exc)
    controller.close()
    return "unexpected-success"


def _successful_start_target(
    rank: int,
    world_size: int,
    cache_root: Path,
    remote_root: Path,
    attempt_log: Path,
) -> dict[str, Any]:
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    selection_dir = cache_root / "selection"
    packed_dir = cache_root / "packed"
    selection_entries = tuple(_entry("selection", index) for index in range(world_size))
    packed_entries = tuple(_entry("packed", index) for index in range(world_size))
    selection_materializer = RemoteSelectionMaterializer(
        local_dir=selection_dir,
        remote_dir="features/selection",
        inventory={entry.name: _reader_entry(entry) for entry in selection_entries},
        loader_certificate={"selection_inventory_aggregate": "a" * 64},
        identity_contract={},
        timeout_seconds=20,
        retry_interval_seconds=0.001,
        trust_existing_cache=False,
    )
    packed_materializer = RemotePackedRowMaterializer(
        local_dir=packed_dir,
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
                for entry in packed_entries
            ],
        },
        cache_shards=2,
        base_shard_stems=[],
        provenance_shard_paths=[],
        timeout_seconds=20,
        retry_interval_seconds=0.001,
        trust_existing_cache=False,
    )
    selection_materializer.validate_cache_namespace()
    packed_materializer.validate_cache_namespace()
    controller = RemoteBodyPrefetcher.start(
        remote=_remote_config(selection_wait_timeout_seconds=20),
        cache_root=cache_root,
        rank=rank,
        world_size=world_size,
        train_selection_dir=selection_dir,
        selection_materializers=[selection_materializer],
        packed_materializer=packed_materializer,
        resume_state_dict=None,
        _backend_factory=_SpawnBackendFactory(
            root=str(remote_root),
            rank=rank,
            barrier=None,
            attempt_log=str(attempt_log),
        ),
    )
    try:
        consumed_selection = [
            selection_materializer.read(selection_dir / entry.name) for entry in selection_entries
        ]
        consumed_packed = [
            packed_materializer.read(packed_dir / entry.name) for entry in packed_entries
        ]
        dist.barrier()
        own_demand_count = len(
            list((controller.control_root / "demands" / f"rank_{rank:05d}").glob("*.json"))
        )
        return {
            "consumed_selection": consumed_selection,
            "consumed_packed": consumed_packed,
            "worker_pid": controller.process.pid,
            "parent_pid": os.getpid(),
            "own_demand_count": own_demand_count,
        }
    finally:
        controller.close()


def _worker_config(
    *,
    cache_root: Path,
    families: tuple[RemoteBodyFamily, ...],
    rank: int,
    world_size: int,
    nonce: str = "d" * 32,
    timeout_seconds: float = 0.1,
) -> _WorkerConfig:
    return _WorkerConfig(
        bucket_id="owner/bucket",
        cache_root=str(cache_root),
        nonce=nonce,
        rank=rank,
        world_size=world_size,
        lookahead_shards_per_rank=2,
        cache_shards=2,
        timeout_seconds=timeout_seconds,
        retry_interval_seconds=0.001,
        families=families,
    )


def _write_rank_state(
    *,
    cfg: _WorkerConfig,
    demand_indices_by_rank: Mapping[int, Sequence[int]],
) -> None:
    control = _control_root(Path(cfg.cache_root), cfg.nonce)
    schedule = cfg.families[0].schedule_id
    n_entries = len(cfg.families[0].entries)
    for rank in range(cfg.world_size):
        _atomic_record(
            {
                "schema_version": REMOTE_PREFETCH_SCHEMA_VERSION,
                "nonce": cfg.nonce,
                "rank": rank,
                "schedule_id": schedule,
                "epoch": 1,
                "inventory_index": 0,
                "generation": 0,
            },
            _cursor_path(control, rank),
        )
        _atomic_record(
            {
                "schema_version": REMOTE_PREFETCH_SCHEMA_VERSION,
                "nonce": cfg.nonce,
                "rank": rank,
                "schedule_id": schedule,
                "generation": -1,
            },
            _progress_path(control, rank),
        )
        for index in demand_indices_by_rank.get(rank, ()):
            assert 0 <= index < n_entries
            _atomic_record(
                {
                    "schema_version": REMOTE_PREFETCH_SCHEMA_VERSION,
                    "nonce": cfg.nonce,
                    "requester_rank": rank,
                    "schedule_id": schedule,
                    "epoch": 1,
                    "inventory_index": index,
                    "generation": index,
                    "created_ns": time.time_ns(),
                },
                _demand_path(
                    control,
                    requester_rank=rank,
                    schedule_id=schedule,
                    inventory_index=index,
                ),
            )


class _SharedBackend:
    def __init__(
        self,
        root: Path,
        *,
        overlap_barrier: threading.Barrier | None = None,
        failing_path: str | None = None,
        block_release: threading.Event | None = None,
    ) -> None:
        self.root = root
        self.overlap_barrier = overlap_barrier
        self.failing_path = failing_path
        self.block_release = block_release
        self.download_started = threading.Event()
        self.lock = threading.Lock()
        self.download_counts: Counter[str] = Counter()
        self.download_order: list[str] = []
        self.attempt_ranks: defaultdict[str, set[int]] = defaultdict(set)
        self.active_by_path: Counter[str] = Counter()
        self.max_active_by_path: Counter[str] = Counter()
        self.global_active = 0
        self.max_global_active = 0

    def _path(self, remote_path: str) -> Path:
        return self.root / remote_path

    def stat(self, remote_path: str) -> RemoteObjectInfo | None:
        path = self._path(remote_path)
        if not path.is_file():
            return None
        xet_hash = str(cast(Any, hf_xet).hash_files([str(path)])[0].hash)
        return RemoteObjectInfo(
            path=remote_path,
            byte_length=path.stat().st_size,
            xet_hash=xet_hash,
        )

    def view(self, rank: int) -> _BackendView:
        return _BackendView(self, rank)


class _BackendView:
    def __init__(self, shared: _SharedBackend, rank: int) -> None:
        self.shared = shared
        self.rank = rank

    def stat(self, remote_path: str) -> RemoteObjectInfo | None:
        return self.shared.stat(remote_path)

    def download(self, remote_path: str, local_path: Path) -> None:
        with self.shared.lock:
            self.shared.download_counts[remote_path] += 1
            self.shared.download_order.append(remote_path)
            self.shared.attempt_ranks[remote_path].add(self.rank)
            self.shared.active_by_path[remote_path] += 1
            self.shared.max_active_by_path[remote_path] = max(
                self.shared.max_active_by_path[remote_path],
                self.shared.active_by_path[remote_path],
            )
            self.shared.global_active += 1
            self.shared.max_global_active = max(
                self.shared.max_global_active,
                self.shared.global_active,
            )
        try:
            if self.shared.failing_path == remote_path:
                raise OSError("injected persistent owner transport failure")
            if self.shared.overlap_barrier is not None:
                self.shared.overlap_barrier.wait(timeout=5)
            if self.shared.block_release is not None:
                self.shared.download_started.set()
                assert self.shared.block_release.wait(timeout=5)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.shared._path(remote_path), local_path)
        finally:
            with self.shared.lock:
                self.shared.active_by_path[remote_path] -= 1
                self.shared.global_active -= 1


class _ThreadProcessAdapter:
    def __init__(self, thread: threading.Thread) -> None:
        self.thread = thread
        self.pid = os.getpid()

    @property
    def exitcode(self) -> int | None:
        return 0 if not self.thread.is_alive() else None

    def is_alive(self) -> bool:
        return self.thread.is_alive()

    def join(self, timeout: float | None = None) -> None:
        self.thread.join(timeout=timeout)

    def terminate(self) -> None:
        raise AssertionError("test worker should stop gracefully")


@dataclass(frozen=True)
class _SpawnBackendFactory:
    root: str
    rank: int
    barrier: Any | None
    attempt_log: str

    def __call__(self, _bucket_id: str) -> _SpawnBackend:
        return _SpawnBackend(self)


class _SpawnBackend:
    def __init__(self, factory: _SpawnBackendFactory) -> None:
        self.factory = factory

    def _path(self, remote_path: str) -> Path:
        return Path(self.factory.root) / remote_path

    def stat(self, remote_path: str) -> RemoteObjectInfo | None:
        path = self._path(remote_path)
        if not path.is_file():
            return None
        xet_hash = str(cast(Any, hf_xet).hash_files([str(path)])[0].hash)
        return RemoteObjectInfo(
            path=remote_path,
            byte_length=path.stat().st_size,
            xet_hash=xet_hash,
        )

    def download(self, remote_path: str, local_path: Path) -> None:
        descriptor = os.open(
            self.factory.attempt_log,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o600,
        )
        try:
            os.write(
                descriptor,
                f"{self.factory.rank}|{remote_path}\n".encode(),
            )
        finally:
            os.close(descriptor)
        if self.factory.barrier is not None:
            self.factory.barrier.wait(timeout=10)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(remote_path), local_path)


class _RetryMutationBackend:
    def __init__(
        self,
        root: Path,
        *,
        block_second_attempt: bool,
    ) -> None:
        self.root = root
        self.block_second_attempt = block_second_attempt
        self.retry_started = threading.Event()
        self.release_retry = threading.Event()
        self.download_count = 0

    def _path(self, remote_path: str) -> Path:
        return self.root / remote_path

    def stat(self, remote_path: str) -> RemoteObjectInfo | None:
        path = self._path(remote_path)
        if not path.is_file():
            return None
        result = cast(Any, hf_xet).hash_files([str(path)])[0]
        return RemoteObjectInfo(
            path=remote_path,
            byte_length=path.stat().st_size,
            xet_hash=str(result.hash),
        )

    def download(self, remote_path: str, local_path: Path) -> None:
        self.download_count += 1
        if self.download_count == 2:
            self.retry_started.set()
            if self.block_second_attempt:
                assert self.release_retry.wait(timeout=5)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(remote_path), local_path)


def _replace_transport_during_xet_hash(
    monkeypatch: pytest.MonkeyPatch,
    *,
    persistent: bool,
) -> list[int]:
    real_hash_files = cast(Any, remote_training.hf_xet).hash_files
    mutations = [0]

    def replacing_hash_files(paths):
        result = real_hash_files(paths)
        path = Path(paths[0])
        if path.parent == Path("/proc/self/fd"):
            path = path.resolve()
        if ".owner-transfer." in path.name and (persistent or mutations[0] == 0):
            replacement = path.with_name(f"{path.name}.replacement-{mutations[0]}")
            replacement.write_bytes(path.read_bytes())
            os.replace(replacement, path)
            mutations[0] += 1
        return result

    monkeypatch.setattr(
        remote_training.hf_xet,
        "hash_files",
        replacing_hash_files,
    )
    return mutations


def _publish_remote_sources(
    *,
    root: Path,
    families: Sequence[RemoteBodyFamily],
) -> None:
    for family in families:
        for index, entry in enumerate(family.entries):
            path = root / family.remote_dir / entry.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_payload(family.kind, index))


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
@pytest.mark.parametrize("inventory_length", [0, 1, 3, 8, 17, 584])
def test_owner_stripes_are_disjoint_and_cover_signed_inventory(
    world_size: int,
    inventory_length: int,
) -> None:
    stripes = [
        set(
            _owned_inventory_indices(
                inventory_length,
                world_size=world_size,
                rank=rank,
            )
        )
        for rank in range(world_size)
    ]
    assert set().union(*stripes) == set(range(inventory_length))
    assert sum(len(stripe) for stripe in stripes) == inventory_length
    for left in range(world_size):
        for right in range(left + 1, world_size):
            assert stripes[left].isdisjoint(stripes[right])
    for index in range(inventory_length):
        assert index in stripes[owner_for_inventory_index(index, world_size)]


def test_packed_and_selection_plan_share_inventory_owner(tmp_path: Path) -> None:
    names = [f"shard_{index:05d}.safetensors.zst" for index in range(5)]
    selection_dir = tmp_path / "selection"
    packed_dir = tmp_path / "packed"
    selection_dir.mkdir()
    packed_dir.mkdir()
    selection = RemoteSelectionMaterializer(
        local_dir=selection_dir,
        remote_dir="features/selection",
        inventory={
            name: {
                "name": name,
                "byte_length": 10 + index,
                "content_sha256": f"{index + 1:064x}",
            }
            for index, name in enumerate(names)
        },
        loader_certificate={"selection_inventory_aggregate": "a" * 64},
        identity_contract={},
        timeout_seconds=1,
        retry_interval_seconds=0.01,
        trust_existing_cache=False,
    )
    packed = RemotePackedRowMaterializer(
        local_dir=packed_dir,
        remote_dir="features/packed",
        inventory={
            "aggregate_sha256": "b" * 64,
            "packed_row_fingerprint": "e" * 64,
            "provenance_fingerprint": "f" * 64,
            "shards": [
                {
                    "name": name,
                    "byte_length": 20 + index,
                    "content_sha256": f"{index + 10:064x}",
                    "file_fingerprint": f"{index + 20:064x}",
                    "total_segments": 1,
                }
                for index, name in enumerate(names)
            ],
        },
        cache_shards=2,
        base_shard_stems=["source_00000"],
        provenance_shard_paths=[],
        timeout_seconds=1,
        retry_interval_seconds=0.01,
        trust_existing_cache=False,
    )

    families, bindings, packed_binding = build_remote_body_plan(
        train_selection_dir=selection_dir,
        selection_materializers=[selection],
        packed_materializer=packed,
    )

    assert len(families) == 2
    selection_binding = bindings[str(selection_dir.resolve())]
    assert packed_binding is not None
    assert selection_binding.schedule_id == packed_binding.schedule_id
    for world_size in (1, 2, 4, 8):
        for name in names:
            selection_index = selection_binding.inventory_index(name)
            packed_index = packed_binding.inventory_index(name)
            assert selection_index == packed_index
            assert owner_for_inventory_index(
                selection_index,
                world_size,
            ) == owner_for_inventory_index(packed_index, world_size)


@pytest.mark.parametrize(
    ("local_world_size", "environment_world_size", "runtime_world_size", "match"),
    [
        (None, "1", 1, "requires torchrun"),
        ("1", None, 1, "requires torchrun"),
        ("not-an-int", "1", 1, "decimal integers"),
        ("1", "2", 1, "disagrees with WORLD_SIZE"),
        ("3", "4", 4, "positive divisor"),
    ],
)
def test_remote_environment_contract_is_strict(
    monkeypatch: pytest.MonkeyPatch,
    local_world_size: str | None,
    environment_world_size: str | None,
    runtime_world_size: int,
    match: str,
) -> None:
    if local_world_size is None:
        monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    else:
        monkeypatch.setenv("LOCAL_WORLD_SIZE", local_world_size)
    if environment_world_size is None:
        monkeypatch.delenv("WORLD_SIZE", raising=False)
    else:
        monkeypatch.setenv("WORLD_SIZE", environment_world_size)

    with pytest.raises(RemoteTrainingError, match=match):
        _parse_local_world_size(runtime_world_size)


@pytest.mark.parametrize(("local_world_size", "world_size"), [(2, 4), (1, 4)])
def test_shared_remote_cache_accepts_multi_node_world_sizes(
    monkeypatch: pytest.MonkeyPatch,
    local_world_size: int,
    world_size: int,
) -> None:
    monkeypatch.setenv("LOCAL_WORLD_SIZE", str(local_world_size))
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    assert _parse_local_world_size(world_size) == local_world_size


def test_start_holds_session_lock_until_all_worker_teardown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("WORLD_SIZE", "1")
    cache_root = tmp_path / "cache"
    selection_dir = cache_root / "selection"
    selection_dir.mkdir(parents=True)
    entry = _entry("selection", 0)
    (selection_dir / entry.name).write_bytes(_payload("selection", 0))
    materializer = RemoteSelectionMaterializer(
        local_dir=selection_dir,
        remote_dir="features/selection",
        inventory={entry.name: _reader_entry(entry)},
        loader_certificate={
            "selection_inventory_aggregate": "a" * 64,
            "max_features_per_segment": 1,
        },
        identity_contract={},
        timeout_seconds=20,
        retry_interval_seconds=0.001,
        trust_existing_cache=False,
    )
    materializer.validate_cache_namespace()
    remote = RemoteTrainingConfig.parse(
        {
            "bucket_id": "owner/bucket",
            "body_prefetch": {"lookahead_shards_per_rank": 2},
            "selection_prefix": "features",
            "checkpoint_prefix": "checkpoints",
            "selection_wait_timeout_seconds": 20,
            "checkpoint_retry_timeout_seconds": 2,
            "retry_interval_seconds": 0.001,
            "resume_latest": False,
        }
    )
    assert remote is not None
    controller = RemoteBodyPrefetcher.start(
        remote=remote,
        cache_root=cache_root,
        rank=0,
        world_size=1,
        train_selection_dir=selection_dir,
        selection_materializers=[materializer],
        packed_materializer=None,
        resume_state_dict=None,
    )
    lock_path = cache_root / ".remote-prefetch.session.lock"
    probe = os.open(lock_path, os.O_RDWR)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert materializer.read(selection_dir / entry.name) == _payload(
            "selection",
            0,
        )
        controller.stop_worker()
        with pytest.raises(BlockingIOError):
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        controller.release_session()
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(probe, fcntl.LOCK_UN)
    finally:
        os.close(probe)
        if not controller._worker_stopped:
            controller.stop_worker()
        if not controller._session_released:
            controller.release_session()


def test_start_rejects_mutated_rank_zero_sentinel_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("WORLD_SIZE", "1")
    cache_root = tmp_path / "cache"
    selection_dir = cache_root / "selection"
    selection_dir.mkdir(parents=True)
    entry = _entry("selection", 0)
    materializer = RemoteSelectionMaterializer(
        local_dir=selection_dir,
        remote_dir="features/selection",
        inventory={entry.name: _reader_entry(entry)},
        loader_certificate={"selection_inventory_aggregate": "a" * 64},
        identity_contract={},
        timeout_seconds=2,
        retry_interval_seconds=0.001,
        trust_existing_cache=False,
    )

    def publish_wrong_bytes(_data: bytes, path: Path) -> None:
        path.write_bytes(b"not-the-random-sentinel")

    monkeypatch.setattr(prefetch, "atomic_write_bytes", publish_wrong_bytes)
    with pytest.raises(RemoteTrainingError, match="exact rank-0 sentinel bytes"):
        RemoteBodyPrefetcher.start(
            remote=_remote_config(),
            cache_root=cache_root,
            rank=0,
            world_size=1,
            train_selection_dir=selection_dir,
            selection_materializers=[materializer],
            packed_materializer=None,
            resume_state_dict=None,
        )

    lock_fd = os.open(
        cache_root / ".remote-prefetch.session.lock",
        os.O_RDWR | os.O_CLOEXEC,
    )
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        os.close(lock_fd)


def test_two_rank_session_lock_failure_is_symmetric_and_bounded(
    tmp_path: Path,
    gloo_multiprocess,
) -> None:
    cache_root = tmp_path / "cache"
    (cache_root / "selection").mkdir(parents=True)
    lock_path = cache_root / ".remote-prefetch.session.lock"
    lock_fd = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | os.O_CLOEXEC,
        0o600,
    )
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        started = time.monotonic()
        results = gloo_multiprocess(
            2,
            _session_lock_failure_target,
            cache_root,
        )
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    assert time.monotonic() - started < 30
    assert len(results) == 2
    assert all(
        "startup phase 'session-lock' failed symmetrically" in result
        and "rank=0 RemoteTrainingError" in result
        for result in results
    )


def test_two_rank_start_runs_full_shared_cache_protocol_and_all_rank_readers(
    tmp_path: Path,
    gloo_multiprocess,
) -> None:
    world_size = 2
    cache_root = tmp_path / "cache"
    selection_dir = cache_root / "selection"
    selection_dir.mkdir(parents=True)
    (cache_root / "packed").mkdir()
    family_plan_root = tmp_path / "family-plan"
    family_plan_root.mkdir()
    _, families = _families(family_plan_root, n_entries=world_size, paired=True)
    remote_root = tmp_path / "remote"
    _publish_remote_sources(root=remote_root, families=families)
    # _families creates its own local cache; only its signed remote layout and
    # deterministic payloads are reused by the real materializers above.
    attempt_log = tmp_path / "start-attempts.log"

    results = gloo_multiprocess(
        world_size,
        _successful_start_target,
        cache_root,
        remote_root,
        attempt_log,
    )

    expected_selection = [_payload("selection", index) for index in range(world_size)]
    expected_packed = [_payload("packed", index) for index in range(world_size)]
    for result in results:
        assert result["consumed_selection"] == expected_selection
        assert result["consumed_packed"] == expected_packed
        assert result["worker_pid"] != result["parent_pid"]
        assert result["own_demand_count"] == 0
    attempts = [line.split("|", 1) for line in attempt_log.read_text(encoding="utf-8").splitlines()]
    assert len(attempts) == 2 * world_size
    paths_by_index: defaultdict[int, list[str]] = defaultdict(list)
    for rank_text, remote_path in attempts:
        index = int(Path(remote_path).name.split("_")[1].split(".")[0])
        assert int(rank_text) == owner_for_inventory_index(index, world_size)
        paths_by_index[index].append(remote_path)
    for index in range(world_size):
        name = f"shard_{index:05d}.safetensors.zst"
        assert paths_by_index[index] == [
            f"features/selection/{name}",
            f"features/packed/{name}",
        ]

    probe = os.open(cache_root / ".remote-prefetch.session.lock", os.O_RDWR)
    try:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(probe, fcntl.LOCK_UN)
    finally:
        os.close(probe)


def test_production_readers_have_no_backend_or_publisher_role() -> None:
    assert "backend" not in RemoteSelectionMaterializer.__dataclass_fields__
    assert "backend" not in RemotePackedRowMaterializer.__dataclass_fields__
    for reader in (
        RemoteSelectionMaterializer.read,
        RemotePackedRowMaterializer.read,
    ):
        source = inspect.getsource(reader)
        assert "_download" not in source
        assert "LOCK_EX" not in source
        assert ".backend" not in source


def test_two_owners_transfer_distinct_paths_concurrently_and_publish_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root, families = _families(tmp_path, n_entries=2, paired=False)
    nonce = "d" * 32
    _create_control_tree(
        cache_root=cache_root,
        nonce=nonce,
        world_size=2,
        families=families,
    )
    cfgs = tuple(
        _worker_config(
            cache_root=cache_root,
            families=families,
            rank=rank,
            world_size=2,
            nonce=nonce,
        )
        for rank in range(2)
    )
    _write_rank_state(
        cfg=cfgs[0],
        demand_indices_by_rank={0: (0, 1), 1: (0, 1)},
    )
    remote_root = tmp_path / "remote"
    _publish_remote_sources(root=remote_root, families=families)
    shared = _SharedBackend(remote_root, overlap_barrier=threading.Barrier(2))

    def backend_factory(_bucket_id: str) -> _BackendView:
        rank = int(threading.current_thread().name.rsplit("-", 1)[1])
        return shared.view(rank)

    monkeypatch.setattr(prefetch, "HfBucketBackend", backend_factory)
    context = mp.get_context("spawn")
    workers: list[tuple[threading.Thread, Any, Any]] = []
    for cfg in cfgs:
        child_command, parent_command = context.Pipe(duplex=False)
        parent_status, child_status = context.Pipe(duplex=False)
        thread = threading.Thread(
            target=_worker_main,
            args=(cfg, child_command, child_status),
            name=f"owner-{cfg.rank}",
        )
        thread.start()
        workers.append((thread, parent_command, parent_status))
    try:
        for _thread, _command, status in workers:
            assert status.poll(5)
            assert status.recv()["state"] == "ready"
        deadline = time.monotonic() + 5
        finals = [Path(families[0].local_dir) / entry.name for entry in families[0].entries]
        ready_paths = [
            _ready_path(
                _control_root(cache_root, nonce),
                schedule_id=families[0].schedule_id,
                inventory_index=index,
            )
            for index in range(len(families[0].entries))
        ]
        while (
            not all(path.is_file() for path in (*finals, *ready_paths))
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert all(path.is_file() for path in (*finals, *ready_paths))
        assert shared.max_global_active == 2
        for index, (entry, final) in enumerate(zip(families[0].entries, finals, strict=True)):
            remote_path = f"{families[0].remote_dir}/{entry.name}"
            assert shared.attempt_ranks[remote_path] == {owner_for_inventory_index(index, 2)}
            assert shared.download_counts[remote_path] == 1
            assert shared.max_active_by_path[remote_path] == 1
            assert final.read_bytes() == _payload("selection", index)
            assert _ready_path(
                _control_root(cache_root, nonce),
                schedule_id=families[0].schedule_id,
                inventory_index=index,
            ).is_file()
    finally:
        for _thread, command, _status in workers:
            command.send({"op": "cancel"})
        for thread, _command, _status in workers:
            thread.join(timeout=5)
            assert not thread.is_alive()


def test_worker_finalizes_lookahead_pairs_without_consumer_demand(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root, families = _families(tmp_path, n_entries=3, paired=True)
    nonce = "7" * 32
    control = _create_control_tree(
        cache_root=cache_root,
        nonce=nonce,
        world_size=1,
        families=families,
    )
    cfg = _worker_config(
        cache_root=cache_root,
        families=families,
        rank=0,
        world_size=1,
        nonce=nonce,
    )
    _write_rank_state(cfg=cfg, demand_indices_by_rank={})
    remote_root = tmp_path / "remote"
    _publish_remote_sources(root=remote_root, families=families)
    shared = _SharedBackend(remote_root)
    monkeypatch.setattr(prefetch, "HfBucketBackend", lambda _bucket_id: shared.view(0))

    context = mp.get_context("spawn")
    child_command, parent_command = context.Pipe(duplex=False)
    parent_status, child_status = context.Pipe(duplex=False)
    worker = threading.Thread(
        target=_worker_main,
        args=(cfg, child_command, child_status),
        name="owner-0",
    )
    worker.start()
    try:
        assert parent_status.poll(5)
        assert parent_status.recv()["state"] == "ready"
        ready_paths = [
            _ready_path(
                control,
                schedule_id=families[0].schedule_id,
                inventory_index=index,
            )
            for index in range(2)
        ]
        deadline = time.monotonic() + 5
        while not all(path.is_file() for path in ready_paths) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert all(path.is_file() for path in ready_paths)

        for family in families:
            for index in range(2):
                entry = family.entries[index]
                assert (Path(family.local_dir) / entry.name).read_bytes() == _payload(
                    family.kind,
                    index,
                )
                assert _receipt_path(
                    control,
                    family_id=family.family_id,
                    inventory_index=index,
                ).is_file()
            assert not (Path(family.local_dir) / family.entries[2].name).exists()
        assert not list((control / "demands" / "rank_00000").glob("*.json"))
    finally:
        parent_command.send({"op": "cancel"})
        worker.join(timeout=5)
        assert not worker.is_alive()
        parent_command.close()
        parent_status.close()


@pytest.mark.parametrize(
    "world_size",
    [
        2,
        pytest.param(4, marks=pytest.mark.slow),
        pytest.param(8, marks=pytest.mark.slow),
    ],
)
def test_spawned_rank_workers_share_cache_and_keep_owner_stripes(
    tmp_path: Path,
    world_size: int,
) -> None:
    cache_root, families = _families(
        tmp_path,
        n_entries=world_size,
        paired=False,
    )
    nonce = "6" * 32
    _create_control_tree(
        cache_root=cache_root,
        nonce=nonce,
        world_size=world_size,
        families=families,
    )
    cfgs = tuple(
        _worker_config(
            cache_root=cache_root,
            families=families,
            rank=rank,
            world_size=world_size,
            nonce=nonce,
        )
        for rank in range(world_size)
    )
    _write_rank_state(
        cfg=cfgs[0],
        demand_indices_by_rank={rank: tuple(range(world_size)) for rank in range(world_size)},
    )
    remote_root = tmp_path / "remote"
    _publish_remote_sources(root=remote_root, families=families)
    attempt_log = tmp_path / "attempts.log"
    context = mp.get_context("spawn")
    barrier = context.Barrier(world_size)
    workers: list[tuple[BaseProcess, Any, Any]] = []
    for cfg in cfgs:
        child_command, parent_command = context.Pipe(duplex=False)
        parent_status, child_status = context.Pipe(duplex=False)
        process = context.Process(
            target=_worker_main,
            args=(
                cfg,
                child_command,
                child_status,
                _SpawnBackendFactory(
                    root=str(remote_root),
                    rank=cfg.rank,
                    barrier=barrier,
                    attempt_log=str(attempt_log),
                ),
            ),
            daemon=False,
        )
        process.start()
        child_command.close()
        child_status.close()
        workers.append((process, parent_command, parent_status))
    try:
        for process, _command, status in workers:
            assert status.poll(20), f"worker {process.pid} did not report ready"
            assert status.recv()["state"] == "ready"
        ready_paths = [
            _ready_path(
                _control_root(cache_root, nonce),
                schedule_id=families[0].schedule_id,
                inventory_index=index,
            )
            for index in range(world_size)
        ]
        deadline = time.monotonic() + 20
        while not all(path.is_file() for path in ready_paths) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert all(path.is_file() for path in ready_paths)
        attempts = [
            line.split("|", 1) for line in attempt_log.read_text(encoding="utf-8").splitlines()
        ]
        assert len(attempts) == world_size
        for rank_text, remote_path in attempts:
            index = int(Path(remote_path).name.split("_")[1].split(".")[0])
            assert int(rank_text) == owner_for_inventory_index(index, world_size)
        for consumer_rank in range(world_size):
            consumed = [
                (Path(families[0].local_dir) / entry.name).read_bytes()
                for entry in families[0].entries
            ]
            assert consumed == [_payload("selection", index) for index in range(world_size)], (
                consumer_rank
            )
    finally:
        for _process, command, _status in workers:
            command.send({"op": "cancel"})
        for process, _command, status in workers:
            process.join(timeout=20)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            assert process.exitcode == 0
            status.close()
            _command.close()


def test_consumer_waits_for_nominal_owner_and_never_performs_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root, families = _families(tmp_path, n_entries=1, paired=False)
    family = families[0]
    nonce = "7" * 32
    control = _create_control_tree(
        cache_root=cache_root,
        nonce=nonce,
        world_size=1,
        families=families,
    )
    cfg = _worker_config(
        cache_root=cache_root,
        families=families,
        rank=0,
        world_size=1,
        nonce=nonce,
        timeout_seconds=1,
    )
    _write_rank_state(cfg=cfg, demand_indices_by_rank={0: (0,)})
    demand_path = _demand_path(
        control,
        requester_rank=0,
        schedule_id=family.schedule_id,
        inventory_index=0,
    )
    initial_demand = demand_path.read_bytes()
    remote_root = tmp_path / "remote"
    _publish_remote_sources(root=remote_root, families=families)
    release = threading.Event()
    shared = _SharedBackend(remote_root, block_release=release)
    monkeypatch.setattr(
        prefetch,
        "HfBucketBackend",
        lambda _bucket_id: shared.view(0),
    )
    context = mp.get_context("spawn")
    child_command, parent_command = context.Pipe(duplex=False)
    parent_status, child_status = context.Pipe(duplex=False)
    worker_thread = threading.Thread(
        target=_worker_main,
        args=(cfg, child_command, child_status),
        name="owner-0",
    )
    worker_thread.start()
    assert parent_status.poll(5)
    assert parent_status.recv()["state"] == "ready"

    entry = family.entries[0]
    binding = RemoteBodyBinding(
        family_id=family.family_id,
        schedule_id=family.schedule_id,
        kind="selection",
        index_by_name=((entry.name, 0),),
        paired=False,
        train_schedule=True,
    )
    remote = RemoteTrainingConfig.parse(
        {
            "bucket_id": "owner/bucket",
            "body_prefetch": {"lookahead_shards_per_rank": 2},
            "selection_prefix": "features",
            "checkpoint_prefix": "checkpoints",
            "selection_wait_timeout_seconds": 2,
            "checkpoint_retry_timeout_seconds": 2,
            "retry_interval_seconds": 0.001,
            "resume_latest": False,
        }
    )
    assert remote is not None
    controller = RemoteBodyPrefetcher(
        remote=remote,
        cache_root=cache_root,
        nonce=nonce,
        rank=0,
        world_size=1,
        families=families,
        bindings={str(Path(family.local_dir).resolve()): binding},
        packed_binding=None,
        train_binding=binding,
        control_root=control,
        process=cast(Any, _ThreadProcessAdapter(worker_thread)),
        command_pipe=parent_command,
        status_pipe=parent_status,
        session_lock_fd=None,
        admission=calculate_remote_prefetch_disk_admission(
            cache_root=cache_root,
            families=families,
            world_size=1,
            cache_shards=1,
            lookahead_shards_per_rank=2,
            safety_reserve_bytes=0,
        ),
    )
    materializer = RemoteSelectionMaterializer(
        local_dir=Path(family.local_dir),
        remote_dir=family.remote_dir,
        inventory={entry.name: _reader_entry(entry)},
        loader_certificate={"selection_inventory_aggregate": family.family_aggregate},
        identity_contract={},
        timeout_seconds=2,
        retry_interval_seconds=0.001,
        trust_existing_cache=False,
    )
    materializer.attach_body_prefetch(controller, binding)
    result: list[bytes] = []
    failures: list[BaseException] = []

    def source():
        yield materializer.read(Path(family.local_dir) / entry.name, epoch=1)

    loader = ClosableBatchPrefetcher(
        source(),
        prefetch_batches=PositiveInt.parse(1),
        stop_callback=controller.cancel_event.set,
        failure_check=controller.check_worker_errors,
    )

    def consume() -> None:
        try:
            result.append(next(loader))
        except BaseException as exc:
            failures.append(exc)

    consumer = threading.Thread(target=consume)
    try:
        consumer.start()
        assert shared.download_started.wait(timeout=5)
        time.sleep(0.05)
        assert consumer.is_alive()
        assert not (Path(family.local_dir) / entry.name).exists()
        # Startup/request-window demand is adopted rather than destructively
        # replaced by the current consumer.
        assert demand_path.read_bytes() == initial_demand
        release.set()
        consumer.join(timeout=5)
        assert not consumer.is_alive()
        assert failures == []
        assert result == [_payload("selection", 0)]
        assert shared.attempt_ranks[f"{family.remote_dir}/{entry.name}"] == {0}
        assert not demand_path.exists()
        assert materializer.read(
            Path(family.local_dir) / entry.name,
            epoch=2,
        ) == _payload("selection", 0)
        assert not demand_path.exists()
    finally:
        release.set()
        try:
            loader.close()
        finally:
            consumer.join(timeout=5)
            try:
                controller.stop_worker()
            finally:
                if controller.worker_is_terminal():
                    controller.release_session()

    assert not loader._thread.is_alive()
    assert not consumer.is_alive()
    assert controller._session_released


def test_paired_owner_failure_rolls_back_selection_and_never_publishes_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root, families = _families(tmp_path, n_entries=1, paired=True)
    nonce = "e" * 32
    control = _create_control_tree(
        cache_root=cache_root,
        nonce=nonce,
        world_size=1,
        families=families,
    )
    cfg = _worker_config(
        cache_root=cache_root,
        families=families,
        rank=0,
        world_size=1,
        nonce=nonce,
        timeout_seconds=0.02,
    )
    _write_rank_state(cfg=cfg, demand_indices_by_rank={0: (0,)})
    remote_root = tmp_path / "remote"
    _publish_remote_sources(root=remote_root, families=families)
    shared = _SharedBackend(
        remote_root,
        failing_path=f"{families[1].remote_dir}/{families[1].entries[0].name}",
    )
    monkeypatch.setattr(
        prefetch,
        "HfBucketBackend",
        lambda _bucket_id: shared.view(0),
    )
    context = mp.get_context("spawn")
    child_command, parent_command = context.Pipe(duplex=False)
    parent_status, child_status = context.Pipe(duplex=False)
    failures: list[BaseException] = []

    def run() -> None:
        try:
            _worker_main(cfg, child_command, child_status)
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=run, name="owner-0")
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(failures) == 1
    assert "persistent owner transport failure" in str(failures[0])
    selection_remote_path = f"{families[0].remote_dir}/{families[0].entries[0].name}"
    packed_remote_path = f"{families[1].remote_dir}/{families[1].entries[0].name}"
    assert shared.download_order[0] == selection_remote_path
    assert shared.download_order[1:]
    assert set(shared.download_order[1:]) == {packed_remote_path}
    for family in families:
        entry = family.entries[0]
        assert not (Path(family.local_dir) / entry.name).exists()
        assert not _receipt_path(
            control,
            family_id=family.family_id,
            inventory_index=0,
        ).exists()
        assert not list(
            (cache_root / ".remote-prefetch" / family.family_id).glob(".*owner-transfer.*")
        )
    assert not _ready_path(
        control,
        schedule_id=families[0].schedule_id,
        inventory_index=0,
    ).exists()
    status = prefetch._load_status(
        _status_path(control, 0),
        nonce=nonce,
        world_size=1,
    )
    assert status["state"] == "error"
    assert status["family_id"] == families[1].family_id
    assert status["inventory_index"] == 0
    assert status["body_name"] == families[1].entries[0].name
    parent_command.close()
    parent_status.close()


def test_cancel_during_transport_leaves_no_stage_final_receipt_or_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root, families = _families(tmp_path, n_entries=1, paired=False)
    nonce = "f" * 32
    control = _create_control_tree(
        cache_root=cache_root,
        nonce=nonce,
        world_size=1,
        families=families,
    )
    cfg = _worker_config(
        cache_root=cache_root,
        families=families,
        rank=0,
        world_size=1,
        nonce=nonce,
        timeout_seconds=1,
    )
    _write_rank_state(cfg=cfg, demand_indices_by_rank={0: (0,)})
    remote_root = tmp_path / "remote"
    _publish_remote_sources(root=remote_root, families=families)
    release = threading.Event()
    shared = _SharedBackend(remote_root, block_release=release)
    monkeypatch.setattr(
        prefetch,
        "HfBucketBackend",
        lambda _bucket_id: shared.view(0),
    )
    context = mp.get_context("spawn")
    child_command, parent_command = context.Pipe(duplex=False)
    parent_status, child_status = context.Pipe(duplex=False)
    thread = threading.Thread(
        target=_worker_main,
        args=(cfg, child_command, child_status),
        name="owner-0",
    )
    thread.start()
    assert parent_status.poll(5)
    assert parent_status.recv()["state"] == "ready"
    assert shared.download_started.wait(timeout=5)
    parent_command.send({"op": "cancel"})
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()

    family = families[0]
    entry = family.entries[0]
    assert not (Path(family.local_dir) / entry.name).exists()
    assert not _receipt_path(
        control,
        family_id=family.family_id,
        inventory_index=0,
    ).exists()
    assert not _ready_path(
        control,
        schedule_id=family.schedule_id,
        inventory_index=0,
    ).exists()
    assert not list((cache_root / ".remote-prefetch" / family.family_id).iterdir())
    status = prefetch._load_status(
        _status_path(control, 0),
        nonce=nonce,
        world_size=1,
    )
    assert status["state"] == "stopped"
    parent_command.close()
    parent_status.close()


def test_disk_admission_counts_larger_validation_body_as_active_temporary(
    tmp_path: Path,
) -> None:
    cache_root, train_families = _families(tmp_path, n_entries=4, paired=True)
    validation_dir = cache_root / "validation-selection"
    validation_dir.mkdir()
    validation_length = 10_000
    validation_family = RemoteBodyFamily(
        family_id="3" * 64,
        family_aggregate="d" * 64,
        kind="selection",
        local_dir=str(validation_dir),
        remote_dir="features/validation-selection",
        entries=(
            RemoteBodyEntry(
                name="shard_00000.safetensors.zst",
                byte_length=validation_length,
                content_sha256="e" * 64,
            ),
        ),
        preexisting_descriptors=(),
        schedule_id="f" * 64,
        train_schedule=False,
    )
    families = (*train_families, validation_family)

    admission = calculate_remote_prefetch_disk_admission(
        cache_root=cache_root,
        families=families,
        world_size=2,
        cache_shards=2,
        lookahead_shards_per_rank=2,
        safety_reserve_bytes=0,
    )

    expected_active = 0
    for rank in range(2):
        expected_active += max(
            entry.byte_length
            for family in families
            for index, entry in enumerate(family.entries)
            if owner_for_inventory_index(index, 2) == rank
        )
    assert admission.active_temporary_bytes == expected_active
    assert admission.active_temporary_bytes >= validation_length


def test_receipt_and_ready_records_reject_booleans_as_integer_fields(
    tmp_path: Path,
) -> None:
    cache_root, families = _families(tmp_path, n_entries=1, paired=False)
    nonce = "4" * 32
    control = _create_control_tree(
        cache_root=cache_root,
        nonce=nonce,
        world_size=1,
        families=families,
    )
    cfg = _worker_config(
        cache_root=cache_root,
        families=families,
        rank=0,
        world_size=1,
        nonce=nonce,
    )
    remote_root = tmp_path / "remote"
    _publish_remote_sources(root=remote_root, families=families)
    child_command, parent_command = mp.get_context("spawn").Pipe(duplex=False)
    prefetch._publish_schedule(
        cfg=cfg,
        backend=cast(Any, _SharedBackend(remote_root).view(0)),
        schedules=prefetch._schedule_map(families),
        schedule_id=families[0].schedule_id,
        inventory_index=0,
        generation=0,
        command_pipe=child_command,
    )
    receipt_path = _receipt_path(
        control,
        family_id=families[0].family_id,
        inventory_index=0,
    )
    ready_path = _ready_path(
        control,
        schedule_id=families[0].schedule_id,
        inventory_index=0,
    )
    valid_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    valid_ready = json.loads(ready_path.read_text(encoding="utf-8"))
    try:
        for field, value in (
            ("schema_version", True),
            ("inventory_index", False),
            ("owner_rank", False),
            ("generation", False),
        ):
            malformed = dict(valid_receipt)
            malformed[field] = value
            _atomic_record(malformed, receipt_path)
            with pytest.raises(RemoteTrainingError):
                prefetch._load_receipt(
                    receipt_path,
                    cfg=cfg,
                    family=families[0],
                    inventory_index=0,
                )
            _atomic_record(valid_receipt, receipt_path)

        for field, value in (
            ("schema_version", True),
            ("inventory_index", False),
            ("owner_rank", False),
            ("generation", False),
        ):
            malformed = dict(valid_ready)
            malformed[field] = value
            _atomic_record(malformed, ready_path)
            with pytest.raises(RemoteTrainingError):
                prefetch._load_ready(
                    ready_path,
                    cfg=cfg,
                    schedule_id=families[0].schedule_id,
                    inventory_index=0,
                    components=families,
                )
            _atomic_record(valid_ready, ready_path)
    finally:
        parent_command.close()
        child_command.close()


def test_worker_status_rejects_partial_identity_and_boolean_integer_fields(
    tmp_path: Path,
) -> None:
    cache_root, families = _families(tmp_path, n_entries=1, paired=False)
    nonce = "3" * 32
    control = _create_control_tree(
        cache_root=cache_root,
        nonce=nonce,
        world_size=1,
        families=families,
    )
    cfg = _worker_config(
        cache_root=cache_root,
        families=families,
        rank=0,
        world_size=1,
        nonce=nonce,
    )
    family = families[0]
    status_path = _status_path(control, 0)
    prefetch._write_worker_status(
        cfg,
        state="staging",
        family_id=family.family_id,
        inventory_index=0,
        body_name=family.entries[0].name,
    )
    valid = json.loads(status_path.read_text(encoding="utf-8"))

    for field in ("schema_version", "rank", "pid", "inventory_index", "updated_ns"):
        malformed = dict(valid)
        malformed[field] = False
        _atomic_record(malformed, status_path)
        with pytest.raises(RemoteTrainingError):
            prefetch._load_status(status_path, nonce=nonce, world_size=1)

    for field in ("family_id", "inventory_index", "body_name"):
        malformed = dict(valid)
        malformed[field] = None
        _atomic_record(malformed, status_path)
        with pytest.raises(RemoteTrainingError, match="all-or-none"):
            prefetch._load_status(status_path, nonce=nonce, world_size=1)

    malformed = dict(valid)
    malformed["body_name"] = "../escaped"
    _atomic_record(malformed, status_path)
    with pytest.raises(RemoteTrainingError, match="direct basename"):
        prefetch._load_status(status_path, nonce=nonce, world_size=1)
