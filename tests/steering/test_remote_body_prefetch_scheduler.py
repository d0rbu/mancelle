from __future__ import annotations

import fcntl
import hashlib
import multiprocessing as mp
import os
import shutil
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import hf_xet
import pytest

import steering.remote_body_prefetch as prefetch
from steering.remote_training import RemoteObjectInfo, RemoteTrainingError

_SCHEDULE_ID = "c" * 64
_NONCE = "d" * 32


def _payload(kind: str, index: int) -> bytes:
    return f"{kind}-signed-body-{index}".encode()


def _entry(kind: str, index: int) -> prefetch.RemoteBodyEntry:
    name = f"shard_{index:05d}.safetensors.zst"
    payload = _payload(kind, index)
    fingerprint = None
    if kind == "packed":
        digest = hashlib.sha256()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(payload)
        fingerprint = digest.hexdigest()
    return prefetch.RemoteBodyEntry(
        name=name,
        byte_length=len(payload),
        content_sha256=hashlib.sha256(payload).hexdigest(),
        file_fingerprint=fingerprint,
    )


def _families(
    tmp_path: Path,
    *,
    n_entries: int,
    paired: bool,
) -> tuple[Path, tuple[prefetch.RemoteBodyFamily, ...]]:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    selection_dir = cache_root / "selection"
    selection_dir.mkdir()
    families = [
        prefetch.RemoteBodyFamily(
            family_id="1" * 64,
            family_aggregate="a" * 64,
            kind="selection",
            local_dir=str(selection_dir),
            remote_dir="features/selection",
            entries=tuple(_entry("selection", index) for index in range(n_entries)),
            preexisting_descriptors=(),
            schedule_id=_SCHEDULE_ID,
            train_schedule=True,
        )
    ]
    if paired:
        packed_dir = cache_root / "packed"
        packed_dir.mkdir()
        families.append(
            prefetch.RemoteBodyFamily(
                family_id="2" * 64,
                family_aggregate="b" * 64,
                kind="packed",
                local_dir=str(packed_dir),
                remote_dir="features/packed",
                entries=tuple(_entry("packed", index) for index in range(n_entries)),
                preexisting_descriptors=(),
                schedule_id=_SCHEDULE_ID,
                train_schedule=True,
            )
        )
    return cache_root, tuple(families)


def _worker_config(
    *,
    cache_root: Path,
    families: tuple[prefetch.RemoteBodyFamily, ...],
    rank: int,
    world_size: int,
    lookahead: int = 2,
    cache_shards: int = 2,
) -> prefetch._WorkerConfig:
    return prefetch._WorkerConfig(
        bucket_id="owner/bucket",
        cache_root=str(cache_root),
        nonce=_NONCE,
        rank=rank,
        world_size=world_size,
        lookahead_shards_per_rank=lookahead,
        cache_shards=cache_shards,
        timeout_seconds=1,
        retry_interval_seconds=0.001,
        families=families,
    )


class _CountingBackend:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.download_counts: Counter[str] = Counter()
        self.download_order: list[str] = []
        self.stat_counts: Counter[str] = Counter()

    def _path(self, remote_path: str) -> Path:
        return self.root / remote_path

    def stat(self, remote_path: str) -> RemoteObjectInfo | None:
        self.stat_counts[remote_path] += 1
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
        self.download_counts[remote_path] += 1
        self.download_order.append(remote_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(remote_path), local_path)


def _publish_remote_sources(
    root: Path,
    families: Sequence[prefetch.RemoteBodyFamily],
) -> None:
    for family in families:
        for index, entry in enumerate(family.entries):
            path = root / family.remote_dir / entry.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_payload(family.kind, index))


@contextmanager
def _command_pipe() -> Iterator[Any]:
    receive, send = mp.get_context("spawn").Pipe(duplex=False)
    try:
        yield receive
    finally:
        receive.close()
        send.close()


def _publish(
    *,
    cfg: prefetch._WorkerConfig,
    backend: _CountingBackend,
    inventory_index: int,
    generation: int,
) -> None:
    with _command_pipe() as command:
        prefetch._publish_schedule(
            cfg=cfg,
            backend=cast(Any, backend),
            schedules=prefetch._schedule_map(cfg.families),
            schedule_id=_SCHEDULE_ID,
            inventory_index=inventory_index,
            generation=generation,
            command_pipe=command,
        )


@pytest.mark.parametrize(
    (
        "inventory_length",
        "world_size",
        "rank",
        "cursor_generations",
        "expected",
    ),
    [
        pytest.param(0, 1, 0, (), (), id="empty-inventory"),
        pytest.param(3, 4, 3, (0,), (), id="short-owner-has-no-entry"),
        pytest.param(
            3,
            4,
            1,
            (0,),
            ((_SCHEDULE_ID, 1, 1),),
            id="short-deduplicates-epoch-wrap",
        ),
        pytest.param(
            5,
            4,
            0,
            (1,),
            (
                (_SCHEDULE_ID, 4, 4),
                (_SCHEDULE_ID, 0, 5),
            ),
            id="uneven-inventory",
        ),
        pytest.param(
            17,
            4,
            2,
            (0, 9),
            (
                (_SCHEDULE_ID, 10, 10),
                (_SCHEDULE_ID, 2, 2),
            ),
            id="divergent-rank-frontiers",
        ),
        pytest.param(
            584,
            8,
            7,
            (579,),
            (
                (_SCHEDULE_ID, 583, 583),
                (_SCHEDULE_ID, 7, 591),
            ),
            id="production-shape-epoch-wrap",
        ),
    ],
)
def test_lookahead_obligations_are_nearest_owned_signed_indices(
    tmp_path: Path,
    inventory_length: int,
    world_size: int,
    rank: int,
    cursor_generations: tuple[int, ...],
    expected: tuple[tuple[str, int, int], ...],
) -> None:
    cache_root, families = _families(
        tmp_path,
        n_entries=inventory_length,
        paired=False,
    )
    cfg = _worker_config(
        cache_root=cache_root,
        families=families,
        rank=rank,
        world_size=world_size,
    )
    cursors = tuple({"generation": generation} for generation in cursor_generations)

    actual = prefetch._lookahead_obligations(
        cfg=cfg,
        cursors=cursors,
        train_schedule_id=_SCHEDULE_ID,
        inventory_length=inventory_length,
    )

    assert actual == list(expected)
    assert len(actual) <= cfg.lookahead_shards_per_rank
    assert len({index for _schedule, index, _generation in actual}) == len(actual)
    assert all(
        prefetch.owner_for_inventory_index(index, world_size) == rank
        for _schedule, index, _generation in actual
    )


def test_lookahead_publishes_pair_before_demand_and_demand_does_not_redownload(
    tmp_path: Path,
) -> None:
    cache_root, families = _families(tmp_path, n_entries=2, paired=True)
    cfg = _worker_config(
        cache_root=cache_root,
        families=families,
        rank=0,
        world_size=1,
    )
    control = prefetch._create_control_tree(
        cache_root=cache_root,
        nonce=cfg.nonce,
        world_size=cfg.world_size,
        families=families,
    )
    remote_root = tmp_path / "remote"
    _publish_remote_sources(remote_root, families)
    backend = _CountingBackend(remote_root)

    _publish(
        cfg=cfg,
        backend=backend,
        inventory_index=0,
        generation=0,
    )

    expected_remote_paths = [f"{family.remote_dir}/{family.entries[0].name}" for family in families]
    assert backend.download_order == expected_remote_paths
    assert backend.download_counts == Counter(dict.fromkeys(expected_remote_paths, 1))
    for family in families:
        entry = family.entries[0]
        final_path = Path(family.local_dir) / entry.name
        assert final_path.read_bytes() == _payload(family.kind, 0)
        receipt = prefetch._load_receipt(
            prefetch._receipt_path(
                control,
                family_id=family.family_id,
                inventory_index=0,
            ),
            cfg=cfg,
            family=family,
            inventory_index=0,
        )
        assert receipt["generation"] == 0
        assert receipt["owner_rank"] == 0
    ready_path = prefetch._ready_path(
        control,
        schedule_id=_SCHEDULE_ID,
        inventory_index=0,
    )
    ready = prefetch._load_ready(
        ready_path,
        cfg=cfg,
        schedule_id=_SCHEDULE_ID,
        inventory_index=0,
        components=families,
    )
    assert ready["generation"] == 0

    demand_path = prefetch._demand_path(
        control,
        requester_rank=0,
        schedule_id=_SCHEDULE_ID,
        inventory_index=0,
    )
    prefetch._atomic_record(
        {
            "schema_version": prefetch.REMOTE_PREFETCH_SCHEMA_VERSION,
            "nonce": cfg.nonce,
            "requester_rank": 0,
            "schedule_id": _SCHEDULE_ID,
            "epoch": 1,
            "inventory_index": 0,
            "generation": 0,
            "created_ns": 1,
        },
        demand_path,
    )
    demands = prefetch._scan_demands(
        cfg=cfg,
        schedules=prefetch._schedule_map(families),
    )
    demand = demands[(_SCHEDULE_ID, 0)]

    _publish(
        cfg=cfg,
        backend=backend,
        inventory_index=0,
        generation=cast(int, demand["generation"]),
    )

    assert backend.download_order == expected_remote_paths
    assert backend.download_counts == Counter(dict.fromkeys(expected_remote_paths, 1))
    for family in families:
        entry = family.entries[0]
        assert (Path(family.local_dir) / entry.name).read_bytes() == _payload(
            family.kind,
            0,
        )
    assert ready["components"] == [family.family_id for family in families]


def _write_progress(
    *,
    control: Path,
    cfg: prefetch._WorkerConfig,
    rank: int,
    generation: int,
) -> None:
    prefetch._atomic_record(
        {
            "schema_version": prefetch.REMOTE_PREFETCH_SCHEMA_VERSION,
            "nonce": cfg.nonce,
            "rank": rank,
            "schedule_id": _SCHEDULE_ID,
            "generation": generation,
        },
        prefetch._progress_path(control, rank),
    )


def _final_path(family: prefetch.RemoteBodyFamily, index: int) -> Path:
    return Path(family.local_dir) / family.entries[index].name


def _receipt_path(
    control: Path,
    family: prefetch.RemoteBodyFamily,
    index: int,
) -> Path:
    return prefetch._receipt_path(
        control,
        family_id=family.family_id,
        inventory_index=index,
    )


def _ready_path(control: Path, index: int) -> Path:
    return prefetch._ready_path(
        control,
        schedule_id=_SCHEDULE_ID,
        inventory_index=index,
    )


def test_owner_eviction_uses_min_progress_and_preserves_demand_lock_and_lookahead(
    tmp_path: Path,
) -> None:
    cache_root, families = _families(tmp_path, n_entries=8, paired=True)
    selection, packed = families
    cfgs = tuple(
        _worker_config(
            cache_root=cache_root,
            families=families,
            rank=rank,
            world_size=2,
            cache_shards=2,
        )
        for rank in range(2)
    )
    control = prefetch._create_control_tree(
        cache_root=cache_root,
        nonce=_NONCE,
        world_size=2,
        families=families,
    )
    remote_root = tmp_path / "remote"
    _publish_remote_sources(remote_root, families)
    backend = _CountingBackend(remote_root)
    for index in range(6):
        _publish(
            cfg=cfgs[index % 2],
            backend=backend,
            inventory_index=index,
            generation=index,
        )
    _publish(
        cfg=cfgs[0],
        backend=backend,
        inventory_index=6,
        generation=6,
    )
    future_final = _final_path(packed, 6)
    assert future_final.read_bytes() == _payload("packed", 6)
    _write_progress(control=control, cfg=cfgs[0], rank=0, generation=7)
    prefetch._evict_finalized_train(cfg=cfgs[0], demands={})
    assert all(_final_path(packed, index).is_file() for index in range(6))

    _write_progress(control=control, cfg=cfgs[0], rank=1, generation=4)
    demand_path = prefetch._demand_path(
        control,
        requester_rank=0,
        schedule_id=_SCHEDULE_ID,
        inventory_index=2,
    )
    prefetch._atomic_record(
        {
            "schema_version": prefetch.REMOTE_PREFETCH_SCHEMA_VERSION,
            "nonce": cfgs[0].nonce,
            "requester_rank": 0,
            "schedule_id": _SCHEDULE_ID,
            "epoch": 1,
            "inventory_index": 2,
            "generation": 2,
            "created_ns": 1,
        },
        demand_path,
    )
    lock_fd = prefetch._open_cache_lock(Path(packed.local_dir) / f".{packed.entries[0].name}.lock")
    fcntl.flock(lock_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    try:
        prefetch._evict_finalized_train(cfg=cfgs[0], demands={})
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    # min(progress)=4 and cache_shards=2 means only generations <3 are
    # eligible. Index 0 is lock-protected, index 2 is demand-protected,
    # index 4 is retained by the trailing window, and odd indices belong to
    # the other owner.
    assert all(_final_path(packed, index).is_file() for index in range(6))
    assert future_final.read_bytes() == _payload("packed", 6)

    # The worker's earlier scan was empty, but the requester-specific record
    # appeared before the exclusive body lock. The under-lock recheck must
    # preserve index 2 while still evicting another eligible owned body.
    prefetch._evict_finalized_train(cfg=cfgs[0], demands={})
    assert not _final_path(packed, 0).exists()
    assert not _receipt_path(control, packed, 0).exists()
    assert not _ready_path(control, 0).exists()
    assert not _final_path(selection, 0).exists()
    assert not _receipt_path(control, selection, 0).exists()
    assert _final_path(packed, 2).is_file()
    assert _final_path(packed, 4).is_file()
    assert all(_final_path(packed, index).is_file() for index in (1, 3, 5))
    assert future_final.read_bytes() == _payload("packed", 6)

    demand_path.unlink()
    prefetch._fsync_directory(demand_path.parent)
    prefetch._evict_finalized_train(cfg=cfgs[0], demands={})
    assert not _final_path(packed, 2).exists()
    assert not _receipt_path(control, packed, 2).exists()
    assert not _ready_path(control, 2).exists()
    assert not _final_path(selection, 2).exists()
    assert not _receipt_path(control, selection, 2).exists()
    assert _final_path(packed, 4).is_file()
    assert future_final.read_bytes() == _payload("packed", 6)

    # Rank 0 never evicts rank 1's bodies. Once rank 1 performs its own pass,
    # generation 1 is eligible while generation 3 is exactly at the boundary.
    prefetch._evict_finalized_train(cfg=cfgs[1], demands={})
    assert not _final_path(packed, 1).exists()
    assert _final_path(packed, 3).is_file()
    assert _final_path(packed, 5).is_file()
    assert future_final.read_bytes() == _payload("packed", 6)


def test_owner_eviction_pairs_selection_and_packed_across_epoch_wrap(
    tmp_path: Path,
) -> None:
    cache_root, families = _families(tmp_path, n_entries=4, paired=True)
    selection, packed = families
    cfgs = tuple(
        _worker_config(
            cache_root=cache_root,
            families=families,
            rank=rank,
            world_size=2,
            cache_shards=2,
        )
        for rank in range(2)
    )
    control = prefetch._create_control_tree(
        cache_root=cache_root,
        nonce=_NONCE,
        world_size=2,
        families=families,
    )
    remote_root = tmp_path / "remote"
    _publish_remote_sources(remote_root, families)
    backend = _CountingBackend(remote_root)
    for index in range(4):
        _publish(
            cfg=cfgs[index % 2],
            backend=backend,
            inventory_index=index,
            generation=index,
        )
    for cfg in cfgs:
        _write_progress(control=control, cfg=cfg, rank=cfg.rank, generation=4)

    # With four inventory entries, progress generation 4 is epoch 2 / shard 0.
    # A two-shard window evicts generations 0-2, retains previous shard 3, and
    # treats each paired selection/packed position as all-or-nothing.
    prefetch._evict_finalized_train(cfg=cfgs[0], demands={})
    for family in families:
        assert not _final_path(family, 0).exists()
        assert _final_path(family, 1).is_file()
        assert not _final_path(family, 2).exists()
        assert _final_path(family, 3).is_file()

    prefetch._evict_finalized_train(cfg=cfgs[1], demands={})
    for family in families:
        assert not _final_path(family, 1).exists()
        assert _final_path(family, 3).is_file()


def test_owner_eviction_defers_atomic_progress_snapshot_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root, families = _families(tmp_path, n_entries=1, paired=True)
    cfg = _worker_config(
        cache_root=cache_root,
        families=families,
        rank=0,
        world_size=1,
        cache_shards=1,
    )
    control = prefetch._create_control_tree(
        cache_root=cache_root,
        nonce=cfg.nonce,
        world_size=1,
        families=families,
    )
    remote_root = tmp_path / "remote"
    _publish_remote_sources(remote_root, families)
    _publish(
        cfg=cfg,
        backend=_CountingBackend(remote_root),
        inventory_index=0,
        generation=0,
    )
    _write_progress(control=control, cfg=cfg, rank=0, generation=2)
    original_load_progress = prefetch._load_progress
    calls = 0

    def race_once(
        path: Path,
        *,
        cfg: prefetch._WorkerConfig,
        train_schedule_id: str,
    ) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RemoteTrainingError(
                f"remote prefetch record changed while reading: {path}"
            )
        return original_load_progress(
            path,
            cfg=cfg,
            train_schedule_id=train_schedule_id,
        )

    monkeypatch.setattr(prefetch, "_load_progress", race_once)
    prefetch._evict_finalized_train(cfg=cfg, demands={})
    assert all(_final_path(family, 0).is_file() for family in families)

    prefetch._evict_finalized_train(cfg=cfg, demands={})
    assert calls == 2
    assert all(not _final_path(family, 0).exists() for family in families)


def test_demand_scan_handles_dead_denied_vanished_and_unexpected_temporaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root, families = _families(tmp_path, n_entries=1, paired=False)
    cfg = _worker_config(
        cache_root=cache_root,
        families=families,
        rank=0,
        world_size=1,
    )
    control = prefetch._create_control_tree(
        cache_root=cache_root,
        nonce=cfg.nonce,
        world_size=1,
        families=families,
    )
    directory = prefetch._demand_dir(control, 0)
    prefix = f"{_SCHEDULE_ID}.00000000.json.tmp"

    dead = directory / f"{prefix}.999999999.aaaaaaaa"
    dead.write_bytes(b"dead")
    assert (
        prefetch._scan_demands(
            cfg=cfg,
            schedules=prefetch._schedule_map(families),
        )
        == {}
    )
    assert not dead.exists()

    denied = directory / f"{prefix}.999999998.bbbbbbbb"
    denied.write_bytes(b"denied")
    with monkeypatch.context() as scoped:
        scoped.setattr(
            prefetch.os,
            "kill",
            lambda *_args: (_ for _ in ()).throw(PermissionError("denied")),
        )
        with pytest.raises(RemoteTrainingError, match="cannot prove"):
            prefetch._scan_demands(
                cfg=cfg,
                schedules=prefetch._schedule_map(families),
            )
    denied.unlink()

    vanished = directory / f"{prefix}.999999997.cccccccc"
    vanished.write_bytes(b"vanished")
    original_lstat = Path.lstat

    def vanish_during_lstat(path: Path) -> os.stat_result:
        if Path(path) == vanished:
            vanished.unlink()
            raise FileNotFoundError(vanished)
        return original_lstat(path)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "lstat", vanish_during_lstat)
        assert (
            prefetch._scan_demands(
                cfg=cfg,
                schedules=prefetch._schedule_map(families),
            )
            == {}
        )

    unreadable = directory / f"{prefix}.999999996.dddddddd"
    unreadable.write_bytes(b"unreadable")

    def deny_lstat(path: Path) -> os.stat_result:
        if Path(path) == unreadable:
            raise PermissionError("injected metadata denial")
        return original_lstat(path)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "lstat", deny_lstat)
        with pytest.raises(RemoteTrainingError, match="cannot inspect volatile"):
            prefetch._scan_demands(
                cfg=cfg,
                schedules=prefetch._schedule_map(families),
            )
    unreadable.unlink()

    unexpected = directory / "unexpected"
    unexpected.write_bytes(b"x")
    with pytest.raises(RemoteTrainingError, match="unexpected remote prefetch demand"):
        prefetch._scan_demands(
            cfg=cfg,
            schedules=prefetch._schedule_map(families),
        )


def test_demand_scan_rejects_persistent_invalid_temp_and_ignores_live_writer(
    tmp_path: Path,
) -> None:
    cache_root, families = _families(tmp_path, n_entries=1, paired=False)
    cfg = _worker_config(
        cache_root=cache_root,
        families=families,
        rank=0,
        world_size=1,
    )
    control = prefetch._create_control_tree(
        cache_root=cache_root,
        nonce=cfg.nonce,
        world_size=1,
        families=families,
    )
    directory = prefetch._demand_dir(control, 0)
    prefix = f"{_SCHEDULE_ID}.00000000.json.tmp"

    invalid = directory / f"{prefix}.{os.getpid()}.aaaaaaaa"
    invalid.mkdir()
    with pytest.raises(RemoteTrainingError, match="direct single-link regular file"):
        prefetch._scan_demands(
            cfg=cfg,
            schedules=prefetch._schedule_map(families),
        )
    invalid.rmdir()

    live = directory / f"{prefix}.{os.getpid()}.bbbbbbbb"
    live.write_bytes(b"publication in progress")
    assert (
        prefetch._scan_demands(
            cfg=cfg,
            schedules=prefetch._schedule_map(families),
        )
        == {}
    )
    assert live.exists()


@pytest.mark.parametrize("operation", ["scan", "has-requester"])
def test_demand_read_tolerates_concurrent_requester_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    cache_root, families = _families(tmp_path, n_entries=1, paired=False)
    cfg = _worker_config(
        cache_root=cache_root,
        families=families,
        rank=0,
        world_size=1,
    )
    control = prefetch._create_control_tree(
        cache_root=cache_root,
        nonce=cfg.nonce,
        world_size=1,
        families=families,
    )
    path = prefetch._demand_path(
        control,
        requester_rank=0,
        schedule_id=_SCHEDULE_ID,
        inventory_index=0,
    )
    prefetch._atomic_record(
        {
            "schema_version": prefetch.REMOTE_PREFETCH_SCHEMA_VERSION,
            "nonce": cfg.nonce,
            "requester_rank": 0,
            "schedule_id": _SCHEDULE_ID,
            "epoch": 1,
            "inventory_index": 0,
            "generation": 0,
            "created_ns": 1,
        },
        path,
    )
    original_open = os.open
    removed = False

    def remove_before_open(target, *args, **kwargs):
        nonlocal removed
        if Path(target) == path and not removed:
            removed = True
            path.unlink()
        return original_open(target, *args, **kwargs)

    monkeypatch.setattr(prefetch.os, "open", remove_before_open)
    schedules = prefetch._schedule_map(families)
    if operation == "scan":
        assert prefetch._scan_demands(cfg=cfg, schedules=schedules) == {}
    else:
        assert not prefetch._has_requester_demand(
            cfg=cfg,
            schedules=schedules,
            schedule_id=_SCHEDULE_ID,
            inventory_index=0,
        )
    assert removed


def test_demand_scan_defers_atomic_snapshot_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root, families = _families(tmp_path, n_entries=1, paired=False)
    cfg = _worker_config(
        cache_root=cache_root,
        families=families,
        rank=0,
        world_size=1,
    )
    control = prefetch._create_control_tree(
        cache_root=cache_root,
        nonce=cfg.nonce,
        world_size=1,
        families=families,
    )
    path = prefetch._demand_path(
        control,
        requester_rank=0,
        schedule_id=_SCHEDULE_ID,
        inventory_index=0,
    )
    prefetch._atomic_record(
        {
            "schema_version": prefetch.REMOTE_PREFETCH_SCHEMA_VERSION,
            "nonce": cfg.nonce,
            "requester_rank": 0,
            "schedule_id": _SCHEDULE_ID,
            "epoch": 1,
            "inventory_index": 0,
            "generation": 0,
            "created_ns": 1,
        },
        path,
    )
    schedules = prefetch._schedule_map(families)
    original_load = prefetch._load_demand_if_present
    calls = 0

    def race_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RemoteTrainingError(
                f"remote prefetch record changed while reading: {path}"
            )
        return original_load(*args, **kwargs)

    monkeypatch.setattr(prefetch, "_load_demand_if_present", race_once)
    assert prefetch._scan_demands(cfg=cfg, schedules=schedules) == {}
    assert prefetch._scan_demands(cfg=cfg, schedules=schedules) == {
        (_SCHEDULE_ID, 0): original_load(
            path,
            cfg=cfg,
            schedules=schedules,
            requester_rank=0,
        )
    }


def test_eviction_recheck_preserves_body_on_atomic_demand_snapshot_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root, families = _families(tmp_path, n_entries=1, paired=False)
    cfg = _worker_config(
        cache_root=cache_root,
        families=families,
        rank=0,
        world_size=1,
    )
    control = prefetch._create_control_tree(
        cache_root=cache_root,
        nonce=cfg.nonce,
        world_size=1,
        families=families,
    )
    path = prefetch._demand_path(
        control,
        requester_rank=0,
        schedule_id=_SCHEDULE_ID,
        inventory_index=0,
    )
    path.write_bytes(b"present")

    def race(*_args, **_kwargs):
        raise RemoteTrainingError(
            f"remote prefetch record changed while reading: {path}"
        )

    monkeypatch.setattr(prefetch, "_load_demand_if_present", race)
    assert prefetch._has_requester_demand(
        cfg=cfg,
        schedules=prefetch._schedule_map(families),
        schedule_id=_SCHEDULE_ID,
        inventory_index=0,
    )


@pytest.mark.parametrize("operation", ["scan", "has-requester"])
def test_demand_read_keeps_malformed_record_fail_loud(
    tmp_path: Path,
    operation: str,
) -> None:
    cache_root, families = _families(tmp_path, n_entries=1, paired=False)
    cfg = _worker_config(
        cache_root=cache_root,
        families=families,
        rank=0,
        world_size=1,
    )
    control = prefetch._create_control_tree(
        cache_root=cache_root,
        nonce=cfg.nonce,
        world_size=1,
        families=families,
    )
    path = prefetch._demand_path(
        control,
        requester_rank=0,
        schedule_id=_SCHEDULE_ID,
        inventory_index=0,
    )
    prefetch._atomic_record(
        {
            "schema_version": prefetch.REMOTE_PREFETCH_SCHEMA_VERSION,
            "nonce": "e" * 32,
            "requester_rank": 0,
            "schedule_id": _SCHEDULE_ID,
            "epoch": 1,
            "inventory_index": 0,
            "generation": 0,
            "created_ns": 1,
        },
        path,
    )
    schedules = prefetch._schedule_map(families)

    with pytest.raises(RemoteTrainingError, match="body demand has stale nonce"):
        if operation == "scan":
            prefetch._scan_demands(cfg=cfg, schedules=schedules)
        else:
            prefetch._has_requester_demand(
                cfg=cfg,
                schedules=schedules,
                schedule_id=_SCHEDULE_ID,
                inventory_index=0,
            )


@pytest.mark.parametrize("operation", ["scan", "has-requester"])
def test_demand_read_keeps_control_io_errors_fail_loud(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    cache_root, families = _families(tmp_path, n_entries=1, paired=False)
    cfg = _worker_config(
        cache_root=cache_root,
        families=families,
        rank=0,
        world_size=1,
    )
    control = prefetch._create_control_tree(
        cache_root=cache_root,
        nonce=cfg.nonce,
        world_size=1,
        families=families,
    )
    path = prefetch._demand_path(
        control,
        requester_rank=0,
        schedule_id=_SCHEDULE_ID,
        inventory_index=0,
    )
    original_open = os.open

    def deny_open(target, *args, **kwargs):
        if Path(target) == path:
            raise PermissionError("injected demand I/O denial")
        return original_open(target, *args, **kwargs)

    monkeypatch.setattr(prefetch.os, "open", deny_open)
    schedules = prefetch._schedule_map(families)
    with pytest.raises(RemoteTrainingError, match="cannot open remote prefetch record"):
        if operation == "scan":
            path.write_bytes(b"{}")
            prefetch._scan_demands(cfg=cfg, schedules=schedules)
        else:
            prefetch._has_requester_demand(
                cfg=cfg,
                schedules=schedules,
                schedule_id=_SCHEDULE_ID,
                inventory_index=0,
            )


def test_cursor_scan_skips_absent_and_rejects_wrong_rank_namespace(
    tmp_path: Path,
) -> None:
    cache_root, families = _families(tmp_path, n_entries=1, paired=False)
    cfg = _worker_config(
        cache_root=cache_root,
        families=families,
        rank=0,
        world_size=2,
    )
    control = prefetch._create_control_tree(
        cache_root=cache_root,
        nonce=cfg.nonce,
        world_size=2,
        families=families,
    )
    assert (
        prefetch._scan_cursors(
            cfg=cfg,
            train_schedule_id=_SCHEDULE_ID,
            inventory_length=1,
        )
        == []
    )
    prefetch._atomic_record(
        {
            "schema_version": prefetch.REMOTE_PREFETCH_SCHEMA_VERSION,
            "nonce": cfg.nonce,
            "rank": 1,
            "schedule_id": _SCHEDULE_ID,
            "epoch": 1,
            "inventory_index": 0,
            "generation": 0,
        },
        prefetch._cursor_path(control, 0),
    )
    with pytest.raises(RemoteTrainingError, match="wrong namespace"):
        prefetch._scan_cursors(
            cfg=cfg,
            train_schedule_id=_SCHEDULE_ID,
            inventory_length=1,
        )


def test_cursor_scan_defers_atomic_snapshot_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root, families = _families(tmp_path, n_entries=1, paired=False)
    cfg = _worker_config(
        cache_root=cache_root,
        families=families,
        rank=0,
        world_size=1,
    )
    control = prefetch._create_control_tree(
        cache_root=cache_root,
        nonce=cfg.nonce,
        world_size=1,
        families=families,
    )
    path = prefetch._cursor_path(control, 0)
    prefetch._atomic_record(
        {
            "schema_version": prefetch.REMOTE_PREFETCH_SCHEMA_VERSION,
            "nonce": cfg.nonce,
            "rank": 0,
            "schedule_id": _SCHEDULE_ID,
            "epoch": 1,
            "inventory_index": 0,
            "generation": 0,
        },
        path,
    )
    original_load = prefetch._load_cursor
    calls = 0

    def race_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RemoteTrainingError(
                f"remote prefetch record changed while reading: {path}"
            )
        return original_load(*args, **kwargs)

    monkeypatch.setattr(prefetch, "_load_cursor", race_once)
    assert (
        prefetch._scan_cursors(
            cfg=cfg,
            train_schedule_id=_SCHEDULE_ID,
            inventory_length=1,
        )
        == []
    )
    cursors = prefetch._scan_cursors(
        cfg=cfg,
        train_schedule_id=_SCHEDULE_ID,
        inventory_length=1,
    )
    assert calls == 2
    assert cursors == [original_load(path, cfg=cfg, train_schedule_id=_SCHEDULE_ID, inventory_length=1)]


def test_owner_eviction_handles_disappearance_and_warm_preexisting_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root, families = _families(tmp_path, n_entries=1, paired=True)
    packed = families[1]
    final = _final_path(packed, 0)
    final.write_bytes(_payload("packed", 0))
    cfg = _worker_config(
        cache_root=cache_root,
        families=families,
        rank=0,
        world_size=1,
        cache_shards=1,
    )
    control = prefetch._create_control_tree(
        cache_root=cache_root,
        nonce=cfg.nonce,
        world_size=1,
        families=families,
    )
    _write_progress(control=control, cfg=cfg, rank=0, generation=2)
    original_open = prefetch._open_cache_lock

    def unlink_before_lock(path: Path) -> int:
        if Path(path).name == f".{packed.entries[0].name}.lock":
            final.unlink(missing_ok=True)
        return original_open(path)

    with monkeypatch.context() as scoped:
        scoped.setattr(prefetch, "_open_cache_lock", unlink_before_lock)
        prefetch._evict_finalized_train(cfg=cfg, demands={})

    final.write_bytes(_payload("packed", 0))
    descriptor = prefetch._descriptor_dict(prefetch._direct_file_stat(final))
    warm_packed = replace(
        packed,
        preexisting_descriptors=(
            (
                packed.entries[0].name,
                prefetch._descriptor_dict_to_tuple(descriptor),
            ),
        ),
    )
    warm_families = (families[0], warm_packed)
    warm_cfg = replace(cfg, families=warm_families)
    prefetch._evict_finalized_train(cfg=warm_cfg, demands={})
    assert not final.exists()
