"""Strict rank-striped remote body prefetch for remote encoder-decoder training.

One spawned process per DDP rank owns transport for inventory positions
``index % world_size == rank``.  Parent processes are cache-only consumers:
they publish requester-specific demand records, wait for the nominal owner,
and read finalized bytes under the existing per-body lock.

This module is operational by design.  Its nonce-scoped control records,
lookahead, receipts, worker state, and disk admission never participate in
semantic run identity or checkpoint payloads.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import re
import shutil
import stat
import threading
import time
import traceback
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import torch.distributed as dist
from beartype import beartype
from loguru import logger

from nanochat.atomic_io import atomic_json_dump, atomic_write_bytes
from steering._common.strict_json import canonical_json
from steering.refined_types import (
    LowercaseSha256,
    NonNegativeInt,
    NonNegativeIntLike,
    PositiveInt,
    PositiveIntLike,
)
from steering.remote_training import (
    HfBucketBackend,
    RemoteMaterializationCancelled,
    RemotePackedRowMaterializer,
    RemoteSelectionMaterializer,
    RemoteTrainingConfig,
    RemoteTrainingError,
    _download_with_retry,
    _open_cache_lock,
    _read_cache_body_bytes,
)

if TYPE_CHECKING:
    _RemoteArtifactMapping = Mapping[str, Any]
    _RemoteBodyFamilySequence = Sequence["RemoteBodyFamily"]
else:
    # Runtime container checks are deliberately outer-only. Parameterized
    # container checking uses randomized sampling and these APIs validate
    # family/artifact relationships explicitly below.
    _RemoteArtifactMapping = Mapping
    _RemoteBodyFamilySequence = Sequence

REMOTE_PREFETCH_SCHEMA_VERSION = 1
REMOTE_PREFETCH_RECEIPT_SCHEMA_VERSION = 1
REMOTE_PREFETCH_DISK_SAFETY_RESERVE_BYTES = 1 << 30
REMOTE_PREFETCH_RECORD_MAX_BYTES = 16 << 10
REMOTE_PREFETCH_RECEIPT_BOUND_BYTES = 4 << 10
REMOTE_PREFETCH_CONTROL_BOUND_BYTES = 4 << 10
REMOTE_PREFETCH_POLL_SECONDS = 0.05
REMOTE_PREFETCH_LOCAL_FAILURE_CHECK_INTERVAL_SECONDS = 1.0
REMOTE_PREFETCH_PEER_STATUS_SWEEP_INTERVAL_SECONDS = 10.0
REMOTE_PREFETCH_GRACEFUL_JOIN_SECONDS = 5.0
REMOTE_PREFETCH_HARD_JOIN_SECONDS = 5.0
REMOTE_PREFETCH_ERROR_TEXT_MAX_CHARS = 2_000

_BODY_KINDS = frozenset({"selection", "packed"})
_STATUS_KEYS = frozenset(
    {
        "schema_version",
        "nonce",
        "rank",
        "pid",
        "state",
        "family_id",
        "inventory_index",
        "body_name",
        "updated_ns",
        "error_type",
        "error_message",
        "traceback",
    }
)
_CURSOR_KEYS = frozenset(
    {
        "schema_version",
        "nonce",
        "rank",
        "schedule_id",
        "epoch",
        "inventory_index",
        "generation",
    }
)
_PROGRESS_KEYS = frozenset(
    {
        "schema_version",
        "nonce",
        "rank",
        "schedule_id",
        "generation",
    }
)
_DEMAND_KEYS = frozenset(
    {
        "schema_version",
        "nonce",
        "requester_rank",
        "schedule_id",
        "epoch",
        "inventory_index",
        "generation",
        "created_ns",
    }
)
_READY_KEYS = frozenset(
    {
        "schema_version",
        "nonce",
        "schedule_id",
        "inventory_index",
        "generation",
        "owner_rank",
        "components",
    }
)
_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "nonce",
        "family_id",
        "family_aggregate",
        "inventory_index",
        "basename",
        "byte_length",
        "content_sha256",
        "file_fingerprint",
        "owner_rank",
        "generation",
    }
)
_DESCRIPTOR_KEYS = frozenset(
    {
        "device",
        "inode",
        "mode",
        "size",
        "mtime_ns",
        "ctime_ns",
        "link_count",
    }
)
_CONTROL_SUBDIRS = (
    "cursors",
    "demands",
    "progress",
    "status",
    "receipts",
    "ready",
)


@dataclass(frozen=True)
class RemoteBodyEntry:
    name: str
    byte_length: int
    content_sha256: str
    file_fingerprint: str | None = None


@dataclass(frozen=True)
class RemoteBodyFamily:
    family_id: str
    family_aggregate: str
    kind: Literal["selection", "packed"]
    local_dir: str
    remote_dir: str
    entries: tuple[RemoteBodyEntry, ...]
    preexisting_descriptors: tuple[tuple[str, tuple[int, ...]], ...]
    schedule_id: str
    train_schedule: bool

    @beartype
    def entry(self, inventory_index: NonNegativeIntLike) -> RemoteBodyEntry:
        if inventory_index >= len(self.entries):
            raise RemoteTrainingError(
                f"remote prefetch inventory index is invalid for {self.family_id}: "
                f"{inventory_index!r}"
            )
        return self.entries[inventory_index]

    def preexisting_descriptor(self, name: str) -> dict[str, int] | None:
        for recorded_name, values in self.preexisting_descriptors:
            if recorded_name == name:
                return _descriptor_tuple_to_dict(values)
        return None


@dataclass(frozen=True)
class RemoteBodyBinding:
    family_id: str
    schedule_id: str
    kind: Literal["selection", "packed"]
    index_by_name: tuple[tuple[str, int], ...]
    paired: bool
    train_schedule: bool

    def inventory_index(self, name: str) -> int:
        for recorded_name, index in self.index_by_name:
            if recorded_name == name:
                return index
        raise RemoteTrainingError(
            f"remote prefetch binding {self.family_id} has no inventory entry {name!r}"
        )


@dataclass(frozen=True)
class RemotePrefetchDiskAdmission:
    peak_bytes: int
    current_managed_bytes: int
    additional_required_bytes: int
    available_bytes: int
    safety_reserve_bytes: int
    selection_final_bytes: int
    packed_final_bytes: int
    active_temporary_bytes: int
    receipt_and_control_bytes: int


@dataclass(frozen=True)
class _WorkerConfig:
    bucket_id: str
    cache_root: str
    nonce: str
    rank: int
    world_size: int
    lookahead_shards_per_rank: int
    cache_shards: int
    timeout_seconds: float
    retry_interval_seconds: float
    families: tuple[RemoteBodyFamily, ...]


@dataclass(frozen=True)
class _NewPublication:
    family_id: str
    inventory_index: int


class _WorkerCancelled(Exception):
    pass


@beartype
def owner_for_inventory_index(
    inventory_index: NonNegativeIntLike,
    world_size: PositiveIntLike,
) -> int:
    """Return the sole transport owner for one signed inventory position."""
    return inventory_index % world_size


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _bounded_error_text(error: BaseException) -> str:
    text = str(error)
    if len(text) <= REMOTE_PREFETCH_ERROR_TEXT_MAX_CHARS:
        return text
    omitted = len(text) - REMOTE_PREFETCH_ERROR_TEXT_MAX_CHARS
    return text[:REMOTE_PREFETCH_ERROR_TEXT_MAX_CHARS] + f"... <{omitted} error characters omitted>"


def _require_sha256(value: Any, *, surface: str) -> LowercaseSha256:
    if not isinstance(value, LowercaseSha256):
        raise RemoteTrainingError(f"{surface} must be 64 lowercase hexadecimal characters")
    return LowercaseSha256.parse(value)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _ensure_direct_directory(path: Path, *, create: bool = False) -> None:
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        info = path.lstat()
    except OSError as exc:
        raise RemoteTrainingError(
            f"remote prefetch directory is unavailable: {path}: {exc}"
        ) from exc
    if not stat.S_ISDIR(info.st_mode) or path.is_symlink():
        raise RemoteTrainingError(f"remote prefetch path must be a direct directory: {path}")


def _direct_file_stat(path: Path) -> os.stat_result:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RemoteTrainingError("remote prefetch requires os.O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RemoteTrainingError(f"cannot open remote prefetch file {path}: {exc}") from exc
    try:
        opened = os.fstat(fd)
        linked = path.lstat()
        if not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(linked.st_mode):
            raise RemoteTrainingError(f"remote prefetch file must be a direct regular file: {path}")
        if (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino):
            raise RemoteTrainingError(f"remote prefetch file changed while opening: {path}")
        if opened.st_nlink != 1 or linked.st_nlink != 1:
            raise RemoteTrainingError(
                f"remote prefetch file must have exactly one hard link: {path}"
            )
        return opened
    finally:
        os.close(fd)


def _volatile_direct_file_exists(path: Path) -> bool:
    """Validate one enumerated writer temp without requiring it to persist."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise RemoteTrainingError(
            f"cannot inspect volatile remote prefetch temporary {path}: {exc}"
        ) from exc
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RemoteTrainingError(
            f"remote prefetch temporary must be a direct single-link regular file: {path}"
        )
    return True


def _descriptor_dict(info: os.stat_result) -> dict[str, int]:
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "mode": info.st_mode,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "link_count": info.st_nlink,
    }


def _descriptor_dict_to_tuple(value: Mapping[str, Any]) -> tuple[int, ...]:
    if set(value) != _DESCRIPTOR_KEYS or any(type(value[key]) is not int for key in value):
        raise RemoteTrainingError("remote prefetch descriptor record is malformed")
    return (
        value["device"],
        value["inode"],
        value["mode"],
        value["size"],
        value["mtime_ns"],
        value["ctime_ns"],
        value["link_count"],
    )


def _descriptor_tuple_to_dict(value: Sequence[int]) -> dict[str, int]:
    if len(value) != 7 or any(type(item) is not int for item in value):
        raise RemoteTrainingError("remote prefetch descriptor tuple is malformed")
    return dict(
        zip(
            (
                "device",
                "inode",
                "mode",
                "size",
                "mtime_ns",
                "ctime_ns",
                "link_count",
            ),
            value,
            strict=True,
        )
    )


def _atomic_snapshot_descriptor_is_stable(
    before: os.stat_result,
    after: os.stat_result,
) -> bool:
    """Allow only the metadata transition caused by unlinking an opened inode."""
    content_identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    content_identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    )
    if (
        content_identity_before != content_identity_after
        or before.st_nlink not in {0, 1}
        or after.st_nlink not in {0, 1}
    ):
        return False
    # Atomic replacement may unlink the old opened inode, changing ctime and
    # nlink without changing its bytes.  If it remained linked, a ctime change
    # is an in-place mutation and must still fail loudly.
    return (before.st_nlink == after.st_nlink and before.st_ctime_ns == after.st_ctime_ns) or (
        before.st_nlink == 1 and after.st_nlink == 0
    )


def _read_opened_direct_bytes(
    path: Path,
    *,
    max_bytes: int,
    require_stable_path: bool,
) -> bytes:
    """Read one direct regular file through a descriptor-pinned snapshot.

    Atomic control-record publication intentionally replaces the pathname's
    inode.  Such readers validate the opened descriptor before and after the
    read but must not require the pathname to keep naming that inode: the old
    descriptor is still a coherent, fully fsynced record snapshot.  Immutable
    artifacts additionally require pathname identity to remain stable.
    """
    if not hasattr(os, "O_NOFOLLOW"):
        raise RemoteTrainingError("remote prefetch requires os.O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RemoteTrainingError(f"cannot open remote prefetch record {path}: {exc}") from exc
    try:
        before = os.fstat(fd)
        valid_link_count = (
            before.st_nlink == 1 if require_stable_path else before.st_nlink in {0, 1}
        )
        if not stat.S_ISREG(before.st_mode) or not valid_link_count:
            raise RemoteTrainingError(
                f"remote prefetch record must be a direct single-link regular file: {path}"
            )
        if require_stable_path:
            linked_before = path.lstat()
            if not stat.S_ISREG(linked_before.st_mode) or linked_before.st_nlink != 1:
                raise RemoteTrainingError(
                    f"remote prefetch record must be a direct single-link regular file: {path}"
                )
            if (before.st_dev, before.st_ino) != (
                linked_before.st_dev,
                linked_before.st_ino,
            ):
                raise RemoteTrainingError(f"remote prefetch record changed while opening: {path}")
        if before.st_size > max_bytes:
            raise RemoteTrainingError(f"remote prefetch record exceeds {max_bytes} bytes: {path}")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            data = handle.read(max_bytes + 1)
        after = os.fstat(fd)
        descriptor_is_stable = (
            _descriptor_dict(before) == _descriptor_dict(after)
            if require_stable_path
            else _atomic_snapshot_descriptor_is_stable(before, after)
        )
        if len(data) > max_bytes or not descriptor_is_stable:
            raise RemoteTrainingError(f"remote prefetch record changed while reading: {path}")
        if require_stable_path:
            linked_after = path.lstat()
            if _descriptor_dict(before) != _descriptor_dict(linked_after):
                raise RemoteTrainingError(f"remote prefetch record changed while reading: {path}")
        return data
    except OSError as exc:
        raise RemoteTrainingError(f"cannot read remote prefetch record {path}: {exc}") from exc
    finally:
        os.close(fd)


def _read_direct_bytes(path: Path, *, max_bytes: int) -> bytes:
    """Read an immutable direct file and require stable pathname identity."""
    return _read_opened_direct_bytes(
        path,
        max_bytes=max_bytes,
        require_stable_path=True,
    )


def _read_atomic_record_snapshot_bytes(path: Path, *, max_bytes: int) -> bytes:
    """Read either coherent generation of an atomically replaced record."""
    return _read_opened_direct_bytes(
        path,
        max_bytes=max_bytes,
        require_stable_path=False,
    )


def _is_transient_mutable_record_read(exc: RemoteTrainingError) -> bool:
    """Return whether a mutable atomic record should be sampled again later."""
    return isinstance(exc.__cause__, FileNotFoundError) or str(exc).startswith(
        "remote prefetch record changed while reading: "
    )


def _load_json_record(
    path: Path,
    *,
    expected_keys: frozenset[str],
    surface: str,
) -> dict[str, Any]:
    """Load one `_atomic_record` control snapshot and validate its exact schema."""
    try:
        raw = json.loads(
            _read_atomic_record_snapshot_bytes(
                path,
                max_bytes=REMOTE_PREFETCH_RECORD_MAX_BYTES,
            ).decode("utf-8")
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RemoteTrainingError(f"{surface} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(raw, dict) or set(raw) != expected_keys:
        keys = set(raw) if isinstance(raw, dict) else set()
        raise RemoteTrainingError(
            f"{surface} must contain exactly {sorted(expected_keys)}; "
            f"got keys={sorted(keys, key=str)}"
        )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != REMOTE_PREFETCH_SCHEMA_VERSION
    ):
        raise RemoteTrainingError(
            f"{surface}.schema_version must be {REMOTE_PREFETCH_SCHEMA_VERSION}"
        )
    return raw


def _atomic_record(value: Mapping[str, Any], path: Path) -> None:
    atomic_json_dump(dict(value), path)
    _fsync_directory(path.parent)


def _safe_remove_tree(path: Path) -> None:
    """Remove one validated stale nonce directory without following links."""
    _ensure_direct_directory(path)
    for child in path.iterdir():
        info = child.lstat()
        if stat.S_ISDIR(info.st_mode) and not child.is_symlink():
            _safe_remove_tree(child)
        elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
            child.unlink()
        else:
            raise RemoteTrainingError(
                f"refusing to remove unsafe stale remote prefetch entry: {child}"
            )
    path.rmdir()


def _family_id(*, kind: str, aggregate: str, remote_dir: str) -> str:
    _require_sha256(aggregate, surface="remote body family aggregate")
    return _sha256_json(
        {
            "schema_version": REMOTE_PREFETCH_SCHEMA_VERSION,
            "kind": kind,
            "aggregate": aggregate,
            "remote_dir": remote_dir,
        }
    )


def _preexisting_tuple(
    descriptors: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        (name, _descriptor_dict_to_tuple(descriptors[name])) for name in sorted(descriptors)
    )


def build_remote_body_plan(
    *,
    train_selection_dir: Path,
    selection_materializers: Sequence[RemoteSelectionMaterializer],
    packed_materializer: RemotePackedRowMaterializer | None,
) -> tuple[
    tuple[RemoteBodyFamily, ...],
    dict[str, RemoteBodyBinding],
    RemoteBodyBinding | None,
]:
    """Build signed family specs and cache-only materializer bindings."""
    if not selection_materializers:
        raise RemoteTrainingError("remote body prefetch requires selection materializers")
    train_key = str(Path(train_selection_dir).resolve())
    selection_by_dir = {
        str(materializer.local_dir.resolve()): materializer
        for materializer in selection_materializers
    }
    if len(selection_by_dir) != len(selection_materializers):
        raise RemoteTrainingError("remote body prefetch received duplicate selection caches")
    train_selection = selection_by_dir.get(train_key)
    if train_selection is None:
        raise RemoteTrainingError(
            f"remote body prefetch has no train selection materializer for {train_selection_dir}"
        )

    families: list[RemoteBodyFamily] = []
    bindings: dict[str, RemoteBodyBinding] = {}

    train_selection_entries = tuple(
        RemoteBodyEntry(
            name=name,
            byte_length=int(entry["byte_length"]),
            content_sha256=str(entry["content_sha256"]),
        )
        for name, entry in train_selection.inventory.items()
    )
    if not train_selection_entries:
        raise RemoteTrainingError("remote body prefetch train inventory must not be empty")
    train_selection_aggregate = str(
        train_selection.loader_certificate["selection_inventory_aggregate"]
    )
    train_selection_family_id = _family_id(
        kind="selection",
        aggregate=train_selection_aggregate,
        remote_dir=train_selection.remote_dir,
    )

    packed_family: RemoteBodyFamily | None = None
    if packed_materializer is not None:
        packed_entries = tuple(
            RemoteBodyEntry(
                name=str(entry["name"]),
                byte_length=int(entry["byte_length"]),
                content_sha256=str(entry["content_sha256"]),
                file_fingerprint=str(entry["file_fingerprint"]),
            )
            for entry in cast(list[dict[str, Any]], packed_materializer.inventory["shards"])
        )
        if tuple(entry.name for entry in packed_entries) != tuple(
            entry.name for entry in train_selection_entries
        ):
            raise RemoteTrainingError(
                "train packed and selection signed inventories must have identical "
                "name-sorted entries for strict paired ownership"
            )
        packed_aggregate = str(packed_materializer.inventory["aggregate_sha256"])
        packed_family_id = _family_id(
            kind="packed",
            aggregate=packed_aggregate,
            remote_dir=packed_materializer.remote_dir,
        )
        schedule_id = _sha256_json(
            {
                "schema_version": REMOTE_PREFETCH_SCHEMA_VERSION,
                "selection_family_id": train_selection_family_id,
                "packed_family_id": packed_family_id,
            }
        )
        packed_family = RemoteBodyFamily(
            family_id=packed_family_id,
            family_aggregate=packed_aggregate,
            kind="packed",
            local_dir=str(packed_materializer.local_dir),
            remote_dir=packed_materializer.remote_dir,
            entries=packed_entries,
            preexisting_descriptors=_preexisting_tuple(packed_materializer.preexisting_descriptors),
            schedule_id=schedule_id,
            train_schedule=True,
        )
    else:
        schedule_id = train_selection_family_id

    train_selection_family = RemoteBodyFamily(
        family_id=train_selection_family_id,
        family_aggregate=train_selection_aggregate,
        kind="selection",
        local_dir=str(train_selection.local_dir),
        remote_dir=train_selection.remote_dir,
        entries=train_selection_entries,
        preexisting_descriptors=_preexisting_tuple(train_selection.preexisting_descriptors),
        schedule_id=schedule_id,
        train_schedule=True,
    )
    families.append(train_selection_family)
    if packed_family is not None:
        families.append(packed_family)

    paired = packed_family is not None
    train_selection_binding = RemoteBodyBinding(
        family_id=train_selection_family.family_id,
        schedule_id=schedule_id,
        kind="selection",
        index_by_name=tuple(
            (entry.name, index) for index, entry in enumerate(train_selection_entries)
        ),
        paired=paired,
        train_schedule=True,
    )
    bindings[train_key] = train_selection_binding
    packed_binding: RemoteBodyBinding | None = None
    if packed_family is not None:
        packed_binding = RemoteBodyBinding(
            family_id=packed_family.family_id,
            schedule_id=schedule_id,
            kind="packed",
            index_by_name=tuple(
                (entry.name, index) for index, entry in enumerate(packed_family.entries)
            ),
            paired=True,
            train_schedule=True,
        )

    for local_key, materializer in sorted(selection_by_dir.items()):
        if local_key == train_key:
            continue
        entries = tuple(
            RemoteBodyEntry(
                name=name,
                byte_length=int(entry["byte_length"]),
                content_sha256=str(entry["content_sha256"]),
            )
            for name, entry in materializer.inventory.items()
        )
        if not entries:
            raise RemoteTrainingError(
                f"remote body prefetch validation inventory must not be empty: {local_key}"
            )
        aggregate = str(materializer.loader_certificate["selection_inventory_aggregate"])
        family_id = _family_id(
            kind="selection",
            aggregate=aggregate,
            remote_dir=materializer.remote_dir,
        )
        family = RemoteBodyFamily(
            family_id=family_id,
            family_aggregate=aggregate,
            kind="selection",
            local_dir=str(materializer.local_dir),
            remote_dir=materializer.remote_dir,
            entries=entries,
            preexisting_descriptors=_preexisting_tuple(materializer.preexisting_descriptors),
            schedule_id=family_id,
            train_schedule=False,
        )
        families.append(family)
        bindings[local_key] = RemoteBodyBinding(
            family_id=family_id,
            schedule_id=family_id,
            kind="selection",
            index_by_name=tuple((entry.name, index) for index, entry in enumerate(entries)),
            paired=False,
            train_schedule=False,
        )

    family_ids = [family.family_id for family in families]
    if len(set(family_ids)) != len(family_ids):
        raise RemoteTrainingError(
            "remote body prefetch family identities collide across configured caches"
        )
    return tuple(families), bindings, packed_binding


def _schedule_map(
    families: Sequence[RemoteBodyFamily],
) -> dict[str, tuple[RemoteBodyFamily, ...]]:
    grouped: dict[str, list[RemoteBodyFamily]] = {}
    for family in families:
        grouped.setdefault(family.schedule_id, []).append(family)
    result: dict[str, tuple[RemoteBodyFamily, ...]] = {}
    for schedule_id, members in grouped.items():
        ordered = sorted(members, key=lambda family: 0 if family.kind == "selection" else 1)
        lengths = {len(member.entries) for member in ordered}
        names = {tuple(entry.name for entry in member.entries) for member in ordered}
        if len(lengths) != 1 or len(names) != 1:
            raise RemoteTrainingError(
                f"remote prefetch schedule {schedule_id} does not have aligned inventories"
            )
        if len(ordered) not in {1, 2} or (
            len(ordered) == 2
            and tuple(member.kind for member in ordered) != ("selection", "packed")
        ):
            raise RemoteTrainingError(
                f"remote prefetch schedule {schedule_id} has invalid components"
            )
        result[schedule_id] = tuple(ordered)
    train_schedules = {family.schedule_id for family in families if family.train_schedule}
    if len(train_schedules) != 1:
        raise RemoteTrainingError(
            "remote body prefetch requires exactly one train inventory schedule"
        )
    return result


def _plan_identity(families: Sequence[RemoteBodyFamily]) -> str:
    return _sha256_json(
        [
            {
                "family_id": family.family_id,
                "family_aggregate": family.family_aggregate,
                "kind": family.kind,
                "schedule_id": family.schedule_id,
                "train_schedule": family.train_schedule,
                "names": [entry.name for entry in family.entries],
            }
            for family in sorted(families, key=lambda item: item.family_id)
        ]
    )


@beartype
def calculate_remote_prefetch_disk_admission(
    *,
    cache_root: Path,
    families: _RemoteBodyFamilySequence,
    world_size: PositiveIntLike,
    cache_shards: PositiveIntLike,
    lookahead_shards_per_rank: PositiveIntLike,
    safety_reserve_bytes: NonNegativeIntLike = REMOTE_PREFETCH_DISK_SAFETY_RESERVE_BYTES,
) -> RemotePrefetchDiskAdmission:
    """Calculate the conservative event-by-event managed-cache peak."""
    schedules = _schedule_map(families)
    # A paused rank can hold the global minimum while every signed body becomes
    # durable. Retention bounds steady-state storage, not this startup peak.
    selection_final_bytes = 0
    for family in families:
        if family.kind != "selection":
            continue
        selection_final_bytes += sum(entry.byte_length for entry in family.entries)
    packed_members = [family for family in families if family.kind == "packed"]
    packed_final_bytes = 0
    if packed_members:
        packed = packed_members[0]
        # Finalized bodies are evicted only after every rank has crossed the
        # configured trailing window. A paused rank can therefore make every
        # signed body durable before the global minimum advances. Count the
        # exact full inventory rather than relying on a shard-count heuristic.
        packed_final_bytes = sum(entry.byte_length for entry in packed.entries)

    active_temporary_bytes = 0
    for owner_rank in range(world_size):
        # Each worker performs at most one transfer at a time, but an explicit
        # validation demand can be larger than every train component. Count
        # the largest body in this owner's complete signed plan.
        active_temporary_bytes += max(
            (
                entry.byte_length
                for family in families
                for index, entry in enumerate(family.entries)
                if owner_for_inventory_index(index, world_size) == owner_rank
            ),
            default=0,
        )

    body_count = sum(len(family.entries) for family in families)
    schedule_entry_count = sum(len(members[0].entries) for members in schedules.values())
    receipt_and_control_bytes = (
        body_count * REMOTE_PREFETCH_RECEIPT_BOUND_BYTES
        + (body_count + world_size * schedule_entry_count + world_size * 4 + len(families) * 4)
        * REMOTE_PREFETCH_CONTROL_BOUND_BYTES
    )
    peak_bytes = (
        selection_final_bytes
        + packed_final_bytes
        + active_temporary_bytes
        + receipt_and_control_bytes
    )

    current_managed_bytes = 0
    for family in families:
        local_dir = Path(family.local_dir)
        for entry in family.entries:
            final_path = local_dir / entry.name
            if final_path.exists() and not final_path.is_symlink():
                current_managed_bytes += _direct_file_stat(final_path).st_size
    additional_required_bytes = max(0, peak_bytes - current_managed_bytes)
    available_bytes = shutil.disk_usage(cache_root).free
    return RemotePrefetchDiskAdmission(
        peak_bytes=peak_bytes,
        current_managed_bytes=current_managed_bytes,
        additional_required_bytes=additional_required_bytes,
        available_bytes=available_bytes,
        safety_reserve_bytes=safety_reserve_bytes,
        selection_final_bytes=selection_final_bytes,
        packed_final_bytes=packed_final_bytes,
        active_temporary_bytes=active_temporary_bytes,
        receipt_and_control_bytes=receipt_and_control_bytes,
    )


def _prefetch_root(cache_root: Path) -> Path:
    return cache_root / ".remote-prefetch"


def _control_root(cache_root: Path, nonce: str) -> Path:
    return _prefetch_root(cache_root) / "control" / nonce


def _status_path(control_root: Path, rank: int) -> Path:
    return control_root / "status" / f"rank_{rank:05d}.json"


def _cursor_path(control_root: Path, rank: int) -> Path:
    return control_root / "cursors" / f"rank_{rank:05d}.json"


def _progress_path(control_root: Path, rank: int) -> Path:
    return control_root / "progress" / f"rank_{rank:05d}.json"


def _demand_dir(control_root: Path, rank: int) -> Path:
    return control_root / "demands" / f"rank_{rank:05d}"


def _demand_path(
    control_root: Path,
    *,
    requester_rank: int,
    schedule_id: str,
    inventory_index: int,
) -> Path:
    return _demand_dir(control_root, requester_rank) / (f"{schedule_id}.{inventory_index:08d}.json")


def _receipt_path(
    control_root: Path,
    *,
    family_id: str,
    inventory_index: int,
) -> Path:
    return control_root / "receipts" / family_id / f"{inventory_index:08d}.json"


def _ready_path(
    control_root: Path,
    *,
    schedule_id: str,
    inventory_index: int,
) -> Path:
    return control_root / "ready" / schedule_id / f"{inventory_index:08d}.json"


def _transport_path(cache_root: Path, family: RemoteBodyFamily, index: int) -> Path:
    entry = family.entry(index)
    return (_prefetch_root(cache_root) / family.family_id) / (
        f".{entry.name}.owner-transfer.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    )


def _embedded_transport_pid(name: str, *, signed_names: Sequence[str]) -> int | None:
    for signed_name in signed_names:
        private_prefix = f".{signed_name}.owner-transfer."
        nested_prefix = f"..{signed_name}.owner-transfer."
        if name.startswith(private_prefix):
            suffix = name.removeprefix(private_prefix)
            parts = suffix.split(".")
            if len(parts) == 2 and re.fullmatch(r"[0-9a-f]{8}", parts[1]):
                try:
                    return int(parts[0])
                except ValueError:
                    return None
        if name.startswith(nested_prefix) and ".remote-tmp." in name:
            parts = name.rsplit(".", 2)
            if len(parts) == 3 and re.fullmatch(r"[0-9a-f]{8}", parts[2]):
                try:
                    return int(parts[1])
                except ValueError:
                    return None
    return None


def _validate_and_sweep_transport_namespace(
    *,
    cache_root: Path,
    family: RemoteBodyFamily,
) -> None:
    transport_dir = _prefetch_root(cache_root) / family.family_id
    _ensure_direct_directory(transport_dir)
    expected = {entry.name for entry in family.entries}
    for path in transport_dir.iterdir():
        if path.name in expected:
            _direct_file_stat(path)
            path.unlink()
            logger.info(f"remote_training.prefetch_obsolete_stage_sweep path={path}")
            continue
        pid = _embedded_transport_pid(path.name, signed_names=tuple(expected))
        if pid is None:
            raise RemoteTrainingError(f"unexpected remote prefetch transport entry: {path}")
        _direct_file_stat(path)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            path.unlink()
            logger.info(f"remote_training.prefetch_dead_transport_sweep path={path}")
        except PermissionError as exc:
            raise RemoteTrainingError(
                "cannot prove that a private remote transport file belongs to a "
                f"dead process: {path}"
            ) from exc
        else:
            raise RemoteTrainingError(
                "private remote transport file belongs to a live PID; refusing "
                f"unsafe startup reuse: {path}"
            )
    _fsync_directory(transport_dir)


def _validate_rank(value: Any, *, world_size: int, surface: str) -> int:
    if not isinstance(value, NonNegativeInt) or value >= world_size:
        raise RemoteTrainingError(
            f"{surface} must be a strict int in [0, {world_size}), got {value!r}"
        )
    return NonNegativeInt.parse(value)


def _validate_epoch_index_generation(
    record: Mapping[str, Any],
    *,
    inventory_length: int,
    surface: str,
) -> tuple[int, int, int]:
    epoch = record["epoch"]
    index = record["inventory_index"]
    generation = record["generation"]
    if not isinstance(epoch, PositiveInt):
        raise RemoteTrainingError(f"{surface}.epoch must be a strict int >= 1")
    if not isinstance(index, NonNegativeInt) or index >= inventory_length:
        raise RemoteTrainingError(f"{surface}.inventory_index must be in [0, {inventory_length})")
    expected_generation = (epoch - 1) * inventory_length + index
    if not isinstance(generation, NonNegativeInt) or generation != expected_generation:
        raise RemoteTrainingError(
            f"{surface}.generation must equal (epoch - 1) * inventory_length + index"
        )
    return epoch, index, generation


def _validate_control_tree(
    *,
    cache_root: Path,
    nonce: str,
    world_size: int,
    families: Sequence[RemoteBodyFamily],
) -> Path:
    root = _prefetch_root(cache_root)
    _ensure_direct_directory(root)
    control_parent = root / "control"
    _ensure_direct_directory(control_parent)
    control = control_parent / nonce
    _ensure_direct_directory(control)
    if {path.name for path in control.iterdir()} != set(_CONTROL_SUBDIRS):
        raise RemoteTrainingError(
            f"remote prefetch control directory has an invalid schema: {control}"
        )
    for name in _CONTROL_SUBDIRS:
        _ensure_direct_directory(control / name)
    expected_rank_dirs = {f"rank_{rank:05d}" for rank in range(world_size)}
    demand_root = control / "demands"
    if {path.name for path in demand_root.iterdir()} != expected_rank_dirs:
        raise RemoteTrainingError(
            f"remote prefetch demand directory has an invalid rank schema: {demand_root}"
        )
    for rank_dir in demand_root.iterdir():
        _ensure_direct_directory(rank_dir)
    expected_family_ids = {family.family_id for family in families}
    receipts_root = control / "receipts"
    if {path.name for path in receipts_root.iterdir()} != expected_family_ids:
        raise RemoteTrainingError(
            f"remote prefetch receipt directory has an invalid family schema: {receipts_root}"
        )
    for family_dir in receipts_root.iterdir():
        _ensure_direct_directory(family_dir)
    expected_schedule_ids = {family.schedule_id for family in families}
    ready_root = control / "ready"
    if {path.name for path in ready_root.iterdir()} != expected_schedule_ids:
        raise RemoteTrainingError(
            f"remote prefetch ready directory has an invalid schedule schema: {ready_root}"
        )
    for schedule_dir in ready_root.iterdir():
        _ensure_direct_directory(schedule_dir)
    return control


def _create_control_tree(
    *,
    cache_root: Path,
    nonce: str,
    world_size: int,
    families: Sequence[RemoteBodyFamily],
) -> Path:
    root = _prefetch_root(cache_root)
    _ensure_direct_directory(root, create=True)
    control_parent = root / "control"
    _ensure_direct_directory(control_parent, create=True)
    for stale in list(control_parent.iterdir()):
        if stale.name == nonce:
            continue
        _safe_remove_tree(stale)
    control = control_parent / nonce
    if control.exists() or control.is_symlink():
        raise RemoteTrainingError(f"remote prefetch nonce already exists: {control}")
    control.mkdir(mode=0o700)
    for name in _CONTROL_SUBDIRS:
        (control / name).mkdir(mode=0o700)
    for rank in range(world_size):
        _demand_dir(control, rank).mkdir(mode=0o700)
    for family in families:
        (control / "receipts" / family.family_id).mkdir(mode=0o700)
        stage_dir = root / family.family_id
        _ensure_direct_directory(stage_dir, create=True)
        _validate_and_sweep_transport_namespace(
            cache_root=cache_root,
            family=family,
        )
    for schedule_id in sorted({family.schedule_id for family in families}):
        (control / "ready" / schedule_id).mkdir(mode=0o700)
    _fsync_directory(control)
    _fsync_directory(control_parent)
    return _validate_control_tree(
        cache_root=cache_root,
        nonce=nonce,
        world_size=world_size,
        families=families,
    )


def _write_worker_status(
    cfg: _WorkerConfig,
    *,
    state: str,
    family_id: str | None = None,
    inventory_index: int | None = None,
    body_name: str | None = None,
    error: BaseException | None = None,
) -> None:
    control = _control_root(Path(cfg.cache_root), cfg.nonce)
    error_message = _bounded_error_text(error) if error is not None else None
    record = {
        "schema_version": REMOTE_PREFETCH_SCHEMA_VERSION,
        "nonce": cfg.nonce,
        "rank": cfg.rank,
        "pid": os.getpid(),
        "state": state,
        "family_id": family_id,
        "inventory_index": inventory_index,
        "body_name": body_name,
        "updated_ns": time.time_ns(),
        "error_type": type(error).__name__ if error is not None else None,
        "error_message": error_message,
        "traceback": (traceback.format_exc(limit=30)[-12000:] if error is not None else None),
    }
    _atomic_record(record, _status_path(control, cfg.rank))


def _load_status(
    path: Path,
    *,
    nonce: str,
    world_size: int,
) -> dict[str, Any]:
    raw = _load_json_record(path, expected_keys=_STATUS_KEYS, surface="worker status")
    if raw["nonce"] != nonce:
        raise RemoteTrainingError(f"worker status has a stale nonce: {path}")
    _validate_rank(raw["rank"], world_size=world_size, surface="worker status rank")
    if type(raw["pid"]) is not int or raw["pid"] <= 0:
        raise RemoteTrainingError("worker status pid must be a positive strict int")
    if raw["state"] not in {
        "starting",
        "idle",
        "staging",
        "publishing",
        "error",
        "stopped",
    }:
        raise RemoteTrainingError(f"worker status has invalid state {raw['state']!r}")
    if type(raw["updated_ns"]) is not int or raw["updated_ns"] <= 0:
        raise RemoteTrainingError("worker status updated_ns must be a positive strict int")
    identity = (
        raw["family_id"],
        raw["inventory_index"],
        raw["body_name"],
    )
    if any(value is not None for value in identity):
        if any(value is None for value in identity):
            raise RemoteTrainingError("worker status active body identity must be all-or-none")
        _require_sha256(raw["family_id"], surface="worker status family_id")
        if type(raw["inventory_index"]) is not int or raw["inventory_index"] < 0:
            raise RemoteTrainingError(
                "worker status inventory_index must be a non-negative strict int"
            )
        body_name = raw["body_name"]
        if not isinstance(body_name, str) or not body_name or Path(body_name).name != body_name:
            raise RemoteTrainingError("worker status body_name must be a direct basename")
    for key in ("error_type", "error_message", "traceback"):
        if raw[key] is not None and not isinstance(raw[key], str):
            raise RemoteTrainingError(f"worker status {key} must be a string or null")
    if raw["state"] == "error" and (
        not isinstance(raw["error_type"], str)
        or not raw["error_type"]
        or not isinstance(raw["error_message"], str)
    ):
        raise RemoteTrainingError("terminal worker error status lacks typed diagnostics")
    return raw


def _load_cursor(
    path: Path,
    *,
    cfg: _WorkerConfig,
    train_schedule_id: str,
    inventory_length: int,
) -> dict[str, Any]:
    raw = _load_json_record(path, expected_keys=_CURSOR_KEYS, surface="rank cursor")
    if raw["nonce"] != cfg.nonce or raw["schedule_id"] != train_schedule_id:
        raise RemoteTrainingError(f"rank cursor has stale run identity: {path}")
    _validate_rank(raw["rank"], world_size=cfg.world_size, surface="rank cursor rank")
    _validate_epoch_index_generation(
        raw,
        inventory_length=inventory_length,
        surface="rank cursor",
    )
    return raw


def _load_progress(
    path: Path,
    *,
    cfg: _WorkerConfig,
    train_schedule_id: str,
) -> dict[str, Any]:
    raw = _load_json_record(path, expected_keys=_PROGRESS_KEYS, surface="rank progress")
    if raw["nonce"] != cfg.nonce or raw["schedule_id"] != train_schedule_id:
        raise RemoteTrainingError(f"rank progress has stale run identity: {path}")
    _validate_rank(raw["rank"], world_size=cfg.world_size, surface="rank progress rank")
    if type(raw["generation"]) is not int or raw["generation"] < -1:
        raise RemoteTrainingError("rank progress generation must be a strict int >= -1")
    return raw


def _load_demand(
    path: Path,
    *,
    cfg: _WorkerConfig,
    schedules: Mapping[str, tuple[RemoteBodyFamily, ...]],
    requester_rank: int,
) -> dict[str, Any]:
    raw = _load_json_record(path, expected_keys=_DEMAND_KEYS, surface="body demand")
    if raw["nonce"] != cfg.nonce:
        raise RemoteTrainingError(f"body demand has stale nonce: {path}")
    if raw["requester_rank"] != requester_rank:
        raise RemoteTrainingError(
            f"body demand is outside its requester-specific namespace: {path}"
        )
    _validate_rank(
        raw["requester_rank"],
        world_size=cfg.world_size,
        surface="body demand requester_rank",
    )
    schedule_id = raw["schedule_id"]
    if schedule_id not in schedules:
        raise RemoteTrainingError(f"body demand names an unknown schedule: {path}")
    _validate_epoch_index_generation(
        raw,
        inventory_length=len(schedules[schedule_id][0].entries),
        surface="body demand",
    )
    if type(raw["created_ns"]) is not int or raw["created_ns"] <= 0:
        raise RemoteTrainingError("body demand created_ns must be a positive strict int")
    expected_name = f"{schedule_id}.{raw['inventory_index']:08d}.json"
    if path.name != expected_name:
        raise RemoteTrainingError(f"body demand filename does not match its record: {path}")
    return raw


def _load_demand_if_present(
    path: Path,
    *,
    cfg: _WorkerConfig,
    schedules: Mapping[str, tuple[RemoteBodyFamily, ...]],
    requester_rank: int,
) -> dict[str, Any] | None:
    """Load one demand unless its requester concurrently satisfied and removed it."""
    try:
        return _load_demand(
            path,
            cfg=cfg,
            schedules=schedules,
            requester_rank=requester_rank,
        )
    except RemoteTrainingError as exc:
        if isinstance(exc.__cause__, FileNotFoundError):
            return None
        raise


def _load_receipt(
    path: Path,
    *,
    cfg: _WorkerConfig,
    family: RemoteBodyFamily,
    inventory_index: int,
) -> dict[str, Any]:
    raw = _load_json_record(
        path,
        expected_keys=_RECEIPT_KEYS,
        surface="publication receipt",
    )
    entry = family.entry(inventory_index)
    expected = {
        "nonce": cfg.nonce,
        "family_id": family.family_id,
        "family_aggregate": family.family_aggregate,
        "inventory_index": inventory_index,
        "basename": entry.name,
        "byte_length": entry.byte_length,
        "content_sha256": entry.content_sha256,
        "file_fingerprint": entry.file_fingerprint,
        "owner_rank": owner_for_inventory_index(inventory_index, cfg.world_size),
    }
    strict_int_keys = {"inventory_index", "byte_length", "owner_rank"}
    for key, value in expected.items():
        if (key in strict_int_keys and type(raw[key]) is not int) or raw[key] != value:
            raise RemoteTrainingError(
                f"publication receipt {path} has mismatched {key}: {raw[key]!r} != {value!r}"
            )
    if (
        type(raw["generation"]) is not int
        or raw["generation"] < inventory_index
        or raw["generation"] % len(family.entries) != inventory_index
    ):
        raise RemoteTrainingError("publication receipt generation is malformed")
    return raw


def _load_ready(
    path: Path,
    *,
    cfg: _WorkerConfig,
    schedule_id: str,
    inventory_index: int,
    components: Sequence[RemoteBodyFamily],
) -> dict[str, Any]:
    raw = _load_json_record(path, expected_keys=_READY_KEYS, surface="pair-ready record")
    generation = raw["generation"]
    inventory_length = len(components[0].entries)
    if (
        raw["nonce"] != cfg.nonce
        or raw["schedule_id"] != schedule_id
        or type(raw["inventory_index"]) is not int
        or raw["inventory_index"] != inventory_index
        or type(raw["owner_rank"]) is not int
        or raw["owner_rank"]
        != owner_for_inventory_index(
            inventory_index,
            cfg.world_size,
        )
        or type(generation) is not int
        or generation < inventory_index
        or generation % inventory_length != inventory_index
        or raw["components"] != [family.family_id for family in components]
    ):
        raise RemoteTrainingError(f"pair-ready record is malformed: {path}")
    return raw


def _same_preexisting_descriptor(
    family: RemoteBodyFamily,
    *,
    inventory_index: int,
) -> bool:
    entry = family.entry(inventory_index)
    expected = family.preexisting_descriptor(entry.name)
    if expected is None:
        return False
    final_path = Path(family.local_dir) / entry.name
    try:
        actual = _descriptor_dict(_direct_file_stat(final_path))
    except RemoteTrainingError:
        return False
    return actual == expected


def _existing_receipt(
    *,
    cfg: _WorkerConfig,
    family: RemoteBodyFamily,
    inventory_index: int,
) -> dict[str, Any] | None:
    path = _receipt_path(
        _control_root(Path(cfg.cache_root), cfg.nonce),
        family_id=family.family_id,
        inventory_index=inventory_index,
    )
    if not path.exists():
        return None
    receipt = _load_receipt(
        path,
        cfg=cfg,
        family=family,
        inventory_index=inventory_index,
    )
    final_path = Path(family.local_dir) / family.entry(inventory_index).name
    try:
        _direct_file_stat(final_path)
    except RemoteTrainingError:
        return None
    return receipt


def _poll_worker_command(command_pipe: Connection) -> None:
    while command_pipe.poll():
        command = command_pipe.recv()
        if command == {"op": "cancel"}:
            raise _WorkerCancelled
        raise RemoteTrainingError(f"remote prefetch worker received invalid command {command!r}")


def _rollback_final_under_lock(
    *,
    cfg: _WorkerConfig,
    family: RemoteBodyFamily,
    inventory_index: int,
) -> None:
    entry = family.entry(inventory_index)
    final_path = Path(family.local_dir) / entry.name
    receipt_path = _receipt_path(
        _control_root(Path(cfg.cache_root), cfg.nonce),
        family_id=family.family_id,
        inventory_index=inventory_index,
    )
    failures: list[BaseException] = []
    try:
        receipt_path.unlink(missing_ok=True)
    except BaseException as exc:
        failures.append(exc)
    try:
        _fsync_directory(receipt_path.parent)
    except BaseException as exc:
        failures.append(exc)
    if final_path.exists() or final_path.is_symlink():
        try:
            final_path.unlink()
        except BaseException as exc:
            failures.append(exc)
        else:
            try:
                _fsync_directory(final_path.parent)
            except BaseException as exc:
                failures.append(exc)
    if failures:
        primary = failures[0]
        for failure in failures[1:]:
            primary.add_note(
                f"remote body rollback also failed: {type(failure).__name__}: {failure}"
            )
        raise primary


def _rollback_new_publication(
    *,
    cfg: _WorkerConfig,
    publication: _NewPublication,
) -> None:
    family = next(
        (candidate for candidate in cfg.families if candidate.family_id == publication.family_id),
        None,
    )
    if family is None:
        raise RemoteTrainingError(f"cannot roll back unknown remote family {publication.family_id}")
    entry = family.entry(publication.inventory_index)
    lock_fd = _open_cache_lock(Path(family.local_dir) / f".{entry.name}.lock")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        _rollback_final_under_lock(
            cfg=cfg,
            family=family,
            inventory_index=publication.inventory_index,
        )
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def _publish_component(
    *,
    cfg: _WorkerConfig,
    backend: HfBucketBackend,
    family: RemoteBodyFamily,
    inventory_index: int,
    generation: int,
    command_pipe: Connection,
) -> _NewPublication | None:
    if owner_for_inventory_index(inventory_index, cfg.world_size) != cfg.rank:
        raise RemoteTrainingError(
            "remote prefetch worker attempted transport for a non-owned inventory index"
        )
    _poll_worker_command(command_pipe)
    entry = family.entry(inventory_index)
    _write_worker_status(
        cfg,
        state="staging",
        family_id=family.family_id,
        inventory_index=inventory_index,
        body_name=entry.name,
    )
    local_dir = Path(family.local_dir)
    final_path = local_dir / entry.name
    published = False
    lock_fd = _open_cache_lock(local_dir / f".{entry.name}.lock")
    try:
        while True:
            _poll_worker_command(command_pipe)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if command_pipe.poll(REMOTE_PREFETCH_POLL_SECONDS):
                    continue

        final_present = final_path.exists() or final_path.is_symlink()
        if final_present:
            unchanged_preexisting = _same_preexisting_descriptor(
                family,
                inventory_index=inventory_index,
            )
            existing_receipt = (
                None
                if unchanged_preexisting
                else _existing_receipt(
                    cfg=cfg,
                    family=family,
                    inventory_index=inventory_index,
                )
            )
            if not unchanged_preexisting and existing_receipt is None:
                raise RemoteTrainingError(
                    "remote prefetch found a fresh final body without its exact "
                    f"nonce-scoped receipt; refusing adoption: {final_path}"
                )
            if existing_receipt is not None and generation > cast(
                int, existing_receipt["generation"]
            ):
                refreshed_receipt = dict(existing_receipt)
                refreshed_receipt["generation"] = generation
                receipt_path = _receipt_path(
                    _control_root(Path(cfg.cache_root), cfg.nonce),
                    family_id=family.family_id,
                    inventory_index=inventory_index,
                )
                try:
                    _atomic_record(refreshed_receipt, receipt_path)
                    _poll_worker_command(command_pipe)
                except BaseException as primary:
                    try:
                        _atomic_record(existing_receipt, receipt_path)
                    except BaseException as rollback_error:
                        primary.add_note(
                            "also failed to restore the prior publication receipt: "
                            f"{type(rollback_error).__name__}: {rollback_error}"
                        )
                    raise
            return None

        transport_path = _transport_path(
            Path(cfg.cache_root),
            family,
            inventory_index,
        )
        try:
            _download_with_retry(
                backend,
                remote_path=f"{family.remote_dir}/{entry.name}",
                final_path=transport_path,
                timeout_seconds=cfg.timeout_seconds,
                retry_interval_seconds=cfg.retry_interval_seconds,
            )
            _poll_worker_command(command_pipe)
            _write_worker_status(
                cfg,
                state="publishing",
                family_id=family.family_id,
                inventory_index=inventory_index,
                body_name=entry.name,
            )
            os.replace(transport_path, final_path)
            published = True
            _fsync_directory(local_dir)
        finally:
            transport_path.unlink(missing_ok=True)

        receipt = {
            "schema_version": REMOTE_PREFETCH_RECEIPT_SCHEMA_VERSION,
            "nonce": cfg.nonce,
            "family_id": family.family_id,
            "family_aggregate": family.family_aggregate,
            "inventory_index": inventory_index,
            "basename": entry.name,
            "byte_length": entry.byte_length,
            "content_sha256": entry.content_sha256,
            "file_fingerprint": entry.file_fingerprint,
            "owner_rank": cfg.rank,
            "generation": generation,
        }
        _poll_worker_command(command_pipe)
        _atomic_record(
            receipt,
            _receipt_path(
                _control_root(Path(cfg.cache_root), cfg.nonce),
                family_id=family.family_id,
                inventory_index=inventory_index,
            ),
        )
        _poll_worker_command(command_pipe)
        return _NewPublication(
            family_id=family.family_id,
            inventory_index=inventory_index,
        )
    except BaseException as primary:
        if published:
            try:
                _rollback_final_under_lock(
                    cfg=cfg,
                    family=family,
                    inventory_index=inventory_index,
                )
            except BaseException as rollback_error:
                primary.add_note(
                    "also failed to roll back component publication: "
                    f"{type(rollback_error).__name__}: {rollback_error}"
                )
        raise
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def _scan_demands(
    *,
    cfg: _WorkerConfig,
    schedules: Mapping[str, tuple[RemoteBodyFamily, ...]],
) -> dict[tuple[str, int], dict[str, Any]]:
    control = _control_root(Path(cfg.cache_root), cfg.nonce)
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for requester_rank in range(cfg.world_size):
        directory = _demand_dir(control, requester_rank)
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if not path.name.endswith(".json"):
                match = re.fullmatch(
                    r"[0-9a-f]{64}\.[0-9]{8}\.json\.tmp\.([1-9][0-9]*)\.[0-9a-f]{8}",
                    path.name,
                )
                if match is not None:
                    if not _volatile_direct_file_exists(path):
                        continue
                    writer_pid = int(match.group(1))
                    try:
                        os.kill(writer_pid, 0)
                    except ProcessLookupError:
                        path.unlink(missing_ok=True)
                        _fsync_directory(directory)
                    except PermissionError as exc:
                        raise RemoteTrainingError(
                            "cannot prove that a requester-specific demand "
                            f"temporary belongs to a live process: {path}"
                        ) from exc
                    else:
                        # atomic_json_dump writes beside the final before one
                        # os.replace. A live writer's exact temporary is not a
                        # control record and must not be parsed mid-publication.
                        pass
                    continue
                raise RemoteTrainingError(f"unexpected remote prefetch demand entry: {path}")
            try:
                demand = _load_demand_if_present(
                    path,
                    cfg=cfg,
                    schedules=schedules,
                    requester_rank=requester_rank,
                )
            except RemoteTrainingError as exc:
                # A demand is authoritative only once one coherent generation
                # is readable. Defer an NFS descriptor transition to the next
                # scan; the canonical record remains in place for that scan.
                if _is_transient_mutable_record_read(exc):
                    continue
                raise
            if demand is None:
                continue
            key = (cast(str, demand["schedule_id"]), cast(int, demand["inventory_index"]))
            prior = latest.get(key)
            if prior is None or demand["generation"] > prior["generation"]:
                latest[key] = demand
    return latest


def _has_requester_demand(
    *,
    cfg: _WorkerConfig,
    schedules: Mapping[str, tuple[RemoteBodyFamily, ...]],
    schedule_id: str,
    inventory_index: int,
) -> bool:
    """Check one body identity after acquiring its exclusive eviction lock."""
    control = _control_root(Path(cfg.cache_root), cfg.nonce)
    for requester_rank in range(cfg.world_size):
        path = _demand_path(
            control,
            requester_rank=requester_rank,
            schedule_id=schedule_id,
            inventory_index=inventory_index,
        )
        try:
            demand = _load_demand_if_present(
                path,
                cfg=cfg,
                schedules=schedules,
                requester_rank=requester_rank,
            )
        except RemoteTrainingError as exc:
            # This is the under-exclusive-lock eviction recheck. If a demand
            # snapshot is indeterminate, preserve the body conservatively and
            # retry eviction on a later worker loop.
            if _is_transient_mutable_record_read(exc):
                return True
            raise
        if demand is not None:
            return True
    return False


def _scan_cursors(
    *,
    cfg: _WorkerConfig,
    train_schedule_id: str,
    inventory_length: int,
) -> list[dict[str, Any]]:
    control = _control_root(Path(cfg.cache_root), cfg.nonce)
    cursors: list[dict[str, Any]] = []
    for rank in range(cfg.world_size):
        path = _cursor_path(control, rank)
        if not path.exists():
            continue
        try:
            cursor = _load_cursor(
                path,
                cfg=cfg,
                train_schedule_id=train_schedule_id,
                inventory_length=inventory_length,
            )
        except RemoteTrainingError as exc:
            # Cursors are advisory lookahead snapshots. Missing one atomic
            # generation merely shortens this worker-loop frontier; demand
            # publication remains authoritative and the next loop resamples.
            if _is_transient_mutable_record_read(exc):
                continue
            raise
        if cursor["rank"] != rank:
            raise RemoteTrainingError(f"rank cursor is in the wrong namespace: {path}")
        cursors.append(cursor)
    return cursors


def _lookahead_obligations(
    *,
    cfg: _WorkerConfig,
    cursors: Sequence[Mapping[str, Any]],
    train_schedule_id: str,
    inventory_length: int,
) -> list[tuple[str, int, int]]:
    """Return this owner's nearest K publishable train obligations."""
    distances: dict[int, tuple[int, int]] = {}
    for cursor in cursors:
        cursor_generation = cast(int, cursor["generation"])
        found = 0
        distance = 0
        # In any non-empty inventory, at most world_size positions separate two
        # entries owned by this rank.  The extra full inventory covers sparse
        # short-inventory ownership without relying on that approximation.
        limit = inventory_length + cfg.lookahead_shards_per_rank * cfg.world_size
        while found < cfg.lookahead_shards_per_rank and distance < limit:
            generation = cursor_generation + distance
            index = generation % inventory_length
            if owner_for_inventory_index(index, cfg.world_size) == cfg.rank:
                prior = distances.get(index)
                candidate = (distance, generation)
                if prior is None or candidate < prior:
                    distances[index] = candidate
                found += 1
            distance += 1
    ordered = sorted(
        ((distance, index, generation) for index, (distance, generation) in distances.items()),
        key=lambda item: (item[0], item[1]),
    )
    return [
        (train_schedule_id, index, generation)
        for _distance, index, generation in ordered[: cfg.lookahead_shards_per_rank]
    ]


def _publish_schedule(
    *,
    cfg: _WorkerConfig,
    backend: HfBucketBackend,
    schedules: Mapping[str, tuple[RemoteBodyFamily, ...]],
    schedule_id: str,
    inventory_index: int,
    generation: int,
    command_pipe: Connection,
) -> None:
    components = schedules[schedule_id]
    ready_path = _ready_path(
        _control_root(Path(cfg.cache_root), cfg.nonce),
        schedule_id=schedule_id,
        inventory_index=inventory_index,
    )
    if ready_path.exists():
        ready = _load_ready(
            ready_path,
            cfg=cfg,
            schedule_id=schedule_id,
            inventory_index=inventory_index,
            components=components,
        )
        if cast(int, ready["generation"]) >= generation:
            complete = True
            for family in components:
                if _same_preexisting_descriptor(
                    family,
                    inventory_index=inventory_index,
                ):
                    continue
                receipt = _existing_receipt(
                    cfg=cfg,
                    family=family,
                    inventory_index=inventory_index,
                )
                if receipt is None or cast(int, receipt["generation"]) < generation:
                    complete = False
                    break
            if complete:
                return

    publications: list[_NewPublication] = []
    try:
        for family in components:
            publication = _publish_component(
                cfg=cfg,
                backend=backend,
                family=family,
                inventory_index=inventory_index,
                generation=generation,
                command_pipe=command_pipe,
            )
            if publication is not None:
                publications.append(publication)
        _poll_worker_command(command_pipe)
        ready = {
            "schema_version": REMOTE_PREFETCH_SCHEMA_VERSION,
            "nonce": cfg.nonce,
            "schedule_id": schedule_id,
            "inventory_index": inventory_index,
            "generation": generation,
            "owner_rank": cfg.rank,
            "components": [family.family_id for family in components],
        }
        _atomic_record(ready, ready_path)
        _poll_worker_command(command_pipe)
    except BaseException as primary:
        cleanup_errors: list[BaseException] = []
        try:
            ready_path.unlink(missing_ok=True)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        try:
            _fsync_directory(ready_path.parent)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        for publication in reversed(publications):
            try:
                _rollback_new_publication(
                    cfg=cfg,
                    publication=publication,
                )
            except BaseException as rollback_error:
                cleanup_errors.append(rollback_error)
        for cleanup_error in cleanup_errors:
            primary.add_note(
                "also failed to roll back owner publication: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise


def _evict_finalized_train(
    *,
    cfg: _WorkerConfig,
    demands: Mapping[tuple[str, int], Mapping[str, Any]],
) -> None:
    train_families = [family for family in cfg.families if family.train_schedule]
    if not train_families:
        return
    train_schedule_id = train_families[0].schedule_id
    schedules = _schedule_map(cfg.families)
    control = _control_root(Path(cfg.cache_root), cfg.nonce)
    progress: list[int] = []
    for rank in range(cfg.world_size):
        path = _progress_path(control, rank)
        if not path.exists():
            return
        try:
            record = _load_progress(
                path,
                cfg=cfg,
                train_schedule_id=train_schedule_id,
            )
        except RemoteTrainingError as exc:
            # Progress is a replace-in-place control snapshot used only to
            # determine which old bodies may be evicted.  On shared NFS, the
            # opened generation can undergo a descriptor transition while its
            # rank atomically publishes the next generation.  A temporarily
            # invisible pathname is the same incomplete-snapshot case.  Keep
            # the cache intact for this pass and sample all ranks again on the
            # next worker loop; every schema, identity, and permission error
            # remains terminal.
            if _is_transient_mutable_record_read(exc):
                return
            raise
        if record["rank"] != rank:
            raise RemoteTrainingError(f"rank progress is in the wrong namespace: {path}")
        progress.append(cast(int, record["generation"]))
    min_progress = min(progress)
    if min_progress < 0:
        return
    members = schedules[train_schedule_id]
    if not members:
        return
    evict_before = max(0, min_progress - cfg.cache_shards + 1)
    demanded_indices = {
        index
        for (schedule_id, index), _demand in demands.items()
        if schedule_id == train_schedule_id
    }
    for index in range(len(members[0].entries)):
        if owner_for_inventory_index(index, cfg.world_size) != cfg.rank or index in demanded_indices:
            continue
        lock_fds: list[int] = []
        try:
            lock_paths = sorted(
                (
                    Path(family.local_dir) / f".{family.entries[index].name}.lock",
                    family,
                )
                for family in members
            )
            locked = True
            for lock_path, _family in lock_paths:
                lock_fd = _open_cache_lock(lock_path)
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(lock_fd)
                    locked = False
                    break
                lock_fds.append(lock_fd)
            if not locked:
                continue
            if _has_requester_demand(
                cfg=cfg,
                schedules=schedules,
                schedule_id=train_schedule_id,
                inventory_index=index,
            ):
                continue
            existing: list[tuple[RemoteBodyFamily, Path, Path, int]] = []
            for family in members:
                entry = family.entries[index]
                body_path = Path(family.local_dir) / entry.name
                if not (body_path.exists() or body_path.is_symlink()):
                    continue
                receipt_path = _receipt_path(
                    control,
                    family_id=family.family_id,
                    inventory_index=index,
                )
                if receipt_path.exists():
                    receipt = _load_receipt(
                        receipt_path,
                        cfg=cfg,
                        family=family,
                        inventory_index=index,
                    )
                    body_generation = cast(int, receipt["generation"])
                elif _same_preexisting_descriptor(family, inventory_index=index):
                    body_generation = index
                else:
                    raise RemoteTrainingError(
                        f"owner eviction found a fresh {family.kind} body without its exact "
                        f"receipt: {body_path}"
                    )
                existing.append((family, body_path, receipt_path, body_generation))
            if not existing:
                continue
            generations = {generation for _family, _body, _receipt, generation in existing}
            if len(generations) != 1:
                raise RemoteTrainingError(
                    f"train body generation mismatch before owner eviction at index {index}: "
                    f"{sorted(generations)}"
                )
            body_generation = generations.pop()
            if body_generation >= evict_before:
                continue
            for _family, body_path, receipt_path, _generation in existing:
                body_path.unlink(missing_ok=True)
                receipt_path.unlink(missing_ok=True)
            ready_path = _ready_path(
                control,
                schedule_id=train_schedule_id,
                inventory_index=index,
            )
            ready_path.unlink(missing_ok=True)
            for family, _body_path, _receipt_path_value, _generation in existing:
                _fsync_directory(Path(family.local_dir))
            for _family, _body_path, receipt_path, _generation in existing:
                _fsync_directory(receipt_path.parent)
            _fsync_directory(ready_path.parent)
            logger.info(
                "remote_training.prefetch_owner_evict "
                f"owner={cfg.rank} index={index} generation={body_generation} "
                f"paths={[str(body_path) for _family, body_path, _receipt, _gen in existing]}"
            )
        finally:
            for lock_fd in reversed(lock_fds):
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)


def _worker_main(
    cfg: _WorkerConfig,
    command_pipe: Connection,
    status_pipe: Connection,
    backend_factory: Any = None,
) -> None:
    try:
        schedules = _schedule_map(cfg.families)
        train_schedule_id = next(
            family.schedule_id for family in cfg.families if family.train_schedule
        )
        train_inventory_length = len(schedules[train_schedule_id][0].entries)
        _validate_control_tree(
            cache_root=Path(cfg.cache_root),
            nonce=cfg.nonce,
            world_size=cfg.world_size,
            families=cfg.families,
        )
        backend = (
            HfBucketBackend(cfg.bucket_id)
            if backend_factory is None
            else backend_factory(cfg.bucket_id)
        )
        _write_worker_status(cfg, state="starting")
        status_pipe.send({"state": "ready", "rank": cfg.rank, "pid": os.getpid()})
        last_idle_status = 0.0
        while True:
            _poll_worker_command(command_pipe)
            demands = _scan_demands(cfg=cfg, schedules=schedules)
            cursors = _scan_cursors(
                cfg=cfg,
                train_schedule_id=train_schedule_id,
                inventory_length=train_inventory_length,
            )
            demanded = sorted(
                (
                    (
                        cast(int, record["created_ns"]),
                        schedule_id,
                        index,
                        cast(int, record["generation"]),
                    )
                    for (schedule_id, index), record in demands.items()
                    if owner_for_inventory_index(index, cfg.world_size) == cfg.rank
                ),
                key=lambda item: (item[0], item[1], item[2]),
            )
            obligations: list[tuple[str, int, int]] = [
                (schedule_id, index, generation)
                for _created, schedule_id, index, generation in demanded
            ]
            demanded_keys = {
                (schedule_id, index) for schedule_id, index, _generation in obligations
            }
            for lookahead in _lookahead_obligations(
                cfg=cfg,
                cursors=cursors,
                train_schedule_id=train_schedule_id,
                inventory_length=train_inventory_length,
            ):
                if (lookahead[0], lookahead[1]) not in demanded_keys:
                    obligations.append(lookahead)

            for schedule_id, index, generation in obligations:
                _poll_worker_command(command_pipe)
                _publish_schedule(
                    cfg=cfg,
                    backend=backend,
                    schedules=schedules,
                    schedule_id=schedule_id,
                    inventory_index=index,
                    generation=generation,
                    command_pipe=command_pipe,
                )
            _evict_finalized_train(
                cfg=cfg,
                demands=demands,
            )
            now = time.monotonic()
            if now - last_idle_status >= 1.0:
                _write_worker_status(cfg, state="idle")
                last_idle_status = now
            if command_pipe.poll(REMOTE_PREFETCH_POLL_SECONDS):
                _poll_worker_command(command_pipe)
    except _WorkerCancelled:
        _write_worker_status(cfg, state="stopped")
        try:
            status_pipe.send({"state": "stopped", "rank": cfg.rank})
        except (BrokenPipeError, EOFError, OSError):
            pass
    except BaseException as exc:
        active_family_id: str | None = None
        active_inventory_index: int | None = None
        active_body_name: str | None = None
        try:
            current_status_path = _status_path(
                _control_root(Path(cfg.cache_root), cfg.nonce),
                cfg.rank,
            )
            if current_status_path.exists():
                current_status = _load_status(
                    current_status_path,
                    nonce=cfg.nonce,
                    world_size=cfg.world_size,
                )
                if current_status["state"] in {"staging", "publishing"}:
                    active_family_id = cast(str | None, current_status["family_id"])
                    active_inventory_index = cast(
                        int | None,
                        current_status["inventory_index"],
                    )
                    active_body_name = cast(str | None, current_status["body_name"])
        except BaseException as status_read_error:
            exc.add_note(
                "also failed to recover active worker identity from status: "
                f"{type(status_read_error).__name__}: {status_read_error}"
            )
        try:
            _write_worker_status(
                cfg,
                state="error",
                family_id=active_family_id,
                inventory_index=active_inventory_index,
                body_name=active_body_name,
                error=exc,
            )
        except BaseException as status_write_error:
            exc.add_note(
                "also failed to persist terminal worker status: "
                f"{type(status_write_error).__name__}: {status_write_error}"
            )
        try:
            status_pipe.send(
                {
                    "state": "error",
                    "rank": cfg.rank,
                    "error_type": type(exc).__name__,
                    "error_message": _bounded_error_text(exc),
                    "family_id": active_family_id,
                    "inventory_index": active_inventory_index,
                    "body_name": active_body_name,
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        command_pipe.close()
        status_pipe.close()


def _all_gather_object(value: Any, *, world_size: int) -> list[Any]:
    if world_size == 1:
        return [value]
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, value)
    return gathered


def _broadcast_object(value: Any, *, rank: int, world_size: int) -> Any:
    if world_size == 1:
        return value
    payload = [value if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _raise_if_startup_phase_failed(
    *,
    phase: str,
    local_error: BaseException | None,
    world_size: int,
) -> None:
    """Make every rank leave a failed startup phase with the same error."""
    local_payload = (
        None
        if local_error is None
        else {
            "error_type": type(local_error).__name__,
            "error_message": str(local_error),
        }
    )
    gathered = _all_gather_object(local_payload, world_size=world_size)
    failures = [
        f"rank={rank} {payload['error_type']}: {payload['error_message']}"
        for rank, payload in enumerate(gathered)
        if isinstance(payload, Mapping)
    ]
    if not failures:
        return
    error = RemoteTrainingError(
        f"remote body prefetch startup phase {phase!r} failed symmetrically: " + "; ".join(failures)
    )
    if local_error is not None:
        raise error from local_error
    raise error


def _parse_local_world_size(world_size: int) -> int:
    raw_local = os.environ.get("LOCAL_WORLD_SIZE")
    raw_world = os.environ.get("WORLD_SIZE")
    if raw_local is None or raw_world is None:
        raise RemoteTrainingError(
            "strict remote body prefetch requires torchrun to publish "
            "LOCAL_WORLD_SIZE and WORLD_SIZE; launch with torchrun --standalone"
        )
    try:
        local_world_size = int(raw_local)
        environment_world_size = int(raw_world)
    except ValueError as exc:
        raise RemoteTrainingError(
            "LOCAL_WORLD_SIZE and WORLD_SIZE must be decimal integers"
        ) from exc
    if environment_world_size != world_size:
        raise RemoteTrainingError(
            "distributed world size disagrees with WORLD_SIZE: "
            f"runtime={world_size}, environment={environment_world_size}"
        )
    if local_world_size < 1 or world_size % local_world_size != 0:
        raise RemoteTrainingError(
            "strict remote body prefetch requires LOCAL_WORLD_SIZE to be a positive "
            "divisor of WORLD_SIZE: "
            f"LOCAL_WORLD_SIZE={local_world_size}, WORLD_SIZE={world_size}"
        )
    return local_world_size


def _initial_resume_demand(
    *,
    resume_state_dict: Mapping[str, Any] | None,
    train_binding: RemoteBodyBinding,
) -> tuple[int, int, list[tuple[int, int]]]:
    name_to_index = dict(train_binding.index_by_name)
    if resume_state_dict is None:
        return 1, 0, [(1, 0)]
    epoch = resume_state_dict.get("epoch", 1)
    if not isinstance(epoch, PositiveInt):
        raise RemoteTrainingError(
            f"remote prefetch resume epoch must be a strict int >= 1, got {epoch!r}"
        )
    shard = resume_state_dict.get("shard")
    if not isinstance(shard, str) or shard not in name_to_index:
        raise RemoteTrainingError(
            f"remote prefetch resume state must name a signed train inventory shard, got {shard!r}"
        )
    cursor_index = name_to_index[shard]
    demands: list[tuple[int, int]] = []
    raw_rows = resume_state_dict.get("batch_rows")
    if raw_rows is not None:
        if not isinstance(raw_rows, list) or not raw_rows:
            raise RemoteTrainingError(
                "remote prefetch exact resume batch_rows must be a non-empty list"
            )
        for row_index, raw in enumerate(raw_rows):
            if not isinstance(raw, (list, tuple)) or len(raw) != 3:
                raise RemoteTrainingError(
                    f"remote prefetch batch_rows[{row_index}] must be [epoch, shard_name, row_idx]"
                )
            row_epoch, row_shard, _row = raw
            if (
                not isinstance(row_epoch, PositiveInt)
                or not isinstance(row_shard, str)
                or row_shard not in name_to_index
            ):
                raise RemoteTrainingError(
                    f"remote prefetch batch_rows[{row_index}] has invalid signed identity"
                )
            demands.append((row_epoch, name_to_index[row_shard]))
    demands.append((epoch, cursor_index))
    return epoch, cursor_index, sorted(set(demands))


class RemoteBodyPrefetcher:
    """Parent-side cache-only coordinator for one rank's spawned worker."""

    def __init__(
        self,
        *,
        remote: RemoteTrainingConfig,
        cache_root: Path,
        nonce: str,
        rank: int,
        world_size: int,
        families: tuple[RemoteBodyFamily, ...],
        bindings: Mapping[str, RemoteBodyBinding],
        packed_binding: RemoteBodyBinding | None,
        train_binding: RemoteBodyBinding,
        control_root: Path,
        process: BaseProcess,
        command_pipe: Connection,
        status_pipe: Connection,
        session_lock_fd: int | None,
        admission: RemotePrefetchDiskAdmission,
    ) -> None:
        self.remote = remote
        self.cache_root = cache_root
        self.nonce = nonce
        self.rank = rank
        self.world_size = world_size
        self.families = families
        self.bindings = dict(bindings)
        self.packed_binding = packed_binding
        self.train_binding = train_binding
        self.control_root = control_root
        self.process = process
        self.command_pipe = command_pipe
        self.status_pipe = status_pipe
        self.session_lock_fd = session_lock_fd
        self.admission = admission
        self.cancel_event = threading.Event()
        self._stopping = False
        self._worker_stopped = False
        self._session_released = False
        self._failure: RemoteTrainingError | None = None
        self._local_failure_lock = threading.RLock()
        self._last_successful_local_failure_check_monotonic: float | None = None
        self._peer_status_sweep_lock = threading.Lock()
        self._last_successful_peer_status_sweep_monotonic: float | None = None
        self._demand_lock = threading.Lock()
        self._families_by_id = {family.family_id: family for family in families}
        self._schedules = _schedule_map(families)
        self._train_inventory_length = len(train_binding.index_by_name)
        self._worker_cfg = _WorkerConfig(
            bucket_id=remote.bucket_id,
            cache_root=str(cache_root),
            nonce=nonce,
            rank=rank,
            world_size=world_size,
            lookahead_shards_per_rank=(remote.body_prefetch.lookahead_shards_per_rank),
            cache_shards=remote.body_prefetch.cache_shards,
            timeout_seconds=remote.selection_wait_timeout_seconds,
            retry_interval_seconds=remote.retry_interval_seconds,
            families=families,
        )

    @classmethod
    def start(
        cls,
        *,
        remote: RemoteTrainingConfig,
        cache_root: Path,
        rank: int,
        world_size: int,
        train_selection_dir: Path,
        selection_materializers: Sequence[RemoteSelectionMaterializer],
        packed_materializer: RemotePackedRowMaterializer | None,
        resume_state_dict: Mapping[str, Any] | None,
        _backend_factory: Any = None,
    ) -> RemoteBodyPrefetcher:
        resolved_cache_root: Path | None = None
        families: tuple[RemoteBodyFamily, ...] | None = None
        bindings: dict[str, RemoteBodyBinding] | None = None
        packed_binding: RemoteBodyBinding | None = None
        setup_error: BaseException | None = None
        try:
            _parse_local_world_size(world_size)
            resolved_cache_root = Path(cache_root).resolve()
            _ensure_direct_directory(resolved_cache_root)
            families, bindings, packed_binding = build_remote_body_plan(
                train_selection_dir=train_selection_dir,
                selection_materializers=selection_materializers,
                packed_materializer=packed_materializer,
            )
        except BaseException as exc:
            setup_error = exc
        _raise_if_startup_phase_failed(
            phase="local-plan",
            local_error=setup_error,
            world_size=world_size,
        )
        assert resolved_cache_root is not None
        assert families is not None
        assert bindings is not None
        cache_root = resolved_cache_root
        train_key = str(Path(train_selection_dir).resolve())
        train_binding = bindings[train_key]
        plan_identity = _plan_identity(families)
        gathered_plans = _all_gather_object(plan_identity, world_size=world_size)
        if any(item != plan_identity for item in gathered_plans):
            raise RemoteTrainingError(
                "remote body prefetch signed family plans differ across ranks"
            )

        session_lock_fd: int | None = None
        session_lock_path = cache_root / ".remote-prefetch.session.lock"
        nonce: str | None = None
        sentinel_bytes: bytes | None = None
        sentinel_path: Path | None = None
        canary_fd: int | None = None
        canary_path: Path | None = None
        process: BaseProcess | None = None
        child_command: Connection | None = None
        parent_command: Connection | None = None
        parent_status: Connection | None = None
        child_status: Connection | None = None
        control: Path | None = None
        try:
            lock_error: BaseException | None = None
            if rank == 0:
                try:
                    session_lock_fd = _open_cache_lock(session_lock_path)
                    fcntl.flock(
                        session_lock_fd,
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                    nonce = uuid.uuid4().hex
                    sentinel_bytes = os.urandom(32)
                except BlockingIOError as exc:
                    lock_error = RemoteTrainingError(
                        "remote body cache namespace is already owned by another "
                        f"job: {cache_root}. Concurrent jobs must use separate cache roots."
                    )
                    lock_error.add_note(f"flock rejected rank 0: {exc}")
                except BaseException as exc:
                    lock_error = exc
            _raise_if_startup_phase_failed(
                phase="session-lock",
                local_error=lock_error,
                world_size=world_size,
            )
            nonce = cast(str, _broadcast_object(nonce, rank=rank, world_size=world_size))
            if not isinstance(nonce, str) or re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
                raise RemoteTrainingError("remote prefetch run nonce broadcast is malformed")

            sentinel_path = cache_root / f".remote-prefetch-sentinel.{nonce}"
            expected_sentinel_hex = cast(
                str,
                _broadcast_object(
                    sentinel_bytes.hex() if sentinel_bytes is not None else None,
                    rank=rank,
                    world_size=world_size,
                ),
            )
            sentinel_write_error: BaseException | None = None
            if rank == 0:
                assert sentinel_bytes is not None
                try:
                    atomic_write_bytes(sentinel_bytes, sentinel_path)
                    _fsync_directory(cache_root)
                except BaseException as exc:
                    sentinel_write_error = exc
            _raise_if_startup_phase_failed(
                phase="sentinel-publish",
                local_error=sentinel_write_error,
                world_size=world_size,
            )
            observed_hex: str | None = None
            sentinel_read_error: BaseException | None = None
            try:
                observed_hex = _read_direct_bytes(sentinel_path, max_bytes=64).hex()
            except BaseException as exc:
                sentinel_read_error = exc
            _raise_if_startup_phase_failed(
                phase="sentinel-observe",
                local_error=sentinel_read_error,
                world_size=world_size,
            )
            gathered_sentinels = _all_gather_object(
                observed_hex,
                world_size=world_size,
            )
            if gathered_sentinels != [expected_sentinel_hex] * world_size:
                raise RemoteTrainingError(
                    "remote body prefetch cache paths did not expose the exact "
                    "rank-0 sentinel bytes through one shared namespace: "
                    f"expected={expected_sentinel_hex}, "
                    f"observations={gathered_sentinels}"
                )
            sentinel_remove_error: BaseException | None = None
            if rank == 0:
                try:
                    sentinel_path.unlink()
                    _fsync_directory(cache_root)
                except BaseException as exc:
                    sentinel_remove_error = exc
            _raise_if_startup_phase_failed(
                phase="sentinel-remove",
                local_error=sentinel_remove_error,
                world_size=world_size,
            )

            canary_path = cache_root / f".remote-prefetch-flock-canary.{nonce}"
            canary_publish_error: BaseException | None = None
            if rank == 0:
                try:
                    canary_fd = _open_cache_lock(canary_path)
                    fcntl.flock(canary_fd, fcntl.LOCK_EX)
                except BaseException as exc:
                    canary_publish_error = exc
            _raise_if_startup_phase_failed(
                phase="flock-canary-publish",
                local_error=canary_publish_error,
                world_size=world_size,
            )
            flock_shared = True
            canary_probe_error: BaseException | None = None
            if rank != 0:
                try:
                    probe_fd = _open_cache_lock(canary_path)
                    try:
                        try:
                            fcntl.flock(probe_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            flock_shared = True
                        else:
                            flock_shared = False
                            fcntl.flock(probe_fd, fcntl.LOCK_UN)
                    finally:
                        os.close(probe_fd)
                except BaseException as exc:
                    canary_probe_error = exc
            _raise_if_startup_phase_failed(
                phase="flock-canary-probe",
                local_error=canary_probe_error,
                world_size=world_size,
            )
            gathered_flock = _all_gather_object(flock_shared, world_size=world_size)
            if gathered_flock != [True] * world_size:
                raise RemoteTrainingError(
                    "remote body prefetch ranks do not share one flock namespace"
                )
            canary_remove_error: BaseException | None = None
            if rank == 0:
                assert canary_fd is not None
                try:
                    fcntl.flock(canary_fd, fcntl.LOCK_UN)
                    os.close(canary_fd)
                    canary_fd = None
                    canary_path.unlink()
                    _fsync_directory(cache_root)
                except BaseException as exc:
                    canary_remove_error = exc
            _raise_if_startup_phase_failed(
                phase="flock-canary-remove",
                local_error=canary_remove_error,
                world_size=world_size,
            )

            control_create_error: BaseException | None = None
            if rank == 0:
                try:
                    _create_control_tree(
                        cache_root=cache_root,
                        nonce=nonce,
                        world_size=world_size,
                        families=families,
                    )
                except BaseException as exc:
                    control_create_error = exc
            _raise_if_startup_phase_failed(
                phase="control-create",
                local_error=control_create_error,
                world_size=world_size,
            )
            control_validate_error: BaseException | None = None
            try:
                control = _validate_control_tree(
                    cache_root=cache_root,
                    nonce=nonce,
                    world_size=world_size,
                    families=families,
                )
            except BaseException as exc:
                control_validate_error = exc
            _raise_if_startup_phase_failed(
                phase="control-validate",
                local_error=control_validate_error,
                world_size=world_size,
            )
            assert control is not None

            cache_shards = remote.body_prefetch.cache_shards
            admission: RemotePrefetchDiskAdmission | None = None
            admission_error: BaseException | None = None
            if rank == 0:
                try:
                    admission = calculate_remote_prefetch_disk_admission(
                        cache_root=cache_root,
                        families=families,
                        world_size=world_size,
                        cache_shards=cache_shards,
                        lookahead_shards_per_rank=(remote.body_prefetch.lookahead_shards_per_rank),
                    )
                    if admission.available_bytes < (
                        admission.additional_required_bytes + admission.safety_reserve_bytes
                    ):
                        raise RemoteTrainingError(
                            "remote body prefetch disk admission failed before "
                            f"worker spawn: peak={admission.peak_bytes}, "
                            f"current_managed={admission.current_managed_bytes}, "
                            f"additional_required={admission.additional_required_bytes}, "
                            f"free={admission.available_bytes}, "
                            f"safety_reserve={admission.safety_reserve_bytes}"
                        )
                except BaseException as exc:
                    admission_error = exc
            _raise_if_startup_phase_failed(
                phase="disk-admission",
                local_error=admission_error,
                world_size=world_size,
            )
            admission = cast(
                RemotePrefetchDiskAdmission,
                _broadcast_object(
                    admission,
                    rank=rank,
                    world_size=world_size,
                ),
            )
            logger.info(
                "remote_training.prefetch_disk_admission "
                f"peak_bytes={admission.peak_bytes} "
                f"additional_required_bytes={admission.additional_required_bytes} "
                f"free_bytes={admission.available_bytes} "
                f"safety_reserve_bytes={admission.safety_reserve_bytes}"
            )

            resume_values: tuple[int, int, list[tuple[int, int]]] | None = None
            resume_error: BaseException | None = None
            try:
                resume_values = _initial_resume_demand(
                    resume_state_dict=resume_state_dict,
                    train_binding=train_binding,
                )
            except BaseException as exc:
                resume_error = exc
            _raise_if_startup_phase_failed(
                phase="resume-demand-parse",
                local_error=resume_error,
                world_size=world_size,
            )
            assert resume_values is not None
            cursor_epoch, cursor_index, initial_demands = resume_values
            local_resume = {
                "rank": rank,
                "epoch": cursor_epoch,
                "inventory_index": cursor_index,
                "demands": initial_demands,
            }
            gathered_resume = _all_gather_object(
                local_resume,
                world_size=world_size,
            )
            resume_gather_error: BaseException | None = None
            try:
                gathered_ranks = {
                    item["rank"] for item in gathered_resume if isinstance(item, Mapping)
                }
                if gathered_ranks != set(range(world_size)) or len(gathered_resume) != world_size:
                    raise RemoteTrainingError(
                        "remote body prefetch resume-demand all-gather is malformed"
                    )
            except BaseException as exc:
                resume_gather_error = exc
            _raise_if_startup_phase_failed(
                phase="resume-demand-gather",
                local_error=resume_gather_error,
                world_size=world_size,
            )
            record_error: BaseException | None = None
            try:
                generation = (cursor_epoch - 1) * len(train_binding.index_by_name) + cursor_index
                _atomic_record(
                    {
                        "schema_version": REMOTE_PREFETCH_SCHEMA_VERSION,
                        "nonce": nonce,
                        "rank": rank,
                        "schedule_id": train_binding.schedule_id,
                        "epoch": cursor_epoch,
                        "inventory_index": cursor_index,
                        "generation": generation,
                    },
                    _cursor_path(control, rank),
                )
                _atomic_record(
                    {
                        "schema_version": REMOTE_PREFETCH_SCHEMA_VERSION,
                        "nonce": nonce,
                        "rank": rank,
                        "schedule_id": train_binding.schedule_id,
                        "generation": generation - 1,
                    },
                    _progress_path(control, rank),
                )
                for demand_epoch, demand_index in initial_demands:
                    demand_generation = (demand_epoch - 1) * len(
                        train_binding.index_by_name
                    ) + demand_index
                    _atomic_record(
                        {
                            "schema_version": REMOTE_PREFETCH_SCHEMA_VERSION,
                            "nonce": nonce,
                            "requester_rank": rank,
                            "schedule_id": train_binding.schedule_id,
                            "epoch": demand_epoch,
                            "inventory_index": demand_index,
                            "generation": demand_generation,
                            "created_ns": time.time_ns(),
                        },
                        _demand_path(
                            control,
                            requester_rank=rank,
                            schedule_id=train_binding.schedule_id,
                            inventory_index=demand_index,
                        ),
                    )
            except BaseException as exc:
                record_error = exc
            _raise_if_startup_phase_failed(
                phase="rank-control-publish",
                local_error=record_error,
                world_size=world_size,
            )

            worker_cfg = _WorkerConfig(
                bucket_id=remote.bucket_id,
                cache_root=str(cache_root),
                nonce=nonce,
                rank=rank,
                world_size=world_size,
                lookahead_shards_per_rank=(remote.body_prefetch.lookahead_shards_per_rank),
                cache_shards=cache_shards,
                timeout_seconds=remote.selection_wait_timeout_seconds,
                retry_interval_seconds=remote.retry_interval_seconds,
                families=families,
            )
            spawn_error: BaseException | None = None
            try:
                context = mp.get_context("spawn")
                child_command, parent_command = context.Pipe(duplex=False)
                parent_status, child_status = context.Pipe(duplex=False)
                process = context.Process(
                    target=_worker_main,
                    args=(
                        worker_cfg,
                        child_command,
                        child_status,
                        _backend_factory,
                    ),
                    name=f"remote-body-prefetch-rank-{rank}",
                    daemon=False,
                )
                process.start()
            except BaseException as exc:
                spawn_error = exc
            finally:
                if child_command is not None:
                    child_command.close()
                    child_command = None
                if child_status is not None:
                    child_status.close()
                    child_status = None
            _raise_if_startup_phase_failed(
                phase="worker-spawn",
                local_error=spawn_error,
                world_size=world_size,
            )
            assert process is not None
            assert parent_command is not None
            assert parent_status is not None
            ready_deadline = time.monotonic() + min(
                60.0,
                remote.selection_wait_timeout_seconds,
            )
            ready_message: Any = None
            ready_error: BaseException | None = None
            try:
                while time.monotonic() < ready_deadline:
                    if parent_status.poll(REMOTE_PREFETCH_POLL_SECONDS):
                        ready_message = parent_status.recv()
                        break
                    if process.exitcode is not None:
                        break
            except BaseException as exc:
                ready_error = exc
            _raise_if_startup_phase_failed(
                phase="worker-ready",
                local_error=ready_error,
                world_size=world_size,
            )
            local_ready = (
                isinstance(ready_message, dict)
                and ready_message.get("state") == "ready"
                and ready_message.get("rank") == rank
                and process.is_alive()
            )
            gathered_ready = _all_gather_object(local_ready, world_size=world_size)
            if gathered_ready != [True] * world_size:
                raise RemoteTrainingError(
                    "remote body prefetch workers did not start symmetrically: "
                    f"{gathered_ready}; local_message={ready_message!r}, "
                    f"exitcode={process.exitcode}"
                )
            controller: RemoteBodyPrefetcher | None = None
            attach_error: BaseException | None = None
            try:
                controller = cls(
                    remote=remote,
                    cache_root=cache_root,
                    nonce=nonce,
                    rank=rank,
                    world_size=world_size,
                    families=families,
                    bindings=bindings,
                    packed_binding=packed_binding,
                    train_binding=train_binding,
                    control_root=control,
                    process=process,
                    command_pipe=parent_command,
                    status_pipe=parent_status,
                    session_lock_fd=session_lock_fd,
                    admission=admission,
                )
                for materializer in selection_materializers:
                    binding = bindings[str(materializer.local_dir.resolve())]
                    materializer.attach_body_prefetch(controller, binding)
                if packed_materializer is not None:
                    assert packed_binding is not None
                    packed_materializer.attach_body_prefetch(controller, packed_binding)
            except BaseException as exc:
                attach_error = exc
            _raise_if_startup_phase_failed(
                phase="reader-attach",
                local_error=attach_error,
                world_size=world_size,
            )
            assert controller is not None
            logger.info(
                "remote_training.prefetch_worker_started "
                f"rank={rank} pid={process.pid} nonce={nonce} "
                f"lookahead={remote.body_prefetch.lookahead_shards_per_rank}"
            )
            return controller
        except BaseException as primary:
            cleanup_errors: list[BaseException] = []
            local_worker_terminal = process is None
            if parent_command is not None:
                try:
                    parent_command.send({"op": "cancel"})
                except (BrokenPipeError, EOFError, OSError):
                    pass
            if process is not None and process.pid is not None:
                try:
                    process.join(timeout=REMOTE_PREFETCH_GRACEFUL_JOIN_SECONDS)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=REMOTE_PREFETCH_HARD_JOIN_SECONDS)
                    local_worker_terminal = not process.is_alive()
                    if not local_worker_terminal:
                        cleanup_errors.append(
                            RemoteTrainingError(
                                "remote prefetch startup worker survived bounded "
                                f"termination: rank={rank}, pid={process.pid}"
                            )
                        )
                except BaseException as exc:
                    local_worker_terminal = False
                    cleanup_errors.append(exc)
            if control is not None and nonce is not None and local_worker_terminal:
                try:
                    status_path = _status_path(control, rank)
                    existing_terminal = False
                    if status_path.exists():
                        try:
                            existing_terminal = _load_status(
                                status_path,
                                nonce=nonce,
                                world_size=world_size,
                            )["state"] in {"stopped", "error"}
                        except BaseException:
                            existing_terminal = False
                    if not existing_terminal:
                        _atomic_record(
                            {
                                "schema_version": REMOTE_PREFETCH_SCHEMA_VERSION,
                                "nonce": nonce,
                                "rank": rank,
                                "pid": (
                                    process.pid
                                    if process is not None and process.pid is not None
                                    else os.getpid()
                                ),
                                "state": "error",
                                "family_id": None,
                                "inventory_index": None,
                                "body_name": None,
                                "updated_ns": time.time_ns(),
                                "error_type": "StartupCleanup",
                                "error_message": _bounded_error_text(primary),
                                "traceback": None,
                            },
                            status_path,
                        )
                except BaseException as exc:
                    cleanup_errors.append(exc)
            for connection in (
                child_command,
                parent_command,
                parent_status,
                child_status,
            ):
                if connection is None:
                    continue
                try:
                    connection.close()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if canary_fd is not None:
                try:
                    fcntl.flock(canary_fd, fcntl.LOCK_UN)
                    os.close(canary_fd)
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if rank == 0:
                for path in (sentinel_path, canary_path):
                    if path is not None:
                        try:
                            path.unlink()
                        except FileNotFoundError:
                            continue
                        except BaseException as exc:
                            cleanup_errors.append(exc)
            all_workers_terminal = local_worker_terminal
            if session_lock_fd is not None and control is not None and nonce is not None:
                deadline = time.monotonic() + REMOTE_PREFETCH_GRACEFUL_JOIN_SECONDS
                all_workers_terminal = False
                while time.monotonic() < deadline:
                    try:
                        statuses = [
                            _load_status(
                                _status_path(control, owner_rank),
                                nonce=nonce,
                                world_size=world_size,
                            )
                            for owner_rank in range(world_size)
                        ]
                    except (FileNotFoundError, RemoteTrainingError):
                        time.sleep(REMOTE_PREFETCH_POLL_SECONDS)
                        continue
                    all_workers_terminal = all(
                        status["state"] in {"stopped", "error"} for status in statuses
                    )
                    if all_workers_terminal:
                        break
                    time.sleep(REMOTE_PREFETCH_POLL_SECONDS)
                if not all_workers_terminal:
                    cleanup_errors.append(
                        RemoteTrainingError(
                            "remote prefetch startup session lock retained because "
                            "not every rank proved its worker terminal"
                        )
                    )
            if session_lock_fd is not None and all_workers_terminal:
                try:
                    fcntl.flock(session_lock_fd, fcntl.LOCK_UN)
                    os.close(session_lock_fd)
                except BaseException as exc:
                    cleanup_errors.append(exc)
            for cleanup_error in cleanup_errors:
                primary.add_note(
                    "remote prefetch startup cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise

    def _sticky_failure(
        self,
        message: str,
        *,
        family_id: str | None = None,
        inventory_index: int | None = None,
        body_name: str | None = None,
    ) -> RemoteTrainingError:
        with self._local_failure_lock:
            if self._failure is None:
                self._failure = RemoteTrainingError(message)
                try:
                    status_path = _status_path(self.control_root, self.rank)
                    existing_terminal = False
                    if status_path.exists():
                        try:
                            existing_terminal = (
                                _load_status(
                                    status_path,
                                    nonce=self.nonce,
                                    world_size=self.world_size,
                                )["state"]
                                == "error"
                            )
                        except BaseException:
                            # Replace malformed local status with the sticky parent
                            # record below; the primary worker error stays intact.
                            existing_terminal = False
                    if not existing_terminal:
                        _atomic_record(
                            {
                                "schema_version": REMOTE_PREFETCH_SCHEMA_VERSION,
                                "nonce": self.nonce,
                                "rank": self.rank,
                                "pid": self.process.pid or -1,
                                "state": "error",
                                "family_id": family_id,
                                "inventory_index": inventory_index,
                                "body_name": body_name,
                                "updated_ns": time.time_ns(),
                                "error_type": "RemoteWorkerExit",
                                "error_message": message,
                                "traceback": None,
                            },
                            status_path,
                        )
                except BaseException as status_error:
                    self._failure.add_note(
                        "also failed to persist sticky worker state: "
                        f"{type(status_error).__name__}: {status_error}"
                    )
            return self._failure

    def _drain_status_pipe(self) -> None:
        with self._local_failure_lock:
            while self.status_pipe.poll():
                try:
                    message = self.status_pipe.recv()
                except EOFError:
                    break
                if isinstance(message, dict) and message.get("state") == "error":
                    self._sticky_failure(
                        "remote body prefetch worker failed: "
                        f"rank={message.get('rank')}, family={message.get('family_id')}, "
                        f"index={message.get('inventory_index')}, "
                        f"body={message.get('body_name')}, "
                        f"{message.get('error_type')}: {message.get('error_message')}",
                        family_id=cast(str | None, message.get("family_id")),
                        inventory_index=cast(int | None, message.get("inventory_index")),
                        body_name=cast(str | None, message.get("body_name")),
                    )

    def check_worker_errors(self) -> None:
        """Raise sticky worker failure from any rank before further training work."""
        with self._local_failure_lock:
            if self._failure is not None:
                raise self._failure
            now = time.monotonic()
            last_check = self._last_successful_local_failure_check_monotonic
            if (
                last_check is None
                or now - last_check >= REMOTE_PREFETCH_LOCAL_FAILURE_CHECK_INTERVAL_SECONDS
            ):
                self._drain_status_pipe()
                if self._failure is not None:
                    raise self._failure
                exitcode = None if self._stopping else self.process.exitcode
                if exitcode is not None:
                    raise self._sticky_failure(
                        "remote body prefetch worker exited unexpectedly: "
                        f"rank={self.rank}, pid={self.process.pid}, "
                        f"exitcode={exitcode}"
                    )
                self._last_successful_local_failure_check_monotonic = time.monotonic()

        if not self._peer_status_sweep_lock.acquire(blocking=False):
            with self._local_failure_lock:
                if self._failure is not None:
                    raise self._failure
            return
        try:
            with self._local_failure_lock:
                if self._failure is not None:
                    raise self._failure
            now = time.monotonic()
            last_sweep = self._last_successful_peer_status_sweep_monotonic
            if (
                last_sweep is not None
                and now - last_sweep < REMOTE_PREFETCH_PEER_STATUS_SWEEP_INTERVAL_SECONDS
            ):
                with self._local_failure_lock:
                    if self._failure is not None:
                        raise self._failure
                return
            for owner_rank in range(self.world_size):
                path = _status_path(self.control_root, owner_rank)
                try:
                    status_record = _load_status(
                        path,
                        nonce=self.nonce,
                        world_size=self.world_size,
                    )
                except RemoteTrainingError as exc:
                    if isinstance(exc.__cause__, FileNotFoundError):
                        continue
                    # Status is a replace-in-place control snapshot, not an
                    # immutable artifact.  On shared NFS a reader can observe
                    # the opened generation's descriptor transition while its
                    # owner atomically publishes the next generation.  The
                    # descriptor-pinned reader rejects that one observation;
                    # defer it to the next peer sweep just like a temporarily
                    # invisible status pathname.  Every other validation error
                    # remains terminal and the local pipe/exit checks above
                    # continue to report this rank's worker failures eagerly.
                    if _is_transient_mutable_record_read(exc):
                        continue
                    with self._local_failure_lock:
                        if self._failure is not None:
                            raise self._failure from None
                    raise
                if status_record["state"] == "error":
                    raise self._sticky_failure(
                        "remote body prefetch owner reported terminal failure: "
                        f"owner={owner_rank}, family={status_record['family_id']}, "
                        f"index={status_record['inventory_index']}, "
                        f"body={status_record['body_name']}, "
                        f"{status_record['error_type']}: "
                        f"{status_record['error_message']}",
                        family_id=cast(str | None, status_record["family_id"]),
                        inventory_index=cast(int | None, status_record["inventory_index"]),
                        body_name=cast(str | None, status_record["body_name"]),
                    )
            self._last_successful_peer_status_sweep_monotonic = time.monotonic()
            with self._local_failure_lock:
                if self._failure is not None:
                    raise self._failure
        finally:
            self._peer_status_sweep_lock.release()

    def _family(self, binding: RemoteBodyBinding) -> RemoteBodyFamily:
        try:
            return self._families_by_id[binding.family_id]
        except KeyError as exc:
            raise RemoteTrainingError(
                f"remote body binding names unknown family {binding.family_id}"
            ) from exc

    def _unchanged_preexisting(
        self,
        *,
        family: RemoteBodyFamily,
        inventory_index: int,
    ) -> bool:
        return _same_preexisting_descriptor(
            family,
            inventory_index=inventory_index,
        )

    @beartype
    def publish_cursor(
        self,
        *,
        binding: RemoteBodyBinding,
        epoch: PositiveIntLike,
        inventory_index: NonNegativeIntLike,
    ) -> None:
        if not binding.train_schedule:
            return
        self.check_worker_errors()
        if inventory_index >= self._train_inventory_length:
            raise RemoteTrainingError(
                f"remote prefetch cursor index is invalid: {inventory_index!r}"
            )
        generation = (epoch - 1) * self._train_inventory_length + inventory_index
        _atomic_record(
            {
                "schema_version": REMOTE_PREFETCH_SCHEMA_VERSION,
                "nonce": self.nonce,
                "rank": self.rank,
                "schedule_id": self.train_binding.schedule_id,
                "epoch": epoch,
                "inventory_index": inventory_index,
                "generation": generation,
            },
            _cursor_path(self.control_root, self.rank),
        )

    @beartype
    def mark_loaded(
        self,
        *,
        binding: RemoteBodyBinding,
        epoch: PositiveIntLike,
        inventory_index: NonNegativeIntLike,
    ) -> None:
        if not binding.train_schedule:
            return
        self.check_worker_errors()
        family = self._family(binding)
        if inventory_index >= len(family.entries):
            raise RemoteTrainingError(
                f"remote prefetch progress index is invalid: {inventory_index!r}"
            )
        generation = (epoch - 1) * self._train_inventory_length + inventory_index
        _atomic_record(
            {
                "schema_version": REMOTE_PREFETCH_SCHEMA_VERSION,
                "nonce": self.nonce,
                "rank": self.rank,
                "schedule_id": self.train_binding.schedule_id,
                "generation": generation,
            },
            _progress_path(self.control_root, self.rank),
        )

    @beartype
    def request_window(
        self,
        *,
        binding: RemoteBodyBinding,
        start_index: NonNegativeIntLike = 0,
    ) -> None:
        """Explicitly demand the validation startup window for this rank."""
        if binding.train_schedule:
            return
        self.check_worker_errors()
        family = self._family(binding)
        if start_index >= len(family.entries):
            raise RemoteTrainingError(
                f"remote prefetch validation start_index is invalid: {start_index!r}"
            )
        for offset in range(
            min(
                len(family.entries),
                self.remote.body_prefetch.lookahead_shards_per_rank * self.world_size,
            )
        ):
            index = (start_index + offset) % len(family.entries)
            self._publish_demand(
                binding=binding,
                epoch=1,
                inventory_index=index,
            )

    @beartype
    def _publish_demand(
        self,
        *,
        binding: RemoteBodyBinding,
        epoch: PositiveIntLike,
        inventory_index: NonNegativeIntLike,
    ) -> int:
        family = self._family(binding)
        if inventory_index >= len(family.entries):
            raise RemoteTrainingError("remote body demand inventory index is invalid")
        generation = (epoch - 1) * len(family.entries) + inventory_index
        eviction_lock_fds: list[int] = []
        if binding.train_schedule:
            for component in sorted(
                self._schedules[binding.schedule_id],
                key=lambda family: (family.local_dir, family.family_id),
            ):
                entry = component.entry(inventory_index)
                lock_fd = _open_cache_lock(
                    Path(component.local_dir) / f".{entry.name}.lock"
                )
                fcntl.flock(lock_fd, fcntl.LOCK_SH)
                eviction_lock_fds.append(lock_fd)
        demand_path = _demand_path(
            self.control_root,
            requester_rank=self.rank,
            schedule_id=binding.schedule_id,
            inventory_index=inventory_index,
        )
        try:
            with self._demand_lock:
                had_existing_demand = demand_path.exists()
                if had_existing_demand:
                    existing = _load_demand(
                        demand_path,
                        cfg=self._worker_cfg,
                        schedules=self._schedules,
                        requester_rank=self.rank,
                    )
                    if cast(int, existing["generation"]) >= generation:
                        return generation
                record = {
                    "schema_version": REMOTE_PREFETCH_SCHEMA_VERSION,
                    "nonce": self.nonce,
                    "requester_rank": self.rank,
                    "schedule_id": binding.schedule_id,
                    "epoch": epoch,
                    "inventory_index": inventory_index,
                    "generation": generation,
                    "created_ns": time.time_ns(),
                }
                try:
                    _atomic_record(record, demand_path)
                except BaseException as primary:
                    # Never delete a record that another logical request may
                    # already depend on. A newly created canonical record can
                    # be removed precisely; an overwritten record is retained
                    # as a conservative demand after uncertain durability.
                    if not had_existing_demand:
                        try:
                            if demand_path.exists():
                                current = _load_demand(
                                    demand_path,
                                    cfg=self._worker_cfg,
                                    schedules=self._schedules,
                                    requester_rank=self.rank,
                                )
                                if current == record:
                                    demand_path.unlink()
                                    _fsync_directory(demand_path.parent)
                        except BaseException as cleanup_error:
                            primary.add_note(
                                "also failed to roll back requester demand: "
                                f"{type(cleanup_error).__name__}: {cleanup_error}"
                            )
                    raise
        finally:
            for eviction_lock_fd in reversed(eviction_lock_fds):
                try:
                    fcntl.flock(eviction_lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(eviction_lock_fd)
        return generation

    def _remove_satisfied_demand(
        self,
        *,
        binding: RemoteBodyBinding,
        inventory_index: int,
        generation: int,
    ) -> None:
        path = _demand_path(
            self.control_root,
            requester_rank=self.rank,
            schedule_id=binding.schedule_id,
            inventory_index=inventory_index,
        )
        with self._demand_lock:
            if not path.exists():
                return
            record = _load_demand(
                path,
                cfg=self._worker_cfg,
                schedules=self._schedules,
                requester_rank=self.rank,
            )
            # A later request may atomically replace the same requester-specific
            # path while this consumer is finishing its descriptor-pinned read.
            if record["generation"] != generation:
                return
            path.unlink()
            _fsync_directory(path.parent)

    def _wait_ready(
        self,
        *,
        binding: RemoteBodyBinding,
        inventory_index: int,
        generation: int,
        cancel_event: threading.Event | None,
    ) -> None:
        deadline = time.monotonic() + self.remote.selection_wait_timeout_seconds
        owner_rank = owner_for_inventory_index(inventory_index, self.world_size)
        components = self._schedules[binding.schedule_id]
        ready_path = _ready_path(
            self.control_root,
            schedule_id=binding.schedule_id,
            inventory_index=inventory_index,
        )
        while True:
            if self.cancel_event.is_set() or (cancel_event is not None and cancel_event.is_set()):
                raise RemoteMaterializationCancelled(
                    "remote cache materialization cancelled while waiting for "
                    f"owner={owner_rank}, family={binding.family_id}, "
                    f"index={inventory_index}"
                )
            self.check_worker_errors()
            if ready_path.exists():
                ready = _load_ready(
                    ready_path,
                    cfg=self._worker_cfg,
                    schedule_id=binding.schedule_id,
                    inventory_index=inventory_index,
                    components=components,
                )
                # The ready path is keyed by body identity, not logical epoch.
                # A valid receipt from an earlier epoch is expected at wrap;
                # wait for the owner to refresh it to this demand generation.
                if cast(int, ready["generation"]) >= generation:
                    return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                status_path = _status_path(self.control_root, owner_rank)
                owner_status: Any = (
                    _load_status(
                        status_path,
                        nonce=self.nonce,
                        world_size=self.world_size,
                    )
                    if status_path.exists()
                    else "missing"
                )
                family = self._family(binding)
                entry = family.entry(inventory_index)
                raise RemoteTrainingError(
                    "timed out waiting for strict remote body owner: "
                    f"family={binding.family_id}, body={entry.name}, "
                    f"inventory_index={inventory_index}, owner={owner_rank}, "
                    f"worker_status={owner_status}"
                )
            time.sleep(min(REMOTE_PREFETCH_POLL_SECONDS, remaining))

    @beartype
    def read_body(
        self,
        *,
        binding: RemoteBodyBinding,
        path: Path,
        entry: _RemoteArtifactMapping,
        trust_existing_cache: bool,
        cancel_event: threading.Event | None,
        epoch: PositiveIntLike,
    ) -> bytes:
        """Request the nominal owner and descriptor-pin one final cache body."""
        self.check_worker_errors()
        family = self._family(binding)
        inventory_index = binding.inventory_index(path.name)
        expected_entry = family.entry(inventory_index)
        if (
            expected_entry.byte_length != entry["byte_length"]
            or expected_entry.content_sha256 != entry["content_sha256"]
            or expected_entry.file_fingerprint != entry.get("file_fingerprint")
        ):
            raise RemoteTrainingError(f"remote body reader entry drift for {path}")
        unchanged_preexisting = self._unchanged_preexisting(
            family=family,
            inventory_index=inventory_index,
        )
        schedule_preexisting = all(
            self._unchanged_preexisting(
                family=component,
                inventory_index=inventory_index,
            )
            for component in self._schedules[binding.schedule_id]
        )
        demand_generation: int | None = None
        try:
            if not schedule_preexisting:
                demand_generation = self._publish_demand(
                    binding=binding,
                    epoch=epoch,
                    inventory_index=inventory_index,
                )
                self._wait_ready(
                    binding=binding,
                    inventory_index=inventory_index,
                    generation=demand_generation,
                    cancel_event=cancel_event,
                )

            if not unchanged_preexisting:
                receipt_path = _receipt_path(
                    self.control_root,
                    family_id=family.family_id,
                    inventory_index=inventory_index,
                )
                if not receipt_path.exists():
                    raise RemoteTrainingError(
                        "fresh remote body is missing its nonce-scoped verification "
                        f"receipt: family={family.family_id}, body={path.name}, "
                        f"owner={owner_for_inventory_index(inventory_index, self.world_size)}"
                    )
                _load_receipt(
                    receipt_path,
                    cfg=self._worker_cfg,
                    family=family,
                    inventory_index=inventory_index,
                )

            lock_fd = _open_cache_lock(Path(family.local_dir) / f".{path.name}.lock")
            try:
                while True:
                    if self.cancel_event.is_set() or (
                        cancel_event is not None and cancel_event.is_set()
                    ):
                        raise RemoteMaterializationCancelled(
                            "remote cache materialization cancelled before descriptor read"
                        )
                    self.check_worker_errors()
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        time.sleep(REMOTE_PREFETCH_POLL_SECONDS)
                data = _read_cache_body_bytes(
                    path,
                    entry=entry,
                    verify_content=not trust_existing_cache,
                )
            finally:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)
            self.check_worker_errors()
        except BaseException as primary:
            if demand_generation is not None:
                try:
                    self._remove_satisfied_demand(
                        binding=binding,
                        inventory_index=inventory_index,
                        generation=demand_generation,
                    )
                except BaseException as cleanup_error:
                    primary.add_note(
                        "also failed to remove requester demand: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            raise
        if demand_generation is not None:
            self._remove_satisfied_demand(
                binding=binding,
                inventory_index=inventory_index,
                generation=demand_generation,
            )
        return data

    def stop_worker(self) -> None:
        if self._worker_stopped:
            return
        self._stopping = True
        self.cancel_event.set()
        try:
            self.command_pipe.send({"op": "cancel"})
        except (BrokenPipeError, EOFError, OSError):
            pass
        self.process.join(timeout=REMOTE_PREFETCH_GRACEFUL_JOIN_SECONDS)
        hard_terminated = False
        if self.process.is_alive():
            logger.warning(
                "remote_training.prefetch_worker_hard_terminate "
                f"rank={self.rank} pid={self.process.pid}"
            )
            self.process.terminate()
            self.process.join(timeout=REMOTE_PREFETCH_HARD_JOIN_SECONDS)
            hard_terminated = not self.process.is_alive()
        if self.process.is_alive():
            raise RemoteTrainingError(
                "remote body prefetch worker survived bounded hard termination: "
                f"rank={self.rank}, pid={self.process.pid}"
            )
        if hard_terminated:
            _atomic_record(
                {
                    "schema_version": REMOTE_PREFETCH_SCHEMA_VERSION,
                    "nonce": self.nonce,
                    "rank": self.rank,
                    "pid": self.process.pid or -1,
                    "state": "stopped",
                    "family_id": None,
                    "inventory_index": None,
                    "body_name": None,
                    "updated_ns": time.time_ns(),
                    "error_type": "HardTerminated",
                    "error_message": ("parent hard-terminated a worker stuck after cancellation"),
                    "traceback": None,
                },
                _status_path(self.control_root, self.rank),
            )
        self._drain_status_pipe()
        self.command_pipe.close()
        self.status_pipe.close()
        self._worker_stopped = True
        if self._failure is not None:
            raise self._failure

    def worker_is_terminal(self) -> bool:
        """Return whether the owned process is dead and its pipes were closed."""
        return self._worker_stopped and not self.process.is_alive()

    def release_session(self) -> None:
        if self._session_released:
            return
        if not self._worker_stopped:
            raise RemoteTrainingError(
                "remote prefetch session lock cannot be released before worker shutdown"
            )
        if self.rank == 0 and self.session_lock_fd is not None:
            deadline = time.monotonic() + REMOTE_PREFETCH_GRACEFUL_JOIN_SECONDS
            terminal = False
            while time.monotonic() < deadline:
                terminal = True
                for owner_rank in range(self.world_size):
                    path = _status_path(self.control_root, owner_rank)
                    if not path.exists():
                        terminal = False
                        break
                    state = _load_status(
                        path,
                        nonce=self.nonce,
                        world_size=self.world_size,
                    )["state"]
                    if state not in {"stopped", "error"}:
                        terminal = False
                        break
                if terminal:
                    break
                time.sleep(REMOTE_PREFETCH_POLL_SECONDS)
            if not terminal:
                raise RemoteTrainingError(
                    "remote body cache session cannot be released while another "
                    "rank worker is still non-terminal"
                )
            fcntl.flock(self.session_lock_fd, fcntl.LOCK_UN)
            os.close(self.session_lock_fd)
            self.session_lock_fd = None
        self._session_released = True

    def close(self) -> None:
        primary: BaseException | None = None
        try:
            self.stop_worker()
        except BaseException as exc:
            primary = exc
        if self.worker_is_terminal():
            try:
                self.release_session()
            except BaseException as release_error:
                if primary is None:
                    raise
                primary.add_note(
                    "remote prefetch session release also failed: "
                    f"{type(release_error).__name__}: {release_error}"
                )
        if primary is not None:
            raise primary
