"""Reusable source-document index for autointerp token examples."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import queue
import random
import re
import shutil
import sqlite3
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor, TimeoutError
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch as t
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation
from loguru import logger

from nanochat.atomic_io import atomic_write_text
from steering.eval.autointerp_filter.path_safety import (
    paths_alias,
)
from steering.eval.autointerp_filter.strict_json import (
    canonical_json_equal,
    load_json_object,
)
from steering.eval.autointerp_filter.types import (
    SCHEMA_VERSION as _SCHEMA_VERSION,
)
from steering.eval.autointerp_filter.types import (
    AutoInterpCacheError,
    AutoInterpDataError,
    SourceDocRef,
)
from steering.feature_io import (
    MetadataKey,
    TensorKey,
    _delta_decode,
    spooled_feature_shard,
)
from steering.refined_types import (
    NonNegativeIntTuple,
    NonNegativeIntTupleLike,
    PositiveInt,
    PositiveIntLike,
)
from steering.segmentation import (
    SegmentMode,
    segment_mode_from_metadata,
)

if TYPE_CHECKING:
    _PathTupleLike = tuple[Path, ...]
    _ScanResultLike = tuple[list[tuple], list[SourceDocRef]]
else:
    # Beartype samples parameterized containers through Python's global RNG.
    # Refined tuple predicates below check every feature ID deterministically;
    # other collections retain their existing explicit structural checks.
    _PathTupleLike = tuple
    _ScanResultLike = tuple

# Bumped v1 -> v2 when the production build moved to the vectorized
# chunk-independent true-reduction scanner. v1 indices were built by the
# legacy chunk-dependent scanner; on real (well-formed) feature shards their
# rows are byte-identical to v2, but the method tag is pinned in the manifest
# so any pre-existing index is loudly rejected and rebuilt rather than
# silently reused across the semantics change (see docs Cache Identity).
SOURCE_INDEX_METHOD = "source_doc_index_v2"
_SOURCE_INDEX_REQUIRED_TENSOR_KEYS = frozenset(
    {
        str(TensorKey.OFFSETS),
        str(TensorKey.FEATURE_IDS),
        str(TensorKey.FEATURE_MAGS),
        str(TensorKey.DOC_IDS),
    }
)
_SOURCE_INDEX_ROW_OPTIONAL_TENSOR_KEYS = (
    TensorKey.SEGMENT_TOKEN_STARTS,
    TensorKey.SEGMENT_TOKEN_ENDS,
    TensorKey.SEGMENT_POSITIONS,
    TensorKey.SEGMENT_INDICES,
)
_VEC_MERGE_PENDING_MIN_ROWS = 1_000_000
_SOURCE_INDEX_PROGRESS_LOG_INTERVAL_S = 60.0


class SourceIndexProgressEventKind(StrEnum):
    """Source-index progress event names."""

    BUILD_START = "build_start"
    BUILD_DONE = "build_done"
    SHARD_START = "shard_start"
    SHARD_SPOOLED = "shard_spooled"
    CHUNK_DONE = "chunk_done"
    SHARD_DONE = "shard_done"


@dataclass(frozen=True)
class SourceIndexProgressEvent:
    """Structured progress event emitted by the source-index builder."""

    kind: SourceIndexProgressEventKind
    shard_index: int | None = None
    shard_count: int | None = None
    shard_name: str | None = None
    chunk_index: int | None = None
    chunk_count: int | None = None
    entries_done: int | None = None
    entries_total: int | None = None
    rows_done: int | None = None
    rows_total: int | None = None
    positive_rows: int | None = None
    bucket_features: int | None = None
    bucket_rows: int | None = None
    intruder_refs: int | None = None
    target_count: int | None = None
    chunk_seconds: float | None = None
    elapsed_seconds: float | None = None
    wait_seconds: float | None = None
    merge_seconds: float | None = None
    entries_per_second: float | None = None
    rss_kib: int | None = None
    shard_bytes: int | None = None
    spool_bytes: int | None = None
    sqlite_bytes: int | None = None


SourceIndexProgressCallback = Callable[[SourceIndexProgressEvent], None]
_WORKER_PROGRESS_POLL_INTERVAL_S = 0.25
_WORKER_PROGRESS_QUEUE: Any | None = None


def _init_scan_worker_progress_queue(progress_queue: Any | None) -> None:
    global _WORKER_PROGRESS_QUEUE
    _WORKER_PROGRESS_QUEUE = progress_queue


def _emit_worker_progress(event: SourceIndexProgressEvent) -> None:
    if _WORKER_PROGRESS_QUEUE is None:
        return
    try:
        _WORKER_PROGRESS_QUEUE.put(event)
    except Exception:
        # Progress is diagnostic; a broken parent-side queue must not turn an
        # otherwise valid shard scan into a data-path failure.
        return


@dataclass(frozen=True, init=False)
class SourceIndexRequest:
    feature_dir: Path
    data_dir: str | None
    index_path: Path
    manifest_path: Path
    shard_paths: tuple[Path, ...]
    parquet_paths: tuple[Path, ...]
    target_feature_ids: NonNegativeIntTuple
    n_sae_features: PositiveInt
    max_positive_refs_per_feature: PositiveInt
    max_intruder_refs: PositiveInt
    seed: int
    feature_fingerprint: str
    data_fingerprint: str
    context_independent_features: bool
    ignore_bos_features: bool
    chunk_size: PositiveInt
    # Scheduling knob (worker processes for the per-shard scan); default 1 =
    # serial. Deliberately NOT part of _manifest_payload / cache identity.
    scan_workers: PositiveInt = PositiveInt.parse(1)
    force_rebuild: bool = False
    shared_cache_dir: Path | None = None
    # Query-only projection of target_feature_ids. None preserves the legacy
    # behavior of loading every indexed target. This field must never enter the
    # manifest or shared-cache key: workers may read disjoint projections of
    # one immutable full-width index.
    load_target_feature_ids: NonNegativeIntTuple | None = None

    @beartype
    def __init__(
        self,
        feature_dir: Path,
        data_dir: str | None,
        index_path: Path,
        manifest_path: Path,
        shard_paths: _PathTupleLike,
        parquet_paths: _PathTupleLike,
        target_feature_ids: NonNegativeIntTupleLike,
        n_sae_features: PositiveIntLike,
        max_positive_refs_per_feature: PositiveIntLike,
        max_intruder_refs: PositiveIntLike,
        seed: int,
        feature_fingerprint: str,
        data_fingerprint: str,
        context_independent_features: bool,
        ignore_bos_features: bool,
        chunk_size: PositiveIntLike,
        scan_workers: PositiveIntLike = 1,
        force_rebuild: bool = False,
        shared_cache_dir: Path | None = None,
        load_target_feature_ids: NonNegativeIntTupleLike | None = None,
    ) -> None:
        object.__setattr__(self, "feature_dir", feature_dir)
        object.__setattr__(self, "data_dir", data_dir)
        object.__setattr__(self, "index_path", index_path)
        object.__setattr__(self, "manifest_path", manifest_path)
        object.__setattr__(self, "shard_paths", shard_paths)
        object.__setattr__(self, "parquet_paths", parquet_paths)
        object.__setattr__(
            self,
            "target_feature_ids",
            NonNegativeIntTuple.parse(target_feature_ids),
        )
        object.__setattr__(self, "n_sae_features", PositiveInt.parse(n_sae_features))
        object.__setattr__(
            self,
            "max_positive_refs_per_feature",
            PositiveInt.parse(max_positive_refs_per_feature),
        )
        object.__setattr__(self, "max_intruder_refs", PositiveInt.parse(max_intruder_refs))
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "feature_fingerprint", feature_fingerprint)
        object.__setattr__(self, "data_fingerprint", data_fingerprint)
        object.__setattr__(
            self,
            "context_independent_features",
            context_independent_features,
        )
        object.__setattr__(self, "ignore_bos_features", ignore_bos_features)
        object.__setattr__(self, "chunk_size", PositiveInt.parse(chunk_size))
        object.__setattr__(self, "scan_workers", PositiveInt.parse(scan_workers))
        object.__setattr__(self, "force_rebuild", force_rebuild)
        object.__setattr__(self, "shared_cache_dir", shared_cache_dir)
        object.__setattr__(
            self,
            "load_target_feature_ids",
            None
            if load_target_feature_ids is None
            else NonNegativeIntTuple.parse(load_target_feature_ids),
        )


@dataclass(frozen=True)
class SourceIndexData:
    positives_by_feature: dict[int, list[tuple[SourceDocRef, float]]]
    intruder_refs: list[SourceDocRef]
    manifest_sha256: str | None = None
    artifact_sha256: str | None = None


@dataclass(frozen=True)
class _SpooledSourceIndexShardData:
    offsets: t.Tensor
    n_entries: int
    doc_ids: t.Tensor
    segment_token_starts: t.Tensor
    segment_token_ends: t.Tensor
    stem: str
    shard_index: int
    intruders: list[SourceDocRef]


@dataclass(frozen=True)
class _DecodedEntryChunk:
    start: int
    end: int
    feature_ids: t.Tensor
    magnitudes: t.Tensor


@dataclass(frozen=True)
class _PositiveRow:
    feature_id: int
    magnitude: float
    stable_key: str
    doc_ref: SourceDocRef


def _current_rss_kib() -> int | None:
    status_path = Path(f"/proc/{os.getpid()}/status")
    try:
        for line in status_path.read_text().splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1])
    except OSError:
        return None
    return None


def _path_size(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        return path.stat().st_size
    except OSError:
        return None


def _assert_unique_shard_stems(shard_paths: tuple[Path, ...]) -> None:
    """Fail loud if two shard paths share a stem. Cross-shard intruder keys
    embed the stem, so unique stems guarantee cross-shard keys never collide —
    the invariant that makes the per-shard dedup set byte-equivalent to a
    global one."""
    seen: set[str] = set()
    for shard_path in shard_paths:
        stem = shard_path.stem.removesuffix(".safetensors")
        if stem in seen:
            raise AutoInterpDataError(
                f"duplicate feature shard stem {stem!r} in source-index shard set; "
                "stems must be unique so cross-shard stable keys cannot collide"
            )
        seen.add(stem)


class _TopKMergeState:
    """Streaming, bounded-memory, torch-vectorized per-feature top-k merge.
    Survivors are held as a Python list of the scanner's 8-field payload tuples
    (numpy-free); each shard combines survivors + the shard's rows and reduces to
    the per-feature top-k by reusing the same candidate reducer as the
    per-shard scanner. The reducer deduplicates ``(feature_id, stable_key)`` by
    max magnitude, then orders by ``(magnitude DESC, stable_key ASC)``.

    Magnitudes are sorted via a float32 tensor (matching scanner state) but are
    STORED from the original payload tuple, NOT the float32 tensor -- so both the
    ordering and the stored magnitude bytes match the reference implementation.
    Parent survivors are deterministically bounded by
    ``target_count * max_positive_refs_per_feature`` rows."""

    def __init__(self, max_positive_refs: int) -> None:
        self._k = max_positive_refs
        self._survivors: list[tuple] = []

    def add_shard(self, positive_rows: list[tuple]) -> int:
        combined = self._survivors + positive_rows
        if not combined:
            self._survivors = []
            return 0
        fids = t.tensor([row[0] for row in combined], dtype=t.int32)
        mags = t.tensor([row[1] for row in combined], dtype=t.float32)
        ranks = _rank_row_stable_keys([row[2] for row in combined])
        rows = t.arange(len(combined), dtype=t.int32)
        _, _, _, selected_rows = _reduce_source_index_candidates(
            fids,
            ranks,
            mags,
            rows,
            max_positive_refs=self._k,
        )
        self._survivors = [combined[i] for i in selected_rows.tolist()]
        return len(self._survivors)

    def materialize(self) -> list[tuple]:
        if not self._survivors:
            return []
        rows: list[tuple] = []
        rank_in_feat = 0
        prev_fid: int | None = None
        for row in self._survivors:
            if row[0] != prev_fid:
                rank_in_feat, prev_fid = 0, row[0]
            rows.append(
                (
                    row[0],
                    rank_in_feat,
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    row[6],
                    row[7],
                )
            )
            rank_in_feat += 1
        return rows


def _bulk_insert_positive_refs(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    """One-shot write of the already-ranked positive_refs rows (replaces
    ``_materialize_positive_refs``'s SELECT-from-positive_work). No per-row
    re-validation — the scanner already validated."""
    if not rows:
        return
    conn.executemany(
        "INSERT INTO positive_refs (feature_id, rank, magnitude, stable_key, "
        "shard_stem, shard_index, doc_id, segment_token_start, "
        "segment_token_end) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


def _percent(done: int | None, total: int | None) -> str:
    if done is None or total is None or total <= 0:
        return "n/a"
    return f"{(100.0 * done / total):.1f}%"


def _rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M/s"
    if value >= 1_000:
        return f"{value / 1_000:.2f}k/s"
    return f"{value:.1f}/s"


def _bytes_gib(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / 1024**3:.2f}GiB"


def _rss_mib(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / 1024:.1f}MiB"


def _seconds_text(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}s"


def _shard_label(event: SourceIndexProgressEvent) -> str:
    if event.shard_index is None:
        return "n/a"
    if event.shard_count is None:
        return str(event.shard_index + 1)
    return f"{event.shard_index + 1}/{event.shard_count}"


class _SourceIndexProgressLogger:
    def __init__(self, *, chunk_interval_s: float) -> None:
        self._chunk_interval_s = chunk_interval_s
        self._last_chunk_log_monotonic_by_shard: dict[int, float] = {}

    def __call__(self, event: SourceIndexProgressEvent) -> None:
        if event.kind == SourceIndexProgressEventKind.BUILD_START:
            logger.info(
                "Source index build start: "
                f"shards={event.shard_count} targets={event.target_count} "
                f"rss={_rss_mib(event.rss_kib)}"
            )
            return
        if event.kind == SourceIndexProgressEventKind.BUILD_DONE:
            logger.info(
                "Source index build done: "
                f"elapsed={event.elapsed_seconds:.1f}s "
                f"sqlite={_bytes_gib(event.sqlite_bytes)} "
                f"rss={_rss_mib(event.rss_kib)}"
            )
            return
        if event.kind == SourceIndexProgressEventKind.SHARD_START:
            logger.info(
                f"Source index shard {_shard_label(event)} "
                f"start: {event.shard_name} "
                f"compressed={_bytes_gib(event.shard_bytes)} "
                f"rss={_rss_mib(event.rss_kib)}"
            )
            return
        if event.kind == SourceIndexProgressEventKind.SHARD_SPOOLED:
            logger.info(
                f"Source index shard {_shard_label(event)} "
                f"spooled: entries={event.entries_total} rows={event.rows_total} "
                f"spool={_bytes_gib(event.spool_bytes)} "
                f"elapsed={event.elapsed_seconds:.1f}s "
                f"rss={_rss_mib(event.rss_kib)}"
            )
            return
        if event.kind == SourceIndexProgressEventKind.CHUNK_DONE:
            self._log_chunk_event(event)
            return
        if event.kind == SourceIndexProgressEventKind.SHARD_DONE:
            logger.info(
                f"Source index shard {_shard_label(event)} "
                f"done: positive_rows={event.positive_rows} "
                f"live_positive_rows={event.rows_done} "
                f"intruder_refs={event.intruder_refs} "
                f"sqlite={_bytes_gib(event.sqlite_bytes)} "
                f"elapsed={event.elapsed_seconds:.1f}s "
                f"wait={_seconds_text(event.wait_seconds)} "
                f"merge={_seconds_text(event.merge_seconds)} "
                f"rss={_rss_mib(event.rss_kib)}"
            )

    def _log_chunk_event(self, event: SourceIndexProgressEvent) -> None:
        if event.shard_index is None:
            return
        now = time.monotonic()
        last = self._last_chunk_log_monotonic_by_shard.get(event.shard_index)
        is_first = event.chunk_index == 1
        is_last = (
            event.chunk_index is not None
            and event.chunk_count is not None
            and event.chunk_index == event.chunk_count
        )
        if (
            not is_first
            and not is_last
            and last is not None
            and now - last < self._chunk_interval_s
        ):
            return
        self._last_chunk_log_monotonic_by_shard[event.shard_index] = now
        logger.info(
            f"Source index shard {_shard_label(event)} "
            f"chunk {event.chunk_index}/{event.chunk_count}: "
            f"entries={event.entries_done}/{event.entries_total} "
            f"({_percent(event.entries_done, event.entries_total)}) "
            f"rate={_rate(event.entries_per_second)} "
            f"chunk_s={event.chunk_seconds:.2f} "
            f"rows_emitted={event.positive_rows} "
            f"bucket_features={event.bucket_features} "
            f"bucket_rows={event.bucket_rows} "
            f"elapsed={event.elapsed_seconds:.1f}s "
            f"rss={_rss_mib(event.rss_kib)}"
        )


def _emit_progress(
    progress_callback: SourceIndexProgressCallback | None,
    event: SourceIndexProgressEvent,
) -> None:
    if progress_callback is not None:
        progress_callback(event)


def fingerprint_paths(paths: list[Path] | tuple[Path, ...]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths):
        stat = path.stat()
        payload = {
            "name": path.name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        h.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def load_or_build_source_index(
    request: SourceIndexRequest,
    *,
    progress_callback: SourceIndexProgressCallback | None = None,
) -> SourceIndexData:
    """Load one source index, building it if the cache is missing or incomplete."""
    entry = ensure_source_index_cache(request, progress_callback=progress_callback)
    manifest = _manifest_payload(entry)
    _validate_published_source_index(entry, manifest)
    return _load_index(entry)


def ensure_source_index_cache(
    request: SourceIndexRequest,
    *,
    progress_callback: SourceIndexProgressCallback | None = None,
) -> SourceIndexRequest:
    """Build or validate a complete source-index pair without loading its rows.

    This is the prepare-time operation for a single-process autointerp run.  It
    deliberately keeps the full index out of Python memory: the read step later
    loads only its assigned feature projection through
    :func:`load_or_build_source_index`.  Atomic publication (temp file plus
    ``os.replace`` inside :func:`_build_index`) is what protects a concurrent
    reader from a half-written index; no cross-process lock is taken.
    """
    entry, manifest = _source_index_entry_and_manifest(request)
    if entry.force_rebuild:
        entry.index_path.unlink(missing_ok=True)
        entry.manifest_path.unlink(missing_ok=True)
    elif _source_index_pair_exists(entry):
        _validate_published_source_index(entry, manifest)
        return entry
    elif entry.index_path.exists() or entry.manifest_path.exists():
        _validate_existing_manifest(entry, manifest)
        # _validate_existing_manifest always raises for a partial pair;
        # retain an explicit fail-closed fallback if that contract changes.
        raise AutoInterpCacheError(
            "source index requires both SQLite and manifest files before a build can proceed"
        )

    progress_logger = _SourceIndexProgressLogger(
        chunk_interval_s=_SOURCE_INDEX_PROGRESS_LOG_INTERVAL_S,
    )

    def _emit_both(event: SourceIndexProgressEvent) -> None:
        progress_logger(event)
        _emit_progress(progress_callback, event)

    _build_index(entry, manifest, progress_callback=_emit_both)
    _validate_published_source_index(entry, manifest)
    return entry


def validate_source_index_cache(request: SourceIndexRequest) -> SourceIndexRequest:
    """Validate an existing source-index cache entry without loading all rows."""
    entry, manifest = _source_index_entry_and_manifest(request)
    if entry.force_rebuild:
        return entry
    if entry.index_path.exists() or entry.manifest_path.exists():
        _validate_published_source_index(entry, manifest)
    return entry


def _source_index_entry_and_manifest(
    request: SourceIndexRequest,
) -> tuple[SourceIndexRequest, dict]:
    _validate_request(request)
    manifest = _manifest_payload(request)
    entry = _resolve_shared_cache_request(request, manifest)
    return _canonical_source_index_entry_request(entry), manifest


def _source_index_pair_exists(request: SourceIndexRequest) -> bool:
    return request.index_path.exists() and request.manifest_path.exists()


def _validate_published_source_index(request: SourceIndexRequest, manifest: dict) -> None:
    _validate_existing_manifest(request, manifest)
    _reject_source_index_sqlite_sidecars(request.index_path)
    _validate_published_source_index_header(request.index_path)
    conn = _connect_read_only(request.index_path)
    try:
        _validate_index_metadata(conn, index_path=request.index_path)
    finally:
        conn.close()


def _canonical_source_index_entry_request(
    request: SourceIndexRequest,
) -> SourceIndexRequest:
    """Resolve storage aliases without changing manifest/cache identity."""
    return replace(
        request,
        index_path=request.index_path.resolve(),
        manifest_path=request.manifest_path.resolve(),
    )


def _source_index_manifest_cache_key(manifest: dict) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_shared_cache_request(
    request: SourceIndexRequest, manifest: dict
) -> SourceIndexRequest:
    if request.shared_cache_dir is None:
        return request
    key = _source_index_manifest_cache_key(manifest)
    cache_dir = request.shared_cache_dir
    return replace(
        request,
        index_path=cache_dir / f"source_doc_index_{key}.sqlite",
        manifest_path=cache_dir / f"source_doc_index_{key}.manifest.json",
    )


def _validate_request(request: SourceIndexRequest) -> None:
    try:
        source_paths_alias = paths_alias(request.index_path, request.manifest_path)
    except OSError as exc:
        raise AutoInterpDataError(
            f"cannot verify source index SQLite/manifest path identity: {exc}"
        ) from exc
    if source_paths_alias:
        raise AutoInterpDataError(
            "source index SQLite and manifest paths must resolve to distinct files "
            "and must not hard-link one inode: "
            f"{request.index_path} and {request.manifest_path}"
        )
    if not request.ignore_bos_features:
        raise AutoInterpDataError(
            "autointerp source index requires feature shards extracted with "
            "ignore_bos_features=true"
        )
    _validate_target_feature_ids(
        request.target_feature_ids,
        n_sae_features=request.n_sae_features,
    )
    _resolved_load_target_feature_ids(request)


def _resolved_load_target_feature_ids(
    request: SourceIndexRequest,
) -> tuple[int, ...]:
    target_feature_ids = request.load_target_feature_ids
    if target_feature_ids is None:
        return request.target_feature_ids
    if len(set(target_feature_ids)) != len(target_feature_ids):
        raise AutoInterpDataError(
            f"source index load target feature IDs must be unique, got {target_feature_ids!r}"
        )
    if tuple(sorted(target_feature_ids)) != target_feature_ids:
        raise AutoInterpDataError(
            "source index load target feature IDs must be sorted ascending, got "
            f"{target_feature_ids!r}"
        )
    _validate_target_feature_ids(
        target_feature_ids,
        n_sae_features=request.n_sae_features,
    )
    indexed = set(request.target_feature_ids)
    missing = [feature_id for feature_id in target_feature_ids if feature_id not in indexed]
    if missing:
        raise AutoInterpDataError(
            "source index load target feature IDs must be a subset of indexed "
            f"target_feature_ids; missing={missing}"
        )
    return target_feature_ids


@beartype
def _validate_target_feature_ids(
    target_feature_ids: NonNegativeIntTupleLike,
    *,
    n_sae_features: PositiveIntLike,
) -> None:
    bad = sorted(
        {
            int(feature_id)
            for feature_id in target_feature_ids
            if int(feature_id) < 0 or int(feature_id) >= n_sae_features
        }
    )
    if bad:
        raise AutoInterpDataError(
            "source index target feature IDs out of range for "
            f"n_sae_features={n_sae_features}: {bad}"
        )


def _target_feature_mask(
    target_feature_ids: tuple[int, ...],
    *,
    n_sae_features: int,
) -> t.Tensor:
    allowed = t.zeros(n_sae_features, dtype=t.bool)
    if target_feature_ids:
        allowed[t.tensor(target_feature_ids, dtype=t.int64)] = True
    return allowed


def _rank_row_stable_keys(row_keys: list[str]) -> t.Tensor:
    """Dense lexicographic ranks of row stable-keys.

    Ranks are assigned in Python ``str`` order (code-point lexicographic,
    shorter-prefix-first), so ordering by ``(magnitude desc, rank asc)`` is
    exactly ordering by ``(magnitude desc, stable_key asc)``. Duplicate keys
    share a rank, which preserves per-(feature, stable_key) dedup group
    identity. Returns dense ranks as ``torch.int32`` (rank < n_rows).
    """
    if len(row_keys) > t.iinfo(t.int32).max:
        raise AutoInterpDataError("source-index row count exceeds int32 rank space")
    if not row_keys:
        return t.zeros(0, dtype=t.int32)
    ranking = {key: rank for rank, key in enumerate(sorted(set(row_keys)))}
    return t.tensor([ranking[key] for key in row_keys], dtype=t.int32)


def _worst_magnitudes_from_state(
    state_fids: t.Tensor,
    state_mags: t.Tensor,
    *,
    n_sae_features: int,
    max_positive_refs: int,
) -> t.Tensor:
    """Per-feature k-th best magnitude of a reduced state, else ``-inf``.

    Requires the state invariant: grouped by ``(fid asc, mag desc,
    rank asc)`` with at most ``max_positive_refs`` rows per feature, so a
    full group's last row is its k-th best magnitude.
    """
    worst = t.full((n_sae_features,), float("-inf"), dtype=t.float32)
    n_state = int(state_fids.shape[0])
    if n_state == 0:
        return worst
    boundary = t.ones(n_state, dtype=t.bool)
    boundary[1:] = state_fids[1:] != state_fids[:-1]
    group_start = boundary.nonzero(as_tuple=False).flatten()
    group_end = t.cat([group_start[1:], t.tensor([n_state], dtype=t.int64)])
    full = (group_end - group_start) >= max_positive_refs
    worst[state_fids[group_start[full]].to(t.int64)] = state_mags[group_end[full] - 1]
    return worst


def _torch_lexsort(keys: tuple[t.Tensor, ...]) -> t.Tensor:
    """Stable multi-key sort; keys ordered least-significant first (the
    last key is primary), returning the permutation that sorts them."""
    order = t.arange(int(keys[0].numel()), dtype=t.int64)
    for key in keys:
        order = order[key[order].argsort(stable=True)]
    return order


def _reduce_source_index_candidates(
    fids: t.Tensor,
    ranks: t.Tensor,
    mags: t.Tensor,
    rows: t.Tensor,
    *,
    max_positive_refs: int,
) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    """Exact dedup + per-feature top-k reduction.

    Dedup keeps the max magnitude per ``(fid, rank)``; top-k keeps the
    first ``max_positive_refs`` per feature under ``(mag desc, rank asc)``.
    Output is globally ordered ``(fid asc, mag desc, rank asc)`` — the
    state invariant consumed by ``_worst_magnitudes_from_state`` and the
    finalizer.
    """
    if int(fids.shape[0]) == 0:
        return fids, ranks, mags, rows

    order = _torch_lexsort((-mags, ranks, fids))
    tf, tr, tm, tw = fids[order], ranks[order], mags[order], rows[order]
    first = t.ones(int(tf.numel()), dtype=t.bool)
    first[1:] = (tf[1:] != tf[:-1]) | (tr[1:] != tr[:-1])
    tf, tr, tm, tw = tf[first], tr[first], tm[first], tw[first]

    order = _torch_lexsort((tr, -tm, tf))
    tf, tr, tm, tw = tf[order], tr[order], tm[order], tw[order]
    boundary = t.ones(int(tf.numel()), dtype=t.bool)
    boundary[1:] = tf[1:] != tf[:-1]
    group_start = boundary.nonzero(as_tuple=False).flatten()
    group_id = boundary.to(t.int64).cumsum(0) - 1
    within = t.arange(int(tf.numel()), dtype=t.int64) - group_start[group_id]
    keep = within < max_positive_refs
    return tf[keep], tr[keep], tm[keep], tw[keep]


def _validated_source_doc_ref(
    *,
    shard_stem: str,
    shard_index: int,
    doc_id: int,
    segment_token_start: int,
    segment_token_end: int,
    stable_key: str | None = None,
    context: str,
    error_cls: type[Exception] = AutoInterpDataError,
) -> SourceDocRef:
    try:
        ref = SourceDocRef(
            shard_stem=str(shard_stem),
            shard_index=int(shard_index),
            doc_id=int(doc_id),
            segment_token_start=int(segment_token_start),
            segment_token_end=int(segment_token_end),
        )
    except AutoInterpDataError as exc:
        raise error_cls(
            f"{context} has invalid token bounds: {segment_token_start}:{segment_token_end}"
        ) from exc
    except (
        BeartypeCallHintParamViolation,
        OverflowError,
        TypeError,
        ValueError,
    ) as exc:
        raise error_cls(f"{context} has invalid source coordinates: {exc}") from exc
    if stable_key is not None and stable_key != ref.stable_key():
        raise error_cls(
            f"{context} stable_key mismatch: stored={stable_key!r}, expected={ref.stable_key()!r}"
        )
    return ref


def _positive_row_from_tuple(
    row: tuple,
    *,
    context: str,
    error_cls: type[Exception] = AutoInterpDataError,
) -> _PositiveRow:
    if len(row) != 8:
        raise error_cls(f"{context} expected 8 columns, got {len(row)}")
    fid = int(row[0])
    mag = float(row[1])
    if fid < 0:
        raise error_cls(f"{context} has negative feature_id={fid}")
    if not math.isfinite(mag):
        raise error_cls(f"{context} has non-finite magnitude={mag}")
    ref = _validated_source_doc_ref(
        shard_stem=str(row[3]),
        shard_index=int(row[4]),
        doc_id=int(row[5]),
        segment_token_start=int(row[6]),
        segment_token_end=int(row[7]),
        stable_key=str(row[2]),
        context=context,
        error_cls=error_cls,
    )
    return _PositiveRow(
        feature_id=fid,
        magnitude=mag,
        stable_key=str(row[2]),
        doc_ref=ref,
    )


def _intruder_row_tuple(rank: int, ref: SourceDocRef) -> tuple:
    validated = _validated_source_doc_ref(
        shard_stem=ref.shard_stem,
        shard_index=ref.shard_index,
        doc_id=ref.doc_id,
        segment_token_start=ref.segment_token_start,
        segment_token_end=ref.segment_token_end,
        context=f"source-index intruder ref rank {rank}",
    )
    return (
        rank,
        validated.stable_key(),
        validated.shard_stem,
        validated.shard_index,
        validated.doc_id,
        validated.segment_token_start,
        validated.segment_token_end,
    )


def _manifest_payload(request: SourceIndexRequest) -> dict:
    return {
        "schema_version": _SCHEMA_VERSION,
        "method": SOURCE_INDEX_METHOD,
        "feature_dir": str(request.feature_dir),
        "data_dir": request.data_dir,
        "feature_fingerprint": request.feature_fingerprint,
        "data_fingerprint": request.data_fingerprint,
        "target_feature_ids": list(request.target_feature_ids),
        "n_sae_features": request.n_sae_features,
        "shard_domain": [path.name for path in request.shard_paths],
        "n_feature_shards": len(request.shard_paths),
        "n_data_shards": len(request.parquet_paths),
        "max_positive_refs_per_feature": request.max_positive_refs_per_feature,
        "max_intruder_refs": request.max_intruder_refs,
        "seed": request.seed,
        "context_independent_features": request.context_independent_features,
        "ignore_bos_features": request.ignore_bos_features,
        "chunk_size": request.chunk_size,
    }


def _validate_source_manifest_mapping(
    existing: dict[str, object],
    manifest: dict,
    *,
    path: Path,
) -> None:
    if canonical_json_equal(existing, manifest):
        return
    existing_method = existing.get("method")
    existing_schema = existing.get("schema_version")
    if existing_method != manifest["method"] or existing_schema != manifest["schema_version"]:
        raise AutoInterpCacheError(
            f"Existing autointerp source index at {path} was built "
            f"by an incompatible generation (method={existing_method!r}, "
            f"schema_version={existing_schema!r}; current "
            f"method={manifest['method']!r}, schema_version={manifest['schema_version']!r}). "
            "Delete it or set source_index.force_rebuild=true to rebuild with the "
            "current scanner."
        )
    raise AutoInterpCacheError(
        f"Existing autointerp source index manifest at {path} "
        "does not match the current feature/data/target configuration"
    )


def _validate_existing_manifest(request: SourceIndexRequest, manifest: dict) -> None:
    if not request.index_path.is_file() or not request.manifest_path.is_file():
        raise AutoInterpCacheError(
            "source index requires both SQLite and manifest files; found "
            f"index={request.index_path.is_file()} manifest={request.manifest_path.is_file()}"
        )
    existing = load_json_object(request.manifest_path, context="source index manifest")
    _validate_source_manifest_mapping(
        existing,
        manifest,
        path=request.manifest_path,
    )


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def _connect_read_only(path: Path) -> sqlite3.Connection:
    """Open an already-validated immutable index without SQLite side effects."""
    _reject_source_index_sqlite_sidecars(path)
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True)


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute(
        "INSERT INTO metadata (key, value) VALUES ('schema_version', ?)",
        (str(_SCHEMA_VERSION),),
    )
    conn.execute(
        "CREATE TABLE positive_refs ("
        " feature_id INTEGER NOT NULL,"
        " rank INTEGER NOT NULL,"
        " magnitude REAL NOT NULL,"
        " stable_key TEXT NOT NULL,"
        " shard_stem TEXT NOT NULL,"
        " shard_index INTEGER NOT NULL,"
        " doc_id INTEGER NOT NULL,"
        " segment_token_start INTEGER NOT NULL,"
        " segment_token_end INTEGER NOT NULL,"
        " PRIMARY KEY (feature_id, rank)"
        ")"
    )
    conn.execute(
        "CREATE TABLE intruder_refs ("
        " rank INTEGER PRIMARY KEY,"
        " stable_key TEXT NOT NULL,"
        " shard_stem TEXT NOT NULL,"
        " shard_index INTEGER NOT NULL,"
        " doc_id INTEGER NOT NULL,"
        " segment_token_start INTEGER NOT NULL,"
        " segment_token_end INTEGER NOT NULL"
        ")"
    )
    conn.execute("CREATE INDEX positive_refs_feature_idx ON positive_refs(feature_id)")
    conn.commit()


@dataclass(frozen=True)
class _ScanTask:
    """Picklable per-shard scan request for a worker process (spawn-safe: no
    closures, no callbacks, only primitives + Path)."""

    shard_path: Path
    target_ids: tuple[int, ...]
    n_sae_features: int
    max_positive_refs: int
    chunk_size: int
    temp_dir: Path
    shard_index: int
    shard_count: int


@dataclass(frozen=True)
class _ScanResult:
    """Picklable per-shard scan result. Intruders are stored as parallel lists
    of primitive ints (all share one stem/shard_index within a shard) so IPC
    pickles ints, not hundreds of thousands of SourceDocRef objects; the parent
    reconstructs them in order, preserving the exact ``stable_key()`` sequence.
    The parent holds at most ``scan_workers + 2`` of these at once (the
    submission window), so peak held-result memory is O(scan_workers)."""

    positive_rows: list[tuple]
    intr_stem: str
    intr_shard_index: int
    intr_doc_ids: list[int]
    intr_tok_starts: list[int]
    intr_tok_ends: list[int]


def _scan_task(
    request: SourceIndexRequest, shard_index: int, target_ids: tuple[int, ...]
) -> _ScanTask:
    return _ScanTask(
        shard_path=request.shard_paths[shard_index],
        target_ids=target_ids,
        n_sae_features=request.n_sae_features,
        max_positive_refs=request.max_positive_refs_per_feature,
        chunk_size=request.chunk_size,
        temp_dir=request.index_path.parent,
        shard_index=shard_index,
        shard_count=len(request.shard_paths),
    )


def _scan_shard_worker(task: _ScanTask) -> _ScanResult:
    """Worker entry: scan one shard with the real production scanner and pack
    the result. Module-level and picklable so it runs under a ``spawn`` context.
    Calls ``_scan_source_index_shard_file_topk_vectorized`` directly so serial
    and spawned builds share the same scanner."""
    positive_rows, intruders = _scan_source_index_shard_file_topk_vectorized(
        shard_path=task.shard_path,
        target_feature_ids=task.target_ids,
        n_sae_features=task.n_sae_features,
        max_positive_refs=task.max_positive_refs,
        chunk_size=task.chunk_size,
        temp_dir=task.temp_dir,
        shard_index=task.shard_index,
        shard_count=task.shard_count,
        progress_callback=(_emit_worker_progress if _WORKER_PROGRESS_QUEUE is not None else None),
    )
    stem = intruders[0].shard_stem if intruders else ""
    shard_index = intruders[0].shard_index if intruders else task.shard_index
    return _ScanResult(
        positive_rows=positive_rows,
        intr_stem=stem,
        intr_shard_index=shard_index,
        intr_doc_ids=[ref.doc_id for ref in intruders],
        intr_tok_starts=[ref.segment_token_start for ref in intruders],
        intr_tok_ends=[ref.segment_token_end for ref in intruders],
    )


def _unpack_intruders(result: _ScanResult) -> list[SourceDocRef]:
    """Reconstruct the shard's intruder refs in exact order from flat arrays."""
    return [
        SourceDocRef(
            shard_stem=result.intr_stem,
            shard_index=result.intr_shard_index,
            doc_id=doc_id,
            segment_token_start=tok_start,
            segment_token_end=tok_end,
        )
        for doc_id, tok_start, tok_end in zip(
            result.intr_doc_ids,
            result.intr_tok_starts,
            result.intr_tok_ends,
            strict=True,
        )
    ]


def _merge_shard_into_index(
    state: _TopKMergeState,
    *,
    max_intruder_refs: int,
    seed: int,
    positive_rows: list[tuple],
    shard_intruders: list[SourceDocRef],
    intruder_refs: list[SourceDocRef],
    intruder_seen: int,
) -> tuple[int, int]:
    """Merge one shard's scan results into the index: vectorized per-feature
    top-k (``state``) for positives, then the intruder-reservoir walk. Shared by
    the serial and the parallel (`scan_workers > 1`) paths so both take an
    identical merge, and called by the parent strictly in shard-index order so
    output is order-deterministic. ``intruder_refs`` is mutated in place; the
    running global ``intruder_seen`` counter is returned, along with the live
    positive-row count held by ``state`` after this shard. The intruder dedup
    set is per-shard (#631): stable keys lead with a per-shard-unique stem, so
    resetting it each shard is byte-equivalent to a global set.
    """
    live_positive_rows = state.add_shard(positive_rows)
    intruder_keys: set[str] = set()
    for doc_ref in shard_intruders:
        key = doc_ref.stable_key()
        if key in intruder_keys:
            continue
        intruder_keys.add(key)
        intruder_seen += 1
        if len(intruder_refs) < max_intruder_refs:
            intruder_refs.append(doc_ref)
            continue
        replacement = random.Random(f"{seed}:intruder:{intruder_seen}").randrange(intruder_seen)
        if replacement < max_intruder_refs:
            intruder_refs[replacement] = doc_ref
    return intruder_seen, live_positive_rows


def _emit_shard_start(
    progress_callback: SourceIndexProgressCallback | None,
    request: SourceIndexRequest,
    shard_idx: int,
    shard_path: Path,
) -> None:
    _emit_progress(
        progress_callback,
        SourceIndexProgressEvent(
            kind=SourceIndexProgressEventKind.SHARD_START,
            shard_index=shard_idx,
            shard_count=len(request.shard_paths),
            shard_name=shard_path.name,
            shard_bytes=_path_size(shard_path),
            rss_kib=_current_rss_kib(),
        ),
    )


def _emit_shard_done(
    progress_callback: SourceIndexProgressCallback | None,
    request: SourceIndexRequest,
    tmp_path: Path,
    shard_idx: int,
    shard_path: Path,
    positive_rows: list[tuple],
    live_positive_rows: int,
    intruder_refs: list[SourceDocRef],
    shard_start: float,
    wait_seconds: float | None = None,
    merge_seconds: float | None = None,
) -> None:
    _emit_progress(
        progress_callback,
        SourceIndexProgressEvent(
            kind=SourceIndexProgressEventKind.SHARD_DONE,
            shard_index=shard_idx,
            shard_count=len(request.shard_paths),
            shard_name=shard_path.name,
            positive_rows=len(positive_rows),
            rows_done=live_positive_rows,
            intruder_refs=len(intruder_refs),
            elapsed_seconds=time.perf_counter() - shard_start,
            wait_seconds=wait_seconds,
            merge_seconds=merge_seconds,
            sqlite_bytes=_path_size(tmp_path),
            rss_kib=_current_rss_kib(),
        ),
    )


# zstd feature shard -> decompressed safetensors spool; observed expansion is
# ~1.75x (a 1.93 GB shard -> ~3.4 GB spool), rounded up to 2x for headroom.
_SPOOL_DECOMPRESSION_FACTOR = 2


def _assert_scan_worker_disk_budget(request: SourceIndexRequest) -> None:
    """Fail loud BEFORE spawning workers if the temp dir cannot hold the peak
    number of concurrent decompressed spools. Only ``min(scan_workers,
    n_shards)`` workers ever run at once (there are only ``n_shards`` scan
    tasks), so more workers than shards does not need more disk. Estimated
    conservatively from the largest compressed shard."""
    temp_dir = request.index_path.parent
    max_compressed = max((shard.stat().st_size for shard in request.shard_paths), default=0)
    est_spool = max_compressed * _SPOOL_DECOMPRESSION_FACTOR
    concurrent = min(request.scan_workers, len(request.shard_paths))
    required = int(concurrent * est_spool * 1.1)
    free = shutil.disk_usage(temp_dir).free
    if free < required:
        raise AutoInterpDataError(
            f"insufficient free disk for source-index scan_workers="
            f"{request.scan_workers}: need ~{required / 1e9:.1f} GB "
            f"({concurrent} concurrent ~{est_spool / 1e9:.1f} GB spools "
            f"+ 10% margin) in {temp_dir}, but only {free / 1e9:.1f} GB free"
        )


def _parallel_scan_and_merge(
    request: SourceIndexRequest,
    tmp_path: Path,
    target_ids: tuple[int, ...],
    intruder_refs: list[SourceDocRef],
    state: _TopKMergeState,
    progress_callback: SourceIndexProgressCallback | None,
) -> int:
    """Scan shards in ``request.scan_workers`` spawned worker processes and merge
    results in the parent STRICTLY in shard-index order (rendezvous), so output
    is byte-identical to the serial path regardless of completion order. A
    bounded submission window (``scan_workers + 2``) caps held-result memory so a
    straggling shard cannot make workers scan far ahead into parent RAM. Any
    worker error propagates from ``.result()`` and aborts the build (the caller's
    ``except`` cleans up); on abort queued shards are cancelled and in-flight
    scans are not awaited, so the failure surfaces promptly.

    Worker-side SHARD_SPOOLED / CHUNK_DONE events are streamed
    back to the parent on a diagnostic queue. The parent still emits SHARD_DONE
    only after the strict in-order merge, preserving byte-identical output while
    making worker scan progress visible during long rendezvous waits.
    """
    _assert_scan_worker_disk_budget(request)
    n = len(request.shard_paths)
    window = request.scan_workers + 2
    ctx = multiprocessing.get_context("spawn")
    intruder_seen = 0
    progress_queue = ctx.Queue() if progress_callback is not None else None
    executor = ProcessPoolExecutor(
        max_workers=request.scan_workers,
        mp_context=ctx,
        initializer=_init_scan_worker_progress_queue,
        initargs=(progress_queue,),
    )
    try:
        futures: dict[int, Future[_ScanResult]] = {}
        next_submit = 0
        while next_submit < min(window, n):
            futures[next_submit] = executor.submit(
                _scan_shard_worker, _scan_task(request, next_submit, target_ids)
            )
            next_submit += 1
        for shard_idx in range(n):
            shard_path = request.shard_paths[shard_idx]
            shard_start = time.perf_counter()
            _emit_shard_start(progress_callback, request, shard_idx, shard_path)
            result = _scan_result_with_progress(
                futures.pop(shard_idx),
                progress_queue=progress_queue,
                progress_callback=progress_callback,
            )  # rendezvous; raises -> abort
            wait_seconds = time.perf_counter() - shard_start
            merge_start = time.perf_counter()
            intruder_seen, live_positive_rows = _merge_shard_into_index(
                state,
                max_intruder_refs=request.max_intruder_refs,
                seed=request.seed,
                positive_rows=result.positive_rows,
                shard_intruders=_unpack_intruders(result),
                intruder_refs=intruder_refs,
                intruder_seen=intruder_seen,
            )
            merge_seconds = time.perf_counter() - merge_start
            if next_submit < n:
                futures[next_submit] = executor.submit(
                    _scan_shard_worker, _scan_task(request, next_submit, target_ids)
                )
                next_submit += 1
            _emit_shard_done(
                progress_callback,
                request,
                tmp_path,
                shard_idx,
                shard_path,
                result.positive_rows,
                live_positive_rows,
                intruder_refs,
                shard_start,
                wait_seconds=wait_seconds,
                merge_seconds=merge_seconds,
            )
    except BaseException:
        _drain_worker_progress(progress_queue, progress_callback)
        # Abort promptly: cancel queued shards and do not block on in-flight
        # scans (default shutdown(wait=True) would finish up to window scans
        # before the failure surfaces). Orphaned spools are swept by the
        # caller's except path.
        executor.shutdown(wait=False, cancel_futures=True)
        if progress_queue is not None:
            progress_queue.close()
            progress_queue.join_thread()
        raise
    executor.shutdown(wait=True)
    _drain_worker_progress(progress_queue, progress_callback)
    if progress_queue is not None:
        progress_queue.close()
        progress_queue.join_thread()
    return intruder_seen


def _cleanup_worker_spools(request: SourceIndexRequest) -> None:
    """Sweep orphaned per-shard spool temp files on build abort. A worker that
    raises cleans its own spool via ``spooled_feature_shard``'s context manager;
    a hard-crashed worker (killed / ``os._exit``) cannot, so the parent removes
    any ``.{shard}.spool.*`` left in the temp dir. Best-effort; never raises, so
    it cannot mask the original build failure it runs alongside on the abort
    path."""
    temp_dir = request.index_path.parent
    try:
        for shard_path in request.shard_paths:
            for spool in temp_dir.glob(f".{shard_path.name}.spool.*"):
                spool.unlink(missing_ok=True)
    except OSError:
        pass


def _drain_worker_progress(
    progress_queue: Any | None,
    progress_callback: SourceIndexProgressCallback | None,
) -> None:
    if progress_queue is None or progress_callback is None:
        return
    while True:
        try:
            event = progress_queue.get_nowait()
        except queue.Empty:
            return
        _emit_progress(progress_callback, event)


def _scan_result_with_progress(
    future: Future[_ScanResult],
    *,
    progress_queue: Any | None,
    progress_callback: SourceIndexProgressCallback | None,
) -> _ScanResult:
    while True:
        _drain_worker_progress(progress_queue, progress_callback)
        try:
            result = future.result(timeout=_WORKER_PROGRESS_POLL_INTERVAL_S)
        except TimeoutError:
            continue
        _drain_worker_progress(progress_queue, progress_callback)
        return result


def _build_index(
    request: SourceIndexRequest,
    manifest: dict,
    *,
    progress_callback: SourceIndexProgressCallback | None = None,
) -> None:
    tmp_path = request.index_path.with_name(f"{request.index_path.name}.tmp.{os.getpid()}")
    tmp_path.unlink(missing_ok=True)
    logger.info(f"Building autointerp source index at {request.index_path}")
    build_start = time.perf_counter()
    _emit_progress(
        progress_callback,
        SourceIndexProgressEvent(
            kind=SourceIndexProgressEventKind.BUILD_START,
            shard_count=len(request.shard_paths),
            target_count=len(set(request.target_feature_ids)),
            rss_kib=_current_rss_kib(),
        ),
    )
    conn = _connect(tmp_path)
    try:
        _create_schema(conn)
        _assert_unique_shard_stems(request.shard_paths)
        intruder_refs: list[SourceDocRef] = []
        intruder_seen = 0
        target_ids = tuple(sorted(set(request.target_feature_ids)))
        state = _TopKMergeState(request.max_positive_refs_per_feature)
        if request.scan_workers == 1:
            for shard_idx, shard_path in enumerate(request.shard_paths):
                shard_start = time.perf_counter()
                _emit_shard_start(progress_callback, request, shard_idx, shard_path)
                positive_rows, shard_intruders = _scan_source_index_shard_file_topk_vectorized(
                    shard_path=shard_path,
                    target_feature_ids=target_ids,
                    n_sae_features=request.n_sae_features,
                    max_positive_refs=request.max_positive_refs_per_feature,
                    chunk_size=request.chunk_size,
                    temp_dir=request.index_path.parent,
                    shard_index=shard_idx,
                    shard_count=len(request.shard_paths),
                    progress_callback=progress_callback,
                )
                merge_start = time.perf_counter()
                intruder_seen, live_positive_rows = _merge_shard_into_index(
                    state,
                    max_intruder_refs=request.max_intruder_refs,
                    seed=request.seed,
                    positive_rows=positive_rows,
                    shard_intruders=shard_intruders,
                    intruder_refs=intruder_refs,
                    intruder_seen=intruder_seen,
                )
                merge_seconds = time.perf_counter() - merge_start
                _emit_shard_done(
                    progress_callback,
                    request,
                    tmp_path,
                    shard_idx,
                    shard_path,
                    positive_rows,
                    live_positive_rows,
                    intruder_refs,
                    shard_start,
                    merge_seconds=merge_seconds,
                )
        else:
            intruder_seen = _parallel_scan_and_merge(
                request,
                tmp_path,
                target_ids,
                intruder_refs,
                state,
                progress_callback,
            )
        _bulk_insert_positive_refs(conn, state.materialize())
        _insert_intruder_refs(conn, intruder_refs)
        conn.commit()
    except BaseException:
        conn.close()
        tmp_path.unlink(missing_ok=True)
        if request.scan_workers > 1:
            # Serial builds have no workers (their spool is cleaned by the
            # scanner's context manager); only sweep on the parallel path, where
            # a hard-crashed worker can orphan a spool.
            _cleanup_worker_spools(request)
        raise
    conn.close()
    try:
        os.replace(tmp_path, request.index_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    atomic_write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        request.manifest_path,
    )
    _emit_progress(
        progress_callback,
        SourceIndexProgressEvent(
            kind=SourceIndexProgressEventKind.BUILD_DONE,
            shard_count=len(request.shard_paths),
            target_count=len(set(request.target_feature_ids)),
            elapsed_seconds=time.perf_counter() - build_start,
            sqlite_bytes=_path_size(request.index_path),
            rss_kib=_current_rss_kib(),
        ),
    )


def _load_spooled_source_index_shard_data(
    shard,
    *,
    shard_path: Path,
) -> _SpooledSourceIndexShardData:
    missing = _SOURCE_INDEX_REQUIRED_TENSOR_KEYS - shard.tensor_keys
    if missing:
        raise AutoInterpDataError(
            f"feature shard {shard_path} is missing required source-index "
            f"tensors: {sorted(missing)}"
        )
    data = shard.parsed_metadata()
    data[TensorKey.OFFSETS] = shard.get_tensor(TensorKey.OFFSETS)
    data[TensorKey.DOC_IDS] = shard.get_tensor(TensorKey.DOC_IDS)
    for key in _SOURCE_INDEX_ROW_OPTIONAL_TENSOR_KEYS:
        if str(key) in shard.tensor_keys:
            data[key] = shard.get_tensor(key)

    offsets = data[TensorKey.OFFSETS].to(t.int64)
    if int(offsets.ndim) != 1 or int(offsets.numel()) == 0:
        raise AutoInterpDataError(f"feature shard {shard_path} has invalid CSR offsets")
    if int(offsets[0].item()) != 0 or bool((offsets[1:] < offsets[:-1]).any()):
        raise AutoInterpDataError(f"feature shard {shard_path} has non-monotonic CSR offsets")
    n_entries = int(offsets[-1].item())
    _validate_spooled_tensor_length(
        shard,
        TensorKey.FEATURE_IDS,
        expected=n_entries,
        shard_path=shard_path,
    )
    _validate_spooled_tensor_length(
        shard,
        TensorKey.FEATURE_MAGS,
        expected=n_entries,
        shard_path=shard_path,
    )
    n_rows = int(offsets.numel()) - 1
    doc_ids = row_aligned_i64_tensor(
        data,
        TensorKey.DOC_IDS,
        shard_path=shard_path,
        n_rows=n_rows,
    )
    if doc_ids is None:
        raise AutoInterpDataError(f"feature shard {shard_path} is missing doc_ids")
    segment_token_starts, segment_token_ends = segment_token_bounds(
        data,
        shard_path=shard_path,
        n_rows=n_rows,
    )
    stem = shard_path.stem.removesuffix(".safetensors")
    shard_index = shard_index_from_stem(stem)
    intruders = [
        _validated_source_doc_ref(
            shard_stem=stem,
            shard_index=shard_index,
            doc_id=int(doc_ids[row_index].item()),
            segment_token_start=int(segment_token_starts[row_index].item()),
            segment_token_end=int(segment_token_ends[row_index].item()),
            context=f"feature shard {shard_path} intruder row {row_index}",
        )
        for row_index in range(n_rows)
    ]
    return _SpooledSourceIndexShardData(
        offsets=offsets,
        n_entries=n_entries,
        doc_ids=doc_ids,
        segment_token_starts=segment_token_starts,
        segment_token_ends=segment_token_ends,
        stem=stem,
        shard_index=shard_index,
        intruders=intruders,
    )


def _vectorized_chunk_survivors(
    *,
    chunk: _DecodedEntryChunk,
    offsets_tail: t.Tensor,
    allowed: t.Tensor,
    n_sae_features: int,
    n_rows: int,
    worst: t.Tensor,
    key_ranks: t.Tensor,
) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor] | None:
    chunk_fids = chunk.feature_ids
    chunk_mags = chunk.magnitudes
    if int(chunk_fids.ndim) != 1 or int(chunk_mags.ndim) != 1:
        raise AutoInterpDataError("source-index entry chunks must be 1-D tensors")
    if int(chunk_fids.numel()) != int(chunk_mags.numel()):
        raise AutoInterpDataError(
            "source-index entry chunk length mismatch: "
            f"feature_ids={int(chunk_fids.numel())}, "
            f"feature_mags={int(chunk_mags.numel())}"
        )
    if chunk.start < 0:
        raise AutoInterpDataError(f"source-index entry chunk has negative start {chunk.start}")
    if chunk.end > int(offsets_tail[-1].item()):
        raise AutoInterpDataError(
            "source-index entry chunk exceeds CSR payload length: "
            f"chunk_end={chunk.end}, n_entries={int(offsets_tail[-1].item())}"
        )
    in_range = (chunk_fids >= 0) & (chunk_fids < n_sae_features)
    if not bool(in_range.any()):
        return None
    match = t.zeros_like(in_range)
    valid_positions = in_range.nonzero(as_tuple=False).flatten()
    match[valid_positions] = allowed[chunk_fids[valid_positions]]
    if not bool(match.any()):
        return None
    local_indices = match.nonzero(as_tuple=False).flatten().to(t.int64)
    entry_indices = local_indices + chunk.start
    matched_fids = chunk_fids[local_indices]
    matched_mags = chunk_mags[local_indices]
    if not bool(t.isfinite(matched_mags).all()):
        raise AutoInterpDataError("source-index entry chunk has non-finite magnitude")
    row_indices = t.searchsorted(offsets_tail, entry_indices, right=True)
    if int(row_indices.numel()) and int(row_indices.max().item()) >= n_rows:
        raise AutoInterpDataError(
            "source-index row lookup exceeded row-aligned tensors: "
            f"max_row={int(row_indices.max().item())}, n_rows={n_rows}"
        )
    keep = matched_mags >= worst[matched_fids]
    if not bool(keep.any()):
        return None
    kept = keep.nonzero(as_tuple=False).flatten()
    kept_fids = matched_fids[kept].to(t.int32)
    kept_mags = matched_mags[kept].to(t.float32)
    kept_rows = row_indices[kept].to(t.int32)
    return (
        kept_fids,
        key_ranks[kept_rows],
        kept_mags,
        kept_rows,
    )


def _finalize_vectorized_rows(
    state: tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor],
    *,
    scan_data: _SpooledSourceIndexShardData,
    row_keys: list[str],
) -> list[tuple]:
    st_fids, _, st_mags, st_rows = state
    if not bool(t.isfinite(st_mags).all()):
        raise AutoInterpDataError(
            "source-index positive row has non-finite magnitude="
            f"{float(st_mags[~t.isfinite(st_mags)][0])}"
        )
    refs = scan_data.intruders
    out: list[tuple] = []
    for fid, mag, row in zip(st_fids.tolist(), st_mags.tolist(), st_rows.tolist(), strict=True):
        ref = refs[row]
        out.append(
            (
                fid,
                mag,
                row_keys[row],
                ref.shard_stem,
                ref.shard_index,
                ref.doc_id,
                ref.segment_token_start,
                ref.segment_token_end,
            )
        )
    return out


@beartype
def _scan_source_index_shard_file_topk_vectorized(
    *,
    shard_path: Path,
    target_feature_ids: NonNegativeIntTupleLike,
    n_sae_features: PositiveIntLike,
    max_positive_refs: PositiveIntLike,
    chunk_size: PositiveIntLike,
    temp_dir: Path | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
    progress_callback: SourceIndexProgressCallback | None = None,
    merge_pending_min_rows: int | None = None,
) -> _ScanResultLike:
    """Vectorized production scanner; output-equal to the survivor scanner.

    ``merge_pending_min_rows`` is a scheduling seam; the merge-threshold
    test matrix proves it cannot change output.
    """
    _validate_target_feature_ids(target_feature_ids, n_sae_features=n_sae_features)

    shard_scan_start = time.perf_counter()
    with spooled_feature_shard(shard_path, temp_dir=temp_dir) as shard:
        scan_data = _load_spooled_source_index_shard_data(
            shard,
            shard_path=shard_path,
        )
        _emit_progress(
            progress_callback,
            SourceIndexProgressEvent(
                kind=SourceIndexProgressEventKind.SHARD_SPOOLED,
                shard_index=shard_index,
                shard_count=shard_count,
                shard_name=shard_path.name,
                entries_total=scan_data.n_entries,
                rows_total=int(scan_data.offsets.numel()) - 1,
                elapsed_seconds=time.perf_counter() - shard_scan_start,
                shard_bytes=_path_size(shard_path),
                spool_bytes=_path_size(getattr(shard, "temp_path", None)),
                rss_kib=_current_rss_kib(),
            ),
        )
        if not target_feature_ids:
            return [], scan_data.intruders

        allowed = _target_feature_mask(
            target_feature_ids,
            n_sae_features=n_sae_features,
        )
        if n_sae_features > t.iinfo(t.int32).max:
            raise AutoInterpDataError("source-index n_sae_features exceeds int32 feature space")
        offsets_tail = scan_data.offsets[1:]
        n_rows = int(offsets_tail.numel())
        row_keys = [ref.stable_key() for ref in scan_data.intruders]
        key_ranks = _rank_row_stable_keys(row_keys)

        state = (
            t.zeros(0, dtype=t.int32),
            t.zeros(0, dtype=t.int32),
            t.zeros(0, dtype=t.float32),
            t.zeros(0, dtype=t.int32),
        )
        worst = _worst_magnitudes_from_state(
            state[0],
            state[2],
            n_sae_features=n_sae_features,
            max_positive_refs=max_positive_refs,
        )
        pending: list[tuple[t.Tensor, ...]] = []
        pending_rows = 0
        total_chunks = max(1, math.ceil(scan_data.n_entries / chunk_size))

        def merge_pending() -> None:
            nonlocal state, worst, pending, pending_rows
            if not pending:
                return
            cat = tuple(t.cat([state[i], *(p[i] for p in pending)]) for i in range(4))
            state = _reduce_source_index_candidates(
                cat[0],
                cat[1],
                cat[2],
                cat[3],
                max_positive_refs=max_positive_refs,
            )
            worst = _worst_magnitudes_from_state(
                state[0],
                state[2],
                n_sae_features=n_sae_features,
                max_positive_refs=max_positive_refs,
            )
            pending = []
            pending_rows = 0

        for chunk_index, chunk in enumerate(
            _iter_spooled_entry_chunks(
                shard,
                scan_data,
                chunk_size=chunk_size,
            ),
            start=1,
        ):
            chunk_start_time = time.perf_counter()
            survivors = _vectorized_chunk_survivors(
                chunk=chunk,
                offsets_tail=offsets_tail,
                allowed=allowed,
                n_sae_features=n_sae_features,
                n_rows=n_rows,
                worst=worst,
                key_ranks=key_ranks,
            )
            chunk_survivors = 0
            if survivors is not None:
                chunk_survivors = int(survivors[0].shape[0])
                pending.append(survivors)
                pending_rows += chunk_survivors
            threshold = (
                merge_pending_min_rows
                if merge_pending_min_rows is not None
                else max(2 * int(state[0].shape[0]), _VEC_MERGE_PENDING_MIN_ROWS)
            )
            if pending_rows >= threshold:
                merge_pending()
            chunk_seconds = time.perf_counter() - chunk_start_time
            elapsed_seconds = time.perf_counter() - shard_scan_start
            entries_done = min(chunk.end, scan_data.n_entries)
            # state[0] is sorted by fid (reduce invariant), so the distinct
            # count is a linear adjacency scan, never an O(n log n) unique.
            state_features = (
                0 if state[0].shape[0] == 0 else int((state[0][1:] != state[0][:-1]).sum()) + 1
            )
            _emit_progress(
                progress_callback,
                SourceIndexProgressEvent(
                    kind=SourceIndexProgressEventKind.CHUNK_DONE,
                    shard_index=shard_index,
                    shard_count=shard_count,
                    shard_name=shard_path.name,
                    chunk_index=chunk_index,
                    chunk_count=total_chunks,
                    entries_done=entries_done,
                    entries_total=scan_data.n_entries,
                    positive_rows=chunk_survivors,
                    bucket_features=state_features,
                    bucket_rows=int(state[0].shape[0]) + pending_rows,
                    chunk_seconds=chunk_seconds,
                    elapsed_seconds=elapsed_seconds,
                    entries_per_second=(
                        entries_done / elapsed_seconds if elapsed_seconds > 0.0 else None
                    ),
                    rss_kib=_current_rss_kib(),
                ),
            )
        merge_pending()
        return _finalize_vectorized_rows(
            state,
            scan_data=scan_data,
            row_keys=row_keys,
        ), scan_data.intruders


def _validate_spooled_tensor_length(
    shard,
    key: TensorKey,
    *,
    expected: int,
    shard_path: Path,
) -> None:
    actual = 1
    for dim in shard.tensor_shape(key):
        actual *= int(dim)
    if actual != expected:
        raise AutoInterpDataError(
            f"feature shard {shard_path} CSR payload length mismatch: "
            f"offsets[-1]={expected}, {key}={actual}"
        )


def _iter_spooled_entry_chunks(
    shard,
    scan_data: _SpooledSourceIndexShardData,
    *,
    chunk_size: int,
) -> Iterator[_DecodedEntryChunk]:
    offsets_tail = scan_data.offsets[1:]
    for chunk_start in range(0, scan_data.n_entries, chunk_size):
        chunk_end = min(chunk_start + chunk_size, scan_data.n_entries)
        decode_start, decode_end, local_offsets = _decode_span_for_entry_chunk(
            scan_data.offsets,
            offsets_tail,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
        )
        decoded_fids = _delta_decode(
            shard.get_tensor_slice(TensorKey.FEATURE_IDS, decode_start, decode_end),
            local_offsets,
        )
        local_start = chunk_start - decode_start
        local_end = chunk_end - decode_start
        feature_ids = decoded_fids[local_start:local_end].to(t.int64)
        magnitudes = shard.get_tensor_slice(
            TensorKey.FEATURE_MAGS,
            chunk_start,
            chunk_end,
        ).to(t.float32)
        expected_len = chunk_end - chunk_start
        if int(feature_ids.numel()) != expected_len or int(magnitudes.numel()) != expected_len:
            raise AutoInterpDataError(
                "decoded source-index chunk length mismatch: "
                f"entries={expected_len}, feature_ids={int(feature_ids.numel())}, "
                f"feature_mags={int(magnitudes.numel())}"
            )
        yield _DecodedEntryChunk(
            start=chunk_start,
            end=chunk_end,
            feature_ids=feature_ids,
            magnitudes=magnitudes,
        )


def _decode_span_for_entry_chunk(
    offsets: t.Tensor,
    offsets_tail: t.Tensor,
    *,
    chunk_start: int,
    chunk_end: int,
) -> tuple[int, int, t.Tensor]:
    if chunk_start < 0 or chunk_end <= chunk_start:
        raise AutoInterpDataError(
            f"invalid source-index entry chunk bounds: {chunk_start}:{chunk_end}"
        )
    if int(offsets.ndim) != 1 or int(offsets_tail.ndim) != 1:
        raise AutoInterpDataError("source-index CSR offsets must be 1-D tensors")
    if int(offsets.numel()) != int(offsets_tail.numel()) + 1:
        raise AutoInterpDataError(
            "source-index offsets_tail must be offsets[1:] "
            f"(offsets={int(offsets.numel())}, tail={int(offsets_tail.numel())})"
        )
    if int(offsets.numel()) == 0:
        raise AutoInterpDataError("source-index CSR offsets are empty")
    n_entries = int(offsets[-1].item())
    if chunk_end > n_entries:
        raise AutoInterpDataError(
            "source-index entry chunk exceeds CSR payload length: "
            f"chunk_end={chunk_end}, n_entries={n_entries}"
        )
    start_row = int(
        t.searchsorted(
            offsets_tail,
            t.tensor(chunk_start, dtype=t.int64),
            right=True,
        ).item()
    )
    end_row = int(
        t.searchsorted(
            offsets_tail,
            t.tensor(chunk_end - 1, dtype=t.int64),
            right=True,
        ).item()
    )
    decode_start = int(offsets[start_row].item())
    decode_end = int(offsets[end_row + 1].item())
    local_offsets = offsets[start_row : end_row + 2] - decode_start
    if not (decode_start <= chunk_start < chunk_end <= decode_end):
        raise AutoInterpDataError(
            "source-index decoded span does not contain requested entry chunk: "
            f"decode={decode_start}:{decode_end}, chunk={chunk_start}:{chunk_end}"
        )
    if int(local_offsets.numel()) < 2 or int(local_offsets[0].item()) != 0:
        raise AutoInterpDataError("source-index decoded span has invalid local offsets")
    if int(local_offsets[-1].item()) != decode_end - decode_start:
        raise AutoInterpDataError(
            "source-index decoded span local offsets do not cover decoded payload"
        )
    if bool((local_offsets[1:] < local_offsets[:-1]).any()):
        raise AutoInterpDataError("source-index decoded span has non-monotonic local offsets")
    return decode_start, decode_end, local_offsets


def _insert_intruder_refs(
    conn: sqlite3.Connection,
    refs: list[SourceDocRef],
) -> None:
    conn.executemany(
        "INSERT INTO intruder_refs (rank, stable_key, shard_stem, shard_index, "
        "doc_id, segment_token_start, segment_token_end) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [_intruder_row_tuple(rank, ref) for rank, ref in enumerate(refs)],
    )


def _validate_index_metadata(
    conn: sqlite3.Connection,
    *,
    index_path: Path,
) -> None:
    row = conn.execute("SELECT value FROM metadata WHERE key = 'schema_version'").fetchone()
    if row is None or row[0] != str(_SCHEMA_VERSION):
        raise AutoInterpCacheError(
            f"{index_path} schema_version={None if row is None else row[0]} != {_SCHEMA_VERSION}"
        )


def _validate_bound_sqlite_single_file_header(file_descriptor: int, *, path: Path) -> None:
    """Require a rollback-format main DB; immutable SQLite masks WAL format."""
    try:
        header = os.pread(file_descriptor, 20, 0)
    except OSError as exc:
        raise AutoInterpCacheError(
            f"cannot read bound source-index SQLite header {path}: {exc}"
        ) from exc
    if len(header) != 20 or header[:16] != b"SQLite format 3\x00":
        raise AutoInterpCacheError(f"source-index artifact is not a SQLite 3 database: {path}")
    write_version = header[18]
    read_version = header[19]
    if (write_version, read_version) != (1, 1):
        raise AutoInterpCacheError(
            "published source index must be a rollback-format SQLite main database "
            "for an exact single-file proof; header write/read versions must both be 1, "
            f"got {(write_version, read_version)!r} for {path}"
        )


def _validate_published_source_index_header(path: Path) -> None:
    """Check the on-disk SQLite format before opening it read-only.

    ``immutable=1`` in :func:`_connect_read_only` prevents an innocent
    read-only probe from materializing WAL sidecars, so a published index must
    be a rollback-format single file; reject anything else up front.
    """
    try:
        file_descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        raise AutoInterpCacheError(f"cannot open source-index SQLite header {path}: {exc}") from exc
    try:
        _validate_bound_sqlite_single_file_header(file_descriptor, path=path)
    finally:
        os.close(file_descriptor)


def _sqlite_sidecar_paths(index_path: Path) -> tuple[Path, Path, Path]:
    return (
        index_path.with_name(f"{index_path.name}-journal"),
        index_path.with_name(f"{index_path.name}-wal"),
        index_path.with_name(f"{index_path.name}-shm"),
    )


def _reject_source_index_sqlite_sidecars(index_path: Path) -> None:
    """Require the pathname-visible published database to be one immutable file."""
    existing: list[Path] = []
    for path in _sqlite_sidecar_paths(index_path):
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise AutoInterpCacheError(
                f"cannot inspect source-index SQLite sidecar path {path}: {exc}"
            ) from exc
        existing.append(path)
    if existing:
        raise AutoInterpCacheError(
            "published source index must be one immutable SQLite file without "
            f"journal/WAL sidecars; found {[str(path) for path in existing]}"
        )


def _load_index(request: SourceIndexRequest) -> SourceIndexData:
    conn = _connect_read_only(request.index_path)
    try:
        _validate_index_metadata(conn, index_path=request.index_path)
        positives: dict[int, list[tuple[SourceDocRef, float]]] = {}
        query = (
            "SELECT feature_id, magnitude, stable_key, shard_stem, shard_index, "
            "doc_id, segment_token_start, segment_token_end FROM positive_refs "
            "WHERE feature_id = ? ORDER BY rank ASC"
        )
        for fid in _resolved_load_target_feature_ids(request):
            rows = conn.execute(query, (int(fid),)).fetchall()
            parsed_rows = [
                _positive_row_from_tuple(
                    tuple(row),
                    context=f"{request.index_path} positive_refs feature_id={fid}",
                    error_cls=AutoInterpCacheError,
                )
                for row in rows
            ]
            positives[int(fid)] = [(parsed.doc_ref, parsed.magnitude) for parsed in parsed_rows]
        intruders = []
        for row in conn.execute(
            "SELECT stable_key, shard_stem, shard_index, doc_id, segment_token_start, "
            "segment_token_end FROM intruder_refs ORDER BY rank ASC"
        ).fetchall():
            intruders.append(
                _validated_source_doc_ref(
                    shard_stem=str(row[1]),
                    shard_index=int(row[2]),
                    doc_id=int(row[3]),
                    segment_token_start=int(row[4]),
                    segment_token_end=int(row[5]),
                    stable_key=str(row[0]),
                    context=f"{request.index_path} intruder_refs row",
                    error_cls=AutoInterpCacheError,
                )
            )
    finally:
        conn.close()
    return SourceIndexData(
        positives_by_feature=positives,
        intruder_refs=intruders,
    )


def shard_index_from_stem(stem: str) -> int:
    match = re.search(r"shard_(\d+)", stem)
    if match is None:
        raise AutoInterpDataError(f"feature shard stem {stem!r} does not contain a shard index")
    return int(match.group(1))


def row_aligned_i64_tensor(
    data: dict,
    key: TensorKey,
    *,
    shard_path: Path,
    n_rows: int,
) -> t.Tensor | None:
    value = data.get(key)
    if value is None:
        return None
    tensor = value.to(t.int64)
    if int(tensor.numel()) != n_rows:
        raise AutoInterpDataError(
            f"feature shard {shard_path} has {key} length {int(tensor.numel())}, expected {n_rows}"
        )
    return tensor


def segment_token_bounds(
    data: dict,
    *,
    shard_path: Path,
    n_rows: int,
) -> tuple[t.Tensor, t.Tensor]:
    starts = row_aligned_i64_tensor(
        data,
        TensorKey.SEGMENT_TOKEN_STARTS,
        shard_path=shard_path,
        n_rows=n_rows,
    )
    ends = row_aligned_i64_tensor(
        data,
        TensorKey.SEGMENT_TOKEN_ENDS,
        shard_path=shard_path,
        n_rows=n_rows,
    )
    if (starts is None) != (ends is None):
        raise AutoInterpDataError(
            f"feature shard {shard_path} must contain both segment_token_starts "
            "and segment_token_ends when either is present"
        )
    if starts is not None and ends is not None:
        if bool((starts < 0).any()) or bool((ends <= starts).any()):
            raise AutoInterpDataError(
                f"feature shard {shard_path} has invalid segment token bounds"
            )
        return starts, ends

    segment_size = int(data.get(MetadataKey.SEGMENT_SIZE, 0))
    if segment_size <= 0:
        raise AutoInterpDataError(
            f"feature shard {shard_path} has invalid segment_size={segment_size}; "
            "token-level autointerp requires positive segment size"
        )

    segment_mode = segment_mode_from_metadata(data, source=shard_path)
    if segment_mode is not SegmentMode.fixed_tokens:
        raise AutoInterpDataError(
            f"feature shard {shard_path} has no explicit segment token provenance (segment_mode={segment_mode.value!r})"
        )

    positions = row_aligned_i64_tensor(
        data,
        TensorKey.SEGMENT_POSITIONS,
        shard_path=shard_path,
        n_rows=n_rows,
    )
    if positions is not None:
        starts = positions
    else:
        segment_indices = row_aligned_i64_tensor(
            data,
            TensorKey.SEGMENT_INDICES,
            shard_path=shard_path,
            n_rows=n_rows,
        )
        if segment_indices is None:
            raise AutoInterpDataError(
                f"feature shard {shard_path} is missing segment token provenance; "
                "expected explicit token bounds, segment_positions, or fixed-token "
                "segment_indices"
            )
        starts = segment_indices * segment_size

    if bool((starts < 0).any()):
        raise AutoInterpDataError(f"feature shard {shard_path} has negative segment token starts")
    return starts, starts + segment_size
