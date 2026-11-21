"""Shared splice-counterfactual helpers for live and packed selection paths."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch as t
from loguru import logger

from nanochat.atomic_io import atomic_json_dump
from steering.conditional.delta import (
    CounterfactualRef,
    FeatureDeltaConfig,
)
from steering.conditional.feature_index import FeatureIndex
from steering.conditional.percentile_table import uint8_to_percentile
from steering.conditional.splice_provenance import (
    SpliceProvenanceError,
    SpliceProvenanceIndex,
)
from steering.feature_io import glob_features, shard_stem_from_path
from steering.gradient_overlay import (
    PROVENANCE_BASE_SHARD_IDX_KEY,
    PROVENANCE_DOC_IDX_KEY,
    PROVENANCE_SEG_ORDINAL_KEY,
    provenance_dir,
)
from steering.immutable_io import is_sha256_hex
from steering.overlay_io import glob_overlay_shards, load_overlay_shard, read_manifest
from steering.packed_batch_io import load_packed_row_shard


@dataclass(frozen=True)
class PackedProvenanceInfo:
    """Validated packed-row provenance identity for a packed-row directory."""

    base_shard_stems: list[str]
    packed_shard_paths: list[Path]
    provenance_shard_paths: list[Path]
    packed_stem_fingerprint: str
    packed_row_fingerprint: str
    provenance_fingerprint: str


_FILE_FINGERPRINT_CACHE_SCHEMA_VERSION = 2
_PACKED_PROVENANCE_CACHE_SCHEMA_VERSION = 2
_PACKED_ROW_FINGERPRINT_CACHE = ".packed-row-fingerprint-cache.json"
_PROVENANCE_FINGERPRINT_CACHE = ".packed-provenance-fingerprint-cache.json"
_DERIVED_FEATURE_FINGERPRINT_CACHE = ".derived-feature-fingerprint-cache.json"
_SPLICE_SIDECAR_FINGERPRINT_CACHE = ".splice-sidecar-fingerprint-cache.json"
_PACKED_PROVENANCE_VALIDATION_CACHE = ".packed-provenance-validation-cache.json"
_STAT_FIELD_NAMES = (
    "mode",
    "ino",
    "size",
    "mtime_ns",
    "ctime_ns",
)


@dataclass(frozen=True)
class _FileSignature:
    """Exact link + target identity used to validate a cached content digest.

    The cache is valid only while every direct input has the same inode, mode,
    size, mtime, and ctime at both the directory-entry and target levels.
    ``st_dev`` is deliberately excluded: on clustered filesystems the same
    immutable file can have a different client-local device number on each
    node. Atomic replacement still invalidates even same-size files via inode
    or ctime, and retargeting a symlink invalidates either its link identity or
    target identity. The target must be a regular file.
    """

    name: str
    lstat_fields: tuple[int, int, int, int, int]
    stat_fields: tuple[int, int, int, int, int]

    @staticmethod
    def _fields(result: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            int(result.st_mode),
            int(result.st_ino),
            int(result.st_size),
            int(result.st_mtime_ns),
            int(result.st_ctime_ns),
        )

    @classmethod
    def capture(cls, path: Path) -> _FileSignature:
        absolute = path.absolute()
        try:
            link_result = absolute.lstat()
            target_result = absolute.stat()
        except OSError as exc:
            raise SpliceProvenanceError(
                f"cannot capture artifact signature for {absolute}: {exc}"
            ) from exc
        if not stat.S_ISREG(target_result.st_mode):
            raise SpliceProvenanceError(
                f"artifact fingerprint input is not a regular file: {absolute}"
            )
        return cls(
            name=path.name,
            lstat_fields=cls._fields(link_result),
            stat_fields=cls._fields(target_result),
        )

    @staticmethod
    def _fields_json(
        fields: tuple[int, int, int, int, int],
    ) -> dict[str, int]:
        return dict(zip(_STAT_FIELD_NAMES, fields, strict=True))

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "lstat": self._fields_json(self.lstat_fields),
            "stat": self._fields_json(self.stat_fields),
        }


@dataclass(frozen=True)
class PackedFingerprintFile:
    """One ordered packed shard's stat signature and content fingerprint."""

    name: str
    signature: _FileSignature
    content_sha256: str


@dataclass(frozen=True)
class PackedFingerprintResolution:
    """Structured result from the strict packed-row cache interface."""

    aggregate_sha256: str
    files: tuple[PackedFingerprintFile, ...]
    cache_path: Path

    def content_inventory(self) -> tuple[tuple[str, str], ...]:
        return tuple((entry.name, entry.content_sha256) for entry in self.files)


def assert_packed_row_fingerprint_resolution_current(
    resolution: PackedFingerprintResolution,
) -> None:
    """Recheck both link and target signatures without opening shard bodies."""

    paths = [
        resolution.cache_path.parent / entry.name for entry in resolution.files
    ]
    expected = [entry.signature for entry in resolution.files]
    if _capture_file_signatures(paths) != expected:
        raise SpliceProvenanceError(
            "packed-row inputs changed after their strict fingerprint "
            "resolution was established"
        )


def assert_packed_row_fingerprint_file_current(
    entry: PackedFingerprintFile,
    path: Path,
) -> None:
    """Recheck one named packed shard against a strict-resolution entry."""

    if path.name != entry.name or _FileSignature.capture(path) != entry.signature:
        raise SpliceProvenanceError(
            f"packed-row input {path} changed after its strict fingerprint "
            "resolution was established"
        )


def _ordered_unique_paths(paths: list[Path]) -> list[Path]:
    ordered = sorted((Path(path) for path in paths), key=lambda path: path.name)
    previous_name: str | None = None
    for path in ordered:
        if path.name == previous_name:
            raise ValueError(
                f"_fingerprint_files received a duplicate basename {path.name!r}; "
                "fingerprint inputs must have unique names (fingerprint one "
                "directory's direct children)."
            )
        previous_name = path.name
    return ordered


def _capture_file_signatures(paths: list[Path]) -> list[_FileSignature]:
    return [_FileSignature.capture(path) for path in paths]


def _signature_from_json(raw: object, *, surface: str) -> _FileSignature:
    if not isinstance(raw, Mapping) or set(raw) != {"name", "lstat", "stat"}:
        raise ValueError(f"{surface} must contain exactly name/lstat/stat")
    parsed_raw = cast(Mapping[str, object], raw)
    name = parsed_raw["name"]
    if not isinstance(name, str) or not name or name != Path(name).name:
        raise ValueError(f"{surface}.name is not a direct filename: {name!r}")

    parsed_fields: list[tuple[int, int, int, int, int]] = []
    for field_name in ("lstat", "stat"):
        raw_fields = parsed_raw[field_name]
        if not isinstance(raw_fields, Mapping) or set(raw_fields) != set(
            _STAT_FIELD_NAMES
        ):
            raise ValueError(
                f"{surface}.{field_name} must contain exactly {_STAT_FIELD_NAMES!r}"
            )
        parsed_raw_fields = cast(Mapping[str, object], raw_fields)
        values: list[int] = []
        for key in _STAT_FIELD_NAMES:
            value = parsed_raw_fields[key]
            if type(value) is not int:
                raise ValueError(
                    f"{surface}.{field_name}.{key} must be a strict integer"
                )
            values.append(value)
        if values[2] < 0:
            raise ValueError(f"{surface}.{field_name}.size must be non-negative")
        parsed_fields.append((values[0], values[1], values[2], values[3], values[4]))
    return _FileSignature(
        name=name,
        lstat_fields=parsed_fields[0],
        stat_fields=parsed_fields[1],
    )


def _fold_file_digests(digests: list[str]) -> str:
    h = hashlib.sha256()
    for digest in digests:
        h.update(digest.encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def canonical_feature_delta_dict(config: FeatureDeltaConfig) -> dict[str, Any]:
    """Canonical manifest representation for a parsed feature_delta config."""
    return {
        "mode": config.mode.value,
        "neg_delta_ratio": float(config.neg_delta_ratio),
    }


def _fingerprint_json(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Streaming chunk for per-file hashing: bounds resident bytes per worker thread
# (vs the pre-#740 ``read_bytes()`` that loaded each whole ~2 GB shard) so many
# files can hash concurrently in a few hundred MB total.
_FINGERPRINT_CHUNK = 16 * 1024 * 1024  # 16 MiB


def _file_fingerprint(path: Path) -> str:
    """SHA-256 hex of one file's name + streamed bytes.

    Issue #740: this is a ONE-TIME streaming hash of ~1.3 TB (source + packed shards) that is not
    re-read afterwards, so it must not pollute the page cache — otherwise ~100+ GB of cache
    accumulates and later starves the shared /dev/shm source store into swap (the parallel run
    then runs ~4.6x slower and OOMs). ``posix_fadvise(DONTNEED)`` on each already-hashed range
    keeps the fingerprint's cache footprint to ~one chunk per thread.
    """
    h = hashlib.sha256()
    h.update(path.name.encode("utf-8"))
    h.update(b"\0")
    with open(path, "rb") as fh:
        fd = fh.fileno()
        offset = 0
        while True:
            block = fh.read(_FINGERPRINT_CHUNK)
            if not block:
                break
            h.update(block)
            n = len(block)
            try:
                os.posix_fadvise(fd, offset, n, os.POSIX_FADV_DONTNEED)
            except (OSError, AttributeError):
                pass  # non-POSIX / unsupported fs: fall back to plain streaming
            offset += n
    return h.hexdigest()


def _file_fingerprint_with_body_validator(
    path: Path,
    validator: Callable[[Path, bytes], None],
) -> str:
    """Hash and validate one exact compressed body from a single file read."""

    h = hashlib.sha256()
    h.update(path.name.encode("utf-8"))
    h.update(b"\0")
    with open(path, "rb") as fh:
        fd = fh.fileno()
        compressed = fh.read()
        try:
            h.update(compressed)
            validator(path, compressed)
        finally:
            try:
                os.posix_fadvise(
                    fd,
                    0,
                    len(compressed),
                    os.POSIX_FADV_DONTNEED,
                )
            except (OSError, AttributeError):
                pass
    return h.hexdigest()


def _drop_page_cache(path: Path) -> None:
    """Advise the kernel to drop a fully-read file from the page cache (issue #740).

    The startup validation streams ~635 GB of packed shards (decompressed once to read their
    segment counts) that it never re-reads; without dropping that cache, ~100+ GB accumulates and
    starves the shared /dev/shm source store into swap during the parallel run. Best-effort:
    unsupported platforms/filesystems silently fall back to leaving the cache in place.
    """
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)  # length 0 ⇒ whole file
    except (OSError, AttributeError):
        pass
    finally:
        os.close(fd)


def _fingerprint_files(paths: list[Path]) -> str:
    """Content fingerprint of a set of files, order-independent of scheduling.

    Issue #740 startup optimization: the pre-#740 implementation folded every
    file's (name + bytes) into ONE serial SHA-256 stream, which fingerprinting
    ~1.34 TB made a 25-72 min single-threaded startup cost re-paid on every run.
    A single SHA stream cannot be parallelized (SHA is sequential), so this hashes
    each file independently (name + streamed bytes) and folds the per-file hex
    digests together in name-sorted order. Because ``hashlib`` releases the GIL on
    large ``update`` calls, a THREAD pool parallelizes BOTH the reads and the
    hashing across cores (no torch, no fork/spawn/pickle hazards).

    This CHANGES the fingerprint VALUE vs the pre-#740 single-stream scheme (a
    one-time migration: existing selection manifests fingerprinted under the old
    scheme read as stale and must be rebuilt), but it remains fully content-based
    — any byte change in any file changes the digest — so staleness detection and
    fail-loud behavior are unchanged. The result is deterministic and independent
    of thread scheduling (folding is name-sorted).

    Preconditions (matching the pre-#740 serial version): inputs must have UNIQUE
    basenames — the fold orders by basename, so a shared basename would be
    order-ambiguous, and every caller fingerprints the direct children of ONE
    directory (unique names). Inputs must also be quiescent (not concurrently
    rewritten) — like any content hash, a writer mutating a file mid-read yields
    an unspecified snapshot; these artifacts are immutable by contract."""
    ordered = _ordered_unique_paths(paths)
    if not ordered:
        return hashlib.sha256().hexdigest()
    max_workers = min(len(ordered), os.cpu_count() or 1)
    if max_workers <= 1 or len(ordered) == 1:
        per_file = [_file_fingerprint(p) for p in ordered]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            # ``map`` yields results in input (name-sorted) order regardless of
            # which thread finishes first, so the fold below is deterministic.
            per_file = list(pool.map(_file_fingerprint, ordered))
    return _fold_file_digests(per_file)


def _load_file_fingerprint_cache(
    cache_path: Path,
    *,
    namespace: str,
) -> tuple[list[_FileSignature], list[str], str] | None:
    """Strictly parse a fingerprint cache; every defect is a visible miss."""
    try:
        cache_path.lstat()
    except FileNotFoundError:
        logger.debug(f"artifact fingerprint cache miss: no cache at {cache_path}")
        return None
    except OSError as exc:
        logger.warning(
            f"artifact fingerprint cache {cache_path} is unreadable "
            f"({exc!r}); recomputing from artifact bytes"
        )
        return None
    if not cache_path.is_file():
        logger.debug(f"artifact fingerprint cache miss: no cache at {cache_path}")
        return None
    try:
        raw = json.loads(cache_path.read_text())
        if not isinstance(raw, Mapping) or set(raw) != {
            "schema_version",
            "namespace",
            "files",
            "aggregate_sha256",
        }:
            raise ValueError("top-level keys are not exact")
        if type(raw["schema_version"]) is not int or (
            raw["schema_version"] != _FILE_FINGERPRINT_CACHE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported schema_version={raw['schema_version']!r}"
            )
        if raw["namespace"] != namespace:
            raise ValueError(
                f"namespace={raw['namespace']!r}, expected {namespace!r}"
            )
        raw_files = raw["files"]
        if not isinstance(raw_files, list) or not raw_files:
            raise ValueError("files must be a non-empty list")
        signatures: list[_FileSignature] = []
        digests: list[str] = []
        previous_name: str | None = None
        for index, raw_entry in enumerate(raw_files):
            if not isinstance(raw_entry, Mapping) or set(raw_entry) != {
                "signature",
                "content_sha256",
            }:
                raise ValueError(f"files[{index}] keys are not exact")
            parsed_entry = cast(Mapping[str, object], raw_entry)
            signature = _signature_from_json(
                parsed_entry["signature"], surface=f"files[{index}].signature"
            )
            if previous_name is not None and signature.name <= previous_name:
                raise ValueError("file names must be unique and strictly sorted")
            previous_name = signature.name
            digest = parsed_entry["content_sha256"]
            if not is_sha256_hex(digest):
                raise ValueError(
                    f"files[{index}].content_sha256 is not lowercase SHA-256"
                )
            signatures.append(signature)
            digests.append(cast(str, digest))
        aggregate = raw["aggregate_sha256"]
        expected_aggregate = _fold_file_digests(digests)
        if not is_sha256_hex(aggregate) or aggregate != expected_aggregate:
            raise ValueError(
                "aggregate_sha256 does not match cached per-file digests"
            )
        return signatures, digests, expected_aggregate
    except (OSError, UnicodeError, ValueError) as exc:
        logger.warning(
            f"artifact fingerprint cache {cache_path} is corrupt or unreadable "
            f"({exc!r}); recomputing from artifact bytes"
        )
        return None


def _store_file_fingerprint_cache(
    cache_path: Path,
    *,
    namespace: str,
    signatures: list[_FileSignature],
    digests: list[str],
    aggregate: str,
) -> None:
    try:
        atomic_json_dump(
            {
                "schema_version": _FILE_FINGERPRINT_CACHE_SCHEMA_VERSION,
                "namespace": namespace,
                "files": [
                    {
                        "signature": signature.to_json(),
                        "content_sha256": digest,
                    }
                    for signature, digest in zip(signatures, digests, strict=True)
                ],
                "aggregate_sha256": aggregate,
            },
            cache_path,
        )
    except OSError as exc:
        logger.warning(
            f"could not write artifact fingerprint cache {cache_path} ({exc!r}); "
            "continuing with the freshly computed content fingerprint"
        )


def _fingerprint_files_cached(
    paths: list[Path],
    *,
    cache_dir: Path | None,
    cache_filename: str,
    namespace: str,
) -> str:
    """Return the canonical fingerprint, using only an exact stat-certified hit.

    A cold/missing/corrupt/stale cache executes the existing full content-hash
    implementation. Signatures are captured both before and after the hash;
    any concurrent artifact mutation fails loud instead of certifying a mixed
    snapshot. Cache writes are best-effort because they are an optimization.
    """
    ordered = _ordered_unique_paths(paths)
    if not ordered:
        return hashlib.sha256().hexdigest()
    if cache_dir is None:
        return _fingerprint_files(ordered)

    before = _capture_file_signatures(ordered)
    cache_path = Path(cache_dir) / cache_filename
    cached = _load_file_fingerprint_cache(cache_path, namespace=namespace)
    if cached is not None:
        cached_signatures, _cached_digests, aggregate = cached
        if cached_signatures == before:
            after_hit = _capture_file_signatures(ordered)
            if after_hit != before:
                raise SpliceProvenanceError(
                    f"artifact input changed while validating cached {namespace} "
                    "fingerprint; retry from quiescent immutable artifacts"
                )
            logger.debug(
                f"artifact fingerprint cache hit for {namespace}: {cache_path}"
            )
            return aggregate
        logger.info(
            f"artifact fingerprint cache {cache_path} is stale: input file "
            "identity changed; recomputing from artifact bytes"
        )

    max_workers = min(len(ordered), os.cpu_count() or 1)
    if max_workers <= 1 or len(ordered) == 1:
        digests = [_file_fingerprint(path) for path in ordered]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            digests = list(pool.map(_file_fingerprint, ordered))
    after = _capture_file_signatures(ordered)
    if after != before:
        changed = [
            signature.name
            for signature, current in zip(before, after, strict=True)
            if signature != current
        ]
        raise SpliceProvenanceError(
            f"artifact input changed while computing {namespace} fingerprint: "
            f"{changed[:5]}; retry from quiescent immutable artifacts"
        )
    aggregate = _fold_file_digests(digests)
    _store_file_fingerprint_cache(
        cache_path,
        namespace=namespace,
        signatures=before,
        digests=digests,
        aggregate=aggregate,
    )
    return aggregate


def resolve_packed_row_fingerprint_strict(
    packed_row_dir: str | Path,
    *,
    force: bool = False,
    allow_cache_write: bool = True,
    body_validator: Callable[[Path, bytes], None] | None = None,
) -> PackedFingerprintResolution:
    """Resolve packed identity with source fences and a best-effort cache.

    Missing, stale, or unreadable caches are recomputed from artifact bytes.
    ``force=True`` ignores the cache and validates each body when requested.
    Cache writes are optional and independent of consumer validation.
    """

    if not isinstance(force, bool):
        raise TypeError(f"force must be a bool, got {force!r}")
    if not isinstance(allow_cache_write, bool):
        raise TypeError(
            f"allow_cache_write must be a bool, got {allow_cache_write!r}"
        )
    if body_validator is not None and (not force or not callable(body_validator)):
        raise TypeError(
            "body_validator must be callable and is supported only with force=True"
        )
    directory = Path(packed_row_dir)
    ordered = _ordered_unique_paths(glob_overlay_shards(directory))
    if not ordered:
        raise SpliceProvenanceError(
            f"no packed row shards found in {directory}."
        )
    before = _capture_file_signatures(ordered)
    cache_path = directory / _PACKED_ROW_FINGERPRINT_CACHE
    cached: tuple[list[_FileSignature], list[str], str] | None = None
    if not force:
        cached = _load_file_fingerprint_cache(
            cache_path,
            namespace="packed_rows",
        )
    if cached is not None:
        cached_signatures, cached_digests, aggregate = cached
        if cached_signatures == before:
            if _capture_file_signatures(ordered) != before:
                raise SpliceProvenanceError(
                    "packed-row inputs changed while validating the strict "
                    "fingerprint cache hit"
                )
            resolution = PackedFingerprintResolution(
                aggregate_sha256=aggregate,
                files=tuple(
                    PackedFingerprintFile(
                        name=signature.name,
                        signature=signature,
                        content_sha256=digest,
                    )
                    for signature, digest in zip(
                        cached_signatures,
                        cached_digests,
                        strict=True,
                    )
                ),
                cache_path=cache_path,
            )
            logger.info(f"packed-row fingerprint cache hit: {cache_path}")
            return resolution

    max_workers = min(len(ordered), os.cpu_count() or 1)
    if body_validator is not None:
        digests = [
            _file_fingerprint_with_body_validator(path, body_validator)
            for path in ordered
        ]
    elif max_workers <= 1:
        digests = [_file_fingerprint(path) for path in ordered]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            digests = list(pool.map(_file_fingerprint, ordered))
    after = _capture_file_signatures(ordered)
    if after != before:
        changed = [
            signature.name
            for signature, current in zip(before, after, strict=True)
            if signature != current
        ]
        raise SpliceProvenanceError(
            "packed-row inputs changed while computing the strict fingerprint: "
            f"{changed[:5]}"
        )
    computed = PackedFingerprintResolution(
        aggregate_sha256=_fold_file_digests(digests),
        files=tuple(
            PackedFingerprintFile(
                name=signature.name,
                signature=signature,
                content_sha256=digest,
            )
            for signature, digest in zip(before, digests, strict=True)
        ),
        cache_path=cache_path,
    )
    if allow_cache_write:
        _store_file_fingerprint_cache(
            cache_path,
            namespace="packed_rows",
            signatures=before,
            digests=digests,
            aggregate=computed.aggregate_sha256,
        )
        assert_packed_row_fingerprint_resolution_current(computed)
    return computed


def packed_row_fingerprint(
    packed_row_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
) -> str:
    """Fingerprint packed-row shard bytes for selection-manifest binding."""
    packed_row_dir = Path(packed_row_dir)
    paths = glob_overlay_shards(packed_row_dir)
    if not paths:
        raise SpliceProvenanceError(f"no packed row shards found in {packed_row_dir}.")
    return _fingerprint_files_cached(
        paths,
        cache_dir=Path(cache_dir) if cache_dir is not None else None,
        cache_filename=_PACKED_ROW_FINGERPRINT_CACHE,
        namespace="packed_rows",
    )


def tensor_content_fingerprint(tensor: t.Tensor) -> str:
    """Fingerprint a tensor's shape, dtype, and exact contiguous CPU bytes."""
    h = hashlib.sha256()
    h.update(str(tensor.dtype).encode("utf-8"))
    h.update(b"\0")
    h.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("utf-8"))
    h.update(b"\0")
    h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def derived_feature_fingerprint(
    feature_dir: str | Path,
    *,
    expected_stems: list[str] | None = None,
    cache_dir: str | Path | None = None,
) -> str:
    """Fingerprint derived feature shard bytes used by splice preselection."""
    feature_dir = Path(feature_dir)
    paths = glob_features(feature_dir)
    if not paths:
        raise SpliceProvenanceError(
            f"no feature shards found in derived feature dir {feature_dir}."
        )
    stems = [shard_stem_from_path(p) for p in paths]
    if expected_stems is not None and stems != expected_stems:
        raise SpliceProvenanceError(
            f"derived feature shard stems for {feature_dir} are {stems}, but "
            f"packed provenance records {expected_stems}. Re-run packed-row and "
            "selected-feature precompute from matching artifacts."
        )
    return _fingerprint_files_cached(
        paths,
        cache_dir=Path(cache_dir) if cache_dir is not None else None,
        cache_filename=_DERIVED_FEATURE_FINGERPRINT_CACHE,
        namespace="derived_features",
    )


def splice_sidecar_fingerprint(
    feature_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
) -> str:
    """Content fingerprint of a spliced feature dir's sidecar JSON files."""
    sidecar_dir = Path(feature_dir) / "_splice_provenance"
    if not sidecar_dir.is_dir():
        raise SpliceProvenanceError(
            f"splice provenance dir not found: {sidecar_dir}. "
            "feature_delta.mode=splice_counterfactual requires a spliced "
            "derived feature directory."
        )
    paths = sorted(sidecar_dir.glob("*.json"))
    if not paths:
        raise SpliceProvenanceError(
            f"splice provenance dir {sidecar_dir} contains no sidecar JSON files."
        )
    return _fingerprint_files_cached(
        paths,
        cache_dir=Path(cache_dir) if cache_dir is not None else None,
        cache_filename=_SPLICE_SIDECAR_FINGERPRINT_CACHE,
        namespace="splice_sidecars",
    )


def _load_packed_provenance_validation_cache(
    cache_path: Path,
    *,
    manifest_fingerprint: str,
    base_shard_stems: list[str],
    recorded_provenance_shards: int,
    packed_paths: list[Path],
    provenance_paths: list[Path],
    packed_signatures: list[_FileSignature],
    provenance_signatures: list[_FileSignature],
) -> PackedProvenanceInfo | None:
    """Return a prior semantic certificate only for an exact artifact identity."""
    if not cache_path.is_file():
        logger.debug(
            f"packed-provenance validation cache miss: no cache at {cache_path}"
        )
        return None
    try:
        raw = json.loads(cache_path.read_text())
        expected_keys = {
            "schema_version",
            "manifest_fingerprint",
            "base_shard_stems",
            "provenance_shards",
            "packed_files",
            "provenance_files",
            "packed_stem_fingerprint",
            "packed_row_fingerprint",
            "provenance_fingerprint",
            "aggregate_sha256",
        }
        if not isinstance(raw, Mapping) or set(raw) != expected_keys:
            raise ValueError("top-level keys are not exact")
        signed_payload = {
            key: raw[key] for key in expected_keys if key != "aggregate_sha256"
        }
        aggregate = raw["aggregate_sha256"]
        expected_aggregate = _fingerprint_json(signed_payload)
        if not is_sha256_hex(aggregate) or aggregate != expected_aggregate:
            raise ValueError(
                "aggregate_sha256 does not match the semantic certificate"
            )
        if type(raw["schema_version"]) is not int or (
            raw["schema_version"] != _PACKED_PROVENANCE_CACHE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported schema_version={raw['schema_version']!r}"
            )
        cached_manifest_fingerprint = raw["manifest_fingerprint"]
        if not is_sha256_hex(cached_manifest_fingerprint):
            raise ValueError("manifest_fingerprint is not lowercase SHA-256")
        cached_stems = raw["base_shard_stems"]
        if not isinstance(cached_stems, list) or not all(
            isinstance(stem, str) for stem in cached_stems
        ):
            raise ValueError("base_shard_stems must be a list of strings")
        cached_count = raw["provenance_shards"]
        if type(cached_count) is not int or cached_count < 0:
            raise ValueError("provenance_shards must be a non-negative strict integer")

        raw_packed = raw["packed_files"]
        if not isinstance(raw_packed, list) or not raw_packed:
            raise ValueError("packed_files must be a non-empty list")
        cached_packed_signatures: list[_FileSignature] = []
        previous_name: str | None = None
        for index, entry in enumerate(raw_packed):
            if not isinstance(entry, Mapping) or set(entry) != {
                "signature",
                "total_segments",
            }:
                raise ValueError(f"packed_files[{index}] keys are not exact")
            parsed_entry = cast(Mapping[str, object], entry)
            signature = _signature_from_json(
                parsed_entry["signature"],
                surface=f"packed_files[{index}].signature",
            )
            if previous_name is not None and signature.name <= previous_name:
                raise ValueError(
                    "packed file names must be unique and strictly sorted"
                )
            previous_name = signature.name
            total_segments = parsed_entry["total_segments"]
            if type(total_segments) is not int or total_segments < 0:
                raise ValueError(
                    f"packed_files[{index}].total_segments must be a "
                    "non-negative strict integer"
                )
            cached_packed_signatures.append(signature)

        raw_provenance = raw["provenance_files"]
        if not isinstance(raw_provenance, list) or not raw_provenance:
            raise ValueError("provenance_files must be a non-empty list")
        cached_provenance_signatures: list[_FileSignature] = []
        previous_name = None
        for index, entry in enumerate(raw_provenance):
            signature = _signature_from_json(
                entry, surface=f"provenance_files[{index}]"
            )
            if previous_name is not None and signature.name <= previous_name:
                raise ValueError(
                    "provenance file names must be unique and strictly sorted"
                )
            previous_name = signature.name
            cached_provenance_signatures.append(signature)

        packed_stem_fingerprint = raw["packed_stem_fingerprint"]
        packed_row_fingerprint_value = raw["packed_row_fingerprint"]
        provenance_fingerprint = raw["provenance_fingerprint"]
        if not all(
            is_sha256_hex(value)
            for value in (
                packed_stem_fingerprint,
                packed_row_fingerprint_value,
                provenance_fingerprint,
            )
        ):
            raise ValueError("one or more cached fingerprints are not lowercase SHA-256")
        if packed_stem_fingerprint != _fingerprint_json(cached_stems):
            raise ValueError(
                "packed_stem_fingerprint does not match cached base_shard_stems"
            )
    except (OSError, ValueError) as exc:
        logger.warning(
            f"packed-provenance validation cache {cache_path} is corrupt or "
            f"unreadable ({exc!r}); repeating full semantic validation"
        )
        return None

    current_identity = (
        manifest_fingerprint,
        base_shard_stems,
        recorded_provenance_shards,
        packed_signatures,
        provenance_signatures,
    )
    cached_identity = (
        cached_manifest_fingerprint,
        cached_stems,
        cached_count,
        cached_packed_signatures,
        cached_provenance_signatures,
    )
    if current_identity != cached_identity:
        logger.info(
            f"packed-provenance validation cache {cache_path} is stale: "
            "manifest or artifact identity changed; repeating full semantic validation"
        )
        return None
    logger.debug(f"packed-provenance validation cache hit: {cache_path}")
    return PackedProvenanceInfo(
        base_shard_stems=list(base_shard_stems),
        packed_shard_paths=packed_paths,
        provenance_shard_paths=provenance_paths,
        packed_stem_fingerprint=packed_stem_fingerprint,
        packed_row_fingerprint=packed_row_fingerprint_value,
        provenance_fingerprint=provenance_fingerprint,
    )


def _store_packed_provenance_validation_cache(
    cache_path: Path,
    *,
    manifest_fingerprint: str,
    base_shard_stems: list[str],
    recorded_provenance_shards: int,
    packed_signatures: list[_FileSignature],
    provenance_signatures: list[_FileSignature],
    packed_total_segments: list[int],
    info: PackedProvenanceInfo,
) -> None:
    try:
        signed_payload = {
            "schema_version": _PACKED_PROVENANCE_CACHE_SCHEMA_VERSION,
            "manifest_fingerprint": manifest_fingerprint,
            "base_shard_stems": base_shard_stems,
            "provenance_shards": recorded_provenance_shards,
            "packed_files": [
                {
                    "signature": signature.to_json(),
                    "total_segments": total_segments,
                }
                for signature, total_segments in zip(
                    packed_signatures, packed_total_segments, strict=True
                )
            ],
            "provenance_files": [
                signature.to_json() for signature in provenance_signatures
            ],
            "packed_stem_fingerprint": info.packed_stem_fingerprint,
            "packed_row_fingerprint": info.packed_row_fingerprint,
            "provenance_fingerprint": info.provenance_fingerprint,
        }
        atomic_json_dump(
            {
                **signed_payload,
                "aggregate_sha256": _fingerprint_json(signed_payload),
            },
            cache_path,
        )
    except OSError as exc:
        logger.warning(
            f"could not write packed-provenance validation cache {cache_path} "
            f"({exc!r}); continuing with the completed full validation"
        )


def validate_packed_provenance(
    packed_row_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
) -> PackedProvenanceInfo:
    """Validate packed-row provenance coverage and return its identity.

    The checks are intentionally done before any selection/loader iteration so
    stale packed rows fail at startup instead of silently falling back to live
    selection or joining against the wrong derived feature shard.
    """
    packed_row_dir = Path(packed_row_dir)
    manifest_path = packed_row_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SpliceProvenanceError(
            f"packed-row manifest missing for {packed_row_dir}; re-run "
            "scripts.preprocess.precompute_packed_rows."
        )
    manifest_signature = _FileSignature.capture(manifest_path)
    manifest = read_manifest(packed_row_dir)
    manifest_fingerprint = _file_fingerprint(manifest_path)
    manifest_signature_after_read = _FileSignature.capture(manifest_path)
    if manifest_signature_after_read != manifest_signature:
        raise SpliceProvenanceError(
            f"packed-row manifest changed while validating {packed_row_dir}; "
            "retry from quiescent immutable artifacts"
        )
    meta = manifest.get("metadata", {}) if manifest else {}
    if not isinstance(meta, dict):
        raise SpliceProvenanceError(
            f"packed-row manifest for {packed_row_dir} has invalid metadata; "
            "re-run scripts.preprocess.precompute_packed_rows."
        )
    stems = meta.get("base_shard_stems")
    recorded_prov_shards = meta.get("provenance_shards")
    prov_dir = provenance_dir(packed_row_dir)
    if manifest is None:
        raise SpliceProvenanceError(
            f"packed-row manifest invalid for {packed_row_dir}; re-run "
            "scripts.preprocess.precompute_packed_rows."
        )
    if not isinstance(stems, list) or not all(isinstance(s, str) for s in stems):
        raise SpliceProvenanceError(
            f"packed-row manifest for {packed_row_dir} is missing "
            "metadata.base_shard_stems; re-run precompute_packed_rows."
        )
    base_shard_stems = [str(s) for s in stems]
    if type(recorded_prov_shards) is not int:
        raise SpliceProvenanceError(
            f"packed-row manifest for {packed_row_dir} is missing integer "
            "metadata.provenance_shards; re-run precompute_packed_rows."
        )
    if not prov_dir.is_dir():
        raise SpliceProvenanceError(
            f"packed-segment provenance dir not found at {prov_dir}; re-run "
            "scripts.preprocess.precompute_packed_rows with resume=true."
        )

    packed_paths = glob_overlay_shards(packed_row_dir)
    prov_paths = glob_overlay_shards(prov_dir)
    if not packed_paths:
        raise SpliceProvenanceError(f"no packed row shards found in {packed_row_dir}.")
    if len(prov_paths) != len(packed_paths) or len(prov_paths) != recorded_prov_shards:
        raise SpliceProvenanceError(
            f"packed provenance at {prov_dir} covers {len(prov_paths)} shard(s) "
            f"but {packed_row_dir} has {len(packed_paths)} packed shard(s) "
            f"(manifest records {recorded_prov_shards}); re-run "
            "precompute_packed_rows with resume=true."
        )

    packed_signatures = _capture_file_signatures(packed_paths)
    provenance_signatures = _capture_file_signatures(prov_paths)
    resolved_cache_dir = Path(cache_dir) if cache_dir is not None else None
    if resolved_cache_dir is not None:
        cached_info = _load_packed_provenance_validation_cache(
            resolved_cache_dir / _PACKED_PROVENANCE_VALIDATION_CACHE,
            manifest_fingerprint=manifest_fingerprint,
            base_shard_stems=base_shard_stems,
            recorded_provenance_shards=recorded_prov_shards,
            packed_paths=packed_paths,
            provenance_paths=prov_paths,
            packed_signatures=packed_signatures,
            provenance_signatures=provenance_signatures,
        )
        if cached_info is not None:
            if (
                _capture_file_signatures(packed_paths) != packed_signatures
                or _capture_file_signatures(prov_paths) != provenance_signatures
                or _FileSignature.capture(manifest_path) != manifest_signature
            ):
                raise SpliceProvenanceError(
                    "packed rows, provenance, or manifest changed while "
                    "validating a cached packed-provenance certificate; retry "
                    "from quiescent immutable artifacts"
                )
            return cached_info

    prov_by_name = {p.name: p for p in prov_paths}
    packed_total_segments: list[int] = []
    for packed_path in packed_paths:
        prov_path = prov_by_name.get(packed_path.name)
        if prov_path is None:
            raise SpliceProvenanceError(
                f"provenance shard missing for packed shard {packed_path.name} "
                f"under {prov_dir}; re-run precompute_packed_rows with resume=true."
            )
        packed = load_packed_row_shard(packed_path)
        prov = load_overlay_shard(prov_path)
        try:
            base_idx = prov[PROVENANCE_BASE_SHARD_IDX_KEY]
            doc_idx = prov[PROVENANCE_DOC_IDX_KEY]
            seg_ordinal = prov[PROVENANCE_SEG_ORDINAL_KEY]
        except KeyError as exc:
            raise SpliceProvenanceError(
                f"{prov_path}: missing provenance tensor key {exc.args[0]!r}."
            ) from exc
        if not (
            len(base_idx)
            == len(doc_idx)
            == len(seg_ordinal)
            == packed.total_segments
        ):
            raise SpliceProvenanceError(
                f"{prov_path}: provenance has lengths "
                f"{len(base_idx)}, {len(doc_idx)}, {len(seg_ordinal)} but "
                f"packed shard {packed_path.name} has {packed.total_segments} "
                "segments."
            )
        if bool((base_idx < 0).any() or (doc_idx < 0).any() or (seg_ordinal < 0).any()):
            raise SpliceProvenanceError(
                f"{prov_path}: corrupt packed provenance contains negative "
                "(base_shard_idx, doc_idx, seg_ordinal) field(s)."
            )
        if bool((base_idx >= len(base_shard_stems)).any()):
            bad = int(base_idx[base_idx >= len(base_shard_stems)][0].item())
            raise SpliceProvenanceError(
                f"{prov_path}: base_shard_idx {bad} is out of range for "
                f"manifest base_shard_stems length {len(base_shard_stems)}."
            )
        packed_total_segments.append(int(packed.total_segments))
        # This 1.5 GB shard was decompressed only to read total_segments and is not re-read;
        # drop its (~635 GB across the corpus) page cache so it doesn't starve the source store.
        del packed
        _drop_page_cache(packed_path)

    packed_row_fingerprint_value = _fingerprint_files_cached(
        packed_paths,
        cache_dir=resolved_cache_dir,
        cache_filename=_PACKED_ROW_FINGERPRINT_CACHE,
        namespace="packed_rows",
    )
    provenance_fingerprint = _fingerprint_files_cached(
        prov_paths,
        cache_dir=resolved_cache_dir,
        cache_filename=_PROVENANCE_FINGERPRINT_CACHE,
        namespace="packed_provenance",
    )
    packed_signatures_after = _capture_file_signatures(packed_paths)
    provenance_signatures_after = _capture_file_signatures(prov_paths)
    manifest_signature_after_validation = _FileSignature.capture(manifest_path)
    if (
        packed_signatures_after != packed_signatures
        or provenance_signatures_after != provenance_signatures
        or manifest_signature_after_validation != manifest_signature
    ):
        raise SpliceProvenanceError(
            "packed rows, provenance, or manifest changed during full packed-"
            "provenance validation; retry from quiescent immutable artifacts"
        )

    info = PackedProvenanceInfo(
        base_shard_stems=base_shard_stems,
        packed_shard_paths=packed_paths,
        provenance_shard_paths=prov_paths,
        packed_stem_fingerprint=_fingerprint_json(base_shard_stems),
        packed_row_fingerprint=packed_row_fingerprint_value,
        provenance_fingerprint=provenance_fingerprint,
    )
    if resolved_cache_dir is not None:
        _store_packed_provenance_validation_cache(
            resolved_cache_dir / _PACKED_PROVENANCE_VALIDATION_CACHE,
            manifest_fingerprint=manifest_fingerprint,
            base_shard_stems=base_shard_stems,
            recorded_provenance_shards=recorded_prov_shards,
            packed_signatures=packed_signatures,
            provenance_signatures=provenance_signatures,
            packed_total_segments=packed_total_segments,
            info=info,
        )
    return info


def resolve_counterfactual_refs(
    splice_index: SpliceProvenanceIndex,
    feature_index: FeatureIndex,
    shard_stem: str,
    doc_idx: int,
    segments: list,
    has_precomputed: bool,
) -> tuple[list[CounterfactualRef | None], set[int]]:
    """Per-segment would-have-followed references for one derived doc."""
    n_prov = splice_index.doc_n_segments(shard_stem, doc_idx)
    if n_prov != len(segments):
        raise SpliceProvenanceError(
            f"{shard_stem} doc {doc_idx}: sidecar records {n_prov} segments but "
            f"the feature shard has {len(segments)} — sidecar/shard drift. "
            f"Re-run scripts.preprocess.splice_feature_shards (skip_existing=false)."
        )
    splices = splice_index.doc_splices(shard_stem, doc_idx)
    expected_positions = set(splices)
    seg_refs: list[CounterfactualRef | None] = [None] * len(segments)
    for pos, sref in splices.items():
        ref_doc, ref_pos = sref.whf_derived
        try:
            ref_doc_segments = feature_index.lookup(shard_stem, ref_doc)
        except KeyError as exc:
            raise SpliceProvenanceError(
                f"{shard_stem} doc {doc_idx} pos {pos}: would-have-followed doc "
                f"{ref_doc} is absent from the feature shard — sidecar/shard drift."
            ) from exc
        if ref_pos >= len(ref_doc_segments):
            raise SpliceProvenanceError(
                f"{shard_stem} doc {doc_idx} pos {pos}: would-have-followed "
                f"position {ref_pos} out of range for derived doc {ref_doc} "
                f"({len(ref_doc_segments)} segments) — sidecar/shard drift."
            )
        ref_seg = ref_doc_segments[ref_pos]
        if len(ref_seg.feature_ids) == 0:
            raise SpliceProvenanceError(
                f"{shard_stem} doc {doc_idx} pos {pos}: would-have-followed "
                f"segment {ref_doc}:{ref_pos} has zero feature ids — corrupt "
                "feature shard."
            )
        ref_pcts = (
            uint8_to_percentile(ref_seg.feature_percentiles)
            if has_precomputed and ref_seg.feature_percentiles is not None
            else None
        )
        seg_refs[pos] = (ref_seg.feature_ids, ref_seg.feature_mags, ref_pcts)
    resolved_positions = {i for i, ref in enumerate(seg_refs) if ref is not None}
    if resolved_positions != expected_positions:
        raise SpliceProvenanceError(
            f"{shard_stem} doc {doc_idx}: resolved counterfactual ref positions "
            f"{sorted(resolved_positions)} != sidecar splice positions "
            f"{sorted(expected_positions)}."
        )
    return seg_refs, expected_positions


def validate_counterfactual_delta_results(
    *,
    shard_stem: str,
    doc_idx: int,
    expected_positions: set[int],
    seg_refs: list[CounterfactualRef | None],
    batch_results: list[Any],
) -> int:
    ref_positions = {i for i, ref in enumerate(seg_refs) if ref is not None}
    if ref_positions != expected_positions:
        raise SpliceProvenanceError(
            f"{shard_stem} doc {doc_idx}: counterfactual ref positions "
            f"{sorted(ref_positions)} != sidecar splice positions "
            f"{sorted(expected_positions)}."
        )
    delta_positions = {
        i for i, result in enumerate(batch_results) if result is not None
    }
    if delta_positions != expected_positions:
        raise SpliceProvenanceError(
            f"{shard_stem} doc {doc_idx}: counterfactual delta-scored segment "
            f"positions {sorted(delta_positions)} != sidecar splice positions "
            f"{sorted(expected_positions)}."
        )
    return len(delta_positions)
