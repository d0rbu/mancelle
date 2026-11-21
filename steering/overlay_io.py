"""Overlay I/O for hierarchical precomputed data.

Overlays store precomputed data (percentiles, token boundaries, etc.) in
subdirectories of the feature directory. The directory structure encodes
the checksum of the data's provenance, making stale data structurally
impossible.

Layout::

    features/.../base/
      shard_00000_features.safetensors.zst   # immutable base
      pct/<pct_hash>/                         # percentile overlay
        manifest.json
        shard_00000.safetensors.zst
      tok/<tok_hash>/                         # tokenizer overlay
        manifest.json
        seg_boundaries/
          shard_00000.safetensors.zst

Mirror subdirectories use file-only symlinks to cross-reference overlays.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch as t
import zstandard
from loguru import logger
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save

from nanochat.atomic_io import atomic_json_dump, atomic_write_bytes, atomic_write_text
from steering.feature_io import (
    CorruptFeatureShardError,
    FeatureDirectoryMismatchError,
    MetadataKey,
    RegularFileIdentity,
    TensorKey,
    read_shard_metadata,
    shard_stem_from_path,
)

COMPRESSED_EXT = ".safetensors.zst"
ZSTD_LEVEL = 12
MANIFEST_FILENAME = "manifest.json"
CHECKSUM_PREFIX_LEN = 16

# Base-shard content-hash cache (issue #529). Opt-in via ``cache_dir``; stat-validated so a
# cache hit returns the EXACT streaming hash value without re-reading any shard bytes.
BASE_SHARD_HASH_CACHE_FILENAME = ".base_shard_hash_cache.json"
BASE_SHARD_HASH_CACHE_VERSION = 1
CONTENT_BOUND_MANIFEST_VERSION = 2
_LOWERCASE_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class OverlayStaleError(RuntimeError):
    """Raised when an overlay manifest no longer matches its base shard bytes."""


class OverlayManifestError(OverlayStaleError):
    """Raised when a present overlay manifest cannot satisfy a requested overlay."""


class OverlayIncompleteError(OverlayManifestError):
    """Raised when a valid overlay manifest is missing required shard payloads."""


class OverlayKind(Enum):
    """Supported overlay types.

    ``PACKED_ROW`` labels manifests for the precomputed packed-row directory
    (``pct/<pct_hash>/tok/<tok_hash>/c{T}/s{N}/manifest.json``). The
    ``overlay_dir`` path helper does **not** accept ``PACKED_ROW`` because
    packed-row dirs use a 4-level nested layout that
    is built directly by ``scripts/preprocess/precompute_packed_rows.py``;
    the enum value is only used as a label in the written manifest so
    consumers can distinguish packed-row manifests from percentile and
    segment-boundary manifests.
    """

    PERCENTILE = "pct"
    SEG_BOUNDARY = "seg_boundaries"
    PACKED_ROW = "packed_rows"
    GRADIENT = "grad"


# ---------------------------------------------------------------------------
# Path construction
# ---------------------------------------------------------------------------


def overlay_dir(feature_dir: Path, kind: OverlayKind, checksum: str) -> Path:
    """Construct the canonical overlay directory path.

    Returns:
        - PERCENTILE: ``feature_dir / pct / <checksum[:16]>``
        - SEG_BOUNDARY: ``feature_dir / tok / <checksum[:16]> / seg_boundaries``
        - GRADIENT: ``feature_dir / grad / <checksum[:16]>``

    Raises:
        ValueError: when ``kind`` is unsupported. ``PACKED_ROW`` is a label-only
            enum value; packed-row directory paths are constructed directly by
            the precompute script (4-level nested layout).
    """
    prefix = checksum[:CHECKSUM_PREFIX_LEN]
    if kind is OverlayKind.PERCENTILE:
        return feature_dir / "pct" / prefix
    if kind is OverlayKind.SEG_BOUNDARY:
        return feature_dir / "tok" / prefix / "seg_boundaries"
    if kind is OverlayKind.GRADIENT:
        return feature_dir / "grad" / prefix
    if kind is OverlayKind.PACKED_ROW:
        raise ValueError(
            "OverlayKind.PACKED_ROW is a manifest label only; packed-row "
            "directory paths are built by scripts/preprocess/precompute_packed_rows.py "
            "as pct/<pct>/tok/<tok>/c{T}/s{N}/, not via overlay_dir()."
        )
    raise ValueError(f"Unknown overlay kind: {kind}")


def expected_overlay_shard_stems(
    base_shard_paths: list[Path],
    kind: OverlayKind,
) -> list[str]:
    """Derive required payload stems without persisting coverage metadata.

    Percentile overlays cover every base shard. Token-boundary overlays omit
    whole-shard ``n_segments == 0`` inputs, matching their producer's existing
    no-payload behavior.
    """
    if kind not in (OverlayKind.PERCENTILE, OverlayKind.SEG_BOUNDARY):
        raise ValueError("expected_overlay_shard_stems supports only percentile and token overlays")
    stems: list[str] = []
    for base_path in base_shard_paths:
        stem = shard_stem_from_path(base_path)
        if kind is OverlayKind.PERCENTILE:
            stems.append(stem)
            continue
        metadata = read_shard_metadata(base_path)
        raw_n_segments = metadata.get(MetadataKey.N_SEGMENTS)
        if raw_n_segments is None:
            raise CorruptFeatureShardError(
                f"Base shard {base_path} lacks {MetadataKey.N_SEGMENTS.value} metadata."
            )
        try:
            n_segments = int(raw_n_segments)
        except ValueError as exc:
            raise CorruptFeatureShardError(
                f"Base shard {base_path} has invalid "
                f"{MetadataKey.N_SEGMENTS.value}={raw_n_segments!r}."
            ) from exc
        if n_segments < 0:
            raise CorruptFeatureShardError(
                f"Base shard {base_path} has negative {MetadataKey.N_SEGMENTS.value}={n_segments}."
            )
        if n_segments > 0:
            stems.append(stem)
    return stems


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _validate_precomputed_base_hash(value: str, *, surface: str) -> None:
    if not isinstance(value, str) or _LOWERCASE_SHA256_RE.fullmatch(value) is None:
        raise ValueError(
            f"{surface}: precomputed_base_hash must be exactly 64 lowercase hex "
            f"characters, got {value!r}"
        )


def discover_overlay(
    feature_dir: Path,
    kind: OverlayKind,
    checksum: str,
    *,
    base_shard_paths: list[Path] | None = None,
    precomputed_base_hash: str | None = None,
    cache_dir: Path | None = None,
) -> Path | None:
    """Find an overlay directory matching *checksum*.

    Checks that the directory exists and its manifest's full checksum matches.
    Accepts prefix matching: the caller may pass either the full 64-char hash
    or a prefix (>= ``CHECKSUM_PREFIX_LEN`` chars).  The manifest always stores
    the full hash, so the match direction is one-way:
    ``manifest_checksum.startswith(checksum)``.

    *cache_dir* forwards to the stat-validated base-shard hash cache (#530); only consulted
    when *base_shard_paths* is given and the manifest is v2. Remote producers may instead
    supply a validated *precomputed_base_hash*. The two identity sources are mutually
    exclusive.

    Returns the overlay directory path, or None if not found / checksum mismatch.
    """
    if base_shard_paths is not None and precomputed_base_hash is not None:
        raise ValueError(
            "discover_overlay: base_shard_paths and precomputed_base_hash are mutually exclusive"
        )
    if precomputed_base_hash is not None:
        _validate_precomputed_base_hash(
            precomputed_base_hash,
            surface="discover_overlay",
        )
        if cache_dir is not None:
            raise ValueError(
                "discover_overlay: cache_dir cannot be used with precomputed_base_hash"
            )
    if not checksum:
        return None
    d = overlay_dir(feature_dir, kind, checksum)
    if not d.is_dir():
        return None
    manifest = read_manifest(d)
    if manifest is None:
        logger.warning(f"Overlay dir exists but no valid manifest: {d}")
        return None
    manifest_checksum = manifest.get("checksum", "")
    if not manifest_checksum:
        logger.warning(f"Overlay manifest has empty checksum in {d}")
        return None
    # One-directional prefix match: the caller's checksum (which may be a
    # prefix) must be a prefix-of-or-equal-to the manifest's full hash.
    # The reverse direction (manifest prefix matching caller) is unsafe
    # because it would validate against potentially truncated data.
    if not manifest_checksum.startswith(checksum):
        logger.warning(
            f"Overlay manifest checksum mismatch in {d}: "
            f"expected {checksum[:16]}..., got {manifest_checksum[:16]}..."
        )
        return None
    if base_shard_paths is not None or precomputed_base_hash is not None:
        manifest_base_hash = manifest.get("base_shard_content_hash")
        if manifest_base_hash is None:
            schema_version = manifest.get("schema_version", 1)
            if isinstance(schema_version, int) and schema_version >= CONTENT_BOUND_MANIFEST_VERSION:
                raise OverlayStaleError(
                    f"Overlay manifest in {d} declares schema_version="
                    f"{schema_version} but lacks base_shard_content_hash; "
                    "rebuild the overlay for the current base shards."
                )
            logger.warning(
                f"Overlay manifest in {d} is legacy and lacks "
                "base_shard_content_hash; accepting without base-shard "
                "integrity validation"
            )
        else:
            current_base_hash = (
                precomputed_base_hash
                if precomputed_base_hash is not None
                else compute_base_shard_content_hash(
                    base_shard_paths or [],
                    cache_dir=cache_dir,
                )
            )
            if current_base_hash != manifest_base_hash:
                raise OverlayStaleError(
                    f"Overlay {d} was built against base shard content hash "
                    f"{manifest_base_hash[:16]}..., but current base shards hash "
                    f"to {current_base_hash[:16]}.... Re-run extraction or "
                    "rebuild the overlay for the current base shards."
                )
    return d


class BaseShardContentHasher:
    """Incremental form of the compatibility-sensitive base-shard digest.

    Each caller supplies shards in canonical order. One shard contributes its
    basename, a NUL byte, its exact compressed body, and a trailing NUL byte.
    ``bytes_read`` counts physical body bytes once.
    """

    def __init__(self) -> None:
        self._hash = hashlib.sha256()
        self.bytes_read = 0
        self._open_shard = False

    def begin_shard(self, shard_name: str) -> None:
        if self._open_shard:
            raise RuntimeError("BaseShardContentHasher already has an open shard")
        self._hash.update(shard_name.encode("utf-8"))
        self._hash.update(b"\0")
        self._open_shard = True

    def update_shard_bytes(self, chunk: bytes) -> None:
        if not self._open_shard:
            raise RuntimeError("BaseShardContentHasher has no open shard")
        self._hash.update(chunk)
        self.bytes_read += len(chunk)

    def end_shard(self) -> None:
        if not self._open_shard:
            raise RuntimeError("BaseShardContentHasher has no open shard")
        self._hash.update(b"\0")
        self._open_shard = False

    def update(self, shard_name: str, compressed_body: bytes) -> None:
        self.begin_shard(shard_name)
        self.update_shard_bytes(compressed_body)
        self.end_shard()

    def hexdigest(self) -> str:
        if self._open_shard:
            raise RuntimeError("BaseShardContentHasher cannot finalize an open shard")
        return self._hash.hexdigest()


def _stream_base_hash(sorted_paths: list[Path]) -> tuple[str, int]:
    """Canonical streaming SHA-256 over each shard's ``(name, bytes)``.

    Returns ``(hexdigest, bytes_read)``. *sorted_paths* must already be sorted by
    ``str(path)`` — the caller owns ordering so the cached and uncached paths hash an
    identical byte stream. This is the single source of truth for the hash value; do not
    change the update order without re-capturing every overlay's ``base_shard_content_hash``.
    """
    hasher = BaseShardContentHasher()
    for path in sorted_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Base shard not found for overlay hash: {path}")
        data = path.read_bytes()
        hasher.update(path.name, data)
    return hasher.hexdigest(), hasher.bytes_read


@dataclass(frozen=True)
class _ShardSig:
    """Cheap per-shard identity used to validate a cached hash without reading bytes.

    A false *hit* (returning a stale value) is the only dangerous failure; it is structurally
    impossible for an immutable base shard re-extracted via atomic rename, which always changes
    ``ino`` and ``ctime_ns`` (and almost always ``size``). Any ambiguity falls through to a miss.
    """

    name: str
    size: int
    mtime_ns: int
    ctime_ns: int
    ino: int

    @staticmethod
    def from_path(path: Path) -> _ShardSig:
        st = path.stat()
        return _ShardSig(path.name, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
            "ino": self.ino,
        }


@dataclass
class BaseHashResult:
    """Result of a base-shard hash computation, with profiling fields."""

    hexdigest: str
    bytes_read: int  # 0 on a cache hit; sum of shard sizes on a miss
    cache_hit: bool


def _load_hash_cache(cache_dir: Path) -> dict | None:
    """Load + structurally validate the cache file. Any defect ⇒ None (treated as a miss).

    Every miss is logged so a silently-ignored cache becomes visible. Severity reflects how
    surprising the miss is: a benign absent cache (first run) at DEBUG; a stale-version cache
    (expected after a format bump) at INFO; any structural corruption (unparseable bytes, wrong
    JSON type, truncated/garbage digest, malformed shard list) at WARNING — corruption is never
    swallowed silently. The routine "base shards legitimately changed" invalidation is a
    signature mismatch handled in ``_compute_base_hash``, not here.
    """
    path = cache_dir / BASE_SHARD_HASH_CACHE_FILENAME
    if not path.is_file():
        logger.debug(f"base-shard hash cache miss: no cache file at {path} (will recompute)")
        return None
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError) as exc:
        # ValueError covers json.JSONDecodeError AND read_text()'s UnicodeDecodeError
        # (a ValueError, not OSError) on a non-UTF-8 corrupt cache — any defect ⇒ miss.
        logger.warning(
            f"base-shard hash cache at {path} is corrupt/unreadable ({exc!r}); recomputing"
        )
        return None
    if not isinstance(data, dict):
        logger.warning(
            f"base-shard hash cache at {path} is not a JSON object "
            f"(got {type(data).__name__}); recomputing"
        )
        return None
    version = data.get("version")
    if version != BASE_SHARD_HASH_CACHE_VERSION:
        logger.info(
            f"base-shard hash cache at {path} has version {version!r} != "
            f"{BASE_SHARD_HASH_CACHE_VERSION}; recomputing"
        )
        return None
    digest = data.get("directory_hash")
    # Validate the digest shape HERE (not as a stripped-under-`-O` assert on the hit path):
    # a structurally-valid cache with a corrupt/truncated hash must fall through to recompute.
    if not isinstance(digest, str) or len(digest) != 64:
        logger.warning(
            f"base-shard hash cache at {path} has a malformed directory_hash "
            f"({digest!r}); recomputing"
        )
        return None
    if not isinstance(data.get("shards"), list):
        logger.warning(
            f"base-shard hash cache at {path} has a non-list 'shards' field; recomputing"
        )
        return None
    return data


def _sigs_match(cached_shards: list, current: list[_ShardSig]) -> bool:
    """True iff the cached signatures cover exactly the same name-set with identical stats."""
    if len(cached_shards) != len(current):
        return False
    cached_by_name = {c.get("name"): c for c in cached_shards if isinstance(c, dict)}
    if set(cached_by_name) != {s.name for s in current}:
        return False
    for s in current:
        c = cached_by_name[s.name]
        if (
            c.get("size") != s.size
            or c.get("mtime_ns") != s.mtime_ns
            or c.get("ctime_ns") != s.ctime_ns
            or c.get("ino") != s.ino
        ):
            return False
    return True


def _store_hash_cache(cache_dir: Path, directory_hash: str, sigs: list[_ShardSig]) -> None:
    # Best-effort: the cache is a pure optimization, so a write failure must NEVER crash the
    # caller — degrade to "no cache" and recompute next run. Issue #530 added a cache write to the
    # load/resolve path; without this guard a
    # read-only feature_dir at train/eval/serving time would turn a previously read-only path into
    # an uncaught OSError. The streamed hexdigest the caller already holds is unaffected.
    try:
        atomic_json_dump(
            {
                "version": BASE_SHARD_HASH_CACHE_VERSION,
                "directory_hash": directory_hash,
                "shards": [s.to_json() for s in sigs],
            },
            cache_dir / BASE_SHARD_HASH_CACHE_FILENAME,
        )
    except OSError as exc:
        logger.warning(
            f"could not write base-shard hash cache to {cache_dir} ({exc!r}); "
            "continuing without caching (hash will be recomputed next run)"
        )


def _compute_base_hash(base_shard_paths: list[Path], *, cache_dir: Path | None) -> BaseHashResult:
    """Compute the base-shard hash, optionally via the stat-validated cache.

    ``cache_dir=None`` ⇒ exact streaming behavior. ``cache_dir`` set ⇒ on a hit (all shard
    signatures match) return the cached value with ``bytes_read=0``; on any miss recompute by
    streaming and rewrite the cache. The returned ``hexdigest`` is identical either way.
    """
    sorted_paths = sorted((Path(p) for p in base_shard_paths), key=lambda p: str(p))
    if cache_dir is None:
        digest, nbytes = _stream_base_hash(sorted_paths)
        return BaseHashResult(hexdigest=digest, bytes_read=nbytes, cache_hit=False)

    for p in sorted_paths:
        if not p.is_file():
            raise FileNotFoundError(f"Base shard not found for overlay hash: {p}")
    current = [_ShardSig.from_path(p) for p in sorted_paths]

    cached = _load_hash_cache(cache_dir)
    if cached is not None:
        if _sigs_match(cached["shards"], current):
            digest = cached["directory_hash"]
            assert isinstance(digest, str) and len(digest) == 64, (
                f"corrupt cached base-shard hash in {cache_dir}: {digest!r}"
            )
            return BaseHashResult(hexdigest=digest, bytes_read=0, cache_hit=True)
        # Cache file is structurally valid but its signatures no longer match the on-disk base
        # shards — the routine "base shards were re-extracted/changed" invalidation, and the most
        # common real-world miss. Log it (INFO) so a warm run that unexpectedly re-streams the
        # whole corpus is diagnosable; _load_hash_cache already logs the absent/corrupt cases.
        logger.info(
            f"base-shard hash cache at {cache_dir / BASE_SHARD_HASH_CACHE_FILENAME} is stale: "
            "base shards changed (size/mtime_ns/ctime_ns/inode differ from the cached "
            "signatures) → recomputing"
        )

    digest, nbytes = _stream_base_hash(sorted_paths)
    _store_hash_cache(cache_dir, digest, current)
    return BaseHashResult(hexdigest=digest, bytes_read=nbytes, cache_hit=False)


def compute_base_shard_content_hash(
    base_shard_paths: list[Path], *, cache_dir: Path | None = None
) -> str:
    """Return a deterministic SHA-256 over base shard names and bytes.

    ``cache_dir=None`` (default) is the exact pre-#529 streaming behavior. Passing a
    ``cache_dir`` enables the stat-validated cache, which returns the *same* value.
    """
    return _compute_base_hash(base_shard_paths, cache_dir=cache_dir).hexdigest


def resolve_single_percentile_overlay(feature_dir: Path) -> tuple[Path, str]:
    """The one TRUE percentile overlay ``(dir, full checksum)`` under
    ``feature_dir/pct/``.

    Only a real percentile overlay counts: a ``pct/<hash>/`` directory that is
    merely the PARENT of a packed-row tree (``pct/<hash>/tok/.../c{T}/s{N}/``)
    carries no percentile manifest of its own (``kind != "pct"``) and is
    skipped. Raises :class:`OverlayStaleError` when absent (run
    ``precompute_percentiles`` first) or ambiguous (more than one) so callers
    that depend on the percentile identity — the gradient producer AND the
    gradient selection — resolve it identically and fail loud on drift.
    """
    cands: list[tuple[Path, dict]] = []
    for p in sorted((Path(feature_dir) / "pct").glob("*")):
        if not p.is_dir():
            continue
        manifest = read_manifest(p)
        if manifest is None or manifest.get("kind") != OverlayKind.PERCENTILE.value:
            continue
        cands.append((p, manifest))
    if not cands:
        raise OverlayStaleError(
            f"no percentile overlay under {Path(feature_dir) / 'pct'} — run "
            f"precompute_percentiles first."
        )
    if len(cands) > 1:
        raise OverlayStaleError(
            f"multiple percentile overlays under {Path(feature_dir) / 'pct'}: "
            + ", ".join(p.name for p, _ in cands)
            + " — ambiguous; keep exactly one."
        )
    cand_dir, manifest = cands[0]
    checksum = manifest.get("checksum")
    if not checksum:
        raise OverlayStaleError(f"percentile overlay {cand_dir} has no manifest checksum.")
    return cand_dir, str(checksum)


# ---------------------------------------------------------------------------
# Shard I/O (simple compressed safetensors, no delta encoding)
# ---------------------------------------------------------------------------


def save_overlay_shard(
    tensors: dict[str, t.Tensor],
    path: Path,
    metadata: dict[str, str] | None = None,
    *,
    zstd_level: int | None = None,
) -> Path:
    """Save an overlay shard as compressed safetensors.

    Unlike ``feature_io.save_features``, this does NO delta encoding or dtype
    optimization — tensors are saved as-is.

    Args:
        tensors: Dict of tensor name → tensor.
        path: Output path. Extension forced to ``.safetensors.zst``.
        metadata: Optional string metadata dict for the safetensors header.
        zstd_level: Compression level (1-22). Defaults to module-level ``ZSTD_LEVEL``.

    Returns:
        Actual path written.
    """
    path = Path(path)
    if not path.name.endswith(COMPRESSED_EXT):
        name = path.name.removesuffix(".safetensors")
        path = path.parent / (name + COMPRESSED_EXT)

    sf_bytes = safetensors_save(tensors, metadata=metadata or {})
    cctx = zstandard.ZstdCompressor(level=zstd_level if zstd_level is not None else ZSTD_LEVEL)
    compressed = cctx.compress(sf_bytes)

    # Atomic write via shared helper: tmp.<pid>.<uuid> + fsync + os.replace.
    # The per-PID unique tmp suffix prevents concurrent DDP overlay writers
    # from stomping each other's tmp file — the old ``.tmp`` suffix collided
    # when two ranks touched the same overlay shard.
    atomic_write_bytes(compressed, path)
    return path


def load_overlay_shard_bytes(compressed: bytes) -> dict[str, t.Tensor]:
    """Decode one complete compressed safetensors overlay from memory."""
    dctx = zstandard.ZstdDecompressor()
    sf_bytes = dctx.decompress(compressed)
    return safetensors_load(sf_bytes)


def load_overlay_shard(path: Path) -> dict[str, t.Tensor]:
    """Load an overlay shard from compressed safetensors.

    Follows symlinks transparently. Returns dict of tensor name → tensor.
    """
    path = Path(path)
    if not path.name.endswith(COMPRESSED_EXT):
        raise ValueError(f"Expected {COMPRESSED_EXT} file, got: {path.name}")

    # Resolve symlinks for reading
    resolved = path.resolve()
    compressed = resolved.read_bytes()
    return load_overlay_shard_bytes(compressed)


def load_overlay_shard_stable(
    path: Path,
    *,
    expected_identity: RegularFileIdentity,
) -> dict[str, t.Tensor]:
    """Load one direct overlay leaf through a descriptor-bound identity fence."""
    path = Path(path)
    if not path.name.endswith(COMPRESSED_EXT):
        raise ValueError(f"Expected {COMPRESSED_EXT} file, got: {path.name}")
    if not hasattr(os, "O_NOFOLLOW"):
        raise CorruptFeatureShardError("stable overlay reads require os.O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FeatureDirectoryMismatchError(
            f"overlay shard changed before its stable read: {path}; {exc}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        opened_identity = RegularFileIdentity.from_stat(opened)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened_identity != expected_identity
        ):
            raise FeatureDirectoryMismatchError(
                f"overlay shard changed before its stable read: {path}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            compressed = handle.read()
        after = os.fstat(descriptor)
        if (
            after.st_nlink != 1
            or RegularFileIdentity.from_stat(after) != expected_identity
            or len(compressed) != expected_identity.size
        ):
            raise FeatureDirectoryMismatchError(
                f"overlay shard changed during its stable read: {path}"
            )
        try:
            linked = path.lstat()
        except OSError as exc:
            raise FeatureDirectoryMismatchError(
                f"overlay shard changed after its stable read: {path}"
            ) from exc
        if linked.st_nlink != 1 or RegularFileIdentity.from_stat(linked) != expected_identity:
            raise FeatureDirectoryMismatchError(
                f"overlay shard changed after its stable read: {path}"
            )
        return load_overlay_shard_bytes(compressed)
    finally:
        os.close(descriptor)


def overlay_content_digest(path: Path) -> str:
    """Hash an overlay's decompressed tensor content deterministically.

    Tensor keys are processed in sorted order and each key contributes its name,
    dtype, shape, and contiguous CPU bytes. This intentionally ignores compressed
    bytes and safetensors metadata ordering, which are not canonical identities.
    """
    return overlay_tensors_content_digest(load_overlay_shard(path))


def overlay_tensors_content_digest(tensors: dict[str, t.Tensor]) -> str:
    """Hash already-loaded overlay tensors using the file digest contract."""
    digest = hashlib.sha256()
    for key in sorted(tensors):
        tensor = tensors[key]
        digest.update(str(key).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.cpu().contiguous().numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_percentile_overlay_shard(path: Path, *, expected_numel: int) -> t.Tensor:
    """Load a percentile payload and enforce its base-derived structure."""
    try:
        data = load_overlay_shard(path)
    except Exception as exc:
        raise CorruptFeatureShardError(
            f"Percentile overlay shard {path} is unreadable "
            f"({type(exc).__name__}: {exc}). Rebuild this overlay shard."
        ) from exc
    expected_key = TensorKey.FEATURE_PERCENTILES.value
    if set(data) != {expected_key}:
        raise CorruptFeatureShardError(
            f"Percentile overlay shard {path} has keys {sorted(data)}, "
            f"expected exactly [{expected_key!r}]. Rebuild this overlay shard."
        )
    values = data[expected_key]
    if values.ndim != 1 or values.dtype != t.uint8 or values.numel() != expected_numel:
        raise CorruptFeatureShardError(
            f"Percentile overlay shard {path} expected a rank-1 torch.uint8 "
            f"{expected_key} tensor with numel={expected_numel}; got "
            f"shape={tuple(values.shape)}, dtype={values.dtype}, "
            f"numel={values.numel()}. Rebuild this overlay shard."
        )
    return values


def validate_percentile_overlay_shard_stable(
    path: Path,
    *,
    expected_numel: int,
    expected_identity: RegularFileIdentity,
) -> t.Tensor:
    """Stable-read variant used by transient remote materializers."""
    try:
        data = load_overlay_shard_stable(path, expected_identity=expected_identity)
    except FeatureDirectoryMismatchError:
        raise
    except Exception as exc:
        raise CorruptFeatureShardError(
            f"Percentile overlay shard {path} is unreadable "
            f"({type(exc).__name__}: {exc}). Rebuild this overlay shard."
        ) from exc
    expected_key = TensorKey.FEATURE_PERCENTILES.value
    if set(data) != {expected_key}:
        raise CorruptFeatureShardError(
            f"Percentile overlay shard {path} has keys {sorted(data)}, "
            f"expected exactly [{expected_key!r}]. Rebuild this overlay shard."
        )
    values = data[expected_key]
    if values.ndim != 1 or values.dtype != t.uint8 or values.numel() != expected_numel:
        raise CorruptFeatureShardError(
            f"Percentile overlay shard {path} expected a rank-1 torch.uint8 "
            f"{expected_key} tensor with numel={expected_numel}; got "
            f"shape={tuple(values.shape)}, dtype={values.dtype}, "
            f"numel={values.numel()}. Rebuild this overlay shard."
        )
    return values


def validate_token_boundary_overlay_shard(
    path: Path, *, expected_n_segments: int
) -> tuple[t.Tensor, t.Tensor]:
    """Load a token-boundary payload and enforce its base-derived structure."""
    try:
        data = load_overlay_shard(path)
    except Exception as exc:
        raise CorruptFeatureShardError(
            f"Token-boundary overlay shard {path} is unreadable "
            f"({type(exc).__name__}: {exc}). Rebuild this overlay shard."
        ) from exc
    start_key = TensorKey.SEGMENT_TOKEN_STARTS.value
    end_key = TensorKey.SEGMENT_TOKEN_ENDS.value
    if set(data) != {start_key, end_key}:
        raise CorruptFeatureShardError(
            f"Token-boundary overlay shard {path} has keys {sorted(data)}, "
            f"expected exactly [{start_key!r}, {end_key!r}]. "
            "Rebuild this overlay shard."
        )
    starts = data[start_key]
    ends = data[end_key]
    for key, values in ((start_key, starts), (end_key, ends)):
        if values.ndim != 1 or values.dtype != t.int32 or values.numel() != expected_n_segments:
            raise CorruptFeatureShardError(
                f"Token-boundary overlay shard {path} expected a rank-1 "
                f"torch.int32 {key} tensor with numel={expected_n_segments}; "
                f"got shape={tuple(values.shape)}, dtype={values.dtype}, "
                f"numel={values.numel()}. Rebuild this overlay shard."
            )
    return starts, ends


def glob_overlay_shards(overlay_dir_path: Path) -> list[Path]:
    """List all ``.safetensors.zst`` files in *overlay_dir_path* (sorted).

    Only returns direct children (not recursive). Includes symlinks.
    """
    if not overlay_dir_path.is_dir():
        return []
    return sorted(overlay_dir_path.glob(f"*{COMPRESSED_EXT}"))


def overlay_shard_path(overlay_dir_path: Path, shard_stem: str) -> Path:
    """Construct the path for an overlay shard file."""
    return overlay_dir_path / f"{shard_stem}{COMPRESSED_EXT}"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def write_manifest(
    overlay_dir_path: Path,
    checksum: str,
    kind: OverlayKind,
    metadata: dict | None = None,
    *,
    base_shard_paths: list[Path] | None = None,
    precomputed_base_hash: str | None = None,
    cache_dir: Path | None = None,
    include_created_at: bool = True,
) -> Path:
    """Write ``manifest.json`` with full checksum and creation metadata.

    When *base_shard_paths* or *precomputed_base_hash* is given, the manifest is
    ``schema_version=2`` and records a ``base_shard_content_hash``.
    *precomputed_base_hash*, when supplied, is validated and stored instead of
    recomputing the hash. Local callers may provide both as an optimization;
    remote callers provide only the precomputed digest because no persistent
    local base-shard paths exist. ``include_created_at=false`` omits the
    non-semantic wall-clock field for deterministic immutable publication.

    *cache_dir*, when set, enables the stat-validated base-shard hash cache at
    ``<cache_dir>/.base_shard_hash_cache.json`` (#530). It is consulted ONLY when
    *base_shard_paths* is given and *precomputed_base_hash* is None (precomputed wins;
    *cache_dir* is then irrelevant). The stored hash value is byte-identical either way.

    Returns:
        Path to the written manifest file.
    """
    if precomputed_base_hash is not None:
        _validate_precomputed_base_hash(
            precomputed_base_hash,
            surface="write_manifest",
        )
    overlay_dir_path.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "checksum": checksum,
        "kind": kind.value,
    }
    if include_created_at:
        manifest["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["checksum_prefix"] = checksum[:CHECKSUM_PREFIX_LEN]
    if metadata:
        manifest["metadata"] = metadata
    if base_shard_paths is not None or precomputed_base_hash is not None:
        manifest["base_shard_content_hash"] = (
            precomputed_base_hash
            if precomputed_base_hash is not None
            else compute_base_shard_content_hash(
                base_shard_paths or [],
                cache_dir=cache_dir,
            )
        )
        manifest["schema_version"] = CONTENT_BOUND_MANIFEST_VERSION

    path = overlay_dir_path / MANIFEST_FILENAME
    # Atomic write via shared helper (tmp.<pid>.<uuid> + fsync + os.replace).
    atomic_write_text(json.dumps(manifest, indent=2) + "\n", path)
    return path


def read_manifest(overlay_dir_path: Path) -> dict | None:
    """Read and parse ``manifest.json``. Returns None if missing or invalid."""
    path = overlay_dir_path / MANIFEST_FILENAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or "checksum" not in data:
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Symlink management
# ---------------------------------------------------------------------------


def create_mirror_symlinks(
    source_dir: Path,
    mirror_dir_path: Path,
    shard_stems: list[str] | None = None,
) -> int:
    """Create per-file symlinks from *mirror_dir_path* to files in *source_dir*.

    Only creates symlinks for ``.safetensors.zst`` files. **Never** creates
    directory symlinks — only individual file symlinks.

    Args:
        source_dir: Directory containing the actual shard files.
        mirror_dir_path: Directory where symlinks will be created.
        shard_stems: If provided, only link these stems. Otherwise link all shards.

    Returns:
        Number of symlinks created.
    """
    mirror_dir_path.mkdir(parents=True, exist_ok=True)

    if shard_stems is not None:
        source_files = [source_dir / f"{stem}{COMPRESSED_EXT}" for stem in shard_stems]
    else:
        source_files = glob_overlay_shards(source_dir)

    created = 0
    for src in source_files:
        if not src.is_file():
            logger.warning(f"Source file not found, skipping symlink: {src}")
            continue
        link_path = mirror_dir_path / src.name
        if link_path.exists() or link_path.is_symlink():
            # Already exists — skip
            continue
        # Use relative symlink for portability
        rel_target = os.path.relpath(src, mirror_dir_path)
        link_path.symlink_to(rel_target)
        created += 1

    return created
