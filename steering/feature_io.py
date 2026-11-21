"""I/O utilities for SAE feature shard files.

Provides compressed read/write/glob for feature shards using the
``.safetensors.zst`` format with sort + delta-encoding + zstd compression.

Generic write path:
    1. Sort ``feature_ids`` ascending within each segment (co-sort magnitudes).
    2. Delta-encode sorted IDs within each segment.
    3. Apply dtype optimization (uint16 IDs, float16 mags, int32 offsets).
    4. Serialize to safetensors bytes.
    5. zstd-compress and write as ``.safetensors.zst``.

``save_features_presorted`` is a narrower writer for derived shards whose
segments are already sorted; it validates that precondition and then skips the
generic per-segment sort.

Read path:
    1. Read ``.safetensors.zst`` bytes.
    2. zstd-decompress.
    3. Parse safetensors from bytes and upcast dtypes.
    4. Delta-decode ``feature_ids`` to restore original values.

Glob:
    - ``glob_features`` discovers ``.safetensors.zst`` shards.
"""

import io
import json
import os
import stat
import struct
import tempfile
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import BinaryIO, Protocol, TypedDict

import torch as t
import zstandard
from loguru import logger
from safetensors import safe_open
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save
from torch import Tensor

from errors import EmptyExtractionSegmentError, ValidationError
from nanochat.atomic_io import atomic_write_bytes
from steering.sae_resolver import L0Tier
from steering.safetensors_io import deterministic_safetensors_bytes


class FeatureDirectoryMismatchError(Exception):
    """Raised when feature file contents don't match their directory path."""


class CorruptFeatureShardError(ValueError):
    """A feature shard file is unreadable: not zstd, truncated, an
    impossible/oversized safetensors header, or a non-JSON header.

    Subclasses ``ValueError`` so existing ``except ValueError`` /
    ``except (FeatureDirectoryMismatchError, ValueError)`` handlers keep
    catching it (``json.JSONDecodeError`` — the exception it replaces — is
    itself a ``ValueError``)."""


class TensorKey(StrEnum):
    """Keys for tensor fields stored in feature shard files."""

    OFFSETS = "offsets"
    FEATURE_IDS = "feature_ids"
    FEATURE_MAGS = "feature_mags"
    FEATURE_PERCENTILES = "feature_percentiles"
    DOC_IDS = "doc_ids"
    SEGMENT_INDICES = "segment_indices"
    SEGMENT_POSITIONS = "segment_positions"
    SEGMENT_CHAR_STARTS = "segment_char_starts"
    SEGMENT_CHAR_ENDS = "segment_char_ends"
    SEGMENT_TOKEN_STARTS = "segment_token_starts"
    SEGMENT_TOKEN_ENDS = "segment_token_ends"
    DOC_TOKEN_COUNTS = "doc_token_counts"


class MetadataKey(StrEnum):
    """Keys for scalar metadata stored in the safetensors header."""

    N_DOCS = "n_docs"
    N_SEGMENTS = "n_segments"
    N_SAE_FEATURES = "n_sae_features"
    LAYER = "layer"
    CONTEXT_WINDOW = "context_window"
    POOLING = "pooling"
    SEGMENT_MODE = "segment_mode"
    SEGMENT_SIZE = "segment_size"
    PERCENTILE_CHECKSUM = "percentile_checksum"
    TOKENIZER_CHECKSUM = "tokenizer_checksum"
    CONTEXT_INDEPENDENT_FEATURES = "context_independent_features"
    L0_TIER = "l0_tier"  # NEW: gemma-scope-2 L0Tier value, or absent for v1.
    IGNORE_BOS_FEATURES = "ignore_bos_features"  # bool; absent → False (legacy)
    EXTRACTION_BACKEND = "extraction_backend"  # concrete backend; absent → unknown
    EXTRACTION_MODEL_ID = "extraction_model_id"
    EXTRACTION_SAE_RELEASE = "extraction_sae_release"
    EXTRACTION_SAE_ID = "extraction_sae_id"
    EXTRACTION_DOCUMENTS_SHA256 = "extraction_documents_sha256"


class ExtractionBackend(StrEnum):
    """Concrete activation implementation recorded in a feature shard."""

    nnterp = "nnterp"
    native = "native"
    transformerlens = "transformerlens"


def extraction_backend_from_metadata(
    metadata: Mapping[str, object], *, source: str | Path,
) -> ExtractionBackend | None:
    """Parse provenance without assigning a backend to a legacy shard."""
    if MetadataKey.EXTRACTION_BACKEND not in metadata:
        return None
    raw = metadata[MetadataKey.EXTRACTION_BACKEND]
    if isinstance(raw, str):
        try:
            return ExtractionBackend(raw)
        except ValueError:
            pass
    raise ValueError(
        f"Invalid extraction_backend {raw!r} in {source}; expected one of "
        f"{[backend.value for backend in ExtractionBackend]}. "
        "Legacy unknown provenance must omit the key; 'auto' is not provenance."
    )


@dataclass(frozen=True)
class ExtractionBackendProvenance:
    """One consistent known backend, plus explicitly counted unknown shards."""

    backend: ExtractionBackend | None
    unknown_shards: int
    n_shards: int

    def warn_if_incomplete(self, source: str | Path) -> None:
        if self.unknown_shards:
            logger.warning(
                f"Incomplete extraction_backend provenance in {source}: "
                f"{self.unknown_shards}/{self.n_shards} shards have unknown backend; "
                f"known backend={self.backend or 'none'}. "
                "Backend uniformity cannot be certified for the legacy shards; "
                "their metadata will not be backfilled."
            )


def summarize_extraction_backends(
    metadata_by_path: Iterable[tuple[Path, Mapping[str, object]]],
) -> ExtractionBackendProvenance:
    """Reject conflicting known values; allow unknown legacy provenance."""
    seen = [
        (path, extraction_backend_from_metadata(metadata, source=path))
        for path, metadata in metadata_by_path
    ]
    known = {backend for _, backend in seen if backend is not None}
    if len(known) > 1:
        details = "\n".join(
            f"  - {path}: extraction_backend={backend or 'unknown'}"
            for path, backend in sorted(seen, key=lambda item: str(item[0]))
        )
        raise FeatureDirectoryMismatchError(
            f"extraction_backend disagrees across shards:\n{details}"
        )
    return ExtractionBackendProvenance(
        backend=next(iter(known), None),
        unknown_shards=sum(backend is None for _, backend in seen),
        n_shards=len(seen),
    )


def read_extraction_backend_provenance(
    shard_paths: Iterable[str | Path],
) -> ExtractionBackendProvenance:
    """Inspect only shard headers, including every shard in a resume directory."""
    return summarize_extraction_backends(
        (Path(path), read_shard_metadata(path)) for path in shard_paths
    )


def assert_extraction_backend_compatible(
    feature_dir: str | Path, backend: ExtractionBackend,
) -> None:
    """Guard append/resume against known conflicts without rewriting legacy data.

    This is a preflight check, not a lock against independent concurrent writers.
    """
    provenance = read_extraction_backend_provenance(glob_features(feature_dir))
    if provenance.backend is not None and provenance.backend != backend:
        raise FeatureDirectoryMismatchError(
            f"extraction_backend mismatch in {feature_dir}: "
            f"existing={provenance.backend}, new={backend}; use a separate output_dir."
        )
    provenance.warn_if_incomplete(feature_dir)


_TENSOR_KEYS = frozenset(TensorKey)
_METADATA_KEYS = frozenset(MetadataKey)

_INT_METADATA = frozenset(
    {
        MetadataKey.N_DOCS,
        MetadataKey.N_SEGMENTS,
        MetadataKey.N_SAE_FEATURES,
        MetadataKey.LAYER,
        MetadataKey.CONTEXT_WINDOW,
        MetadataKey.SEGMENT_SIZE,
    }
)

COMPRESSED_EXT = ".safetensors.zst"
ZSTD_LEVEL = 12
# zstandard compression levels are documented in [1, 22]; level 0 is invalid
# (the library raises a `ZstdError` rather than silently using a default).
# Surface a clean `ValueError` from the wrapper to give callers a useful
# trace.
_MIN_ZSTD_LEVEL = 1
_MAX_ZSTD_LEVEL = 22

# fp16 max is 65504.0; we guard below that with margin so callers get a loud
# failure before silent saturation to +Inf on disk. If a model genuinely
# produces activations at this scale, switch feature_mags storage to fp32.
FP16_SAFE_MAX = 60000.0

# Upper bound on safetensors header size we are willing to allocate. The
# real header is typically a few KB; a corrupted shard claiming 2**62 bytes
# would otherwise OOM the process. 100 MB is a generous cap (any genuine
# header above that is itself a bug).
MAX_SAFETENSORS_HEADER_SIZE = 100 * 1024 * 1024


@dataclass(frozen=True)
class RegularFileIdentity:
    """Descriptor-bindable identity for one direct regular file."""

    mode: int
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def capture(cls, path: str | Path) -> "RegularFileIdentity":
        resolved = Path(path)
        result = resolved.lstat()
        if not stat.S_ISREG(result.st_mode):
            raise FeatureDirectoryMismatchError(
                f"feature shard must be a direct regular file, got {resolved}"
            )
        return cls(
            mode=int(result.st_mode),
            device=int(result.st_dev),
            inode=int(result.st_ino),
            size=int(result.st_size),
            mtime_ns=int(result.st_mtime_ns),
            ctime_ns=int(result.st_ctime_ns),
        )

    @classmethod
    def from_stat(cls, result: os.stat_result) -> "RegularFileIdentity":
        return cls(
            mode=int(result.st_mode),
            device=int(result.st_dev),
            inode=int(result.st_ino),
            size=int(result.st_size),
            mtime_ns=int(result.st_mtime_ns),
            ctime_ns=int(result.st_ctime_ns),
        )

    def with_verified_size(
        self,
        *,
        verified_byte_length: int,
        path: str | Path,
    ) -> "RegularFileIdentity":
        """Rebind the size field to a full-read-verified byte length.

        Lustre can transiently report a stale st_size for a recently written
        file, and stable reads verify strict identity equality, so an identity
        captured right after publication must carry the length the verification
        pass actually read, never the bare stat size.
        """
        if self.size == verified_byte_length:
            return self
        logger.warning(
            f"published file stat size {self.size} disagrees with verified byte "
            f"length {verified_byte_length}; binding the verified length: {path}"
        )
        return replace(self, size=verified_byte_length)

    @classmethod
    def from_verified_publication(
        cls,
        result: os.stat_result,
        *,
        verified_byte_length: int,
        path: str | Path,
    ) -> "RegularFileIdentity":
        """Identity for a just-published file whose content a full read verified."""
        return cls.from_stat(result).with_verified_size(
            verified_byte_length=verified_byte_length,
            path=path,
        )

    def to_json(self, *, name: str) -> dict[str, int | str]:
        return {
            "name": name,
            "mode": self.mode,
            "device": self.device,
            "inode": self.inode,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
        }


@dataclass(frozen=True)
class EncodedPercentileFeatures:
    """Storage-width tensors required to compute one percentile overlay."""

    offsets: Tensor
    feature_id_deltas: Tensor
    feature_magnitudes: Tensor
    metadata: dict[str, object]

    @property
    def n_values(self) -> int:
        """Number of encoded feature values in the shard."""
        return int(self.feature_id_deltas.numel())


class _ProfileDetailSink(Protocol):
    def time_detail(self, key: str) -> AbstractContextManager[None]: ...


def save_features(
    data: dict,
    path: str | Path,
    *,
    zstd_level: int | None = None,
) -> Path:
    """Save a feature shard dict to ``.safetensors.zst`` with compression.

    Pipeline:
        1. Sort feature_ids ascending within each segment, co-sort magnitudes
           (and percentiles if present).
        2. Delta-encode sorted feature_ids.
        3. Apply dtype optimization (uint16 IDs, float16 mags, uint8 pcts, int32 offsets).
        4. Serialize to safetensors bytes with metadata.
        5. zstd-compress and write to disk.

    Args:
        data: Shard dict with the standard keys (offsets, feature_ids, etc.).
        path: Output path. Extension is forced to ``.safetensors.zst``.
        zstd_level: Compression level (1-22). Defaults to module-level ``ZSTD_LEVEL``.

    Returns:
        The actual path written (with ``.safetensors.zst`` extension).

    Raises:
        ValueError: If ``zstd_level`` is outside the valid zstd range
            ``[1, 22]`` or if ``feature_mags`` saturates fp16.
        ValidationError: If a positive ``feature_mags`` value would underflow
            to zero in fp16.
    """
    _validate_zstd_level("save_features", zstd_level)
    path = _feature_output_path(path)

    offsets = data["offsets"]
    fids = data["feature_ids"]
    mags = data["feature_mags"]

    _assert_no_empty_segments("save_features", offsets)

    # Sort within segments and delta-encode (co-sort percentiles if present)
    extra_arrays: list[t.Tensor] = []
    has_pcts = TensorKey.FEATURE_PERCENTILES in data and isinstance(
        data[TensorKey.FEATURE_PERCENTILES], t.Tensor
    )
    if has_pcts:
        extra_arrays.append(data[TensorKey.FEATURE_PERCENTILES])

    sort_result = _sort_within_segments(fids, mags, offsets, *extra_arrays)
    sorted_fids = sort_result[0]
    sorted_mags = sort_result[1]
    delta_fids = _delta_encode(sorted_fids, offsets)

    # Build modified data dict
    modified_data = dict(data)
    modified_data["feature_ids"] = delta_fids
    modified_data["feature_mags"] = sorted_mags
    if has_pcts:
        modified_data[TensorKey.FEATURE_PERCENTILES] = sort_result[2]

    return _write_feature_payload(modified_data, path, zstd_level)


def save_features_presorted(
    data: dict,
    path: str | Path,
    *,
    zstd_level: int | None = None,
    zstd_threads: int = 0,
    profile: _ProfileDetailSink | None = None,
    deterministic: bool = False,
) -> Path:
    """Save an already feature-id-sorted shard without the generic per-segment sort.

    This is intended for post-extraction writers that build shards from already
    sorted source segments. It preserves :func:`save_features`'s on-disk format
    and loader behavior, but it validates that every segment is non-empty and
    sorted ascending before using the faster path. General extraction writers
    should keep using :func:`save_features`.

    Remote publication opts into ``deterministic`` to reproduce the exact bytes
    pinned by immutable upload claims across interrupted runs. The default keeps
    the ordinary writer's serialization behavior.
    ``zstd_threads=0`` retains single-threaded compression; positive values opt
    into zstd worker threads, which preserve decoded values but can change bytes.
    """
    _validate_zstd_level("save_features_presorted", zstd_level)
    if isinstance(zstd_threads, bool) or not isinstance(zstd_threads, int) or zstd_threads < 0:
        raise ValueError("save_features_presorted: zstd_threads must be an integer >= 0")
    path = _feature_output_path(path)

    offsets = data["offsets"]
    fids = data["feature_ids"]
    _assert_no_empty_segments("save_features_presorted", offsets)
    _assert_segments_presorted(fids, offsets)

    modified_data = dict(data)
    if profile is None:
        modified_data["feature_ids"] = _delta_encode_presorted(fids, offsets)
    else:
        with profile.time_detail("save.delta_encode"):
            modified_data["feature_ids"] = _delta_encode_presorted(fids, offsets)
    return _write_feature_payload(
        modified_data, path, zstd_level, profile=profile, deterministic=deterministic,
        zstd_threads=zstd_threads,
    )


def _profile_elapsed(profile: dict[str, float | int] | None, key: str, started: float) -> None:
    if profile is not None:
        profile[key] = float(profile.get(key, 0.0)) + time.perf_counter() - started


def _profile_cpu_elapsed(profile: dict[str, float] | None, key: str, started: float) -> None:
    if profile is not None:
        profile[key] = float(profile.get(key, 0.0)) + time.thread_time() - started


def _decompress_feature_safetensors_bytes(compressed: bytes) -> bytes:
    """Decompress a feature payload whether or not its frame declares its size."""
    frame_size = zstandard.frame_content_size(compressed)
    if frame_size in {zstandard.CONTENTSIZE_UNKNOWN, -1}:
        with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(compressed)) as reader:
            return reader.read()
    return zstandard.ZstdDecompressor().decompress(compressed)


def load_features(
    path: str | Path,
    *,
    keys: set[str] | None = None,
    profile: dict[str, float | int] | None = None,
    profile_cpu: dict[str, float] | None = None,
) -> dict:
    """Load a compressed feature shard from ``.safetensors.zst``.

    Pipeline:
        1. Read and zstd-decompress bytes.
        2. Parse safetensors from bytes.
        3. Upcast tensors to original dtypes.
        4. Delta-decode feature_ids to restore original values.

    Args:
        keys: When given, only these **tensor** fields are upcast/delta-decoded and returned;
            the unrequested tensors (typically the large ``feature_ids``/``feature_mags``) are
            skipped, saving the dtype upcast + the ``feature_ids`` cumsum. Scalar metadata is
            always parsed. ``feature_ids`` delta-decoding needs ``offsets`` — request both or
            neither. A requested key absent from the shard's tensors fails loud.

    Returns:
        Dict with tensor and scalar metadata fields.
    """
    return decode_feature_safetensors_bytes(
        load_feature_safetensors_bytes(
            path,
            profile=profile,
            profile_cpu=profile_cpu,
        ),
        keys=keys,
        profile=profile,
        profile_cpu=profile_cpu,
    )


def load_feature_safetensors_bytes(
    path: str | Path,
    *,
    profile: dict[str, float | int] | None = None,
    profile_cpu: dict[str, float] | None = None,
) -> bytes:
    """Read and zstd-decompress one shard into plain safetensors bytes."""
    path = Path(path)
    if not path.name.endswith(COMPRESSED_EXT):
        raise ValueError(f"Unknown feature file format: {path.name!r} (expected {COMPRESSED_EXT})")
    started = time.perf_counter() if profile is not None else 0.0
    cpu_started = time.thread_time() if profile_cpu is not None else 0.0
    compressed = path.read_bytes()
    _profile_elapsed(profile, "compressed_read", started)
    _profile_cpu_elapsed(profile_cpu, "compressed_read", cpu_started)
    if profile is not None:
        profile["compressed_bytes"] = len(compressed)
    started = time.perf_counter() if profile is not None else 0.0
    cpu_started = time.thread_time() if profile_cpu is not None else 0.0
    decompressed = _decompress_feature_safetensors_bytes(compressed)
    _profile_elapsed(profile, "zstd_decompress", started)
    _profile_cpu_elapsed(profile_cpu, "zstd_decompress", cpu_started)
    if profile is not None:
        profile["decompressed_bytes"] = len(decompressed)
    return decompressed


def load_features_stable(
    path: str | Path,
    *,
    expected_identity: RegularFileIdentity,
    keys: set[str] | None = None,
    profile: dict[str, float | int] | None = None,
    profile_cpu: dict[str, float] | None = None,
) -> dict:
    """Load one shard through the exact regular file sealed by an inventory.

    The opened descriptor, complete read, and final pathname are all required
    to retain ``expected_identity``.  This is intentionally separate from the
    legacy symlink-transparent :func:`load_features` API.
    """
    return decode_feature_safetensors_bytes(
        load_feature_safetensors_bytes_stable(
            path,
            expected_identity=expected_identity,
            profile=profile,
            profile_cpu=profile_cpu,
        ),
        keys=keys,
        profile=profile,
        profile_cpu=profile_cpu,
    )


def load_feature_safetensors_bytes_stable(
    path: str | Path,
    *,
    expected_identity: RegularFileIdentity,
    profile: dict[str, float | int] | None = None,
    profile_cpu: dict[str, float] | None = None,
) -> bytes:
    """Stable-read and zstd-decompress one shard into safetensors bytes."""
    started = time.perf_counter() if profile is not None else 0.0
    cpu_started = time.thread_time() if profile_cpu is not None else 0.0
    compressed = read_feature_compressed_bytes_stable(
        path,
        expected_identity=expected_identity,
    )
    _profile_elapsed(profile, "compressed_read", started)
    _profile_cpu_elapsed(profile_cpu, "compressed_read", cpu_started)
    if profile is not None:
        profile["compressed_bytes"] = len(compressed)
    started = time.perf_counter() if profile is not None else 0.0
    cpu_started = time.thread_time() if profile_cpu is not None else 0.0
    decompressed = _decompress_feature_safetensors_bytes(compressed)
    _profile_elapsed(profile, "zstd_decompress", started)
    _profile_cpu_elapsed(profile_cpu, "zstd_decompress", cpu_started)
    if profile is not None:
        profile["decompressed_bytes"] = len(decompressed)
    return decompressed


def read_feature_compressed_bytes_stable(
    path: str | Path,
    *,
    expected_identity: RegularFileIdentity,
) -> bytes:
    """Return exact compressed bytes through a descriptor-sealed regular file read."""
    path = Path(path)
    if not path.name.endswith(COMPRESSED_EXT):
        raise ValueError(f"Unknown feature file format: {path.name!r} (expected {COMPRESSED_EXT})")
    observed_before = RegularFileIdentity.capture(path)
    if observed_before != expected_identity:
        raise FeatureDirectoryMismatchError(
            f"feature shard changed before stable read: {path}; "
            f"expected={expected_identity!r}, observed={observed_before!r}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if opened.st_nlink != 1 or RegularFileIdentity.from_stat(opened) != expected_identity:
            raise FeatureDirectoryMismatchError(
                f"feature shard changed while being opened for stable read: {path}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            compressed = handle.read()
        after_read = os.fstat(descriptor)
        if (
            after_read.st_nlink != 1
            or RegularFileIdentity.from_stat(after_read) != expected_identity
            or len(compressed) != expected_identity.size
        ):
            raise FeatureDirectoryMismatchError(f"feature shard changed during stable read: {path}")
    finally:
        os.close(descriptor)
    observed_after = RegularFileIdentity.capture(path)
    if observed_after != expected_identity:
        raise FeatureDirectoryMismatchError(
            f"feature shard pathname changed after stable read: {path}"
        )
    return compressed


def decode_feature_safetensors_bytes(
    sf_bytes: bytes,
    *,
    keys: set[str] | None = None,
    profile: dict[str, float | int] | None = None,
    profile_cpu: dict[str, float] | None = None,
) -> dict:
    """Parse, restore, and delta-decode plain safetensors shard bytes."""
    # Load tensors from bytes
    started = time.perf_counter() if profile is not None else 0.0
    cpu_started = time.thread_time() if profile_cpu is not None else 0.0
    tensors = safetensors_load(sf_bytes)
    _profile_elapsed(profile, "safetensors_deserialize", started)
    _profile_cpu_elapsed(profile_cpu, "safetensors_deserialize", cpu_started)

    # Read metadata from the raw safetensors header
    started = time.perf_counter() if profile is not None else 0.0
    cpu_started = time.thread_time() if profile_cpu is not None else 0.0
    metadata = _read_safetensors_metadata_from_bytes(sf_bytes)
    _profile_elapsed(profile, "metadata_parse", started)
    _profile_cpu_elapsed(profile_cpu, "metadata_parse", cpu_started)

    if keys is not None:
        missing = keys - set(tensors)
        if missing:
            raise ValueError(
                f"load_features: requested keys not present in shard tensors: {sorted(missing)}"
            )
        # feature_ids are delta-encoded against offsets; the decode below needs both. Requesting
        # feature_ids without offsets would silently return raw deltas — reject it loudly.
        if "feature_ids" in keys and "offsets" not in keys:
            raise ValueError(
                "load_features: keys requesting 'feature_ids' must also include 'offsets' "
                "(feature_ids are delta-encoded and undecodable without offsets)"
            )

    # Upcast tensors (only the requested subset when ``keys`` is given)
    started = time.perf_counter() if profile is not None else 0.0
    cpu_started = time.thread_time() if profile_cpu is not None else 0.0
    result: dict = {}
    for key, tensor in tensors.items():
        if keys is not None and key not in keys:
            continue
        result[key] = _upcast(key, tensor)
    _profile_elapsed(profile, "tensor_upcast", started)
    _profile_cpu_elapsed(profile_cpu, "tensor_upcast", cpu_started)

    # Delta-decode feature_ids
    if "feature_ids" in result and "offsets" in result:
        started = time.perf_counter() if profile is not None else 0.0
        cpu_started = time.thread_time() if profile_cpu is not None else 0.0
        feature_ids = result["feature_ids"]
        if feature_ids is tensors["feature_ids"]:
            feature_ids = feature_ids.clone()
        result["feature_ids"] = _delta_decode_inplace(feature_ids, result["offsets"])
        _profile_elapsed(profile, "feature_id_delta_decode", started)
        _profile_cpu_elapsed(profile_cpu, "feature_id_delta_decode", cpu_started)

    started = time.perf_counter() if profile is not None else 0.0
    cpu_started = time.thread_time() if profile_cpu is not None else 0.0
    _add_parsed_metadata(result, metadata)
    _profile_elapsed(profile, "metadata_parse", started)
    _profile_cpu_elapsed(profile_cpu, "metadata_parse", cpu_started)
    return result


def decode_percentile_feature_safetensors_bytes(sf_bytes: bytes) -> dict:
    """Decode only the tensors required to build a percentile overlay.

    Feature IDs retain the generic loader's restored integer dtype, but are
    delta-decoded in one writable restored-dtype buffer instead of cloning an
    already-upcast tensor. Feature magnitudes intentionally retain their exact
    storage dtype; the percentile lookup converts them to float64 on its current
    device.
    """
    tensors = safetensors_load(sf_bytes)
    metadata = _read_safetensors_metadata_from_bytes(sf_bytes)
    del sf_bytes

    required = {
        TensorKey.OFFSETS.value,
        TensorKey.FEATURE_IDS.value,
        TensorKey.FEATURE_MAGS.value,
    }
    missing = required - set(tensors)
    if missing:
        raise ValueError(
            "decode_percentile_feature_safetensors_bytes: required tensor(s) "
            f"missing from feature shard: {sorted(missing)}"
        )

    raw_feature_ids = tensors.pop(TensorKey.FEATURE_IDS.value)
    feature_magnitudes = tensors.pop(TensorKey.FEATURE_MAGS.value)
    offsets = _upcast(
        TensorKey.OFFSETS.value,
        tensors.pop(TensorKey.OFFSETS.value),
    )
    del tensors

    feature_ids = _upcast(TensorKey.FEATURE_IDS.value, raw_feature_ids)
    if feature_ids is raw_feature_ids:
        feature_ids = raw_feature_ids.clone()
    del raw_feature_ids
    _delta_decode_inplace(feature_ids, offsets)

    result = {
        TensorKey.OFFSETS.value: offsets,
        TensorKey.FEATURE_IDS.value: feature_ids,
        TensorKey.FEATURE_MAGS.value: feature_magnitudes,
    }
    _add_parsed_metadata(result, metadata)
    return result


def parse_percentile_feature_safetensors_bytes(
    sf_bytes: bytes,
) -> EncodedPercentileFeatures:
    """Parse storage-width percentile inputs without delta-decoding IDs."""
    tensors = safetensors_load(sf_bytes)
    raw_metadata = _read_safetensors_metadata_from_bytes(sf_bytes)
    del sf_bytes

    required = {
        TensorKey.OFFSETS.value,
        TensorKey.FEATURE_IDS.value,
        TensorKey.FEATURE_MAGS.value,
    }
    missing = required - set(tensors)
    if missing:
        raise ValueError(
            "parse_percentile_feature_safetensors_bytes: required tensor(s) "
            f"missing from feature shard: {sorted(missing)}"
        )

    feature_id_deltas = tensors.pop(TensorKey.FEATURE_IDS.value)
    feature_magnitudes = tensors.pop(TensorKey.FEATURE_MAGS.value)
    offsets = _upcast(
        TensorKey.OFFSETS.value,
        tensors.pop(TensorKey.OFFSETS.value),
    )
    del tensors

    metadata: dict[str, object] = {}
    _add_parsed_metadata(metadata, raw_metadata)
    return EncodedPercentileFeatures(
        offsets=offsets,
        feature_id_deltas=feature_id_deltas,
        feature_magnitudes=feature_magnitudes,
        metadata=metadata,
    )


class SpooledFeatureShard:
    """Low-memory view over one compressed feature shard.

    The shard body is decompressed to a temporary plain safetensors file by
    :func:`spooled_feature_shard`; tensor reads then go through
    ``safetensors.safe_open`` so callers can request bounded slices instead of
    materialising the whole shard.
    """

    def __init__(self, *, path: Path, temp_path: Path, reader) -> None:
        self.path = path
        self.temp_path = temp_path
        self._reader = reader
        self.tensor_keys = {str(key) for key in reader.keys()}
        self.raw_metadata = dict(reader.metadata() or {})

    def require_tensor_keys(self, keys: set[str]) -> None:
        missing = set(keys) - self.tensor_keys
        if missing:
            raise ValueError(
                "spooled_feature_shard: requested keys not present in shard "
                f"tensors: {sorted(missing)}"
            )

    def parsed_metadata(self) -> dict:
        result: dict = {}
        _add_parsed_metadata(result, self.raw_metadata)
        return result

    def get_tensor(self, key: str | TensorKey) -> Tensor:
        name = str(key)
        self.require_tensor_keys({name})
        tensor = _upcast(name, self._reader.get_tensor(name)).clone()
        if name == str(TensorKey.FEATURE_IDS):
            self.require_tensor_keys({str(TensorKey.OFFSETS)})
            offsets = _upcast(
                TensorKey.OFFSETS,
                self._reader.get_tensor(str(TensorKey.OFFSETS)),
            ).clone()
            if int(tensor.ndim) != 1 or int(offsets.ndim) != 1:
                raise ValueError(
                    "spooled_feature_shard: feature_ids and offsets must be 1-D "
                    "to decode feature IDs"
                )
            if int(offsets.numel()) == 0 or int(offsets[0].item()) != 0:
                raise ValueError("spooled_feature_shard: offsets must be non-empty and start at 0")
            if bool((offsets[1:] < offsets[:-1]).any()):
                raise ValueError(
                    "spooled_feature_shard: offsets must be monotonic to decode feature IDs"
                )
            if int(offsets[-1].item()) != int(tensor.numel()):
                raise ValueError(
                    "spooled_feature_shard: offsets[-1] must match feature_ids "
                    f"length ({int(offsets[-1].item())} != {int(tensor.numel())})"
                )
            tensor = _delta_decode(tensor, offsets)
        return tensor

    def tensor_shape(self, key: str | TensorKey) -> list[int]:
        name = str(key)
        self.require_tensor_keys({name})
        return list(self._reader.get_slice(name).get_shape())

    def get_tensor_slice(self, key: str | TensorKey, start: int, end: int) -> Tensor:
        """Read a storage-order tensor slice.

        ``feature_ids`` slices are still delta-encoded because decoding requires
        segment-boundary context from ``offsets``. Callers that slice
        ``feature_ids`` must decode the selected segment span themselves.
        """
        name = str(key)
        self.require_tensor_keys({name})
        if start < 0 or end < start:
            raise ValueError(f"spooled_feature_shard: invalid slice for {name}: {start}:{end}")
        shape = self.tensor_shape(name)
        if len(shape) != 1:
            raise ValueError(
                "spooled_feature_shard: get_tensor_slice only supports 1-D "
                f"tensors, got {name} shape={shape}"
            )
        if end > shape[0]:
            raise ValueError(
                "spooled_feature_shard: slice out of bounds for "
                f"{name}: {start}:{end} with length {shape[0]}"
            )
        sliced = self._reader.get_slice(name)[start:end]
        return _upcast(name, sliced).clone()


@contextmanager
def spooled_feature_shard(
    path: str | Path,
    *,
    temp_dir: str | Path | None = None,
) -> Iterator[SpooledFeatureShard]:
    """Open a compressed feature shard through a temporary safetensors spool.

    This trades temporary disk for lower peak RAM. The existing
    :func:`load_features` eager reader remains the reference implementation.
    """
    path = Path(path)
    if not path.name.endswith(COMPRESSED_EXT):
        raise ValueError(f"Unknown feature file format: {path.name!r} (expected {COMPRESSED_EXT})")
    parent = Path(temp_dir) if temp_dir is not None else path.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.spool.",
            suffix=".safetensors",
            dir=parent,
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            dctx = zstandard.ZstdDecompressor()
            with open(path, "rb") as src:
                with dctx.stream_reader(src) as reader:
                    while True:
                        chunk = reader.read(16 * 1024 * 1024)
                        if not chunk:
                            break
                        tmp.write(chunk)
        with safe_open(str(tmp_path), framework="pt", device="cpu") as reader:
            yield SpooledFeatureShard(path=path, temp_path=tmp_path, reader=reader)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _add_parsed_metadata(result: dict, metadata: dict[str, str]) -> None:
    # Parse scalar metadata from strings
    for key in _METADATA_KEYS:
        if key in metadata:
            result[key] = _parse_metadata_value(key, metadata[key])

    # context_independent_features: the _METADATA_KEYS loop above already
    # parsed this as a bool via _parse_metadata_value when present. Legacy
    # shards lack the key → default False so callers never need to
    # distinguish "absent" vs "false".
    if MetadataKey.CONTEXT_INDEPENDENT_FEATURES not in result:
        result[MetadataKey.CONTEXT_INDEPENDENT_FEATURES] = False

    # Legacy shards lack ignore_bos_features → default False so callers never
    # need to distinguish "absent" vs "false".
    if MetadataKey.IGNORE_BOS_FEATURES not in result:
        result[MetadataKey.IGNORE_BOS_FEATURES] = False

    # Expose checksums if present (not in _METADATA_KEYS to avoid
    # breaking existing load paths, but included for callers that need it)
    if MetadataKey.PERCENTILE_CHECKSUM in metadata:
        result[MetadataKey.PERCENTILE_CHECKSUM] = metadata[MetadataKey.PERCENTILE_CHECKSUM]
    if MetadataKey.TOKENIZER_CHECKSUM in metadata:
        result[MetadataKey.TOKENIZER_CHECKSUM] = metadata[MetadataKey.TOKENIZER_CHECKSUM]


def glob_features(directory: str | Path) -> list[Path]:
    """Find all compressed feature shard files in *directory*.

    Returns:
        Sorted list of paths matching ``*_features.safetensors.zst``.
    """
    directory = Path(directory)
    return sorted(directory.glob(f"*_features{COMPRESSED_EXT}"))


def feature_path(directory: str | Path, shard_name: str) -> Path:
    """Construct the output path for a new feature shard (compressed)."""
    return Path(directory) / f"{shard_name}_features{COMPRESSED_EXT}"


def feature_path_exists(directory: str | Path, shard_name: str) -> bool:
    """Check if a compressed feature shard exists."""
    d = Path(directory)
    return (d / f"{shard_name}_features{COMPRESSED_EXT}").is_file()


def shard_stem_from_path(path: str | Path) -> str:
    """Extract shard stem from a feature file path.

    ``shard_00001_features.safetensors.zst`` -> ``shard_00001``
    """
    name = Path(path).name
    # Strip known extensions (check longest first)
    for ext in (COMPRESSED_EXT, ".safetensors", ".pt"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return name.replace("_features", "")


def empty_segment_indices(offsets: Tensor) -> list[int]:
    """Return indices ``i`` where ``offsets[i] == offsets[i+1]`` — i.e. a base
    segment with zero extracted features.

    The base-extraction CSR contract (issue #522) is that every emitted
    segment of a shard with ``n_segments > 0`` is non-empty. A whole-shard
    ``n_segments == 0`` (``offsets == [0]``) is exempt and yields ``[]``.

    Single source of truth shared by the writer (save_features), the
    validators (shard_validation), the load contract (FeatureIndex), and the
    base-reading producers. It lives here in the leaf I/O module so
    save_features can use it without a circular import.
    """
    if offsets.numel() <= 1:
        return []
    diffs = offsets[1:] - offsets[:-1]
    return (diffs == 0).nonzero(as_tuple=True)[0].tolist()


def _validate_zstd_level(caller: str, zstd_level: int | None) -> None:
    if zstd_level is not None and not (_MIN_ZSTD_LEVEL <= zstd_level <= _MAX_ZSTD_LEVEL):
        raise ValueError(
            f"{caller}: zstd_level {zstd_level} is outside the valid "
            f"range [{_MIN_ZSTD_LEVEL}, {_MAX_ZSTD_LEVEL}] (zstandard library limit)"
        )


def _feature_output_path(path: str | Path) -> Path:
    path = Path(path)
    name = path.name
    for ext in (COMPRESSED_EXT, ".safetensors", ".pt"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    if not name.endswith("_features"):
        name = name + "_features"
    return path.parent / (name + COMPRESSED_EXT)


def _assert_no_empty_segments(caller: str, offsets: Tensor) -> None:
    # Base-extraction invariant (issue #522): no emitted segment may be empty.
    # Downstream sparse formats that can legitimately contain empty data use
    # separate overlay/selection writers.
    empties = empty_segment_indices(offsets)
    if empties:
        i = empties[0]
        raise EmptyExtractionSegmentError(
            f"{caller}: segment {i} has zero features "
            f"(offsets[{i}]==offsets[{i + 1}]={int(offsets[i].item())}); refusing "
            f"to persist an empty base segment — see issue #522."
        )


def _assert_segments_presorted(feature_ids: Tensor, offsets: Tensor) -> None:
    n_segs = offsets.numel() - 1
    for i in range(n_segs):
        s, e = int(offsets[i].item()), int(offsets[i + 1].item())
        if e - s < 2:
            continue
        seg = feature_ids[s:e]
        if bool((seg[1:] < seg[:-1]).any().item()):
            raise ValueError(
                f"save_features_presorted: segment {i} feature_ids are not sorted ascending."
            )


def _write_feature_payload(
    data: dict,
    path: Path,
    zstd_level: int | None,
    *,
    profile: _ProfileDetailSink | None = None,
    deterministic: bool = False,
    zstd_threads: int = 0,
) -> Path:
    extraction_backend_from_metadata(data, source=path)
    serialize = deterministic_safetensors_bytes if deterministic else safetensors_save

    def _assemble() -> tuple[dict[str, t.Tensor], dict[str, str]]:
        tensors: dict[str, t.Tensor] = {}
        metadata: dict[str, str] = {}

        for key, value in data.items():
            if key in _TENSOR_KEYS and isinstance(value, t.Tensor):
                optimized = _optimize_dtype(key, value)
                if key == TensorKey.FEATURE_MAGS:
                    positive_underflow = (value > 0) & (optimized == 0)
                    if bool(positive_underflow.any().item()):
                        count = int(positive_underflow.sum().item())
                        raise ValidationError(
                            f"save_features: {path}: {count} positive feature_mags "
                            "underflow during float16 conversion to zero; refusing "
                            "to write an invalid shard."
                        )
                tensors[key] = optimized
            elif isinstance(value, bool):
                # bools must be encoded before the str() call below to avoid
                # "True"/"False" casing drift; the loader expects "true"/"false".
                metadata[key] = "true" if value else "false"
            else:
                metadata[key] = str(value)
        return tensors, metadata

    if profile is None:
        tensors, metadata = _assemble()
        sf_bytes = serialize(tensors, metadata=metadata)
        cctx = zstandard.ZstdCompressor(
            level=zstd_level if zstd_level is not None else ZSTD_LEVEL, threads=zstd_threads
        )
        compressed = cctx.compress(sf_bytes)
    else:
        with profile.time_detail("save.assemble_payload"):
            tensors, metadata = _assemble()
        with profile.time_detail("save.safetensors_serialize"):
            sf_bytes = serialize(tensors, metadata=metadata)
        with profile.time_detail("save.zstd_compress"):
            cctx = zstandard.ZstdCompressor(
                level=zstd_level if zstd_level is not None else ZSTD_LEVEL, threads=zstd_threads
            )
            compressed = cctx.compress(sf_bytes)

    # Atomic write: tmp + fsync + os.replace via shared helper. A crash
    # mid-write can never leave a torn final shard that glob_features would
    # pick up; at worst a ``.tmp.<pid>.<uuid>`` file is orphaned.
    if profile is None:
        atomic_write_bytes(compressed, path)
    else:
        with profile.time_detail("save.atomic_write"):
            atomic_write_bytes(compressed, path)
    return path


# ---------------------------------------------------------------------------
# Sort and delta-encoding helpers
# ---------------------------------------------------------------------------


def _sort_within_segments(
    feature_ids: Tensor,
    feature_mags: Tensor,
    offsets: Tensor,
    *extra_arrays: Tensor,
) -> tuple[Tensor, ...]:
    """Sort feature_ids ascending within each segment, co-sort magnitudes and extras.

    Args:
        feature_ids: Flat 1-D tensor of feature indices.
        feature_mags: Flat 1-D tensor of activation magnitudes (same length).
        offsets: 1-D tensor of segment boundary offsets (length = n_segments + 1).
        *extra_arrays: Additional flat 1-D tensors to co-sort (same length as feature_ids).

    Returns:
        Tuple of (sorted_feature_ids, sorted_feature_mags, *sorted_extras).
    """
    sorted_ids = feature_ids.clone()
    sorted_mags = feature_mags.clone()
    sorted_extras = [arr.clone() for arr in extra_arrays]
    n_segs = offsets.numel() - 1

    for i in range(n_segs):
        s, e = offsets[i].item(), offsets[i + 1].item()
        if e - s < 2:
            continue
        seg_ids = feature_ids[s:e]
        order = seg_ids.argsort()
        sorted_ids[s:e] = seg_ids[order]
        sorted_mags[s:e] = feature_mags[s:e][order]
        for j, arr in enumerate(extra_arrays):
            sorted_extras[j][s:e] = arr[s:e][order]

    return (sorted_ids, sorted_mags, *sorted_extras)


def _delta_encode(sorted_ids: Tensor, offsets: Tensor) -> Tensor:
    """Delta-encode sorted feature IDs within each segment.

    The first ID in each segment is kept as-is. Subsequent IDs become
    deltas from their predecessor: ``[id0, id1-id0, id2-id1, ...]``.

    Args:
        sorted_ids: Feature IDs already sorted ascending within segments.
        offsets: Segment boundary offsets (length = n_segments + 1).

    Returns:
        Delta-encoded tensor (same shape and dtype as input).
    """
    delta = sorted_ids.clone()
    n_segs = offsets.numel() - 1

    for i in range(n_segs):
        s, e = offsets[i].item(), offsets[i + 1].item()
        if e - s < 2:
            continue
        delta[s + 1 : e] = sorted_ids[s + 1 : e] - sorted_ids[s : e - 1]

    return delta


def _delta_encode_presorted(sorted_ids: Tensor, offsets: Tensor) -> Tensor:
    """Vectorized delta encoding for ids already sorted within each segment."""
    delta = sorted_ids.clone()
    if delta.numel() < 2:
        return delta
    delta[1:].sub_(sorted_ids[:-1])
    starts = offsets[:-1].to(t.int64)
    if starts.numel() > 0:
        delta[starts] = sorted_ids[starts]
    return delta


def _delta_decode(delta_ids: Tensor, offsets: Tensor) -> Tensor:
    """Reverse delta encoding using cumulative sum within each segment.

    Args:
        delta_ids: Delta-encoded feature IDs.
        offsets: Segment boundary offsets (length = n_segments + 1).

    Returns:
        Reconstructed feature IDs (sorted ascending within segments).
    """
    return _delta_decode_inplace(delta_ids.clone(), offsets)


def decode_feature_id_deltas(delta_ids: Tensor, offsets: Tensor) -> Tensor:
    """Restore one complete-segment span into a writable integer buffer."""
    feature_ids = _upcast(TensorKey.FEATURE_IDS.value, delta_ids)
    if feature_ids is delta_ids:
        feature_ids = delta_ids.clone()
    return _delta_decode_inplace(feature_ids, offsets)


def _delta_decode_inplace(ids: Tensor, offsets: Tensor) -> Tensor:
    """Delta-decode IDs in their existing writable result buffer."""
    # Avoid two Tensor.item() dispatches for every segment.
    boundaries = offsets.tolist()
    for s, e in pairwise(boundaries):
        if e - s < 2:
            continue
        ids[s:e] = ids[s:e].cumsum(0)

    return ids


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _optimize_dtype(key: str, tensor: t.Tensor) -> t.Tensor:
    """Downcast a tensor for compact storage."""
    if key == TensorKey.FEATURE_IDS:
        if tensor.numel() == 0 or tensor.max().item() <= 65535:
            return tensor.to(t.uint16)
        return tensor  # keep int32 for large SAE widths
    if key == TensorKey.FEATURE_MAGS:
        max_abs = tensor.abs().max().item() if tensor.numel() > 0 else 0.0
        if max_abs >= FP16_SAFE_MAX:
            raise ValueError(
                f"save_features: feature_mags max abs value {max_abs:.1f} "
                f"exceeds fp16 safe threshold {FP16_SAFE_MAX}; would saturate "
                f"to +Inf on disk. Consider storing as fp32."
            )
        return tensor.to(t.float16)
    if key == TensorKey.FEATURE_PERCENTILES:
        return tensor.to(t.uint8)
    if key == TensorKey.OFFSETS:
        if tensor.numel() == 0 or tensor.max().item() <= 2**31 - 1:
            return tensor.to(t.int32)
        return tensor  # keep int64 for very large shards
    return tensor


def _upcast(key: str, tensor: t.Tensor) -> t.Tensor:
    """Restore tensor to the dtype expected by downstream code."""
    if key == TensorKey.FEATURE_IDS and tensor.dtype in (t.uint16, t.int16):
        return tensor.to(t.int32)
    if key == TensorKey.FEATURE_MAGS and tensor.dtype == t.float16:
        return tensor.to(t.float32)
    if key == TensorKey.FEATURE_PERCENTILES:
        return tensor  # keep as uint8 — caller decodes with .float() / 255.0
    if key == TensorKey.OFFSETS and tensor.dtype == t.int32:
        return tensor.to(t.int64)
    return tensor


class _Readable(Protocol):
    """Structural protocol for objects with a ``read(n)`` method."""

    def read(self, n: int, /) -> bytes: ...


ShardHeader = TypedDict(
    "ShardHeader",
    {"__metadata__": dict[str, str], "__tensor_keys__": set[str]},
)


def _read_exact(reader: _Readable, n: int) -> bytes:
    """Read exactly ``n`` bytes from ``reader``, looping until ``n`` bytes
    are accumulated or ``reader.read`` returns empty (true EOF).

    Decouples header parsing from whether the underlying decompression
    reader returns ``n`` or fewer bytes per call (python-zstandard's
    documented contract permits short reads even though 0.25.0 does not).
    Never swallows exceptions — a ``zstandard.ZstdError`` from ``read``
    propagates to ``read_shard_header``'s single error boundary. The caller
    decides whether a short result (``len < n``) is fatal. A non-positive
    ``n`` returns ``b""`` (the loop body never runs).
    """
    buf = bytearray()
    while len(buf) < n:
        chunk = reader.read(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return bytes(buf)


def _read_shard_header_from_file(path: Path, handle: BinaryIO) -> ShardHeader:
    """Parse one compressed shard header from an already-open file."""
    parsed = _read_raw_shard_header_from_file(path, handle)
    raw_metadata = parsed.pop("__metadata__", {})
    if not isinstance(raw_metadata, dict):
        raise CorruptFeatureShardError(
            f"read_shard_header: shard {path} has a non-dict __metadata__ "
            f"(got {type(raw_metadata).__name__}). Likely a corrupted shard."
        )
    metadata: dict[str, str] = {}
    for k, v in raw_metadata.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise CorruptFeatureShardError(
                f"read_shard_header: shard {path} has a non-string "
                f"__metadata__ entry ({k!r}: {v!r}). Likely a corrupted shard."
            )
        metadata[k] = v
    tensor_keys: set[str] = {str(k) for k in parsed}
    return {"__metadata__": metadata, "__tensor_keys__": tensor_keys}


def _read_raw_shard_header_from_file(
    path: Path,
    handle: BinaryIO,
) -> dict[str, object]:
    """Parse and return the complete safetensors JSON header object."""
    dctx = zstandard.ZstdDecompressor()
    try:
        reader = dctx.stream_reader(handle)
        size_bytes = _read_exact(reader, 8)
        if len(size_bytes) < 8:
            raise CorruptFeatureShardError(
                f"read_shard_header: shard {path} is empty or truncated "
                f"(got {len(size_bytes)} of 8 safetensors header-length "
                f"bytes). Likely a corrupted or partially written shard."
            )
        # Slice to 8: for conforming readers len==8 (no-op); a hypothetical
        # over-reading reader is then surfaced as the loud "ended mid-header"
        # CorruptFeatureShardError downstream, never a bare struct.error.
        header_size = struct.unpack("<Q", size_bytes[:8])[0]
        if header_size == 0:
            raise CorruptFeatureShardError(
                f"read_shard_header: shard {path} declares a zero-length "
                f"safetensors header. Likely a corrupted shard."
            )
        if header_size > MAX_SAFETENSORS_HEADER_SIZE:
            raise CorruptFeatureShardError(
                f"read_shard_header: refusing to allocate {header_size} "
                f"bytes for safetensors header (cap: "
                f"{MAX_SAFETENSORS_HEADER_SIZE}). This usually means the "
                f"shard at {path} is corrupted."
            )
        header_bytes = _read_exact(reader, header_size)
    except zstandard.ZstdError as exc:
        raise CorruptFeatureShardError(
            f"read_shard_header: shard {path} is not a readable zstd stream "
            f"(decode failed: {exc}). Likely a corrupted or truncated shard."
        ) from exc

    if len(header_bytes) < header_size:
        raise CorruptFeatureShardError(
            f"read_shard_header: shard {path} ended mid-header "
            f"(got {len(header_bytes)} of {header_size} declared header "
            f"bytes). Likely a truncated shard."
        )
    try:
        parsed = json.loads(header_bytes)
    except json.JSONDecodeError as exc:
        raise CorruptFeatureShardError(
            f"read_shard_header: shard {path} has a non-JSON safetensors "
            f"header ({exc}). Likely a corrupted shard."
        ) from exc
    if not isinstance(parsed, dict):
        raise CorruptFeatureShardError(
            f"read_shard_header: shard {path} has a non-object safetensors "
            f"header (got {type(parsed).__name__}). Likely a corrupted shard."
        )
    return parsed


def read_feature_value_count(path: str | Path) -> int:
    """Stream one raw shard header and return its rank-1 feature-ID length."""
    resolved = Path(path)
    with resolved.open("rb") as handle:
        parsed = _read_raw_shard_header_from_file(resolved, handle)
    feature_ids = parsed.get(TensorKey.FEATURE_IDS.value)
    if not isinstance(feature_ids, dict):
        raise CorruptFeatureShardError(
            f"feature shard {resolved} has no feature_ids tensor descriptor"
        )
    shape = feature_ids.get("shape")
    if not isinstance(shape, list) or len(shape) != 1 or type(shape[0]) is not int or shape[0] < 0:
        raise CorruptFeatureShardError(
            f"feature shard {resolved} has malformed feature_ids shape {shape!r}"
        )
    return shape[0]


def read_shard_header(path: str | Path) -> ShardHeader:
    """Read metadata and tensor keys from a shard via streaming decompression.

    Decompresses only enough bytes to parse the safetensors header (typically
    a few KB), never the multi-GB body. Behavior-identical to a full
    decompress for valid shards; any unreadable shard (not zstd, truncated,
    impossible/oversized header, non-JSON header, non-object header,
    non-dict __metadata__, or non-string __metadata__ entry) raises
    :class:`CorruptFeatureShardError` naming the shard path — it is never
    silently treated as a metadata-less shard.

    Returns:
        ``{"__metadata__": <scalar metadata dict>,
           "__tensor_keys__": <set of tensor names, excluding __metadata__>}``.
    """
    path = Path(path)
    with open(path, "rb") as handle:
        return _read_shard_header_from_file(path, handle)


def read_shard_header_stable(
    path: str | Path,
    *,
    expected_identity: RegularFileIdentity,
) -> ShardHeader:
    """Read a header through the exact regular file sealed by an inventory."""
    path = Path(path)
    observed_before = RegularFileIdentity.capture(path)
    if observed_before != expected_identity:
        raise FeatureDirectoryMismatchError(
            f"feature shard changed before stable header read: {path}; "
            f"expected={expected_identity!r}, observed={observed_before!r}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        if RegularFileIdentity.from_stat(os.fstat(descriptor)) != expected_identity:
            raise FeatureDirectoryMismatchError(
                f"feature shard changed while opening stable header: {path}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            header = _read_shard_header_from_file(path, handle)
        if RegularFileIdentity.from_stat(os.fstat(descriptor)) != expected_identity:
            raise FeatureDirectoryMismatchError(
                f"feature shard changed during stable header read: {path}"
            )
    finally:
        os.close(descriptor)
    observed_after = RegularFileIdentity.capture(path)
    if observed_after != expected_identity:
        raise FeatureDirectoryMismatchError(
            f"feature shard pathname changed after stable header read: {path}"
        )
    return header


def read_shard_metadata(path: str | Path) -> dict[str, str]:
    """Stream only the safetensors header and return its scalar metadata.

    Reads orders of magnitude less than :func:`load_features` (header only,
    never the multi-GB body). A valid shard whose header carries no
    ``__metadata__`` returns ``{}``; an *unreadable* shard (corrupt,
    truncated, empty, non-zstd) raises :class:`CorruptFeatureShardError`
    rather than returning empty metadata, so a corrupt shard can never be
    silently mistaken for a metadata-less (v1) shard by the l0-tier guards.
    """
    return read_shard_header(path)["__metadata__"]


def _read_safetensors_metadata_from_bytes(sf_bytes: bytes) -> dict[str, str]:
    """Read metadata header from raw safetensors bytes."""
    header_size = struct.unpack("<Q", sf_bytes[:8])[0]
    if header_size > MAX_SAFETENSORS_HEADER_SIZE:
        raise ValueError(
            f"_read_safetensors_metadata_from_bytes: refusing to allocate "
            f"{header_size} bytes for safetensors header "
            f"(cap: {MAX_SAFETENSORS_HEADER_SIZE}). Likely a corrupted shard."
        )
    header_bytes = sf_bytes[8 : 8 + header_size]
    header = json.loads(header_bytes)
    return header.get("__metadata__", {})


def _parse_metadata_value(key: str, raw: str) -> int | str | bool:
    """Convert a string-encoded metadata value back to its proper type."""
    if key in _INT_METADATA:
        return int(raw)
    if key == MetadataKey.CONTEXT_INDEPENDENT_FEATURES:
        return raw.lower() == "true"
    if key == MetadataKey.IGNORE_BOS_FEATURES:
        return raw.lower() == "true"
    if key == MetadataKey.EXTRACTION_BACKEND:
        backend = extraction_backend_from_metadata({key: raw}, source="shard metadata")
        assert backend is not None
        return backend.value
    return raw


def read_l0_tier_from_shard(path: str | Path) -> L0Tier | None:
    """Return the L0Tier recorded in a shard's metadata, or None for v1 shards.

    Raises:
        ValueError: if the metadata value is present but not a recognized
            L0Tier name.
    """
    resolved = Path(path)
    return _l0_tier_from_metadata(resolved, read_shard_metadata(resolved))


def _l0_tier_from_metadata(
    path: Path,
    metadata: Mapping[str, str],
) -> L0Tier | None:
    raw = metadata.get(MetadataKey.L0_TIER)
    if raw is None:
        return None
    try:
        return L0Tier(raw)
    except ValueError as exc:
        raise ValueError(
            f"Unknown L0Tier {raw!r} in shard metadata of {path}. "
            f"Valid: {[t.value for t in L0Tier]}."
        ) from exc


def assert_uniform_l0_tier_metadata(
    metadata_by_path: Iterable[tuple[Path, Mapping[str, str]]],
) -> L0Tier | None:
    """Apply the shared L0 rule to caller-supplied, identity-bound metadata."""
    seen: dict[Path, L0Tier | None] = {}
    for path, metadata in metadata_by_path:
        seen[path] = _l0_tier_from_metadata(path, metadata)
    if not seen:
        return None
    unique = set(seen.values())
    if len(unique) > 1:
        lines = ["l0_tier disagrees across shards:"]
        for path, tier in sorted(seen.items(), key=lambda item: str(item[0])):
            lines.append(f"  - {path}: l0_tier={getattr(tier, 'value', tier)}")
        raise FeatureDirectoryMismatchError("\n".join(lines))
    return next(iter(unique))


def assert_uniform_l0_tier(
    shard_paths: Iterable[str | Path],
) -> L0Tier | None:
    """Return the single L0 tier shared by every shard, or None if all v1.

    Reads each shard's header and applies the same parser as
    :func:`read_l0_tier_from_shard` (the enum form — never the raw-str
    ``load_features`` value). This is the single cross-shard l0-consistency
    rule, consumed by both
    :func:`steering.feature_paths.verify_feature_dir` and
    ``steering.compute_statistics.compute_statistics``.

    Args:
        shard_paths: Shard file paths (e.g. ``glob_features(feature_dir)``).

    Returns:
        The agreed ``L0Tier`` if every shard carries the same v2 tier,
        ``None`` if every shard is v1 (no tier) or the iterable is empty.

    Raises:
        FeatureDirectoryMismatchError: If shards disagree (any two distinct
            tiers, including v1/None vs a v2 tier). The message lists every
            shard path and its observed tier.
        ValueError: Propagated from ``read_l0_tier_from_shard`` if a shard's
            metadata holds an unrecognized tier value.
    """
    paths = [Path(raw_path) for raw_path in shard_paths]
    return assert_uniform_l0_tier_metadata((path, read_shard_metadata(path)) for path in paths)


def assert_l0_tier_compatible(
    feature_dir: str | Path,
    new_l0_tier: L0Tier | None,
) -> None:
    """Refuse to write into a feature directory whose existing shards have
    a different L0 tier than ``new_l0_tier``.

    Args:
        feature_dir: Directory the caller is about to write into.
        new_l0_tier: The tier being written, or ``None`` for v1 (no tier).

    Raises:
        FeatureDirectoryMismatchError: If any existing shard disagrees with
            ``new_l0_tier``. The message lists every shard that mismatched.
    """
    feature_dir = Path(feature_dir)
    if not feature_dir.is_dir():
        return
    existing = glob_features(feature_dir)
    if not existing:
        return
    mismatches: list[str] = []
    for shard_path in existing:
        observed = read_l0_tier_from_shard(shard_path)
        if observed != new_l0_tier:
            if observed is None and new_l0_tier is not None:
                kind = "v1 (no l0_tier) → v2"
            elif observed is not None and new_l0_tier is None:
                kind = "v2 → v1 (no l0_tier)"
            else:
                kind = "tier mismatch"
            mismatches.append(
                f"{shard_path.name}: l0_tier "
                f"existing={getattr(observed, 'value', observed)} "
                f"new={getattr(new_l0_tier, 'value', new_l0_tier)} ({kind})"
            )
    if mismatches:
        raise FeatureDirectoryMismatchError(
            f"l0_tier mismatch in {feature_dir}:\n" + "\n".join(f"  - {m}" for m in mismatches)
        )


def read_ignore_bos_from_shard(path: str | Path) -> bool:
    """Return the shard's ignore_bos_features flag.

    Absent on a *valid* shard (legacy shards) reads as ``False`` — unlike
    l0_tier, "absent" is not a distinct unknown state: a legacy shard and an
    explicit ``false`` shard are semantically identical, so they must never
    trip a uniformity mismatch.

    Streams only the safetensors header via :func:`read_shard_metadata`
    (never the multi-GB body), exactly like :func:`read_l0_tier_from_shard`.
    An *unreadable* shard (corrupt, truncated, empty, non-zstd) raises
    :class:`CorruptFeatureShardError` (a ``ValueError``) naming the shard —
    a corrupt shard is never silently mistaken for a ``false`` / legacy
    shard, which would otherwise hide corruption behind a wrong uniformity
    result.
    """
    resolved = Path(path)
    return _ignore_bos_from_metadata(resolved, read_shard_metadata(resolved))


def _ignore_bos_from_metadata(
    path: Path,
    metadata: Mapping[str, str],
) -> bool:
    raw = metadata.get(MetadataKey.IGNORE_BOS_FEATURES)
    if raw is None:
        return False
    if raw == "true":
        return True
    if raw == "false":
        return False
    raise ValueError(
        f"Invalid ignore_bos_features {raw!r} in shard metadata of {path}; "
        f"expected 'true' or 'false'."
    )


def assert_uniform_ignore_bos_metadata(
    metadata_by_path: Iterable[tuple[Path, Mapping[str, str]]],
) -> bool:
    """Apply the shared BOS rule to caller-supplied, identity-bound metadata."""
    seen: dict[Path, bool] = {}
    for path, metadata in metadata_by_path:
        seen[path] = _ignore_bos_from_metadata(path, metadata)
    if not seen:
        return False
    unique = set(seen.values())
    if len(unique) > 1:
        lines = ["ignore_bos_features disagrees across shards:"]
        for path, value in sorted(seen.items(), key=lambda item: str(item[0])):
            lines.append(f"  - {path}: ignore_bos_features={value}")
        raise FeatureDirectoryMismatchError("\n".join(lines))
    return next(iter(unique))


def assert_uniform_ignore_bos(shard_paths: Iterable[str | Path]) -> bool:
    """Return the single ignore_bos_features value shared by every shard.

    Returns ``False`` for an empty iterable or an all-legacy directory. Raises
    :class:`FeatureDirectoryMismatchError` if shards disagree. Mirrors
    :func:`assert_uniform_l0_tier`; consumed by
    ``steering.compute_statistics.compute_statistics`` and
    :func:`steering.feature_paths.verify_feature_dir`.
    """
    paths = [Path(raw_path) for raw_path in shard_paths]
    return assert_uniform_ignore_bos_metadata((path, read_shard_metadata(path)) for path in paths)


def warn_ignore_bos_cfg_key_ignored(supplied: object, *, resolved: bool) -> None:
    """Loudly log (non-fatal) that a present-but-dead ``ignore_bos_features``
    config/CLI key was IGNORED.

    BOS-feature exclusion is no longer a configurable knob: it is frozen
    always-on at extraction (``extract_features.main``) and inherited from the
    base shards by every downstream producer. The single shared message keeps
    the two callers (extraction freeze + gradient overlay producer) from
    drifting. Callers do the struct-mode-safe ``"ignore_bos_features" in cfg``
    presence check and pass the supplied value + the value actually resolved.
    """
    logger.error(
        "ignore_bos_features is no longer a configurable option (resolved to "
        f"{resolved}); the value supplied via config/CLI ({supplied!r}) is "
        "IGNORED. Remove the key from your config / command line."
    )
