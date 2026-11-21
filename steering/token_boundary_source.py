"""Deterministic compact input contract for remote token-boundary preprocessing."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath
from typing import cast

import torch as t
import zstandard
from safetensors.torch import load as safetensors_load

from nanochat.atomic_io import atomic_write_bytes
from steering._common.strict_json import canonical_json, strict_json_object
from steering.feature_io import (
    COMPRESSED_EXT,
    RegularFileIdentity,
    read_feature_compressed_bytes_stable,
)
from steering.immutable_io import is_sha256_hex
from steering.remote_types import NormalizedRemotePrefix
from steering.safetensors_io import (
    deterministic_safetensors_bytes,
    read_safetensors_metadata_from_bytes,
)

TOKEN_BOUNDARY_SOURCE_SCHEMA_VERSION = 1
TOKEN_BOUNDARY_SOURCE_ZSTD_LEVEL = 12
TOKEN_BOUNDARY_SOURCE_MANIFEST_FILENAME = "manifest.json"
TOKEN_BOUNDARY_SOURCE_DIRNAME = "_token_boundary_source"

_TENSOR_SPECS = {
    "offsets": (t.int64, 1),
    "doc_ids": (t.int32, 1),
    "segment_indices": (t.int32, 1),
    "segment_char_starts": (t.int32, 1),
    "segment_char_ends": (t.int32, 1),
    "doc_token_counts": (t.int32, 1),
    "n_docs": (t.int64, 0),
    "n_segments": (t.int64, 0),
    "context_window": (t.int64, 0),
    "segment_size": (t.int64, 0),
}
TOKEN_BOUNDARY_SOURCE_VECTOR_KEYS = frozenset(
    key for key, (_, ndim) in _TENSOR_SPECS.items() if ndim == 1
)
TOKEN_BOUNDARY_SOURCE_SCALAR_KEYS = frozenset(
    key for key, (_, ndim) in _TENSOR_SPECS.items() if ndim == 0
)
TOKEN_BOUNDARY_SOURCE_KEYS = frozenset(_TENSOR_SPECS)
_SEGMENT_KEYS = (
    "doc_ids",
    "segment_indices",
    "segment_char_starts",
    "segment_char_ends",
)


class TokenBoundarySourceError(ValueError):
    """A compact token-boundary source is malformed or unauthenticated."""


class MissingTokenBoundarySourceError(TokenBoundarySourceError):
    """The derived compact source has not been committed with a manifest."""


@dataclass(frozen=True)
class TokenBoundarySourceManifest:
    base_shard_content_hash: str
    raw_inventory_sha256: str
    schema_version: int
    shard_count: int
    source_inventory_sha256: str


_MANIFEST_KEYS = frozenset(field.name for field in fields(TokenBoundarySourceManifest))


def _validate_sha256(value: object, *, surface: str) -> str:
    if not is_sha256_hex(value):
        raise TokenBoundarySourceError(
            f"{surface} must be exactly 64 lowercase hexadecimal characters"
        )
    return cast(str, value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TokenBoundarySourceError(f"compact source {message}")


def derive_token_boundary_source_prefix(
    raw_prefix: NormalizedRemotePrefix,
    raw_inventory_sha256: str,
) -> NormalizedRemotePrefix:
    """Derive the immutable v1 compact namespace from one raw inventory."""
    digest = _validate_sha256(
        raw_inventory_sha256,
        surface="raw_inventory_sha256",
    )
    return NormalizedRemotePrefix.parse(
        PurePosixPath(
            str(raw_prefix),
            TOKEN_BOUNDARY_SOURCE_DIRNAME,
            f"v{TOKEN_BOUNDARY_SOURCE_SCHEMA_VERSION}",
            digest,
        ).as_posix()
    )


def token_boundary_source_manifest_path(prefix: NormalizedRemotePrefix) -> str:
    return PurePosixPath(
        str(prefix), TOKEN_BOUNDARY_SOURCE_MANIFEST_FILENAME
    ).as_posix()


def token_boundary_source_shard_path(
    prefix: NormalizedRemotePrefix,
    index: int,
) -> str:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"compact source shard index must be a non-negative int, got {index!r}")
    return PurePosixPath(
        str(prefix), f"shard_{index:05d}_features{COMPRESSED_EXT}"
    ).as_posix()


def _require_tensor(
    payload: Mapping[str, object],
    key: str,
    *,
    dtype: t.dtype,
    ndim: int,
) -> t.Tensor:
    value = payload[key]
    if not isinstance(value, t.Tensor):
        raise TokenBoundarySourceError(f"compact source {key} must be a tensor")
    if value.dtype != dtype or value.ndim != ndim:
        raise TokenBoundarySourceError(
            f"compact source {key} must have dtype={dtype} rank={ndim}, "
            f"got dtype={value.dtype} rank={value.ndim}"
        )
    return value


def _tensor_has(predicate: t.Tensor) -> bool:
    return bool(predicate.any().item())


def validate_token_boundary_source_payload(
    payload: Mapping[str, object],
) -> dict[str, t.Tensor]:
    """Validate exactly the ten fields consumed by token-boundary preprocessing."""
    keys = set(payload)
    if keys != TOKEN_BOUNDARY_SOURCE_KEYS:
        raise TokenBoundarySourceError(
            "compact source must contain exactly "
            f"{sorted(TOKEN_BOUNDARY_SOURCE_KEYS)}; got keys={sorted(keys)}"
        )

    tensors = {
        key: _require_tensor(payload, key, dtype=dtype, ndim=ndim)
        for key, (dtype, ndim) in _TENSOR_SPECS.items()
    }
    offsets = tensors["offsets"]
    doc_ids = tensors["doc_ids"]
    segment_indices = tensors["segment_indices"]
    segment_char_starts = tensors["segment_char_starts"]
    segment_char_ends = tensors["segment_char_ends"]
    doc_token_counts = tensors["doc_token_counts"]
    n_docs = int(tensors["n_docs"].item())
    n_segments = int(tensors["n_segments"].item())
    context_window = int(tensors["context_window"].item())
    segment_size = int(tensors["segment_size"].item())
    _require(n_docs >= 0, "n_docs must be non-negative")
    _require(n_segments >= 0, "n_segments must be non-negative")
    _require(context_window > 0, "context_window must be positive")
    _require(segment_size > 0, "segment_size must be positive")

    expected_segment_lengths = {key: tensors[key].numel() for key in _SEGMENT_KEYS}
    mismatched = {
        key: length
        for key, length in expected_segment_lengths.items()
        if length != n_segments
    }
    _require(
        not mismatched,
        f"per-segment lengths must equal n_segments={n_segments}; got {mismatched}",
    )
    _require(
        doc_token_counts.numel() == n_docs,
        f"doc_token_counts length must equal n_docs={n_docs}, "
        f"got {doc_token_counts.numel()}",
    )
    _require(offsets.numel() > 0, "offsets must not be empty")
    _require(
        offsets.numel() == n_segments + 1,
        f"offsets length must equal n_segments + 1; "
        f"expected={n_segments + 1}, got={offsets.numel()}",
    )
    _require(int(offsets[0].item()) == 0, "offsets[0] must equal 0")
    offset_deltas = offsets[1:] - offsets[:-1]
    _require(not _tensor_has(offset_deltas < 0), "offsets must be non-decreasing")
    _require(
        n_segments == 0 or not _tensor_has(offset_deltas == 0),
        "nonempty shards require strictly increasing offsets",
    )
    _require(
        doc_ids.numel() == 0
        or not (_tensor_has(doc_ids < 0) or _tensor_has(doc_ids >= n_docs)),
        "doc_ids must be in [0, n_docs)",
    )
    _require(not _tensor_has(segment_indices < 0), "segment_indices must be non-negative")
    _require(
        not _tensor_has(segment_char_starts < 0),
        "segment_char_starts must be non-negative",
    )
    _require(
        not _tensor_has(segment_char_ends < segment_char_starts),
        "character ends must not precede starts",
    )
    _require(
        not _tensor_has(doc_token_counts < 0),
        "doc_token_counts must be non-negative",
    )

    return {key: tensors[key] for key in sorted(tensors)}


def compact_token_boundary_payload_from_raw(
    raw_data: Mapping[str, object],
) -> dict[str, t.Tensor]:
    """Select and clone the compact payload from one decoded raw feature shard."""
    missing = TOKEN_BOUNDARY_SOURCE_KEYS - set(raw_data)
    if missing:
        raise TokenBoundarySourceError(
            f"decoded raw feature shard lacks compact source fields: {sorted(missing)}"
        )
    selected: dict[str, object] = {
        key: raw_data[key] for key in TOKEN_BOUNDARY_SOURCE_VECTOR_KEYS
    }
    for key in TOKEN_BOUNDARY_SOURCE_SCALAR_KEYS:
        value = raw_data[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TokenBoundarySourceError(
                f"decoded raw feature scalar {key} must be a strict int, got {value!r}"
            )
        selected[key] = t.tensor(value, dtype=t.int64)
    validated = validate_token_boundary_source_payload(selected)
    return validate_token_boundary_source_payload(
        {key: value.clone() for key, value in validated.items()}
    )


def save_token_boundary_source_shard(
    payload: Mapping[str, object],
    path: str | Path,
) -> Path:
    """Write one deterministic empty-metadata, zstd-level-12 compact shard."""
    tensors = validate_token_boundary_source_payload(payload)
    resolved = Path(path)
    if not resolved.name.endswith(COMPRESSED_EXT):
        resolved = resolved.with_name(resolved.name + COMPRESSED_EXT)
    ordered = {key: tensors[key] for key in sorted(tensors)}
    safetensors_bytes = deterministic_safetensors_bytes(ordered, metadata={})
    compressed = zstandard.ZstdCompressor(
        level=TOKEN_BOUNDARY_SOURCE_ZSTD_LEVEL
    ).compress(safetensors_bytes)
    atomic_write_bytes(compressed, resolved)
    return resolved


def decode_token_boundary_source_bytes(compressed: bytes) -> dict[str, t.Tensor]:
    try:
        safetensors_bytes = zstandard.ZstdDecompressor().decompress(compressed)
    except zstandard.ZstdError as exc:
        raise TokenBoundarySourceError("compact source body is not valid zstd") from exc
    try:
        metadata = read_safetensors_metadata_from_bytes(safetensors_bytes)
        tensors = safetensors_load(safetensors_bytes)
    except Exception as exc:
        raise TokenBoundarySourceError("compact source safetensors body is invalid") from exc
    if metadata not in (None, {}):
        raise TokenBoundarySourceError("compact source safetensors metadata must be empty")
    return validate_token_boundary_source_payload(tensors)


def load_token_boundary_source_shard_stable(
    path: str | Path,
    *,
    expected_identity: RegularFileIdentity,
) -> dict[str, t.Tensor]:
    compressed = read_feature_compressed_bytes_stable(
        path,
        expected_identity=expected_identity,
    )
    return decode_token_boundary_source_bytes(compressed)


def assert_token_boundary_source_tensor_equal(
    expected: Mapping[str, t.Tensor],
    observed: Mapping[str, t.Tensor],
) -> None:
    if set(expected) != set(observed):
        raise TokenBoundarySourceError("compact source reload changed its tensor key set")
    for key in sorted(expected):
        if not t.equal(expected[key], observed[key]):
            raise TokenBoundarySourceError(
                f"compact source reload changed tensor {key}"
            )


def serialize_token_boundary_source_manifest(
    manifest: TokenBoundarySourceManifest,
) -> bytes:
    validate_token_boundary_source_manifest(manifest)
    return (canonical_json(asdict(manifest)) + "\n").encode("ascii")


def validate_token_boundary_source_manifest(
    manifest: TokenBoundarySourceManifest,
    *,
    expected_shard_count: int | None = None,
    expected_raw_inventory_sha256: str | None = None,
    expected_source_inventory_sha256: str | None = None,
) -> None:
    _require(
        type(manifest.schema_version) is int
        and manifest.schema_version == TOKEN_BOUNDARY_SOURCE_SCHEMA_VERSION,
        "manifest schema_version must be strict integer 1",
    )
    _require(
        type(manifest.shard_count) is int and manifest.shard_count > 0,
        "manifest shard_count must be a positive strict integer",
    )
    for name in (
        "base_shard_content_hash",
        "raw_inventory_sha256",
        "source_inventory_sha256",
    ):
        _validate_sha256(
            getattr(manifest, name),
            surface=f"compact source manifest {name}",
        )
    _require(
        expected_shard_count is None or manifest.shard_count == expected_shard_count,
        f"manifest shard_count mismatch: expected={expected_shard_count}, "
        f"got={manifest.shard_count}",
    )
    _require(
        expected_raw_inventory_sha256 is None
        or manifest.raw_inventory_sha256 == expected_raw_inventory_sha256,
        "manifest raw inventory digest mismatch",
    )
    _require(
        expected_source_inventory_sha256 is None
        or manifest.source_inventory_sha256 == expected_source_inventory_sha256,
        "manifest compact inventory digest mismatch",
    )


def parse_token_boundary_source_manifest(
    payload: bytes,
    *,
    expected_shard_count: int | None = None,
    expected_raw_inventory_sha256: str | None = None,
    expected_source_inventory_sha256: str | None = None,
) -> TokenBoundarySourceManifest:
    try:
        raw = json.loads(payload, object_pairs_hook=strict_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise TokenBoundarySourceError(
            "compact source manifest is not valid UTF-8 JSON"
        ) from exc
    keys = set(raw) if isinstance(raw, Mapping) else set()
    if not isinstance(raw, Mapping) or keys != _MANIFEST_KEYS:
        raise TokenBoundarySourceError(
            "compact source manifest must contain exactly "
            f"{sorted(_MANIFEST_KEYS)}; got keys={sorted(keys, key=str)}"
        )
    manifest = TokenBoundarySourceManifest(**raw)
    validate_token_boundary_source_manifest(
        manifest,
        expected_shard_count=expected_shard_count,
        expected_raw_inventory_sha256=expected_raw_inventory_sha256,
        expected_source_inventory_sha256=expected_source_inventory_sha256,
    )
    if serialize_token_boundary_source_manifest(manifest) != payload:
        raise TokenBoundarySourceError(
            "compact source manifest must use canonical JSON serialization"
        )
    return manifest
