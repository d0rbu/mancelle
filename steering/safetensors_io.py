"""Deterministic safetensors serialization for content-addressed artifacts.

The safetensors Rust serializer writes its JSON header through a hash map.  The
tensor payload is stable, but metadata key order can therefore differ between
processes.  That is harmless for ordinary model files and harmful for artifacts
whose complete bytes are hashed or compared during distributed resume.

Callers must opt in explicitly. Legacy writers continue to use
``atomic_safetensors_save`` unless their artifact contract explicitly requires
canonical bytes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import torch as t
from safetensors.torch import load, save

from steering.immutable_io import stream_hash_file_descriptor_bound


def _strict_header_object(payload: bytes, *, label: str) -> tuple[dict[str, object], int]:
    if len(payload) < 8:
        raise ValueError(f"{label} is too short to contain a header")
    header_length = int.from_bytes(payload[:8], byteorder="little", signed=False)
    header_end = 8 + header_length
    if header_length <= 0 or header_end > len(payload):
        raise ValueError(f"{label} has an invalid header length")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        header = json.loads(
            payload[8:header_end].decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} has malformed header JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"{label} has a non-object header")
    return header, header_end


def canonicalize_safetensors_bytes(payload: bytes) -> bytes:
    """Return ``payload`` with a strict, sorted, minimally encoded JSON header.

    Tensor data offsets in safetensors are relative to the data section. The
    data section is therefore copied byte-for-byte; only the JSON header and
    its required eight-byte padding are rebuilt.
    """

    header, header_end = _strict_header_object(payload, label="safetensors payload")
    canonical_header = json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    canonical_header += b" " * ((-len(canonical_header)) % 8)
    return (
        len(canonical_header).to_bytes(8, byteorder="little", signed=False)
        + canonical_header
        + payload[header_end:]
    )


def deterministic_safetensors_bytes(
    tensors: Mapping[str, t.Tensor],
    *,
    metadata: Mapping[str, str] | None = None,
) -> bytes:
    """Return safetensors bytes with a canonical JSON header.

    Safetensors data offsets are relative to the data section, so sorting and
    repadding only the header preserves the tensor payload and its offsets while
    making the complete file byte-stable across processes.
    """

    encoded = bytes(save(dict(tensors), metadata=None if metadata is None else dict(metadata)))
    try:
        return canonicalize_safetensors_bytes(encoded)
    except ValueError as exc:
        raise ValueError(f"safetensors serializer returned invalid bytes: {exc}") from exc


def read_safetensors_metadata_from_bytes(payload: bytes) -> dict[str, str] | None:
    """Return the ``__metadata__`` mapping parsed from an in-memory snapshot.

    The bytes-based sibling of ``safe_open(path).metadata()``.  A caller that
    has already read and digest-verified a file must parse it from *those same
    bytes*: reopening the path issues an independent read, so the digest it
    verified and the metadata it acts on can come from different file contents.
    That hash-one-snapshot / parse-another split is a defect this codebase has
    hit repeatedly, and it is why this function takes ``bytes`` and never a
    ``Path``.

    Validation mirrors ``deterministic_safetensors_bytes`` on the write side, so
    a reader cannot silently accept a header the writer would never emit.

    Returns ``None`` when the header carries no ``__metadata__`` section at all,
    matching ``safe_open(...).metadata()`` and keeping that case distinguishable
    from a present-but-empty mapping.  Collapsing the two would turn a precise
    "this file has no metadata" diagnostic into a generic key-set mismatch.
    """

    header, _header_end = _strict_header_object(payload, label="safetensors payload")
    if "__metadata__" not in header:
        return None
    metadata = header["__metadata__"]
    if not isinstance(metadata, dict):
        raise ValueError("safetensors payload has a non-object __metadata__ section")
    if any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()
    ):
        raise ValueError("safetensors payload metadata keys and values must be strings")
    return metadata


def load_safetensors_file_snapshot(
    path: Path,
) -> tuple[dict[str, t.Tensor], dict[str, str] | None]:
    """Load tensors and metadata from one descriptor-bound file snapshot.

    The path must name a stable regular file (not a symlink). Both the tensor
    decoder and metadata parser consume the exact bytes that were read and
    mutation-checked through one open file descriptor.
    """

    digest = stream_hash_file_descriptor_bound(path, retain_payload=True)
    payload = digest.payload
    if payload is None:  # pragma: no cover - guaranteed by retain_payload=True
        raise RuntimeError("descriptor-bound safetensors read did not retain its payload")
    metadata = read_safetensors_metadata_from_bytes(payload)
    tensors = dict(load(payload))
    return tensors, metadata


__all__ = [
    "canonicalize_safetensors_bytes",
    "deterministic_safetensors_bytes",
    "load_safetensors_file_snapshot",
    "read_safetensors_metadata_from_bytes",
]
