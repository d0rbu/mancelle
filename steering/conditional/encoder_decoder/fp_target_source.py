"""Integrity certificate for packed-row feature-prediction target sources.

Compact FP targets consume raw packed CSR feature IDs.  Unlike the legacy
full-width indexed-assignment target, cluster consolidation intentionally maps
distinct siblings to one representative and max-reduces them.  Exact duplicate
*original* IDs in a source segment are therefore corruption, not another value
to reduce.  This module certifies that invariant once over an immutable shard
family and binds the result to the exact compressed bytes in its packed manifest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch as t
from beartype import beartype

from errors import PackedRowInvariantError
from nanochat.atomic_io import atomic_write_text
from steering._common.strict_json import canonical_json
from steering.conditional.splice_counterfactual import _fold_file_digests, packed_row_fingerprint
from steering.immutable_io import is_sha256_hex
from steering.overlay_io import MANIFEST_FILENAME, glob_overlay_shards, read_manifest
from steering.packed_batch_io import PackedRowShard, load_packed_row_shard
from steering.refined_types import PositiveInt

FP_TARGET_SOURCE_INTEGRITY_KEY = "fp_target_source_integrity"
FP_TARGET_SOURCE_INTEGRITY_VERSION = 1

_CERTIFICATE_KEYS = frozenset(
    {
        "semantic_version",
        "n_sae_features",
        "packed_row_fingerprint",
        "ordered_shard_inventory",
        "ordered_shard_content_aggregate_sha256",
        "rows_checked",
        "segments_checked",
        "feature_entries_checked",
        "csr_structure_valid",
        "bounds_valid",
        "unique_original_ids_per_segment",
        "scan_complete",
    }
)
_INVENTORY_ENTRY_KEYS = frozenset({"name", "byte_length"})
_EMPTY_INVENTORY_FINGERPRINT = hashlib.sha256().hexdigest()


def fp_target_source_certificate_digest(certificate: Mapping[str, Any]) -> str:
    """Return the resume identity of one strictly parsed certificate."""
    parsed = _parse_certificate(certificate, surface="FP source certificate digest")
    return hashlib.sha256(canonical_json(parsed).encode("utf-8")).hexdigest()


def _ordered_inventory(packed_row_dir: Path) -> list[dict[str, Any]]:
    paths = glob_overlay_shards(packed_row_dir)
    entries: list[dict[str, Any]] = []
    previous_name: str | None = None
    for path in paths:
        if previous_name is not None and path.name <= previous_name:
            raise PackedRowInvariantError(
                "packed-row shard inventory must have unique, strictly sorted names"
            )
        previous_name = path.name
        stat_result = path.stat()
        if not path.is_file():
            raise PackedRowInvariantError(f"packed-row shard is not a regular file: {path}")
        entries.append({"name": path.name, "byte_length": int(stat_result.st_size)})
    return entries


def _packed_source_fingerprint(packed_row_dir: Path) -> str:
    """Fingerprint a packed shard family, including the valid empty family."""
    if not glob_overlay_shards(packed_row_dir):
        return _EMPTY_INVENTORY_FINGERPRINT
    return packed_row_fingerprint(packed_row_dir, cache_dir=packed_row_dir)


def _validate_integer_tensor(
    tensor: t.Tensor,
    *,
    name: str,
    ndim: int,
    shard_path: Path,
) -> None:
    if (
        tensor.ndim != ndim
        or tensor.dtype is t.bool
        or (tensor.is_floating_point() or tensor.is_complex())
    ):
        raise PackedRowInvariantError(
            f"{shard_path}: {name} must be a {ndim}-D integer tensor, got "
            f"dtype={tensor.dtype}, shape={tuple(tensor.shape)}"
        )


def _validate_offsets(
    offsets: t.Tensor,
    *,
    name: str,
    expected_entries: int,
    terminal: int,
    shard_path: Path,
) -> None:
    _validate_integer_tensor(offsets, name=name, ndim=1, shard_path=shard_path)
    if offsets.numel() != expected_entries:
        raise PackedRowInvariantError(
            f"{shard_path}: {name} has {offsets.numel()} entries, expected {expected_entries}"
        )
    values = offsets.to(t.int64)
    if int(values[0].item()) != 0:
        raise PackedRowInvariantError(f"{shard_path}: {name}[0] must be 0")
    if bool((values[1:] < values[:-1]).any()):
        raise PackedRowInvariantError(f"{shard_path}: {name} must be nondecreasing")
    if int(values[-1].item()) != terminal:
        raise PackedRowInvariantError(
            f"{shard_path}: {name} terminal offset is {int(values[-1])}, expected {terminal}"
        )


def _validate_shard(
    shard: PackedRowShard,
    *,
    shard_path: Path,
    n_sae_features: int,
) -> tuple[int, int, int]:
    expected_dtypes = {
        "tokens": t.uint16,
        "row_seg_offsets": t.int32,
        "seg_token_start": t.uint16,
        "seg_token_end": t.uint16,
        "seg_doc_id": t.uint8,
        "seg_feat_offsets": t.int32,
        "feature_ids": t.uint16,
        "feature_pcts": t.uint8,
    }
    for name, expected_dtype in expected_dtypes.items():
        actual_dtype = getattr(shard, name).dtype
        if actual_dtype is not expected_dtype:
            raise PackedRowInvariantError(
                f"{shard_path}: {name} must have packed dtype {expected_dtype}, got {actual_dtype}"
            )
    if shard.tokens.ndim != 2 or shard.tokens.shape[1] < 2:
        raise PackedRowInvariantError(
            f"{shard_path}: tokens must have shape [rows, T+1] with T>=1, got "
            f"{tuple(shard.tokens.shape)}"
        )
    _validate_integer_tensor(shard.tokens, name="tokens", ndim=2, shard_path=shard_path)
    n_rows = int(shard.tokens.shape[0])
    n_segments = int(shard.seg_token_start.numel())
    n_features = int(shard.feature_ids.numel())

    _validate_offsets(
        shard.row_seg_offsets,
        name="row_seg_offsets",
        expected_entries=n_rows + 1,
        terminal=n_segments,
        shard_path=shard_path,
    )
    _validate_offsets(
        shard.seg_feat_offsets,
        name="seg_feat_offsets",
        expected_entries=n_segments + 1,
        terminal=n_features,
        shard_path=shard_path,
    )

    for name in ("seg_token_start", "seg_token_end", "seg_doc_id"):
        tensor = getattr(shard, name)
        _validate_integer_tensor(tensor, name=name, ndim=1, shard_path=shard_path)
        if tensor.numel() != n_segments:
            raise PackedRowInvariantError(
                f"{shard_path}: {name} has {tensor.numel()} entries, expected {n_segments}"
            )
    _validate_integer_tensor(shard.feature_ids, name="feature_ids", ndim=1, shard_path=shard_path)
    _validate_integer_tensor(shard.feature_pcts, name="feature_pcts", ndim=1, shard_path=shard_path)
    if shard.feature_pcts.numel() != n_features:
        raise PackedRowInvariantError(
            f"{shard_path}: feature_ids/feature_pcts alignment mismatch: "
            f"{n_features} != {shard.feature_pcts.numel()}"
        )

    token_starts = shard.seg_token_start.to(t.int64)
    token_ends = shard.seg_token_end.to(t.int64)
    if n_segments > 0 and bool(
        ((token_starts < 0) | (token_ends < token_starts) | (token_ends > shard.context_len)).any()
    ):
        raise PackedRowInvariantError(
            f"{shard_path}: segment token boundaries are negative, reversed, or "
            f"outside [0, {shard.context_len}]"
        )

    feature_ids = shard.feature_ids.to(t.int64)
    if n_features > 0:
        invalid = (feature_ids < 0) | (feature_ids >= n_sae_features)
        if bool(invalid.any()):
            bad = int(feature_ids[invalid][0].item())
            raise PackedRowInvariantError(
                f"{shard_path}: feature ID {bad} is outside [0, {n_sae_features})"
            )

    feature_offsets = shard.seg_feat_offsets.to(t.int64)
    for segment_idx in range(n_segments):
        start = int(feature_offsets[segment_idx].item())
        end = int(feature_offsets[segment_idx + 1].item())
        segment_ids = feature_ids[start:end]
        if segment_ids.numel() > 1 and (t.unique(segment_ids).numel() != segment_ids.numel()):
            values, counts = t.unique(segment_ids, return_counts=True)
            duplicate = int(values[counts > 1][0].item())
            raise PackedRowInvariantError(
                f"{shard_path}: segment {segment_idx} contains duplicate "
                f"original SAE feature ID {duplicate}"
            )
    return n_rows, n_segments, n_features


def _parse_certificate(
    raw: Any,
    *,
    surface: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _CERTIFICATE_KEYS:
        keys = set(raw) if isinstance(raw, Mapping) else set()
        raise PackedRowInvariantError(
            f"{surface}: {FP_TARGET_SOURCE_INTEGRITY_KEY} must contain exactly "
            f"{sorted(_CERTIFICATE_KEYS)}, got keys={sorted(keys, key=str)}"
        )
    cert = cast(Mapping[str, Any], raw)
    if type(cert["semantic_version"]) is not int or (
        cert["semantic_version"] != FP_TARGET_SOURCE_INTEGRITY_VERSION
    ):
        raise PackedRowInvariantError(
            f"{surface}: unknown FP target source semantic_version "
            f"{cert['semantic_version']!r}; expected "
            f"{FP_TARGET_SOURCE_INTEGRITY_VERSION}"
        )
    if type(cert["n_sae_features"]) is not int or cert["n_sae_features"] <= 0:
        raise PackedRowInvariantError(f"{surface}: n_sae_features must be a positive strict int")
    for key in (
        "packed_row_fingerprint",
        "ordered_shard_content_aggregate_sha256",
    ):
        if not is_sha256_hex(cert[key]):
            raise PackedRowInvariantError(
                f"{surface}: {key} must be 64 lowercase hexadecimal characters"
            )
    if cert["ordered_shard_content_aggregate_sha256"] != cert["packed_row_fingerprint"]:
        raise PackedRowInvariantError(
            f"{surface}: ordered shard content aggregate and packed-row fingerprint disagree"
        )

    inventory = cert["ordered_shard_inventory"]
    if not isinstance(inventory, list):
        raise PackedRowInvariantError(f"{surface}: ordered_shard_inventory must be a list")
    parsed_inventory: list[dict[str, Any]] = []
    previous_name: str | None = None
    for index, entry_raw in enumerate(inventory):
        if not isinstance(entry_raw, Mapping) or set(entry_raw) != _INVENTORY_ENTRY_KEYS:
            keys = set(entry_raw) if isinstance(entry_raw, Mapping) else set()
            raise PackedRowInvariantError(
                f"{surface}: inventory entry {index} must contain exactly "
                f"{sorted(_INVENTORY_ENTRY_KEYS)}, got keys={sorted(keys, key=str)}"
            )
        entry = cast(Mapping[str, Any], entry_raw)
        name = entry["name"]
        if (
            not isinstance(name, str)
            or not name
            or name != Path(name).name
            or (previous_name is not None and name <= previous_name)
        ):
            raise PackedRowInvariantError(
                f"{surface}: inventory filenames must be direct, unique, and strictly sorted"
            )
        previous_name = name
        byte_length = entry["byte_length"]
        if type(byte_length) is not int or byte_length < 0:
            raise PackedRowInvariantError(
                f"{surface}: inventory byte_length must be a non-negative strict int"
            )
        parsed_inventory.append({"name": name, "byte_length": byte_length})

    for key in ("rows_checked", "segments_checked", "feature_entries_checked"):
        if type(cert[key]) is not int or cert[key] < 0:
            raise PackedRowInvariantError(f"{surface}: {key} must be a non-negative strict int")
    if not parsed_inventory and (
        cert["rows_checked"] != 0
        or cert["segments_checked"] != 0
        or cert["feature_entries_checked"] != 0
        or cert["packed_row_fingerprint"] != _EMPTY_INVENTORY_FINGERPRINT
    ):
        raise PackedRowInvariantError(
            f"{surface}: an empty shard inventory must certify zero rows, "
            "segments, and feature entries with the canonical empty fingerprint"
        )
    for key in (
        "csr_structure_valid",
        "bounds_valid",
        "unique_original_ids_per_segment",
        "scan_complete",
    ):
        if cert[key] is not True:
            raise PackedRowInvariantError(f"{surface}: {key} must be the literal boolean true")
    return {**dict(cert), "ordered_shard_inventory": parsed_inventory}


@dataclass(frozen=True)
class PackedShardAttestation:
    """One packed body's certified scan, captured while its bytes were local.

    ``certify_fp_target_source`` opens every local shard, which a family that
    is published to a bucket and evicted AS IT IS PRODUCED can never satisfy —
    and an empty local directory certifies vacuously (zero rows scanned, the
    canonical empty fingerprint), which would be a false statement about a real
    family. Capturing the same per-shard scan at publication time, the last
    moment the bytes are local, keeps the certificate true instead.
    """

    name: str
    byte_length: int
    file_fingerprint: str
    rows_checked: int
    segments_checked: int
    feature_entries_checked: int


@beartype
def attest_packed_shard(
    shard_path: str | Path,
    *,
    n_sae_features: PositiveInt,
) -> tuple[int, int, int]:
    """Run the certificate's per-shard scan over one packed body.

    Returns ``(rows, segments, feature_entries)`` exactly as the whole-family
    scan accumulates them, and raises on any CSR/bounds/duplicate-ID defect.
    """
    path = Path(shard_path)
    return _validate_shard(
        load_packed_row_shard(path),
        shard_path=path,
        n_sae_features=n_sae_features,
    )


@beartype
def attested_inventory(attestations: Sequence[PackedShardAttestation]) -> dict[str, Any]:
    """The verified-inventory view of an attested family.

    Shaped for ``validate_fp_target_source_certificate_remote``, whose local
    counterpart derives the same two values by reopening every body. The fold
    matches ``_fingerprint_files`` by construction (name-sorted per-file
    digests), which is what lets an evicted family re-validate.
    """
    ordered = sorted(attestations, key=lambda attestation: attestation.name)
    return {
        "shards": [
            {"name": attestation.name, "byte_length": attestation.byte_length}
            for attestation in ordered
        ],
        "packed_row_fingerprint": _fold_file_digests(
            [attestation.file_fingerprint for attestation in ordered]
        ),
    }


@beartype
def certify_fp_target_source_from_attestations(
    packed_row_dir: str | Path,
    *,
    n_sae_features: PositiveInt,
    attestations: Sequence[PackedShardAttestation],
) -> dict[str, Any]:
    """Publish a certificate for a family whose bodies are no longer local.

    Every field is derived from per-shard attestations taken before eviction,
    so the published certificate states exactly what was scanned. The
    ``packed_row_fingerprint`` fold matches ``_fingerprint_files`` by
    construction (name-sorted per-file digests), which is what lets
    ``validate_fp_target_source_certificate_remote`` re-validate it against the
    published inventory.
    """
    packed_dir = Path(packed_row_dir)
    manifest = read_manifest(packed_dir)
    if manifest is None:
        raise PackedRowInvariantError(
            f"packed-row manifest is missing or invalid: {packed_dir / MANIFEST_FILENAME}"
        )
    manifest_before = canonical_json(manifest)
    if not attestations:
        # An empty inventory certifies vacuously -- zero rows scanned with the
        # canonical empty fingerprint -- which is the exact false statement this
        # path exists to avoid.
        raise PackedRowInvariantError(
            f"certifying {packed_dir} from attestations requires at least one attested shard"
        )
    # Any body still on disk must be one the attestations cover: certifying a
    # family that also holds unattested local shards would understate it.
    attested_sizes = {attestation.name: attestation.byte_length for attestation in attestations}
    for entry in _ordered_inventory(packed_dir):
        if attested_sizes.get(entry["name"]) != entry["byte_length"]:
            raise PackedRowInvariantError(
                f"packed-row shard {entry['name']} in {packed_dir} is not covered by a "
                "matching attestation; refusing to certify a family that was not fully scanned"
            )

    # Field shapes, name uniqueness and ordering, and the empty-family rule are
    # all enforced by _parse_certificate below, exactly as for a local scan.
    inventory = attested_inventory(attestations)
    fingerprint = cast(str, inventory["packed_row_fingerprint"])
    certificate: dict[str, Any] = {
        "semantic_version": FP_TARGET_SOURCE_INTEGRITY_VERSION,
        "n_sae_features": n_sae_features,
        "packed_row_fingerprint": fingerprint,
        "ordered_shard_inventory": inventory["shards"],
        "ordered_shard_content_aggregate_sha256": fingerprint,
        "rows_checked": sum(a.rows_checked for a in attestations),
        "segments_checked": sum(a.segments_checked for a in attestations),
        "feature_entries_checked": sum(a.feature_entries_checked for a in attestations),
        "csr_structure_valid": True,
        "bounds_valid": True,
        "unique_original_ids_per_segment": True,
        "scan_complete": True,
    }
    parsed = _parse_certificate(certificate, surface=f"new certificate for {packed_dir}")
    current_manifest = read_manifest(packed_dir)
    if current_manifest is None or canonical_json(current_manifest) != manifest_before:
        raise PackedRowInvariantError(
            "packed-row manifest changed while certifying from attestations; "
            "refusing to overwrite concurrent metadata"
        )
    metadata_raw = current_manifest.get("metadata")
    if metadata_raw is None:
        metadata: dict[str, Any] = {}
    elif isinstance(metadata_raw, Mapping):
        metadata = dict(metadata_raw)
    else:
        raise PackedRowInvariantError(
            f"packed-row manifest metadata must be a mapping in {packed_dir}"
        )
    metadata[FP_TARGET_SOURCE_INTEGRITY_KEY] = parsed
    atomic_write_text(
        json.dumps({**current_manifest, "metadata": metadata}, indent=2) + "\n",
        packed_dir / MANIFEST_FILENAME,
    )
    return parsed


@beartype
def certify_fp_target_source(
    packed_row_dir: str | Path,
    *,
    n_sae_features: PositiveInt,
) -> dict[str, Any]:
    """Stream every packed shard and atomically append a completed certificate.

    The manifest is not written until all shards pass. If a shard or the source
    inventory changes during the scan, the function raises and leaves the
    manifest byte-for-byte untouched.
    """
    packed_dir = Path(packed_row_dir)
    manifest = read_manifest(packed_dir)
    if manifest is None:
        raise PackedRowInvariantError(
            f"packed-row manifest is missing or invalid: {packed_dir / MANIFEST_FILENAME}"
        )
    manifest_before = canonical_json(manifest)
    inventory_before = _ordered_inventory(packed_dir)
    try:
        fingerprint_before = _packed_source_fingerprint(packed_dir)
    except Exception as exc:
        raise PackedRowInvariantError(
            f"could not fingerprint packed-row source {packed_dir}: {exc}"
        ) from exc

    rows_checked = 0
    segments_checked = 0
    feature_entries_checked = 0
    for entry in inventory_before:
        shard_path = packed_dir / entry["name"]
        shard = load_packed_row_shard(shard_path)
        rows, segments, features = _validate_shard(
            shard,
            shard_path=shard_path,
            n_sae_features=n_sae_features,
        )
        rows_checked += rows
        segments_checked += segments
        feature_entries_checked += features

    inventory_after = _ordered_inventory(packed_dir)
    try:
        fingerprint_after = _packed_source_fingerprint(packed_dir)
    except Exception as exc:
        raise PackedRowInvariantError(
            f"could not re-fingerprint packed-row source {packed_dir}: {exc}"
        ) from exc
    if inventory_after != inventory_before or fingerprint_after != fingerprint_before:
        raise PackedRowInvariantError(
            "packed-row source changed during the FP target integrity scan; "
            "retry from quiescent immutable artifacts"
        )
    current_manifest = read_manifest(packed_dir)
    if current_manifest is None or canonical_json(current_manifest) != manifest_before:
        raise PackedRowInvariantError(
            "packed-row manifest changed during the FP target integrity scan; "
            "refusing to overwrite concurrent metadata"
        )

    certificate: dict[str, Any] = {
        "semantic_version": FP_TARGET_SOURCE_INTEGRITY_VERSION,
        "n_sae_features": n_sae_features,
        "packed_row_fingerprint": fingerprint_after,
        "ordered_shard_inventory": inventory_after,
        "ordered_shard_content_aggregate_sha256": fingerprint_after,
        "rows_checked": rows_checked,
        "segments_checked": segments_checked,
        "feature_entries_checked": feature_entries_checked,
        "csr_structure_valid": True,
        "bounds_valid": True,
        "unique_original_ids_per_segment": True,
        "scan_complete": True,
    }
    parsed = _parse_certificate(certificate, surface=f"new certificate for {packed_dir}")
    metadata_raw = current_manifest.get("metadata")
    if metadata_raw is None:
        metadata: dict[str, Any] = {}
    elif isinstance(metadata_raw, Mapping):
        metadata = dict(metadata_raw)
    else:
        raise PackedRowInvariantError(
            f"packed-row manifest metadata must be a mapping in {packed_dir}"
        )
    metadata[FP_TARGET_SOURCE_INTEGRITY_KEY] = parsed
    updated_manifest = {**current_manifest, "metadata": metadata}
    atomic_write_text(
        json.dumps(updated_manifest, indent=2) + "\n",
        packed_dir / MANIFEST_FILENAME,
    )
    return parsed


def validate_fp_target_source_certificate(
    packed_row_dir: str | Path,
    *,
    n_sae_features: int,
) -> tuple[dict[str, Any], str]:
    """Validate the certificate against the current exact shard family."""
    packed_dir = Path(packed_row_dir)
    manifest = read_manifest(packed_dir)
    if manifest is None:
        raise PackedRowInvariantError(
            f"packed-row manifest is missing or invalid: {packed_dir / MANIFEST_FILENAME}"
        )
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise PackedRowInvariantError(f"packed-row manifest metadata is missing in {packed_dir}")
    if FP_TARGET_SOURCE_INTEGRITY_KEY not in metadata:
        raise PackedRowInvariantError(
            f"compact feature prediction requires a completed "
            f"{FP_TARGET_SOURCE_INTEGRITY_KEY} certificate in "
            f"{packed_dir / MANIFEST_FILENAME}. Run "
            "scripts.preprocess.validate_fp_target_source before training."
        )
    certificate = _parse_certificate(
        metadata[FP_TARGET_SOURCE_INTEGRITY_KEY],
        surface=f"packed-row manifest {packed_dir / MANIFEST_FILENAME}",
    )
    if certificate["n_sae_features"] != n_sae_features:
        raise PackedRowInvariantError(
            "FP target source certificate SAE-width mismatch: "
            f"certificate={certificate['n_sae_features']}, configured="
            f"{n_sae_features}"
        )
    current_inventory = _ordered_inventory(packed_dir)
    if current_inventory != certificate["ordered_shard_inventory"]:
        raise PackedRowInvariantError(
            "FP target source certificate shard inventory is stale: "
            f"certificate={certificate['ordered_shard_inventory']!r}, "
            f"current={current_inventory!r}"
        )
    try:
        current_fingerprint = _packed_source_fingerprint(packed_dir)
    except Exception as exc:
        raise PackedRowInvariantError(
            f"could not validate packed-row fingerprint for {packed_dir}: {exc}"
        ) from exc
    if current_fingerprint != certificate["packed_row_fingerprint"]:
        raise PackedRowInvariantError(
            "FP target source certificate packed-row fingerprint is stale: "
            f"certificate={certificate['packed_row_fingerprint']}, "
            f"current={current_fingerprint}"
        )
    return certificate, fp_target_source_certificate_digest(certificate)


def validate_fp_target_source_certificate_remote(
    packed_row_dir: str | Path,
    *,
    n_sae_features: int,
    inventory: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Validate a compact-FP certificate against a verified remote inventory.

    The local validator above derives the ordered body inventory and packed-row
    fingerprint by opening every local compressed shard. Remote packed-row
    streaming has already authenticated those same values through
    ``remote_packed_inventory.json``; this variant deliberately substitutes
    only that verified source while preserving certificate parsing, SAE-width,
    and digest semantics exactly.
    """
    packed_dir = Path(packed_row_dir)
    manifest = read_manifest(packed_dir)
    if manifest is None:
        raise PackedRowInvariantError(
            f"packed-row manifest is missing or invalid: {packed_dir / MANIFEST_FILENAME}"
        )
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise PackedRowInvariantError(f"packed-row manifest metadata is missing in {packed_dir}")
    if FP_TARGET_SOURCE_INTEGRITY_KEY not in metadata:
        raise PackedRowInvariantError(
            f"compact feature prediction requires a completed "
            f"{FP_TARGET_SOURCE_INTEGRITY_KEY} certificate in "
            f"{packed_dir / MANIFEST_FILENAME}. Run "
            "scripts.preprocess.validate_fp_target_source before "
            "scripts.preprocess.publish_remote_packed_inventory."
        )
    certificate = _parse_certificate(
        metadata[FP_TARGET_SOURCE_INTEGRITY_KEY],
        surface=f"packed-row manifest {packed_dir / MANIFEST_FILENAME}",
    )
    if certificate["n_sae_features"] != n_sae_features:
        raise PackedRowInvariantError(
            "FP target source certificate SAE-width mismatch: "
            f"certificate={certificate['n_sae_features']}, configured="
            f"{n_sae_features}"
        )

    raw_shards = inventory.get("shards")
    if not isinstance(raw_shards, list):
        raise PackedRowInvariantError("verified remote packed inventory shards must be a list")
    remote_ordered_inventory: list[dict[str, Any]] = []
    previous_name: str | None = None
    for index, raw_entry in enumerate(raw_shards):
        if not isinstance(raw_entry, Mapping):
            raise PackedRowInvariantError(
                f"verified remote packed inventory shard {index} must be a mapping"
            )
        name = raw_entry.get("name")
        byte_length = raw_entry.get("byte_length")
        if (
            not isinstance(name, str)
            or not name
            or name != Path(name).name
            or (previous_name is not None and name <= previous_name)
        ):
            raise PackedRowInvariantError(
                "verified remote packed inventory filenames must be direct, "
                "unique, and strictly sorted"
            )
        if type(byte_length) is not int or byte_length < 0:
            raise PackedRowInvariantError(
                "verified remote packed inventory byte_length must be a non-negative strict int"
            )
        previous_name = name
        remote_ordered_inventory.append({"name": name, "byte_length": byte_length})

    if certificate["ordered_shard_inventory"] != remote_ordered_inventory:
        raise PackedRowInvariantError(
            "FP target source certificate shard inventory is stale against "
            "the verified remote packed inventory: "
            f"certificate={certificate['ordered_shard_inventory']!r}, "
            f"remote={remote_ordered_inventory!r}"
        )
    remote_fingerprint = inventory.get("packed_row_fingerprint")
    if not is_sha256_hex(remote_fingerprint):
        raise PackedRowInvariantError(
            "verified remote packed inventory packed_row_fingerprint must be "
            "64 lowercase hexadecimal characters"
        )
    if certificate["packed_row_fingerprint"] != remote_fingerprint:
        raise PackedRowInvariantError(
            "FP target source certificate packed-row fingerprint is stale "
            "against the verified remote packed inventory: "
            f"certificate={certificate['packed_row_fingerprint']}, "
            f"remote={remote_fingerprint}"
        )
    return certificate, fp_target_source_certificate_digest(certificate)
