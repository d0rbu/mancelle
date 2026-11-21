"""Producer-authority certificate for clustered selection artifacts.

The certificate binds the exact compressed selection inventory to the packed
row family and to the semantic validation results produced before publication.
Consumers still rehash the compact selection bytes on every launch.  They may
skip tensor decompression only after this certificate and the independent
packed-row stat cache both validate.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from errors import FeatureClustersError
from steering.conditional.delta import (
    CLUSTER_DELTA_SEMANTICS_KEY,
    CLUSTERED_SPLICE_COUNTERFACTUAL_MANIFEST_SCHEMA_VERSION,
    CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS,
)
from steering.conditional.feature_clusters import (
    FeatureClusterDescriptor,
    parse_clustered_selection_shard_inventory,
)
from steering.conditional.splice_selection_certificate import (
    SPLICE_RESUME_METADATA_KEY,
    SpliceSelectionStats,
    parse_splice_resume_metadata,
)
from steering.immutable_io import is_sha256_hex
from steering.remote_training import (
    REMOTE_LOADER_KEY,
    RemoteTrainingError,
    parse_remote_loader_certificate,
)

CLUSTERED_SELECTION_VALIDATION_KEY = "clustered_selection_validation"
CLUSTERED_SELECTION_VALIDATION_SCHEMA_VERSION = 1
CLUSTERED_SELECTION_SEMANTICS_VERSION = "clustered_selection_validation_v1"
_SHARD_SUFFIX = ".safetensors.zst"


@dataclass(frozen=True)
class ClusteredSelectionValidationAuthority:
    """Strictly parsed producer proof used by clustered consumers."""

    kind: str
    selection_checksum: str
    selection_inventory_aggregate: str
    packed_row_fingerprint: str
    effective_buckets: int
    segment_counts_by_name: tuple[tuple[str, int], ...]
    max_features_by_name: tuple[tuple[str, int], ...] | None
    packed_content_by_name: tuple[tuple[str, str], ...] | None
    max_features_per_segment: int


@dataclass(frozen=True)
class CompletedSpliceValidation:
    """Shared completed splice-v3 proof parsed identically by every consumer."""

    stats_by_stem: tuple[tuple[str, SpliceSelectionStats], ...]
    validated_max_features_per_segment: int


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _strict_non_negative_int(value: object, *, surface: str) -> int:
    if type(value) is not int or value < 0:
        raise FeatureClustersError(
            f"{surface} must be a non-negative strict integer, got {value!r}"
        )
    return value


def _strict_effective_buckets(value: object, *, surface: str) -> int:
    if type(value) is not int or not (1 <= value <= 256):
        raise FeatureClustersError(
            f"{surface} must be a strict integer in [1, 256], got {value!r}"
        )
    return value


def _strict_direct_shard_name(value: object, *, surface: str) -> str:
    if (
        not isinstance(value, str)
        or not value.endswith(_SHARD_SUFFIX)
        or value != Path(value).name
    ):
        raise FeatureClustersError(
            f"{surface} must be a direct {_SHARD_SUFFIX!r} filename, got {value!r}"
        )
    return value


def _strict_sha256(value: object, *, surface: str) -> str:
    if not is_sha256_hex(value):
        raise FeatureClustersError(
            f"{surface} must be 64 lowercase hexadecimal characters, got {value!r}"
        )
    return cast(str, value)


def _canonical_feature_delta(value: object, *, surface: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"mode", "neg_delta_ratio"}:
        raise FeatureClustersError(
            f"{surface} must be null or contain exactly mode/neg_delta_ratio"
        )
    parsed_value = cast(Mapping[str, object], value)
    mode = parsed_value["mode"]
    ratio = parsed_value["neg_delta_ratio"]
    if not isinstance(mode, str) or not mode:
        raise FeatureClustersError(f"{surface}.mode must be a non-empty string")
    if isinstance(ratio, bool) or not isinstance(ratio, int | float):
        raise FeatureClustersError(
            f"{surface}.neg_delta_ratio must be a real number, got {ratio!r}"
        )
    canonical_ratio = float(ratio)
    if not math.isfinite(canonical_ratio):
        raise FeatureClustersError(
            f"{surface}.neg_delta_ratio must be finite, got {ratio!r}"
        )
    return {"mode": mode, "neg_delta_ratio": canonical_ratio}


def _canonical_cluster_descriptor(
    value: object,
    *,
    surface: str,
) -> dict[str, Any]:
    try:
        return FeatureClusterDescriptor.from_dict(value).canonical_dict()
    except FeatureClustersError as exc:
        raise FeatureClustersError(
            f"{surface} is not a canonical feature cluster descriptor: {exc}"
        ) from exc


def build_clustered_selection_validation(
    *,
    selection_checksum: str,
    selection_integrity: Mapping[str, Any],
    packed_row_fingerprint: str,
    packed_content_by_name: Mapping[str, str],
    feature_clusters_descriptor: Mapping[str, Any],
    feature_delta: Mapping[str, Any] | None,
    effective_buckets: int,
    segment_counts_by_stem: Mapping[str, int],
    max_features_by_stem: Mapping[str, int],
) -> dict[str, Any]:
    """Build one complete schema-v1 certificate from stat-fenced evidence."""

    checksum = _strict_sha256(selection_checksum, surface="selection checksum")
    packed_fingerprint = _strict_sha256(
        packed_row_fingerprint,
        surface="packed-row fingerprint",
    )
    buckets = _strict_effective_buckets(
        effective_buckets,
        surface="effective_buckets",
    )
    inventory = parse_clustered_selection_shard_inventory(
        selection_integrity,
        surface="clustered selection validation",
    )
    descriptor = _canonical_cluster_descriptor(
        feature_clusters_descriptor,
        surface="feature_clusters_descriptor",
    )
    delta = _canonical_feature_delta(
        feature_delta,
        surface="feature_delta",
    )
    selection_by_stem = {
        entry["name"].removesuffix(_SHARD_SUFFIX): entry
        for entry in inventory["shards"]
    }
    if (
        set(selection_by_stem) != set(packed_content_by_name)
        or set(selection_by_stem) != set(segment_counts_by_stem)
        or set(selection_by_stem) != set(max_features_by_stem)
    ):
        raise FeatureClustersError(
            "clustered selection validation evidence must have identical "
            "selection, packed, segment-count, and width stem domains"
        )

    shards: list[dict[str, Any]] = []
    for stem in sorted(selection_by_stem):
        selection_entry = selection_by_stem[stem]
        packed_digest = _strict_sha256(
            packed_content_by_name[stem],
            surface=f"packed content digest for {stem!r}",
        )
        shards.append(
            {
                "selection_name": selection_entry["name"],
                "selection_content_sha256": selection_entry["content_sha256"],
                "packed_name": f"{stem}{_SHARD_SUFFIX}",
                "packed_content_sha256": packed_digest,
                "segment_count": _strict_non_negative_int(
                    segment_counts_by_stem[stem],
                    surface=f"segment count for {stem!r}",
                ),
                "max_features_per_segment": _strict_non_negative_int(
                    max_features_by_stem[stem],
                    surface=f"maximum selected width for {stem!r}",
                ),
            }
        )
    signed = {
        "schema_version": CLUSTERED_SELECTION_VALIDATION_SCHEMA_VERSION,
        "semantics_version": CLUSTERED_SELECTION_SEMANTICS_VERSION,
        "status": "complete",
        "selection_checksum": checksum,
        "selection_inventory_aggregate": inventory["aggregate_sha256"],
        "packed_row_fingerprint": packed_fingerprint,
        "feature_clusters_descriptor": descriptor,
        "feature_delta": delta,
        "effective_buckets": buckets,
        "shards": shards,
    }
    return {**signed, "aggregate_sha256": _sha256_json(signed)}


def parse_clustered_selection_validation(
    raw: object,
    *,
    selection_checksum: str,
    selection_integrity: Mapping[str, Any],
    packed_row_fingerprint: str,
    feature_clusters_descriptor: Mapping[str, Any],
    feature_delta: Mapping[str, Any] | None,
    effective_buckets: int,
    surface: str,
) -> ClusteredSelectionValidationAuthority:
    """Strictly parse a complete schema-v1 certificate without opening shards."""

    top_keys = {
        "schema_version",
        "semantics_version",
        "status",
        "selection_checksum",
        "selection_inventory_aggregate",
        "packed_row_fingerprint",
        "feature_clusters_descriptor",
        "feature_delta",
        "effective_buckets",
        "shards",
        "aggregate_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != top_keys:
        keys = set(raw) if isinstance(raw, Mapping) else set()
        raise FeatureClustersError(
            f"{surface} {CLUSTERED_SELECTION_VALIDATION_KEY} must contain "
            f"exactly {sorted(top_keys)}; got keys={sorted(keys, key=str)}"
        )
    parsed_raw = cast(Mapping[str, Any], raw)
    schema = parsed_raw["schema_version"]
    if (
        type(schema) is not int
        or schema != CLUSTERED_SELECTION_VALIDATION_SCHEMA_VERSION
    ):
        raise FeatureClustersError(
            f"{surface} clustered validation schema_version must be strict "
            f"integer {CLUSTERED_SELECTION_VALIDATION_SCHEMA_VERSION}"
        )
    if parsed_raw["semantics_version"] != CLUSTERED_SELECTION_SEMANTICS_VERSION:
        raise FeatureClustersError(
            f"{surface} clustered validation semantics_version is unsupported"
        )
    if parsed_raw["status"] != "complete":
        raise FeatureClustersError(
            f"{surface} clustered validation status must be 'complete'"
        )
    inventory = parse_clustered_selection_shard_inventory(
        selection_integrity,
        surface=surface,
    )
    expected_identity = {
        "selection_checksum": _strict_sha256(
            selection_checksum,
            surface=f"{surface} expected selection checksum",
        ),
        "selection_inventory_aggregate": inventory["aggregate_sha256"],
        "packed_row_fingerprint": _strict_sha256(
            packed_row_fingerprint,
            surface=f"{surface} expected packed-row fingerprint",
        ),
        "feature_clusters_descriptor": _canonical_cluster_descriptor(
            feature_clusters_descriptor,
            surface=f"{surface} expected feature cluster descriptor",
        ),
        "feature_delta": _canonical_feature_delta(
            feature_delta,
            surface=f"{surface} expected feature delta",
        ),
        "effective_buckets": _strict_effective_buckets(
            effective_buckets,
            surface=f"{surface} expected effective buckets",
        ),
    }
    parsed_identity = {
        "selection_checksum": _strict_sha256(
            parsed_raw["selection_checksum"],
            surface=f"{surface} clustered validation selection_checksum",
        ),
        "selection_inventory_aggregate": _strict_sha256(
            parsed_raw["selection_inventory_aggregate"],
            surface=(
                f"{surface} clustered validation "
                "selection_inventory_aggregate"
            ),
        ),
        "packed_row_fingerprint": _strict_sha256(
            parsed_raw["packed_row_fingerprint"],
            surface=f"{surface} clustered validation packed_row_fingerprint",
        ),
        "feature_clusters_descriptor": _canonical_cluster_descriptor(
            parsed_raw["feature_clusters_descriptor"],
            surface=(
                f"{surface} clustered validation "
                "feature_clusters_descriptor"
            ),
        ),
        "feature_delta": _canonical_feature_delta(
            parsed_raw["feature_delta"],
            surface=f"{surface} clustered validation feature_delta",
        ),
        "effective_buckets": _strict_effective_buckets(
            parsed_raw["effective_buckets"],
            surface=f"{surface} clustered validation effective_buckets",
        ),
    }
    for key, expected in expected_identity.items():
        if parsed_identity[key] != expected:
            raise FeatureClustersError(
                f"{surface} clustered validation {key}={parsed_raw[key]!r} "
                f"does not match expected {expected!r}"
            )

    raw_shards = parsed_raw["shards"]
    if not isinstance(raw_shards, list) or not raw_shards:
        raise FeatureClustersError(
            f"{surface} clustered validation shards must be a non-empty list"
        )
    entry_keys = {
        "selection_name",
        "selection_content_sha256",
        "packed_name",
        "packed_content_sha256",
        "segment_count",
        "max_features_per_segment",
    }
    inventory_by_name = {
        entry["name"]: entry["content_sha256"] for entry in inventory["shards"]
    }
    signed_entries: list[dict[str, Any]] = []
    segments: list[tuple[str, int]] = []
    widths: list[tuple[str, int]] = []
    packed: list[tuple[str, str]] = []
    previous_selection_name: str | None = None
    for index, raw_entry in enumerate(raw_shards):
        if not isinstance(raw_entry, Mapping) or set(raw_entry) != entry_keys:
            keys = set(raw_entry) if isinstance(raw_entry, Mapping) else set()
            raise FeatureClustersError(
                f"{surface} clustered validation shard {index} must contain "
                f"exactly {sorted(entry_keys)}; got keys={sorted(keys, key=str)}"
            )
        entry = cast(Mapping[str, Any], raw_entry)
        selection_name = _strict_direct_shard_name(
            entry["selection_name"],
            surface=f"{surface} shard {index} selection_name",
        )
        packed_name = _strict_direct_shard_name(
            entry["packed_name"],
            surface=f"{surface} shard {index} packed_name",
        )
        if (
            previous_selection_name is not None
            and selection_name <= previous_selection_name
        ):
            raise FeatureClustersError(
                f"{surface} clustered validation shard entries must be "
                "strictly sorted by unique selection_name"
            )
        previous_selection_name = selection_name
        if selection_name != packed_name:
            raise FeatureClustersError(
                f"{surface} clustered validation selection/packed filenames "
                f"must match one-to-one, got {selection_name!r}/{packed_name!r}"
            )
        selection_digest = _strict_sha256(
            entry["selection_content_sha256"],
            surface=f"{surface} {selection_name} selection digest",
        )
        if inventory_by_name.get(selection_name) != selection_digest:
            raise FeatureClustersError(
                f"{surface} clustered validation selection digest for "
                f"{selection_name!r} disagrees with the certified inventory"
            )
        packed_digest = _strict_sha256(
            entry["packed_content_sha256"],
            surface=f"{surface} {packed_name} packed digest",
        )
        segment_count = _strict_non_negative_int(
            entry["segment_count"],
            surface=f"{surface} {selection_name} segment_count",
        )
        max_features = _strict_non_negative_int(
            entry["max_features_per_segment"],
            surface=f"{surface} {selection_name} max_features_per_segment",
        )
        signed_entry = {
            "selection_name": selection_name,
            "selection_content_sha256": selection_digest,
            "packed_name": packed_name,
            "packed_content_sha256": packed_digest,
            "segment_count": segment_count,
            "max_features_per_segment": max_features,
        }
        signed_entries.append(signed_entry)
        segments.append((selection_name, segment_count))
        widths.append((selection_name, max_features))
        packed.append((packed_name, packed_digest))
    if set(inventory_by_name) != {name for name, _count in segments}:
        raise FeatureClustersError(
            f"{surface} clustered validation shard domain differs from the "
            "selection inventory"
        )
    signed = {
        "schema_version": schema,
        "semantics_version": parsed_raw["semantics_version"],
        "status": parsed_raw["status"],
        **parsed_identity,
        "shards": signed_entries,
    }
    aggregate = parsed_raw["aggregate_sha256"]
    expected_aggregate = _sha256_json(signed)
    if not is_sha256_hex(aggregate) or aggregate != expected_aggregate:
        raise FeatureClustersError(
            f"{surface} clustered validation aggregate mismatch: manifest has "
            f"{aggregate!r}, fields hash to {expected_aggregate!r}"
        )
    return ClusteredSelectionValidationAuthority(
        kind="schema_v1",
        selection_checksum=expected_identity["selection_checksum"],
        selection_inventory_aggregate=inventory["aggregate_sha256"],
        packed_row_fingerprint=expected_identity["packed_row_fingerprint"],
        effective_buckets=expected_identity["effective_buckets"],
        segment_counts_by_name=tuple(segments),
        max_features_by_name=tuple(widths),
        packed_content_by_name=tuple(packed),
        max_features_per_segment=max(width for _name, width in widths),
    )


def parse_completed_splice_validation(
    metadata: Mapping[str, Any],
    *,
    selection_integrity: Mapping[str, Any],
    packed_row_fingerprint: str,
    effective_buckets: int,
    feature_clusters_descriptor: Mapping[str, Any],
    expected_stems: set[str],
    surface: str,
) -> CompletedSpliceValidation:
    """Parse the common completed splice-v3 proof and all summary bindings."""

    buckets = _strict_effective_buckets(
        effective_buckets,
        surface=f"{surface} expected effective buckets",
    )
    fingerprint = _strict_sha256(
        packed_row_fingerprint,
        surface=f"{surface} expected packed-row fingerprint",
    )
    clusters = metadata.get("feature_clusters")
    splice = metadata.get("splice_counterfactual")
    if (
        not isinstance(clusters, Mapping)
        or clusters.get("status") != "complete"
        or _canonical_cluster_descriptor(
            clusters.get("descriptor"),
            surface=f"{surface} feature_clusters.descriptor",
        )
        != _canonical_cluster_descriptor(
            feature_clusters_descriptor,
            surface=f"{surface} expected feature cluster descriptor",
        )
        or not isinstance(splice, Mapping)
        or splice.get("status") != "complete"
        or type(splice.get("schema_version")) is not int
        or splice.get("schema_version")
        != CLUSTERED_SPLICE_COUNTERFACTUAL_MANIFEST_SCHEMA_VERSION
        or splice.get(CLUSTER_DELTA_SEMANTICS_KEY)
        != CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS
    ):
        raise FeatureClustersError(
            f"{surface} legacy proof requires complete cluster and splice statuses"
        )
    (
        stats_by_stem,
        outputs_validated,
        _storage_validated_by_stem,
        validated_max,
    ) = parse_splice_resume_metadata(
        metadata.get(SPLICE_RESUME_METADATA_KEY),
        selection_integrity=selection_integrity,
        packed_rows_hash=fingerprint,
        effective_buckets=buckets,
        manifest_status="complete",
        surface=surface,
    )
    # ``manifest_status="complete"`` makes these non-optional by the shared
    # splice-v3 parser's contract.
    assert outputs_validated and validated_max is not None
    if set(stats_by_stem) != expected_stems:
        raise FeatureClustersError(
            f"{surface} completed splice-v3 proof does not cover the exact "
            "packed/selection shard domain"
        )
    certified_totals = {
        "total_packed_segments_selected": sum(
            stats.total_packed_segments for stats in stats_by_stem.values()
        ),
        "total_sidecar_splice_positions_referenced": sum(
            stats.total_sidecar_splice_positions_referenced
            for stats in stats_by_stem.values()
        ),
        "total_delta_positions_selected": sum(
            stats.total_delta_positions_selected for stats in stats_by_stem.values()
        ),
    }
    for key, expected in certified_totals.items():
        recorded = splice.get(key)
        if type(recorded) is not int or recorded != expected:
            raise FeatureClustersError(
                f"{surface} {key}={recorded!r} disagrees with the completed "
                f"splice-v3 shard total {expected}"
            )
    return CompletedSpliceValidation(
        stats_by_stem=tuple(sorted(stats_by_stem.items())),
        validated_max_features_per_segment=validated_max,
    )


def parse_legacy_splice_validation(
    metadata: Mapping[str, Any],
    *,
    selection_checksum: str,
    selection_integrity: Mapping[str, Any],
    packed_row_fingerprint: str,
    effective_buckets: int,
    feature_clusters_descriptor: Mapping[str, Any],
    feature_delta: Mapping[str, Any] | None,
    surface: str,
) -> ClusteredSelectionValidationAuthority:
    """Accept only the complete splice-v3 + remote-loader-v2 proof."""

    checksum = _strict_sha256(
        selection_checksum,
        surface=f"{surface} expected selection checksum",
    )
    fingerprint = _strict_sha256(
        packed_row_fingerprint,
        surface=f"{surface} expected packed-row fingerprint",
    )
    buckets = _strict_effective_buckets(
        effective_buckets,
        surface=f"{surface} expected effective buckets",
    )
    if _canonical_feature_delta(
        feature_delta,
        surface=f"{surface} feature_delta",
    ) != {
        "mode": "splice_counterfactual",
        "neg_delta_ratio": 1.0,
    }:
        raise FeatureClustersError(
            f"{surface} has no schema-v1 clustered selection validation and "
            "is not eligible for the legacy splice-v3 proof"
        )
    inventory = parse_clustered_selection_shard_inventory(
        selection_integrity,
        surface=surface,
    )
    names = [entry["name"] for entry in inventory["shards"]]
    expected_stems = {name.removesuffix(_SHARD_SUFFIX) for name in names}
    completed = parse_completed_splice_validation(
        metadata,
        selection_integrity=selection_integrity,
        packed_row_fingerprint=fingerprint,
        effective_buckets=buckets,
        feature_clusters_descriptor=feature_clusters_descriptor,
        expected_stems=expected_stems,
        surface=surface,
    )
    stats_by_stem = dict(completed.stats_by_stem)
    validated_max = completed.validated_max_features_per_segment
    try:
        remote = parse_remote_loader_certificate(
            metadata.get(REMOTE_LOADER_KEY),
            surface=surface,
        )
    except RemoteTrainingError as exc:
        raise FeatureClustersError(str(exc)) from exc
    expected_remote = {
        "selection_checksum": checksum,
        "selection_inventory_aggregate": inventory["aggregate_sha256"],
        "packed_row_fingerprint": fingerprint,
        "max_features_per_segment": validated_max,
        CLUSTER_DELTA_SEMANTICS_KEY: CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS,
    }
    for key, expected in expected_remote.items():
        if remote.get(key) != expected:
            raise FeatureClustersError(
                f"{surface} legacy splice-v3 and remote-loader certificates "
                f"disagree on {key}: {remote.get(key)!r} != {expected!r}"
            )
    return ClusteredSelectionValidationAuthority(
        kind="legacy_splice_v3",
        selection_checksum=checksum,
        selection_inventory_aggregate=inventory["aggregate_sha256"],
        packed_row_fingerprint=fingerprint,
        effective_buckets=buckets,
        segment_counts_by_name=tuple(
            (
                f"{stem}{_SHARD_SUFFIX}",
                stats_by_stem[stem].total_packed_segments,
            )
            for stem in sorted(stats_by_stem)
        ),
        max_features_by_name=None,
        packed_content_by_name=None,
        max_features_per_segment=validated_max,
    )


def resolve_clustered_selection_validation(
    metadata: Mapping[str, Any],
    *,
    selection_checksum: str,
    selection_integrity: Mapping[str, Any],
    packed_row_fingerprint: str,
    feature_clusters_descriptor: Mapping[str, Any],
    feature_delta: Mapping[str, Any] | None,
    effective_buckets: int,
    surface: str,
) -> ClusteredSelectionValidationAuthority:
    """Resolve schema-v1 authority or the one admitted legacy equivalent."""

    raw = metadata.get(CLUSTERED_SELECTION_VALIDATION_KEY)
    if raw is not None:
        authority = parse_clustered_selection_validation(
            raw,
            selection_checksum=selection_checksum,
            selection_integrity=selection_integrity,
            packed_row_fingerprint=packed_row_fingerprint,
            feature_clusters_descriptor=feature_clusters_descriptor,
            feature_delta=feature_delta,
            effective_buckets=effective_buckets,
            surface=surface,
        )
        remote_raw = metadata.get(REMOTE_LOADER_KEY)
        if remote_raw is not None:
            try:
                remote = parse_remote_loader_certificate(
                    remote_raw,
                    surface=surface,
                )
            except RemoteTrainingError as exc:
                raise FeatureClustersError(str(exc)) from exc
            expected_remote = {
                "selection_checksum": authority.selection_checksum,
                "selection_inventory_aggregate": (
                    authority.selection_inventory_aggregate
                ),
                "packed_row_fingerprint": authority.packed_row_fingerprint,
                "max_features_per_segment": (
                    authority.max_features_per_segment
                ),
                CLUSTER_DELTA_SEMANTICS_KEY: (
                    CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS
                    if _canonical_feature_delta(
                        feature_delta,
                        surface=f"{surface} feature_delta",
                    )
                    == {
                        "mode": "splice_counterfactual",
                        "neg_delta_ratio": 1.0,
                    }
                    else None
                ),
            }
            for key, expected in expected_remote.items():
                if remote.get(key) != expected:
                    raise FeatureClustersError(
                        f"{surface} clustered validation and remote-loader "
                        f"certificates disagree on {key}: "
                        f"{remote.get(key)!r} != {expected!r}"
                    )
        splice_raw = metadata.get(SPLICE_RESUME_METADATA_KEY)
        if splice_raw is not None:
            expected_stems = {
                name.removesuffix(_SHARD_SUFFIX)
                for name, _count in authority.segment_counts_by_name
            }
            completed = parse_completed_splice_validation(
                metadata,
                selection_integrity=selection_integrity,
                packed_row_fingerprint=packed_row_fingerprint,
                effective_buckets=effective_buckets,
                feature_clusters_descriptor=feature_clusters_descriptor,
                expected_stems=expected_stems,
                surface=surface,
            )
            splice_counts = tuple(
                (
                    f"{stem}{_SHARD_SUFFIX}",
                    stats.total_packed_segments,
                )
                for stem, stats in completed.stats_by_stem
            )
            if (
                completed.validated_max_features_per_segment
                != authority.max_features_per_segment
                or splice_counts != authority.segment_counts_by_name
            ):
                raise FeatureClustersError(
                    f"{surface} clustered validation and splice-v3 "
                    "certificates disagree on per-shard counts or width"
                )
        return authority
    return parse_legacy_splice_validation(
        metadata,
        selection_checksum=selection_checksum,
        selection_integrity=selection_integrity,
        packed_row_fingerprint=packed_row_fingerprint,
        effective_buckets=effective_buckets,
        feature_clusters_descriptor=feature_clusters_descriptor,
        feature_delta=feature_delta,
        surface=surface,
    )


def assert_validation_packed_inventory(
    authority: ClusteredSelectionValidationAuthority,
    packed_files: Sequence[tuple[str, str]],
    *,
    surface: str,
) -> None:
    """Bind schema-v1 per-file packed evidence to the strict cache result."""

    current = tuple(packed_files)
    if authority.packed_content_by_name is not None and (
        authority.packed_content_by_name != current
    ):
        raise FeatureClustersError(
            f"{surface} clustered validation packed-file inventory does not "
            "match the current stat-certified packed cache"
        )


__all__ = [
    "CLUSTERED_SELECTION_SEMANTICS_VERSION",
    "CLUSTERED_SELECTION_VALIDATION_KEY",
    "CLUSTERED_SELECTION_VALIDATION_SCHEMA_VERSION",
    "CompletedSpliceValidation",
    "ClusteredSelectionValidationAuthority",
    "assert_validation_packed_inventory",
    "build_clustered_selection_validation",
    "parse_completed_splice_validation",
    "parse_clustered_selection_validation",
    "parse_legacy_splice_validation",
    "resolve_clustered_selection_validation",
]
