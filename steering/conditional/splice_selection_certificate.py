"""Signed resume certificate for clustered splice selection artifacts.

The producer records per-shard semantic scan results only after writing and
byte-certifying each selection shard.  Consumers may use a completed
certificate to avoid reopening the much larger packed-row bodies, but only
after validating every identity link in this module.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from errors import FeatureClustersError
from steering.conditional.delta import (
    CLUSTER_DELTA_SEMANTICS_KEY,
    CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS,
)

SPLICE_RESUME_METADATA_KEY = "splice_resume"
SPLICE_RESUME_SCHEMA_VERSION = 3
SELECTION_SHARD_SUFFIX = ".safetensors.zst"
_LOWER_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class SpliceSelectionStats:
    total_packed_segments: int = 0
    total_sidecar_splice_positions_referenced: int = 0
    total_delta_positions_selected: int = 0

    def __add__(
        self,
        other: "SpliceSelectionStats",  # noqa: UP037 - preserve public annotation
    ) -> "SpliceSelectionStats":  # noqa: UP037
        return SpliceSelectionStats(
            total_packed_segments=(
                self.total_packed_segments + other.total_packed_segments
            ),
            total_sidecar_splice_positions_referenced=(
                self.total_sidecar_splice_positions_referenced
                + other.total_sidecar_splice_positions_referenced
            ),
            total_delta_positions_selected=(
                self.total_delta_positions_selected
                + other.total_delta_positions_selected
            ),
        )


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in _LOWER_HEX for char in value)
    )


def build_splice_resume_metadata(
    *,
    stats_by_stem: Mapping[str, SpliceSelectionStats],
    selection_integrity: Mapping[str, Any],
    storage_validated_by_stem: Mapping[str, str],
    validated_max_features_per_segment: int | None,
    packed_rows_hash: str,
    effective_buckets: int,
    status: str,
) -> dict[str, Any]:
    """Bind stats and post-serialization validation to exact shard bytes."""
    if status not in {"in_progress", "complete"}:
        raise ValueError(f"unsupported splice resume status {status!r}")
    raw_integrity_entries = selection_integrity.get("shards")
    selection_aggregate = selection_integrity.get("aggregate_sha256")
    if not isinstance(raw_integrity_entries, list) or not raw_integrity_entries:
        raise FeatureClustersError(
            "cannot build splice resume metadata without selection integrity entries"
        )
    if not _is_lower_sha256(selection_aggregate):
        raise FeatureClustersError(
            "cannot build splice resume metadata from malformed selection aggregate"
        )
    entries: list[dict[str, Any]] = []
    expected_stems: set[str] = set()
    for raw_entry in raw_integrity_entries:
        if not isinstance(raw_entry, Mapping):
            raise FeatureClustersError(
                "cannot build splice resume metadata from malformed selection entry"
            )
        name = raw_entry.get("name")
        selection_content_sha256 = raw_entry.get("content_sha256")
        if not isinstance(name, str) or not name.endswith(SELECTION_SHARD_SUFFIX):
            raise FeatureClustersError(
                f"cannot build splice resume metadata for invalid shard name {name!r}"
            )
        if not _is_lower_sha256(selection_content_sha256):
            raise FeatureClustersError(
                f"cannot build splice resume metadata for malformed digest of {name!r}"
            )
        stem = name.removesuffix(SELECTION_SHARD_SUFFIX)
        stats = stats_by_stem.get(stem)
        if stats is None:
            raise FeatureClustersError(
                f"cannot certify splice resume stats: {name!r} has no per-shard stats"
            )
        expected_stems.add(stem)
        storage_validation_sha256 = storage_validated_by_stem.get(stem)
        if storage_validation_sha256 is not None and (
            not _is_lower_sha256(storage_validation_sha256)
            or storage_validation_sha256 != selection_content_sha256
        ):
            raise FeatureClustersError(
                f"cannot certify splice storage validation for {name!r}: "
                "the validation digest must equal the selection content digest"
            )
        entries.append(
            {
                "name": name,
                "selection_content_sha256": selection_content_sha256,
                "storage_validation_content_sha256": storage_validation_sha256,
                "total_packed_segments": stats.total_packed_segments,
                "total_sidecar_splice_positions_referenced": (
                    stats.total_sidecar_splice_positions_referenced
                ),
                "total_delta_positions_selected": (
                    stats.total_delta_positions_selected
                ),
            }
        )
    if set(stats_by_stem) != expected_stems:
        raise FeatureClustersError(
            "cannot certify splice resume stats whose shard names differ from "
            "the selection integrity certificate"
        )
    unexpected_validations = set(storage_validated_by_stem) - expected_stems
    if unexpected_validations:
        raise FeatureClustersError(
            "cannot certify splice storage validation outside the selection "
            f"integrity domain: {sorted(unexpected_validations)[:5]}"
        )
    if status == "complete" and set(storage_validated_by_stem) != expected_stems:
        missing = sorted(expected_stems - set(storage_validated_by_stem))
        raise FeatureClustersError(
            "cannot complete splice resume metadata before every serialized "
            f"selection shard is validated; missing={missing[:5]}"
        )
    if validated_max_features_per_segment is not None and (
        type(validated_max_features_per_segment) is not int
        or validated_max_features_per_segment < 0
    ):
        raise FeatureClustersError(
            "cannot certify splice maximum features per segment unless it is "
            "a non-negative strict integer"
        )
    if bool(storage_validated_by_stem) != (
        validated_max_features_per_segment is not None
    ):
        raise FeatureClustersError(
            "cannot certify splice maximum features per segment without the "
            "same non-empty post-serialization validation domain"
        )
    output_validation = {
        "status": status,
        "selection_aggregate_sha256": selection_aggregate,
        "packed_row_fingerprint": packed_rows_hash,
        "effective_buckets": effective_buckets,
        "validated_max_features_per_segment": (
            validated_max_features_per_segment
        ),
    }
    signed_payload = {
        "schema_version": SPLICE_RESUME_SCHEMA_VERSION,
        CLUSTER_DELTA_SEMANTICS_KEY: CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS,
        "shards": entries,
        "output_validation": output_validation,
    }
    return {
        **signed_payload,
        "aggregate_sha256": _canonical_json_sha256(signed_payload),
    }


def parse_splice_resume_metadata(
    raw: object,
    *,
    selection_integrity: Mapping[str, Any],
    packed_rows_hash: str,
    effective_buckets: int,
    manifest_status: str,
    surface: str,
) -> tuple[dict[str, SpliceSelectionStats], bool, dict[str, str], int | None]:
    """Strictly validate a resume certificate already bound by shard integrity."""
    expected_top_keys = {
        "schema_version",
        CLUSTER_DELTA_SEMANTICS_KEY,
        "shards",
        "output_validation",
        "aggregate_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_top_keys:
        raise FeatureClustersError(
            f"{surface} {SPLICE_RESUME_METADATA_KEY} must contain exactly "
            f"{sorted(expected_top_keys)}"
        )
    parsed_raw = cast(Mapping[str, Any], raw)
    schema_version = parsed_raw["schema_version"]
    if (
        type(schema_version) is not int
        or schema_version != SPLICE_RESUME_SCHEMA_VERSION
    ):
        raise FeatureClustersError(
            f"{surface} splice resume schema_version must be strict integer "
            f"{SPLICE_RESUME_SCHEMA_VERSION}, got {schema_version!r}"
        )
    if parsed_raw[CLUSTER_DELTA_SEMANTICS_KEY] != CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS:
        raise FeatureClustersError(
            f"{surface} splice resume {CLUSTER_DELTA_SEMANTICS_KEY} must equal "
            "the current canonical clustered splice semantics, got "
            f"{parsed_raw[CLUSTER_DELTA_SEMANTICS_KEY]!r}"
        )
    raw_entries = parsed_raw["shards"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise FeatureClustersError(
            f"{surface} splice resume shards must be a non-empty list"
        )
    raw_integrity_entries = selection_integrity.get("shards")
    selection_aggregate = selection_integrity.get("aggregate_sha256")
    if not isinstance(raw_integrity_entries, list) or not _is_lower_sha256(
        selection_aggregate
    ):
        raise FeatureClustersError(
            f"{surface} has malformed clustered selection integrity metadata"
        )
    integrity_by_name = {
        entry.get("name"): entry.get("content_sha256")
        for entry in raw_integrity_entries
        if isinstance(entry, Mapping)
    }
    stats_by_stem: dict[str, SpliceSelectionStats] = {}
    storage_validated_by_stem: dict[str, str] = {}
    parsed_entries: list[dict[str, Any]] = []
    previous_name: str | None = None
    expected_entry_keys = {
        "name",
        "selection_content_sha256",
        "storage_validation_content_sha256",
        "total_packed_segments",
        "total_sidecar_splice_positions_referenced",
        "total_delta_positions_selected",
    }
    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, Mapping) or set(entry) != expected_entry_keys:
            raise FeatureClustersError(
                f"{surface} splice resume shard entry {index} must contain "
                f"exactly {sorted(expected_entry_keys)}"
            )
        parsed_entry = cast(Mapping[str, Any], entry)
        name = parsed_entry["name"]
        if (
            not isinstance(name, str)
            or not name.endswith(SELECTION_SHARD_SUFFIX)
            or name != Path(name).name
            or (previous_name is not None and name <= previous_name)
        ):
            raise FeatureClustersError(
                f"{surface} splice resume shard names must be direct, unique, "
                f"strictly sorted selection filenames; got {name!r}"
            )
        previous_name = name
        content_sha = parsed_entry["selection_content_sha256"]
        if not _is_lower_sha256(content_sha):
            raise FeatureClustersError(
                f"{surface} splice resume digest for {name!r} is malformed"
            )
        if integrity_by_name.get(name) != content_sha:
            raise FeatureClustersError(
                f"{surface} splice resume stats for {name!r} are not bound to "
                "the certified selection shard content"
            )
        stat_values: list[int] = []
        for key in (
            "total_packed_segments",
            "total_sidecar_splice_positions_referenced",
            "total_delta_positions_selected",
        ):
            value = parsed_entry[key]
            if type(value) is not int or value < 0:
                raise FeatureClustersError(
                    f"{surface} splice resume {name!r} {key} must be a "
                    f"non-negative strict integer, got {value!r}"
                )
            stat_values.append(value)
        stem = name.removesuffix(SELECTION_SHARD_SUFFIX)
        stats_by_stem[stem] = SpliceSelectionStats(*stat_values)
        storage_validation_sha256 = parsed_entry[
            "storage_validation_content_sha256"
        ]
        if storage_validation_sha256 is not None:
            if (
                not _is_lower_sha256(storage_validation_sha256)
                or storage_validation_sha256 != content_sha
            ):
                raise FeatureClustersError(
                    f"{surface} splice storage validation for {name!r} must "
                    "equal the certified selection content digest"
                )
            storage_validated_by_stem[stem] = storage_validation_sha256
        parsed_entries.append(dict(parsed_entry))
    if set(integrity_by_name) != {entry["name"] for entry in parsed_entries}:
        raise FeatureClustersError(
            f"{surface} splice resume shard names differ from the clustered "
            "selection integrity certificate"
        )

    output_validation = parsed_raw["output_validation"]
    expected_validation_keys = {
        "status",
        "selection_aggregate_sha256",
        "packed_row_fingerprint",
        "effective_buckets",
        "validated_max_features_per_segment",
    }
    if not isinstance(output_validation, Mapping) or set(
        output_validation
    ) != expected_validation_keys:
        raise FeatureClustersError(
            f"{surface} splice resume output_validation keys are malformed"
        )
    parsed_output_validation = cast(Mapping[str, Any], output_validation)
    status = parsed_output_validation["status"]
    if status not in {"in_progress", "complete"} or status != manifest_status:
        raise FeatureClustersError(
            f"{surface} splice resume status {status!r} does not match manifest "
            f"status {manifest_status!r}"
        )
    if parsed_output_validation["selection_aggregate_sha256"] != selection_aggregate:
        raise FeatureClustersError(
            f"{surface} splice resume output validation is not bound to the "
            "current selection integrity aggregate"
        )
    if parsed_output_validation["packed_row_fingerprint"] != packed_rows_hash:
        raise FeatureClustersError(
            f"{surface} splice resume output validation is stale for current packed rows"
        )
    recorded_buckets = parsed_output_validation["effective_buckets"]
    if type(recorded_buckets) is not int or recorded_buckets != effective_buckets:
        raise FeatureClustersError(
            f"{surface} splice resume effective_buckets={recorded_buckets!r}, "
            f"expected {effective_buckets}"
        )
    validated_max_features_per_segment: int | None = None
    raw_validated_max = parsed_output_validation[
        "validated_max_features_per_segment"
    ]
    if raw_validated_max is not None and (
        type(raw_validated_max) is not int or raw_validated_max < 0
    ):
        raise FeatureClustersError(
            f"{surface} splice resume validated maximum features per segment "
            f"must be a non-negative strict integer, got {raw_validated_max!r}"
        )
    validated_max_features_per_segment = raw_validated_max
    if bool(storage_validated_by_stem) != (
        validated_max_features_per_segment is not None
    ):
        raise FeatureClustersError(
            f"{surface} splice resume validated maximum features per segment "
            "does not match its post-serialization validation domain"
        )
    if (
        status == "complete"
        and set(storage_validated_by_stem) != set(stats_by_stem)
    ):
        missing = sorted(set(stats_by_stem) - set(storage_validated_by_stem))
        raise FeatureClustersError(
            f"{surface} complete splice resume certificate has unvalidated "
            f"serialized selection shards: {missing[:5]}"
        )
    signed_payload = {
        "schema_version": schema_version,
        CLUSTER_DELTA_SEMANTICS_KEY: parsed_raw[CLUSTER_DELTA_SEMANTICS_KEY],
        "shards": parsed_entries,
        "output_validation": dict(parsed_output_validation),
    }
    aggregate = parsed_raw["aggregate_sha256"]
    expected_aggregate = _canonical_json_sha256(signed_payload)
    if not _is_lower_sha256(aggregate) or aggregate != expected_aggregate:
        raise FeatureClustersError(
            f"{surface} splice resume aggregate mismatch: manifest has "
            f"{aggregate!r}, entries hash to {expected_aggregate!r}"
        )
    outputs_validated = (
        status == "complete"
        and set(storage_validated_by_stem) == set(stats_by_stem)
    )
    return (
        stats_by_stem,
        outputs_validated,
        storage_validated_by_stem,
        validated_max_features_per_segment,
    )
