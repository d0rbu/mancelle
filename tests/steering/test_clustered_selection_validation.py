"""Schema and cache gates for certified clustered evaluation."""

from __future__ import annotations

import builtins
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

import pytest

import steering.conditional.splice_counterfactual as splice
from errors import FeatureClustersError
from steering.conditional.clustered_selection_validation import (
    CLUSTERED_SELECTION_VALIDATION_KEY,
    assert_validation_packed_inventory,
    build_clustered_selection_validation,
    parse_clustered_selection_validation,
    parse_completed_splice_validation,
    resolve_clustered_selection_validation,
)
from steering.conditional.delta import (
    CLUSTER_DELTA_SEMANTICS_KEY,
    CLUSTERED_SPLICE_COUNTERFACTUAL_MANIFEST_SCHEMA_VERSION,
    CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS,
)
from steering.conditional.feature_clusters import (
    build_clustered_selection_shard_integrity,
)
from steering.conditional.splice_counterfactual import (
    SpliceProvenanceError,
    assert_packed_row_fingerprint_file_current,
    assert_packed_row_fingerprint_resolution_current,
    resolve_packed_row_fingerprint_strict,
)
from steering.conditional.splice_selection_certificate import (
    SPLICE_RESUME_METADATA_KEY,
    SpliceSelectionStats,
    build_splice_resume_metadata,
)
from steering.remote_training import (
    REMOTE_LOADER_KEY,
    build_remote_loader_certificate,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_CHECKSUM = "c" * 64
_DESCRIPTOR = {
    "schema_version": 1,
    "semantics": "post_mask_post_delta_max_percentile_v1",
    "n_sae_features": 8,
    "content_sha256": "d" * 64,
}


def _write_shards(tmp_path: Path) -> tuple[Path, Path]:
    packed_dir = tmp_path / "packed"
    selection_dir = tmp_path / "selection"
    packed_dir.mkdir()
    selection_dir.mkdir()
    packed = packed_dir / "shard_00000.safetensors.zst"
    selection = selection_dir / "shard_00000.safetensors.zst"
    packed.write_bytes(b"packed-body")
    selection.write_bytes(b"selection-body")
    return packed, selection


def _certificate(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    packed, selection = _write_shards(tmp_path)
    inventory = build_clustered_selection_shard_integrity([selection])
    proof = resolve_packed_row_fingerprint_strict(packed.parent)
    certificate = build_clustered_selection_validation(
        selection_checksum=_CHECKSUM,
        selection_integrity=inventory,
        packed_row_fingerprint=proof.aggregate_sha256,
        packed_content_by_name={
            entry.name.removesuffix(".safetensors.zst"): entry.content_sha256
            for entry in proof.files
        },
        feature_clusters_descriptor=_DESCRIPTOR,
        feature_delta=None,
        effective_buckets=1,
        segment_counts_by_stem={"shard_00000": 3},
        max_features_by_stem={"shard_00000": 7},
    )
    return certificate, inventory, proof.aggregate_sha256


def _legacy_metadata(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    _packed, selection = _write_shards(tmp_path)
    inventory = build_clustered_selection_shard_integrity([selection])
    proof = resolve_packed_row_fingerprint_strict(selection.parent.parent / "packed")
    digest = inventory["shards"][0]["content_sha256"]
    stats = SpliceSelectionStats(
        total_packed_segments=3,
        total_sidecar_splice_positions_referenced=5,
        total_delta_positions_selected=7,
    )
    metadata = {
        "feature_clusters": {
            "status": "complete",
            "descriptor": dict(_DESCRIPTOR),
        },
        "splice_counterfactual": {
            "schema_version": (
                CLUSTERED_SPLICE_COUNTERFACTUAL_MANIFEST_SCHEMA_VERSION
            ),
            "status": "complete",
            CLUSTER_DELTA_SEMANTICS_KEY: (
                CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS
            ),
            "total_packed_segments_selected": 3,
            "total_sidecar_splice_positions_referenced": 5,
            "total_delta_positions_selected": 7,
        },
        SPLICE_RESUME_METADATA_KEY: build_splice_resume_metadata(
            stats_by_stem={"shard_00000": stats},
            selection_integrity=inventory,
            storage_validated_by_stem={"shard_00000": digest},
            validated_max_features_per_segment=7,
            packed_rows_hash=proof.aggregate_sha256,
            effective_buckets=1,
            status="complete",
        ),
        REMOTE_LOADER_KEY: build_remote_loader_certificate(
            selection_checksum=_CHECKSUM,
            selection_inventory_aggregate=inventory["aggregate_sha256"],
            packed_row_fingerprint=proof.aggregate_sha256,
            max_features_per_segment=7,
            cluster_delta_semantics=(
                CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS
            ),
        ),
    }
    return metadata, inventory, proof.aggregate_sha256


def _resign_clustered_validation(certificate: dict[str, Any]) -> None:
    signed = {
        key: value
        for key, value in certificate.items()
        if key != "aggregate_sha256"
    }
    certificate["aggregate_sha256"] = hashlib.sha256(
        json.dumps(signed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_clustered_validation_schema_round_trip(tmp_path: Path) -> None:
    certificate, inventory, packed_fingerprint = _certificate(tmp_path)
    parsed = parse_clustered_selection_validation(
        certificate,
        selection_checksum=_CHECKSUM,
        selection_integrity=inventory,
        packed_row_fingerprint=packed_fingerprint,
        feature_clusters_descriptor=_DESCRIPTOR,
        feature_delta=None,
        effective_buckets=1,
        surface="test manifest",
    )
    assert parsed.kind == "schema_v1"
    assert parsed.segment_counts_by_name == (
        ("shard_00000.safetensors.zst", 3),
    )
    assert parsed.max_features_per_segment == 7


def test_clustered_validation_parses_independent_literal_golden_document() -> None:
    """Pin the wire format without invoking either production certificate builder."""

    descriptor = {
        "schema_version": 1,
        "semantics": "post_mask_post_delta_max_percentile_v1",
        "n_sae_features": 8,
        "content_sha256": "5" * 64,
    }
    inventory = {
        "schema_version": 1,
        "shards": [
            {
                "name": "shard_00000.safetensors.zst",
                "byte_length": 17,
                "content_sha256": "1" * 64,
            }
        ],
        "aggregate_sha256": (
            "d1814dcc5a5284932fec453014b598a0c8dd9e01b0ea77021fe95a6b5d13de44"
        ),
    }
    certificate = {
        "schema_version": 1,
        "semantics_version": "clustered_selection_validation_v1",
        "status": "complete",
        "selection_checksum": "3" * 64,
        "selection_inventory_aggregate": inventory["aggregate_sha256"],
        "packed_row_fingerprint": "4" * 64,
        "feature_clusters_descriptor": descriptor,
        "feature_delta": None,
        "effective_buckets": 1,
        "shards": [
            {
                "selection_name": "shard_00000.safetensors.zst",
                "selection_content_sha256": "1" * 64,
                "packed_name": "shard_00000.safetensors.zst",
                "packed_content_sha256": "2" * 64,
                "segment_count": 3,
                "max_features_per_segment": 7,
            }
        ],
        "aggregate_sha256": (
            "a97f487324d62e60048652b5b14d46b5f31cad98c3eba91b619f7e7487c63bdd"
        ),
    }

    parsed = parse_clustered_selection_validation(
        certificate,
        selection_checksum="3" * 64,
        selection_integrity=inventory,
        packed_row_fingerprint="4" * 64,
        feature_clusters_descriptor=descriptor,
        feature_delta=None,
        effective_buckets=1,
        surface="literal golden certificate",
    )

    assert parsed.kind == "schema_v1"
    assert parsed.segment_counts_by_name == (
        ("shard_00000.safetensors.zst", 3),
    )
    assert parsed.max_features_by_name == (
        ("shard_00000.safetensors.zst", 7),
    )
    assert parsed.packed_content_by_name == (
        ("shard_00000.safetensors.zst", "2" * 64),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("schema_version", 1.0),
        ("n_sae_features", True),
        ("n_sae_features", 8.0),
        ("content_sha256", "not-a-digest"),
    ],
)
def test_clustered_validation_rejects_resigned_descriptor_type_aliases(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    certificate, inventory, packed_fingerprint = _certificate(tmp_path)
    certificate["feature_clusters_descriptor"][field] = value
    _resign_clustered_validation(certificate)

    with pytest.raises(
        FeatureClustersError,
        match="canonical feature cluster descriptor",
    ):
        parse_clustered_selection_validation(
            certificate,
            selection_checksum=_CHECKSUM,
            selection_integrity=inventory,
            packed_row_fingerprint=packed_fingerprint,
            feature_clusters_descriptor=_DESCRIPTOR,
            feature_delta=None,
            effective_buckets=1,
            surface="strict descriptor fixture",
        )


def test_clustered_validation_rejects_resigned_descriptor_extra_key(
    tmp_path: Path,
) -> None:
    certificate, inventory, packed_fingerprint = _certificate(tmp_path)
    certificate["feature_clusters_descriptor"]["extra"] = None
    _resign_clustered_validation(certificate)

    with pytest.raises(FeatureClustersError, match="contain exactly"):
        parse_clustered_selection_validation(
            certificate,
            selection_checksum=_CHECKSUM,
            selection_integrity=inventory,
            packed_row_fingerprint=packed_fingerprint,
            feature_clusters_descriptor=_DESCRIPTOR,
            feature_delta=None,
            effective_buckets=1,
            surface="strict descriptor fixture",
        )


def test_legacy_splice_rejects_bool_aliased_cluster_descriptor(
    tmp_path: Path,
) -> None:
    metadata, inventory, packed_fingerprint = _legacy_metadata(tmp_path)
    metadata["feature_clusters"]["descriptor"]["schema_version"] = True

    with pytest.raises(
        FeatureClustersError,
        match="canonical feature cluster descriptor",
    ):
        resolve_clustered_selection_validation(
            metadata,
            selection_checksum=_CHECKSUM,
            selection_integrity=inventory,
            packed_row_fingerprint=packed_fingerprint,
            feature_clusters_descriptor=_DESCRIPTOR,
            feature_delta={
                "mode": "splice_counterfactual",
                "neg_delta_ratio": 1.0,
            },
            effective_buckets=1,
            surface="legacy strict descriptor fixture",
        )


def test_schema_v1_resolver_accepts_certificate_without_optional_twins(
    tmp_path: Path,
) -> None:
    certificate, inventory, packed_fingerprint = _certificate(tmp_path)
    authority = resolve_clustered_selection_validation(
        {CLUSTERED_SELECTION_VALIDATION_KEY: certificate},
        selection_checksum=_CHECKSUM,
        selection_integrity=inventory,
        packed_row_fingerprint=packed_fingerprint,
        feature_clusters_descriptor=_DESCRIPTOR,
        feature_delta=None,
        effective_buckets=1,
        surface="test manifest",
    )
    assert authority.kind == "schema_v1"


@pytest.mark.parametrize("effective_buckets", [False, 0, 257])
def test_clustered_validation_rejects_effective_buckets_outside_storage_domain(
    tmp_path: Path,
    effective_buckets: object,
) -> None:
    _packed, selection = _write_shards(tmp_path)
    inventory = build_clustered_selection_shard_integrity([selection])
    with pytest.raises(FeatureClustersError, match=r"\[1, 256\]"):
        build_clustered_selection_validation(
            selection_checksum=_CHECKSUM,
            selection_integrity=inventory,
            packed_row_fingerprint=_SHA_A,
            packed_content_by_name={"shard_00000": _SHA_B},
            feature_clusters_descriptor=_DESCRIPTOR,
            feature_delta=None,
            effective_buckets=effective_buckets,  # ty: ignore[invalid-argument-type]
            segment_counts_by_stem={"shard_00000": 3},
            max_features_by_stem={"shard_00000": 7},
        )


@pytest.mark.parametrize("ratio", [float("nan"), float("inf"), float("-inf")])
def test_clustered_validation_rejects_non_finite_delta_ratio(
    tmp_path: Path,
    ratio: float,
) -> None:
    _packed, selection = _write_shards(tmp_path)
    inventory = build_clustered_selection_shard_integrity([selection])
    with pytest.raises(FeatureClustersError, match="must be finite"):
        build_clustered_selection_validation(
            selection_checksum=_CHECKSUM,
            selection_integrity=inventory,
            packed_row_fingerprint=_SHA_A,
            packed_content_by_name={"shard_00000": _SHA_B},
            feature_clusters_descriptor=_DESCRIPTOR,
            feature_delta={
                "mode": "splice_counterfactual",
                "neg_delta_ratio": ratio,
            },
            effective_buckets=1,
            segment_counts_by_stem={"shard_00000": 3},
            max_features_by_stem={"shard_00000": 7},
        )


def test_clustered_validation_rejects_non_json_cluster_descriptor(
    tmp_path: Path,
) -> None:
    _packed, selection = _write_shards(tmp_path)
    inventory = build_clustered_selection_shard_integrity([selection])
    with pytest.raises(FeatureClustersError, match="canonical feature cluster descriptor"):
        build_clustered_selection_validation(
            selection_checksum=_CHECKSUM,
            selection_integrity=inventory,
            packed_row_fingerprint=_SHA_A,
            packed_content_by_name={"shard_00000": _SHA_B},
            feature_clusters_descriptor={"bad": object()},
            feature_delta=None,
            effective_buckets=1,
            segment_counts_by_stem={"shard_00000": 3},
            max_features_by_stem={"shard_00000": 7},
        )


@pytest.mark.parametrize(
    ("feature_delta", "match"),
    [
        ({"mode": "splice_counterfactual"}, "exactly"),
        ({"mode": "", "neg_delta_ratio": 1.0}, "non-empty"),
        (
            {"mode": "splice_counterfactual", "neg_delta_ratio": True},
            "real number",
        ),
    ],
)
def test_clustered_validation_rejects_malformed_delta_descriptor(
    tmp_path: Path,
    feature_delta: dict[str, Any],
    match: str,
) -> None:
    _packed, selection = _write_shards(tmp_path)
    inventory = build_clustered_selection_shard_integrity([selection])
    with pytest.raises(FeatureClustersError, match=match):
        build_clustered_selection_validation(
            selection_checksum=_CHECKSUM,
            selection_integrity=inventory,
            packed_row_fingerprint=_SHA_A,
            packed_content_by_name={"shard_00000": _SHA_B},
            feature_clusters_descriptor=_DESCRIPTOR,
            feature_delta=feature_delta,
            effective_buckets=1,
            segment_counts_by_stem={"shard_00000": 3},
            max_features_by_stem={"shard_00000": 7},
        )


def test_clustered_validation_rejects_non_mapping_cluster_descriptor(
    tmp_path: Path,
) -> None:
    _packed, selection = _write_shards(tmp_path)
    inventory = build_clustered_selection_shard_integrity([selection])
    with pytest.raises(FeatureClustersError, match="canonical feature cluster descriptor"):
        build_clustered_selection_validation(
            selection_checksum=_CHECKSUM,
            selection_integrity=inventory,
            packed_row_fingerprint=_SHA_A,
            packed_content_by_name={"shard_00000": _SHA_B},
            feature_clusters_descriptor=[],  # ty: ignore[invalid-argument-type]
            feature_delta=None,
            effective_buckets=1,
            segment_counts_by_stem={"shard_00000": 3},
            max_features_by_stem={"shard_00000": 7},
        )


def test_clustered_validation_builder_requires_identical_shard_domains(
    tmp_path: Path,
) -> None:
    _packed, selection = _write_shards(tmp_path)
    inventory = build_clustered_selection_shard_integrity([selection])
    with pytest.raises(FeatureClustersError, match="identical"):
        build_clustered_selection_validation(
            selection_checksum=_CHECKSUM,
            selection_integrity=inventory,
            packed_row_fingerprint=_SHA_A,
            packed_content_by_name={},
            feature_clusters_descriptor=_DESCRIPTOR,
            feature_delta=None,
            effective_buckets=1,
            segment_counts_by_stem={"shard_00000": 3},
            max_features_by_stem={"shard_00000": 7},
        )


def test_complete_legacy_splice_v3_proof_is_accepted(tmp_path: Path) -> None:
    metadata, inventory, packed_fingerprint = _legacy_metadata(tmp_path)
    authority = resolve_clustered_selection_validation(
        metadata,
        selection_checksum=_CHECKSUM,
        selection_integrity=inventory,
        packed_row_fingerprint=packed_fingerprint,
        feature_clusters_descriptor=_DESCRIPTOR,
        feature_delta={
            "mode": "splice_counterfactual",
            "neg_delta_ratio": 1.0,
        },
        effective_buckets=1,
        surface="legacy fixture",
    )
    assert authority.kind == "legacy_splice_v3"
    assert authority.segment_counts_by_name == (
        ("shard_00000.safetensors.zst", 3),
    )
    assert authority.max_features_per_segment == 7


def _add_schema_v1_certificate_to_legacy_metadata(
    metadata: dict[str, Any],
    inventory: dict[str, Any],
    packed_fingerprint: str,
    packed_dir: Path,
    *,
    max_features_per_segment: int = 7,
) -> None:
    proof = resolve_packed_row_fingerprint_strict(packed_dir)
    assert proof.aggregate_sha256 == packed_fingerprint
    metadata[CLUSTERED_SELECTION_VALIDATION_KEY] = (
        build_clustered_selection_validation(
            selection_checksum=_CHECKSUM,
            selection_integrity=inventory,
            packed_row_fingerprint=packed_fingerprint,
            packed_content_by_name={
                "shard_00000": proof.files[0].content_sha256,
            },
            feature_clusters_descriptor=_DESCRIPTOR,
            feature_delta={
                "mode": "splice_counterfactual",
                "neg_delta_ratio": 1.0,
            },
            effective_buckets=1,
            segment_counts_by_stem={"shard_00000": 3},
            max_features_by_stem={
                "shard_00000": max_features_per_segment,
            },
        )
    )


def test_schema_v1_and_legacy_splice_proofs_must_agree_exactly(
    tmp_path: Path,
) -> None:
    metadata, inventory, packed_fingerprint = _legacy_metadata(tmp_path)
    _add_schema_v1_certificate_to_legacy_metadata(
        metadata,
        inventory,
        packed_fingerprint,
        tmp_path / "packed",
    )
    authority = resolve_clustered_selection_validation(
        metadata,
        selection_checksum=_CHECKSUM,
        selection_integrity=inventory,
        packed_row_fingerprint=packed_fingerprint,
        feature_clusters_descriptor=_DESCRIPTOR,
        feature_delta={
            "mode": "splice_counterfactual",
            "neg_delta_ratio": 1.0,
        },
        effective_buckets=1,
        surface="cross-proof fixture",
    )
    assert authority.kind == "schema_v1"

    metadata.pop(REMOTE_LOADER_KEY)
    _add_schema_v1_certificate_to_legacy_metadata(
        metadata,
        inventory,
        packed_fingerprint,
        tmp_path / "packed",
        max_features_per_segment=8,
    )
    with pytest.raises(FeatureClustersError, match="disagree"):
        resolve_clustered_selection_validation(
            metadata,
            selection_checksum=_CHECKSUM,
            selection_integrity=inventory,
            packed_row_fingerprint=packed_fingerprint,
            feature_clusters_descriptor=_DESCRIPTOR,
            feature_delta={
                "mode": "splice_counterfactual",
                "neg_delta_ratio": 1.0,
            },
            effective_buckets=1,
            surface="cross-proof fixture",
        )


def test_legacy_proof_rejects_non_splice_delta_mode(tmp_path: Path) -> None:
    metadata, inventory, packed_fingerprint = _legacy_metadata(tmp_path)
    with pytest.raises(FeatureClustersError, match="not eligible"):
        resolve_clustered_selection_validation(
            metadata,
            selection_checksum=_CHECKSUM,
            selection_integrity=inventory,
            packed_row_fingerprint=packed_fingerprint,
            feature_clusters_descriptor=_DESCRIPTOR,
            feature_delta=None,
            effective_buckets=1,
            surface="legacy fixture",
        )


def test_legacy_proof_rejects_remote_loader_disagreement(tmp_path: Path) -> None:
    metadata, inventory, packed_fingerprint = _legacy_metadata(tmp_path)
    metadata[REMOTE_LOADER_KEY] = build_remote_loader_certificate(
        selection_checksum=_CHECKSUM,
        selection_inventory_aggregate=inventory["aggregate_sha256"],
        packed_row_fingerprint=packed_fingerprint,
        max_features_per_segment=8,
        cluster_delta_semantics=CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS,
    )
    with pytest.raises(FeatureClustersError, match="disagree"):
        resolve_clustered_selection_validation(
            metadata,
            selection_checksum=_CHECKSUM,
            selection_integrity=inventory,
            packed_row_fingerprint=packed_fingerprint,
            feature_clusters_descriptor=_DESCRIPTOR,
            feature_delta={
                "mode": "splice_counterfactual",
                "neg_delta_ratio": 1.0,
            },
            effective_buckets=1,
            surface="legacy fixture",
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", 1, "statuses"),
        ("status", "in_progress", "statuses"),
        (
            CLUSTER_DELTA_SEMANTICS_KEY,
            "old-semantics",
            "statuses",
        ),
        (
            "total_packed_segments_selected",
            4,
            "total_packed_segments_selected",
        ),
        (
            "total_sidecar_splice_positions_referenced",
            6,
            "total_sidecar_splice_positions_referenced",
        ),
        (
            "total_delta_positions_selected",
            8,
            "total_delta_positions_selected",
        ),
    ],
)
def test_complete_splice_parser_rejects_every_summary_or_identity_disagreement(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    metadata, inventory, packed_fingerprint = _legacy_metadata(tmp_path)
    metadata["splice_counterfactual"][field] = value
    with pytest.raises(FeatureClustersError, match=match):
        parse_completed_splice_validation(
            metadata,
            selection_integrity=inventory,
            packed_row_fingerprint=packed_fingerprint,
            effective_buckets=1,
            feature_clusters_descriptor=_DESCRIPTOR,
            expected_stems={"shard_00000"},
            surface="legacy fixture",
        )


def test_complete_splice_parser_rejects_cluster_descriptor_disagreement(
    tmp_path: Path,
) -> None:
    metadata, inventory, packed_fingerprint = _legacy_metadata(tmp_path)
    metadata["feature_clusters"]["descriptor"] = {
        **_DESCRIPTOR,
        "content_sha256": _SHA_A,
    }
    with pytest.raises(FeatureClustersError, match="statuses"):
        parse_completed_splice_validation(
            metadata,
            selection_integrity=inventory,
            packed_row_fingerprint=packed_fingerprint,
            effective_buckets=1,
            feature_clusters_descriptor=_DESCRIPTOR,
            expected_stems={"shard_00000"},
            surface="legacy fixture",
        )


def test_complete_splice_parser_rejects_expected_stem_disagreement(
    tmp_path: Path,
) -> None:
    metadata, inventory, packed_fingerprint = _legacy_metadata(tmp_path)
    with pytest.raises(FeatureClustersError, match="exact"):
        parse_completed_splice_validation(
            metadata,
            selection_integrity=inventory,
            packed_row_fingerprint=packed_fingerprint,
            effective_buckets=1,
            feature_clusters_descriptor=_DESCRIPTOR,
            expected_stems={"different"},
            surface="legacy fixture",
        )


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("schema_version",), True, "schema_version"),
        (("semantics_version",), "old", "semantics_version"),
        (("status",), "in_progress", "status"),
        (("selection_checksum",), _SHA_A, "selection_checksum"),
        (
            ("selection_inventory_aggregate",),
            _SHA_A,
            "selection_inventory_aggregate",
        ),
        (("packed_row_fingerprint",), _SHA_A, "packed_row_fingerprint"),
        (
            ("feature_clusters_descriptor", "content_sha256"),
            _SHA_A,
            "feature_clusters_descriptor",
        ),
        (
            ("feature_delta",),
            {"mode": "other", "neg_delta_ratio": 1.0},
            "feature_delta",
        ),
        (("effective_buckets",), True, "effective_buckets"),
        (("shards", 0, "segment_count"), True, "segment_count"),
        (("shards", 0, "max_features_per_segment"), -1, "max_features"),
        (("shards", 0, "selection_name"), "../escape.safetensors.zst", "direct"),
        (("shards", 0, "packed_name"), "other.safetensors.zst", "must match"),
        (
            ("shards", 0, "selection_content_sha256"),
            _SHA_A,
            "certified inventory",
        ),
        (
            ("shards", 0, "packed_content_sha256"),
            "bad",
            "packed digest",
        ),
        (("aggregate_sha256",), _SHA_B, "aggregate mismatch"),
    ],
)
def test_clustered_validation_rejects_malformed_fields(
    tmp_path: Path,
    path: tuple[str | int, ...],
    value: object,
    match: str,
) -> None:
    certificate, inventory, packed_fingerprint = _certificate(tmp_path)
    current: Any = certificate
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = value
    with pytest.raises(FeatureClustersError, match=match):
        parse_clustered_selection_validation(
            certificate,
            selection_checksum=_CHECKSUM,
            selection_integrity=inventory,
            packed_row_fingerprint=packed_fingerprint,
            feature_clusters_descriptor=_DESCRIPTOR,
            feature_delta=None,
            effective_buckets=1,
            surface="test manifest",
        )


def test_clustered_validation_rejects_extra_key(tmp_path: Path) -> None:
    certificate, inventory, packed_fingerprint = _certificate(tmp_path)
    certificate["unexpected"] = None
    with pytest.raises(FeatureClustersError, match="must contain exactly"):
        parse_clustered_selection_validation(
            certificate,
            selection_checksum=_CHECKSUM,
            selection_integrity=inventory,
            packed_row_fingerprint=packed_fingerprint,
            feature_clusters_descriptor=_DESCRIPTOR,
            feature_delta=None,
            effective_buckets=1,
            surface="test manifest",
        )


def test_clustered_validation_rejects_empty_or_duplicate_shard_entries(
    tmp_path: Path,
) -> None:
    certificate, inventory, packed_fingerprint = _certificate(tmp_path)
    empty = {**certificate, "shards": []}
    with pytest.raises(FeatureClustersError, match="non-empty"):
        parse_clustered_selection_validation(
            empty,
            selection_checksum=_CHECKSUM,
            selection_integrity=inventory,
            packed_row_fingerprint=packed_fingerprint,
            feature_clusters_descriptor=_DESCRIPTOR,
            feature_delta=None,
            effective_buckets=1,
            surface="test manifest",
        )

    duplicate = {
        **certificate,
        "shards": [certificate["shards"][0], certificate["shards"][0]],
    }
    with pytest.raises(FeatureClustersError, match="strictly sorted"):
        parse_clustered_selection_validation(
            duplicate,
            selection_checksum=_CHECKSUM,
            selection_integrity=inventory,
            packed_row_fingerprint=packed_fingerprint,
            feature_clusters_descriptor=_DESCRIPTOR,
            feature_delta=None,
            effective_buckets=1,
            surface="test manifest",
        )


def test_clustered_validation_rejects_malformed_shard_entry_shape(
    tmp_path: Path,
) -> None:
    certificate, inventory, packed_fingerprint = _certificate(tmp_path)
    certificate["shards"][0]["extra"] = None
    with pytest.raises(FeatureClustersError, match="must contain exactly"):
        parse_clustered_selection_validation(
            certificate,
            selection_checksum=_CHECKSUM,
            selection_integrity=inventory,
            packed_row_fingerprint=packed_fingerprint,
            feature_clusters_descriptor=_DESCRIPTOR,
            feature_delta=None,
            effective_buckets=1,
            surface="test manifest",
        )


def test_clustered_validation_rejects_partial_certificate_shard_domain(
    tmp_path: Path,
) -> None:
    packed, selection = _write_shards(tmp_path)
    second_packed = packed.parent / "shard_00001.safetensors.zst"
    second_selection = selection.parent / "shard_00001.safetensors.zst"
    second_packed.write_bytes(b"second-packed-body")
    second_selection.write_bytes(b"second-selection-body")
    inventory = build_clustered_selection_shard_integrity(
        [selection, second_selection]
    )
    proof = resolve_packed_row_fingerprint_strict(packed.parent)
    certificate = build_clustered_selection_validation(
        selection_checksum=_CHECKSUM,
        selection_integrity=inventory,
        packed_row_fingerprint=proof.aggregate_sha256,
        packed_content_by_name={
            entry.name.removesuffix(".safetensors.zst"): entry.content_sha256
            for entry in proof.files
        },
        feature_clusters_descriptor=_DESCRIPTOR,
        feature_delta=None,
        effective_buckets=1,
        segment_counts_by_stem={
            "shard_00000": 3,
            "shard_00001": 4,
        },
        max_features_by_stem={
            "shard_00000": 7,
            "shard_00001": 8,
        },
    )
    certificate["shards"].pop()
    with pytest.raises(FeatureClustersError, match="domain differs"):
        parse_clustered_selection_validation(
            certificate,
            selection_checksum=_CHECKSUM,
            selection_integrity=inventory,
            packed_row_fingerprint=proof.aggregate_sha256,
            feature_clusters_descriptor=_DESCRIPTOR,
            feature_delta=None,
            effective_buckets=1,
            surface="partial fixture",
        )


def test_clustered_validation_rejects_cross_disagreeing_remote_evidence(
    tmp_path: Path,
) -> None:
    certificate, inventory, packed_fingerprint = _certificate(tmp_path)
    metadata = {
        CLUSTERED_SELECTION_VALIDATION_KEY: certificate,
        REMOTE_LOADER_KEY: build_remote_loader_certificate(
            selection_checksum=_CHECKSUM,
            selection_inventory_aggregate=inventory["aggregate_sha256"],
            packed_row_fingerprint=packed_fingerprint,
            max_features_per_segment=8,
        ),
    }

    with pytest.raises(FeatureClustersError, match="disagree"):
        resolve_clustered_selection_validation(
            metadata,
            selection_checksum=_CHECKSUM,
            selection_integrity=inventory,
            packed_row_fingerprint=packed_fingerprint,
            feature_clusters_descriptor=_DESCRIPTOR,
            feature_delta=None,
            effective_buckets=1,
            surface="test manifest",
        )


@pytest.mark.parametrize("schema_v1", [False, True])
def test_clustered_validation_wraps_malformed_remote_loader_evidence(
    tmp_path: Path,
    schema_v1: bool,
) -> None:
    if schema_v1:
        certificate, inventory, packed_fingerprint = _certificate(tmp_path)
        metadata = {
            CLUSTERED_SELECTION_VALIDATION_KEY: certificate,
            REMOTE_LOADER_KEY: {"schema_version": "bad"},
        }
        feature_delta = None
    else:
        metadata, inventory, packed_fingerprint = _legacy_metadata(tmp_path)
        metadata[REMOTE_LOADER_KEY] = {"schema_version": "bad"}
        feature_delta = {
            "mode": "splice_counterfactual",
            "neg_delta_ratio": 1.0,
        }

    with pytest.raises(FeatureClustersError, match="remote_loader.schema_version"):
        resolve_clustered_selection_validation(
            metadata,
            selection_checksum=_CHECKSUM,
            selection_integrity=inventory,
            packed_row_fingerprint=packed_fingerprint,
            feature_clusters_descriptor=_DESCRIPTOR,
            feature_delta=feature_delta,
            effective_buckets=1,
            surface="remote fixture",
        )


def test_schema_authority_rejects_current_packed_inventory_disagreement(
    tmp_path: Path,
) -> None:
    certificate, inventory, packed_fingerprint = _certificate(tmp_path)
    authority = parse_clustered_selection_validation(
        certificate,
        selection_checksum=_CHECKSUM,
        selection_integrity=inventory,
        packed_row_fingerprint=packed_fingerprint,
        feature_clusters_descriptor=_DESCRIPTOR,
        feature_delta=None,
        effective_buckets=1,
        surface="test manifest",
    )
    assert_validation_packed_inventory(
        authority,
        authority.packed_content_by_name or (),
        surface="test manifest",
    )
    with pytest.raises(FeatureClustersError, match="packed-file inventory"):
        assert_validation_packed_inventory(
            authority,
            (("shard_00000.safetensors.zst", _SHA_A),),
            surface="test manifest",
        )


def test_strict_packed_cache_hit_never_hashes_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed, _selection = _write_shards(tmp_path)
    initial = resolve_packed_row_fingerprint_strict(packed.parent)

    def forbidden(_path: Path) -> str:
        raise AssertionError("warm cache hit hashed packed body")

    monkeypatch.setattr(splice, "_file_fingerprint", forbidden)
    warm = resolve_packed_row_fingerprint_strict(packed.parent)
    assert warm.aggregate_sha256 == initial.aggregate_sha256


@pytest.mark.parametrize("allow_cache_write", [False, True])
def test_packed_cache_corruption_recomputes_from_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allow_cache_write: bool,
) -> None:
    packed, _selection = _write_shards(tmp_path)
    initial = resolve_packed_row_fingerprint_strict(packed.parent)
    initial.cache_path.write_text("{truncated", encoding="utf-8")
    real_hash = splice._file_fingerprint
    calls = 0

    def recording(path: Path) -> str:
        nonlocal calls
        calls += 1
        return real_hash(path)

    monkeypatch.setattr(splice, "_file_fingerprint", recording)
    repaired = resolve_packed_row_fingerprint_strict(
        packed.parent,
        allow_cache_write=allow_cache_write,
    )
    assert calls == 1
    assert repaired.aggregate_sha256 == initial.aggregate_sha256
    if allow_cache_write:
        assert json.loads(repaired.cache_path.read_text())["aggregate_sha256"] == (
            repaired.aggregate_sha256
        )
    else:
        assert repaired.cache_path.read_text() == "{truncated"


def test_forced_repair_replaces_a_symlink_cache_without_following_it(
    tmp_path: Path,
) -> None:
    packed, _selection = _write_shards(tmp_path)
    cache_path = packed.parent / ".packed-row-fingerprint-cache.json"
    outside = tmp_path / "outside-cache-target"
    outside.write_bytes(b"must remain untouched")
    cache_path.symlink_to(outside)

    resolve_packed_row_fingerprint_strict(
        packed.parent,
        force=True,
    )

    assert not cache_path.is_symlink()
    assert cache_path.is_file()
    assert outside.read_bytes() == b"must remain untouched"


def test_strict_packed_cache_stale_signature_refreshes(tmp_path: Path) -> None:
    packed, _selection = _write_shards(tmp_path)
    initial = resolve_packed_row_fingerprint_strict(packed.parent)
    packed.write_bytes(b"changed-packed-body")
    refreshed = resolve_packed_row_fingerprint_strict(packed.parent)
    assert refreshed.aggregate_sha256 != initial.aggregate_sha256


def test_strict_packed_cache_stale_read_only_hashes_without_publishing(
    tmp_path: Path,
) -> None:
    packed, _selection = _write_shards(tmp_path)
    initial = resolve_packed_row_fingerprint_strict(packed.parent)
    prior_cache_bytes = initial.cache_path.read_bytes()
    packed.write_bytes(b"changed-packed-body")

    refreshed = resolve_packed_row_fingerprint_strict(
        packed.parent,
        allow_cache_write=False,
    )

    assert refreshed.aggregate_sha256 != initial.aggregate_sha256
    assert initial.cache_path.read_bytes() == prior_cache_bytes


def test_strict_packed_file_fence_checks_name_replacement_and_symlink_target(
    tmp_path: Path,
) -> None:
    packed, _selection = _write_shards(tmp_path)
    resolution = resolve_packed_row_fingerprint_strict(packed.parent)
    entry = resolution.files[0]
    assert_packed_row_fingerprint_file_current(entry, packed)

    with pytest.raises(SpliceProvenanceError, match="changed after"):
        assert_packed_row_fingerprint_file_current(
            entry,
            packed.with_name("other.safetensors.zst"),
        )

    replacement = tmp_path / "replacement"
    replacement.write_bytes(packed.read_bytes())
    os.replace(replacement, packed)
    with pytest.raises(SpliceProvenanceError, match="changed after"):
        assert_packed_row_fingerprint_file_current(entry, packed)

    target = tmp_path / "retarget"
    target.write_bytes(packed.read_bytes())
    packed.unlink()
    packed.symlink_to(target)
    with pytest.raises(SpliceProvenanceError, match="changed after"):
        assert_packed_row_fingerprint_file_current(entry, packed)


def test_strict_packed_cache_detects_symlink_retarget(tmp_path: Path) -> None:
    packed_dir = tmp_path / "packed"
    packed_dir.mkdir()
    target_a = tmp_path / "target-a"
    target_b = tmp_path / "target-b"
    target_a.write_bytes(b"a")
    target_b.write_bytes(b"b")
    link = packed_dir / "shard_00000.safetensors.zst"
    link.symlink_to(target_a)
    initial = resolve_packed_row_fingerprint_strict(packed_dir)
    link.unlink()
    link.symlink_to(target_b)
    refreshed = resolve_packed_row_fingerprint_strict(packed_dir)
    assert refreshed.aggregate_sha256 != initial.aggregate_sha256


def test_strict_packed_resolution_final_fence_checks_symlink_target(
    tmp_path: Path,
) -> None:
    packed_dir = tmp_path / "packed"
    packed_dir.mkdir()
    target = tmp_path / "target"
    target.write_bytes(b"original")
    (packed_dir / "shard_00000.safetensors.zst").symlink_to(target)
    resolution = resolve_packed_row_fingerprint_strict(packed_dir)

    target.write_bytes(b"changed target bytes")

    with pytest.raises(SpliceProvenanceError, match="changed after"):
        assert_packed_row_fingerprint_resolution_current(resolution)


def test_packed_cache_write_failure_is_nonfatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed, _selection = _write_shards(tmp_path)
    expected = resolve_packed_row_fingerprint_strict(
        packed.parent, allow_cache_write=False
    )

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("read-only mount")

    monkeypatch.setattr(splice, "atomic_json_dump", fail_write)
    actual = resolve_packed_row_fingerprint_strict(packed.parent)
    assert actual == expected
    assert not actual.cache_path.exists()


def test_strict_packed_cache_read_only_missing_does_not_publish(
    tmp_path: Path,
) -> None:
    packed, _selection = _write_shards(tmp_path)
    result = resolve_packed_row_fingerprint_strict(
        packed.parent,
        allow_cache_write=False,
    )
    assert not result.cache_path.exists()


def test_forced_strict_resolution_hashes_and_validates_one_exact_body_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed, _selection = _write_shards(tmp_path)
    expected_body = packed.read_bytes()
    real_open = builtins.open
    opens = 0
    validated: list[tuple[Path, bytes]] = []

    def recording_open(path: Any, *args: Any, **kwargs: Any):
        nonlocal opens
        if isinstance(path, (str, os.PathLike)) and Path(path) == packed:
            opens += 1
        return real_open(path, *args, **kwargs)

    def validate(path: Path, compressed: bytes) -> None:
        validated.append((path, compressed))

    monkeypatch.setattr(builtins, "open", recording_open)
    monkeypatch.setattr(
        splice,
        "_file_fingerprint",
        lambda _path: pytest.fail("forced fused path used a second body hash"),
    )
    result = resolve_packed_row_fingerprint_strict(
        packed.parent,
        force=True,
        allow_cache_write=False,
        body_validator=validate,
    )

    assert opens == 1
    assert validated == [(packed, expected_body)]
    assert not result.cache_path.exists()


def test_forced_strict_resolution_fuses_every_shard_into_one_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed, _selection = _write_shards(tmp_path)
    second = packed.with_name("shard_00001.safetensors.zst")
    second.write_bytes(b"second-packed-body")
    expected = [(packed, packed.read_bytes()), (second, second.read_bytes())]
    real_open = builtins.open
    opens = {path: 0 for path, _body in expected}
    validated: list[tuple[Path, bytes]] = []

    def recording_open(path: Any, *args: Any, **kwargs: Any):
        resolved = Path(path) if isinstance(path, (str, os.PathLike)) else None
        if resolved in opens:
            opens[resolved] += 1
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", recording_open)
    monkeypatch.setattr(
        splice,
        "_file_fingerprint",
        lambda _path: pytest.fail("forced fused path used a second body hash"),
    )
    result = resolve_packed_row_fingerprint_strict(
        packed.parent,
        force=True,
        allow_cache_write=False,
        body_validator=lambda path, body: validated.append((path, body)),
    )

    assert opens == {packed: 1, second: 1}
    assert validated == expected
    assert [entry.name for entry in result.files] == [
        "shard_00000.safetensors.zst",
        "shard_00001.safetensors.zst",
    ]
    assert not result.cache_path.exists()


def test_forced_multishard_validator_failure_never_publishes_partial_evidence(
    tmp_path: Path,
) -> None:
    packed, _selection = _write_shards(tmp_path)
    second = packed.with_name("shard_00001.safetensors.zst")
    second.write_bytes(b"second-packed-body")
    validated: list[str] = []

    def reject_second(path: Path, _body: bytes) -> None:
        validated.append(path.name)
        if path == second:
            raise FeatureClustersError("second shard semantic failure")

    with pytest.raises(FeatureClustersError, match="second shard"):
        resolve_packed_row_fingerprint_strict(
            packed.parent,
            force=True,
            body_validator=reject_second,
        )

    assert validated == [
        "shard_00000.safetensors.zst",
        "shard_00001.safetensors.zst",
    ]
    assert not (packed.parent / ".packed-row-fingerprint-cache.json").exists()


def test_forced_fused_hash_tolerates_unsupported_page_cache_advice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed, _selection = _write_shards(tmp_path)

    def unsupported(*_args: object) -> None:
        raise OSError("filesystem does not support fadvise")

    monkeypatch.setattr(splice.os, "posix_fadvise", unsupported)
    resolve_packed_row_fingerprint_strict(
        packed.parent,
        force=True,
        allow_cache_write=False,
        body_validator=lambda _path, _compressed: None,
    )


def test_failed_forced_body_validation_never_publishes_cache(
    tmp_path: Path,
) -> None:
    packed, _selection = _write_shards(tmp_path)

    def reject(_path: Path, _compressed: bytes) -> None:
        raise FeatureClustersError("semantic body validation failed")

    with pytest.raises(FeatureClustersError, match="semantic body"):
        resolve_packed_row_fingerprint_strict(
            packed.parent,
            force=True,
            body_validator=reject,
        )
    assert not (packed.parent / ".packed-row-fingerprint-cache.json").exists()


def test_body_validator_is_rejected_without_forced_revalidation(
    tmp_path: Path,
) -> None:
    packed, _selection = _write_shards(tmp_path)
    with pytest.raises(TypeError, match="only with force=True"):
        resolve_packed_row_fingerprint_strict(
            packed.parent,
            body_validator=lambda _path, _compressed: None,
        )


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("force", 1),
        ("allow_cache_write", 1),
    ],
)
def test_strict_packed_resolution_rejects_bool_lookalikes(
    tmp_path: Path,
    keyword: str,
    value: object,
) -> None:
    packed, _selection = _write_shards(tmp_path)
    bad_value = cast(Any, value)
    with pytest.raises(TypeError, match=f"{keyword} must be a bool"):
        if keyword == "force":
            resolve_packed_row_fingerprint_strict(
                packed.parent,
                force=bad_value,
            )
        else:
            resolve_packed_row_fingerprint_strict(
                packed.parent,
                allow_cache_write=bad_value,
            )


def test_packed_cache_write_retains_packed_source_fence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed, _selection = _write_shards(tmp_path)
    real_store = splice._store_file_fingerprint_cache

    def store_then_mutate(*args: Any, **kwargs: Any) -> None:
        real_store(*args, **kwargs)
        packed.write_bytes(b"changed during cache write")

    monkeypatch.setattr(splice, "_store_file_fingerprint_cache", store_then_mutate)
    with pytest.raises(SpliceProvenanceError, match="inputs changed"):
        resolve_packed_row_fingerprint_strict(packed.parent)


def test_strict_packed_cache_rejects_non_regular_target(tmp_path: Path) -> None:
    packed_dir = tmp_path / "packed"
    packed_dir.mkdir()
    directory_target = tmp_path / "directory-target"
    directory_target.mkdir()
    (packed_dir / "shard_00000.safetensors.zst").symlink_to(directory_target)
    with pytest.raises(SpliceProvenanceError, match="not a regular file"):
        resolve_packed_row_fingerprint_strict(packed_dir)


def test_strict_packed_cache_signature_includes_ctime(tmp_path: Path) -> None:
    packed, _selection = _write_shards(tmp_path)
    initial = resolve_packed_row_fingerprint_strict(packed.parent)
    cache = json.loads(initial.cache_path.read_text())
    cache["files"][0]["signature"]["stat"]["ctime_ns"] += 1
    initial.cache_path.write_text(json.dumps(cache), encoding="utf-8")
    refreshed = resolve_packed_row_fingerprint_strict(packed.parent)
    assert refreshed.files == initial.files
    assert json.loads(refreshed.cache_path.read_text())["files"][0]["signature"] == (
        refreshed.files[0].signature.to_json()
    )


@pytest.mark.parametrize("section", ["lstat", "stat"])
@pytest.mark.parametrize(
    "field",
    ["mode", "ino", "size", "mtime_ns", "ctime_ns"],
)
def test_strict_packed_cache_invalidates_every_signature_field(
    tmp_path: Path,
    section: str,
    field: str,
) -> None:
    packed, _selection = _write_shards(tmp_path)
    initial = resolve_packed_row_fingerprint_strict(packed.parent)
    cache = json.loads(initial.cache_path.read_text())
    cache["files"][0]["signature"][section][field] += 1
    initial.cache_path.write_text(json.dumps(cache), encoding="utf-8")

    refreshed = resolve_packed_row_fingerprint_strict(packed.parent)
    assert refreshed.files == initial.files
    assert json.loads(refreshed.cache_path.read_text())["files"][0]["signature"] == (
        refreshed.files[0].signature.to_json()
    )


def test_packed_cache_wrong_namespace_is_recomputed(tmp_path: Path) -> None:
    packed, _selection = _write_shards(tmp_path)
    initial = resolve_packed_row_fingerprint_strict(packed.parent)
    cache = json.loads(initial.cache_path.read_text())
    cache["namespace"] = "not_packed_rows"
    initial.cache_path.write_text(json.dumps(cache), encoding="utf-8")
    repaired = resolve_packed_row_fingerprint_strict(packed.parent)
    assert repaired == initial
    assert json.loads(repaired.cache_path.read_text())["namespace"] == "packed_rows"


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", 999), ("aggregate_sha256", _SHA_A)],
)
def test_packed_cache_internal_corruption_is_recomputed(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    packed, _selection = _write_shards(tmp_path)
    initial = resolve_packed_row_fingerprint_strict(packed.parent)
    cache = json.loads(initial.cache_path.read_text())
    cache[field] = value
    initial.cache_path.write_text(json.dumps(cache), encoding="utf-8")
    repaired = resolve_packed_row_fingerprint_strict(packed.parent)
    assert repaired == initial
    assert json.loads(repaired.cache_path.read_text())[field] != value


def test_packed_cache_unreadable_file_is_recomputed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed, _selection = _write_shards(tmp_path)
    initial = resolve_packed_row_fingerprint_strict(packed.parent)
    real_read_text = Path.read_text

    def fail_cache_read(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == initial.cache_path:
            raise PermissionError("injected unreadable cache")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_cache_read)
    actual = resolve_packed_row_fingerprint_strict(packed.parent)
    assert actual == initial


def test_packed_cache_lstat_failure_is_recomputed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed, _selection = _write_shards(tmp_path)
    initial = resolve_packed_row_fingerprint_strict(packed.parent)
    real_lstat = Path.lstat

    def fail_cache_lstat(path: Path):
        if path == initial.cache_path:
            raise OSError("injected lstat failure")
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", fail_cache_lstat)
    actual = resolve_packed_row_fingerprint_strict(packed.parent)
    assert actual == initial


def test_legacy_cache_lstat_failure_remains_a_visible_cache_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_path = tmp_path / "legacy-cache.json"
    real_lstat = Path.lstat

    def fail_cache_lstat(path: Path):
        if path == cache_path:
            raise OSError("injected legacy lstat failure")
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", fail_cache_lstat)
    assert (
        splice._load_file_fingerprint_cache(
            cache_path,
            namespace="legacy",
        )
        is None
    )


@pytest.mark.parametrize("cache_kind", ["non_file", "corrupt"])
def test_legacy_cache_defects_remain_visible_cache_misses(
    tmp_path: Path,
    cache_kind: str,
) -> None:
    cache_path = tmp_path / "legacy-cache.json"
    if cache_kind == "non_file":
        cache_path.mkdir()
    else:
        cache_path.write_bytes(b"not-json")

    assert (
        splice._load_file_fingerprint_cache(
            cache_path,
            namespace="legacy",
        )
        is None
    )


def test_packed_cache_directory_is_a_nonfatal_miss(tmp_path: Path) -> None:
    packed, _selection = _write_shards(tmp_path)
    initial = resolve_packed_row_fingerprint_strict(packed.parent)
    initial.cache_path.unlink()
    initial.cache_path.mkdir()
    actual = resolve_packed_row_fingerprint_strict(packed.parent)
    assert actual == initial
    assert actual.cache_path.is_dir()


def test_strict_packed_cache_hit_rejects_input_change_during_signature_fence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed, _selection = _write_shards(tmp_path)
    resolve_packed_row_fingerprint_strict(packed.parent)
    real_capture = splice._capture_file_signatures
    calls = 0

    def mutate_before_second_capture(paths: list[Path]):
        nonlocal calls
        calls += 1
        if calls == 2:
            packed.write_bytes(b"changed-during-cache-hit")
        return real_capture(paths)

    monkeypatch.setattr(
        splice,
        "_capture_file_signatures",
        mutate_before_second_capture,
    )
    with pytest.raises(SpliceProvenanceError, match="cache hit"):
        resolve_packed_row_fingerprint_strict(packed.parent)


def test_strict_packed_cache_addition_is_stale(tmp_path: Path) -> None:
    packed, _selection = _write_shards(tmp_path)
    resolve_packed_row_fingerprint_strict(packed.parent)
    second = packed.parent / "shard_00001.safetensors.zst"
    second.write_bytes(os.urandom(8))
    refreshed = resolve_packed_row_fingerprint_strict(packed.parent)
    assert [entry.name for entry in refreshed.files] == [packed.name, second.name]


def test_strict_packed_cache_rename_is_stale(tmp_path: Path) -> None:
    packed, _selection = _write_shards(tmp_path)
    initial = resolve_packed_row_fingerprint_strict(packed.parent)
    renamed = packed.with_name("shard_00001.safetensors.zst")
    packed.rename(renamed)
    refreshed = resolve_packed_row_fingerprint_strict(renamed.parent)
    assert refreshed.aggregate_sha256 != initial.aggregate_sha256


def test_strict_packed_cache_replacement_is_stale(tmp_path: Path) -> None:
    packed, _selection = _write_shards(tmp_path)
    initial = resolve_packed_row_fingerprint_strict(packed.parent)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(packed.read_bytes())
    os.replace(replacement, packed)
    refreshed = resolve_packed_row_fingerprint_strict(packed.parent)
    assert refreshed.aggregate_sha256 == initial.aggregate_sha256
    assert refreshed.files[0].signature != initial.files[0].signature


def test_strict_packed_cache_removal_cannot_be_hidden_by_cached_list(
    tmp_path: Path,
) -> None:
    packed, _selection = _write_shards(tmp_path)
    resolve_packed_row_fingerprint_strict(packed.parent)
    packed.unlink()
    with pytest.raises(SpliceProvenanceError, match="no packed row shards"):
        resolve_packed_row_fingerprint_strict(packed.parent)
