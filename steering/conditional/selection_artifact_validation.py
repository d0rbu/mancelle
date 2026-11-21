"""Shared read-only validation for complete packed selection artifacts.

The precomputed training loader and clustered inline steering both consume the
same ``packed rows + sel/<hash>`` family.  This module owns their common
startup certificate check so selection identity, splice provenance, and static
cluster rules cannot drift between the two consumers.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from errors import FeatureMaskError
from steering.conditional.delta import (
    CLUSTER_DELTA_SEMANTICS_KEY,
    DeltaMode,
    parse_feature_delta,
    splice_counterfactual_manifest_identity,
)
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    assert_clustered_selection_shard_integrity,
    assert_feature_clusters_selection_metadata,
)
from steering.conditional.splice_counterfactual import (
    canonical_feature_delta_dict,
    derived_feature_fingerprint,
    packed_row_fingerprint,
    splice_sidecar_fingerprint,
    validate_packed_provenance,
)
from steering.conditional.splice_selection_batching import (
    canonical_splice_selection_batching,
)
from steering.overlay_io import glob_overlay_shards, read_manifest


def validate_complete_selection_dir(
    *,
    packed_row_dir: Path,
    selection_hash: str,
    configured_mask: dict[str, Any] | None,
    label: str,
    feature_clusters: FeatureClusterConsolidation | None = None,
    delta_config: dict[str, Any] | None = None,
    splice_feature_dir: Path | None = None,
    boundary_fingerprint: str | None = None,
    splice_batch_size: object = None,
    remote_materializer: Any | None = None,
    packed_materializer: Any | None = None,
    remote_packed_streaming: bool = False,
    remote_derived_feature_fingerprint: str | None = None,
) -> Path:
    """Require exact packed/selection coverage and immutable identity.

    ``boundary_fingerprint`` is already derived from the actual runtime bucket
    boundaries.  Passing the digest instead of a tensor keeps this seam
    read-only and prevents consumers from inventing a second fingerprint rule.
    """
    packed_row_dir = Path(packed_row_dir)
    sel_dir = packed_row_dir / "sel" / selection_hash
    if not sel_dir.is_dir() or (
        remote_materializer is None and not glob_overlay_shards(sel_dir)
    ):
        raise FileNotFoundError(
            f"{label} requires precomputed selection shards at {sel_dir}; run "
            f"precompute_selected_features with packed_row_dir={packed_row_dir}."
        )
    manifest = read_manifest(sel_dir)
    if manifest is None:
        raise ValueError(
            f"{label} selection dir {sel_dir} is missing manifest.json; re-run "
            "precompute_selected_features."
        )
    manifest_selection_hash = manifest.get("checksum")
    if manifest_selection_hash != selection_hash:
        raise ValueError(
            f"{label} selection dir {sel_dir} has manifest selection checksum "
            f"{manifest_selection_hash!r}, expected {selection_hash!r}. The "
            "directory was copied, renamed, or produced for a different "
            "selection config; re-run precompute_selected_features."
        )
    meta = manifest.get("metadata", {})
    if not isinstance(meta, dict):
        raise ValueError(
            f"{label} selection dir {sel_dir} has invalid manifest metadata; "
            "re-run precompute_selected_features."
        )
    manifest_mask = meta.get("feature_mask")
    if manifest_mask != configured_mask:
        raise FeatureMaskError(
            f"{label} sel/ manifest feature_mask mismatch for {sel_dir}: "
            f"manifest has {manifest_mask!r}, configured {configured_mask!r}. "
            "Re-run precompute_selected_features with the current config."
        )
    assert_feature_clusters_selection_metadata(
        feature_clusters,
        meta,
        surface=f"{label} sel/ manifest {sel_dir}",
    )
    if feature_clusters is not None and remote_materializer is None:
        assert_clustered_selection_shard_integrity(
            sel_dir,
            meta,
            surface=f"{label} sel/ manifest {sel_dir}",
        )
    manifest_packed = meta.get("packed_rows")
    if packed_materializer is not None and not remote_packed_streaming:
        raise ValueError(
            f"{label}: packed_materializer requires remote_packed_streaming"
        )
    expected_packed_hash = (
        packed_materializer.packed_row_fingerprint
        if packed_materializer is not None
        else packed_row_fingerprint(
            packed_row_dir,
            cache_dir=packed_row_dir,
        )
    )
    actual_packed_hash = (
        manifest_packed.get("packed_row_fingerprint")
        if isinstance(manifest_packed, dict)
        else None
    )
    if actual_packed_hash != expected_packed_hash:
        raise ValueError(
            f"{label} sel/ manifest packed row fingerprint mismatch for "
            f"{sel_dir}: manifest has {actual_packed_hash!r}, current packed "
            f"rows hash to {expected_packed_hash!r}. Re-run "
            "precompute_selected_features after regenerating packed rows."
        )

    splice_meta: dict[str, Any] | None = None
    if delta_config is None:
        if meta.get("feature_delta") is not None:
            raise ValueError(
                f"{label} selection dir {sel_dir} records feature_delta "
                f"{meta.get('feature_delta')!r}, but this packed validation pass "
                "is delta-free. Re-run precompute_selected_features with "
                "feature_delta=null and no feature_selection.delta_weights."
            )
    else:
        parsed_delta = parse_feature_delta(delta_config)
        if parsed_delta is None:
            raise ValueError(f"{label}: delta_config unexpectedly parsed to None")
        expected_delta = canonical_feature_delta_dict(parsed_delta)
        if meta.get("feature_delta") != expected_delta:
            raise ValueError(
                f"{label} sel/ manifest feature_delta mismatch for {sel_dir}: "
                f"manifest has {meta.get('feature_delta')!r}, configured "
                f"{expected_delta!r}. Re-run precompute_selected_features."
            )
        if parsed_delta.mode is DeltaMode.splice_counterfactual:
            splice_meta_raw = meta.get("splice_counterfactual")
            if not isinstance(splice_meta_raw, dict):
                raise ValueError(
                    f"{label} sel/ manifest {sel_dir} is missing "
                    "splice_counterfactual metadata. Re-run "
                    "precompute_selected_features."
                )
            splice_meta = splice_meta_raw
            expected_splice_identity = splice_counterfactual_manifest_identity(
                clustered=feature_clusters is not None
            )
            if any(
                splice_meta.get(key) != expected
                for key, expected in expected_splice_identity.items()
            ) or (
                feature_clusters is None
                and CLUSTER_DELTA_SEMANTICS_KEY in splice_meta
            ):
                raise ValueError(
                    f"{label} sel/ manifest cluster-delta semantics mismatch "
                    f"for {sel_dir}: manifest has schema/semantics "
                    f"{splice_meta.get('schema_version')!r}/"
                    f"{splice_meta.get(CLUSTER_DELTA_SEMANTICS_KEY)!r}, "
                    f"configured path requires {expected_splice_identity!r}. "
                    "Re-run precompute_selected_features."
                )
            expected_batching = canonical_splice_selection_batching(
                delta_config,
                splice_batch_size,
            )
            batching_present = "selection_batching" in splice_meta
            recorded_batching = splice_meta.get("selection_batching")
            if (expected_batching is None and batching_present) or (
                expected_batching is not None
                and recorded_batching != expected_batching
            ):
                raise ValueError(
                    f"{label} sel/ manifest selection batching mismatch for "
                    f"{sel_dir}: manifest has {recorded_batching!r}, configured "
                    f"mode requires {expected_batching!r}. Re-run "
                    "precompute_selected_features with the current config."
                )
        if splice_feature_dir is not None:
            if splice_meta is None:
                raise ValueError(
                    f"{label}: splice_feature_dir is valid only for "
                    "feature_delta.mode=splice_counterfactual"
                )
            packed_info = (
                packed_materializer.packed_provenance_info()
                if packed_materializer is not None
                else validate_packed_provenance(
                    packed_row_dir,
                    cache_dir=packed_row_dir,
                )
            )
            manifest_prov = meta.get("packed_provenance")
            expected_prov = {
                "packed_stem_fingerprint": packed_info.packed_stem_fingerprint,
                "packed_row_fingerprint": packed_info.packed_row_fingerprint,
                "provenance_shard_count": len(packed_info.provenance_shard_paths),
                "provenance_fingerprint": packed_info.provenance_fingerprint,
            }
            if manifest_prov != expected_prov:
                raise ValueError(
                    f"{label} sel/ manifest packed_provenance mismatch for "
                    f"{sel_dir}: manifest has {manifest_prov!r}, current "
                    f"packed rows require {expected_prov!r}. Re-run "
                    "precompute_selected_features after regenerating packed rows."
                )
            if splice_meta.get("derived_shard_stems") != packed_info.base_shard_stems:
                raise ValueError(
                    f"{label} sel/ manifest derived_shard_stems mismatch for "
                    f"{sel_dir}. Re-run precompute_selected_features."
                )
            expected_sidecar = splice_sidecar_fingerprint(
                splice_feature_dir,
                cache_dir=packed_row_dir,
            )
            actual_sidecar = splice_meta.get("sidecar_fingerprint")
            if actual_sidecar != expected_sidecar:
                raise ValueError(
                    f"{label} sel/ manifest sidecar_fingerprint mismatch for "
                    f"{sel_dir}: manifest has {actual_sidecar!r}, current "
                    f"sidecars hash to {expected_sidecar!r}. Re-run "
                    "precompute_selected_features after regenerating splice sidecars."
                )
            actual_derived = splice_meta.get("derived_feature_fingerprint")
            if remote_packed_streaming:
                if (
                    not isinstance(actual_derived, str)
                    or re.fullmatch(r"[0-9a-f]{64}", actual_derived) is None
                ):
                    raise ValueError(
                        f"{label} sel/ manifest "
                        "derived_feature_fingerprint mismatch for "
                        f"{sel_dir}: manifest has {actual_derived!r}, expected "
                        "64 lowercase hexadecimal characters for the remote "
                        "cross-family lineage lock. Re-run "
                        "precompute_selected_features and republish the "
                        "selection manifest."
                    )
                if (
                    remote_derived_feature_fingerprint is not None
                    and actual_derived != remote_derived_feature_fingerprint
                ):
                    raise ValueError(
                        f"{label} sel/ manifest derived_feature_fingerprint "
                        f"mismatch for {sel_dir}: manifest has "
                        f"{actual_derived!r}, verified remote startup contracts "
                        f"require {remote_derived_feature_fingerprint!r}. Re-run "
                        "precompute_selected_features and republish the remote "
                        "selection manifest."
                    )
            else:
                if remote_derived_feature_fingerprint is not None:
                    raise ValueError(
                        f"{label}: remote_derived_feature_fingerprint requires "
                        "remote_packed_streaming"
                    )
                expected_derived = derived_feature_fingerprint(
                    splice_feature_dir,
                    expected_stems=packed_info.base_shard_stems,
                    cache_dir=packed_row_dir,
                )
                if actual_derived != expected_derived:
                    raise ValueError(
                        f"{label} sel/ manifest derived_feature_fingerprint "
                        f"mismatch for {sel_dir}: manifest has "
                        f"{actual_derived!r}, current derived features hash to "
                        f"{expected_derived!r}. Re-run "
                        "precompute_selected_features after regenerating "
                        "derived feature shards."
                    )
            if boundary_fingerprint is not None:
                actual_boundaries = splice_meta.get("boundary_fingerprint")
                if actual_boundaries != boundary_fingerprint:
                    raise ValueError(
                        f"{label} sel/ manifest boundary_fingerprint mismatch "
                        f"for {sel_dir}: manifest has {actual_boundaries!r}, "
                        f"current boundaries hash to {boundary_fingerprint!r}. "
                        "Re-run precompute_selected_features after changing "
                        "statistics, buckets, or threshold settings."
                    )

    packed_stems = (
        packed_materializer.stems()
        if packed_materializer is not None
        else {
            path.stem.split(".")[0]
            for path in glob_overlay_shards(packed_row_dir)
        }
    )
    selection_stems = (
        set(remote_materializer.shard_paths())
        if remote_materializer is not None
        else {path.stem.split(".")[0] for path in glob_overlay_shards(sel_dir)}
    )
    missing = sorted(packed_stems - selection_stems)
    extra = sorted(selection_stems - packed_stems)
    if missing or extra:
        raise FileNotFoundError(
            f"{label} selection shards at {sel_dir} must exactly cover packed "
            f"dir {packed_row_dir}; missing={missing[:5]}, extra={extra[:5]}. "
            "Re-run precompute_selected_features (resume fills missing shards)."
        )
    return sel_dir
