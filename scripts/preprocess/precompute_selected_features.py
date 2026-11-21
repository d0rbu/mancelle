"""Precompute selected features from base packed row shards.

Applies a specific feature selection config (thresholding/sampling +
discretization + negative sampling) to the raw features in base packed
row shards, writing the results to ``sel/<selection_hash>/`` subdirectory.

At training time, the precomputed loader can read these shards to skip
all runtime feature processing.

Usage::

    uv run python -m scripts.preprocess.precompute_selected_features \
        packed_row_dir=pct/.../tok/.../c1024/sall \
        feature_delta=null feature_selection.delta_weights=null
"""

from __future__ import annotations

import faulthandler
import hashlib
import itertools
import json
import multiprocessing
import os
import re
import resource
import shutil
import stat
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, cast

import hydra
import psutil
import torch as t
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from errors import FeatureClustersError, GradientAttributionError
from nanochat.atomic_io import atomic_write_bytes
from steering.conditional.cluster_segment_selection import select_clustered_segment
from steering.conditional.clustered_selection_validation import (
    CLUSTERED_SELECTION_VALIDATION_KEY,
    assert_validation_packed_inventory,
    build_clustered_selection_validation,
    parse_clustered_selection_validation,
    resolve_clustered_selection_validation,
)
from steering.conditional.delta import (
    CLUSTER_DELTA_SEMANTICS_KEY,
    CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS,
    ClusteredCounterfactualDeltaSegResult,
    DeltaMode,
    DeltaSamplingWeights,
    compute_doc_clustered_counterfactual_deltas,
    compute_doc_counterfactual_deltas,
    parse_feature_delta,
    sample_delta_negatives,
    splice_counterfactual_manifest_identity,
)
from steering.conditional.discretization import (
    compute_percentiles,
    discretize_activations,
    percentile_to_bucket,
)
from steering.conditional.feature_clusters import (
    SELECTION_SHARD_INTEGRITY_KEY,
    FeatureClusterConsolidation,
    assert_clustered_selection_shard_integrity,
    assert_feature_clusters_selection_metadata,
    build_clustered_selection_shard_integrity,
    canonical_feature_clusters_descriptor,
    extend_clustered_selection_shard_integrity,
    parse_clustered_selection_shard_inventory,
    resolve_feature_clusters,
    validated_clustered_selection_shard_names,
)
from steering.conditional.feature_index import (
    FeatureIndex,
    FeatureIndexCatalog,
    FeatureIndexCatalogError,
    TransportFeatureIndexCatalog,
)
from steering.conditional.feature_mask import canonical_dict, resolve_feature_mask
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    SamplingBundle,
    ThresholdSelectionConfig,
    build_feature_pipeline_from_cfg_block,
    negative_sampling_mask,
)
from steering.conditional.freq_blacklist import maybe_apply_frequency_blacklist
from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
    magnitude_to_percentile,
    normalized_percentile_to_uint8,
    uint8_to_percentile,
)
from steering.conditional.sampling import (
    NegativeBundle,
    sample_features_batched,
    sample_inactive_features_batched,
    sample_k,
)
from steering.conditional.shared_stem_store import SharedStemStore
from steering.conditional.shared_store_lifecycle import (
    ParentStoreLifecycle,
    register_worker_store_lease,
    sweep_orphaned_stores,
)
from steering.conditional.splice_counterfactual import (
    PackedFingerprintFile,
    PackedFingerprintResolution,
    PackedProvenanceInfo,
    assert_packed_row_fingerprint_file_current,
    assert_packed_row_fingerprint_resolution_current,
    canonical_feature_delta_dict,
    derived_feature_fingerprint,
    packed_row_fingerprint,
    resolve_counterfactual_refs,
    resolve_packed_row_fingerprint_strict,
    splice_sidecar_fingerprint,
    tensor_content_fingerprint,
    validate_counterfactual_delta_results,
    validate_packed_provenance,
)
from steering.conditional.splice_provenance import (
    SpliceProvenanceError,
    SpliceProvenanceIndex,
)
from steering.conditional.splice_resource_admission import (
    AdmissionPolicy,
    SpliceResourceAdmissionError,
    admit_splice_resources,
    capture_capacity_snapshot,
    evaluate_splice_resource_admission_from_layouts,
)
from steering.conditional.splice_selection_batching import (
    SPLICE_SELECTION_BATCHING_ALGORITHM,
    canonical_splice_selection_batching,
    resolve_splice_selection_batch_size,
)
from steering.conditional.splice_selection_certificate import (
    SELECTION_SHARD_SUFFIX as _SELECTION_SHARD_SUFFIX,
)
from steering.conditional.splice_selection_certificate import (
    SPLICE_RESUME_METADATA_KEY,
    SpliceSelectionStats,
    _is_lower_sha256,
)
from steering.conditional.splice_selection_certificate import (
    build_splice_resume_metadata as _build_splice_resume_metadata,
)
from steering.conditional.splice_selection_certificate import (
    parse_splice_resume_metadata as _parse_splice_resume_metadata,
)
from steering.conditional.thresholding import (
    ThresholdConfig,
    ThresholdType,
    apply_threshold,
    select_from_packed_percentiles,
)
from steering.feature_io import RegularFileIdentity, glob_features, read_shard_metadata
from steering.feature_paths import (
    PRODUCER_IGNORE_BOS_FEATURES,
    PoolingMode,
    parse_feature_dir,
    resolve_output_dir,
    verify_segment_size,
)
from steering.feature_shard_splice import PROVENANCE_DIRNAME
from steering.gradient_overlay import (
    GRAD_DOC_SEG_OFFSETS_KEY,
    GRAD_OFFSETS_KEY,
    GRAD_SCORES_KEY,
    PROVENANCE_BASE_SHARD_IDX_KEY,
    PROVENANCE_DOC_IDX_KEY,
    PROVENANCE_SEG_ORDINAL_KEY,
    GradientRankingSpec,
    assert_descriptor_matches,
    canonical_gradient_descriptor,
    canonical_gradient_selection_dict,
    compute_gradient_overlay_checksum,
    provenance_dir,
    resolve_gradient_spec_from_feature_selection,
)
from steering.overlay_io import (
    MANIFEST_FILENAME,
    OverlayKind,
    OverlayStaleError,
    discover_overlay,
    glob_overlay_shards,
    load_overlay_shard,
    overlay_shard_path,
    read_manifest,
    resolve_single_percentile_overlay,
    write_manifest,
)
from steering.packed_batch_io import (
    PackedRowShard,
    PackedRowShardHandle,
    SelectionShard,
    compute_selection_hash,
    load_packed_row_shard,
    load_selection_shard,
    open_packed_row_shard,
    save_selection_shard,
)
from steering.refined_types import AtLeastTwoInt, NonNegativeInt, PositiveInt
from steering.remote_preselection import (
    MaterializedPackedShard,
    RemoteFeatureShardMaterializer,
    RemotePreselectionConfig,
    RemotePreselectionError,
    build_remote_preselection_receipt,
    build_remote_unspliced_preselection_receipt,
    build_transport_feature_catalog,
    build_trusted_remote_preselection_source,
    discover_remote_preselection_receipts,
    discover_remote_unspliced_preselection_receipts,
    download_remote_object_bytes,
    download_remote_packed_preselection_inventory,
    local_object_info_for_remote_path,
    materialize_remote_packed_shard,
    publish_remote_preselection_manifest,
    publish_remote_preselection_shard,
    publish_remote_unspliced_preselection_shard,
    resolve_remote_preselection_prefixes,
    selection_integrity_from_receipts,
    selection_integrity_from_unspliced_receipts,
    validate_local_remote_packed_controls,
    validate_local_remote_preselection_controls,
)
from steering.remote_training import (
    REMOTE_LOADER_KEY,
    REMOTE_PACKED_INVENTORY_FILENAME,
    BucketBackend,
    HfBucketBackend,
    RemoteObjectInfo,
    RemoteTrainingConfig,
    build_remote_loader_certificate,
    parse_remote_loader_certificate,
)
from steering.removed_features import reject_feature_clustering_config
from steering.sae_resolver import ActivationType, SAEWidth
from steering.safetensors_layout import (
    parse_semantic_layout,
    read_compressed_safetensors_layout,
)
from steering.stats_io import find_statistics_path, load_statistics
from steering.stats_types import ExtractionStatistics

_ATOMIC_SELECTION_TEMP_RE = re.compile(
    r"^(?P<stem>.+)\.safetensors\.zst\.tmp\."
    r"(?P<pid>[1-9][0-9]*)\.(?P<nonce>[0-9a-f]{8})$"
)
# Threshold types that are safe to evaluate against percentile-only inputs in
# this precompute script (where ``feature_mags`` are not reconstructed from the
# base packed shard and a tensor of zeros is passed in their place). All others
# require real magnitudes — we raise rather than silently miscompute. See
# slice 5 map P1 #3 + consolidation manifest §4.10.
_PERCENTILE_ONLY_THRESHOLD_TYPES: frozenset[ThresholdType] = frozenset(
    {
        ThresholdType.feature_percentile,
        ThresholdType.feature_percentile_segment_topk,
    }
)


def _select_from_packed_percentiles(
    *,
    fids: t.Tensor,
    pcts: t.Tensor,
    selection: ThresholdSelectionConfig,
) -> t.Tensor:
    """Apply threshold selection to packed rows in percentile space.

    Packed row shards intentionally do not store raw activation magnitudes.
    Their row-aligned percentile bytes are enough for the two threshold modes
    allowed by this producer: feature-percentile filtering, optionally followed
    by a segment top-k cap. The shared math itself lives in
    ``steering.conditional.thresholding.select_from_packed_percentiles``; this
    wrapper only owns the producer-specific unsupported-type error (a raw-
    magnitude threshold type has no meaning against percentile-only packed
    rows, and the message here points callers at live feature selection
    instead).
    """
    th_type = selection.selection.threshold_type
    threshold = selection.selection.threshold
    if th_type not in _PERCENTILE_ONLY_THRESHOLD_TYPES:
        supported = sorted(item.value for item in _PERCENTILE_ONLY_THRESHOLD_TYPES)
        raise ValueError(
            f"Threshold type {th_type!r} requires raw feature magnitudes, but "
            "precompute_selected_features only stores percentiles. Supported "
            f"types in precompute: {supported}. Run live feature selection at "
            "training time instead."
        )
    return select_from_packed_percentiles(fids, pcts, th_type, threshold)


def _lookup_selected_percentiles(
    *,
    selected_fids: t.Tensor,
    candidate_fids: t.Tensor,
    candidate_pcts: t.Tensor,
) -> t.Tensor:
    if selected_fids.numel() == 0:
        raise ValueError(
            "_lookup_selected_percentiles requires at least one selected feature; "
            "empty selections must be handled by the caller before percentile "
            "lookup."
        )
    if candidate_fids.numel() == 0:
        first = int(selected_fids[0].item())
        raise ValueError(
            f"selected feature id {first} has 0 matching packed-row percentile "
            "entries; expected exactly one. The selection output and packed "
            "row are out of sync."
        )
    is_strictly_sorted = candidate_fids.numel() == 1 or bool(
        (candidate_fids[1:] > candidate_fids[:-1]).all()
    )
    if is_strictly_sorted:
        positions = t.searchsorted(candidate_fids, selected_fids)
        in_bounds = positions < candidate_fids.numel()
        safe_positions = positions.clamp(max=candidate_fids.numel() - 1)
        matched = in_bounds & (candidate_fids[safe_positions] == selected_fids)
        if not bool(matched.all()):
            first_bad = int((~matched).nonzero(as_tuple=True)[0][0].item())
            feature_id = int(selected_fids[first_bad].item())
            raise ValueError(
                f"selected feature id {feature_id} has 0 matching packed-row "
                "percentile entries; expected exactly one. The selection output "
                "and packed row are out of sync."
            )
        # ``tolist`` + a fresh float32 CPU tensor deliberately preserves the
        # reference function's dtype/device/rounding contract exactly.
        return t.tensor(
            candidate_pcts[positions].tolist(),
            dtype=t.float32,
        )
    vals: list[float] = []
    for sf in selected_fids.tolist():
        idx = (candidate_fids == sf).nonzero(as_tuple=True)[0]
        if len(idx) != 1:
            raise ValueError(
                "selected feature id "
                f"{int(sf)} has {len(idx)} matching packed-row percentile entries; "
                "expected exactly one. The selection output and packed row are "
                "out of sync."
            )
        vals.append(float(candidate_pcts[idx[0]].item()))
    return t.tensor(vals, dtype=t.float32)


def _pipeline_consumes_selection_rng(pipeline_config: FeaturePipelineConfig) -> bool:
    return (
        isinstance(pipeline_config.selection_config, SamplingBundle)
        or pipeline_config.negative_bundle is not None
    )


def assert_precompute_supports_pipeline(pipeline_config: FeaturePipelineConfig) -> None:
    """Reject selection precompute modes that packed rows cannot represent."""
    if pipeline_config.delta_bundle is None:
        return
    if pipeline_config.delta_bundle.config.mode is DeltaMode.splice_counterfactual:
        return
    raise ValueError(
        "precompute_selected_features does not support feature_delta modes "
        "prev_segment or cum_max. Selection shards are aligned to packed rows, "
        "and those modes require carried document-history state that packed "
        "rows do not encode. Only feature_delta.mode=splice_counterfactual can "
        "be precomputed from packed provenance."
    )


def assert_gradient_ranking_supported(
    pipeline_config: FeaturePipelineConfig,
    grad_spec: GradientRankingSpec | None,
) -> None:
    """v1 restrictions for gradient-attribution ranking (parse already
    guarantees SamplingSelection + mode=top_k_percentile + no delta_weights)."""
    if grad_spec is None:
        return
    if not isinstance(pipeline_config.selection_config, SamplingBundle):
        raise GradientAttributionError(
            "feature_selection.gradient requires the sampling selection shape "
            "(SamplingBundle); threshold selections have no gradient channel."
        )
    if pipeline_config.delta_bundle is not None:
        raise GradientAttributionError(
            "feature_selection.gradient cannot be combined with feature_delta."
        )


def assert_feature_clusters_precompute_supported(
    pipeline_config: FeaturePipelineConfig,
    grad_spec: GradientRankingSpec | None,
    feature_clusters: FeatureClusterConsolidation | None,
) -> None:
    """Fail before output creation for undefined clustered selection semantics."""
    if feature_clusters is None:
        return
    if grad_spec is not None:
        raise FeatureClustersError(
            "feature_clusters cannot be combined with feature_selection.gradient in "
            "V1 because no reviewed cluster-level gradient aggregation contract exists"
        )
    selection = pipeline_config.selection_config
    if pipeline_config.delta_bundle is not None:
        if not isinstance(selection, SamplingBundle):
            raise FeatureClustersError(
                "feature_clusters with splice_counterfactual supports sampling "
                "selection only in V1; clustered threshold semantics are undefined"
            )
        return
    if isinstance(selection, SamplingBundle):
        return
    if isinstance(selection, ThresholdSelectionConfig):
        threshold_type = selection.selection.threshold_type
        if threshold_type in _PERCENTILE_ONLY_THRESHOLD_TYPES:
            return
        supported = sorted(item.value for item in _PERCENTILE_ONLY_THRESHOLD_TYPES)
        raise FeatureClustersError(
            f"feature_clusters cannot apply raw-magnitude threshold {threshold_type.value!r}; "
            f"ordinary clustered precompute supports only {supported}"
        )
    raise FeatureClustersError(
        f"feature_clusters does not support selection config {type(selection).__name__}"
    )


def _assert_feature_clusters_sel_manifest_current(
    output_dir: Path,
    feature_clusters: FeatureClusterConsolidation | None,
    *,
    selection_hash: str,
    packed_rows_hash: str,
    resume: bool,
) -> set[str]:
    """Validate artifact identity and return byte-certified reusable shard stems.

    ``resume=False`` still validates the selection, packed-row, mask, and
    cluster identities before touching output, but deliberately does not
    require the old shard bytes or their progress certificate to be healthy:
    every in-domain shard will be atomically recomputed. This makes the
    explicit full-recompute mode capable of repairing damaged outputs without
    weakening the default resume path's fail-loud integrity checks.
    """
    if not isinstance(resume, bool):
        raise TypeError(f"resume must be a bool, got {resume!r}")
    prior_manifest = read_manifest(output_dir)
    prior_shards = glob_overlay_shards(output_dir)
    if prior_manifest is None:
        if feature_clusters is not None and prior_shards:
            raise FeatureClustersError(
                f"clustered sel dir {output_dir} contains selection shards but no "
                "valid manifest; their cluster identity cannot be verified. Remove "
                "or relocate the directory and rerun precompute_selected_features"
            )
        return (
            {path.name.removesuffix(".safetensors.zst") for path in prior_shards}
            if feature_clusters is None and resume
            else set()
        )
    metadata = prior_manifest.get("metadata", {})
    assert_feature_clusters_selection_metadata(
        feature_clusters,
        metadata,
        surface=f"sel/ manifest {output_dir}",
        require_complete=False,
    )
    prior_selection_hash = prior_manifest.get("checksum")
    if feature_clusters is not None and prior_selection_hash != selection_hash:
        raise FeatureClustersError(
            f"clustered sel/ manifest {output_dir} records selection checksum "
            f"{prior_selection_hash!r}, but the current selection identity is "
            f"{selection_hash!r}. Remove or relocate the directory and rerun "
            "precompute_selected_features"
        )
    prior_packed = metadata.get("packed_rows") if isinstance(metadata, Mapping) else None
    prior_packed_hash = (
        prior_packed.get("packed_row_fingerprint") if isinstance(prior_packed, Mapping) else None
    )
    if feature_clusters is not None and prior_packed_hash != packed_rows_hash:
        raise FeatureClustersError(
            f"clustered sel/ manifest {output_dir} records packed-row fingerprint "
            f"{prior_packed_hash!r}, but current packed rows hash to "
            f"{packed_rows_hash!r}. Existing selection shards are stale; remove "
            "or relocate the directory and rerun precompute_selected_features"
        )
    if feature_clusters is None:
        return (
            {path.name.removesuffix(".safetensors.zst") for path in prior_shards}
            if resume
            else set()
        )
    if not resume:
        return set()
    cluster_metadata = metadata.get("feature_clusters")
    assert isinstance(cluster_metadata, Mapping)
    if cluster_metadata["status"] == "in_progress":
        # Schema-v1 interrupted manifests predate incremental certification.
        # Their plausible-looking shard bytes have no expected digest, so they
        # remain untrusted and are overwritten once. New producers checkpoint
        # exact compressed-byte identities after every atomic shard publish.
        if SELECTION_SHARD_INTEGRITY_KEY not in metadata:
            return set()
        certified_names = validated_clustered_selection_shard_names(
            output_dir,
            metadata,
            surface=f"in-progress sel/ manifest {output_dir}",
            allow_untracked=True,
        )
        return {name.removesuffix(".safetensors.zst") for name in certified_names}
    assert_clustered_selection_shard_integrity(
        output_dir,
        metadata,
        surface=f"sel/ manifest {output_dir}",
    )
    return {path.name.removesuffix(".safetensors.zst") for path in prior_shards}


def _pid_may_be_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _sweep_dead_atomic_selection_temps(
    output_dir: Path,
    expected_stems: set[str],
) -> tuple[str, ...]:
    """Remove only dead-writer atomic debris in the current shard domain.

    ``nanochat.atomic_io`` deliberately leaves ``<final>.tmp.<pid>.<uuid8>``
    after an uncatchable worker death. A live PID means another writer may
    still own the file and is a hard concurrency error. Malformed names,
    symlinks, directories, and out-of-domain stems are never guessed at.
    """
    if not output_dir.is_dir():
        return ()
    removed: list[str] = []
    for path in sorted(output_dir.iterdir()):
        if ".safetensors.zst.tmp." not in path.name:
            continue
        match = _ATOMIC_SELECTION_TEMP_RE.fullmatch(path.name)
        if match is None:
            raise RuntimeError(f"selection output contains malformed atomic temporary file: {path}")
        stem = match.group("stem")
        if stem not in expected_stems:
            raise RuntimeError(
                "selection output contains an atomic temporary file outside "
                f"the current packed-row domain: {path}"
            )
        result = path.lstat()
        if not stat.S_ISREG(result.st_mode):
            raise RuntimeError(
                f"selection atomic temporary path must be a real regular file: {path}"
            )
        pid = int(match.group("pid"))
        if _pid_may_be_alive(pid):
            raise RuntimeError(
                f"selection atomic temporary file {path} belongs to live PID {pid}; "
                "refusing concurrent output mutation"
            )
        path.unlink()
        removed.append(path.name)
    if removed:
        directory_fd = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        logger.info(f"Removed {len(removed)} dead-writer atomic selection temporary file(s)")
    return tuple(removed)


def _selection_shard_file_identity(
    path: Path,
    *,
    surface: str,
) -> tuple[int, int, int, int, int, int]:
    """Capture direct-file identity for a validation-to-hash race guard."""
    try:
        result = path.lstat()
    except OSError as exc:
        raise FeatureClustersError(
            f"{surface} cannot inspect serialized selection shard {path}"
        ) from exc
    if not stat.S_ISREG(result.st_mode):
        raise FeatureClustersError(
            f"{surface} serialized selection shard must be a direct regular file, got {path}"
        )
    return (
        int(result.st_mode),
        int(result.st_dev),
        int(result.st_ino),
        int(result.st_size),
        int(result.st_mtime_ns),
        int(result.st_ctime_ns),
    )


def _selection_max_features_per_segment(selection: SelectionShard) -> int:
    """Exact maximum positive-plus-negative CSR width after validation."""
    widths = (
        selection.sel_offsets[1:].to(t.int64)
        - selection.sel_offsets[:-1].to(t.int64)
        + selection.neg_offsets[1:].to(t.int64)
        - selection.neg_offsets[:-1].to(t.int64)
    )
    return int(widths.max().item()) if widths.numel() > 0 else 0


def _validate_and_fingerprint_clustered_selection_shard(
    *,
    prior_integrity: dict[str, Any] | None,
    selection_path: Path,
    expected_segments: int,
    feature_clusters: FeatureClusterConsolidation,
    effective_buckets: int,
    surface: str,
) -> tuple[dict[str, Any], str, int]:
    """Validate serialized tensors and bind that result to exact file bytes.

    The file identity must remain unchanged across decompression, semantic
    validation, and compressed-byte hashing. This closes the otherwise subtle
    gap where a path could be replaced after its tensors passed validation but
    before a digest was recorded in the progress manifest.
    """
    if type(expected_segments) is not int or expected_segments < 0:
        raise FeatureClustersError(
            f"{surface} expected_segments must be a non-negative strict int, "
            f"got {expected_segments!r}"
        )
    before = _selection_shard_file_identity(selection_path, surface=surface)
    selection = load_selection_shard(selection_path)
    after_load = _selection_shard_file_identity(selection_path, surface=surface)
    if after_load != before:
        raise FeatureClustersError(
            f"{surface} serialized selection shard changed while it was loaded: {selection_path}"
        )
    feature_clusters.assert_selection_shard(
        sel_offsets=selection.sel_offsets,
        sel_fids=selection.sel_fids,
        sel_buckets=selection.sel_buckets,
        sel_pcts=selection.sel_pcts,
        neg_offsets=selection.neg_offsets,
        neg_fids=selection.neg_fids,
        expected_segments=expected_segments,
        effective_buckets=effective_buckets,
        surface=f"serialized clustered selection shard {selection_path.name}",
    )
    after_validation = _selection_shard_file_identity(
        selection_path,
        surface=surface,
    )
    if after_validation != before:
        raise FeatureClustersError(
            f"{surface} serialized selection shard changed while it was validated: {selection_path}"
        )
    integrity = extend_clustered_selection_shard_integrity(
        prior_integrity,
        selection_path,
        surface=surface,
    )
    after_hash = _selection_shard_file_identity(selection_path, surface=surface)
    if after_hash != before:
        raise FeatureClustersError(
            f"{surface} serialized selection shard changed between validation "
            f"and fingerprinting: {selection_path}"
        )
    matching_entries = [
        entry for entry in integrity["shards"] if entry["name"] == selection_path.name
    ]
    if len(matching_entries) != 1:
        raise FeatureClustersError(
            f"{surface} fingerprint does not contain exactly one entry for {selection_path.name!r}"
        )
    content_sha256 = matching_entries[0]["content_sha256"]
    return (
        integrity,
        cast(str, content_sha256),
        _selection_max_features_per_segment(selection),
    )


def _validate_clustered_selection_outputs(
    *,
    output_dir: Path,
    base_shard_paths: list[Path],
    packed_resolution: PackedFingerprintResolution,
    expected_selection_integrity: Mapping[str, Any],
    expected_packed_rows_hash: str,
    feature_clusters: FeatureClusterConsolidation,
    effective_buckets: int,
    segment_counts_by_stem: dict[str, int] | None = None,
    max_features_by_stem: dict[str, int] | None = None,
) -> int:
    """Require exact complete representative-only shards before completion.

    Returns the exact global positive-plus-negative CSR width used by the
    remote loader as its static ``pad_F`` certificate.
    """
    selection_paths = glob_overlay_shards(output_dir)
    selection_by_stem = {
        path.name.removesuffix(_SELECTION_SHARD_SUFFIX): path for path in selection_paths
    }
    base_by_stem = {
        path.name.removesuffix(_SELECTION_SHARD_SUFFIX): path for path in base_shard_paths
    }
    missing = sorted(set(base_by_stem) - set(selection_by_stem))
    extra = sorted(set(selection_by_stem) - set(base_by_stem))
    if missing or extra:
        raise FeatureClustersError(
            f"clustered selection shards at {output_dir} must match packed shards "
            f"exactly; missing={missing[:5]}, extra={extra[:5]}"
        )
    if packed_resolution.aggregate_sha256 != expected_packed_rows_hash:
        raise FeatureClustersError(
            "clustered selection validation packed-row fingerprint disagrees "
            "with the selection manifest identity"
        )
    packed_file_by_name = {entry.name: entry for entry in packed_resolution.files}
    if set(packed_file_by_name) != {path.name for path in base_shard_paths}:
        raise FeatureClustersError(
            "clustered selection validation strict packed-row proof does not "
            "cover the exact packed shard domain"
        )
    rebuilt_integrity: dict[str, Any] | None = None
    max_features_per_segment = 0
    for stem in sorted(selection_by_stem):
        selection_path = selection_by_stem[stem]
        packed_path = base_by_stem[stem]
        packed_file = packed_file_by_name[packed_path.name]
        assert_packed_row_fingerprint_file_current(packed_file, packed_path)
        packed = load_packed_row_shard(packed_path)
        assert_packed_row_fingerprint_file_current(packed_file, packed_path)
        rebuilt_integrity, _digest, shard_max_features = (
            _validate_and_fingerprint_clustered_selection_shard(
                prior_integrity=rebuilt_integrity,
                selection_path=selection_path,
                expected_segments=packed.total_segments,
                feature_clusters=feature_clusters,
                effective_buckets=effective_buckets,
                surface=f"clustered selection finalization {output_dir}",
            )
        )
        assert_packed_row_fingerprint_file_current(packed_file, packed_path)
        if segment_counts_by_stem is not None:
            segment_counts_by_stem[stem] = packed.total_segments
        if max_features_by_stem is not None:
            max_features_by_stem[stem] = shard_max_features
        max_features_per_segment = max(
            max_features_per_segment,
            shard_max_features,
        )
    if rebuilt_integrity != expected_selection_integrity:
        raise FeatureClustersError(
            "clustered selection finalization compressed-byte inventory "
            "changed after in-progress evidence was recorded"
        )
    assert_packed_row_fingerprint_resolution_current(packed_resolution)
    return max_features_per_segment


@dataclass(frozen=True)
class _CompletedClusteredValidationEvidence:
    certificate: dict[str, Any]
    selection_signatures: tuple[tuple[Path, tuple[int, int, int, int, int, int]], ...]
    packed_resolution: PackedFingerprintResolution


def _assert_completed_clustered_validation_evidence_current(
    evidence: _CompletedClusteredValidationEvidence,
    *,
    output_dir: Path,
    packed_row_dir: Path,
) -> None:
    """Recheck the exact certified shard domains and every stat signature."""

    expected_selection_names = tuple(
        path.name for path, _signature in evidence.selection_signatures
    )
    actual_selection_names = tuple(sorted(path.name for path in glob_overlay_shards(output_dir)))
    if actual_selection_names != expected_selection_names:
        raise FeatureClustersError(
            "serialized selection shard inventory changed before completed manifest publication"
        )
    expected_packed_names = tuple(entry.name for entry in evidence.packed_resolution.files)
    actual_packed_names = tuple(sorted(path.name for path in glob_overlay_shards(packed_row_dir)))
    if actual_packed_names != expected_packed_names:
        raise FeatureClustersError(
            "packed-row shard inventory changed before completed selection manifest publication"
        )
    for selection_path, signature in evidence.selection_signatures:
        if (
            _selection_shard_file_identity(
                selection_path,
                surface=f"publishing clustered selection manifest {output_dir}",
            )
            != signature
        ):
            raise FeatureClustersError(
                "serialized selection shard changed immediately before "
                f"completed manifest publication: {selection_path}"
            )
    assert_packed_row_fingerprint_resolution_current(evidence.packed_resolution)


def _build_completed_clustered_selection_validation(
    *,
    output_dir: Path,
    packed_proof: PackedFingerprintResolution,
    selection_hash: str,
    selection_integrity: Mapping[str, Any],
    packed_rows_hash: str,
    feature_clusters: FeatureClusterConsolidation,
    feature_delta: Mapping[str, Any] | None,
    effective_buckets: int,
    expected_segments_by_stem: Mapping[str, int],
    validated_max_features_by_stem: Mapping[str, int],
    prior_metadata: Mapping[str, Any] | None,
) -> _CompletedClusteredValidationEvidence:
    """Revalidate compact outputs and construct one complete producer proof."""

    selection_paths = glob_overlay_shards(output_dir)
    selection_by_stem = {
        path.name.removesuffix(_SELECTION_SHARD_SUFFIX): path for path in selection_paths
    }
    if set(selection_by_stem) != set(expected_segments_by_stem):
        raise FeatureClustersError(
            "completed clustered validation requires exact selection/segment stem coverage"
        )
    selection_signatures = tuple(
        (
            selection_by_stem[stem],
            _selection_shard_file_identity(
                selection_by_stem[stem],
                surface=f"final clustered validation {output_dir}",
            ),
        )
        for stem in sorted(selection_by_stem)
    )
    prior_authority = None
    if prior_metadata is not None and CLUSTERED_SELECTION_VALIDATION_KEY in prior_metadata:
        prior_clusters = prior_metadata.get("feature_clusters")
        parser_kwargs = {
            "selection_checksum": selection_hash,
            "selection_integrity": selection_integrity,
            "packed_row_fingerprint": packed_rows_hash,
            "feature_clusters_descriptor": (feature_clusters.resolved.descriptor_dict()),
            "feature_delta": feature_delta,
            "effective_buckets": effective_buckets,
            "surface": f"prior clustered selection manifest {output_dir}",
        }
        if isinstance(prior_clusters, Mapping) and prior_clusters.get("status") == "complete":
            prior_authority = resolve_clustered_selection_validation(
                prior_metadata,
                **parser_kwargs,
            )
        else:
            # A retry first replaces its old manifest with certificate-free
            # in-progress metadata. If it started from a manually downgraded
            # formerly complete manifest, the independently signed schema-v1
            # block captured before that replacement may still avoid a tensor
            # replay after its exact identity and compressed inventory pass.
            prior_authority = parse_clustered_selection_validation(
                prior_metadata[CLUSTERED_SELECTION_VALIDATION_KEY],
                **parser_kwargs,
            )

    rebuilt_integrity: dict[str, Any] | None = None
    widths_by_stem: dict[str, int] = {}
    if prior_authority is not None and prior_authority.max_features_by_name is not None:
        prior_segments = {
            name.removesuffix(_SELECTION_SHARD_SUFFIX): count
            for name, count in prior_authority.segment_counts_by_name
        }
        widths_by_stem = {
            name.removesuffix(_SELECTION_SHARD_SUFFIX): width
            for name, width in prior_authority.max_features_by_name
        }
        if prior_segments != dict(expected_segments_by_stem) or set(widths_by_stem) != set(
            expected_segments_by_stem
        ):
            raise FeatureClustersError(
                "prior clustered validation certificate disagrees with the "
                "current per-shard segment-count domain"
            )
    elif set(validated_max_features_by_stem) == set(expected_segments_by_stem):
        widths_by_stem = dict(validated_max_features_by_stem)

    if widths_by_stem:
        # A complete schema-v1 certificate or this run's in-memory semantic
        # evidence already binds the canonical tensor result. Reissue/build
        # after hashing the exact compressed bytes under the same direct-file
        # stat fence, without reopening the selection tensors.
        for stem in sorted(selection_by_stem):
            selection_path = selection_by_stem[stem]
            before = _selection_shard_file_identity(
                selection_path,
                surface=f"final clustered validation {output_dir}",
            )
            rebuilt_integrity = extend_clustered_selection_shard_integrity(
                rebuilt_integrity,
                selection_path,
                surface=f"final clustered validation {output_dir}",
            )
            after = _selection_shard_file_identity(
                selection_path,
                surface=f"final clustered validation {output_dir}",
            )
            if after != before:
                raise FeatureClustersError(
                    "serialized selection shard changed while its completed "
                    f"certificate was reissued: {selection_path}"
                )
    else:
        for stem in sorted(selection_by_stem):
            rebuilt_integrity, _digest, width = _validate_and_fingerprint_clustered_selection_shard(
                prior_integrity=rebuilt_integrity,
                selection_path=selection_by_stem[stem],
                expected_segments=expected_segments_by_stem[stem],
                feature_clusters=feature_clusters,
                effective_buckets=effective_buckets,
                surface=f"final clustered validation {output_dir}",
            )
            widths_by_stem[stem] = width
    if rebuilt_integrity != selection_integrity:
        raise FeatureClustersError(
            "completed clustered validation rebuilt a selection inventory "
            "different from the final manifest inventory"
        )
    assert_packed_row_fingerprint_resolution_current(packed_proof)
    if packed_proof.aggregate_sha256 != packed_rows_hash:
        raise FeatureClustersError(
            "completed clustered validation packed cache disagrees with the "
            "selection manifest packed-row fingerprint"
        )
    if prior_authority is not None:
        assert_validation_packed_inventory(
            prior_authority,
            tuple((entry.name, entry.content_sha256) for entry in packed_proof.files),
            surface=f"final clustered validation {output_dir}",
        )
    packed_content_by_stem = {
        entry.name.removesuffix(_SELECTION_SHARD_SUFFIX): entry.content_sha256
        for entry in packed_proof.files
    }
    certificate = build_clustered_selection_validation(
        selection_checksum=selection_hash,
        selection_integrity=selection_integrity,
        packed_row_fingerprint=packed_rows_hash,
        packed_content_by_name=packed_content_by_stem,
        feature_clusters_descriptor=feature_clusters.resolved.descriptor_dict(),
        feature_delta=feature_delta,
        effective_buckets=effective_buckets,
        segment_counts_by_stem=expected_segments_by_stem,
        max_features_by_stem=widths_by_stem,
    )
    for selection_path, signature in selection_signatures:
        if (
            _selection_shard_file_identity(
                selection_path,
                surface=f"final clustered validation {output_dir}",
            )
            != signature
        ):
            raise FeatureClustersError(
                "serialized selection shard changed while the completed "
                f"certificate was assembled: {selection_path}"
            )
    assert_packed_row_fingerprint_resolution_current(packed_proof)
    return _CompletedClusteredValidationEvidence(
        certificate=certificate,
        selection_signatures=selection_signatures,
        packed_resolution=packed_proof,
    )


class GradScoreJoiner:
    """Joins base-aligned gradient overlay shards to packed segments via the
    packer-recorded provenance mapping.

    Each grad shard is self-contained: ``grad_scores`` (fp16, flat in base
    CSR order) plus ``offsets`` and ``doc_seg_offsets`` copies, so locating
    segment ``(doc_idx, seg_ordinal)`` never touches base shards. Access is
    near-SEQUENTIAL: a packed shard only interleaves docs from the packing
    buffer's window, so it references a handful of adjacent base shards —
    ``retain()`` loads exactly one packed shard's referenced set up front and
    drops the rest. No eviction policy needed (or pretended).
    """

    def __init__(self, grad_overlay_dir: Path, base_shard_stems: list[str]):
        self.grad_overlay_dir = Path(grad_overlay_dir)
        self.base_shard_stems = base_shard_stems
        self._loaded: dict[int, dict[str, t.Tensor]] = {}

    def _load(self, base_idx: int) -> dict[str, t.Tensor]:
        stem = self.base_shard_stems[base_idx]
        path = self.grad_overlay_dir / f"{stem}.safetensors.zst"
        if not path.is_file():
            raise GradientAttributionError(
                f"gradient overlay shard missing for base shard {stem!r} at "
                f"{path}; re-run scripts.preprocess.precompute_gradient_attribution "
                f"(resume=true fills gaps)."
            )
        return load_overlay_shard(path)

    def retain(self, base_indices: set[int]) -> None:
        """Hold exactly the given base shards' tensors (load missing, drop
        the rest) — called once per packed shard with its referenced set."""
        self._loaded = {
            b: self._loaded[b] if b in self._loaded else self._load(b) for b in base_indices
        }

    def scores(self, base_idx: int, doc_idx: int, seg_ordinal: int) -> t.Tensor:
        """fp32 score slice for one base segment."""
        data = self._loaded.get(base_idx)
        if data is None:
            data = self._load(base_idx)
            self._loaded[base_idx] = data
        dso = data[GRAD_DOC_SEG_OFFSETS_KEY]
        if not (0 <= doc_idx < len(dso) - 1):
            raise GradientAttributionError(
                f"provenance doc_idx {doc_idx} out of range for base shard "
                f"{self.base_shard_stems[base_idx]!r} ({len(dso) - 1} docs)"
            )
        if seg_ordinal < 0:
            # A negative ordinal would silently index the PREVIOUS doc's last
            # segment via the offset arithmetic below.
            raise GradientAttributionError(
                f"provenance seg_ordinal {seg_ordinal} is negative for doc "
                f"{doc_idx} of base shard {self.base_shard_stems[base_idx]!r} "
                f"— corrupt provenance."
            )
        gseg = int(dso[doc_idx]) + seg_ordinal
        if gseg >= int(dso[doc_idx + 1]):
            raise GradientAttributionError(
                f"provenance seg_ordinal {seg_ordinal} out of range for doc "
                f"{doc_idx} of base shard {self.base_shard_stems[base_idx]!r}"
            )
        offsets = data[GRAD_OFFSETS_KEY]
        lo, hi = int(offsets[gseg]), int(offsets[gseg + 1])
        return data[GRAD_SCORES_KEY][lo:hi].to(t.float32)


class ShardGradientJoin:
    """Per-packed-shard gradient lookup: this shard's provenance triplets +
    the shared joiner (pre-loaded with exactly this shard's referenced base
    shards)."""

    def __init__(self, joiner: GradScoreJoiner, prov: dict[str, t.Tensor]):
        self.joiner = joiner
        self.base_idx = prov[PROVENANCE_BASE_SHARD_IDX_KEY]
        self.doc_idx = prov[PROVENANCE_DOC_IDX_KEY]
        self.seg_ordinal = prov[PROVENANCE_SEG_ORDINAL_KEY]
        # Every packed segment maps to a real base segment — the packer raises
        # rather than emit a filler/sentinel triple, so any negative field is
        # corruption, not a legitimate "no base segment" marker.
        if bool(
            (self.base_idx < 0).any() or (self.doc_idx < 0).any() or (self.seg_ordinal < 0).any()
        ):
            raise GradientAttributionError(
                "corrupt provenance: negative (base_shard_idx, doc_idx, "
                "seg_ordinal) field(s) — the packer only writes non-negative "
                "triples (every packed segment has a real base segment)."
            )
        joiner.retain({int(b) for b in self.base_idx.unique()})

    @property
    def total_segments(self) -> int:
        return len(self.base_idx)

    def scores_for_segment(self, seg_idx: int, n_feat: int) -> t.Tensor:
        """fp32 scores aligned with the packed segment's fid list."""
        b = int(self.base_idx[seg_idx])
        g = self.joiner.scores(b, int(self.doc_idx[seg_idx]), int(self.seg_ordinal[seg_idx]))
        if g.numel() != n_feat:
            raise GradientAttributionError(
                f"gradient/provenance misalignment at packed segment {seg_idx}: "
                f"base segment carries {g.numel()} stored features but the "
                f"packed segment has {n_feat}. The packed dir, provenance, or "
                f"grad overlay are out of sync — re-run the producers."
            )
        return g


def _assert_sel_overlay_unchanged(output_dir: Path, grad_overlay_checksum: str | None) -> None:
    """Refuse unauthenticated or stale gradient-ranked selection shards.

    The selection hash is config-only — the resolved grad overlay checksum is
    NOT in it (the four ANTI-DESYNC sites compute it from config alone) — so a
    re-run after the grad overlay changed (e.g. a new percentile overlay => a
    new ``grad/<hash>/`` via the content-addressed descriptor) would otherwise
    SKIP the existing per-shard sel files (``out_path.exists()``) and silently
    serve selections made from the OLD overlay's scores. The sel manifest
    records the overlay it was built against; if it disagrees with the current
    one, the dir is stale — fail loud (delete it to rebuild). A fresh empty dir
    is a no-op; an interrupted dir with reusable shards must be authenticated.
    """
    if grad_overlay_checksum is None:
        return
    prior_overlay = (
        (read_manifest(output_dir) or {})
        .get("metadata", {})
        .get("gradient_attribution", {})
        .get("overlay_checksum")
    )
    if glob_overlay_shards(output_dir) and (
        not isinstance(prior_overlay, str) or not prior_overlay
    ):
        raise GradientAttributionError(
            f"sel dir {output_dir} already contains selection shards but its "
            "gradient overlay identity cannot be authenticated from manifest.json. "
            f"Delete {output_dir} and re-run precompute_selected_features."
        )
    if prior_overlay is not None and prior_overlay != grad_overlay_checksum:
        prior_preview = (
            prior_overlay[:16] if isinstance(prior_overlay, str) else repr(prior_overlay)
        )
        raise GradientAttributionError(
            f"sel dir {output_dir} was built against gradient overlay "
            f"{prior_preview}..., but the current overlay is "
            f"{grad_overlay_checksum[:16]}... (the percentile overlay or the "
            f"gradient artifacts changed). Its selections are stale — delete "
            f"{output_dir} and re-run precompute_selected_features."
        )


def _assert_splice_counterfactual_sel_manifest_current(
    output_dir: Path,
    pipeline_config: FeaturePipelineConfig,
    packed_info: PackedProvenanceInfo,
    splice_sidecar_hash: str,
    derived_feature_hash: str,
    boundary_hash: str,
    feature_clusters: FeatureClusterConsolidation | None = None,
    splice_batch_size: int = 1,
) -> None:
    """Refuse to reuse splice-counterfactual selections from old artifacts.

    The splice selection hash is config-only. Packed provenance and splice
    sidecars live outside that hash, so a rerun after either artifact changes
    must not skip old selection shards and then stamp them with a fresh
    manifest.
    """
    prior_manifest = read_manifest(output_dir)
    prior_shards = glob_overlay_shards(output_dir)
    if prior_manifest is None:
        if prior_shards:
            raise SpliceProvenanceError(
                f"splice_counterfactual sel dir {output_dir} already contains "
                "selection shards but has no valid manifest.json. Its "
                "sidecar/provenance identity cannot be verified; delete the "
                "dir and re-run precompute_selected_features."
            )
        return

    metadata = prior_manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise SpliceProvenanceError(
            f"splice_counterfactual sel dir {output_dir} has no manifest "
            "metadata. Delete the dir and re-run precompute_selected_features."
        )
    if pipeline_config.delta_bundle is None:
        raise SpliceProvenanceError(
            "internal error: splice_counterfactual manifest check called for a "
            "pipeline without feature_delta."
        )

    expected_delta = canonical_feature_delta_dict(pipeline_config.delta_bundle.config)
    if metadata.get("feature_delta") != expected_delta:
        raise SpliceProvenanceError(
            f"splice_counterfactual sel dir {output_dir} records feature_delta "
            f"{metadata.get('feature_delta')!r}, but the current config is "
            f"{expected_delta!r}. Delete the dir and re-run "
            "precompute_selected_features."
        )

    splice_meta = metadata.get("splice_counterfactual")
    if not isinstance(splice_meta, dict):
        raise SpliceProvenanceError(
            f"splice_counterfactual sel dir {output_dir} is missing "
            "splice_counterfactual manifest metadata. Delete the dir and "
            "re-run precompute_selected_features."
        )
    expected_splice = {
        **splice_counterfactual_manifest_identity(clustered=feature_clusters is not None),
        "derived_shard_stems": packed_info.base_shard_stems,
        "derived_feature_fingerprint": derived_feature_hash,
        "boundary_fingerprint": boundary_hash,
        "sidecar_fingerprint": splice_sidecar_hash,
    }
    for key, expected in expected_splice.items():
        if splice_meta.get(key) != expected:
            raise SpliceProvenanceError(
                f"splice_counterfactual sel dir {output_dir} is stale: "
                f"manifest splice_counterfactual.{key}={splice_meta.get(key)!r}, "
                f"current value is {expected!r}. Delete the dir and re-run "
                "precompute_selected_features."
            )
    expected_batching = canonical_splice_selection_batching(
        expected_delta,
        splice_batch_size,
    )
    recorded_batching_present = "selection_batching" in splice_meta
    recorded_batching = splice_meta.get("selection_batching")
    if (expected_batching is None and recorded_batching_present) or (
        expected_batching is not None and recorded_batching != expected_batching
    ):
        raise SpliceProvenanceError(
            f"splice_counterfactual sel dir {output_dir} records selection "
            f"batching {recorded_batching!r}, but the current config resolves "
            f"to {expected_batching!r}. Delete the dir and re-run "
            "precompute_selected_features."
        )

    packed_meta = metadata.get("packed_provenance")
    if not isinstance(packed_meta, dict):
        raise SpliceProvenanceError(
            f"splice_counterfactual sel dir {output_dir} is missing "
            "packed_provenance manifest metadata. Delete the dir and re-run "
            "precompute_selected_features."
        )
    expected_packed = {
        "packed_stem_fingerprint": packed_info.packed_stem_fingerprint,
        "packed_row_fingerprint": packed_info.packed_row_fingerprint,
        "provenance_shard_count": len(packed_info.provenance_shard_paths),
        "provenance_fingerprint": packed_info.provenance_fingerprint,
    }
    for key, expected in expected_packed.items():
        if packed_meta.get(key) != expected:
            raise SpliceProvenanceError(
                f"splice_counterfactual sel dir {output_dir} is stale: "
                f"manifest packed_provenance.{key}={packed_meta.get(key)!r}, "
                f"current value is {expected!r}. Delete the dir and re-run "
                "precompute_selected_features."
            )


def _base_selection_metadata(
    *,
    feature_mask,
    feature_clusters: FeatureClusterConsolidation | None,
    grad_spec: GradientRankingSpec | None,
    grad_overlay_checksum: str | None,
    packed_rows_hash: str,
    status: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "packed_rows": {
            "packed_row_fingerprint": packed_rows_hash,
        }
    }
    if feature_mask is not None:
        metadata["feature_mask"] = canonical_dict(feature_mask)
    if feature_clusters is not None:
        cluster_metadata = feature_clusters.metadata_dict()
        if status is not None:
            cluster_metadata["status"] = status
        metadata["feature_clusters"] = cluster_metadata
    if grad_spec is not None:
        metadata["gradient_attribution"] = {
            **canonical_gradient_selection_dict(grad_spec),
            "overlay_checksum": grad_overlay_checksum,
        }
    return metadata


def _splice_selection_metadata(
    *,
    pipeline: FeaturePipelineConfig,
    packed_info: PackedProvenanceInfo,
    splice_stats: SpliceSelectionStats,
    splice_sidecar_hash: str,
    splice_derived_feature_hash: str,
    splice_boundary_hash: str,
    feature_mask,
    feature_clusters: FeatureClusterConsolidation | None,
    grad_spec: GradientRankingSpec | None,
    grad_overlay_checksum: str | None,
    status: str,
    splice_batch_size: int = 1,
) -> dict[str, Any]:
    assert pipeline.delta_bundle is not None
    metadata = _base_selection_metadata(
        feature_mask=feature_mask,
        feature_clusters=feature_clusters,
        grad_spec=grad_spec,
        grad_overlay_checksum=grad_overlay_checksum,
        packed_rows_hash=packed_info.packed_row_fingerprint,
        status=status,
    )
    metadata["feature_delta"] = canonical_feature_delta_dict(pipeline.delta_bundle.config)
    splice_metadata: dict[str, Any] = {
        **splice_counterfactual_manifest_identity(clustered=feature_clusters is not None),
        "status": status,
        "derived_shard_stems": packed_info.base_shard_stems,
        "total_packed_segments_selected": splice_stats.total_packed_segments,
        "total_sidecar_splice_positions_referenced": (
            splice_stats.total_sidecar_splice_positions_referenced
        ),
        "total_delta_positions_selected": (splice_stats.total_delta_positions_selected),
        "derived_feature_fingerprint": splice_derived_feature_hash,
        "boundary_fingerprint": splice_boundary_hash,
        "sidecar_fingerprint": splice_sidecar_hash,
    }
    splice_batching = canonical_splice_selection_batching(
        canonical_feature_delta_dict(pipeline.delta_bundle.config),
        splice_batch_size,
    )
    if splice_batching is not None:
        splice_metadata["selection_batching"] = splice_batching
    metadata["splice_counterfactual"] = splice_metadata
    metadata["packed_provenance"] = {
        "packed_stem_fingerprint": packed_info.packed_stem_fingerprint,
        "packed_row_fingerprint": packed_info.packed_row_fingerprint,
        "provenance_shard_count": len(packed_info.provenance_shard_paths),
        "provenance_fingerprint": packed_info.provenance_fingerprint,
    }
    return metadata


def resolve_gradient_artifacts(
    cfg: DictConfig,
    feature_dir: Path,
    packed_row_dir: Path,
    grad_spec: GradientRankingSpec,
) -> tuple[GradScoreJoiner, Path, str]:
    """Locate + verify the grad overlay and provenance for this run.

    Returns ``(joiner, provenance_dir_path, overlay_checksum)``. Every
    failure is a typed error naming the producer command to run.
    """
    base_paths = glob_features(feature_dir)
    if not base_paths:
        raise GradientAttributionError(f"no base shards in {feature_dir}")
    base_meta = read_shard_metadata(base_paths[0])
    ignore_bos = base_meta.get("ignore_bos_features", "false") == "true"

    # Resolve the SAME single percentile overlay the producer used: the stored
    # score is pct * sensitivity, so the pct identity is part of the grad
    # overlay's content hash. A changed pct overlay re-addresses the grad dir,
    # so a stale overlay surfaces here as "gradient overlay not found".
    try:
        _, pct_checksum = resolve_single_percentile_overlay(feature_dir)
    except OverlayStaleError as exc:
        raise GradientAttributionError(f"{exc} The gradient score is pct * sensitivity.") from exc

    descriptor = canonical_gradient_descriptor(
        attribution_mode=grad_spec.mode.value,
        model=cfg.sae.model,
        model_dtype=grad_spec.model_dtype,
        layer=int(cfg.sae.layer),
        sae_release=str(cfg.sae.release),
        sae_width=str(cfg.sae.width),
        sae_l0=cfg.sae.get("l0"),
        activation_type=str(cfg.sae.activation_type),
        ignore_bos_features=ignore_bos,
        min_final_segment_tokens=grad_spec.min_final_segment_tokens,
        percentile_checksum=pct_checksum,
    )
    checksum = compute_gradient_overlay_checksum(descriptor)
    overlay = discover_overlay(
        feature_dir, OverlayKind.GRADIENT, checksum, base_shard_paths=base_paths
    )
    if overlay is None:
        raise GradientAttributionError(
            f"gradient overlay not found for descriptor checksum "
            f"{checksum[:16]}... under {feature_dir / 'grad'}. Run:\n"
            f"  uv run python -m scripts.preprocess.precompute_gradient_attribution \\\n"
            f"      attribution_mode={grad_spec.mode.value} "
            f"min_final_segment_tokens={grad_spec.min_final_segment_tokens} "
            f"model_dtype={grad_spec.model_dtype}"
        )
    overlay_manifest = read_manifest(overlay)
    assert overlay_manifest is not None  # discover_overlay verified it
    manifest_descriptor = {
        k: v for k, v in overlay_manifest.get("metadata", {}).items() if k in descriptor
    }
    assert_descriptor_matches(
        descriptor, manifest_descriptor, artifact=f"gradient overlay {overlay}"
    )

    # Provenance is written by the packer in the same pass as the packed
    # shards (scripts.preprocess.precompute_packed_rows), so the two can only
    # drift via manual edits or a dir packed before provenance integration —
    # both detected here via the manifest marker + shard-count equality.
    prov_dir = provenance_dir(packed_row_dir)
    packed_manifest = read_manifest(packed_row_dir)
    packed_meta = (packed_manifest or {}).get("metadata", {})
    stems = packed_meta.get("base_shard_stems")
    regen_hint = (
        "re-run scripts.preprocess.precompute_packed_rows with resume=true — "
        "existing packed shards are kept byte-identical and the missing "
        "provenance is regenerated alongside them"
    )
    if packed_manifest is None or stems is None or not prov_dir.is_dir():
        raise GradientAttributionError(
            f"packed-segment provenance not available for {packed_row_dir} "
            f"(missing {'manifest' if packed_manifest is None else 'provenance record'}) "
            f"— the dir predates packer-integrated provenance; {regen_hint}."
        )
    n_packed = len(glob_overlay_shards(packed_row_dir))
    n_prov = len(glob_overlay_shards(prov_dir))
    if n_prov != n_packed or n_prov != packed_meta.get("provenance_shards"):
        raise GradientAttributionError(
            f"provenance at {prov_dir} covers {n_prov} shards but "
            f"{packed_row_dir} has {n_packed} packed shards (manifest records "
            f"{packed_meta.get('provenance_shards')}) — incomplete or stale "
            f"provenance; {regen_hint}."
        )
    joiner = GradScoreJoiner(overlay, stems)
    return joiner, prov_dir, checksum


def select_features_for_shard(
    base_shard_path: Path,
    pipeline_config: FeaturePipelineConfig,
    output_dir: Path,
    shard_name: str,
    gradient_join: ShardGradientJoin | None = None,
    feature_clusters: FeatureClusterConsolidation | None = None,
) -> bool:
    """Apply feature selection to a base packed row shard.

    Loads the base shard, applies selection + discretization + negative sampling
    per segment, and writes a SelectionShard aligned 1:1 with the base.

    When ``gradient_join`` is provided (gradient-attribution ranking), each
    segment's ranking key is its grad-overlay score row instead of the
    activation percentile; only features with score > 0 are ELIGIBLE — if a
    segment has fewer eligible features than the drawn k, the selection is
    capped at what's available (aggregate-logged), and zero eligible features
    yield an empty selection. ``sel_pcts``/buckets stay percentile-based.

    Returns True if the shard was written.
    """
    assert_precompute_supports_pipeline(pipeline_config)

    t0 = time.monotonic()
    base = load_packed_row_shard(base_shard_path)
    t_load = time.monotonic() - t0

    total_segs = base.total_segments
    feat_offsets = base.seg_feat_offsets.tolist()

    if gradient_join is not None and gradient_join.total_segments != total_segs:
        raise GradientAttributionError(
            f"{shard_name}: provenance has {gradient_join.total_segments} "
            f"segments but the packed shard has {total_segs}"
        )
    n_zero_eligible = 0
    n_capped = 0

    # Accumulators for selection shard
    sel_fids_flat: list[int] = []
    sel_buckets_flat: list[int] = []
    sel_pcts_flat: list[int] = []
    sel_counts: list[int] = []  # per-segment count

    neg_fids_flat: list[int] = []
    neg_counts: list[int] = []  # per-segment count

    use_sampling = isinstance(pipeline_config.selection_config, SamplingBundle)

    t1 = time.monotonic()

    for seg_idx in range(total_segs):
        f_start = feat_offsets[seg_idx]
        f_end = feat_offsets[seg_idx + 1]
        n_feat = f_end - f_start

        if n_feat == 0:
            sel_counts.append(0)
            neg_counts.append(0)
            continue

        fids = base.feature_ids[f_start:f_end].to(t.int64)

        # Clustered ordinary preselection is shared with BPB-Adv context
        # conversion.  The no-cluster path below remains the legacy body so a
        # null feature_clusters config cannot perturb its order, RNG, or bytes.
        if isinstance(feature_clusters, FeatureClusterConsolidation):
            if gradient_join is not None:
                raise FeatureClustersError(
                    "internal error: clustered ordinary selection reached the "
                    "gradient-attribution path after the startup support gate"
                )
            # Do not bypass consolidation when the descriptor becomes an
            # identity after masking. Supported packed rows are ascending, and
            # mask filtering preserves that order, so unconditional identity
            # consolidation is byte-identical. An inert-cluster bypass would
            # instead change currently accepted unsorted artifacts in-place;
            # reject that corrupt input before selection or output mutation.
            if feature_clusters.is_identity_after_mask and (
                fids.numel() > 1 and bool((fids[1:] < fids[:-1]).any())
            ):
                raise FeatureClustersError(
                    f"{shard_name} segment {seg_idx} has descending packed feature "
                    "IDs; identity-after-mask cluster consolidation requires "
                    "ascending packed feature IDs"
                )
            clustered = select_clustered_segment(
                fids,
                base.feature_pcts[f_start:f_end],
                clusters=feature_clusters,
                pipeline=pipeline_config,
            )
            sel_fids_flat.extend(clustered.positive_ids.tolist())
            sel_buckets_flat.extend(clustered.positive_buckets.tolist())
            sel_pcts_flat.extend(clustered.positive_percentiles_uint8.tolist())
            sel_counts.append(int(clustered.positive_ids.numel()))
            neg_fids_flat.extend(clustered.negative_ids.tolist())
            neg_counts.append(int(clustered.negative_ids.numel()))
            continue

        pcts = base.feature_pcts[f_start:f_end].float() / 255.0
        grad_scores: t.Tensor | None = None
        if gradient_join is not None:
            grad_scores = gradient_join.scores_for_segment(seg_idx, n_feat)

        if pipeline_config.feature_mask is not None:
            if grad_scores is not None:
                fids, pcts, grad_scores = pipeline_config.feature_mask.apply_to_positive(
                    fids, pcts, grad_scores
                )
            else:
                fids, pcts = pipeline_config.feature_mask.apply_to_positive(fids, pcts)
            n_feat = int(fids.numel())
            if n_feat == 0:
                sel_counts.append(0)
                neg_counts.append(0)
                continue

        if feature_clusters is not None:
            if grad_scores is not None:
                fids, pcts, grad_scores = feature_clusters.consolidate_segment(
                    fids,
                    pcts,
                    grad_scores,
                )
            else:
                fids, pcts = feature_clusters.consolidate_segment(fids, pcts)
            n_feat = int(fids.numel())
            if n_feat == 0:
                sel_counts.append(0)
                neg_counts.append(0)
                continue

        # --- Feature selection ---
        if use_sampling:
            bundle = pipeline_config.selection_config
            if grad_scores is not None:
                # Gradient-attribution ranking: STRICT eligibility — only
                # features whose attribution score is > 0 may be selected (no
                # percentile fallback, no random fill). The k drawn by the
                # sampling machinery is implicitly capped at the eligible
                # count by its existing clamp.
                eligible = grad_scores > 0
                elig_fids = fids[eligible]
                elig_pcts = pcts[eligible]
                elig_scores = grad_scores[eligible]
                n_eligible = int(elig_fids.numel())
                if n_eligible == 0:
                    n_zero_eligible += 1
                    sampled = []
                else:
                    sampled = sample_features_batched(
                        all_feature_ids=elig_fids,
                        all_feature_mags=t.zeros_like(elig_fids, dtype=t.float32),
                        segment_sizes=[n_eligible],
                        percentile_table=bundle.percentile_table,
                        config=bundle.selection,
                        rng=bundle.rng,
                        override_weights_list=[elig_scores],
                        precomputed_percentiles=elig_pcts,
                        gradient_rank_scores_supplied=True,
                    )
            else:
                # Sample k features
                sampled = sample_features_batched(
                    all_feature_ids=fids,
                    all_feature_mags=t.zeros_like(
                        fids, dtype=t.float32
                    ),  # mags not used when pcts provided
                    segment_sizes=[n_feat],
                    percentile_table=bundle.percentile_table,
                    config=bundle.selection,
                    rng=bundle.rng,
                    precomputed_percentiles=pcts,
                )
            if sampled:
                selected_fids, _selected_mags = sampled[0]
            else:
                selected_fids = t.tensor([], dtype=t.int64)
            if grad_scores is not None and len(selected_fids) > 0:
                # k >= available: the drawn k was capped by the eligible pool.
                if int(selected_fids.numel()) == int((grad_scores > 0).sum()):
                    n_capped += 1
        else:
            selected_fids = _select_from_packed_percentiles(
                fids=fids,
                pcts=pcts,
                selection=pipeline_config.selection_config,
            )

        # Defense-in-depth (H-assert): assert finalized positive ids are clean.
        # Cheap: boolean membership over selected_fids (tens–hundreds of ids).
        # Strictly mask-gated (I5: no-op when feature_mask is None).
        #
        # This MUST precede _lookup_selected_percentiles: a masked id can only reach
        # here if a downstream bug re-introduced it after apply_to_positive removed
        # it from ``fids`` (issue #663's guard). Such an id is therefore absent from
        # ``fids``, so the percentile lookup would otherwise raise a generic
        # "out of sync" error and shadow this mask-specific guard, misdiagnosing a
        # mask leak as a data-sync fault.
        if pipeline_config.feature_mask is not None:
            pipeline_config.feature_mask.assert_no_masked(selected_fids)
        if feature_clusters is not None:
            feature_clusters.assert_representatives_only(
                selected_fids,
                surface=f"{shard_name} positive selection",
            )

        # --- Discretization ---
        if len(selected_fids) > 0:
            selected_pcts = _lookup_selected_percentiles(
                selected_fids=selected_fids.to(t.int64),
                candidate_fids=fids.to(t.int64),
                candidate_pcts=pcts,
            )
            sel_pcts_vals = cast(
                list[int],
                normalized_percentile_to_uint8(selected_pcts).tolist(),
            )
            buckets = percentile_to_bucket(
                selected_pcts,
                pipeline_config.effective_buckets,
            )
        else:
            buckets = t.tensor([], dtype=t.int64)
            sel_pcts_vals = []

        sel_fids_flat.extend(selected_fids.tolist())
        sel_buckets_flat.extend(buckets.tolist())
        sel_pcts_flat.extend(sel_pcts_vals)
        sel_counts.append(len(selected_fids))

        # --- Negative sampling ---
        if pipeline_config.negative_bundle is not None:
            neg_bundle = pipeline_config.negative_bundle
            neg_sampled = sample_inactive_features_batched(
                [fids],
                neg_bundle.n_sae_features,
                neg_bundle.selection,
                neg_bundle.rng,
                feature_mask=(
                    feature_clusters.effective_negative_mask()
                    if feature_clusters is not None
                    else negative_sampling_mask(pipeline_config)
                ),
            )
            if neg_sampled:
                # Defense-in-depth (H-assert): assert finalized negative ids are clean.
                if pipeline_config.feature_mask is not None:
                    pipeline_config.feature_mask.assert_no_masked(neg_sampled[0])
                if feature_clusters is not None:
                    feature_clusters.assert_representatives_only(
                        neg_sampled[0],
                        surface=f"{shard_name} negative selection",
                    )
                neg_fids_flat.extend(neg_sampled[0].tolist())
                neg_counts.append(len(neg_sampled[0]))
            else:
                neg_counts.append(0)
        else:
            neg_counts.append(0)

    t_compute = time.monotonic() - t1

    # Build CSR offsets
    sel_offsets = [0]
    for c in sel_counts:
        sel_offsets.append(sel_offsets[-1] + c)
    neg_offsets = [0]
    for c in neg_counts:
        neg_offsets.append(neg_offsets[-1] + c)

    sel_shard = SelectionShard(
        sel_offsets=t.tensor(sel_offsets, dtype=t.int32),
        sel_fids=t.tensor(sel_fids_flat, dtype=t.uint16)
        if sel_fids_flat
        else t.tensor([], dtype=t.uint16),
        sel_buckets=t.tensor(sel_buckets_flat, dtype=t.uint8)
        if sel_buckets_flat
        else t.tensor([], dtype=t.uint8),
        sel_pcts=t.tensor(sel_pcts_flat, dtype=t.uint8)
        if sel_pcts_flat
        else t.tensor([], dtype=t.uint8),
        neg_offsets=t.tensor(neg_offsets, dtype=t.int32),
        neg_fids=t.tensor(neg_fids_flat, dtype=t.uint16)
        if neg_fids_flat
        else t.tensor([], dtype=t.uint16),
    )

    t2 = time.monotonic()
    out_path = overlay_shard_path(output_dir, shard_name)
    save_selection_shard(sel_shard, out_path)
    t_save = time.monotonic() - t2

    if gradient_join is not None and (n_zero_eligible or n_capped):
        logger.warning(
            f"  {shard_name}: gradient eligibility — {n_zero_eligible} segments "
            f"with ZERO positive-score features (empty selection), {n_capped} "
            f"segments capped at the eligible pool (drawn k >= available)."
        )
    logger.info(
        f"  {shard_name}: {total_segs} segs, {sel_shard.total_selected} sel, "
        f"{sel_shard.total_negative} neg ({time.monotonic() - t0:.1f}s: "
        f"load={t_load:.1f}s, compute={t_compute:.1f}s, save={t_save:.1f}s)"
    )
    return True


@dataclass(frozen=True)
class _SpliceResumeState:
    stats_by_stem: dict[str, SpliceSelectionStats]
    outputs_validated: bool
    selection_integrity: dict[str, Any] | None
    storage_validated_by_stem: dict[str, str]
    validated_max_features_per_segment: int | None


def _load_splice_resume_state(
    output_dir: Path,
    *,
    certified_existing_stems: set[str],
    packed_rows_hash: str,
    effective_buckets: int,
) -> _SpliceResumeState:
    """Load optional splice resume metadata after shard bytes were certified."""
    manifest = read_manifest(output_dir)
    if manifest is None:
        if certified_existing_stems:
            raise FeatureClustersError(
                f"sel/ dir {output_dir} has byte-certified reusable shards but "
                "no readable manifest during splice resume"
            )
        return _SpliceResumeState({}, False, None, {}, None)
    metadata = manifest.get("metadata", {}) if manifest is not None else {}
    if not isinstance(metadata, Mapping):
        raise FeatureClustersError(f"sel/ manifest {output_dir} has invalid metadata during resume")
    raw_resume = metadata.get(SPLICE_RESUME_METADATA_KEY)
    if not certified_existing_stems:
        if raw_resume is not None:
            raise FeatureClustersError(
                f"sel/ manifest {output_dir} advertises "
                f"{SPLICE_RESUME_METADATA_KEY} without any byte-certified "
                "reusable selection shards"
            )
        return _SpliceResumeState({}, False, None, {}, None)
    raw_integrity = metadata.get(SELECTION_SHARD_INTEGRITY_KEY)
    if not isinstance(raw_integrity, Mapping):
        raise FeatureClustersError(
            f"sel/ manifest {output_dir} lost its validated selection integrity metadata"
        )
    selection_integrity = parse_clustered_selection_shard_inventory(
        raw_integrity,
        surface=f"sel/ manifest {output_dir} during splice resume",
    )
    if raw_resume is None:
        # A current-semantic in-progress manifest may predate its first resume
        # checkpoint. Replay existing shard stats and validate the compact
        # selection artifacts before publishing a certificate.
        return _SpliceResumeState({}, False, selection_integrity, {}, None)
    cluster_metadata = metadata.get("feature_clusters")
    splice_metadata = metadata.get("splice_counterfactual")
    cluster_status = (
        cluster_metadata.get("status") if isinstance(cluster_metadata, Mapping) else None
    )
    splice_status = splice_metadata.get("status") if isinstance(splice_metadata, Mapping) else None
    if cluster_status != splice_status or cluster_status not in {
        "in_progress",
        "complete",
    }:
        raise FeatureClustersError(
            f"sel/ manifest {output_dir} has inconsistent clustered/splice status "
            f"({cluster_status!r} vs {splice_status!r})"
        )
    (
        stats_by_stem,
        outputs_validated,
        storage_validated_by_stem,
        validated_max_features_per_segment,
    ) = _parse_splice_resume_metadata(
        raw_resume,
        selection_integrity=selection_integrity,
        packed_rows_hash=packed_rows_hash,
        effective_buckets=effective_buckets,
        manifest_status=cluster_status,
        surface=f"sel/ manifest {output_dir}",
    )
    if set(stats_by_stem) != certified_existing_stems:
        raise FeatureClustersError(
            f"sel/ manifest {output_dir} splice resume stats do not exactly cover "
            "the byte-certified reusable shards"
        )
    return _SpliceResumeState(
        stats_by_stem=stats_by_stem,
        outputs_validated=outputs_validated,
        selection_integrity=selection_integrity,
        storage_validated_by_stem=storage_validated_by_stem,
        validated_max_features_per_segment=validated_max_features_per_segment,
    )


def _seed_from_parts(global_seed: int, shard_name: str, stream: str) -> int:
    payload = f"{global_seed}:{shard_name}:{stream}".encode("utf-8")  # noqa: UP012
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def _ordered_source_stems(
    ordered_group_keys: list[tuple[int, int]],
    packed_info: PackedProvenanceInfo,
    shard_name: str,
) -> list[str]:
    """Source-shard stems in reference first-need order (dedup-preserving).

    Bounds are already guaranteed by startup validate_packed_provenance; the
    re-check keeps error identity if that pass is ever bypassed.
    """
    stems: list[str] = []
    seen: set[str] = set()
    for base_shard_idx, _doc in ordered_group_keys:
        if not (0 <= base_shard_idx < len(packed_info.base_shard_stems)):
            raise SpliceProvenanceError(
                f"{shard_name}: base_shard_idx {base_shard_idx} out of range "
                f"for {len(packed_info.base_shard_stems)} manifest stems."
            )
        stem = packed_info.base_shard_stems[base_shard_idx]
        if stem not in seen:
            seen.add(stem)
            stems.append(stem)
    return stems


def _splice_shard_pipeline(
    pipeline_config: FeaturePipelineConfig,
    *,
    global_seed: int,
    shard_name: str,
) -> FeaturePipelineConfig:
    """Clone the pipeline with per-shard RNGs for deterministic resume."""
    selection_config = pipeline_config.selection_config
    if isinstance(selection_config, SamplingBundle):
        selection_config = SamplingBundle(
            selection=selection_config.selection,
            percentile_table=selection_config.percentile_table,
            rng=t.Generator().manual_seed(_seed_from_parts(global_seed, shard_name, "positive")),
        )
    negative_bundle = pipeline_config.negative_bundle
    if negative_bundle is not None:
        negative_bundle = NegativeBundle(
            selection=negative_bundle.selection,
            n_sae_features=negative_bundle.n_sae_features,
            rng=t.Generator().manual_seed(_seed_from_parts(global_seed, shard_name, "negative")),
            feature_mask=negative_bundle.feature_mask,
        )
    return FeaturePipelineConfig(
        selection_config=selection_config,
        boundaries=pipeline_config.boundaries,
        negative_bundle=negative_bundle,
        effective_buckets=pipeline_config.effective_buckets,
        delta_bundle=pipeline_config.delta_bundle,
        feature_mask=pipeline_config.feature_mask,
    )


def _selected_pcts_uint8(
    *,
    selected_fids: t.Tensor,
    selected_mags: t.Tensor,
    boundaries: t.Tensor,
) -> t.Tensor:
    if selected_fids.numel() == 0:
        return t.tensor([], dtype=t.uint8)
    return normalized_percentile_to_uint8(
        compute_percentiles(selected_fids, selected_mags, boundaries)
    )


def _require_clustered_delta_integer_cpu_vector(
    name: str,
    value: Any,
) -> t.Tensor:
    if not isinstance(value, t.Tensor):
        raise FeatureClustersError(f"{name} must be a tensor, got {type(value).__name__}")
    if value.device.type != "cpu" or value.ndim != 1:
        raise FeatureClustersError(
            f"{name} must be a one-dimensional CPU tensor, got "
            f"shape={tuple(value.shape)}, device={value.device}"
        )
    if value.dtype is t.bool or value.is_floating_point() or value.is_complex():
        raise FeatureClustersError(f"{name} must use an integer dtype, got {value.dtype}")
    return value.to(t.int64)


def _validate_clustered_delta_negative_contract(
    *,
    raw_fids: t.Tensor,
    positive_candidate_fids: t.Tensor,
    negative_candidate_fids: t.Tensor,
    negative_candidate_weights: t.Tensor,
    ref_mask: t.Tensor,
    feature_clusters: FeatureClusterConsolidation,
    n_sae_features: int,
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """Validate the representative-space contract used by the delta fast path.

    ``compute_doc_clustered_counterfactual_deltas`` has already masked and
    consolidated both sides. Its positive and signed-negative candidates are
    disjoint mask-allowed representatives, while ``ref_mask`` is their complete
    current/reference representative-space presence union. Therefore:

    * filtering the signed-negative candidates again is an identity;
    * projecting ``ref_mask`` again is an identity;
    * projecting raw-current IDs only adds bits already present in ``ref_mask``;
    * positive candidates cannot overlap the signed-negative pool.

    Validate every premise before the selection RNGs advance. A malformed
    internal result fails loudly instead of being silently repaired by the
    redundant transformations this guard replaces.
    """
    if type(n_sae_features) is not int or n_sae_features <= 0:
        raise FeatureClustersError(
            "clustered delta negative SAE width must be a positive strict int, "
            f"got {n_sae_features!r}"
        )
    if n_sae_features != feature_clusters.n_sae_features:
        raise FeatureClustersError(
            "clustered delta negative SAE width does not match the cluster "
            f"descriptor: negative={n_sae_features}, "
            f"clusters={feature_clusters.n_sae_features}"
        )

    raw_ids = _require_clustered_delta_integer_cpu_vector(
        "clustered delta raw current feature IDs",
        raw_fids,
    )
    positive_ids = _require_clustered_delta_integer_cpu_vector(
        "clustered delta positive candidate IDs",
        positive_candidate_fids,
    )
    negative_ids = _require_clustered_delta_integer_cpu_vector(
        "clustered delta negative candidate IDs",
        negative_candidate_fids,
    )
    if not isinstance(negative_candidate_weights, t.Tensor):
        raise FeatureClustersError(
            "clustered delta negative candidate weights must be a tensor, got "
            f"{type(negative_candidate_weights).__name__}"
        )
    if negative_candidate_weights.device.type != "cpu" or negative_candidate_weights.ndim != 1:
        raise FeatureClustersError(
            "clustered delta negative candidate weights must be a "
            "one-dimensional CPU tensor, got "
            f"shape={tuple(negative_candidate_weights.shape)}, "
            f"device={negative_candidate_weights.device}"
        )
    if not negative_candidate_weights.is_floating_point():
        raise FeatureClustersError(
            "clustered delta negative candidate weights must be floating point, "
            f"got {negative_candidate_weights.dtype}"
        )
    if negative_candidate_weights.shape != negative_ids.shape:
        raise FeatureClustersError(
            "clustered delta negative candidate IDs and weights must be aligned, "
            f"got ids={tuple(negative_ids.shape)}, "
            f"weights={tuple(negative_candidate_weights.shape)}"
        )
    if not bool(t.isfinite(negative_candidate_weights).all()):
        raise FeatureClustersError(
            "clustered delta negative candidate weights contain a non-finite value"
        )
    if bool((negative_candidate_weights < 0).any()):
        raise FeatureClustersError(
            "clustered delta negative candidate weights must be non-negative"
        )
    negative_weights = negative_candidate_weights.float()

    if not isinstance(ref_mask, t.Tensor):
        raise FeatureClustersError(
            f"clustered delta reference presence must be a tensor, got {type(ref_mask).__name__}"
        )
    if ref_mask.device.type != "cpu":
        raise FeatureClustersError(
            f"clustered delta reference presence must be on CPU, got device={ref_mask.device}"
        )
    feature_clusters.assert_representative_presence(
        ref_mask,
        surface="clustered delta reference presence",
    )
    feature_clusters.assert_representatives_only(
        positive_ids,
        surface="clustered delta positive candidates",
    )
    feature_clusters.assert_representatives_only(
        negative_ids,
        surface="clustered delta negative candidates",
    )

    for name, ids in (
        ("positive", positive_ids),
        ("negative", negative_ids),
    ):
        if ids.numel() != t.unique(ids).numel():
            raise FeatureClustersError(
                f"clustered delta {name} candidates contain duplicate feature IDs"
            )
        if ids.numel() and not bool(ref_mask[ids].all()):
            bad = int(ids[~ref_mask[ids]][0].item())
            raise FeatureClustersError(
                f"clustered delta {name} candidate feature ID {bad} is absent "
                "from the current/reference presence union"
            )
    if positive_ids.numel() and negative_ids.numel():
        overlap = t.isin(negative_ids, positive_ids)
        if bool(overlap.any()):
            bad = int(negative_ids[overlap][0].item())
            raise FeatureClustersError(
                f"clustered delta feature ID {bad} appears in both positive "
                "and negative candidate pools"
            )

    if raw_ids.numel():
        if int(raw_ids.min().item()) < 0 or int(raw_ids.max().item()) >= n_sae_features:
            invalid = (raw_ids < 0) | (raw_ids >= n_sae_features)
            bad = int(raw_ids[invalid][0].item())
            raise FeatureClustersError(
                f"clustered delta raw current feature ID {bad} is out of range "
                f"[0, {n_sae_features})"
            )
        allowed_raw = raw_ids[feature_clusters.allowed_universe[raw_ids]]
        if allowed_raw.numel():
            raw_representatives = feature_clusters.rep_of[allowed_raw]
            missing = ~ref_mask[raw_representatives]
            if bool(missing.any()):
                bad = int(raw_representatives[missing][0].item())
                raise FeatureClustersError(
                    f"clustered delta raw current representative {bad} is absent "
                    "from the current/reference presence union"
                )

    return negative_ids, negative_weights, ref_mask


def _materialize_splice_segment_selection(
    *,
    raw_fids: t.Tensor,
    raw_mags: t.Tensor,
    raw_pcts: t.Tensor | None,
    delta_result: Any | None,
    pipeline_config: FeaturePipelineConfig,
    feature_clusters: FeatureClusterConsolidation | None = None,
    tensor_output: bool = False,
) -> (
    tuple[list[int], list[int], list[int], list[int]]
    | tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]
):
    """Select positives/negatives for one packed segment."""
    has_delta = delta_result is not None
    # An active descriptor remains part of artifact identity and final output
    # validation even when the resolved manual mask leaves at most one member
    # per cluster. Computationally that state is a strict identity: route it
    # through every legacy selection/materialization branch so active-inert
    # clustering cannot perturb buckets, serialized percentiles, or RNG state.
    # The bool is cached at cluster build time; never scan the full SAE-width
    # consolidated-away tensor in this per-segment hot path.
    cluster_consolidation_active = (
        feature_clusters is not None and not feature_clusters.is_identity_after_mask
    )
    clustered_delta_result = isinstance(
        delta_result,
        ClusteredCounterfactualDeltaSegResult,
    )
    if has_delta and cluster_consolidation_active != clustered_delta_result:
        expected = (
            "representative-level clustered" if cluster_consolidation_active else "raw-feature"
        )
        raise FeatureClustersError(
            "splice-counterfactual materialization received the wrong delta "
            f"currency: expected {expected} results for the configured cluster "
            "state"
        )
    delta_values: t.Tensor | None = None
    neg_delta_fids: t.Tensor | None = None
    neg_delta_weights: t.Tensor | None = None
    ref_mask: t.Tensor | None = None

    if delta_result is None:
        fids = raw_fids.to(t.int64)
        mags = raw_mags.float()
        pcts = raw_pcts
        if pipeline_config.feature_mask is not None:
            if pcts is not None:
                fids, mags, pcts = pipeline_config.feature_mask.apply_to_positive(fids, mags, pcts)
            else:
                fids, mags = pipeline_config.feature_mask.apply_to_positive(fids, mags)
    else:
        (
            fids,
            mags,
            delta_values,
            neg_delta_fids,
            neg_delta_weights,
            ref_mask,
            pcts,
        ) = delta_result
        if clustered_delta_result and pipeline_config.negative_bundle is not None:
            assert feature_clusters is not None
            assert neg_delta_fids is not None
            assert neg_delta_weights is not None
            assert ref_mask is not None
            neg_delta_fids, neg_delta_weights, ref_mask = (
                _validate_clustered_delta_negative_contract(
                    raw_fids=raw_fids,
                    positive_candidate_fids=fids,
                    negative_candidate_fids=neg_delta_fids,
                    negative_candidate_weights=neg_delta_weights,
                    ref_mask=ref_mask,
                    feature_clusters=feature_clusters,
                    n_sae_features=pipeline_config.negative_bundle.n_sae_features,
                )
            )
        fids = fids.to(t.int64)
        mags = mags.float()
        if pipeline_config.feature_mask is not None and not clustered_delta_result:
            keep = pipeline_config.feature_mask.keep_mask(fids)
            fids = fids[keep]
            mags = mags[keep]
            if delta_values is not None:
                delta_values = delta_values[keep]
            if pcts is not None:
                pcts = pcts[keep]

    if feature_clusters is not None:
        if not isinstance(pipeline_config.selection_config, SamplingBundle):
            raise FeatureClustersError(
                "internal error: clustered splice materialization requires "
                "SamplingBundle after the startup support gate"
            )
        if cluster_consolidation_active and not clustered_delta_result:
            if pcts is None:
                pcts = (
                    magnitude_to_percentile(
                        fids.to(t.int64),
                        mags.float(),
                        pipeline_config.selection_config.percentile_table,
                    )
                    / 100.0
                ).float()
            if delta_values is not None:
                fids, pcts, delta_values = feature_clusters.consolidate_segment(
                    fids,
                    pcts,
                    delta_values,
                )
            else:
                fids, pcts = feature_clusters.consolidate_segment(fids, pcts)
            # A representative has no meaningful raw magnitude when a sibling
            # supplied its maximum percentile. Sampling consumes the explicitly
            # supplied percentile/score channels; keep a shape-aligned neutral
            # magnitude tensor so no downstream branch can reinterpret a sibling's
            # raw scale as representative evidence.
            mags = t.zeros_like(pcts, dtype=t.float32)

    if isinstance(pipeline_config.selection_config, SamplingBundle):
        bundle = pipeline_config.selection_config
        override_weights: t.Tensor | None = None
        if has_delta and delta_values is not None and len(delta_values) > 0:
            if bundle.selection.delta_weights is DeltaSamplingWeights.percentile_delta:
                override_weights = delta_values
        sampled = sample_features_batched(
            all_feature_ids=fids,
            all_feature_mags=mags,
            segment_sizes=[int(fids.numel())],
            percentile_table=bundle.percentile_table,
            config=bundle.selection,
            rng=bundle.rng,
            override_weights_list=[override_weights],
            precomputed_percentiles=pcts,
        )
        if sampled:
            selected_fids, selected_mags = sampled[0]
        else:
            selected_fids = t.tensor([], dtype=t.int64)
            selected_mags = t.tensor([], dtype=t.float32)
    elif isinstance(pipeline_config.selection_config, ThresholdSelectionConfig):
        th_cfg = ThresholdConfig(
            pipeline_config.selection_config.selection.threshold_type,
            pipeline_config.selection_config.selection.threshold,
        )
        selected_fids, selected_mags = apply_threshold(
            th_cfg,
            fids,
            mags,
            precomputed=pipeline_config.selection_config.precomputed_thresholds,
        )
    else:
        raise TypeError(
            f"Unsupported selection config: {type(pipeline_config.selection_config).__name__}"
        )

    if selected_fids.numel() > 0 and cluster_consolidation_active:
        assert pcts is not None
        selected_pcts = _lookup_selected_percentiles(
            selected_fids=selected_fids.to(t.int64),
            candidate_fids=fids.to(t.int64),
            candidate_pcts=pcts.float(),
        )
        buckets = percentile_to_bucket(
            selected_pcts,
            pipeline_config.effective_buckets,
        )
        sel_pcts_vals = normalized_percentile_to_uint8(selected_pcts)
    elif selected_fids.numel() > 0:
        buckets = discretize_activations(
            selected_fids.to(t.int64),
            selected_mags.float(),
            pipeline_config.boundaries,
        )
        sel_pcts_vals = _selected_pcts_uint8(
            selected_fids=selected_fids.to(t.int64),
            selected_mags=selected_mags.float(),
            boundaries=pipeline_config.boundaries,
        )
    else:
        buckets = t.tensor([], dtype=t.int64)
        sel_pcts_vals = t.tensor([], dtype=t.uint8)

    neg_fids = t.tensor([], dtype=t.int64)
    if pipeline_config.negative_bundle is not None:
        neg_bundle = pipeline_config.negative_bundle
        neg_mask = (
            feature_clusters.effective_negative_mask()
            if cluster_consolidation_active and feature_clusters is not None
            else negative_sampling_mask(pipeline_config)
        )
        if has_delta and neg_delta_fids is not None and ref_mask is not None:
            assert pipeline_config.delta_bundle is not None
            assert neg_delta_weights is not None
            neg_k = sample_k(
                neg_bundle.selection.distribution,
                neg_bundle.selection.mean,
                neg_bundle.selection.std,
                neg_bundle.rng,
                min_k=neg_bundle.selection.min_k,
                max_k=neg_bundle.selection.max_k,
            )
            allowed = (
                neg_mask.allowed_universe_mask(neg_bundle.n_sae_features)
                if neg_mask is not None
                else None
            )
            raw_feature_ids_for_negatives = raw_fids.to(t.int64)
            ref_mask_for_negatives = ref_mask
            positive_ids_for_negatives = selected_fids.to(t.int64)
            if cluster_consolidation_active:
                assert feature_clusters is not None
                # The pre-RNG contract guard proved that this clustered result
                # already contains disjoint representative-space candidates,
                # that ref_mask is already projected, and that it covers both
                # the raw-current representatives and every positive candidate.
                # Empty redundant exclusion lists preserve the exact eligible
                # pools while avoiding two full-width projections and a
                # full-width positive-membership filter.
                raw_feature_ids_for_negatives = fids[:0]
                ref_mask_for_negatives = ref_mask
                positive_ids_for_negatives = fids[:0]
            if allowed is not None and cluster_consolidation_active:
                uniform_capacity = int(
                    (allowed.to(ref_mask_for_negatives.device) & ~ref_mask_for_negatives)
                    .sum()
                    .item()
                )
                # The filtered delta pool is disjoint from the uniform pool:
                # every delta candidate is current/reference-present, while
                # uniform candidates are exactly outside projected ref_mask.
                # Include both capacities. Capping only to uniform capacity
                # would incorrectly suppress valid representative-attached
                # delta negatives when the reference covers the whole allowed
                # universe.
                allowed_pool = int(neg_delta_fids.numel()) + uniform_capacity
                neg_k = max(0, min(neg_k, allowed_pool))
            elif allowed is not None:
                if len(raw_fids) > 0:
                    active_bits = allowed.clone()
                    active_bits[raw_fids.long()] = False
                    allowed_pool = int(active_bits.sum().item())
                else:
                    allowed_pool = int(allowed.sum().item())
                neg_k = max(0, min(neg_k, allowed_pool))
            else:
                neg_k = max(
                    0,
                    min(neg_k, neg_bundle.n_sae_features - len(raw_fids)),
                )
            neg_fids = sample_delta_negatives(
                k=neg_k,
                neg_delta_ratio=pipeline_config.delta_bundle.config.neg_delta_ratio,
                neg_delta_fids=neg_delta_fids,
                neg_delta_weights=neg_delta_weights,
                raw_feature_ids=raw_feature_ids_for_negatives,
                ref_mask=ref_mask_for_negatives,
                selected_positive_ids=positive_ids_for_negatives,
                n_sae_features=neg_bundle.n_sae_features,
                rng=neg_bundle.rng,
                feature_mask=neg_mask,
            )
        else:
            sampled_neg = sample_inactive_features_batched(
                [fids.to(t.int64) if cluster_consolidation_active else raw_fids.to(t.int64)],
                neg_bundle.n_sae_features,
                neg_bundle.selection,
                neg_bundle.rng,
                feature_mask=neg_mask,
            )
            if sampled_neg:
                neg_fids = sampled_neg[0]

    if feature_clusters is None and pipeline_config.feature_mask is not None:
        pipeline_config.feature_mask.assert_no_masked(selected_fids.to(t.int64))
        pipeline_config.feature_mask.assert_no_masked(neg_fids.to(t.int64))
    if feature_clusters is not None:
        feature_clusters.assert_representatives_only(
            selected_fids.to(t.int64),
            surface="splice positive selection",
        )
        feature_clusters.assert_representatives_only(
            neg_fids.to(t.int64),
            surface="splice negative selection",
        )

    selected_fids = selected_fids.to(t.int64)
    buckets = buckets.to(t.int64)
    sel_pcts_vals = sel_pcts_vals.to(t.int64)
    neg_fids = neg_fids.to(t.int64)
    if tensor_output:
        return selected_fids, buckets, sel_pcts_vals, neg_fids
    return (
        selected_fids.tolist(),
        buckets.tolist(),
        cast(list[int], sel_pcts_vals.tolist()),
        neg_fids.tolist(),
    )


@dataclass(frozen=True)
class _OrdinarySpliceSelectionInput:
    """One non-delta segment waiting for tensor-native batched selection."""

    segment_index: int
    raw_fids: t.Tensor
    raw_mags: t.Tensor
    raw_pcts: t.Tensor | None


@dataclass(frozen=True)
class _TensorSelectionBatch:
    """Tensor-native CSR rows in local batch order.

    ``segment_indices`` maps each local CSR row back to its packed-shard row.
    Feature payloads never pass through Python scalars or lists.
    """

    segment_indices: tuple[int, ...]
    sel_counts: t.Tensor
    sel_fids: t.Tensor
    sel_buckets: t.Tensor
    sel_pcts: t.Tensor
    neg_counts: t.Tensor
    neg_fids: t.Tensor


def _empty_cpu_tensor(dtype: t.dtype) -> t.Tensor:
    return t.empty(0, dtype=dtype, device="cpu")


def _cat_cpu_tensors(values: list[t.Tensor], dtype: t.dtype) -> t.Tensor:
    if not values or all(value.numel() == 0 for value in values):
        return _empty_cpu_tensor(dtype)
    return t.cat([value.to(dtype=dtype, device="cpu") for value in values])


def _require_cpu_vector(name: str, value: t.Tensor) -> None:
    if value.device.type != "cpu" or value.ndim != 1:
        raise SpliceProvenanceError(
            f"{name} must be a one-dimensional CPU tensor, got "
            f"shape={tuple(value.shape)}, device={value.device}"
        )


def _require_integer_cpu_vector(name: str, value: t.Tensor) -> None:
    _require_cpu_vector(name, value)
    if value.dtype is t.bool or value.is_floating_point() or value.is_complex():
        raise SpliceProvenanceError(f"{name} must use an integer dtype, got {value.dtype}")


def _require_storage_range(name: str, value: t.Tensor, maximum: int) -> None:
    # CPU PyTorch does not implement min/max reductions for uint16. Widening
    # is exact for every supported selection storage type and also makes this
    # validator safe for directly-constructed, already-narrowed batches.
    range_values = value.to(t.int64)
    if range_values.numel() > 0 and (
        int(range_values.min().item()) < 0 or int(range_values.max().item()) > maximum
    ):
        raise SpliceProvenanceError(f"{name} is outside [0, {maximum}]")


def _tensor_selection_batch(
    segment_indices: list[int] | tuple[int, ...],
    rows: list[tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]],
) -> _TensorSelectionBatch:
    """Pack per-row tensor results without tensor -> list -> tensor copies."""
    indices = tuple(segment_indices)
    if (
        not indices
        or any(type(index) is not int for index in indices)
        or len(indices) != len(rows)
        or len(set(indices)) != len(indices)
    ):
        raise SpliceProvenanceError(
            "tensor selection batch requires a non-empty row set and one "
            "unique segment index (an int) per row"
        )

    sel_counts: list[int] = []
    neg_counts: list[int] = []
    sel_fids: list[t.Tensor] = []
    sel_buckets: list[t.Tensor] = []
    sel_pcts: list[t.Tensor] = []
    neg_fids: list[t.Tensor] = []
    for row_idx, (row_fids, row_buckets, row_pcts, row_neg) in enumerate(rows):
        for name, value in (
            ("selected feature IDs", row_fids),
            ("selected buckets", row_buckets),
            ("selected percentiles", row_pcts),
            ("negative feature IDs", row_neg),
        ):
            _require_integer_cpu_vector(f"batch row {row_idx} {name}", value)
        if not (row_fids.numel() == row_buckets.numel() == row_pcts.numel()):
            raise SpliceProvenanceError(
                f"batch row {row_idx} has misaligned positive tensors: "
                f"{row_fids.numel()} IDs, {row_buckets.numel()} buckets, "
                f"{row_pcts.numel()} percentiles"
            )
        sel_counts.append(int(row_fids.numel()))
        neg_counts.append(int(row_neg.numel()))
        sel_fids.append(row_fids)
        sel_buckets.append(row_buckets)
        sel_pcts.append(row_pcts)
        neg_fids.append(row_neg)

    packed_sel_fids = _cat_cpu_tensors(sel_fids, t.int64)
    packed_sel_buckets = _cat_cpu_tensors(sel_buckets, t.int64)
    packed_sel_pcts = _cat_cpu_tensors(sel_pcts, t.int64)
    packed_neg_fids = _cat_cpu_tensors(neg_fids, t.int64)
    _require_storage_range(
        "selected feature ID",
        packed_sel_fids,
        t.iinfo(t.uint16).max,
    )
    _require_storage_range(
        "negative feature ID",
        packed_neg_fids,
        t.iinfo(t.uint16).max,
    )
    _require_storage_range(
        "bucket",
        packed_sel_buckets,
        t.iinfo(t.uint8).max,
    )
    _require_storage_range(
        "percentile",
        packed_sel_pcts,
        t.iinfo(t.uint8).max,
    )
    return _TensorSelectionBatch(
        segment_indices=indices,
        sel_counts=t.tensor(sel_counts, dtype=t.int32),
        sel_fids=packed_sel_fids.to(t.uint16),
        sel_buckets=packed_sel_buckets.to(t.uint8),
        sel_pcts=packed_sel_pcts.to(t.uint8),
        neg_counts=t.tensor(neg_counts, dtype=t.int32),
        neg_fids=packed_neg_fids.to(t.uint16),
    )


class _TensorSelectionShardBuilder:
    """Assemble out-of-order tensor batches into packed-segment CSR order."""

    def __init__(self, total_segments: int) -> None:
        if type(total_segments) is not int or total_segments < 0:
            raise ValueError("total_segments must be a non-negative int")
        self._total_segments = total_segments
        self._covered: set[int] = set()
        self._batches: list[_TensorSelectionBatch] = []

    def add(self, batch: _TensorSelectionBatch) -> None:
        if (
            not batch.segment_indices
            or any(type(index) is not int for index in batch.segment_indices)
            or len(set(batch.segment_indices)) != len(batch.segment_indices)
        ):
            raise SpliceProvenanceError(
                "selection batch must carry non-empty, unique integer segment indices"
            )
        for name, counts in (
            ("selected counts", batch.sel_counts),
            ("negative counts", batch.neg_counts),
        ):
            _require_integer_cpu_vector(f"selection batch {name}", counts)
            if counts.numel() != len(batch.segment_indices):
                raise SpliceProvenanceError(
                    f"selection batch {name} has {counts.numel()} rows for "
                    f"{len(batch.segment_indices)} segment indices"
                )
            if counts.numel() and int(counts.min().item()) < 0:
                raise SpliceProvenanceError(f"selection batch {name} must be non-negative")
        for name, payload in (
            ("selected feature IDs", batch.sel_fids),
            ("selected buckets", batch.sel_buckets),
            ("selected percentiles", batch.sel_pcts),
            ("negative feature IDs", batch.neg_fids),
        ):
            _require_integer_cpu_vector(f"selection batch {name}", payload)
        selected_count = int(batch.sel_counts.to(t.int64).sum().item())
        negative_count = int(batch.neg_counts.to(t.int64).sum().item())
        if not (
            batch.sel_fids.numel()
            == batch.sel_buckets.numel()
            == batch.sel_pcts.numel()
            == selected_count
        ):
            raise SpliceProvenanceError(
                "selection batch positive counts and payload tensors are misaligned"
            )
        if batch.neg_fids.numel() != negative_count:
            raise SpliceProvenanceError(
                "selection batch negative counts and payload tensor are misaligned"
            )
        for name, payload, storage_dtype, maximum in (
            (
                "selected feature ID",
                batch.sel_fids,
                t.uint16,
                t.iinfo(t.uint16).max,
            ),
            ("selected bucket", batch.sel_buckets, t.uint8, t.iinfo(t.uint8).max),
            (
                "selected percentile",
                batch.sel_pcts,
                t.uint8,
                t.iinfo(t.uint8).max,
            ),
            (
                "negative feature ID",
                batch.neg_fids,
                t.uint16,
                t.iinfo(t.uint16).max,
            ),
        ):
            # The normal producer batch is already narrowed and its dtype is
            # itself a range proof. Directly-constructed wider batches still
            # receive the full fail-loud validation without adding a second
            # feature-payload widening pass to the hot path.
            if payload.dtype != storage_dtype:
                _require_storage_range(f"selection batch {name}", payload, maximum)

        out_of_range = next(
            (index for index in batch.segment_indices if not 0 <= index < self._total_segments),
            None,
        )
        if out_of_range is not None:
            raise SpliceProvenanceError(
                f"selection batch segment index {out_of_range} is outside "
                f"[0, {self._total_segments})"
            )
        overlap = self._covered.intersection(batch.segment_indices)
        if overlap:
            repeated = min(overlap)
            raise SpliceProvenanceError(
                f"selection batch materialized packed segment {repeated} more than once"
            )
        # Commit only after the complete batch has passed validation. A caller
        # that catches a rejected add can still safely use the builder.
        self._covered.update(batch.segment_indices)
        self._batches.append(batch)

    @staticmethod
    def _row_targets(
        *,
        payload_size: int,
        local_counts: t.Tensor,
        global_offsets: t.Tensor,
        segment_indices: tuple[int, ...],
    ) -> t.Tensor:
        counts = local_counts.to(t.int64)
        total = int(counts.sum().item())
        if total == 0:
            if payload_size != 0:
                raise SpliceProvenanceError("zero-count selection batch carries non-empty payload")
            return t.empty(0, dtype=t.int64)
        if payload_size != total:
            raise SpliceProvenanceError(
                f"selection batch counts sum to {total}, but payload has {payload_size} values"
            )
        local_offsets = t.cat((t.zeros(1, dtype=t.int64), counts.cumsum(0)[:-1]))
        repeated_local_offsets = t.repeat_interleave(local_offsets, counts)
        within_row = t.arange(total, dtype=t.int64) - repeated_local_offsets
        global_starts = global_offsets[t.tensor(segment_indices, dtype=t.int64)]
        return t.repeat_interleave(global_starts, counts) + within_row

    def build(self) -> SelectionShard:
        expected = set(range(self._total_segments))
        if self._covered != expected:
            missing = sorted(expected - self._covered)
            extra = sorted(self._covered - expected)
            raise SpliceProvenanceError(
                "tensor selection batches do not exactly cover the packed "
                f"segments: missing={missing[:8]}, extra={extra[:8]}"
            )

        sel_counts = t.zeros(self._total_segments, dtype=t.int64)
        neg_counts = t.zeros(self._total_segments, dtype=t.int64)
        for batch in self._batches:
            indices = t.tensor(batch.segment_indices, dtype=t.int64)
            sel_counts[indices] = batch.sel_counts.to(t.int64)
            neg_counts[indices] = batch.neg_counts.to(t.int64)

        sel_offsets64 = t.cat((t.zeros(1, dtype=t.int64), sel_counts.cumsum(0)))
        neg_offsets64 = t.cat((t.zeros(1, dtype=t.int64), neg_counts.cumsum(0)))
        int32_max = t.iinfo(t.int32).max
        if int(sel_offsets64[-1].item()) > int32_max:
            raise SpliceProvenanceError("selected-feature CSR exceeds int32 offsets")
        if int(neg_offsets64[-1].item()) > int32_max:
            raise SpliceProvenanceError("negative-feature CSR exceeds int32 offsets")

        # PyTorch CPU advanced assignment is not implemented for uint16.
        # Scatter IDs in int64, then perform one bulk narrowing conversion;
        # range checks happened when each batch was packed.
        sel_fids_work = t.empty(int(sel_offsets64[-1].item()), dtype=t.int64)
        sel_buckets = t.empty(sel_fids_work.numel(), dtype=t.uint8)
        sel_pcts = t.empty(sel_fids_work.numel(), dtype=t.uint8)
        neg_fids_work = t.empty(int(neg_offsets64[-1].item()), dtype=t.int64)
        for batch in self._batches:
            sel_targets = self._row_targets(
                payload_size=batch.sel_fids.numel(),
                local_counts=batch.sel_counts,
                global_offsets=sel_offsets64,
                segment_indices=batch.segment_indices,
            )
            if not (batch.sel_buckets.numel() == batch.sel_pcts.numel() == batch.sel_fids.numel()):
                raise SpliceProvenanceError(
                    "selection batch positive payload tensors are misaligned"
                )
            sel_fids_work[sel_targets] = batch.sel_fids.to(t.int64)
            sel_buckets[sel_targets] = batch.sel_buckets
            sel_pcts[sel_targets] = batch.sel_pcts
            neg_targets = self._row_targets(
                payload_size=batch.neg_fids.numel(),
                local_counts=batch.neg_counts,
                global_offsets=neg_offsets64,
                segment_indices=batch.segment_indices,
            )
            neg_fids_work[neg_targets] = batch.neg_fids.to(t.int64)

        return SelectionShard(
            sel_offsets=sel_offsets64.to(t.int32),
            sel_fids=sel_fids_work.to(t.uint16),
            sel_buckets=sel_buckets,
            sel_pcts=sel_pcts,
            neg_offsets=neg_offsets64.to(t.int32),
            neg_fids=neg_fids_work.to(t.uint16),
        )


def _materialize_ordinary_splice_selection_batch(
    inputs: list[_OrdinarySpliceSelectionInput],
    *,
    pipeline_config: FeaturePipelineConfig,
    feature_clusters: FeatureClusterConsolidation | None = None,
) -> _TensorSelectionBatch:
    """Select one execution-order run of ordinary, non-delta segments."""
    if not inputs:
        raise ValueError("ordinary splice selection batch must not be empty")
    segment_indices = tuple(item.segment_index for item in inputs)
    if any(type(index) is not int for index in segment_indices) or len(set(segment_indices)) != len(
        segment_indices
    ):
        raise SpliceProvenanceError(
            "ordinary splice selection batch requires unique integer segment indices"
        )
    cluster_consolidation_active = (
        feature_clusters is not None and not feature_clusters.is_identity_after_mask
    )
    if feature_clusters is not None and not isinstance(
        pipeline_config.selection_config,
        SamplingBundle,
    ):
        raise FeatureClustersError(
            "internal error: clustered splice materialization requires "
            "SamplingBundle after the startup support gate"
        )

    candidate_fids: list[t.Tensor] = []
    candidate_mags: list[t.Tensor] = []
    candidate_pcts: list[t.Tensor | None] = []
    for item in inputs:
        _require_integer_cpu_vector("ordinary raw feature IDs", item.raw_fids)
        _require_cpu_vector("ordinary raw feature magnitudes", item.raw_mags)
        if item.raw_pcts is not None:
            _require_cpu_vector("ordinary raw percentiles", item.raw_pcts)
        if item.raw_fids.numel() != item.raw_mags.numel() or (
            item.raw_pcts is not None and item.raw_fids.numel() != item.raw_pcts.numel()
        ):
            raise SpliceProvenanceError(
                f"ordinary packed segment {item.segment_index} has misaligned "
                f"raw tensors: {item.raw_fids.numel()} IDs, "
                f"{item.raw_mags.numel()} magnitudes, "
                + (
                    "no percentile overlay"
                    if item.raw_pcts is None
                    else f"{item.raw_pcts.numel()} percentiles"
                )
            )
        fids = item.raw_fids.to(t.int64)
        mags = item.raw_mags.float()
        pcts = item.raw_pcts
        if pipeline_config.feature_mask is not None:
            if pcts is not None:
                fids, mags, pcts = pipeline_config.feature_mask.apply_to_positive(
                    fids,
                    mags,
                    pcts,
                )
            else:
                fids, mags = pipeline_config.feature_mask.apply_to_positive(
                    fids,
                    mags,
                )
        if cluster_consolidation_active:
            assert feature_clusters is not None
            assert isinstance(pipeline_config.selection_config, SamplingBundle)
            if pcts is None:
                pcts = (
                    magnitude_to_percentile(
                        fids,
                        mags,
                        pipeline_config.selection_config.percentile_table,
                    )
                    / 100.0
                ).float()
            fids, pcts = feature_clusters.consolidate_segment(fids, pcts)
            mags = t.zeros_like(pcts, dtype=t.float32)
        candidate_fids.append(fids)
        candidate_mags.append(mags)
        candidate_pcts.append(pcts)

    segment_sizes = [int(fids.numel()) for fids in candidate_fids]
    all_fids = _cat_cpu_tensors(candidate_fids, t.int64)
    all_mags = _cat_cpu_tensors(candidate_mags, t.float32)
    have_precomputed = [pcts is not None for pcts in candidate_pcts]
    if any(have_precomputed) and not all(have_precomputed):
        raise SpliceProvenanceError(
            "ordinary selection batch mixed segments with and without precomputed percentiles"
        )
    all_pcts = (
        _cat_cpu_tensors(
            [cast(t.Tensor, pcts) for pcts in candidate_pcts],
            t.float32,
        )
        if all(have_precomputed)
        else None
    )

    if isinstance(pipeline_config.selection_config, SamplingBundle):
        bundle = pipeline_config.selection_config
        selected = sample_features_batched(
            all_feature_ids=all_fids,
            all_feature_mags=all_mags,
            segment_sizes=segment_sizes,
            percentile_table=bundle.percentile_table,
            config=bundle.selection,
            rng=bundle.rng,
            override_weights_list=[None] * len(inputs),
            precomputed_percentiles=all_pcts,
        )
    elif isinstance(
        pipeline_config.selection_config,
        ThresholdSelectionConfig,
    ):
        th_cfg = ThresholdConfig(
            pipeline_config.selection_config.selection.threshold_type,
            pipeline_config.selection_config.selection.threshold,
        )
        selected = [
            apply_threshold(
                th_cfg,
                fids,
                mags,
                precomputed=pipeline_config.selection_config.precomputed_thresholds,
            )
            for fids, mags in zip(candidate_fids, candidate_mags, strict=True)
        ]
    else:
        raise TypeError(
            f"Unsupported selection config: {type(pipeline_config.selection_config).__name__}"
        )

    selected_rows: list[tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]] = []
    neg_rows: list[t.Tensor]
    if pipeline_config.negative_bundle is None:
        neg_rows = [_empty_cpu_tensor(t.int64) for _ in inputs]
    else:
        neg_bundle = pipeline_config.negative_bundle
        neg_mask = (
            feature_clusters.effective_negative_mask()
            if cluster_consolidation_active and feature_clusters is not None
            else negative_sampling_mask(pipeline_config)
        )
        negative_active = [
            fids if cluster_consolidation_active else item.raw_fids.to(t.int64)
            for item, fids in zip(inputs, candidate_fids, strict=True)
        ]
        neg_rows = sample_inactive_features_batched(
            negative_active,
            neg_bundle.n_sae_features,
            neg_bundle.selection,
            neg_bundle.rng,
            feature_mask=neg_mask,
        )

    for row_idx, ((selected_fids, selected_mags), neg_fids) in enumerate(
        zip(selected, neg_rows, strict=True)
    ):
        fids = candidate_fids[row_idx]
        pcts = candidate_pcts[row_idx]
        if selected_fids.numel() > 0 and cluster_consolidation_active:
            assert pcts is not None
            selected_pcts = _lookup_selected_percentiles(
                selected_fids=selected_fids.to(t.int64),
                candidate_fids=fids,
                candidate_pcts=pcts.float(),
            )
            buckets = percentile_to_bucket(
                selected_pcts,
                pipeline_config.effective_buckets,
            )
            selected_pcts_uint8 = normalized_percentile_to_uint8(selected_pcts)
        elif selected_fids.numel() > 0:
            buckets = discretize_activations(
                selected_fids.to(t.int64),
                selected_mags.float(),
                pipeline_config.boundaries,
            )
            selected_pcts_uint8 = _selected_pcts_uint8(
                selected_fids=selected_fids.to(t.int64),
                selected_mags=selected_mags.float(),
                boundaries=pipeline_config.boundaries,
            )
        else:
            buckets = _empty_cpu_tensor(t.int64)
            selected_pcts_uint8 = _empty_cpu_tensor(t.uint8)

        if feature_clusters is None and pipeline_config.feature_mask is not None:
            pipeline_config.feature_mask.assert_no_masked(selected_fids.to(t.int64))
            pipeline_config.feature_mask.assert_no_masked(neg_fids.to(t.int64))
        selected_rows.append(
            (
                selected_fids.to(t.int64),
                buckets.to(t.int64),
                selected_pcts_uint8.to(t.int64),
                neg_fids.to(t.int64),
            )
        )

    return _tensor_selection_batch(
        segment_indices,
        selected_rows,
    )


def _seg_mags_or_zeros(seg: Any) -> t.Tensor:
    """Segment feature magnitudes as float32, or right-shaped ZEROS when the source
    FeatureIndex was built with skip_feature_mags (issue #740 PR-E — the mags tensor
    is an empty placeholder to halve resident RAM). A no-op when mags ARE loaded
    (returns the real values, byte-identical). skip_feature_mags is only enabled on
    the cluster-consolidation splice path, where magnitudes are provably unused
    (zeroed at materialization; deltas + refs use precomputed percentiles), so the
    zeros substitution is byte-identical there."""
    mags = seg.feature_mags
    if mags.numel() == 0 and seg.feature_ids.numel() > 0:
        return t.zeros(int(seg.feature_ids.numel()), dtype=t.float32)
    return mags.float()


def _validate_packed_segment_matches_feature_index(
    *,
    shard_name: str,
    seg_idx: int,
    packed,
    packed_feat_offsets: list[int],
    segments: list,
    seg_ordinal: int,
) -> tuple[t.Tensor, t.Tensor, t.Tensor | None]:
    if not (0 <= seg_ordinal < len(segments)):
        raise SpliceProvenanceError(
            f"{shard_name}: packed segment {seg_idx} points to seg_ordinal "
            f"{seg_ordinal}, but the feature-index doc has {len(segments)} "
            "segments."
        )
    seg = segments[seg_ordinal]
    f_start = packed_feat_offsets[seg_idx]
    f_end = packed_feat_offsets[seg_idx + 1]
    packed_fids = packed.feature_ids[f_start:f_end].to(t.int64)
    current_fids = seg.feature_ids.to(t.int64)
    if not t.equal(packed_fids, current_fids):
        raise SpliceProvenanceError(
            f"{shard_name}: packed segment {seg_idx} feature ids differ from "
            f"FeatureIndex segment {seg_ordinal}."
        )
    current_pcts: t.Tensor | None = None
    if seg.feature_percentiles is not None:
        packed_pcts = packed.feature_pcts[f_start:f_end].to(t.uint8)
        if not t.equal(packed_pcts, seg.feature_percentiles.to(t.uint8)):
            raise SpliceProvenanceError(
                f"{shard_name}: packed segment {seg_idx} percentiles differ "
                f"from FeatureIndex segment {seg_ordinal}."
            )
        current_pcts = uint8_to_percentile(seg.feature_percentiles)
    return current_fids, _seg_mags_or_zeros(seg), current_pcts


def splice_counterfactual_stats_for_packed_shard(
    base_shard_path: Path,
    *,
    provenance_path: Path,
    packed_info: PackedProvenanceInfo,
    splice_index: SpliceProvenanceIndex,
    packed_spool_dir: str | Path | None = None,
) -> SpliceSelectionStats:
    """Cheap metadata stats for an already-materialized splice selection shard."""
    with open_packed_row_shard(base_shard_path, spool_dir=packed_spool_dir) as packed_handle:
        return _splice_counterfactual_stats_for_loaded_packed_shard(
            base_shard_path,
            packed=packed_handle.shard,
            provenance_path=provenance_path,
            packed_info=packed_info,
            splice_index=splice_index,
        )


def _splice_counterfactual_stats_for_loaded_packed_shard(
    base_shard_path: Path,
    *,
    packed: PackedRowShard,
    provenance_path: Path,
    packed_info: PackedProvenanceInfo,
    splice_index: SpliceProvenanceIndex,
) -> SpliceSelectionStats:
    prov = load_overlay_shard(provenance_path)
    base_idx = prov[PROVENANCE_BASE_SHARD_IDX_KEY]
    doc_idx = prov[PROVENANCE_DOC_IDX_KEY]
    seg_ordinal = prov[PROVENANCE_SEG_ORDINAL_KEY]
    if not (len(base_idx) == len(doc_idx) == len(seg_ordinal) == packed.total_segments):
        raise SpliceProvenanceError(
            f"{base_shard_path.name}: provenance has lengths {len(base_idx)}, "
            f"{len(doc_idx)}, {len(seg_ordinal)} but packed shard has "
            f"{packed.total_segments} segments."
        )
    docs: set[tuple[int, int]] = set()
    for seg_idx in range(packed.total_segments):
        b = int(base_idx[seg_idx].item())
        d = int(doc_idx[seg_idx].item())
        if not (0 <= b < len(packed_info.base_shard_stems)):
            raise SpliceProvenanceError(
                f"{base_shard_path.name}: base_shard_idx {b} out of range "
                f"for {len(packed_info.base_shard_stems)} manifest stems."
            )
        docs.add((b, d))
    n_sidecar = 0
    for b, d in sorted(docs):
        stem = packed_info.base_shard_stems[b]
        n_sidecar += len(splice_index.doc_splices(stem, d))
    return SpliceSelectionStats(
        total_packed_segments=packed.total_segments,
        total_sidecar_splice_positions_referenced=n_sidecar,
        total_delta_positions_selected=n_sidecar,
    )


def select_splice_counterfactual_features_for_shard(
    base_shard_path: Path,
    pipeline_config: FeaturePipelineConfig,
    output_dir: Path,
    shard_name: str,
    *,
    provenance_path: Path,
    packed_info: PackedProvenanceInfo,
    feature_index: FeatureIndex,
    splice_index: SpliceProvenanceIndex,
    seed: int,
    feature_clusters: FeatureClusterConsolidation | None = None,
    packed_spool_dir: str | Path | None = None,
    splice_batch_size: int = 1,
) -> SpliceSelectionStats:
    """Precompute final selections for one packed shard under splice deltas."""
    splice_batch_size = resolve_splice_selection_batch_size(splice_batch_size)
    assert pipeline_config.delta_bundle is not None
    assert pipeline_config.delta_bundle.config.mode is DeltaMode.splice_counterfactual
    pipeline = _splice_shard_pipeline(
        pipeline_config,
        global_seed=seed,
        shard_name=shard_name,
    )
    delta_bundle = pipeline.delta_bundle
    assert delta_bundle is not None

    t0 = time.monotonic()
    feature_cache_before = feature_index.cache_stats.snapshot()
    sidecar_cache_before = splice_index.cache_stats.snapshot()
    with open_packed_row_shard(base_shard_path, spool_dir=packed_spool_dir) as packed_handle:
        return _select_splice_counterfactual_features_for_loaded_shard(
            pipeline=pipeline,
            packed_handle=packed_handle,
            output_dir=output_dir,
            shard_name=shard_name,
            provenance_path=provenance_path,
            packed_info=packed_info,
            feature_index=feature_index,
            splice_index=splice_index,
            feature_clusters=feature_clusters,
            splice_batch_size=splice_batch_size,
            t0=t0,
            feature_cache_before=feature_cache_before,
            sidecar_cache_before=sidecar_cache_before,
        )


def _select_splice_counterfactual_features_for_loaded_shard(
    *,
    pipeline: FeaturePipelineConfig,
    packed_handle: PackedRowShardHandle,
    output_dir: Path,
    shard_name: str,
    provenance_path: Path,
    packed_info: PackedProvenanceInfo,
    feature_index: FeatureIndex,
    splice_index: SpliceProvenanceIndex,
    feature_clusters: FeatureClusterConsolidation | None,
    splice_batch_size: int,
    t0: float,
    feature_cache_before: dict[str, int | float],
    sidecar_cache_before: dict[str, int | float],
) -> SpliceSelectionStats:
    packed = packed_handle.shard
    delta_bundle = pipeline.delta_bundle
    assert delta_bundle is not None
    prov = load_overlay_shard(provenance_path)
    base_idx = prov[PROVENANCE_BASE_SHARD_IDX_KEY]
    doc_idx = prov[PROVENANCE_DOC_IDX_KEY]
    seg_ordinal = prov[PROVENANCE_SEG_ORDINAL_KEY]
    if not (len(base_idx) == len(doc_idx) == len(seg_ordinal) == packed.total_segments):
        raise SpliceProvenanceError(
            f"{shard_name}: provenance has lengths {len(base_idx)}, "
            f"{len(doc_idx)}, {len(seg_ordinal)} but packed shard has "
            f"{packed.total_segments} segments."
        )
    load_done = time.monotonic()

    groups: dict[tuple[int, int], list[int]] = {}
    for seg_idx in range(packed.total_segments):
        key = (int(base_idx[seg_idx].item()), int(doc_idx[seg_idx].item()))
        groups.setdefault(key, []).append(seg_idx)
    grouping_done = time.monotonic()

    distinct_source_stems = len({b for (b, _d) in groups})
    ordered_groups = sorted(groups.items(), key=lambda item: min(item[1]))
    ordered_group_keys = [key for key, _segs in ordered_groups]
    source_stem_transitions = sum(
        left[0] != right[0]
        for left, right in itertools.pairwise(ordered_group_keys)
    )
    source_stem_runs = source_stem_transitions + bool(ordered_group_keys)
    logger.info(
        f"  {shard_name}: {len(groups)} (source_shard, doc) groups across "
        f"{distinct_source_stems} distinct source shard(s), "
        f"{source_stem_runs} source-stem run(s); "
        f"RSS {psutil.Process().memory_info().rss / 1e9:.1f} GB"
    )

    legacy_segment_selection = splice_batch_size == 1
    sel_by_seg: list[tuple[list[int], list[int], list[int], list[int]] | None] | None = (
        [None] * packed.total_segments if legacy_segment_selection else None
    )
    tensor_builder = (
        None if legacy_segment_selection else _TensorSelectionShardBuilder(packed.total_segments)
    )
    pending_ordinary: list[_OrdinarySpliceSelectionInput] = []
    ordinary_batch_count = 0
    ordinary_segment_count = 0
    maximum_ordinary_batch_size = 0
    serial_delta_segment_count = 0

    def flush_pending_ordinary() -> None:
        nonlocal ordinary_batch_count
        nonlocal ordinary_segment_count
        nonlocal maximum_ordinary_batch_size
        if not pending_ordinary:
            return
        assert tensor_builder is not None
        batch_segments = len(pending_ordinary)
        tensor_builder.add(
            _materialize_ordinary_splice_selection_batch(
                pending_ordinary,
                pipeline_config=pipeline,
                feature_clusters=feature_clusters,
            )
        )
        ordinary_batch_count += 1
        ordinary_segment_count += batch_segments
        maximum_ordinary_batch_size = max(
            maximum_ordinary_batch_size,
            batch_segments,
        )
        pending_ordinary.clear()

    packed_feat_offsets = packed.seg_feat_offsets.tolist()
    total_sidecar_positions = 0
    total_delta_positions = 0

    # PR-E mags-as-zeros is only sound when every segment carries a percentile overlay; guard the
    # per-segment lookups below against a missing overlay (issue #740 Codex review). NOTE: this
    # guards NEWLY-processed shards; a shard already written by a pre-guard build is trusted on
    # resume (stat-only path). That only matters for the corrupt-input case this guards — the
    # production data always has full percentile coverage, so the guard never fires there; an
    # operator who suspects a stale artifact must delete the sel/ dir to force reprocessing.
    skip_feature_mags_active = bool(getattr(feature_index, "_skip_feature_mags", False))

    f_cache_on = feature_index.max_resident_shards > 1
    s_cache_on = splice_index.max_resident_shards > 1
    if f_cache_on or s_cache_on:
        stems_first_need = _ordered_source_stems(ordered_group_keys, packed_info, shard_name)
        if f_cache_on:
            feature_index.drop_except(stems_first_need)
        if s_cache_on:
            splice_index.drop_except(stems_first_need)
        # Pre-seed feature-shard-then-sidecar per stem, in first-need order,
        # bounded by the SMALLER capacity, so the pre-seeded touches are a strict
        # PREFIX of the capacity-1 reference access sequence. Preserves first-error
        # identity under UNEQUAL capacities (incl. the shipped default 5!=16) and
        # multi-artifact corruption. Ensuring BOTH per stem is required: a per-index
        # bound lets the larger cache pre-load a stem whose paired artifact on the
        # smaller index has not been touched yet, reordering which corrupt artifact
        # fails first. Do NOT gate these two calls by f_cache_on/s_cache_on -- when a
        # cache is off (cap 1), ensure_resident is a benign single-slot load-and-hold
        # of exactly what the group loop would load first, and seed_bound is then 1.
        seed_bound = min(feature_index.max_resident_shards, splice_index.max_resident_shards)
        for stem in stems_first_need[:seed_bound]:
            feature_index.ensure_resident(stem)
            splice_index.ensure_resident(stem)
        n_stems = len(stems_first_need)
        # Warn on the SOURCE (feature-shard) cache specifically: its reloads are the
        # expensive ~GB decompress+materialize, whereas the sidecar cache is cheap
        # (~tens of MB). Fires only when source caching is enabled (cap>1) but is
        # still smaller than this shard's working set -- the operator-facing signal
        # that selection_source_cache_shards is undersized for this corpus (the
        # tail feature shards reload; output stays byte-identical).
        if f_cache_on and n_stems > feature_index.max_resident_shards:
            logger.warning(
                f"  {shard_name}: references {n_stems} source shards > source "
                f"cache capacity {feature_index.max_resident_shards}; the tail "
                f"reloads feature shards with LRU eviction."
            )

    for (base_shard_idx, doc_id), seg_indices in ordered_groups:
        if not (0 <= base_shard_idx < len(packed_info.base_shard_stems)):
            raise SpliceProvenanceError(
                f"{shard_name}: base_shard_idx {base_shard_idx} out of range "
                f"for {len(packed_info.base_shard_stems)} manifest stems."
            )
        derived_stem = packed_info.base_shard_stems[base_shard_idx]
        try:
            segments = feature_index.lookup(derived_stem, doc_id)
        except KeyError as exc:
            raise SpliceProvenanceError(
                f"{shard_name}: packed provenance points to missing derived "
                f"doc {derived_stem}:{doc_id}."
            ) from exc

        decoded_pcts: list[t.Tensor | None] = []
        for seg in segments:
            if seg.feature_percentiles is not None:
                decoded_pcts.append(uint8_to_percentile(seg.feature_percentiles))
            elif skip_feature_mags_active:
                # PR-E mags-as-zeros REQUIRES percentiles: with real magnitudes skipped, a segment
                # lacking a percentile overlay would silently fall back to zeros and diverge from
                # the sequential reference. Fail loud rather than write a wrong artifact (issue
                # #740 Codex review). A missing/short overlay ⇒ FeatureIndex omits the field here.
                raise SpliceProvenanceError(
                    f"{shard_name}: skip_feature_mags is active (cluster consolidation) but a "
                    f"segment of derived doc {derived_stem}:{doc_id} has NO percentile overlay; "
                    "the mags-as-zeros path would substitute zeros and diverge from the reference. "
                    "Restore full percentile-overlay coverage (re-run token-boundary/percentile "
                    "precompute) or disable feature clustering for this run."
                )
            else:
                decoded_pcts.append(None)

        seg_refs, expected_positions = resolve_counterfactual_refs(
            splice_index,
            feature_index,
            derived_stem,
            doc_id,
            segments,
            any(p is not None for p in decoded_pcts),
        )
        segment_fids = [seg.feature_ids.to(t.int64) for seg in segments]
        segment_mags = [_seg_mags_or_zeros(seg) for seg in segments]
        if feature_clusters is not None and not feature_clusters.is_identity_after_mask:
            batch_results = compute_doc_clustered_counterfactual_deltas(
                segment_fids,
                segment_mags,
                decoded_pcts,
                seg_refs,
                feature_clusters,
                delta_bundle.percentile_table,
            )
        else:
            batch_results = compute_doc_counterfactual_deltas(
                segment_fids,
                segment_mags,
                decoded_pcts,
                seg_refs,
                delta_bundle.percentile_table.shape[0],
                delta_bundle.percentile_table,
            )
        total_sidecar_positions += len(expected_positions)
        total_delta_positions += validate_counterfactual_delta_results(
            shard_stem=derived_stem,
            doc_idx=doc_id,
            expected_positions=expected_positions,
            seg_refs=seg_refs,
            batch_results=batch_results,
        )

        for seg_idx in seg_indices:
            ordinal = int(seg_ordinal[seg_idx].item())
            raw_fids, raw_mags, raw_pcts = _validate_packed_segment_matches_feature_index(
                shard_name=shard_name,
                seg_idx=seg_idx,
                packed=packed,
                packed_feat_offsets=packed_feat_offsets,
                segments=segments,
                seg_ordinal=ordinal,
            )
            delta_result = batch_results[ordinal]
            if legacy_segment_selection:
                assert sel_by_seg is not None
                result = _materialize_splice_segment_selection(
                    raw_fids=raw_fids,
                    raw_mags=raw_mags,
                    raw_pcts=raw_pcts,
                    delta_result=delta_result,
                    pipeline_config=pipeline,
                    feature_clusters=feature_clusters,
                )
                assert isinstance(result[0], list)
                sel_by_seg[seg_idx] = cast(
                    tuple[list[int], list[int], list[int], list[int]],
                    result,
                )
                continue

            assert tensor_builder is not None
            if delta_result is not None:
                # Preserve each delta row's specialized negative sampler and
                # its position in both shard-local RNG streams. Only ordinary
                # executions between delta rows are grouped.
                flush_pending_ordinary()
                serial_delta_segment_count += 1
                tensor_result = _materialize_splice_segment_selection(
                    raw_fids=raw_fids,
                    raw_mags=raw_mags,
                    raw_pcts=raw_pcts,
                    delta_result=delta_result,
                    pipeline_config=pipeline,
                    feature_clusters=feature_clusters,
                    tensor_output=True,
                )
                assert isinstance(tensor_result[0], t.Tensor)
                tensor_builder.add(
                    _tensor_selection_batch(
                        [seg_idx],
                        [
                            cast(
                                tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor],
                                tensor_result,
                            )
                        ],
                    )
                )
                continue

            # sample_features_batched has one optional precomputed-percentile
            # tensor for the complete batch. Split at a representation change
            # so a mixed batch can never silently ignore one row's overlay.
            if pending_ordinary and (pending_ordinary[-1].raw_pcts is None) != (raw_pcts is None):
                flush_pending_ordinary()
            pending_ordinary.append(
                _OrdinarySpliceSelectionInput(
                    segment_index=seg_idx,
                    raw_fids=raw_fids,
                    raw_mags=raw_mags,
                    raw_pcts=raw_pcts,
                )
            )
            if len(pending_ordinary) == splice_batch_size:
                flush_pending_ordinary()

    flush_pending_ordinary()
    if not legacy_segment_selection and (
        ordinary_segment_count + serial_delta_segment_count != packed.total_segments
    ):
        raise SpliceProvenanceError(
            f"{shard_name}: batched selection accounting covers "
            f"{ordinary_segment_count} ordinary + "
            f"{serial_delta_segment_count} delta segments, expected "
            f"{packed.total_segments}"
        )

    selection_done = time.monotonic()
    if legacy_segment_selection:
        assert sel_by_seg is not None
        sel_fids_flat: list[int] = []
        sel_buckets_flat: list[int] = []
        sel_pcts_flat: list[int] = []
        sel_counts: list[int] = []
        neg_fids_flat: list[int] = []
        neg_counts: list[int] = []
        for seg_idx, result in enumerate(sel_by_seg):
            if result is None:
                raise SpliceProvenanceError(
                    f"{shard_name}: no selection materialized for packed segment "
                    f"{seg_idx}; corrupt packed provenance."
                )
            seg_sel_fids, seg_buckets, seg_pcts, seg_neg_fids = result
            sel_fids_flat.extend(seg_sel_fids)
            sel_buckets_flat.extend(seg_buckets)
            sel_pcts_flat.extend(seg_pcts)
            sel_counts.append(len(seg_sel_fids))
            neg_fids_flat.extend(seg_neg_fids)
            neg_counts.append(len(seg_neg_fids))

        sel_offsets = [0]
        for count in sel_counts:
            sel_offsets.append(sel_offsets[-1] + count)
        neg_offsets = [0]
        for count in neg_counts:
            neg_offsets.append(neg_offsets[-1] + count)

        sel_shard = SelectionShard(
            sel_offsets=t.tensor(sel_offsets, dtype=t.int32),
            sel_fids=(
                t.tensor(sel_fids_flat, dtype=t.uint16)
                if sel_fids_flat
                else t.tensor([], dtype=t.uint16)
            ),
            sel_buckets=(
                t.tensor(sel_buckets_flat, dtype=t.uint8)
                if sel_buckets_flat
                else t.tensor([], dtype=t.uint8)
            ),
            sel_pcts=(
                t.tensor(sel_pcts_flat, dtype=t.uint8)
                if sel_pcts_flat
                else t.tensor([], dtype=t.uint8)
            ),
            neg_offsets=t.tensor(neg_offsets, dtype=t.int32),
            neg_fids=(
                t.tensor(neg_fids_flat, dtype=t.uint16)
                if neg_fids_flat
                else t.tensor([], dtype=t.uint16)
            ),
        )
    else:
        assert tensor_builder is not None
        sel_shard = tensor_builder.build()
    out_path = overlay_shard_path(output_dir, shard_name)
    save_selection_shard(sel_shard, out_path)
    save_done = time.monotonic()
    profile = {
        "schema_version": 1,
        "event": "shard",
        "shard": shard_name,
        "packed_segments": packed.total_segments,
        "source_doc_groups": len(groups),
        "distinct_source_stems": distinct_source_stems,
        "source_stem_transitions": source_stem_transitions,
        "source_stem_runs": source_stem_runs,
        "packed_storage": {
            "mode": packed_handle.storage_mode,
            "compressed_bytes": packed_handle.compressed_bytes,
            "tensor_bytes": packed_handle.tensor_bytes,
            "spool_bytes": packed_handle.spool_bytes,
        },
        "stages_seconds": {
            "packed_and_provenance_load": load_done - t0,
            "grouping": grouping_done - load_done,
            "selection": selection_done - grouping_done,
            "flatten_and_save": save_done - selection_done,
            "total": save_done - t0,
        },
        "feature_cache_delta": feature_index.cache_stats.delta(feature_cache_before),
        "sidecar_cache_delta": splice_index.cache_stats.delta(sidecar_cache_before),
    }
    if not legacy_segment_selection:
        profile["selection_batching"] = {
            "algorithm": SPLICE_SELECTION_BATCHING_ALGORITHM,
            "configured_batch_size": splice_batch_size,
            "ordinary_batches": ordinary_batch_count,
            "ordinary_segments": ordinary_segment_count,
            "mean_ordinary_batch_size": (
                ordinary_segment_count / ordinary_batch_count if ordinary_batch_count else 0.0
            ),
            "maximum_ordinary_batch_size": maximum_ordinary_batch_size,
            "serial_delta_segments": serial_delta_segment_count,
            "tensor_native_csr": True,
        }
    logger.info(
        "SPLICE_RESOURCE_PROFILE " + json.dumps(profile, sort_keys=True, separators=(",", ":"))
    )
    logger.info(
        f"  {shard_name}: splice_counterfactual {packed.total_segments} segs, "
        f"{sel_shard.total_selected} sel, {sel_shard.total_negative} neg "
        f"({save_done - t0:.1f}s)"
    )
    return SpliceSelectionStats(
        total_packed_segments=packed.total_segments,
        total_sidecar_splice_positions_referenced=total_sidecar_positions,
        total_delta_positions_selected=total_delta_positions,
    )


# --- Parallel splice-selection workers (issue #740 PR-D) ---------------------
# A packed shard's selection output is a pure function of (shard data, RNG seeded
# from shard_name, config) and is written to its own file, so processing shards
# across a process pool is BYTE-IDENTICAL to the sequential loop (there is no
# cross-shard state on the splice path: RNG is per shard_name, stats aggregate by
# commutative addition, and replay_skipped_for_rng is gated to the non-splice
# path). "spawn" is used, not "fork": the parent has already run torch CPU ops
# (percentile table, cluster build) so fork-after-threads could deadlock — each
# worker instead rebuilds its OWN pipeline + index caches from the picklable cfg
# (mutable caches can't be shared across processes). The parent still does the
# one-time startup (validate_packed_provenance + fingerprints + manifest); workers
# only process shards. Default selection_num_workers=1 keeps the single-process
# path unchanged.
_SPLICE_WORKER_STATE: dict[str, Any] = {}


def _init_splice_worker(payload: dict[str, Any]) -> None:
    """Rebuild this worker's pipeline + per-worker index caches exactly once."""
    init_started = time.monotonic()
    # A spawn worker that dies on a C-level fault (segfault/abort) otherwise surfaces only as an
    # opaque BrokenProcessPool with no location. faulthandler dumps the faulting C+Python traceback
    # to stderr (→ the run log) so such crashes are diagnosable (issue #740 shared-store bring-up).
    faulthandler.enable()
    store_dir = payload.get("shared_store_dir")
    store_lease = register_worker_store_lease(Path(store_dir)) if store_dir else None
    lease_done = time.monotonic()
    t.set_num_threads(max(1, int(payload["torch_threads"])))
    cfg = OmegaConf.create(payload["cfg"])
    feature_dir = Path(payload["feature_dir"])
    stats_path = find_statistics_path(feature_dir)
    assert stats_path is not None, f"no statistics under {feature_dir}"
    feature_stats = load_statistics(stats_path)
    n_sae_features = feature_stats.metadata.n_sae_features
    feature_mask = resolve_feature_mask(cfg, n_sae_features)
    resolved_clusters = resolve_feature_clusters(cfg, n_sae_features)
    feature_clusters = (
        FeatureClusterConsolidation.build(resolved_clusters, feature_mask)
        if resolved_clusters is not None
        else None
    )
    feature_delta_raw = (
        cast(dict[str, Any], OmegaConf.to_container(cfg.feature_delta, resolve=True))
        if cfg.get("feature_delta") is not None
        else None
    )
    pipeline = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=feature_stats.per_feature_stats,
        n_sae_features=n_sae_features,
        num_buckets=cfg.num_buckets,
        seed=cfg.get("seed", 42),
        feature_delta=parse_feature_delta(feature_delta_raw),
        feature_mask=feature_mask,
    )
    pipeline_done = time.monotonic()
    # PR-F: optional shared /dev/shm source store — every worker maps one physical copy per
    # stem instead of holding its own, so per-worker RAM stops bounding the worker count.
    shared_store: SharedStemStore | None = None
    if store_dir:
        shared_store = SharedStemStore(
            store_dir,
            capacity=int(payload["shared_store_capacity"]),
            min_free_bytes=int(payload.get("shared_store_min_free_bytes", 0)),
        )
    remote_source_inventory = payload.get("remote_source_inventory")
    if remote_source_inventory is None:
        feature_index_catalog = cast(FeatureIndexCatalog, payload["feature_index_catalog"])
    else:
        feature_index_catalog = build_transport_feature_catalog(
            cast(Mapping[str, object], remote_source_inventory),
            staging_dir=Path(payload["remote_source_staging_root"]),
            segment_size=int(payload["segment_size"]),
        )
    expected_feature_fingerprint = cast(str, payload["splice_derived_feature_hash"])
    if feature_index_catalog.validated_content_fingerprint != expected_feature_fingerprint:
        raise FeatureIndexCatalogError(
            "spawn-worker FeatureIndex catalog fingerprint does not match the "
            "parent's validated derived-feature fingerprint"
        )
    max_resident_shards = 1 if shared_store is not None else int(payload["cache_shards"])
    skip_feature_mags = feature_clusters is not None and not feature_clusters.is_identity_after_mask
    if isinstance(feature_index_catalog, TransportFeatureIndexCatalog):
        backend_factory = cast(
            Callable[[str], BucketBackend] | None,
            payload.get("remote_worker_backend_factory"),
        )
        materializer = RemoteFeatureShardMaterializer(
            bucket_id=payload["remote_bucket_id"],
            inventory=cast(Mapping[str, object], remote_source_inventory),
            catalog=feature_index_catalog,
            retry_timeout_seconds=float(payload["remote_retry_timeout_seconds"]),
            retry_interval_seconds=float(payload["remote_retry_interval_seconds"]),
            backend=(
                backend_factory(str(payload["remote_bucket_id"]))
                if backend_factory is not None
                else None
            ),
        )
        feature_index = FeatureIndex.from_transport_catalog(
            feature_index_catalog,
            materializer=materializer,
            max_resident_shards=max_resident_shards,
            skip_feature_mags=skip_feature_mags,
            compact_feature_ids=True,
            shared_store=shared_store,
            lazy_shared_docs=shared_store is not None,
        )
    else:
        feature_index = FeatureIndex.from_catalog(
            feature_index_catalog,
            # With a shared store, one mapped source stem per worker bounds the
            # steady local cache. The store-published compact document locator
            # makes LRU-1 switches cheap.
            max_resident_shards=max_resident_shards,
            skip_feature_mags=skip_feature_mags,
            compact_feature_ids=True,
            shared_store=shared_store,
            lazy_shared_docs=shared_store is not None,
        )
    splice_index = SpliceProvenanceIndex(
        feature_dir,
        max_resident_shards=int(payload["sidecar_shards"]),
        compact=True,
    )
    indices_done = time.monotonic()
    _SPLICE_WORKER_STATE.clear()
    _SPLICE_WORKER_STATE.update(
        pipeline=pipeline,
        feature_clusters=feature_clusters,
        feature_index=feature_index,
        splice_index=splice_index,
        packed_info=payload["packed_info"],
        output_dir=Path(payload["output_dir"]),
        provenance_dir=Path(payload["provenance_dir"]),
        seed=int(payload["seed"]),
        existing_stems=frozenset(payload["existing_stems"]),
        packed_spool_dir=payload.get("packed_spool_dir"),
        splice_batch_size=resolve_splice_selection_batch_size(
            cfg.get("selection_splice_batch_size")
        ),
        packed_fingerprint_files={
            entry.name: entry
            for entry in cast(
                tuple[PackedFingerprintFile, ...],
                payload["packed_fingerprint_files"],
            )
        },
        store_lease=store_lease,
    )
    state_done = time.monotonic()
    logger.info(
        "SPLICE_RESOURCE_PROFILE "
        + json.dumps(
            {
                "schema_version": 1,
                "event": "worker_initialization",
                "pid": os.getpid(),
                "shared_store": shared_store is not None,
                "stages_seconds": {
                    "lease_registration": lease_done - init_started,
                    "runtime_pipeline": pipeline_done - lease_done,
                    "catalog_and_indices": indices_done - pipeline_done,
                    "state_publish": state_done - indices_done,
                    "total": state_done - init_started,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _process_splice_shard_in_worker(
    shard_path_str: str,
) -> tuple[str, bool, SpliceSelectionStats]:
    """Process one packed shard (or stat-only if already materialized).

    Returns ``(shard_name, was_written, stats)``. Mirrors the sequential loop
    body's splice branch exactly, using this worker's own caches."""
    st = _SPLICE_WORKER_STATE
    shard_path = Path(shard_path_str)
    shard_name = shard_path.stem.split(".")[0]
    prov_path = st["provenance_dir"] / shard_path.name
    packed_files = cast(
        dict[str, PackedFingerprintFile],
        st["packed_fingerprint_files"],
    )
    packed_file = packed_files.get(shard_path.name)
    if st["feature_clusters"] is not None and packed_file is None:
        raise FeatureClustersError(
            f"spawn worker has no strict packed-row proof for {shard_path.name!r}"
        )
    if packed_file is not None:
        assert_packed_row_fingerprint_file_current(packed_file, shard_path)
    if shard_name in st["existing_stems"]:
        stats = splice_counterfactual_stats_for_packed_shard(
            shard_path,
            provenance_path=prov_path,
            packed_info=st["packed_info"],
            splice_index=st["splice_index"],
            packed_spool_dir=st["packed_spool_dir"],
        )
        if packed_file is not None:
            assert_packed_row_fingerprint_file_current(packed_file, shard_path)
        return shard_name, False, stats
    stats = select_splice_counterfactual_features_for_shard(
        shard_path,
        st["pipeline"],
        st["output_dir"],
        shard_name,
        provenance_path=prov_path,
        packed_info=st["packed_info"],
        feature_index=st["feature_index"],
        splice_index=st["splice_index"],
        seed=st["seed"],
        feature_clusters=st["feature_clusters"],
        packed_spool_dir=st["packed_spool_dir"],
        splice_batch_size=st["splice_batch_size"],
    )
    if packed_file is not None:
        assert_packed_row_fingerprint_file_current(packed_file, shard_path)
    return shard_name, True, stats


def _process_remote_splice_shard_in_worker(
    task: tuple[str, RegularFileIdentity],
) -> tuple[str, SpliceSelectionStats]:
    """Run the existing selector against one parent-staged packed body."""
    st = _SPLICE_WORKER_STATE
    shard_path = Path(task[0])
    expected_identity = task[1]
    if RegularFileIdentity.capture(shard_path) != expected_identity:
        raise RemotePreselectionError(
            f"remote staged packed body changed before worker selection: {shard_path}"
        )
    shard_name = shard_path.name.removesuffix(_SELECTION_SHARD_SUFFIX)
    stats = select_splice_counterfactual_features_for_shard(
        shard_path,
        st["pipeline"],
        st["output_dir"],
        shard_name,
        provenance_path=st["provenance_dir"] / shard_path.name,
        packed_info=st["packed_info"],
        feature_index=st["feature_index"],
        splice_index=st["splice_index"],
        seed=st["seed"],
        feature_clusters=st["feature_clusters"],
        packed_spool_dir=st["packed_spool_dir"],
        splice_batch_size=st["splice_batch_size"],
    )
    if RegularFileIdentity.capture(shard_path) != expected_identity:
        raise RemotePreselectionError(
            f"remote staged packed body changed during worker selection: {shard_path}"
        )
    return shard_name, stats


def _run_remote_splice_shards(
    *,
    entries: Sequence[Mapping[str, object]],
    cfg: DictConfig,
    feature_dir: Path,
    source_inventory: Mapping[str, object],
    source_staging_root: Path,
    packed_staging_dir: Path,
    output_staging_dir: Path,
    packed_prefix: str,
    bucket_id: str,
    backend,
    packed_info: PackedProvenanceInfo,
    splice_derived_feature_hash: str,
    cache_shards: int,
    sidecar_shards: int,
    seed: int,
    num_workers: int,
    retry_timeout_seconds: float,
    retry_interval_seconds: float,
    packed_spool_dir: str | None,
    shared_store_root: str | None,
    shared_store_capacity: int,
    on_completed_shard: Callable[[str, SpliceSelectionStats], None],
    remote_worker_backend_factory: Callable[[str], BucketBackend] | None = None,
) -> None:
    """Bound packed transport around the unchanged splice selector."""
    if not entries:
        return
    active_workers = min(num_workers, len(entries))
    n_cpu = os.process_cpu_count() or os.cpu_count() or active_workers
    payload: dict[str, Any] = {
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "feature_dir": str(feature_dir),
        "remote_source_inventory": dict(source_inventory),
        "remote_source_staging_root": str(source_staging_root),
        "remote_bucket_id": bucket_id,
        "remote_retry_timeout_seconds": retry_timeout_seconds,
        "remote_retry_interval_seconds": retry_interval_seconds,
        "segment_size": int(cfg.segment_size),
        "splice_derived_feature_hash": splice_derived_feature_hash,
        "cache_shards": cache_shards,
        "sidecar_shards": sidecar_shards,
        "packed_info": packed_info,
        "packed_fingerprint_files": (),
        "output_dir": str(output_staging_dir),
        "provenance_dir": str(Path(packed_info.provenance_shard_paths[0]).parent),
        "seed": seed,
        "existing_stems": (),
        "torch_threads": max(1, n_cpu // active_workers),
        "packed_spool_dir": packed_spool_dir,
    }
    if remote_worker_backend_factory is not None:
        payload["remote_worker_backend_factory"] = remote_worker_backend_factory
    store_lifecycle: ParentStoreLifecycle | None = None
    if shared_store_root is not None:
        _sweep_orphaned_shm_stores(shared_store_root)
        shared_store_dir = tempfile.mkdtemp(
            prefix=f"nanochat-sel-{os.getpid()}-", dir=shared_store_root
        )
        try:
            store_lifecycle = ParentStoreLifecycle.start(
                Path(shared_store_dir), Path(shared_store_root)
            )
        except BaseException:
            shutil.rmtree(shared_store_dir)
            raise
        payload["shared_store_dir"] = shared_store_dir
        payload["shared_store_capacity"] = shared_store_capacity

    def _stage(
        entry: Mapping[str, object],
    ) -> tuple[AbstractContextManager[MaterializedPackedShard], MaterializedPackedShard]:
        context = materialize_remote_packed_shard(
            backend,
            packed_prefix=packed_prefix,
            entry=entry,
            staging_dir=packed_staging_dir,
            timeout_seconds=retry_timeout_seconds,
            retry_interval_seconds=retry_interval_seconds,
        )
        return context, context.__enter__()

    primary_error: BaseException | None = None
    process_pool: ProcessPoolExecutor | None = None
    staged: list[
        tuple[
            AbstractContextManager[MaterializedPackedShard],
            MaterializedPackedShard,
            Future[tuple[str, SpliceSelectionStats]] | None,
        ]
    ] = []
    download_future: (
        Future[
            tuple[
                AbstractContextManager[MaterializedPackedShard],
                MaterializedPackedShard,
            ]
        ]
        | None
    ) = None
    try:
        if active_workers > 1:
            process_pool = ProcessPoolExecutor(
                max_workers=active_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_splice_worker,
                initargs=(payload,),
            )
        else:
            _init_splice_worker(payload)

        with ThreadPoolExecutor(max_workers=1) as downloader:
            position = 0
            download_future = downloader.submit(_stage, entries[position])
            position += 1

            def _submit_one() -> None:
                nonlocal download_future, position
                assert download_future is not None
                completed_download = download_future
                download_future = None
                context, materialized = completed_download.result()
                try:
                    worker_future = (
                        None
                        if process_pool is None
                        else process_pool.submit(
                            _process_remote_splice_shard_in_worker,
                            (str(materialized.path), materialized.identity),
                        )
                    )
                except BaseException:
                    context.__exit__(None, None, None)
                    raise
                staged.append((context, materialized, worker_future))
                if position < len(entries):
                    download_future = downloader.submit(_stage, entries[position])
                    position += 1

            while download_future is not None and len(staged) < active_workers:
                _submit_one()
            while staged:
                context, materialized, worker_future = staged[0]
                if worker_future is None:
                    shard_name, stats = _process_remote_splice_shard_in_worker(
                        (str(materialized.path), materialized.identity)
                    )
                else:
                    shard_name, stats = worker_future.result()
                if RegularFileIdentity.capture(materialized.path) != materialized.identity:
                    raise RemotePreselectionError(
                        f"staged packed body changed before parent commit: {materialized.path}"
                    )
                on_completed_shard(shard_name, stats)
                staged.pop(0)
                context.__exit__(None, None, None)
                if download_future is not None:
                    _submit_one()
        if process_pool is not None:
            process_pool.shutdown(wait=True)
            process_pool = None
    except BaseException as exc:
        primary_error = exc
        if store_lifecycle is not None and not store_lifecycle.closed:
            try:
                store_lifecycle.abort_after_worker_failure()
            except BaseException as cleanup_error:
                exc.add_note(f"shared-store abort also failed: {cleanup_error!r}")
        if process_pool is not None:
            process_pool.shutdown(wait=True, cancel_futures=True)
        raise
    finally:
        if download_future is not None and not download_future.cancel():
            try:
                pending_context, _pending_materialized = download_future.result()
                pending_context.__exit__(None, None, None)
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(f"packed lookahead cleanup also failed: {cleanup_error!r}")
        for context, _materialized, _future in staged:
            try:
                context.__exit__(None, None, None)
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(f"packed staging cleanup also failed: {cleanup_error!r}")
        if store_lifecycle is not None and not store_lifecycle.closed:
            try:
                store_lifecycle.cleanup_after_workers()
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(f"shared-store cleanup also failed: {cleanup_error!r}")


def _run_splice_shards_parallel(
    *,
    base_shard_paths: list[Path],
    existing_stems: set[str],
    cfg: DictConfig,
    feature_dir: Path,
    pct_checksum: str,
    feature_index_catalog: FeatureIndexCatalog,
    splice_derived_feature_hash: str,
    cache_shards: int,
    sidecar_shards: int,
    packed_info: PackedProvenanceInfo,
    packed_fingerprint_files: tuple[PackedFingerprintFile, ...],
    output_dir: Path,
    packed_row_dir: Path,
    seed: int,
    num_workers: int,
    packed_spool_dir: str | None = None,
    shared_store_root: str | None = None,
    shared_store_capacity: int = 0,
    on_completed_shard: Callable[[str, bool, SpliceSelectionStats], None] | None = None,
) -> SpliceSelectionStats:
    """Process the packed shards across ``num_workers`` spawn workers, byte-
    identically to the sequential loop. Aggregates the per-shard stats.

    When ``shared_store_dir`` is set, all workers share one /dev/shm copy per source stem
    (PR-F); the directory is removed on exit. Byte-identical either way (the store is a
    read-only cache in front of the same loader)."""
    # Divide the CPUs AVAILABLE TO THIS PROCESS (respects Slurm/cgroup affinity via
    # os.process_cpu_count, Python 3.13+) across the actually-concurrent workers so
    # torch intra-op threads don't oversubscribe the allocation.
    n_cpu = os.process_cpu_count() or os.cpu_count() or num_workers
    concurrent = min(num_workers, len(base_shard_paths)) or 1
    torch_threads = max(1, n_cpu // concurrent)
    if feature_index_catalog.validated_content_fingerprint != splice_derived_feature_hash:
        raise FeatureIndexCatalogError(
            "parallel FeatureIndex catalog fingerprint does not match the "
            "validated derived-feature fingerprint"
        )
    feature_index_catalog.validate()
    payload: dict[str, Any] = {
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "feature_dir": str(feature_dir),
        "pct_checksum": pct_checksum,
        "feature_index_catalog": feature_index_catalog,
        "splice_derived_feature_hash": splice_derived_feature_hash,
        "cache_shards": cache_shards,
        "sidecar_shards": sidecar_shards,
        "packed_info": packed_info,
        "packed_fingerprint_files": packed_fingerprint_files,
        "output_dir": str(output_dir),
        "provenance_dir": str(provenance_dir(packed_row_dir)),
        "seed": seed,
        "existing_stems": sorted(existing_stems),
        "torch_threads": torch_threads,
        "packed_spool_dir": packed_spool_dir,
    }
    shared_store_dir: str | None = None
    store_lifecycle: ParentStoreLifecycle | None = None
    if shared_store_root:
        _sweep_orphaned_shm_stores(shared_store_root)
        # Atomic EXCLUSIVE creation (mkdtemp): the OS guarantees a brand-new dir, so a crashed
        # run's stale store can never be reopened via a PID+suffix collision (issue #740 review).
        shared_store_dir = tempfile.mkdtemp(
            prefix=f"nanochat-sel-{os.getpid()}-", dir=shared_store_root
        )
        try:
            store_lifecycle = ParentStoreLifecycle.start(
                Path(shared_store_dir), Path(shared_store_root)
            )
        except BaseException:
            # No pool exists yet, so no worker can hold a mapping or lease.
            shutil.rmtree(shared_store_dir)
            raise
        payload["shared_store_dir"] = shared_store_dir
        payload["shared_store_capacity"] = shared_store_capacity
        logger.info(
            f"Shared source store: {shared_store_dir} (capacity {shared_store_capacity} "
            f"stems); workers map one copy per stem."
        )
    logger.info(
        f"Processing {len(base_shard_paths)} shards across {num_workers} spawn "
        f"worker(s), {torch_threads} torch thread(s) each..."
    )
    mp_ctx = multiprocessing.get_context("spawn")
    total = SpliceSelectionStats()
    shard_strs = [str(p) for p in base_shard_paths]
    primary_error: BaseException | None = None
    pool: ProcessPoolExecutor | None = None
    try:
        pool = ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp_ctx,
            initializer=_init_splice_worker,
            initargs=(payload,),
        )
        for shard_name, was_written, stats in pool.map(_process_splice_shard_in_worker, shard_strs):
            total += stats
            if on_completed_shard is not None:
                # A writing worker atomically published before returning. The
                # parent alone advances both byte identity and per-shard stats;
                # existing legacy shards also checkpoint their replayed stats.
                on_completed_shard(shard_name, was_written, stats)
            logger.info(f"  {shard_name}: {'written' if was_written else 'existing'} (worker)")
        pool.shutdown(wait=True)
        pool = None
    except BaseException as exc:
        primary_error = exc
        # ProcessPoolExecutor's context-manager exit waits for every already-
        # running task. A first-shard failure could therefore leave unrelated
        # multi-minute production tasks alive before store cleanup begins. The
        # janitor owns exact registered identities and terminates only this
        # store's workers before the executor is reaped.
        if store_lifecycle is not None and not store_lifecycle.closed:
            try:
                store_lifecycle.abort_after_worker_failure()
            except BaseException as cleanup_error:
                exc.add_note(
                    "shared-store worker-abort cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
                logger.exception(
                    "Shared-store worker-abort cleanup failed while handling a primary error"
                )
        if pool is not None:
            try:
                pool.shutdown(wait=True, cancel_futures=True)
            except BaseException as shutdown_error:
                exc.add_note(
                    "process-pool shutdown also failed: "
                    f"{type(shutdown_error).__name__}: {shutdown_error}"
                )
        try:
            _sweep_dead_atomic_selection_temps(
                output_dir,
                {path.stem.split(".")[0] for path in base_shard_paths},
            )
        except BaseException as temporary_cleanup_error:
            exc.add_note(
                "selection atomic-temporary cleanup also failed: "
                f"{type(temporary_cleanup_error).__name__}: "
                f"{temporary_cleanup_error}"
            )
        raise
    finally:
        if store_lifecycle is not None and not store_lifecycle.closed:
            try:
                store_lifecycle.cleanup_after_workers()
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    "shared-store cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
                logger.exception("Shared-store cleanup failed while handling a primary error")
    return total


def _sweep_orphaned_shm_stores(shm_root: str) -> None:
    """Reap only schema-valid dead-owner stores whose exclusive lease is free."""
    try:
        result = sweep_orphaned_stores(Path(shm_root))
    except OSError as exc:
        logger.warning(f"Could not inspect shared-store root {shm_root}: {exc}")
        return
    for path in result.removed:
        logger.info(f"Reclaimed orphaned shared source store {path}")
    for path in result.skipped_live:
        logger.debug(f"Preserving live shared source store {path}")
    for path in result.skipped_unsafe:
        logger.warning(
            f"Refusing to reap unverified or leased shared source store {path}; "
            "inspect its owner/lease state manually"
        )


def _remote_receipt_stats(receipt: Mapping[str, object]) -> SpliceSelectionStats:
    raw = cast(Mapping[str, object], receipt["splice_stats"])
    return SpliceSelectionStats(
        total_packed_segments=cast(int, raw["total_packed_segments"]),
        total_sidecar_splice_positions_referenced=cast(
            int, raw["total_sidecar_splice_positions_referenced"]
        ),
        total_delta_positions_selected=cast(int, raw["total_delta_positions_selected"]),
    )


def _selection_rng_state_sha256(pipeline: FeaturePipelineConfig) -> str:
    """Hash the exact shared sampling-generator state without making a draw."""
    selection = pipeline.selection_config
    if not isinstance(selection, SamplingBundle):
        raise RemotePreselectionError(
            "remote unspliced pre-selection requires a SamplingBundle RNG"
        )
    if pipeline.negative_bundle is not None and pipeline.negative_bundle.rng is not selection.rng:
        raise RemotePreselectionError(
            "remote unspliced positive and negative sampling must share one generator"
        )
    state = selection.rng.get_state()
    if state.device.type != "cpu" or state.dtype is not t.uint8 or state.ndim != 1:
        raise RemotePreselectionError(
            "remote unspliced selection RNG state must be one-dimensional CPU uint8"
        )
    return hashlib.sha256(state.contiguous().numpy().tobytes()).hexdigest()


def _stable_local_file_sha256(
    path: Path, *, surface: str
) -> tuple[RegularFileIdentity, str]:
    """Hash one direct local control under descriptor and pathname fences."""
    identity = RegularFileIdentity.capture(path)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RemotePreselectionError(f"{surface} requires os.O_NOFOLLOW")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    digest = hashlib.sha256()
    try:
        if RegularFileIdentity.from_stat(os.fstat(descriptor)) != identity:
            raise RemotePreselectionError(f"{surface} changed before hashing: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while block := handle.read(16 * 1024 * 1024):
                digest.update(block)
        if RegularFileIdentity.from_stat(os.fstat(descriptor)) != identity:
            raise RemotePreselectionError(f"{surface} changed while hashing: {path}")
    finally:
        os.close(descriptor)
    if RegularFileIdentity.capture(path) != identity:
        raise RemotePreselectionError(f"{surface} path changed while hashing: {path}")
    return identity, digest.hexdigest()


def _assert_remote_logical_selection_dir(
    output_dir: Path, *, remote_manifest_bytes: bytes | None
) -> None:
    """Allow only an absent/empty logical dir or the exact final manifest."""
    if not output_dir.exists():
        return
    try:
        directory_info = output_dir.lstat()
    except OSError as exc:
        raise RemotePreselectionError(
            f"cannot inspect logical local selection directory: {output_dir}"
        ) from exc
    if not stat.S_ISDIR(directory_info.st_mode):
        raise RemotePreselectionError(
            f"logical local selection path must be a direct directory: {output_dir}"
        )
    entries = list(output_dir.iterdir())
    if not entries:
        return
    if len(entries) != 1 or entries[0].name != MANIFEST_FILENAME:
        raise RemotePreselectionError(
            "remote pre-selection refuses local bodies, temporaries, or unknown files in "
            f"{output_dir}"
        )
    manifest_path = entries[0]
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RemotePreselectionError(
            f"logical local selection manifest must be a direct regular file: {manifest_path}"
        )
    if remote_manifest_bytes is None:
        raise RemotePreselectionError(
            "a local selection manifest exists without a final remote manifest"
        )
    if manifest_path.read_bytes() != remote_manifest_bytes:
        raise RemotePreselectionError(
            f"local selection manifest conflicts with the final remote manifest: {manifest_path}"
        )


def _run_remote_unspliced_shards(
    *,
    entries: Sequence[Mapping[str, object]],
    backend: BucketBackend,
    packed_prefix: str,
    packed_staging_dir: Path,
    output_staging_dir: Path,
    pipeline: FeaturePipelineConfig,
    feature_clusters: FeatureClusterConsolidation,
    replay_names: set[str],
    retry_timeout_seconds: float,
    retry_interval_seconds: float,
    on_selected_shard: Callable[[Mapping[str, object], Path, bool], None],
) -> None:
    """Serial packed transport around the unchanged ordinary selector."""
    if not entries:
        return

    def _stage(
        entry: Mapping[str, object],
    ) -> tuple[AbstractContextManager[MaterializedPackedShard], MaterializedPackedShard]:
        context = materialize_remote_packed_shard(
            backend,
            packed_prefix=packed_prefix,
            entry=entry,
            staging_dir=packed_staging_dir,
            timeout_seconds=retry_timeout_seconds,
            retry_interval_seconds=retry_interval_seconds,
        )
        return context, context.__enter__()

    download_future: Future[
        tuple[AbstractContextManager[MaterializedPackedShard], MaterializedPackedShard]
    ] | None = None
    active_context: AbstractContextManager[MaterializedPackedShard] | None = None
    primary_error: BaseException | None = None
    try:
        with ThreadPoolExecutor(max_workers=1) as downloader:
            current_index = 0
            download_future = downloader.submit(_stage, entries[current_index])
            while download_future is not None:
                completed = download_future
                download_future = None
                active_context, materialized = completed.result()
                entry = entries[current_index]
                current_index += 1
                if current_index < len(entries):
                    download_future = downloader.submit(_stage, entries[current_index])

                name = entry.get("name")
                if not isinstance(name, str) or materialized.path.name != name:
                    raise RemotePreselectionError(
                        "packed inventory name and staged packed filename disagree"
                    )
                if not name.endswith(_SELECTION_SHARD_SUFFIX):
                    raise RemotePreselectionError(
                        f"packed inventory name is not canonical: {name!r}"
                    )
                shard_stem = name.removesuffix(_SELECTION_SHARD_SUFFIX)
                if RegularFileIdentity.capture(materialized.path) != materialized.identity:
                    raise RemotePreselectionError(
                        f"staged packed body changed before selection: {materialized.path}"
                    )
                if any(output_staging_dir.iterdir()):
                    raise RemotePreselectionError(
                        "ordinary remote selection output staging was not empty before selection"
                    )
                written = select_features_for_shard(
                    materialized.path,
                    pipeline,
                    output_staging_dir,
                    shard_stem,
                    gradient_join=None,
                    feature_clusters=feature_clusters,
                )
                if RegularFileIdentity.capture(materialized.path) != materialized.identity:
                    raise RemotePreselectionError(
                        f"staged packed body changed during selection: {materialized.path}"
                    )
                output_paths = list(output_staging_dir.iterdir())
                selection_path = output_staging_dir / name
                if (
                    not written
                    or output_paths != [selection_path]
                    or not selection_path.is_file()
                    or selection_path.is_symlink()
                ):
                    raise RemotePreselectionError(
                        f"ordinary selector did not produce exactly {name!r}"
                    )
                on_selected_shard(entry, selection_path, name in replay_names)
                if RegularFileIdentity.capture(materialized.path) != materialized.identity:
                    raise RemotePreselectionError(
                        f"staged packed body changed before cleanup: {materialized.path}"
                    )
                if any(output_staging_dir.iterdir()):
                    raise RemotePreselectionError(
                        "ordinary remote selection output staging was not empty after commit"
                    )
                active_context.__exit__(None, None, None)
                active_context = None
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if active_context is not None:
            try:
                active_context.__exit__(None, None, None)
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    f"active packed staging cleanup also failed: {cleanup_error!r}"
                )
        if download_future is not None and not download_future.cancel():
            try:
                pending_context, _pending = download_future.result()
                pending_context.__exit__(None, None, None)
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    f"packed lookahead cleanup also failed: {cleanup_error!r}"
                )


def _remote_preselection_staging_estimate(
    *,
    packed_sizes: Sequence[int],
    packed_concurrency: int,
    source_pair_sizes: Sequence[int],
    source_concurrency: int,
    output_concurrency: int,
    max_output_shard_bytes: int,
    staging_free_bytes: int,
    has_selection_work: bool,
) -> dict[str, int]:
    """Pure staging arithmetic shared by the two remote transport modes."""
    if not has_selection_work:
        return {
            "staged_packed_compressed": 0,
            "staged_source_compressed": 0,
            "staged_selection_output_bound": 0,
            "staging_free_reserve": 0,
            "staging_required_with_reserve": 0,
        }
    packed = sum(sorted(packed_sizes, reverse=True)[:packed_concurrency])
    source = sum(sorted(source_pair_sizes, reverse=True)[:source_concurrency])
    output = output_concurrency * max_output_shard_bytes
    reserve = min(
        8 * 1024**3,
        max(512 * 1024**2, staging_free_bytes // 20),
    )
    return {
        "staged_packed_compressed": packed,
        "staged_source_compressed": source,
        "staged_selection_output_bound": output,
        "staging_free_reserve": reserve,
        "staging_required_with_reserve": packed + source + output + reserve,
    }


def _admit_remote_preselection_staging(
    *,
    config: RemotePreselectionConfig,
    packed_inventory: Mapping[str, object],
    source_inventory: Mapping[str, object],
    packed_info: PackedProvenanceInfo,
    feature_dir: Path,
    n_sae_features: int,
    skip_feature_mags: bool,
    active_workers: int,
    source_cache_shards: int,
    sidecar_cache_shards: int,
    shared_store_root: str | None,
    shared_store_capacity: int,
    packed_spool_dir: str | None,
    has_selection_work: bool,
) -> None:
    """Fail closed on the bounded remote transport's point-in-time envelope."""
    packed_entries = cast(list[dict[str, object]], packed_inventory["shards"])
    source_entries = cast(list[dict[str, object]], source_inventory["shards"])
    packed_layouts = [
        parse_semantic_layout(entry["layout"], surface=f"remote packed layout {entry['name']}")
        for entry in packed_entries
    ]
    provenance_by_name = {path.name: path for path in packed_info.provenance_shard_paths}
    provenance_layouts = [
        read_compressed_safetensors_layout(provenance_by_name[cast(str, entry["name"])])
        for entry in packed_entries
    ]
    source_layouts = (
        [
            (
                cast(str, entry["stem"]),
                parse_semantic_layout(
                    cast(Mapping[str, object], entry["raw"])["layout"],
                    surface=f"remote raw layout {entry['stem']}",
                ),
                parse_semantic_layout(
                    cast(Mapping[str, object], entry["percentile"])["layout"],
                    surface=f"remote percentile layout {entry['stem']}",
                ),
            )
            for entry in source_entries
        ]
        if has_selection_work
        else []
    )
    sidecar_sizes = (
        [
            (feature_dir / PROVENANCE_DIRNAME / f"{entry['stem']}.json").stat().st_size
            for entry in source_entries
        ]
        if has_selection_work
        else []
    )
    shared_store_enabled = has_selection_work and shared_store_root is not None
    packed_spool_enabled = has_selection_work and packed_spool_dir is not None
    capacities = capture_capacity_snapshot(
        shared_store_root=shared_store_root if shared_store_enabled else None,
        packed_spool_dir=packed_spool_dir if packed_spool_enabled else None,
    )
    base_report = evaluate_splice_resource_admission_from_layouts(
        packed_layouts=packed_layouts,
        provenance_layouts=provenance_layouts,
        source_layouts=source_layouts,
        sidecar_sizes=sidecar_sizes,
        n_sae_features=n_sae_features,
        skip_feature_mags=skip_feature_mags,
        requested_workers=active_workers,
        source_cache_shards=source_cache_shards,
        sidecar_cache_shards=sidecar_cache_shards,
        has_selection_work=has_selection_work,
        requires_packed_runtime=has_selection_work,
        requires_provenance_runtime=True,
        requires_source_runtime=True,
        shared_store_enabled=shared_store_enabled,
        shared_store_capacity=(shared_store_capacity if shared_store_enabled else 0),
        packed_spool_enabled=packed_spool_enabled,
        capacities=capacities,
        policy=AdmissionPolicy(),
    )

    packed_sizes = [cast(int, entry["byte_length"]) for entry in packed_entries]
    source_pair_sizes = [
            cast(int, cast(Mapping[str, object], entry["raw"])["byte_length"])
            + cast(
                int,
                cast(Mapping[str, object], entry["percentile"])["byte_length"],
            )
            for entry in source_entries
    ]
    packed_concurrency = min(len(packed_sizes), active_workers + 1)
    source_concurrency = min(len(source_pair_sizes), 1 if shared_store_enabled else active_workers)
    free = shutil.disk_usage(config.staging_dir).free
    staging = _remote_preselection_staging_estimate(
        packed_sizes=packed_sizes,
        packed_concurrency=packed_concurrency,
        source_pair_sizes=source_pair_sizes,
        source_concurrency=source_concurrency,
        output_concurrency=active_workers,
        max_output_shard_bytes=config.max_output_shard_bytes,
        staging_free_bytes=free,
        has_selection_work=has_selection_work,
    )
    staging_required = staging["staging_required_with_reserve"]

    violations = list(base_report.violations)
    if staging_required > free:
        violations.append(
            f"staging filesystem requires {staging_required} bytes including "
            f"reserve, but only {free} bytes are available"
        )
    report = base_report.to_dict()
    report["transport_mode"] = "remote_streaming"
    exact_bytes = cast(dict[str, int], report["exact_bytes"])
    exact_bytes.update(
        {
            "staged_packed_compressed": staging["staged_packed_compressed"],
            "staged_source_compressed": staging["staged_source_compressed"],
            "staged_selection_output_bound": staging["staged_selection_output_bound"],
        }
    )
    estimated_bytes = cast(dict[str, int], report["estimated_bytes"])
    estimated_bytes.update(
        {
            "staging_free_reserve": staging["staging_free_reserve"],
            "staging_required_with_reserve": staging_required,
        }
    )
    report_capacities = cast(dict[str, object], report["capacities"])
    report_capacities["staging_available_bytes"] = free
    report["decision"] = "reject" if violations else "admit"
    report["violations"] = violations
    logger.info(
        "SPLICE_RESOURCE_ADMISSION " + json.dumps(report, sort_keys=True, separators=(",", ":"))
    )
    if violations:
        raise RemotePreselectionError("; ".join(violations))


def _admit_remote_unspliced_preselection_staging(
    *,
    config: RemotePreselectionConfig,
    packed_inventory: Mapping[str, object],
    packed_info: PackedProvenanceInfo,
    n_sae_features: int,
    has_selection_work: bool,
) -> None:
    """Admit the serial packed-only ordinary transport before run staging."""
    packed_entries = cast(list[dict[str, object]], packed_inventory["shards"])
    packed_layouts = [
        parse_semantic_layout(entry["layout"], surface=f"remote packed layout {entry['name']}")
        for entry in packed_entries
    ]
    provenance_by_name = {path.name: path for path in packed_info.provenance_shard_paths}
    provenance_layouts = [
        read_compressed_safetensors_layout(provenance_by_name[cast(str, entry["name"])])
        for entry in packed_entries
    ]
    capacities = capture_capacity_snapshot(
        shared_store_root=None,
        packed_spool_dir=None,
    )
    base_report = evaluate_splice_resource_admission_from_layouts(
        packed_layouts=packed_layouts,
        provenance_layouts=provenance_layouts,
        source_layouts=[],
        sidecar_sizes=[],
        n_sae_features=n_sae_features,
        skip_feature_mags=False,
        requested_workers=1,
        source_cache_shards=1,
        sidecar_cache_shards=1,
        has_selection_work=has_selection_work,
        requires_packed_runtime=has_selection_work,
        requires_provenance_runtime=False,
        requires_source_runtime=False,
        shared_store_enabled=False,
        shared_store_capacity=0,
        packed_spool_enabled=False,
        capacities=capacities,
        policy=AdmissionPolicy(),
    )
    packed_sizes = [cast(int, entry["byte_length"]) for entry in packed_entries]
    free = shutil.disk_usage(config.staging_dir).free
    staging = _remote_preselection_staging_estimate(
        packed_sizes=packed_sizes,
        packed_concurrency=min(2, len(packed_sizes)),
        source_pair_sizes=[],
        source_concurrency=0,
        output_concurrency=1,
        max_output_shard_bytes=config.max_output_shard_bytes,
        staging_free_bytes=free,
        has_selection_work=has_selection_work,
    )
    staging_required = staging["staging_required_with_reserve"]
    violations = list(base_report.violations)
    if staging_required > free:
        violations.append(
            f"staging filesystem requires {staging_required} bytes including "
            f"reserve, but only {free} bytes are available"
        )
    report = base_report.to_dict()
    report["transport_mode"] = "remote_streaming_unspliced"
    exact_bytes = cast(dict[str, int], report["exact_bytes"])
    exact_bytes.update(
        {
            "staged_packed_compressed": staging["staged_packed_compressed"],
            "staged_source_compressed": 0,
            "staged_selection_output_bound": staging["staged_selection_output_bound"],
        }
    )
    estimated_bytes = cast(dict[str, int], report["estimated_bytes"])
    estimated_bytes.update(
        {
            "staging_free_reserve": staging["staging_free_reserve"],
            "staging_required_with_reserve": staging_required,
        }
    )
    report_capacities = cast(dict[str, object], report["capacities"])
    report_capacities["staging_available_bytes"] = free
    report["decision"] = "reject" if violations else "admit"
    report["violations"] = violations
    logger.info(
        "SPLICE_RESOURCE_ADMISSION "
        + json.dumps(report, sort_keys=True, separators=(",", ":"))
    )
    if violations:
        raise RemotePreselectionError("; ".join(violations))


def _build_remote_splice_selection_manifest(
    *,
    directory: Path,
    selection_hash: str,
    packed_inventory: Mapping[str, object],
    receipts: Mapping[str, Mapping[str, object]],
    pipeline: FeaturePipelineConfig,
    packed_info: PackedProvenanceInfo,
    splice_sidecar_hash: str,
    splice_derived_feature_hash: str,
    splice_boundary_hash: str,
    feature_mask,
    feature_clusters: FeatureClusterConsolidation,
    splice_batch_size: int,
) -> Path:
    assert pipeline.delta_bundle is not None
    selection_integrity = selection_integrity_from_receipts(receipts)
    stats_by_stem: dict[str, SpliceSelectionStats] = {}
    widths_by_stem: dict[str, int] = {}
    total_stats = SpliceSelectionStats()
    for name in sorted(receipts):
        stem = name.removesuffix(_SELECTION_SHARD_SUFFIX)
        receipt = receipts[name]
        stats = _remote_receipt_stats(receipt)
        stats_by_stem[stem] = stats
        total_stats += stats
        widths_by_stem[stem] = cast(int, receipt["max_features_per_segment"])
    max_width = max(widths_by_stem.values(), default=0)
    storage = {
        name.removesuffix(_SELECTION_SHARD_SUFFIX): cast(
            str,
            cast(Mapping[str, object], receipts[name]["selection_body"])["content_sha256"],
        )
        for name in sorted(receipts)
    }
    metadata = _splice_selection_metadata(
        pipeline=pipeline,
        packed_info=packed_info,
        splice_stats=total_stats,
        splice_sidecar_hash=splice_sidecar_hash,
        splice_derived_feature_hash=splice_derived_feature_hash,
        splice_boundary_hash=splice_boundary_hash,
        feature_mask=feature_mask,
        feature_clusters=feature_clusters,
        grad_spec=None,
        grad_overlay_checksum=None,
        status="complete",
        splice_batch_size=splice_batch_size,
    )
    metadata[SELECTION_SHARD_INTEGRITY_KEY] = selection_integrity
    metadata[SPLICE_RESUME_METADATA_KEY] = _build_splice_resume_metadata(
        stats_by_stem=stats_by_stem,
        selection_integrity=selection_integrity,
        storage_validated_by_stem=storage,
        validated_max_features_per_segment=max_width,
        packed_rows_hash=packed_info.packed_row_fingerprint,
        effective_buckets=pipeline.effective_buckets,
        status="complete",
    )
    metadata[REMOTE_LOADER_KEY] = build_remote_loader_certificate(
        selection_checksum=selection_hash,
        selection_inventory_aggregate=cast(str, selection_integrity["aggregate_sha256"]),
        packed_row_fingerprint=packed_info.packed_row_fingerprint,
        max_features_per_segment=max_width,
        cluster_delta_semantics=CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS,
    )
    packed_entries = cast(list[dict[str, object]], packed_inventory["shards"])
    packed_by_stem = {
        cast(str, entry["name"]).removesuffix(_SELECTION_SHARD_SUFFIX): cast(
            str, entry["file_fingerprint"]
        )
        for entry in packed_entries
    }
    segment_counts = {
        cast(str, entry["name"]).removesuffix(_SELECTION_SHARD_SUFFIX): cast(
            int, entry["total_segments"]
        )
        for entry in packed_entries
    }
    metadata[CLUSTERED_SELECTION_VALIDATION_KEY] = build_clustered_selection_validation(
        selection_checksum=selection_hash,
        selection_integrity=selection_integrity,
        packed_row_fingerprint=packed_info.packed_row_fingerprint,
        packed_content_by_name=packed_by_stem,
        feature_clusters_descriptor=feature_clusters.resolved.descriptor_dict(),
        feature_delta=canonical_feature_delta_dict(pipeline.delta_bundle.config),
        effective_buckets=pipeline.effective_buckets,
        segment_counts_by_stem=segment_counts,
        max_features_by_stem=widths_by_stem,
    )
    resolve_clustered_selection_validation(
        metadata,
        selection_checksum=selection_hash,
        selection_integrity=selection_integrity,
        packed_row_fingerprint=packed_info.packed_row_fingerprint,
        feature_clusters_descriptor=feature_clusters.resolved.descriptor_dict(),
        feature_delta=canonical_feature_delta_dict(pipeline.delta_bundle.config),
        effective_buckets=pipeline.effective_buckets,
        surface="remote pre-selection completed manifest",
    )
    return write_manifest(
        directory,
        selection_hash,
        OverlayKind.PACKED_ROW,
        metadata=metadata,
        precomputed_base_hash=cast(str, packed_inventory["base_shard_content_hash"]),
        include_created_at=False,
    )


def _build_remote_unspliced_selection_manifest(
    *,
    directory: Path,
    selection_hash: str,
    packed_inventory: Mapping[str, object],
    receipts: Mapping[str, Mapping[str, object]],
    pipeline: FeaturePipelineConfig,
    packed_info: PackedProvenanceInfo,
    feature_mask,
    feature_clusters: FeatureClusterConsolidation,
) -> Path:
    """Build the ordinary timestamp-free manifest from packed v2 receipts."""
    if pipeline.delta_bundle is not None:
        raise RemotePreselectionError(
            "ordinary remote manifest construction requires feature_delta=null"
        )
    selection_integrity = selection_integrity_from_unspliced_receipts(receipts)
    widths_by_stem: dict[str, int] = {}
    for name in sorted(receipts):
        receipt = receipts[name]
        widths_by_stem[name.removesuffix(_SELECTION_SHARD_SUFFIX)] = cast(
            int, receipt["max_features_per_segment"]
        )
    max_width = max(widths_by_stem.values(), default=0)
    metadata = _base_selection_metadata(
        feature_mask=feature_mask,
        feature_clusters=feature_clusters,
        grad_spec=None,
        grad_overlay_checksum=None,
        packed_rows_hash=packed_info.packed_row_fingerprint,
        status="complete",
    )
    metadata[SELECTION_SHARD_INTEGRITY_KEY] = selection_integrity
    metadata[REMOTE_LOADER_KEY] = build_remote_loader_certificate(
        selection_checksum=selection_hash,
        selection_inventory_aggregate=cast(str, selection_integrity["aggregate_sha256"]),
        packed_row_fingerprint=packed_info.packed_row_fingerprint,
        max_features_per_segment=max_width,
        cluster_delta_semantics=None,
    )
    packed_entries = cast(list[dict[str, object]], packed_inventory["shards"])
    packed_by_stem = {
        cast(str, entry["name"]).removesuffix(_SELECTION_SHARD_SUFFIX): cast(
            str, entry["file_fingerprint"]
        )
        for entry in packed_entries
    }
    segment_counts = {
        cast(str, entry["name"]).removesuffix(_SELECTION_SHARD_SUFFIX): cast(
            int, entry["total_segments"]
        )
        for entry in packed_entries
    }
    metadata[CLUSTERED_SELECTION_VALIDATION_KEY] = build_clustered_selection_validation(
        selection_checksum=selection_hash,
        selection_integrity=selection_integrity,
        packed_row_fingerprint=packed_info.packed_row_fingerprint,
        packed_content_by_name=packed_by_stem,
        feature_clusters_descriptor=feature_clusters.resolved.descriptor_dict(),
        feature_delta=None,
        effective_buckets=pipeline.effective_buckets,
        segment_counts_by_stem=segment_counts,
        max_features_by_stem=widths_by_stem,
    )
    resolve_clustered_selection_validation(
        metadata,
        selection_checksum=selection_hash,
        selection_integrity=selection_integrity,
        packed_row_fingerprint=packed_info.packed_row_fingerprint,
        feature_clusters_descriptor=feature_clusters.resolved.descriptor_dict(),
        feature_delta=None,
        effective_buckets=pipeline.effective_buckets,
        surface="remote ordinary pre-selection completed manifest",
    )
    return write_manifest(
        directory,
        selection_hash,
        OverlayKind.PACKED_ROW,
        metadata=metadata,
        precomputed_base_hash=cast(str, packed_inventory["base_shard_content_hash"]),
        include_created_at=False,
    )


def _run_remote_splice_preselection(
    *,
    cfg: DictConfig,
    remote_config: RemotePreselectionConfig,
    remote_training: RemoteTrainingConfig,
    feature_dir: Path,
    packed_row_dir: Path,
    output_dir: Path,
    selection_hash: str,
    pipeline: FeaturePipelineConfig,
    feature_mask,
    feature_clusters: FeatureClusterConsolidation,
    n_sae_features: int,
    selection_num_workers: int,
    selection_source_cache_shards: int,
    selection_sidecar_cache_shards: int,
    selection_splice_batch_size: int,
    selection_shared_store: bool,
    selection_shared_store_capacity: int,
    selection_packed_spool_dir: str | None,
    seed: int,
) -> None:
    """Produce clustered splice selection bodies through remote transport."""
    if remote_training.train_packed is None:
        raise RemotePreselectionError("remote_preselection requires remote_training.train_packed")
    packed_manifest = read_manifest(packed_row_dir)
    packed_metadata = (packed_manifest or {}).get("metadata")
    if not isinstance(packed_metadata, Mapping):
        raise RemotePreselectionError("local packed manifest metadata is invalid")
    pct_checksum = packed_metadata.get("pct_checksum")
    if not isinstance(pct_checksum, str) or not _is_lower_sha256(pct_checksum):
        raise RemotePreselectionError(
            "remote pre-selection requires a packed manifest percentile checksum"
        )
    prefixes = resolve_remote_preselection_prefixes(
        remote=remote_training,
        feature_root=cfg.feature_root,
        feature_dir=feature_dir,
        packed_row_dir=packed_row_dir,
        selection_checksum=selection_hash,
    )

    shared_store_root: str | None = None
    shared_store_capacity = 0
    if selection_num_workers > 1 and selection_shared_store:
        shared_store_root = os.environ.get("NANOCHAT_SHM_DIR", "/dev/shm")
        if not Path(shared_store_root).is_dir():
            raise RemotePreselectionError(
                f"selection shared-store root is absent: {shared_store_root}"
            )
        shared_store_capacity = (
            selection_shared_store_capacity
            if selection_shared_store_capacity > 0
            else selection_num_workers + 4
        )
    remote_config.assert_disjoint(
        feature_dir,
        packed_row_dir,
        output_dir,
        selection_packed_spool_dir,
        shared_store_root,
    )
    backend = HfBucketBackend(remote_training.bucket_id)
    timeout = float(remote_training.checkpoint_retry_timeout_seconds)
    retry = float(remote_training.retry_interval_seconds)
    packed_inventory = download_remote_packed_preselection_inventory(
        backend,
        prefixes=prefixes,
        staging_dir=remote_config.staging_dir,
        timeout_seconds=timeout,
        retry_interval_seconds=retry,
    )
    packed_staging_logical = remote_config.staging_dir / "packed"
    packed_info = validate_local_remote_packed_controls(
        packed_row_dir=packed_row_dir,
        packed_staging_dir=packed_staging_logical,
        packed_inventory=packed_inventory,
    )
    splice_sidecar_hash = splice_sidecar_fingerprint(
        feature_dir,
        cache_dir=packed_row_dir,
    )
    splice_derived_feature_hash = derived_feature_fingerprint(
        feature_dir,
        expected_stems=packed_info.base_shard_stems,
        cache_dir=packed_row_dir,
    )
    source_inventory = build_trusted_remote_preselection_source(
        backend,
        bucket_id=remote_training.bucket_id,
        prefixes=prefixes,
        percentile_checksum=pct_checksum,
        feature_dir=feature_dir,
        expected_stems=packed_info.base_shard_stems,
        sidecar_fingerprint=splice_sidecar_hash,
        derived_feature_fingerprint=splice_derived_feature_hash,
    )
    packed_info = validate_local_remote_preselection_controls(
        packed_row_dir=packed_row_dir,
        feature_dir=feature_dir,
        packed_staging_dir=packed_staging_logical,
        packed_inventory=packed_inventory,
        source_inventory=source_inventory,
        expected_segment_size=int(cfg.segment_size),
    )
    splice_boundary_hash = tensor_content_fingerprint(pipeline.boundaries)
    expected_names = [
        cast(str, entry["name"])
        for entry in cast(list[dict[str, object]], packed_inventory["shards"])
    ]
    receipts, prior_manifest = discover_remote_preselection_receipts(
        backend,
        output_prefix=prefixes.selection_output,
        expected_names=expected_names,
        selection_checksum=selection_hash,
        packed_inventory_aggregate=cast(str, packed_inventory["aggregate_sha256"]),
        source_inventory_aggregate=cast(str, source_inventory["aggregate_sha256"]),
        packed_row_fingerprint=packed_info.packed_row_fingerprint,
        staging_dir=remote_config.staging_dir,
        timeout_seconds=timeout,
        retry_interval_seconds=retry,
    )
    packed_segments_by_name = {
        cast(str, entry["name"]): cast(int, entry["total_segments"])
        for entry in cast(list[dict[str, object]], packed_inventory["shards"])
    }
    for name, receipt in receipts.items():
        if receipt["expected_packed_segments"] != packed_segments_by_name[name]:
            raise RemotePreselectionError(
                f"remote pre-selection receipt segment count drift for {name}"
            )
    if prior_manifest is not None and set(receipts) != set(expected_names):
        raise RemotePreselectionError(
            "final remote manifest exists without exact body/receipt coverage"
        )
    missing_entries = [
        entry
        for entry in cast(list[dict[str, object]], packed_inventory["shards"])
        if entry["name"] not in receipts
    ]
    active_workers = min(selection_num_workers, max(1, len(missing_entries)))
    _admit_remote_preselection_staging(
        config=remote_config,
        packed_inventory=packed_inventory,
        source_inventory=source_inventory,
        packed_info=packed_info,
        feature_dir=feature_dir,
        n_sae_features=n_sae_features,
        skip_feature_mags=not feature_clusters.is_identity_after_mask,
        active_workers=active_workers,
        source_cache_shards=selection_source_cache_shards,
        sidecar_cache_shards=selection_sidecar_cache_shards,
        shared_store_root=shared_store_root,
        shared_store_capacity=shared_store_capacity,
        packed_spool_dir=selection_packed_spool_dir,
        has_selection_work=bool(missing_entries),
    )

    input_paths = {
        PurePosixPath(str(prefixes.packed_rows), name).as_posix() for name in expected_names
    }
    input_paths.add(
        PurePosixPath(str(prefixes.packed_rows), REMOTE_PACKED_INVENTORY_FILENAME).as_posix()
    )
    source_manifest = cast(
        Mapping[str, object],
        cast(Mapping[str, object], source_inventory["percentile"])["manifest"],
    )
    source_statistics = cast(Mapping[str, object], source_inventory["feature_statistics"])
    input_paths.add(cast(str, source_manifest["path"]))
    input_paths.add(cast(str, source_statistics["path"]))
    for sidecar in cast(
        list[dict[str, object]],
        cast(Mapping[str, object], source_inventory["sidecars"])["files"],
    ):
        input_paths.add(cast(str, sidecar["path"]))
        for entry in cast(list[dict[str, object]], source_inventory["shards"]):
            raw_entry = cast(Mapping[str, object], entry["raw"])
            percentile_entry = cast(Mapping[str, object], entry["percentile"])
            input_paths.add(cast(str, raw_entry["path"]))
            input_paths.add(cast(str, percentile_entry["path"]))
    input_snapshot = {path: backend.stat(path) for path in sorted(input_paths)}
    if any(info is None for info in input_snapshot.values()):
        raise RemotePreselectionError("an inventoried remote input disappeared")
    percentile_prefix = PurePosixPath(str(prefixes.raw_source), "pct", pct_checksum[:16]).as_posix()
    output_prefix = str(prefixes.selection_output)

    def _input_namespace(prefix: str) -> tuple[object, ...]:
        return tuple(
            info
            for info in backend.list(prefix)
            if info.path != output_prefix and not info.path.startswith(f"{output_prefix}/")
        )

    source_namespace_snapshot = _input_namespace(str(prefixes.raw_source))
    percentile_namespace_snapshot = _input_namespace(percentile_prefix)
    packed_namespace_snapshot = _input_namespace(str(prefixes.packed_rows))

    def _assert_remote_input_fence() -> None:
        for path, expected in input_snapshot.items():
            if backend.stat(path) != expected:
                raise RemotePreselectionError(
                    f"remote input changed before final manifest publication: {path}"
                )
        if _input_namespace(str(prefixes.raw_source)) != source_namespace_snapshot:
            raise RemotePreselectionError(
                "remote raw-source namespace changed during pre-selection"
            )
        if _input_namespace(percentile_prefix) != percentile_namespace_snapshot:
            raise RemotePreselectionError(
                "remote percentile namespace changed during pre-selection"
            )
        current_packed = _input_namespace(str(prefixes.packed_rows))
        if current_packed != packed_namespace_snapshot:
            raise RemotePreselectionError(
                "remote packed input namespace changed during pre-selection"
            )

    with tempfile.TemporaryDirectory(
        prefix="remote-preselection-run-", dir=remote_config.staging_dir
    ) as run_name:
        run_root = Path(run_name)
        packed_stage = run_root / "packed"
        source_stage = run_root / "source"
        output_stage = run_root / "selection"
        for directory in (packed_stage, source_stage, output_stage):
            directory.mkdir()

        def _commit(shard_name: str, stats: SpliceSelectionStats) -> None:
            name = f"{shard_name}{_SELECTION_SHARD_SUFFIX}"
            packed_entry = next(
                entry
                for entry in cast(list[dict[str, object]], packed_inventory["shards"])
                if entry["name"] == name
            )
            if stats.total_packed_segments != packed_entry["total_segments"]:
                raise RemotePreselectionError(
                    f"selector segment count differs from packed inventory for {name}"
                )
            selection_path = output_stage / name
            before = _selection_shard_file_identity(
                selection_path, surface="remote pre-selection output"
            )
            _integrity, content_sha, width = _validate_and_fingerprint_clustered_selection_shard(
                prior_integrity=None,
                selection_path=selection_path,
                expected_segments=cast(int, packed_entry["total_segments"]),
                feature_clusters=feature_clusters,
                effective_buckets=pipeline.effective_buckets,
                surface="remote pre-selection output",
            )
            if selection_path.stat().st_size > remote_config.max_output_shard_bytes:
                raise RemotePreselectionError(
                    f"selection body {name} exceeds max_output_shard_bytes"
                )
            body_remote_path = PurePosixPath(str(prefixes.selection_output), name).as_posix()
            local_info, stable_sha = local_object_info_for_remote_path(
                selection_path, remote_path=body_remote_path
            )
            if (
                stable_sha != content_sha
                or _selection_shard_file_identity(
                    selection_path, surface="remote pre-selection output"
                )
                != before
            ):
                raise RemotePreselectionError(
                    f"selection body changed across validation for {name}"
                )
            receipt = build_remote_preselection_receipt(
                selection_checksum=selection_hash,
                packed_inventory_aggregate=cast(str, packed_inventory["aggregate_sha256"]),
                source_inventory_aggregate=cast(str, source_inventory["aggregate_sha256"]),
                packed_row_fingerprint=packed_info.packed_row_fingerprint,
                shard_name=name,
                expected_packed_segments=stats.total_packed_segments,
                selection_body=local_info,
                content_sha256=content_sha,
                splice_stats={
                    "total_packed_segments": stats.total_packed_segments,
                    "total_sidecar_splice_positions_referenced": (
                        stats.total_sidecar_splice_positions_referenced
                    ),
                    "total_delta_positions_selected": (stats.total_delta_positions_selected),
                },
                max_features_per_segment=width,
            )
            publish_remote_preselection_shard(
                backend,
                selection_path=selection_path,
                receipt=receipt,
                output_prefix=prefixes.selection_output,
                timeout_seconds=timeout,
                retry_interval_seconds=retry,
            )
            receipts[name] = receipt
            if (
                _selection_shard_file_identity(
                    selection_path, surface="remote pre-selection cleanup"
                )
                != before
            ):
                raise RemotePreselectionError(f"selection body changed before cleanup for {name}")
            selection_path.unlink()

        _run_remote_splice_shards(
            entries=missing_entries,
            cfg=cfg,
            feature_dir=feature_dir,
            source_inventory=source_inventory,
            source_staging_root=source_stage,
            packed_staging_dir=packed_stage,
            output_staging_dir=output_stage,
            packed_prefix=str(prefixes.packed_rows),
            bucket_id=str(remote_training.bucket_id),
            backend=backend,
            packed_info=replace(
                packed_info,
                packed_shard_paths=[packed_stage / name for name in expected_names],
            ),
            splice_derived_feature_hash=splice_derived_feature_hash,
            cache_shards=selection_source_cache_shards,
            sidecar_shards=selection_sidecar_cache_shards,
            seed=seed,
            num_workers=selection_num_workers,
            retry_timeout_seconds=timeout,
            retry_interval_seconds=retry,
            packed_spool_dir=selection_packed_spool_dir,
            shared_store_root=shared_store_root,
            shared_store_capacity=shared_store_capacity,
            on_completed_shard=_commit,
        )
        if set(receipts) != set(expected_names):
            raise RemotePreselectionError("remote receipt coverage is incomplete after selection")
        _assert_remote_input_fence()
        manifest_stage = run_root / "manifest"
        manifest_path = _build_remote_splice_selection_manifest(
            directory=manifest_stage,
            selection_hash=selection_hash,
            packed_inventory=packed_inventory,
            receipts=receipts,
            pipeline=pipeline,
            packed_info=packed_info,
            splice_sidecar_hash=splice_sidecar_hash,
            splice_derived_feature_hash=splice_derived_feature_hash,
            splice_boundary_hash=splice_boundary_hash,
            feature_mask=feature_mask,
            feature_clusters=feature_clusters,
            splice_batch_size=selection_splice_batch_size,
        )
        publish_remote_preselection_manifest(
            backend,
            manifest_path=manifest_path,
            output_prefix=prefixes.selection_output,
            staging_dir=remote_config.staging_dir,
            timeout_seconds=timeout,
            retry_interval_seconds=retry,
        )
        final_receipts, final_manifest = discover_remote_preselection_receipts(
            backend,
            output_prefix=prefixes.selection_output,
            expected_names=expected_names,
            selection_checksum=selection_hash,
            packed_inventory_aggregate=cast(str, packed_inventory["aggregate_sha256"]),
            source_inventory_aggregate=cast(str, source_inventory["aggregate_sha256"]),
            packed_row_fingerprint=packed_info.packed_row_fingerprint,
            staging_dir=remote_config.staging_dir,
            timeout_seconds=timeout,
            retry_interval_seconds=retry,
        )
        if final_manifest is None or final_receipts != receipts:
            raise RemotePreselectionError(
                "final remote manifest/receipt fence changed after publication"
            )
        _assert_remote_input_fence()
        manifest_bytes = manifest_path.read_bytes()
        output_dir.mkdir(parents=True, exist_ok=True)
        local_manifest = output_dir / MANIFEST_FILENAME
        if local_manifest.exists() and local_manifest.read_bytes() != manifest_bytes:
            raise RemotePreselectionError(
                f"refusing to replace a conflicting local manifest: {local_manifest}"
            )
        atomic_write_bytes(manifest_bytes, local_manifest)
    logger.info(
        "Remote selection manifest published last at "
        f"{prefixes.selection_output}/{MANIFEST_FILENAME}"
    )


def _run_remote_unspliced_preselection(
    *,
    cfg: DictConfig,
    remote_config: RemotePreselectionConfig,
    remote_training: RemoteTrainingConfig,
    feature_dir: Path,
    packed_row_dir: Path,
    output_dir: Path,
    selection_hash: str,
    pipeline: FeaturePipelineConfig,
    feature_mask,
    feature_clusters: FeatureClusterConsolidation,
    n_sae_features: int,
    feature_statistics: ExtractionStatistics,
    statistics_path: Path,
    statistics_loaded_identity: RegularFileIdentity | None,
    selection_packed_spool_dir: str | None,
) -> None:
    """Produce clustered ordinary selection from authenticated packed bodies."""
    if remote_training.train_packed is None:
        raise RemotePreselectionError("remote_preselection requires remote_training.train_packed")
    packed_manifest = read_manifest(packed_row_dir)
    packed_metadata = (packed_manifest or {}).get("metadata")
    if (
        packed_manifest is None
        or packed_manifest.get("kind") != OverlayKind.PACKED_ROW.value
        or not isinstance(packed_metadata, Mapping)
    ):
        raise RemotePreselectionError("local packed manifest is invalid")
    pct_checksum = packed_metadata.get("pct_checksum")
    if not isinstance(pct_checksum, str) or not _is_lower_sha256(pct_checksum):
        raise RemotePreselectionError(
            "remote unspliced pre-selection requires a lowercase packed percentile checksum"
        )
    statistics_pct_checksum = compute_percentile_checksum(
        build_percentile_table(
            feature_statistics.per_feature_stats,
            feature_statistics.metadata.n_sae_features,
        )
    )
    if statistics_pct_checksum != pct_checksum:
        raise RemotePreselectionError(
            "local feature statistics produce a different percentile checksum than packed rows"
        )
    try:
        parsed_feature_dir = parse_feature_dir(feature_dir)
    except ValueError as exc:
        raise RemotePreselectionError(
            f"remote unspliced feature_dir is not canonical: {feature_dir}"
        ) from exc
    if parsed_feature_dir.segment_size != int(cfg.segment_size):
        raise RemotePreselectionError(
            "remote unspliced feature-directory segment size differs from cfg.segment_size"
        )
    if feature_clusters.n_sae_features != n_sae_features:
        raise RemotePreselectionError(
            "remote unspliced cluster SAE width differs from feature statistics"
        )

    prefixes = resolve_remote_preselection_prefixes(
        remote=remote_training,
        feature_root=cfg.feature_root,
        feature_dir=feature_dir,
        packed_row_dir=packed_row_dir,
        selection_checksum=selection_hash,
    )
    remote_config.assert_disjoint(
        feature_dir,
        packed_row_dir,
        output_dir,
        selection_packed_spool_dir,
    )
    backend = HfBucketBackend(remote_training.bucket_id)
    timeout = float(remote_training.checkpoint_retry_timeout_seconds)
    retry = float(remote_training.retry_interval_seconds)
    packed_inventory_path = PurePosixPath(
        str(prefixes.packed_rows), REMOTE_PACKED_INVENTORY_FILENAME
    ).as_posix()
    packed_inventory_identity = backend.stat(packed_inventory_path)
    if packed_inventory_identity is None:
        raise RemotePreselectionError(
            f"remote packed inventory is absent: {packed_inventory_path}"
        )
    packed_inventory = download_remote_packed_preselection_inventory(
        backend,
        prefixes=prefixes,
        staging_dir=remote_config.staging_dir,
        timeout_seconds=timeout,
        retry_interval_seconds=retry,
    )
    packed_info = validate_local_remote_packed_controls(
        packed_row_dir=packed_row_dir,
        packed_staging_dir=remote_config.staging_dir / "remote-unspliced-owned-packed",
        packed_inventory=packed_inventory,
    )
    packed_entries = cast(list[dict[str, object]], packed_inventory["shards"])
    expected_names = [cast(str, entry["name"]) for entry in packed_entries]
    if expected_names != sorted(set(expected_names)) or not expected_names:
        raise RemotePreselectionError(
            "remote unspliced packed inventory names must be non-empty, sorted, and unique"
        )
    provenance_names = [path.name for path in packed_info.provenance_shard_paths]
    if provenance_names != expected_names:
        raise RemotePreselectionError(
            "remote unspliced packed manifest, inventory, and provenance domains differ"
        )

    packed_manifest_path = packed_row_dir / MANIFEST_FILENAME
    manifest_start_identity, manifest_start_sha256 = _stable_local_file_sha256(
        packed_manifest_path, surface="local packed manifest"
    )
    statistics_start_identity, statistics_start_sha256 = _stable_local_file_sha256(
        statistics_path, surface="local feature statistics"
    )
    if (
        statistics_loaded_identity is None
        or statistics_start_identity != statistics_loaded_identity
    ):
        raise RemotePreselectionError(
            "local feature statistics changed after the selection pipeline was built"
        )
    receipts, prior_manifest, _body_only = (
        discover_remote_unspliced_preselection_receipts(
            backend,
            output_prefix=prefixes.selection_output,
            expected_names=expected_names,
            selection_checksum=selection_hash,
            packed_inventory_aggregate=cast(str, packed_inventory["aggregate_sha256"]),
            packed_row_fingerprint=packed_info.packed_row_fingerprint,
            staging_dir=remote_config.staging_dir,
            timeout_seconds=timeout,
            retry_interval_seconds=retry,
        )
    )
    packed_segments_by_name = {
        cast(str, entry["name"]): cast(int, entry["total_segments"])
        for entry in packed_entries
    }
    for name, receipt in receipts.items():
        if receipt["expected_packed_segments"] != packed_segments_by_name[name]:
            raise RemotePreselectionError(
                f"remote ordinary pre-selection receipt segment count drift for {name}"
            )
    if prior_manifest is not None and set(receipts) != set(expected_names):
        raise RemotePreselectionError(
            "final remote manifest exists without exact body/receipt coverage"
        )
    remote_manifest_bytes: bytes | None = None
    if prior_manifest is not None:
        downloaded_manifest, remote_manifest_bytes = download_remote_object_bytes(
            backend,
            remote_path=PurePosixPath(
                str(prefixes.selection_output), MANIFEST_FILENAME
            ).as_posix(),
            staging_dir=remote_config.staging_dir,
            timeout_seconds=timeout,
            retry_interval_seconds=retry,
            surface="final remote ordinary selection manifest",
        )
        if downloaded_manifest != prior_manifest:
            raise RemotePreselectionError("final remote manifest identity changed during download")
    _assert_remote_logical_selection_dir(
        output_dir,
        remote_manifest_bytes=remote_manifest_bytes,
    )

    has_selection_work = len(receipts) != len(expected_names)
    _admit_remote_unspliced_preselection_staging(
        config=remote_config,
        packed_inventory=packed_inventory,
        packed_info=packed_info,
        n_sae_features=n_sae_features,
        has_selection_work=has_selection_work,
    )

    input_paths = {
        packed_inventory_path,
        *{
            PurePosixPath(str(prefixes.packed_rows), name).as_posix()
            for name in expected_names
        },
    }
    input_snapshot = {path: backend.stat(path) for path in sorted(input_paths)}
    if any(info is None for info in input_snapshot.values()):
        raise RemotePreselectionError("an inventoried remote packed input disappeared")
    if input_snapshot[packed_inventory_path] != packed_inventory_identity:
        raise RemotePreselectionError("remote packed inventory changed during startup")
    output_prefix = str(prefixes.selection_output)

    def _packed_namespace() -> tuple[RemoteObjectInfo, ...]:
        listing = tuple(
            info
            for info in backend.list(str(prefixes.packed_rows))
            if info.path != output_prefix and not info.path.startswith(f"{output_prefix}/")
        )
        if len({info.path for info in listing}) != len(listing):
            raise RemotePreselectionError("remote packed namespace contains duplicate paths")
        return tuple(sorted(listing, key=lambda info: info.path))

    packed_namespace_snapshot = _packed_namespace()

    def _assert_local_controls_current() -> None:
        current_info = validate_local_remote_packed_controls(
            packed_row_dir=packed_row_dir,
            packed_staging_dir=remote_config.staging_dir / "remote-unspliced-owned-packed",
            packed_inventory=packed_inventory,
        )
        if current_info != packed_info:
            raise RemotePreselectionError(
                "local packed controls changed during remote unspliced selection"
            )
        manifest_identity, manifest_sha256 = _stable_local_file_sha256(
            packed_manifest_path, surface="local packed manifest fence"
        )
        if (
            manifest_identity != manifest_start_identity
            or manifest_sha256 != manifest_start_sha256
        ):
            raise RemotePreselectionError("local packed manifest changed during selection")
        statistics_identity, statistics_sha256 = _stable_local_file_sha256(
            statistics_path, surface="local feature statistics fence"
        )
        if (
            statistics_identity != statistics_start_identity
            or statistics_sha256 != statistics_start_sha256
        ):
            raise RemotePreselectionError("local feature statistics changed during selection")
        current_statistics = load_statistics(statistics_path)
        current_checksum = compute_percentile_checksum(
            build_percentile_table(
                current_statistics.per_feature_stats,
                current_statistics.metadata.n_sae_features,
            )
        )
        if current_checksum != pct_checksum or current_checksum != statistics_pct_checksum:
            raise RemotePreselectionError(
                "feature-statistics percentile checksum changed during selection"
            )

    def _assert_remote_inputs_current() -> None:
        for path, expected in input_snapshot.items():
            if backend.stat(path) != expected:
                raise RemotePreselectionError(
                    f"remote packed input changed before manifest publication: {path}"
                )
        if _packed_namespace() != packed_namespace_snapshot:
            raise RemotePreselectionError(
                "remote packed namespace changed during ordinary pre-selection"
            )
        _assert_local_controls_current()

    with tempfile.TemporaryDirectory(
        prefix="remote-unspliced-preselection-run-", dir=remote_config.staging_dir
    ) as run_name:
        run_root = Path(run_name)
        packed_stage = run_root / "packed"
        output_stage = run_root / "selection"
        packed_stage.mkdir()
        output_stage.mkdir()

        if has_selection_work:
            receipt_prefix_names = set(receipts)

            def _commit_or_replay(
                entry: Mapping[str, object],
                selection_path: Path,
                replay: bool,
            ) -> None:
                name = cast(str, entry["name"])
                before = _selection_shard_file_identity(
                    selection_path, surface="remote ordinary pre-selection output"
                )
                _integrity, content_sha256, width = (
                    _validate_and_fingerprint_clustered_selection_shard(
                        prior_integrity=None,
                        selection_path=selection_path,
                        expected_segments=cast(int, entry["total_segments"]),
                        feature_clusters=feature_clusters,
                        effective_buckets=pipeline.effective_buckets,
                        surface="remote ordinary pre-selection output",
                    )
                )
                byte_length = selection_path.stat().st_size
                if byte_length <= 0:
                    raise RemotePreselectionError(
                        f"ordinary selection body {name} must have positive byte length"
                    )
                post_rng_sha256 = _selection_rng_state_sha256(pipeline)
                if replay:
                    receipt = receipts[name]
                    body = cast(Mapping[str, object], receipt["selection_body"])
                    if (
                        byte_length != body["byte_length"]
                        or content_sha256 != body["content_sha256"]
                        or width != receipt["max_features_per_segment"]
                        or post_rng_sha256 != receipt["post_selection_rng_sha256"]
                    ):
                        raise RemotePreselectionError(
                            f"ordinary selection prefix replay drift for {name}"
                        )
                else:
                    if byte_length > remote_config.max_output_shard_bytes:
                        raise RemotePreselectionError(
                            f"selection body {name} exceeds max_output_shard_bytes"
                        )
                    body_remote_path = PurePosixPath(
                        str(prefixes.selection_output), name
                    ).as_posix()
                    local_info, stable_sha256 = local_object_info_for_remote_path(
                        selection_path,
                        remote_path=body_remote_path,
                    )
                    if stable_sha256 != content_sha256:
                        raise RemotePreselectionError(
                            f"ordinary selection compressed digest drift for {name}"
                        )
                    receipt = build_remote_unspliced_preselection_receipt(
                        selection_checksum=selection_hash,
                        packed_inventory_aggregate=cast(
                            str, packed_inventory["aggregate_sha256"]
                        ),
                        packed_row_fingerprint=packed_info.packed_row_fingerprint,
                        shard_name=name,
                        expected_packed_segments=cast(int, entry["total_segments"]),
                        selection_body=local_info,
                        content_sha256=content_sha256,
                        max_features_per_segment=width,
                        post_selection_rng_sha256=post_rng_sha256,
                    )
                    publish_remote_unspliced_preselection_shard(
                        backend,
                        selection_path=selection_path,
                        receipt=receipt,
                        output_prefix=prefixes.selection_output,
                        timeout_seconds=timeout,
                        retry_interval_seconds=retry,
                    )
                    receipts[name] = receipt
                if (
                    _selection_shard_file_identity(
                        selection_path,
                        surface="remote ordinary pre-selection cleanup",
                    )
                    != before
                ):
                    raise RemotePreselectionError(
                        f"ordinary selection body changed before cleanup for {name}"
                    )
                selection_path.unlink()

            _run_remote_unspliced_shards(
                entries=packed_entries,
                backend=backend,
                packed_prefix=str(prefixes.packed_rows),
                packed_staging_dir=packed_stage,
                output_staging_dir=output_stage,
                pipeline=pipeline,
                feature_clusters=feature_clusters,
                replay_names=receipt_prefix_names,
                retry_timeout_seconds=timeout,
                retry_interval_seconds=retry,
                on_selected_shard=_commit_or_replay,
            )
        if set(receipts) != set(expected_names):
            raise RemotePreselectionError(
                "remote ordinary receipt coverage is incomplete after selection"
            )

        _assert_remote_inputs_current()
        manifest_stage = run_root / "manifest"
        expected_manifest_path = _build_remote_unspliced_selection_manifest(
            directory=manifest_stage,
            selection_hash=selection_hash,
            packed_inventory=packed_inventory,
            receipts=receipts,
            pipeline=pipeline,
            packed_info=packed_info,
            feature_mask=feature_mask,
            feature_clusters=feature_clusters,
        )
        expected_manifest_bytes = expected_manifest_path.read_bytes()
        if remote_manifest_bytes is not None:
            if remote_manifest_bytes != expected_manifest_bytes:
                raise RemotePreselectionError(
                    "final remote ordinary selection manifest conflicts with current inputs"
                )
        else:
            publish_remote_preselection_manifest(
                backend,
                manifest_path=expected_manifest_path,
                output_prefix=prefixes.selection_output,
                staging_dir=remote_config.staging_dir,
                timeout_seconds=timeout,
                retry_interval_seconds=retry,
            )
            final_receipts, final_manifest, final_body_only = (
                discover_remote_unspliced_preselection_receipts(
                    backend,
                    output_prefix=prefixes.selection_output,
                    expected_names=expected_names,
                    selection_checksum=selection_hash,
                    packed_inventory_aggregate=cast(
                        str, packed_inventory["aggregate_sha256"]
                    ),
                    packed_row_fingerprint=packed_info.packed_row_fingerprint,
                    staging_dir=remote_config.staging_dir,
                    timeout_seconds=timeout,
                    retry_interval_seconds=retry,
                )
            )
            if (
                final_manifest is None
                or final_receipts != receipts
                or final_body_only
            ):
                raise RemotePreselectionError(
                    "final ordinary remote manifest/receipt fence changed after publication"
                )
        fenced_receipts, fenced_manifest, fenced_body_only = (
            discover_remote_unspliced_preselection_receipts(
                backend,
                output_prefix=prefixes.selection_output,
                expected_names=expected_names,
                selection_checksum=selection_hash,
                packed_inventory_aggregate=cast(
                    str, packed_inventory["aggregate_sha256"]
                ),
                packed_row_fingerprint=packed_info.packed_row_fingerprint,
                staging_dir=remote_config.staging_dir,
                timeout_seconds=timeout,
                retry_interval_seconds=retry,
            )
        )
        if (
            fenced_manifest is None
            or fenced_receipts != receipts
            or fenced_body_only
        ):
            raise RemotePreselectionError(
                "final ordinary remote namespace changed before local publication"
            )
        fenced_manifest_info, fenced_manifest_bytes = download_remote_object_bytes(
            backend,
            remote_path=PurePosixPath(
                str(prefixes.selection_output), MANIFEST_FILENAME
            ).as_posix(),
            staging_dir=remote_config.staging_dir,
            timeout_seconds=timeout,
            retry_interval_seconds=retry,
            surface="final remote ordinary selection manifest fence",
        )
        if (
            fenced_manifest_info != fenced_manifest
            or fenced_manifest_bytes != expected_manifest_bytes
        ):
            raise RemotePreselectionError(
                "final ordinary remote manifest changed before local publication"
            )
        _assert_remote_inputs_current()
        _assert_remote_logical_selection_dir(
            output_dir,
            remote_manifest_bytes=(
                remote_manifest_bytes if remote_manifest_bytes is not None else None
            ),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        local_manifest_path = output_dir / MANIFEST_FILENAME
        if (
            local_manifest_path.exists()
            and local_manifest_path.read_bytes() != expected_manifest_bytes
        ):
            raise RemotePreselectionError(
                f"refusing to replace a conflicting local manifest: {local_manifest_path}"
            )
        atomic_write_bytes(expected_manifest_bytes, local_manifest_path)
    logger.info(
        "Remote ordinary selection manifest published last at "
        f"{prefixes.selection_output}/{MANIFEST_FILENAME}"
    )


@hydra.main(
    config_path="../../configs/training",
    config_name="conditional_encdec_train",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Precompute selection shards for a packed row directory.

    Uses the training config to build the feature pipeline (same selection,
    discretization, and negative sampling as training), then applies it to
    each base shard.

    Usage::

        uv run python -m scripts.preprocess.precompute_selected_features \
            packed_row_dir=path/to/packed_rows \
            feature_delta=null feature_selection.delta_weights=null

    The packed_row_dir override is REQUIRED. All other params come from
    the training config (feature_selection, negative_selection, etc.).
    """
    reject_feature_clustering_config(cfg, "precompute_selected_features")

    remote_preselection_raw = (
        OmegaConf.to_container(cfg.remote_preselection, resolve=True)
        if cfg.get("remote_preselection") is not None
        else None
    )
    remote_preselection = RemotePreselectionConfig.parse(remote_preselection_raw)
    remote_training: RemoteTrainingConfig | None = None
    if remote_preselection is not None:
        remote_training_raw = (
            OmegaConf.to_container(cfg.remote_training, resolve=True)
            if cfg.get("remote_training") is not None
            else None
        )
        remote_training = RemoteTrainingConfig.parse(remote_training_raw)
        if remote_training is None:
            raise RemotePreselectionError(
                "remote_preselection requires a non-null remote_training mapping"
            )

    packed_row_dir = cfg.get("packed_row_dir")
    if packed_row_dir is None:
        logger.error("packed_row_dir is required (pass packed_row_dir=path/to/packed_rows)")
        sys.exit(1)

    packed_row_dir = Path(packed_row_dir)
    if not packed_row_dir.is_dir():
        logger.error(f"packed_row_dir not found: {packed_row_dir}")
        sys.exit(1)

    # Resolve feature_dir (same logic as training script).
    # Note: this script reads the *training* config (configs/training/...),
    # whose SAE keys live under cfg.sae rather than at the top level — that
    # is why the shared ``resolve_feature_dir`` helper used by the other
    # precompute scripts is NOT a drop-in replacement here.
    if cfg.get("feature_dir") is not None:
        feature_dir = Path(cfg.feature_dir)
    else:
        feature_dir = resolve_output_dir(
            output_dir=None,
            feature_root=cfg.feature_root,
            segment_size=cfg.segment_size,
            layer=cfg.sae.layer,
            activation_type=ActivationType(cfg.sae.activation_type),
            pooling=PoolingMode(cfg.pooling),
            width=SAEWidth(cfg.sae.width),
            model=cfg.sae.model,
            dataset_phase=cfg.get("dataset_phase"),
            ignore_bos_features=PRODUCER_IGNORE_BOS_FEATURES,
        )

    # Cross-stage segment_size check: the training cfg, the resolved feature
    # directory convention, and the base shard metadata must agree.
    # ``precompute_selected_features`` previously skipped this check (slice 5
    # map P1 #5) — pointing the script at a stale packed-row dir from a
    # different segment_size run produced silent misalignment.
    if remote_preselection is None:
        verify_segment_size(feature_dir, expected_segment_size=int(cfg.segment_size))

    stats_path = find_statistics_path(feature_dir)
    if stats_path is None:
        logger.error(f"No statistics found in {feature_dir}")
        sys.exit(1)
    statistics_load_identity = (
        RegularFileIdentity.capture(Path(stats_path))
        if remote_preselection is not None and cfg.get("feature_delta") is None
        else None
    )
    feature_stats = load_statistics(stats_path)
    if (
        statistics_load_identity is not None
        and RegularFileIdentity.capture(Path(stats_path)) != statistics_load_identity
    ):
        raise RemotePreselectionError(
            "local feature statistics changed while the selection pipeline was built"
        )
    per_feature_stats = feature_stats.per_feature_stats
    n_sae_features = feature_stats.metadata.n_sae_features
    logger.info(f"Loaded {n_sae_features} SAE features from {stats_path}")

    # --- Read raw cfg blocks for compute_selection_hash (used below) ---
    feature_selection_raw = cast(
        dict[str, Any],
        OmegaConf.to_container(cfg.feature_selection, resolve=True),
    )
    negative_selection_raw = (
        cast(
            dict[str, Any],
            OmegaConf.to_container(cfg.negative_selection, resolve=True),
        )
        if cfg.get("negative_selection") is not None
        else None
    )
    feature_delta_raw = (
        cast(
            dict[str, Any],
            OmegaConf.to_container(cfg.feature_delta, resolve=True),
        )
        if cfg.get("feature_delta") is not None
        else None
    )
    feature_delta = parse_feature_delta(feature_delta_raw)
    if feature_delta is not None and feature_delta.mode is not DeltaMode.splice_counterfactual:
        if remote_preselection is not None:
            raise RemotePreselectionError(
                "remote pre-selection supports feature_delta=null or "
                "feature_delta.mode=splice_counterfactual only"
            )
        raise ValueError(
            "precompute_selected_features does not support "
            f"feature_delta.mode={feature_delta.mode.value!r} for packed rows; "
            "only splice_counterfactual is supported. prev_segment or cum_max "
            "require carried document-history state that packed rows do not "
            "encode."
        )

    maybe_apply_frequency_blacklist(
        cfg,
        str(feature_dir),
        n_sae_features,
    )

    feature_mask = resolve_feature_mask(cfg, n_sae_features)
    resolved_feature_clusters = resolve_feature_clusters(cfg, n_sae_features)
    feature_clusters = (
        FeatureClusterConsolidation.build(resolved_feature_clusters, feature_mask)
        if resolved_feature_clusters is not None
        else None
    )
    if feature_clusters is not None:
        assert resolved_feature_clusters is not None
        cluster_counts = feature_clusters.metadata_dict()["counts"]
        logger.info(
            "Static feature clusters: "
            f"{cluster_counts['source_clusters']} source clusters, "
            f"{cluster_counts['consolidated_away']} allowed members consolidated away, "
            f"descriptor {resolved_feature_clusters.descriptor.identity_sha256()[:16]}..."
        )

    seed = cfg.get("seed", 42)
    num_buckets = cfg.num_buckets

    # --- Splice-counterfactual source cache capacities (issue #740) ---
    # Producer-only; NOT part of compute_selection_hash — every existing
    # sel/<hash>/ dir stays valid regardless of these values. Only consumed
    # below when the feature_delta.mode=splice_counterfactual branch builds
    # the two indices.
    selection_source_cache_shards = cfg.get("selection_source_cache_shards", 5)
    selection_sidecar_cache_shards = cfg.get("selection_sidecar_cache_shards", 16)
    # Producer-only shard parallelism (issue #740 PR-D): default 1 = single-process
    # (unchanged, byte-identical). Validated HERE, before any output-dir / manifest
    # write, so a rejected value never mutates an existing complete sel/ manifest.
    selection_num_workers = cfg.get("selection_num_workers", 1)
    for name, value in (
        ("selection_source_cache_shards", selection_source_cache_shards),
        ("selection_sidecar_cache_shards", selection_sidecar_cache_shards),
        ("selection_num_workers", selection_num_workers),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an int >= 1, got {value!r}")
    selection_splice_batch_size = resolve_splice_selection_batch_size(
        cfg.get("selection_splice_batch_size")
    )
    # Producer-only shared /dev/shm source store (issue #740 PR-F): when true AND running
    # parallel splice, workers share one physical copy per source stem. Default off ⇒ unchanged.
    # Capacity 0 ⇒ auto (num_workers + 4, matching the measured locality target).
    # The count is not a byte bound; admission accounts exact runtime dtypes,
    # evicted mappings, in-flight worker generations, and publication storage.
    selection_shared_store_raw = cfg.get("selection_shared_store", False)
    if not isinstance(selection_shared_store_raw, bool):
        raise ValueError(
            f"selection_shared_store must be a bool, got {selection_shared_store_raw!r}"
        )
    selection_shared_store = selection_shared_store_raw
    selection_shared_store_capacity = cfg.get("selection_shared_store_capacity", 0)
    if (
        isinstance(selection_shared_store_capacity, bool)
        or not isinstance(selection_shared_store_capacity, int)
        or selection_shared_store_capacity < 0
    ):
        raise ValueError(
            "selection_shared_store_capacity must be an int >= 0 (0 = auto), got "
            f"{selection_shared_store_capacity!r}"
        )
    selection_packed_spool_raw = cfg.get("selection_packed_spool_dir")
    selection_packed_spool_dir: str | None = None
    if selection_packed_spool_raw is not None:
        if (
            not isinstance(selection_packed_spool_raw, str)
            or not selection_packed_spool_raw.strip()
        ):
            raise ValueError(
                "selection_packed_spool_dir must be null or a non-empty path "
                f"string, got {selection_packed_spool_raw!r}"
            )
        spool_path = Path(selection_packed_spool_raw).expanduser().absolute()
        if not spool_path.is_dir():
            raise ValueError(
                f"selection_packed_spool_dir must already exist as a directory, got {spool_path}"
            )
        selection_packed_spool_dir = str(spool_path)
    selection_resume = cfg.get("selection_resume", True)
    if not isinstance(selection_resume, bool):
        raise ValueError(f"selection_resume must be a bool, got {selection_resume!r}")
    logger.info(
        f"Splice source caches: {selection_source_cache_shards} feature shard(s) + "
        f"{selection_sidecar_cache_shards} sidecar(s) resident "
        f"(default 1 = legacy single-slot reference)."
    )
    logger.info(
        "Selection resume: "
        + (
            "enabled (reuse legacy unclustered shards; clustered shards require "
            "identity-validated byte certificates)"
            if selection_resume
            else "disabled (recompute every selection shard)"
        )
    )

    # --- Build pipeline (delegates parse + build to shared helper) ---
    pipeline = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=per_feature_stats,
        n_sae_features=n_sae_features,
        num_buckets=num_buckets,
        seed=seed,
        feature_delta=feature_delta,
        feature_mask=feature_mask,
    )
    logger.info(
        f"Pipeline: effective_buckets={pipeline.effective_buckets}, "
        f"selection_type={type(pipeline.selection_config).__name__}"
    )
    assert_precompute_supports_pipeline(pipeline)
    is_splice_counterfactual = (
        pipeline.delta_bundle is not None
        and pipeline.delta_bundle.config.mode is DeltaMode.splice_counterfactual
    )
    if selection_splice_batch_size == 1:
        batching_log = "legacy one-segment RNG schedule"
    elif is_splice_counterfactual:
        batching_log = (
            f"up to {selection_splice_batch_size} ordinary executions per "
            "batch (versioned selection identity)"
        )
    else:
        batching_log = (
            f"requested size {selection_splice_batch_size} is inert because "
            "feature_delta.mode is not splice_counterfactual"
        )
    logger.info(f"Splice selection batching: {batching_log}")
    requires_parallel_admission = is_splice_counterfactual and selection_num_workers > 1

    # --- Gradient-attribution ranking artifacts (optional) ---
    grad_spec = resolve_gradient_spec_from_feature_selection(feature_selection_raw)
    assert_gradient_ranking_supported(pipeline, grad_spec)
    assert_feature_clusters_precompute_supported(pipeline, grad_spec, feature_clusters)
    grad_joiner: GradScoreJoiner | None = None
    grad_prov_dir: Path | None = None
    grad_overlay_checksum: str | None = None
    if grad_spec is not None:
        grad_joiner, grad_prov_dir, grad_overlay_checksum = resolve_gradient_artifacts(
            cfg, feature_dir, packed_row_dir, grad_spec
        )
        logger.info(
            f"Gradient ranking: mode={grad_spec.mode.value}, overlay checksum "
            f"{grad_overlay_checksum[:16]}..., provenance at {grad_prov_dir}"
        )

    # --- Compute selection hash for output directory ---
    # ANTI-DESYNC: this selection-hash is the precompute-side ``compute_selection_hash`` call at:
    #   1. scripts/preprocess/precompute_selected_features.py (THIS site)
    #   2. scripts/train/conditional_encdec_train.py:build_shared_state
    #   3. scripts/eval/base_eval.py:_resolve_bpb_selection_dir
    #   4. steering/eval/feature_selection_hash.py:compute_canonical_feature_selection_hash (cache identity)
    # All four must agree on feature-mask, cluster, and splice-batching semantics.
    # Change ALL FOUR or none.
    selection_hash = compute_selection_hash(
        positive_selection=feature_selection_raw,
        negative_selection=negative_selection_raw,
        delta_config=feature_delta_raw,
        num_buckets=num_buckets,
        seed=seed,
        ddp_rank=0,
        feature_mask=canonical_dict(feature_mask),
        feature_clusters=canonical_feature_clusters_descriptor(resolved_feature_clusters),
        splice_batch_size=selection_splice_batch_size,
    )
    output_dir = packed_row_dir / "sel" / selection_hash
    if remote_preselection is not None:
        assert remote_training is not None
        common_unsupported_reason: str | None = None
        if feature_clusters is None:
            common_unsupported_reason = "static feature_clusters are required"
        elif grad_spec is not None:
            common_unsupported_reason = "gradient-attribution ranking is not supported"
        elif not isinstance(pipeline.selection_config, SamplingBundle):
            common_unsupported_reason = "sampling-based feature selection is required"
        elif cfg.get("use_precomputed_batches") is not True:
            common_unsupported_reason = "use_precomputed_batches=true is required"
        elif remote_training.train_packed is None:
            common_unsupported_reason = "remote_training.train_packed is required"
        if common_unsupported_reason is not None:
            raise RemotePreselectionError(
                "remote pre-selection supports only clustered, precomputed, "
                "sampling-based selection: " + common_unsupported_reason
            )
        assert feature_clusters is not None
        if is_splice_counterfactual:
            _run_remote_splice_preselection(
                cfg=cfg,
                remote_config=remote_preselection,
                remote_training=remote_training,
                feature_dir=feature_dir,
                packed_row_dir=packed_row_dir,
                output_dir=output_dir,
                selection_hash=selection_hash,
                pipeline=pipeline,
                feature_mask=feature_mask,
                feature_clusters=feature_clusters,
                n_sae_features=n_sae_features,
                selection_num_workers=selection_num_workers,
                selection_source_cache_shards=selection_source_cache_shards,
                selection_sidecar_cache_shards=selection_sidecar_cache_shards,
                selection_splice_batch_size=selection_splice_batch_size,
                selection_shared_store=selection_shared_store,
                selection_shared_store_capacity=selection_shared_store_capacity,
                selection_packed_spool_dir=selection_packed_spool_dir,
                seed=seed,
            )
        elif pipeline.delta_bundle is None:
            if selection_num_workers != 1:
                raise RemotePreselectionError(
                    "remote unspliced pre-selection supports only clustered, precomputed, "
                    "sampling-based selection with feature_delta=null and "
                    "selection_num_workers=1"
                )
            logger.info(
                "Remote unspliced pre-selection: source/sidecar caches, shared-store, "
                "shared-store capacity, packed spool, and splice-batch settings are inert"
            )
            logger.info(
                "Remote unspliced receipt discovery is authoritative; selection_resume "
                "does not authorize overwrite, deletion, or ignoring completed receipts"
            )
            _run_remote_unspliced_preselection(
                cfg=cfg,
                remote_config=remote_preselection,
                remote_training=remote_training,
                feature_dir=feature_dir,
                packed_row_dir=packed_row_dir,
                output_dir=output_dir,
                selection_hash=selection_hash,
                pipeline=pipeline,
                feature_mask=feature_mask,
                feature_clusters=feature_clusters,
                n_sae_features=n_sae_features,
                feature_statistics=feature_stats,
                statistics_path=Path(stats_path),
                statistics_loaded_identity=statistics_load_identity,
                selection_packed_spool_dir=selection_packed_spool_dir,
            )
        else:
            raise RemotePreselectionError(
                "remote pre-selection supports feature_delta=null or "
                "feature_delta.mode=splice_counterfactual only"
            )
        return
    prior_completed_cluster_metadata: Mapping[str, Any] | None = None
    prior_output_manifest = read_manifest(output_dir)
    if selection_resume and isinstance(prior_output_manifest, Mapping):
        raw_prior_metadata = prior_output_manifest.get("metadata")
        if (
            isinstance(raw_prior_metadata, Mapping)
            and CLUSTERED_SELECTION_VALIDATION_KEY in raw_prior_metadata
        ):
            prior_completed_cluster_metadata = raw_prior_metadata
    _assert_sel_overlay_unchanged(output_dir, grad_overlay_checksum)
    # W=1 and every non-splice mode preserve the legacy output-creation point.
    # High-parallel splice runs delay their first output mutation until after
    # resource admission has proved the requested topology is inside the
    # supported envelope.
    if not requires_parallel_admission:
        output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {output_dir}")
    logger.info(f"Selection hash: {selection_hash[:16]}...")

    packed_info: PackedProvenanceInfo | None = None
    splice_index: SpliceProvenanceIndex | None = None
    splice_feature_index: FeatureIndex | None = None
    splice_feature_catalog: FeatureIndexCatalog | None = None
    splice_sidecar_hash: str | None = None
    splice_derived_feature_hash: str | None = None
    splice_boundary_hash: str | None = None
    pct_checksum: str | None = None
    shared_store_root: str | None = None
    shared_store_capacity = 0
    splice_stats = SpliceSelectionStats()
    if is_splice_counterfactual:
        startup_started = time.monotonic()
        packed_info = validate_packed_provenance(
            packed_row_dir,
            cache_dir=packed_row_dir,
        )
        packed_validation_done = time.monotonic()
        packed_manifest = read_manifest(packed_row_dir)
        packed_meta = (packed_manifest or {}).get("metadata", {})
        pct_checksum = packed_meta.get("pct_checksum")
        if not isinstance(pct_checksum, str) or not pct_checksum:
            raise SpliceProvenanceError(
                f"splice_counterfactual packed rows at {packed_row_dir} are "
                "missing metadata.pct_checksum. Re-run precompute_packed_rows "
                "from feature shards with a percentile overlay."
            )
        # PR-E (#740): halve source-shard resident RAM by not materializing the
        # feature magnitudes when they are provably unused — the cluster-
        # consolidation splice path zeros them at materialization and deltas/refs
        # use precomputed percentiles. Gated OFF when clustering is inert
        # (is_identity_after_mask) or absent, where the non-cluster path DOES read
        # magnitudes for discretization.
        splice_feature_index = FeatureIndex(
            str(feature_dir),
            percentile_checksum=pct_checksum,
            max_resident_shards=selection_source_cache_shards,
            skip_feature_mags=(
                feature_clusters is not None and not feature_clusters.is_identity_after_mask
            ),
            # PR-E2: keep feature_ids int32 resident (the splice path converts on use) —
            # byte-identical on ALL splice paths, so unconditional. ~halves the biggest tensor.
            compact_feature_ids=True,
        )
        feature_scan_done = time.monotonic()
        # Snapshot immediately after the parent's validating scan. The second validation
        # below, after the full content hash, proves that the hash and scan refer to the
        # identical filesystem objects; workers then revalidate this snapshot before use.
        splice_feature_catalog = splice_feature_index.export_catalog()
        catalog_export_done = time.monotonic()
        splice_index = SpliceProvenanceIndex(
            feature_dir,
            max_resident_shards=selection_sidecar_cache_shards,
        )
        splice_sidecar_hash = splice_sidecar_fingerprint(
            feature_dir,
            cache_dir=packed_row_dir,
        )
        sidecar_fingerprint_done = time.monotonic()
        splice_derived_feature_hash = derived_feature_fingerprint(
            feature_dir,
            expected_stems=packed_info.base_shard_stems,
            cache_dir=packed_row_dir,
        )
        feature_fingerprint_done = time.monotonic()
        splice_feature_catalog.validate()
        splice_feature_catalog = replace(
            splice_feature_catalog,
            validated_content_fingerprint=splice_derived_feature_hash,
        )
        splice_feature_catalog.validate()
        catalog_binding_done = time.monotonic()
        splice_boundary_hash = tensor_content_fingerprint(pipeline.boundaries)
        _assert_splice_counterfactual_sel_manifest_current(
            output_dir,
            pipeline,
            packed_info,
            splice_sidecar_hash,
            splice_derived_feature_hash,
            splice_boundary_hash,
            feature_clusters=feature_clusters,
            splice_batch_size=selection_splice_batch_size,
        )
        startup_done = time.monotonic()
        logger.info(
            "SPLICE_RESOURCE_PROFILE "
            + json.dumps(
                {
                    "schema_version": 1,
                    "event": "parent_startup",
                    "packed_shards": len(packed_info.packed_shard_paths),
                    "source_stems": len(packed_info.base_shard_stems),
                    "stages_seconds": {
                        "packed_provenance_validation": (packed_validation_done - startup_started),
                        "feature_index_scan": (feature_scan_done - packed_validation_done),
                        "catalog_export": catalog_export_done - feature_scan_done,
                        "sidecar_fingerprint": (sidecar_fingerprint_done - catalog_export_done),
                        "derived_feature_fingerprint": (
                            feature_fingerprint_done - sidecar_fingerprint_done
                        ),
                        "catalog_fingerprint_binding": (
                            catalog_binding_done - feature_fingerprint_done
                        ),
                        "boundary_and_manifest_validation": (startup_done - catalog_binding_done),
                        "total": startup_done - startup_started,
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        logger.info(
            f"splice_counterfactual: validated packed provenance for "
            f"{len(packed_info.packed_shard_paths)} packed shard(s); "
            f"sidecar fingerprint {splice_sidecar_hash[:16]}..."
        )
        if requires_parallel_admission and selection_shared_store:
            shared_store_capacity = (
                selection_shared_store_capacity
                if selection_shared_store_capacity > 0
                else selection_num_workers + 4
            )
            shared_store_root = os.environ.get("NANOCHAT_SHM_DIR", "/dev/shm")
            if not os.path.isdir(shared_store_root):
                raise RuntimeError(
                    f"selection_shared_store=true but shared-memory dir "
                    f"{shared_store_root!r} is absent; set NANOCHAT_SHM_DIR or "
                    "disable the shared store."
                )

    # --- Process each base shard ---
    base_shard_paths = (
        packed_info.packed_shard_paths
        if packed_info is not None
        else glob_overlay_shards(packed_row_dir)
    )
    if not base_shard_paths:
        logger.error(f"No base shards found in {packed_row_dir}")
        sys.exit(1)
    selection_packed_rows_hash = (
        packed_info.packed_row_fingerprint
        if packed_info is not None
        else packed_row_fingerprint(packed_row_dir, cache_dir=packed_row_dir)
    )
    producer_packed_resolution: PackedFingerprintResolution | None = None
    producer_packed_file_by_name: dict[str, PackedFingerprintFile] = {}
    if feature_clusters is not None:
        producer_packed_resolution = resolve_packed_row_fingerprint_strict(
            packed_row_dir,
            force=False,
        )
        if producer_packed_resolution.aggregate_sha256 != selection_packed_rows_hash:
            raise FeatureClustersError(
                "clustered selection producer strict packed-row proof "
                "disagrees with the selection manifest identity"
            )
        producer_packed_file_by_name = {
            entry.name: entry for entry in producer_packed_resolution.files
        }
        if set(producer_packed_file_by_name) != {path.name for path in base_shard_paths}:
            raise FeatureClustersError(
                "clustered selection producer strict packed-row proof does not "
                "cover the exact packed shard domain"
            )

    def _assert_current_packed_shard(shard_path: Path) -> None:
        if producer_packed_resolution is None:
            return
        assert_packed_row_fingerprint_file_current(
            producer_packed_file_by_name[shard_path.name],
            shard_path,
        )

    certified_existing_stems = _assert_feature_clusters_sel_manifest_current(
        output_dir,
        feature_clusters,
        selection_hash=selection_hash,
        packed_rows_hash=selection_packed_rows_hash,
        resume=selection_resume,
    )
    splice_resume_state = _SpliceResumeState({}, False, None, {}, None)
    if is_splice_counterfactual and feature_clusters is not None and selection_resume:
        splice_resume_state = _load_splice_resume_state(
            output_dir,
            certified_existing_stems=certified_existing_stems,
            packed_rows_hash=selection_packed_rows_hash,
            effective_buckets=pipeline.effective_buckets,
        )
        for cached_stats in splice_resume_state.stats_by_stem.values():
            splice_stats += cached_stats
    base_stems = [p.stem.split(".")[0] for p in base_shard_paths]
    base_stem_set = set(base_stems)
    if feature_clusters is not None:
        actual_existing_stems = {
            path.name.removesuffix(".safetensors.zst") for path in glob_overlay_shards(output_dir)
        }
        unexpected = sorted(actual_existing_stems - base_stem_set)
        if unexpected:
            raise FeatureClustersError(
                f"clustered sel dir {output_dir} contains shard names outside "
                f"the current packed-row domain: {unexpected[:5]}"
            )
        unexpected_certified = sorted(certified_existing_stems - base_stem_set)
        if unexpected_certified:
            raise FeatureClustersError(
                f"clustered sel dir {output_dir} certifies shard names outside "
                f"the current packed-row domain: {unexpected_certified[:5]}"
            )
    existing_stems = certified_existing_stems if selection_resume else set()
    missing_stems = set(base_stems) - existing_stems
    stats_certified_stems = set(splice_resume_state.stats_by_stem)
    splice_paths_to_process = [
        path for path in base_shard_paths if path.stem.split(".")[0] not in stats_certified_stems
    ]
    complete_certified_splice_noop = (
        is_splice_counterfactual
        and feature_clusters is not None
        and selection_resume
        and not missing_stems
        and stats_certified_stems == existing_stems
        and splice_resume_state.outputs_validated
        and set(splice_resume_state.storage_validated_by_stem) == existing_stems
    )

    if requires_parallel_admission:
        assert packed_info is not None
        assert splice_feature_catalog is not None
        skip_feature_mags = (
            feature_clusters is not None and not feature_clusters.is_identity_after_mask
        )
        try:
            admission = admit_splice_resources(
                packed_shard_paths=packed_info.packed_shard_paths,
                provenance_shard_paths=packed_info.provenance_shard_paths,
                base_shard_stems=packed_info.base_shard_stems,
                feature_index_catalog=splice_feature_catalog,
                n_sae_features=PositiveInt.parse(n_sae_features),
                skip_feature_mags=skip_feature_mags,
                num_workers=AtLeastTwoInt.parse(selection_num_workers),
                source_cache_shards=PositiveInt.parse(selection_source_cache_shards),
                sidecar_cache_shards=PositiveInt.parse(selection_sidecar_cache_shards),
                has_selection_work=bool(missing_stems),
                shared_store_enabled=selection_shared_store,
                shared_store_capacity=NonNegativeInt.parse(shared_store_capacity),
                packed_spool_dir=selection_packed_spool_dir,
                shared_store_root=shared_store_root,
            )
        except SpliceResourceAdmissionError as exc:
            logger.error("SPLICE_RESOURCE_ADMISSION " + exc.report.to_json())
            raise
        logger.info("SPLICE_RESOURCE_ADMISSION " + admission.to_json())
        output_dir.mkdir(parents=True, exist_ok=True)

    # A killed atomic writer can leave only its uniquely named temporary file,
    # never a torn final shard. Sweep that debris after high-parallel resource
    # admission (so a rejected configuration remains mutation-free) and before
    # writing the next progress manifest. Live-writer evidence is a hard error.
    _sweep_dead_atomic_selection_temps(output_dir, base_stem_set)

    progress_integrity: dict[str, Any] | None = None
    if splice_resume_state.selection_integrity is not None:
        progress_integrity = splice_resume_state.selection_integrity
    elif feature_clusters is not None and existing_stems:
        progress_integrity = build_clustered_selection_shard_integrity(
            [overlay_shard_path(output_dir, stem) for stem in sorted(existing_stems)]
        )
    progress_splice_stats = dict(splice_resume_state.stats_by_stem)
    progress_storage_validated = dict(splice_resume_state.storage_validated_by_stem)
    progress_validated_max_features = splice_resume_state.validated_max_features_per_segment
    validation_max_features_by_stem: dict[str, int] = {}

    def _write_in_progress_manifest(current_splice_stats: SpliceSelectionStats) -> None:
        if is_splice_counterfactual:
            assert packed_info is not None
            assert splice_sidecar_hash is not None
            assert splice_derived_feature_hash is not None
            assert splice_boundary_hash is not None
            metadata = _splice_selection_metadata(
                pipeline=pipeline,
                packed_info=packed_info,
                splice_stats=current_splice_stats,
                splice_sidecar_hash=splice_sidecar_hash,
                splice_derived_feature_hash=splice_derived_feature_hash,
                splice_boundary_hash=splice_boundary_hash,
                feature_mask=feature_mask,
                feature_clusters=feature_clusters,
                grad_spec=grad_spec,
                grad_overlay_checksum=grad_overlay_checksum,
                status="in_progress",
                splice_batch_size=selection_splice_batch_size,
            )
        else:
            assert feature_clusters is not None or grad_spec is not None
            metadata = _base_selection_metadata(
                feature_mask=feature_mask,
                feature_clusters=feature_clusters,
                grad_spec=grad_spec,
                grad_overlay_checksum=grad_overlay_checksum,
                packed_rows_hash=selection_packed_rows_hash,
                status="in_progress",
            )
        if progress_integrity is not None:
            metadata[SELECTION_SHARD_INTEGRITY_KEY] = progress_integrity
            integrity_names = {
                entry["name"].removesuffix(_SELECTION_SHARD_SUFFIX)
                for entry in progress_integrity["shards"]
            }
            if (
                is_splice_counterfactual
                and feature_clusters is not None
                and set(progress_splice_stats) == integrity_names
            ):
                metadata[SPLICE_RESUME_METADATA_KEY] = _build_splice_resume_metadata(
                    stats_by_stem=progress_splice_stats,
                    selection_integrity=progress_integrity,
                    storage_validated_by_stem=progress_storage_validated,
                    validated_max_features_per_segment=(progress_validated_max_features),
                    packed_rows_hash=selection_packed_rows_hash,
                    effective_buckets=pipeline.effective_buckets,
                    status="in_progress",
                )
        write_manifest(
            output_dir,
            selection_hash,
            OverlayKind.PACKED_ROW,
            metadata=metadata,
            base_shard_paths=base_shard_paths,
            cache_dir=packed_row_dir,
        )

    def _checkpoint_completed_clustered_shard(
        shard_name: str,
        was_written: bool,
        shard_splice_stats: SpliceSelectionStats,
    ) -> None:
        nonlocal progress_integrity, progress_validated_max_features
        if feature_clusters is None:
            return
        if shard_name not in base_stem_set:
            raise FeatureClustersError(
                f"refusing to certify unexpected clustered selection shard "
                f"{shard_name!r} outside the packed-row domain"
            )
        if is_splice_counterfactual:
            progress_splice_stats[shard_name] = shard_splice_stats
            if shard_name not in progress_storage_validated:
                progress_integrity, validated_digest, shard_max_features = (
                    _validate_and_fingerprint_clustered_selection_shard(
                        prior_integrity=progress_integrity,
                        selection_path=overlay_shard_path(output_dir, shard_name),
                        expected_segments=shard_splice_stats.total_packed_segments,
                        feature_clusters=feature_clusters,
                        effective_buckets=pipeline.effective_buckets,
                        surface=f"in-progress sel/ manifest {output_dir}",
                    )
                )
                progress_storage_validated[shard_name] = validated_digest
                validation_max_features_by_stem[shard_name] = shard_max_features
                progress_validated_max_features = max(
                    progress_validated_max_features or 0,
                    shard_max_features,
                )
            elif was_written:
                raise FeatureClustersError(
                    f"refusing to replace already storage-validated splice "
                    f"selection shard {shard_name!r}"
                )
        elif was_written:
            progress_integrity = extend_clustered_selection_shard_integrity(
                progress_integrity,
                overlay_shard_path(output_dir, shard_name),
                surface=f"in-progress sel/ manifest {output_dir}",
            )
        current_stats = SpliceSelectionStats()
        for completed_stats in progress_splice_stats.values():
            current_stats += completed_stats
        _write_in_progress_manifest(current_stats)

    # A current in-progress certificate can bind trustworthy stats before every
    # shard has post-serialization tensor validation. Finish those validations
    # from compact selection artifacts only; the signed total_packed_segments
    # value supplies the CSR row count, so no packed-row body is needed.
    if is_splice_counterfactual and feature_clusters is not None:
        for shard_name in sorted(stats_certified_stems - set(progress_storage_validated)):
            _checkpoint_completed_clustered_shard(
                shard_name,
                False,
                progress_splice_stats[shard_name],
            )

    if (
        is_splice_counterfactual
        or feature_clusters is not None
        or (grad_spec is not None and bool(missing_stems))
    ) and not complete_certified_splice_noop:
        _write_in_progress_manifest(splice_stats)

    logger.info(f"Processing {len(base_shard_paths)} base shards...")
    replay_skipped_for_rng = (
        bool(missing_stems)
        and not is_splice_counterfactual
        and _pipeline_consumes_selection_rng(pipeline)
    )
    t_start = time.monotonic()
    if is_splice_counterfactual and selection_num_workers > 1 and splice_paths_to_process:
        assert packed_info is not None
        assert splice_feature_index is not None
        assert splice_feature_catalog is not None
        assert splice_index is not None
        assert splice_derived_feature_hash is not None
        assert isinstance(pct_checksum, str)
        splice_stats += _run_splice_shards_parallel(
            base_shard_paths=splice_paths_to_process,
            existing_stems=existing_stems,
            cfg=cfg,
            feature_dir=feature_dir,
            pct_checksum=pct_checksum,
            feature_index_catalog=splice_feature_catalog,
            splice_derived_feature_hash=splice_derived_feature_hash,
            cache_shards=selection_source_cache_shards,
            sidecar_shards=selection_sidecar_cache_shards,
            packed_info=packed_info,
            packed_fingerprint_files=tuple(
                producer_packed_file_by_name[name] for name in sorted(producer_packed_file_by_name)
            ),
            output_dir=output_dir,
            packed_row_dir=packed_row_dir,
            seed=seed,
            num_workers=selection_num_workers,
            packed_spool_dir=selection_packed_spool_dir,
            shared_store_root=shared_store_root,
            shared_store_capacity=shared_store_capacity,
            on_completed_shard=_checkpoint_completed_clustered_shard,
        )
    else:
        with tempfile.TemporaryDirectory(
            prefix="precompute_selected_features_rng_advance_"
        ) as tmp_selection_dir:
            tmp_selection_path = Path(tmp_selection_dir)
            for shard_path in base_shard_paths:
                shard_name = shard_path.stem.split(".")[0]
                gradient_join: ShardGradientJoin | None = None
                needs_gradient_join = grad_joiner is not None and (
                    shard_name not in existing_stems or replay_skipped_for_rng
                )
                if needs_gradient_join:
                    assert grad_joiner is not None
                    assert grad_prov_dir is not None
                    prov_path = grad_prov_dir / shard_path.name
                    if not prov_path.is_file():
                        raise GradientAttributionError(
                            f"provenance shard missing for {shard_name} at {prov_path}; "
                            f"re-run scripts.preprocess.precompute_packed_rows with "
                            f"resume=true (heals missing provenance in place)."
                        )
                    gradient_join = ShardGradientJoin(grad_joiner, load_overlay_shard(prov_path))
                if shard_name in existing_stems:
                    logger.info(f"  {shard_name}: already exists, skipping")
                    if is_splice_counterfactual:
                        if shard_name in stats_certified_stems:
                            continue
                        assert packed_info is not None
                        assert splice_index is not None
                        _assert_current_packed_shard(shard_path)
                        shard_stats = splice_counterfactual_stats_for_packed_shard(
                            shard_path,
                            provenance_path=provenance_dir(packed_row_dir) / shard_path.name,
                            packed_info=packed_info,
                            splice_index=splice_index,
                            packed_spool_dir=selection_packed_spool_dir,
                        )
                        _assert_current_packed_shard(shard_path)
                        splice_stats += shard_stats
                        _checkpoint_completed_clustered_shard(
                            shard_name,
                            False,
                            shard_stats,
                        )
                    elif replay_skipped_for_rng:
                        logger.info(
                            f"  {shard_name}: replaying skipped shard to advance "
                            "selection RNG for deterministic resume"
                        )
                        _assert_current_packed_shard(shard_path)
                        select_features_for_shard(
                            shard_path,
                            pipeline,
                            tmp_selection_path,
                            shard_name,
                            gradient_join=gradient_join,
                            feature_clusters=feature_clusters,
                        )
                        _assert_current_packed_shard(shard_path)
                    continue
                if is_splice_counterfactual:
                    assert packed_info is not None
                    assert splice_feature_index is not None
                    assert splice_index is not None
                    _assert_current_packed_shard(shard_path)
                    shard_stats = select_splice_counterfactual_features_for_shard(
                        shard_path,
                        pipeline,
                        output_dir,
                        shard_name,
                        provenance_path=provenance_dir(packed_row_dir) / shard_path.name,
                        packed_info=packed_info,
                        feature_index=splice_feature_index,
                        splice_index=splice_index,
                        seed=seed,
                        feature_clusters=feature_clusters,
                        packed_spool_dir=selection_packed_spool_dir,
                        splice_batch_size=selection_splice_batch_size,
                    )
                    _assert_current_packed_shard(shard_path)
                    splice_stats += shard_stats
                    _checkpoint_completed_clustered_shard(
                        shard_name,
                        True,
                        shard_stats,
                    )
                else:
                    _assert_current_packed_shard(shard_path)
                    was_written = select_features_for_shard(
                        shard_path,
                        pipeline,
                        output_dir,
                        shard_name,
                        gradient_join=gradient_join,
                        feature_clusters=feature_clusters,
                    )
                    _assert_current_packed_shard(shard_path)
                    if was_written:
                        _checkpoint_completed_clustered_shard(
                            shard_name,
                            True,
                            SpliceSelectionStats(),
                        )

    elapsed = time.monotonic() - t_start
    logger.info(f"Done: {len(base_shard_paths)} shards in {elapsed:.0f}s")
    # ru_maxrss is in KiB on Linux; /1e6 converts to GB (approx).
    logger.info(
        f"Peak RSS this process: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB"
    )

    remote_loader_max_features: int | None = None
    reused_remote_loader_certificate = False
    validation_segment_counts: dict[str, int] = {}
    if feature_clusters is not None:
        prior_remote_loader: dict[str, Any] | None = None
        if complete_certified_splice_noop:
            prior_manifest = read_manifest(output_dir)
            prior_metadata = (
                prior_manifest.get("metadata") if isinstance(prior_manifest, Mapping) else None
            )
            prior_raw = (
                prior_metadata.get(REMOTE_LOADER_KEY)
                if isinstance(prior_metadata, Mapping)
                else None
            )
            if prior_raw is not None:
                prior_remote_loader = parse_remote_loader_certificate(
                    prior_raw,
                    surface=f"completed clustered selection manifest {output_dir}",
                )
                if (
                    progress_integrity is None
                    or prior_remote_loader["selection_checksum"] != selection_hash
                    or prior_remote_loader["selection_inventory_aggregate"]
                    != progress_integrity["aggregate_sha256"]
                    or prior_remote_loader["packed_row_fingerprint"] != selection_packed_rows_hash
                    or prior_remote_loader.get(CLUSTER_DELTA_SEMANTICS_KEY)
                    != (
                        CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS
                        if is_splice_counterfactual
                        else None
                    )
                ):
                    raise FeatureClustersError(
                        "completed clustered selection remote-loader certificate "
                        "does not match the current selection, inventory, and packed rows"
                    )
                remote_loader_max_features = prior_remote_loader["max_features_per_segment"]
                reused_remote_loader_certificate = True

        if is_splice_counterfactual:
            if set(progress_storage_validated) != base_stem_set:
                missing_validation = sorted(base_stem_set - set(progress_storage_validated))
                extra_validation = sorted(set(progress_storage_validated) - base_stem_set)
                raise FeatureClustersError(
                    "post-serialization clustered selection validation must exactly "
                    f"cover the packed-row domain; missing={missing_validation[:5]}, "
                    f"extra={extra_validation[:5]}"
                )
            if progress_integrity is None:
                raise FeatureClustersError(
                    "post-serialization clustered selection validation completed "
                    "without a compressed-byte integrity certificate"
                )
            if remote_loader_max_features is not None:
                if progress_validated_max_features is None:
                    # A current remote-loader certificate is bound to these exact
                    # selection bytes and can restore a missing in-progress
                    # running-width aggregate without reading shard bodies.
                    progress_validated_max_features = remote_loader_max_features
                elif progress_validated_max_features != remote_loader_max_features:
                    raise FeatureClustersError(
                        "completed clustered selection certificates disagree on "
                        "max_features_per_segment"
                    )
            # The splice-certificate parser requires a persisted global width
            # whenever its non-empty per-shard storage-validation domain is
            # present. This transaction has already required that domain to
            # equal the non-empty packed domain above, and every current-run
            # insertion updates the width in the same checkpoint operation.
            # Therefore a missing width here is an internal invariant failure,
            # not a recoverable legacy state.
            assert progress_validated_max_features is not None
            remote_loader_max_features = progress_validated_max_features
            validation_segment_counts = {
                stem: stats.total_packed_segments for stem, stats in progress_splice_stats.items()
            }
            logger.info(
                "splice_counterfactual: every serialized selection shard was "
                "validated and bound to its exact compressed digest; "
                "skipped final packed-row body replay"
            )
        elif remote_loader_max_features is None:
            assert producer_packed_resolution is not None
            assert progress_integrity is not None
            remote_loader_max_features = _validate_clustered_selection_outputs(
                output_dir=output_dir,
                base_shard_paths=base_shard_paths,
                packed_resolution=producer_packed_resolution,
                expected_selection_integrity=progress_integrity,
                expected_packed_rows_hash=selection_packed_rows_hash,
                feature_clusters=feature_clusters,
                effective_buckets=pipeline.effective_buckets,
                segment_counts_by_stem=validation_segment_counts,
                max_features_by_stem=validation_max_features_by_stem,
            )
    if complete_certified_splice_noop:
        logger.info(
            "splice_counterfactual resume: all exact selection bytes, per-shard "
            "stats, and prior output validation are current; "
            + (
                "reused the remote-loader certificate and skipped body reads"
                if reused_remote_loader_certificate
                else "reused the signed splice v3 width and skipped body reads"
            )
        )

    # --- Write manifest for the sel/ dir (defense-in-depth: I4) ---
    # Records the canonical feature_mask so that training-time discovery can
    # assert the loaded sel/ dir was built with the same mask (guards hash
    # collisions and moved/mislabeled dirs).
    # Back-compat (I5): when no mask is configured, omit the key entirely —
    # do NOT write `feature_mask: null`.  Legacy sel/ dirs without this key
    # plus no configured mask must keep working byte-identically.
    completed_validation_evidence: _CompletedClusteredValidationEvidence | None = None
    sel_metadata = _base_selection_metadata(
        feature_mask=feature_mask,
        feature_clusters=feature_clusters,
        grad_spec=grad_spec,
        grad_overlay_checksum=grad_overlay_checksum,
        packed_rows_hash=selection_packed_rows_hash,
        status="complete" if feature_clusters is not None else None,
    )
    if is_splice_counterfactual:
        assert packed_info is not None
        assert splice_sidecar_hash is not None
        assert splice_derived_feature_hash is not None
        assert splice_boundary_hash is not None
        sel_metadata = _splice_selection_metadata(
            pipeline=pipeline,
            packed_info=packed_info,
            splice_stats=splice_stats,
            splice_sidecar_hash=splice_sidecar_hash,
            splice_derived_feature_hash=splice_derived_feature_hash,
            splice_boundary_hash=splice_boundary_hash,
            feature_mask=feature_mask,
            feature_clusters=feature_clusters,
            grad_spec=grad_spec,
            grad_overlay_checksum=grad_overlay_checksum,
            status="complete",
            splice_batch_size=selection_splice_batch_size,
        )
    if feature_clusters is not None:
        if progress_integrity is None:
            progress_integrity = build_clustered_selection_shard_integrity(
                glob_overlay_shards(output_dir)
            )
        final_integrity_stems = {
            entry["name"].removesuffix(_SELECTION_SHARD_SUFFIX)
            for entry in progress_integrity["shards"]
        }
        if final_integrity_stems != base_stem_set:
            raise FeatureClustersError(
                "final clustered selection integrity does not exactly cover the "
                "packed-row shard domain"
            )
        sel_metadata[SELECTION_SHARD_INTEGRITY_KEY] = progress_integrity
        assert remote_loader_max_features is not None
        sel_metadata[REMOTE_LOADER_KEY] = build_remote_loader_certificate(
            selection_checksum=selection_hash,
            selection_inventory_aggregate=progress_integrity["aggregate_sha256"],
            packed_row_fingerprint=selection_packed_rows_hash,
            max_features_per_segment=remote_loader_max_features,
            cluster_delta_semantics=(
                CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS if is_splice_counterfactual else None
            ),
        )
        if is_splice_counterfactual:
            if set(progress_splice_stats) != base_stem_set:
                raise FeatureClustersError(
                    "final splice resume stats do not exactly cover the packed-row shard domain"
                )
            sel_metadata[SPLICE_RESUME_METADATA_KEY] = _build_splice_resume_metadata(
                stats_by_stem=progress_splice_stats,
                selection_integrity=progress_integrity,
                storage_validated_by_stem=progress_storage_validated,
                validated_max_features_per_segment=(progress_validated_max_features),
                packed_rows_hash=selection_packed_rows_hash,
                effective_buckets=pipeline.effective_buckets,
                status="complete",
            )
        if set(validation_segment_counts) != base_stem_set:
            raise FeatureClustersError(
                "final clustered semantic-validation segment counts do not "
                "exactly cover the packed-row shard domain"
            )
        assert producer_packed_resolution is not None
        completed_validation_evidence = _build_completed_clustered_selection_validation(
            output_dir=output_dir,
            packed_proof=producer_packed_resolution,
            selection_hash=selection_hash,
            selection_integrity=progress_integrity,
            packed_rows_hash=selection_packed_rows_hash,
            feature_clusters=feature_clusters,
            feature_delta=(
                canonical_feature_delta_dict(pipeline.delta_bundle.config)
                if pipeline.delta_bundle is not None
                else None
            ),
            effective_buckets=pipeline.effective_buckets,
            expected_segments_by_stem=validation_segment_counts,
            validated_max_features_by_stem=(validation_max_features_by_stem),
            prior_metadata=prior_completed_cluster_metadata,
        )
        sel_metadata[CLUSTERED_SELECTION_VALIDATION_KEY] = completed_validation_evidence.certificate
        # Resolve the fully assembled document through the exact consumer seam
        # before completed-manifest publication. This makes the
        # schema-v1, splice-v3, and remote-loader proofs one transaction: a
        # resumed in-progress global width (or any other cross-proof field)
        # cannot disagree with the freshly validated per-shard evidence.
        resolve_clustered_selection_validation(
            sel_metadata,
            selection_checksum=selection_hash,
            selection_integrity=progress_integrity,
            packed_row_fingerprint=selection_packed_rows_hash,
            feature_clusters_descriptor=(feature_clusters.resolved.descriptor_dict()),
            feature_delta=(
                canonical_feature_delta_dict(pipeline.delta_bundle.config)
                if pipeline.delta_bundle is not None
                else None
            ),
            effective_buckets=pipeline.effective_buckets,
            surface=f"completed clustered selection manifest {output_dir}",
        )
    if completed_validation_evidence is not None:
        _assert_completed_clustered_validation_evidence_current(
            completed_validation_evidence,
            output_dir=output_dir,
            packed_row_dir=packed_row_dir,
        )
    write_manifest(
        output_dir,
        selection_hash,
        OverlayKind.PACKED_ROW,
        metadata=sel_metadata,
        base_shard_paths=base_shard_paths,
        cache_dir=packed_row_dir,
    )

    logger.info(f"Selection shards at: {output_dir}")


if __name__ == "__main__":
    main()
