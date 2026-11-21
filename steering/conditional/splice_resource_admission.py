"""Fail-closed resource admission for parallel splice preselection.

Parallel splice selection combines several individually large representations:
eager packed-row buffers, source-feature decode transients, optional shared
tmpfs mappings, provenance tensors, compact sidecars, and Python selection
state.  Worker count and store directory-count are therefore not sufficient
resource bounds.

This module derives exact fixed tensor/file sizes from validated safetensors
headers and combines them with an explicit, versioned measurement-derived
private-memory envelope.  A pass means the requested configuration is inside
that supported envelope at the instant sampled; it is deliberately *not* a
mathematical guarantee against later unrelated host pressure.  A configuration
outside the envelope is rejected before output or shared-store creation.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NotRequired, TypedDict, Unpack

import psutil
from beartype import beartype
from phantom.sized import NonEmptyStr

from steering.conditional.feature_index import FeatureIndexCatalog
from steering.feature_shard_splice import PROVENANCE_DIRNAME
from steering.gradient_overlay import (
    PROVENANCE_BASE_SHARD_IDX_KEY,
    PROVENANCE_DOC_IDX_KEY,
    PROVENANCE_SEG_ORDINAL_KEY,
)
from steering.refined_types import (
    AtLeastTwoInt,
    NonNegativeInt,
    PartsPerMillion,
    PositiveInt,
)
from steering.safetensors_layout import (
    CompressedSafetensorsLayout,
    SafetensorsLayoutError,
    SemanticSafetensorsLayout,
    TensorLayout,
    read_compressed_safetensors_layout,
    semantic_layout_from_compressed,
)

SafetensorsLayout = CompressedSafetensorsLayout | SemanticSafetensorsLayout

MIB = 1024**2
GIB = 1024**3
ADMISSION_SCHEMA_VERSION = 1
_PACKED_DTYPES: dict[str, tuple[str, int]] = {
    "tokens": ("U16", 2),
    "row_seg_offsets": ("I32", 1),
    "seg_token_start": ("U16", 1),
    "seg_token_end": ("U16", 1),
    "seg_doc_id": ("U8", 1),
    "seg_feat_offsets": ("I32", 1),
    "feature_ids": ("U16", 1),
    "feature_pcts": ("U8", 1),
}
_SOURCE_SEGMENT_KEYS = (
    "doc_ids",
    "segment_indices",
    "segment_positions",
    "segment_char_starts",
    "segment_char_ends",
)
_INTEGER_DTYPES = frozenset({"U8", "I8", "U16", "I16", "U32", "I32", "U64", "I64"})
_TMPFS_TYPES = frozenset({"tmpfs", "ramfs"})
_MOUNT_ESCAPE_RE = re.compile(r"\\([0-7]{3})")


class SpliceResourceAdmissionError(RuntimeError):
    """The requested parallel configuration is outside the supported envelope."""

    def __init__(self, report: SpliceResourceAdmissionReport):
        self.report = report
        joined = "; ".join(report.violations)
        super().__init__(
            "parallel splice resource admission rejected the requested "
            f"configuration: {joined}. Report: {report.to_json()}"
        )


@beartype
@dataclass(frozen=True)
class AdmissionPolicy:
    """Versioned constants for non-structural admission reserves.

    ``worker_packed_ratio`` comes from the production W=12 sample: private
    worker memory beyond the retained packed buffer was roughly 0.6--0.7 of
    packed serialized bytes.  The 0.75 envelope plus an explicit sidecar term
    is intentionally conservative.  Tiny fixtures use the floor rather than a
    multi-GiB constant so the model scales with actual artifacts.
    """

    store_free_reserve_bytes: NonNegativeInt = NonNegativeInt.parse(512 * MIB)
    spool_min_free_reserve_bytes: NonNegativeInt = NonNegativeInt.parse(512 * MIB)
    spool_max_free_reserve_bytes: NonNegativeInt = NonNegativeInt.parse(8 * GIB)
    spool_free_reserve_fraction_ppm: PartsPerMillion = PartsPerMillion.parse(50_000)
    host_min_reserve_bytes: NonNegativeInt = NonNegativeInt.parse(512 * MIB)
    host_max_reserve_bytes: NonNegativeInt = NonNegativeInt.parse(8 * GIB)
    host_reserve_fraction_ppm: PartsPerMillion = PartsPerMillion.parse(100_000)
    worker_private_floor_bytes: NonNegativeInt = NonNegativeInt.parse(64 * MIB)
    worker_packed_ratio_numerator: NonNegativeInt = NonNegativeInt.parse(3)
    worker_packed_ratio_denominator: PositiveInt = PositiveInt.parse(4)
    parent_private_reserve_bytes: NonNegativeInt = NonNegativeInt.parse(256 * MIB)
    sidecar_file_size_multiplier: NonNegativeInt = NonNegativeInt.parse(1)

    def validate(self) -> None:
        """Validate relationships that cannot be expressed by individual field types."""
        if self.spool_min_free_reserve_bytes > self.spool_max_free_reserve_bytes:
            raise ValueError("spool minimum reserve exceeds spool maximum reserve")
        if self.host_min_reserve_bytes > self.host_max_reserve_bytes:
            raise ValueError("host minimum reserve exceeds host maximum reserve")


@dataclass(frozen=True)
class MountInfo:
    mount_point: str
    fs_type: str


@beartype
@dataclass(frozen=True)
class FilesystemCapacity:
    path: NonEmptyStr
    mount_point: NonEmptyStr
    fs_type: NonEmptyStr
    available_bytes: NonNegativeInt
    block_size: PositiveInt


@beartype
@dataclass(frozen=True)
class CapacitySnapshot:
    host_mem_available_bytes: NonNegativeInt
    cgroup_memory_available_bytes: NonNegativeInt | None
    effective_memory_available_bytes: NonNegativeInt
    shared_store: FilesystemCapacity | None
    packed_spool: FilesystemCapacity | None


class SpliceResourceAdmissionInputs(TypedDict):
    """Statically checked inputs shared by the evaluate-and-raise wrapper."""

    packed_shard_paths: list[Any]
    provenance_shard_paths: list[Any]
    base_shard_stems: list[Any]
    feature_index_catalog: FeatureIndexCatalog
    n_sae_features: PositiveInt
    skip_feature_mags: bool
    num_workers: AtLeastTwoInt
    source_cache_shards: PositiveInt
    sidecar_cache_shards: PositiveInt
    has_selection_work: bool
    shared_store_enabled: bool
    shared_store_capacity: NonNegativeInt
    packed_spool_dir: str | Path | None
    shared_store_root: str | Path | None
    capacities: NotRequired[CapacitySnapshot | None]
    policy: NotRequired[AdmissionPolicy | None]


@dataclass(frozen=True)
class SourceStemEstimate:
    stem: str
    compressed_input_bytes: int
    serialized_input_bytes: int
    shared_tensor_bytes: int
    shared_allocated_bytes: int
    shared_doc_index_tensor_bytes: int
    shared_doc_index_build_peak_bytes: int
    private_resident_tensor_bytes: int
    materialization_private_peak_bytes: int
    private_build_peak_bytes: int


@dataclass(frozen=True)
class SourceLayoutCounts:
    """Semantic counts proven by one raw/percentile layout pair."""

    documents: int
    segments: int
    features: int


@dataclass(frozen=True)
class SpliceResourceAdmissionReport:
    schema_version: int
    decision: str
    basis: str
    requested_workers: int
    active_workers: int
    has_selection_work: bool
    shared_store_enabled: bool
    shared_store_capacity: int
    packed_storage_mode: str
    exact_bytes: dict[str, int]
    estimated_bytes: dict[str, int]
    capacities: CapacitySnapshot
    source_stems: tuple[SourceStemEstimate, ...]
    policy: AdmissionPolicy
    violations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _decode_mount_field(value: str) -> str:
    return _MOUNT_ESCAPE_RE.sub(lambda match: chr(int(match.group(1), 8)), value)


def parse_mountinfo(text: str, path: str | Path) -> MountInfo:
    """Return the longest matching Linux mountinfo entry for ``path``."""
    target = Path(path).resolve()
    matches: list[MountInfo] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        left, separator, right = line.partition(" - ")
        if not separator:
            raise ValueError(f"mountinfo line {line_number} has no separator")
        left_fields = left.split()
        right_fields = right.split()
        if len(left_fields) < 5 or len(right_fields) < 1:
            raise ValueError(f"mountinfo line {line_number} is truncated")
        mount_point = Path(_decode_mount_field(left_fields[4]))
        try:
            target.relative_to(mount_point)
        except ValueError:
            continue
        matches.append(MountInfo(mount_point=str(mount_point), fs_type=right_fields[0]))
    if not matches:
        raise ValueError(f"no mountinfo entry contains {target}")
    return max(matches, key=lambda item: len(item.mount_point))


def _filesystem_capacity(path: Path, mountinfo_text: str) -> FilesystemCapacity:
    mount = parse_mountinfo(mountinfo_text, path)
    statvfs = os.statvfs(path)
    usage = shutil.disk_usage(path)
    block_size = int(statvfs.f_frsize or statvfs.f_bsize)
    if block_size < 1:
        raise RuntimeError(f"filesystem at {path} reported invalid block size {block_size}")
    return FilesystemCapacity(
        path=NonEmptyStr.parse(str(path.resolve())),
        mount_point=NonEmptyStr.parse(mount.mount_point),
        fs_type=NonEmptyStr.parse(mount.fs_type),
        available_bytes=NonNegativeInt.parse(int(usage.free)),
        block_size=PositiveInt.parse(block_size),
    )


def _read_cgroup_available_bytes(
    *,
    proc_cgroup_path: Path = Path("/proc/self/cgroup"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> int | None:
    try:
        cgroup_text = proc_cgroup_path.read_text()
    except OSError:
        return None
    relative: str | None = None
    for line in cgroup_text.splitlines():
        hierarchy, controllers, path = line.split(":", 2)
        if hierarchy == "0" and controllers == "":
            relative = path
            break
    if relative is None:
        return None
    group = cgroup_root / relative.lstrip("/")
    try:
        raw_max = (group / "memory.max").read_text().strip()
        current = int((group / "memory.current").read_text().strip())
    except (OSError, ValueError):
        return None
    if raw_max == "max":
        return None
    try:
        maximum = int(raw_max)
    except ValueError:
        return None
    return max(0, maximum - current)


def capture_capacity_snapshot(
    *,
    shared_store_root: str | Path | None,
    packed_spool_dir: str | Path | None,
    mountinfo_path: Path = Path("/proc/self/mountinfo"),
) -> CapacitySnapshot:
    """Capture current host/cgroup/filesystem capacity without mutating it."""
    try:
        mountinfo_text = mountinfo_path.read_text()
    except OSError as exc:
        raise RuntimeError(f"cannot read Linux mountinfo at {mountinfo_path}: {exc}") from exc
    host_available = int(psutil.virtual_memory().available)
    if host_available < 0:
        raise RuntimeError(f"psutil reported negative MemAvailable: {host_available}")
    cgroup_available = _read_cgroup_available_bytes()
    effective = (
        min(host_available, cgroup_available) if cgroup_available is not None else host_available
    )
    shared = (
        _filesystem_capacity(Path(shared_store_root), mountinfo_text)
        if shared_store_root is not None
        else None
    )
    spool = (
        _filesystem_capacity(Path(packed_spool_dir), mountinfo_text)
        if packed_spool_dir is not None
        else None
    )
    return CapacitySnapshot(
        host_mem_available_bytes=NonNegativeInt.parse(host_available),
        cgroup_memory_available_bytes=(
            NonNegativeInt.parse(cgroup_available) if cgroup_available is not None else None
        ),
        effective_memory_available_bytes=NonNegativeInt.parse(effective),
        shared_store=shared,
        packed_spool=spool,
    )


def _require_tensor(
    layout: SafetensorsLayout,
    name: str,
    *,
    dtypes: frozenset[str],
    rank: int,
) -> TensorLayout:
    try:
        tensor = layout.tensor(name)
    except KeyError as exc:
        raise SafetensorsLayoutError(f"{layout.path}: missing tensor {name!r}") from exc
    if tensor.dtype not in dtypes or len(tensor.shape) != rank:
        raise SafetensorsLayoutError(
            f"{layout.path}: tensor {name!r} expected rank {rank} and dtype in "
            f"{sorted(dtypes)}, got shape={tensor.shape}, dtype={tensor.dtype}"
        )
    return tensor


def validate_packed_layout(
    layout: SafetensorsLayout,
) -> tuple[int, int]:
    if layout.tensor_names != frozenset(_PACKED_DTYPES):
        raise SafetensorsLayoutError(
            f"{layout.path}: packed tensor keys are {sorted(layout.tensor_names)}, "
            f"expected {sorted(_PACKED_DTYPES)}"
        )
    tensors: dict[str, TensorLayout] = {}
    for name, (dtype, rank) in _PACKED_DTYPES.items():
        tensors[name] = _require_tensor(layout, name, dtypes=frozenset({dtype}), rank=rank)
    tokens = tensors["tokens"]
    rows = tokens.shape[0]
    if tokens.shape[1] < 2:
        raise SafetensorsLayoutError(f"{layout.path}: packed tokens width must be >= 2")
    n_segments = tensors["seg_token_start"].numel
    n_features = tensors["feature_ids"].numel
    expected_shapes = {
        "row_seg_offsets": (rows + 1,),
        "seg_token_end": (n_segments,),
        "seg_doc_id": (n_segments,),
        "seg_feat_offsets": (n_segments + 1,),
        "feature_pcts": (n_features,),
    }
    for name, expected in expected_shapes.items():
        if tensors[name].shape != expected:
            raise SafetensorsLayoutError(
                f"{layout.path}: packed tensor {name!r} has shape "
                f"{tensors[name].shape}, expected {expected}"
            )
    return n_segments, n_features


def _validate_provenance_layout(layout: SafetensorsLayout, *, expected_segments: int) -> None:
    expected_names = frozenset(
        {
            PROVENANCE_BASE_SHARD_IDX_KEY,
            PROVENANCE_DOC_IDX_KEY,
            PROVENANCE_SEG_ORDINAL_KEY,
        }
    )
    if layout.tensor_names != expected_names:
        raise SafetensorsLayoutError(
            f"{layout.path}: provenance tensor keys are {sorted(layout.tensor_names)}, "
            f"expected {sorted(expected_names)}"
        )
    for name in expected_names:
        tensor = _require_tensor(layout, name, dtypes=frozenset({"I32"}), rank=1)
        if tensor.shape != (expected_segments,):
            raise SafetensorsLayoutError(
                f"{layout.path}: provenance tensor {name!r} has shape "
                f"{tensor.shape}, expected {(expected_segments,)}"
            )


def _metadata_int(layout: SafetensorsLayout, key: str) -> int:
    metadata = dict(layout.metadata)
    raw = metadata.get(key)
    try:
        value = int(raw) if raw is not None else None
    except ValueError as exc:
        raise SafetensorsLayoutError(
            f"{layout.path}: metadata {key!r} is not an integer: {raw!r}"
        ) from exc
    if value is None or value < 0:
        raise SafetensorsLayoutError(
            f"{layout.path}: metadata {key!r} must be an integer >= 0, got {raw!r}"
        )
    return value


def _round_file_bytes(size: int, block_size: int) -> int:
    if size == 0:
        return 0
    return ((size + block_size - 1) // block_size) * block_size


def validate_source_layout_pair(
    base: SafetensorsLayout,
    percentile: SafetensorsLayout,
) -> SourceLayoutCounts:
    """Apply the source invariants shared by local and remote admission."""
    offsets = _require_tensor(base, "offsets", dtypes=frozenset({"I32", "I64"}), rank=1)
    feature_ids = _require_tensor(
        base,
        "feature_ids",
        dtypes=frozenset({"U16", "I16", "I32"}),
        rank=1,
    )
    feature_mags = _require_tensor(base, "feature_mags", dtypes=frozenset({"F16", "F32"}), rank=1)
    if feature_mags.numel != feature_ids.numel:
        raise SafetensorsLayoutError(
            f"{base.path}: feature_mags length {feature_mags.numel} does not "
            f"match feature_ids length {feature_ids.numel}"
        )
    n_segments = _metadata_int(base, "n_segments")
    n_docs = _metadata_int(base, "n_docs")
    if offsets.shape != (n_segments + 1,):
        raise SafetensorsLayoutError(
            f"{base.path}: offsets shape {offsets.shape} does not match n_segments={n_segments}"
        )
    segment_tensors: list[TensorLayout] = []
    for name in _SOURCE_SEGMENT_KEYS:
        tensor = _require_tensor(base, name, dtypes=_INTEGER_DTYPES, rank=1)
        if tensor.shape != (n_segments,):
            raise SafetensorsLayoutError(
                f"{base.path}: {name} shape {tensor.shape} does not match n_segments={n_segments}"
            )
        segment_tensors.append(tensor)

    if percentile.tensor_names != frozenset({"feature_percentiles"}):
        raise SafetensorsLayoutError(
            f"{percentile.path}: percentile overlay keys are "
            f"{sorted(percentile.tensor_names)}, expected ['feature_percentiles']"
        )
    percentiles = _require_tensor(
        percentile, "feature_percentiles", dtypes=frozenset({"U8"}), rank=1
    )
    if percentiles.numel != feature_ids.numel:
        raise SafetensorsLayoutError(
            f"{percentile.path}: percentile length {percentiles.numel} does not "
            f"match source feature_ids length {feature_ids.numel}"
        )
    return SourceLayoutCounts(
        documents=n_docs,
        segments=n_segments,
        features=feature_ids.numel,
    )


def _source_estimate(
    *,
    stem: str,
    base: SafetensorsLayout,
    percentile: SafetensorsLayout,
    n_sae_features: int,
    skip_feature_mags: bool,
    shared_block_size: int,
    include_shared_doc_index: bool,
) -> SourceStemEstimate:
    counts = validate_source_layout_pair(base, percentile)
    offsets = base.tensor("offsets")
    feature_ids = base.tensor("feature_ids")
    feature_mags = base.tensor("feature_mags")
    percentiles = percentile.tensor("feature_percentiles")
    n_segments = counts.segments
    segment_tensors = [base.tensor(name) for name in _SOURCE_SEGMENT_KEYS]

    # The resolved SAE vocabulary is the producer/consumer contract.  It proves
    # every valid ID fits uint16 only at widths <= 65,536.  Wider vocabularies
    # are costed at int32 even if one shard happens to contain only small IDs.
    fid_runtime_bytes = feature_ids.numel * (2 if n_sae_features <= 65_536 else 4)
    offsets_runtime_bytes = offsets.numel * 8
    mags_runtime_bytes = 0 if skip_feature_mags else feature_mags.numel * 4
    # Issue #758: shared workers publish the stable eager row order and its
    # sorted int64 doc IDs.  Both have exactly n_segments elements, so their
    # payload and tmpfs allocation are header-derived rather than sampled.
    shared_doc_index_tensor_bytes = 2 * n_segments * 8 if include_shared_doc_index else 0
    shared_doc_index_build_peak_bytes = 40 * n_segments if include_shared_doc_index else 0
    shared_tensors = [
        offsets_runtime_bytes,
        fid_runtime_bytes,
        mags_runtime_bytes,
        percentiles.nbytes,
        *(tensor.nbytes for tensor in segment_tensors),
        *((n_segments * 8, n_segments * 8) if include_shared_doc_index else ()),
    ]
    shared_tensor_bytes = sum(shared_tensors)
    # One raw tensor file per nonempty tensor plus manifest and directory
    # allowance.  The store's files are fully written, so block-rounding is the
    # relevant tmpfs allocation unit rather than sparse logical length.
    shared_allocated_bytes = (
        sum(_round_file_bytes(size, shared_block_size) for size in shared_tensors)
        + 2 * shared_block_size
    )

    private_resident_tensor_bytes = fid_runtime_bytes + mags_runtime_bytes + percentiles.nbytes
    offset_allocation = offsets_runtime_bytes if offsets.dtype == "I32" else 0
    mags_allocation = (
        mags_runtime_bytes if not skip_feature_mags and feature_mags.dtype == "F16" else 0
    )
    final_allocations = offset_allocation + fid_runtime_bytes + mags_allocation

    # load_features holds compressed + complete decompressed bytes while the
    # delta decoder can hold two int32 ID arrays (upcast input + decoded clone).
    base_decode_peak = (
        base.compressed_bytes
        + base.serialized_bytes
        + 2 * feature_ids.numel * 4
        + offset_allocation
        + mags_allocation
    )
    overlay_peak = (
        base.serialized_bytes
        + final_allocations
        + percentile.compressed_bytes
        + percentile.serialized_bytes
    )
    publish_private_peak = base.serialized_bytes + percentile.serialized_bytes + final_allocations
    materialization_private_peak = max(
        base_decode_peak,
        overlay_peak,
        publish_private_peak,
        publish_private_peak + shared_doc_index_build_peak_bytes,
    )
    private_build_peak = max(
        base_decode_peak,
        overlay_peak + private_resident_tensor_bytes,
        publish_private_peak + private_resident_tensor_bytes,
    )
    return SourceStemEstimate(
        stem=stem,
        compressed_input_bytes=base.compressed_bytes + percentile.compressed_bytes,
        serialized_input_bytes=base.serialized_bytes + percentile.serialized_bytes,
        shared_tensor_bytes=shared_tensor_bytes,
        shared_allocated_bytes=shared_allocated_bytes,
        shared_doc_index_tensor_bytes=shared_doc_index_tensor_bytes,
        shared_doc_index_build_peak_bytes=shared_doc_index_build_peak_bytes,
        private_resident_tensor_bytes=private_resident_tensor_bytes,
        materialization_private_peak_bytes=materialization_private_peak,
        private_build_peak_bytes=private_build_peak,
    )


def _sum_largest(values: list[int], count: int) -> int:
    if count <= 0:
        return 0
    return sum(sorted(values, reverse=True)[:count])


def _bounded_fraction(
    value: int, *, numerator: int, denominator: int, minimum: int, maximum: int
) -> int:
    scaled = (value * numerator + denominator - 1) // denominator
    return min(maximum, max(minimum, scaled))


def _validate_capacity_relationships(snapshot: CapacitySnapshot) -> None:
    cgroup = snapshot.cgroup_memory_available_bytes
    expected_effective = (
        min(snapshot.host_mem_available_bytes, cgroup)
        if cgroup is not None
        else snapshot.host_mem_available_bytes
    )
    if snapshot.effective_memory_available_bytes != expected_effective:
        raise ValueError(
            "CapacitySnapshot.effective_memory_available_bytes must equal the "
            "minimum of host and finite cgroup availability"
        )


def evaluate_splice_resource_admission_from_layouts(
    *,
    packed_layouts: Sequence[SafetensorsLayout],
    provenance_layouts: Sequence[SafetensorsLayout],
    source_layouts: Sequence[tuple[str, SafetensorsLayout, SafetensorsLayout]],
    sidecar_sizes: Sequence[int],
    n_sae_features: int,
    skip_feature_mags: bool,
    requested_workers: int,
    source_cache_shards: int,
    sidecar_cache_shards: int,
    has_selection_work: bool,
    requires_packed_runtime: bool,
    requires_provenance_runtime: bool,
    requires_source_runtime: bool,
    shared_store_enabled: bool,
    shared_store_capacity: int,
    packed_spool_enabled: bool,
    capacities: CapacitySnapshot,
    policy: AdmissionPolicy,
) -> SpliceResourceAdmissionReport:
    """Evaluate the common local/remote envelope from semantic layouts only."""
    policy.validate()
    _validate_capacity_relationships(capacities)
    for label, value in (
        ("n_sae_features", n_sae_features),
        ("requested_workers", requested_workers),
        ("source_cache_shards", source_cache_shards),
        ("sidecar_cache_shards", sidecar_cache_shards),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a strict integer >= 1")
    if not all(
        isinstance(value, bool)
        for value in (
            skip_feature_mags,
            has_selection_work,
            requires_packed_runtime,
            requires_provenance_runtime,
            requires_source_runtime,
            shared_store_enabled,
            packed_spool_enabled,
        )
    ):
        raise ValueError("admission mode switches must be booleans")
    if len(packed_layouts) != len(provenance_layouts) or not packed_layouts:
        raise ValueError("packed/provenance layouts must be non-empty and pair exactly")
    if shared_store_enabled and shared_store_capacity < 1:
        raise ValueError("shared_store_capacity must be >= 1 when the store is enabled")
    if not shared_store_enabled and shared_store_capacity != 0:
        raise ValueError("shared_store_capacity must be 0 when the store is disabled")
    if shared_store_enabled and capacities.shared_store is None:
        raise ValueError("capacity snapshot is missing shared-store filesystem data")
    if packed_spool_enabled and capacities.packed_spool is None:
        raise ValueError("capacity snapshot is missing packed-spool filesystem data")
    if any(
        isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in sidecar_sizes
    ):
        raise ValueError("sidecar sizes must be non-negative strict integers")

    semantic_packed: list[SemanticSafetensorsLayout] = []
    semantic_provenance: list[SemanticSafetensorsLayout] = []
    for packed, provenance in zip(packed_layouts, provenance_layouts, strict=True):
        packed_semantic = (
            semantic_layout_from_compressed(packed)
            if isinstance(packed, CompressedSafetensorsLayout)
            else packed
        )
        provenance_semantic = (
            semantic_layout_from_compressed(provenance)
            if isinstance(provenance, CompressedSafetensorsLayout)
            else provenance
        )
        n_segments, _n_features = validate_packed_layout(packed_semantic)
        _validate_provenance_layout(provenance_semantic, expected_segments=n_segments)
        semantic_packed.append(packed_semantic)
        semantic_provenance.append(provenance_semantic)

    if has_selection_work and requires_source_runtime and not source_layouts:
        raise ValueError("selection work requires at least one source layout pair")
    shared_block_size = (
        capacities.shared_store.block_size if capacities.shared_store is not None else 4096
    )
    source_estimates = (
        [
            _source_estimate(
                stem=stem,
                base=base,
                percentile=percentile,
                n_sae_features=n_sae_features,
                skip_feature_mags=skip_feature_mags,
                shared_block_size=shared_block_size,
                include_shared_doc_index=shared_store_enabled,
            )
            for stem, base, percentile in source_layouts
        ]
        if has_selection_work and requires_source_runtime
        else []
    )

    active_workers = min(requested_workers, len(semantic_packed))
    runtime_workers = active_workers if requires_packed_runtime else 0
    packed_serialized = [layout.serialized_bytes for layout in semantic_packed]
    packed_eager_peaks = [
        layout.compressed_bytes + layout.serialized_bytes for layout in semantic_packed
    ]
    provenance_peaks = [
        layout.compressed_bytes + layout.serialized_bytes for layout in semantic_provenance
    ]
    packed_resident_bytes = _sum_largest(packed_serialized, runtime_workers)
    packed_load_peak_bytes = (
        packed_resident_bytes
        if packed_spool_enabled
        else _sum_largest(packed_eager_peaks, runtime_workers)
    )
    provenance_runtime_workers = active_workers if requires_provenance_runtime else 0
    provenance_load_peak_bytes = _sum_largest(
        provenance_peaks, provenance_runtime_workers
    )
    per_worker_sidecar_estimate = (
        _sum_largest(list(sidecar_sizes), min(sidecar_cache_shards, len(sidecar_sizes)))
        * policy.sidecar_file_size_multiplier
    )
    sidecar_cache_estimate = (
        active_workers * per_worker_sidecar_estimate
        if requires_source_runtime
        else 0
    )

    shared_store_bytes = 0
    shared_store_tensor_bytes = 0
    private_source_cache_bytes = 0
    source_materialization_bytes = 0
    store_union_stems = 0
    if has_selection_work and requires_source_runtime and source_estimates:
        if shared_store_enabled:
            store_union_stems = min(
                len(source_estimates),
                shared_store_capacity + 2 * active_workers + 1,
            )
            shared_store_bytes = _sum_largest(
                [item.shared_allocated_bytes for item in source_estimates],
                store_union_stems,
            )
            shared_store_tensor_bytes = _sum_largest(
                [item.shared_tensor_bytes for item in source_estimates],
                store_union_stems,
            )
            source_materialization_bytes = max(
                item.materialization_private_peak_bytes for item in source_estimates
            )
        else:
            cache_stems = min(source_cache_shards, len(source_estimates))
            per_worker_resident = _sum_largest(
                [item.private_resident_tensor_bytes for item in source_estimates],
                cache_stems,
            )
            per_worker_build = max(item.private_build_peak_bytes for item in source_estimates)
            private_source_cache_bytes = active_workers * per_worker_resident
            source_materialization_bytes = active_workers * per_worker_build

    largest_packed = max(packed_serialized)
    per_worker_private = max(
        policy.worker_private_floor_bytes,
        (
            largest_packed * policy.worker_packed_ratio_numerator
            + policy.worker_packed_ratio_denominator
            - 1
        )
        // policy.worker_packed_ratio_denominator,
    )
    worker_private_estimate = runtime_workers * per_worker_private
    estimated_incremental = sum(
        (
            packed_load_peak_bytes,
            provenance_load_peak_bytes,
            shared_store_bytes,
            private_source_cache_bytes,
            source_materialization_bytes,
            sidecar_cache_estimate,
            worker_private_estimate,
            policy.parent_private_reserve_bytes,
        )
    )
    host_reserve = _bounded_fraction(
        capacities.effective_memory_available_bytes,
        numerator=policy.host_reserve_fraction_ppm,
        denominator=1_000_000,
        minimum=policy.host_min_reserve_bytes,
        maximum=policy.host_max_reserve_bytes,
    )
    exact_bytes = {
        "packed_serialized_resident": packed_resident_bytes,
        "packed_eager_compressed_plus_serialized": _sum_largest(
            packed_eager_peaks, runtime_workers
        ),
        "provenance_compressed_plus_serialized": provenance_load_peak_bytes,
        "shared_store_union_tensor_payload": shared_store_tensor_bytes,
        "shared_store_union_stems": store_union_stems,
        "packed_spool_required": packed_resident_bytes if packed_spool_enabled else 0,
    }
    estimated_bytes = {
        "packed_memory_peak": packed_load_peak_bytes,
        "provenance_memory_peak": provenance_load_peak_bytes,
        "private_source_cache": private_source_cache_bytes,
        "source_materialization_peak": source_materialization_bytes,
        "shared_store_union_filesystem": shared_store_bytes,
        "compact_sidecar_cache": sidecar_cache_estimate,
        "worker_private_residual": worker_private_estimate,
        "parent_private_reserve": policy.parent_private_reserve_bytes,
        "incremental_peak_before_host_reserve": estimated_incremental,
        "host_os_cgroup_reserve": host_reserve,
        "required_with_host_reserve": estimated_incremental + host_reserve,
    }

    violations: list[str] = []
    if shared_store_enabled:
        assert capacities.shared_store is not None
        store_required = shared_store_bytes + policy.store_free_reserve_bytes
        if store_required > capacities.shared_store.available_bytes:
            violations.append(
                "shared-store filesystem cannot fit the unlink-aware source "
                f"union plus reserve ({store_required} required, "
                f"{capacities.shared_store.available_bytes} available)"
            )
    if packed_spool_enabled:
        assert capacities.packed_spool is not None
        if capacities.packed_spool.fs_type in _TMPFS_TYPES:
            violations.append(
                "selection_packed_spool_dir is memory-backed "
                f"({capacities.packed_spool.fs_type}); use a disk-backed filesystem"
            )
        spool_reserve = _bounded_fraction(
            capacities.packed_spool.available_bytes,
            numerator=policy.spool_free_reserve_fraction_ppm,
            denominator=1_000_000,
            minimum=policy.spool_min_free_reserve_bytes,
            maximum=policy.spool_max_free_reserve_bytes,
        )
        spool_required = packed_resident_bytes + spool_reserve
        estimated_bytes["packed_spool_free_reserve"] = spool_reserve
        estimated_bytes["packed_spool_required_with_reserve"] = spool_required
        if spool_required > capacities.packed_spool.available_bytes:
            violations.append(
                "packed spool filesystem cannot fit one concurrent shard per "
                f"active worker plus reserve ({spool_required} required, "
                f"{capacities.packed_spool.available_bytes} available)"
            )
    if estimated_incremental + host_reserve > capacities.effective_memory_available_bytes:
        violations.append(
            "current host/cgroup memory headroom is below the supported "
            f"parallel envelope ({estimated_incremental + host_reserve} required, "
            f"{capacities.effective_memory_available_bytes} available)"
        )

    return SpliceResourceAdmissionReport(
        schema_version=ADMISSION_SCHEMA_VERSION,
        decision="reject" if violations else "admit",
        basis=(
            "exact header-declared fixed payloads plus a conservative, "
            "measurement-derived private-memory envelope; admission is a "
            "point-in-time support check, not a guarantee against later "
            "unrelated host pressure"
        ),
        requested_workers=requested_workers,
        active_workers=active_workers,
        has_selection_work=has_selection_work,
        shared_store_enabled=shared_store_enabled,
        shared_store_capacity=shared_store_capacity,
        packed_storage_mode="disk_spool" if packed_spool_enabled else "eager",
        exact_bytes=exact_bytes,
        estimated_bytes=estimated_bytes,
        capacities=capacities,
        source_stems=tuple(source_estimates),
        policy=policy,
        violations=tuple(violations),
    )


# Beartype's implemented container strategy samples elements, so the three
# collection parameters stay outer-container-only here and are scanned
# exhaustively below. Scalar/refined parameters are fully checked by beartype.
@beartype
def evaluate_splice_resource_admission(
    *,
    packed_shard_paths: list[Any],
    provenance_shard_paths: list[Any],
    base_shard_stems: list[Any],
    feature_index_catalog: FeatureIndexCatalog,
    n_sae_features: PositiveInt,
    skip_feature_mags: bool,
    num_workers: AtLeastTwoInt,
    source_cache_shards: PositiveInt,
    sidecar_cache_shards: PositiveInt,
    has_selection_work: bool,
    shared_store_enabled: bool,
    shared_store_capacity: NonNegativeInt,
    packed_spool_dir: str | Path | None,
    shared_store_root: str | Path | None,
    capacities: CapacitySnapshot | None = None,
    policy: AdmissionPolicy | None = None,
) -> SpliceResourceAdmissionReport:
    """Build a complete admission report without changing filesystem state."""
    if policy is None:
        policy = AdmissionPolicy()
    policy.validate()
    if not all(isinstance(path, Path) for path in packed_shard_paths):
        raise ValueError("packed_shard_paths must contain only pathlib.Path values")
    if not all(isinstance(path, Path) for path in provenance_shard_paths):
        raise ValueError("provenance_shard_paths must contain only pathlib.Path values")
    if not all(isinstance(stem, str) and stem for stem in base_shard_stems):
        raise ValueError("base_shard_stems must contain only non-empty strings")
    if len(set(base_shard_stems)) != len(base_shard_stems):
        raise ValueError("base_shard_stems must not contain duplicates")
    if len(packed_shard_paths) != len(provenance_shard_paths):
        raise ValueError("packed and provenance path counts must match")
    if not packed_shard_paths:
        raise ValueError("parallel splice admission requires at least one packed shard")
    packed_names = [path.name for path in packed_shard_paths]
    provenance_names = [path.name for path in provenance_shard_paths]
    if len(set(packed_names)) != len(packed_names):
        raise ValueError("packed_shard_paths must have unique basenames")
    if len(set(provenance_names)) != len(provenance_names):
        raise ValueError("provenance_shard_paths must have unique basenames")
    if set(packed_names) != set(provenance_names):
        raise ValueError("packed and provenance shard basenames must match exactly")
    if shared_store_enabled and shared_store_root is None:
        raise ValueError("shared_store_root is required when shared_store_enabled")
    if shared_store_enabled and shared_store_capacity < 1:
        raise ValueError("shared_store_capacity must be >= 1 when the store is enabled")
    if not shared_store_enabled and shared_store_capacity != 0:
        raise ValueError("shared_store_capacity must be 0 when the store is disabled")

    feature_index_catalog.validate()
    if capacities is None:
        capacities = capture_capacity_snapshot(
            shared_store_root=shared_store_root if shared_store_enabled else None,
            packed_spool_dir=packed_spool_dir,
        )
    _validate_capacity_relationships(capacities)
    if shared_store_enabled and capacities.shared_store is None:
        raise ValueError("capacity snapshot is missing shared-store filesystem data")
    if packed_spool_dir is not None and capacities.packed_spool is None:
        raise ValueError("capacity snapshot is missing packed-spool filesystem data")

    packed_layouts: list[SemanticSafetensorsLayout] = []
    provenance_layouts: list[SemanticSafetensorsLayout] = []
    provenance_by_name = {path.name: path for path in provenance_shard_paths}
    for packed_path in packed_shard_paths:
        packed_layout = semantic_layout_from_compressed(
            read_compressed_safetensors_layout(packed_path)
        )
        n_segments, _n_features = validate_packed_layout(packed_layout)
        try:
            provenance_path = provenance_by_name[packed_path.name]
        except KeyError as exc:
            raise ValueError(f"no provenance path matches packed shard {packed_path.name}") from exc
        provenance_layout = semantic_layout_from_compressed(
            read_compressed_safetensors_layout(provenance_path)
        )
        _validate_provenance_layout(provenance_layout, expected_segments=n_segments)
        packed_layouts.append(packed_layout)
        provenance_layouts.append(provenance_layout)

    source_layouts: list[tuple[str, SemanticSafetensorsLayout, SemanticSafetensorsLayout]] = []
    if has_selection_work:
        shard_paths = dict(feature_index_catalog.shard_entries)
        percentile_paths = dict(feature_index_catalog.percentile_overlay_entries)
        if feature_index_catalog.percentile_overlay_dir is None:
            raise SafetensorsLayoutError(
                "parallel splice admission requires a percentile overlay catalog"
            )
        for stem in base_shard_stems:
            try:
                base_path = Path(shard_paths[stem])
                percentile_path = Path(percentile_paths[stem])
            except KeyError as exc:
                raise SafetensorsLayoutError(
                    "FeatureIndex catalog has no complete source/percentile entry "
                    f"for packed base stem {stem!r}"
                ) from exc
            source_layouts.append(
                (
                    stem,
                    semantic_layout_from_compressed(read_compressed_safetensors_layout(base_path)),
                    semantic_layout_from_compressed(
                        read_compressed_safetensors_layout(percentile_path)
                    ),
                )
            )
        feature_index_catalog.validate()

    sidecar_sizes: list[int] = []
    for stem in base_shard_stems:
        path = Path(feature_index_catalog.feature_dir) / PROVENANCE_DIRNAME / f"{stem}.json"
        try:
            sidecar_sizes.append(path.stat().st_size)
        except OSError as exc:
            raise RuntimeError(f"cannot stat splice sidecar {path}: {exc}") from exc
    return evaluate_splice_resource_admission_from_layouts(
        packed_layouts=packed_layouts,
        provenance_layouts=provenance_layouts,
        source_layouts=source_layouts,
        sidecar_sizes=sidecar_sizes,
        n_sae_features=n_sae_features,
        skip_feature_mags=skip_feature_mags,
        requested_workers=num_workers,
        source_cache_shards=source_cache_shards,
        sidecar_cache_shards=sidecar_cache_shards,
        has_selection_work=has_selection_work,
        requires_packed_runtime=True,
        requires_provenance_runtime=True,
        requires_source_runtime=True,
        shared_store_enabled=shared_store_enabled,
        shared_store_capacity=shared_store_capacity,
        packed_spool_enabled=packed_spool_dir is not None,
        capacities=capacities,
        policy=policy,
    )


def admit_splice_resources(
    **kwargs: Unpack[SpliceResourceAdmissionInputs],
) -> SpliceResourceAdmissionReport:
    """Evaluate and raise before side effects when the envelope cannot fit."""
    report = evaluate_splice_resource_admission(**kwargs)
    if report.violations:
        raise SpliceResourceAdmissionError(report)
    return report
