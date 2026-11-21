"""Corpus-wide SAE feature statistics from per-shard feature files.

Aggregates across all *_features.safetensors.zst shard files to produce per-feature
and corpus-level statistics with 10B token extrapolation.

Usage:
    uv run python -m steering.compute_statistics
    uv run python -m steering.compute_statistics feature_dir=path/to/features
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import os
import random
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import hydra
import numpy as np
import torch as t
from loguru import logger
from omegaconf import DictConfig
from torch.utils._python_dispatch import is_in_torch_dispatch_mode
from tqdm import tqdm

from errors import EmptyExtractionSegmentError, ValidationError
from steering.exact_statistics import (
    POSITIVE_FINITE_F16_COUNT,
    ExactMagnitudeCounts,
    ExactStatisticsError,
    WeightedDistribution,
    count_table_nbytes,
)
from steering.feature_health import (
    assert_feature_statistics_not_degenerate,
    firing_health,
)
from steering.feature_io import (
    assert_uniform_ignore_bos,
    assert_uniform_ignore_bos_metadata,
    assert_uniform_l0_tier,
    assert_uniform_l0_tier_metadata,
    decode_feature_safetensors_bytes,
    empty_segment_indices,
    glob_features,
    load_feature_safetensors_bytes,
    load_feature_safetensors_bytes_stable,
    load_features,
    load_features_stable,
    read_shard_header_stable,
)
from steering.feature_paths import (
    PRODUCER_IGNORE_BOS_FEATURES,
    PoolingMode,
    parse_feature_dir,
    resolve_output_dir,
    verify_segment_size,
)
from steering.profiling.feature_statistics import StatisticsProfile, statistics_profiler
from steering.remote_feature_inventory import resolve_remote_feature_prefix
from steering.remote_types import HfBucketId, NormalizedRemotePrefix
from steering.sae_resolver import SAE_WIDTH_TO_NUM_FEATURES, ActivationType, SAEWidth
from steering.segmentation import SegmentMode
from steering.sft.mixture_fingerprint import (
    SFT_MIXTURE_FINGERPRINT_METADATA_KEY,
    _uniform_sft_mixture_fingerprint_json,
    _uniform_sft_mixture_fingerprint_metadata_json,
)
from steering.statistics_checkpoint import (
    LoadedStatisticsCheckpoint,
    ShardInventoryEntry,
    StatisticsCheckpointError,
    StatisticsCheckpointStore,
    assert_shard_inventory_unchanged,
    build_checkpoint_contract,
    capture_shard_inventory,
    fsync_parent_directory,
    numeric_runtime_fingerprint,
)
from steering.statistics_remote import (
    RemoteStatisticsConfig,
    RemoteStatisticsError,
    RemoteStatisticsSource,
)
from steering.stats_io import save_statistics
from steering.stats_types import (
    CorpusStats,
    ExtractionStatistics,
    Extrapolation,
    MetadataStats,
    PerFeatureStat,
    SegmentFeatureStats,
    StorageStats,
)

_TENSOR_GETITEM = t.Tensor.__getitem__
_TENSOR_ITEM = t.Tensor.item
_TENSOR_TOLIST = t.Tensor.tolist
_BUILTIN_ROUND = round

PERCENTILES = tuple(range(1, 100))
CORPUS_PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
P99_SUBSAMPLE_TARGET = 10_000_000
_CPU_DEVICE = t.device("cpu")
_CUDA_DEVICE = t.device("cuda")


def _assert_nonempty_base_shard(data: dict, shard_path: Path) -> None:
    """Defense-in-depth empty-segment guard for base-reading producers (site f).

    ``compute_statistics`` reads base shards directly via ``load_features``
    (NOT through ``FeatureIndex``), so it does not inherit the FeatureIndex
    load check. After loading a *raw* base shard (before any percentile
    filtering, which legitimately empties segments), reject any shard with an
    empty segment per the issue #522 non-empty-base-segment contract.

    The legitimate whole-shard ``n_segments == 0`` skip is automatically
    exempt: ``empty_segment_indices`` returns ``[]`` for ``offsets == [0]``.
    """
    empties = empty_segment_indices(data["offsets"])
    if empties:
        i = empties[0]
        raise EmptyExtractionSegmentError(
            f"compute_statistics: base shard {shard_path.name} segment {i} has "
            f"zero features (offsets[{i}]==offsets[{i + 1}]); base shards must "
            f"be non-empty (issue #522). Re-extract."
        )


def _stats_domain_shards(feature_dir: str | Path, exclude_last_shard: bool) -> list[Path]:
    """Return the shard list whose headers define the stats consistency domain."""
    stats_shards = glob_features(feature_dir)
    if exclude_last_shard and len(stats_shards) > 1:
        stats_shards = stats_shards[:-1]
    return stats_shards


def _percentile_dict(
    data: t.Tensor,
    percentiles: tuple[int, ...],
    round_digits: int | None = None,
    as_int: bool = False,
) -> dict[str, float | int]:
    """Compute named percentiles from a 1-D tensor using **nearest-rank**.

    Method: ``idx = min(int(p / 100 * n), n - 1)`` of the sorted values. This
    is the corpus-level percentile convention used throughout this module
    (``corpus_pcts``, ``magnitude_stats.percentiles``, ``count_pcts``,
    ``active_per_segment_percentiles``, ``segment_length_percentiles``,
    ``features_per_token_percentiles``).

    .. warning::

        The PER-FEATURE percentile path uses **linear interpolation** between
        sorted ranks, NOT nearest-rank. The two conventions are
        intentionally distinct — corpus-level percentiles describe a
        population summary that downstream tooling indexes by integer
        percentile, while per-feature percentiles drive
        :func:`steering.conditional.percentile_table.magnitude_to_percentile`
        which interpolates the per-feature row. Tooling that compares the
        two (e.g. ``viz/`` plotters) MUST account for this method drift.

        Documented as map P2 #2 / consolidation manifest §4.2.

    Args:
        data: 1-D tensor of values.
        percentiles: Percentile values (0-100 scale, e.g. 50 for median).
        round_digits: If set, round float results to this many decimals.
        as_int: If True, cast results to int instead of float.

    Returns:
        Dict like ``{"p10": 0.42, "p50": 1.23, ...}``.
    """
    sorted_vals = data.sort().values
    n = len(sorted_vals)
    result: dict[str, float | int] = {}
    for p in percentiles:
        idx = min(int(p / 100 * n), n - 1)
        val = sorted_vals[idx].item()
        if as_int:
            result[f"p{p}"] = int(val)
        elif round_digits is not None:
            result[f"p{p}"] = round(val, round_digits)
        else:
            result[f"p{p}"] = val
    return result


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Shard loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardMetadata:
    """Extraction metadata from the first shard file."""

    pooling: str
    segment_mode: SegmentMode
    segment_size: int
    context_window: int
    layer: int


@dataclass
class LoadedShards:
    """Result of loading and concatenating all shard feature files."""

    feature_ids: t.Tensor
    feature_mags: t.Tensor
    total_docs: int
    total_tokens: int
    total_storage: int
    metadata: ShardMetadata
    shard_data_list: list[dict]
    n_shards: int


@dataclass
class ExactPass1Result:
    """One-pass aggregate state without corpus-sized magnitude spills."""

    acc: Accumulators
    seg_acc: ExactSegmentStatsAccumulator
    magnitude_counts: ExactMagnitudeCounts | None
    mag_subsample: t.Tensor
    global_mag_min: float
    metadata: ShardMetadata
    total_docs: int
    total_tokens: int
    total_storage: int
    n_shards: int
    n_sae_features: int
    shard_paths: list[Path]


def load_all_shards(
    feature_dir: str,
    *,
    exclude_last_shard: bool = False,
) -> LoadedShards:
    """Load all feature shard files from *feature_dir*.

    Concatenates ``feature_ids`` and ``feature_mags`` across shards and
    collects aggregate metadata.

    Args:
        feature_dir: Directory containing ``*_features.safetensors.zst`` shard files.
        exclude_last_shard: When True, drop the final (validation) shard from the
            aggregation if more than one shard is present. Mirrors the streaming-path
            exact streaming flag so eager + streaming produce identical stats.
    """
    logger.warning(
        "load_all_shards() loads all shard data into memory. "
        "Use exact_pass1() for O(1-shard) input memory."
    )
    pt_files = glob_features(feature_dir)
    if not pt_files:
        raise FileNotFoundError(f"No feature shard files found in {feature_dir}")
    if exclude_last_shard and len(pt_files) > 1:
        logger.info(f"Excluding last shard (validation): {pt_files[-1].name}")
        pt_files = pt_files[:-1]

    shard_data_list: list[dict] = []
    all_ids: list[t.Tensor] = []
    all_mags: list[t.Tensor] = []
    total_docs = 0
    total_tokens = 0
    total_storage = 0

    for pt_path in pt_files:
        data = load_features(pt_path)
        _assert_nonempty_base_shard(data, pt_path)
        shard_data_list.append(data)
        all_ids.append(data["feature_ids"])
        all_mags.append(data["feature_mags"])
        total_docs += data["n_docs"]
        total_tokens += int(data["doc_token_counts"].sum().item())
        total_storage += pt_path.stat().st_size

    first = shard_data_list[0]

    return LoadedShards(
        feature_ids=t.cat(all_ids) if all_ids else t.tensor([], dtype=t.int32),
        feature_mags=t.cat(all_mags) if all_mags else t.tensor([], dtype=t.float32),
        total_docs=total_docs,
        total_tokens=total_tokens,
        total_storage=total_storage,
        metadata=ShardMetadata(
            pooling=first["pooling"],
            segment_mode=SegmentMode(first["segment_mode"]),
            segment_size=first["segment_size"],
            context_window=first["context_window"],
            layer=first["layer"],
        ),
        shard_data_list=shard_data_list,
        n_shards=len(pt_files),
    )


# ---------------------------------------------------------------------------
# Accumulators
# ---------------------------------------------------------------------------


@dataclass
class Accumulators:
    """Dense accumulators for vectorized scatter_add_ aggregation."""

    occurrence: t.Tensor
    mag_sum: t.Tensor
    mag_sum_sq: t.Tensor
    mag_min: t.Tensor
    mag_max: t.Tensor


def init_accumulators(n: int) -> Accumulators:
    """Create zero-initialized accumulators for *n* features."""
    return Accumulators(
        occurrence=t.zeros(n, dtype=t.int64),
        mag_sum=t.zeros(n, dtype=t.float64),
        mag_sum_sq=t.zeros(n, dtype=t.float64),
        mag_min=t.full((n,), float("inf"), dtype=t.float64),
        mag_max=t.full((n,), float("-inf"), dtype=t.float64),
    )


def _expand_accumulators(acc: Accumulators, new_size: int) -> None:
    """Expand all accumulator tensors to *new_size*, preserving existing data."""
    acc.occurrence = t.cat([acc.occurrence, t.zeros(new_size - len(acc.occurrence), dtype=t.int64)])
    acc.mag_sum = t.cat([acc.mag_sum, t.zeros(new_size - len(acc.mag_sum), dtype=t.float64)])
    acc.mag_sum_sq = t.cat(
        [acc.mag_sum_sq, t.zeros(new_size - len(acc.mag_sum_sq), dtype=t.float64)]
    )
    acc.mag_min = t.cat(
        [
            acc.mag_min,
            t.full((new_size - len(acc.mag_min),), float("inf"), dtype=t.float64),
        ]
    )
    acc.mag_max = t.cat(
        [
            acc.mag_max,
            t.full((new_size - len(acc.mag_max),), float("-inf"), dtype=t.float64),
        ]
    )


def accumulate_shard(acc: Accumulators, feat_ids: t.Tensor, feat_mags: t.Tensor) -> None:
    """Update *acc* in-place with one shard's features.

    Auto-expands every accumulator tensor if *feat_ids* contains indices
    beyond the current size.
    """
    if len(feat_ids) == 0:
        return

    assert len(feat_ids) == len(feat_mags), (
        f"feat_ids length {len(feat_ids)} != feat_mags length {len(feat_mags)}"
    )

    max_id = int(feat_ids.max().item())
    if max_id >= len(acc.occurrence):
        _expand_accumulators(acc, max_id + 1)

    ids_long = feat_ids.to(t.int64)
    mags_f64 = feat_mags.to(t.float64)
    acc.occurrence.scatter_add_(0, ids_long, t.ones_like(ids_long))
    acc.mag_sum.scatter_add_(0, ids_long, mags_f64)
    acc.mag_sum_sq.scatter_add_(0, ids_long, mags_f64**2)
    acc.mag_min.scatter_reduce_(0, ids_long, mags_f64, reduce="amin")
    acc.mag_max.scatter_reduce_(0, ids_long, mags_f64, reduce="amax")


# Default GPU chunk sizes (activations per chunk)
GPU_CHUNK_4090 = 100_000_000  # 24GB VRAM
GPU_CHUNK_DEFAULT = 500_000_000  # 80GB VRAM (H100/H200)
GPU_CHUNK_LARGE = 800_000_000  # 141GB+ VRAM (B200)
# Device-first dtype conversion briefly retains one 4-byte transfer tensor
# alongside the int64 IDs, float64 magnitudes, and 8-byte reduction temporary.
_BYTES_PER_ACT_ACCUM = 28


def gpu_accumulate_shard(
    acc: Accumulators,
    feat_ids: t.Tensor,
    feat_mags: t.Tensor,
    gpu_chunk_size: int = GPU_CHUNK_DEFAULT,
) -> None:
    """GPU-accelerated version of accumulate_shard with VRAM-safe chunking.

    Moves accumulators to GPU, processes feature data in chunks to stay within
    VRAM limits, then moves results back to CPU.

    Args:
        acc: Dense accumulators (will be mutated in place, kept on CPU).
        feat_ids: 1-D int tensor of feature indices.
        feat_mags: 1-D float tensor of activation magnitudes.
        gpu_chunk_size: Max activations per GPU chunk.
    """
    if len(feat_ids) == 0:
        return

    assert len(feat_ids) == len(feat_mags), (
        f"feat_ids length {len(feat_ids)} != feat_mags length {len(feat_mags)}"
    )

    # Expand accumulators if needed (same as CPU path)
    max_id = int(feat_ids.max().item())
    if max_id >= len(acc.occurrence):
        _expand_accumulators(acc, max_id + 1)

    device = t.device("cuda")

    # Move accumulators to GPU
    acc.occurrence = acc.occurrence.to(device)
    acc.mag_sum = acc.mag_sum.to(device)
    acc.mag_sum_sq = acc.mag_sum_sq.to(device)
    acc.mag_min = acc.mag_min.to(device)
    acc.mag_max = acc.mag_max.to(device)

    n = len(feat_ids)
    for start in range(0, n, gpu_chunk_size):
        end = min(start + gpu_chunk_size, n)
        chunk_ids = feat_ids[start:end].to(device).to(t.int64)
        chunk_mags = feat_mags[start:end].to(device).to(t.float64)

        acc.occurrence.scatter_add_(0, chunk_ids, t.ones_like(chunk_ids))
        acc.mag_sum.scatter_add_(0, chunk_ids, chunk_mags)
        acc.mag_sum_sq.scatter_add_(0, chunk_ids, chunk_mags**2)
        acc.mag_min.scatter_reduce_(0, chunk_ids, chunk_mags, reduce="amin")
        acc.mag_max.scatter_reduce_(0, chunk_ids, chunk_mags, reduce="amax")

    # Move back to CPU
    acc.occurrence = acc.occurrence.cpu()
    acc.mag_sum = acc.mag_sum.cpu()
    acc.mag_sum_sq = acc.mag_sum_sq.cpu()
    acc.mag_min = acc.mag_min.cpu()
    acc.mag_max = acc.mag_max.cpu()


# ---------------------------------------------------------------------------
# Segment stats
# ---------------------------------------------------------------------------


@dataclass
class SegmentStatsAccumulator:
    """Accumulated per-segment distribution data across shards."""

    features_per_segment: list[int]
    tokens_per_doc: list[int]
    segment_lengths: list[int]
    features_per_token_ratios: list[float]


@dataclass
class ExactSegmentStatsAccumulator:
    """Bounded-state segment distributions for the exact streaming path."""

    features_per_segment: WeightedDistribution
    tokens_per_doc: WeightedDistribution
    segment_lengths: WeightedDistribution
    features_per_token_ratios: WeightedDistribution


def init_segment_stats() -> SegmentStatsAccumulator:
    """Create an empty segment stats accumulator."""
    return SegmentStatsAccumulator(
        features_per_segment=[],
        tokens_per_doc=[],
        segment_lengths=[],
        features_per_token_ratios=[],
    )


def init_exact_segment_stats() -> ExactSegmentStatsAccumulator:
    """Create empty bounded-state segment distributions."""
    return ExactSegmentStatsAccumulator(
        features_per_segment=WeightedDistribution(),
        tokens_per_doc=WeightedDistribution(),
        segment_lengths=WeightedDistribution(),
        features_per_token_ratios=WeightedDistribution(),
    )


def compute_segment_lengths(shard_data: dict) -> list[int]:
    """Compute token lengths from stored segment positions for every mode."""
    seg_mode = shard_data["segment_mode"]
    seg_size = shard_data["segment_size"]
    try:
        SegmentMode(str(seg_mode))
    except ValueError as exc:
        raise ValidationError(
            f"Unknown segment_mode {seg_mode!r}; valid values are "
            f"{[mode.value for mode in SegmentMode]}"
        ) from exc
    if seg_size <= 0:
        raise ValidationError(f"compute_segment_lengths requires segment_size > 0, got {seg_size}")

    n_segments = shard_data["n_segments"]
    pos = shard_data["segment_positions"][:n_segments].to(t.int64)
    dids = shard_data["doc_ids"][:n_segments].to(t.int64)
    ctx_win = shard_data["context_window"]
    doc_tokens = shard_data["doc_token_counts"]
    kept = doc_tokens.to(t.int64)[dids]
    if ctx_win > 0:
        kept = t.clamp(kept, max=ctx_win)
    is_last = t.ones(n_segments, dtype=t.bool)
    is_last[:-1] = dids[1:] != dids[:-1]
    nxt = t.empty_like(pos)
    nxt[:-1] = pos[1:]
    nxt[is_last] = kept[is_last]
    lengths = nxt - pos
    bad = (lengths <= 0).nonzero(as_tuple=True)[0]
    if len(bad) > 0:
        i = int(bad[0])
        raise ValidationError(
            f"Computed non-positive segment length {int(lengths[i])} at segment {i} "
            f"(doc_id={int(dids[i])}, position={int(pos[i])}, segment_mode={seg_mode!r})"
        )
    return lengths.tolist()


def accumulate_segment_stats(
    acc: SegmentStatsAccumulator,
    shard_data: dict,
) -> list[int]:
    """Update *acc* in-place with one shard's segment-level statistics.

    Returns the shard's segment lengths (needed by threshold accumulation).
    """
    offsets = shard_data["offsets"]
    doc_tokens = shard_data["doc_token_counts"]

    segment_feature_counts = (offsets[1:] - offsets[:-1]).tolist()
    acc.features_per_segment.extend(segment_feature_counts)
    acc.tokens_per_doc.extend(doc_tokens.tolist())

    shard_seg_lengths = compute_segment_lengths(shard_data)
    acc.segment_lengths.extend(shard_seg_lengths)

    for feat_count, seg_len in zip(segment_feature_counts, shard_seg_lengths, strict=True):
        if seg_len > 0:
            acc.features_per_token_ratios.append(feat_count / seg_len)

    return shard_seg_lengths


def accumulate_exact_segment_stats(
    acc: ExactSegmentStatsAccumulator,
    shard_data: dict,
) -> list[int]:
    """Update exact weighted segment distributions from one shard."""
    offsets = shard_data["offsets"]
    segment_feature_counts = (offsets[1:] - offsets[:-1]).tolist()
    shard_segment_lengths = compute_segment_lengths(shard_data)

    acc.features_per_segment.add(segment_feature_counts)
    acc.tokens_per_doc.add(shard_data["doc_token_counts"].tolist())
    acc.segment_lengths.add(shard_segment_lengths)
    acc.features_per_token_ratios.add(
        feature_count / segment_length
        for feature_count, segment_length in zip(
            segment_feature_counts,
            shard_segment_lengths,
            strict=True,
        )
        if segment_length > 0
    )
    return shard_segment_lengths


def _weighted_segment_stats(
    acc: SegmentStatsAccumulator | ExactSegmentStatsAccumulator,
) -> ExactSegmentStatsAccumulator:
    """Return one exact representation for eager and streaming finalization."""
    if isinstance(acc, ExactSegmentStatsAccumulator):
        return acc
    weighted = init_exact_segment_stats()
    weighted.features_per_segment.add(acc.features_per_segment)
    weighted.tokens_per_doc.add(acc.tokens_per_doc)
    weighted.segment_lengths.add(acc.segment_lengths)
    weighted.features_per_token_ratios.add(acc.features_per_token_ratios)
    return weighted


@dataclass(frozen=True)
class FeaturePercentiles:
    """Per-feature magnitude percentiles, median, and histogram counts."""

    percentiles: dict[str, float]
    median: float
    histogram: list[int] | None


def compute_per_feature_percentiles(
    feat_ids: t.Tensor,
    feat_mags: t.Tensor,
    percentiles: tuple[int, ...] = PERCENTILES,
) -> tuple[dict[int, FeaturePercentiles], list[float] | None]:
    """Compute per-feature magnitude percentiles, medians, and histograms.

    Fully vectorized implementation using double-sort trick:
    1. Sort by magnitude, then stable sort by feature ID → gives sorted mags within each group
    2. Quantiles via direct index lookup (no per-feature torch.quantile calls)
    3. Histograms via bucketize + scatter_add (no per-feature torch.histogram calls)

    Falls back to GPU when available for large inputs (>10M activations).

    Returns a tuple of:
        - ``{feature_id: FeaturePercentiles}`` for every unique feature
        - shared histogram bin edges (list of floats), or None if no data
    """
    if len(feat_ids) == 0:
        return {}, None

    native_quantile_rows = (
        type(feat_ids) is t.Tensor
        and type(feat_mags) is t.Tensor
        and not t.overrides.has_torch_function((feat_ids, feat_mags))
        and not is_in_torch_dispatch_mode()
        and not dict.__len__(vars(feat_ids))
        and not dict.__len__(vars(feat_mags))
        and type(percentiles) is tuple
        and all(type(p) is int for p in percentiles)
    )

    # Use GPU if available and data is large enough to benefit
    use_gpu = t.cuda.is_available() and len(feat_ids) > 10_000_000
    device = t.device("cuda") if use_gpu else t.device("cpu")

    ids_dev = feat_ids.to(device)
    mags_dev = feat_mags.to(t.float64).to(device)

    # ── Shared histogram bin edges clipped at p99 ──
    n_hist_bins = 50
    mag_lo = mags_dev.min().item()
    # Exact p99 via kthvalue — no sampling needed
    k99 = max(1, int(len(mags_dev) * 0.99))
    mag_p99 = mags_dev.kthvalue(k99).values.item()
    mag_hi = max(mag_p99, mag_lo + 1e-6)
    shared_edges = t.linspace(mag_lo, mag_hi, n_hist_bins + 1, dtype=t.float64, device=device)

    # ── Double-sort: sort by mag first, then stable sort by feat_id ──
    # This yields contiguous groups (by feat_id) with sorted magnitudes within each group.
    sort_by_mag = mags_dev.argsort()
    ids_by_mag = ids_dev[sort_by_mag]
    mags_by_mag = mags_dev[sort_by_mag]
    del sort_by_mag

    sort_by_id = ids_by_mag.argsort(stable=True)
    sorted_ids = ids_by_mag[sort_by_id]
    sorted_mags = mags_by_mag[sort_by_id]
    del ids_by_mag, mags_by_mag, sort_by_id

    # ── Group boundaries ──
    unique_ids, counts = t.unique_consecutive(sorted_ids, return_counts=True)
    n_features = len(unique_ids)
    starts = t.zeros(n_features + 1, dtype=t.int64, device=device)
    starts[1:] = counts.cumsum(0)
    feature_starts = starts[:-1]

    # ── Vectorized quantile computation ──
    # With sorted magnitudes within each group, quantile is just interpolated index lookup.
    # Linear interpolation: index = (count - 1) * p
    pct_fracs = t.tensor([p / 100.0 for p in percentiles], dtype=t.float64, device=device)
    counts_f = counts.to(t.float64)

    # [n_features, n_percentiles]
    continuous_idx = (counts_f.unsqueeze(1) - 1) * pct_fracs.unsqueeze(0)
    lower_idx = continuous_idx.floor().long()
    upper_idx = continuous_idx.ceil().long().clamp(max=(counts.unsqueeze(1) - 1))
    frac = continuous_idx - lower_idx.to(t.float64)

    global_lower = feature_starts.unsqueeze(1) + lower_idx
    global_upper = feature_starts.unsqueeze(1) + upper_idx
    lower_vals = sorted_mags[global_lower]
    upper_vals = sorted_mags[global_upper]
    quantile_values = lower_vals + frac * (upper_vals - lower_vals)  # [n_features, n_percentiles]

    # Median: index = (count - 1) * 0.5
    median_continuous = (counts_f - 1) * 0.5
    median_lower = median_continuous.floor().long()
    median_upper = median_continuous.ceil().long().clamp(max=(counts - 1))
    median_frac = median_continuous - median_lower.to(t.float64)
    medians = sorted_mags[feature_starts + median_lower] + median_frac * (
        sorted_mags[feature_starts + median_upper] - sorted_mags[feature_starts + median_lower]
    )

    # ── Vectorized histogram computation ──
    # Bucketize all activations at once, then scatter into per-feature bins.
    clamped = sorted_mags.clamp(shared_edges[0], shared_edges[-1])
    bin_idx = t.bucketize(clamped, shared_edges[1:])  # [N], values in [0, n_hist_bins-1]
    bin_idx = bin_idx.clamp(max=n_hist_bins - 1)

    feature_membership = t.repeat_interleave(
        t.arange(n_features, dtype=t.int64, device=device),
        counts,
    )
    combined_idx = feature_membership * n_hist_bins + bin_idx
    histogram_flat = t.zeros(n_features * n_hist_bins, dtype=t.int64, device=device)
    histogram_flat.scatter_add_(0, combined_idx, t.ones_like(combined_idx))
    histograms = histogram_flat.reshape(n_features, n_hist_bins)  # [n_features, n_hist_bins]

    # ── Build result dict (transfer back to CPU for serialization) ──
    unique_ids_cpu = unique_ids.cpu()
    quantile_values_cpu = quantile_values.cpu()
    medians_cpu = medians.cpu()
    histograms_cpu = histograms.cpu()

    result: dict[int, FeaturePercentiles] = {}
    for i in range(n_features):
        fid = int(unique_ids_cpu[i].item())
        if (
            native_quantile_rows
            and type(quantile_values_cpu) is t.Tensor
            and not t.overrides.has_torch_function((quantile_values_cpu,))
            and not is_in_torch_dispatch_mode()
            and quantile_values_cpu.device.type == "cpu"
            and quantile_values_cpu.dtype is t.float64
            and quantile_values_cpu.layout is t.strided
            and tuple(quantile_values_cpu.shape) == (n_features, len(percentiles))
            and round is _BUILTIN_ROUND
            and t.Tensor.__getitem__ is _TENSOR_GETITEM
            and t.Tensor.item is _TENSOR_ITEM
            and t.Tensor.tolist is _TENSOR_TOLIST
            and not dict.__len__(vars(quantile_values_cpu))
            and not quantile_values_cpu.requires_grad
        ):
            quantile_row = quantile_values_cpu[i].tolist()
            pcts = {
                f"p{p}": round(quantile_row[j], 6) for j, p in enumerate(percentiles)
            }
        else:
            pcts = {
                f"p{p}": round(quantile_values_cpu[i, j].item(), 6) for j, p in enumerate(percentiles)
            }
        result[fid] = FeaturePercentiles(
            percentiles=pcts,
            median=round(medians_cpu[i].item(), 6),
            histogram=histograms_cpu[i].tolist(),
        )

    shared_bin_edges = [round(v, 6) for v in shared_edges.cpu().tolist()]
    return result, shared_bin_edges


def compute_per_feature_thresholds(
    feat_ids: t.Tensor,
    feat_mags: t.Tensor,
    percentile: float,
) -> t.Tensor:
    """Compute per-feature magnitude threshold at a given percentile.

    Vectorized implementation using double-sort trick: sort by magnitude then
    stable sort by feature ID to get sorted magnitudes within each group.
    Quantile is then a direct index lookup.

    For each feature, finds the *percentile*-th quantile among its **non-zero**
    activations.  Features with no non-zero activations get ``inf`` (nothing
    passes the filter).

    Args:
        feat_ids: 1-D integer tensor of feature indices.
        feat_mags: 1-D float tensor of activation magnitudes (same length).
        percentile: Quantile in [0, 1].

    Returns:
        Float64 tensor of shape ``[max_feature_id + 1]`` with per-feature
        threshold values.
    """
    assert 0.0 <= percentile <= 1.0, f"percentile must be in [0, 1], got {percentile}"

    if len(feat_ids) == 0:
        return t.tensor([], dtype=t.float64)

    # Filter to non-zero activations first
    nonzero_mask = feat_mags > 0
    nz_ids = feat_ids[nonzero_mask]
    nz_mags = feat_mags[nonzero_mask].to(t.float64)

    max_id = int(feat_ids.max().item())
    thresholds = t.full((max_id + 1,), float("inf"), dtype=t.float64)

    if len(nz_ids) == 0:
        return thresholds

    # Use GPU if available and data is large enough
    use_gpu = t.cuda.is_available() and len(nz_ids) > 10_000_000
    device = t.device("cuda") if use_gpu else t.device("cpu")
    nz_ids_dev = nz_ids.to(device)
    nz_mags_dev = nz_mags.to(device)

    # Double-sort: by mag then stable by feat_id → sorted mags within each group
    sort_by_mag = nz_mags_dev.argsort()
    ids_by_mag = nz_ids_dev[sort_by_mag]
    mags_by_mag = nz_mags_dev[sort_by_mag]
    del sort_by_mag

    sort_by_id = ids_by_mag.argsort(stable=True)
    sorted_ids = ids_by_mag[sort_by_id]
    sorted_mags = mags_by_mag[sort_by_id]
    del ids_by_mag, mags_by_mag, sort_by_id

    # Group boundaries
    unique_ids, counts = t.unique_consecutive(sorted_ids, return_counts=True)
    n_features = len(unique_ids)
    feature_starts = t.zeros(n_features, dtype=t.int64, device=device)
    feature_starts[1:] = counts[:-1].cumsum(0)

    # Vectorized quantile: index = (count - 1) * percentile
    counts_f = counts.to(t.float64)
    continuous_idx = (counts_f - 1) * percentile
    lower_idx = continuous_idx.floor().long()
    upper_idx = continuous_idx.ceil().long().clamp(max=(counts - 1))
    frac = continuous_idx - lower_idx.to(t.float64)

    global_lower = feature_starts + lower_idx
    global_upper = feature_starts + upper_idx
    threshold_vals = sorted_mags[global_lower] + frac * (
        sorted_mags[global_upper] - sorted_mags[global_lower]
    )

    # Write back to dense thresholds tensor
    unique_ids_cpu = unique_ids.cpu()
    threshold_vals_cpu = threshold_vals.cpu()
    thresholds[unique_ids_cpu.long()] = threshold_vals_cpu

    return thresholds


def filter_shard_by_percentile(
    shard_data: dict,
    per_feature_thresholds: t.Tensor,
) -> dict:
    """Filter a shard's activations by per-feature percentile thresholds.

    For each activation ``(feature_id, magnitude)``, keeps it only when
    ``magnitude > per_feature_thresholds[feature_id]``.  Recomputes offsets
    so segment boundaries remain valid.

    Feature IDs beyond the threshold tensor are **not** kept (treated as
    having ``threshold = inf``).

    Args:
        shard_data: Dict from a ``*_features.safetensors.zst`` shard file.
        per_feature_thresholds: Float64 tensor of shape ``[n_features]``.

    Returns:
        Shallow copy of *shard_data* with filtered ``feature_ids``,
        ``feature_mags``, and recomputed ``offsets``.
    """
    feat_ids = shard_data["feature_ids"]
    feat_mags = shard_data["feature_mags"]
    offsets = shard_data["offsets"]
    n_segments = shard_data["n_segments"]

    if len(feat_ids) == 0:
        return dict(shard_data)

    # Look up each activation's feature threshold
    n_thresh = len(per_feature_thresholds)
    thresholds_for_each = t.full((len(feat_ids),), float("inf"), dtype=t.float64)
    in_range = feat_ids.to(t.int64) < n_thresh
    thresholds_for_each[in_range] = per_feature_thresholds[feat_ids[in_range].to(t.int64)]

    keep_mask = feat_mags.to(t.float64) > thresholds_for_each

    # Recompute offsets: count kept activations per segment
    seg_lengths = (offsets[1:] - offsets[:-1]).to(t.int64)
    seg_membership = t.repeat_interleave(
        t.arange(n_segments, dtype=t.int64),
        seg_lengths,
    )
    kept_seg_ids = seg_membership[keep_mask]
    new_counts = t.zeros(n_segments, dtype=t.int64)
    if len(kept_seg_ids) > 0:
        new_counts.scatter_add_(0, kept_seg_ids, t.ones_like(kept_seg_ids))
    new_offsets = t.zeros(n_segments + 1, dtype=offsets.dtype)
    new_offsets[1:] = new_counts.cumsum(0)

    result = dict(shard_data)
    result["feature_ids"] = feat_ids[keep_mask]
    result["feature_mags"] = feat_mags[keep_mask]
    result["offsets"] = new_offsets
    assert result["offsets"][0] == 0, "Filtered offsets[0] must be 0"
    assert result["offsets"][-1] == len(result["feature_ids"]), (
        f"Filtered offsets[-1]={result['offsets'][-1]} != len(feature_ids)={len(result['feature_ids'])}"
    )
    assert len(result["feature_ids"]) == len(result["feature_mags"]), (
        f"Filtered feature_ids length {len(result['feature_ids'])} != "
        f"feature_mags length {len(result['feature_mags'])}"
    )
    return result


# ---------------------------------------------------------------------------
# Production exact streaming reduction
# ---------------------------------------------------------------------------


_CHECKPOINT_DISTRIBUTIONS = (
    ("features_per_segment", t.int64),
    ("tokens_per_doc", t.int64),
    ("segment_lengths", t.int64),
    ("features_per_token_ratios", t.float64),
)
_CHECKPOINT_SCALAR_KEYS = frozenset(
    {
        "metadata",
        "total_docs",
        "total_tokens",
        "total_storage",
        "subsample_step",
        "magnitude_total_count",
    }
)


def _resolve_exact_runtime(device: t.device, gpu_chunk_size: int) -> tuple[t.device, int]:
    """Resolve the exact-pass device and VRAM-safe moment chunk size once."""
    resolved_device = (
        device if device == _CUDA_DEVICE and t.cuda.is_available() else _CPU_DEVICE
    )
    if resolved_device == _CUDA_DEVICE:
        free_vram, _ = t.cuda.mem_get_info()
        vram_budget = max(1_000_000, int(free_vram * 0.5 / _BYTES_PER_ACT_ACCUM))
        gpu_chunk_size = min(gpu_chunk_size, vram_budget)
    return resolved_device, gpu_chunk_size


def _distribution_checkpoint_tensors(
    tensors: dict[str, t.Tensor],
    *,
    prefix: str,
    distribution: WeightedDistribution,
    value_dtype: t.dtype,
) -> None:
    ordered = sorted(distribution.counts.items(), key=lambda item: item[0])
    tensors[f"{prefix}_values"] = t.tensor(
        [item[0] for item in ordered],
        dtype=value_dtype,
    )
    tensors[f"{prefix}_multiplicities"] = t.tensor(
        [item[1] for item in ordered],
        dtype=t.int64,
    )


def _checkpoint_distribution(
    tensors: dict[str, t.Tensor],
    *,
    prefix: str,
    value_dtype: t.dtype,
) -> WeightedDistribution:
    values = tensors[f"{prefix}_values"]
    multiplicities = tensors[f"{prefix}_multiplicities"]
    if (
        values.ndim != 1
        or values.dtype != value_dtype
        or multiplicities.ndim != 1
        or multiplicities.dtype != t.int64
        or values.shape != multiplicities.shape
    ):
        raise StatisticsCheckpointError(
            f"checkpoint distribution {prefix!r} has invalid value/multiplicity tensors"
        )
    if bool((multiplicities <= 0).any().item()):
        raise StatisticsCheckpointError(
            f"checkpoint distribution {prefix!r} has nonpositive multiplicities"
        )
    python_values = values.tolist()
    python_multiplicities = multiplicities.tolist()
    if len(set(python_values)) != len(python_values):
        raise StatisticsCheckpointError(
            f"checkpoint distribution {prefix!r} contains duplicate values"
        )
    if python_values != sorted(python_values):
        raise StatisticsCheckpointError(
            f"checkpoint distribution {prefix!r} values are not sorted"
        )
    counts = Counter(
        {
            value: int(multiplicity)
            for value, multiplicity in zip(
                python_values,
                python_multiplicities,
                strict=True,
            )
        }
    )
    try:
        return WeightedDistribution(
            counts=counts,
            population_size=sum(counts.values()),
        )
    except ExactStatisticsError as exc:
        raise StatisticsCheckpointError(
            f"checkpoint distribution {prefix!r} is outside its numeric domain: {exc}"
        ) from exc


def _build_exact_checkpoint_state(
    *,
    acc: Accumulators,
    seg_acc: ExactSegmentStatsAccumulator,
    magnitude_counts: ExactMagnitudeCounts | None,
    subsample_parts: list[t.Tensor],
    subsample_step: int,
    metadata: ShardMetadata,
    total_docs: int,
    total_tokens: int,
    total_storage: int,
) -> tuple[dict[str, object], dict[str, t.Tensor]]:
    tensors = {
        "occurrence": acc.occurrence,
        "mag_sum": acc.mag_sum,
        "mag_sum_sq": acc.mag_sum_sq,
        "mag_min": acc.mag_min,
        "mag_max": acc.mag_max,
        "mag_subsample": (
            t.cat(subsample_parts) if subsample_parts else t.empty(0, dtype=t.float32)
        ),
    }
    if magnitude_counts is not None:
        tensors["magnitude_counts"] = magnitude_counts.counts
    for prefix, value_dtype in _CHECKPOINT_DISTRIBUTIONS:
        _distribution_checkpoint_tensors(
            tensors,
            prefix=prefix,
            distribution=getattr(seg_acc, prefix),
            value_dtype=value_dtype,
        )
    scalars: dict[str, object] = {
        "metadata": {
            "pooling": metadata.pooling,
            "segment_mode": metadata.segment_mode.value,
            "segment_size": metadata.segment_size,
            "context_window": metadata.context_window,
            "layer": metadata.layer,
        },
        "total_docs": total_docs,
        "total_tokens": total_tokens,
        "total_storage": total_storage,
        "subsample_step": subsample_step,
        "magnitude_total_count": (
            magnitude_counts.total_count if magnitude_counts is not None else None
        ),
    }
    return scalars, tensors


def _restore_exact_checkpoint_state(
    checkpoint: LoadedStatisticsCheckpoint,
    *,
    n_sae_features: int,
    skip_per_feature_detail: bool,
    inventory_byte_lengths: Sequence[int],
) -> tuple[
    Accumulators,
    ExactSegmentStatsAccumulator,
    ExactMagnitudeCounts | None,
    list[t.Tensor],
    int,
    ShardMetadata,
    int,
    int,
    int,
]:
    tensors = checkpoint.tensors
    expected_keys = {
        "occurrence",
        "mag_sum",
        "mag_sum_sq",
        "mag_min",
        "mag_max",
        "mag_subsample",
    }
    for prefix, _value_dtype in _CHECKPOINT_DISTRIBUTIONS:
        expected_keys.update({f"{prefix}_values", f"{prefix}_multiplicities"})
    if not skip_per_feature_detail:
        expected_keys.add("magnitude_counts")
    if set(tensors) != expected_keys:
        raise StatisticsCheckpointError(
            "checkpoint tensor keys differ: "
            f"expected={sorted(expected_keys)}, observed={sorted(tensors)}"
        )
    for name, dtype in (
        ("occurrence", t.int64),
        ("mag_sum", t.float64),
        ("mag_sum_sq", t.float64),
        ("mag_min", t.float64),
        ("mag_max", t.float64),
    ):
        tensor = tensors[name]
        if tensor.shape != (n_sae_features,) or tensor.dtype != dtype:
            raise StatisticsCheckpointError(
                f"checkpoint tensor {name!r} must have shape ({n_sae_features},) "
                f"and dtype {dtype}, got {tuple(tensor.shape)} {tensor.dtype}"
            )
    sample = tensors["mag_subsample"]
    if sample.ndim != 1 or sample.dtype != t.float32:
        raise StatisticsCheckpointError(
            "checkpoint mag_subsample must be a 1-D float32 tensor"
        )
    acc = Accumulators(
        occurrence=tensors["occurrence"],
        mag_sum=tensors["mag_sum"],
        mag_sum_sq=tensors["mag_sum_sq"],
        mag_min=tensors["mag_min"],
        mag_max=tensors["mag_max"],
    )
    distributions = {
        prefix: _checkpoint_distribution(
            tensors,
            prefix=prefix,
            value_dtype=value_dtype,
        )
        for prefix, value_dtype in _CHECKPOINT_DISTRIBUTIONS
    }
    seg_acc = ExactSegmentStatsAccumulator(**distributions)
    if set(checkpoint.scalars) != _CHECKPOINT_SCALAR_KEYS:
        raise StatisticsCheckpointError(
            "checkpoint scalar keys differ: "
            f"expected={sorted(_CHECKPOINT_SCALAR_KEYS)}, "
            f"observed={sorted(checkpoint.scalars)}"
        )
    total_docs = checkpoint.scalars["total_docs"]
    total_tokens = checkpoint.scalars["total_tokens"]
    total_storage = checkpoint.scalars["total_storage"]
    subsample_step = checkpoint.scalars["subsample_step"]
    for name, value in (
        ("total_docs", total_docs),
        ("total_tokens", total_tokens),
        ("total_storage", total_storage),
    ):
        if type(value) is not int or value < 0:
            raise StatisticsCheckpointError(
                f"checkpoint scalar {name} must be a nonnegative int, got {value!r}"
            )
    if type(subsample_step) is not int or subsample_step <= 0:
        raise StatisticsCheckpointError(
            f"checkpoint subsample_step must be positive, got {subsample_step!r}"
        )
    expected_storage = sum(inventory_byte_lengths[: checkpoint.processed_shards])
    if total_storage != expected_storage:
        raise StatisticsCheckpointError(
            "checkpoint total_storage disagrees with its processed inventory: "
            f"state={total_storage}, inventory={expected_storage}"
        )
    metadata_raw = checkpoint.scalars["metadata"]
    if not isinstance(metadata_raw, dict) or set(metadata_raw) != {
        "pooling",
        "segment_mode",
        "segment_size",
        "context_window",
        "layer",
    }:
        raise StatisticsCheckpointError("checkpoint shard metadata is malformed")
    if (
        type(metadata_raw["pooling"]) is not str
        or type(metadata_raw["segment_mode"]) is not str
    ):
        raise StatisticsCheckpointError("checkpoint shard metadata strings are malformed")
    for key in ("segment_size", "context_window", "layer"):
        if type(metadata_raw[key]) is not int:
            raise StatisticsCheckpointError(
                f"checkpoint shard metadata {key} must be an int"
            )
    try:
        metadata = ShardMetadata(
            pooling=metadata_raw["pooling"],
            segment_mode=SegmentMode(metadata_raw["segment_mode"]),
            segment_size=metadata_raw["segment_size"],
            context_window=metadata_raw["context_window"],
            layer=metadata_raw["layer"],
        )
    except (TypeError, ValueError) as exc:
        raise StatisticsCheckpointError(
            f"checkpoint shard metadata is invalid: {metadata_raw!r}"
        ) from exc
    occurrence_total = int(acc.occurrence.sum().item())
    magnitude_counts: ExactMagnitudeCounts | None = None
    magnitude_total = checkpoint.scalars["magnitude_total_count"]
    if skip_per_feature_detail:
        if magnitude_total is not None:
            raise StatisticsCheckpointError(
                "skip-detail checkpoint unexpectedly records a magnitude count total"
            )
    else:
        counts = tensors["magnitude_counts"]
        if (
            counts.shape != (n_sae_features, POSITIVE_FINITE_F16_COUNT)
            or counts.dtype != t.int64
        ):
            raise StatisticsCheckpointError(
                "checkpoint magnitude_counts has an invalid shape or dtype: "
                f"{tuple(counts.shape)} {counts.dtype}"
            )
        if type(magnitude_total) is not int or magnitude_total < 0:
            raise StatisticsCheckpointError(
                f"checkpoint magnitude_total_count is invalid: {magnitude_total!r}"
            )
        table_total = 0
        for start in range(0, n_sae_features, 128):
            rows = counts[start : start + 128]
            if bool((rows < 0).any().item()):
                raise StatisticsCheckpointError("checkpoint count table contains negatives")
            table_total += int(rows.sum().item())
        if table_total != magnitude_total:
            raise StatisticsCheckpointError(
                f"checkpoint count-table total {table_total} != tracked {magnitude_total}"
            )
        magnitude_counts = ExactMagnitudeCounts(counts=counts, total_count=magnitude_total)
    if occurrence_total != (magnitude_total if magnitude_total is not None else occurrence_total):
        raise StatisticsCheckpointError(
            "checkpoint occurrence and magnitude totals disagree: "
            f"occurrence={occurrence_total}, magnitude={magnitude_total}"
        )
    n_segments = seg_acc.features_per_segment.population_size
    if seg_acc.segment_lengths.population_size != n_segments:
        raise StatisticsCheckpointError(
            "checkpoint segment distributions have different populations"
        )
    if seg_acc.tokens_per_doc.population_size != total_docs:
        raise StatisticsCheckpointError(
            "checkpoint token distribution population disagrees with total_docs"
        )
    if seg_acc.tokens_per_doc.exact_integer_sum() != total_tokens:
        raise StatisticsCheckpointError(
            "checkpoint token distribution sum disagrees with total_tokens"
        )
    if seg_acc.features_per_segment.exact_integer_sum() != occurrence_total:
        raise StatisticsCheckpointError(
            "checkpoint segment feature-count sum disagrees with occurrence total"
        )
    if seg_acc.features_per_token_ratios.population_size != n_segments:
        raise StatisticsCheckpointError(
            "checkpoint features/token population disagrees with segment population"
        )
    active = acc.occurrence > 0
    if bool((acc.occurrence < 0).any().item()):
        raise StatisticsCheckpointError("checkpoint occurrence contains negatives")
    if bool(active.any().item()):
        if not bool(t.isfinite(acc.mag_min[active]).all().item()) or not bool(
            t.isfinite(acc.mag_max[active]).all().item()
        ):
            raise StatisticsCheckpointError("checkpoint active min/max contains non-finite values")
        if not bool(t.isfinite(acc.mag_sum[active]).all().item()) or not bool(
            t.isfinite(acc.mag_sum_sq[active]).all().item()
        ):
            raise StatisticsCheckpointError("checkpoint active moments contain non-finite values")
    inactive = ~active
    if bool(inactive.any().item()) and (
        not bool((acc.mag_sum[inactive] == 0).all().item())
        or not bool((acc.mag_sum_sq[inactive] == 0).all().item())
        or not bool(t.isposinf(acc.mag_min[inactive]).all().item())
        or not bool(t.isneginf(acc.mag_max[inactive]).all().item())
    ):
        raise StatisticsCheckpointError("checkpoint inactive moment state is not canonical")
    return (
        acc,
        seg_acc,
        magnitude_counts,
        [sample],
        subsample_step,
        metadata,
        total_docs,
        total_tokens,
        total_storage,
    )


def _canonical_feature_width(feature_dir: str | Path) -> int | None:
    """Resolve a canonical feature directory's declared SAE width."""
    try:
        components = parse_feature_dir(feature_dir)
    except ValueError:
        return None
    return SAE_WIDTH_TO_NUM_FEATURES[components.width]


def _should_prefetch_exact_loads(
    *,
    resolved_device: t.device,
    magnitude_counts: ExactMagnitudeCounts | None,
    remaining_shards: int,
) -> bool:
    """Whether read+zstd work can overlap the exact CUDA reducer."""
    return (
        resolved_device == _CUDA_DEVICE
        and magnitude_counts is not None
        and remaining_shards >= 2
    )


def _prepare_exact_shard_safetensors(
    *,
    shard_index: int,
    shard_path: Path,
    shard_inventory: Sequence[ShardInventoryEntry] | None,
    remote_source: RemoteStatisticsSource | None,
) -> bytes:
    """Read and decompress one exact-pass shard without creating tensors."""
    if remote_source is None:
        if shard_inventory is None:
            return load_feature_safetensors_bytes(shard_path)
        return load_feature_safetensors_bytes_stable(
            shard_path,
            expected_identity=shard_inventory[shard_index].identity,
        )

    prepared = remote_source.prepare(shard_index)
    load_error: BaseException | None = None
    try:
        return load_feature_safetensors_bytes_stable(
            prepared.local_path,
            expected_identity=prepared.file_identity,
        )
    except BaseException as exc:
        load_error = exc
        raise
    finally:
        try:
            remote_source.release(prepared)
        except BaseException as release_exc:
            if load_error is None:
                raise
            if release_exc is not load_error:
                load_error.add_note(
                    "remote statistics release also failed after loader "
                    f"failure: {release_exc!r}"
                )


def _shutdown_exact_load_executor(
    executor: ThreadPoolExecutor,
    prepared_future: Future[bytes] | None,
    *,
    primary_error: BaseException | None,
) -> None:
    """Cancel and join the bounded loader without masking a primary failure."""
    if prepared_future is not None:
        prepared_future.cancel()
    shutdown_error: BaseException | None = None
    try:
        executor.shutdown(wait=True, cancel_futures=True)
    except BaseException as exc:
        shutdown_error = exc

    preparation_error: BaseException | None = None
    if (
        prepared_future is not None
        and prepared_future.done()
        and not prepared_future.cancelled()
    ):
        preparation_error = prepared_future.exception()

    if primary_error is None:
        if preparation_error is not None:
            raise preparation_error
        if shutdown_error is not None:
            raise shutdown_error
        return
    if preparation_error is not None and preparation_error is not primary_error:
        primary_error.add_note(
            f"read+zstd lookahead also failed during cleanup: {preparation_error!r}"
        )
    if shutdown_error is not None and shutdown_error is not primary_error:
        primary_error.add_note(
            f"read+zstd executor shutdown also failed: {shutdown_error!r}"
        )


def exact_pass1(
    feature_dir: str,
    n_sae_features: int | None = None,
    *,
    device: t.device = _CPU_DEVICE,
    gpu_chunk_size: int = GPU_CHUNK_DEFAULT,
    exclude_last_shard: bool = False,
    skip_per_feature_detail: bool = False,
    shard_inventory: Sequence[ShardInventoryEntry] | None = None,
    checkpoint_store: StatisticsCheckpointStore | None = None,
    checkpoint_contract: Mapping[str, object] | None = None,
    checkpoint_interval_shards: int = 50,
    _runtime_resolved: bool = False,
    profile: StatisticsProfile | None = None,
    remote_source: RemoteStatisticsSource | None = None,
) -> ExactPass1Result:
    """Accumulate one shard at a time into exact bounded-state reducers.

    Canonical feature directories and explicit callers allocate the final
    count-table width before any shard body is loaded. Noncanonical library
    callers retain historical family-wide auto-detection through a discovery
    prepass rather than risk a multi-GiB reallocation after aggregation starts.
    """
    if not isinstance(device, t.device):
        raise TypeError(f"device must be a torch.device, got {device!r}")
    if shard_inventory is not None and remote_source is not None:
        raise StatisticsCheckpointError(
            "exact_pass1 local shard_inventory and remote_source are mutually exclusive"
        )
    if remote_source is not None:
        shard_paths = remote_source.shard_paths
    elif shard_inventory is not None:
        shard_paths = [entry.path for entry in shard_inventory]
    else:
        shard_paths = glob_features(feature_dir)
    if not shard_paths:
        raise FileNotFoundError(f"No feature shard files in {feature_dir}")
    if (
        shard_inventory is None
        and remote_source is None
        and exclude_last_shard
        and len(shard_paths) > 1
    ):
        logger.info(f"Excluding last shard (validation): {shard_paths[-1].name}")
        shard_paths = shard_paths[:-1]
    if checkpoint_store is not None:
        if (shard_inventory is None and remote_source is None) or checkpoint_contract is None:
            raise StatisticsCheckpointError(
                "checkpointed exact_pass1 requires a sealed inventory and contract"
            )
        if type(checkpoint_interval_shards) is not int or checkpoint_interval_shards <= 0:
            raise StatisticsCheckpointError(
                "checkpoint_interval_shards must be a positive int, got "
                f"{checkpoint_interval_shards!r}"
            )

    declared_width = n_sae_features
    if declared_width is None:
        declared_width = _canonical_feature_width(feature_dir)
    elif skip_per_feature_detail:
        count_table_nbytes(declared_width)

    noncanonical_has_activations = True
    if declared_width is None and not skip_per_feature_detail:
        logger.warning(
            "Noncanonical feature directory with no explicit n_sae_features; "
            "performing a compatibility width-discovery prepass. Canonical "
            "production paths and explicit widths read every shard body once."
        )
        detected_max = -1
        for shard_path in tqdm(shard_paths, desc="Discover noncanonical width"):
            discovery_data = load_features(shard_path)
            _assert_nonempty_base_shard(discovery_data, shard_path)
            discovery_ids = discovery_data["feature_ids"]
            if len(discovery_ids) == 0:
                del discovery_ids, discovery_data
                continue
            min_feature_id = int(discovery_ids.min().item())
            if min_feature_id < 0:
                raise ExactStatisticsError(
                    f"feature IDs must be nonnegative in {shard_path.name}; "
                    f"minimum={min_feature_id}"
                )
            detected_max = max(detected_max, int(discovery_ids.max().item()))
            del discovery_ids, discovery_data
        noncanonical_has_activations = detected_max >= 0
        declared_width = detected_max + 1 if noncanonical_has_activations else 24_576

    if _runtime_resolved:
        resolved_device = device
    else:
        resolved_device, gpu_chunk_size = _resolve_exact_runtime(device, gpu_chunk_size)
    use_gpu = resolved_device == _CUDA_DEVICE

    restored = None
    if checkpoint_store is not None and checkpoint_contract is not None:
        restore_started = profile.start() if profile is not None else 0
        restored = checkpoint_store.restore(checkpoint_contract)
        if profile is not None:
            profile.finish_run_region(
                "statistics.pass1.checkpoint_restore",
                restore_started,
            )
    if restored is None:
        magnitude_counts: ExactMagnitudeCounts | None = None
        if (
            declared_width is not None
            and not skip_per_feature_detail
            and noncanonical_has_activations
        ):
            allocate_started = profile.start() if profile is not None else 0
            magnitude_counts = ExactMagnitudeCounts.allocate(declared_width)
            if profile is not None:
                profile.finish_run_region(
                    "statistics.pass1.count_table_allocate",
                    allocate_started,
                )
        acc = init_accumulators(declared_width if declared_width is not None else 1)
        seg_acc = init_exact_segment_stats()
        metadata: ShardMetadata | None = None
        total_docs = 0
        total_tokens = 0
        total_storage = 0
        subsample_parts: list[t.Tensor] = []
        subsample_step = 1
        start_shard = 0
    else:
        if declared_width is None:
            raise StatisticsCheckpointError(
                "checkpoint restore requires a resolved SAE dictionary width"
            )
        state_restore_started = profile.start() if profile is not None else 0
        if remote_source is not None:
            restored.scalars = remote_source.restore_checkpoint_metadata(restored.scalars)
        (
            acc,
            seg_acc,
            magnitude_counts,
            subsample_parts,
            subsample_step,
            restored_metadata,
            total_docs,
            total_tokens,
            total_storage,
        ) = _restore_exact_checkpoint_state(
            restored,
            n_sae_features=declared_width,
            skip_per_feature_detail=skip_per_feature_detail,
            inventory_byte_lengths=(
                remote_source.compressed_byte_lengths
                if remote_source is not None
                else tuple(entry.identity.size for entry in (shard_inventory or ()))
            ),
        )
        if profile is not None:
            profile.finish_run_region(
                "statistics.pass1.state_restore",
                state_restore_started,
            )
        metadata = restored_metadata
        if (
            remote_source is not None
            and metadata.segment_size != remote_source.config.expected_segment_size
        ):
            raise StatisticsCheckpointError(
                "restored shard metadata segment_size differs from the resolved "
                "remote configuration: "
                f"checkpoint={metadata.segment_size}, "
                f"expected={remote_source.config.expected_segment_size}"
            )
        start_shard = restored.processed_shards
    if profile is not None:
        profile.config["n_input_shards"] = len(shard_paths)
        profile.config["resumed_from_shard"] = start_shard

    progress_paths = shard_paths[start_shard:]
    progress = tqdm(
        progress_paths,
        desc="Exact pass: reduce shards",
        initial=start_shard,
        total=len(shard_paths),
    )
    prefetch_exact_loads = _should_prefetch_exact_loads(
        resolved_device=resolved_device,
        magnitude_counts=magnitude_counts,
        remaining_shards=len(progress_paths),
    )
    if profile is not None:
        profile.config["read_zstd_prefetch_enabled"] = prefetch_exact_loads
    if prefetch_exact_loads:
        logger.info(
            "Exact pass read+zstd prefetch enabled: one raw lookahead shard, "
            "main-thread safetensors decode"
        )
    load_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="statistics-read-zstd")
        if prefetch_exact_loads
        else None
    )
    prepared_future: Future[bytes] | None = None
    pipeline_error: BaseException | None = None
    try:
        for shard_index, shard_path in enumerate(progress, start=start_shard):
            shard_profile = (
                profile.shard(shard_index, shard_path.name)
                if profile is not None
                else None
            )
            phase_started = shard_profile.start() if shard_profile is not None else 0
            if load_executor is not None:
                if prepared_future is None:
                    prepared_future = load_executor.submit(
                        _prepare_exact_shard_safetensors,
                        shard_index=shard_index,
                        shard_path=shard_path,
                        shard_inventory=shard_inventory,
                        remote_source=remote_source,
                    )
                prepared_bytes = prepared_future.result()
                prepared_future = None
                try:
                    data = decode_feature_safetensors_bytes(prepared_bytes)
                finally:
                    del prepared_bytes
                if shard_profile is not None:
                    shard_profile.finish("load", phase_started)
            elif remote_source is not None:
                prepared = remote_source.prepare(shard_index)
                load_error: BaseException | None = None
                try:
                    data = load_features_stable(
                        prepared.local_path,
                        expected_identity=prepared.file_identity,
                    )
                    if shard_profile is not None:
                        shard_profile.finish("load", phase_started)
                except BaseException as exc:
                    load_error = exc
                    raise
                finally:
                    try:
                        remote_source.release(prepared)
                    except BaseException as release_exc:
                        if load_error is None:
                            raise
                        if release_exc is not load_error:
                            load_error.add_note(
                                "remote statistics release also failed after loader "
                                f"failure: {release_exc!r}"
                            )
            elif shard_inventory is None:
                data = load_features(shard_path)
            else:
                inventory_entry = shard_inventory[shard_index]
                data = load_features_stable(
                    shard_path,
                    expected_identity=inventory_entry.identity,
                )
            if load_executor is None and remote_source is None and shard_profile is not None:
                shard_profile.finish("load", phase_started)

            processed_shards = shard_index + 1
            checkpoint_due = checkpoint_store is not None and (
                processed_shards == len(shard_paths)
                or processed_shards % checkpoint_interval_shards == 0
            )
            if (
                load_executor is not None
                and processed_shards < len(shard_paths)
                and not checkpoint_due
            ):
                prepared_future = load_executor.submit(
                    _prepare_exact_shard_safetensors,
                    shard_index=processed_shards,
                    shard_path=shard_paths[processed_shards],
                    shard_inventory=shard_inventory,
                    remote_source=remote_source,
                )

            feature_ids = data["feature_ids"]
            feature_mags = data["feature_mags"]
            n_activations = len(feature_ids)
            n_segments_in_shard = int(data["n_segments"])
            if shard_profile is not None:
                shard_profile.record_workload(
                    n_activations=n_activations,
                    n_segments=n_segments_in_shard,
                )

            phase_started = shard_profile.start() if shard_profile is not None else 0
            _assert_nonempty_base_shard(data, shard_path)
            if metadata is None:
                metadata = ShardMetadata(
                    pooling=data["pooling"],
                    segment_mode=SegmentMode(data["segment_mode"]),
                    segment_size=data["segment_size"],
                    context_window=data["context_window"],
                    layer=data["layer"],
                )
            elif remote_source is not None:
                observed_shard_metadata = ShardMetadata(
                    pooling=data["pooling"],
                    segment_mode=SegmentMode(data["segment_mode"]),
                    segment_size=data["segment_size"],
                    context_window=data["context_window"],
                    layer=data["layer"],
                )
                if observed_shard_metadata != metadata:
                    raise RemoteStatisticsError(
                        "remote shard structural metadata differs across the statistics "
                        f"domain at {shard_path}: expected={metadata!r}, "
                        f"observed={observed_shard_metadata!r}"
                    )
            if remote_source is not None and (
                metadata.segment_size != remote_source.config.expected_segment_size
            ):
                raise RemoteStatisticsError(
                    f"remote shard {shard_path} segment_size={metadata.segment_size} "
                    "differs from resolved segment_size="
                    f"{remote_source.config.expected_segment_size}"
                )
            if n_activations > 0 and declared_width is not None:
                min_feature_id = int(feature_ids.min().item())
                max_feature_id = int(feature_ids.max().item())
                if min_feature_id < 0 or max_feature_id >= declared_width:
                    raise ExactStatisticsError(
                        "feature IDs lie outside the declared SAE dictionary: "
                        f"shard={shard_path.name}, min={min_feature_id}, "
                        f"max={max_feature_id}, valid=[0, {declared_width})"
                    )
            if shard_profile is not None:
                shard_profile.finish("validate", phase_started)

            if magnitude_counts is not None:
                phase_started = shard_profile.start() if shard_profile is not None else 0
                magnitude_counts.add(
                    feature_ids,
                    feature_mags,
                    reduction_device=resolved_device,
                )
                if shard_profile is not None:
                    shard_profile.finish("exact_counts", phase_started)

            phase_started = shard_profile.start() if shard_profile is not None else 0
            if use_gpu:
                gpu_accumulate_shard(acc, feature_ids, feature_mags, gpu_chunk_size)
            else:
                accumulate_shard(acc, feature_ids, feature_mags)
            if shard_profile is not None:
                shard_profile.finish("moments", phase_started)

            phase_started = shard_profile.start() if shard_profile is not None else 0
            accumulate_exact_segment_stats(seg_acc, data)
            if shard_profile is not None:
                shard_profile.finish("segment_stats", phase_started)

            phase_started = shard_profile.start() if shard_profile is not None else 0
            total_docs += data["n_docs"]
            total_tokens += int(data["doc_token_counts"].sum().item())
            if remote_source is not None:
                shard_bytes = remote_source.compressed_byte_lengths[shard_index]
            elif shard_inventory is not None:
                shard_bytes = shard_inventory[shard_index].identity.size
            else:
                shard_bytes = shard_path.stat().st_size
            total_storage += shard_bytes

            if n_activations > 0 and not skip_per_feature_detail:
                if shard_index == 0:
                    estimated_total = n_activations * len(shard_paths)
                    subsample_step = max(1, estimated_total // P99_SUBSAMPLE_TARGET)
                subsample_parts.append(feature_mags[::subsample_step].clone())
            if shard_profile is not None:
                shard_profile.record_workload(
                    input_bytes=shard_bytes,
                    n_activations=n_activations,
                    n_segments=n_segments_in_shard,
                )
                shard_profile.finish("bookkeeping", phase_started)

            # The next iteration may already hold plain safetensors bytes, but
            # it cannot decode them until this shard's tensors are released.
            del feature_ids, feature_mags, data

            if checkpoint_due:
                phase_started = shard_profile.start() if shard_profile is not None else 0
                assert metadata is not None
                scalars, checkpoint_tensors = _build_exact_checkpoint_state(
                    acc=acc,
                    seg_acc=seg_acc,
                    magnitude_counts=magnitude_counts,
                    subsample_parts=subsample_parts,
                    subsample_step=subsample_step,
                    metadata=metadata,
                    total_docs=total_docs,
                    total_tokens=total_tokens,
                    total_storage=total_storage,
                )
                if remote_source is not None:
                    scalars.update(remote_source.checkpoint_metadata())
                assert checkpoint_contract is not None
                checkpoint_store.commit(
                    contract=checkpoint_contract,
                    processed_shards=processed_shards,
                    scalars=scalars,
                    tensors=checkpoint_tensors,
                )
                del scalars, checkpoint_tensors
                if shard_profile is not None:
                    shard_profile.finish("checkpoint", phase_started)
    except BaseException as exc:
        pipeline_error = exc
        if load_executor is not None and remote_source is not None:
            remote_source.request_cancel()
        raise
    finally:
        if load_executor is not None:
            _shutdown_exact_load_executor(
                load_executor,
                prepared_future,
                primary_error=pipeline_error,
            )

    assert metadata is not None
    final_width = declared_width if declared_width is not None else len(acc.occurrence)
    if magnitude_counts is not None and magnitude_counts.n_sae_features != final_width:
        raise RuntimeError(
            f"exact count width {magnitude_counts.n_sae_features} != accumulator "
            f"width {final_width} after width resolution"
        )

    finish_started = profile.start() if profile is not None else 0
    mag_subsample = t.cat(subsample_parts) if subsample_parts else t.empty(0, dtype=t.float32)
    active = acc.occurrence > 0
    global_mag_min = float(acc.mag_min[active].min().item()) if bool(active.any().item()) else 0.0
    total_activations = int(acc.occurrence.sum().item())
    if magnitude_counts is not None and magnitude_counts.total_count != total_activations:
        raise RuntimeError(
            "exact count-table total disagrees with moment accumulator: "
            f"counts={magnitude_counts.total_count}, moments={total_activations}"
        )
    if profile is not None:
        profile.finish_run_region("statistics.pass1.finish", finish_started)

    logger.info(
        f"Exact pass complete: width={final_width:,}, "
        f"active={int(active.sum().item()):,}, "
        f"activations={total_activations:,}, shards={len(shard_paths):,}; "
        "wrote 0 magnitude-spill bytes"
    )
    return ExactPass1Result(
        acc=acc,
        seg_acc=seg_acc,
        magnitude_counts=magnitude_counts,
        mag_subsample=mag_subsample,
        global_mag_min=global_mag_min,
        metadata=metadata,
        total_docs=total_docs,
        total_tokens=total_tokens,
        total_storage=total_storage,
        n_shards=len(shard_paths),
        n_sae_features=final_width,
        shard_paths=shard_paths,
    )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def _validate_checkpoint_request(
    *,
    feature_dir: str,
    output_path: str,
    checkpoint_dir: str,
    checkpoint_interval_shards: int,
    streaming: bool,
    activation_percentile: float,
    preloaded: LoadedShards | None,
) -> None:
    if checkpoint_dir == "":
        raise StatisticsCheckpointError("checkpoint_dir must not be empty")
    checkpoint_path = Path(checkpoint_dir).resolve()
    for label, candidate in (
        ("feature_dir", Path(feature_dir).resolve()),
        ("output_path", Path(output_path).resolve()),
    ):
        if candidate == checkpoint_path or candidate.is_relative_to(checkpoint_path):
            raise StatisticsCheckpointError(
                "checkpoint_dir must be a dedicated directory that does not "
                f"contain {label}: checkpoint={checkpoint_path}, {label}={candidate}"
            )
    if not streaming:
        raise StatisticsCheckpointError("statistics checkpoints require streaming=True")
    if preloaded is not None:
        raise StatisticsCheckpointError(
            "statistics checkpoints are incompatible with _preloaded"
        )
    if activation_percentile != 0.0:
        raise StatisticsCheckpointError(
            "statistics checkpoints require activation_percentile=0.0"
        )
    if type(checkpoint_interval_shards) is not int or checkpoint_interval_shards <= 0:
        raise StatisticsCheckpointError(
            "checkpoint_interval_shards must be a positive int, got "
            f"{checkpoint_interval_shards!r}"
        )


def _compute_statistics_impl(
    feature_dir: str,
    output_path: str,
    n_sae_features: int | None = None,
    activation_percentile: float = 0.0,
    skip_per_feature_detail: bool = False,
    _preloaded: LoadedShards | None = None,
    streaming: bool = True,
    device: t.device = _CPU_DEVICE,
    gpu_chunk_size: int = GPU_CHUNK_DEFAULT,
    exclude_last_shard: bool = False,
    allow_degenerate_statistics: bool = False,
    checkpoint_dir: str | None = None,
    checkpoint_interval_shards: int = 50,
    _checkpoint_store: StatisticsCheckpointStore | None = None,
    _profile: StatisticsProfile | None = None,
    remote_config: RemoteStatisticsConfig | None = None,
) -> ExtractionStatistics:
    """Compute corpus-wide feature statistics from per-shard feature files.

    Aggregates across all *_features.safetensors.zst shard files to produce
    per-feature and corpus-level statistics with 10B token extrapolation.

    The default streaming path reads every shard body once. It retains the
    existing vectorized moment accumulation and replaces corpus-sized sorted
    magnitude spills with exact integer counts over the positive finite
    float16 domain.

    Args:
        feature_dir: Directory containing *_features.safetensors.zst shard files.
        output_path: Where to write the stats file.
        n_sae_features: Number of SAE features (auto-detected from data if None).
        skip_per_feature_detail: If True, skip the expensive per-feature
            percentile and histogram computation (Pass 2).  Corpus-level
            aggregates are still computed.  Per-feature stats will contain
            count, mean_mag, std_mag, min/max but no median, percentiles,
            or histograms.
        _preloaded: Pre-loaded shard data to avoid redundant I/O when
            computing stats for multiple percentiles on the same combo_dir.
            When provided, a deep copy is made so filtering doesn't mutate
            the original.
        streaming: If True, use streaming passes (O(1-shard) memory) instead
            of loading all shards into memory.  Incompatible with _preloaded
            and activation_percentile > 0.
        device: Resolved CPU or CUDA device for streaming acceleration.
        gpu_chunk_size: Max activations per GPU moment chunk (for VRAM safety).
        checkpoint_dir: Opt-in exact-pass checkpoint directory. A compatible
            committed generation is resumed automatically.
        checkpoint_interval_shards: Number of newly completed shards between
            checkpoint generations. The final shard is always checkpointed.
    """
    if streaming:
        device, gpu_chunk_size = _resolve_exact_runtime(device, gpu_chunk_size)
    if _profile is not None:
        _profile.config["numeric_runtime"] = numeric_runtime_fingerprint(
            device=device,
            gpu_chunk_size=gpu_chunk_size,
        )

    # Canonical paths declare the SAE dictionary independently of which
    # aggregation mode runs. Resolve that contract before branching so eager
    # execution (including activation-percentile filtering) cannot shrink the
    # artifact to only the IDs that happened to fire.
    if n_sae_features is None:
        n_sae_features = _canonical_feature_width(feature_dir)
    if checkpoint_dir is not None and n_sae_features is None:
        raise StatisticsCheckpointError(
            "statistics checkpoints require an explicit n_sae_features or a "
            "canonical feature directory whose width can be resolved before body reads"
        )
    if remote_config is not None and n_sae_features is None:
        raise RemoteStatisticsError(
            "remote statistics requires an explicit n_sae_features or a canonical "
            "feature directory whose width can be resolved before body reads"
        )

    # Fail fast on cross-shard header metadata before reading/sorting shard
    # bodies. The domain intentionally matches the stats domain: when
    # exclude_last_shard=True, the validation shard is ignored here exactly as
    # exact_pass1/load_all_shards ignore it during aggregation.
    preflight_started = _profile.start() if _profile is not None else 0
    remote_source: RemoteStatisticsSource | None = None
    checkpoint_inventory: tuple[ShardInventoryEntry, ...] | None = None
    if remote_config is not None:
        if _profile is not None:
            _profile.remote_transport = {
                "schema_version": 1,
                "status": "running",
                "bucket_id": str(remote_config.bucket_id),
                "remote_prefix": str(remote_config.remote_prefix),
            }
        try:
            remote_source = RemoteStatisticsSource.snapshot(
                remote_config,
                exclude_last_shard=exclude_last_shard,
                profile=_profile,
            )
        except BaseException as exc:
            if _profile is not None:
                assert isinstance(_profile.remote_transport, dict)
                _profile.remote_transport["status"] = "failed"
                _profile.remote_transport["terminal_error"] = {
                    "stage": "inventory_snapshot",
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "index": None,
                    "path": None,
                }
            raise
        stats_l0_tier = None
        stats_ignore_bos = False
        stats_sft_mixture_fingerprint_json = None
    else:
        stats_shards = _stats_domain_shards(feature_dir, exclude_last_shard)
        checkpoint_inventory = (
            capture_shard_inventory(stats_shards) if checkpoint_dir is not None else None
        )
        if checkpoint_inventory is None:
            stats_l0_tier = assert_uniform_l0_tier(stats_shards)
            stats_ignore_bos = assert_uniform_ignore_bos(stats_shards)
            stats_sft_mixture_fingerprint_json = _uniform_sft_mixture_fingerprint_json(
                stats_shards,
                operation="compute_statistics",
            )
        else:
            metadata_by_path = [
                (
                    entry.path,
                    read_shard_header_stable(
                        entry.path,
                        expected_identity=entry.identity,
                    )["__metadata__"],
                )
                for entry in checkpoint_inventory
            ]
            stats_l0_tier = assert_uniform_l0_tier_metadata(metadata_by_path)
            stats_ignore_bos = assert_uniform_ignore_bos_metadata(metadata_by_path)
            stats_sft_mixture_fingerprint_json = (
                _uniform_sft_mixture_fingerprint_metadata_json(
                    metadata_by_path,
                    operation="compute_statistics",
                )
            )
            assert_shard_inventory_unchanged(checkpoint_inventory)
    if _profile is not None:
        _profile.finish_run_region("statistics.preflight", preflight_started)

    checkpoint_contract: dict[str, object] | None = None
    if remote_source is not None and checkpoint_dir is not None:
        assert n_sae_features is not None
        contract_started = _profile.start() if _profile is not None else 0
        checkpoint_contract = remote_source.checkpoint_contract(
            n_sae_features=n_sae_features,
            skip_per_feature_detail=skip_per_feature_detail,
            numeric_runtime=numeric_runtime_fingerprint(
                device=device,
                gpu_chunk_size=gpu_chunk_size,
            ),
        )
        if _profile is not None:
            _profile.finish_run_region("statistics.checkpoint.contract", contract_started)
    elif checkpoint_inventory is not None:
        assert n_sae_features is not None
        contract_started = _profile.start() if _profile is not None else 0
        checkpoint_contract = build_checkpoint_contract(
            inventory=checkpoint_inventory,
            n_sae_features=n_sae_features,
            exclude_last_shard=exclude_last_shard,
            skip_per_feature_detail=skip_per_feature_detail,
            l0_tier=stats_l0_tier.value if stats_l0_tier is not None else None,
            ignore_bos_features=stats_ignore_bos,
            sft_mixture_fingerprint_json=stats_sft_mixture_fingerprint_json,
            numeric_runtime=numeric_runtime_fingerprint(
                device=device,
                gpu_chunk_size=gpu_chunk_size,
            ),
        )
        if _profile is not None:
            _profile.finish_run_region("statistics.checkpoint.contract", contract_started)

    # ── Streaming path: exact one-pass bounded-state reduction ──
    loaded: LoadedShards | None = None
    exact_stream: ExactPass1Result | None = None

    with contextlib.ExitStack():
        checkpoint_store = _checkpoint_store
        if streaming:
            assert _preloaded is None, "Cannot use _preloaded with streaming=True"
            assert activation_percentile == 0.0, (
                "activation_percentile filtering is not supported in streaming mode. "
                "Set activation_percentile=0.0 or streaming=False."
            )
            pass1_started = _profile.start() if _profile is not None else 0
            try:
                exact_stream = exact_pass1(
                    feature_dir,
                    n_sae_features,
                    device=device,
                    gpu_chunk_size=gpu_chunk_size,
                    exclude_last_shard=exclude_last_shard,
                    skip_per_feature_detail=skip_per_feature_detail,
                    shard_inventory=checkpoint_inventory,
                    checkpoint_store=checkpoint_store,
                    checkpoint_contract=checkpoint_contract,
                    checkpoint_interval_shards=checkpoint_interval_shards,
                    _runtime_resolved=True,
                    profile=_profile,
                    remote_source=remote_source,
                )
            except BaseException as exc:
                if remote_source is not None:
                    remote_source.record_terminal_error(
                        stage="reducer",
                        error=exc,
                    )
                    try:
                        remote_source.finish_transport()
                    except BaseException as transport_exc:
                        if transport_exc is not exc:
                            exc.add_note(
                                "remote statistics transport shutdown also failed: "
                                f"{transport_exc!r}"
                            )
                raise
            else:
                if remote_source is not None:
                    remote_source.finish_transport()
            if _profile is not None:
                _profile.finish_run_region("statistics.pass1.wall", pass1_started)
            n_sae_features = exact_stream.n_sae_features
            if _profile is not None:
                _profile.config["resolved_n_sae_features"] = n_sae_features
            acc = exact_stream.acc
            seg_acc = exact_stream.seg_acc
            n_shards = exact_stream.n_shards
            total_docs = exact_stream.total_docs
            total_tokens = exact_stream.total_tokens
            total_storage = exact_stream.total_storage
            metadata_src = exact_stream.metadata
            if remote_source is not None:
                (
                    stats_l0_tier,
                    stats_ignore_bos,
                    stats_sft_mixture_fingerprint_json,
                ) = remote_source.resolved_metadata()
            per_feat_thresholds_dict: dict[str, float] | None = None
        else:
            # ── Eager path: load all shards into memory ──
            if _preloaded is not None:
                loaded = copy.copy(_preloaded)
                # Deep copy shard_data_list so percentile filtering doesn't mutate original
                loaded.shard_data_list = [dict(sd) for sd in _preloaded.shard_data_list]
                loaded.feature_ids = _preloaded.feature_ids.clone()
                loaded.feature_mags = _preloaded.feature_mags.clone()
            else:
                # Honor exclude_last_shard symmetrically with the streaming path
                # (the retired reference pass also drops the validation shard). Without this,
                # config default `exclude_last_shard: true` was silently ignored
                # whenever streaming=False was requested (e.g. activation_percentile>0).
                loaded = load_all_shards(feature_dir, exclude_last_shard=exclude_last_shard)

            # ── Optional percentile filtering ──
            per_feat_thresholds_dict = None
            if activation_percentile > 0:
                thresholds_tensor = compute_per_feature_thresholds(
                    loaded.feature_ids,
                    loaded.feature_mags,
                    activation_percentile,
                )
                logger.info(
                    f"Filtering activations at percentile {activation_percentile} "
                    f"({len(thresholds_tensor)} features)"
                )

                # Build per-feature threshold dict for output
                per_feat_thresholds_dict = {}
                for fid in range(len(thresholds_tensor)):
                    val = thresholds_tensor[fid].item()
                    if val != float("inf"):
                        per_feat_thresholds_dict[str(fid)] = round(val, 6)

                # Filter each shard
                filtered_shards = [
                    filter_shard_by_percentile(sd, thresholds_tensor)
                    for sd in loaded.shard_data_list
                ]
                loaded.shard_data_list = filtered_shards

                # Rebuild concatenated tensors from filtered shards
                all_ids = [sd["feature_ids"] for sd in filtered_shards]
                all_mags = [sd["feature_mags"] for sd in filtered_shards]
                loaded.feature_ids = t.cat(all_ids) if all_ids else t.tensor([], dtype=t.int32)
                loaded.feature_mags = t.cat(all_mags) if all_mags else t.tensor([], dtype=t.float32)

            # Detect n_sae_features
            if n_sae_features is None:
                if len(loaded.feature_ids) > 0:
                    n_sae_features = int(loaded.feature_ids.max().item()) + 1
                else:
                    n_sae_features = 24576  # default for gpt2-small-res-jb

            # ── Pass 1: aggregate shard stats ──
            seg_acc = init_segment_stats()
            acc = init_accumulators(n_sae_features)

            for data in tqdm(
                loaded.shard_data_list,
                desc="Pass 1: aggregating shard stats",
                position=0,
                total=len(loaded.shard_data_list),
            ):
                feat_ids = data["feature_ids"]
                feat_mags = data["feature_mags"]

                # Accumulate segment-level stats (features/segment, lengths, ratios)
                accumulate_segment_stats(seg_acc, data)

                # Accumulate main per-feature stats
                accumulate_shard(acc, feat_ids, feat_mags)

            n_shards = loaded.n_shards
            total_docs = loaded.total_docs
            total_tokens = loaded.total_tokens
            total_storage = loaded.total_storage
            metadata_src = loaded.metadata

        finalization_started = _profile.start() if _profile is not None else 0
        start_time = time.monotonic()

        metadata = MetadataStats(
            pooling_mode=metadata_src.pooling,
            segment_mode=metadata_src.segment_mode,
            segment_size=metadata_src.segment_size,
            context_window=metadata_src.context_window,
            layer=metadata_src.layer,
            n_sae_features=n_sae_features,
            generated_at=datetime.now(UTC).isoformat(),
            activation_percentile=activation_percentile,
        )

        # ── Distributions from pass 1 ──
        segment_distributions = _weighted_segment_stats(seg_acc)
        fpd = segment_distributions.features_per_segment
        tpd = segment_distributions.tokens_per_doc

        fps_histogram: dict[str, object] = {}
        fps_percentiles: dict[str, float] = {}
        if fpd.population_size > 0:
            fps_hist = fpd.histogram(n_bins=50)
            fps_histogram = dataclasses.asdict(fps_hist) if fps_hist is not None else {}
            fps_percentiles = fpd.percentile_dict(
                CORPUS_PERCENTILES,
                round_digits=4,
            )

        seg_len_histogram: dict[str, object] | None = None
        seg_len_percentiles: dict[str, float] | None = None
        segment_lengths = segment_distributions.segment_lengths
        if segment_lengths.population_size > 0:
            sl_hist = segment_lengths.histogram(n_bins=50)
            seg_len_histogram = dataclasses.asdict(sl_hist) if sl_hist is not None else None
            seg_len_percentiles = segment_lengths.percentile_dict(
                CORPUS_PERCENTILES,
                round_digits=4,
            )

        fpt_histogram: dict[str, object] | None = None
        fpt_percentiles: dict[str, float] | None = None
        features_per_token = segment_distributions.features_per_token_ratios
        if features_per_token.population_size > 0:
            fpt_hist = features_per_token.histogram(n_bins=50)
            fpt_histogram = dataclasses.asdict(fpt_hist) if fpt_hist is not None else None
            fpt_percentiles = features_per_token.percentile_dict(
                CORPUS_PERCENTILES,
                round_digits=4,
            )

        # ── Aggregate per-feature stats from accumulators ──
        occurrence = acc.occurrence
        mag_sum = acc.mag_sum
        mag_sum_sq = acc.mag_sum_sq
        mag_min = acc.mag_min
        mag_max = acc.mag_max

        active_mask = occurrence > 0
        active_indices = active_mask.nonzero(as_tuple=True)[0]
        n_features_seen = int(active_mask.sum().item())

        # ── Corpus-wide degeneracy metrics (issue #549). Computed here while all
        # inputs are in hand; the FAIL-FAST gate *check* itself runs just below in
        # this same finalization region (these metrics are also persisted into
        # health_metrics further down). Per-shard placement would mis-fire (a
        # healthy corpus can look degenerate in one small shard).
        n_segments = fpd.population_size
        counts = occurrence[active_indices]
        assert n_sae_features is not None  # set by both streaming/eager paths above
        median_firing_freq, n_firing_gt_99pct = firing_health(counts, n_segments)

        active_counts = occurrence[active_mask].float()
        count_pcts: dict[str, float | int] = {}
        if len(active_counts) > 0:
            count_pcts = _percentile_dict(active_counts, PERCENTILES, as_int=True)

        # NOTE (manifest §4.2 / map P2 #2): the corpus-level magnitude percentiles
        # below use NEAREST-RANK via ``_percentile_dict``, NOT the LINEAR-INTERP
        # method used by the per-feature path (``compute_per_feature_percentiles``,
        # the retained sort-based test references).
        # Downstream tooling that compares the two (e.g. viz/) MUST account for
        # the method drift. Documented at the producer site (`_percentile_dict`).
        mean_mags_all = mag_sum[active_mask] / occurrence[active_mask].double()
        mag_stats: dict[str, object] = {}
        if len(mean_mags_all) > 0:
            mag_pcts = _percentile_dict(mean_mags_all, PERCENTILES, round_digits=6)
            mag_stats = {
                "mean": round(mean_mags_all.mean().item(), 6),
                "std": round(mean_mags_all.std().item(), 6),
                "min": round(mean_mags_all.min().item(), 6),
                "max": round(mean_mags_all.max().item(), 6),
                "percentiles": mag_pcts,
            }

        corpus = CorpusStats(
            n_shards_processed=n_shards,
            n_docs=total_docs,
            n_tokens_gpt2=total_tokens,
            mean_tokens_per_doc=round(total_tokens / max(total_docs, 1), 1),
            median_tokens_per_doc=int(tpd.percentile_dict((50,))["p50"])
            if tpd.population_size > 0
            else 0,
        )

        storage = StorageStats(
            total_bytes=total_storage,
            mean_bytes_per_doc=round(total_storage / max(total_docs, 1), 1),
            mean_bytes_per_token=round(total_storage / max(total_tokens, 1), 3),
        )

        # ── Per-feature magnitude percentiles ──
        # Degeneracy gate (issue #549) — fail before exact finalization.
        assert_feature_statistics_not_degenerate(
            median_firing_frequency=median_firing_freq,
            n_features_firing_gt_99pct=n_firing_gt_99pct,
            n_features_seen=n_features_seen,
            n_sae_features=n_sae_features,
            n_segments=n_segments,
            allow_degenerate=allow_degenerate_statistics,
        )
        if skip_per_feature_detail:
            logger.info("Skipping per-feature detail (percentiles/histograms)")
            per_feat_detail: dict[int, FeaturePercentiles] = {}
            shared_bin_edges: list[float] | None = None
        elif streaming:
            assert exact_stream is not None
            if exact_stream.magnitude_counts is None:
                per_feat_detail = {}
                shared_bin_edges = None
            else:
                exact_finalize_started = _profile.start() if _profile is not None else 0
                exact_detail, shared_bin_edges = exact_stream.magnitude_counts.finalize(
                    occurrence=acc.occurrence,
                    mag_subsample=exact_stream.mag_subsample,
                    global_mag_min=exact_stream.global_mag_min,
                    percentiles=PERCENTILES,
                )
                if _profile is not None:
                    _profile.finish_run_region(
                        "statistics.finalize.exact_magnitudes",
                        exact_finalize_started,
                    )
                per_feat_detail = {
                    fid: FeaturePercentiles(
                        percentiles=detail.percentiles,
                        median=detail.median,
                        histogram=detail.histogram,
                    )
                    for fid, detail in exact_detail.items()
                }
        else:
            assert loaded is not None
            per_feat_detail, shared_bin_edges = compute_per_feature_percentiles(
                loaded.feature_ids,
                loaded.feature_mags,
            )

    feature_stats = SegmentFeatureStats(
        n_segments=n_segments,
        n_features_seen=n_features_seen,
        mean_active_per_segment=round(
            fpd.exact_integer_sum() / fpd.population_size,
            1,
        )
        if fpd.population_size > 0
        else 0,
        median_active_per_segment=int(fpd.percentile_dict((50,))["p50"])
        if fpd.population_size > 0
        else 0,
        p90_active_per_segment=int(fpd.percentile_dict((90,))["p90"])
        if fpd.population_size > 0
        else 0,
        max_active_per_segment=int(fpd.maximum()) if fpd.population_size > 0 else 0,
        min_active_per_segment=int(fpd.minimum()) if fpd.population_size > 0 else 0,
        n_total_sae_features=n_sae_features,
        pct_features_active=round(n_features_seen / max(n_sae_features, 1) * 100, 2),
        count_percentiles=count_pcts,
        magnitude_stats=mag_stats,
        active_per_segment_histogram=fps_histogram,
        active_per_segment_percentiles=fps_percentiles,
        mag_histogram_bin_edges=shared_bin_edges,
        segment_length_histogram=seg_len_histogram,
        segment_length_percentiles=seg_len_percentiles,
        features_per_token_histogram=fpt_histogram,
        features_per_token_percentiles=fpt_percentiles,
    )

    # ── Build enriched per-feature stats ──
    mean_mags = mag_sum[active_indices] / counts.double()
    ids_list = active_indices.tolist()
    counts_list = counts.tolist()
    mags_list = mean_mags.tolist()

    mean_mags_incl_zeros = mag_sum[active_indices] / max(n_segments, 1)
    mags_incl_zeros_list = mean_mags_incl_zeros.tolist()

    per_feat_items: list[tuple[str, PerFeatureStat]] = []
    for feature_id, count, mean_mag, mean_mag_iz in zip(
        ids_list, counts_list, mags_list, mags_incl_zeros_list, strict=True
    ):
        fid = int(feature_id)
        count_val = int(count)

        # std from sum-of-squares: std = sqrt(E[x^2] - E[x]^2)
        mean_sq = mag_sum_sq[fid].item() / count_val
        variance = max(mean_sq - mean_mag**2, 0.0)
        std_val = variance**0.5

        detail = per_feat_detail.get(fid)
        hist_data: dict[str, object] | None = (
            {"counts": detail.histogram} if detail and detail.histogram else None
        )

        per_feat_items.append(
            (
                str(fid),
                PerFeatureStat(
                    count=count_val,
                    pct_segments_active=round(count_val / max(n_segments, 1) * 100, 4),
                    mean_mag=round(mean_mag, 6),
                    mean_mag_incl_zeros=round(mean_mag_iz, 6),
                    std_mag=round(std_val, 6),
                    min_mag=round(mag_min[fid].item(), 6),
                    max_mag=round(mag_max[fid].item(), 6),
                    median_mag=detail.median if detail else 0.0,
                    percentiles=detail.percentiles if detail else {},
                    histogram=hist_data,
                ),
            )
        )

    # Sort by count descending
    per_feat_items.sort(key=lambda x: x[1].count, reverse=True)
    per_feat = dict(per_feat_items)

    elapsed = time.monotonic() - start_time
    extrapolation = estimate_10b(corpus, storage, elapsed_seconds=elapsed)

    stats = ExtractionStatistics(
        metadata=metadata,
        corpus=corpus,
        features=feature_stats,
        storage=storage,
        per_feature_stats=per_feat,
        extrapolation_10b=extrapolation,
        per_feature_thresholds=per_feat_thresholds_dict,
        l0_tier=stats_l0_tier,
        ignore_bos_features=stats_ignore_bos,
        health_metrics={
            "median_feature_firing_frequency": str(median_firing_freq),
            "n_features_firing_gt_99pct": str(n_firing_gt_99pct),
        },
    )
    if _profile is not None:
        _profile.finish_run_region("statistics.finalize.wall", finalization_started)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    extra_metadata = (
        {SFT_MIXTURE_FINGERPRINT_METADATA_KEY: (stats_sft_mixture_fingerprint_json)}
        if stats_sft_mixture_fingerprint_json is not None
        else None
    )
    if remote_source is not None:
        fence_started = _profile.start() if _profile is not None else 0
        try:
            remote_source.assert_final_inventory_unchanged()
        except BaseException as exc:
            remote_source.record_terminal_error(
                stage="final_inventory_fence",
                error=exc,
            )
            raise
        if _profile is not None:
            _profile.finish_run_region("statistics.output.inventory_fence", fence_started)
    elif checkpoint_inventory is not None:
        # Fence the complete source family at the publication boundary. Stable
        # per-shard reads detect changes while each body is consumed, but a
        # processed shard can still be replaced or a new shard appended later
        # in a long run. This performs only one fresh listing plus lstat identity
        # checks; it never rereads or hashes shard bodies.
        fence_started = _profile.start() if _profile is not None else 0
        assert_shard_inventory_unchanged(
            checkpoint_inventory,
            observed_paths=_stats_domain_shards(feature_dir, exclude_last_shard),
        )
        if _profile is not None:
            _profile.finish_run_region("statistics.output.inventory_fence", fence_started)
    serialize_started = _profile.start() if _profile is not None else 0
    try:
        actual_path = save_statistics(
            stats,
            output_path,
            extra_metadata=extra_metadata,
        )
    except BaseException as exc:
        if remote_source is not None:
            remote_source.record_terminal_error(stage="output_serialize", error=exc)
        raise
    if _profile is not None:
        _profile.finish_run_region("statistics.output.serialize", serialize_started)
    if checkpoint_store is not None:
        cleanup_started = _profile.start() if _profile is not None else 0
        try:
            fsync_parent_directory(actual_path)
            checkpoint_store.cleanup_after_output_commit()
        except BaseException as exc:
            if remote_source is not None:
                remote_source.record_terminal_error(stage="output_commit", error=exc)
            raise
        if _profile is not None:
            _profile.finish_run_region(
                "statistics.output.checkpoint_cleanup",
                cleanup_started,
            )
    logger.info(f"Wrote statistics to {actual_path}")

    log_summary(stats)

    if not skip_per_feature_detail and per_feat and not streaming:
        sanity_check_percentiles(feature_dir, per_feat, exclude_last_shard=exclude_last_shard)
    elif not skip_per_feature_detail and per_feat:
        logger.info(
            "Skipped raw-shard percentile resampling: exact integer-count "
            "finalization already checked every feature count against occurrence"
        )

    return stats


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _validate_remote_statistics_request(
    *,
    remote_config: RemoteStatisticsConfig,
    feature_dir: str,
    output_path: str,
    n_sae_features: int | None,
    activation_percentile: float,
    preloaded: LoadedShards | None,
    streaming: bool,
    checkpoint_dir: str | None,
    profile_output: str | None,
) -> None:
    """Fail before backend construction for every unsupported remote mode."""
    if not streaming:
        raise RemoteStatisticsError("remote statistics requires streaming=True")
    if activation_percentile != 0.0:
        raise RemoteStatisticsError(
            "remote statistics requires activation_percentile=0.0"
        )
    if preloaded is not None:
        raise RemoteStatisticsError("remote statistics is incompatible with _preloaded")
    if (
        isinstance(n_sae_features, bool)
        or not isinstance(n_sae_features, int)
        or n_sae_features <= 0
    ):
        raise RemoteStatisticsError(
            "remote statistics requires an explicit positive n_sae_features, got "
            f"{n_sae_features!r}"
        )
    try:
        components = parse_feature_dir(feature_dir)
    except ValueError as exc:
        raise RemoteStatisticsError(
            "remote statistics feature_dir must be a canonical parseable feature path"
        ) from exc
    if components.segment_size != remote_config.expected_segment_size:
        raise RemoteStatisticsError(
            "remote statistics segment size differs between feature_dir and config: "
            f"path={components.segment_size}, "
            f"config={remote_config.expected_segment_size}"
        )

    actual_output = Path(output_path).with_suffix(".safetensors").resolve()
    protected: list[tuple[str, Path]] = [
        ("feature_dir", Path(feature_dir).resolve()),
        ("output_path", actual_output),
    ]
    if checkpoint_dir is not None:
        protected.append(("checkpoint_dir", Path(checkpoint_dir).resolve()))
    if profile_output is not None:
        protected.append(("profile_output", Path(profile_output).resolve()))
    for label, path in protected:
        if _paths_overlap(remote_config.staging_dir, path):
            raise RemoteStatisticsError(
                "remote statistics staging_dir must be disjoint in both containment "
                f"directions from {label}: staging={remote_config.staging_dir}, "
                f"{label}={path}"
            )


def compute_statistics(
    feature_dir: str,
    output_path: str,
    n_sae_features: int | None = None,
    activation_percentile: float = 0.0,
    skip_per_feature_detail: bool = False,
    _preloaded: LoadedShards | None = None,
    streaming: bool = True,
    device: str = "cpu",
    gpu_chunk_size: int = GPU_CHUNK_DEFAULT,
    exclude_last_shard: bool = False,
    allow_degenerate_statistics: bool = False,
    checkpoint_dir: str | None = None,
    checkpoint_interval_shards: int = 50,
    profile_output: str | None = None,
    remote_config: RemoteStatisticsConfig | None = None,
) -> ExtractionStatistics:
    """Compute statistics, optionally writing a performance timing sidecar."""
    profile_config: dict[str, object] = {}
    if profile_output is not None:
        if not streaming:
            raise ValueError("statistics profiling requires streaming=True")
        profile_path = Path(profile_output).resolve()
        statistics_path = Path(output_path).with_suffix(".safetensors").resolve()
        if profile_path == statistics_path:
            raise ValueError("profile_output must differ from the statistics output path")
        profile_config = {
            "feature_dir": str(Path(feature_dir).resolve()),
            "output_path": str(statistics_path),
            "n_sae_features": n_sae_features,
            "streaming": streaming,
            "requested_device": device,
            "requested_gpu_chunk_size": gpu_chunk_size,
            "exclude_last_shard": exclude_last_shard,
            "skip_per_feature_detail": skip_per_feature_detail,
            "checkpoint_enabled": checkpoint_dir is not None,
            "checkpoint_interval_shards": checkpoint_interval_shards,
            "read_zstd_prefetch_enabled": False,
        }
        if remote_config is not None:
            profile_config["remote_enabled"] = True
            profile_config["remote_bucket_id"] = str(remote_config.bucket_id)
            profile_config["remote_prefix"] = str(remote_config.remote_prefix)
    if remote_config is not None:
        _validate_remote_statistics_request(
            remote_config=remote_config,
            feature_dir=feature_dir,
            output_path=output_path,
            n_sae_features=n_sae_features,
            activation_percentile=activation_percentile,
            preloaded=_preloaded,
            streaming=streaming,
            checkpoint_dir=checkpoint_dir,
            profile_output=profile_output,
        )
    configured_device = t.device(
        ("cuda" if t.cuda.is_available() else "cpu") if device == "auto" else device
    )
    with statistics_profiler(
        output_path=profile_output,
        config=profile_config,
    ) as profile:
        if checkpoint_dir is None:
            return _compute_statistics_impl(
                feature_dir=feature_dir,
                output_path=output_path,
                n_sae_features=n_sae_features,
                activation_percentile=activation_percentile,
                skip_per_feature_detail=skip_per_feature_detail,
                _preloaded=_preloaded,
                streaming=streaming,
                device=configured_device,
                gpu_chunk_size=gpu_chunk_size,
                exclude_last_shard=exclude_last_shard,
                allow_degenerate_statistics=allow_degenerate_statistics,
                checkpoint_dir=None,
                checkpoint_interval_shards=checkpoint_interval_shards,
                _checkpoint_store=None,
                _profile=profile,
                remote_config=remote_config,
            )
        _validate_checkpoint_request(
            feature_dir=feature_dir,
            output_path=output_path,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval_shards=checkpoint_interval_shards,
            streaming=streaming,
            activation_percentile=activation_percentile,
            preloaded=_preloaded,
        )
        with StatisticsCheckpointStore(checkpoint_dir) as checkpoint_store:
            return _compute_statistics_impl(
                feature_dir=feature_dir,
                output_path=output_path,
                n_sae_features=n_sae_features,
                activation_percentile=activation_percentile,
                skip_per_feature_detail=skip_per_feature_detail,
                _preloaded=_preloaded,
                streaming=streaming,
                device=configured_device,
                gpu_chunk_size=gpu_chunk_size,
                exclude_last_shard=exclude_last_shard,
                allow_degenerate_statistics=allow_degenerate_statistics,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval_shards=checkpoint_interval_shards,
                _checkpoint_store=checkpoint_store,
                _profile=profile,
                remote_config=remote_config,
            )


# ---------------------------------------------------------------------------
# Percentile sanity check
# ---------------------------------------------------------------------------


def sanity_check_percentiles(
    feature_dir: str,
    per_feature_stats: dict[str, PerFeatureStat],
    *,
    n_shards_to_check: int = 3,
    n_features_to_check: int = 50,
    tolerance: float = 0.02,
    seed: int = 42,
    exclude_last_shard: bool = False,
) -> bool:
    """Quick sanity check: verify computed percentiles against raw data.

    Loads a subset of shards, picks a random sample of features, and for each
    feature+percentile pair checks that the actual fraction of magnitudes at or
    below the percentile value is close to the expected fraction.

    Returns True if all checks pass, False otherwise (with warnings logged).
    """
    paths = glob_features(feature_dir)
    if exclude_last_shard and len(paths) > 1:
        paths = paths[:-1]
    if not paths:
        logger.warning("Sanity check: no shard files found, skipping")
        return True

    rng = random.Random(seed)

    # Pick shards to check
    check_paths = paths if len(paths) <= n_shards_to_check else rng.sample(paths, n_shards_to_check)

    # Pick features to check (only those with percentiles)
    candidates = [
        (fid, stat)
        for fid, stat in per_feature_stats.items()
        if stat.percentiles and stat.count > 0
    ]
    if not candidates:
        logger.warning("Sanity check: no features with percentiles, skipping")
        return True

    check_features = (
        candidates
        if len(candidates) <= n_features_to_check
        else rng.sample(candidates, n_features_to_check)
    )
    check_fids = {int(fid) for fid, _ in check_features}
    check_stats = {int(fid): stat for fid, stat in check_features}

    # Collect magnitudes per feature from the sampled shards
    mags_by_feature: dict[int, list[float]] = {fid: [] for fid in check_fids}
    for p in check_paths:
        data = load_features(p)
        fids = data["feature_ids"]
        fmags = data["feature_mags"]
        for target_fid in check_fids:
            mask = fids == target_fid
            if mask.any():
                mags_by_feature[target_fid].extend(fmags[mask].tolist())
        del data

    # Validate each feature's percentiles
    n_checked = 0
    failures: list[str] = []
    for fid in check_fids:
        mags = mags_by_feature[fid]
        if len(mags) < 100:
            continue  # too few samples for a meaningful check

        stat = check_stats[fid]
        mags_arr = np.array(mags, dtype=np.float64)
        n = len(mags_arr)

        for pkey, pval in stat.percentiles.items():
            # pkey is like "p10", "p90", etc.
            expected_frac = int(pkey[1:]) / 100.0
            # Use bracket test: for quantized data many percentiles map to
            # the same value, so fraction(data <= v) can far exceed the
            # nominal rank.  The valid condition is:
            #   fraction(data < v) <= expected_frac <= fraction(data <= v)
            # with tolerance on both sides.
            frac_below = float(np.sum(mags_arr < pval - 1e-7)) / n
            frac_at_or_below = float(np.sum(mags_arr <= pval + 1e-7)) / n

            if expected_frac < frac_below - tolerance:
                failures.append(
                    f"feature {fid} {pkey}: value {pval:.6f} too high — "
                    f"expected ~{expected_frac:.0%} but {frac_below:.1%} "
                    f"are strictly below (delta={expected_frac - frac_below:+.1%})"
                )
            elif expected_frac > frac_at_or_below + tolerance:
                failures.append(
                    f"feature {fid} {pkey}: value {pval:.6f} too low — "
                    f"expected ~{expected_frac:.0%} but only {frac_at_or_below:.1%} "
                    f"are at or below (delta={expected_frac - frac_at_or_below:+.1%})"
                )
        n_checked += 1

    if failures:
        logger.warning(
            f"Sanity check FAILED: {len(failures)} violations in {n_checked} features "
            f"(checked {len(check_paths)} of {len(paths)} shards, tolerance={tolerance:.0%})"
        )
        for f in failures[:10]:
            logger.warning(f"  {f}")
        if len(failures) > 10:
            logger.warning(f"  ... and {len(failures) - 10} more")
        return False

    logger.info(
        f"Sanity check PASSED: {n_checked} features verified across "
        f"{len(check_paths)} shards (tolerance={tolerance:.0%})"
    )
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def estimate_10b(
    corpus: CorpusStats,
    storage: StorageStats,
    target_tokens: int = 10_000_000_000,
    elapsed_seconds: float = 0.0,
) -> Extrapolation:
    """Extrapolate profiling results to a target token count."""
    if corpus.mean_tokens_per_doc <= 0:
        raise ValueError("Cannot extrapolate with 0 tokens per doc")

    scale_factor = target_tokens / max(corpus.n_tokens_gpt2, 1)
    estimated_docs = int(target_tokens / corpus.mean_tokens_per_doc)
    estimated_storage_bytes = estimated_docs * storage.mean_bytes_per_doc
    estimated_storage_gb = round(estimated_storage_bytes / (1024**3), 1)
    estimated_hours = (
        round(elapsed_seconds * scale_factor / 3600, 1) if elapsed_seconds > 0 else 0.0
    )

    return Extrapolation(
        target_tokens=target_tokens,
        estimated_docs=estimated_docs,
        estimated_storage_gb=estimated_storage_gb,
        estimated_wall_time_hours=estimated_hours,
    )


def log_summary(stats: ExtractionStatistics):
    """Log a human-readable summary of extraction statistics."""
    meta = stats.metadata
    corpus = stats.corpus
    features = stats.features
    storage = stats.storage
    extrap = stats.extrapolation_10b

    logger.info("=" * 50)
    logger.info("SAE Feature Extraction Statistics")
    logger.info("=" * 50)
    logger.info(
        f"Pooling: {meta.pooling_mode} | Segment: {meta.segment_mode} "
        f"({meta.segment_size} tokens) | Layer: {meta.layer}"
    )
    logger.info(f"Shards processed: {corpus.n_shards_processed}")
    logger.info(f"Documents: {corpus.n_docs:,}")
    logger.info(f"GPT-2 tokens: {corpus.n_tokens_gpt2:,}")
    logger.info(f"Mean tokens/doc: {corpus.mean_tokens_per_doc:.0f}")
    logger.info(f"Median tokens/doc: {corpus.median_tokens_per_doc}")
    logger.info(f"Total segments: {features.n_segments:,}")
    logger.info(
        f"Features per segment — Mean: {features.mean_active_per_segment:.0f} | "
        f"Median: {features.median_active_per_segment} | "
        f"P90: {features.p90_active_per_segment} | "
        f"Max: {features.max_active_per_segment}"
    )
    logger.info(
        f"Unique features seen: {features.n_features_seen:,} / "
        f"{features.n_total_sae_features:,} ({features.pct_features_active}%)"
    )
    if features.count_percentiles:
        pcts_str = " | ".join(f"{k}: {v}" for k, v in features.count_percentiles.items())
        logger.info(f"Count percentiles: {pcts_str}")
    if features.magnitude_stats:
        ms = features.magnitude_stats
        logger.info(
            f"Magnitude stats — Mean: {ms.get('mean', 'N/A')} | "
            f"Std: {ms.get('std', 'N/A')} | "
            f"Min: {ms.get('min', 'N/A')} | Max: {ms.get('max', 'N/A')}"
        )
    logger.info(
        f"Storage — Total: {storage.total_bytes / (1024**2):.1f} MB "
        f"({corpus.n_shards_processed} shards) | "
        f"Per doc: {storage.mean_bytes_per_doc:.1f} bytes | "
        f"Per 1K tokens: {storage.mean_bytes_per_token * 1000:.1f} bytes"
    )
    logger.info("=" * 50)
    logger.info(f"Extrapolation to {extrap.target_tokens / 1e9:.0f}B tokens")
    logger.info("=" * 50)
    logger.info(f"  Estimated documents: {extrap.estimated_docs:,}")
    logger.info(f"  Estimated storage: {extrap.estimated_storage_gb:.1f} GB")
    if extrap.estimated_wall_time_hours > 0:
        logger.info(f"  Estimated wall time: {extrap.estimated_wall_time_hours:.1f} hours")


# ---------------------------------------------------------------------------
# Hydra entrypoint
# ---------------------------------------------------------------------------


def _remote_statistics_enabled(cfg: DictConfig) -> bool:
    remote = cfg.get("remote", None)
    if remote is None:
        return False
    enabled = remote.get("enabled", False)
    if type(enabled) is not bool:
        raise RemoteStatisticsError(
            f"remote.enabled must be a bool, got {enabled!r}"
        )
    return enabled


@hydra.main(
    config_path="../configs/extraction",
    config_name="compute_statistics",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Compute feature statistics from extracted shard files."""
    remote_enabled = _remote_statistics_enabled(cfg)
    # Resolve feature directory (auto-construct or explicit override)
    if cfg.feature_dir is not None:
        feature_dir = cfg.feature_dir
    else:
        activation_type = ActivationType(cfg.activation_type)
        pooling = PoolingMode(cfg.pooling)
        width = SAEWidth(cfg.sae_width)
        feature_dir = str(
            resolve_output_dir(
                output_dir=None,
                feature_root=cfg.feature_root,
                segment_size=cfg.segment_size,
                layer=cfg.layer,
                activation_type=activation_type,
                pooling=pooling,
                width=width,
                model=cfg.model,
                dataset_phase=cfg.dataset_phase,
                ignore_bos_features=PRODUCER_IGNORE_BOS_FEATURES,
            )
        )

    # Cross-stage segment_size check (Wave 3D task 4).
    feature_dir_path = Path(feature_dir)
    if (
        not remote_enabled
        and feature_dir_path.is_dir()
        and glob_features(feature_dir_path)
    ):
        verify_segment_size(feature_dir_path, expected_segment_size=int(cfg.segment_size))

    output_path = cfg.output_path
    if output_path is None:
        output_path = str(Path(feature_dir) / "feature_statistics.safetensors")

    n_sae_features = int(cfg.n_sae_features) if cfg.n_sae_features is not None else None
    activation_percentile = float(cfg.activation_percentile)
    streaming = bool(cfg.streaming)

    device = str(cfg.device)
    gpu_chunk_size_raw = cfg.gpu_chunk_size
    if gpu_chunk_size_raw == "auto" or gpu_chunk_size_raw is None:
        gpu_chunk_size = GPU_CHUNK_DEFAULT
        if device in ("cuda", "auto") and t.cuda.is_available():
            vram_gb = t.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb < 30:
                gpu_chunk_size = GPU_CHUNK_4090
            elif vram_gb > 100:
                gpu_chunk_size = GPU_CHUNK_LARGE
    else:
        gpu_chunk_size = int(gpu_chunk_size_raw)

    logger.info(f"Computing statistics from {feature_dir}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Device: {device} | GPU chunk size: {gpu_chunk_size:,}")
    if activation_percentile > 0:
        logger.info(f"Activation percentile filter: {activation_percentile}")

    exclude_last_shard = bool(cfg.exclude_last_shard)
    allow_degenerate_statistics = bool(cfg.allow_degenerate_statistics)
    checkpoint_dir = str(cfg.checkpoint_dir) if cfg.checkpoint_dir is not None else None
    checkpoint_interval_raw = cfg.checkpoint_interval_shards
    if isinstance(checkpoint_interval_raw, bool) or not isinstance(
        checkpoint_interval_raw,
        int,
    ) or checkpoint_interval_raw <= 0:
        raise StatisticsCheckpointError(
            "checkpoint_interval_shards must be a positive int, got "
            f"{checkpoint_interval_raw!r}"
        )
    checkpoint_interval_shards = checkpoint_interval_raw

    profile_output: str | None = None
    if bool(cfg.get("profile", False)):
        configured_profile_output = cfg.get("profile_summary_output", None)
        profile_output = (
            str(configured_profile_output)
            if configured_profile_output is not None
            else "out/traces/compute_statistics_profile.json"
        )
        logger.info(f"Statistics profile: {profile_output}")

    remote_config: RemoteStatisticsConfig | None = None
    if remote_enabled:
        remote = cfg.remote
        if remote.staging_dir is None:
            raise RemoteStatisticsError(
                "remote.staging_dir is required when remote.enabled=true"
            )
        try:
            components = parse_feature_dir(feature_dir)
        except ValueError as exc:
            raise RemoteStatisticsError(
                "remote feature_dir must be a canonical parseable feature path"
            ) from exc
        configured_segment_size = int(cfg.segment_size)
        if components.segment_size != configured_segment_size:
            raise RemoteStatisticsError(
                "remote segment_size differs between config and feature path: "
                f"config={configured_segment_size}, path={components.segment_size}"
            )
        remote_root = NormalizedRemotePrefix.parse(str(remote.remote_features_root))
        remote_prefix = resolve_remote_feature_prefix(
            feature_dir,
            local_features_root=Path(str(remote.local_features_root)),
            remote_features_root=remote_root,
            surface="feature_dir",
        )
        expected_count_raw = remote.expected_shard_count
        remote_config = RemoteStatisticsConfig(
            bucket_id=HfBucketId.parse(str(remote.bucket_id)),
            remote_prefix=remote_prefix,
            staging_dir=Path(str(remote.staging_dir)),
            expected_shard_count=(
                expected_count_raw if expected_count_raw is not None else None
            ),
            retry_timeout_seconds=remote.retry_timeout_seconds,
            retry_interval_seconds=remote.retry_interval_seconds,
            expected_segment_size=configured_segment_size,
        )
        logger.info(
            "Remote statistics input: "
            f"bucket={remote_config.bucket_id} prefix={remote_config.remote_prefix} "
            f"staging={remote_config.staging_dir} "
            f"expected_shard_count={remote_config.expected_shard_count}"
        )

    compute_statistics(
        feature_dir=feature_dir,
        output_path=output_path,
        n_sae_features=n_sae_features,
        activation_percentile=activation_percentile,
        streaming=streaming,
        device=device,
        gpu_chunk_size=gpu_chunk_size,
        exclude_last_shard=exclude_last_shard,
        allow_degenerate_statistics=allow_degenerate_statistics,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval_shards=checkpoint_interval_shards,
        profile_output=profile_output,
        remote_config=remote_config,
    )


if __name__ == "__main__":
    main()
