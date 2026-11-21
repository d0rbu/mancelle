"""Retired statistics implementations retained as test parity references."""

from dataclasses import dataclass
from pathlib import Path

import torch as t
from loguru import logger
from tqdm import tqdm

from nanochat.common import DeviceType
from steering.compute_statistics import (
    _BYTES_PER_ACT_ACCUM,
    GPU_CHUNK_DEFAULT,
    P99_SUBSAMPLE_TARGET,
    PERCENTILES,
    Accumulators,
    FeaturePercentiles,
    SegmentMode,
    SegmentStatsAccumulator,
    ShardMetadata,
    _assert_nonempty_base_shard,
    accumulate_segment_stats,
    accumulate_shard,
    gpu_accumulate_shard,
    init_accumulators,
    init_segment_stats,
)
from steering.feature_io import glob_features, load_features
from steering.stats_schema import HistogramData

GPU_MERGE_VRAM_BUDGET_BYTES = 4 * 1024**3


@dataclass
class Pass1Result:
    """Result from the retired streaming Pass 1 implementation."""

    acc: Accumulators
    seg_acc: SegmentStatsAccumulator
    metadata: ShardMetadata
    total_docs: int
    total_tokens: int
    total_storage: int
    n_shards: int
    n_sae_features: int
    shard_paths: list[Path]


def _histogram_counts(
    sorted_mags: t.Tensor,
    edges: t.Tensor,
    n_bins: int,
) -> t.Tensor:
    """Compute the retired reference histogram from shared edges."""
    clamped = sorted_mags.clamp(edges[0], edges[-1])
    bin_idx = t.bucketize(clamped, edges[1:]).clamp(max=n_bins - 1)
    hist_counts = t.zeros(n_bins, dtype=t.int64, device=sorted_mags.device)
    hist_counts.scatter_add_(0, bin_idx, t.ones_like(bin_idx))
    return hist_counts


def build_histogram(
    data: t.Tensor,
    n_bins: int = 50,
) -> HistogramData | None:
    """Build the retired eager histogram used as an exact test reference."""
    if len(data) == 0:
        return None
    counts, edges = t.histogram(data.to(t.float64), bins=n_bins)
    return HistogramData(
        bin_edges=[round(value, 4) for value in edges.tolist()],
        counts=counts.tolist(),
    )


# Retired sorted-spill reference implementation
#
# These helpers were the production sorted-spill implementation before the
# exact-count reducer replaced it. They remain here solely as an independent
# parity oracle; production has no resource-admission fallback to this path.
# ---------------------------------------------------------------------------


@dataclass
class SortedShardInfo:
    """Reference to a shard's data sorted by feature_id in a temp file."""

    mags_path: Path | None  # torch.save fp16/fp32 spill (None if shard empty)
    n_activations: int
    per_feature_counts: t.Tensor  # int64, dense counts per feature


FP16_SPILL_CHECK_CHUNK_SIZE = 16_000_000  # bounded exactness-check scratch

# Bytes-per-activation estimates for VRAM budget calculations
_BYTES_PER_ACT_SORT = 40  # ids (4B) + mags (4B) + argsort output (8B) + workspace (~24B)
_BYTES_PER_ACT_PASS2 = 48  # ids (8B) + mags (4B) + argsort (8B) + sort output (8B) + workspace


def _cpu_sort_shard(
    feat_ids: t.Tensor,
    feat_mags: t.Tensor,
) -> tuple[t.Tensor, t.Tensor]:
    """Sort a shard by feature_id on CPU.

    Returns:
        (sorted_mags, per_feature_counts) both on CPU.
    """
    sort_order = feat_ids.argsort()
    sorted_mags = feat_mags[sort_order]
    sorted_ids = feat_ids[sort_order]
    del sort_order

    max_id = int(sorted_ids[-1].item())
    shard_counts = t.zeros(max_id + 1, dtype=t.int64)
    shard_counts.scatter_add_(
        0,
        sorted_ids.long(),
        t.ones(len(feat_ids), dtype=t.int64),
    )
    del sorted_ids
    return sorted_mags, shard_counts


def _gpu_sort_shard(
    feat_ids: t.Tensor,
    feat_mags: t.Tensor,
) -> tuple[t.Tensor, t.Tensor]:
    """Sort a shard by feature_id on GPU with VRAM-aware chunking.

    If the whole shard fits in VRAM, does a single GPU argsort.
    Otherwise, sorts in GPU-sized chunks then merges via counting sort
    (O(n) merge exploiting bounded integer feature IDs).

    Returns:
        (sorted_mags, per_feature_counts) both on CPU.
    """
    n_act = len(feat_ids)
    gpu_dev = t.device("cuda")

    free_vram, _ = t.cuda.mem_get_info()
    chunk_size = max(1_000_000, int(free_vram * 0.6 / _BYTES_PER_ACT_SORT))

    if chunk_size >= n_act:
        return _gpu_sort_shard_single(feat_ids, feat_mags, n_act, gpu_dev)

    return _gpu_sort_shard_chunked(feat_ids, feat_mags, n_act, chunk_size, gpu_dev)


def _gpu_sort_shard_single(
    feat_ids: t.Tensor,
    feat_mags: t.Tensor,
    n_act: int,
    gpu_dev: t.device,
) -> tuple[t.Tensor, t.Tensor]:
    """Whole shard fits in VRAM - single GPU argsort."""
    gpu_ids = feat_ids.to(gpu_dev)
    gpu_mags = feat_mags.to(gpu_dev)
    sort_order = gpu_ids.argsort(stable=True)
    sorted_mags = gpu_mags[sort_order].cpu()
    sorted_ids = gpu_ids[sort_order]

    max_id = int(sorted_ids[-1].item())
    shard_counts = t.zeros(max_id + 1, dtype=t.int64, device=gpu_dev)
    shard_counts.scatter_add_(
        0,
        sorted_ids.long(),
        t.ones(n_act, dtype=t.int64, device=gpu_dev),
    )
    shard_counts = shard_counts.cpu()
    del sort_order, gpu_ids, gpu_mags, sorted_ids
    t.cuda.empty_cache()
    return sorted_mags, shard_counts


def _gpu_sort_shard_chunked(
    feat_ids: t.Tensor,
    feat_mags: t.Tensor,
    n_act: int,
    chunk_size: int,
    gpu_dev: t.device,
) -> tuple[t.Tensor, t.Tensor]:
    """Chunked GPU sort + counting-sort merge for large shards.

    Sorts each chunk on GPU independently, then merges via per-feature
    offset placement (O(n) since feature IDs are bounded integers).
    """
    n_chunks = (n_act + chunk_size - 1) // chunk_size
    max_id = int(feat_ids.max().item())
    shard_counts = t.zeros(max_id + 1, dtype=t.int64)
    sorted_chunks: list[tuple[t.Tensor, t.Tensor, t.Tensor]] = []

    for ci in range(n_chunks):
        cs = ci * chunk_size
        ce = min(cs + chunk_size, n_act)
        c_ids = feat_ids[cs:ce].to(gpu_dev)
        c_mags = feat_mags[cs:ce].to(gpu_dev)
        order = c_ids.argsort(stable=True)
        s_ids = c_ids[order].cpu()
        s_mags = c_mags[order].cpu()
        c_max = int(s_ids[-1].item())
        c_counts = t.zeros(c_max + 1, dtype=t.int64)
        c_counts.scatter_add_(
            0,
            s_ids.long(),
            t.ones(ce - cs, dtype=t.int64),
        )
        shard_counts[: c_max + 1] += c_counts
        sorted_chunks.append((s_ids, s_mags, c_counts))
        del c_ids, c_mags, order
        t.cuda.empty_cache()

    # Counting-sort merge: place each chunk's per-feature data at the
    # correct global offset
    global_offsets = t.zeros(max_id + 2, dtype=t.int64)
    global_offsets[1:] = shard_counts.cumsum(0)
    write_cursor = global_offsets[:-1].clone()

    sorted_mags = t.empty(n_act, dtype=t.float32)
    for s_ids, s_mags, c_counts in sorted_chunks:
        c_offsets = t.zeros(len(c_counts) + 1, dtype=t.int64)
        c_offsets[1:] = c_counts.cumsum(0)
        for fid in range(len(c_counts)):
            n_f = c_counts[fid].item()
            if n_f == 0:
                continue
            src = c_offsets[fid].item()
            dst = write_cursor[fid].item()
            sorted_mags[dst : dst + n_f] = s_mags[src : src + n_f]
            write_cursor[fid] += n_f
        del s_ids, s_mags

    del sorted_chunks, global_offsets, write_cursor
    return sorted_mags, shard_counts


# ---------------------------------------------------------------------------


# Retired sorted-spill finalization references
# ---------------------------------------------------------------------------


def optimized_pass1(
    feature_dir: str,
    tmp_dir: str | Path,
    n_sae_features: int | None = None,
    device: str = "cpu",
    gpu_chunk_size: int = GPU_CHUNK_DEFAULT,
    exclude_last_shard: bool = False,
) -> tuple[Pass1Result, list[SortedShardInfo], t.Tensor, float]:
    """Optimized Pass 1: accumulate stats + sort shards + write temp files.

    Eliminates the double-read of all shards by letting accumulators
    auto-expand on first encounter of each feature ID.  Simultaneously
    sorts each shard by feature_id and writes sorted magnitudes to temp
    binary files for the merge-based Pass 2.

    Returns:
        (pass1_result, sorted_shards, mag_subsample, global_mag_min)
    """
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    pt_files = glob_features(feature_dir)
    if not pt_files:
        raise FileNotFoundError(f"No feature shard files in {feature_dir}")
    if exclude_last_shard and len(pt_files) > 1:
        logger.info(f"Excluding last shard (validation): {pt_files[-1].name}")
        pt_files = pt_files[:-1]

    use_gpu = device == "cuda" and t.cuda.is_available()

    # Extract metadata from first shard
    first_data = load_features(pt_files[0])
    metadata = ShardMetadata(
        pooling=first_data["pooling"],
        segment_mode=SegmentMode(first_data["segment_mode"]),
        segment_size=first_data["segment_size"],
        context_window=first_data["context_window"],
        layer=first_data["layer"],
    )
    del first_data

    initial_size = n_sae_features if n_sae_features is not None else 1
    acc = init_accumulators(initial_size)
    seg_acc = init_segment_stats()
    total_docs = 0
    total_tokens = 0
    total_storage = 0
    sorted_shards: list[SortedShardInfo] = []
    subsample_parts: list[t.Tensor] = []

    subsample_step = 1  # updated after first shard

    # Auto-size gpu_chunk_size based on free VRAM when GPU is available.
    if use_gpu:
        free_vram, _ = t.cuda.mem_get_info()
        vram_budget = max(1_000_000, int(free_vram * 0.5 / _BYTES_PER_ACT_ACCUM))
        if vram_budget < gpu_chunk_size:
            gpu_chunk_size = vram_budget

    for i, pt_path in enumerate(tqdm(pt_files, desc="Pass 1: accumulate + sort")):
        data = load_features(pt_path)
        _assert_nonempty_base_shard(data, pt_path)
        feat_ids = data["feature_ids"]
        feat_mags = data["feature_mags"]

        # Accumulate (auto-expands accumulators if needed)
        if use_gpu:
            gpu_accumulate_shard(acc, feat_ids, feat_mags, gpu_chunk_size)
        else:
            accumulate_shard(acc, feat_ids, feat_mags)
        accumulate_segment_stats(seg_acc, data)
        total_docs += data["n_docs"]
        total_tokens += int(data["doc_token_counts"].sum().item())
        total_storage += pt_path.stat().st_size

        n_act = len(feat_ids)
        if n_act > 0:
            # Update subsample step based on first shard
            if i == 0:
                estimated_total = n_act * len(pt_files)
                subsample_step = max(1, estimated_total // P99_SUBSAMPLE_TARGET)

            # Subsample magnitudes for p99 / bin-edge estimation
            subsample_parts.append(feat_mags[::subsample_step].clone())

            # Sort by feature_id and compute per-feature counts
            sort_fn = _gpu_sort_shard if use_gpu else _cpu_sort_shard
            sorted_mags, shard_counts = sort_fn(feat_ids, feat_mags)

            # Spill sorted magnitudes via torch.save so Pass 2 can memory-map
            # them. The helper uses fp16 only when that is an exact finite
            # round-trip for the values in this shard, otherwise fp32.
            mags_path = tmp_dir / f"shard_{i:05d}_mags.pt"
            _spill_sorted_mags(sorted_mags, mags_path)

            sorted_shards.append(
                SortedShardInfo(
                    mags_path=mags_path,
                    n_activations=n_act,
                    per_feature_counts=shard_counts,
                )
            )
            del sorted_mags
        else:
            sorted_shards.append(
                SortedShardInfo(
                    mags_path=None,
                    n_activations=0,
                    per_feature_counts=t.zeros(0, dtype=t.int64),
                )
            )

        del data

    n_sae_features_detected = len(acc.occurrence)

    mag_subsample = t.cat(subsample_parts) if subsample_parts else t.tensor([], dtype=t.float32)
    active = acc.occurrence > 0
    global_mag_min = float(acc.mag_min[active].min().item()) if active.any() else 0.0

    logger.info(
        f"Pass 1 complete: {n_sae_features_detected} features, "
        f"{int(acc.occurrence.sum().item()):,} total activations, "
        f"{len(mag_subsample):,} subsample values, "
        f"temp files: {_sorted_spill_bytes(sorted_shards) / 1e9:.1f} GB"
    )

    p1 = Pass1Result(
        acc=acc,
        seg_acc=seg_acc,
        metadata=metadata,
        total_docs=total_docs,
        total_tokens=total_tokens,
        total_storage=total_storage,
        n_shards=len(pt_files),
        n_sae_features=n_sae_features_detected,
        shard_paths=pt_files,
    )
    return p1, sorted_shards, mag_subsample, global_mag_min


def merge_per_feature_percentiles(
    sorted_shards: list[SortedShardInfo],
    n_sae_features: int,
    occurrence: t.Tensor,
    mag_subsample: t.Tensor,
    global_mag_min: float,
    percentiles: tuple[int, ...] = PERCENTILES,
) -> tuple[dict[int, FeaturePercentiles], list[float] | None]:
    """Compute exact per-feature percentiles by merging sorted temp files.

    For each feature, reads its magnitudes from all shard temp files via
    pre-computed offsets, sorts them, and computes exact percentiles with
    linear interpolation.

    Memory: O(max_feature_occurrence + n_sae_features).
    I/O: One sequential pass through all sorted temp data.

    Args:
        sorted_shards: Sorted shard references from optimized_pass1.
        n_sae_features: Total number of SAE features.
        occurrence: Dense occurrence counts from Pass 1.
        mag_subsample: Magnitude subsample for p99/bin-edge estimation.
        global_mag_min: Global minimum magnitude from Pass 1.
        percentiles: Percentile values to compute.
    """
    active_ids, shard_offsets, shard_memmaps, shared_edges, mag_lo, mag_hi = _setup_merge_context(
        sorted_shards,
        n_sae_features,
        occurrence,
        mag_subsample,
        global_mag_min,
    )
    if len(active_ids) == 0:
        return {}, None

    n_hist_bins = 50

    logger.info(
        f"Pass 2: processing {len(active_ids)} active features, "
        f"bin edges [{mag_lo:.4f}, {mag_hi:.4f}]"
    )

    pct_fracs = t.tensor([p / 100.0 for p in percentiles], dtype=t.float64)
    result: dict[int, FeaturePercentiles] = {}

    for fid_t in tqdm(active_ids, desc="Pass 2: merge percentiles"):
        fid = int(fid_t.item())

        mags_f32 = _read_feature_mags(fid, shard_offsets, shard_memmaps)
        if mags_f32 is None:
            continue

        mags = mags_f32.to(t.float64)
        count = len(mags)

        assert count == int(occurrence[fid].item()), (
            f"Feature {fid}: merge count {count} != occurrence {int(occurrence[fid].item())}"
        )

        sorted_mags = mags.sort().values

        # ── Percentiles (linear interpolation) ──
        continuous_idx = (count - 1) * pct_fracs
        lower_idx = continuous_idx.floor().long()
        upper_idx = continuous_idx.ceil().long().clamp(max=count - 1)
        frac = continuous_idx - lower_idx.to(t.float64)
        quantile_vals = sorted_mags[lower_idx] + frac * (
            sorted_mags[upper_idx] - sorted_mags[lower_idx]
        )

        # ── Median ──
        median_cont = (count - 1) * 0.5
        median_lo = int(median_cont)
        median_hi = min(median_lo + 1, count - 1)
        median_frac = median_cont - median_lo
        median_val = sorted_mags[median_lo].item() + median_frac * (
            sorted_mags[median_hi].item() - sorted_mags[median_lo].item()
        )

        # ── Histogram ──
        hist_counts = _histogram_counts(sorted_mags, shared_edges, n_hist_bins)

        pcts = {f"p{p}": round(quantile_vals[j].item(), 6) for j, p in enumerate(percentiles)}
        result[fid] = FeaturePercentiles(
            percentiles=pcts,
            median=round(median_val, 6),
            histogram=hist_counts.tolist(),
        )

    # Clean up memmaps
    for mm in shard_memmaps:
        if mm is not None:
            del mm

    shared_bin_edges = [round(v, 6) for v in shared_edges.tolist()]
    return result, shared_bin_edges


def _setup_merge_context(
    sorted_shards: list[SortedShardInfo],
    n_sae_features: int,
    occurrence: t.Tensor,
    mag_subsample: t.Tensor,
    global_mag_min: float,
) -> tuple[
    t.Tensor,  # active_ids
    list[t.Tensor],  # shard_offsets
    list[t.Tensor | None],  # shard_mmaps (memory-mapped spill tensors)
    t.Tensor,  # shared_edges (float64, n_hist_bins+1)
    float,  # mag_lo
    float,  # mag_hi
]:
    """Shared setup for merge-based Pass 2 variants (CPU and GPU)."""
    active_mask = occurrence > 0
    active_ids = active_mask.nonzero(as_tuple=True)[0]

    shard_offsets: list[t.Tensor] = []
    shard_mmaps: list[t.Tensor | None] = []

    for si in sorted_shards:
        counts = si.per_feature_counts
        if len(counts) < n_sae_features:
            counts = t.cat(
                [
                    counts,
                    t.zeros(n_sae_features - len(counts), dtype=t.int64),
                ]
            )
        offsets = t.zeros(n_sae_features + 1, dtype=t.int64)
        offsets[1:] = counts.cumsum(0)

        assert offsets[-1].item() == si.n_activations, (
            f"Offset mismatch: offsets[-1]={offsets[-1].item()} vs n_activations={si.n_activations}"
        )
        shard_offsets.append(offsets)

        if si.mags_path is not None and si.n_activations > 0:
            shard_mmaps.append(t.load(si.mags_path, mmap=True, weights_only=True))
        else:
            shard_mmaps.append(None)

    n_hist_bins = 50
    if len(mag_subsample) > 0:
        sample_f64 = mag_subsample.to(t.float64)
        mag_lo = global_mag_min
        k99 = max(1, int(len(sample_f64) * 0.99))
        mag_p99 = sample_f64.kthvalue(k99).values.item()
        mag_hi = max(mag_p99, mag_lo + 1e-6)
    else:
        mag_lo, mag_hi = 0.0, 1.0
    shared_edges = t.linspace(mag_lo, mag_hi, n_hist_bins + 1, dtype=t.float64)

    return active_ids, shard_offsets, shard_mmaps, shared_edges, mag_lo, mag_hi


def _sorted_spill_bytes(sorted_shards: list[SortedShardInfo]) -> int:
    """Return actual bytes used by Pass-1 magnitude spill files."""
    total = 0
    for si in sorted_shards:
        if si.mags_path is not None and si.mags_path.exists():
            total += si.mags_path.stat().st_size
    return total


def _can_spill_as_fp16_losslessly(
    mags32: t.Tensor,
) -> bool:
    """Return True iff finite fp32 values round-trip through fp16 exactly.

    The check is intentionally chunked: production shards can contain billions
    of activations, and a whole-tensor fp16 plus fp32 round-trip copy would add
    avoidable peak memory to the sort path.
    """
    if mags32.numel() == 0:
        return False

    flat = mags32.view(-1)
    for start in range(0, flat.numel(), FP16_SPILL_CHECK_CHUNK_SIZE):
        chunk = flat[start : start + FP16_SPILL_CHECK_CHUNK_SIZE]
        if not bool(t.isfinite(chunk).all().item()):
            return False
        chunk16 = chunk.to(t.float16)
        roundtrip = chunk16.to(t.float32)
        is_exact = t.equal(roundtrip, chunk)
        del chunk16, roundtrip
        if not is_exact:
            return False
    return True


def _spill_sorted_mags(sorted_mags: t.Tensor, path: Path) -> None:
    """Spill sorted magnitudes for Pass-2 in the smallest *bitwise-lossless*
    precision: fp16 iff ``fp32 -> fp16 -> fp32`` is the exact identity for
    every element, else fp32.

    Why this is safe and halves Pass-1 scratch:

    Shard ``feature_mags`` are stored fp16 (``feature_io._optimize_dtype``
    *raises* rather than ever writing fp32 — ``FP16_SAFE_MAX`` guard), upcast
    fp16->fp32 on load, then only *permuted* by the sort (no arithmetic). So
    every spilled value in the production pipeline is exactly
    fp16-representable and the fp16 spill round-trips bit-for-bit at half the
    temp-file size — the fp32 spill doubled scratch for zero numerical gain
    (≈2.9 TB vs ≈1.45 TB on a 727 B-activation corpus).

    The original audit hazard — fp16 saturating magnitudes >65504 to +Inf and
    flushing sub-fp16-precision values — only bites when the input carries
    real fp32 precision (e.g. a future fp32 mags-storage mode, or a non-shard
    caller). The exact round-trip check below detects exactly that case and
    falls back to fp32, so the audit guarantee
    (phase2/03-numerical-correctness.md §1 P0 #1;
    phase1/08-compute-statistics.md) is preserved unconditionally — see the
    regression tests in tests/steering/test_compute_statistics_fp16_spill.py.

    ``_read_feature_mags`` already upcasts whatever dtype it reads to fp32,
    so the Pass-2 read path is unchanged either way.
    """
    # Pass 2 loads these files with ``mmap=True`` and slices CPU storage.
    # Keep the spill format CPU-backed even if a future caller hands this
    # helper a CUDA tensor.
    mags32 = sorted_mags.detach().to(device=t.device("cpu"), dtype=t.float32).contiguous()
    if _can_spill_as_fp16_losslessly(mags32):
        t.save(mags32.to(t.float16), path)
        return
    t.save(mags32, path)


def _read_feature_mags(
    fid: int,
    shard_offsets: list[t.Tensor],
    shard_mmaps: list[t.Tensor | None],
) -> t.Tensor | None:
    """Read all magnitudes for a single feature from memory-mapped shard files."""
    parts: list[t.Tensor] = []
    for si_idx in range(len(shard_mmaps)):
        start = int(shard_offsets[si_idx][fid].item())
        end = int(shard_offsets[si_idx][fid + 1].item())
        mmap = shard_mmaps[si_idx]
        if start < end and mmap is not None:
            # clone() forces a read from memory-mapped storage. Spills may be
            # fp16 or fp32; Pass 2 always computes from fp32 values.
            parts.append(mmap[start:end].clone().to(t.float32))
    if not parts:
        return None
    return t.cat(parts) if len(parts) > 1 else parts[0]


def vectorized_pass2_percentiles(
    sorted_shards: list[SortedShardInfo],
    n_sae_features: int,
    occurrence: t.Tensor,
    mag_subsample: t.Tensor,
    global_mag_min: float,
    percentiles: tuple[int, ...] = PERCENTILES,
    gpu_chunk_size: int = GPU_CHUNK_DEFAULT,
    device: DeviceType = DeviceType.CUDA,
) -> tuple[dict[int, FeaturePercentiles], list[float] | None]:
    """Vectorized Pass 2: batch per-feature percentiles using double-sort trick.

    Instead of processing one feature at a time (as in merge_per_feature_percentiles),
    this function batches many features together and uses the double-sort trick to
    compute percentiles, medians, and histograms for the entire batch at once on GPU.

    Algorithm:
    1. Use _setup_merge_context for shared setup (active IDs, memmaps, bin edges).
    2. Pre-allocate a CPU buffer with per-feature contiguous layout.
    3. Fill the buffer from memmaps (each feature gets a contiguous slice).
    4. Process batches of features: transfer to device, apply double-sort trick,
       vectorized percentile + histogram computation.

    Falls back to CPU when CUDA is not available.

    Args:
        sorted_shards: Sorted shard references from optimized_pass1.
        n_sae_features: Total number of SAE features.
        occurrence: Dense occurrence counts from Pass 1.
        mag_subsample: Magnitude subsample for p99/bin-edge estimation.
        global_mag_min: Global minimum magnitude from Pass 1.
        percentiles: Percentile values to compute.
        gpu_chunk_size: Max activations per GPU chunk.
        device: ``DeviceType.CUDA`` (default, vectorized GPU path) or
            ``DeviceType.CPU`` (feature-by-feature, avoids VRAM OOM on
            features with 100M+ activations). ``DeviceType.MPS`` is not
            currently supported by this function.

    Returns:
        Same as merge_per_feature_percentiles: (result_dict, shared_bin_edges).
    """
    if device is DeviceType.MPS:
        raise ValueError(
            f"vectorized_pass2_percentiles does not support {device!r}; "
            "use DeviceType.CUDA or DeviceType.CPU"
        )
    active_ids, shard_offsets, shard_memmaps, shared_edges, mag_lo, mag_hi = _setup_merge_context(
        sorted_shards,
        n_sae_features,
        occurrence,
        mag_subsample,
        global_mag_min,
    )
    if len(active_ids) == 0:
        return {}, None

    n_hist_bins = 50
    use_gpu = device is DeviceType.CUDA
    torch_device = t.device("cuda") if use_gpu else t.device("cpu")

    # Auto-size gpu_chunk_size based on free VRAM if GPU is available.
    if use_gpu:
        free_vram, _ = t.cuda.mem_get_info()
        vram_budget = max(1_000_000, int(free_vram * 0.6 / _BYTES_PER_ACT_PASS2))
        if vram_budget < gpu_chunk_size:
            gpu_chunk_size = vram_budget

    total_activations = int(occurrence.sum().item())

    logger.info(
        f"Vectorized Pass 2: streaming from memmaps for {len(active_ids)} active features "
        f"({total_activations:,} total activations), "
        f"bin edges [{mag_lo:.4f}, {mag_hi:.4f}]"
    )

    # ── Process features in batches using double-sort trick ──
    pct_fracs = t.tensor([p / 100.0 for p in percentiles], dtype=t.float64, device=torch_device)
    shared_edges_dev = shared_edges.to(torch_device)
    result: dict[int, FeaturePercentiles] = {}

    # Determine batch boundaries: group consecutive active features so total
    # activations per batch fit within gpu_chunk_size
    active_list = active_ids.tolist()
    n_active = len(active_list)
    batch_start_idx = 0
    batch_num = 0

    while batch_start_idx < n_active:
        # Accumulate features until we hit the chunk size limit
        batch_end_idx = batch_start_idx
        batch_total = 0
        while batch_end_idx < n_active:
            fid = active_list[batch_end_idx]
            fid_count = int(occurrence[fid].item())
            if batch_total + fid_count > gpu_chunk_size and batch_end_idx > batch_start_idx:
                break
            batch_total += fid_count
            batch_end_idx += 1

        if batch_total == 0:
            batch_start_idx = batch_end_idx
            continue

        batch_fids = active_list[batch_start_idx:batch_end_idx]
        n_batch = len(batch_fids)
        batch_num += 1
        logger.info(
            f"Pass 2 batch {batch_num}: features {batch_start_idx}-{batch_end_idx} "
            f"({n_batch} features, {batch_total:,} activations, "
            f"{batch_start_idx}/{n_active} done)"
        )

        # Extract contiguous magnitude data for this batch
        # Build counts and feature ID tensors for the batch
        batch_counts = t.tensor(
            [int(occurrence[fid].item()) for fid in batch_fids],
            dtype=t.int64,
        )
        batch_n = int(batch_counts.sum().item())

        # Gather magnitudes from memmaps into a contiguous tensor
        batch_mags = t.zeros(batch_n, dtype=t.float32)
        batch_ids = t.zeros(batch_n, dtype=t.int64)
        pos = 0
        for local_idx, fid in enumerate(batch_fids):
            mags_f32 = _read_feature_mags(fid, shard_offsets, shard_memmaps)
            if mags_f32 is not None:
                cnt = len(mags_f32)
                batch_mags[pos : pos + cnt] = mags_f32
                batch_ids[pos : pos + cnt] = local_idx
                pos += cnt

        # Transfer to device
        mags_dev = batch_mags.to(t.float64).to(torch_device)
        ids_dev = batch_ids.to(torch_device)
        del batch_mags, batch_ids

        # ── Double-sort: sort by magnitude, then stable sort by local feature index ──
        sort_by_mag = mags_dev.argsort()
        ids_by_mag = ids_dev[sort_by_mag]
        mags_by_mag = mags_dev[sort_by_mag]
        del sort_by_mag

        sort_by_id = ids_by_mag.argsort(stable=True)
        sorted_ids = ids_by_mag[sort_by_id]
        sorted_mags = mags_by_mag[sort_by_id]
        del ids_by_mag, mags_by_mag, sort_by_id, ids_dev, mags_dev

        # ── Group boundaries ──
        unique_local_ids, counts = t.unique_consecutive(sorted_ids, return_counts=True)
        n_groups = len(unique_local_ids)
        starts = t.zeros(n_groups + 1, dtype=t.int64, device=torch_device)
        starts[1:] = counts.cumsum(0)
        feature_starts = starts[:-1]

        # ── Vectorized quantile computation ──
        counts_f = counts.to(t.float64)
        continuous_idx = (counts_f.unsqueeze(1) - 1) * pct_fracs.unsqueeze(0)
        lower_idx = continuous_idx.floor().long()
        upper_idx = continuous_idx.ceil().long().clamp(max=(counts.unsqueeze(1) - 1))
        frac = continuous_idx - lower_idx.to(t.float64)

        global_lower = feature_starts.unsqueeze(1) + lower_idx
        global_upper = feature_starts.unsqueeze(1) + upper_idx
        lower_vals = sorted_mags[global_lower]
        upper_vals = sorted_mags[global_upper]
        quantile_values = lower_vals + frac * (upper_vals - lower_vals)

        # ── Median computation ──
        median_continuous = (counts_f - 1) * 0.5
        median_lower = median_continuous.floor().long()
        median_upper = median_continuous.ceil().long().clamp(max=(counts - 1))
        median_frac = median_continuous - median_lower.to(t.float64)
        medians = sorted_mags[feature_starts + median_lower] + median_frac * (
            sorted_mags[feature_starts + median_upper] - sorted_mags[feature_starts + median_lower]
        )

        # ── Vectorized histogram computation ──
        clamped = sorted_mags.clamp(shared_edges_dev[0], shared_edges_dev[-1])
        bin_idx = t.bucketize(clamped, shared_edges_dev[1:])
        bin_idx = bin_idx.clamp(max=n_hist_bins - 1)

        feature_membership = t.repeat_interleave(
            t.arange(n_groups, dtype=t.int64, device=torch_device),
            counts,
        )
        combined_idx = feature_membership * n_hist_bins + bin_idx
        histogram_flat = t.zeros(n_groups * n_hist_bins, dtype=t.int64, device=torch_device)
        histogram_flat.scatter_add_(0, combined_idx, t.ones_like(combined_idx))
        histograms = histogram_flat.reshape(n_groups, n_hist_bins)

        # ── Build result entries (transfer to CPU) ──
        unique_local_ids_cpu = unique_local_ids.cpu()
        quantile_values_cpu = quantile_values.cpu()
        medians_cpu = medians.cpu()
        histograms_cpu = histograms.cpu()

        for i in range(n_groups):
            local_idx = int(unique_local_ids_cpu[i].item())
            fid = batch_fids[local_idx]
            pcts = {
                f"p{p}": round(quantile_values_cpu[i, j].item(), 6)
                for j, p in enumerate(percentiles)
            }
            result[fid] = FeaturePercentiles(
                percentiles=pcts,
                median=round(medians_cpu[i].item(), 6),
                histogram=histograms_cpu[i].tolist(),
            )

        del sorted_ids, sorted_mags, unique_local_ids, counts
        del quantile_values, medians, histograms, histogram_flat
        del feature_membership, combined_idx, clamped, bin_idx
        if use_gpu:
            t.cuda.empty_cache()

        batch_start_idx = batch_end_idx

    # Clean up memmaps
    for mm in shard_memmaps:
        if mm is not None:
            del mm

    shared_bin_edges = [round(v, 6) for v in shared_edges.tolist()]
    return result, shared_bin_edges


# ---------------------------------------------------------------------------


def stream_pass1(
    feature_dir: str,
    n_sae_features: int | None = None,
    device: str = "cpu",
    gpu_chunk_size: int = GPU_CHUNK_DEFAULT,
) -> Pass1Result:
    """Streaming Pass 1: aggregate stats by loading one shard at a time.

    Equivalent to load_all_shards() + accumulation loop but uses O(1 shard) memory.

    Args:
        feature_dir: Directory containing feature shard files.
        n_sae_features: Number of SAE features (auto-detected if None).
        device: ``"cpu"`` or ``"cuda"`` for GPU-accelerated scatter.
        gpu_chunk_size: Max activations per GPU chunk (ignored when device="cpu").
    """
    pt_files = glob_features(feature_dir)
    if not pt_files:
        raise FileNotFoundError(f"No feature shard files in {feature_dir}")

    use_gpu = device == "cuda" and t.cuda.is_available()

    # Extract metadata from first shard
    first_data = load_features(pt_files[0])
    metadata = ShardMetadata(
        pooling=first_data["pooling"],
        segment_mode=SegmentMode(first_data["segment_mode"]),
        segment_size=first_data["segment_size"],
        context_window=first_data["context_window"],
        layer=first_data["layer"],
    )

    # Determine n_sae_features if not provided
    if n_sae_features is None:
        max_id = (
            int(first_data["feature_ids"].max().item()) if len(first_data["feature_ids"]) > 0 else 0
        )
        del first_data
        for pt_path in pt_files[1:]:
            data = load_features(pt_path)
            if len(data["feature_ids"]) > 0:
                max_id = max(max_id, int(data["feature_ids"].max().item()))
            del data
        n_sae_features = max_id + 1
    else:
        del first_data

    acc = init_accumulators(n_sae_features)
    seg_acc = init_segment_stats()
    total_docs = 0
    total_tokens = 0
    total_storage = 0

    desc = "Pass 1 (GPU): aggregating shard stats" if use_gpu else "Pass 1: aggregating shard stats"

    for pt_path in tqdm(pt_files, desc=desc):
        data = load_features(pt_path)
        _assert_nonempty_base_shard(data, pt_path)
        if use_gpu:
            gpu_accumulate_shard(acc, data["feature_ids"], data["feature_mags"], gpu_chunk_size)
        else:
            accumulate_shard(acc, data["feature_ids"], data["feature_mags"])
        accumulate_segment_stats(seg_acc, data)
        total_docs += data["n_docs"]
        total_tokens += int(data["doc_token_counts"].sum().item())
        total_storage += pt_path.stat().st_size
        del data

    return Pass1Result(
        acc=acc,
        seg_acc=seg_acc,
        metadata=metadata,
        total_docs=total_docs,
        total_tokens=total_tokens,
        total_storage=total_storage,
        n_shards=len(pt_files),
        n_sae_features=n_sae_features,
        shard_paths=pt_files,
    )


def stream_per_feature_percentiles(
    shard_paths: list[Path],
    occurrence: t.Tensor,
    percentiles: tuple[int, ...] = PERCENTILES,
    batch_size: int = 1000,
) -> tuple[dict[int, FeaturePercentiles], list[float] | None]:
    """Compute exact per-feature percentiles by streaming shards in batches.

    Processes features in groups of *batch_size*, re-reading all shards for each
    group.  Memory usage: O(batch_size * max_occurrences_in_batch + 1 shard).

    Args:
        shard_paths: Sorted list of shard file paths.
        occurrence: Dense tensor of per-feature occurrence counts from Pass 1.
        percentiles: Percentile values to compute (e.g. (10, 20, ..., 99)).
        batch_size: Number of features to process per pass over shards.

    Returns:
        Same as compute_per_feature_percentiles: (result_dict, shared_bin_edges).
    """
    active_mask = occurrence > 0
    active_ids = active_mask.nonzero(as_tuple=True)[0]
    if len(active_ids) == 0:
        return {}, None

    # ── First pass: collect ALL magnitudes to compute shared histogram bin edges ──
    # TODO: For very large datasets (hundreds of shards), this O(total_mags) pass
    # may not fit in memory. Consider streaming min/kthvalue estimation instead.
    all_mags_list: list[t.Tensor] = []
    for pt_path in tqdm(shard_paths, desc="Pass 2a: collecting magnitudes for bin edges"):
        data = load_features(pt_path)
        all_mags_list.append(data["feature_mags"].to(t.float64))
        del data

    all_mags = t.cat(all_mags_list)
    del all_mags_list

    n_hist_bins = 50
    mag_lo = all_mags.min().item()
    k99 = max(1, int(len(all_mags) * 0.99))
    mag_p99 = all_mags.kthvalue(k99).values.item()
    mag_hi = max(mag_p99, mag_lo + 1e-6)
    shared_edges_tensor = t.linspace(mag_lo, mag_hi, n_hist_bins + 1, dtype=t.float64)
    del all_mags

    # ── Process features in batches ──
    pct_fracs = t.tensor([p / 100.0 for p in percentiles], dtype=t.float64)
    result: dict[int, FeaturePercentiles] = {}

    n_active = len(active_ids)
    for batch_start in tqdm(
        range(0, n_active, batch_size),
        desc="Pass 2b: batched per-feature percentiles",
        total=(n_active + batch_size - 1) // batch_size,
    ):
        batch_end = min(batch_start + batch_size, n_active)
        batch_fids = active_ids[batch_start:batch_end]  # feature IDs in this batch
        n_batch = len(batch_fids)

        # Pre-allocate magnitude storage for this batch
        batch_buffers: list[list[float]] = [[] for _ in range(n_batch)]

        # Build a lookup: feature_id -> batch index (for fast membership testing)
        fid_to_batch_idx = t.full((len(occurrence),), -1, dtype=t.int64)
        fid_to_batch_idx[batch_fids] = t.arange(n_batch, dtype=t.int64)

        # Stream all shards, collecting magnitudes for this batch
        for pt_path in shard_paths:
            data = load_features(pt_path)
            shard_ids = data["feature_ids"].to(t.int64)
            shard_mags = data["feature_mags"].to(t.float64)
            del data

            # Vectorized masking: which activations belong to this batch?
            # Clamp to valid range to avoid index-out-of-bounds
            safe_ids = shard_ids.clamp(max=len(occurrence) - 1)
            batch_indices = fid_to_batch_idx[safe_ids]
            mask = batch_indices >= 0

            if mask.any():
                matched_indices = batch_indices[mask]
                matched_mags = shard_mags[mask]
                for bi, mag in zip(matched_indices.tolist(), matched_mags.tolist(), strict=True):
                    batch_buffers[bi].append(mag)

            del shard_ids, shard_mags

        # Compute percentiles and histograms for this batch
        for local_idx in range(n_batch):
            fid = int(batch_fids[local_idx].item())
            mags_list = batch_buffers[local_idx]
            if not mags_list:
                continue

            mags_t = t.tensor(mags_list, dtype=t.float64)
            mags_sorted = mags_t.sort().values
            count = len(mags_sorted)

            # Quantile computation (linear interpolation, matching reference)
            pcts: dict[str, float] = {}
            for j, p in enumerate(percentiles):
                continuous_idx = (count - 1) * pct_fracs[j].item()
                lower = int(continuous_idx)
                upper = min(lower + 1, count - 1)
                frac = continuous_idx - lower
                val = mags_sorted[lower].item() + frac * (
                    mags_sorted[upper].item() - mags_sorted[lower].item()
                )
                pcts[f"p{p}"] = round(val, 6)

            # Median
            median_continuous = (count - 1) * 0.5
            median_lower = int(median_continuous)
            median_upper = min(median_lower + 1, count - 1)
            median_frac = median_continuous - median_lower
            median_val = mags_sorted[median_lower].item() + median_frac * (
                mags_sorted[median_upper].item() - mags_sorted[median_lower].item()
            )

            # Histogram using shared bin edges
            hist_counts = _histogram_counts(mags_sorted, shared_edges_tensor, n_hist_bins)

            result[fid] = FeaturePercentiles(
                percentiles=pcts,
                median=round(median_val, 6),
                histogram=hist_counts.tolist(),
            )

        del batch_buffers, fid_to_batch_idx

    shared_bin_edges = [round(v, 6) for v in shared_edges_tensor.tolist()]
    return result, shared_bin_edges


def gpu_exact_percentiles(
    shard_paths: list[Path],
    occurrence: t.Tensor,
    percentiles: tuple[int, ...] = PERCENTILES,
    gpu_chunk_size: int = GPU_CHUNK_DEFAULT,
) -> tuple[dict[int, FeaturePercentiles], list[float] | None]:
    """Compute exact per-feature percentiles via GPU-accelerated single-pass.

    Algorithm:
    1. Pre-allocate CPU buffer for all feature magnitudes (size known from Pass 1).
    2. Stream each shard once: GPU-sort by feature_id, write contiguous groups
       into the pre-allocated buffer (no I/O amplification).
    3. For each feature: GPU-sort its magnitudes and compute exact percentiles
       via linear interpolation.

    ~20x faster than stream_per_feature_percentiles on 3 shards (77s vs 25 min).

    Args:
        shard_paths: Sorted list of shard file paths.
        occurrence: Dense tensor of per-feature occurrence counts from Pass 1.
        percentiles: Percentile values to compute.
        gpu_chunk_size: Max activations per GPU chunk during collection.

    Returns:
        Same as compute_per_feature_percentiles: (result_dict, shared_bin_edges).
    """
    active_mask = occurrence > 0
    active_ids = active_mask.nonzero(as_tuple=True)[0]
    if len(active_ids) == 0:
        return {}, None

    n_features_total = len(occurrence)
    total_activations = int(occurrence.sum().item())

    # ── Pre-allocate CPU buffer for all magnitudes ──
    # Each feature gets a contiguous slice: [offsets[fid] : offsets[fid] + occurrence[fid]]
    offsets = t.zeros(n_features_total + 1, dtype=t.int64)
    offsets[1:] = occurrence.cumsum(0)

    logger.info(
        f"GPU Pass 2: allocating {total_activations * 4 / 1e9:.2f} GB CPU buffer "
        f"for {int(active_mask.sum().item())} active features"
    )
    mag_buffer = t.zeros(total_activations, dtype=t.float32)
    fill_counts = t.zeros(n_features_total, dtype=t.int64)

    # ── Phase 1: Collect magnitudes from all shards (single pass) ──
    for shard_path in tqdm(shard_paths, desc="GPU Pass 2a: collecting magnitudes"):
        data = load_features(shard_path)
        shard_ids = data["feature_ids"]
        shard_mags = data["feature_mags"]
        del data

        n_shard = len(shard_ids)
        for chunk_start in range(0, n_shard, gpu_chunk_size):
            chunk_end = min(chunk_start + gpu_chunk_size, n_shard)
            chunk_ids = shard_ids[chunk_start:chunk_end].to(t.int64)
            chunk_mags = shard_mags[chunk_start:chunk_end]

            # GPU sort by feature_id for contiguous groups
            device = t.device("cuda")
            gpu_ids = chunk_ids.to(device)
            gpu_mags = chunk_mags.to(device)
            sort_order = gpu_ids.argsort(stable=True)
            sorted_ids = gpu_ids[sort_order]
            sorted_mags = gpu_mags[sort_order].cpu()
            sorted_ids_cpu = sorted_ids.cpu()
            del gpu_ids, gpu_mags, sort_order, sorted_ids

            # Get contiguous groups
            unique_fids, group_counts = t.unique_consecutive(sorted_ids_cpu, return_counts=True)

            # Write each group's magnitudes into the pre-allocated buffer
            group_start = 0
            for raw_fid, cnt in zip(unique_fids.tolist(), group_counts.tolist(), strict=True):
                fid = int(raw_fid)
                write_start = int(offsets[fid].item() + fill_counts[fid].item())
                mag_buffer[write_start : write_start + cnt] = sorted_mags[
                    group_start : group_start + cnt
                ]
                fill_counts[fid] += cnt
                group_start += cnt

    # Sanity check: all magnitudes collected
    assert t.equal(fill_counts, occurrence), (
        "fill_counts mismatch: not all magnitudes were collected"
    )

    # ── Phase 2: Compute shared histogram bin edges ──
    n_hist_bins = 50
    buf = mag_buffer[:total_activations]
    mag_lo = buf.min().item()

    # GPU kthvalue for p99 (avoids O(n log n) CPU sort of 2.6B+ elements)
    k99 = max(1, int(total_activations * 0.99))
    device = t.device("cuda")
    # Process in chunks to fit GPU VRAM
    KTHVAL_CHUNK = gpu_chunk_size
    if total_activations <= KTHVAL_CHUNK:
        mag_p99 = buf.to(device).kthvalue(k99).values.item()
    else:
        # Chunked approach: sort each chunk, keep top 1%, merge
        top_n = total_activations - k99 + 1  # how many values >= p99
        candidates = []
        for i in range(0, total_activations, KTHVAL_CHUNK):
            chunk = buf[i : i + KTHVAL_CHUNK].to(device)
            # Keep the top_n largest from each chunk
            n_keep = min(top_n, len(chunk))
            topk_vals = chunk.topk(n_keep).values
            candidates.append(topk_vals.cpu())
            del chunk, topk_vals
        merged = t.cat(candidates).to(device)
        # p99 = the smallest of the top 1%
        mag_p99 = merged.kthvalue(min(top_n, len(merged))).values.item()
        del merged, candidates

    mag_hi = max(mag_p99, mag_lo + 1e-6)
    shared_edges = t.linspace(mag_lo, mag_hi, n_hist_bins + 1, dtype=t.float64)
    del buf

    # ── Phase 3: Per-feature percentiles + histograms on GPU ──
    device = t.device("cuda")
    pct_fracs = t.tensor([p / 100.0 for p in percentiles], dtype=t.float64, device=device)
    shared_edges_gpu = shared_edges.to(device)
    result: dict[int, FeaturePercentiles] = {}

    # Process features in batches for GPU efficiency
    FEATURE_BATCH = 512
    active_list = active_ids.tolist()

    for batch_start in tqdm(
        range(0, len(active_list), FEATURE_BATCH),
        desc="GPU Pass 2b: per-feature percentiles",
        total=(len(active_list) + FEATURE_BATCH - 1) // FEATURE_BATCH,
    ):
        batch_fids = active_list[batch_start : batch_start + FEATURE_BATCH]

        for raw_fid in batch_fids:
            fid = int(raw_fid)
            start = int(offsets[fid].item())
            end = int(offsets[fid + 1].item())
            count = end - start
            if count == 0:
                continue

            # Move feature magnitudes to GPU and sort
            feature_mags = mag_buffer[start:end].to(t.float64).to(device)
            sorted_mags = feature_mags.sort().values

            # Quantiles via linear interpolation
            continuous_idx = (count - 1) * pct_fracs
            lower_idx = continuous_idx.floor().long()
            upper_idx = continuous_idx.ceil().long().clamp(max=count - 1)
            frac = continuous_idx - lower_idx.to(t.float64)
            quantile_vals = sorted_mags[lower_idx] + frac * (
                sorted_mags[upper_idx] - sorted_mags[lower_idx]
            )

            # Median
            median_cont = (count - 1) * 0.5
            median_lo = int(median_cont)
            median_hi = min(median_lo + 1, count - 1)
            median_frac = median_cont - median_lo
            median_val = sorted_mags[median_lo].item() + median_frac * (
                sorted_mags[median_hi].item() - sorted_mags[median_lo].item()
            )

            # Histogram
            hist_counts = _histogram_counts(sorted_mags, shared_edges_gpu, n_hist_bins)

            pcts = {f"p{p}": round(quantile_vals[j].item(), 6) for j, p in enumerate(percentiles)}
            result[fid] = FeaturePercentiles(
                percentiles=pcts,
                median=round(median_val, 6),
                histogram=hist_counts.cpu().tolist(),
            )

    shared_bin_edges = [round(v, 6) for v in shared_edges.tolist()]
    return result, shared_bin_edges


def gpu_merge_per_feature_percentiles(
    sorted_shards: list[SortedShardInfo],
    n_sae_features: int,
    occurrence: t.Tensor,
    mag_subsample: t.Tensor,
    global_mag_min: float,
    percentiles: tuple[int, ...] = PERCENTILES,
    vram_budget_bytes: int = GPU_MERGE_VRAM_BUDGET_BYTES,
) -> tuple[dict[int, FeaturePercentiles], list[float] | None]:
    """GPU-accelerated per-feature percentiles from sorted temp files.

    Same interface and exact results as merge_per_feature_percentiles but
    transfers each feature's magnitudes to GPU for sorting, giving a
    significant speedup for features with many activations.

    Features with fewer than GPU_SORT_THRESHOLD activations are sorted on
    CPU to avoid transfer overhead dominating.

    Memory: O(vram_budget_bytes) GPU + O(max_feature_occurrence) CPU.
    I/O: Same single-pass memmap reads as CPU merge.
    """
    active_ids, shard_offsets, shard_memmaps, shared_edges, mag_lo, mag_hi = _setup_merge_context(
        sorted_shards,
        n_sae_features,
        occurrence,
        mag_subsample,
        global_mag_min,
    )
    if len(active_ids) == 0:
        return {}, None

    n_hist_bins = 50
    device = t.device("cuda")
    pct_fracs = t.tensor([p / 100.0 for p in percentiles], dtype=t.float64, device=device)
    shared_edges_gpu = shared_edges.to(device)

    # Threshold: features with fewer activations use CPU sort
    GPU_SORT_THRESHOLD = 50_000

    logger.info(
        f"GPU Pass 2: processing {len(active_ids)} active features, "
        f"bin edges [{mag_lo:.4f}, {mag_hi:.4f}], "
        f"GPU sort threshold: {GPU_SORT_THRESHOLD:,}"
    )

    result: dict[int, FeaturePercentiles] = {}

    # Process features in VRAM-budget-limited batches
    # Group features by occurrence and batch so total mags fit in VRAM budget
    batch_fids: list[int] = []
    batch_bytes: int = 0
    bytes_per_elem = 8  # float64

    def _process_batch(fids: list[int]) -> None:
        """Process a batch of features: read mags, GPU sort, compute percentiles."""
        for fid in fids:
            mags_f32 = _read_feature_mags(fid, shard_offsets, shard_memmaps)
            if mags_f32 is None:
                continue

            count = len(mags_f32)
            assert count == int(occurrence[fid].item()), (
                f"Feature {fid}: merge count {count} != occurrence {int(occurrence[fid].item())}"
            )

            if count >= GPU_SORT_THRESHOLD:
                # GPU sort path
                mags_gpu = mags_f32.to(t.float64).to(device)
                sorted_mags = mags_gpu.sort().values

                # Percentiles on GPU
                continuous_idx = (count - 1) * pct_fracs
                lower_idx = continuous_idx.floor().long()
                upper_idx = continuous_idx.ceil().long().clamp(max=count - 1)
                frac = continuous_idx - lower_idx.to(t.float64)
                quantile_vals = sorted_mags[lower_idx] + frac * (
                    sorted_mags[upper_idx] - sorted_mags[lower_idx]
                )

                # Median on GPU
                median_cont = (count - 1) * 0.5
                median_lo = int(median_cont)
                median_hi = min(median_lo + 1, count - 1)
                median_frac = median_cont - median_lo
                median_val = sorted_mags[median_lo].item() + median_frac * (
                    sorted_mags[median_hi].item() - sorted_mags[median_lo].item()
                )

                # Histogram on GPU
                hist_counts = _histogram_counts(sorted_mags, shared_edges_gpu, n_hist_bins)

                pcts = {
                    f"p{p}": round(quantile_vals[j].item(), 6) for j, p in enumerate(percentiles)
                }
                result[fid] = FeaturePercentiles(
                    percentiles=pcts,
                    median=round(median_val, 6),
                    histogram=hist_counts.cpu().tolist(),
                )
                del mags_gpu, sorted_mags
            else:
                # CPU sort path (faster for small features)
                mags_f64 = mags_f32.to(t.float64)
                sorted_mags = mags_f64.sort().values

                pct_fracs_cpu = t.tensor(
                    [p / 100.0 for p in percentiles],
                    dtype=t.float64,
                )
                continuous_idx = (count - 1) * pct_fracs_cpu
                lower_idx = continuous_idx.floor().long()
                upper_idx = continuous_idx.ceil().long().clamp(max=count - 1)
                frac = continuous_idx - lower_idx.to(t.float64)
                quantile_vals = sorted_mags[lower_idx] + frac * (
                    sorted_mags[upper_idx] - sorted_mags[lower_idx]
                )

                median_cont = (count - 1) * 0.5
                median_lo = int(median_cont)
                median_hi = min(median_lo + 1, count - 1)
                median_frac = median_cont - median_lo
                median_val = sorted_mags[median_lo].item() + median_frac * (
                    sorted_mags[median_hi].item() - sorted_mags[median_lo].item()
                )

                hist_counts = _histogram_counts(sorted_mags, shared_edges, n_hist_bins)

                pcts = {
                    f"p{p}": round(quantile_vals[j].item(), 6) for j, p in enumerate(percentiles)
                }
                result[fid] = FeaturePercentiles(
                    percentiles=pcts,
                    median=round(median_val, 6),
                    histogram=hist_counts.tolist(),
                )

    for fid_t in tqdm(active_ids, desc="GPU Pass 2: merge percentiles"):
        fid = int(fid_t.item())
        feat_bytes = int(occurrence[fid].item()) * bytes_per_elem
        if batch_bytes + feat_bytes > vram_budget_bytes and batch_fids:
            _process_batch(batch_fids)
            batch_fids = []
            batch_bytes = 0
        batch_fids.append(fid)
        batch_bytes += feat_bytes

    if batch_fids:
        _process_batch(batch_fids)

    # Clean up memmaps
    for mm in shard_memmaps:
        if mm is not None:
            del mm

    shared_bin_edges = [round(v, 6) for v in shared_edges.tolist()]
    return result, shared_bin_edges
