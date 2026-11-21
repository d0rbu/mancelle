# steering/eval/axbench_bpb/context_cache.py
"""Stage 1: prepare_context_cache — extract per-prefix SAE features.

Reuses steering.extract_features for the actual Gemma+SAE forward.
"""
from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from safetensors.torch import load_file, save_file

from steering.conditional.discretization import discretize_activations
from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.eval.axbench_bpb.paths import OutputPaths
from steering.eval.axbench_bpb.sample_loader import collect_unique_prefixes
from steering.eval.axbench_bpb.types import AxBenchBpbError
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.extract_features import extract_document_features
from steering.feature_paths import PoolingMode
from steering.segmentation import SegmentMode


def threshold_and_discretize_features(
    feature_ids: torch.Tensor,
    feature_mags: torch.Tensor,
    *,
    boundaries: torch.Tensor,
    extraction_threshold: float,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Apply AxBench context-cache thresholding and magnitude bucketing.

    This is the single sparse-feature conversion used by context-cache segments
    and roofline general-val output-window bags when a non-empty sparse signal is
    REQUIRED. It preserves the input feature order, keeps only ``mag >
    extraction_threshold``, and raises if the raw or thresholded signal is empty.
    """

    if feature_ids.ndim != 1 or feature_mags.ndim != 1:
        raise AxBenchBpbError(
            "sparse feature signal must be one-dimensional: got "
            f"feature_ids shape={tuple(feature_ids.shape)} and "
            f"feature_mags shape={tuple(feature_mags.shape)}"
        )
    if feature_ids.numel() != feature_mags.numel():
        raise AxBenchBpbError(
            "sparse feature signal has mismatched feature id/magnitude counts: "
            f"{feature_ids.numel()} ids vs {feature_mags.numel()} magnitudes"
        )
    if feature_ids.numel() == 0:
        raise AxBenchBpbError(
            "cannot threshold/discretize an empty sparse feature signal"
        )
    if feature_ids.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise AxBenchBpbError(
            "sparse feature ids must use an integer dtype: got "
            f"{feature_ids.dtype}"
        )
    if not feature_mags.is_floating_point():
        raise AxBenchBpbError(
            "sparse feature magnitudes must use a floating-point dtype: got "
            f"{feature_mags.dtype}"
        )
    if not bool(torch.isfinite(feature_mags).all().item()):
        n_bad = int((~torch.isfinite(feature_mags)).sum().item())
        raise AxBenchBpbError(
            f"sparse feature signal contains {n_bad} non-finite magnitude(s)"
        )
    if boundaries.ndim != 2:
        raise AxBenchBpbError(
            "bucket boundaries must be a two-dimensional tensor: got "
            f"shape={tuple(boundaries.shape)}"
        )
    if bool((feature_ids < 0).any().item()):
        bad = feature_ids[feature_ids < 0].detach().to("cpu").tolist()[:5]
        raise AxBenchBpbError(
            f"sparse feature ids must be non-negative; first bad ids: {bad}"
        )
    n_boundary_features = int(boundaries.shape[0])
    if bool((feature_ids >= n_boundary_features).any().item()):
        bad = (
            feature_ids[feature_ids >= n_boundary_features]
            .detach()
            .to("cpu")
            .tolist()[:5]
        )
        raise AxBenchBpbError(
            "sparse feature ids exceed bucket boundary table size "
            f"{n_boundary_features}; first bad ids: {bad}"
        )

    mask = feature_mags > float(extraction_threshold)
    kept_fids = feature_ids[mask].to(torch.int64)
    kept_mags = feature_mags[mask].to(torch.float32)
    if kept_fids.numel() == 0:
        raise AxBenchBpbError(
            "sparse feature signal has zero features above "
            f"extraction_threshold={float(extraction_threshold)!r}"
        )

    buckets = discretize_activations(
        kept_fids,
        kept_mags,
        boundaries,
    ).to(torch.int64)
    return (
        tuple(int(x) for x in kept_fids.tolist()),
        tuple(int(x) for x in buckets.tolist()),
    )


def serialize_context_segments(
    segments: tuple[ContextFeatureSegment, ...], path: Path
) -> None:
    """Atomic write of context segments as safetensors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tensors: dict[str, torch.Tensor] = {}
    for i, seg in enumerate(segments):
        tensors[f"seg_{i}_feature_ids"] = torch.tensor(
            seg.feature_ids, dtype=torch.int64
        )
        tensors[f"seg_{i}_buckets"] = torch.tensor(seg.buckets, dtype=torch.int64)
        tensors[f"seg_{i}_char_bounds"] = torch.tensor(
            [seg.char_start, seg.char_end], dtype=torch.int64
        )
    tensors["n_segments"] = torch.tensor([len(segments)], dtype=torch.int64)
    save_file(tensors, str(tmp))
    tmp.rename(path)


def load_context_segments(path: Path) -> tuple[ContextFeatureSegment, ...]:
    """Reload context segments from safetensors."""
    tensors = load_file(str(path))
    n = int(tensors["n_segments"].item())
    segs: list[ContextFeatureSegment] = []
    for i in range(n):
        fids = tuple(int(x) for x in tensors[f"seg_{i}_feature_ids"].tolist())
        bkts = tuple(int(x) for x in tensors[f"seg_{i}_buckets"].tolist())
        start, end = tensors[f"seg_{i}_char_bounds"].tolist()
        segs.append(
            ContextFeatureSegment(
                feature_ids=fids,
                buckets=bkts,
                char_start=int(start),
                char_end=int(end),
            )
        )
    return tuple(segs)


def extract_prefix_features(
    *,
    prefix_text: str,
    base_model: Any,
    base_sae: Any,
    boundaries: torch.Tensor,
    segment_size: int,
    sae_layer: int,
    extraction_max_tokens: int,
    extraction_threshold: float,
) -> tuple[ContextFeatureSegment, ...]:
    """Run base Gemma + SAE over ``prefix_text`` and return per-segment
    ``ContextFeatureSegment`` tuples ready for the context cache.

    The extraction pipeline is delegated end-to-end to
    :func:`steering.extract_features.extract_document_features` — the exact
    function used at training time — so there is no drift risk between
    training and evaluation. ``extract_document_features`` handles
    head-truncation, a single forward pass through the base model, SAE
    activation capture, segment boundary precompute, per-segment pooling,
    and sparse feature extraction.

    After extraction, each segment is (1) magnitude-thresholded by
    ``extraction_threshold`` (keep features with ``mag > threshold``)
    and (2) discretized into bucket indices via
    :func:`steering.conditional.discretization.discretize_activations`
    using the supplied ``boundaries`` table.

    Args:
        prefix_text: Raw prefix text to extract features from.
        base_model: HookedTransformer instance for the base Gemma model.
        base_sae: SAELens SAE to run on the base model activations.
        boundaries: ``[n_features, num_buckets - 1]`` float64 per-feature
            bucket boundaries. Forwarded to ``discretize_activations``.
        segment_size: Token window size for fixed-token segmentation.
        sae_layer: Residual stream layer index the SAE was trained on.
        extraction_max_tokens: Max SAE tokens per forward pass. Clamped
            by ``extract_document_features`` to ``min(max_tokens,
            base_model.cfg.n_ctx)``.
        extraction_threshold: Magnitude cutoff applied per segment —
            features with ``mag > extraction_threshold`` survive. The
            default ``0.0`` matches the canonical axbench extraction
            setting (drop zero activations only).

    Returns:
        Tuple of ``ContextFeatureSegment`` — one per extraction segment,
        ordered by token position. Every segment must carry at least one
        surviving SAE feature after thresholding; an empty or fully filtered
        segment raises because it indicates corrupt or mismatched extraction
        inputs.
    """
    all_feat_ids, all_feat_mags, _positions, char_starts, char_ends, _n_tokens = (
        extract_document_features(
            model=base_model,
            sae=base_sae,
            text=prefix_text,
            layer=sae_layer,
            context_window=extraction_max_tokens,
            pooling=PoolingMode.max_pool,
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=segment_size,
        )
    )

    threshold = float(extraction_threshold)
    segments: list[ContextFeatureSegment] = []
    for seg_fids, seg_mags, char_start, char_end in zip(
        all_feat_ids, all_feat_mags, char_starts, char_ends, strict=True,
    ):
        if seg_fids.ndim != 1 or seg_mags.ndim != 1:
            raise AxBenchBpbError(
                "extract_document_features returned non-1D feature ids/magnitudes "
                f"for char span [{int(char_start)}, {int(char_end)}): "
                f"feature_ids shape={tuple(seg_fids.shape)}, "
                f"feature_mags shape={tuple(seg_mags.shape)}"
            )
        if seg_fids.numel() != seg_mags.numel():
            raise AxBenchBpbError(
                "extract_document_features returned mismatched feature id/magnitude "
                f"counts for char span [{int(char_start)}, {int(char_end)}): "
                f"{seg_fids.numel()} ids vs {seg_mags.numel()} magnitudes"
            )
        feature_ids, buckets = threshold_and_discretize_features(
            seg_fids,
            seg_mags,
            boundaries=boundaries,
            extraction_threshold=threshold,
        )

        segments.append(
            ContextFeatureSegment(
                feature_ids=feature_ids,
                buckets=buckets,
                char_start=int(char_start),
                char_end=int(char_end),
            )
        )

    return tuple(segments)


def build_context_cache(
    *,
    cfg: Any,  # AxBenchBpbConfig; Any to avoid the top-level import cycle
    output_paths: OutputPaths,
    source_positive: Path,
    source_negative: Path,
    base_model: Any,
    base_sae: Any,
    boundaries: torch.Tensor,
    segment_size: int,
    sae_layer: int,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    concept_ids: Collection[int] | None = None,
) -> Path | None:
    """Stage entry point: build the context cache from both synth parquets.

    Idempotent: skips any prefix whose .safetensors already exists (unless
    ``cfg.context_cache.force_rebuild``).

    Single-process (``ddp_world_size == 1``): extracts every prefix on
    rank 0 and writes ``context_cache/manifest.json`` at the end mapping
    ``prefix_hash`` to ``{n_segments, n_features_total}``. Returns the
    manifest path.

    Multi-rank (``ddp_world_size > 1``): each rank handles every
    ``ddp_world_size``-th prefix (round-robin by sorted ``prefix_hash``,
    so each rank's slice is deterministic and disjoint). The manifest is
    NOT written by this function under DDP — the caller is responsible
    for inserting a barrier between this call and a separate
    :func:`write_context_cache_manifest` call on rank 0 so all per-rank
    safetensors files are durable on disk before rank 0 reads them back.
    Returns ``None`` under DDP for every rank.
    """
    unique = collect_unique_prefixes(
        source_positive, source_negative,
        filter_unaccepted=cfg.filter_unaccepted_samples,
        concept_ids=concept_ids,
    )
    record_counter("axbench_bpb.context_cache.unique_prefixes", len(unique))
    cache_dir = output_paths.context_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic order: sort by prefix_hash so each rank's slice is
    # independent of dict-iteration order. Round-robin sharding spreads
    # the workload uniformly when prefix-extraction cost is roughly
    # constant (true here — every prefix runs one Gemma+SAE forward pass).
    items = sorted(unique.items(), key=lambda kv: kv[1])
    my_items = items[ddp_rank::ddp_world_size]
    record_counter(
        "axbench_bpb.context_cache.rank_assigned_prefixes",
        len(my_items),
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    )

    if my_items:
        log_every = max(1, len(my_items) // 20)
        n_skipped = 0
        n_rebuilt = 0
        n_segments = 0
        n_features = 0
        for i, (prefix_text, prefix_hash) in enumerate(my_items):
            out_path = output_paths.context_cache_file(prefix_hash)
            if out_path.exists() and not cfg.context_cache.force_rebuild:
                n_skipped += 1
            else:
                with profile_region(
                    "axbench_bpb.context_cache.extract_prefix",
                    ddp_rank=ddp_rank,
                ):
                    segs = extract_prefix_features(
                        prefix_text=prefix_text,
                        base_model=base_model,
                        base_sae=base_sae,
                        boundaries=boundaries,
                        segment_size=segment_size,
                        sae_layer=sae_layer,
                        extraction_max_tokens=cfg.extraction_max_tokens,
                        extraction_threshold=cfg.extraction_threshold,
                    )
                with profile_region(
                    "axbench_bpb.context_cache.write_prefix",
                    ddp_rank=ddp_rank,
                    n_segments=len(segs),
                ):
                    serialize_context_segments(segs, out_path)
                n_rebuilt += 1
                n_segments += len(segs)
                n_features += sum(len(seg.feature_ids) for seg in segs)
            if (i + 1) % log_every == 0 or i == len(my_items) - 1:
                logger.info(
                    f"[axbench-bpb] context_cache rank {ddp_rank}/"
                    f"{ddp_world_size}: {i + 1}/{len(my_items)} prefixes"
                )
        record_counter(
            "axbench_bpb.context_cache.skipped_existing",
            n_skipped,
            ddp_rank=ddp_rank,
        )
        record_counter(
            "axbench_bpb.context_cache.rebuilt",
            n_rebuilt,
            ddp_rank=ddp_rank,
        )
        record_counter(
            "axbench_bpb.context_cache.rebuilt_segments",
            n_segments,
            ddp_rank=ddp_rank,
        )
        record_counter(
            "axbench_bpb.context_cache.rebuilt_features",
            n_features,
            ddp_rank=ddp_rank,
        )

    # Under DDP, the caller must barrier + invoke write_context_cache_manifest
    # on rank 0 explicitly. Returning None here forces that pattern.
    if ddp_world_size > 1:
        return None

    return write_context_cache_manifest(
        unique_prefixes=unique, output_paths=output_paths,
    )


def write_context_cache_manifest(
    *,
    unique_prefixes: dict[str, str],
    output_paths: OutputPaths,
) -> Path:
    """Read back every per-prefix safetensors file and write the manifest JSON.

    Must only be called on rank 0 under DDP, AFTER a barrier ensures all
    ranks have finished writing their slices via :func:`build_context_cache`.
    Reads each ``context_cache/<prefix_hash>.safetensors`` to populate
    ``{n_segments, n_features_total}``, then atomically writes the manifest
    via tmp + rename.

    Returns the manifest path.
    """
    manifest: dict[str, dict[str, int]] = {}
    with profile_region(
        "axbench_bpb.context_cache.manifest_readback",
        rows=len(unique_prefixes),
    ):
        for _prefix_text, prefix_hash in sorted(
            unique_prefixes.items(), key=lambda kv: kv[1],
        ):
            out_path = output_paths.context_cache_file(prefix_hash)
            segs = load_context_segments(out_path)
            manifest[prefix_hash] = {
                "n_segments": len(segs),
                "n_features_total": sum(len(s.feature_ids) for s in segs),
            }

    manifest_path = output_paths.context_cache_manifest()
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with profile_region(
        "axbench_bpb.context_cache.manifest_write",
        rows=len(manifest),
    ):
        manifest_tmp.write_text(json.dumps(manifest, sort_keys=True, indent=2))
        manifest_tmp.rename(manifest_path)
    record_counter("axbench_bpb.context_cache.manifest_entries", len(manifest))
    try:
        record_counter(
            "axbench_bpb.context_cache.manifest_bytes",
            manifest_path.stat().st_size,
        )
    except OSError:
        pass
    return manifest_path


def load_manifest(output_paths: OutputPaths) -> dict[str, dict[str, int]]:
    """Return the manifest dict. Raises AxBenchBpbError if missing."""
    p = output_paths.context_cache_manifest()
    if not p.exists():
        raise AxBenchBpbError(
            f"context cache manifest not found at {p}; run stage=prepare_context_cache"
        )
    return json.loads(p.read_text())
