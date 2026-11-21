"""Precompute magnitude-to-percentile values as overlay shards.

For each base feature shard, computes uint8 percentiles and writes them to a
separate overlay directory: ``pct/<checksum[:16]>/shard_NNNNN.safetensors.zst``.
Base shards are never modified.

The overlay directory name encodes the SHA-256 checksum of the percentile table,
and a ``manifest.json`` stores the full checksum for exact validation.

Usage::

    # Auto-construct path from SAE parameters:
    uv run python -m scripts.preprocess.precompute_percentiles

    # Override specific params:
    uv run python -m scripts.preprocess.precompute_percentiles layer=12 sae_width=32k

    # Explicit feature dir:
    uv run python -m scripts.preprocess.precompute_percentiles feature_dir=path/to/features

    # GPU acceleration + resume:
    uv run python -m scripts.preprocess.precompute_percentiles device=cuda resume=true

    # Multi-GPU (4 GPUs via torchrun):
    torchrun --standalone --nproc_per_node=4 -m scripts.preprocess.precompute_percentiles resume=true

Reads ``RANK``, ``WORLD_SIZE``, and ``LOCAL_RANK`` from the environment
(set automatically by torchrun, or manually). Each rank processes a
disjoint subset of shards via round-robin striping. Only rank 0 writes
the manifest. Resume works across rank changes because it checks for
existing overlay files regardless of which rank wrote them.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import hydra
import torch as t
from loguru import logger
from omegaconf import DictConfig

from scripts.preprocess._common import resolve_feature_dir
from scripts.preprocess._pct_profile import PercentileProfile, PercentileShardTiming
from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
)
from steering.feature_io import (
    TensorKey,
    decode_percentile_feature_safetensors_bytes,
    glob_features,
    load_feature_safetensors_bytes,
    parse_percentile_feature_safetensors_bytes,
    read_feature_value_count,
    shard_stem_from_path,
)
from steering.feature_paths import verify_segment_size
from steering.overlay_io import (
    OverlayIncompleteError,
    OverlayKind,
    discover_overlay,
    expected_overlay_shard_stems,
    overlay_dir,
    overlay_shard_path,
    save_overlay_shard,
    validate_percentile_overlay_shard,
    write_manifest,
)
from steering.percentile_precompute import (
    PercentileChunkTiming,
    compute_percentile_overlay_values,
    compute_percentile_overlay_values_from_encoded,
)
from steering.percentile_remote import (
    remote_percentiles_enabled,
    run_remote_percentiles,
)
from steering.stats_io import (
    find_statistics_path,
    load_statistics,
    save_percentile_checksum,
)


def _validate_percentile_payload(shard_path: Path, payload_path: Path) -> None:
    """Validate one payload against its base shard's feature count."""
    validate_percentile_overlay_shard(
        payload_path,
        expected_numel=read_feature_value_count(shard_path),
    )


def precompute_shard(
    shard_path: Path,
    percentile_table: t.Tensor,
    output_dir: Path,
    shard_stem: str,
    dry_run: bool = False,
    device: t.device | str = "cpu",
    zstd_level: int | None = None,
    timing: PercentileShardTiming | None = None,
    *,
    _table_is_finite: bool = False,
) -> bool:
    """Precompute percentiles for a single shard and write as overlay.

    Reads the base shard (immutable), computes uint8 percentiles, and writes
    them to ``output_dir/shard_stem.safetensors.zst``. The base shard is
    never modified.

    Args:
        shard_path: Path to the base ``.safetensors.zst`` shard.
        percentile_table: [n_features, 99] float64 percentile lookup table.
        output_dir: Overlay directory to write the percentile shard.
        shard_stem: Stem name for the output file (e.g., ``shard_00000``).
        dry_run: If True, compute but don't write.
        device: Compute device ("cpu" or "cuda").
        zstd_level: Compression level (1-22). Defaults to overlay_io.ZSTD_LEVEL.

    Returns:
        True if the overlay shard was written (or would be on dry_run).
    """
    t0 = time.monotonic()
    if t.device(device).type == "cuda":
        encoded = parse_percentile_feature_safetensors_bytes(
            load_feature_safetensors_bytes(shard_path)
        )
        parse_seconds = time.monotonic() - t0
        n_features = encoded.n_values
        chunk_timing = PercentileChunkTiming()
        pcts_uint8 = compute_percentile_overlay_values_from_encoded(
            encoded,
            percentile_table,
            device=device,
            source_name=shard_path.name,
            timing=chunk_timing,
        )
        t_load = parse_seconds + chunk_timing.decode_seconds
        t_compute = chunk_timing.lookup_seconds
    else:
        data = decode_percentile_feature_safetensors_bytes(
            load_feature_safetensors_bytes(shard_path)
        )
        t_load = time.monotonic() - t0
        n_features = len(data[TensorKey.FEATURE_IDS.value])
        compute_started = time.monotonic()
        pcts_uint8 = compute_percentile_overlay_values(
            data,
            percentile_table,
            device=device,
            source_name=shard_path.name,
            _table_is_finite=_table_is_finite,
        )
        t_compute = time.monotonic() - compute_started

    if timing is not None:
        timing.n_values = n_features
        timing.load_seconds = t_load
        timing.load_bytes = shard_path.stat().st_size
        timing.compute_seconds = t_compute

    if dry_run:
        logger.info(
            f"  [DRY-RUN] {shard_path.name}: {n_features:,} features, "
            f"load={t_load:.1f}s, compute={t_compute:.1f}s"
        )
        return True

    t2 = time.monotonic()
    out_path = overlay_shard_path(output_dir, shard_stem)
    save_overlay_shard(
        {TensorKey.FEATURE_PERCENTILES: pcts_uint8},
        out_path,
        zstd_level=zstd_level,
    )
    t_save = time.monotonic() - t2
    if timing is not None:
        timing.save_seconds = t_save
        timing.overlay_bytes = out_path.stat().st_size
    t_total = time.monotonic() - t0

    logger.info(
        f"  {shard_stem}: {n_features:,} features in {t_total:.0f}s "
        f"(load={t_load:.0f}s, compute={t_compute:.0f}s, save={t_save:.0f}s)"
    )
    return True


def run(cfg: DictConfig) -> int:
    """Run the pipeline from a complete Hydra config or equivalent test fixture."""
    feature_dir = resolve_feature_dir(cfg)
    if remote_percentiles_enabled(cfg):
        return run_remote_percentiles(cfg, feature_dir)
    if not feature_dir.is_dir():
        logger.error(f"Feature directory not found: {feature_dir}")
        return 1

    # Cross-stage segment_size check (Wave 3D task 4): Hydra cfg, directory
    # convention, and shard metadata must agree.
    verify_segment_size(feature_dir, expected_segment_size=int(cfg.segment_size))

    # Multi-GPU: read RANK/WORLD_SIZE from environment (set by torchrun or manually)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    # Device selection
    device_str = cfg.device
    if device_str == "auto":
        if t.cuda.is_available():
            device = t.device(f"cuda:{local_rank % t.cuda.device_count()}")
        else:
            device = t.device("cpu")
    else:
        device = t.device(device_str)

    rank_prefix = f"[rank {rank}/{world_size}] " if world_size > 1 else ""
    logger.info(f"{rank_prefix}Using device: {device}")

    stats_path = find_statistics_path(feature_dir)
    if stats_path is None:
        logger.error(f"No feature_statistics.safetensors found in {feature_dir}")
        return 1

    logger.info(f"Loading statistics from {stats_path}")
    stats = load_statistics(stats_path)
    n_sae = stats.metadata.n_sae_features
    logger.info(f"Building percentile table ({n_sae} features)")
    pct_table = build_percentile_table(stats.per_feature_stats, n_sae)
    checksum = compute_percentile_checksum(pct_table)
    table_is_finite = device.type == "cpu"
    logger.info(f"Percentile table checksum: {checksum[:16]}...")

    dry_run = bool(cfg.dry_run)
    resume = bool(cfg.resume)
    force = bool(cfg.force)
    zstd_level = cfg.zstd_level
    profile = bool(cfg.get("profile", False))
    profile_summary_output = cfg.get("profile_summary_output", None)

    if force and resume:
        logger.warning("Both force=true and resume=true set; force takes precedence")
        resume = False

    profiler = (
        PercentileProfile(
            config={
                "feature_dir": str(feature_dir),
                "device": str(device),
                "resume": resume,
                "force": force,
                "zstd_level": zstd_level,
                "world_size": world_size,
                "rank": rank,
            }
        )
        if profile
        else None
    )

    # Resolve overlay directory
    output = overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum)
    logger.info(f"Overlay directory: {output}")

    all_shard_paths = glob_features(feature_dir)
    logger.info(f"{rank_prefix}Found {len(all_shard_paths)} feature shards in {feature_dir}")
    expected_stems = expected_overlay_shard_stems(
        all_shard_paths, OverlayKind.PERCENTILE
    )

    # Authenticate an existing v2 manifest against the current base bytes before
    # any output, statistics, or cache write. Legacy v1 remains accepted by the
    # ordinary discovery contract (without base authentication).
    existing_overlay = None
    if not force:
        existing_overlay = discover_overlay(
            feature_dir,
            OverlayKind.PERCENTILE,
            checksum,
            base_shard_paths=all_shard_paths,
            cache_dir=None,
        )
    if existing_overlay is not None and not force:
        logger.info(f"Overlay already exists at {existing_overlay}")
        missing_stems = [
            stem
            for stem in expected_stems
            if not overlay_shard_path(existing_overlay, stem).is_file()
        ]
        if not missing_stems and not resume:
            if not dry_run:
                save_percentile_checksum(stats_path, checksum)
                logger.info(f"Saved percentile checksum to {stats_path}")
            return 0
        if not resume:
            raise OverlayIncompleteError(
                f"Percentile overlay at {existing_overlay} is incomplete: "
                f"missing {len(missing_stems)} expected shard(s). "
                "Use resume=true to fill them."
            )

    # Write manifest (rank 0 only to avoid races). Pass base_shard_paths so the
    # written manifest is schema_version=2 with a base_shard_content_hash —
    # discover_overlay() will then reject the overlay if the base shards are
    # silently re-extracted underneath. See consolidation manifest §4.5.
    if not dry_run and rank == 0 and (existing_overlay is None or force):
        write_manifest(
            output, checksum, OverlayKind.PERCENTILE,
            metadata={
                "n_sae_features": n_sae,
                "stats_file": stats_path.name,
            },
            base_shard_paths=all_shard_paths,
            cache_dir=feature_dir,
        )
        logger.info(f"{rank_prefix}Wrote manifest to {output}")

    # Stripe shards across ranks (round-robin)
    shard_paths = all_shard_paths[rank::world_size]
    if world_size > 1:
        logger.info(
            f"{rank_prefix}{len(shard_paths)} shards assigned "
            f"(of {len(all_shard_paths)} total)"
        )

    if not shard_paths:
        logger.warning(f"{rank_prefix}No shards assigned, nothing to do.")
        return 0

    n_processed = 0
    n_skipped = 0
    n_total = len(shard_paths)
    t_start = time.monotonic()

    try:
        for i, shard_path in enumerate(shard_paths):
            stem = shard_stem_from_path(shard_path)

            # Resume: skip if overlay shard already exists
            if resume and not force:
                overlay_path = overlay_shard_path(output, stem)
                if overlay_path.is_file():
                    _validate_percentile_payload(shard_path, overlay_path)
                    n_skipped += 1
                    if n_skipped <= 5 or n_skipped % 50 == 0:
                        logger.info(f"  [{i+1}/{n_total}] Skipped {stem} (overlay exists)")
                    continue

            timing = (
                PercentileShardTiming(stem=stem, n_values=0)
                if profiler is not None
                else None
            )
            precompute_shard(
                shard_path, pct_table, output, stem,
                dry_run=dry_run, device=device, zstd_level=zstd_level,
                timing=timing,
                _table_is_finite=table_is_finite,
            )
            if profiler is not None and timing is not None:
                profiler.add_shard(timing)
            n_processed += 1

            elapsed = time.monotonic() - t_start
            remaining = n_total - i - 1
            if n_processed > 0:
                avg_per_shard = elapsed / (n_processed + n_skipped)
                eta_s = remaining * avg_per_shard
                eta_str = f"{eta_s/60:.0f}m" if eta_s > 60 else f"{eta_s:.0f}s"
            else:
                eta_str = "?"
            logger.info(
                f"  [{i+1}/{n_total}] {stem} done "
                f"(processed={n_processed}, skipped={n_skipped}, "
                f"elapsed={elapsed/60:.1f}m, ETA={eta_str})"
            )
    finally:
        if profiler is not None:
            profiler.totals = {
                "wall_seconds": time.monotonic() - t_start,
                "n_processed": n_processed,
                "n_skipped": n_skipped,
            }
            sidecar_path = (
                Path("out/traces/pct_profile.json")
                if profile_summary_output is None
                else Path(profile_summary_output)
            )
            if world_size > 1:
                # Every rank writes its own stripe; an unsuffixed shared path
                # would be last-writer-wins under torchrun (review finding,
                # issue #1269).
                sidecar_path = sidecar_path.with_name(
                    f"{sidecar_path.stem}.rank{rank}{sidecar_path.suffix}"
                )
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            profiler.write(sidecar_path)
            logger.info(f"Profile sidecar → {sidecar_path}")

    elapsed_total = time.monotonic() - t_start
    logger.info(
        f"{rank_prefix}Done: {n_processed} processed, {n_skipped} skipped "
        f"in {elapsed_total/60:.1f}m (overlay={output}, checksum={checksum[:16]}...)"
    )

    if not dry_run:
        for path in shard_paths:
            stem = shard_stem_from_path(path)
            payload_path = overlay_shard_path(output, stem)
            if not payload_path.is_file():
                raise OverlayIncompleteError(
                    f"Percentile overlay at {output} is incomplete for "
                    f"rank {rank}: missing assigned shard {stem}."
                )
            _validate_percentile_payload(path, payload_path)

    if not dry_run:
        save_percentile_checksum(stats_path, checksum)
        logger.info(f"Saved percentile checksum to {stats_path}")

    return 0


@hydra.main(
    config_path="../../configs/extraction",
    config_name="precompute_percentiles",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    sys.exit(run(cfg))


if __name__ == "__main__":
    main()
