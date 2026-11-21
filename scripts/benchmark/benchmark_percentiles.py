"""Benchmark one percentile overlay shard without modifying the feature directory.

Usage::

    uv run python -m scripts.benchmark.benchmark_percentiles \
        feature_dir=... out=out/bench/pct_bench.json \
        shard=shard_00000 scratch_dir=/tmp/pct-bench device=cpu

The benchmark records load, direct-gather compute, compressed save, wall-clock, and
peak resident-set size. ``compare_to`` hashes a reference overlay after the result is
written and exits nonzero on a content mismatch while retaining the JSON evidence.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import psutil
import torch as t
from omegaconf import DictConfig, OmegaConf

from scripts.preprocess._pct_profile import PercentileShardTiming
from scripts.preprocess.precompute_percentiles import precompute_shard
from steering.conditional.percentile_table import build_percentile_table
from steering.feature_io import glob_features, shard_stem_from_path
from steering.overlay_io import overlay_content_digest, overlay_shard_path
from steering.stats_io import find_statistics_path, load_statistics


def _sample_peak_rss(fn, interval_s: float = 0.2):
    """Run ``fn`` while a daemon sampler records this process's peak RSS."""
    process = psutil.Process()
    peak = [0]
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            try:
                peak[0] = max(peak[0], process.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            stop.wait(interval_s)

    sampler = threading.Thread(target=sample, name="pct-rss-sampler", daemon=True)
    sampler.start()
    started = time.perf_counter()
    try:
        result = fn()
    finally:
        stop.set()
        sampler.join()
        try:
            peak[0] = max(peak[0], process.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return result, time.perf_counter() - started, peak[0]


def run_bench(cfg: DictConfig) -> dict:
    """Benchmark one shard and write the JSON result."""
    feature_dir = Path(str(cfg.feature_dir))
    out_path = Path(str(cfg.out))
    shard_name = cfg.get("shard", None)
    shard_paths = glob_features(feature_dir)
    if shard_name is not None:
        shard_paths = [
            path
            for path in shard_paths
            if shard_stem_from_path(path) == str(shard_name)
        ]
    if len(shard_paths) != 1:
        raise ValueError(
            f"benchmark_percentiles expected exactly one shard, found {len(shard_paths)}"
        )
    shard_path = shard_paths[0]
    stem = shard_stem_from_path(shard_path)

    stats_path = find_statistics_path(feature_dir)
    if stats_path is None:
        raise FileNotFoundError(f"No feature statistics found in {feature_dir}")
    stats = load_statistics(stats_path)
    percentile_table = build_percentile_table(
        stats.per_feature_stats, stats.metadata.n_sae_features
    )
    scratch_dir = Path(
        str(cfg.get("scratch_dir", out_path.parent / "pct_benchmark_scratch"))
    )
    scratch_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.get("device", "cpu")
    zstd_level = cfg.get("zstd_level", 12)
    timing = PercentileShardTiming(stem=stem, n_values=0)

    def compute() -> bool:
        return precompute_shard(
            shard_path,
            percentile_table,
            scratch_dir,
            stem,
            dry_run=False,
            device=device,
            zstd_level=zstd_level,
            timing=timing,
        )

    _, wall_seconds, peak_rss_bytes = _sample_peak_rss(compute)
    output_overlay = overlay_shard_path(scratch_dir, stem)
    overlay_digest = overlay_content_digest(output_overlay)

    compare_to = cfg.get("compare_to", None)
    compare_to_digest = (
        overlay_content_digest(Path(str(compare_to))) if compare_to is not None else None
    )
    digests_match = (
        overlay_digest == compare_to_digest if compare_to_digest is not None else None
    )
    result = {
        "shard": stem,
        "n_values": timing.n_values,
        "load_seconds": timing.load_seconds,
        "compute_seconds": timing.compute_seconds,
        "save_seconds": timing.save_seconds,
        "wall_seconds": wall_seconds,
        "peak_rss_bytes": peak_rss_bytes,
        "overlay_digest": overlay_digest,
        "compare_to_digest": compare_to_digest,
        "digests_match": digests_match,
        "torch_num_threads": t.get_num_threads(),
        "device": str(device),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if compare_to_digest is not None and not digests_match:
        raise SystemExit(1)
    return result


def main() -> None:
    cfg = OmegaConf.from_cli()
    missing = [key for key in ("feature_dir", "out") if key not in cfg]
    if missing:
        raise SystemExit(
            "benchmark_percentiles: required arg(s) missing: "
            f"{' '.join(f'{key}=...' for key in missing)}"
        )
    print(json.dumps(run_bench(cfg), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
