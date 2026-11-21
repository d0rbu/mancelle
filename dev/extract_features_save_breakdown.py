"""Decompose `save_features` time into sort / delta / serialize / compress / write phases.

Hypothesis: the save thread does substantial CPU work (Python sort loops,
delta-encode, zstd-12 compress) BEFORE the disk write begins. Quantifying each phase
tells us whether the right fix is more concurrency, faster compression, or vectorized
sort/delta.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import torch as t
import zstandard
from dev.extract_features_intershard_repro import make_realistic_shard  # type: ignore
from safetensors.torch import save as safetensors_save

from nanochat.atomic_io import atomic_write_bytes
from steering.feature_io import (
    COMPRESSED_EXT,
    ZSTD_LEVEL,
    TensorKey,
    _delta_encode,
    _optimize_dtype,
    _sort_within_segments,
)


def profile_save(data: dict, out_stem: Path, zstd_level: int = ZSTD_LEVEL) -> dict:
    timings: dict[str, float] = {}

    offsets = data["offsets"]
    fids = data["feature_ids"]
    mags = data["feature_mags"]

    t0 = time.perf_counter()
    sort_result = _sort_within_segments(fids, mags, offsets)
    sorted_fids, sorted_mags = sort_result[0], sort_result[1]
    t1 = time.perf_counter()
    timings["sort_within_segments"] = t1 - t0

    t0 = time.perf_counter()
    delta_fids = _delta_encode(sorted_fids, offsets)
    t1 = time.perf_counter()
    timings["delta_encode"] = t1 - t0

    modified = dict(data)
    modified["feature_ids"] = delta_fids
    modified["feature_mags"] = sorted_mags

    t0 = time.perf_counter()
    tensors: dict[str, t.Tensor] = {}
    metadata: dict[str, str] = {}
    _TENSOR_KEYS = {k.value for k in TensorKey}
    for key, value in modified.items():
        if key in _TENSOR_KEYS and isinstance(value, t.Tensor):
            tensors[key] = _optimize_dtype(key, value)
        elif isinstance(value, bool):
            metadata[key] = "true" if value else "false"
        else:
            metadata[key] = str(value)
    t1 = time.perf_counter()
    timings["optimize_dtype"] = t1 - t0

    t0 = time.perf_counter()
    sf_bytes = safetensors_save(tensors, metadata=metadata)
    t1 = time.perf_counter()
    timings["safetensors_serialize"] = t1 - t0
    timings["serialized_mb"] = len(sf_bytes) / 1024 / 1024

    t0 = time.perf_counter()
    cctx = zstandard.ZstdCompressor(level=zstd_level)
    compressed = cctx.compress(sf_bytes)
    t1 = time.perf_counter()
    timings[f"zstd_compress_l{zstd_level}"] = t1 - t0
    timings["compressed_mb"] = len(compressed) / 1024 / 1024

    out_path = Path(str(out_stem) + COMPRESSED_EXT)
    t0 = time.perf_counter()
    atomic_write_bytes(compressed, out_path)
    t1 = time.perf_counter()
    timings["atomic_write_bytes"] = t1 - t0
    timings["total"] = sum(v for k, v in timings.items() if k not in ("serialized_mb", "compressed_mb"))
    return timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", type=int, default=30_000)
    ap.add_argument("--feats-per-seg", type=int, default=800)
    ap.add_argument("--n-features", type=int, default=16_384)
    args = ap.parse_args()

    print(f"Generating shard ({args.segments} segments × {args.feats_per_seg} feats)...")
    data = make_realistic_shard(
        n_segments=args.segments,
        avg_feats_per_seg=args.feats_per_seg,
        n_features_vocab=args.n_features,
        seed=0,
    )

    for level in (3, 6, 9, 12, 19):
        with tempfile.TemporaryDirectory(prefix="save_breakdown_") as tmp:
            out = Path(tmp) / "shard_00000_features"
            timings = profile_save(data, out, zstd_level=level)
        print(f"\n=== zstd_level={level} ===")
        total = timings["total"]
        for k in sorted(timings):
            if k.endswith("_mb"):
                print(f"  {k:30s}: {timings[k]:8.2f} MB")
            else:
                pct = 100 * timings[k] / total if total > 0 else 0
                print(f"  {k:30s}: {timings[k]:6.3f}s ({pct:4.1f}%)")


if __name__ == "__main__":
    main()
