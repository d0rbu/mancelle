"""Benchmark the real conditioned-SFT data generator on synthetic shards.

Example production-like CPU geometry::

    uv run python scripts/benchmark/benchmark_sft_data_pipeline.py \
        --shards 6 --docs-per-shard 100 --feats-per-seg 1500 --batches 120
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

# ``python scripts/benchmark/...`` otherwise exposes only this directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch as t

from scripts.train import conditional_encdec_sft as sft
from steering.conditional.encoder_decoder.sft_features import load_sft_features
from steering.conditional.sampling import (
    Distribution,
    NegativeBundle,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
)
from steering.feature_io import save_features
from steering.profiling.markers import TimingRecorder, timing_tags, use_recorder
from steering.sft_schema import SFTSplit
from tasks.common import Task
from tests.steering.test_encdec_sft_conditioned_integration import SFTMockTokenizer
from tests.steering.test_sft_features import _write_parquet

_N_SAE_FEATURES = 65_536
_SEQUENCE_LENGTH = 128
_BUFFER_SIZE = 100


@dataclass(frozen=True)
class BenchmarkGeometry:
    shards: int
    docs_per_shard: int
    feats_per_seg: int
    segments_per_doc: int
    batches: int
    device_batch_size: int
    warmup_batches: int
    device: str
    # Production runs feature prediction on, and when a sampling bundle exists
    # the FP table IS the sampling table object (conditional_encdec_sft.py:4236),
    # which turns on both the per-segment FP percentile lookup and the
    # sampling-weight reuse branch. Off by default so the pre-existing baseline
    # geometries keep their recorded numbers.
    feature_prediction: bool


class _BenchmarkTask(Task):
    def __init__(self, n: int) -> None:
        super().__init__()
        self._n = n

    def num_examples(self) -> int:
        return self._n

    def get_example(self, index: int) -> dict:
        return {
            "messages": [
                {"role": "user", "content": f"benchmark question {index:08d}"},
                {
                    "role": "assistant",
                    "content": f"benchmark answer {index:08d} " + "x" * 32,
                },
            ]
        }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}")
    return parsed


def _percentile(values: list[float], q: float) -> float:
    """Return a nearest-rank percentile; at tiny sample counts p99 equals max."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def _write_shard(
    *,
    feature_dir: Path,
    parquet_dir: Path,
    stem: str,
    original_indices: list[int],
    geometry: BenchmarkGeometry,
    total_chars: int,
) -> None:
    n_docs = len(original_indices)
    n_segments = n_docs * geometry.segments_per_doc
    feature_offsets = t.arange(geometry.feats_per_seg, dtype=t.int64)
    max_base = _N_SAE_FEATURES - geometry.feats_per_seg + 1
    bases = (t.arange(n_segments, dtype=t.int64) * 17 % max_base).unsqueeze(1)
    feature_ids = (bases + feature_offsets).to(t.int32).reshape(-1)
    magnitudes = (
        t.linspace(0.5, 12.5, geometry.feats_per_seg, dtype=t.float32)
        .repeat(n_segments)
        .contiguous()
    )
    doc_ids = t.arange(n_docs, dtype=t.int32).repeat_interleave(geometry.segments_per_doc)
    segment_indices = t.arange(geometry.segments_per_doc, dtype=t.int32).repeat(n_docs)
    segment_width = max(1, total_chars // geometry.segments_per_doc)
    char_starts = segment_indices * segment_width
    char_ends = (segment_indices + 1) * segment_width
    char_ends[segment_indices == geometry.segments_per_doc - 1] = total_chars
    shard = {
        "offsets": t.arange(
            0,
            (n_segments + 1) * geometry.feats_per_seg,
            geometry.feats_per_seg,
            dtype=t.int64,
        ),
        "feature_ids": feature_ids,
        "feature_mags": magnitudes,
        "doc_ids": doc_ids,
        "segment_indices": segment_indices,
        "segment_positions": segment_indices * 128,
        "segment_char_starts": char_starts,
        "segment_char_ends": char_ends,
        "doc_token_counts": t.full((n_docs,), 256, dtype=t.int32),
        "n_docs": n_docs,
        "n_segments": n_segments,
        "layer": 22,
        "context_window": 1024,
        "pooling": "mean_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 128,
    }
    save_features(shard, feature_dir / f"{stem}_features")
    _write_parquet(parquet_dir, stem, original_indices)


def _build_artifacts(
    root: Path,
    geometry: BenchmarkGeometry,
) -> tuple[_BenchmarkTask, SFTMockTokenizer, Path, Path]:
    feature_dir = root / "features"
    parquet_dir = root / "parquet"
    feature_dir.mkdir()
    parquet_dir.mkdir()
    task = _BenchmarkTask(geometry.shards * geometry.docs_per_shard)
    tokenizer = SFTMockTokenizer()
    ids, _mask = tokenizer.render_conversation(task[0])
    total_chars = len(tokenizer.decode(ids))
    for shard_index in range(geometry.shards):
        first = shard_index * geometry.docs_per_shard
        original_indices = list(range(first, first + geometry.docs_per_shard))
        _write_shard(
            feature_dir=feature_dir,
            parquet_dir=parquet_dir,
            stem=f"shard_{shard_index:05d}",
            original_indices=original_indices,
            geometry=geometry,
            total_chars=total_chars,
        )
    (parquet_dir / "conversion_meta.json").write_text(
        json.dumps(
            {
                "mixture": "capabilities",
                "split": "train",
                "format": "full_nanochat_text",
                "completed_shards": geometry.shards,
                "total_shards": geometry.shards,
                "total_conversations": len(task),
            }
        ),
        encoding="utf-8",
    )
    return task, tokenizer, feature_dir, parquet_dir


def _timing_aggregates(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    elapsed_by_name: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        elapsed_by_name[str(row["name"])].append(float(row["elapsed_ns"]) / 1e6)
    output: dict[str, dict[str, float | int]] = {}
    for name, values in sorted(elapsed_by_name.items()):
        output[name] = {
            "count": len(values),
            "total_ms": sum(values),
            "mean_ms": sum(values) / len(values),
            "p50_ms": _percentile(values, 0.50),
            "p99_ms": _percentile(values, 0.99),
            "max_ms": max(values),
        }
    return output


def _selection_summary(timing: dict[str, dict[str, float | int]]) -> dict[str, float | int]:
    fsd = timing.get("sft/feature_select_discretize", {})
    conversations = int(fsd.get("count", 0))
    total_ms = float(fsd.get("total_ms", 0.0))
    return {
        "conversations": conversations,
        "total_ms": total_ms,
        "ms_per_conversation": total_ms / conversations if conversations else 0.0,
    }


def _refill_summary(timing: dict[str, dict[str, float | int]]) -> dict[str, float | int]:
    refill = timing.get("sft/refill_buffer", {})
    return {
        "calls": int(refill.get("count", 0)),
        "total_ms": float(refill.get("total_ms", 0.0)),
        "p99_ms": float(refill.get("p99_ms", 0.0)),
        "max_ms": float(refill.get("max_ms", 0.0)),
    }


def _load_profile_aggregates(records: list[dict]) -> dict[str, Any]:
    by_phase: dict[str, dict[str, Any]] = {}
    for record in records:
        phase = str(record["phase"])
        aggregate = by_phase.setdefault(
            phase,
            {
                "count": 0,
                "total_seconds": 0.0,
                "compressed_bytes": 0,
                "decompressed_bytes": 0,
                "detail_seconds": defaultdict(float),
                "detail_cpu_seconds": defaultdict(float),
            },
        )
        aggregate["count"] += 1
        aggregate["total_seconds"] += float(record["total_seconds"])
        aggregate["compressed_bytes"] += int(record.get("compressed_bytes", 0))
        aggregate["decompressed_bytes"] += int(record.get("decompressed_bytes", 0))
        for key, value in record.get("detail_seconds", {}).items():
            aggregate["detail_seconds"][str(key)] += float(value)
        for key, value in record.get("detail_cpu_seconds", {}).items():
            aggregate["detail_cpu_seconds"][str(key)] += float(value)
    for aggregate in by_phase.values():
        aggregate["detail_seconds"] = dict(sorted(aggregate["detail_seconds"].items()))
        aggregate["detail_cpu_seconds"] = dict(sorted(aggregate["detail_cpu_seconds"].items()))
    return by_phase


def _current_rss_bytes() -> int:
    """Current resident set size, not the process-lifetime high-water mark."""
    with open("/proc/self/statm", encoding="ascii") as statm:
        resident_pages = int(statm.read().split()[1])
    return resident_pages * os.sysconf("SC_PAGESIZE")


def run_benchmark(geometry: BenchmarkGeometry) -> dict[str, Any]:
    if geometry.feats_per_seg > _N_SAE_FEATURES:
        raise ValueError(
            f"feats_per_seg={geometry.feats_per_seg} exceeds the synthetic SAE "
            f"width {_N_SAE_FEATURES}"
        )
    if geometry.warmup_batches >= geometry.batches:
        raise ValueError(
            f"warmup_batches={geometry.warmup_batches} must be strictly less than "
            f"batches={geometry.batches}; steady state needs at least one measured batch"
        )
    try:
        t.device(geometry.device)
    except RuntimeError as exc:
        raise ValueError(f"invalid benchmark device {geometry.device!r}") from exc
    use_cuda = geometry.device.startswith("cuda")
    if use_cuda and not t.cuda.is_available():
        raise RuntimeError(
            f"benchmark device {geometry.device!r} requested but CUDA is unavailable"
        )
    t.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="benchmark-sft-data-pipeline-") as tmp:
        task, tokenizer, feature_dir, parquet_dir = _build_artifacts(Path(tmp), geometry)
        store = load_sft_features(str(feature_dir), str(parquet_dir))
        feature_index = store._feature_index
        feature_index._profile_loads = True

        percentile_table = (
            t.linspace(0.5, 99.5, 99, dtype=t.float64)
            .unsqueeze(0)
            .expand(_N_SAE_FEATURES, -1)
            .contiguous()
        )
        rng = t.Generator().manual_seed(1504)
        sampling_bundle = SamplingBundle(
            selection=SamplingSelection(
                distribution=Distribution.geometric,
                mean=12.0,
                std=12.0,
                min_k=1,
                max_k=64,
                mode=SamplingMode.weighted,
            ),
            percentile_table=percentile_table,
            rng=rng,
        )
        negative_bundle = NegativeBundle(
            selection=NegativeSamplingSelection(
                distribution=Distribution.geometric,
                mean=3.0,
                std=3.0,
                min_k=1,
                max_k=8,
            ),
            n_sae_features=_N_SAE_FEATURES,
            rng=rng,
        )
        boundaries = (
            t.tensor([25.0, 50.0, 75.0], dtype=t.float64)
            .unsqueeze(0)
            .expand(_N_SAE_FEATURES, -1)
            .contiguous()
        )
        state = sft.SFTState()
        generator = sft.make_sft_data_generator(
            task=task,
            tokenizer=tokenizer,
            B=geometry.device_batch_size,
            T=_SEQUENCE_LENGTH,
            device=geometry.device,
            sft_state=state,
            buffer_size=_BUFFER_SIZE,
            feature_dir=str(feature_dir),
            parquet_dir=str(parquet_dir),
            sampling_bundle=sampling_bundle,
            negative_bundle=negative_bundle,
            boundaries=boundaries,
            fp_percentile_table=(percentile_table if geometry.feature_prediction else None),
            n_sae_features=(_N_SAE_FEATURES if geometry.feature_prediction else None),
            min_loss_tokens=8,
            alignment_probe_samples=0,
            profile_markers=True,
            expected_split=SFTSplit.TRAIN,
        )
        fixture_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        rss_current_before = _current_rss_bytes()
        overall_cache_start = feature_index.cache_stats.snapshot()
        recorder = TimingRecorder()
        started = time.perf_counter()
        # Nested CUDA timing regions synchronize the CURRENT device, so a
        # non-default --device (e.g. cuda:1) must be current for the whole
        # timed loop or async transfers would be under-timed.
        device_context = t.cuda.device(geometry.device) if use_cuda else nullcontext()
        with (
            device_context,
            patch.object(sft, "load_sft_features", return_value=store),
            patch.object(
                sft,
                "_assert_sft_mixture_fingerprint_matches_task",
                return_value=None,
            ),
            use_recorder(recorder),
            timing_tags(source="benchmark_sft_data_pipeline"),
        ):
            for _ in range(geometry.warmup_batches):
                next(generator)
            if use_cuda:
                t.cuda.synchronize(geometry.device)
            steady_started = time.perf_counter()
            steady_row_start = len(recorder.rows())
            steady_profile_start = len(feature_index.load_profiles)
            steady_cache_start = feature_index.cache_stats.snapshot()
            for _ in range(geometry.batches - geometry.warmup_batches):
                next(generator)
            if use_cuda:
                t.cuda.synchronize(geometry.device)
            steady_wall_seconds = time.perf_counter() - steady_started
        wall_seconds = time.perf_counter() - started

        rows = recorder.rows()
        timing = _timing_aggregates(rows)
        steady_timing = _timing_aggregates(rows[steady_row_start:])
        cache_stats = feature_index.cache_stats.snapshot()
        overall_cache_delta = feature_index.cache_stats.delta(overall_cache_start)
        steady_cache_delta = feature_index.cache_stats.delta(steady_cache_start)
        load_profiles = feature_index.load_profiles
        overall_load_profiles = _load_profile_aggregates(load_profiles)
        steady_load_profiles = _load_profile_aggregates(load_profiles[steady_profile_start:])
        feature_index.close()

        rss_current_after = _current_rss_bytes()
        max_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        steady_batches = geometry.batches - geometry.warmup_batches
        return {
            "geometry": {
                **geometry.__dict__,
                "n_sae_features": _N_SAE_FEATURES,
                "sequence_length": _SEQUENCE_LENGTH,
                "buffer_size": _BUFFER_SIZE,
            },
            "throughput": {
                "wall_seconds": wall_seconds,
                "batches_per_second": geometry.batches / wall_seconds,
                "batches": geometry.batches,
                "consumed_conversations": state.consumed,
            },
            "feature_select_discretize": _selection_summary(timing),
            "refill_buffer": _refill_summary(timing),
            "timing_regions": timing,
            "feature_index": {
                "cache_stats": cache_stats,
                "load_profiles": overall_load_profiles,
                "resident_loads_during_timed_batches": int(overall_cache_delta["loads"]),
                "resident_loads_during_steady_state_batches": int(steady_cache_delta["loads"]),
            },
            "peak_rss": {
                "ru_maxrss_kib": max_rss_kib,
                "bytes": max_rss_kib * 1024,
                "fixture_ru_maxrss_kib": fixture_rss_kib,
                # High-water-mark increment: 0 does NOT mean no growth — the
                # fixture-build phase may simply have peaked higher than the
                # run. Use the rss_current_* fields for growth attribution.
                "ru_maxrss_increment_kib": max_rss_kib - fixture_rss_kib,
                "rss_current_before_bytes": rss_current_before,
                "rss_current_after_bytes": rss_current_after,
                "rss_current_delta_bytes": rss_current_after - rss_current_before,
            },
            "steady_state": {
                "throughput": {
                    "wall_seconds": steady_wall_seconds,
                    "batches_per_second": steady_batches / steady_wall_seconds,
                    "batches": steady_batches,
                    "consumed_conversations": steady_batches * geometry.device_batch_size,
                },
                "feature_select_discretize": _selection_summary(steady_timing),
                "refill_buffer": _refill_summary(steady_timing),
                "timing_regions": steady_timing,
                "feature_index": {
                    "cache_stats_delta": steady_cache_delta,
                    "load_profiles": steady_load_profiles,
                    "resident_loads_during_steady_state_batches": int(steady_cache_delta["loads"]),
                },
            },
        }


def _parse_args() -> BenchmarkGeometry:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=_positive_int, default=3)
    parser.add_argument("--docs-per-shard", type=_positive_int, default=20)
    parser.add_argument("--feats-per-seg", type=_positive_int, default=64)
    parser.add_argument("--segments-per-doc", type=_positive_int, default=3)
    parser.add_argument("--batches", type=_positive_int, default=5)
    parser.add_argument("--device-batch-size", type=_positive_int, default=2)
    parser.add_argument("--warmup-batches", type=_non_negative_int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--feature-prediction", action="store_true")
    args = parser.parse_args()
    return BenchmarkGeometry(
        shards=args.shards,
        docs_per_shard=args.docs_per_shard,
        feats_per_seg=args.feats_per_seg,
        segments_per_doc=args.segments_per_doc,
        batches=args.batches,
        device_batch_size=args.device_batch_size,
        warmup_batches=args.warmup_batches,
        device=args.device,
        feature_prediction=args.feature_prediction,
    )


def main() -> None:
    print(json.dumps(run_benchmark(_parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
