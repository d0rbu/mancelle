"""Smoke tests for the issue #1269 percentile benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch as t
from omegaconf import OmegaConf

from scripts.benchmark.benchmark_percentiles import run_bench
from scripts.benchmark.verify_overlay_cache_byte_identity import _hash_overlay_tensors
from scripts.preprocess.precompute_percentiles import run as run_percentiles
from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
)
from steering.feature_io import save_features
from steering.overlay_io import (
    load_overlay_shard,
    overlay_content_digest,
    overlay_shard_path,
    save_overlay_shard,
)
from steering.segmentation import SegmentMode
from steering.stats_io import load_statistics, save_statistics
from steering.stats_types import (
    CorpusStats,
    ExtractionStatistics,
    Extrapolation,
    MetadataStats,
    PerFeatureStat,
    SegmentFeatureStats,
    StorageStats,
)


def _make_env(root: Path) -> Path:
    feature_dir = root / "features" / "s16"
    feature_dir.mkdir(parents=True)
    stats = ExtractionStatistics(
        metadata=MetadataStats(
            pooling_mode="mean_pool",
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=16,
            context_window=1024,
            layer=8,
            n_sae_features=32,
            generated_at="test",
            activation_percentile=0.0,
        ),
        corpus=CorpusStats(
            n_shards_processed=1,
            n_docs=2,
            n_tokens_gpt2=100,
            mean_tokens_per_doc=50.0,
            median_tokens_per_doc=50,
        ),
        features=SegmentFeatureStats(
            n_segments=2,
            n_features_seen=8,
            mean_active_per_segment=4.0,
            median_active_per_segment=4,
            p90_active_per_segment=5,
            max_active_per_segment=6,
            min_active_per_segment=1,
            n_total_sae_features=32,
            pct_features_active=0.5,
            count_percentiles={},
            magnitude_stats={},
            active_per_segment_histogram={},
            active_per_segment_percentiles={},
        ),
        storage=StorageStats(
            total_bytes=1000,
            mean_bytes_per_doc=500.0,
            mean_bytes_per_token=5.0,
        ),
        per_feature_stats={
            str(fid): PerFeatureStat(
                count=100,
                pct_segments_active=0.5,
                mean_mag=1.0,
                mean_mag_incl_zeros=0.5,
                std_mag=0.5,
                min_mag=0.1,
                max_mag=2.0,
                median_mag=1.0,
                percentiles={
                    f"p{k}": float(k) * 0.1 + fid * 0.01 for k in range(1, 100)
                },
            )
            for fid in range(8)
        },
        extrapolation_10b=Extrapolation(
            target_tokens=10_000_000_000,
            estimated_docs=1_000_000,
            estimated_storage_gb=100.0,
            estimated_wall_time_hours=10.0,
        ),
    )
    save_statistics(stats, feature_dir / "feature_statistics")
    save_features(
        {
            "offsets": t.tensor([0, 6, 12], dtype=t.int64),
            "feature_ids": t.tensor([fid % 8 for fid in range(12)], dtype=t.int32),
            "feature_mags": t.tensor(
                [0.2 + 0.01 * fid for fid in range(12)], dtype=t.float32
            ),
            "doc_ids": t.tensor([0, 1], dtype=t.int32),
            "segment_indices": t.tensor([0, 0], dtype=t.int32),
            "segment_positions": t.tensor([0, 0], dtype=t.int32),
            "doc_token_counts": t.tensor([16, 16], dtype=t.int32),
            "segment_char_starts": t.tensor([0, 0], dtype=t.int32),
            "segment_char_ends": t.tensor([16, 16], dtype=t.int32),
            "n_docs": 2,
            "n_segments": 2,
            "layer": 8,
            "context_window": 1024,
            "pooling": "mean_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 16,
        },
        feature_dir / "shard_00000_features",
    )
    return feature_dir


def _cfg(feature_dir: Path, **overrides):
    values = {
        "feature_dir": str(feature_dir),
        "segment_size": 16,
        "device": "cpu",
        "dry_run": False,
        "resume": False,
        "force": False,
        "zstd_level": 12,
    }
    values.update(overrides)
    return OmegaConf.create(values)


def _pct_overlay_path(feature_dir: Path, stem: str) -> Path:
    stats = load_statistics(feature_dir / "feature_statistics.safetensors")
    table = build_percentile_table(stats.per_feature_stats, stats.metadata.n_sae_features)
    checksum = compute_percentile_checksum(table)
    return overlay_shard_path(feature_dir / "pct" / checksum[:16], stem)


def test_percentile_benchmark_smoke_and_digest_mismatch(tmp_path: Path) -> None:
    feature_dir = _make_env(tmp_path / "input")
    assert run_percentiles(_cfg(feature_dir)) == 0
    compare_to = _pct_overlay_path(feature_dir, "shard_00000")

    out_path = tmp_path / "bench.json"
    result = run_bench(
        OmegaConf.create(
            {
                "feature_dir": str(feature_dir),
                "out": str(out_path),
                "shard": "shard_00000",
                "scratch_dir": str(tmp_path / "scratch"),
                "device": "cpu",
                "zstd_level": 12,
                "compare_to": str(compare_to),
            }
        )
    )
    assert out_path.is_file()
    payload = json.loads(out_path.read_text())
    assert set(payload) == {
        "shard",
        "n_values",
        "load_seconds",
        "compute_seconds",
        "save_seconds",
        "wall_seconds",
        "peak_rss_bytes",
        "overlay_digest",
        "compare_to_digest",
        "digests_match",
        "torch_num_threads",
        "device",
    }
    assert result["n_values"] > 0
    assert result["peak_rss_bytes"] > 0
    assert result["digests_match"] is True
    assert overlay_content_digest(compare_to) == _hash_overlay_tensors(compare_to)

    corrupted = load_overlay_shard(compare_to)
    percentiles = corrupted[next(iter(corrupted))].clone()
    percentiles[0] = t.tensor((int(percentiles[0]) + 1) % 256, dtype=percentiles.dtype)
    corrupt_path = tmp_path / "corrupt.safetensors.zst"
    save_overlay_shard({next(iter(corrupted)): percentiles}, corrupt_path)
    with pytest.raises(SystemExit, match="1"):
        run_bench(
            OmegaConf.create(
                {
                    "feature_dir": str(feature_dir),
                    "out": str(tmp_path / "mismatch.json"),
                    "shard": "shard_00000",
                    "scratch_dir": str(tmp_path / "scratch-mismatch"),
                    "device": "cpu",
                    "compare_to": str(corrupt_path),
                }
            )
        )
    assert json.loads((tmp_path / "mismatch.json").read_text())["digests_match"] is False
