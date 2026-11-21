"""Tests for percentile producer profiling (issue #1269)."""

from __future__ import annotations

import json
from pathlib import Path

import torch as t
from omegaconf import OmegaConf

from scripts.preprocess.precompute_percentiles import run
from steering.feature_io import save_features
from steering.overlay_io import load_overlay_shard
from steering.segmentation import SegmentMode
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

_N_SAE = 32
_SEGMENT_SIZE = 16
_N_SHARDS = 2


def _make_env(root: Path, n_shards: int = _N_SHARDS) -> Path:
    feature_dir = root / "features" / f"s{_SEGMENT_SIZE}"
    feature_dir.mkdir(parents=True)
    per_feature_stats = {
        str(fid): PerFeatureStat(
            count=100,
            pct_segments_active=0.5,
            mean_mag=1.0,
            mean_mag_incl_zeros=0.5,
            std_mag=0.5,
            min_mag=0.1,
            max_mag=2.0,
            median_mag=1.0,
            percentiles={f"p{k}": float(k) * 0.1 + fid * 0.01 for k in range(1, 100)},
        )
        for fid in range(8)
    }
    stats = ExtractionStatistics(
        metadata=MetadataStats(
            pooling_mode="mean_pool",
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=_SEGMENT_SIZE,
            context_window=1024,
            layer=8,
            n_sae_features=_N_SAE,
            generated_at="test",
            activation_percentile=0.0,
        ),
        corpus=CorpusStats(
            n_shards_processed=n_shards,
            n_docs=2 * n_shards,
            n_tokens_gpt2=100,
            mean_tokens_per_doc=50.0,
            median_tokens_per_doc=50,
        ),
        features=SegmentFeatureStats(
            n_segments=2 * n_shards,
            n_features_seen=8,
            mean_active_per_segment=4.0,
            median_active_per_segment=4,
            p90_active_per_segment=5,
            max_active_per_segment=6,
            min_active_per_segment=1,
            n_total_sae_features=_N_SAE,
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
        per_feature_stats=per_feature_stats,
        extrapolation_10b=Extrapolation(
            target_tokens=10_000_000_000,
            estimated_docs=1_000_000,
            estimated_storage_gb=100.0,
            estimated_wall_time_hours=10.0,
        ),
    )
    save_statistics(stats, feature_dir / "feature_statistics")

    for shard_idx in range(n_shards):
        n_features = 12
        save_features(
            {
                "offsets": t.tensor([0, 6, n_features], dtype=t.int64),
                "feature_ids": t.tensor(
                    [(fid + shard_idx) % 8 for fid in range(n_features)], dtype=t.int32
                ),
                "feature_mags": t.tensor(
                    [0.2 + 0.01 * fid for fid in range(n_features)], dtype=t.float32
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
                "segment_size": _SEGMENT_SIZE,
            },
            feature_dir / f"shard_{shard_idx:05d}_features",
        )
    return feature_dir


def _cfg(feature_dir: Path, **overrides):
    values = {
        "feature_dir": str(feature_dir),
        "segment_size": _SEGMENT_SIZE,
        "device": "cpu",
        "dry_run": False,
        "resume": False,
        "force": False,
        "zstd_level": 12,
    }
    values.update(overrides)
    return OmegaConf.create(values)


def _overlay_tensors(feature_dir: Path) -> dict[str, dict[str, t.Tensor]]:
    pct_dirs = list((feature_dir / "pct").iterdir())
    assert len(pct_dirs) == 1
    return {
        path.name: load_overlay_shard(path)
        for path in sorted(pct_dirs[0].glob("*.safetensors.zst"))
    }


def test_profile_sidecar_written(tmp_path: Path) -> None:
    feature_dir = _make_env(tmp_path)
    sidecar = tmp_path / "pct.json"
    assert run(_cfg(feature_dir, profile=True, profile_summary_output=str(sidecar))) == 0

    payload = json.loads(sidecar.read_text())
    assert set(payload) == {"config", "per_shard", "totals"}
    assert len(payload["per_shard"]) == payload["totals"]["n_processed"]
    assert payload["totals"]["n_processed"] == _N_SHARDS
    assert {record["n_values"] for record in payload["per_shard"]} == {12}
    assert all(
        set(record)
        == {
            "stem",
            "n_values",
            "load_seconds",
            "load_bytes",
            "compute_seconds",
            "save_seconds",
            "overlay_bytes",
        }
        for record in payload["per_shard"]
    )


def test_profile_false_writes_no_sidecar(tmp_path: Path) -> None:
    feature_dir = _make_env(tmp_path)
    sidecar = tmp_path / "should-not-exist.json"
    assert run(_cfg(feature_dir, profile_summary_output=str(sidecar))) == 0
    assert not sidecar.exists()


def test_outputs_unchanged_by_profiling(tmp_path: Path) -> None:
    profiled = _make_env(tmp_path / "profiled")
    unprofiled = _make_env(tmp_path / "unprofiled")
    assert (
        run(
            _cfg(
                profiled,
                profile=True,
                profile_summary_output=str(tmp_path / "profile.json"),
            )
        )
        == 0
    )
    assert run(_cfg(unprofiled)) == 0

    profiled_shards = _overlay_tensors(profiled)
    unprofiled_shards = _overlay_tensors(unprofiled)
    assert profiled_shards.keys() == unprofiled_shards.keys()
    for stem in profiled_shards:
        assert profiled_shards[stem].keys() == unprofiled_shards[stem].keys()
        for key in profiled_shards[stem]:
            assert t.equal(profiled_shards[stem][key], unprofiled_shards[stem][key])


def test_resume_skips_counted(tmp_path: Path) -> None:
    feature_dir = _make_env(tmp_path)
    assert run(_cfg(feature_dir)) == 0
    sidecar = tmp_path / "resume.json"
    assert (
        run(
            _cfg(
                feature_dir,
                resume=True,
                profile=True,
                profile_summary_output=str(sidecar),
            )
        )
        == 0
    )
    totals = json.loads(sidecar.read_text())["totals"]
    assert totals["n_skipped"] == _N_SHARDS
    assert totals["n_processed"] == 0
    assert json.loads(sidecar.read_text())["per_shard"] == []
