"""Production write paths emit ``schema_version=2`` manifests.

``precompute_percentiles.run`` must thread
``base_shard_paths=glob_features(feature_dir)`` into
``write_manifest`` so the resulting overlay manifest carries a
``base_shard_content_hash`` and a ``schema_version=2`` field. ``discover_overlay``
then rejects the overlay if the base shards are silently re-extracted.

The library-level test suite at
``tests/steering/test_overlay_base_shard_integrity.py`` already exercises the
schema-v2 contract with hand-built manifests; this regression test exercises the
*production write path* end-to-end (the bug surface we're closing).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch as t
from omegaconf import DictConfig, OmegaConf

from scripts.preprocess.precompute_percentiles import run as run_percentiles
from steering.feature_io import save_features
from steering.overlay_io import (
    BASE_SHARD_HASH_CACHE_FILENAME,
    MANIFEST_FILENAME,
    OverlayKind,
    OverlayStaleError,
    discover_overlay,
    glob_overlay_shards,
    overlay_dir,
    overlay_shard_path,
)
from steering.segmentation import SegmentMode
from steering.stats_io import load_percentile_checksum, save_statistics
from steering.stats_types import (
    CorpusStats,
    ExtractionStatistics,
    Extrapolation,
    MetadataStats,
    PerFeatureStat,
    SegmentFeatureStats,
    StorageStats,
)

N_SAE = 64
N_FEATURES_SEEN = 8
SEGMENT_SIZE = 16


def _make_stats_file(feature_dir: Path) -> Path:
    """Minimal valid stats file (mirrors tests/scripts/test_checksum_pipeline_wiring.py)."""
    per_feature_stats: dict[str, PerFeatureStat] = {}
    for fid in range(N_FEATURES_SEEN):
        pcts = {f"p{k}": float(k) * 0.1 + fid * 0.01 for k in range(1, 100)}
        per_feature_stats[str(fid)] = PerFeatureStat(
            count=100,
            pct_segments_active=0.5,
            mean_mag=1.0,
            mean_mag_incl_zeros=0.5,
            std_mag=0.5,
            min_mag=0.1,
            max_mag=2.0,
            median_mag=1.0,
            percentiles=pcts,
        )

    stats = ExtractionStatistics(
        metadata=MetadataStats(
            pooling_mode="mean_pool",
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=SEGMENT_SIZE,
            context_window=1024,
            layer=8,
            n_sae_features=N_SAE,
            generated_at="test",
            activation_percentile=0.0,
        ),
        corpus=CorpusStats(
            n_shards_processed=1, n_docs=2, n_tokens_gpt2=200,
            mean_tokens_per_doc=100.0, median_tokens_per_doc=100,
        ),
        features=SegmentFeatureStats(
            n_segments=10, n_features_seen=N_FEATURES_SEEN,
            mean_active_per_segment=5.0, median_active_per_segment=5,
            p90_active_per_segment=8, max_active_per_segment=10,
            min_active_per_segment=1, n_total_sae_features=N_SAE,
            pct_features_active=0.5, count_percentiles={},
            magnitude_stats={}, active_per_segment_histogram={},
            active_per_segment_percentiles={},
        ),
        storage=StorageStats(
            total_bytes=1000, mean_bytes_per_doc=500.0, mean_bytes_per_token=5.0,
        ),
        per_feature_stats=per_feature_stats,
        extrapolation_10b=Extrapolation(
            target_tokens=10_000_000_000,
            estimated_docs=1_000_000,
            estimated_storage_gb=100.0,
            estimated_wall_time_hours=10.0,
        ),
    )
    return save_statistics(stats, feature_dir / "feature_statistics")


def _make_feature_shard(feature_dir: Path, shard_name: str, seed: int) -> Path:
    t.manual_seed(seed)
    n_features = 30
    # Manifest tests require a healthy percentile producer input: every
    # emitted feature ID must have a statistics row.
    fids = t.randint(0, N_FEATURES_SEEN, (n_features,), dtype=t.int32)
    mags = t.rand(n_features) + 0.1
    offsets = t.tensor([0, 15, n_features], dtype=t.int64)
    data = {
        "offsets": offsets,
        "feature_ids": fids,
        "feature_mags": mags,
        "doc_ids": t.tensor([0, 1], dtype=t.int32),
        "segment_indices": t.tensor([0, 0], dtype=t.int32),
        "segment_positions": t.tensor([0, 0], dtype=t.int32),
        # One segment per doc, so each doc keeps exactly ``segment_size`` tokens;
        # anything larger would be geometry no fixed_tokens extraction can emit.
        "doc_token_counts": t.tensor([SEGMENT_SIZE, SEGMENT_SIZE], dtype=t.int32),
        "segment_char_starts": t.tensor([0, 0], dtype=t.int32),
        "segment_char_ends": t.tensor([64, 64], dtype=t.int32),
        "n_docs": 2,
        "n_segments": 2,
        "layer": 8,
        "context_window": 1024,
        "pooling": "mean_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": SEGMENT_SIZE,
    }
    return save_features(data, feature_dir / shard_name)


def _make_env(tmp_path: Path, n_shards: int = 2) -> tuple[Path, Path]:
    # ``verify_segment_size`` requires a parent dir component matching ``s{N}``;
    # build a minimal canonical path under tmp.
    feature_dir = tmp_path / "features" / f"s{SEGMENT_SIZE}"
    feature_dir.mkdir(parents=True)
    stats_path = _make_stats_file(feature_dir)
    for i in range(n_shards):
        _make_feature_shard(feature_dir, f"shard_{i:05d}", seed=42 + i)
    return feature_dir, stats_path


def _percentile_cfg(feature_dir: Path) -> DictConfig:
    cfg = OmegaConf.create({
        "feature_dir": str(feature_dir),
        "segment_size": SEGMENT_SIZE,
        "device": "cpu",
        "dry_run": False,
        "resume": False,
        "force": False,
        "zstd_level": 12,
    })
    assert isinstance(cfg, DictConfig)
    return cfg


class TestPercentileManifestSchemaVersion2:
    def test_run_writes_v2_manifest_with_base_shard_hash(self, tmp_path):
        feature_dir, stats_path = _make_env(tmp_path)
        ret = run_percentiles(_percentile_cfg(feature_dir))
        assert ret == 0

        checksum = load_percentile_checksum(stats_path)
        assert checksum is not None
        d = overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum)
        manifest_path = d / MANIFEST_FILENAME
        assert manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text())
        assert manifest.get("schema_version") == 2, (
            "production write path must emit schema_version=2 manifests "
            "(consolidation manifest §4.5)"
        )
        assert isinstance(manifest.get("base_shard_content_hash"), str)
        assert len(manifest["base_shard_content_hash"]) == 64

    def test_discover_overlay_with_base_shard_paths_succeeds_after_run(self, tmp_path):
        from steering.feature_io import glob_features

        feature_dir, stats_path = _make_env(tmp_path)
        ret = run_percentiles(_percentile_cfg(feature_dir))
        assert ret == 0
        checksum = load_percentile_checksum(stats_path)
        assert checksum is not None

        # discover_overlay with base_shard_paths=... must accept the manifest
        # written by the production path (no OverlayStaleError raised).
        result = discover_overlay(
            feature_dir, OverlayKind.PERCENTILE, checksum,
            base_shard_paths=glob_features(feature_dir),
        )
        assert result == overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum)

    def test_discover_overlay_raises_when_base_shards_mutated(self, tmp_path):
        from steering.feature_io import glob_features

        feature_dir, stats_path = _make_env(tmp_path)
        ret = run_percentiles(_percentile_cfg(feature_dir))
        assert ret == 0
        checksum = load_percentile_checksum(stats_path)
        assert checksum is not None  # narrow for type checker

        # Re-extract a base shard with new bytes (same path, different seed).
        all_shards = sorted(glob_features(feature_dir))
        target = all_shards[0]
        target.unlink()
        _make_feature_shard(feature_dir, target.name.split(".")[0], seed=999)

        # discover_overlay against the (immutable) overlay must now reject.
        with pytest.raises(OverlayStaleError):
            discover_overlay(
                feature_dir, OverlayKind.PERCENTILE, checksum,
                base_shard_paths=glob_features(feature_dir),
            )

    def test_overlay_shard_files_written(self, tmp_path):
        # Sanity: the run actually wrote shard files (otherwise the v2 fix
        # would be cosmetic).
        feature_dir, stats_path = _make_env(tmp_path, n_shards=2)
        ret = run_percentiles(_percentile_cfg(feature_dir))
        assert ret == 0
        checksum = load_percentile_checksum(stats_path)
        assert checksum is not None  # narrow for type checker
        d = overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum)
        shards = glob_overlay_shards(d)
        assert len(shards) == 2


def test_percentile_producer_rejects_stale_v2_before_mutation(tmp_path):
    """F2: producer discovery authenticates v2 before touching outputs."""
    from steering.feature_io import glob_features

    feature_dir, stats_path = _make_env(tmp_path)
    cfg = _percentile_cfg(feature_dir)
    assert run_percentiles(cfg) == 0
    checksum = load_percentile_checksum(stats_path)
    assert checksum is not None
    output = overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum)

    target = glob_features(feature_dir)[0]
    target.unlink()
    _make_feature_shard(feature_dir, target.name.split(".")[0], seed=999)

    watched = [
        output / MANIFEST_FILENAME,
        *glob_overlay_shards(output),
        stats_path,
        feature_dir / BASE_SHARD_HASH_CACHE_FILENAME,
    ]
    before = {path: path.read_bytes() for path in watched}
    with pytest.raises(OverlayStaleError):
        run_percentiles(cfg)
    assert {path: path.read_bytes() for path in watched} == before


@pytest.mark.parametrize("legacy_v1", [False, True])
def test_percentile_resume_preserves_existing_manifest(tmp_path, legacy_v1):
    """F2: resume fills payloads without restamping either v1 or v2."""
    feature_dir, stats_path = _make_env(tmp_path)
    cfg = _percentile_cfg(feature_dir)
    assert run_percentiles(cfg) == 0
    checksum = load_percentile_checksum(stats_path)
    assert checksum is not None
    output = overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum)
    manifest_path = output / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["created_at"] = "2001-02-03T04:05:06Z"
    if legacy_v1:
        manifest.pop("base_shard_content_hash")
        manifest.pop("schema_version")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_before = manifest_path.read_bytes()

    missing = overlay_shard_path(output, "shard_00001")
    missing.unlink()
    cfg.resume = True
    assert run_percentiles(cfg) == 0
    assert missing.is_file()
    assert manifest_path.read_bytes() == manifest_before
