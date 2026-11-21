"""Tests for steering.stats_io safetensors statistics I/O."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch as t

from steering.compute_statistics import (
    compute_statistics,
)
from steering.feature_io import save_features
from steering.segmentation import SegmentMode
from steering.stats_io import (
    content_sha256,
    find_statistics_path,
    glob_statistics,
    load_percentile_checksum,
    load_statistics,
    load_statistics_and_checksum_from_bytes,
    save_percentile_checksum,
    save_statistics,
    statistics_path,
    verify_statistics_sha256,
)
from steering.stats_types import (
    CorpusStats,
    ExtractionStatistics,
    Extrapolation,
    MetadataStats,
    SegmentFeatureStats,
    StorageStats,
)


def _write_synthetic_shards(tmp_dir: Path, n_shards: int = 2) -> None:
    """Write synthetic shard files for integration tests."""
    for shard_idx in range(n_shards):
        n_segments = 5
        feat_ids = []
        feat_mags = []
        offsets = [0]
        for seg in range(n_segments):
            seg_feats = [0, 1, 2]
            seg_mags = [
                float(seg + 1),
                float((seg + 1) * 10),
                float((seg + 1) * 0.1),
            ]
            feat_ids.extend(seg_feats)
            feat_mags.extend(seg_mags)
            offsets.append(offsets[-1] + len(seg_feats))

        shard = {
            "feature_ids": t.tensor(feat_ids, dtype=t.int32),
            "feature_mags": t.tensor(feat_mags, dtype=t.float32),
            "offsets": t.tensor(offsets, dtype=t.int64),
            "n_segments": n_segments,
            "n_docs": 1,
            "doc_ids": t.zeros(n_segments, dtype=t.int32),
            "doc_token_counts": t.tensor([640], dtype=t.int32),
            "segment_indices": t.arange(n_segments, dtype=t.int32),
            "segment_positions": t.arange(n_segments, dtype=t.int32) * 128,
            "segment_mode": "fixed_tokens",
            "segment_size": 128,
            "context_window": 1024,
            "pooling": "mean_pool",
            "layer": 8,
        }
        save_features(shard, tmp_dir / f"shard_{shard_idx}_features")


class TestSaveLoadRoundTrip:
    """Verify save_statistics -> load_statistics preserves all data."""

    def test_basic_round_trip(self, tmp_path):
        """Save ExtractionStatistics then load back -- all fields match."""
        _write_synthetic_shards(tmp_path)
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )

        sf_path = save_statistics(stats, tmp_path / "stats")
        assert sf_path.suffix == ".safetensors"
        assert sf_path.exists()

        loaded = load_statistics(sf_path)

        # Metadata
        assert loaded.metadata.n_sae_features == stats.metadata.n_sae_features
        assert loaded.metadata.pooling_mode == stats.metadata.pooling_mode
        assert loaded.metadata.layer == stats.metadata.layer

        # Corpus
        assert loaded.corpus.n_docs == stats.corpus.n_docs
        assert loaded.corpus.n_tokens_gpt2 == stats.corpus.n_tokens_gpt2

        # Per-feature stats
        assert len(loaded.per_feature_stats) == len(stats.per_feature_stats)
        for fid, fstat in stats.per_feature_stats.items():
            loaded_fstat = loaded.per_feature_stats[fid]
            assert loaded_fstat.count == fstat.count
            assert abs(loaded_fstat.mean_mag - fstat.mean_mag) < 1e-5
            assert abs(loaded_fstat.median_mag - fstat.median_mag) < 1e-5
            for pk, pv in fstat.percentiles.items():
                assert abs(loaded_fstat.percentiles[pk] - pv) < 1e-5

    def test_streaming_stats_round_trip(self, tmp_path):
        """Streaming pipeline output also round-trips correctly."""
        _write_synthetic_shards(tmp_path, n_shards=3)
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=True,
        )

        sf_path = save_statistics(stats, tmp_path / "stats")
        loaded = load_statistics(sf_path)

        for fid, fstat in stats.per_feature_stats.items():
            loaded_fstat = loaded.per_feature_stats[fid]
            assert loaded_fstat.count == fstat.count
            for pk, pv in fstat.percentiles.items():
                assert abs(loaded_fstat.percentiles[pk] - pv) < 1e-5


class TestLoadStatisticsAndChecksumFromBytes:
    """``load_statistics_and_checksum_from_bytes`` parses tensors + metadata
    + checksum from an in-memory snapshot with no filesystem access.
    Pins the ``_load_safetensors_stats`` / ``_reconstruct_statistics``
    refactor (Codex fix batch finding 3) as behavior-preserving: the
    bytes-based path must agree EXACTLY with the pre-existing
    path-based ``load_statistics`` + ``load_percentile_checksum`` pair.
    """

    def test_matches_path_based_load_exactly(self, tmp_path):
        _write_synthetic_shards(tmp_path)
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )
        sf_path = save_statistics(stats, tmp_path / "stats")
        save_percentile_checksum(sf_path, "f" * 64)

        path_based_stats = load_statistics(sf_path)
        path_based_checksum = load_percentile_checksum(sf_path)

        bytes_based_stats, bytes_based_checksum, raw_feature_ids = (
            load_statistics_and_checksum_from_bytes(sf_path.read_bytes())
        )

        assert bytes_based_checksum == path_based_checksum == "f" * 64
        assert bytes_based_stats == path_based_stats
        assert raw_feature_ids is not None
        assert sorted(raw_feature_ids.tolist()) == sorted(
            int(fid) for fid in stats.per_feature_stats
        )

    def test_raw_feature_ids_is_none_when_no_active_features(self, tmp_path):
        stats = ExtractionStatistics(
            metadata=MetadataStats(
                pooling_mode="max_pool",
                segment_mode=SegmentMode.fixed_tokens,
                segment_size=128,
                context_window=1024,
                layer=8,
                n_sae_features=16,
                generated_at="2026-06-30",
                activation_percentile=0.0,
            ),
            corpus=CorpusStats(
                n_shards_processed=0,
                n_docs=0,
                n_tokens_gpt2=0,
                mean_tokens_per_doc=0.0,
                median_tokens_per_doc=0,
            ),
            features=SegmentFeatureStats(
                n_segments=0,
                n_features_seen=0,
                mean_active_per_segment=0.0,
                median_active_per_segment=0,
                p90_active_per_segment=0,
                max_active_per_segment=0,
                min_active_per_segment=0,
                n_total_sae_features=16,
                pct_features_active=0.0,
                count_percentiles={},
                magnitude_stats={},
                active_per_segment_histogram={},
                active_per_segment_percentiles={},
            ),
            storage=StorageStats(total_bytes=0, mean_bytes_per_doc=0.0, mean_bytes_per_token=0.0),
            per_feature_stats={},
            extrapolation_10b=Extrapolation(
                target_tokens=0,
                estimated_docs=0,
                estimated_storage_gb=0.0,
                estimated_wall_time_hours=0.0,
            ),
        )
        sf_path = save_statistics(stats, tmp_path / "stats")

        _, _, raw_feature_ids = load_statistics_and_checksum_from_bytes(sf_path.read_bytes())

        assert raw_feature_ids is None

    def test_checksum_is_none_when_absent(self, tmp_path):
        _write_synthetic_shards(tmp_path)
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )
        sf_path = save_statistics(stats, tmp_path / "stats")

        _, checksum, _ = load_statistics_and_checksum_from_bytes(sf_path.read_bytes())

        assert checksum is None


class TestLoadRejectsNonSafetensors:
    """Verify load_statistics() rejects non-safetensors files."""

    def test_unknown_extension_raises(self, tmp_path):
        """load_statistics() raises for unknown extensions."""
        bad_path = tmp_path / "stats.xyz"
        bad_path.write_text("{}")
        try:
            load_statistics(bad_path)
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert ".xyz" in str(e)


class TestPathHelpers:
    """Test statistics_path and find_statistics_path."""

    def test_statistics_path(self, tmp_path):
        p = statistics_path(tmp_path)
        assert p == tmp_path / "feature_statistics.safetensors"

    def test_find_safetensors(self, tmp_path):
        (tmp_path / "feature_statistics.safetensors").write_bytes(b"")
        found = find_statistics_path(tmp_path)
        assert found is not None
        assert found.suffix == ".safetensors"

    def test_find_ignores_json(self, tmp_path):
        """JSON files are not found."""
        (tmp_path / "feature_statistics.json").write_text("{}")
        assert find_statistics_path(tmp_path) is None

    def test_find_returns_none(self, tmp_path):
        assert find_statistics_path(tmp_path) is None


class TestGlobStatistics:
    """Test recursive statistics file discovery."""

    def test_glob_finds_safetensors(self, tmp_path):
        d1 = tmp_path / "a" / "b"
        d1.mkdir(parents=True)
        d2 = tmp_path / "c"
        d2.mkdir()
        (d1 / "feature_statistics.safetensors").write_bytes(b"")
        (d2 / "feature_statistics.safetensors").write_bytes(b"")

        found = glob_statistics(tmp_path)
        assert len(found) == 2

    def test_glob_ignores_json(self, tmp_path):
        """JSON files are not discovered."""
        (tmp_path / "feature_statistics.json").write_text("{}")
        found = glob_statistics(tmp_path)
        assert len(found) == 0


class TestPerFeatureThresholds:
    """Test that per_feature_thresholds round-trip correctly."""

    def test_thresholds_preserved(self, tmp_path):
        _write_synthetic_shards(tmp_path)
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            activation_percentile=0.5,
            streaming=False,
        )

        sf_path = save_statistics(stats, tmp_path / "stats")
        loaded = load_statistics(sf_path)

        assert loaded.per_feature_thresholds is not None
        for fid, thresh in stats.per_feature_thresholds.items():  # ty: ignore[unresolved-attribute]
            assert abs(loaded.per_feature_thresholds[fid] - thresh) < 1e-5

    def test_null_thresholds(self, tmp_path):
        _write_synthetic_shards(tmp_path)
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            activation_percentile=0.0,
        )

        sf_path = save_statistics(stats, tmp_path / "stats")
        loaded = load_statistics(sf_path)
        assert loaded.per_feature_thresholds is None


class TestContentSha256:
    """content_sha256 must be stable across independent re-saves of equal
    decoded content, unlike a raw file-byte hash (the safetensors crate
    randomizes its metadata HashMap's serialized key order per save)."""

    def test_stable_across_independent_saves_of_equal_content(self, tmp_path):
        _write_synthetic_shards(tmp_path)
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )

        path_a = save_statistics(stats, tmp_path / "stats_a")
        path_b = save_statistics(stats, tmp_path / "stats_b")

        assert content_sha256(path_a) == content_sha256(path_b)

    def test_reproducible_for_the_same_file(self, tmp_path):
        _write_synthetic_shards(tmp_path)
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )
        sf_path = save_statistics(stats, tmp_path / "stats")

        assert content_sha256(sf_path) == content_sha256(sf_path)

    def test_sensitive_to_tensor_content_changes(self, tmp_path):
        _write_synthetic_shards(tmp_path)
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )
        sf_path = save_statistics(stats, tmp_path / "stats")
        baseline = content_sha256(sf_path)

        _write_synthetic_shards(tmp_path / "more", n_shards=3)
        other_stats = compute_statistics(
            feature_dir=str(tmp_path / "more"),
            output_path=str(tmp_path / "other_stats"),
            streaming=False,
        )
        other_path = save_statistics(other_stats, tmp_path / "other_stats")

        assert content_sha256(other_path) != baseline


class TestVerifyStatisticsSha256:
    def test_accepts_canonical_content_hash(self, tmp_path):
        _write_synthetic_shards(tmp_path)
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )
        path = save_statistics(stats, tmp_path / "stats")
        expected = content_sha256(path)

        assert (
            verify_statistics_sha256(path, expected, surface="test statistics")
            == expected
        )

    def test_accepts_exact_legacy_raw_file_hash(self, tmp_path):
        _write_synthetic_shards(tmp_path)
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )
        path = save_statistics(stats, tmp_path / "stats")
        expected = hashlib.sha256(path.read_bytes()).hexdigest()

        assert expected != content_sha256(path)
        assert (
            verify_statistics_sha256(path, expected, surface="test statistics")
            == expected
        )

    def test_rejects_digest_matching_neither_identity(self, tmp_path):
        _write_synthetic_shards(tmp_path)
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )
        path = save_statistics(stats, tmp_path / "stats")

        with pytest.raises(
            ValueError,
            match="actual_content=.*actual_file=",
        ):
            verify_statistics_sha256(path, "0" * 64, surface="test statistics")
