"""Exhaustive statistics roundtrip tests.

Tests the full pipeline: compute_statistics -> save_statistics ->
load_statistics -> build_boundaries_from_stats / build_thresholds_from_stats.

Does NOT duplicate coverage in tests/test_stats_io.py.
"""

from __future__ import annotations

from pathlib import Path

import torch as t

from steering.compute_statistics import compute_statistics
from steering.conditional.discretization import build_boundaries_from_stats
from steering.conditional.thresholding import (
    ThresholdType,
    build_thresholds_from_stats,
)
from steering.feature_io import save_features
from steering.stats_io import load_statistics, save_statistics
from tests.steering._discretization_reference import compute_bucket_boundaries

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _write_synthetic_shards(
    tmp_dir: Path,
    *,
    shard_configs: list[dict] | None = None,
    n_segments: int = 5,
    n_docs: int = 1,
    doc_token_counts: list[int] | None = None,
) -> None:
    """Write feature shard files with controllable data.

    If *shard_configs* is provided, each element must have::

        {
            "feature_ids": list[int],   # flat, per-segment features concatenated
            "feature_mags": list[float],
            "n_features_per_segment": list[int],  # len == n_segments
        }

    Otherwise writes two default shards identical to test_stats_io.
    """
    if shard_configs is None:
        # Default: 2 identical shards with features 0,1,2 per segment
        shard_configs = []
        for _ in range(2):
            feat_ids: list[int] = []
            feat_mags: list[float] = []
            nfps: list[int] = []
            for seg in range(n_segments):
                seg_feats = [0, 1, 2]
                seg_mags = [
                    float(seg + 1),
                    float((seg + 1) * 10),
                    float((seg + 1) * 0.1),
                ]
                feat_ids.extend(seg_feats)
                feat_mags.extend(seg_mags)
                nfps.append(len(seg_feats))
            shard_configs.append(
                {
                    "feature_ids": feat_ids,
                    "feature_mags": feat_mags,
                    "n_features_per_segment": nfps,
                }
            )

    for shard_idx, cfg in enumerate(shard_configs):
        flat_ids = cfg["feature_ids"]
        flat_mags = cfg["feature_mags"]
        nfps = cfg["n_features_per_segment"]
        n_seg = len(nfps)

        offsets = [0]
        for c in nfps:
            offsets.append(offsets[-1] + c)

        if doc_token_counts is None:
            dtc = [n_seg * 128]
        else:
            dtc = doc_token_counts

        shard = {
            "feature_ids": t.tensor(flat_ids, dtype=t.int32),
            "feature_mags": t.tensor(flat_mags, dtype=t.float32),
            "offsets": t.tensor(offsets, dtype=t.int64),
            "n_segments": n_seg,
            "n_docs": n_docs,
            "doc_ids": t.zeros(n_seg, dtype=t.int32),
            "doc_token_counts": t.tensor(dtc, dtype=t.int32),
            "segment_indices": t.arange(n_seg, dtype=t.int32),
            "segment_positions": t.arange(n_seg, dtype=t.int32) * 128,
            "segment_mode": "fixed_tokens",
            "segment_size": 128,
            "context_window": 1024,
            "pooling": "mean_pool",
            "layer": 8,
        }
        save_features(shard, tmp_dir / f"shard_{shard_idx}_features")


# ---------------------------------------------------------------------------
# 1. TestAggregationCorrectness
# ---------------------------------------------------------------------------


class TestAggregationCorrectness:
    """Verify aggregation logic across shards."""

    def test_counts_sum_across_shards(self, tmp_path: Path) -> None:
        """Feature 0 appears 5 times in each of 2 shards -> count=10."""
        shards = []
        for _ in range(2):
            ids = [0] * 5
            mags = [1.0] * 5
            shards.append(
                {
                    "feature_ids": ids,
                    "feature_mags": mags,
                    "n_features_per_segment": [5],
                }
            )

        _write_synthetic_shards(
            tmp_path,
            shard_configs=shards,
            n_segments=1,
            n_docs=1,
            doc_token_counts=[128],
        )
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )
        assert stats.per_feature_stats["0"].count == 10

    def test_min_max_are_global(self, tmp_path: Path) -> None:
        """Shard 1 mags [0.5, 0.8], shard 2 mags [0.1, 0.3] -> min=0.1, max=0.8."""
        shards = [
            {
                "feature_ids": [0, 0],
                "feature_mags": [0.5, 0.8],
                "n_features_per_segment": [2],
            },
            {
                "feature_ids": [0, 0],
                "feature_mags": [0.1, 0.3],
                "n_features_per_segment": [2],
            },
        ]
        _write_synthetic_shards(
            tmp_path,
            shard_configs=shards,
            n_segments=1,
            n_docs=1,
            doc_token_counts=[128],
        )
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )
        fstat = stats.per_feature_stats["0"]
        # Tolerance accounts for float16 quantization in shard storage
        assert abs(fstat.min_mag - 0.1) < 1e-2
        assert abs(fstat.max_mag - 0.8) < 1e-2

    def test_mean_is_weighted_by_count(self, tmp_path: Path) -> None:
        """10 activations at 1.0 and 2 at 5.0 -> mean = (10+10)/(12) = 1.667."""
        shards = [
            {
                "feature_ids": [0] * 10,
                "feature_mags": [1.0] * 10,
                "n_features_per_segment": [10],
            },
            {
                "feature_ids": [0] * 2,
                "feature_mags": [5.0] * 2,
                "n_features_per_segment": [2],
            },
        ]
        _write_synthetic_shards(
            tmp_path,
            shard_configs=shards,
            n_segments=1,
            n_docs=1,
            doc_token_counts=[128],
        )
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )
        expected_mean = (10 * 1.0 + 2 * 5.0) / 12
        actual_mean = stats.per_feature_stats["0"].mean_mag
        assert abs(actual_mean - expected_mean) < 1e-4, (
            f"Expected ~{expected_mean:.4f}, got {actual_mean}"
        )

    def test_metadata_consistent(self, tmp_path: Path) -> None:
        """n_docs, n_tokens should match sum across shards."""
        shards = [
            {
                "feature_ids": [0, 1],
                "feature_mags": [1.0, 2.0],
                "n_features_per_segment": [2],
            },
            {
                "feature_ids": [0, 2],
                "feature_mags": [3.0, 4.0],
                "n_features_per_segment": [2],
            },
        ]
        _write_synthetic_shards(
            tmp_path,
            shard_configs=shards,
            n_segments=1,
            n_docs=1,
            doc_token_counts=[256],
        )
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )
        # 2 shards, each 1 doc with 256 tokens
        assert stats.corpus.n_docs == 2
        assert stats.corpus.n_tokens_gpt2 == 512

    def test_streaming_matches_non_streaming(self, tmp_path: Path) -> None:
        """Same shards with streaming=True and False produce identical stats."""
        shards = []
        for _ in range(2):
            ids = [0, 1, 2, 0, 1]
            mags = [1.0, 2.0, 0.5, 3.0, 1.5]
            shards.append(
                {
                    "feature_ids": ids,
                    "feature_mags": mags,
                    "n_features_per_segment": [3, 2],
                }
            )
        _write_synthetic_shards(
            tmp_path,
            shard_configs=shards,
            n_segments=2,
            n_docs=1,
            doc_token_counts=[256],
        )

        dir_eager = tmp_path / "eager"
        dir_eager.mkdir()
        dir_stream = tmp_path / "stream"
        dir_stream.mkdir()

        stats_eager = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(dir_eager / "stats"),
            streaming=False,
        )
        stats_stream = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(dir_stream / "stats"),
            streaming=True,
        )

        # Compare per-feature stats
        assert set(stats_eager.per_feature_stats.keys()) == set(
            stats_stream.per_feature_stats.keys()
        )
        for fid in stats_eager.per_feature_stats:
            e = stats_eager.per_feature_stats[fid]
            s = stats_stream.per_feature_stats[fid]
            assert e.count == s.count, f"feature {fid} count mismatch"
            assert abs(e.mean_mag - s.mean_mag) < 1e-4, (
                f"feature {fid} mean_mag: {e.mean_mag} vs {s.mean_mag}"
            )
            assert abs(e.min_mag - s.min_mag) < 1e-4
            assert abs(e.max_mag - s.max_mag) < 1e-4
            assert abs(e.median_mag - s.median_mag) < 1e-4
            for pk in e.percentiles:
                assert abs(e.percentiles[pk] - s.percentiles[pk]) < 1e-4, (
                    f"feature {fid} {pk}: {e.percentiles[pk]} vs {s.percentiles[pk]}"
                )


# ---------------------------------------------------------------------------
# 2. TestPercentileAccuracy
# ---------------------------------------------------------------------------


class TestPercentileAccuracy:
    """Verify percentile values are correct."""

    def test_percentile_p50_is_median(self, tmp_path: Path) -> None:
        """For known uniform-ish data, p50 should approximate the median_mag."""
        # Create many distinct values for a single feature so percentiles are meaningful
        n = 100
        ids = list(range(n))  # feature 0..99, one activation each -- not useful
        # Better: single feature, many activations
        ids = [0] * n
        mags = [float(i + 1) for i in range(n)]  # 1.0 .. 100.0

        shards = [
            {
                "feature_ids": ids,
                "feature_mags": mags,
                "n_features_per_segment": [n],
            }
        ]
        _write_synthetic_shards(
            tmp_path,
            shard_configs=shards,
            n_segments=1,
            n_docs=1,
            doc_token_counts=[128],
        )
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )
        fstat = stats.per_feature_stats["0"]
        # p50 and median_mag should be very close
        assert abs(fstat.percentiles["p50"] - fstat.median_mag) < 1e-3

    def test_percentile_monotonic(self, tmp_path: Path) -> None:
        """For any feature, p10 <= p20 <= ... <= p99."""
        n = 200
        ids = [0] * n
        mags = [float(i) * 0.5 + 0.1 for i in range(n)]

        shards = [
            {
                "feature_ids": ids,
                "feature_mags": mags,
                "n_features_per_segment": [n],
            }
        ]
        _write_synthetic_shards(
            tmp_path,
            shard_configs=shards,
            n_segments=1,
            n_docs=1,
            doc_token_counts=[128],
        )
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )
        fstat = stats.per_feature_stats["0"]
        pct_keys = sorted(fstat.percentiles.keys(), key=lambda k: int(k[1:]))
        values = [fstat.percentiles[k] for k in pct_keys]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1] + 1e-9, (
                f"{pct_keys[i]}={values[i]} > {pct_keys[i + 1]}={values[i + 1]}"
            )

    def test_percentile_roundtrip_precision(self, tmp_path: Path) -> None:
        """compute -> save -> load. Loaded percentile values match within 1e-5."""
        n = 50
        ids = [0] * n
        mags = [float(i) * 1.23 + 0.01 for i in range(n)]

        shards = [
            {
                "feature_ids": ids,
                "feature_mags": mags,
                "n_features_per_segment": [n],
            }
        ]
        _write_synthetic_shards(
            tmp_path,
            shard_configs=shards,
            n_segments=1,
            n_docs=1,
            doc_token_counts=[128],
        )
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )
        sf_path = save_statistics(stats, tmp_path / "rt_stats")
        loaded = load_statistics(sf_path)

        orig = stats.per_feature_stats["0"]
        loaded_f = loaded.per_feature_stats["0"]
        for pk, pv in orig.percentiles.items():
            assert abs(loaded_f.percentiles[pk] - pv) < 1e-5, (
                f"{pk}: orig={pv}, loaded={loaded_f.percentiles[pk]}"
            )


# ---------------------------------------------------------------------------
# 3. TestPerFeatureStats
# ---------------------------------------------------------------------------


class TestPerFeatureStats:
    """Verify per-feature stat properties."""

    def test_features_with_few_occurrences_get_percentiles(
        self, tmp_path: Path
    ) -> None:
        """Feature with only 2 activations still gets valid percentiles."""
        shards = [
            {
                "feature_ids": [0, 0],
                "feature_mags": [1.0, 2.0],
                "n_features_per_segment": [2],
            }
        ]
        _write_synthetic_shards(
            tmp_path,
            shard_configs=shards,
            n_segments=1,
            n_docs=1,
            doc_token_counts=[128],
        )
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )
        fstat = stats.per_feature_stats["0"]
        assert fstat.count == 2
        # Should have percentiles, none should be NaN
        assert len(fstat.percentiles) > 0
        for pk, pv in fstat.percentiles.items():
            assert pv == pv, f"NaN found at {pk}"  # NaN != NaN

    def test_feature_never_activating_absent_from_stats(self, tmp_path: Path) -> None:
        """Feature IDs that never appear should NOT be in per_feature_stats."""
        shards = [
            {
                "feature_ids": [0, 2],
                "feature_mags": [1.0, 2.0],
                "n_features_per_segment": [2],
            }
        ]
        _write_synthetic_shards(
            tmp_path,
            shard_configs=shards,
            n_segments=1,
            n_docs=1,
            doc_token_counts=[128],
        )
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )
        assert "1" not in stats.per_feature_stats
        assert "0" in stats.per_feature_stats
        assert "2" in stats.per_feature_stats

    def test_per_feature_stats_sorted_by_count_desc(self, tmp_path: Path) -> None:
        """per_feature_stats dict should be ordered by count descending."""
        # Feature 0: 1 activation, Feature 1: 5, Feature 2: 3
        shards = [
            {
                "feature_ids": [0] + [1] * 5 + [2] * 3,
                "feature_mags": [1.0] + [2.0] * 5 + [3.0] * 3,
                "n_features_per_segment": [9],
            }
        ]
        _write_synthetic_shards(
            tmp_path,
            shard_configs=shards,
            n_segments=1,
            n_docs=1,
            doc_token_counts=[128],
        )
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )
        counts = [fs.count for fs in stats.per_feature_stats.values()]
        assert counts == sorted(counts, reverse=True), (
            f"Expected descending order, got {counts}"
        )


# ---------------------------------------------------------------------------
# 4. TestBoundariesFromStatsRoundtrip
# ---------------------------------------------------------------------------


class TestBoundariesFromStatsRoundtrip:
    """Full pipeline: shards -> stats -> save -> load -> boundaries."""

    def test_compute_save_load_build_boundaries_matches_raw(
        self, tmp_path: Path
    ) -> None:
        """Boundaries from loaded stats match those from raw data."""
        # Generate data with enough activations per feature for meaningful quantiles
        n_feats = 4
        n_per_feat = 50
        all_ids: list[int] = []
        all_mags: list[float] = []
        for fid in range(n_feats):
            all_ids.extend([fid] * n_per_feat)
            all_mags.extend([float(i + 1) * (fid + 1) * 0.1 for i in range(n_per_feat)])

        shards = [
            {
                "feature_ids": all_ids,
                "feature_mags": all_mags,
                "n_features_per_segment": [len(all_ids)],
            }
        ]
        _write_synthetic_shards(
            tmp_path,
            shard_configs=shards,
            n_segments=1,
            n_docs=1,
            doc_token_counts=[128],
        )

        # Step 1-3: compute -> save -> load
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )
        sf_path = save_statistics(stats, tmp_path / "saved_stats")
        loaded = load_statistics(sf_path)

        num_buckets = 4
        n_sae = stats.metadata.n_sae_features

        # Step 4a: boundaries from loaded stats
        stats_boundaries = build_boundaries_from_stats(
            loaded.per_feature_stats, n_sae, num_buckets
        )

        # Step 4b: boundaries from raw data
        raw_boundaries = compute_bucket_boundaries(
            t.tensor(all_ids, dtype=t.int32),
            t.tensor(all_mags, dtype=t.float32),
            num_buckets,
        )

        # Compare for features that exist (raw boundaries may differ in size)
        for fid in range(n_feats):
            stats_b = stats_boundaries[fid]
            raw_b = raw_boundaries[fid]
            # The stats-based approach uses stored percentile keys (p25,p50,p75 for 4 buckets)
            # while raw uses torch.quantile -- allow moderate tolerance
            for j in range(num_buckets - 1):
                assert abs(stats_b[j].item() - raw_b[j].item()) < 0.5, (
                    f"Feature {fid}, boundary {j}: stats={stats_b[j].item():.4f} "
                    f"vs raw={raw_b[j].item():.4f}"
                )


# ---------------------------------------------------------------------------
# 5. TestThresholdsFromStatsRoundtrip
# ---------------------------------------------------------------------------


class TestThresholdsFromStatsRoundtrip:
    """Full pipeline: shards -> stats -> save -> load -> thresholds."""

    def test_compute_save_load_build_thresholds_matches_raw(
        self, tmp_path: Path
    ) -> None:
        """Thresholds from loaded stats match those from raw data."""
        n_feats = 3
        n_per_feat = 100
        all_ids: list[int] = []
        all_mags: list[float] = []
        for fid in range(n_feats):
            all_ids.extend([fid] * n_per_feat)
            all_mags.extend([float(i + 1) * 0.05 for i in range(n_per_feat)])

        shards = [
            {
                "feature_ids": all_ids,
                "feature_mags": all_mags,
                "n_features_per_segment": [len(all_ids)],
            }
        ]
        _write_synthetic_shards(
            tmp_path,
            shard_configs=shards,
            n_segments=1,
            n_docs=1,
            doc_token_counts=[128],
        )

        # Compute stats with activation_percentile
        percentile = 0.5
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            activation_percentile=percentile,
            streaming=False,
        )
        sf_path = save_statistics(stats, tmp_path / "saved_stats")
        loaded = load_statistics(sf_path)

        n_sae = stats.metadata.n_sae_features

        # Build thresholds from loaded stats
        stats_thresholds = build_thresholds_from_stats(
            loaded.per_feature_stats,
            n_sae,
            ThresholdType.feature_percentile,
            percentile,
        )

        # build_thresholds_from_stats should read the stored per-feature
        # percentile values exactly.
        pct_key = f"p{int(percentile * 100)}"
        assert isinstance(stats_thresholds, t.Tensor)
        for fid_str, fstat in loaded.per_feature_stats.items():
            fid = int(fid_str)
            if fid < n_sae:
                expected = fstat.percentiles[pct_key]
                actual = stats_thresholds[fid].item()
                assert abs(actual - expected) < 1e-4, (
                    f"Feature {fid}: stats threshold={actual} vs stored percentile={expected}"
                )


# ---------------------------------------------------------------------------
# 6. TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for the statistics pipeline."""

    def test_single_shard_roundtrip(self, tmp_path: Path) -> None:
        """Only 1 shard, still works end-to-end."""
        shards = [
            {
                "feature_ids": [0, 1, 2],
                "feature_mags": [1.0, 2.0, 3.0],
                "n_features_per_segment": [3],
            }
        ]
        _write_synthetic_shards(
            tmp_path,
            shard_configs=shards,
            n_segments=1,
            n_docs=1,
            doc_token_counts=[128],
        )
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )
        sf_path = save_statistics(stats, tmp_path / "rt_stats")
        loaded = load_statistics(sf_path)

        assert loaded.corpus.n_shards_processed == 1
        assert len(loaded.per_feature_stats) == 3

    def test_single_feature_per_segment(self, tmp_path: Path) -> None:
        """Each segment has exactly 1 feature activation."""
        n_segments = 4
        ids: list[int] = []
        mags: list[float] = []
        nfps: list[int] = []
        for seg in range(n_segments):
            ids.append(seg % 2)  # alternating features 0, 1
            mags.append(float(seg + 1))
            nfps.append(1)

        shards = [
            {
                "feature_ids": ids,
                "feature_mags": mags,
                "n_features_per_segment": nfps,
            }
        ]
        _write_synthetic_shards(
            tmp_path,
            shard_configs=shards,
            n_segments=n_segments,
            n_docs=1,
            doc_token_counts=[n_segments * 128],
        )
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )
        sf_path = save_statistics(stats, tmp_path / "rt_stats")
        loaded = load_statistics(sf_path)

        assert loaded.per_feature_stats["0"].count == 2
        assert loaded.per_feature_stats["1"].count == 2

    def test_many_features_few_segments(self, tmp_path: Path) -> None:
        """1000 unique features, 3 segments -- some features appear only once."""
        n_features = 1000
        n_segments = 3
        ids: list[int] = []
        mags: list[float] = []
        nfps: list[int] = []

        # Segment 0: features 0..499
        seg0_ids = list(range(500))
        seg0_mags = [float(i + 1) * 0.01 for i in range(500)]
        ids.extend(seg0_ids)
        mags.extend(seg0_mags)
        nfps.append(500)

        # Segment 1: features 250..749
        seg1_ids = list(range(250, 750))
        seg1_mags = [float(i + 1) * 0.02 for i in range(500)]
        ids.extend(seg1_ids)
        mags.extend(seg1_mags)
        nfps.append(500)

        # Segment 2: features 500..999
        seg2_ids = list(range(500, 1000))
        seg2_mags = [float(i + 1) * 0.03 for i in range(500)]
        ids.extend(seg2_ids)
        mags.extend(seg2_mags)
        nfps.append(500)

        shards = [
            {
                "feature_ids": ids,
                "feature_mags": mags,
                "n_features_per_segment": nfps,
            }
        ]
        _write_synthetic_shards(
            tmp_path,
            shard_configs=shards,
            n_segments=n_segments,
            n_docs=1,
            doc_token_counts=[n_segments * 128],
        )
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )
        sf_path = save_statistics(stats, tmp_path / "rt_stats")
        loaded = load_statistics(sf_path)

        # Features 0..249 and 750..999 appear only once
        assert loaded.per_feature_stats["0"].count == 1
        assert loaded.per_feature_stats["999"].count == 1
        # Features 250..499 appear in segments 0 and 1 -> count=2
        assert loaded.per_feature_stats["300"].count == 2
        # Features 500..749 appear in segments 1 and 2 -> count=2
        assert loaded.per_feature_stats["600"].count == 2
        # Total features seen
        assert loaded.features.n_features_seen == n_features


# ---------------------------------------------------------------------------
# 7. TestSaveLoadPrecision
# ---------------------------------------------------------------------------


class TestSaveLoadPrecision:
    """Verify numeric precision of save/load cycle."""

    def test_float64_precision_preserved(self, tmp_path: Path) -> None:
        """Save float64 values, verify they load back with <= 1e-5 error."""
        # Use values that stress float precision
        n = 30
        ids = [0] * n
        mags = [1.0 / 3.0 * (i + 1) for i in range(n)]

        shards = [
            {
                "feature_ids": ids,
                "feature_mags": mags,
                "n_features_per_segment": [n],
            }
        ]
        _write_synthetic_shards(
            tmp_path,
            shard_configs=shards,
            n_segments=1,
            n_docs=1,
            doc_token_counts=[128],
        )
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )
        sf_path = save_statistics(stats, tmp_path / "rt_stats")
        loaded = load_statistics(sf_path)

        orig = stats.per_feature_stats["0"]
        loaded_f = loaded.per_feature_stats["0"]

        assert abs(loaded_f.mean_mag - orig.mean_mag) < 1e-5
        assert abs(loaded_f.std_mag - orig.std_mag) < 1e-5
        assert abs(loaded_f.min_mag - orig.min_mag) < 1e-5
        assert abs(loaded_f.max_mag - orig.max_mag) < 1e-5
        assert abs(loaded_f.median_mag - orig.median_mag) < 1e-5

    def test_histogram_roundtrip(self, tmp_path: Path) -> None:
        """Histograms present in stats should roundtrip correctly."""
        n = 100
        ids = [0] * n
        mags = [float(i + 1) * 0.1 for i in range(n)]

        shards = [
            {
                "feature_ids": ids,
                "feature_mags": mags,
                "n_features_per_segment": [n],
            }
        ]
        _write_synthetic_shards(
            tmp_path,
            shard_configs=shards,
            n_segments=1,
            n_docs=1,
            doc_token_counts=[128],
        )
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )
        sf_path = save_statistics(stats, tmp_path / "rt_stats")
        loaded = load_statistics(sf_path)

        orig = stats.per_feature_stats["0"]
        loaded_f = loaded.per_feature_stats["0"]

        # Both should have histograms
        assert orig.histogram is not None, "Original should have histogram"
        assert loaded_f.histogram is not None, "Loaded should have histogram"

        # Histogram counts should match exactly
        orig_counts = orig.histogram["counts"]
        loaded_counts = loaded_f.histogram["counts"]
        assert orig_counts == loaded_counts, (
            f"Histogram counts differ: {orig_counts} vs {loaded_counts}"
        )

    def test_count_percentiles_are_int(self, tmp_path: Path) -> None:
        """Count percentiles should be integers after loading."""
        shards = []
        for _ in range(2):
            ids = [0, 1, 2, 0, 1]
            mags = [1.0, 2.0, 3.0, 4.0, 5.0]
            shards.append(
                {
                    "feature_ids": ids,
                    "feature_mags": mags,
                    "n_features_per_segment": [3, 2],
                }
            )
        _write_synthetic_shards(
            tmp_path,
            shard_configs=shards,
            n_segments=2,
            n_docs=1,
            doc_token_counts=[256],
        )
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )
        sf_path = save_statistics(stats, tmp_path / "rt_stats")
        loaded = load_statistics(sf_path)

        for key, val in loaded.features.count_percentiles.items():
            assert isinstance(val, int), (
                f"count_percentiles[{key!r}] should be int, got {type(val).__name__}"
            )
