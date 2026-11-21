"""Tests for the DistractorSampler and DistractorSet."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import torch as t
from safetensors.torch import save_file

from steering.eval.distractor_sampler import DistractorSampler, DistractorSet


def _make_stats(
    tmp_path: Path,
    feature_ids: list[int] | None = None,
    counts: list[int] | None = None,
    n_features: int = 100,
) -> Path:
    """Create a minimal safetensors stats file."""
    if feature_ids is None:
        feature_ids = list(range(n_features))
    if counts is None:
        counts = [i + 1 for i in range(len(feature_ids))]

    stats_path = tmp_path / "stats.safetensors"
    tensors = {
        "feature_ids": t.tensor(feature_ids, dtype=t.int32),
        "counts": t.tensor(counts, dtype=t.int64),
    }
    save_file(tensors, str(stats_path))
    return stats_path


class TestDistractorSet:
    """Tests for DistractorSet dataclass."""

    def test_to_feature_conditions(self) -> None:
        ds = DistractorSet(feature_ids=(10, 20, 30), magnitudes=(1, 5, 3))
        conds = ds.to_feature_conditions()
        assert conds == {10: 1, 20: 5, 30: 3}

    def test_empty_set(self) -> None:
        ds = DistractorSet(feature_ids=(), magnitudes=())
        assert ds.to_feature_conditions() == {}


class TestDistractorSampler:
    """Tests for DistractorSampler."""

    def test_correct_count(self, tmp_path: Path) -> None:
        stats_path = _make_stats(tmp_path)
        sampler = DistractorSampler(
            stats_path=stats_path,
            magnitude_levels=[0, 1, 2, 3],
            excluded_feature_ids=set(),
        )
        result = sampler.sample(5, sample_seed=123)
        assert len(result.feature_ids) == 5
        assert len(result.magnitudes) == 5

    def test_blacklisted_features_never_appear(self, tmp_path: Path) -> None:
        blacklisted = {10, 20, 30}
        stats_path = _make_stats(tmp_path, n_features=50)
        sampler = DistractorSampler(
            stats_path=stats_path,
            magnitude_levels=[0, 1, 2],
            excluded_feature_ids=blacklisted,
        )
        for seed in range(50):
            result = sampler.sample(10, sample_seed=seed)
            for fid in result.feature_ids:
                assert fid not in blacklisted, f"Blacklisted feature {fid} appeared"

    def test_deterministic_with_same_seed(self, tmp_path: Path) -> None:
        stats_path = _make_stats(tmp_path)
        sampler = DistractorSampler(
            stats_path=stats_path,
            magnitude_levels=[0, 1, 2, 3],
            excluded_feature_ids={0, 1},
        )
        r1 = sampler.sample(10, sample_seed=999)
        r2 = sampler.sample(10, sample_seed=999)
        assert r1.feature_ids == r2.feature_ids
        assert r1.magnitudes == r2.magnitudes

    def test_different_seeds_differ(self, tmp_path: Path) -> None:
        stats_path = _make_stats(tmp_path)
        sampler = DistractorSampler(
            stats_path=stats_path,
            magnitude_levels=[0, 1, 2, 3],
            excluded_feature_ids=set(),
        )
        r1 = sampler.sample(10, sample_seed=1)
        r2 = sampler.sample(10, sample_seed=2)
        # Very unlikely to be the same
        assert r1.feature_ids != r2.feature_ids or r1.magnitudes != r2.magnitudes

    def test_magnitudes_in_range(self, tmp_path: Path) -> None:
        mag_levels = [0, 3, 7]
        stats_path = _make_stats(tmp_path)
        sampler = DistractorSampler(
            stats_path=stats_path,
            magnitude_levels=mag_levels,
            excluded_feature_ids=set(),
        )
        for seed in range(20):
            result = sampler.sample(5, sample_seed=seed)
            for m in result.magnitudes:
                assert m in mag_levels, f"Magnitude {m} not in {mag_levels}"

    def test_frequency_weighting_statistical(self, tmp_path: Path) -> None:
        """Features with higher counts should appear more often on average."""
        # Feature 99 has count 100, feature 0 has count 1
        feature_ids = list(range(100))
        counts = [i + 1 for i in range(100)]
        stats_path = _make_stats(tmp_path, feature_ids=feature_ids, counts=counts)

        sampler = DistractorSampler(
            stats_path=stats_path,
            magnitude_levels=[0],
            excluded_feature_ids=set(),
        )

        appearances: Counter[int] = Counter()
        n_trials = 500
        for seed in range(n_trials):
            result = sampler.sample(1, sample_seed=seed)
            appearances[result.feature_ids[0]] += 1

        # Feature 99 (count=100) should appear more than feature 0 (count=1)
        # With 500 trials this should be very reliable
        high_freq = sum(appearances.get(fid, 0) for fid in range(80, 100))
        low_freq = sum(appearances.get(fid, 0) for fid in range(20))
        assert high_freq > low_freq, (
            f"High-freq features appeared {high_freq} times vs low-freq {low_freq}"
        )

    def test_no_duplicate_features_in_single_sample(self, tmp_path: Path) -> None:
        stats_path = _make_stats(tmp_path, n_features=50)
        sampler = DistractorSampler(
            stats_path=stats_path,
            magnitude_levels=[0, 1, 2],
            excluded_feature_ids=set(),
        )
        for seed in range(30):
            result = sampler.sample(20, sample_seed=seed)
            assert len(set(result.feature_ids)) == len(result.feature_ids), (
                f"Duplicate features in sample: {result.feature_ids}"
            )

    def test_raises_on_n_greater_than_pool(self, tmp_path: Path) -> None:
        stats_path = _make_stats(tmp_path, n_features=10)
        sampler = DistractorSampler(
            stats_path=stats_path,
            magnitude_levels=[0, 1],
            excluded_feature_ids=set(),
        )
        with pytest.raises(ValueError, match="pool only has"):
            sampler.sample(11, sample_seed=1)

    def test_raises_when_all_features_blacklisted(self, tmp_path: Path) -> None:
        n = 10
        stats_path = _make_stats(tmp_path, n_features=n)
        with pytest.raises(ValueError, match="empty"):
            DistractorSampler(
                stats_path=stats_path,
                magnitude_levels=[0],
                excluded_feature_ids=set(range(n)),
            )

    def test_zero_count_features_excluded(self, tmp_path: Path) -> None:
        feature_ids = [0, 1, 2, 3, 4]
        counts = [10, 0, 5, 0, 3]
        stats_path = _make_stats(tmp_path, feature_ids=feature_ids, counts=counts)

        sampler = DistractorSampler(
            stats_path=stats_path,
            magnitude_levels=[0],
            excluded_feature_ids=set(),
        )
        assert sampler.pool_size == 3  # features 0, 2, 4

        for seed in range(20):
            result = sampler.sample(2, sample_seed=seed)
            for fid in result.feature_ids:
                assert fid in {0, 2, 4}, f"Zero-count feature {fid} appeared"

    def test_sample_zero_returns_empty(self, tmp_path: Path) -> None:
        stats_path = _make_stats(tmp_path)
        sampler = DistractorSampler(
            stats_path=stats_path,
            magnitude_levels=[0, 1],
            excluded_feature_ids=set(),
        )
        result = sampler.sample(0, sample_seed=1)
        assert result.feature_ids == ()
        assert result.magnitudes == ()
        assert result.to_feature_conditions() == {}

    def test_pool_size(self, tmp_path: Path) -> None:
        stats_path = _make_stats(tmp_path, n_features=50)
        sampler = DistractorSampler(
            stats_path=stats_path,
            magnitude_levels=[0],
            excluded_feature_ids={10, 20, 30},
        )
        assert sampler.pool_size == 47
