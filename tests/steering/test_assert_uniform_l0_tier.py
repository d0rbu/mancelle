"""Unit tests for steering.feature_io.assert_uniform_l0_tier."""

from pathlib import Path

import pytest

from steering.feature_io import (
    FeatureDirectoryMismatchError,
    assert_uniform_l0_tier,
    glob_features,
    save_features,
)
from steering.sae_resolver import L0Tier


def _sample_shard(l0_tier: str | None = None) -> dict:
    import torch as t

    data: dict = {
        "feature_ids": t.tensor([1, 2, 3], dtype=t.long),
        "feature_mags": t.tensor([0.5, 0.7, 0.3], dtype=t.float32),
        "offsets": t.tensor([0, 3], dtype=t.long),
        "layer": 22,
        "segment_size": 128,
        "context_window": 1024,
        "pooling": "max_pool",
        "segment_mode": "fixed",
        "n_docs": 1,
        "n_segments": 1,
        "context_independent_features": False,
    }
    if l0_tier is not None:
        data["l0_tier"] = l0_tier
    return data


def _write(tmp_path: Path, name: str, l0_tier: str | None) -> Path:
    save_features(_sample_shard(l0_tier), tmp_path / name)
    return tmp_path / name


class TestAssertUniformL0Tier:
    def test_empty_iterable_returns_none(self):
        assert assert_uniform_l0_tier([]) is None

    def test_single_v1_shard_returns_none(self, tmp_path):
        _write(tmp_path, "shard_00001", None)
        assert assert_uniform_l0_tier(glob_features(tmp_path)) is None

    def test_single_v2_shard_returns_tier(self, tmp_path):
        _write(tmp_path, "shard_00001", "medium")
        assert assert_uniform_l0_tier(glob_features(tmp_path)) is L0Tier.medium

    @pytest.mark.parametrize(
        "tier", ["small", "medium", "medium_seed_1", "big"]
    )
    def test_uniform_v2_each_tier(self, tmp_path, tier):
        _write(tmp_path, "shard_00001", tier)
        _write(tmp_path, "shard_00002", tier)
        assert assert_uniform_l0_tier(glob_features(tmp_path)) == L0Tier(tier)

    def test_uniform_v1_multi_shard_returns_none(self, tmp_path):
        _write(tmp_path, "shard_00001", None)
        _write(tmp_path, "shard_00002", None)
        assert assert_uniform_l0_tier(glob_features(tmp_path)) is None

    def test_mixed_tier_raises_listing_all_offenders(self, tmp_path):
        _write(tmp_path, "shard_00001", "medium")
        _write(tmp_path, "shard_00002", "big")
        with pytest.raises(FeatureDirectoryMismatchError) as ei:
            assert_uniform_l0_tier(glob_features(tmp_path))
        msg = str(ei.value)
        assert "medium" in msg and "big" in msg
        assert "shard_00001" in msg and "shard_00002" in msg

    def test_mixed_v1_v2_raises(self, tmp_path):
        _write(tmp_path, "shard_00001", None)
        _write(tmp_path, "shard_00002", "medium")
        with pytest.raises(FeatureDirectoryMismatchError):
            assert_uniform_l0_tier(glob_features(tmp_path))

    def test_unparseable_tier_propagates_valueerror(self, tmp_path):
        _write(tmp_path, "shard_00001", "totally_bogus")
        with pytest.raises(ValueError, match="Unknown L0Tier"):
            assert_uniform_l0_tier(glob_features(tmp_path))

    def test_accepts_a_single_use_generator(self, tmp_path):
        """Pins the iterate-once contract: the helper consumes
        ``shard_paths`` exactly once, so a single-use generator (not just a
        list) works. Guards against a future refactor that re-iterates the
        argument and would silently break generator callers."""
        _write(tmp_path, "shard_00001", "big")
        _write(tmp_path, "shard_00002", "big")
        gen = (p for p in glob_features(tmp_path))  # single-use generator
        assert assert_uniform_l0_tier(gen) is L0Tier.big

    def test_three_distinct_tiers_all_listed(self, tmp_path):
        """More than two distinct tiers: every offender appears in the
        message (not just the first pair)."""
        _write(tmp_path, "shard_00001", "small")
        _write(tmp_path, "shard_00002", "medium")
        _write(tmp_path, "shard_00003", "big")
        with pytest.raises(FeatureDirectoryMismatchError) as ei:
            assert_uniform_l0_tier(glob_features(tmp_path))
        msg = str(ei.value)
        assert "small" in msg and "medium" in msg and "big" in msg
        assert msg.count("l0_tier=") == 3
