"""Tests for the l0_tier metadata + overwrite guard."""

import dataclasses
import json
import struct
from pathlib import Path

import pytest
import torch as t

from steering.compute_statistics import compute_statistics
from steering.feature_io import (
    MetadataKey,
    assert_l0_tier_compatible,
    load_features,
    read_l0_tier_from_shard,
    save_features,
)
from steering.feature_paths import (
    ActivationType,
    DatasetPhase,
    FeatureDirectoryMismatchError,
    PoolingMode,
    build_feature_dir,
    verify_feature_dir,
)
from steering.sae_resolver import L0Tier, SAEWidth
from steering.stats_io import _parse_l0_tier, load_statistics, save_statistics
from steering.stats_types import ExtractionStatistics


def _sample_shard(l0_tier: str | None = None) -> dict:
    """Build a minimal feature shard dict.

    Carries the full field set ``compute_statistics`` consumes
    (``doc_token_counts``/``doc_ids``/``segment_indices``/
    ``segment_positions``) so the same fixture drives both the raw
    metadata round-trip tests and the end-to-end ``compute_statistics``
    wiring test. ``l0_tier`` is only attached when not ``None`` so the
    absent-stays-absent contract is exercised.
    """
    n_segments = 1
    data: dict = {
        "feature_ids": t.tensor([1, 2, 3], dtype=t.long),
        "feature_mags": t.tensor([0.5, 0.7, 0.3], dtype=t.float32),
        "offsets": t.tensor([0, 3], dtype=t.long),
        "layer": 22,
        "segment_size": 128,
        "context_window": 1024,
        "pooling": "max_pool",
        "segment_mode": "fixed_tokens",
        "n_docs": 1,
        "n_segments": n_segments,
        "doc_ids": t.zeros(n_segments, dtype=t.int32),
        "doc_token_counts": t.tensor([640], dtype=t.int32),
        "segment_indices": t.arange(n_segments, dtype=t.int32),
        "segment_positions": t.arange(n_segments, dtype=t.int32) * 128,
        "segment_char_starts": t.zeros(n_segments, dtype=t.int32),
        "segment_char_ends": t.full((n_segments,), 64, dtype=t.int32),
        "context_independent_features": False,
    }
    if l0_tier is not None:
        data["l0_tier"] = l0_tier
    return data


def _minimal_stats_dict(
    tmp_path: Path, l0_tier: L0Tier | None = None
) -> ExtractionStatistics:
    """Build a minimal ``ExtractionStatistics`` for a stats round-trip.

    The real ``stats_io`` API serializes the typed ``ExtractionStatistics``
    dataclass (not a plain dict), so the minimal fixture is produced via the
    same synthetic-shard + ``compute_statistics`` path used by
    ``tests/steering/test_stats_io.py``. ``l0_tier`` is the ``L0Tier`` enum
    and is only attached when not ``None`` so the absent-stays-absent
    contract is exercised.
    """
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    n_segments = 5
    feat_ids: list[int] = []
    feat_mags: list[float] = []
    offsets = [0]
    for seg in range(n_segments):
        seg_feats = [0, 1, 2]
        seg_mags = [float(seg + 1), float((seg + 1) * 10), float((seg + 1) * 0.1)]
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
    save_features(shard, shard_dir / "shard_0_features")
    stats = compute_statistics(
        feature_dir=str(shard_dir),
        output_path=str(tmp_path / "stats"),
        streaming=False,
    )
    if l0_tier is not None:
        stats = dataclasses.replace(stats, l0_tier=l0_tier)
    return stats


class TestL0TierMetadataWrite:
    def test_l0_tier_round_trips(self, tmp_path):
        out_path = tmp_path / "shard_00001"
        save_features(_sample_shard(l0_tier="medium"), out_path)
        loaded = load_features(tmp_path / "shard_00001_features.safetensors.zst")
        assert loaded[MetadataKey.L0_TIER] == "medium"

    def test_l0_tier_absent_for_v1_shard(self, tmp_path):
        out_path = tmp_path / "shard_00001"
        save_features(_sample_shard(l0_tier=None), out_path)
        loaded = load_features(tmp_path / "shard_00001_features.safetensors.zst")
        # Absent metadata stays absent (not coerced to a default).
        assert MetadataKey.L0_TIER not in loaded


class TestReadL0TierFromShard:
    def test_reads_tier_when_present(self, tmp_path):
        out_path = tmp_path / "shard_00001"
        save_features(_sample_shard(l0_tier="big"), out_path)
        shard_path = tmp_path / "shard_00001_features.safetensors.zst"
        assert read_l0_tier_from_shard(shard_path) == L0Tier.big

    def test_returns_none_when_absent(self, tmp_path):
        out_path = tmp_path / "shard_00001"
        save_features(_sample_shard(l0_tier=None), out_path)
        shard_path = tmp_path / "shard_00001_features.safetensors.zst"
        assert read_l0_tier_from_shard(shard_path) is None

    def test_unknown_tier_raises(self, tmp_path):
        out_path = tmp_path / "shard_00001"
        save_features(_sample_shard(l0_tier="huge"), out_path)
        shard_path = tmp_path / "shard_00001_features.safetensors.zst"
        with pytest.raises(ValueError, match=r"Unknown L0Tier"):
            read_l0_tier_from_shard(shard_path)


class TestAssertL0TierCompatible:
    def test_matching_tier_passes(self, tmp_path):
        save_features(_sample_shard(l0_tier="medium"), tmp_path / "shard_00001")
        # Same tier — must not raise.
        assert_l0_tier_compatible(tmp_path, L0Tier.medium)

    def test_mismatched_tier_raises(self, tmp_path):
        save_features(_sample_shard(l0_tier="medium"), tmp_path / "shard_00001")
        with pytest.raises(FeatureDirectoryMismatchError, match=r"l0_tier.*medium.*big"):
            assert_l0_tier_compatible(tmp_path, L0Tier.big)

    def test_nonexistent_dir_passes(self, tmp_path):
        # Gap 5 (feature_io.py:~538): the COMMON first-run case. The output
        # directory does not exist yet, so the not-a-directory early return
        # must fire and return None without raising — a fresh output dir must
        # never block the first write.
        missing = tmp_path / "does_not_exist_yet"
        assert not missing.exists()
        assert assert_l0_tier_compatible(missing, L0Tier.medium) is None

    def test_empty_dir_passes(self, tmp_path):
        # An empty directory has no shards to conflict — must not raise.
        # (Covers the second early return: dir exists but glob_features is
        # empty — distinct from the not-a-dir branch above.)
        assert_l0_tier_compatible(tmp_path, L0Tier.medium)

    def test_v1_dir_v1_write_passes(self, tmp_path):
        save_features(_sample_shard(l0_tier=None), tmp_path / "shard_00001")
        # New write also v1 (None) — must not raise.
        assert_l0_tier_compatible(tmp_path, None)

    def test_v2_into_v1_dir_raises(self, tmp_path):
        save_features(_sample_shard(l0_tier=None), tmp_path / "shard_00001")
        with pytest.raises(FeatureDirectoryMismatchError, match=r"v1.*v2"):
            assert_l0_tier_compatible(tmp_path, L0Tier.medium)

    def test_v1_into_v2_dir_raises(self, tmp_path):
        save_features(_sample_shard(l0_tier="medium"), tmp_path / "shard_00001")
        with pytest.raises(FeatureDirectoryMismatchError, match=r"v2.*v1"):
            assert_l0_tier_compatible(tmp_path, None)


class TestVerifyFeatureDirL0Consistency:
    def test_mismatched_tier_reported(self, tmp_path):
        feature_dir = build_feature_dir(
            root=tmp_path, segment_size=128, layer=22,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool, width=SAEWidth.W_16K,
            model="gemma-3-4b-it", dataset_phase=DatasetPhase.BASE,
        )
        feature_dir.mkdir(parents=True)
        save_features(_sample_shard(l0_tier="medium"), feature_dir / "shard_00001")
        save_features(_sample_shard(l0_tier="big"),    feature_dir / "shard_00002")
        with pytest.raises(FeatureDirectoryMismatchError) as exc:
            verify_feature_dir(feature_dir)
        # The message must surface BOTH shard names and BOTH tiers.
        assert "shard_00001" in str(exc.value)
        assert "shard_00002" in str(exc.value)
        assert "medium" in str(exc.value) and "big" in str(exc.value)


class TestStatsL0Tier:
    """l0_tier must round-trip through the feature-statistics file.

    ``stats_io`` serializes the typed ``ExtractionStatistics`` dataclass, so
    the round-trip is asserted on ``loaded.l0_tier`` (the dataclass attribute).
    Absent stays absent: a v1 stats file loads back with ``l0_tier is None``.
    """

    def test_stats_round_trip_includes_l0_tier(self, tmp_path):
        stats_path = tmp_path / "feature_statistics.safetensors"
        data = _minimal_stats_dict(tmp_path, l0_tier=L0Tier.medium)
        save_statistics(data, stats_path)
        loaded = load_statistics(stats_path)
        # The enum round-trips as the enum itself, not the bare string.
        assert loaded.l0_tier is L0Tier.medium

    def test_stats_round_trip_with_no_l0_tier(self, tmp_path):
        stats_path = tmp_path / "feature_statistics.safetensors"
        data = _minimal_stats_dict(tmp_path, l0_tier=None)
        save_statistics(data, stats_path)
        loaded = load_statistics(stats_path)
        # Absent metadata stays absent (not coerced to a default).
        assert loaded.l0_tier is None

    def test_unknown_l0_tier_in_stats_metadata_raises(self):
        # Mirrors feature_io.read_l0_tier_from_shard's contract: a present
        # but unrecognized tier must raise a clear ValueError listing the
        # valid tiers.
        with pytest.raises(ValueError, match=r"Unknown L0Tier.*'bogus'"):
            _parse_l0_tier("bogus")

    def test_parse_l0_tier_absent_returns_none(self):
        assert _parse_l0_tier(None) is None

    def test_parse_l0_tier_known_returns_enum(self):
        assert _parse_l0_tier("medium") is L0Tier.medium

    def test_stats_round_trip_with_bogus_on_disk_metadata_raises(
        self, tmp_path
    ):
        # End-to-end: hand-corrupt the on-disk safetensors metadata so the
        # READ path (load_statistics) is what surfaces the ValueError.
        stats_path = tmp_path / "feature_statistics.safetensors"
        data = _minimal_stats_dict(tmp_path, l0_tier=L0Tier.medium)
        save_statistics(data, stats_path)

        raw = stats_path.read_bytes()
        header_size = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8 : 8 + header_size])
        header["__metadata__"]["l0_tier"] = "bogus"
        new_header = json.dumps(header).encode("utf-8")
        stats_path.write_bytes(
            struct.pack("<Q", len(new_header))
            + new_header
            + raw[8 + header_size :]
        )

        with pytest.raises(ValueError, match=r"Unknown L0Tier.*'bogus'"):
            load_statistics(stats_path)


class TestComputeStatisticsPopulatesL0Tier:
    def _stamp_dir(self, tmp_path, tiers):
        d = tmp_path / "shards"
        d.mkdir(parents=True, exist_ok=True)
        for i, tier in enumerate(tiers):
            save_features(_sample_shard(l0_tier=tier), d / f"shard_{i:05d}_features")
        return d

    def test_v2_shards_stamp_stats_l0_tier(self, tmp_path):
        d = self._stamp_dir(tmp_path, ["medium", "medium"])
        stats = compute_statistics(
            feature_dir=str(d),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )
        assert stats.l0_tier is L0Tier.medium
        sf_path = save_statistics(stats, tmp_path / "stats")
        assert load_statistics(sf_path).l0_tier is L0Tier.medium

    def test_v1_shards_leave_l0_tier_none(self, tmp_path):
        d = self._stamp_dir(tmp_path, [None, None])
        stats = compute_statistics(
            feature_dir=str(d),
            output_path=str(tmp_path / "stats"),
            streaming=False,
        )
        assert stats.l0_tier is None
        sf_path = save_statistics(stats, tmp_path / "stats")
        assert load_statistics(sf_path).l0_tier is None

    def test_mixed_shards_raise_fail_fast(self, tmp_path):
        d = self._stamp_dir(tmp_path, ["medium", "big"])
        with pytest.raises(FeatureDirectoryMismatchError):
            compute_statistics(
                feature_dir=str(d),
                output_path=str(tmp_path / "stats"),
                streaming=False,
            )

    @pytest.mark.parametrize("streaming", [False, True])
    def test_mixed_shards_raise_before_body_load(
        self, tmp_path, monkeypatch, streaming
    ):
        d = self._stamp_dir(tmp_path, ["medium", "big"])

        def _fail_body_load(*_args, **_kwargs):
            raise AssertionError("compute_statistics loaded a shard body before fail-fast")

        monkeypatch.setattr("steering.compute_statistics.load_features", _fail_body_load)
        with pytest.raises(FeatureDirectoryMismatchError):
            compute_statistics(
                feature_dir=str(d),
                output_path=str(tmp_path / "stats"),
                streaming=streaming,
                exclude_last_shard=False,
            )

    @pytest.mark.parametrize("streaming", [False, True])
    def test_exclude_last_shard_divergent_excluded_shard_does_not_fail(
        self, tmp_path, streaming
    ):
        """The l0-consistency domain must match the STATS domain: with
        exclude_last_shard, a divergent *excluded* validation shard must
        NOT hard-fail the run, and the recorded tier reflects the included
        shards. Regression-locks the exclude_last_shard fix on BOTH the
        streaming (optimized_pass1) and eager (load_all_shards) paths."""
        d = self._stamp_dir(tmp_path, ["medium", "medium", "big"])
        stats = compute_statistics(
            feature_dir=str(d),
            output_path=str(tmp_path / "stats"),
            streaming=streaming,
            exclude_last_shard=True,
        )
        assert stats.l0_tier is L0Tier.medium
        sf_path = save_statistics(stats, tmp_path / "stats")
        assert load_statistics(sf_path).l0_tier is L0Tier.medium

    def test_exclude_last_shard_divergent_included_shard_still_fails(
        self, tmp_path
    ):
        """The fix must NOT weaken real-corruption detection: a divergent
        tier among the INCLUDED shards still fails fast even with
        exclude_last_shard (here the excluded last shard is uniform, so
        the mix is genuinely inside the stats domain)."""
        d = self._stamp_dir(tmp_path, ["medium", "big", "medium"])
        with pytest.raises(FeatureDirectoryMismatchError):
            compute_statistics(
                feature_dir=str(d),
                output_path=str(tmp_path / "stats"),
                streaming=False,
                exclude_last_shard=True,
            )
