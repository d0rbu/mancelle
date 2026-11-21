"""Slice 04 — eager-path ``exclude_last_shard`` honor (map P1 #1).

The streaming path (``optimized_pass1``) drops the final shard when
``exclude_last_shard=True``; the eager path (``load_all_shards``) used to silently
include it. Anyone running ``compute_statistics(streaming=False, …)`` while the
Hydra config defaulted ``exclude_last_shard: true`` saw the validation shard
poison the corpus statistics.

These tests pin the post-fix invariant: the eager path must honor the flag
identically to streaming.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch as t

from steering.compute_statistics import compute_statistics, load_all_shards
from steering.feature_io import save_features


def _write_synthetic_shard(
    tmp_dir: Path,
    shard_idx: int,
    feature_ids: list[int],
    *,
    n_segments: int = 2,
    segment_size: int = 128,
) -> None:
    """Write a small synthetic feature shard with the given feature ids.

    Each shard records ``len(feature_ids)`` features per segment with magnitude 1.0
    so the per-feature ``count`` (occurrence) cleanly identifies which shards were
    aggregated.
    """
    feat_ids: list[int] = []
    feat_mags: list[float] = []
    offsets: list[int] = [0]
    for _seg in range(n_segments):
        feat_ids.extend(feature_ids)
        feat_mags.extend([1.0] * len(feature_ids))
        offsets.append(offsets[-1] + len(feature_ids))

    shard = {
        "feature_ids": t.tensor(feat_ids, dtype=t.int32),
        "feature_mags": t.tensor(feat_mags, dtype=t.float32),
        "offsets": t.tensor(offsets, dtype=t.int64),
        "n_segments": n_segments,
        "n_docs": 1,
        "doc_ids": t.zeros(n_segments, dtype=t.int32),
        "doc_token_counts": t.tensor([n_segments * segment_size], dtype=t.int32),
        "segment_indices": t.arange(n_segments, dtype=t.int32),
        "segment_positions": t.arange(n_segments, dtype=t.int32) * segment_size,
        "segment_mode": "fixed_tokens",
        "segment_size": segment_size,
        "context_window": 1024,
        "pooling": "mean_pool",
        "layer": 8,
    }
    save_features(shard, tmp_dir / f"shard_{shard_idx:05d}_features")


# ---------------------------------------------------------------------------
# Eager path
# ---------------------------------------------------------------------------


class TestEagerExcludeLastShard:
    """Eager path (``streaming=False``) must honor ``exclude_last_shard``."""

    def test_three_shards_excludes_last(self, tmp_path: Path) -> None:
        """Per-feature stats must NOT include features that ONLY appear in the last shard."""
        # Three shards each with disjoint feature ids — any feature
        # that appears in stats unambiguously identifies which shard was aggregated.
        _write_synthetic_shard(tmp_path, 0, feature_ids=[0, 1])
        _write_synthetic_shard(tmp_path, 1, feature_ids=[2, 3])
        _write_synthetic_shard(tmp_path, 2, feature_ids=[4, 5])

        out = tmp_path / "stats"
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(out),
            n_sae_features=8,
            streaming=False,
            exclude_last_shard=True,
        )

        active = set(stats.per_feature_stats.keys())
        # Shards 0 and 1 → features {0, 1, 2, 3}. Shard 2 (excluded) had {4, 5}.
        assert active == {"0", "1", "2", "3"}, (
            f"eager path with exclude_last_shard=True must drop the last shard; "
            f"got active features {sorted(active)}"
        )

    def test_three_shards_includes_last_when_flag_false(self, tmp_path: Path) -> None:
        """``exclude_last_shard=False`` is the historical default and must include all shards."""
        _write_synthetic_shard(tmp_path, 0, feature_ids=[0, 1])
        _write_synthetic_shard(tmp_path, 1, feature_ids=[2, 3])
        _write_synthetic_shard(tmp_path, 2, feature_ids=[4, 5])

        out = tmp_path / "stats"
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(out),
            n_sae_features=8,
            streaming=False,
            exclude_last_shard=False,
        )

        active = set(stats.per_feature_stats.keys())
        assert active == {"0", "1", "2", "3", "4", "5"}, (
            f"eager path with exclude_last_shard=False must include all shards; "
            f"got active features {sorted(active)}"
        )

    def test_n_shards_processed_reflects_exclusion(self, tmp_path: Path) -> None:
        """``CorpusStats.n_shards_processed`` must equal the number of *included* shards."""
        _write_synthetic_shard(tmp_path, 0, feature_ids=[0])
        _write_synthetic_shard(tmp_path, 1, feature_ids=[1])
        _write_synthetic_shard(tmp_path, 2, feature_ids=[2])

        out = tmp_path / "stats"
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(out),
            n_sae_features=4,
            streaming=False,
            exclude_last_shard=True,
        )
        assert stats.corpus.n_shards_processed == 2

        out2 = tmp_path / "stats_all"
        stats_all = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(out2),
            n_sae_features=4,
            streaming=False,
            exclude_last_shard=False,
        )
        assert stats_all.corpus.n_shards_processed == 3

    def test_single_shard_flag_is_noop(self, tmp_path: Path) -> None:
        """``exclude_last_shard=True`` on a single shard MUST keep that shard.

        Matches ``optimized_pass1``'s ``if exclude_last_shard and len(pt_files) > 1``
        guard — when there's only one shard the flag is a no-op so we don't drop the
        entire corpus.
        """
        _write_synthetic_shard(tmp_path, 0, feature_ids=[0, 1])

        out = tmp_path / "stats"
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(out),
            n_sae_features=4,
            streaming=False,
            exclude_last_shard=True,
        )
        assert stats.corpus.n_shards_processed == 1
        assert set(stats.per_feature_stats.keys()) == {"0", "1"}


# ---------------------------------------------------------------------------
# load_all_shards plumbing
# ---------------------------------------------------------------------------


class TestLoadAllShardsExcludeLastShard:
    """``load_all_shards`` accepts the flag and drops the final shard when set."""

    def test_default_includes_all(self, tmp_path: Path) -> None:
        _write_synthetic_shard(tmp_path, 0, feature_ids=[0])
        _write_synthetic_shard(tmp_path, 1, feature_ids=[1])
        _write_synthetic_shard(tmp_path, 2, feature_ids=[2])
        loaded = load_all_shards(str(tmp_path))
        assert loaded.n_shards == 3

    def test_exclude_last(self, tmp_path: Path) -> None:
        _write_synthetic_shard(tmp_path, 0, feature_ids=[0])
        _write_synthetic_shard(tmp_path, 1, feature_ids=[1])
        _write_synthetic_shard(tmp_path, 2, feature_ids=[2])
        loaded = load_all_shards(str(tmp_path), exclude_last_shard=True)
        assert loaded.n_shards == 2

    def test_exclude_last_single_shard_noop(self, tmp_path: Path) -> None:
        _write_synthetic_shard(tmp_path, 0, feature_ids=[0])
        loaded = load_all_shards(str(tmp_path), exclude_last_shard=True)
        # Single-shard corpora keep the only shard.
        assert loaded.n_shards == 1


class TestEagerExcludeLastShardEmptyDir:
    """Empty feature dir surfaces a clear ``FileNotFoundError`` (map §7.2)."""

    def test_empty_dir_raises_filenotfound(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="No feature shard"):
            load_all_shards(str(empty))
