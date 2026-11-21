"""Tests for the train/val split ordering invariant used by precompute_packed_rows.

Critical invariant
------------------
When ``max_shards`` is set, the truncation ``parquet_paths[:max_shards]``
MUST happen BEFORE the ``[:-1]`` val-exclusion split.  If the order were
reversed the val shard would be excluded first and ``max_shards`` would
silently include a different number of training shards than intended.

The canonical code lives in ``scripts/precompute_packed_rows.py``::

    if max_shards is not None:
        parquet_paths = parquet_paths[:max_shards]
    parquet_paths = parquet_paths[:-1]   # training split (exclude last = val)

These tests exercise the pure list-slicing logic (no live dataloader needed).
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nanochat.dataset import list_parquet_files


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_parquet_shard(data_dir: Path, stem: str, n_docs: int = 5) -> Path:
    texts = [f"doc {i}" for i in range(n_docs)]
    path = data_dir / f"{stem}.parquet"
    pq.write_table(pa.table({"text": texts}), str(path))
    return path


def _make_multi_shard_env(tmp_path: Path, n_shards: int = 10) -> str:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i in range(n_shards):
        _create_parquet_shard(data_dir, f"shard_{i:05d}")
    return str(data_dir)


def _apply_split(all_paths: list, max_shards: int | None = None):
    """Replicate the precompute script's split logic."""
    if max_shards is not None:
        all_paths = all_paths[:max_shards]
    train_paths = all_paths[:-1]
    val_paths = all_paths[-1:]
    return train_paths, val_paths


# ---------------------------------------------------------------------------
# Category 2 – Precompute split ordering
# ---------------------------------------------------------------------------

class TestPrecomputeSplitOrdering:
    """Verify that max_shards truncation happens before val exclusion."""

    def test_precompute_excludes_val_shard(self, tmp_path):
        """With 10 shards and max_shards=5, train should use shards 0-3
        and val should be shard 4."""
        data_dir = _make_multi_shard_env(tmp_path, n_shards=10)
        all_paths = list_parquet_files(data_dir)

        train_paths, val_paths = _apply_split(all_paths, max_shards=5)

        assert len(train_paths) == 4
        assert len(val_paths) == 1
        # Train shards are the first 4 of the original 10
        for i, p in enumerate(train_paths):
            assert p == all_paths[i]
        # Val shard is the 5th (index 4)
        assert val_paths[0] == all_paths[4]

    @pytest.mark.parametrize("max_shards", [3, 5, 7, 10])
    def test_precompute_val_consistency(self, tmp_path, max_shards):
        """For various max_shards values, train should be [0:max_shards-1]
        and val should be the shard at index max_shards-1."""
        data_dir = _make_multi_shard_env(tmp_path, n_shards=10)
        all_paths = list_parquet_files(data_dir)

        train_paths, val_paths = _apply_split(all_paths, max_shards=max_shards)

        assert len(train_paths) == max_shards - 1
        assert len(val_paths) == 1
        assert train_paths == all_paths[: max_shards - 1]
        assert val_paths[0] == all_paths[max_shards - 1]


# ---------------------------------------------------------------------------
# Category 3 – Edge cases (pure list slicing)
# ---------------------------------------------------------------------------

class TestMaxShardsEdgeCases:
    """Edge-case behaviour for max_shards truncation + val split."""

    def test_max_shards_one_train_empty_val_single(self, tmp_path):
        """max_shards=1 should yield an empty train set and exactly one val
        shard (the first shard)."""
        data_dir = _make_multi_shard_env(tmp_path, n_shards=10)
        all_paths = list_parquet_files(data_dir)

        train_paths, val_paths = _apply_split(all_paths, max_shards=1)

        assert train_paths == []
        assert len(val_paths) == 1
        assert val_paths[0] == all_paths[0]

    def test_max_shards_two_gives_one_train_one_val(self, tmp_path):
        """max_shards=2 should give exactly 1 training shard and 1 val
        shard."""
        data_dir = _make_multi_shard_env(tmp_path, n_shards=10)
        all_paths = list_parquet_files(data_dir)

        train_paths, val_paths = _apply_split(all_paths, max_shards=2)

        assert len(train_paths) == 1
        assert len(val_paths) == 1
        assert train_paths[0] == all_paths[0]
        assert val_paths[0] == all_paths[1]

    def test_max_shards_exceeds_available(self, tmp_path):
        """When max_shards exceeds the number of available shards, all
        shards should be used (no error, just a no-op truncation)."""
        n_shards = 3
        data_dir = _make_multi_shard_env(tmp_path, n_shards=n_shards)
        all_paths = list_parquet_files(data_dir)

        train_paths, val_paths = _apply_split(all_paths, max_shards=100)

        assert len(train_paths) == n_shards - 1
        assert len(val_paths) == 1
        assert train_paths == all_paths[:-1]
        assert val_paths[0] == all_paths[-1]
