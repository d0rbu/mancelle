"""Issue #521: SFTFeatureStore fails loud on a corrupt/missing feature row."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

from errors import DataLoadError
from steering.conditional.encoder_decoder.sft_features import (
    SFTFeatureStore,
    load_sft_features,
)
from steering.feature_io import save_features


def _write_feature_shard(feature_dir: Path, stem: str = "shard_00000") -> None:
    """Write one minimal real feature shard for loader fail-fast tests."""
    save_features(
        {
            "offsets": t.tensor([0, 1], dtype=t.int64),
            "feature_ids": t.tensor([10], dtype=t.int32),
            "feature_mags": t.tensor([1.0], dtype=t.float32),
            "doc_ids": t.tensor([0], dtype=t.int32),
            "segment_indices": t.tensor([0], dtype=t.int32),
            "segment_positions": t.tensor([0], dtype=t.int32),
            "doc_token_counts": t.tensor([256], dtype=t.int32),
            "segment_char_starts": t.tensor([0], dtype=t.int32),
            "segment_char_ends": t.tensor([768], dtype=t.int32),
            "n_docs": 1,
            "n_segments": 1,
            "layer": 12,
            "context_window": 1024,
            "pooling": "mean_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 128,
        },
        feature_dir / f"{stem}_features",
    )


class _StubIndex:
    """FeatureIndex double whose lookup mirrors the real raise-contract."""

    def __init__(self, present: dict[tuple[str, int], list]):
        self._present = present

    def lookup(self, shard_stem, doc_idx):
        key = (shard_stem, doc_idx)
        if key not in self._present:
            raise KeyError(
                f"Document {doc_idx} not found in shard {shard_stem!r} (shard has 0 docs)"
            )
        return self._present[key]


def _store(present, locations):
    store = SFTFeatureStore.__new__(SFTFeatureStore)
    store._feature_index = _StubIndex(present)  # ty: ignore[invalid-assignment]
    store._locations = locations
    return store


def test_getitem_present_returns_segments():
    segs = [object()]
    store = _store({("shard_00000", 3): segs}, {7: ("shard_00000", 3)})
    assert store[7] is segs


def test_getitem_corrupt_row_raises_keyerror_naming_shard_and_doc():
    # key 7 IS in _locations, but its (shard, doc) is absent from the index → corruption.
    store = _store({}, {7: ("shard_00000", 3)})
    with pytest.raises(KeyError, match=r"Document 3 not found in shard 'shard_00000'"):
        store[7]


def test_getitem_unknown_key_raises_plain_keyerror():
    store = _store({}, {7: ("shard_00000", 3)})
    with pytest.raises(KeyError):
        store[999]  # not in _locations → _locations[key] raises KeyError(999)


def test_get_swallows_keyerror_documenting_why_callers_must_subscript():
    """Mapping.get() catches KeyError -> returns default. This documents WHY the
    conditioned-SFT runtime must subscript (store[idx]), not store.get(idx)."""
    store = _store({}, {7: ("shard_00000", 3)})
    assert store.get(7) is None  # .get() swallows -> None (the trap)
    with pytest.raises(KeyError):
        _ = store[7]  # subscript -> loud (the fix)


def test_empty_parquet_dir_raises_a_path_dataloaderror(tmp_path: Path):
    """An empty parquet dir means "check the path", not "regenerate".

    Reporting it through the stem set-difference ("orphan feature shards …
    regenerate the matched artifact family") prescribes a multi-hour
    conversion for what is normally a typo'd or unmounted ``sft_parquet_dir``
    (review finding 9, PR #1311).
    """
    feature_dir = tmp_path / "features"
    parquet_dir = tmp_path / "parquet"
    feature_dir.mkdir()
    parquet_dir.mkdir()
    _write_feature_shard(feature_dir)

    with pytest.raises(
        DataLoadError, match=r"contains no \*\.parquet shards.*check sft_parquet_dir"
    ):
        load_sft_features(str(feature_dir), str(parquet_dir))


def test_partially_missing_parquet_shard_still_reports_divergence(tmp_path: Path):
    """A genuine per-shard divergence still gets the regeneration message."""
    feature_dir = tmp_path / "features"
    parquet_dir = tmp_path / "parquet"
    feature_dir.mkdir()
    parquet_dir.mkdir()
    _write_feature_shard(feature_dir, stem="shard_00000")
    _write_feature_shard(feature_dir, stem="shard_00001")
    pq.write_table(
        pa.table({"original_index": pa.array([0], type=pa.int64())}),
        parquet_dir / "shard_00000.parquet",
    )

    with pytest.raises(DataLoadError, match=r"diverged.*shard_00001"):
        load_sft_features(str(feature_dir), str(parquet_dir))


def test_missing_original_index_raises_dataloaderror(tmp_path: Path):
    feature_dir = tmp_path / "features"
    parquet_dir = tmp_path / "parquet"
    feature_dir.mkdir()
    parquet_dir.mkdir()
    _write_feature_shard(feature_dir)
    pq.write_table(
        pa.table({"text": pa.array(["doc-0"])}),
        parquet_dir / "shard_00000.parquet",
    )

    with pytest.raises(DataLoadError, match=r"original_index"):
        load_sft_features(str(feature_dir), str(parquet_dir))
