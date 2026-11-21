"""Coverage for the lazy SFT feature store.

Targets ``steering.conditional.encoder_decoder.sft_features`` — the loader that
maps an SFT-mixture index to its pre-extracted segment features via a parquet
``original_index`` column plus on-disk feature shards.

All fixtures are *real* on-disk artifacts: feature shards written with
``steering.feature_io.save_features`` (the exact format the extractor emits) and
parquet shards carrying an ``original_index`` column.  ``load_sft_features`` is
exercised for real — never patched — so the parquet↔shard wiring, the duplicate
guard, the weakref cache, and the lazy lookup are all genuinely tested.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from safetensors.torch import save_file

from errors import DataLoadError
from steering.conditional.encoder_decoder import sft_features as sft_features_module
from steering.conditional.encoder_decoder.sft_features import (
    SFTFeatureStore,
    load_sft_features,
)
from steering.conditional.feature_index import SegmentFeatures
from steering.feature_io import save_features
from steering.segmentation import SegmentMode
from steering.sft import mixture_fingerprint as mixture_fingerprint_module
from steering.sft_schema import SFTSplit

# ---------------------------------------------------------------------------
# Real on-disk fixtures
# ---------------------------------------------------------------------------


def _write_shard(
    feature_dir: Path,
    stem: str,
    *,
    doc_ids: list[int],
    feats_per_seg: int = 2,
    fid_base: int = 10,
    metadata: dict[str, str] | None = None,
    include_n_docs: bool = True,
) -> dict[int, list[int]]:
    """Write one real feature shard with ``len(doc_ids)`` single-segment docs.

    Returns a ``{doc_id: [feature_ids...]}`` map so callers can assert the
    round-trip recovers the exact ids.  Uses the same dict schema
    ``save_features`` consumes in production (and that
    ``tests/steering/test_feature_index.py`` builds its synthetic shards with).
    """
    n = len(doc_ids)
    offsets = list(range(0, n * feats_per_seg + 1, feats_per_seg))
    feature_ids: list[int] = []
    feature_mags: list[float] = []
    expected: dict[int, list[int]] = {}
    for seg_i, doc_id in enumerate(doc_ids):
        fids = [fid_base + seg_i * feats_per_seg + j for j in range(feats_per_seg)]
        feature_ids.extend(fids)
        feature_mags.extend(float(feats_per_seg - j) for j in range(feats_per_seg))
        expected[doc_id] = fids
    shard = {
        "offsets": t.tensor(offsets, dtype=t.int64),
        "feature_ids": t.tensor(feature_ids, dtype=t.int32),
        "feature_mags": t.tensor(feature_mags, dtype=t.float32),
        "doc_ids": t.tensor(doc_ids, dtype=t.int32),
        "segment_indices": t.tensor([0] * n, dtype=t.int32),
        "segment_positions": t.tensor([0] * n, dtype=t.int32),
        "doc_token_counts": t.tensor([256] * (max(doc_ids) + 1), dtype=t.int32),
        "segment_char_starts": t.tensor([0] * n, dtype=t.int32),
        "segment_char_ends": t.tensor([768] * n, dtype=t.int32),
        "n_segments": n,
        "layer": 12,
        "context_window": 1024,
        "pooling": "mean_pool",
        "segment_size": 128,
    }
    if include_n_docs:
        shard["n_docs"] = max(doc_ids) + 1
    if metadata is not None:
        shard.update(metadata)
    save_features(shard, feature_dir / f"{stem}_features")
    return expected


def _write_statistics_mode(feature_dir: Path, segment_mode: SegmentMode | str) -> None:
    save_file(
        {"placeholder": t.zeros(1)},
        feature_dir / "feature_statistics.safetensors",
        metadata={
            "segment_mode": (
                segment_mode.value
                if isinstance(segment_mode, SegmentMode)
                else segment_mode
            )
        },
    )


def _write_parquet(parquet_dir: Path, stem: str, original_indices: list[int]) -> None:
    """Write a parquet shard with an ``original_index`` column (+ a text column,
    mirroring the real SFT parquet layout)."""
    table = pa.table(
        {
            "original_index": pa.array(original_indices, type=pa.int64()),
            "text": pa.array([f"doc-{i}" for i in original_indices]),
        }
    )
    pq.write_table(table, parquet_dir / f"{stem}.parquet")


@pytest.fixture()
def single_shard_store(tmp_path: Path):
    """Build one real shard + parquet and return ``(store, expected, dirs)``.

    ``original_index`` maps local doc 0 -> 100 and local doc 1 -> 200, so the
    store keys are the mixture indices (100, 200) while the shard's own doc_ids
    are (0, 1).  This distinction is the whole point of the loader.
    """
    feature_dir = tmp_path / "features"
    parquet_dir = tmp_path / "parquet"
    feature_dir.mkdir()
    parquet_dir.mkdir()

    doc_fids = _write_shard(
        feature_dir, "shard_00000", doc_ids=[0, 1], feats_per_seg=2, fid_base=10
    )
    _write_parquet(parquet_dir, "shard_00000", original_indices=[100, 200])

    store = load_sft_features(str(feature_dir), str(parquet_dir))
    # expected keyed by mixture idx: local doc 0 (fids 10,11) -> 100, etc.
    expected = {100: doc_fids[0], 200: doc_fids[1]}
    return store, expected, (feature_dir, parquet_dir)


@pytest.fixture()
def feature_parquet_dirs(tmp_path: Path) -> tuple[Path, Path]:
    feature_dir = tmp_path / "features"
    parquet_dir = tmp_path / "parquet"
    feature_dir.mkdir()
    parquet_dir.mkdir()
    return feature_dir, parquet_dir


# ---------------------------------------------------------------------------
# Happy-path round trip
# ---------------------------------------------------------------------------


class TestLoadSftFeaturesRoundTrip:
    """A built store maps mixture indices to the right ``SegmentFeatures``."""

    def test_keys_are_mixture_indices(self, single_shard_store) -> None:
        store, _expected, _dirs = single_shard_store
        assert sorted(store.keys()) == [100, 200]
        assert len(store) == 2
        assert 100 in store
        assert 999 not in store

    @pytest.mark.parametrize(
        ("shard_metadata", "statistics_mode", "expected_mode"),
        [
            pytest.param(None, None, SegmentMode.fixed_tokens, id="absent"),
            pytest.param(
                {"segment_mode": SegmentMode.fixed_tokens.value},
                None,
                SegmentMode.fixed_tokens,
                id="explicit_fixed",
            ),
            pytest.param(
                {"segment_mode": SegmentMode.turn_delimited_fixed_tokens.value},
                SegmentMode.turn_delimited_fixed_tokens,
                SegmentMode.turn_delimited_fixed_tokens,
                id="turn",
            ),
        ],
    )
    def test_store_exposes_shard_segment_mode(
        self,
        feature_parquet_dirs: tuple[Path, Path],
        shard_metadata: dict[str, str] | None,
        statistics_mode: SegmentMode | None,
        expected_mode: SegmentMode,
    ) -> None:
        feature_dir, parquet_dir = feature_parquet_dirs
        _write_shard(
            feature_dir,
            "shard_00000",
            doc_ids=[0],
            metadata=shard_metadata,
        )
        if statistics_mode is not None:
            _write_statistics_mode(feature_dir, statistics_mode)
        _write_parquet(parquet_dir, "shard_00000", original_indices=[0])

        store = load_sft_features(str(feature_dir), str(parquet_dir))

        assert store.segment_mode is expected_mode

    def test_mixed_shard_segment_modes_name_disagreeing_shards(
        self,
        feature_parquet_dirs: tuple[Path, Path],
    ) -> None:
        feature_dir, parquet_dir = feature_parquet_dirs
        _write_shard(feature_dir, "shard_00000", doc_ids=[0])
        _write_shard(
            feature_dir,
            "shard_00001",
            doc_ids=[0],
            metadata={"segment_mode": SegmentMode.turn_delimited_fixed_tokens.value},
        )
        _write_parquet(parquet_dir, "shard_00000", original_indices=[0])
        _write_parquet(parquet_dir, "shard_00001", original_indices=[1])

        with pytest.raises(
            DataLoadError,
            match=r"segment_mode differs.*shard_00000.*shard_00001",
        ):
            load_sft_features(str(feature_dir), str(parquet_dir))

    def test_unknown_shard_segment_mode_names_shard(
        self,
        feature_parquet_dirs: tuple[Path, Path],
    ) -> None:
        feature_dir, parquet_dir = feature_parquet_dirs
        _write_shard(
            feature_dir,
            "shard_00000",
            doc_ids=[0],
            metadata={"segment_mode": "unknown_mode"},
        )
        _write_parquet(parquet_dir, "shard_00000", original_indices=[0])

        with pytest.raises(
            DataLoadError,
            match=r"shard_00000.*unknown segment_mode.*unknown_mode",
        ):
            load_sft_features(str(feature_dir), str(parquet_dir))

    @pytest.mark.parametrize(
        ("shard_mode", "statistics_mode"),
        [
            (SegmentMode.turn_delimited_fixed_tokens, SegmentMode.fixed_tokens),
            (SegmentMode.fixed_tokens, SegmentMode.turn_delimited_fixed_tokens),
        ],
    )
    def test_shard_statistics_segment_mode_disagreement_raises(
        self,
        feature_parquet_dirs: tuple[Path, Path],
        shard_mode: SegmentMode,
        statistics_mode: SegmentMode,
    ) -> None:
        feature_dir, parquet_dir = feature_parquet_dirs
        _write_shard(
            feature_dir,
            "shard_00000",
            doc_ids=[0],
            metadata={"segment_mode": shard_mode.value},
        )
        _write_statistics_mode(feature_dir, statistics_mode)
        _write_parquet(parquet_dir, "shard_00000", original_indices=[0])

        with pytest.raises(
            DataLoadError,
            match=(
                rf"feature shards record '{shard_mode.value}'.*"
                rf"statistics record '{statistics_mode.value}'"
            ),
        ):
            load_sft_features(str(feature_dir), str(parquet_dir))

    def test_unknown_statistics_segment_mode_names_statistics(
        self,
        feature_parquet_dirs: tuple[Path, Path],
    ) -> None:
        feature_dir, parquet_dir = feature_parquet_dirs
        _write_shard(feature_dir, "shard_00000", doc_ids=[0])
        _write_statistics_mode(feature_dir, "unknown_mode")
        _write_parquet(parquet_dir, "shard_00000", original_indices=[0])

        with pytest.raises(
            DataLoadError,
            match=r"statistics metadata.*unknown segment_mode.*unknown_mode",
        ):
            load_sft_features(str(feature_dir), str(parquet_dir))

    def test_getitem_returns_correct_segment_features(self, single_shard_store) -> None:
        store, expected, _dirs = single_shard_store
        for mixture_idx, fids in expected.items():
            segments = store[mixture_idx]
            assert isinstance(segments, list)
            assert len(segments) == 1
            seg = segments[0]
            assert isinstance(seg, SegmentFeatures)
            assert seg.feature_ids.tolist() == fids, (
                f"mixture idx {mixture_idx}: expected {fids}, got {seg.feature_ids.tolist()}"
            )
            assert seg.char_start == 0
            assert seg.char_end == 768

    def test_multi_segment_doc_round_trips(self, tmp_path: Path) -> None:
        """A doc with two segments returns both, looked up by its single
        parquet row.

        ``doc_ids=[0, 0]`` encodes ONE document (doc id 0) carrying two
        segments; the parquet has a single row, so local doc 0 maps to mixture
        index 55 and ``store[55]`` must return both segments in order.
        """
        feature_dir = tmp_path / "features"
        parquet_dir = tmp_path / "parquet"
        feature_dir.mkdir()
        parquet_dir.mkdir()
        # One doc (id 0) with two segments; fids [10,11] then [12,13].
        _write_shard(feature_dir, "shard_00000", doc_ids=[0, 0], fid_base=10)
        _write_parquet(parquet_dir, "shard_00000", original_indices=[55])

        store = load_sft_features(str(feature_dir), str(parquet_dir))
        assert sorted(store.keys()) == [55]
        segments = store[55]
        assert len(segments) == 2
        assert [s.feature_ids.tolist() for s in segments] == [[10, 11], [12, 13]]

    def test_conversion_identity_is_carried_by_store(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "features"
        parquet_dir = tmp_path / "parquet"
        feature_dir.mkdir()
        parquet_dir.mkdir()
        _write_shard(feature_dir, "shard_00000", doc_ids=[0])
        _write_parquet(parquet_dir, "shard_00000", original_indices=[0])
        (parquet_dir / "conversion_meta.json").write_text(
            json.dumps(
                {
                    "mixture": "safety",
                    "split": "val",
                    "format": "full_nanochat_text",
                    "completed_shards": 1,
                    "total_shards": 1,
                    "total_conversations": 1,
                }
            )
        )

        store = load_sft_features(str(feature_dir), str(parquet_dir))

        assert store.conversion_mixture == "safety"
        assert store.conversion_split is SFTSplit.VAL
        assert store.conversion_total_conversations == 1

    def test_malformed_conversion_split_fails_loudly(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "features"
        parquet_dir = tmp_path / "parquet"
        feature_dir.mkdir()
        parquet_dir.mkdir()
        _write_shard(feature_dir, "shard_00000", doc_ids=[0])
        _write_parquet(parquet_dir, "shard_00000", original_indices=[0])
        (parquet_dir / "conversion_meta.json").write_text(
            json.dumps(
                {
                    "mixture": "safety",
                    "split": "bogus",
                    "format": "full_nanochat_text",
                    "completed_shards": 1,
                    "total_shards": 1,
                    "total_conversations": 1,
                }
            )
        )

        with pytest.raises(ValueError, match=r"Invalid SFT split 'bogus'.*'train'.*'val'"):
            load_sft_features(str(feature_dir), str(parquet_dir))

    def test_absent_conversion_meta_is_exposed_as_none(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "features"
        parquet_dir = tmp_path / "parquet"
        feature_dir.mkdir()
        parquet_dir.mkdir()
        _write_shard(feature_dir, "shard_00000", doc_ids=[0])
        _write_parquet(parquet_dir, "shard_00000", original_indices=[0])

        store = load_sft_features(str(feature_dir), str(parquet_dir))

        assert store.conversion_mixture is None
        assert store.conversion_split is None
        assert store.conversion_total_conversations is None

    def test_feature_and_parquet_document_counts_must_match(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "features"
        parquet_dir = tmp_path / "parquet"
        feature_dir.mkdir()
        parquet_dir.mkdir()
        _write_shard(feature_dir, "shard_00000", doc_ids=[0])
        _write_parquet(parquet_dir, "shard_00000", original_indices=[0, 1])

        with pytest.raises(
            DataLoadError,
            match=r"declares n_docs=1, but parquet shard .* contains 2",
        ):
            load_sft_features(str(feature_dir), str(parquet_dir))

    def test_missing_feature_shard_for_parquet_is_rejected(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "features"
        parquet_dir = tmp_path / "parquet"
        feature_dir.mkdir()
        parquet_dir.mkdir()
        _write_shard(feature_dir, "shard_00000", doc_ids=[0])
        _write_parquet(parquet_dir, "shard_00000", original_indices=[0])
        _write_parquet(parquet_dir, "shard_00001", original_indices=[1])

        with pytest.raises(DataLoadError, match=r"missing feature shards.*shard_00001"):
            load_sft_features(str(feature_dir), str(parquet_dir))

    def test_parquet_original_indices_are_exposed_as_the_reference_set(
        self, tmp_path: Path
    ) -> None:
        """The store must publish the parquet's own original_index set."""
        feature_dir = tmp_path / "features"
        parquet_dir = tmp_path / "parquet"
        feature_dir.mkdir()
        parquet_dir.mkdir()
        _write_shard(feature_dir, "shard_00000", doc_ids=[0, 1])
        _write_parquet(parquet_dir, "shard_00000", original_indices=[7, 9])
        _write_shard(feature_dir, "shard_00001", doc_ids=[0])
        _write_parquet(parquet_dir, "shard_00001", original_indices=[4])

        store = load_sft_features(str(feature_dir), str(parquet_dir))

        assert store.parquet_original_indices == frozenset({4, 7, 9})
        assert isinstance(store.parquet_original_indices, frozenset)

    def test_missing_parquet_dir_says_check_the_path(self, tmp_path: Path) -> None:
        """A typo'd/unmounted parquet dir is a path fix, not a regeneration.

        Reporting it through the symmetric stem set-difference would say
        "orphan feature shards … regenerate the matched artifact family" and
        send the operator off to redo a multi-hour conversion (finding 9).
        """
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _write_shard(feature_dir, "shard_00000", doc_ids=[0])

        with pytest.raises(DataLoadError, match=r"does not exist.*check sft_parquet_dir"):
            load_sft_features(str(feature_dir), str(tmp_path / "typo_parquet"))

    def test_empty_parquet_dir_says_check_the_path(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "features"
        parquet_dir = tmp_path / "parquet"
        feature_dir.mkdir()
        parquet_dir.mkdir()
        _write_shard(feature_dir, "shard_00000", doc_ids=[0])

        with pytest.raises(
            DataLoadError,
            match=r"contains no \*\.parquet shards.*check sft_parquet_dir",
        ):
            load_sft_features(str(feature_dir), str(parquet_dir))

    def test_no_feature_shards_for_parquet_is_rejected(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "features"
        parquet_dir = tmp_path / "parquet"
        feature_dir.mkdir()
        parquet_dir.mkdir()
        _write_parquet(parquet_dir, "shard_00000", original_indices=[0])

        with pytest.raises(DataLoadError, match=r"No feature shard files found"):
            load_sft_features(str(feature_dir), str(parquet_dir))

    def test_orphan_feature_shard_is_rejected(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "features"
        parquet_dir = tmp_path / "parquet"
        feature_dir.mkdir()
        parquet_dir.mkdir()
        _write_shard(feature_dir, "shard_00000", doc_ids=[0])
        _write_shard(feature_dir, "shard_00001", doc_ids=[0])
        _write_parquet(parquet_dir, "shard_00000", original_indices=[0])

        with pytest.raises(DataLoadError, match=r"orphan feature shards.*shard_00001"):
            load_sft_features(str(feature_dir), str(parquet_dir))

    def test_orphan_feature_shard_precedes_missing_n_docs(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "features"
        parquet_dir = tmp_path / "parquet"
        feature_dir.mkdir()
        parquet_dir.mkdir()
        _write_shard(feature_dir, "shard_00000", doc_ids=[0])
        _write_shard(
            feature_dir,
            "shard_00001",
            doc_ids=[0],
            include_n_docs=False,
        )
        _write_parquet(parquet_dir, "shard_00000", original_indices=[0])

        with pytest.raises(DataLoadError, match=r"orphan feature shards.*shard_00001"):
            load_sft_features(str(feature_dir), str(parquet_dir))

    def test_missing_n_docs_metadata_is_rejected(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "features"
        parquet_dir = tmp_path / "parquet"
        feature_dir.mkdir()
        parquet_dir.mkdir()
        _write_shard(feature_dir, "shard_00000", doc_ids=[0], include_n_docs=False)
        _write_parquet(parquet_dir, "shard_00000", original_indices=[0])

        with pytest.raises(DataLoadError, match=r"missing required n_docs metadata"):
            load_sft_features(str(feature_dir), str(parquet_dir))

    def test_incomplete_conversion_is_rejected(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "features"
        parquet_dir = tmp_path / "parquet"
        feature_dir.mkdir()
        parquet_dir.mkdir()
        _write_shard(feature_dir, "shard_00000", doc_ids=[0])
        _write_parquet(parquet_dir, "shard_00000", original_indices=[0])
        (parquet_dir / "conversion_meta.json").write_text(
            json.dumps(
                {
                    "mixture": "safety",
                    "split": "train",
                    "format": "full_nanochat_text",
                    "completed_shards": 1,
                    "total_shards": 2,
                    "total_conversations": 2,
                }
            )
        )

        with pytest.raises(DataLoadError, match=r"conversion .* incomplete"):
            load_sft_features(str(feature_dir), str(parquet_dir))


# ---------------------------------------------------------------------------
# Duplicate original_index across shards
# ---------------------------------------------------------------------------


class TestDuplicateOriginalIndex:
    """Two parquet rows mapping to the same ``original_index`` must raise."""

    def test_duplicate_across_two_shards_raises(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "features"
        parquet_dir = tmp_path / "parquet"
        feature_dir.mkdir()
        parquet_dir.mkdir()

        _write_shard(feature_dir, "shard_00000", doc_ids=[0], fid_base=10)
        _write_shard(feature_dir, "shard_00001", doc_ids=[0], fid_base=50)
        # Both shards' local doc 0 claim mixture index 42.
        _write_parquet(parquet_dir, "shard_00000", original_indices=[42])
        _write_parquet(parquet_dir, "shard_00001", original_indices=[42])

        with pytest.raises(
            ValueError,
            match=r"Duplicate original_index=42 across SFT parquet shards",
        ):
            load_sft_features(str(feature_dir), str(parquet_dir))


# ---------------------------------------------------------------------------
# __getitem__ KeyError when the index lacks a doc
# ---------------------------------------------------------------------------


class TestStoreGetItemMissingDoc:
    """``SFTFeatureStore.__getitem__`` raises ``KeyError`` when the shard has
    no segments for the located doc id."""

    def test_getitem_missing_doc_raises_key_error(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _write_shard(feature_dir, "shard_00000", doc_ids=[0])

        # Hand-build a store whose location for key 7 points at a doc id (999)
        # that is absent from the shard.  ``FeatureIndex.lookup`` now RAISES a
        # descriptive ``KeyError`` (naming shard+doc), and ``__getitem__`` lets
        # it propagate rather than masking it as ``KeyError(7)`` (issue #521).
        store = SFTFeatureStore(
            str(feature_dir),
            {7: ("shard_00000", 999)},
            segment_mode=SegmentMode.fixed_tokens,
            parquet_original_indices=frozenset({7}),
        )
        with pytest.raises(KeyError, match=r"Document 999 not found in shard 'shard_00000'"):
            store[7]

    def test_getitem_missing_key_raises_key_error(self, tmp_path: Path) -> None:
        """A key with no location entry at all also raises ``KeyError`` (the
        ``self._locations[key]`` lookup), confirming the Mapping contract."""
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _write_shard(feature_dir, "shard_00000", doc_ids=[0])
        store = SFTFeatureStore(
            str(feature_dir),
            {7: ("shard_00000", 0)},
            segment_mode=SegmentMode.fixed_tokens,
            parquet_original_indices=frozenset({7}),
        )
        with pytest.raises(KeyError):
            store[123456]


# ---------------------------------------------------------------------------
# Weakref cache reuse
# ---------------------------------------------------------------------------


class TestWeakRefCache:
    """Repeated ``load_sft_features`` for the same dirs returns the SAME store.

    The cache key is ``(resolved feature_dir, resolved parquet_dir)``; a live
    reference keeps the WeakValueDictionary entry alive, so the second call must
    hand back the identical object rather than re-scanning the parquet shards.
    """

    def test_same_dirs_return_identical_object(self, single_shard_store) -> None:
        store, _expected, (feature_dir, parquet_dir) = single_shard_store
        store_again = load_sft_features(str(feature_dir), str(parquet_dir))
        assert store_again is store

    def test_each_shard_header_is_read_once_on_miss_and_hit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Count total header I/O across both module-level binding sites."""
        feature_dir = tmp_path / "features"
        parquet_dir = tmp_path / "parquet"
        feature_dir.mkdir()
        parquet_dir.mkdir()
        for shard_index in range(4):
            stem = f"shard_{shard_index:05d}"
            _write_shard(feature_dir, stem, doc_ids=[0])
            _write_parquet(parquet_dir, stem, original_indices=[shard_index])

        calls: list[Path] = []
        real_sft_read = sft_features_module.read_shard_header
        real_fingerprint_read = mixture_fingerprint_module.read_shard_header

        def counted_sft(path: Path):
            calls.append(path)
            return real_sft_read(path)

        def counted_fingerprint(path: Path):
            calls.append(path)
            return real_fingerprint_read(path)

        monkeypatch.setattr(sft_features_module, "read_shard_header", counted_sft)
        monkeypatch.setattr(
            mixture_fingerprint_module,
            "read_shard_header",
            counted_fingerprint,
        )
        store = load_sft_features(str(feature_dir), str(parquet_dir))
        assert len(calls) == 4
        assert sorted(calls) == sorted(feature_dir.glob("*_features.safetensors.zst"))

        calls.clear()
        assert load_sft_features(str(feature_dir), str(parquet_dir)) is store
        assert len(calls) == 4
        assert sorted(calls) == sorted(feature_dir.glob("*_features.safetensors.zst"))

    def test_cache_key_resolves_paths(self, single_shard_store) -> None:
        """A non-normalised path to the same directory still hits the cache,
        because the key resolves paths before comparing."""
        store, _expected, (feature_dir, parquet_dir) = single_shard_store
        messy_feature = str(feature_dir) + "/./"
        messy_parquet = str(parquet_dir) + "/../" + parquet_dir.name
        store_again = load_sft_features(messy_feature, messy_parquet)
        assert store_again is store

    def test_distinct_dirs_return_distinct_objects(self, tmp_path: Path) -> None:
        """Different (feature_dir, parquet_dir) keys must NOT alias."""
        dirs = []
        stores = []
        for tag in ("a", "b"):
            feature_dir = tmp_path / f"features_{tag}"
            parquet_dir = tmp_path / f"parquet_{tag}"
            feature_dir.mkdir()
            parquet_dir.mkdir()
            _write_shard(feature_dir, "shard_00000", doc_ids=[0])
            _write_parquet(parquet_dir, "shard_00000", original_indices=[0])
            dirs.append((feature_dir, parquet_dir))
            stores.append(load_sft_features(str(feature_dir), str(parquet_dir)))
        assert stores[0] is not stores[1]

    def test_cache_key_distinguishes_segment_mode(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "features"
        parquet_dir = tmp_path / "parquet"
        feature_dir.mkdir()
        parquet_dir.mkdir()
        _write_shard(feature_dir, "shard_00000", doc_ids=[0])
        _write_parquet(parquet_dir, "shard_00000", original_indices=[0])
        fixed_store = load_sft_features(str(feature_dir), str(parquet_dir))

        _write_shard(
            feature_dir,
            "shard_00000",
            doc_ids=[0],
            metadata={"segment_mode": SegmentMode.turn_delimited_fixed_tokens.value},
        )
        _write_statistics_mode(feature_dir, SegmentMode.turn_delimited_fixed_tokens)
        turn_store = load_sft_features(str(feature_dir), str(parquet_dir))

        assert fixed_store is not turn_store
        assert fixed_store.segment_mode is SegmentMode.fixed_tokens
        assert turn_store.segment_mode is SegmentMode.turn_delimited_fixed_tokens

    def test_same_feature_dir_with_distinct_parquet_dirs_does_not_alias(
        self, tmp_path: Path
    ) -> None:
        """The cache key already includes parquet_dir, which carries split identity."""
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _write_shard(feature_dir, "shard_00000", doc_ids=[0])
        stores = []
        for split in SFTSplit:
            parquet_dir = tmp_path / f"parquet_{split.value}"
            parquet_dir.mkdir()
            _write_parquet(parquet_dir, "shard_00000", original_indices=[0])
            (parquet_dir / "conversion_meta.json").write_text(
                json.dumps(
                    {
                        "mixture": "safety",
                        "split": split.value,
                        "format": "full_nanochat_text",
                        "completed_shards": 1,
                        "total_shards": 1,
                        "total_conversations": 1,
                    }
                )
            )
            stores.append(load_sft_features(str(feature_dir), str(parquet_dir)))

        assert stores[0] is not stores[1]
        assert stores[0].conversion_split is SFTSplit.TRAIN
        assert stores[1].conversion_split is SFTSplit.VAL
