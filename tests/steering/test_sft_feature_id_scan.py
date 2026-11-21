"""ID-only percentile-scan path (issue #1495): loader, store scan, dispatcher.

Fixtures are REAL on-disk artifact families (save_features shards + parquet
with original_index) — the dispatcher test additionally uses a plain-mapping
double to pin the reference route — and every fast-path result is pinned
against scan_queried_feature_ids_reference, the moved-but-unchanged
production reference scanner, including exception type and message equality
on the anomaly paths.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch as t

import steering.conditional.encoder_decoder.sft_features as sft_features_module
import steering.conditional.feature_index as feature_index_module
from errors import DataLoadError
from steering.conditional.encoder_decoder.sft_features import (
    load_sft_features,
    scan_queried_feature_ids_reference,
    scan_required_feature_id_bitmap,
)
from steering.conditional.feature_index import (
    _ID_SCAN_SHARD_KEYS,
    FeatureIndex,
    SegmentFeatures,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.feature_io import CorruptFeatureShardError, save_features
from tests.steering.test_sft_features import _write_parquet, _write_shard


def _write_custom_shard(
    feature_dir: Path,
    stem: str,
    *,
    doc_ids: list[int],
    segment_indices: list[int],
    seg_feature_ids: list[list[int]],
    seg_feature_mags: list[list[float]] | None = None,
    n_docs: int | None = None,
) -> None:
    """A real multi-segment shard with explicit per-segment feature ids."""
    assert len(doc_ids) == len(segment_indices) == len(seg_feature_ids)
    offsets = [0]
    flat_ids: list[int] = []
    flat_mags: list[float] = []
    for seg_i, fids in enumerate(seg_feature_ids):
        flat_ids.extend(fids)
        mags = (
            seg_feature_mags[seg_i]
            if seg_feature_mags is not None
            else [1.0 + j for j in range(len(fids))]
        )
        flat_mags.extend(mags)
        offsets.append(len(flat_ids))
    n = len(doc_ids)
    shard = {
        "offsets": t.tensor(offsets, dtype=t.int64),
        "feature_ids": t.tensor(flat_ids, dtype=t.int32),
        "feature_mags": t.tensor(flat_mags, dtype=t.float32),
        "doc_ids": t.tensor(doc_ids, dtype=t.int32),
        "segment_indices": t.tensor(segment_indices, dtype=t.int32),
        "segment_positions": t.tensor([i * 128 for i in range(n)], dtype=t.int32),
        "segment_char_starts": t.tensor([0] * n, dtype=t.int32),
        "segment_char_ends": t.tensor([768] * n, dtype=t.int32),
        "n_segments": n,
        "n_docs": (max(doc_ids) + 1 if n_docs is None else n_docs),
        "layer": 12,
        "context_window": 1024,
        "pooling": "mean_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 128,
    }
    save_features(shard, feature_dir / f"{stem}_features")


class TestLoadIdScanArrays:
    def test_returns_only_id_scan_tensors_and_scalars(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _write_shard(feature_dir, "shard_00000", doc_ids=[0, 1], feats_per_seg=3)
        index = FeatureIndex(str(feature_dir))
        arrays = index.load_id_scan_arrays("shard_00000")
        assert set(arrays) >= {"offsets", "doc_ids", "feature_ids"}
        assert "feature_mags" not in arrays
        assert "segment_char_starts" not in arrays
        assert arrays["offsets"].dtype is t.int64
        assert arrays["doc_ids"].dtype is t.int32
        assert arrays["feature_ids"].dtype is t.int32
        assert int(arrays["n_segments"]) == 2

    def test_feature_ids_are_delta_decoded_to_real_values(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _write_custom_shard(
            feature_dir,
            "shard_00000",
            doc_ids=[0, 0],
            segment_indices=[0, 1],
            seg_feature_ids=[[7, 3, 3], [90, 2]],
        )
        index = FeatureIndex(str(feature_dir))
        arrays = index.load_id_scan_arrays("shard_00000")
        # save_features sorts ids ascending within each segment.
        assert arrays["feature_ids"].tolist() == [3, 3, 7, 2, 90]

    def test_unknown_stem_raises_keyerror(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _write_shard(feature_dir, "shard_00000", doc_ids=[0])
        index = FeatureIndex(str(feature_dir))
        with pytest.raises(KeyError, match="'shard_99999'"):
            index.load_id_scan_arrays("shard_99999")

    def test_requests_exactly_the_id_scan_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _write_shard(feature_dir, "shard_00000", doc_ids=[0, 1])
        index = FeatureIndex(str(feature_dir))
        seen_keys: list[set[str] | None] = []
        real = feature_index_module.load_features

        def spy(path, *, keys=None, **kwargs):
            seen_keys.append(keys)
            return real(path, keys=keys, **kwargs)

        monkeypatch.setattr(feature_index_module, "load_features", spy)
        index.load_id_scan_arrays("shard_00000")
        assert seen_keys == [set(_ID_SCAN_SHARD_KEYS)]

    def test_corrupt_empty_segment_matches_full_loader_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The shared validator must raise the identical message on both paths."""
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _write_shard(feature_dir, "shard_00000", doc_ids=[0, 1], feats_per_seg=2)
        real = feature_index_module.load_features

        def doctor(path, *, keys=None, **kwargs):
            data = real(path, keys=keys, **kwargs)
            # Make segment 1 empty: offsets [0,2,4] -> [0,2,2] and trim ids.
            data["offsets"] = t.tensor([0, 2, 2], dtype=t.int64)
            data["feature_ids"] = data["feature_ids"][:2]
            if "feature_mags" in data:
                data["feature_mags"] = data["feature_mags"][:2]
            return data

        index = FeatureIndex(str(feature_dir))
        monkeypatch.setattr(feature_index_module, "load_features", doctor)
        with pytest.raises(CorruptFeatureShardError) as fast_exc:
            index.load_id_scan_arrays("shard_00000")
        with pytest.raises(CorruptFeatureShardError) as full_exc:
            index._load_shard("shard_00000")
        assert str(fast_exc.value) == str(full_exc.value)
        assert "segment 1 has zero" in str(fast_exc.value)


N_SAE = 128


def _dirs(tmp_path: Path) -> tuple[Path, Path]:
    feature_dir = tmp_path / "features"
    parquet_dir = tmp_path / "parquet"
    feature_dir.mkdir()
    parquet_dir.mkdir()
    return feature_dir, parquet_dir


def _single_shard_store(tmp_path: Path, *, original_indices: list[int], **shard_kwargs):
    """One ``_write_custom_shard`` + parquet family, loaded for real."""
    feature_dir, parquet_dir = _dirs(tmp_path)
    _write_custom_shard(feature_dir, "shard_00000", **shard_kwargs)
    _write_parquet(parquet_dir, "shard_00000", original_indices=original_indices)
    return load_sft_features(str(feature_dir), str(parquet_dir))


def _count_vectorized_attempts(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record each entry into the vectorized pass without changing behavior."""
    attempts: list[int] = []
    real_vectorized = sft_features_module.SFTFeatureStore._vectorized_id_scan

    def counting_vectorized(self, *args, **kwargs):
        attempts.append(1)
        return real_vectorized(self, *args, **kwargs)

    monkeypatch.setattr(
        sft_features_module.SFTFeatureStore, "_vectorized_id_scan", counting_vectorized
    )
    return attempts


def _force_float32_doc_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Doctor every shard load with float32 doc_ids (forces the fallback)."""
    real = feature_index_module.load_features

    def doctor(path, *, keys=None, **kwargs):
        data = real(path, keys=keys, **kwargs)
        data["doc_ids"] = data["doc_ids"].to(t.float32)
        return data

    monkeypatch.setattr(feature_index_module, "load_features", doctor)


def _build_family(tmp_path: Path):
    """Three real shards, multi-segment docs, non-contiguous original_index.

    shard_00000: docs {0: segs[[0,1],[5,5,7]], 1: segs[[127]]}     -> orig 5, 100
    shard_00001: docs {0: segs[[3],[64,90]]}                        -> orig 7
    shard_00002: docs {0: segs[[0]], 1: segs[[2,3],[9]], 2: [[11]]} -> orig 300, 12, 44
    """
    feature_dir, parquet_dir = _dirs(tmp_path)
    _write_custom_shard(
        feature_dir,
        "shard_00000",
        doc_ids=[0, 0, 1],
        segment_indices=[0, 1, 0],
        seg_feature_ids=[[0, 1], [5, 5, 7], [127]],
        n_docs=2,
    )
    _write_parquet(parquet_dir, "shard_00000", original_indices=[5, 100])
    _write_custom_shard(
        feature_dir,
        "shard_00001",
        doc_ids=[0, 0],
        segment_indices=[0, 1],
        seg_feature_ids=[[3], [64, 90]],
        n_docs=1,
    )
    _write_parquet(parquet_dir, "shard_00001", original_indices=[7])
    _write_custom_shard(
        feature_dir,
        "shard_00002",
        doc_ids=[0, 1, 1, 2],
        segment_indices=[0, 0, 1, 0],
        seg_feature_ids=[[0], [2, 3], [9], [11]],
        n_docs=3,
    )
    _write_parquet(parquet_dir, "shard_00002", original_indices=[300, 12, 44])
    return load_sft_features(str(feature_dir), str(parquet_dir))


class TestFastScanEqualsReference:
    def test_full_coverage_bitmap_is_byte_identical(self, tmp_path: Path) -> None:
        store = _build_family(tmp_path)
        indices = sorted(store)
        fast = store.scan_queried_feature_id_bitmap(indices, n_sae_features=N_SAE, source="train")
        reference = scan_queried_feature_ids_reference(
            store, indices, n_sae_features=N_SAE, source="train"
        )
        assert fast.dtype is t.bool and fast.shape == (N_SAE,)
        assert t.equal(fast, reference)
        assert fast.nonzero().squeeze(1).tolist() == [0, 1, 2, 3, 5, 7, 9, 11, 64, 90, 127]

    def test_subset_and_duplicate_indices_match_reference(self, tmp_path: Path) -> None:
        store = _build_family(tmp_path)
        # Non-contiguous subset spanning shards 0 and 2, with a duplicate.
        indices = [100, 12, 100, 300]
        fast = store.scan_queried_feature_id_bitmap(indices, n_sae_features=N_SAE, source="train")
        reference = scan_queried_feature_ids_reference(
            store, indices, n_sae_features=N_SAE, source="train"
        )
        assert t.equal(fast, reference)
        assert fast.nonzero().squeeze(1).tolist() == [0, 2, 3, 9, 127]

    def test_boundary_ids_zero_and_max_are_seen(self, tmp_path: Path) -> None:
        store = _build_family(tmp_path)
        fast = store.scan_queried_feature_id_bitmap(
            sorted(store), n_sae_features=N_SAE, source="train"
        )
        assert bool(fast[0]) and bool(fast[N_SAE - 1])

    def test_dispatcher_routes_store_fast_and_mapping_reference(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _build_family(tmp_path)
        calls: list[str] = []
        real_fast = sft_features_module.SFTFeatureStore.scan_queried_feature_id_bitmap

        def spy(self, *args, **kwargs):
            calls.append("fast")
            return real_fast(self, *args, **kwargs)

        monkeypatch.setattr(
            sft_features_module.SFTFeatureStore, "scan_queried_feature_id_bitmap", spy
        )
        scan_required_feature_id_bitmap(store, sorted(store), n_sae_features=N_SAE, source="train")
        assert calls == ["fast"]
        # A plain dict store must take the reference walk without error.
        plain = {
            0: [
                SegmentFeatures(
                    segment_idx=0,
                    position=0,
                    feature_ids=t.tensor([4], dtype=t.int32),
                    feature_mags=t.tensor([1.0]),
                    char_start=0,
                    char_end=8,
                )
            ]
        }
        out = scan_required_feature_id_bitmap(plain, [0], n_sae_features=N_SAE, source="train")
        assert out.nonzero().squeeze(1).tolist() == [4]
        assert calls == ["fast"]

    def test_randomized_layouts_match_reference(self, tmp_path: Path) -> None:
        """Seeded sweep over random shard layouts (sizes, segs, ids, subsets)."""
        for seed in range(5):
            rng = random.Random(seed)
            case_dir = tmp_path / f"case_{seed}"
            feature_dir = case_dir / "features"
            parquet_dir = case_dir / "parquet"
            feature_dir.mkdir(parents=True)
            parquet_dir.mkdir(parents=True)
            next_orig = 0
            all_orig: list[int] = []
            for shard_i in range(rng.randint(1, 3)):
                n_docs = rng.randint(1, 4)
                doc_ids: list[int] = []
                segment_indices: list[int] = []
                seg_ids: list[list[int]] = []
                for doc in range(n_docs):
                    for seg in range(rng.randint(1, 3)):
                        doc_ids.append(doc)
                        segment_indices.append(seg)
                        seg_ids.append([rng.randrange(N_SAE) for _ in range(rng.randint(1, 6))])
                stem = f"shard_{shard_i:05d}"
                _write_custom_shard(
                    feature_dir,
                    stem,
                    doc_ids=doc_ids,
                    segment_indices=segment_indices,
                    seg_feature_ids=seg_ids,
                    n_docs=n_docs,
                )
                originals = []
                for _ in range(n_docs):
                    next_orig += rng.randint(1, 7)
                    originals.append(next_orig)
                _write_parquet(parquet_dir, stem, original_indices=originals)
                all_orig.extend(originals)
            store = load_sft_features(str(feature_dir), str(parquet_dir))
            subset = sorted(rng.sample(all_orig, k=rng.randint(1, len(all_orig))))
            for indices in (sorted(all_orig), subset):
                fast = store.scan_queried_feature_id_bitmap(
                    indices, n_sae_features=N_SAE, source="train"
                )
                reference = scan_queried_feature_ids_reference(
                    store, indices, n_sae_features=N_SAE, source="train"
                )
                assert t.equal(fast, reference), f"seed={seed} indices={indices}"


class TestFastScanErrorParity:
    def test_out_of_range_id_raises_the_exact_reference_error(self, tmp_path: Path) -> None:
        store = _single_shard_store(
            tmp_path,
            original_indices=[10, 20],
            doc_ids=[0, 1],
            segment_indices=[0, 0],
            seg_feature_ids=[[1, 2], [3, N_SAE + 9]],
            n_docs=2,
        )
        with pytest.raises(DataLoadError) as fast_exc:
            store.scan_queried_feature_id_bitmap([10, 20], n_sae_features=N_SAE, source="train")
        with pytest.raises(DataLoadError) as ref_exc:
            scan_queried_feature_ids_reference(
                store, [10, 20], n_sae_features=N_SAE, source="train"
            )
        assert str(fast_exc.value) == str(ref_exc.value)
        assert "document 20" in str(fast_exc.value)
        assert f"outside [0, {N_SAE})" in str(fast_exc.value)

    def test_negative_id_raises_the_exact_reference_error(self, tmp_path: Path) -> None:
        """ids > 65535 keep int32 storage, so a negative id round-trips.

        VERIFIED against feature_io: _optimize_dtype only downcasts to uint16
        when max <= 65535, and _sort_within_segments/_delta_encode perform no
        sign validation — so [-3, 70000] survives save/load bit-exact. If this
        ever changes, doctor via a load_features monkeypatch shared by BOTH
        paths instead (see the empty-segment corruption test).
        """
        store = _single_shard_store(
            tmp_path,
            original_indices=[0],
            doc_ids=[0],
            segment_indices=[0],
            seg_feature_ids=[[-3, 70000]],
            n_docs=1,
        )
        with pytest.raises(DataLoadError) as fast_exc:
            store.scan_queried_feature_id_bitmap([0], n_sae_features=N_SAE, source="train")
        with pytest.raises(DataLoadError) as ref_exc:
            scan_queried_feature_ids_reference(store, [0], n_sae_features=N_SAE, source="train")
        assert str(fast_exc.value) == str(ref_exc.value)
        assert "[-3, 70000]" in str(fast_exc.value)

    def test_index_missing_from_store_raises_the_exact_keyerror(self, tmp_path: Path) -> None:
        store = _build_family(tmp_path)
        with pytest.raises(KeyError) as fast_exc:
            store.scan_queried_feature_id_bitmap([5, 999], n_sae_features=N_SAE, source="train")
        with pytest.raises(KeyError) as ref_exc:
            scan_queried_feature_ids_reference(
                store, [5, 999], n_sae_features=N_SAE, source="train"
            )
        assert fast_exc.value.args == ref_exc.value.args == (999,)

    def test_document_absent_from_shard_raises_the_exact_keyerror(self, tmp_path: Path) -> None:
        """n_docs metadata says 2 docs but the shard only has rows for doc 0."""
        store = _single_shard_store(
            tmp_path,
            original_indices=[10, 11],
            doc_ids=[0, 0],
            segment_indices=[0, 1],
            seg_feature_ids=[[1], [2]],
            n_docs=2,
        )
        with pytest.raises(KeyError) as fast_exc:
            store.scan_queried_feature_id_bitmap([10, 11], n_sae_features=N_SAE, source="train")
        with pytest.raises(KeyError) as ref_exc:
            scan_queried_feature_ids_reference(
                store, [10, 11], n_sae_features=N_SAE, source="train"
            )
        assert str(fast_exc.value) == str(ref_exc.value)
        assert "Document 1 not found in shard" in str(fast_exc.value)

    def test_missing_doc_ids_tensor_surfaces_the_reference_keyerror(self, tmp_path: Path) -> None:
        """A shard without doc_ids: the fast keys-load raises a ValueError,
        the fallback must surface the reference's KeyError('doc_ids') instead."""
        feature_dir, parquet_dir = _dirs(tmp_path)
        shard = {
            "offsets": t.tensor([0, 2], dtype=t.int64),
            "feature_ids": t.tensor([1, 2], dtype=t.int32),
            "feature_mags": t.tensor([1.0, 2.0], dtype=t.float32),
            # doc_ids deliberately absent — save_features does not require it.
            "segment_indices": t.tensor([0], dtype=t.int32),
            "segment_positions": t.tensor([0], dtype=t.int32),
            "segment_char_starts": t.tensor([0], dtype=t.int32),
            "segment_char_ends": t.tensor([8], dtype=t.int32),
            "n_segments": 1,
            "n_docs": 1,
            "layer": 12,
            "context_window": 1024,
            "pooling": "mean_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 128,
        }
        save_features(shard, feature_dir / "shard_00000_features")
        _write_parquet(parquet_dir, "shard_00000", original_indices=[10])
        store = load_sft_features(str(feature_dir), str(parquet_dir))
        with pytest.raises(KeyError) as fast_exc:
            store.scan_queried_feature_id_bitmap([10], n_sae_features=N_SAE, source="train")
        with pytest.raises(KeyError) as ref_exc:
            scan_queried_feature_ids_reference(store, [10], n_sae_features=N_SAE, source="train")
        assert fast_exc.value.args == ref_exc.value.args == ("doc_ids",)

    def test_nonzero_csr_origin_surfaces_the_reference_split_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """offsets=[1,3] over 3 ids passes today's structural asserts but makes
        _build_docs' split_with_sizes raise; the fast path must surface exactly
        that error via fallback, never a silent 3-id bitmap."""
        store = _single_shard_store(
            tmp_path,
            original_indices=[10],
            doc_ids=[0],
            segment_indices=[0],
            seg_feature_ids=[[1, 2, 3]],
            n_docs=1,
        )
        real = feature_index_module.load_features

        def doctor(path, *, keys=None, **kwargs):
            data = real(path, keys=keys, **kwargs)
            data["offsets"] = t.tensor([1, 3], dtype=t.int64)
            return data

        monkeypatch.setattr(feature_index_module, "load_features", doctor)
        with pytest.raises(RuntimeError) as fast_exc:
            store.scan_queried_feature_id_bitmap([10], n_sae_features=N_SAE, source="train")
        with pytest.raises(RuntimeError) as ref_exc:
            scan_queried_feature_ids_reference(store, [10], n_sae_features=N_SAE, source="train")
        assert str(fast_exc.value) == str(ref_exc.value)

    def test_two_corrupt_shards_surface_the_reference_order_error(self, tmp_path: Path) -> None:
        """Fast scans stems in sorted order, the reference in dataset order.
        With shard_00001 zstd-garbage and shard_00002 carrying an out-of-range
        id, dataset order [0(clean shard_00000), 1(shard_00002), 2(shard_00001)]
        must surface shard_00002's DataLoadError — proving the fallback
        restores reference-first error selection."""
        feature_dir, parquet_dir = _dirs(tmp_path)
        _write_custom_shard(
            feature_dir,
            "shard_00000",
            doc_ids=[0],
            segment_indices=[0],
            seg_feature_ids=[[1]],
            n_docs=1,
        )
        _write_parquet(parquet_dir, "shard_00000", original_indices=[0])
        _write_custom_shard(
            feature_dir,
            "shard_00001",
            doc_ids=[0],
            segment_indices=[0],
            seg_feature_ids=[[2]],
            n_docs=1,
        )
        _write_parquet(parquet_dir, "shard_00001", original_indices=[2])
        _write_custom_shard(
            feature_dir,
            "shard_00002",
            doc_ids=[0],
            segment_indices=[0],
            seg_feature_ids=[[3, N_SAE + 50]],
            n_docs=1,
        )
        _write_parquet(parquet_dir, "shard_00002", original_indices=[1])
        store = load_sft_features(str(feature_dir), str(parquet_dir))
        # Corrupt AFTER the store is built (construction full-loads only the
        # first sorted shard, shard_00000, which stays clean).
        shard1 = next(feature_dir.glob("shard_00001*"))
        shard1.write_bytes(b"not a zstd stream")
        with pytest.raises(DataLoadError) as fast_exc:
            store.scan_queried_feature_id_bitmap([0, 1, 2], n_sae_features=N_SAE, source="train")
        with pytest.raises(DataLoadError) as ref_exc:
            scan_queried_feature_ids_reference(
                store, [0, 1, 2], n_sae_features=N_SAE, source="train"
            )
        assert str(fast_exc.value) == str(ref_exc.value)
        assert "document 1" in str(fast_exc.value)

    def test_integral_float_doc_ids_fall_back_and_match_reference(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """float32 doc_ids with integral values: the reference SUCCEEDS
        (0.0 == 0 as a dict key), so the fast path must attempt the vectorized
        pass, fall back, and return the reference's bitmap — never coerce,
        never fail."""
        store = _single_shard_store(
            tmp_path,
            original_indices=[10, 20],
            doc_ids=[0, 1],
            segment_indices=[0, 0],
            seg_feature_ids=[[4, 6], [9]],
            n_docs=2,
        )
        _force_float32_doc_ids(monkeypatch)
        reference_calls: list[int] = []
        real_reference = sft_features_module.scan_queried_feature_ids_reference

        def counting_reference(*args, **kwargs):
            reference_calls.append(1)
            return real_reference(*args, **kwargs)

        monkeypatch.setattr(
            sft_features_module,
            "scan_queried_feature_ids_reference",
            counting_reference,
        )
        fast_attempts = _count_vectorized_attempts(monkeypatch)
        fast = store.scan_queried_feature_id_bitmap([10, 20], n_sae_features=N_SAE, source="train")
        assert fast_attempts == [1], "the vectorized pass must have been attempted"
        assert reference_calls == [1], "must have fallen back exactly once"
        assert fast.nonzero().squeeze(1).tolist() == [4, 6, 9]

    def test_healthy_store_never_falls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deviation 5's 'healthy runs never fall back' claim, pinned."""
        store = _build_family(tmp_path)

        def forbidden_reference(*args, **kwargs):
            raise AssertionError("reference walk must not run on a healthy store")

        monkeypatch.setattr(
            sft_features_module,
            "scan_queried_feature_ids_reference",
            forbidden_reference,
        )
        bitmap = store.scan_queried_feature_id_bitmap(
            sorted(store), n_sae_features=N_SAE, source="train"
        )
        assert int(bitmap.sum()) > 0

    def test_failing_warning_logger_never_outranks_the_reference(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even if emitting the fallback WARNING itself raises, the reference
        outcome must still be returned — telemetry never outranks the
        reference walk."""
        store = _single_shard_store(
            tmp_path,
            original_indices=[10, 20],
            doc_ids=[0, 1],
            segment_indices=[0, 0],
            seg_feature_ids=[[4, 6], [9]],
            n_docs=2,
        )
        _force_float32_doc_ids(monkeypatch)

        class _ExplodingLogger:
            def warning(self, *args, **kwargs):
                raise RuntimeError("logger failed")

            def info(self, *args, **kwargs):
                return None

        monkeypatch.setattr(sft_features_module, "logger", _ExplodingLogger())
        fast_attempts = _count_vectorized_attempts(monkeypatch)
        fast = store.scan_queried_feature_id_bitmap([10, 20], n_sae_features=N_SAE, source="train")
        assert fast_attempts == [1], "the vectorized pass must have been attempted"
        assert fast.nonzero().squeeze(1).tolist() == [4, 6, 9]

    def test_empty_dataset_indices_match_reference(self, tmp_path: Path) -> None:
        store = _build_family(tmp_path)
        fast = store.scan_queried_feature_id_bitmap([], n_sae_features=N_SAE, source="train")
        reference = scan_queried_feature_ids_reference(
            store, [], n_sae_features=N_SAE, source="train"
        )
        assert t.equal(fast, reference)
        assert int(fast.sum()) == 0

    def test_zero_segment_shard_with_empty_parquet_is_never_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reachable 'empty document' analog: a zero-segment shard paired
        with a zero-row parquet contributes no locations, so neither scanner
        touches it and the bitmaps still match. (A document with zero segments
        inside a nonempty shard is impossible: extraction rejects it, and an
        absent row is the KeyError case tested above.)"""
        feature_dir, parquet_dir = _dirs(tmp_path)
        _write_custom_shard(
            feature_dir,
            "shard_00000",
            doc_ids=[0],
            segment_indices=[0],
            seg_feature_ids=[[7]],
            n_docs=1,
        )
        _write_parquet(parquet_dir, "shard_00000", original_indices=[10])
        empty = {
            "offsets": t.tensor([0], dtype=t.int64),
            "feature_ids": t.tensor([], dtype=t.int32),
            "feature_mags": t.tensor([], dtype=t.float32),
            "doc_ids": t.tensor([], dtype=t.int32),
            "segment_indices": t.tensor([], dtype=t.int32),
            "segment_positions": t.tensor([], dtype=t.int32),
            "segment_char_starts": t.tensor([], dtype=t.int32),
            "segment_char_ends": t.tensor([], dtype=t.int32),
            "n_segments": 0,
            "n_docs": 0,
            "layer": 12,
            "context_window": 1024,
            "pooling": "mean_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 128,
        }
        save_features(empty, feature_dir / "shard_00001_features")
        _write_parquet(parquet_dir, "shard_00001", original_indices=[])
        store = load_sft_features(str(feature_dir), str(parquet_dir))
        stems_read: list[str] = []
        real_loader = feature_index_module.FeatureIndex.load_id_scan_arrays

        def recording_loader(self, shard_stem, **kwargs):
            stems_read.append(shard_stem)
            return real_loader(self, shard_stem, **kwargs)

        monkeypatch.setattr(
            feature_index_module.FeatureIndex, "load_id_scan_arrays", recording_loader
        )
        fast = store.scan_queried_feature_id_bitmap([10], n_sae_features=N_SAE, source="train")
        reference = scan_queried_feature_ids_reference(
            store, [10], n_sae_features=N_SAE, source="train"
        )
        assert stems_read == ["shard_00000"], stems_read
        assert t.equal(fast, reference)
        assert fast.nonzero().squeeze(1).tolist() == [7]


class TestFastScanReadsNothingItDoesNotNeed:
    def test_no_segmentfeatures_and_no_full_doc_build(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _build_family(tmp_path)  # store construction loads shard 0 once

        def forbidden(self, *args, **kwargs):
            raise AssertionError("_build_docs must not run during the ID-only scan")

        monkeypatch.setattr(feature_index_module.FeatureIndex, "_build_docs", forbidden)
        fast = store.scan_queried_feature_id_bitmap(
            sorted(store), n_sae_features=N_SAE, source="train"
        )
        assert int(fast.sum()) > 0

    def test_mask_projection_matches_contract(self, tmp_path: Path) -> None:
        """masked_required == raw_required[keep_mask(raw_required)], exactly."""
        store = _build_family(tmp_path)
        bitmap = store.scan_queried_feature_id_bitmap(
            sorted(store), n_sae_features=N_SAE, source="train"
        )
        raw_required = t.nonzero(bitmap, as_tuple=False).squeeze(1)
        mask = FeatureMask(mode=FeatureMaskMode.whitelist, ids=(0, 5, 9, 127))
        masked_required = raw_required[mask.keep_mask(raw_required)]
        assert masked_required.tolist() == [0, 5, 9, 127]
