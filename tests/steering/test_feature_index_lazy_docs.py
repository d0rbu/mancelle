"""Lazy per-document shard materialization (issue #1504, PR-3 Task 2.2).

``lazy_docs=True`` replaces the eager ``_build_docs`` object graph with a
``Mapping`` that materializes one document at a time.  Everything a consumer can
observe — values, dtypes, storage independence, segment order, iteration order,
``len``, the absent-document ``KeyError``, the ``_EMPTY_MAGS`` singleton, and the
load-time failure of a corrupt shard — must match the eager path exactly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch as t

from steering.conditional import feature_index as feature_index_module
from steering.conditional.feature_index import FeatureIndex, SegmentFeatures
from steering.conditional.shared_stem_store import SharedStemStore
from steering.feature_io import TensorKey, load_features, save_features
from steering.overlay_io import (
    OverlayKind,
    overlay_dir,
    overlay_shard_path,
    save_overlay_shard,
    write_manifest,
)
from tests.steering._helpers import save_raw_shard

PCT_CHECKSUM = "a" * 64
TOK_CHECKSUM = "b" * 64


# Sparse, unsorted document ids with a different segment count each: doc ids are
# neither dense nor ascending on disk, and no document's span can be inferred
# from a uniform stride. Ascending-id order is therefore 2, 4, 7, 9 with 1, 2, 3
# and 4 segments respectively.
_DOC_IDS: tuple[int, ...] = (7, 2, 9, 4)
_SEGS_PER_DOC: tuple[int, ...] = (3, 1, 4, 2)
_N_SEGMENTS = sum(_SEGS_PER_DOC)


def _shard_data(
    *,
    seed: int,
    doc_ids: tuple[int, ...] = _DOC_IDS,
    segs_per_doc: tuple[int, ...] = _SEGS_PER_DOC,
) -> dict:
    """A valid shard whose on-disk rows are segment-major AND segment-descending.

    Segment-major makes the ``(doc_id, segment_idx)`` sort a real permutation of
    the stored rows; segment-*descending* additionally makes the compound key
    load-bearing — a stable sort on ``doc_id`` alone would leave each document's
    segments in descending order, which the equality assertions then catch.
    Document ids are sparse and unsorted and their segment counts differ, so the
    document spans cannot be right by accident.
    """
    generator = t.Generator().manual_seed(seed)
    n_docs = len(doc_ids)
    rows = [
        (seg, doc)
        for seg in reversed(range(max(segs_per_doc)))
        for doc, n_segs in zip(doc_ids, segs_per_doc, strict=True)
        if seg < n_segs
    ]
    sizes = [2 + ((doc + seg + seed) % 4) for seg, doc in rows]
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    feature_ids: list[int] = []
    feature_mags: list[float] = []
    for size in sizes:
        feature_ids.extend(sorted(t.randperm(4096, generator=generator)[:size].tolist()))
        feature_mags.extend((t.rand(size, generator=generator) + 0.01).tolist())
    return {
        "offsets": t.tensor(offsets, dtype=t.int64),
        "feature_ids": t.tensor(feature_ids, dtype=t.int32),
        "feature_mags": t.tensor(feature_mags, dtype=t.float32),
        "doc_ids": t.tensor([doc for _seg, doc in rows], dtype=t.int32),
        "segment_indices": t.tensor([seg for seg, _doc in rows], dtype=t.int32),
        "segment_positions": t.tensor([seg * 128 for seg, _doc in rows], dtype=t.int32),
        "segment_char_starts": t.tensor([seg * 16 for seg, _doc in rows], dtype=t.int32),
        "segment_char_ends": t.tensor([seg * 16 + 16 for seg, _doc in rows], dtype=t.int32),
        "doc_token_counts": t.tensor([1000] * n_docs, dtype=t.int32),
        "n_docs": n_docs,
        "n_segments": len(rows),
        "layer": 8,
        "context_window": 1024,
        "pooling": "mean_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 128,
    }


def _build_family(root: Path, *, n_shards: int = 2, overlays: bool = True) -> Path:
    feature_dir = root / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    stems = []
    for shard in range(n_shards):
        stem = f"shard_{shard:05d}"
        save_features(_shard_data(seed=shard), feature_dir / f"{stem}_features")
        stems.append(stem)
    if not overlays:
        return feature_dir
    pct_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, PCT_CHECKSUM)
    tok_dir = overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, TOK_CHECKSUM)
    write_manifest(pct_dir, PCT_CHECKSUM, OverlayKind.PERCENTILE)
    write_manifest(tok_dir, TOK_CHECKSUM, OverlayKind.SEG_BOUNDARY)
    for stem in stems:
        raw = load_features(feature_dir / f"{stem}_features.safetensors.zst")
        n_segments = int(raw["n_segments"])
        save_overlay_shard(
            {
                TensorKey.FEATURE_PERCENTILES: raw["feature_ids"]
                .to(t.int64)
                .remainder(256)
                .to(t.uint8)
            },
            overlay_shard_path(pct_dir, stem),
        )
        save_overlay_shard(
            {
                TensorKey.SEGMENT_TOKEN_STARTS: t.arange(n_segments, dtype=t.int32) * 7,
                TensorKey.SEGMENT_TOKEN_ENDS: t.arange(1, n_segments + 1, dtype=t.int32) * 7,
            },
            overlay_shard_path(tok_dir, stem),
        )
    return feature_dir


def _index(
    feature_dir: Path,
    *,
    lazy_docs: bool,
    overlays: bool = True,
    skip_feature_mags: bool = False,
) -> FeatureIndex:
    return FeatureIndex(
        str(feature_dir),
        lazy_docs=lazy_docs,
        skip_feature_mags=skip_feature_mags,
        percentile_checksum=PCT_CHECKSUM if overlays else None,
        tokenizer_checksum=TOK_CHECKSUM if overlays else None,
    )


def _assert_owns_its_storage(tensor: t.Tensor) -> None:
    """A clone owns exactly its own bytes; a view into a shard array does not."""
    assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()


def _assert_segment_identical(actual: SegmentFeatures, reference: SegmentFeatures) -> None:
    assert actual.segment_idx == reference.segment_idx
    assert type(actual.segment_idx) is type(reference.segment_idx)
    assert actual.position == reference.position
    assert actual.char_start == reference.char_start
    assert actual.char_end == reference.char_end
    assert actual.token_start == reference.token_start
    assert actual.token_end == reference.token_end
    assert actual.feature_ids.dtype == reference.feature_ids.dtype
    assert t.equal(actual.feature_ids, reference.feature_ids)
    assert actual.feature_mags.dtype == reference.feature_mags.dtype
    assert t.equal(actual.feature_mags, reference.feature_mags)
    assert (actual.feature_percentiles is None) == (reference.feature_percentiles is None)
    if actual.feature_percentiles is not None and reference.feature_percentiles is not None:
        assert actual.feature_percentiles.dtype == reference.feature_percentiles.dtype
        assert t.equal(actual.feature_percentiles, reference.feature_percentiles)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_lazy_docs_defaults_to_false(tmp_path: Path) -> None:
    index = _index(_build_family(tmp_path, n_shards=1), lazy_docs=False)
    assert index._lazy_docs is False


def test_lazy_docs_rejects_non_bool(tmp_path: Path) -> None:
    feature_dir = _build_family(tmp_path, n_shards=1, overlays=False)
    with pytest.raises(ValueError, match=r"lazy_docs must be a bool"):
        FeatureIndex(str(feature_dir), lazy_docs=1)  # ty: ignore[invalid-argument-type]


def test_lazy_docs_rejects_shared_store(tmp_path: Path) -> None:
    feature_dir = _build_family(tmp_path, n_shards=1, overlays=False)
    store = SharedStemStore(tmp_path / "shm")
    with pytest.raises(ValueError, match=r"lazy_docs does not support a shared_store"):
        FeatureIndex(str(feature_dir), lazy_docs=True, shared_store=store)


def test_lazy_docs_rejects_lazy_shared_docs(tmp_path: Path) -> None:
    feature_dir = _build_family(tmp_path, n_shards=1, overlays=False)
    store = SharedStemStore(tmp_path / "shm")
    with pytest.raises(ValueError, match=r"lazy_docs and lazy_shared_docs are mutually exclusive"):
        FeatureIndex(
            str(feature_dir),
            lazy_docs=True,
            lazy_shared_docs=True,
            shared_store=store,
            compact_feature_ids=True,
        )


def test_lazy_docs_rejects_compact_feature_ids(tmp_path: Path) -> None:
    feature_dir = _build_family(tmp_path, n_shards=1, overlays=False)
    with pytest.raises(ValueError, match=r"lazy_docs does not support compact_feature_ids"):
        FeatureIndex(str(feature_dir), lazy_docs=True, compact_feature_ids=True)


# ---------------------------------------------------------------------------
# Byte-identity against the eager path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("overlays", [True, False])
@pytest.mark.parametrize("skip_feature_mags", [False, True])
def test_lazy_documents_are_identical_to_eager(
    tmp_path: Path, overlays: bool, skip_feature_mags: bool
) -> None:
    feature_dir = _build_family(tmp_path, n_shards=2, overlays=overlays)
    eager = _index(
        feature_dir, lazy_docs=False, overlays=overlays, skip_feature_mags=skip_feature_mags
    )
    lazy = _index(
        feature_dir, lazy_docs=True, overlays=overlays, skip_feature_mags=skip_feature_mags
    )
    for stem in eager.shard_stems:
        eager_docs = eager._load_shard(stem)
        lazy_docs = lazy._load_shard(stem)
        assert len(lazy_docs) == len(eager_docs)
        assert list(iter(lazy_docs)) == list(iter(eager_docs))
        for doc in eager_docs:
            assert doc in lazy_docs
            eager_segments = eager_docs[doc]
            lazy_segments = lazy_docs[doc]
            assert len(lazy_segments) == len(eager_segments)
            assert [seg.segment_idx for seg in lazy_segments] == sorted(
                seg.segment_idx for seg in lazy_segments
            )
            for actual, reference in zip(lazy_segments, eager_segments, strict=True):
                _assert_segment_identical(actual, reference)
                _assert_owns_its_storage(actual.feature_ids)
                if not skip_feature_mags:
                    _assert_owns_its_storage(actual.feature_mags)
                if actual.feature_percentiles is not None:
                    _assert_owns_its_storage(actual.feature_percentiles)


def test_lazy_lookup_matches_eager_lookup(tmp_path: Path) -> None:
    feature_dir = _build_family(tmp_path, n_shards=2)
    eager = _index(feature_dir, lazy_docs=False)
    lazy = _index(feature_dir, lazy_docs=True)
    for stem in eager.shard_stems:
        for doc in _DOC_IDS:
            for actual, reference in zip(
                lazy.lookup(stem, doc), eager.lookup(stem, doc), strict=True
            ):
                _assert_segment_identical(actual, reference)


def test_skip_feature_mags_reuses_the_empty_singleton(tmp_path: Path) -> None:
    feature_dir = _build_family(tmp_path, n_shards=1)
    lazy = _index(feature_dir, lazy_docs=True, skip_feature_mags=True)
    for segment in lazy.lookup("shard_00000", _DOC_IDS[0]):
        assert segment.feature_mags is feature_index_module._EMPTY_MAGS


def test_repeat_lookups_return_equal_but_independent_objects(tmp_path: Path) -> None:
    """The lazy map deliberately does not memoize (the RSS gate rules it out).

    The eager dict hands back the same list object; the lazy map hands back an
    equal, independently owned one. Its only consumer (``SFTFeatureStore``) looks
    each document up once per epoch and never observes object identity, so this
    is the contract that is pinned instead.
    """
    feature_dir = _build_family(tmp_path, n_shards=1)
    lazy = _index(feature_dir, lazy_docs=True)
    first = lazy.lookup("shard_00000", _DOC_IDS[0])
    second = lazy.lookup("shard_00000", _DOC_IDS[0])
    assert first is not second
    assert first[0].feature_ids is not second[0].feature_ids
    for actual, reference in zip(second, first, strict=True):
        _assert_segment_identical(actual, reference)
        _assert_owns_its_storage(actual.feature_ids)


def test_reload_after_eviction_produces_equal_documents(tmp_path: Path) -> None:
    feature_dir = _build_family(tmp_path, n_shards=2)
    lazy = _index(feature_dir, lazy_docs=True)
    first = lazy.lookup("shard_00000", _DOC_IDS[0])
    lazy.lookup("shard_00001", _DOC_IDS[0])  # evicts shard_00000 (LRU-1)
    reloaded = lazy.lookup("shard_00000", _DOC_IDS[0])
    assert reloaded is not first
    for actual, reference in zip(reloaded, first, strict=True):
        _assert_segment_identical(actual, reference)


def test_absent_document_keyerror_message_matches_eager(tmp_path: Path) -> None:
    feature_dir = _build_family(tmp_path, n_shards=1)
    eager = _index(feature_dir, lazy_docs=False)
    lazy = _index(feature_dir, lazy_docs=True)
    with pytest.raises(KeyError) as eager_error:
        eager.lookup("shard_00000", 999)
    with pytest.raises(KeyError) as lazy_error:
        lazy.lookup("shard_00000", 999)
    assert str(lazy_error.value) == str(eager_error.value)


def test_absent_document_raises_plain_keyerror_from_the_mapping(tmp_path: Path) -> None:
    feature_dir = _build_family(tmp_path, n_shards=1)
    docs = _index(feature_dir, lazy_docs=True)._load_shard("shard_00000")
    assert 999 not in docs
    assert (1 << 200) not in docs  # overflowing key must not build an int64 scalar
    with pytest.raises(KeyError):
        docs[999]


def test_zero_segment_shard_is_an_empty_mapping(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    save_features(
        {
            "offsets": t.tensor([0], dtype=t.int64),
            "feature_ids": t.empty(0, dtype=t.int32),
            "feature_mags": t.empty(0, dtype=t.float32),
            "doc_ids": t.empty(0, dtype=t.int32),
            "segment_indices": t.empty(0, dtype=t.int32),
            "segment_positions": t.empty(0, dtype=t.int32),
            "segment_char_starts": t.empty(0, dtype=t.int32),
            "segment_char_ends": t.empty(0, dtype=t.int32),
            "doc_token_counts": t.empty(0, dtype=t.int32),
            "n_docs": 0,
            "n_segments": 0,
            "layer": 8,
            "context_window": 1024,
            "pooling": "mean_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 128,
        },
        feature_dir / "shard_00000_features",
    )
    eager_docs = FeatureIndex(str(feature_dir), lazy_docs=False)._load_shard("shard_00000")
    lazy_docs = FeatureIndex(str(feature_dir), lazy_docs=True)._load_shard("shard_00000")
    assert len(eager_docs) == 0
    assert len(lazy_docs) == 0
    assert list(iter(lazy_docs)) == []
    assert 0 not in lazy_docs


def test_lookup_constructs_only_the_requested_document(tmp_path: Path, monkeypatch) -> None:
    feature_dir = _build_family(tmp_path, n_shards=1)
    constructed: list[int] = []
    original_init = SegmentFeatures.__init__

    def counting_init(self_segment, *args, **kwargs):
        constructed.append(1)
        original_init(self_segment, *args, **kwargs)

    monkeypatch.setattr(SegmentFeatures, "__init__", counting_init)

    lazy = _index(feature_dir, lazy_docs=True)
    docs = lazy._load_shard("shard_00000")
    assert constructed == []  # nothing materialized at load time
    segments = docs[9]
    assert len(constructed) == len(segments) == 4  # this document only, not the other 3
    docs[9]  # not memoized: exactly this document's segments again, never the shard's
    assert len(constructed) == 8

    eager = _index(feature_dir, lazy_docs=False)
    constructed.clear()
    eager._load_shard("shard_00000")
    assert len(constructed) == _N_SEGMENTS  # every document's segments, all up front


# ---------------------------------------------------------------------------
# Load-time integrity (fail-loud parity with split_with_sizes)
# ---------------------------------------------------------------------------


def _corrupt_shard_dir(tmp_path: Path, mutate) -> Path:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    # A small fixed 2-doc x 2-segment shape (12 feature ids) so the corrupt-tail
    # assertions below can name exact counts.
    data = _shard_data(seed=0, doc_ids=(0, 1), segs_per_doc=(2, 2))
    # Raw-write so the corruption survives save_features' strict writer.
    mutate(data)
    save_raw_shard(feature_dir, "shard_00000", data)
    return feature_dir


def test_trailing_feature_id_surplus_fails_at_load_on_both_paths(tmp_path: Path) -> None:
    def mutate(data: dict) -> None:
        data["feature_ids"] = t.cat([data["feature_ids"], t.tensor([1, 2, 3], dtype=t.int32)])

    feature_dir = _corrupt_shard_dir(tmp_path, mutate)
    # Eager rejects it inside split_with_sizes_copy; lazy must reject it at the
    # same point in the load, not silently slice the truncated span.
    with pytest.raises(RuntimeError, match=r"split_with_sizes expects split_sizes to sum"):
        FeatureIndex(str(feature_dir), lazy_docs=False)._load_shard("shard_00000")
    with pytest.raises(RuntimeError, match=r"CSR span \[0, 12\] does not cover its 15"):
        FeatureIndex(str(feature_dir), lazy_docs=True)._load_shard("shard_00000")


def test_leading_feature_id_offset_fails_at_load_on_the_lazy_path(tmp_path: Path) -> None:
    """``offsets[0] != 0`` makes eager (relative sizes) and lazy (absolute
    offsets) disagree on which bytes a segment owns, so it must be rejected."""

    def mutate(data: dict) -> None:
        data["offsets"] = data["offsets"] + 1
        data["feature_ids"] = t.cat([t.tensor([0], dtype=t.int32), data["feature_ids"]])
        data["feature_mags"] = t.cat([t.tensor([0.0], dtype=t.float32), data["feature_mags"]])

    feature_dir = _corrupt_shard_dir(tmp_path, mutate)
    with pytest.raises(RuntimeError, match=r"CSR span"):
        FeatureIndex(str(feature_dir), lazy_docs=True)._load_shard("shard_00000")


def test_trailing_magnitude_surplus_fails_at_load_on_both_paths(tmp_path: Path) -> None:
    def mutate(data: dict) -> None:
        data["feature_mags"] = t.cat([data["feature_mags"], t.zeros(3, dtype=t.float32)])

    feature_dir = _corrupt_shard_dir(tmp_path, mutate)
    with pytest.raises(RuntimeError, match=r"split_with_sizes expects split_sizes to sum"):
        FeatureIndex(str(feature_dir), lazy_docs=False)._load_shard("shard_00000")
    with pytest.raises(RuntimeError, match=r"has 15 feature_mags for 12 feature_ids"):
        FeatureIndex(str(feature_dir), lazy_docs=True)._load_shard("shard_00000")


@pytest.mark.parametrize("field", ["segment_char_starts", "segment_char_ends"])
def test_short_char_bounds_fail_at_load_on_both_paths(tmp_path: Path, field: str) -> None:
    """Nothing else validates the char-bound lengths.

    Eager indexes bulk ``.tolist()`` copies by row, so a short array is an
    ``IndexError`` DURING ``_build_docs``. Without a load-time check the lazy
    mapping publishes happily and only fails on whichever document happens to
    gather the missing row — a fail-loud regression. Both paths must raise
    inside ``_load_shard``.
    """

    def mutate(data: dict) -> None:
        data[field] = data[field][:-1]

    feature_dir = _corrupt_shard_dir(tmp_path, mutate)
    with pytest.raises(IndexError):
        FeatureIndex(str(feature_dir), lazy_docs=False)._load_shard("shard_00000")
    with pytest.raises(
        RuntimeError, match=r"segment_char_starts and \d+ segment_char_ends for 4 segments"
    ):
        FeatureIndex(str(feature_dir), lazy_docs=True)._load_shard("shard_00000")


def test_one_sided_token_overlay_yields_none_on_both_ends(tmp_path: Path) -> None:
    """Eager decides token bounds by a conjunction (``_build_docs``); so must lazy.

    A half-present overlay is not constructible through ``_materialize_data``
    (both tensors are set together), so drive the mapping directly — this is the
    exact divergence the shared base class exists to prevent.
    """
    feature_dir = _build_family(tmp_path, n_shards=1)
    index = _index(feature_dir, lazy_docs=True)
    data = index._materialize_data("shard_00000")
    assert data["seg_token_starts"] is not None
    del data["seg_token_ends"]

    eager_docs = index._build_docs(dict(data), copy=True)
    lazy_docs = feature_index_module._LazyShardDocs(data)
    for doc in eager_docs:
        for actual, reference in zip(lazy_docs[doc], eager_docs[doc], strict=True):
            assert reference.token_start is None and reference.token_end is None
            assert actual.token_start is None and actual.token_end is None


def test_contains_and_getitem_agree_on_non_int_keys(tmp_path: Path) -> None:
    """``_bounds`` screens non-int keys, so membership and lookup cannot disagree."""
    feature_dir = _build_family(tmp_path, n_shards=1)
    docs = _index(feature_dir, lazy_docs=True)._load_shard("shard_00000")
    present = _DOC_IDS[0]
    assert present in docs
    for key in ("7", None, 3.5, float(present), (1 << 200)):
        assert key not in docs
        with pytest.raises(KeyError):
            docs[key]  # ty: ignore[invalid-argument-type]


def test_percentile_length_mismatch_is_rejected_before_publication(tmp_path: Path) -> None:
    """``validate_percentile_overlay_shard`` already enforces this upstream, so
    drive the mapping directly to prove the guard is real, not vacuous."""
    feature_dir = _build_family(tmp_path, n_shards=1)
    index = _index(feature_dir, lazy_docs=True)
    data = index._materialize_data("shard_00000")
    assert data["feature_percentiles"].numel() == data["feature_ids"].numel()
    data["feature_percentiles"] = t.cat([data["feature_percentiles"], t.zeros(2, dtype=t.uint8)])
    with pytest.raises(RuntimeError, match=r"feature_percentiles"):
        feature_index_module._LazyShardDocs(data)


# ---------------------------------------------------------------------------
# doc_ids_for_shard costs one index scan under lazy docs
# ---------------------------------------------------------------------------


def test_doc_ids_for_shard_matches_eager(tmp_path: Path, monkeypatch) -> None:
    feature_dir = _build_family(tmp_path, n_shards=2)
    eager = _index(feature_dir, lazy_docs=False)
    lazy = _index(feature_dir, lazy_docs=True)
    constructed: list[int] = []
    original_init = SegmentFeatures.__init__

    def counting_init(self_segment, *args, **kwargs):
        constructed.append(1)
        original_init(self_segment, *args, **kwargs)

    expected = eager.doc_ids_for_shard("shard_00001")
    monkeypatch.setattr(SegmentFeatures, "__init__", counting_init)
    assert lazy.doc_ids_for_shard("shard_00001") == expected
    assert constructed == []  # lazy: pure index scan, no doc build at all
