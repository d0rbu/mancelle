"""Byte-identity of FeatureIndex shared-store mode vs the normal load path (issue #740 PR-F
increment 2): shared-mode lookup() must return per-segment identical SegmentFeatures, built as
zero-copy views into one /dev/shm copy per stem."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch as t

import steering.conditional.feature_index as feature_index_module
from steering.conditional.feature_index import FeatureIndex
from steering.conditional.shared_stem_store import SharedStemStore
from steering.feature_io import save_features
from tests.steering._splice_fixtures import make_base_shard
from tests.steering._splice_test_tokenizers import CharMergeTokenizer

SEG = 2
TEXTS = [
    ["alpha beta gamma", "delta epsilon zeta", "eta theta iota", "kappa lambda mu"],
    ["nu xi omicron", "pi rho sigma", "tau upsilon phi"],
    ["chi psi omega", "ab cd ef gh", "ij kl mn op", "qr st uv wx", "yz zy xw"],
]


def _build_dir(tmp_path: Path) -> Path:
    base = tmp_path / "features" / f"s{SEG}" / "base"
    base.mkdir(parents=True)
    tok = CharMergeTokenizer(bos_id=500)
    for i, texts in enumerate(TEXTS):
        shard = make_base_shard(texts, tok, SEG)
        save_features(shard, base / f"shard_{i:05d}")
    return base


def _assert_segments_identical(a_segs, b_segs) -> None:
    assert len(a_segs) == len(b_segs)
    for a, b in zip(a_segs, b_segs, strict=True):
        assert a.segment_idx == b.segment_idx
        assert a.position == b.position
        assert a.char_start == b.char_start and a.char_end == b.char_end
        assert a.token_start == b.token_start and a.token_end == b.token_end
        assert a.feature_ids.dtype == b.feature_ids.dtype
        assert t.equal(a.feature_ids, b.feature_ids)
        assert a.feature_mags.dtype == b.feature_mags.dtype
        assert t.equal(a.feature_mags, b.feature_mags)
        if a.feature_percentiles is None:
            assert b.feature_percentiles is None
        else:
            assert t.equal(a.feature_percentiles, b.feature_percentiles)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("skip_mags", [False, True])
def test_shared_lookup_byte_identical(tmp_path, compact, skip_mags):
    feature_dir = _build_dir(tmp_path)
    normal = FeatureIndex(
        str(feature_dir), skip_feature_mags=skip_mags, compact_feature_ids=compact,
        max_resident_shards=4,
    )
    shared = FeatureIndex(
        str(feature_dir), skip_feature_mags=skip_mags, compact_feature_ids=compact,
        max_resident_shards=4, shared_store=SharedStemStore(tmp_path / "shm"),
    )
    for stem in sorted(normal.shard_stems):
        # every doc id present in this shard
        n_texts = len(TEXTS[int(stem.split("_")[1])])
        for doc in range(n_texts):
            _assert_segments_identical(normal.lookup(stem, doc), shared.lookup(stem, doc))


def test_second_index_gets_shared_hit(tmp_path):
    """A second FeatureIndex over the same store must HIT (no re-materialize) and stay identical
    — the multi-worker case, where one worker materializes and the rest alias."""
    feature_dir = _build_dir(tmp_path)
    store = SharedStemStore(tmp_path / "shm")
    a = FeatureIndex(str(feature_dir), compact_feature_ids=True, max_resident_shards=4,
                     shared_store=store)
    stem = sorted(a.shard_stems)[0]
    a.lookup(stem, 0)  # materializes
    assert store.has(stem)
    b = FeatureIndex(str(feature_dir), compact_feature_ids=True, max_resident_shards=4,
                     shared_store=store)
    b.lookup(stem, 0)  # must be a shared hit
    assert getattr(b._stats, "shared_hits", 0) >= 1
    _assert_segments_identical(a.lookup(stem, 0), b.lookup(stem, 0))


def test_shared_segments_alias_one_copy(tmp_path):
    """The shared-mode segment feature_ids are VIEWS into the /dev/shm buffer: two indices over
    the same store share storage (same data_ptr for the same segment)."""
    feature_dir = _build_dir(tmp_path)
    store = SharedStemStore(tmp_path / "shm")
    a = FeatureIndex(str(feature_dir), compact_feature_ids=True, max_resident_shards=4,
                     shared_store=store)
    b = FeatureIndex(str(feature_dir), compact_feature_ids=True, max_resident_shards=4,
                     shared_store=store)
    stem = sorted(a.shard_stems)[0]
    sa = a.lookup(stem, 0)[0]
    sb = b.lookup(stem, 0)[0]
    # both are views into a mapping of the same underlying file; identical bytes
    assert t.equal(sa.feature_ids, sb.feature_ids)
    # a view has a nonzero-length storage larger than (or equal to) the slice
    assert sa.feature_ids.untyped_storage().nbytes() >= sa.feature_ids.numel() * sa.feature_ids.element_size()


@pytest.mark.parametrize("skip_mags", [False, True])
def test_lazy_shared_lookup_matches_eager_for_every_stem_and_document(
    tmp_path: Path,
    skip_mags: bool,
) -> None:
    feature_dir = _build_dir(tmp_path)
    eager = FeatureIndex(
        str(feature_dir),
        skip_feature_mags=skip_mags,
        compact_feature_ids=True,
        max_resident_shards=3,
        shared_store=SharedStemStore(tmp_path / "shm-eager"),
    )
    lazy = FeatureIndex(
        str(feature_dir),
        skip_feature_mags=skip_mags,
        compact_feature_ids=True,
        max_resident_shards=1,
        shared_store=SharedStemStore(tmp_path / "shm-lazy"),
        lazy_shared_docs=True,
    )

    for stem in eager.shard_stems:
        expected_doc_ids = eager.doc_ids_for_shard(stem)
        assert lazy.doc_ids_for_shard(stem) == expected_doc_ids
        for doc_id in expected_doc_ids:
            expected = eager.lookup(stem, doc_id)
            actual = lazy.lookup(stem, doc_id)
            _assert_segments_identical(expected, actual)
            assert actual is not lazy.lookup(stem, doc_id)
            assert actual[0].feature_ids.data_ptr() == lazy.lookup(stem, doc_id)[0].feature_ids.data_ptr()


def test_lazy_shared_document_locator_preserves_stable_eager_order(
    tmp_path: Path,
) -> None:
    index = FeatureIndex.__new__(FeatureIndex)
    index._initialize_empty(
        prefetch=False,
        percentile_checksum=None,
        tokenizer_checksum=None,
        skip_feature_mags=False,
        compact_feature_ids=True,
        max_resident_shards=1,
        shared_store=SharedStemStore(tmp_path / "shm-locator"),
        lazy_shared_docs=True,
        required_token_boundary_stems=None,
    )
    data = {
        "n_segments": 4,
        "offsets": t.tensor([0, 1, 3, 4, 6], dtype=t.int64),
        "doc_ids": t.tensor([5, 3, 5, 3], dtype=t.int64),
        "segment_indices": t.tensor([1, 1, 0, 0], dtype=t.int64),
        "segment_positions": t.tensor([11, 31, 10, 30], dtype=t.int64),
        "segment_char_starts": t.tensor([110, 310, 100, 300], dtype=t.int64),
        "segment_char_ends": t.tensor([119, 319, 109, 309], dtype=t.int64),
        "feature_ids": t.tensor([51, 31, 32, 50, 30, 33], dtype=t.uint16),
        "feature_mags": t.tensor([5.1, 3.1, 3.2, 5.0, 3.0, 3.3]),
        "feature_percentiles": t.tensor([51, 31, 32, 50, 30, 33], dtype=t.uint8),
        "seg_token_starts": t.tensor([111, 311, 101, 301], dtype=t.int64),
        "seg_token_ends": t.tensor([112, 312, 102, 302], dtype=t.int64),
    }
    index._attach_shared_doc_index(data)
    eager = index._build_docs(data, copy=False)
    lazy = feature_index_module._SharedLazyDocs(data)

    assert len(lazy) == len(eager) == 2
    assert list(lazy) == sorted(eager) == [3, 5]
    assert 3 in lazy and 4 not in lazy
    assert (1 << 100) not in lazy and -(1 << 100) not in lazy
    with pytest.raises(KeyError, match=str(1 << 100)):
        lazy[1 << 100]
    for doc_id in eager:
        _assert_segments_identical(eager[doc_id], lazy[doc_id])
        assert [segment.segment_idx for segment in lazy[doc_id]] == [0, 1]


def test_lazy_shared_lru1_interleaving_never_rebuilds_eager_docs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = _build_dir(tmp_path)
    reference = FeatureIndex(
        str(feature_dir), compact_feature_ids=True, max_resident_shards=3
    )
    expected = {
        (stem, doc_id): reference.lookup(stem, doc_id)
        for stem in reference.shard_stems
        for doc_id in range(len(TEXTS[int(stem.split("_")[1])]))
    }
    lazy = FeatureIndex(
        str(feature_dir),
        compact_feature_ids=True,
        max_resident_shards=1,
        shared_store=SharedStemStore(tmp_path / "shm", capacity=3),
        lazy_shared_docs=True,
    )

    def forbidden_eager_build(*_args, **_kwargs):
        raise AssertionError("lazy shared mode rebuilt the eager full-stem object graph")

    monkeypatch.setattr(lazy, "_build_docs", forbidden_eager_build)
    sequence = [
        ("shard_00000", 0),
        ("shard_00001", 0),
        ("shard_00000", 1),
        ("shard_00002", 0),
        ("shard_00001", 1),
        ("shard_00002", 1),
    ] * 2
    for key in sequence:
        _assert_segments_identical(expected[key], lazy.lookup(*key))

    # Only the three cold store publications read/decode source artifacts.
    # The nine later LRU switches are pure shared remaps of the published raw
    # tensors + locator, even though the FeatureIndex resident map is size one.
    assert lazy.cache_stats.loads == 3
    assert lazy.cache_stats.materializations == 3
    assert lazy.cache_stats.publications == 3
    assert lazy.cache_stats.shared_hits == len(sequence)
    assert lazy.cache_stats.evictions == len(sequence) - 1


def test_eager_shared_lru1_interleaving_rebuilds_full_docs_on_every_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Causal oracle for issue #758's production slowdown.

    The raw shared tensors are decoded only once per stem, but the pre-fix
    eager facade reconstructs every segment object after each LRU-1 remap.
    Keep this contrast explicit so the lazy regression cannot pass merely by
    changing cache counters or source I/O.
    """
    feature_dir = _build_dir(tmp_path)
    index = FeatureIndex(
        str(feature_dir),
        compact_feature_ids=True,
        max_resident_shards=1,
        shared_store=SharedStemStore(tmp_path / "shm", capacity=3),
    )
    real_build = index._build_docs
    builds = 0

    def counted_build(data: dict, *, copy: bool):
        nonlocal builds
        builds += 1
        return real_build(data, copy=copy)

    monkeypatch.setattr(index, "_build_docs", counted_build)
    sequence = [
        ("shard_00000", 0),
        ("shard_00001", 0),
        ("shard_00000", 1),
        ("shard_00002", 0),
        ("shard_00001", 1),
        ("shard_00002", 1),
    ] * 2
    for stem, doc_id in sequence:
        assert index.lookup(stem, doc_id)

    assert builds == len(sequence)
    assert index.cache_stats.loads == 3
    assert index.cache_stats.materializations == 3
    assert index.cache_stats.shared_hits == len(sequence)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lazy_shared_docs": True}, "requires a shared_store"),
        (
            {
                "shared_store": "store",
                "lazy_shared_docs": True,
                "compact_feature_ids": False,
            },
            "requires compact_feature_ids=True",
        ),
        (
            {
                "shared_store": "store",
                "lazy_shared_docs": True,
                "compact_feature_ids": True,
                "prefetch": True,
            },
            "does not support background prefetch",
        ),
    ],
)
def test_lazy_shared_mode_rejects_unsupported_contracts(
    tmp_path: Path,
    kwargs: dict,
    message: str,
) -> None:
    feature_dir = _build_dir(tmp_path)
    resolved = dict(kwargs)
    if resolved.get("shared_store") == "store":
        resolved["shared_store"] = SharedStemStore(tmp_path / "shm")
    with pytest.raises(ValueError, match=message):
        FeatureIndex(str(feature_dir), **resolved)


def test_lazy_shared_mode_fails_loud_on_store_without_locator(tmp_path: Path) -> None:
    feature_dir = _build_dir(tmp_path)
    store = SharedStemStore(tmp_path / "shm")
    eager = FeatureIndex(
        str(feature_dir),
        compact_feature_ids=True,
        shared_store=store,
    )
    eager.lookup("shard_00000", 0)
    lazy = FeatureIndex(
        str(feature_dir),
        compact_feature_ids=True,
        shared_store=store,
        lazy_shared_docs=True,
    )

    with pytest.raises(RuntimeError, match="missing its int64 document-order tensor"):
        lazy.lookup("shard_00000", 0)


@pytest.mark.parametrize("capacity", [1, 2])
def test_shared_lookup_identical_under_tiny_capacity(tmp_path, capacity):
    """Even when the store capacity is smaller than the working set (constant eviction + churn),
    shared-mode lookups stay byte-identical to the normal path — the eviction race + fallback are
    correctness-preserving."""
    feature_dir = _build_dir(tmp_path)
    normal = FeatureIndex(str(feature_dir), compact_feature_ids=True, max_resident_shards=4)
    shared = FeatureIndex(
        str(feature_dir), compact_feature_ids=True, max_resident_shards=4,
        shared_store=SharedStemStore(tmp_path / "shm", capacity=capacity),
    )
    stems = sorted(normal.shard_stems)
    # interleave stems so the store thrashes below capacity
    for _round in range(2):
        for stem in stems:
            n_texts = len(TEXTS[int(stem.split("_")[1])])
            for doc in range(n_texts):
                _assert_segments_identical(normal.lookup(stem, doc), shared.lookup(stem, doc))


def test_cross_process_shared_lookup_identical(tmp_path):
    """A separate process that maps the same store produces identical lookups (spawn workers)."""
    feature_dir = _build_dir(tmp_path)
    shm = tmp_path / "shm"
    # materialize in this process first
    seed = FeatureIndex(str(feature_dir), compact_feature_ids=True, max_resident_shards=4,
                        shared_store=SharedStemStore(shm))
    ref = seed.lookup("shard_00000", 0)
    ref_ids = [s.feature_ids.tolist() for s in ref]
    code = (
        "import torch as t;"
        "from steering.conditional.feature_index import FeatureIndex;"
        "from steering.conditional.shared_stem_store import SharedStemStore;"
        f"idx=FeatureIndex({str(feature_dir)!r}, compact_feature_ids=True, "
        f"max_resident_shards=4, shared_store=SharedStemStore({str(shm)!r}));"
        "segs=idx.lookup('shard_00000', 0);"
        "import json,sys;print(json.dumps([s.feature_ids.tolist() for s in segs]))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=180, check=False)
    assert out.returncode == 0, out.stderr
    import json
    assert json.loads(out.stdout.strip().splitlines()[-1]) == ref_ids


def test_skipping_store_materializes_once_not_thrice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex review #3: when the tmpfs space guard declines to publish, _shared_get_or_materialize
    must build the private copy EXACTLY ONCE per load — not 2-3x (the old fallback re-materialized,
    peaking at multiple private stem copies per worker, which can itself OOM)."""
    feature_dir = _build_dir(tmp_path)
    store = SharedStemStore(tmp_path / "shm", min_free_bytes=1 << 60)  # guard always declines
    idx = FeatureIndex(str(feature_dir), compact_feature_ids=True, max_resident_shards=4,
                       shared_store=store)
    calls = {"n": 0}
    orig = idx._materialize_data

    def counting(stem):
        calls["n"] += 1
        return orig(stem)

    monkeypatch.setattr(idx, "_materialize_data", counting)
    segs = idx.lookup("shard_00000", 0)  # single fresh load of this stem
    assert calls["n"] == 1, f"materialized {calls['n']}x (expected exactly 1)"
    assert not store.has("shard_00000")  # space guard really declined to publish
    # and the private-fallback result is byte-identical to the normal path
    normal = FeatureIndex(str(feature_dir), compact_feature_ids=True, max_resident_shards=4)
    _assert_segments_identical(segs, normal.lookup("shard_00000", 0))


def test_compact_narrows_to_uint16_for_65k_range_ids(tmp_path):
    """PR-H: a compact index whose feature_ids all fit uint16 (65k SAE) materializes them as
    uint16 (½ the int32 bytes), byte-identical in value to the non-compact int64 path."""
    feature_dir = _build_dir(tmp_path)  # make_base_shard ⇒ ids well under 65536
    compact = FeatureIndex(str(feature_dir), compact_feature_ids=True, max_resident_shards=4)
    plain = FeatureIndex(str(feature_dir), compact_feature_ids=False, max_resident_shards=4)
    seg_c = compact.lookup("shard_00000", 0)[0]
    seg_p = plain.lookup("shard_00000", 0)[0]
    assert seg_c.feature_ids.dtype == t.uint16
    assert seg_p.feature_ids.dtype == t.int64
    assert t.equal(seg_c.feature_ids.to(t.int64), seg_p.feature_ids)  # same values


def test_compact_keeps_int32_when_an_id_exceeds_uint16(tmp_path):
    """The uint16 narrowing is GUARDED by the actual max: a shard with an id >= 65536 (wider SAE)
    stays int32 — never a silent uint16 overflow."""
    tok = CharMergeTokenizer(bos_id=500)
    shard = make_base_shard(["alpha beta gamma", "delta epsilon"], tok, SEG, n_features=200000)
    shard["feature_ids"][0] = 70000  # force one id beyond uint16 range
    base = tmp_path / "features" / f"s{SEG}" / "base"
    base.mkdir(parents=True)
    save_features(shard, base / "shard_00000")
    idx = FeatureIndex(str(base), compact_feature_ids=True, max_resident_shards=2)
    segs = idx.lookup("shard_00000", 0)
    assert any(s.feature_ids.numel() for s in segs)
    assert all(s.feature_ids.dtype == t.int32 for s in segs)  # guarded ⇒ int32, not uint16
    assert 70000 in [int(v) for s in segs for v in s.feature_ids.to(t.int64)]


def test_compact_keeps_int32_when_an_id_is_negative(tmp_path, monkeypatch):
    """The lossless guard covers both ends of the uint16 interval."""
    feature_dir = _build_dir(tmp_path)
    idx = FeatureIndex(
        str(feature_dir),
        compact_feature_ids=True,
        max_resident_shards=2,
    )
    real_load = feature_index_module.load_features

    def load_with_negative_id(path, *, keys=None):
        loaded = real_load(path, keys=keys)
        loaded["feature_ids"][0] = -1
        return loaded

    monkeypatch.setattr(feature_index_module, "load_features", load_with_negative_id)
    docs = idx._load_shard(idx.shard_stems[0])
    segments = [segment for doc in docs.values() for segment in doc]

    assert segments
    assert all(segment.feature_ids.dtype == t.int32 for segment in segments)
    assert any(-1 in segment.feature_ids.tolist() for segment in segments)
