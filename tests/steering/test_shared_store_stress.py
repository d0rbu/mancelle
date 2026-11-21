"""Concurrency stress for the shared /dev/shm store (issue #740 PR-F): many spawn workers hammer
a store whose capacity is SMALLER than the working set, forcing constant eviction + re-materialize
churn (as real shards referencing more source stems than the cache capacity do). Asserts every
worker survives (no segfault / abrupt death) AND returns byte-identical results — the eviction
path (atomic rename-aside + unlink-while-mapped + the materialize lock) must be correct under real
cross-process contention, not just in the single-process unit tests."""

from __future__ import annotations

import faulthandler
import multiprocessing

import torch as t

from steering.conditional.feature_index import FeatureIndex
from steering.conditional.shared_stem_store import SharedStemStore
from steering.feature_io import save_features
from tests.steering._splice_fixtures import make_base_shard
from tests.steering._splice_test_tokenizers import CharMergeTokenizer

_SEG = 2
_N_STEMS = 12
_CAPACITY = 4       # << N_STEMS ⇒ the store thrashes (constant eviction)
_N_WORKERS = 4
_ITERS = 12


def _stress_worker(args) -> str:
    faulthandler.enable()
    feature_dir, shm_dir, wid = args
    store = SharedStemStore(shm_dir, capacity=_CAPACITY)
    idx = FeatureIndex(feature_dir, compact_feature_ids=True, max_resident_shards=3,
                       shared_store=store)
    stems = sorted(idx.shard_stems)
    total = 0
    for it in range(_ITERS):
        rot = (wid + it) % len(stems)
        for stem in stems[rot:] + stems[:rot]:  # overlapping, rotated ⇒ cross-worker eviction
            for doc in range(3):
                try:
                    segs = idx.lookup(stem, doc)
                except KeyError:
                    continue
                for s in segs:  # TOUCH shared views: an evicted-mapping segfault would fire here
                    total += int(s.feature_ids.to(t.int64).sum().item())
    return f"{total}"


def test_shared_store_survives_concurrent_eviction_churn(tmp_path):
    base = tmp_path / "features" / f"s{_SEG}" / "base"
    base.mkdir(parents=True)
    tok = CharMergeTokenizer(bos_id=500)
    for i in range(_N_STEMS):
        texts = [f"word{i}_{j} tok{i}_{j} chunk{i}_{j} piece{i}_{j}" for j in range(24)]
        save_features(make_base_shard(texts, tok, _SEG), base / f"shard_{i:05d}")
    shm = tmp_path / "shm"
    shm.mkdir()

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(_N_WORKERS) as pool:
        results = pool.map(
            _stress_worker, [(str(base), str(shm), w) for w in range(_N_WORKERS)]
        )
    # all workers survived (map would raise BrokenProcessPool on a crash) AND agree byte-for-byte
    assert len(results) == _N_WORKERS
    assert len(set(results)) == 1, f"workers disagree under churn: {results}"
