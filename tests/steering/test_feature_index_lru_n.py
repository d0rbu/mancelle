"""Bounded multi-shard LRU for FeatureIndex (issue #740)."""

import logging
import pickle
from pathlib import Path

import pytest

from steering.conditional.feature_index import FeatureIndex
from tests.steering._helpers import save_valid_shard


def _make_dir(tmp_path: Path, n: int) -> Path:
    # save_valid_shard(tmp_path, name, **make_valid_shard_data-kwargs) writes
    # <name>_features.safetensors.zst (stem == <name>) and returns (path, dict);
    # FeatureIndex discovers shards by globbing the dir, so the return is unused.
    d = tmp_path / "feat"
    d.mkdir()
    for i in range(n):
        save_valid_shard(d, f"shard_{i:05d}", n_docs=2, segs_per_doc=2, seed=i)
    return d


def _mru_stem(index: FeatureIndex) -> str | None:
    return next(reversed(index._resident)) if index._resident else None


def test_default_capacity_is_one(tmp_path: Path) -> None:
    idx = FeatureIndex(str(_make_dir(tmp_path, 3)))
    assert idx.max_resident_shards == 1


def test_invalid_capacity_raises(tmp_path: Path) -> None:
    d = _make_dir(tmp_path, 1)
    for bad in (0, -1, True, 1.0):
        with pytest.raises(ValueError):
            FeatureIndex(str(d), max_resident_shards=bad)  # ty: ignore[invalid-argument-type]


def test_capacity_two_keeps_both_resident(tmp_path: Path) -> None:
    idx = FeatureIndex(str(_make_dir(tmp_path, 3)), max_resident_shards=2)
    idx.lookup("shard_00000", 0)
    idx.lookup("shard_00001", 0)
    # both resident: a second lookup of 00000 is a hit, no reload
    before = idx.cache_stats.loads
    idx.lookup("shard_00000", 0)
    assert idx.cache_stats.loads == before
    assert idx.cache_stats.hits >= 1


def test_capacity_two_evicts_lru_on_third(tmp_path: Path) -> None:
    idx = FeatureIndex(str(_make_dir(tmp_path, 3)), max_resident_shards=2)
    idx.lookup("shard_00000", 0)
    idx.lookup("shard_00001", 0)
    idx.lookup("shard_00002", 0)  # evicts 00000 (LRU)
    # Exact counts (not >=): three cold lookups so far, all misses, exactly
    # one over-capacity admission -> exactly one eviction.
    assert idx.cache_stats.evictions == 1
    assert idx.cache_stats.hits == 0
    loads = idx.cache_stats.loads
    idx.lookup("shard_00000", 0)  # reload (evicts 00001)
    assert idx.cache_stats.loads == loads + 1
    assert idx.cache_stats.evictions == 2


def test_resident_order_tracks_mru(tmp_path: Path) -> None:
    idx = FeatureIndex(str(_make_dir(tmp_path, 3)), max_resident_shards=2)
    idx.lookup("shard_00000", 0)
    idx.lookup("shard_00001", 0)
    assert _mru_stem(idx) == "shard_00001"
    idx.lookup("shard_00000", 0)  # move_to_end makes 00000 MRU
    assert _mru_stem(idx) == "shard_00000"


def test_retain_drops_and_seeds(tmp_path: Path) -> None:
    idx = FeatureIndex(str(_make_dir(tmp_path, 4)), max_resident_shards=2)
    idx.lookup("shard_00003", 0)
    idx.retain(["shard_00000", "shard_00001"])
    # 00003 dropped; 00000/00001 resident (seeded up to capacity)
    loads = idx.cache_stats.loads
    idx.lookup("shard_00000", 0)
    idx.lookup("shard_00001", 0)
    assert idx.cache_stats.loads == loads  # both hits
    idx.lookup("shard_00003", 0)
    assert idx.cache_stats.loads == loads + 1  # dropped -> reload


def test_retain_warns_on_overflow(tmp_path: Path, caplog) -> None:
    idx = FeatureIndex(str(_make_dir(tmp_path, 4)), max_resident_shards=2)
    with caplog.at_level(logging.WARNING):
        idx.retain(["shard_00000", "shard_00001", "shard_00002"])
    assert any("exceeds max_resident_shards" in r.message for r in caplog.records)


def test_ensure_resident_bounds(tmp_path: Path) -> None:
    idx = FeatureIndex(str(_make_dir(tmp_path, 4)), max_resident_shards=2)
    for stem in ("shard_00000", "shard_00001", "shard_00002"):
        idx.ensure_resident(stem)
    # only 2 resident (capacity); 00000 evicted
    assert _mru_stem(idx) == "shard_00002"
    assert idx.cache_stats.evictions >= 1


# ---------------------------------------------------------------------------
# Hardening pass A: LRU-vs-FIFO, retain() dedup (R1), and _schedule_prefetch
# resident guard (R2).
# ---------------------------------------------------------------------------


def test_resident_state_is_empty_before_load(tmp_path: Path) -> None:
    idx = FeatureIndex(str(_make_dir(tmp_path, 2)))
    assert not idx._resident
    idx.lookup("shard_00000", 0)
    assert list(idx._resident) == ["shard_00000"]


def test_ensure_resident_unknown_stem_raises(tmp_path: Path) -> None:
    idx = FeatureIndex(str(_make_dir(tmp_path, 1)))
    with pytest.raises(KeyError, match="Unknown shard stem"):
        idx.ensure_resident("no_such_shard")


def test_ensure_resident_noop_on_pinned_shard(tmp_path: Path) -> None:
    idx = FeatureIndex(str(_make_dir(tmp_path, 2)), max_resident_shards=2)
    idx.pin_shard("shard_00000")
    loads = idx.cache_stats.loads
    idx.ensure_resident("shard_00000")  # pinned -> early return, no LRU admit
    assert idx.cache_stats.loads == loads
    assert "shard_00000" not in idx._resident


def test_ensure_resident_hit_on_already_resident_stem(tmp_path: Path) -> None:
    """ensure_resident() has its own inline 'already resident' hit-check
    (distinct physical lines from lookup()'s equivalent branch, which is
    covered separately) -- pin it directly: a second ensure_resident call on
    an already-resident, non-pinned stem must be a hit (no reload), not a
    silent re-admit."""
    idx = FeatureIndex(str(_make_dir(tmp_path, 2)), max_resident_shards=2)
    idx.ensure_resident("shard_00000")
    loads = idx.cache_stats.loads
    hits = idx.cache_stats.hits
    idx.ensure_resident("shard_00000")  # already resident -> hit, not reload
    assert idx.cache_stats.loads == loads
    assert idx.cache_stats.hits == hits + 1


def test_lru_not_fifo_reaccess_then_evict(tmp_path: Path) -> None:
    """Distinguishes true LRU from FIFO: re-accessing A between B/C and D must
    promote A to MRU so D's admission evicts B (the true LRU), not A (which a
    FIFO/insertion-order eviction policy would incorrectly pick)."""
    idx = FeatureIndex(str(_make_dir(tmp_path, 4)), max_resident_shards=3)
    idx.lookup("shard_00000", 0)  # A
    idx.lookup("shard_00001", 0)  # B
    idx.lookup("shard_00002", 0)  # C -- resident order: [A, B, C], A is LRU
    idx.lookup("shard_00000", 0)  # re-access A -> MRU; order: [B, C, A]
    loads = idx.cache_stats.loads
    idx.lookup("shard_00003", 0)  # D -> must evict B (true LRU), not A
    assert idx.cache_stats.loads == loads + 1  # D loaded

    loads = idx.cache_stats.loads
    idx.lookup("shard_00001", 0)  # B: evicted -> must reload
    assert idx.cache_stats.loads == loads + 1

    loads = idx.cache_stats.loads
    idx.lookup("shard_00000", 0)  # A: still resident -> must NOT reload
    assert idx.cache_stats.loads == loads


def test_fresh_pickle_roundtrip_cap_gt_one(tmp_path: Path) -> None:
    idx = FeatureIndex(str(_make_dir(tmp_path, 3)), max_resident_shards=3)
    idx.lookup("shard_00000", 0)
    idx.lookup("shard_00001", 0)
    idx.lookup("shard_00002", 0)  # MRU order: [00000, 00001, 00002]
    loads_before = idx.cache_stats.loads
    hits_before = idx.cache_stats.hits
    state = idx.__getstate__()
    assert {
        "_resident",
        "_max_resident_shards",
        "_stats",
        "_required_token_boundary_stems",
    } <= state.keys()
    assert {"_cached_stem", "_cached_docs"}.isdisjoint(state)

    idx2 = pickle.loads(pickle.dumps(idx))

    assert idx2.max_resident_shards == 3
    assert list(idx2._resident.keys()) == ["shard_00000", "shard_00001", "shard_00002"]
    assert _mru_stem(idx2) == "shard_00002"  # MRU order preserved
    assert idx2.cache_stats.loads == loads_before
    assert idx2.cache_stats.hits == hits_before

    # Every resident stem must still be a hit post-unpickle (no reload).
    loads_after_pickle = idx2.cache_stats.loads
    idx2.lookup("shard_00000", 0)
    idx2.lookup("shard_00001", 0)
    idx2.lookup("shard_00002", 0)
    assert idx2.cache_stats.loads == loads_after_pickle


def test_retain_dedups_stems(tmp_path: Path, caplog) -> None:
    """Proves R1: retain() must dedup stems BEFORE slicing/counting against
    max_resident_shards. Before the fix, retain(["A","A","B"]) at capacity 2
    sliced the RAW list to ["A","A"] (seeding A twice, never seeding B) and
    compared len(["A","A","B"])=3 > 2 to fire a false overflow warning despite
    only 2 distinct shards."""
    idx = FeatureIndex(str(_make_dir(tmp_path, 2)), max_resident_shards=2)
    with caplog.at_level(logging.WARNING):
        idx.retain(["shard_00000", "shard_00000", "shard_00001"])

    loads = idx.cache_stats.loads
    idx.lookup("shard_00000", 0)
    idx.lookup("shard_00001", 0)
    assert idx.cache_stats.loads == loads, (
        "both distinct stems must have been seeded by retain() -> hits only"
    )
    assert not any("exceeds max_resident_shards" in r.message for r in caplog.records), (
        "2 distinct shards at capacity 2 must never warn about exceeding capacity"
    )


def test_pin_shard_plus_retain(tmp_path: Path, caplog) -> None:
    idx = FeatureIndex(str(_make_dir(tmp_path, 2)), max_resident_shards=2)
    idx.pin_shard("shard_00000")
    with caplog.at_level(logging.WARNING):
        idx.retain(["shard_00000", "shard_00001"])
    assert "shard_00000" in idx._pinned
    loads = idx.cache_stats.loads
    idx.lookup("shard_00001", 0)
    assert idx.cache_stats.loads == loads  # seeded by retain -> hit
    assert not any("exceeds max_resident_shards" in r.message for r in caplog.records)


def test_retain_leaves_last_seeded_stem_mru(tmp_path: Path) -> None:
    idx = FeatureIndex(str(_make_dir(tmp_path, 2)), max_resident_shards=2)
    idx.retain(["shard_00000", "shard_00001"])
    assert _mru_stem(idx) == "shard_00001"


def test_schedule_prefetch_skips_resident_stem(tmp_path: Path) -> None:
    """Proves R2: the "already cached" guard in _schedule_prefetch must catch
    ANY resident (or pinned) shard, not just the MRU one. Before the fix, the
    guard only checked the single MRU entry, so a shard that
    is resident-but-not-MRU under capacity > 1 was NOT recognized and a
    redundant background reload was spawned.

    Arranged directly against ``_schedule_prefetch`` (per the task brief's
    documented fallback) rather than via natural lookup-driven timing:
    landing "next_stem resident but not MRU" through real lookups requires
    racing a background prefetch thread deterministically (the prefetch
    worker never itself admits into ``_resident``; only a subsequent miss-path
    admit does, and any such admit would make that stem the new MRU, not a
    non-MRU resident). Directly seeding ``_resident`` sidesteps that race and
    isolates exactly the guard condition under test.
    """
    idx = FeatureIndex(
        str(_make_dir(tmp_path, 3)), max_resident_shards=2, prefetch=True
    )
    # sorted_stems == [shard_00000, shard_00001, shard_00002], so the stem
    # _schedule_prefetch("shard_00000") predicts as "next" is shard_00001.
    idx._resident["shard_00001"] = {0: []}  # resident, LRU (not MRU)
    idx._resident["shard_00000"] = {0: []}  # resident, MRU
    assert _mru_stem(idx) == "shard_00000"
    assert "shard_00001" in idx._resident

    idx._schedule_prefetch("shard_00000")

    assert idx._prefetch_thread is None, (
        "R2 regression: _schedule_prefetch spawned a redundant background load "
        "for shard_00001, which was already resident (just not MRU)"
    )
    idx.close()


def test_schedule_prefetch_skips_pinned_stem(tmp_path: Path) -> None:
    """R2's guard also covers pinned shards (``stem in self._pinned``),
    which the old MRU-only check never recognized either."""
    idx = FeatureIndex(
        str(_make_dir(tmp_path, 3)), max_resident_shards=2, prefetch=True
    )
    idx.pin_shard("shard_00001")  # predicted "next" of shard_00000
    idx._resident["shard_00000"] = {0: []}
    assert _mru_stem(idx) == "shard_00000"

    idx._schedule_prefetch("shard_00000")

    assert idx._prefetch_thread is None, (
        "R2 regression: _schedule_prefetch spawned a redundant background load "
        "for shard_00001, which was pinned"
    )
    idx.close()
