"""Bounded multi-shard LRU for SpliceProvenanceIndex (issue #740)."""

import logging
from pathlib import Path

import pytest
from beartype.roar import BeartypeCallHintParamViolation

from steering.conditional.splice_provenance import SpliceProvenanceIndex
from tests.steering._splice_cf_cache_fixtures import _TEXTS_BY_SHARD as _MULTI_SHARD_TEXTS
from tests.steering._splice_cf_fixtures import build_derived_env


def _index(tmp_path: Path, cap: int) -> SpliceProvenanceIndex:
    env = build_derived_env(tmp_path, cut_fraction=0.6)
    return SpliceProvenanceIndex(Path(env["feature_dir"]), max_resident_shards=cap)


def _index_n(tmp_path: Path, cap: int, n_shards: int) -> SpliceProvenanceIndex:
    """Like _index but over ``n_shards`` sidecars (up to 7) instead of the
    default env's 2, reusing the proven-working multi-shard corpus from
    _splice_cf_cache_fixtures (cheaper than build_interleaved_env: no packed
    rows / overlays / statistics, just the derived shards + v2 sidecars
    build_derived_env already produces)."""
    env = build_derived_env(
        tmp_path, cut_fraction=0.6, texts_by_shard=_MULTI_SHARD_TEXTS[:n_shards]
    )
    return SpliceProvenanceIndex(Path(env["feature_dir"]), max_resident_shards=cap)


def _mru_stem(index: SpliceProvenanceIndex) -> str | None:
    return next(reversed(index._resident)) if index._resident else None


def test_default_capacity_one(tmp_path: Path) -> None:
    assert _index(tmp_path, 1).max_resident_shards == 1


def test_invalid_capacity_raises(tmp_path: Path) -> None:
    env = build_derived_env(tmp_path, cut_fraction=0.6)
    # 1.0 added beyond the brief's (0, -1, True) to mirror the FeatureIndex
    # LRU-N suite's coverage of the "int, not merely int-valued" contract.
    for bad in (0, -1, True, 1.0):
        with pytest.raises(
            BeartypeCallHintParamViolation,
            match="max_resident_shards",
        ):
            SpliceProvenanceIndex(Path(env["feature_dir"]), max_resident_shards=bad)  # ty: ignore[invalid-argument-type]


def test_capacity_two_avoids_reparse(tmp_path: Path) -> None:
    idx = _index(tmp_path, 2)
    idx.shard_n_docs("shard_00000")
    idx.shard_n_docs("shard_00001")
    parses = idx.cache_stats.loads
    idx.shard_n_docs("shard_00000")  # resident hit — no reparse
    assert idx.cache_stats.loads == parses
    assert idx.cache_stats.hits >= 1


def test_capacity_one_reparses_on_switch(tmp_path: Path) -> None:
    idx = _index(tmp_path, 1)
    idx.shard_n_docs("shard_00000")
    idx.shard_n_docs("shard_00001")
    parses = idx.cache_stats.loads
    idx.shard_n_docs("shard_00000")  # capacity 1 → reparse
    assert idx.cache_stats.loads == parses + 1


def test_retain_and_drop(tmp_path: Path) -> None:
    idx = _index(tmp_path, 2)
    idx.shard_n_docs("shard_00001")
    idx.retain(["shard_00000"])
    parses = idx.cache_stats.loads
    idx.shard_n_docs("shard_00000")  # seeded by retain → hit
    assert idx.cache_stats.loads == parses


# -- extra coverage beyond the brief's Step-1 minimum, mirroring
#    tests/steering/test_feature_index_lru_n.py for the sibling class --


def test_ensure_resident_bounds(tmp_path: Path) -> None:
    idx = _index(tmp_path, 1)
    idx.ensure_resident("shard_00000")
    idx.ensure_resident("shard_00001")  # evicts shard_00000 (capacity 1)
    assert _mru_stem(idx) == "shard_00001"
    assert idx.cache_stats.evictions >= 1


def test_drop_except_evicts_non_kept(tmp_path: Path) -> None:
    idx = _index(tmp_path, 2)
    idx.shard_n_docs("shard_00000")
    idx.shard_n_docs("shard_00001")
    idx.drop_except(["shard_00001"])
    assert idx.cache_stats.evictions >= 1
    parses = idx.cache_stats.loads
    idx.shard_n_docs("shard_00001")  # still resident → hit
    assert idx.cache_stats.loads == parses
    idx.shard_n_docs("shard_00000")  # dropped → reparse
    assert idx.cache_stats.loads == parses + 1


def test_resident_order_tracks_mru(tmp_path: Path) -> None:
    idx = _index(tmp_path, 2)
    n0 = idx.shard_n_docs("shard_00000")
    n1 = idx.shard_n_docs("shard_00001")
    assert _mru_stem(idx) == "shard_00001"
    assert idx._resident["shard_00001"].n_docs == n1
    idx.shard_n_docs("shard_00000")  # move_to_end -> MRU becomes shard_00000
    assert _mru_stem(idx) == "shard_00000"
    assert idx._resident["shard_00000"].n_docs == n0


def test_resident_state_is_empty_before_any_load(tmp_path: Path) -> None:
    idx = _index(tmp_path, 2)
    assert not idx._resident


def test_retain_increments_retain_calls_stat(tmp_path: Path) -> None:
    idx = _index(tmp_path, 2)
    idx.retain(["shard_00000"])
    idx.retain(["shard_00001"])
    assert idx.cache_stats.retain_calls == 2


def test_retain_warns_on_overflow(tmp_path: Path, caplog) -> None:
    idx = _index(tmp_path, 1)
    with caplog.at_level(logging.WARNING):
        idx.retain(["shard_00000", "shard_00001"])
    assert any("exceeds max_resident_shards" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Hardening pass A: LRU-vs-FIFO pin + retain() dedup (R1), mirroring the
# FeatureIndex LRU-N suite for this sibling class.
# ---------------------------------------------------------------------------


def test_lru_not_fifo_reaccess_then_evict(tmp_path: Path) -> None:
    """Distinguishes true LRU from FIFO: re-parsing A between B/C and D must
    promote A to MRU so D's admission evicts B (the true LRU), not A."""
    idx = _index_n(tmp_path, cap=3, n_shards=4)
    idx.shard_n_docs("shard_00000")  # A
    idx.shard_n_docs("shard_00001")  # B
    idx.shard_n_docs("shard_00002")  # C -- resident order: [A, B, C], A is LRU
    idx.shard_n_docs("shard_00000")  # re-access A -> MRU; order: [B, C, A]
    parses = idx.cache_stats.loads
    idx.shard_n_docs("shard_00003")  # D -> must evict B (true LRU), not A
    assert idx.cache_stats.loads == parses + 1

    parses = idx.cache_stats.loads
    idx.shard_n_docs("shard_00001")  # B: evicted -> must reparse
    assert idx.cache_stats.loads == parses + 1

    parses = idx.cache_stats.loads
    idx.shard_n_docs("shard_00000")  # A: still resident -> must NOT reparse
    assert idx.cache_stats.loads == parses


def test_retain_dedups_stems(tmp_path: Path, caplog) -> None:
    """Proves R1 for SpliceProvenanceIndex (mirrors the FeatureIndex proof):
    retain() must dedup stems BEFORE slicing/counting against
    max_resident_shards, or a duplicate-laden call at capacity silently
    under-seeds the working set and fires a false overflow warning."""
    idx = _index(tmp_path, 2)
    with caplog.at_level(logging.WARNING):
        idx.retain(["shard_00000", "shard_00000", "shard_00001"])

    parses = idx.cache_stats.loads
    idx.shard_n_docs("shard_00000")
    idx.shard_n_docs("shard_00001")
    assert idx.cache_stats.loads == parses, (
        "both distinct stems must have been seeded by retain() -> hits only"
    )
    assert not any("exceeds max_resident_shards" in r.message for r in caplog.records), (
        "2 distinct shards at capacity 2 must never warn about exceeding capacity"
    )
