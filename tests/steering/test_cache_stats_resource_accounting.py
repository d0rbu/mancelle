"""Strict accounting transitions for source/shared/sidecar cache diagnostics."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
import torch as t

from steering.conditional.feature_index import CacheStats, FeatureIndex
from steering.conditional.shared_stem_store import SharedStemStore
from steering.conditional.splice_provenance import SpliceProvenanceIndex
from tests.steering._splice_cf_fixtures import build_derived_env
from tests.steering.test_feature_index_shared_store import _build_dir


def test_cache_stats_snapshot_has_fixed_complete_schema() -> None:
    stats = CacheStats()
    assert stats.snapshot() == {
        "loads": 0,
        "hits": 0,
        "evictions": 0,
        "retain_calls": 0,
        "load_seconds": 0.0,
        "bytes_read": 0,
        "shared_hits": 0,
        "shared_misses": 0,
        "shared_get_races": 0,
        "materializations": 0,
        "materialize_seconds": 0.0,
        "materialize_lock_wait_seconds": 0.0,
        "publications": 0,
        "publish_declines": 0,
        "private_fallbacks": 0,
        "shared_mapped_bytes": 0,
    }


def test_cache_stats_delta_subtracts_every_counter_without_mutating_snapshot() -> None:
    stats = CacheStats(loads=2, load_seconds=1.5, shared_hits=3, bytes_read=100)
    before = stats.snapshot()
    stats.loads += 4
    stats.load_seconds += 0.25
    stats.shared_hits += 5
    stats.bytes_read += 7

    delta = stats.delta(before)

    assert delta == {
        "loads": 4,
        "hits": 0,
        "evictions": 0,
        "retain_calls": 0,
        "load_seconds": 0.25,
        "bytes_read": 7,
        "shared_hits": 5,
        "shared_misses": 0,
        "shared_get_races": 0,
        "materializations": 0,
        "materialize_seconds": 0.0,
        "materialize_lock_wait_seconds": 0.0,
        "publications": 0,
        "publish_declines": 0,
        "private_fallbacks": 0,
        "shared_mapped_bytes": 0,
    }
    assert before["loads"] == 2
    assert before["shared_hits"] == 3


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"loads": 0},
        {**CacheStats().snapshot(), "unknown_future_field": 0},
        {key: value for key, value in CacheStats().snapshot().items() if key != "bytes_read"},
    ],
)
def test_cache_stats_delta_rejects_partial_or_future_schema(bad: dict[str, int | float]) -> None:
    with pytest.raises(ValueError, match="schema mismatch"):
        CacheStats().delta(bad)


def test_first_shared_lookup_records_one_miss_materialization_publication_and_mapping(
    tmp_path: Path,
) -> None:
    feature_dir = _build_dir(tmp_path)
    store = SharedStemStore(tmp_path / "shm")
    index = FeatureIndex(
        str(feature_dir),
        compact_feature_ids=True,
        max_resident_shards=4,
        shared_store=store,
    )
    before = index.cache_stats.snapshot()

    segments = index.lookup("shard_00000", 0)
    delta = index.cache_stats.delta(before)

    assert segments
    assert delta["loads"] == 1
    assert delta["shared_misses"] == 1
    assert delta["materializations"] == 1
    assert delta["publications"] == 1
    assert delta["shared_hits"] == 1
    assert delta["publish_declines"] == 0
    assert delta["private_fallbacks"] == 0
    assert delta["shared_get_races"] == 0
    assert delta["shared_mapped_bytes"] > 0
    assert delta["materialize_seconds"] >= 0
    assert delta["materialize_lock_wait_seconds"] >= 0


def test_second_index_records_pure_shared_hit_without_materialization(tmp_path: Path) -> None:
    feature_dir = _build_dir(tmp_path)
    store = SharedStemStore(tmp_path / "shm")
    seed = FeatureIndex(
        str(feature_dir),
        compact_feature_ids=True,
        max_resident_shards=4,
        shared_store=store,
    )
    seed.lookup("shard_00000", 0)
    index = FeatureIndex(
        str(feature_dir),
        compact_feature_ids=True,
        max_resident_shards=4,
        shared_store=store,
    )

    index.lookup("shard_00000", 0)
    stats = index.cache_stats

    assert stats.shared_hits == 1
    assert stats.shared_misses == 0
    assert stats.materializations == 0
    assert stats.publications == 0
    assert stats.private_fallbacks == 0
    assert stats.shared_mapped_bytes > 0


def test_space_guard_decline_records_exact_private_fallback_transition(tmp_path: Path) -> None:
    feature_dir = _build_dir(tmp_path)
    store = SharedStemStore(tmp_path / "shm", min_free_bytes=1 << 60)
    index = FeatureIndex(
        str(feature_dir),
        compact_feature_ids=True,
        max_resident_shards=4,
        shared_store=store,
    )

    segments = index.lookup("shard_00000", 0)
    stats = index.cache_stats

    assert segments
    assert not store.has("shard_00000")
    assert stats.shared_misses == 1
    assert stats.materializations == 1
    assert stats.publications == 0
    assert stats.publish_declines == 1
    assert stats.private_fallbacks == 1
    assert stats.shared_hits == 0
    assert stats.shared_mapped_bytes == 0


class _ScriptedStore:
    def __init__(
        self,
        *,
        has_values: list[bool],
        get_values: list[dict | KeyError],
        put_result: bool = True,
    ) -> None:
        self.has_values = has_values
        self.get_values = get_values
        self.put_result = put_result
        self.put_calls = 0
        self.lock_entries = 0
        self.lock_active = False
        self.get_lock_states: list[bool] = []

    def has(self, key: str) -> bool:
        assert key == "stem"
        if not self.has_values:
            raise AssertionError("unexpected has() call")
        return self.has_values.pop(0)

    def get(self, key: str) -> dict:
        assert key == "stem"
        self.get_lock_states.append(self.lock_active)
        if not self.get_values:
            raise AssertionError("unexpected get() call")
        value = self.get_values.pop(0)
        if isinstance(value, KeyError):
            raise value
        return value

    def put(self, key: str, tensors: dict[str, t.Tensor], scalars: dict[str, int]) -> bool:
        assert key == "stem"
        assert tensors
        assert scalars == {"n_segments": 1}
        self.put_calls += 1
        return self.put_result

    @contextmanager
    def materialize_lock(self):
        self.lock_entries += 1
        assert not self.lock_active
        self.lock_active = True
        try:
            yield
        finally:
            self.lock_active = False


def _bare_index(materialized: dict) -> tuple[FeatureIndex, dict[str, int]]:
    index = FeatureIndex.__new__(FeatureIndex)
    index._initialize_empty(
        prefetch=False,
        percentile_checksum=None,
        tokenizer_checksum=None,
        skip_feature_mags=False,
        compact_feature_ids=False,
        max_resident_shards=1,
        shared_store=None,
        lazy_shared_docs=False,
        required_token_boundary_stems=None,
    )
    calls = {"materialize": 0}

    def materialize(stem: str) -> dict:
        assert stem == "stem"
        calls["materialize"] += 1
        return materialized

    index._materialize_data = materialize  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]
    return index, calls


def test_has_get_race_is_counted_then_published_without_double_materialization() -> None:
    data = {"feature_ids": t.tensor([1, 2], dtype=t.uint16), "n_segments": 1}
    mapped = {"feature_ids": t.tensor([1, 2], dtype=t.uint16), "n_segments": 1}
    store = _ScriptedStore(
        has_values=[True, False],
        get_values=[KeyError("evicted"), mapped],
    )
    index, calls = _bare_index(data)

    result = index._shared_get_or_materialize(
        store,  # ty: ignore[invalid-argument-type] -- protocol test double
        "stem",
    )

    assert result is mapped
    assert calls["materialize"] == 1
    assert store.put_calls == 1
    assert index.cache_stats.shared_get_races == 1
    assert index.cache_stats.shared_misses == 1
    assert index.cache_stats.materializations == 1
    assert index.cache_stats.publications == 1
    assert index.cache_stats.shared_hits == 1
    assert index.cache_stats.shared_mapped_bytes == 4
    assert store.get_lock_states == [False, True]


def test_peer_publication_inside_lock_is_shared_hit_without_local_materialization() -> None:
    mapped = {"feature_ids": t.tensor([3], dtype=t.int32), "n_segments": 1}
    store = _ScriptedStore(has_values=[False, True], get_values=[mapped])
    index, calls = _bare_index(mapped)

    result = index._shared_get_or_materialize(
        store,  # ty: ignore[invalid-argument-type] -- protocol test double
        "stem",
    )

    assert result is mapped
    assert calls["materialize"] == 0
    assert store.put_calls == 0
    assert index.cache_stats.shared_misses == 1
    assert index.cache_stats.materializations == 0
    assert index.cache_stats.publications == 0
    assert index.cache_stats.shared_hits == 1
    assert index.cache_stats.shared_mapped_bytes == 4
    assert store.get_lock_states == [True]


def test_post_peer_publication_race_records_one_last_resort_private_materialization() -> None:
    data = {"feature_ids": t.tensor([4, 5, 6], dtype=t.int32), "n_segments": 1}
    store = _ScriptedStore(
        has_values=[False, True],
        get_values=[KeyError("evicted after peer publication")],
    )
    index, calls = _bare_index(data)
    original_materialize = index._materialize_data

    def assert_locked_materialize(stem: str) -> dict:
        assert store.lock_active, "fallback decompressed outside materialization lease"
        return original_materialize(stem)

    index._materialize_data = (  # ty: ignore[invalid-assignment] -- instrumented bound callable
        assert_locked_materialize
    )

    result = index._shared_get_or_materialize(
        store,  # ty: ignore[invalid-argument-type] -- protocol test double
        "stem",
    )

    assert result is data
    assert calls["materialize"] == 1
    assert index.cache_stats.shared_misses == 1
    assert index.cache_stats.shared_get_races == 1
    assert index.cache_stats.private_fallbacks == 1
    assert index.cache_stats.materializations == 1
    assert index.cache_stats.publications == 0
    assert index.cache_stats.shared_hits == 0
    assert store.get_lock_states == [True]
    assert store.lock_active is False


def test_sidecar_load_records_exact_logical_bytes_and_time_but_hits_do_not(
    tmp_path: Path,
) -> None:
    env = build_derived_env(tmp_path, cut_fraction=0.6)
    feature_dir = Path(env["feature_dir"])
    sidecar = feature_dir / "_splice_provenance" / "shard_00000.json"
    index = SpliceProvenanceIndex(feature_dir, max_resident_shards=2)

    before = index.cache_stats.snapshot()
    index.shard_n_docs("shard_00000")
    first = index.cache_stats.delta(before)

    assert first["loads"] == 1
    assert first["hits"] == 0
    assert first["bytes_read"] == sidecar.stat().st_size
    assert first["load_seconds"] >= 0
    assert first["shared_hits"] == 0
    assert first["materializations"] == 0

    before_hit = index.cache_stats.snapshot()
    index.shard_n_docs("shard_00000")
    hit = index.cache_stats.delta(before_hit)
    assert hit["hits"] == 1
    assert hit["loads"] == 0
    assert hit["bytes_read"] == 0
    assert hit["load_seconds"] == 0
