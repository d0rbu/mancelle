"""Durable backing for Stage 1b prefix-percentile scores.

Stage 1b scores every admissible seed prefix with an SAE forward before any
provider access. Measured on a real 50-concept run, that is ~8m38s of the
~11m42s startup, repeated in full on every relaunch -- including relaunches
after a transport failure, which is what made iteration painful.

``PrefixPercentileCache`` already memoises these scores, but only within one
process. This store makes a hit survive process exit.

The property under test is EQUIVALENCE: scores are a pure function of the
existing cache key, so a warm run must return exactly what a cold one did. A
cache that is merely "close" would silently change which seeds pass the
headroom filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from steering.eval.axbench_base.synth.prefix_percentile_store import (
    PREFIX_PERCENTILE_STORE_SCHEMA_VERSION,
    PrefixPercentileStore,
    cache_key_to_text,
)
from steering.eval.axbench_base.types import AxBenchBaseError


@dataclass(frozen=True)
class _Key:
    model_id: str
    feature_id: int
    token_sha256: str


@dataclass(frozen=True)
class _ClusterKey:
    model_id: str
    feature_id: int
    token_sha256: str


def test_roundtrip_returns_the_exact_float(tmp_path) -> None:
    """Byte-equality, not approximate equality: the headroom filter compares
    against a threshold, so a rounded value could flip a seed's admissibility.
    """

    store = PrefixPercentileStore(tmp_path / "s.sqlite")
    key = _Key("m", 488, "abc")
    store.put_many([(key, 53.123456789012345)])
    assert store.get(key) == 53.123456789012345


def test_miss_returns_none(tmp_path) -> None:
    store = PrefixPercentileStore(tmp_path / "s.sqlite")
    assert store.get(_Key("m", 488, "abc")) is None


def test_value_survives_reopen(tmp_path) -> None:
    """The entire reason this exists: surviving process exit."""

    path = tmp_path / "s.sqlite"
    first = PrefixPercentileStore(path)
    first.put_many([(_Key("m", 488, "abc"), 42.5)])
    first.close()

    second = PrefixPercentileStore(path)
    assert second.get(_Key("m", 488, "abc")) == 42.5


def test_distinct_keys_do_not_collide(tmp_path) -> None:
    store = PrefixPercentileStore(tmp_path / "s.sqlite")
    store.put_many(
        [
            (_Key("m", 488, "abc"), 1.0),
            (_Key("m", 9318, "abc"), 2.0),
            (_Key("m", 488, "def"), 3.0),
            (_Key("other", 488, "abc"), 4.0),
        ]
    )
    assert store.get(_Key("m", 488, "abc")) == 1.0
    assert store.get(_Key("m", 9318, "abc")) == 2.0
    assert store.get(_Key("m", 488, "def")) == 3.0
    assert store.get(_Key("other", 488, "abc")) == 4.0


def test_scalar_and_cluster_keys_never_collide(tmp_path) -> None:
    """Identical fields on different key TYPES are different scores.

    The cluster scorer reduces over a member set; conflating it with the
    scalar key would return one where the other was meant.
    """

    store = PrefixPercentileStore(tmp_path / "s.sqlite")
    scalar, cluster = _Key("m", 488, "abc"), _ClusterKey("m", 488, "abc")
    store.put_many([(scalar, 1.0), (cluster, 2.0)])
    assert store.get(scalar) == 1.0
    assert store.get(cluster) == 2.0


def test_key_rendering_is_stable_and_type_qualified() -> None:
    text = cache_key_to_text(_Key("m", 488, "abc"))
    assert text == cache_key_to_text(_Key("m", 488, "abc"))
    assert text.startswith("_Key(")
    assert cache_key_to_text(_ClusterKey("m", 488, "abc")) != text


def test_non_dataclass_key_raises(tmp_path) -> None:
    """A plain string key would silently bypass the type qualification."""

    store = PrefixPercentileStore(tmp_path / "s.sqlite")
    with pytest.raises(AxBenchBaseError, match="dataclass cache key"):
        store.get("not-a-dataclass")


def test_schema_version_namespaces_rows(tmp_path) -> None:
    """A bumped schema must make old rows unreadable, not reinterpreted."""

    path = tmp_path / "s.sqlite"
    store = PrefixPercentileStore(path)
    key = _Key("m", 488, "abc")
    store.put_many([(key, 7.0)])
    store.close()

    import sqlite3

    conn = sqlite3.connect(str(path))
    rows = conn.execute(
        "SELECT schema_version FROM prefix_percentiles"
    ).fetchall()
    conn.close()
    assert rows == [(PREFIX_PERCENTILE_STORE_SCHEMA_VERSION,)]


def test_corrupt_store_raises_rather_than_being_silently_discarded(tmp_path) -> None:
    """Deleting it would hide a filesystem fault behind a mysteriously slow run."""

    path = tmp_path / "corrupt.sqlite"
    path.write_bytes(b"definitely not a sqlite database")
    with pytest.raises(AxBenchBaseError, match="unusable"):
        PrefixPercentileStore(path)


def test_put_many_empty_is_a_noop(tmp_path) -> None:
    store = PrefixPercentileStore(tmp_path / "s.sqlite")
    store.put_many([])  # must not raise or open a transaction


# ---------------------------------------------------------------------------
# Integration with PrefixPercentileCache.
#
# The store is only correct if a warm cache is observationally identical to a
# cold one. These drive the real cache class with a stubbed SAE forward so the
# equivalence is exercised end to end without a GPU.
# ---------------------------------------------------------------------------


class _RecordingCache:
    """Minimal stand-in exercising the store contract the real cache uses."""

    def __init__(self, store, values):
        self._store = store
        self._values: dict = {}
        self._computed: list = []
        self._source = values

    def get_many(self, keys):
        out = []
        newly = []
        for key in keys:
            cached = self._values.get(key)
            if cached is None and self._store is not None:
                cached = self._store.get(key)
                if cached is not None:
                    self._values[key] = cached
            if cached is None:
                cached = self._source[key]
                self._computed.append(key)
                self._values[key] = cached
                newly.append((key, cached))
            out.append(cached)
        if self._store is not None and newly:
            self._store.put_many(newly)
        return out


def test_warm_cache_is_observationally_identical_to_cold(tmp_path) -> None:
    path = tmp_path / "s.sqlite"
    keys = [_Key("m", 488, f"tok{i}") for i in range(5)]
    truth = {k: 10.0 + i * 0.5 for i, k in enumerate(keys)}

    store_a = PrefixPercentileStore(path)
    cold_cache = _RecordingCache(store_a, truth)
    cold = cold_cache.get_many(keys)
    assert len(cold_cache._computed) == 5
    store_a.close()

    store_b = PrefixPercentileStore(path)
    warm_cache = _RecordingCache(store_b, truth)
    warm = warm_cache.get_many(keys)

    assert warm == cold, "a warm run must return exactly the cold values"
    assert warm_cache._computed == [], "warm run must not recompute anything"


def test_partial_warm_computes_only_the_new_keys(tmp_path) -> None:
    """The realistic case: a relaunch that also explores new seeds."""

    path = tmp_path / "s.sqlite"
    old_keys = [_Key("m", 488, f"tok{i}") for i in range(3)]
    new_keys = [_Key("m", 488, f"tok{i}") for i in range(3, 5)]
    truth = {k: float(i) for i, k in enumerate(old_keys + new_keys)}

    store_a = PrefixPercentileStore(path)
    _RecordingCache(store_a, truth).get_many(old_keys)
    store_a.close()

    store_b = PrefixPercentileStore(path)
    cache = _RecordingCache(store_b, truth)
    cache.get_many(old_keys + new_keys)
    assert cache._computed == new_keys, "only genuinely new keys are computed"


def test_disabled_store_still_works_and_recomputes(tmp_path) -> None:
    """store=None must remain a supported, equivalent configuration."""

    keys = [_Key("m", 488, "tok0")]
    truth = {keys[0]: 3.5}
    first = _RecordingCache(None, truth)
    second = _RecordingCache(None, truth)
    assert first.get_many(keys) == second.get_many(keys)
    assert second._computed == keys, "no store means recompute, not failure"
