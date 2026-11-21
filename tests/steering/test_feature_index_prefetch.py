"""Single-slot background shard prefetch (issue #1504).

One in-flight daemon thread and one published-but-unconsumed shard map, never
both.  These tests pin value identity (against ``prefetch=False``), the
run-ahead behaviour, failure parity, the documented residency bound, the
close/join/pickle lifecycle, and composition with ``lazy_docs``.
"""

from __future__ import annotations

import pickle
import random
import threading
import time
from io import StringIO
from pathlib import Path

import pytest
import torch as t
from loguru import logger

from steering.conditional import feature_index as feature_index_module
from steering.conditional.feature_index import CacheStats, FeatureIndex, SegmentFeatures
from steering.conditional.shared_stem_store import SharedStemStore
from steering.feature_io import CorruptFeatureShardError
from tests.steering._helpers import make_valid_shard_data, save_raw_shard, save_valid_shard

_PREFETCH_THREAD_PREFIX = "FeatureIndex-prefetch"


def _make_dir(tmp_path: Path, n_shards: int, *, n_docs: int = 3, segs_per_doc: int = 2) -> str:
    directory = tmp_path / "feat"
    directory.mkdir(exist_ok=True)
    for i in range(n_shards):
        save_valid_shard(
            directory,
            f"shard_{i:05d}",
            n_docs=n_docs,
            segs_per_doc=segs_per_doc,
            seed=i,
        )
    return str(directory)


def _write_corrupt_shard(directory: str, stem: str) -> None:
    """Raw-write a shard whose segment 1 is empty (offsets [0, 2, 2, 4])."""
    data = make_valid_shard_data(n_docs=1, segs_per_doc=3, feats_per_seg=2)
    data["offsets"] = t.tensor([0, 2, 2, 4], dtype=t.int64)
    data["feature_ids"] = t.tensor([3, 5, 7, 9], dtype=t.int32)
    data["feature_mags"] = t.ones(4, dtype=t.float32)
    data["segment_char_starts"] = t.tensor([0, 6, 12], dtype=t.int32)
    data["segment_char_ends"] = t.tensor([6, 12, 18], dtype=t.int32)
    save_raw_shard(Path(directory), stem, data)


def _assert_segments_equal(got: list[SegmentFeatures], expected: list[SegmentFeatures]) -> None:
    assert len(got) == len(expected)
    for actual, reference in zip(got, expected, strict=True):
        assert actual.segment_idx == reference.segment_idx
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
            assert t.equal(actual.feature_percentiles, reference.feature_percentiles)


def _live_prefetch_threads() -> list[threading.Thread]:
    return [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith(_PREFETCH_THREAD_PREFIX) and thread.is_alive()
    ]


def _published_stem(index: FeatureIndex) -> str | None:
    with index._prefetch_lock:
        return index._prefetch_stem if index._prefetch_result is not None else None


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_from_catalog_gets_prefetch_fields(tmp_path: Path) -> None:
    """``from_catalog`` bypasses ``__init__``; ``_initialize_empty`` must still
    populate every prefetch field (else the first lookup would AttributeError)."""
    source = FeatureIndex(_make_dir(tmp_path, 3))
    catalog = source.export_catalog()
    index = FeatureIndex.from_catalog(catalog, prefetch=True)
    try:
        assert index._prefetch_thread is None
        assert index._prefetch_result is None
        assert index._prefetch_stem is None
        assert index._prefetch_closed is False
        index.lookup("shard_00000", 0)
        index._join_inflight()
        assert _published_stem(index) == "shard_00001"
    finally:
        index.close()


# ---------------------------------------------------------------------------
# Value identity
# ---------------------------------------------------------------------------


def test_sequential_lookups_identical_to_eager(tmp_path: Path) -> None:
    directory = _make_dir(tmp_path, 5)
    eager = FeatureIndex(directory, prefetch=False)
    prefetching = FeatureIndex(directory, prefetch=True)
    try:
        for shard in range(5):
            stem = f"shard_{shard:05d}"
            for doc in range(3):
                _assert_segments_equal(prefetching.lookup(stem, doc), eager.lookup(stem, doc))
    finally:
        prefetching.close()


def test_non_monotone_lookups_identical_to_eager(tmp_path: Path) -> None:
    """Epoch-wrap / reposition access patterns must not change any value."""
    directory = _make_dir(tmp_path, 5)
    eager = FeatureIndex(directory, prefetch=False)
    prefetching = FeatureIndex(directory, prefetch=True)
    order = [0, 1, 4, 2, 0, 3, 3, 1, 4, 0]
    try:
        for shard in order:
            stem = f"shard_{shard:05d}"
            for doc in range(3):
                _assert_segments_equal(prefetching.lookup(stem, doc), eager.lookup(stem, doc))
    finally:
        prefetching.close()


def test_stress_random_access_identical_to_eager(tmp_path: Path, monkeypatch) -> None:
    """Randomized load latency + 200 mixed lookups: values still exact.

    The sleep bound is 10 ms, already two orders of magnitude above the real
    per-load cost of these tiny fixtures, so every publish/consume race window
    is widened well past the work it guards.
    """
    directory = _make_dir(tmp_path, 5)
    original_load = FeatureIndex._load_shard
    rng = random.Random(1504)

    def jittered_load(self_index, stem):
        time.sleep(rng.uniform(0.0, 0.01))
        return original_load(self_index, stem)

    monkeypatch.setattr(FeatureIndex, "_load_shard", jittered_load)

    eager = FeatureIndex(directory, prefetch=False)
    prefetching = FeatureIndex(directory, prefetch=True)
    pattern = random.Random(99)
    try:
        for _ in range(200):
            stem = f"shard_{pattern.randrange(5):05d}"
            doc = pattern.randrange(3)
            _assert_segments_equal(prefetching.lookup(stem, doc), eager.lookup(stem, doc))
            assert len(_live_prefetch_threads()) <= 1
    finally:
        prefetching.close()


# ---------------------------------------------------------------------------
# The prefetch actually runs ahead
# ---------------------------------------------------------------------------


def test_next_stem_loads_off_the_consumer_thread(tmp_path: Path, monkeypatch) -> None:
    directory = _make_dir(tmp_path, 5)
    original_load_features = feature_index_module.load_features
    calls: list[tuple[str, str]] = []
    calls_lock = threading.Lock()

    def recording_load_features(path, *args, **kwargs):
        with calls_lock:
            calls.append((Path(path).name.split("_features")[0], threading.current_thread().name))
        return original_load_features(path, *args, **kwargs)

    monkeypatch.setattr(feature_index_module, "load_features", recording_load_features)

    index = FeatureIndex(directory, prefetch=True)
    try:
        index.lookup("shard_00000", 0)
        index._join_inflight()
        with calls_lock:
            observed = dict(calls)
        assert observed["shard_00000"] == threading.current_thread().name
        assert observed["shard_00001"] == f"{_PREFETCH_THREAD_PREFIX}-shard_00001"
        # Exactly one stem of lookahead, never the one after it.
        assert "shard_00002" not in observed
        assert _published_stem(index) == "shard_00001"
    finally:
        index.close()


def test_resident_and_pinned_next_stems_are_not_reloaded(tmp_path: Path) -> None:
    index = FeatureIndex(_make_dir(tmp_path, 4), prefetch=True, max_resident_shards=2)
    try:
        index.pin_shard("shard_00002")
        index._resident["shard_00001"] = {}
        index._schedule_prefetch("shard_00000")  # next stem is resident
        assert index._prefetch_thread is None
        index._schedule_prefetch("shard_00001")  # next stem is pinned
        assert index._prefetch_thread is None
        assert _published_stem(index) is None
    finally:
        index.close()


def test_schedule_prefetch_ignores_unknown_stem(tmp_path: Path) -> None:
    index = FeatureIndex(_make_dir(tmp_path, 3), prefetch=True)
    try:
        index._schedule_prefetch("not-a-stem")
        assert index._prefetch_thread is None
    finally:
        index.close()


# ---------------------------------------------------------------------------
# Stale slots and the documented residency bound
# ---------------------------------------------------------------------------


def test_prefetch_miss_discards_the_stale_slot(tmp_path: Path) -> None:
    index = FeatureIndex(_make_dir(tmp_path, 5), prefetch=True)
    try:
        index.lookup("shard_00000", 0)
        index._join_inflight()
        assert _published_stem(index) == "shard_00001"

        # Jump past the published stem: its map is unreachable and must not
        # survive alongside the shard being admitted.
        index.lookup("shard_00003", 0)
        index._join_inflight()
        assert _published_stem(index) == "shard_00004"
    finally:
        index.close()


def test_sync_load_never_exceeds_the_documented_shard_bound(tmp_path: Path, monkeypatch) -> None:
    """``max_resident_shards + 1`` must bound live shard maps.

    A prefetch miss sync-loads a shard while the slot may still hold the map it
    is leaving, so without discarding that map first the process transiently
    holds ``max_resident + 2`` decompressed maps — contradicting the bound
    stated in the module docstring and in
    docs/pipelines/encoder-decoder-training.md.
    """
    max_resident = 1
    index = FeatureIndex(
        _make_dir(tmp_path, 6),
        prefetch=True,
        max_resident_shards=max_resident,
    )
    samples: list[int] = []
    samples_lock = threading.Lock()
    original_load = FeatureIndex._load_shard

    def sampling_load(self_index, stem):
        # Only one load runs at a time (both schedule and resolve join the
        # previous worker first), so the map being built right now is the +1.
        with self_index._prefetch_lock:
            held = (
                len(self_index._resident)
                + len(self_index._pinned)
                + (1 if self_index._prefetch_result is not None else 0)
                + 1
            )
        with samples_lock:
            samples.append(held)
        return original_load(self_index, stem)

    monkeypatch.setattr(FeatureIndex, "_load_shard", sampling_load)
    try:
        index.lookup("shard_00000", 0)
        index._join_inflight()
        index.lookup("shard_00003", 0)  # prefetch miss: sync load with a full slot held
    finally:
        index.close()

    with samples_lock:
        assert samples
        assert max(samples) <= max_resident + 1, samples


# ---------------------------------------------------------------------------
# Failure parity (fail loud)
# ---------------------------------------------------------------------------


def test_failed_prefetch_warns_and_reraises_on_sync_load(tmp_path: Path) -> None:
    directory = _make_dir(tmp_path, 3)
    _write_corrupt_shard(directory, "shard_00001")

    eager = FeatureIndex(directory, prefetch=False)
    with pytest.raises(CorruptFeatureShardError) as eager_error:
        eager.lookup("shard_00001", 0)

    log_output = StringIO()
    handler_id = logger.add(log_output, format="{level} | {message}", level="WARNING")
    index = FeatureIndex(directory, prefetch=True)
    try:
        index.lookup("shard_00000", 0)
        index._join_inflight()
        # The failed background load published nothing.
        assert _published_stem(index) is None
        with pytest.raises(CorruptFeatureShardError) as prefetched_error:
            index.lookup("shard_00001", 0)
        assert str(prefetched_error.value) == str(eager_error.value)
    finally:
        index.close()
        logger.remove(handler_id)

    assert "Prefetch of shard_00001 failed" in log_output.getvalue()


# ---------------------------------------------------------------------------
# Lifecycle: close, join-vs-shutdown split, pickle
# ---------------------------------------------------------------------------


def test_close_joins_clears_and_is_idempotent(tmp_path: Path) -> None:
    before = len(_live_prefetch_threads())
    index = FeatureIndex(_make_dir(tmp_path, 5), prefetch=True)
    index.lookup("shard_00000", 0)
    index.close()
    assert index._prefetch_thread is None
    assert index._prefetch_result is None
    assert index._prefetch_stem is None
    assert index._prefetch_closed is True
    assert len(_live_prefetch_threads()) == before
    index.close()
    assert len(_live_prefetch_threads()) == before


def test_close_with_an_in_flight_load(tmp_path: Path, monkeypatch) -> None:
    directory = _make_dir(tmp_path, 4)
    original_load = FeatureIndex._load_shard

    def slow_load(self_index, stem):
        time.sleep(0.05)
        return original_load(self_index, stem)

    monkeypatch.setattr(FeatureIndex, "_load_shard", slow_load)
    index = FeatureIndex(directory, prefetch=True)
    index.lookup("shard_00000", 0)
    index.close()
    assert _live_prefetch_threads() == []
    assert index._prefetch_thread is None


def test_worker_publish_after_close_is_discarded(tmp_path: Path, monkeypatch) -> None:
    """The only window a publish can outlive ``close()`` is a close FROM a worker.

    ``__del__`` fires on the prefetch thread when the worker's bound target holds
    the last reference; ``_shutdown_prefetch`` then skips joining itself and
    clears the slot, so this worker reaches its publish afterwards. Without the
    closed flag it would repopulate a shut-down slot with a whole shard map.
    """
    directory = _make_dir(tmp_path, 3)
    original_load = FeatureIndex._load_shard

    def closing_load(self_index, stem):
        result = original_load(self_index, stem)
        if threading.current_thread().name.startswith(_PREFETCH_THREAD_PREFIX):
            self_index.close()
        return result

    monkeypatch.setattr(FeatureIndex, "_load_shard", closing_load)
    index = FeatureIndex(directory, prefetch=True)
    index.lookup("shard_00000", 0)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and _live_prefetch_threads():
        time.sleep(0.01)
    assert _live_prefetch_threads() == []
    assert index._prefetch_closed is True
    assert index._prefetch_result is None
    assert index._prefetch_stem is None
    assert index._prefetch_thread is None


def test_join_inflight_keeps_the_published_result(tmp_path: Path) -> None:
    """``doc_ids_for_shard`` relies on join-only semantics: no consume, no start."""
    index = FeatureIndex(_make_dir(tmp_path, 3), prefetch=True)
    try:
        index.lookup("shard_00000", 0)
        index._join_inflight()
        assert _published_stem(index) == "shard_00001"
        # A non-resident stem takes the join-then-sync-load path.
        assert index.doc_ids_for_shard("shard_00002") == [0, 1, 2]
        assert _published_stem(index) == "shard_00001"
        assert index._prefetch_thread is None
    finally:
        index.close()


def test_close_from_a_prefetch_worker_thread_does_not_raise(tmp_path: Path) -> None:
    """``__del__`` can fire on the worker; joining itself would RuntimeError."""
    index = FeatureIndex(_make_dir(tmp_path, 3), prefetch=True)
    errors: list[RuntimeError] = []

    def call_close() -> None:
        try:
            index.close()
        except RuntimeError as exc:  # pragma: no cover - regression guard
            errors.append(exc)

    thread = threading.Thread(
        target=call_close, daemon=True, name=f"{_PREFETCH_THREAD_PREFIX}-shard_00001"
    )
    with index._prefetch_lock:
        index._prefetch_thread = thread
    thread.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert not errors
    assert index._prefetch_thread is None


def test_pickle_roundtrip_with_an_active_prefetch(tmp_path: Path) -> None:
    directory = _make_dir(tmp_path, 4)
    index = FeatureIndex(directory, prefetch=True)
    try:
        index.lookup("shard_00000", 0)
        state = index.__getstate__()
        assert state["_prefetch_thread"] is None
        assert state["_prefetch_result"] is None
        assert state["_prefetch_stem"] is None
        assert state["_prefetch_lock"] is None
        assert state["_stats_lock"] is None

        revived = pickle.loads(pickle.dumps(index))
    finally:
        index.close()
    try:
        assert isinstance(revived._prefetch_lock, type(threading.Lock()))
        assert isinstance(revived._stats_lock, type(threading.Lock()))
        assert revived._prefetch_thread is None
        assert revived._prefetch_result is None
        eager = FeatureIndex(directory, prefetch=False)
        _assert_segments_equal(revived.lookup("shard_00002", 1), eager.lookup("shard_00002", 1))
    finally:
        revived.close()


def test_pickling_a_closed_index_yields_an_open_slot(tmp_path: Path) -> None:
    directory = _make_dir(tmp_path, 4)
    index = FeatureIndex(directory, prefetch=True)
    index.lookup("shard_00000", 0)
    index.close()
    assert index._prefetch_closed is True

    revived = pickle.loads(pickle.dumps(index))
    try:
        assert revived._prefetch_closed is False
        # shard_00000 rode across the pickle in _resident, so miss on a fresh
        # stem to prove the revived index actually schedules again.
        revived.lookup("shard_00002", 0)
        revived._join_inflight()
        assert _published_stem(revived) == "shard_00003"
    finally:
        revived.close()


# ---------------------------------------------------------------------------
# Diagnostic counters survive the concurrent worker
# ---------------------------------------------------------------------------


def test_stats_increments_are_serialized_across_threads(tmp_path: Path, monkeypatch) -> None:
    """The prefetch worker writes these counters while the consumer does too.

    The read-modify-write window of every ``self._stats.<counter> += ...`` is
    widened so an unguarded increment deterministically loses updates; with
    ``_stats_lock`` held across each read and write the totals stay exact. Both
    the private load path and the shared-store path are covered — the shared
    store is a permitted companion to background prefetch.
    """
    slow_counters = {"loads", "bytes_read", "shared_hits", "shared_mapped_bytes"}

    def slow_getattribute(self_stats, name):
        value = object.__getattribute__(self_stats, name)
        if name in slow_counters:
            time.sleep(0.005)
        return value

    def run_concurrently(index: FeatureIndex, stem: str, per_thread: int) -> None:
        def worker() -> None:
            for _ in range(per_thread):
                index._load_shard(stem)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60.0)
            assert not thread.is_alive()

    private_root = tmp_path / "private"
    private_root.mkdir()
    shared_root = tmp_path / "shared"
    shared_root.mkdir()
    private = FeatureIndex(_make_dir(private_root, 2), prefetch=False)
    store = SharedStemStore(tmp_path / "shm")
    shared = FeatureIndex(_make_dir(shared_root, 2), prefetch=False, shared_store=store)
    # Publish + map the stem once so every later call takes the deterministic
    # shared-hit fast path (exactly one shared_hits increment per call).
    shared._load_shard("shard_00000")

    private_loads = private.cache_stats.loads
    private_bytes = private.cache_stats.bytes_read
    shared_hits = shared.cache_stats.shared_hits
    shared_bytes = shared.cache_stats.shared_mapped_bytes

    monkeypatch.setattr(CacheStats, "__getattribute__", slow_getattribute)
    try:
        run_concurrently(private, "shard_00000", 5)
        run_concurrently(shared, "shard_00000", 5)
    finally:
        monkeypatch.undo()

    assert private.cache_stats.loads == private_loads + 10
    assert private.cache_stats.bytes_read > private_bytes
    assert (private.cache_stats.bytes_read - private_bytes) % 10 == 0
    assert shared.cache_stats.shared_hits == shared_hits + 10
    assert (shared.cache_stats.shared_mapped_bytes - shared_bytes) % 10 == 0


# ---------------------------------------------------------------------------
# Prefetch and lazy documents compose
# ---------------------------------------------------------------------------


def test_prefetch_with_lazy_docs_matches_the_eager_reference(tmp_path: Path) -> None:
    """The production SFT wiring is prefetch + lazy docs together."""
    directory = _make_dir(tmp_path, 5)
    reference = FeatureIndex(directory, prefetch=False, lazy_docs=False)
    prefetching = FeatureIndex(directory, prefetch=True, lazy_docs=True)
    try:
        for shard in (0, 1, 4, 2, 0, 3):
            stem = f"shard_{shard:05d}"
            for doc in range(3):
                _assert_segments_equal(prefetching.lookup(stem, doc), reference.lookup(stem, doc))
    finally:
        prefetching.close()
