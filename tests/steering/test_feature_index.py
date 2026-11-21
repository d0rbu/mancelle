import gc
import sys
import tempfile
import threading
import time
import tracemalloc
from pathlib import Path

import pytest
import torch as t
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from steering.conditional.feature_index import FeatureIndex, SegmentFeatures
from steering.feature_io import (
    CorruptFeatureShardError,
    TensorKey,
    load_features,
    save_features,
)
from steering.overlay_io import (
    OverlayKind,
    overlay_dir,
    overlay_shard_path,
    save_overlay_shard,
    write_manifest,
)
from tests.steering._helpers import make_valid_shard_data, save_raw_shard


def _make_shard(
    doc_ids,
    segment_indices,
    segment_positions,
    offsets,
    feature_ids,
    feature_mags,
    segment_size=128,
):
    """Helper to build a synthetic shard dict."""
    seg_pos = t.tensor(segment_positions, dtype=t.int32)
    seg_char_starts = seg_pos * 6
    seg_char_ends = (seg_pos + segment_size) * 6
    return {
        "offsets": t.tensor(offsets, dtype=t.int64),
        "feature_ids": t.tensor(feature_ids, dtype=t.int32),
        "feature_mags": t.tensor(feature_mags, dtype=t.float32),
        "doc_ids": t.tensor(doc_ids, dtype=t.int32),
        "segment_indices": t.tensor(segment_indices, dtype=t.int32),
        "segment_positions": seg_pos,
        "doc_token_counts": t.tensor([256] * max(max(doc_ids) + 1, 1), dtype=t.int32),
        "segment_char_starts": seg_char_starts,
        "segment_char_ends": seg_char_ends,
        "n_docs": max(doc_ids) + 1 if doc_ids else 0,
        "n_segments": len(doc_ids),
        "layer": 12,
        "context_window": 1024,
        "pooling": "mean_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": segment_size,
    }


@pytest.fixture
def three_shard_dir(tmp_path):
    """Create 3 synthetic shards for prefetch testing."""
    for i in range(3):
        n_docs = 3
        n_segs = n_docs * 2  # 2 segments per doc
        offsets = list(range(0, n_segs * 3 + 1, 3))  # 3 features per segment
        feature_ids = list(range(i * 100, i * 100 + n_segs * 3))
        feature_mags = [0.1 * (j + 1) for j in range(n_segs * 3)]
        doc_ids = [d for d in range(n_docs) for _ in range(2)]
        segment_indices = [s for _ in range(n_docs) for s in range(2)]
        segment_positions = [s * 128 for _ in range(n_docs) for s in range(2)]
        shard = _make_shard(
            doc_ids=doc_ids,
            segment_indices=segment_indices,
            segment_positions=segment_positions,
            offsets=offsets,
            feature_ids=feature_ids,
            feature_mags=feature_mags,
        )
        save_features(shard, tmp_path / f"shard_{i:05d}_features")
    return str(tmp_path)


class TestFeatureIndex:
    @pytest.fixture
    def feature_dir(self, tmp_path):
        """Create a temp dir with two synthetic feature shard files."""
        # Shard 1: "shard_00001" with 2 docs, 3 segments
        shard1 = {
            "offsets": t.tensor([0, 2, 5, 7], dtype=t.int64),
            "feature_ids": t.tensor([10, 20, 30, 40, 50, 60, 70], dtype=t.int32),
            "feature_mags": t.tensor(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=t.float32
            ),
            "doc_ids": t.tensor([0, 0, 1], dtype=t.int32),
            "segment_indices": t.tensor([0, 1, 0], dtype=t.int32),
            "segment_positions": t.tensor([0, 128, 0], dtype=t.int32),
            "doc_token_counts": t.tensor([256, 128], dtype=t.int32),
            "segment_char_starts": t.tensor([0, 768, 0], dtype=t.int32),
            "segment_char_ends": t.tensor([768, 1536, 768], dtype=t.int32),
            "n_docs": 2,
            "n_segments": 3,
            "layer": 12,
            "context_window": 1024,
            "pooling": "mean_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 128,
        }
        save_features(shard1, tmp_path / "shard_00001_features")

        # Shard 2: "shard_00002" with 1 doc, 1 segment
        shard2 = {
            "offsets": t.tensor([0, 3], dtype=t.int64),
            "feature_ids": t.tensor([100, 200, 300], dtype=t.int32),
            "feature_mags": t.tensor([0.8, 0.9, 1.0], dtype=t.float32),
            "doc_ids": t.tensor([0], dtype=t.int32),
            "segment_indices": t.tensor([0], dtype=t.int32),
            "segment_positions": t.tensor([0], dtype=t.int32),
            "doc_token_counts": t.tensor([64], dtype=t.int32),
            "segment_char_starts": t.tensor([0], dtype=t.int32),
            "segment_char_ends": t.tensor([768], dtype=t.int32),
            "n_docs": 1,
            "n_segments": 1,
            "layer": 12,
            "context_window": 1024,
            "pooling": "mean_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 128,
        }
        save_features(shard2, tmp_path / "shard_00002_features")
        return str(tmp_path)

    def test_loads_all_shards(self, feature_dir):
        idx = FeatureIndex(feature_dir)
        assert idx.n_shards == 2

    def test_lookup_existing_doc(self, feature_dir):
        idx = FeatureIndex(feature_dir)
        segments = idx.lookup("shard_00001", doc_idx=0)
        assert segments is not None
        assert len(segments) == 2  # doc 0 in shard_00001 has 2 segments
        assert isinstance(segments[0], SegmentFeatures)
        assert segments[0].feature_ids.tolist() == [10, 20]
        assert segments[0].feature_mags.tolist() == pytest.approx([0.1, 0.2], abs=1e-3)
        assert segments[1].feature_ids.tolist() == [30, 40, 50]
        assert segments[1].feature_mags.tolist() == pytest.approx(
            [0.3, 0.4, 0.5], abs=1e-3
        )

    def test_lookup_second_doc(self, feature_dir):
        idx = FeatureIndex(feature_dir)
        segments = idx.lookup("shard_00001", doc_idx=1)
        assert segments is not None
        assert len(segments) == 1
        assert segments[0].feature_ids.tolist() == [60, 70]

    def test_lookup_different_shard(self, feature_dir):
        idx = FeatureIndex(feature_dir)
        segments = idx.lookup("shard_00002", doc_idx=0)
        assert segments is not None
        assert len(segments) == 1
        assert segments[0].feature_ids.tolist() == [100, 200, 300]
        assert segments[0].feature_mags.tolist() == pytest.approx(
            [0.8, 0.9, 1.0], abs=1e-3
        )

    def test_segment_positions_preserved(self, feature_dir):
        idx = FeatureIndex(feature_dir)
        segments = idx.lookup("shard_00001", doc_idx=0)
        assert segments is not None
        assert segments[0].position == 0
        assert segments[1].position == 128

    def test_segment_indices_preserved(self, feature_dir):
        idx = FeatureIndex(feature_dir)
        segments = idx.lookup("shard_00001", doc_idx=0)
        assert segments is not None
        assert segments[0].segment_idx == 0
        assert segments[1].segment_idx == 1

    def test_empty_dir_warns(self, tmp_path):
        """An empty directory should produce zero shards, not crash."""
        idx = FeatureIndex(str(tmp_path))
        assert idx.n_shards == 0
        with pytest.raises(KeyError, match=r"Unknown shard stem: 'anything'"):
            idx.lookup("anything", doc_idx=0)

    def test_segments_sorted_by_segment_idx(self, tmp_path):
        """Segments should be sorted by segment_idx even if stored out of order."""
        shard = {
            "offsets": t.tensor([0, 1, 2, 3], dtype=t.int64),
            "feature_ids": t.tensor([10, 20, 30], dtype=t.int32),
            "feature_mags": t.tensor([0.1, 0.2, 0.3], dtype=t.float32),
            "doc_ids": t.tensor([0, 0, 0], dtype=t.int32),
            # Out-of-order segment indices
            "segment_indices": t.tensor([2, 0, 1], dtype=t.int32),
            "segment_positions": t.tensor([256, 0, 128], dtype=t.int32),
            "doc_token_counts": t.tensor([384], dtype=t.int32),
            "segment_char_starts": t.tensor([1536, 0, 768], dtype=t.int32),
            "segment_char_ends": t.tensor([2304, 768, 1536], dtype=t.int32),
            "n_docs": 1,
            "n_segments": 3,
            "layer": 12,
            "context_window": 1024,
            "pooling": "mean_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 128,
        }
        save_features(shard, tmp_path / "shard_00000_features")

        idx = FeatureIndex(str(tmp_path))
        segments = idx.lookup("shard_00000", doc_idx=0)
        assert segments is not None
        assert [s.segment_idx for s in segments] == [0, 1, 2]
        assert [s.position for s in segments] == [0, 128, 256]

    def test_frozen_dataclass(self, feature_dir):
        """SegmentFeatures should be immutable."""
        idx = FeatureIndex(feature_dir)
        segments = idx.lookup("shard_00001", doc_idx=0)
        assert segments is not None
        with pytest.raises(AttributeError):
            segments[0].segment_idx = 99  # ty: ignore[invalid-assignment]

    # ------------------------------------------------------------------
    # New tests for lazy single-shard loading
    # ------------------------------------------------------------------

    def test_init_does_not_load_all_shards_into_memory(self, tmp_path):
        """Creating a FeatureIndex should NOT load all shard data into memory.

        We create several large-ish shards and verify that peak memory after
        __init__ stays well below the total size of all shards combined.
        """
        n_shards = 5
        features_per_shard = 100_000  # ~400KB per tensor * 2 tensors = ~800KB
        for i in range(n_shards):
            shard = _make_shard(
                doc_ids=[0],
                segment_indices=[0],
                segment_positions=[0],
                offsets=[0, features_per_shard],
                feature_ids=list(range(features_per_shard)),
                feature_mags=[0.1] * features_per_shard,
            )
            save_features(shard, tmp_path / f"shard_{i:05d}_features")

        # Total data across all shards: ~4MB (5 * 800KB)
        # With lazy loading, only ONE shard (the first, for segment_size) is loaded
        # then discarded. Peak should be well under total.
        tracemalloc.start()
        _idx = FeatureIndex(str(tmp_path))
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        total_all_shards_bytes = n_shards * features_per_shard * 4 * 2  # int32+float32
        # Peak memory should be much less than loading all shards
        assert peak < total_all_shards_bytes * 0.5, (
            f"Peak memory {peak:,} bytes is too high; "
            f"total shard data would be {total_all_shards_bytes:,} bytes. "
            f"Looks like all shards were loaded eagerly."
        )

    def test_shard_eviction_on_switch(self, feature_dir):
        """Accessing shard 1, then shard 2, then shard 1 again should all succeed.

        The LRU-1 cache should evict the old shard when a new one is requested,
        and re-load when the old shard is needed again.
        """
        idx = FeatureIndex(feature_dir)

        # Access shard 1
        segs1 = idx.lookup("shard_00001", doc_idx=0)
        assert segs1 is not None
        assert segs1[0].feature_ids.tolist() == [10, 20]

        # Access shard 2 (evicts shard 1 from cache)
        segs2 = idx.lookup("shard_00002", doc_idx=0)
        assert segs2 is not None
        assert segs2[0].feature_ids.tolist() == [100, 200, 300]

        # Access shard 1 again (must re-load, evicts shard 2)
        segs1_again = idx.lookup("shard_00001", doc_idx=0)
        assert segs1_again is not None
        assert segs1_again[0].feature_ids.tolist() == [10, 20]
        assert segs1_again[1].feature_ids.tolist() == [30, 40, 50]

    def test_lookup_returns_cloned_tensors(self, feature_dir):
        """Tensor slices from lookup must be cloned (independent storage).

        This ensures that when the parent shard tensor is freed, the sliced
        tensors remain valid and don't hold a reference to the full storage.
        """
        idx = FeatureIndex(feature_dir)
        segments = idx.lookup("shard_00001", doc_idx=0)
        assert segments is not None

        for seg in segments:
            # A properly cloned tensor has storage_offset() == 0
            assert seg.feature_ids.storage_offset() == 0, (
                f"feature_ids storage_offset={seg.feature_ids.storage_offset()}, "
                "expected 0 (tensor was not cloned)"
            )
            assert seg.feature_mags.storage_offset() == 0, (
                f"feature_mags storage_offset={seg.feature_mags.storage_offset()}, "
                "expected 0 (tensor was not cloned)"
            )
            # Cloned tensors should have their own storage sized to their numel
            assert (
                seg.feature_ids.untyped_storage().size()
                == seg.feature_ids.numel() * seg.feature_ids.element_size()
            ), "feature_ids storage size doesn't match numel - not independently stored"
            assert (
                seg.feature_mags.untyped_storage().size()
                == seg.feature_mags.numel() * seg.feature_mags.element_size()
            ), (
                "feature_mags storage size doesn't match numel - not independently stored"
            )

    def test_segment_size_from_first_shard(self, feature_dir):
        """segment_size should be available after __init__ even without a lookup."""
        idx = FeatureIndex(feature_dir)
        assert idx.segment_size == 128

    def test_segment_size_mismatch_raises(self, tmp_path):
        """Loading a shard with a different segment_size should raise an assertion."""
        shard1 = _make_shard(
            doc_ids=[0],
            segment_indices=[0],
            segment_positions=[0],
            offsets=[0, 1],
            feature_ids=[10],
            feature_mags=[0.1],
            segment_size=128,
        )
        shard2 = _make_shard(
            doc_ids=[0],
            segment_indices=[0],
            segment_positions=[0],
            offsets=[0, 1],
            feature_ids=[20],
            feature_mags=[0.2],
            segment_size=256,  # different!
        )
        save_features(shard1, tmp_path / "shard_00001_features")
        save_features(shard2, tmp_path / "shard_00002_features")

        idx = FeatureIndex(str(tmp_path))
        # First shard loads fine during __init__ (to get segment_size)
        # Second shard should fail validation when loaded
        with pytest.raises(AssertionError, match="Segment size mismatch"):
            idx.lookup("shard_00002", doc_idx=0)

    def test_all_features_removed(self, feature_dir):
        """all_features() method should no longer exist."""
        idx = FeatureIndex(feature_dir)
        assert not hasattr(idx, "all_features"), (
            "all_features() should have been removed"
        )

    def test_empty_dir_segment_size_none(self, tmp_path):
        """An empty directory should have segment_size == None."""
        idx = FeatureIndex(str(tmp_path))
        assert idx.segment_size is None

    def test_prefetch_false_creates_no_threads(self, feature_dir, monkeypatch):
        """prefetch=False must never spawn threads."""
        thread_calls = []
        orig_init = threading.Thread.__init__

        def tracking_init(self_thread, *args, **kwargs):
            thread_calls.append(1)
            orig_init(self_thread, *args, **kwargs)

        monkeypatch.setattr(threading.Thread, "__init__", tracking_init)

        idx = FeatureIndex(feature_dir, prefetch=False)
        idx.lookup("shard_00001", doc_idx=0)
        idx.lookup("shard_00002", doc_idx=0)
        idx.lookup("shard_00001", doc_idx=0)
        assert len(thread_calls) == 0, f"Expected 0 threads, got {len(thread_calls)}"

    def test_prefetch_results_identical_to_no_prefetch(self, three_shard_dir):
        """Lookup results must be bit-identical with and without prefetch."""
        results_no_prefetch = {}
        idx = FeatureIndex(three_shard_dir, prefetch=False)
        for stem in idx.shard_stems:
            for doc in idx.doc_ids_for_shard(stem):
                result = idx.lookup(stem, doc)
                results_no_prefetch[(stem, doc)] = [
                    (s.feature_ids.tolist(), s.feature_mags.tolist())
                    for s in result
                ]

        results_prefetch = {}
        idx2 = FeatureIndex(three_shard_dir, prefetch=True)
        for stem in idx2.shard_stems:
            for doc in idx2.doc_ids_for_shard(stem):
                result = idx2.lookup(stem, doc)
                results_prefetch[(stem, doc)] = [
                    (s.feature_ids.tolist(), s.feature_mags.tolist())
                    for s in result
                ]
        idx2.close()

        assert results_no_prefetch == results_prefetch

    def test_prefetch_predicts_next_shard(self, three_shard_dir, monkeypatch):
        """After loading shard_00000, prefetch should load shard_00001 in background."""
        load_calls = []
        original_load = FeatureIndex._load_shard

        def tracking_load(self_idx, stem):
            load_calls.append(stem)
            return original_load(self_idx, stem)

        monkeypatch.setattr(FeatureIndex, "_load_shard", tracking_load)

        idx = FeatureIndex(three_shard_dir, prefetch=True)
        idx.lookup("shard_00000", doc_idx=0)
        idx._join_inflight()

        idx.lookup("shard_00001", doc_idx=0)
        idx.close()

        assert load_calls[:2] == ["shard_00000", "shard_00001"], (
            f"Expected first two calls ['shard_00000', 'shard_00001'], got {load_calls}"
        )

    def test_prefetch_miss_falls_back_to_sync(self, three_shard_dir):
        """Skipping the predicted shard falls back to sync load correctly."""
        idx = FeatureIndex(three_shard_dir, prefetch=True)

        result_0 = idx.lookup("shard_00000", doc_idx=0)
        assert result_0 is not None

        result_2 = idx.lookup("shard_00002", doc_idx=0)
        assert result_2 is not None

        idx_verify = FeatureIndex(three_shard_dir, prefetch=False)
        expected = idx_verify.lookup("shard_00002", doc_idx=0)
        assert expected is not None
        for got, exp in zip(result_2, expected, strict=True):
            assert got.feature_ids.tolist() == exp.feature_ids.tolist()
            assert got.feature_mags.tolist() == exp.feature_mags.tolist()
        idx.close()

    def test_cache_eviction_with_prefetch(self, three_shard_dir):
        """Full cycle: hits, misses, evictions with prefetch enabled."""
        idx = FeatureIndex(three_shard_dir, prefetch=True)
        verify = FeatureIndex(three_shard_dir, prefetch=False)

        r0 = idx.lookup("shard_00000", doc_idx=0)
        assert r0 is not None

        r0_again = idx.lookup("shard_00000", doc_idx=1)
        e0_again = verify.lookup("shard_00000", doc_idx=1)
        if r0_again is not None:
            assert e0_again is not None
            for got, exp in zip(r0_again, e0_again, strict=True):
                assert got.feature_ids.tolist() == exp.feature_ids.tolist()

        r1 = idx.lookup("shard_00001", doc_idx=0)
        e1 = verify.lookup("shard_00001", doc_idx=0)
        assert r1 is not None and e1 is not None
        for got, exp in zip(r1, e1, strict=True):
            assert got.feature_ids.tolist() == exp.feature_ids.tolist()

        r0_reload = idx.lookup("shard_00000", doc_idx=0)
        e0_reload = verify.lookup("shard_00000", doc_idx=0)
        assert r0_reload is not None and e0_reload is not None
        for got, exp in zip(r0_reload, e0_reload, strict=True):
            assert got.feature_ids.tolist() == exp.feature_ids.tolist()
        idx.close()

    def test_last_shard_no_prefetch_crash(self, three_shard_dir):
        """Looking up the last shard should not crash or start a thread."""
        idx = FeatureIndex(three_shard_dir, prefetch=True)
        result = idx.lookup("shard_00002", doc_idx=0)
        assert result is not None
        assert idx._prefetch_thread is None
        idx.close()

    def test_close_joins_prefetch_thread(self, three_shard_dir):
        """close() should join the in-flight prefetch thread."""
        idx = FeatureIndex(three_shard_dir, prefetch=True)
        idx.lookup("shard_00000", doc_idx=0)
        idx.close()
        assert idx._prefetch_thread is None
        assert idx._prefetch_result is None
        idx.close()

    def test_concurrent_lookup_prefetch_no_corruption(self, three_shard_dir, monkeypatch):
        """Verify lock prevents corruption when prefetch is in-flight during lookup."""
        original_load = FeatureIndex._load_shard

        def slow_load(self_idx, stem):
            time.sleep(0.1)  # Widen race window
            return original_load(self_idx, stem)

        monkeypatch.setattr(FeatureIndex, "_load_shard", slow_load)

        idx = FeatureIndex(three_shard_dir, prefetch=True)
        r0 = idx.lookup("shard_00000", doc_idx=0)
        assert r0 is not None

        r1 = idx.lookup("shard_00001", doc_idx=0)
        assert r1 is not None

        verify = FeatureIndex(three_shard_dir, prefetch=False)
        e1 = verify.lookup("shard_00001", doc_idx=0)
        assert e1 is not None
        for got, exp in zip(r1, e1, strict=True):
            assert got.feature_ids.tolist() == exp.feature_ids.tolist()
        idx.close()

    def test_rapid_transitions_no_thread_leak(self, three_shard_dir):
        """Quickly alternating shards should never have more than 1 prefetch thread.

        Scheduling joins the previous worker before spawning, so the single-slot
        bound holds; the worker names carry their stem suffix.
        """
        idx = FeatureIndex(three_shard_dir, prefetch=True)
        stems = [f"shard_{i:05d}" for i in range(3)]

        for _ in range(5):
            for stem in stems:
                result = idx.lookup(stem, doc_idx=0)
                assert result is not None
                active_threads = [
                    th for th in threading.enumerate()
                    if th.name.startswith("FeatureIndex-prefetch") and th.is_alive()
                ]
                assert len(active_threads) <= 1, (
                    f"Expected at most 1 prefetch thread, found {len(active_threads)}"
                )

        idx.close()

    def test_doc_ids_for_shard_does_not_break_prefetch(self, three_shard_dir):
        """doc_ids_for_shard interleaved with prefetch lookups works correctly."""
        idx = FeatureIndex(three_shard_dir, prefetch=True)

        # Lookup shard_00000 triggers prefetch of shard_00001
        idx.lookup("shard_00000", doc_idx=0)
        idx._join_inflight()

        # Call doc_ids_for_shard on shard_00001 (loads via LRU)
        doc_ids = idx.doc_ids_for_shard("shard_00001")
        assert len(doc_ids) > 0

        # Subsequent normal lookup on shard_00001 doc 0 should still work
        result = idx.lookup("shard_00001", doc_idx=0)
        assert result is not None

        idx.close()

    def test_lookup_unknown_shard_raises_keyerror(self, feature_dir):
        idx = FeatureIndex(feature_dir)
        with pytest.raises(KeyError, match=r"Unknown shard stem: 'nonexistent'"):
            idx.lookup("nonexistent", doc_idx=0)

    def test_lookup_missing_doc_raises_keyerror_naming_shard_and_doc(self, feature_dir):
        idx = FeatureIndex(feature_dir)
        with pytest.raises(KeyError, match=r"Document 99 not found in shard 'shard_00001'"):
            idx.lookup("shard_00001", doc_idx=99)

    def test_lookup_unknown_shard_message_matches_siblings(self, feature_dir):
        """lookup, pin_shard, doc_ids_for_shard raise identical text for unknown shard."""
        idx = FeatureIndex(feature_dir)
        msgs = []
        for call in (
            lambda: idx.lookup("ghost", doc_idx=0),
            lambda: idx.pin_shard("ghost"),
            lambda: idx.doc_ids_for_shard("ghost"),
        ):
            with pytest.raises(KeyError) as ei:
                call()
            msgs.append(str(ei.value))
        assert msgs[0] == msgs[1] == msgs[2]

    def test_lookup_missing_doc_cache_hit_path_raises(self, feature_dir):
        idx = FeatureIndex(feature_dir)
        assert len(idx.lookup("shard_00001", doc_idx=0)) == 2  # populate cache
        with pytest.raises(KeyError, match=r"Document 5 not found in shard 'shard_00001'"):
            idx.lookup("shard_00001", doc_idx=5)

    def test_lookup_missing_doc_pinned_path_raises(self, feature_dir):
        idx = FeatureIndex(feature_dir)
        idx.pin_shard("shard_00002")
        assert len(idx.lookup("shard_00002", doc_idx=0)) == 1
        with pytest.raises(KeyError, match=r"Document 7 not found in shard 'shard_00002'"):
            idx.lookup("shard_00002", doc_idx=7)

    def test_lookup_missing_doc_prefetch_enabled_raises(self, three_shard_dir):
        idx = FeatureIndex(three_shard_dir, prefetch=True)
        assert idx.lookup("shard_00000", doc_idx=0) is not None
        with pytest.raises(KeyError, match=r"Document 42 not found in shard 'shard_00000'"):
            idx.lookup("shard_00000", doc_idx=42)
        idx.close()

    def test_lookup_present_doc_byte_identical_to_load_shard(self, feature_dir):
        """Exact-equality vs source of truth: lookup(present) == _load_shard()[doc]."""
        idx = FeatureIndex(feature_dir)
        docs = idx._load_shard("shard_00001")
        for doc_id, expected_segs in docs.items():
            got = idx.lookup("shard_00001", doc_idx=doc_id)
            assert len(got) == len(expected_segs)
            for g, e in zip(got, expected_segs, strict=True):
                assert g.segment_idx == e.segment_idx
                assert g.position == e.position
                assert g.char_start == e.char_start and g.char_end == e.char_end
                assert t.equal(g.feature_ids, e.feature_ids)
                assert t.equal(g.feature_mags, e.feature_mags)


@pytest.mark.slow
class TestFeatureIndexPrefetchPerformance:
    def test_prefetch_hides_latency(self, three_shard_dir, monkeypatch):
        """Prefetch hides shard load latency behind simulated GPU compute.

        Real usage: ~500 batches of GPU compute between shard transitions gives
        the prefetch thread time to complete. This test simulates that gap.
        """
        load_delay = 0.1  # 100ms simulated shard load
        compute_gap = 0.15  # 150ms simulated GPU compute between transitions

        original_load = FeatureIndex._load_shard

        def slow_load(self_idx, stem):
            time.sleep(load_delay)
            return original_load(self_idx, stem)

        monkeypatch.setattr(FeatureIndex, "_load_shard", slow_load)

        # Without prefetch: each transition costs load_delay, plus compute gap
        idx_no = FeatureIndex(three_shard_dir, prefetch=False)
        t0 = time.perf_counter()
        for i in range(3):
            idx_no.lookup(f"shard_{i:05d}", doc_idx=0)
            time.sleep(compute_gap)
        time_no_prefetch = time.perf_counter() - t0

        # With prefetch: first load blocks, rest hidden behind compute gap
        idx_yes = FeatureIndex(three_shard_dir, prefetch=True)
        t0 = time.perf_counter()
        for i in range(3):
            idx_yes.lookup(f"shard_{i:05d}", doc_idx=0)
            time.sleep(compute_gap)
        time_prefetch = time.perf_counter() - t0
        idx_yes.close()

        # Both: 3 * compute_gap = 450ms baseline
        # No prefetch adds: 3 * load_delay = 300ms → ~750ms total
        # Prefetch adds: ~1 * load_delay = 100ms → ~550ms total
        # Speedup ~1.36x; use 1.2x threshold for CI stability
        speedup = time_no_prefetch / time_prefetch
        assert speedup > 1.2, (
            f"Speedup {speedup:.2f}x too low. "
            f"no_prefetch={time_no_prefetch:.3f}s, prefetch={time_prefetch:.3f}s"
        )

    def test_memory_bounded_with_prefetch(self, tmp_path):
        """Peak memory with prefetch should stay under 3 shards worth."""
        n_shards = 5
        features_per_shard = 50_000
        for i in range(n_shards):
            shard = _make_shard(
                doc_ids=[0],
                segment_indices=[0],
                segment_positions=[0],
                offsets=[0, features_per_shard],
                feature_ids=list(range(features_per_shard)),
                feature_mags=[0.1] * features_per_shard,
            )
            save_features(shard, tmp_path / f"shard_{i:05d}_features")

        gc.collect()
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        tracemalloc.start()

        try:
            idx = FeatureIndex(str(tmp_path), prefetch=True)
            for i in range(n_shards):
                idx.lookup(f"shard_{i:05d}", doc_idx=0)
            idx.close()

            current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        one_shard_bytes = features_per_shard * 8
        three_shards = one_shard_bytes * 3
        assert peak < three_shards, (
            f"Peak {peak:,} bytes exceeds 3-shard budget ({three_shards:,} bytes). "
            f"Memory is not bounded at 2 shards."
        )


class TestFeatureIndexPrefetchRegression:
    """Re-run core tests with prefetch=True to verify no regressions."""

    @pytest.fixture
    def feature_dir(self, tmp_path):
        """Same as TestFeatureIndex.feature_dir."""
        shard1 = {
            "offsets": t.tensor([0, 2, 5, 7], dtype=t.int64),
            "feature_ids": t.tensor([10, 20, 30, 40, 50, 60, 70], dtype=t.int32),
            "feature_mags": t.tensor(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=t.float32
            ),
            "doc_ids": t.tensor([0, 0, 1], dtype=t.int32),
            "segment_indices": t.tensor([0, 1, 0], dtype=t.int32),
            "segment_positions": t.tensor([0, 128, 0], dtype=t.int32),
            "doc_token_counts": t.tensor([256, 128], dtype=t.int32),
            "segment_char_starts": t.tensor([0, 768, 0], dtype=t.int32),
            "segment_char_ends": t.tensor([768, 1536, 768], dtype=t.int32),
            "n_docs": 2,
            "n_segments": 3,
            "layer": 12,
            "context_window": 1024,
            "pooling": "mean_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 128,
        }
        save_features(shard1, tmp_path / "shard_00001_features")

        shard2 = {
            "offsets": t.tensor([0, 3], dtype=t.int64),
            "feature_ids": t.tensor([100, 200, 300], dtype=t.int32),
            "feature_mags": t.tensor([0.8, 0.9, 1.0], dtype=t.float32),
            "doc_ids": t.tensor([0], dtype=t.int32),
            "segment_indices": t.tensor([0], dtype=t.int32),
            "segment_positions": t.tensor([0], dtype=t.int32),
            "doc_token_counts": t.tensor([64], dtype=t.int32),
            "segment_char_starts": t.tensor([0], dtype=t.int32),
            "segment_char_ends": t.tensor([768], dtype=t.int32),
            "n_docs": 1,
            "n_segments": 1,
            "layer": 12,
            "context_window": 1024,
            "pooling": "mean_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 128,
        }
        save_features(shard2, tmp_path / "shard_00002_features")
        return str(tmp_path)

    def test_lookup_existing_doc_prefetch(self, feature_dir):
        idx = FeatureIndex(feature_dir, prefetch=True)
        segments = idx.lookup("shard_00001", doc_idx=0)
        assert segments is not None
        assert len(segments) == 2
        assert segments[0].feature_ids.tolist() == [10, 20]
        assert segments[0].feature_mags.tolist() == pytest.approx([0.1, 0.2], abs=1e-3)
        idx.close()

    def test_lookup_different_shard_prefetch(self, feature_dir):
        idx = FeatureIndex(feature_dir, prefetch=True)
        segments = idx.lookup("shard_00002", doc_idx=0)
        assert segments is not None
        assert segments[0].feature_ids.tolist() == [100, 200, 300]
        idx.close()

    def test_shard_eviction_prefetch(self, feature_dir):
        idx = FeatureIndex(feature_dir, prefetch=True)
        segs1 = idx.lookup("shard_00001", doc_idx=0)
        assert segs1 is not None
        segs2 = idx.lookup("shard_00002", doc_idx=0)
        assert segs2 is not None
        segs1_again = idx.lookup("shard_00001", doc_idx=0)
        assert segs1_again is not None
        assert segs1_again[0].feature_ids.tolist() == [10, 20]
        idx.close()

    def test_missing_shard_prefetch(self, feature_dir):
        idx = FeatureIndex(feature_dir, prefetch=True)
        with pytest.raises(KeyError, match=r"Unknown shard stem: 'nonexistent'"):
            idx.lookup("nonexistent", doc_idx=0)
        idx.close()

    def test_missing_doc_prefetch(self, feature_dir):
        idx = FeatureIndex(feature_dir, prefetch=True)
        with pytest.raises(KeyError, match=r"Document 99 not found in shard 'shard_00001'"):
            idx.lookup("shard_00001", doc_idx=99)
        idx.close()

    def test_cloned_tensors_prefetch(self, feature_dir):
        idx = FeatureIndex(feature_dir, prefetch=True)
        segments = idx.lookup("shard_00001", doc_idx=0)
        assert segments is not None
        for seg in segments:
            assert seg.feature_ids.storage_offset() == 0
            assert seg.feature_mags.storage_offset() == 0
        idx.close()

    def test_lookup_second_doc_prefetch(self, feature_dir):
        idx = FeatureIndex(feature_dir, prefetch=True)
        segments = idx.lookup("shard_00001", doc_idx=1)
        assert segments is not None
        assert len(segments) == 1
        assert segments[0].feature_ids.tolist() == [60, 70]
        idx.close()

    def test_segment_positions_preserved_prefetch(self, feature_dir):
        idx = FeatureIndex(feature_dir, prefetch=True)
        segments = idx.lookup("shard_00001", doc_idx=0)
        assert segments is not None
        assert segments[0].position == 0
        assert segments[1].position == 128
        idx.close()

    def test_segment_indices_preserved_prefetch(self, feature_dir):
        idx = FeatureIndex(feature_dir, prefetch=True)
        segments = idx.lookup("shard_00001", doc_idx=0)
        assert segments is not None
        assert segments[0].segment_idx == 0
        assert segments[1].segment_idx == 1
        idx.close()

    def test_empty_dir_warns_prefetch(self, tmp_path):
        idx = FeatureIndex(str(tmp_path), prefetch=True)
        assert idx.n_shards == 0
        with pytest.raises(KeyError, match=r"Unknown shard stem: 'anything'"):
            idx.lookup("anything", doc_idx=0)
        idx.close()

    def test_segments_sorted_by_segment_idx_prefetch(self, tmp_path):
        shard = {
            "offsets": t.tensor([0, 1, 2, 3], dtype=t.int64),
            "feature_ids": t.tensor([10, 20, 30], dtype=t.int32),
            "feature_mags": t.tensor([0.1, 0.2, 0.3], dtype=t.float32),
            "doc_ids": t.tensor([0, 0, 0], dtype=t.int32),
            "segment_indices": t.tensor([2, 0, 1], dtype=t.int32),
            "segment_positions": t.tensor([256, 0, 128], dtype=t.int32),
            "doc_token_counts": t.tensor([384], dtype=t.int32),
            "segment_char_starts": t.tensor([1536, 0, 768], dtype=t.int32),
            "segment_char_ends": t.tensor([2304, 768, 1536], dtype=t.int32),
            "n_docs": 1,
            "n_segments": 3,
            "layer": 12,
            "context_window": 1024,
            "pooling": "mean_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 128,
        }
        save_features(shard, tmp_path / "shard_00000_features")
        idx = FeatureIndex(str(tmp_path), prefetch=True)
        segments = idx.lookup("shard_00000", doc_idx=0)
        assert segments is not None
        assert [s.segment_idx for s in segments] == [0, 1, 2]
        assert [s.position for s in segments] == [0, 128, 256]
        idx.close()

    def test_frozen_dataclass_prefetch(self, feature_dir):
        idx = FeatureIndex(feature_dir, prefetch=True)
        segments = idx.lookup("shard_00001", doc_idx=0)
        assert segments is not None
        with pytest.raises(AttributeError):
            segments[0].segment_idx = 99  # ty: ignore[invalid-assignment]
        idx.close()

    def test_segment_size_from_first_shard_prefetch(self, feature_dir):
        idx = FeatureIndex(feature_dir, prefetch=True)
        assert idx.segment_size == 128
        idx.close()

    def test_empty_dir_segment_size_none_prefetch(self, tmp_path):
        idx = FeatureIndex(str(tmp_path), prefetch=True)
        assert idx.segment_size is None
        idx.close()

    def test_doc_ids_for_shard_prefetch(self, feature_dir):
        """doc_ids_for_shard returns correct IDs with prefetch=True."""
        idx = FeatureIndex(feature_dir, prefetch=True)
        doc_ids = idx.doc_ids_for_shard("shard_00001")
        assert doc_ids == [0, 1]
        idx.close()

    def test_doc_ids_for_shard_after_prefetch_miss(self, feature_dir):
        """doc_ids_for_shard returns correct IDs after a prefetch has fired."""
        idx = FeatureIndex(feature_dir, prefetch=True)
        # Lookup shard_00001 triggers prefetch of shard_00002
        idx.lookup("shard_00001", doc_idx=0)
        idx._join_inflight()
        # shard_00001 is still the LRU-cached shard
        doc_ids = idx.doc_ids_for_shard("shard_00001")
        assert doc_ids == [0, 1]
        idx.close()

    def test_doc_ids_for_shard_unknown_raises_prefetch(self, feature_dir):
        """doc_ids_for_shard raises KeyError for unknown stem even with prefetch=True."""
        idx = FeatureIndex(feature_dir, prefetch=True)
        with pytest.raises(KeyError, match="Unknown shard stem"):
            idx.doc_ids_for_shard("nonexistent")
        idx.close()


# ---------------------------------------------------------------------------
# Hypothesis property-based tests for _load_shard invariants
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Hypothesis property-based tests
# ---------------------------------------------------------------------------


_MIN_POSITIVE_FP16 = float(
    t.nextafter(
        t.tensor(0.0, dtype=t.float16),
        t.tensor(1.0, dtype=t.float16),
    ).item()
)


@st.composite
def random_shard(draw):
    """Generate a random shard dict for property-based testing."""
    n_docs = draw(st.integers(min_value=1, max_value=20))
    segs_per_doc = draw(
        st.lists(
            st.integers(min_value=1, max_value=5),
            min_size=n_docs,
            max_size=n_docs,
        )
    )

    doc_ids: list[int] = []
    segment_indices: list[int] = []
    for doc_id, n_segs in enumerate(segs_per_doc):
        for seg_idx in range(n_segs):
            doc_ids.append(doc_id)
            segment_indices.append(seg_idx)

    n_segments = len(doc_ids)

    # min_value=1: base extracted shards must have non-empty segments
    # (issue #522). save_features now rejects empty segments at the writer,
    # so the round-trip/sort properties below only apply to valid shards.
    feats_per_seg = draw(
        st.lists(
            st.integers(min_value=1, max_value=20),
            min_size=n_segments,
            max_size=n_segments,
        )
    )
    offsets = [0]
    for n_f in feats_per_seg:
        offsets.append(offsets[-1] + n_f)
    total = offsets[-1]

    feature_ids = draw(
        st.lists(
            st.integers(min_value=0, max_value=10000),
            min_size=total,
            max_size=total,
        )
    )
    feature_mags = draw(
        st.lists(
            st.one_of(
                st.just(0.0),
                st.floats(
                    min_value=_MIN_POSITIVE_FP16,
                    max_value=10.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            ),
            min_size=total,
            max_size=total,
        )
    )

    segment_positions = [idx * 128 for idx in segment_indices]

    # Optionally shuffle segment order to stress the sort path
    if draw(st.booleans()):
        import random

        order = list(range(n_segments))
        rng = random.Random(draw(st.integers(0, 10000)))
        rng.shuffle(order)
        doc_ids = [doc_ids[i] for i in order]
        segment_indices = [segment_indices[i] for i in order]
        segment_positions = [segment_positions[i] for i in order]

        # Reorder features to match shuffled segments
        new_offsets = [0]
        new_fids: list[int] = []
        new_fmags: list[float] = []
        for i in order:
            start = offsets[i]
            end = offsets[i + 1]
            new_fids.extend(feature_ids[start:end])
            new_fmags.extend(feature_mags[start:end])
            new_offsets.append(new_offsets[-1] + (end - start))
        offsets = new_offsets
        feature_ids = new_fids
        feature_mags = new_fmags

    return _make_shard(
        doc_ids=doc_ids,
        segment_indices=segment_indices,
        segment_positions=segment_positions,
        offsets=offsets,
        feature_ids=feature_ids,
        feature_mags=feature_mags,
    )


@given(
    missing_doc=st.integers(min_value=1000, max_value=10_000),
    missing_stem=st.text(alphabet="abcdef0123456789_", min_size=1, max_size=12),
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_property_present_returns_list_absent_raises(three_shard_dir, missing_doc, missing_stem):
    idx = FeatureIndex(three_shard_dir)
    for stem in idx.shard_stems:
        for doc in idx.doc_ids_for_shard(stem):
            segs = idx.lookup(stem, doc)
            assert isinstance(segs, list) and len(segs) >= 1
    known = idx.shard_stems[0]
    with pytest.raises(KeyError):
        idx.lookup(known, doc_idx=missing_doc)
    if missing_stem not in idx.shard_stems:
        with pytest.raises(KeyError):
            idx.lookup(missing_stem, doc_idx=0)
    idx.close()


class TestLoadShardHypothesis:
    """Property-based tests for _load_shard invariants."""

    @given(shard_data=random_shard())
    @settings(max_examples=50, deadline=None)
    def test_storage_independence_property(self, shard_data):
        """Every tensor from fast path has independent storage."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            save_features(shard_data, tmp_path / "shard_00000_features")
            idx = FeatureIndex(tmp)
            docs = idx._load_shard("shard_00000")
            for segs in docs.values():
                for seg in segs:
                    assert seg.feature_ids.storage_offset() == 0
                    assert seg.feature_mags.storage_offset() == 0
                    if seg.feature_ids.numel() > 0:
                        assert (
                            seg.feature_ids.untyped_storage().size()
                            == seg.feature_ids.numel()
                            * seg.feature_ids.element_size()
                        )
                    if seg.feature_mags.numel() > 0:
                        assert (
                            seg.feature_mags.untyped_storage().size()
                            == seg.feature_mags.numel()
                            * seg.feature_mags.element_size()
                        )

    @given(shard_data=random_shard())
    @settings(max_examples=50, deadline=None)
    def test_sort_order_property(self, shard_data):
        """Segments within each doc are sorted by segment_idx."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            save_features(shard_data, tmp_path / "shard_00000_features")
            idx = FeatureIndex(tmp)
            docs = idx._load_shard("shard_00000")
            for doc_id, segs in docs.items():
                indices = [s.segment_idx for s in segs]
                assert indices == sorted(indices), (
                    f"Doc {doc_id}: segments not sorted: {indices}"
                )


# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestLoadShardPerformance:
    def test_dict_build_under_budget(self, tmp_path):
        """Dict-build phase for 350K segments should complete in under 2s."""
        n_docs = 350
        segs_per_doc = 1000
        feats_per_seg = 10
        n_segments = n_docs * segs_per_doc  # 350K

        doc_ids = [d for d in range(n_docs) for _ in range(segs_per_doc)]
        segment_indices = [s for _ in range(n_docs) for s in range(segs_per_doc)]
        segment_positions = [s * 128 for s in segment_indices]
        offsets = list(range(0, n_segments * feats_per_seg + 1, feats_per_seg))
        total_feats = n_segments * feats_per_seg
        feature_ids = [i % 16384 for i in range(total_feats)]
        feature_mags = [0.1] * total_feats

        shard = _make_shard(
            doc_ids=doc_ids,
            segment_indices=segment_indices,
            segment_positions=segment_positions,
            offsets=offsets,
            feature_ids=feature_ids,
            feature_mags=feature_mags,
        )
        save_features(shard, tmp_path / "shard_00000_features")

        idx = FeatureIndex(str(tmp_path))
        shard_path = idx._shard_paths["shard_00000"]

        # Warm up OS file cache
        _ = idx._load_shard("shard_00000")

        # Measure I/O baseline (load_features only)
        io_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = load_features(shard_path)
            io_times.append(time.perf_counter() - t0)
        io_baseline = min(io_times)

        # Time _load_shard (best of 3)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = idx._load_shard("shard_00000")
            times.append(time.perf_counter() - t0)

        total = min(times)
        dict_build = total - io_baseline

        print(
            f"\n  _load_shard performance ({n_segments:,} segments):"
            f"\n    I/O baseline: {io_baseline:.3f}s"
            f"\n    total:        {total:.3f}s"
            f"\n    dict-build:   {dict_build:.3f}s"
        )
        assert dict_build < 2.0, (
            f"Dict-build took {dict_build:.3f}s, expected < 2.0s for "
            f"{n_segments:,} segments"
        )


class TestFeatureIndexDelCleanup:
    """Tests for __del__ / close() thread cleanup safety.

    When GC collects a FeatureIndex from within a prefetch thread (because the
    bound ``_prefetch_worker`` target holds the last reference to ``self``),
    ``__del__`` → ``close()`` → ``_shutdown_prefetch()`` must not call
    ``thread.join()`` on the current thread, which raises
    ``RuntimeError: cannot join current thread``.
    """

    def test_close_from_own_prefetch_thread(self, three_shard_dir):
        """close() from the thread that IS ``_prefetch_thread`` must not raise.

        Direct reproduction: we make a thread, register it as the in-flight
        worker, then have that thread call close() on itself — exactly what
        happens when __del__ fires on the prefetch thread during GC.
        """
        idx = FeatureIndex(three_shard_dir, prefetch=True)
        errors: list[RuntimeError] = []

        def call_close():
            try:
                idx.close()
            except RuntimeError as e:
                errors.append(e)

        th = threading.Thread(
            target=call_close, daemon=True, name="FeatureIndex-prefetch-shard_00001"
        )
        with idx._prefetch_lock:
            idx._prefetch_thread = th
        th.start()
        th.join(timeout=5.0)

        assert not th.is_alive(), "Thread did not finish within timeout"
        assert not errors, f"close() raised RuntimeError: {errors[0]}"
        assert idx._prefetch_thread is None, "_prefetch_thread not cleaned up"

    def test_del_from_prefetch_thread_no_unraisable(
        self, three_shard_dir, monkeypatch
    ):
        """Full lifecycle: __del__ triggered by GC on the prefetch thread.

        Scenario:
        1. lookup() starts a prefetch thread (holds ``self`` via its bound target)
        2. Main thread drops its reference (``del idx``)
        3. Prefetch finishes → Thread cleanup releases closure → last ref gone
        4. ``__del__`` fires on the prefetch thread
        5. Must not raise RuntimeError from thread.join()
        """
        unraisable: list[sys.UnraisableHookArgs] = []
        monkeypatch.setattr(sys, "unraisablehook", lambda u: unraisable.append(u))

        original_load = FeatureIndex._load_shard
        prefetch_running = threading.Event()
        proceed = threading.Event()

        def coordinated_load(self_idx, stem):
            result = original_load(self_idx, stem)
            if stem == "shard_00001":  # Only pause the prefetched shard
                prefetch_running.set()
                proceed.wait(timeout=5.0)
            return result

        monkeypatch.setattr(FeatureIndex, "_load_shard", coordinated_load)

        idx = FeatureIndex(three_shard_dir, prefetch=True)
        idx.lookup("shard_00000", doc_idx=0)  # sync load + starts prefetch

        # Wait for prefetch thread to be mid-load
        prefetch_running.wait(timeout=5.0)

        # Drop the main thread's reference while prefetch holds the only one
        del idx

        # Let prefetch finish — when Thread cleans up its target, the closure
        # releases the last reference to the FeatureIndex, triggering __del__
        proceed.set()

        # Wait for the prefetch thread to exit
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            alive = [
                th
                for th in threading.enumerate()
                if th.name.startswith("FeatureIndex-prefetch") and th.is_alive()
            ]
            if not alive:
                break
            time.sleep(0.05)

        gc.collect()

        runtime_errors = [
            e for e in unraisable if isinstance(e.exc_value, RuntimeError)
        ]
        assert not runtime_errors, (
            f"__del__ raised RuntimeError from prefetch thread: "
            f"{runtime_errors[0].exc_value}"
        )


class TestLoadShardRejectsEmptySegments:
    """Issue #522 site (e): the per-shard load contract must refuse a base
    shard that carries a zero-extracted-feature segment.

    The corrupt shard is written RAW (bypassing ``save_features``, which now
    rejects empty segments at the writer) so that ``FeatureIndex._load_shard``
    is the unit under test.
    """

    def test_feature_index_raises_on_empty_segment(self, tmp_path):
        # n_segments=3 with offsets [0, 2, 2, 4]: segment 1 is empty
        # (offsets[1] == offsets[2]). The CSR is otherwise consistent
        # (4 features total) so the ONLY corruption is the empty segment.
        data = make_valid_shard_data(n_docs=1, segs_per_doc=3, feats_per_seg=2)
        data["offsets"] = t.tensor([0, 2, 2, 4], dtype=t.int64)
        data["feature_ids"] = t.tensor([3, 5, 7, 9], dtype=t.int32)
        data["feature_mags"] = t.ones(4, dtype=t.float32)
        # make_valid_shard_data omits char boundaries; the current loader
        # contract requires them, so add length-n_segments tensors.
        data["segment_char_starts"] = t.tensor([0, 6, 12], dtype=t.int32)
        data["segment_char_ends"] = t.tensor([6, 12, 18], dtype=t.int32)
        save_raw_shard(tmp_path, "shard_00000", data)

        idx = FeatureIndex(str(tmp_path))
        with pytest.raises(CorruptFeatureShardError, match=r"segment 1\b"):
            idx.lookup("shard_00000", 0)


def _write_corrupt_empty_segment_shard(tmp_path, stem: str) -> None:
    """Raw-write a base shard whose segment 1 is empty (offsets [0,2,2,4]).

    Bypasses ``save_features`` (now strict) so ``FeatureIndex._load_shard``
    is the unit that must reject the empty segment. Mirrors
    ``TestLoadShardRejectsEmptySegments.test_feature_index_raises_on_empty_segment``.
    """
    data = make_valid_shard_data(n_docs=1, segs_per_doc=3, feats_per_seg=2)
    data["offsets"] = t.tensor([0, 2, 2, 4], dtype=t.int64)
    data["feature_ids"] = t.tensor([3, 5, 7, 9], dtype=t.int32)
    data["feature_mags"] = t.ones(4, dtype=t.float32)
    data["segment_char_starts"] = t.tensor([0, 6, 12], dtype=t.int32)
    data["segment_char_ends"] = t.tensor([6, 12, 18], dtype=t.int32)
    save_raw_shard(tmp_path, stem, data)


class TestPrefetchOfCorruptNextShard:
    """Issue #522: prefetch must NOT mask a corrupt next shard.

    Contract being pinned (``feature_index.py`` ``_prefetch_worker`` +
    ``_resolve_prefetch_or_load``): the background prefetch worker catches
    every exception from ``_load_shard`` and *logs* it at failure time — it
    never publishes a result for a failed load. So when the caller later does a
    synchronous ``lookup`` of that same (corrupt) shard,
    ``_resolve_prefetch_or_load`` first joins the worker (deterministic — no
    sleeps/polling), finds no prefetch result, and falls through to a
    synchronous ``_load_shard`` which re-raises ``CorruptFeatureShardError``.

    A future refactor that wrote prefetch state *before* the load (or that let
    the worker's swallowed error satisfy the cache) would silently turn a
    corrupt shard into a None/empty lookup — this test makes that regress loudly.
    """

    def test_prefetch_then_sync_lookup_reraises_corrupt(self, tmp_path):
        # shard_00000 VALID, shard_00001 CORRUPT (empty segment 1).
        # Sorted-stem order makes shard_00001 the prefetch target of shard_00000.
        valid = _make_shard(
            doc_ids=[0, 0],
            segment_indices=[0, 1],
            segment_positions=[0, 128],
            offsets=[0, 2, 4],
            feature_ids=[10, 20, 30, 40],
            feature_mags=[0.1, 0.2, 0.3, 0.4],
        )
        save_features(valid, tmp_path / "shard_00000_features")
        _write_corrupt_empty_segment_shard(tmp_path, "shard_00001")

        idx = FeatureIndex(str(tmp_path), prefetch=True)
        try:
            # Valid first lookup succeeds AND kicks off the background prefetch
            # of shard_00001, whose _load_shard raises inside the worker
            # (swallowed + logged, no prefetch state written).
            segs = idx.lookup("shard_00000", doc_idx=0)
            assert segs is not None
            assert segs[0].feature_ids.tolist() == [10, 20]

            # Synchronous lookup of the corrupt shard: _resolve_prefetch_or_load
            # joins the worker (deterministic), sees no prefetch hit, and
            # re-raises from the sync _load_shard. The join inside
            # _resolve_prefetch_or_load removes any race on worker completion.
            with pytest.raises(CorruptFeatureShardError, match=r"segment 1\b"):
                idx.lookup("shard_00001", doc_idx=0)
        finally:
            idx.close()


class TestDocIdsForShardRejectsEmptySegment:
    """Issue #522: doc_ids_for_shard loads via _load_shard, so it must reject
    a base shard with an empty segment exactly like lookup() does."""

    def test_doc_ids_for_shard_raises_on_empty_segment(self, tmp_path):
        _write_corrupt_empty_segment_shard(tmp_path, "shard_00000")
        idx = FeatureIndex(str(tmp_path))
        with pytest.raises(CorruptFeatureShardError, match=r"segment 1\b"):
            idx.doc_ids_for_shard("shard_00000")


class TestLoadShardDefensiveCoverage:
    """Hardening pass A (issue #740 review): both of these were flagged as
    possibly-unreachable defensive lines in _load_shard. Empirically, both
    are reachable through legitimate on-disk shard states, so both get real
    tests here rather than `# pragma: no cover`.
    """

    def test_stat_oserror_is_swallowed(self, tmp_path, monkeypatch):
        """`bytes_read` is a best-effort diagnostic counter (never a
        correctness signal — see CacheStats' docstring), so the
        `except OSError: pass` guarding `shard_path.stat().st_size` must
        swallow a transient stat() failure (e.g. an ENOENT/EACCES race with a
        concurrent scrub) without blocking the load or propagating."""
        data = make_valid_shard_data(n_docs=1, segs_per_doc=1, feats_per_seg=2)
        save_features(data, tmp_path / "shard_00000_features")
        idx = FeatureIndex(str(tmp_path))
        target = idx._shard_paths["shard_00000"]
        original_stat = Path.stat

        def flaky_stat(self_path, *a, **kw):
            if self_path == target:
                raise OSError("simulated transient stat() failure")
            return original_stat(self_path, *a, **kw)

        monkeypatch.setattr(Path, "stat", flaky_stat)

        bytes_before = idx.cache_stats.bytes_read
        segs = idx.lookup("shard_00000", 0)  # must not raise despite stat() failing
        assert segs is not None and len(segs) == 1
        assert idx.cache_stats.bytes_read == bytes_before, (
            "stat() failed -> bytes_read must stay unchanged, not partially updated"
        )

    def test_zero_segment_shard_returns_empty_dict(self, tmp_path):
        """A whole-shard `n_segments == 0` (`offsets == [0]`) is a legitimate,
        constructible on-disk shard, NOT a state the extraction pipeline
        rejects: `empty_segment_indices` (steering/feature_io.py) explicitly
        documents "A whole-shard n_segments == 0 ... is exempt" from the
        empty-segment guard, and `save_features` round-trips it cleanly
        (verified directly against `make_valid_shard_data(n_docs=0)`).
        `_load_shard`'s n_segments==0 fast-return must produce an empty
        per-doc index without raising, while still accounting load_seconds.
        """
        data = make_valid_shard_data(n_docs=0, segs_per_doc=0)
        save_features(data, tmp_path / "shard_00000_features")
        idx = FeatureIndex(str(tmp_path))
        seconds_before = idx.cache_stats.load_seconds

        docs = idx._load_shard("shard_00000")

        assert docs == {}
        assert idx.cache_stats.load_seconds >= seconds_before
        assert idx.cache_stats.loads == 1
        # Fail-fast contract is unchanged for a genuinely empty shard: any
        # doc_idx lookup raises KeyError, never a silent None/empty result.
        with pytest.raises(KeyError, match="Document 0 not found"):
            idx.lookup("shard_00000", 0)

    def test_zero_segment_shard_still_validates_requested_percentiles(self, tmp_path):
        """F4: the percentile validator precedes the zero-segment return."""
        data = make_valid_shard_data(n_docs=0, segs_per_doc=0)
        base_path = save_features(data, tmp_path / "shard_00000_features")
        checksum = "a" * 64
        pct_dir = overlay_dir(tmp_path, OverlayKind.PERCENTILE, checksum)
        write_manifest(
            pct_dir,
            checksum,
            OverlayKind.PERCENTILE,
            base_shard_paths=[base_path],
        )
        save_overlay_shard(
            {TensorKey.FEATURE_PERCENTILES: t.ones(1, dtype=t.uint8)},
            overlay_shard_path(pct_dir, "shard_00000"),
        )
        idx = FeatureIndex(str(tmp_path), percentile_checksum=checksum)

        with pytest.raises(CorruptFeatureShardError, match="numel=0"):
            idx._load_shard("shard_00000")
