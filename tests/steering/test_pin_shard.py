"""Tests for FeatureIndex.pin_shard() and val shard pinning in training.

Verifies:
1. Pinned shards are always available and never evicted by LRU
2. Pinned lookups don't interfere with LRU cache or prefetch
3. LRU transitions don't evict pinned shards
4. pin_shard raises on unknown stems
5. Double-pinning is idempotent
6. Pinned shard data matches normal load
7. Prefetch stays on track when pinned shards are accessed between training shards
8. Pickle round-trip preserves pinned data
"""


import pytest
import torch as t

from steering.conditional.feature_index import FeatureIndex
from steering.feature_io import (
    CorruptFeatureShardError,
    save_features,
)
from tests.steering._helpers import make_valid_shard_data, save_raw_shard

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEGMENT_SIZE = 16
N_DOCS = 10
N_SEGS_PER_DOC = 2
N_FEATS_PER_SEG = 5


def _mru_stem(index: FeatureIndex) -> str | None:
    return next(reversed(index._resident)) if index._resident else None


def _make_feature_dir(tmp_path, n_shards=3):
    """Create synthetic feature shards for testing."""
    feature_dir = tmp_path / "features"
    feature_dir.mkdir(parents=True)

    for shard_idx in range(n_shards):
        shard_stem = f"shard_{shard_idx:05d}"
        total_segs = N_DOCS * N_SEGS_PER_DOC
        total_feats = total_segs * N_FEATS_PER_SEG

        all_fids = t.zeros(total_feats, dtype=t.int32)
        all_mags = t.zeros(total_feats, dtype=t.float32)
        offsets = t.zeros(total_segs + 1, dtype=t.int64)
        doc_ids = t.zeros(total_segs, dtype=t.int32)
        seg_indices = t.zeros(total_segs, dtype=t.int32)
        seg_positions = t.zeros(total_segs, dtype=t.int32)
        seg_char_starts = t.zeros(total_segs, dtype=t.int32)
        seg_char_ends = t.zeros(total_segs, dtype=t.int32)

        feat_offset = 0
        seg_global = 0
        for doc_idx in range(N_DOCS):
            for seg_idx in range(N_SEGS_PER_DOC):
                doc_ids[seg_global] = doc_idx
                seg_indices[seg_global] = seg_idx
                seg_positions[seg_global] = seg_idx * SEGMENT_SIZE
                offsets[seg_global] = feat_offset
                seg_char_starts[seg_global] = seg_idx * SEGMENT_SIZE * 6
                seg_char_ends[seg_global] = (seg_idx + 1) * SEGMENT_SIZE * 6

                for f in range(N_FEATS_PER_SEG):
                    # Use shard_idx in feature IDs so we can verify correct shard
                    fid = shard_idx * 1000 + doc_idx * 10 + seg_idx * 5 + f
                    all_fids[feat_offset + f] = fid
                    all_mags[feat_offset + f] = 0.1 * (f + 1) + 0.01 * shard_idx

                feat_offset += N_FEATS_PER_SEG
                seg_global += 1

        offsets[total_segs] = feat_offset

        shard_data = {
            "offsets": offsets,
            "feature_ids": all_fids,
            "feature_mags": all_mags,
            "doc_ids": doc_ids,
            "segment_indices": seg_indices,
            "segment_positions": seg_positions,
            "doc_token_counts": t.full((N_DOCS,), 41, dtype=t.int32),
            "segment_char_starts": seg_char_starts,
            "segment_char_ends": seg_char_ends,
            "n_docs": N_DOCS,
            "n_segments": total_segs,
            "layer": 8,
            "context_window": 1024,
            "pooling": "max_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": SEGMENT_SIZE,
        }
        save_features(shard_data, feature_dir / f"{shard_stem}_features")

    return str(feature_dir)


def _make_corrupt_empty_segment_dir(tmp_path):
    """Create a dir whose single shard has an empty middle segment (issue #522).

    ``offsets=[0,2,2,4]`` → 3 segments, segment 1 empty. Written RAW (bypassing
    the now-strict ``save_features``) so ``_load_shard`` is the guard under test.
    ``_scan`` only reads metadata via ``load_features`` (no empty-segment guard),
    so construction succeeds and the shard is discoverable as ``shard_00000``.
    """
    feature_dir = tmp_path / "features"
    feature_dir.mkdir(parents=True)
    data = make_valid_shard_data(n_docs=1, segs_per_doc=3, feats_per_seg=2)
    data["offsets"] = t.tensor([0, 2, 2, 4], dtype=t.int64)
    data["feature_ids"] = t.tensor([3, 5, 7, 9], dtype=t.int32)
    data["feature_mags"] = t.ones(4, dtype=t.float32)
    # _scan asserts char boundaries exist; make_valid_shard_data omits them.
    data["segment_char_starts"] = t.tensor([0, 6, 12], dtype=t.int32)
    data["segment_char_ends"] = t.tensor([6, 12, 18], dtype=t.int32)
    save_raw_shard(feature_dir, "shard_00000", data)
    return str(feature_dir)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPinShardEmptySegment:
    """Issue #522: pinning a corrupt base shard (empty segment) must raise.

    ``pin_shard`` loads the shard eagerly via ``_load_shard``, so the
    empty-segment guard fires at pin time — the corrupt shard never gets
    silently pinned with a hole.
    """

    def test_pin_shard_raises_on_empty_segment(self, tmp_path):
        fd = _make_corrupt_empty_segment_dir(tmp_path)
        idx = FeatureIndex(fd)
        with pytest.raises(CorruptFeatureShardError, match=r"segment 1\b"):
            idx.pin_shard("shard_00000")


class TestPinShard:
    """Tests for the pin_shard() method."""

    def test_pinned_lookup_returns_data(self, tmp_path):
        """Pinned shard is immediately available for lookup."""
        fd = _make_feature_dir(tmp_path)
        idx = FeatureIndex(fd)

        idx.pin_shard("shard_00002")
        segs = idx.lookup("shard_00002", 0)
        assert segs is not None
        assert len(segs) == N_SEGS_PER_DOC

    def test_pinned_lookup_matches_normal_load(self, tmp_path):
        """Pinned shard returns identical data to a normal cached lookup."""
        fd = _make_feature_dir(tmp_path)

        # Load via normal path
        idx1 = FeatureIndex(fd)
        normal_segs = idx1.lookup("shard_00001", 3)

        # Load via pinned path
        idx2 = FeatureIndex(fd)
        idx2.pin_shard("shard_00001")
        pinned_segs = idx2.lookup("shard_00001", 3)

        assert normal_segs is not None and pinned_segs is not None
        assert len(normal_segs) == len(pinned_segs)
        for ns, ps in zip(normal_segs, pinned_segs, strict=True):
            assert t.equal(ns.feature_ids, ps.feature_ids)
            assert t.equal(ns.feature_mags, ps.feature_mags)
            assert ns.segment_idx == ps.segment_idx
            assert ns.char_start == ps.char_start
            assert ns.char_end == ps.char_end

    def test_pinned_not_evicted_by_lru(self, tmp_path):
        """Loading another shard into LRU doesn't evict the pinned shard."""
        fd = _make_feature_dir(tmp_path)
        idx = FeatureIndex(fd)

        idx.pin_shard("shard_00002")

        # Load shard 0 into LRU (evicts nothing — LRU was empty)
        idx.lookup("shard_00000", 0)
        # Load shard 1 into LRU (evicts shard 0)
        idx.lookup("shard_00001", 0)

        # Pinned shard still available
        segs = idx.lookup("shard_00002", 0)
        assert segs is not None
        # Verify it's the right shard by checking feature IDs
        first_fid = segs[0].feature_ids[0].item()
        assert first_fid >= 2000, f"Expected shard_00002 fids (>=2000), got {first_fid}"

    def test_lru_not_evicted_by_pinned_lookup(self, tmp_path):
        """Looking up a pinned shard doesn't evict the LRU-cached training shard."""
        fd = _make_feature_dir(tmp_path)
        idx = FeatureIndex(fd)

        idx.pin_shard("shard_00002")

        # Load shard 0 into LRU
        idx.lookup("shard_00000", 0)
        assert _mru_stem(idx) == "shard_00000"

        # Lookup pinned shard — should NOT change LRU
        idx.lookup("shard_00002", 0)
        assert _mru_stem(idx) == "shard_00000", (
            "Pinned lookup should not evict LRU cache"
        )

        # LRU shard still cached (instant lookup, no reload)
        segs = idx.lookup("shard_00000", 5)
        assert segs is not None

    def test_pin_unknown_shard_raises(self, tmp_path):
        """Pinning a non-existent shard raises KeyError."""
        fd = _make_feature_dir(tmp_path)
        idx = FeatureIndex(fd)

        import pytest
        with pytest.raises(KeyError, match="Unknown shard stem"):
            idx.pin_shard("shard_99999")

    def test_double_pin_is_idempotent(self, tmp_path):
        """Pinning the same shard twice is safe and doesn't duplicate data."""
        fd = _make_feature_dir(tmp_path)
        idx = FeatureIndex(fd)

        idx.pin_shard("shard_00001")
        idx.pin_shard("shard_00001")  # should not raise or duplicate

        assert len(idx._pinned) == 1
        segs = idx.lookup("shard_00001", 0)
        assert segs is not None

    def test_pinned_lookup_is_fast(self, tmp_path):
        """Pinned lookup is O(1) dict access, not disk I/O."""
        fd = _make_feature_dir(tmp_path)
        idx = FeatureIndex(fd)
        idx.pin_shard("shard_00002")

        import time
        t0 = time.monotonic()
        for _ in range(10000):
            idx.lookup("shard_00002", 0)
        elapsed = time.monotonic() - t0

        # 10K lookups should take < 100ms (pure dict.get)
        assert elapsed < 0.1, f"10K pinned lookups took {elapsed:.3f}s (expected < 0.1s)"

    def test_pinned_survives_pickle_roundtrip(self, tmp_path):
        """Pinned data survives pickle (for multiprocessing)."""
        import pickle

        fd = _make_feature_dir(tmp_path)
        idx = FeatureIndex(fd)
        idx.pin_shard("shard_00001")

        # Pickle round-trip
        data = pickle.dumps(idx)
        idx2 = pickle.loads(data)

        # Pinned shard still there
        assert "shard_00001" in idx2._pinned
        segs = idx2.lookup("shard_00001", 0)
        assert segs is not None
        assert len(segs) == N_SEGS_PER_DOC

    def test_pinned_with_prefetch_no_interference(self, tmp_path):
        """Prefetch operates normally when pinned shards are accessed."""
        fd = _make_feature_dir(tmp_path)
        idx = FeatureIndex(fd, prefetch=True)
        idx.pin_shard("shard_00002")

        # Load shard 0 (triggers prefetch of shard 1)
        idx.lookup("shard_00000", 0)

        # Access pinned shard — should NOT interfere with prefetch
        idx.lookup("shard_00002", 0)

        # Prefetch should still be for shard 1
        idx._join_inflight()
        with idx._prefetch_lock:
            assert idx._prefetch_stem == "shard_00001"
            assert idx._prefetch_result is not None

    def test_val_shard_simulation(self, tmp_path):
        """Simulate the train/val pattern: alternating training and val lookups."""
        fd = _make_feature_dir(tmp_path, n_shards=5)
        idx = FeatureIndex(fd, prefetch=True)

        # Pin shard 4 as "val"
        idx.pin_shard("shard_00004")

        # Simulate training on shards 0, 1, 2, 3 with val lookups between
        for train_shard_idx in range(4):
            train_stem = f"shard_{train_shard_idx:05d}"

            # Training lookups
            for doc_idx in range(N_DOCS):
                segs = idx.lookup(train_stem, doc_idx)
                assert segs is not None

            # Val lookup (should use pinned, not evict training shard)
            val_segs = idx.lookup("shard_00004", 0)
            assert val_segs is not None

            # Training shard should still be cached
            assert _mru_stem(idx) == train_stem, (
                f"Expected LRU to hold {train_stem}, got {_mru_stem(idx)}"
            )

    def test_doc_ids_for_shard_with_pinned(self, tmp_path):
        """doc_ids_for_shard works correctly for pinned shards."""
        fd = _make_feature_dir(tmp_path)
        idx = FeatureIndex(fd)
        idx.pin_shard("shard_00001")

        # doc_ids_for_shard goes through a different code path
        doc_ids = idx.doc_ids_for_shard("shard_00001")
        assert doc_ids == list(range(N_DOCS))

    def test_doc_ids_for_shard_uses_lru_not_pinned(self, tmp_path):
        """doc_ids_for_shard loads via LRU even when the shard is also pinned."""

        fd = _make_feature_dir(tmp_path)
        idx = FeatureIndex(fd)

        # Pin shard_00001 (it sits in _pinned, not LRU)
        idx.pin_shard("shard_00001")

        # Load shard_00000 into LRU
        idx.lookup("shard_00000", doc_idx=0)
        assert _mru_stem(idx) == "shard_00000"

        # doc_ids_for_shard("shard_00001") should load via LRU, evicting shard_00000
        doc_ids = idx.doc_ids_for_shard("shard_00001")
        assert _mru_stem(idx) == "shard_00001"
        assert doc_ids == list(range(N_DOCS))
