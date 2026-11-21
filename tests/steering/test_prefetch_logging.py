"""Tests verifying that prefetch hit/miss messages use DEBUG level (not INFO),
and that pin_shard() stays at INFO level.

After commit 04c8283, prefetch diagnostic messages were downgraded from INFO
to DEBUG so they don't pollute training logs during normal operation.
"""

from io import StringIO

import pytest
import torch as t
from loguru import logger

from steering.conditional.feature_index import FeatureIndex
from steering.feature_io import save_features

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEGMENT_SIZE = 16
N_DOCS = 2
N_SEGS_PER_DOC = 2
N_FEATS_PER_SEG = 5


def _make_shards(tmp_path, n_shards=3, n_docs=N_DOCS):
    """Create minimal synthetic shards for prefetch testing."""
    feature_dir = tmp_path / "features"
    feature_dir.mkdir(parents=True)

    for shard_idx in range(n_shards):
        shard_stem = f"shard_{shard_idx:05d}"
        total_segs = n_docs * N_SEGS_PER_DOC
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
        for doc_idx in range(n_docs):
            for seg_idx in range(N_SEGS_PER_DOC):
                doc_ids[seg_global] = doc_idx
                seg_indices[seg_global] = seg_idx
                seg_positions[seg_global] = seg_idx * SEGMENT_SIZE
                offsets[seg_global] = feat_offset
                seg_char_starts[seg_global] = seg_idx * SEGMENT_SIZE * 6
                seg_char_ends[seg_global] = (seg_idx + 1) * SEGMENT_SIZE * 6

                for f in range(N_FEATS_PER_SEG):
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
            "doc_token_counts": t.full((n_docs,), 41, dtype=t.int32),
            "segment_char_starts": seg_char_starts,
            "segment_char_ends": seg_char_ends,
            "n_docs": n_docs,
            "n_segments": total_segs,
            "layer": 8,
            "context_window": 1024,
            "pooling": "max_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": SEGMENT_SIZE,
        }
        save_features(shard_data, feature_dir / f"{shard_stem}_features")

    return str(feature_dir)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPrefetchLogLevels:
    """Verify prefetch hit/miss messages use DEBUG, pin_shard uses INFO."""

    @pytest.fixture
    def shard_dir(self, tmp_path):
        return _make_shards(tmp_path, n_shards=3)

    def test_prefetch_hit_logged_at_debug(self, shard_dir):
        """Sequential access (shard_00000 -> join -> shard_00001) triggers a
        prefetch hit, which must be logged at DEBUG, not INFO."""
        idx = FeatureIndex(shard_dir, prefetch=True)
        log_output = StringIO()
        handler_id = logger.add(log_output, format="{level} | {message}", level="DEBUG")
        try:
            # First lookup kicks off prefetch of shard_00001
            idx.lookup("shard_00000", doc_idx=0)
            # Wait for prefetch to complete before the next lookup
            idx._join_inflight()
            # This lookup should consume the prefetch result -> hit
            idx.lookup("shard_00001", doc_idx=0)
        finally:
            idx.close()
            logger.remove(handler_id)

        captured = log_output.getvalue()
        hit_lines = [ln for ln in captured.splitlines() if "Prefetch hit" in ln]
        if hit_lines:
            for line in hit_lines:
                assert "DEBUG" in line, (
                    f"Expected 'Prefetch hit' at DEBUG level, got: {line!r}"
                )

    def test_prefetch_miss_logged_at_debug(self, shard_dir):
        """Skipping from shard_00000 to shard_00002 causes a prefetch miss,
        which must be logged at DEBUG, not INFO."""
        idx = FeatureIndex(shard_dir, prefetch=True)
        log_output = StringIO()
        handler_id = logger.add(log_output, format="{level} | {message}", level="DEBUG")
        try:
            # First lookup kicks off prefetch of shard_00001
            idx.lookup("shard_00000", doc_idx=0)
            # Wait for the prefetch thread to finish
            idx._join_inflight()
            # Skip to shard_00002 — prefetch had shard_00001, causing a miss
            idx.lookup("shard_00002", doc_idx=0)
        finally:
            idx.close()
            logger.remove(handler_id)

        captured = log_output.getvalue()
        miss_lines = [ln for ln in captured.splitlines() if "Prefetch miss" in ln]
        if miss_lines:
            for line in miss_lines:
                assert "DEBUG" in line, (
                    f"Expected 'Prefetch miss' at DEBUG level, got: {line!r}"
                )

    def test_no_info_prefetch_messages_during_normal_access(self, shard_dir):
        """A full sequential scan must not emit any prefetch-related INFO messages."""
        idx = FeatureIndex(shard_dir, prefetch=True)
        log_output = StringIO()
        # Capture only INFO and above — DEBUG messages will be excluded
        handler_id = logger.add(log_output, format="{level} | {message}", level="INFO")
        try:
            for shard_idx in range(3):
                shard_stem = f"shard_{shard_idx:05d}"
                for doc_idx in range(N_DOCS):
                    idx.lookup(shard_stem, doc_idx=doc_idx)
            idx._join_inflight()
        finally:
            idx.close()
            logger.remove(handler_id)

        captured = log_output.getvalue()
        # Filter out known legitimate INFO messages
        info_lines = [
            ln for ln in captured.splitlines()
            if ln.strip()
            and "scanned" not in ln.lower()
            and "pinning" not in ln.lower()
        ]
        prefetch_lines = [
            ln for ln in info_lines
            if "prefetch" in ln.lower()
        ]
        assert not prefetch_lines, (
            "Unexpected prefetch-related INFO messages found:\n"
            + "\n".join(prefetch_lines)
        )

    def test_pin_shard_logged_at_info(self, shard_dir):
        """pin_shard() must emit a 'Pinning' message at INFO level."""
        idx = FeatureIndex(shard_dir)
        log_output = StringIO()
        handler_id = logger.add(log_output, format="{level} | {message}", level="INFO")
        try:
            idx.pin_shard("shard_00001")
        finally:
            idx.close()
            logger.remove(handler_id)

        captured = log_output.getvalue()
        pinning_lines = [ln for ln in captured.splitlines() if "Pinning" in ln]
        assert pinning_lines, "Expected at least one 'Pinning' log message at INFO level"
        for line in pinning_lines:
            assert "INFO" in line, (
                f"Expected 'Pinning' at INFO level, got: {line!r}"
            )
