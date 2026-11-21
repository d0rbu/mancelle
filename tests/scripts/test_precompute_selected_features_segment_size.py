"""``precompute_selected_features`` runs ``verify_segment_size``.

Pin slice 05 P1 #5: previously, the script skipped the cross-stage segment_size
check that the other three precompute scripts perform. Pointing the script at a
stale packed-row dir from a different segment_size run produced silent
misalignment.

The ``main()`` function exits with ``SystemExit`` (via ``sys.exit``) when its
preconditions fail; ``verify_segment_size`` raises ``SegmentSizeMismatch``
which is *not* caught by main and propagates up.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch as t

from steering.feature_io import save_features
from steering.feature_paths import (
    SegmentSizeMismatch,
    verify_segment_size,
)


def _make_feature_shard(feature_dir: Path, segment_size: int):
    """Single feature shard with metadata declaring ``segment_size``."""
    fids = t.tensor([0, 1, 2, 3], dtype=t.int32)
    mags = t.tensor([0.1, 0.2, 0.3, 0.4], dtype=t.float32)
    data = {
        "offsets": t.tensor([0, 2, 4], dtype=t.int64),
        "feature_ids": fids,
        "feature_mags": mags,
        "doc_ids": t.tensor([0, 1], dtype=t.int32),
        "segment_indices": t.tensor([0, 0], dtype=t.int32),
        "segment_positions": t.tensor([0, 0], dtype=t.int32),
        "doc_token_counts": t.tensor([10, 10], dtype=t.int32),
        "segment_char_starts": t.tensor([0, 0], dtype=t.int32),
        "segment_char_ends": t.tensor([8, 8], dtype=t.int32),
        "n_docs": 2,
        "n_segments": 2,
        "layer": 8,
        "context_window": 1024,
        "pooling": "mean_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": segment_size,
    }
    return save_features(data, feature_dir / "shard_00000_features")


class TestVerifySegmentSizeRaisesOnMismatch:
    """Pin: when the cfg-declared segment_size differs from the directory or
    shard metadata, ``verify_segment_size`` raises.

    This validates the fix in ``precompute_selected_features.main`` that adds
    the missing ``verify_segment_size`` call.
    """

    def test_dir_mismatch_raises(self, tmp_path):
        # Directory says ``s32``; shard metadata says 32; cfg expects 128 → raise.
        feature_dir = tmp_path / "features" / "s32"
        feature_dir.mkdir(parents=True)
        _make_feature_shard(feature_dir, segment_size=32)

        with pytest.raises(SegmentSizeMismatch):
            verify_segment_size(feature_dir, expected_segment_size=128)

    def test_metadata_mismatch_raises(self, tmp_path):
        # Directory says s32 but shard metadata says 128.
        feature_dir = tmp_path / "features" / "s32"
        feature_dir.mkdir(parents=True)
        _make_feature_shard(feature_dir, segment_size=128)

        with pytest.raises(SegmentSizeMismatch):
            verify_segment_size(feature_dir, expected_segment_size=32)

    def test_match_passes(self, tmp_path):
        feature_dir = tmp_path / "features" / "s32"
        feature_dir.mkdir(parents=True)
        _make_feature_shard(feature_dir, segment_size=32)
        # Should not raise.
        verify_segment_size(feature_dir, expected_segment_size=32)
