"""Memory profiling tests for streaming feature loading.

Verifies that FeatureIndex and compute_statistics use O(1 shard) memory
by creating multiple shards and measuring peak allocation with tracemalloc.
"""

import gc
import tracemalloc

import torch as t

from steering.conditional.feature_index import FeatureIndex
from steering.feature_io import save_features
from tests.steering._compute_statistics_reference import stream_pass1


def _make_large_shard(
    tmp_path, shard_idx: int, n_segments: int = 1000, features_per_seg: int = 100
):
    """Create a shard with known size for memory testing."""
    total_features = n_segments * features_per_seg
    seg_positions = t.arange(n_segments, dtype=t.int32) * 128
    seg_char_starts = seg_positions * 6
    seg_char_ends = (seg_positions + 128) * 6
    shard = {
        "offsets": t.arange(0, total_features + 1, features_per_seg, dtype=t.int64),
        "feature_ids": t.randint(0, 1000, (total_features,), dtype=t.int32),
        "feature_mags": t.rand(total_features, dtype=t.float32),
        "doc_ids": t.zeros(n_segments, dtype=t.int32),
        "segment_indices": t.arange(n_segments, dtype=t.int32),
        "segment_positions": seg_positions,
        "doc_token_counts": t.tensor([n_segments * 128], dtype=t.int32),
        "segment_char_starts": seg_char_starts,
        "segment_char_ends": seg_char_ends,
        "n_docs": 1,
        "n_segments": n_segments,
        "layer": 12,
        # Wide enough to keep every emitted segment position: a narrower window
        # would describe geometry no extraction run can produce.
        "context_window": n_segments * 128,
        "pooling": "mean_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 128,
    }
    save_features(shard, tmp_path / f"shard_{shard_idx:05d}_features")


class TestFeatureIndexMemory:
    def test_lazy_loading_uses_constant_memory(self, tmp_path):
        """Peak memory should NOT scale with number of shards."""
        n_shards = 5
        for i in range(n_shards):
            _make_large_shard(tmp_path, i, n_segments=5000, features_per_seg=50)

        gc.collect()
        tracemalloc.start()

        idx = FeatureIndex(str(tmp_path))

        # Access each shard sequentially
        for i in range(n_shards):
            shard_stem = f"shard_{i:05d}"
            result = idx.lookup(shard_stem, 0)
            assert result is not None

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Each shard: 5000 segs * 50 features * (4+4) bytes = ~2MB raw tensors
        # With lazy loading, peak should be ~1 shard + overhead, NOT n_shards * shard_size
        # Allow generous margin but should be well under n_shards * 2MB = 10MB
        one_shard_estimate_mb = 5000 * 50 * 8 / (1024 * 1024)  # ~1.9MB
        assert peak / (1024 * 1024) < one_shard_estimate_mb * 3, (
            f"Peak {peak / (1024 * 1024):.1f}MB exceeds 3x one-shard estimate "
            f"({one_shard_estimate_mb:.1f}MB). Lazy loading may not be working."
        )


class TestComputeStatisticsMemory:
    def test_stream_pass1_constant_memory(self, tmp_path):
        """stream_pass1 should use O(accumulators) memory, not O(shards)."""
        n_shards = 5
        for i in range(n_shards):
            _make_large_shard(tmp_path, i, n_segments=2000, features_per_seg=50)

        gc.collect()
        tracemalloc.start()

        # Pass n_sae_features explicitly to avoid the extra full scan to
        # discover max feature id (which would load each shard an extra time).
        result = stream_pass1(str(tmp_path), n_sae_features=1000)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Accumulators: 1000 features * 5 tensors * 8 bytes = ~40KB
        # One shard load: 2000 * 50 * 8 = ~800KB
        # Peak should be well under 5 * 800KB = 4MB (that would be loading all shards)
        assert peak / (1024 * 1024) < 4, (
            f"Peak {peak / (1024 * 1024):.1f}MB seems too high for streaming pass1"
        )
        assert result.n_shards == n_shards
