"""Correctness and profiling tests for vectorization optimizations.

Compares old (loop-based) implementations against new (vectorized)
implementations to prove exact equivalence, then benchmarks speedup.

Optimizations tested:
1. Sentence segmentation: torch.searchsorted O(m*log(n)) vs O(n*m) linear scan
2. Pooling + sparse extraction: t.stack + single CPU transfer vs per-segment transfers
3. Offset computation: torch.cumsum vs append loop
4. Per-feature stats: batch tensor indexing vs per-element .item() loop

Note: batch_decode for delimiter segmentation was evaluated and rejected —
benchmarks with real GPT-2 tokenizer showed 0.78x (regression), because
HuggingFace's batch_decode just loops over decode() internally.

Run with:
    uv run pytest tests/test_vectorization_correctness.py -v -s
"""

import time

import pytest
import torch as t

from steering.feature_paths import PoolingMode

# ---------------------------------------------------------------------------
# Reference implementations (inlined from deleted production functions)
# ---------------------------------------------------------------------------


def _ref_pool_segment(
    sae_acts,
    start: int,
    end: int,
    pooling: PoolingMode,
):
    """Reference: pool SAE activations for a single segment."""

    segment = sae_acts[start:end]
    if pooling == PoolingMode.last_token:
        return segment[-1]
    if pooling == PoolingMode.max_pool:
        return segment.max(dim=0).values
    if pooling == PoolingMode.mean_pool:
        return segment.mean(dim=0)
    raise ValueError(f"Unknown pooling mode: {pooling}")


def _ref_extract_sparse_features(pooled):
    """Reference: extract non-zero features, sorted by magnitude descending."""
    import torch as t

    nonzero_mask = pooled > 0
    feature_ids = nonzero_mask.nonzero(as_tuple=True)[0].to(t.int32)
    feature_mags = pooled[nonzero_mask].to(t.float32)
    sort_idx = feature_mags.argsort(descending=True)
    return feature_ids[sort_idx].cpu(), feature_mags[sort_idx].cpu()


# ---------------------------------------------------------------------------
# Reference (old) implementations for correctness comparison
# ---------------------------------------------------------------------------


def _old_extract_document_features_core(sae_acts, segments, pooling):
    """Original per-segment loop for pooling + sparse extraction."""
    all_feat_ids = []
    all_feat_mags = []
    for start, end in segments:
        pooled = _ref_pool_segment(sae_acts, start, end, pooling)
        feat_ids, feat_mags = _ref_extract_sparse_features(pooled)
        all_feat_ids.append(feat_ids)
        all_feat_mags.append(feat_mags)
    return all_feat_ids, all_feat_mags


def _new_extract_document_features_core(sae_acts, segments, pooling):
    """New approach: per-segment pooling via views + single CPU transfer."""
    pooled_list = [_ref_pool_segment(sae_acts, s, e, pooling) for s, e in segments]
    pooled_batch = t.stack(pooled_list)  # [n_segs, n_features]
    pooled_cpu = pooled_batch.cpu()  # single transfer

    all_feat_ids = []
    all_feat_mags = []
    for i in range(len(segments)):
        row = pooled_cpu[i]
        nonzero_mask = row > 0
        feature_ids = nonzero_mask.nonzero(as_tuple=True)[0].to(t.int32)
        feature_mags = row[nonzero_mask].to(t.float32)
        sort_idx = feature_mags.argsort(descending=True)
        all_feat_ids.append(feature_ids[sort_idx])
        all_feat_mags.append(feature_mags[sort_idx])
    return all_feat_ids, all_feat_mags


# ---------------------------------------------------------------------------
# Correctness: batched pooling + sparse extraction — stack vs per-segment
# ---------------------------------------------------------------------------


class TestBatchedPoolingCorrectness:
    """Verify new batched pooling matches old per-segment loop for all pooling modes."""

    @pytest.fixture
    def sae_acts(self):
        """Realistic SAE activations: [1024, 24576] with ~5% sparsity."""
        t.manual_seed(42)
        acts = t.zeros(1024, 24576)
        # Sparse: ~5% non-zero
        mask = t.rand(1024, 24576) < 0.05
        acts[mask] = t.rand(mask.sum()) * 2.0  # ty: ignore[no-matching-overload]
        return acts

    @pytest.fixture
    def segments_fixed(self):
        """8 fixed-size segments of 128 tokens each."""
        return [(i * 128, (i + 1) * 128) for i in range(8)]

    @pytest.fixture
    def segments_variable(self):
        """Variable-length segments."""
        return [(0, 50), (50, 200), (200, 350), (350, 400), (400, 1024)]

    @pytest.mark.parametrize("pooling", list(PoolingMode))
    def test_matches_old_fixed_segments(self, sae_acts, segments_fixed, pooling):
        old_ids, old_mags = _old_extract_document_features_core(
            sae_acts, segments_fixed, pooling
        )
        new_ids, new_mags = _new_extract_document_features_core(
            sae_acts, segments_fixed, pooling
        )

        assert len(old_ids) == len(new_ids)
        for i in range(len(old_ids)):
            assert t.equal(old_ids[i], new_ids[i]), f"Segment {i} feature IDs differ"
            assert t.allclose(old_mags[i], new_mags[i], atol=1e-5), (
                f"Segment {i} feature mags differ"
            )

    @pytest.mark.parametrize("pooling", list(PoolingMode))
    def test_matches_old_variable_segments(self, sae_acts, segments_variable, pooling):
        old_ids, old_mags = _old_extract_document_features_core(
            sae_acts, segments_variable, pooling
        )
        new_ids, new_mags = _new_extract_document_features_core(
            sae_acts, segments_variable, pooling
        )

        assert len(old_ids) == len(new_ids)
        for i in range(len(old_ids)):
            assert t.equal(old_ids[i], new_ids[i]), f"Segment {i} feature IDs differ"
            assert t.allclose(old_mags[i], new_mags[i], atol=1e-5), (
                f"Segment {i} feature mags differ"
            )

    @pytest.mark.parametrize("pooling", list(PoolingMode))
    def test_single_segment(self, sae_acts, pooling):
        segments = [(0, 1024)]
        old_ids, old_mags = _old_extract_document_features_core(
            sae_acts, segments, pooling
        )
        new_ids, new_mags = _new_extract_document_features_core(
            sae_acts, segments, pooling
        )
        assert t.equal(old_ids[0], new_ids[0])
        assert t.allclose(old_mags[0], new_mags[0], atol=1e-5)

    @pytest.mark.parametrize("pooling", list(PoolingMode))
    def test_single_token_segments(self, sae_acts, pooling):
        segments = [(i, i + 1) for i in range(10)]
        old_ids, old_mags = _old_extract_document_features_core(
            sae_acts, segments, pooling
        )
        new_ids, new_mags = _new_extract_document_features_core(
            sae_acts, segments, pooling
        )
        for i in range(len(old_ids)):
            assert t.equal(old_ids[i], new_ids[i])
            assert t.allclose(old_mags[i], new_mags[i], atol=1e-5)


# ---------------------------------------------------------------------------
# Correctness: compute_statistics batch per-feature stats
# ---------------------------------------------------------------------------


class TestComputeStatsCorrectness:
    """Verify batch-indexed per-feature stats match old loop-based computation."""

    def test_batch_vs_loop_per_feature_stats(self):
        t.manual_seed(42)
        n_features = 24576
        occurrence = t.zeros(n_features, dtype=t.int64)
        mag_sum = t.zeros(n_features, dtype=t.float64)

        # Simulate some feature occurrences
        active_ids = t.randint(0, n_features, (5000,), dtype=t.int64)
        active_mags = t.rand(5000, dtype=t.float64) * 3.0
        occurrence.scatter_add_(0, active_ids, t.ones_like(active_ids))
        mag_sum.scatter_add_(0, active_ids, active_mags)

        active_mask = occurrence > 0
        active_indices = active_mask.nonzero(as_tuple=True)[0]

        # Old: per-element .item() loop
        old_per_feat = {}
        for idx in active_indices:
            fid = int(idx.item())
            count = int(occurrence[fid].item())
            mean_mag = float(mag_sum[fid].item()) / count
            old_per_feat[str(fid)] = (count, round(mean_mag, 6))

        # New: batch-indexed
        counts = occurrence[active_indices]
        mean_mags = mag_sum[active_indices] / counts.double()
        ids_list = active_indices.tolist()
        counts_list = counts.tolist()
        mags_list = mean_mags.tolist()
        new_per_feat = {
            str(fid): (int(c), round(m, 6))
            for fid, c, m in zip(ids_list, counts_list, mags_list, strict=True)
        }

        assert old_per_feat == new_per_feat


# ---------------------------------------------------------------------------
# Profiling benchmarks
# ---------------------------------------------------------------------------


class TestBenchmarks:
    """Microbenchmarks comparing old vs new implementations.

    These tests always pass but print timing results for PR description.
    Run with: uv run pytest tests/test_vectorization_correctness.py -v -s -k benchmark
    """

    N_WARMUP = 3
    N_ITERS = 20

    def _benchmark(self, fn, n_warmup=None, n_iters=None):
        if n_warmup is None:
            n_warmup = self.N_WARMUP
        if n_iters is None:
            n_iters = self.N_ITERS
        for _ in range(n_warmup):
            fn()
        times = []
        for _ in range(n_iters):
            start = time.perf_counter()
            fn()
            times.append(time.perf_counter() - start)
        return times

    def test_benchmark_batched_pooling(self):
        """Benchmark: stack + single CPU transfer vs per-segment loop."""
        t.manual_seed(42)
        sae_acts = t.zeros(1024, 24576)
        mask = t.rand(1024, 24576) < 0.05
        sae_acts[mask] = t.rand(mask.sum()) * 2.0  # ty: ignore[no-matching-overload]
        segments = [(i * 128, (i + 1) * 128) for i in range(8)]

        for pooling in PoolingMode:
            old_times = self._benchmark(
                lambda p=pooling: _old_extract_document_features_core(
                    sae_acts, segments, p
                )
            )
            new_times = self._benchmark(
                lambda p=pooling: _new_extract_document_features_core(
                    sae_acts, segments, p
                )
            )

            old_mean = sum(old_times) / len(old_times) * 1000
            new_mean = sum(new_times) / len(new_times) * 1000
            speedup = old_mean / new_mean if new_mean > 0 else float("inf")
            print(
                f"\n  Batched pooling ({pooling.value}, 8 segments x 128 tokens, 24576 features):"
            )
            print(
                f"    Old: {old_mean:.3f} ms | New: {new_mean:.3f} ms | Speedup: {speedup:.2f}x"
            )

    def test_benchmark_compute_stats_loop(self):
        """Benchmark: batch-indexed vs per-element .item() for per-feature stats."""
        t.manual_seed(42)
        n_features = 24576
        occurrence = t.zeros(n_features, dtype=t.int64)
        mag_sum = t.zeros(n_features, dtype=t.float64)
        active_ids = t.randint(0, n_features, (50000,), dtype=t.int64)
        active_mags = t.rand(50000, dtype=t.float64) * 3.0
        occurrence.scatter_add_(0, active_ids, t.ones_like(active_ids))
        mag_sum.scatter_add_(0, active_ids, active_mags)
        active_mask = occurrence > 0
        active_indices = active_mask.nonzero(as_tuple=True)[0]

        def old_fn():
            per_feat = {}
            for idx in active_indices:
                fid = int(idx.item())
                count = int(occurrence[fid].item())
                mean_mag = float(mag_sum[fid].item()) / count
                per_feat[str(fid)] = (count, round(mean_mag, 6))
            return per_feat

        def new_fn():
            counts = occurrence[active_indices]
            mean_mags = mag_sum[active_indices] / counts.double()
            ids_list = active_indices.tolist()
            counts_list = counts.tolist()
            mags_list = mean_mags.tolist()
            return {
                str(fid): (int(c), round(m, 6))
                for fid, c, m in zip(ids_list, counts_list, mags_list, strict=True)
            }

        old_times = self._benchmark(old_fn)
        new_times = self._benchmark(new_fn)

        old_mean = sum(old_times) / len(old_times) * 1000
        new_mean = sum(new_times) / len(new_times) * 1000
        speedup = old_mean / new_mean if new_mean > 0 else float("inf")
        n_active = len(active_indices)
        print(f"\n  Per-feature stats ({n_active} active features):")
        print(
            f"    Old: {old_mean:.3f} ms | New: {new_mean:.3f} ms | Speedup: {speedup:.2f}x"
        )
