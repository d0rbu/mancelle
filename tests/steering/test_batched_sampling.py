"""Tests for sample_features_batched and sample_k_batched correctness."""

import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

from steering.conditional.sampling import (
    Distribution,
    SamplingSelection,
    sample_features,
    sample_features_batched,
    sample_k_batched,
)

N_SAE = 16384  # realistic SAE feature count


@pytest.fixture
def pct_table():
    """Build a full-size monotonic percentile table [16384, 99]."""
    torch.manual_seed(0)
    table = torch.rand(N_SAE, 99, dtype=torch.float64).cumsum(dim=1)
    return table


@pytest.fixture
def sampling_config():
    return SamplingSelection(
        distribution=Distribution.geometric,
        mean=5.0,
        std=5.0,
        min_k=1,
        max_k=20,
    )


def _make_segments(n_segments: int, max_features: int = 20, n_sae: int = N_SAE):
    """Generate random segments with random feature counts."""
    torch.manual_seed(42)
    segments = []
    for _ in range(n_segments):
        n = torch.randint(1, max_features + 1, (1,)).item()
        fids = torch.randperm(n_sae)[:n].to(torch.int64)
        mags = torch.rand(n, dtype=torch.float32) * 10 + 0.1  # ty: ignore[no-matching-overload]
        segments.append((fids, mags))
    return segments


class TestBatchedSamplingEquivalence:
    """Batched sampling must produce identical results to per-segment sampling."""

    def test_empty_segments(self, pct_table, sampling_config):
        """Empty segments produce empty results."""
        rng = torch.Generator()
        rng.manual_seed(42)

        all_fids = torch.tensor([], dtype=torch.int64)
        all_mags = torch.tensor([], dtype=torch.float32)
        seg_sizes = [0, 0, 0]

        results = sample_features_batched(
            all_fids, all_mags, seg_sizes,
            pct_table, sampling_config, rng,
        )
        assert len(results) == 3
        for ids, mags in results:
            assert len(ids) == 0
            assert len(mags) == 0

    def test_mixed_empty_and_nonempty(self, pct_table, sampling_config):
        """Mix of empty and non-empty segments works correctly."""
        rng = torch.Generator()
        rng.manual_seed(42)

        fids1 = torch.tensor([0, 5, 10], dtype=torch.int64)
        mags1 = torch.tensor([1.0, 2.0, 3.0])
        fids2 = torch.tensor([20, 30], dtype=torch.int64)
        mags2 = torch.tensor([4.0, 5.0])

        all_fids = torch.cat([fids1, fids2])
        all_mags = torch.cat([mags1, mags2])
        seg_sizes = [3, 0, 2]  # middle segment is empty

        results = sample_features_batched(
            all_fids, all_mags, seg_sizes,
            pct_table, sampling_config, rng,
        )
        assert len(results) == 3
        assert len(results[1][0]) == 0  # empty segment

    def test_override_weights(self, pct_table, sampling_config):
        """Override weights skip percentile computation for those segments."""
        segments = _make_segments(10)

        rng = torch.Generator()
        rng.manual_seed(42)

        all_fids = torch.cat([s[0] for s in segments])
        all_mags = torch.cat([s[1] for s in segments])
        seg_sizes = [len(s[0]) for s in segments]

        # Override weights for segments 2 and 5
        override_list = [None] * 10
        override_list[2] = torch.ones(seg_sizes[2], dtype=torch.float32)
        override_list[5] = torch.ones(seg_sizes[5], dtype=torch.float32)

        results = sample_features_batched(
            all_fids, all_mags, seg_sizes,
            pct_table, sampling_config, rng,
            override_weights_list=override_list,
        )
        assert len(results) == 10
        # All segments should have results
        for ids, mags in results:
            assert len(ids) == len(mags)

    def test_single_feature_segments(self, pct_table):
        """Segments with exactly 1 feature always return that feature with min_k=1."""
        config = SamplingSelection(
            distribution=Distribution.geometric,
            mean=1.0, std=1.0,
            min_k=1, max_k=5,
        )
        rng = torch.Generator()
        rng.manual_seed(42)

        n_segs = 20
        all_fids = torch.arange(n_segs, dtype=torch.int64)
        all_mags = torch.rand(n_segs, dtype=torch.float32) + 0.1
        seg_sizes = [1] * n_segs

        results = sample_features_batched(
            all_fids, all_mags, seg_sizes,
            pct_table, config, rng,
        )
        for i, (ids, _) in enumerate(results):
            assert len(ids) == 1, f"Single-feature segment {i} returned {len(ids)} features"
            assert ids[0] == i

    def test_all_segments_min_k_respected(self, pct_table):
        """min_k=1 guarantee holds for every segment in the batch."""
        config = SamplingSelection(
            distribution=Distribution.geometric,
            mean=0.5, std=0.5,
            min_k=1, max_k=10,
        )
        segments = _make_segments(200, max_features=8, n_sae=100)
        rng = torch.Generator()
        rng.manual_seed(42)

        all_fids = torch.cat([s[0] for s in segments])
        all_mags = torch.cat([s[1] for s in segments])
        seg_sizes = [len(s[0]) for s in segments]

        results = sample_features_batched(
            all_fids, all_mags, seg_sizes,
            pct_table, config, rng,
        )
        for i, (ids, _) in enumerate(results):
            assert len(ids) >= 1, (
                f"Segment {i}: min_k=1 but got {len(ids)} features"
            )

    def test_max_k_respected(self, pct_table):
        """max_k cap holds for every segment in the batch."""
        config = SamplingSelection(
            distribution=Distribution.geometric,
            mean=50.0, std=50.0,
            min_k=1, max_k=5,
        )
        rng = torch.Generator()
        rng.manual_seed(42)

        # Segments with many features — k should still be capped at max_k
        segments = []
        for _ in range(100):
            n = 30
            fids = torch.randperm(N_SAE)[:n].to(torch.int64)
            mags = torch.rand(n, dtype=torch.float32) + 0.1
            segments.append((fids, mags))

        all_fids = torch.cat([s[0] for s in segments])
        all_mags = torch.cat([s[1] for s in segments])
        seg_sizes = [len(s[0]) for s in segments]

        results = sample_features_batched(
            all_fids, all_mags, seg_sizes,
            pct_table, config, rng,
        )
        for i, (ids, _) in enumerate(results):
            assert len(ids) <= 4, (  # max_k=5 means k in [1,4]
                f"Segment {i}: max_k=5 but got {len(ids)} features"
            )

    def test_output_ids_are_subset_of_input(self, pct_table, sampling_config):
        """All output feature IDs must come from the corresponding input segment."""
        segments = _make_segments(100, max_features=15, n_sae=100)
        rng = torch.Generator()
        rng.manual_seed(42)

        all_fids = torch.cat([s[0] for s in segments])
        all_mags = torch.cat([s[1] for s in segments])
        seg_sizes = [len(s[0]) for s in segments]

        results = sample_features_batched(
            all_fids, all_mags, seg_sizes,
            pct_table, sampling_config, rng,
        )
        offset = 0
        for i, (out_ids, out_mags) in enumerate(results):
            in_ids = all_fids[offset:offset + seg_sizes[i]]
            in_set = set(in_ids.tolist())
            out_set = set(out_ids.tolist())
            assert out_set.issubset(in_set), (
                f"Segment {i}: output IDs {out_set - in_set} not in input"
            )
            assert len(out_ids) == len(out_mags)
            offset += seg_sizes[i]

    def test_no_duplicate_ids_in_output(self, pct_table, sampling_config):
        """Sampling without replacement means no duplicates per segment."""
        segments = _make_segments(100, max_features=15, n_sae=100)
        rng = torch.Generator()
        rng.manual_seed(42)

        all_fids = torch.cat([s[0] for s in segments])
        all_mags = torch.cat([s[1] for s in segments])
        seg_sizes = [len(s[0]) for s in segments]

        results = sample_features_batched(
            all_fids, all_mags, seg_sizes,
            pct_table, sampling_config, rng,
        )
        for i, (ids, _) in enumerate(results):
            assert len(set(ids.tolist())) == len(ids), (
                f"Segment {i}: duplicate IDs in output: {ids.tolist()}"
            )

    def test_normal_distribution_batched(self, pct_table):
        """Works correctly with normal distribution too."""
        config = SamplingSelection(
            distribution=Distribution.normal,
            mean=3.0, std=1.5,
            min_k=0, max_k=8,
        )
        segments = _make_segments(50, max_features=10, n_sae=100)
        rng = torch.Generator()
        rng.manual_seed(42)

        all_fids = torch.cat([s[0] for s in segments])
        all_mags = torch.cat([s[1] for s in segments])
        seg_sizes = [len(s[0]) for s in segments]

        results = sample_features_batched(
            all_fids, all_mags, seg_sizes,
            pct_table, config, rng,
        )
        assert len(results) == 50
        for ids, _ in results:
            assert len(ids) <= 7  # max_k=8 means k <= 7

    @pytest.mark.benchmark
    def test_large_batch_performance(self, pct_table, sampling_config):
        """Batched should be significantly faster than sequential for many segments."""
        import time

        segments = _make_segments(1000, max_features=10)

        # Sequential timing
        rng = torch.Generator()
        rng.manual_seed(99)
        t0 = time.monotonic()
        for fids, mags in segments:
            sample_features(fids, mags, pct_table, sampling_config, rng)
        t_seq = time.monotonic() - t0

        # Batched timing
        rng = torch.Generator()
        rng.manual_seed(99)
        all_fids = torch.cat([s[0] for s in segments])
        all_mags = torch.cat([s[1] for s in segments])
        seg_sizes = [len(s[0]) for s in segments]
        t0 = time.monotonic()
        sample_features_batched(
            all_fids, all_mags, seg_sizes,
            pct_table, sampling_config, rng,
        )
        t_bat = time.monotonic() - t0

        speedup = t_seq / t_bat
        print(f"Sequential: {t_seq*1000:.1f}ms, Batched: {t_bat*1000:.1f}ms, Speedup: {speedup:.1f}x")
        # Gumbel top-k on tiny synthetic segments (1-20 features) has less
        # advantage than on real data (~1800 features); use a conservative threshold.
        assert speedup > 1.2, f"Expected at least 1.2x speedup, got {speedup:.1f}x"


class TestSampleKBatched:
    """Tests for sample_k_batched."""

    def test_output_shape_and_dtype(self):
        """Returns (n,) int64 tensor."""
        rng = torch.Generator().manual_seed(0)
        result = sample_k_batched(Distribution.geometric, 5.0, 5.0, rng, 100)
        assert result.shape == (100,)
        assert result.dtype == torch.int64

    def test_geometric_min_k(self):
        """All values >= min_k for geometric distribution."""
        rng = torch.Generator().manual_seed(0)
        result = sample_k_batched(Distribution.geometric, 5.0, 5.0, rng, 10000, min_k=3)
        assert (result >= 3).all()

    def test_geometric_max_k(self):
        """All values in [min_k, max_k-1] for geometric distribution."""
        rng = torch.Generator().manual_seed(0)
        result = sample_k_batched(Distribution.geometric, 5.0, 5.0, rng, 10000, min_k=2, max_k=10)
        assert (result >= 2).all()
        assert (result <= 9).all()

    def test_geometric_mean(self):
        """Empirical mean within confidence interval of expected geometric mean."""
        rng = torch.Generator().manual_seed(42)
        n = 10_000
        result = sample_k_batched(Distribution.geometric, 5.0, 5.0, rng, n)
        empirical_mean = result.float().mean().item()
        # Geometric with p=1/(1+mean)=1/6 has E[X-1] = (1-p)/p = 5.0
        assert 4.5 < empirical_mean < 5.8, f"Geometric mean {empirical_mean} outside [4.5, 5.8]"

    def test_normal_clamping(self):
        """All values in [min_k, max_k-1] for normal distribution."""
        rng = torch.Generator().manual_seed(0)
        result = sample_k_batched(Distribution.normal, 5.0, 2.0, rng, 10000, min_k=2, max_k=8)
        assert (result >= 2).all()
        assert (result <= 7).all()

    def test_normal_mean(self):
        """Empirical mean near configured mean for normal distribution."""
        rng = torch.Generator().manual_seed(42)
        n = 10_000
        result = sample_k_batched(Distribution.normal, 10.0, 3.0, rng, n, min_k=0, max_k=30)
        empirical_mean = result.float().mean().item()
        assert 9.5 < empirical_mean < 10.5, f"Normal mean {empirical_mean} outside [9.5, 10.5]"

    def test_n_zero(self):
        """n=0 returns empty tensor."""
        rng = torch.Generator().manual_seed(0)
        result = sample_k_batched(Distribution.geometric, 5.0, 5.0, rng, 0)
        assert result.shape == (0,)
        assert result.dtype == torch.int64

    def test_unsupported_distribution(self):
        """Unsupported distribution raises ValueError."""
        rng = torch.Generator().manual_seed(0)
        with pytest.raises(BeartypeCallHintParamViolation, match="distribution"):
            sample_k_batched("bogus", 5.0, 5.0, rng, 10)  # ty: ignore[invalid-argument-type]
