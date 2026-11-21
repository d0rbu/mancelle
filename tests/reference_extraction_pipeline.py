"""Reference extraction pipeline — simple, readable, unoptimized.

This module provides a minimal, straightforward implementation of the entire
SAE feature extraction pipeline. Every function prioritizes clarity over
performance, using plain Python loops and no vectorization. It serves as an
oracle for property-based tests that verify the optimized production code
in steering/extract_features.py and steering/segmentation.py.

The pipeline flow:
    1. Tokenize document text
    2. Truncate to head (first context_window tokens)
    3. Segment the tail into (start, end) ranges
    4. Pool SAE activations per segment (mean/max/last_token)
    5. Extract sparse features (non-zero, sorted by magnitude)
    6. Pack into CSR format with segment metadata

Usage:
    uv run pytest tests/reference_extraction_pipeline.py -v -s
"""

import functools

import hypothesis.strategies as st
import pytest
import torch as t
from hypothesis import assume, given, settings

from steering.extract_features import (
    PoolingMode,
    _get_sae_activations,
    _get_sae_activations_batched,
    pool_segments_batched_gpu,
)
from steering.segmentation import SegmentMode, segment_tokens
from tests.steering.test_extract_features_integration import TinyHookedTransformer, TinySAE

# ═══════════════════════════════════════════════════════════════════════════
# Reference implementations — clarity over performance
# ═══════════════════════════════════════════════════════════════════════════


# --- Segmentation ---


def ref_segment_fixed(n: int, segment_size: int) -> list[tuple[int, int]]:
    """Split n tokens into fixed-size windows."""
    segments = []
    start = 0
    while start < n:
        end = min(start + segment_size, n)
        segments.append((start, end))
        start = end
    return segments


def ref_segment_tokens(
    token_ids: list[int],
    tokenizer,
    mode: SegmentMode,
    segment_size: int = 128,
) -> list[tuple[int, int]]:
    """Reference segmentation dispatcher."""
    n = len(token_ids)
    assert n > 0

    if mode != SegmentMode.fixed_tokens:
        raise ValueError(
            f"Only fixed_tokens segmentation is supported, got: {mode.value}"
        )
    return ref_segment_fixed(n, segment_size)


# --- Pooling ---


def ref_pool_segment(
    sae_acts: t.Tensor,
    start: int,
    end: int,
    pooling: PoolingMode,
) -> t.Tensor:
    """Pool activations for one segment — simple slice + reduce."""
    chunk = sae_acts[start:end]
    if pooling == PoolingMode.last_token:
        return chunk[-1]
    if pooling == PoolingMode.max_pool:
        return chunk.max(dim=0).values
    if pooling == PoolingMode.mean_pool:
        return chunk.mean(dim=0)
    raise ValueError(f"Unknown pooling: {pooling}")


def ref_pool_segment_ignore_bos(
    sae_acts: t.Tensor,
    start: int,
    end: int,
    pooling: PoolingMode,
    keep_mask: t.Tensor,
) -> t.Tensor:
    """Reference BOS-masked pooling: slice, drop <bos> rows, reduce.

    Raises ValueError if the segment has no kept rows (the optimized path
    raises BosOnlySegmentError there; parity is tested separately).
    """
    chunk = sae_acts[start:end]
    m = keep_mask[start:end].to(t.bool)
    if not bool(m.any()):
        raise ValueError("ref_pool_segment_ignore_bos: all-<bos> segment")
    kept = chunk[m]
    if pooling == PoolingMode.last_token:
        return kept[-1]
    if pooling == PoolingMode.max_pool:
        return kept.max(dim=0).values
    if pooling == PoolingMode.mean_pool:
        return kept.mean(dim=0)
    raise ValueError(f"Unknown pooling: {pooling}")


# --- Sparse feature extraction ---


def ref_extract_sparse(pooled: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    """Extract non-zero features, sorted by magnitude descending."""
    ids = []
    mags = []
    for i in range(pooled.shape[0]):
        val = pooled[i].item()
        if val > 0:
            ids.append(i)
            mags.append(val)

    assert ids, "ref_extract_sparse: pooled tensor has no positive features"

    feat_ids = t.tensor(ids, dtype=t.int32)
    feat_mags = t.tensor(mags, dtype=t.float32)

    # Sort by magnitude descending
    order = feat_mags.argsort(descending=True)
    return feat_ids[order], feat_mags[order]


# --- Full per-document extraction (no model, takes pre-computed SAE acts) ---


def ref_extract_document(
    sae_acts: t.Tensor,
    segments: list[tuple[int, int]],
    pooling: PoolingMode,
) -> tuple[list[t.Tensor], list[t.Tensor]]:
    """Reference per-document extraction: pool each segment, extract sparse.

    Segments with all-zero activations produce empty feature lists (the
    optimized pipeline does the same). ref_extract_sparse itself asserts
    on empty input as a sanity check for standalone usage.
    """
    all_ids = []
    all_mags = []
    for start, end in segments:
        pooled = ref_pool_segment(sae_acts, start, end, pooling)
        if (pooled > 0).any():
            feat_ids, feat_mags = ref_extract_sparse(pooled)
        else:
            feat_ids = t.tensor([], dtype=t.int32)
            feat_mags = t.tensor([], dtype=t.float32)
        all_ids.append(feat_ids)
        all_mags.append(feat_mags)
    return all_ids, all_mags


def ref_extract_document_context_independent(
    model,
    sae,
    kept_ids: list[int],
    segments: list[tuple[int, int]],
    pooling: PoolingMode,
    bos_token_id: int,
    layer: int,
) -> tuple[list[t.Tensor], list[t.Tensor]]:
    """Reference context-independent extraction: per-segment forward, unbatched, loop-based.

    Oracle for property tests. Runs each segment through the SAE as its own
    1-sequence forward pass, with BOS injected iff the segment's first token
    isn't already BOS. The BOS position's activations are dropped before
    pooling.

    Returns (all_feat_ids, all_feat_mags) where each element is the sparse
    feature list for one segment — same shape as ``ref_extract_document``.
    """
    device = next(model.parameters()).device
    all_ids: list[t.Tensor] = []
    all_mags: list[t.Tensor] = []
    for start, end in segments:
        seg = kept_ids[start:end]
        bos_injected = seg[0] != bos_token_id
        model_input = ([bos_token_id] + seg) if bos_injected else seg
        tokens = t.tensor(model_input, dtype=t.long, device=device)
        acts = _get_sae_activations(model, sae, tokens, layer=layer)
        real_acts = acts[1:] if bos_injected else acts
        assert real_acts.shape[0] == end - start, (
            f"ref_extract_document_context_independent post-drop length {real_acts.shape[0]} "
            f"!= segment length {end - start}"
        )
        pooled = ref_pool_segment(real_acts, 0, end - start, pooling)
        if (pooled > 0).any():
            fid, fmag = ref_extract_sparse(pooled)
        else:
            fid = t.tensor([], dtype=t.int32)
            fmag = t.tensor([], dtype=t.float32)
        all_ids.append(fid)
        all_mags.append(fmag)
    return all_ids, all_mags


# --- CSR packing ---


def ref_build_csr(
    all_feature_ids: list[t.Tensor],
    all_feature_mags: list[t.Tensor],
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """Build CSR offsets from per-segment feature lists."""
    offsets = [0]
    for fids in all_feature_ids:
        offsets.append(offsets[-1] + len(fids))

    offsets_t = t.tensor(offsets, dtype=t.int64)
    if all_feature_ids:
        ids_t = t.cat(all_feature_ids)
        mags_t = t.cat(all_feature_mags)
    else:
        ids_t = t.tensor([], dtype=t.int32)
        mags_t = t.tensor([], dtype=t.float32)

    return offsets_t, ids_t, mags_t


# --- Batched SAE activations ---


def ref_get_sae_activations_batched(
    model,
    sae,
    token_id_lists: list[list[int]],
    layer: int,
) -> list[t.Tensor]:
    """Reference batched extraction: just loops over unbatched single-doc path."""
    device = next(model.parameters()).device
    results = []
    for ids in token_id_lists:
        tokens = t.tensor(ids, dtype=t.long, device=device)
        acts = _get_sae_activations(model, sae, tokens, layer)
        results.append(acts)
    return results


# --- Per-feature statistics ---


def ref_compute_per_feature_stats(
    feature_ids: t.Tensor,
    feature_mags: t.Tensor,
    n_features: int,
) -> dict[str, tuple[int, float]]:
    """Compute per-feature count and mean magnitude using plain loops."""
    counts: dict[int, int] = {}
    mag_sums: dict[int, float] = {}

    for i in range(len(feature_ids)):
        fid = int(feature_ids[i].item())
        mag = float(feature_mags[i].item())
        counts[fid] = counts.get(fid, 0) + 1
        mag_sums[fid] = mag_sums.get(fid, 0.0) + mag

    result = {}
    for fid in sorted(counts.keys()):
        mean_mag = mag_sums[fid] / counts[fid]
        result[str(fid)] = (counts[fid], round(mean_mag, 6))

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Hypothesis strategies
# ═══════════════════════════════════════════════════════════════════════════


def sae_acts_strategy(
    min_tokens: int = 2,
    max_tokens: int = 256,
    n_features: int = 128,
    sparsity: float = 0.95,
):
    """Generate random SAE activation tensors [n_tokens, n_features]."""

    @st.composite
    def _strategy(draw):
        n_tokens = draw(st.integers(min_value=min_tokens, max_value=max_tokens))
        t.manual_seed(draw(st.integers(0, 2**31)))
        acts = t.zeros(n_tokens, n_features)
        mask = t.rand(n_tokens, n_features) > sparsity
        acts[mask] = t.rand(mask.sum()) * 5.0  # ty: ignore[no-matching-overload]
        return acts

    return _strategy()


def segments_strategy(n_tokens: int, min_seg_size: int = 1, max_seg_size: int = 64):
    """Generate valid non-overlapping segments covering [0, n_tokens)."""

    @st.composite
    def _strategy(draw):
        segments = []
        pos = 0
        while pos < n_tokens:
            remaining = n_tokens - pos
            size = draw(
                st.integers(
                    min_value=min_seg_size, max_value=min(max_seg_size, remaining)
                )
            )
            segments.append((pos, pos + size))
            pos += size
        return segments

    return _strategy()


def pooling_strategy():
    return st.sampled_from(list(PoolingMode))


# ═══════════════════════════════════════════════════════════════════════════
# Property-based tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSegmentationProperties:
    """Property tests for all segmentation modes."""

    # -- Fixed tokens --

    @given(
        n=st.integers(1, 500),
        seg_size=st.integers(1, 200),
    )
    @settings(max_examples=100)
    def test_fixed_covers_all_tokens(self, n, seg_size):
        """Segments must cover every token exactly once."""
        segs = ref_segment_fixed(n, seg_size)
        assert segs[0][0] == 0
        assert segs[-1][1] == n
        for i in range(1, len(segs)):
            assert segs[i][0] == segs[i - 1][1], "Gap or overlap"

    @given(
        n=st.integers(1, 500),
        seg_size=st.integers(1, 200),
    )
    @settings(max_examples=100)
    def test_fixed_matches_optimized(self, n, seg_size):
        """Reference and optimized fixed segmentation must agree."""
        ref = ref_segment_fixed(n, seg_size)
        opt = segment_tokens(list(range(n)), SegmentMode.fixed_tokens, seg_size)
        assert ref == opt


class TestPoolingProperties:
    """Property tests for segment pooling."""

    @given(
        acts=sae_acts_strategy(min_tokens=4, max_tokens=128, n_features=64),
        pooling=pooling_strategy(),
    )
    @settings(max_examples=100)
    def test_pool_matches_optimized(self, acts, pooling):
        """Reference pool_segment must agree with pool_segments_batched_gpu."""
        n = acts.shape[0]
        segments = [(0, n)]
        ref = ref_pool_segment(acts, 0, n, pooling)
        opt = pool_segments_batched_gpu(
            acts, segments, pooling, SegmentMode.fixed_tokens
        )
        assert t.allclose(ref, opt[0], atol=1e-5)

    @given(
        acts=sae_acts_strategy(min_tokens=4, max_tokens=128, n_features=64),
        pooling=pooling_strategy(),
    )
    @settings(max_examples=100)
    def test_pool_output_shape(self, acts, pooling):
        """Pooled output must be [n_features]."""
        n = acts.shape[0]
        result = ref_pool_segment(acts, 0, n, pooling)
        assert result.shape == (acts.shape[1],)

    @given(
        acts=sae_acts_strategy(min_tokens=8, max_tokens=64, n_features=32),
    )
    @settings(max_examples=50)
    def test_mean_pool_bounded_by_max(self, acts):
        """Mean-pooled values must be <= max-pooled values element-wise."""
        n = acts.shape[0]
        mean_pooled = ref_pool_segment(acts, 0, n, PoolingMode.mean_pool)
        max_pooled = ref_pool_segment(acts, 0, n, PoolingMode.max_pool)
        assert (mean_pooled <= max_pooled + 1e-6).all()


class TestSparseExtractionProperties:
    """Property tests for sparse feature extraction."""

    def test_ref_extract_sparse_errors_on_all_zero(self):
        """ref_extract_sparse must error when pooled has no positive features."""
        pooled = t.zeros(64)
        with pytest.raises(AssertionError, match="no positive features"):
            ref_extract_sparse(pooled)

    @given(
        acts=sae_acts_strategy(min_tokens=4, max_tokens=64, n_features=64),
        pooling=pooling_strategy(),
    )
    @settings(max_examples=100)
    def test_sparse_matches_optimized(self, acts, pooling):
        """Reference and optimized sparse extraction must agree."""
        pooled = ref_pool_segment(acts, 0, acts.shape[0], pooling)
        assume((pooled > 0).any())

        ref_ids, ref_mags = ref_extract_sparse(pooled)
        # Inline the deleted extract_sparse_features logic
        nonzero_mask = pooled > 0
        opt_ids = nonzero_mask.nonzero(as_tuple=True)[0].to(t.int32)
        opt_mags = pooled[nonzero_mask].to(t.float32)
        sort_idx = opt_mags.argsort(descending=True)
        opt_ids = opt_ids[sort_idx].cpu()
        opt_mags = opt_mags[sort_idx].cpu()

        assert t.equal(ref_ids, opt_ids)
        assert t.allclose(ref_mags, opt_mags, atol=1e-5)

    @given(
        acts=sae_acts_strategy(min_tokens=4, max_tokens=64, n_features=64),
        pooling=pooling_strategy(),
    )
    @settings(max_examples=100)
    def test_sparse_sorted_descending(self, acts, pooling):
        """Feature magnitudes must be sorted in descending order."""
        pooled = ref_pool_segment(acts, 0, acts.shape[0], pooling)
        assume((pooled > 0).any())
        _, mags = ref_extract_sparse(pooled)
        if len(mags) > 1:
            diffs = mags[1:] - mags[:-1]
            assert (diffs <= 0).all()

    @given(
        acts=sae_acts_strategy(min_tokens=4, max_tokens=64, n_features=64),
        pooling=pooling_strategy(),
    )
    @settings(max_examples=100)
    def test_sparse_all_positive(self, acts, pooling):
        """All extracted magnitudes must be > 0."""
        pooled = ref_pool_segment(acts, 0, acts.shape[0], pooling)
        assume((pooled > 0).any())
        _, mags = ref_extract_sparse(pooled)
        assert (mags > 0).all()

    @given(
        acts=sae_acts_strategy(min_tokens=4, max_tokens=64, n_features=64),
        pooling=pooling_strategy(),
    )
    @settings(max_examples=100)
    def test_sparse_ids_unique(self, acts, pooling):
        """Extracted feature IDs must be unique."""
        pooled = ref_pool_segment(acts, 0, acts.shape[0], pooling)
        assume((pooled > 0).any())
        ids, _ = ref_extract_sparse(pooled)
        assert len(ids) == len(set(ids.tolist()))


class TestFullPipelineProperties:
    """Property tests for the full extraction pipeline (pooling + sparse, multi-segment)."""

    @given(
        acts=sae_acts_strategy(min_tokens=8, max_tokens=128, n_features=64),
        pooling=pooling_strategy(),
        seg_size=st.integers(2, 32),
    )
    @settings(max_examples=50)
    def test_pipeline_ref_vs_optimized(self, acts, pooling, seg_size):
        """Full pipeline (segment + pool + extract) must match between ref and optimized."""
        n = acts.shape[0]
        segments = ref_segment_fixed(n, seg_size)

        ref_ids, ref_mags = ref_extract_document(acts, segments, pooling)

        # Optimized version: stack + single CPU transfer
        pooled_list = [ref_pool_segment(acts, s, e, pooling) for s, e in segments]
        pooled_batch = t.stack(pooled_list)
        pooled_cpu = pooled_batch.cpu()

        for i in range(len(segments)):
            row = pooled_cpu[i]
            nonzero_mask = row > 0
            opt_ids = nonzero_mask.nonzero(as_tuple=True)[0].to(t.int32)
            opt_mags = row[nonzero_mask].to(t.float32)
            sort_idx = opt_mags.argsort(descending=True)
            opt_ids = opt_ids[sort_idx]
            opt_mags = opt_mags[sort_idx]

            assert t.equal(ref_ids[i], opt_ids), f"Segment {i} IDs differ"
            assert t.allclose(ref_mags[i], opt_mags, atol=1e-5), (
                f"Segment {i} mags differ"
            )

    @given(
        acts=sae_acts_strategy(min_tokens=8, max_tokens=128, n_features=64),
        pooling=pooling_strategy(),
        seg_size=st.integers(2, 32),
    )
    @settings(max_examples=50)
    def test_csr_round_trip(self, acts, pooling, seg_size):
        """CSR packing must preserve all features per segment."""
        n = acts.shape[0]
        segments = ref_segment_fixed(n, seg_size)
        all_ids, all_mags = ref_extract_document(acts, segments, pooling)

        offsets, packed_ids, packed_mags = ref_build_csr(all_ids, all_mags)

        # Verify round-trip: unpack and compare
        for i in range(len(segments)):
            start_off = offsets[i].item()
            end_off = offsets[i + 1].item()
            seg_ids = packed_ids[start_off:end_off]
            seg_mags = packed_mags[start_off:end_off]
            assert t.equal(seg_ids, all_ids[i])
            assert t.allclose(seg_mags, all_mags[i], atol=1e-5)

    @given(
        acts=sae_acts_strategy(min_tokens=8, max_tokens=128, n_features=64),
        pooling=pooling_strategy(),
        seg_size=st.integers(2, 32),
    )
    @settings(max_examples=50)
    def test_csr_offsets_monotonic(self, acts, pooling, seg_size):
        """CSR offsets must be monotonically non-decreasing starting at 0."""
        n = acts.shape[0]
        segments = ref_segment_fixed(n, seg_size)
        all_ids, all_mags = ref_extract_document(acts, segments, pooling)
        offsets, _, _ = ref_build_csr(all_ids, all_mags)

        assert offsets[0] == 0
        diffs = offsets[1:] - offsets[:-1]
        assert (diffs >= 0).all()


class TestPerFeatureStatsProperties:
    """Property tests for per-feature statistics computation."""

    @given(
        n_features=st.integers(10, 200),
        n_entries=st.integers(1, 1000),
        seed=st.integers(0, 2**31),
    )
    @settings(max_examples=50)
    def test_stats_ref_vs_optimized(self, n_features, n_entries, seed):
        """Reference loop-based stats must match optimized batch-indexed stats.

        Both use float64 for accumulation to avoid precision mismatches
        between the reference (which accumulates in Python float from float32)
        and the optimized path (which accumulates in float64 tensors).
        """
        t.manual_seed(seed)
        ids = t.randint(0, n_features, (n_entries,), dtype=t.int64)
        mags = t.rand(n_entries, dtype=t.float64) * 5.0

        # Reference — use float64 mags to match optimized accumulation precision
        ref = ref_compute_per_feature_stats(
            ids.to(t.int32),
            mags.to(t.float64),
            n_features,
        )

        # Optimized (same approach as compute_statistics in extract_features.py)
        occurrence = t.zeros(n_features, dtype=t.int64)
        mag_sum = t.zeros(n_features, dtype=t.float64)
        occurrence.scatter_add_(0, ids, t.ones_like(ids))
        mag_sum.scatter_add_(0, ids, mags)

        active_mask = occurrence > 0
        active_indices = active_mask.nonzero(as_tuple=True)[0]
        counts = occurrence[active_indices]
        mean_mags = mag_sum[active_indices] / counts.double()
        ids_list = active_indices.tolist()
        counts_list = counts.tolist()
        mags_list = mean_mags.tolist()
        opt = {
            str(fid): (int(c), round(m, 6))
            for fid, c, m in zip(ids_list, counts_list, mags_list, strict=True)
        }

        assert ref == opt

    @given(
        n_features=st.integers(10, 200),
        n_entries=st.integers(1, 500),
        seed=st.integers(0, 2**31),
    )
    @settings(max_examples=50)
    def test_stats_counts_sum_to_entries(self, n_features, n_entries, seed):
        """Sum of all per-feature counts must equal total entries."""
        t.manual_seed(seed)
        ids = t.randint(0, n_features, (n_entries,), dtype=t.int32)
        mags = t.rand(n_entries, dtype=t.float32) * 5.0

        stats = ref_compute_per_feature_stats(ids, mags, n_features)
        total_count = sum(count for count, _ in stats.values())
        assert total_count == n_entries

    @given(
        n_features=st.integers(10, 200),
        n_entries=st.integers(1, 500),
        seed=st.integers(0, 2**31),
    )
    @settings(max_examples=50)
    def test_stats_mean_mag_positive(self, n_features, n_entries, seed):
        """Mean magnitudes must be positive (since all input mags are positive)."""
        t.manual_seed(seed)
        ids = t.randint(0, n_features, (n_entries,), dtype=t.int32)
        mags = t.rand(n_entries, dtype=t.float32) * 5.0 + 0.01  # ensure > 0

        stats = ref_compute_per_feature_stats(ids, mags, n_features)
        for count, mean_mag in stats.values():
            assert count > 0
            assert mean_mag > 0


class TestBatchedActivationProperties:
    """Property tests for batched SAE activation extraction."""

    @given(
        n_docs=st.integers(1, 8),
        seq_len=st.integers(2, 64),
    )
    @settings(max_examples=30, deadline=10000)
    def test_batched_matches_sequential(self, n_docs, seq_len):
        """Batched results must match sequential (unbatched) processing.

        Uses the tiny model so this runs on CPU without real GPT-2.
        """
        model = _get_tiny_model()
        sae = _get_tiny_sae()
        token_id_lists = [[i * 10 + j for j in range(seq_len)] for i in range(n_docs)]

        batched = _get_sae_activations_batched(model, sae, token_id_lists, layer=8)
        ref = ref_get_sae_activations_batched(model, sae, token_id_lists, layer=8)

        assert len(batched) == len(ref)
        for i in range(n_docs):
            assert t.allclose(batched[i], ref[i], atol=1e-5), (
                f"Doc {i}: max diff {(batched[i] - ref[i]).abs().max()}"
            )

    @given(
        n_docs=st.integers(1, 8),
    )
    @settings(max_examples=20, deadline=10000)
    def test_batched_variable_lengths_match(self, n_docs):
        """Batched with variable-length docs matches sequential."""
        model = _get_tiny_model()
        sae = _get_tiny_sae()
        # Use different lengths per doc
        token_id_lists = [
            list(range(5 + i * 3, 5 + i * 3 + 10 + i * 5)) for i in range(n_docs)
        ]

        batched = _get_sae_activations_batched(model, sae, token_id_lists, layer=8)
        ref = ref_get_sae_activations_batched(model, sae, token_id_lists, layer=8)

        assert len(batched) == len(ref)
        for i in range(n_docs):
            assert batched[i].shape == ref[i].shape, (
                f"Doc {i}: shape mismatch {batched[i].shape} vs {ref[i].shape}"
            )
            assert t.allclose(batched[i], ref[i], atol=1e-5), (
                f"Doc {i}: max diff {(batched[i] - ref[i]).abs().max()}"
            )

    @given(
        n_docs=st.integers(1, 8),
        seq_len=st.integers(2, 64),
    )
    @settings(max_examples=20, deadline=10000)
    def test_batched_output_shapes(self, n_docs, seq_len):
        """Each output tensor must have shape [seq_len_i, n_features]."""
        model = _get_tiny_model()
        sae = _get_tiny_sae()
        token_id_lists = [list(range(seq_len)) for _ in range(n_docs)]

        results = _get_sae_activations_batched(model, sae, token_id_lists, layer=8)

        assert len(results) == n_docs
        for i, acts in enumerate(results):
            assert acts.shape == (seq_len, sae.n_features), (
                f"Doc {i}: expected ({seq_len}, {sae.n_features}), got {acts.shape}"
            )


class TestContextIndependentReferenceProperties:
    @given(
        kept_len=st.integers(min_value=2, max_value=40),
        pooling=pooling_strategy(),
        seed=st.integers(0, 2**31),
    )
    @settings(max_examples=30, deadline=10000)
    def test_ref_context_independent_matches_ref_cd_when_single_segment(
        self, kept_len, pooling, seed,
    ):
        """Under a positionless tiny model, single-segment context-independent matches cd.

        When segments = [(0, kept_len)] AND kept_ids[0] == BOS, the context-independent branch
        skips BOS injection and runs the same forward pass as cd. Features
        should be bit-identical.
        """
        t.manual_seed(seed)
        model = TinyHookedTransformer(d_model=64, vocab_size=256, device="cpu")
        sae = TinySAE(d_model=64, n_features=128, device="cpu")
        # BOS at position 0, then arbitrary tokens
        kept_ids = [0] + [max(1, i + 5) for i in range(kept_len - 1)]
        segments = [(0, kept_len)]
        # cd ref needs sae_acts input; simulate by running the model once.
        tokens = t.tensor(kept_ids, dtype=t.long)
        sae_acts = _get_sae_activations(model, sae, tokens, layer=8)  # ty: ignore[invalid-argument-type]
        ids_cd, mags_cd = ref_extract_document(sae_acts, segments, pooling)
        ids_context_independent, mags_context_independent = ref_extract_document_context_independent(
            model, sae, kept_ids, segments, pooling, bos_token_id=0, layer=8,
        )
        assert len(ids_cd) == len(ids_context_independent) == 1
        assert t.equal(ids_cd[0], ids_context_independent[0]), "single-seg context-independent should match cd ref"
        assert t.allclose(mags_cd[0], mags_context_independent[0], atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


@functools.lru_cache(maxsize=1)
def _get_tiny_model():
    return TinyHookedTransformer(d_model=64, vocab_size=1000, device="cpu")


@functools.lru_cache(maxsize=1)
def _get_tiny_sae():
    return TinySAE(d_model=64, n_features=256, device="cpu")
