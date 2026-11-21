"""Optimized masked pooling == simple reference oracle (property + examples)."""

import random

import torch as t
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from steering.extract_features import pool_segments_batched_gpu
from steering.feature_paths import PoolingMode
from steering.segmentation import SegmentMode
from tests.reference_extraction_pipeline import ref_pool_segment_ignore_bos


@settings(max_examples=200, deadline=None)
@given(
    seq=st.integers(min_value=2, max_value=60),
    seg_size=st.integers(min_value=1, max_value=15),
    base=st.integers(min_value=0, max_value=10),
    n_bos=st.integers(min_value=0, max_value=4),
    seed=st.integers(min_value=0, max_value=50_000),
    pm=st.sampled_from(list(PoolingMode)),
)
def test_optimized_matches_oracle(seq, seg_size, base, n_bos, seed, pm):
    """Oracle parity for multi-BOS positions, SFT-style base>0 offsets, and all 3 pooling modes.

    Alignment note: pool_segments_batched_gpu uses ``base = segments[0][0]`` and slices
    ``sae_acts[base : base + trimmed_len]`` / ``keep_mask[base : base + trimmed_len]`` with
    absolute indices.  ref_pool_segment_ignore_bos also slices ``sae_acts[start:end]`` /
    ``keep_mask[start:end]`` with absolute indices.  Both sides use absolute coordinates so
    the parity comparison is well-defined even when base > 0.
    """
    assume(seq > base)
    g = t.Generator().manual_seed(seed)
    acts = t.rand(seq, 3, generator=g)  # post-ReLU non-negative
    mask = t.ones(seq, dtype=t.bool)
    available = list(range(base, seq))
    assume(len(available) >= n_bos)
    rng = random.Random(seed)
    for pos in rng.sample(available, n_bos):
        mask[pos] = False
    segs, s = [], base
    while s < seq:
        segs.append((s, min(s + seg_size, seq)))
        s += seg_size
    assume(segs)
    assume(not any(not bool(mask[a:b].any()) for a, b in segs))
    opt = pool_segments_batched_gpu(acts, segs, pm, SegmentMode.fixed_tokens, keep_mask=mask)
    ref = t.stack([ref_pool_segment_ignore_bos(acts, a, b, pm, mask) for a, b in segs])
    assert t.allclose(opt, ref, atol=1e-5), (pm, seq, seg_size, base, n_bos, seed)


def test_flag_off_equals_legacy_property():
    g = t.Generator().manual_seed(1)
    acts = t.rand(20, 4, generator=g)
    segs = [(0, 8), (8, 16), (16, 20)]
    for pm in PoolingMode:
        legacy = pool_segments_batched_gpu(acts, segs, pm, SegmentMode.fixed_tokens)
        masked_all_true = pool_segments_batched_gpu(
            acts, segs, pm, SegmentMode.fixed_tokens,
            keep_mask=t.ones(20, dtype=t.bool),
        )
        assert t.allclose(legacy, masked_all_true), pm
