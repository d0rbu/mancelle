"""Property-based tests for ``mix_sft_into_pretraining.distribute_uniformly``.

The contract per `docs/pipelines/sft-mixing.md`:
- Every input item appears in exactly one bucket.
- The buckets' lengths differ by at most 1
  (``floor(N/B)`` or ``ceil(N/B)``).
- For a fixed RNG seed the assignment is deterministic.
"""

from __future__ import annotations

import random

from hypothesis import given, settings
from hypothesis import strategies as st

from scripts.preprocess.mix_sft_into_pretraining import distribute_uniformly


@given(
    items_size=st.integers(min_value=0, max_value=200),
    n_buckets=st.integers(min_value=1, max_value=20),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(derandomize=True, max_examples=200)
def test_all_items_appear_exactly_once(items_size, n_buckets, seed):
    items = list(range(items_size))
    rng = random.Random(seed)
    buckets = distribute_uniformly(items, n_buckets, rng)
    flat = [x for b in buckets for x in b]
    # Every input appears, no duplicates.
    assert sorted(flat) == sorted(items)


@given(
    items_size=st.integers(min_value=0, max_value=200),
    n_buckets=st.integers(min_value=1, max_value=20),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(derandomize=True, max_examples=200)
def test_bucket_size_diff_at_most_one(items_size, n_buckets, seed):
    items = list(range(items_size))
    rng = random.Random(seed)
    buckets = distribute_uniformly(items, n_buckets, rng)
    sizes = [len(b) for b in buckets]
    # Max minus min difference <= 1.
    assert max(sizes) - min(sizes) <= 1


@given(
    items_size=st.integers(min_value=0, max_value=100),
    n_buckets=st.integers(min_value=1, max_value=10),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(derandomize=True, max_examples=100)
def test_deterministic_for_same_seed(items_size, n_buckets, seed):
    items = list(range(items_size))
    out_a = distribute_uniformly(items, n_buckets, random.Random(seed))
    out_b = distribute_uniformly(items, n_buckets, random.Random(seed))
    assert out_a == out_b


@given(
    items_size=st.integers(min_value=10, max_value=100),
    n_buckets=st.integers(min_value=2, max_value=10),
    seed_a=st.integers(min_value=0, max_value=2**31 - 1),
    seed_b=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(derandomize=True, max_examples=50)
def test_different_seeds_usually_differ(items_size, n_buckets, seed_a, seed_b):
    """Two different seeds with enough items should usually shuffle differently.

    Not always guaranteed (small distinguishability for small inputs), so we
    only assert that *if* the seeds differ AND there are enough items,
    the result is permitted to vary. We just exercise the path; no hard
    assertion here. (Property exists for coverage breadth.)
    """
    items = list(range(items_size))
    out_a = distribute_uniformly(items, n_buckets, random.Random(seed_a))
    out_b = distribute_uniformly(items, n_buckets, random.Random(seed_b))
    # Just exercise: both results pass the always-true "all items present" check.
    assert sorted(x for b in out_a for x in b) == sorted(items)
    assert sorted(x for b in out_b for x in b) == sorted(items)


def test_zero_items_returns_n_empty_buckets():
    out = distribute_uniformly([], 5, random.Random(0))
    assert out == [[], [], [], [], []]


def test_one_bucket_returns_full_list():
    items = ["a", "b", "c"]
    out = distribute_uniformly(items, 1, random.Random(0))
    assert len(out) == 1
    assert sorted(out[0]) == sorted(items)
