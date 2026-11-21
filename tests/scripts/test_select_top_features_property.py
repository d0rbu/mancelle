"""Property-based tests for
``scripts.preprocess.extract_common_features.select_top_feature_ids_by_score``.

Contracts:
- Output length == ``min(k, len(scores))``.
- Output is sorted by ``(-score, fid)``: highest score first, ties
  broken by fid ascending.
- Determinism under input dict-order permutation.
- Empty / k <= 0 returns ``[]``.
- Non-finite scores raise ``ValueError``.
"""

from __future__ import annotations

import itertools
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scripts.preprocess.extract_common_features import (
    select_top_feature_ids_by_score,
)


@given(
    pairs=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=10_000),  # fid
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        ),
        unique_by=lambda p: p[0],  # unique fids
        min_size=0,
        max_size=50,
    ),
    k=st.integers(min_value=-5, max_value=60),
)
@settings(derandomize=True, max_examples=200)
def test_output_length_equals_min_k_len(pairs, k):
    scores = dict(pairs)
    out = select_top_feature_ids_by_score(scores, k)
    expected_len = max(0, min(k, len(scores)))
    assert len(out) == expected_len


@given(
    pairs=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=10_000),
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        ),
        unique_by=lambda p: p[0],
        min_size=1,
        max_size=50,
    ),
    k=st.integers(min_value=1, max_value=60),
)
@settings(derandomize=True, max_examples=200)
def test_output_sorted_by_neg_score_then_fid(pairs, k):
    scores = dict(pairs)
    out = select_top_feature_ids_by_score(scores, k)
    if len(out) < 2:
        return
    pairs_out = [(scores[fid], fid) for fid in out]
    # Each consecutive pair should satisfy (-s_i, fid_i) <= (-s_{i+1}, fid_{i+1}).
    for a, b in itertools.pairwise(pairs_out):
        assert (-a[0], a[1]) <= (-b[0], b[1])


@given(
    pairs=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=1000),
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        ),
        unique_by=lambda p: p[0],
        min_size=2,
        max_size=20,
    ),
    k=st.integers(min_value=1, max_value=20),
    seed=st.integers(min_value=0, max_value=2**16 - 1),
)
@settings(derandomize=True, max_examples=200)
def test_invariant_under_dict_order_permutation(pairs, k, seed):
    """The output must not depend on dict insertion order."""
    rng = random.Random(seed)
    pairs_a = list(pairs)
    pairs_b = list(pairs)
    rng.shuffle(pairs_b)
    out_a = select_top_feature_ids_by_score(dict(pairs_a), k)
    out_b = select_top_feature_ids_by_score(dict(pairs_b), k)
    assert out_a == out_b


def test_empty_returns_empty():
    assert select_top_feature_ids_by_score({}, k=10) == []


def test_k_zero_returns_empty():
    assert select_top_feature_ids_by_score({1: 0.5, 2: 0.7}, k=0) == []


def test_negative_k_returns_empty():
    assert select_top_feature_ids_by_score({1: 0.5, 2: 0.7}, k=-3) == []


def test_nan_score_raises():
    with pytest.raises(ValueError, match=r"non-finite"):
        select_top_feature_ids_by_score({1: 0.1, 2: float("nan")}, k=2)


def test_inf_score_raises():
    with pytest.raises(ValueError, match=r"non-finite"):
        select_top_feature_ids_by_score({1: 0.1, 2: float("inf")}, k=2)


def test_tie_break_by_fid_ascending():
    """Equal scores -> lower fid first."""
    out = select_top_feature_ids_by_score({3: 1.0, 1: 1.0, 2: 1.0}, k=3)
    assert out == [1, 2, 3]
