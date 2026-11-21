"""Property tests for partition modifiers in ``encoder_input``.

Every modifier is a pure function over two ``dict[int, int]`` partitions
plus some keyword-only arguments. Five INVARIANTS must hold across
arbitrary inputs regardless of modifier-specific behavior:

1. Positive-mode modifiers never produce a key that appears twice in
   the returned ``active`` dict. (Dicts can't, but this guards against
   future refactors to list-of-pairs or similar.)
2. ``modify_neg_ours_remove`` guarantees no top-K fid appears in the
   returned ``active`` dict.
3. ``modify_neg_ours_add_to_negatives`` moves every top-K fid into
   ``inactive`` at the ``inactive_bucket`` index.
4. Every modifier is deterministic: calling it twice with identical
   inputs returns the same output (no hidden global state, no
   clock-dependent randomness).
5. Every modifier is non-mutating: the caller's ``active`` and
   ``inactive`` dicts are not changed by the call.

These are INVARIANTS that hold across infinite inputs, which is the
exact case Hypothesis-based property tests are designed for.
"""

from __future__ import annotations

import copy
import random

from hypothesis import assume, given
from hypothesis import strategies as st

from steering.eval.axbench_base.encoder_input import (
    modify_neg_ours_add_to_negatives,
    modify_neg_ours_remove,
    modify_neg_ours_replace_all_negatives,
    modify_neg_ours_replace_random_negatives,
    modify_ours_extract_and_add,
    modify_ours_replace_all,
    modify_ours_replace_random,
)


def _assume_disjoint(active: dict[int, int], inactive: dict[int, int]) -> None:
    """Precondition: modifiers assert disjoint (active, inactive) inputs.

    Batch R1 added end-of-function invariant assertions requiring
    modifier outputs to be disjoint. Most modifiers pass through the
    caller's ``inactive`` dict unchanged, so disjoint OUTPUT demands
    disjoint INPUT. Property tests must respect this contract.
    """
    assume(set(active.keys()).isdisjoint(inactive.keys()))


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


_partition_strategy = st.dictionaries(
    keys=st.integers(min_value=0, max_value=1000),
    values=st.integers(min_value=0, max_value=15),
    max_size=20,
)

_top_k_strategy = st.lists(
    st.integers(min_value=0, max_value=1000),
    min_size=0,
    max_size=10,
    unique=True,
)

_coefficient_strategy = st.integers(min_value=0, max_value=15)

_inactive_bucket_strategy = st.integers(min_value=0, max_value=15)

_seed_strategy = st.integers(min_value=0, max_value=2**31 - 1)


# ---------------------------------------------------------------------------
# Property 1: positive-mode modifiers produce no duplicate fids in active
# ---------------------------------------------------------------------------


@given(
    active=_partition_strategy,
    inactive=_partition_strategy,
    top_k=_top_k_strategy,
    coefficient=_coefficient_strategy,
    seed=_seed_strategy,
)
def test_positive_modifiers_no_duplicates_in_active(
    active: dict[int, int],
    inactive: dict[int, int],
    top_k: list[int],
    coefficient: int,
    seed: int,
) -> None:
    _assume_disjoint(active, inactive)
    rng = random.Random(seed)
    # ``modify_ours_extract_and_add`` — injects top-K
    new_active, _ = modify_ours_extract_and_add(
        active, inactive, top_k_concept_fids=top_k, coefficient=coefficient,
    )
    assert len(new_active) == len(set(new_active.keys()))
    # ``modify_ours_replace_random`` — drops K, adds K
    new_active, _ = modify_ours_replace_random(
        active,
        inactive,
        top_k_concept_fids=top_k,
        coefficient=coefficient,
        rng=rng,
    )
    assert len(new_active) == len(set(new_active.keys()))
    # ``modify_ours_replace_all`` — replaces active entirely
    new_active, _ = modify_ours_replace_all(
        active, inactive, top_k_concept_fids=top_k, coefficient=coefficient,
    )
    assert len(new_active) == len(set(new_active.keys()))


# ---------------------------------------------------------------------------
# Property 2: negative-mode ``remove`` strips top-K from active
# ---------------------------------------------------------------------------


@given(
    active=_partition_strategy,
    inactive=_partition_strategy,
    top_k=_top_k_strategy,
)
def test_negative_modifier_removes_top_k_from_active(
    active: dict[int, int],
    inactive: dict[int, int],
    top_k: list[int],
) -> None:
    _assume_disjoint(active, inactive)
    new_active, _ = modify_neg_ours_remove(
        active, inactive, top_k_concept_fids=top_k,
    )
    for fid in top_k:
        assert fid not in new_active


# ---------------------------------------------------------------------------
# Property 3: ``add_to_negatives`` moves top-K to inactive at the bucket
# ---------------------------------------------------------------------------


@given(
    active=_partition_strategy,
    inactive=_partition_strategy,
    top_k=_top_k_strategy,
    inactive_bucket=_inactive_bucket_strategy,
)
def test_add_to_negatives_moves_top_k_to_inactive(
    active: dict[int, int],
    inactive: dict[int, int],
    top_k: list[int],
    inactive_bucket: int,
) -> None:
    _assume_disjoint(active, inactive)
    new_active, new_inactive = modify_neg_ours_add_to_negatives(
        active,
        inactive,
        top_k_concept_fids=top_k,
        inactive_bucket=inactive_bucket,
    )
    for fid in top_k:
        assert fid not in new_active
        assert new_inactive[fid] == inactive_bucket


# ---------------------------------------------------------------------------
# Property 4: modifiers are pure (double-application is idempotent w.r.t. output)
# ---------------------------------------------------------------------------


@given(
    active=_partition_strategy,
    inactive=_partition_strategy,
    top_k=_top_k_strategy,
    coefficient=_coefficient_strategy,
    inactive_bucket=_inactive_bucket_strategy,
    seed=_seed_strategy,
)
def test_modifiers_are_deterministic(
    active: dict[int, int],
    inactive: dict[int, int],
    top_k: list[int],
    coefficient: int,
    inactive_bucket: int,
    seed: int,
) -> None:
    _assume_disjoint(active, inactive)
    # ``modify_ours_extract_and_add``
    a1, i1 = modify_ours_extract_and_add(
        active, inactive, top_k_concept_fids=top_k, coefficient=coefficient,
    )
    a2, i2 = modify_ours_extract_and_add(
        active, inactive, top_k_concept_fids=top_k, coefficient=coefficient,
    )
    assert a1 == a2
    assert i1 == i2

    # ``modify_ours_replace_random`` with FRESH rng each call (same seed)
    a1, i1 = modify_ours_replace_random(
        active,
        inactive,
        top_k_concept_fids=top_k,
        coefficient=coefficient,
        rng=random.Random(seed),
    )
    a2, i2 = modify_ours_replace_random(
        active,
        inactive,
        top_k_concept_fids=top_k,
        coefficient=coefficient,
        rng=random.Random(seed),
    )
    assert a1 == a2
    assert i1 == i2

    # ``modify_ours_replace_all``
    a1, i1 = modify_ours_replace_all(
        active, inactive, top_k_concept_fids=top_k, coefficient=coefficient,
    )
    a2, i2 = modify_ours_replace_all(
        active, inactive, top_k_concept_fids=top_k, coefficient=coefficient,
    )
    assert a1 == a2
    assert i1 == i2

    # ``modify_neg_ours_remove``
    a1, i1 = modify_neg_ours_remove(
        active, inactive, top_k_concept_fids=top_k,
    )
    a2, i2 = modify_neg_ours_remove(
        active, inactive, top_k_concept_fids=top_k,
    )
    assert a1 == a2
    assert i1 == i2

    # ``modify_neg_ours_add_to_negatives``
    a1, i1 = modify_neg_ours_add_to_negatives(
        active,
        inactive,
        top_k_concept_fids=top_k,
        inactive_bucket=inactive_bucket,
    )
    a2, i2 = modify_neg_ours_add_to_negatives(
        active,
        inactive,
        top_k_concept_fids=top_k,
        inactive_bucket=inactive_bucket,
    )
    assert a1 == a2
    assert i1 == i2

    # ``modify_neg_ours_replace_random_negatives`` — fresh rng with same seed
    a1, i1 = modify_neg_ours_replace_random_negatives(
        active,
        inactive,
        top_k_concept_fids=top_k,
        inactive_bucket=inactive_bucket,
        rng=random.Random(seed),
    )
    a2, i2 = modify_neg_ours_replace_random_negatives(
        active,
        inactive,
        top_k_concept_fids=top_k,
        inactive_bucket=inactive_bucket,
        rng=random.Random(seed),
    )
    assert a1 == a2
    assert i1 == i2

    # ``modify_neg_ours_replace_all_negatives``
    a1, i1 = modify_neg_ours_replace_all_negatives(
        active,
        inactive,
        top_k_concept_fids=top_k,
        inactive_bucket=inactive_bucket,
    )
    a2, i2 = modify_neg_ours_replace_all_negatives(
        active,
        inactive,
        top_k_concept_fids=top_k,
        inactive_bucket=inactive_bucket,
    )
    assert a1 == a2
    assert i1 == i2


# ---------------------------------------------------------------------------
# Property 5: modifiers do not mutate caller's partitions
# ---------------------------------------------------------------------------


@given(
    active=_partition_strategy,
    inactive=_partition_strategy,
    top_k=_top_k_strategy,
    coefficient=_coefficient_strategy,
    inactive_bucket=_inactive_bucket_strategy,
    seed=_seed_strategy,
)
def test_modifier_input_not_mutated(
    active: dict[int, int],
    inactive: dict[int, int],
    top_k: list[int],
    coefficient: int,
    inactive_bucket: int,
    seed: int,
) -> None:
    _assume_disjoint(active, inactive)
    active_before = copy.deepcopy(active)
    inactive_before = copy.deepcopy(inactive)

    modify_ours_extract_and_add(
        active, inactive, top_k_concept_fids=top_k, coefficient=coefficient,
    )
    assert active == active_before
    assert inactive == inactive_before

    modify_ours_replace_random(
        active,
        inactive,
        top_k_concept_fids=top_k,
        coefficient=coefficient,
        rng=random.Random(seed),
    )
    assert active == active_before
    assert inactive == inactive_before

    modify_ours_replace_all(
        active, inactive, top_k_concept_fids=top_k, coefficient=coefficient,
    )
    assert active == active_before
    assert inactive == inactive_before

    modify_neg_ours_remove(active, inactive, top_k_concept_fids=top_k)
    assert active == active_before
    assert inactive == inactive_before

    modify_neg_ours_add_to_negatives(
        active,
        inactive,
        top_k_concept_fids=top_k,
        inactive_bucket=inactive_bucket,
    )
    assert active == active_before
    assert inactive == inactive_before

    modify_neg_ours_replace_random_negatives(
        active,
        inactive,
        top_k_concept_fids=top_k,
        inactive_bucket=inactive_bucket,
        rng=random.Random(seed),
    )
    assert active == active_before
    assert inactive == inactive_before

    modify_neg_ours_replace_all_negatives(
        active,
        inactive,
        top_k_concept_fids=top_k,
        inactive_bucket=inactive_bucket,
    )
    assert active == active_before
    assert inactive == inactive_before
