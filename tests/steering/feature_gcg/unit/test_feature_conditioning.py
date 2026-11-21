"""Tests for FeatureConditioning immutable bag value-object (Task 2).

Coverage:
- canonical (sorted) order after from_dict / with_added / construction
- duplicate feature_id rejection at construction and mutation
- with_replaced/with_removed semantics
- digest: string, permutation-invariant, content-sensitive
- feature_ids / __len__
- range validation (fid >= 0, bucket >= 0)
- Hypothesis property tests: sorted+dup-free, digest permutation-invariance
"""

from __future__ import annotations

import dataclasses

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_PROFILE = settings(max_examples=200, derandomize=True, deadline=None)


def _valid_bag_strategy():
    return st.dictionaries(
        keys=st.integers(min_value=0, max_value=1000),
        values=st.integers(min_value=0, max_value=63),
        min_size=0,
        max_size=20,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Canonical order
# ─────────────────────────────────────────────────────────────────────────────


def test_from_dict_canonical_order():
    fc = FeatureConditioning.from_dict({3: 1, 1: 2, 5: 0})
    assert fc.entries == ((1, 2), (3, 1), (5, 0))


def test_with_added_canonical_order():
    fc = FeatureConditioning.from_dict({3: 1})
    fc2 = fc.with_added(1, 2)
    assert fc2.entries == ((1, 2), (3, 1))


def test_direct_construction_sorts_entries():
    fc = FeatureConditioning(entries=((5, 0), (1, 2), (3, 1)))
    assert fc.entries == ((1, 2), (3, 1), (5, 0))


def test_from_dict_empty():
    fc = FeatureConditioning.from_dict({})
    assert fc.entries == ()


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate feature_id rejection
# ─────────────────────────────────────────────────────────────────────────────


def test_construction_duplicate_fid_raises():
    with pytest.raises(BeartypeCallHintParamViolation, match="entries"):
        FeatureConditioning(entries=((1, 2), (1, 3)))


def test_with_added_duplicate_fid_raises():
    fc = FeatureConditioning.from_dict({1: 2})
    with pytest.raises(FeatureGcgError):
        fc.with_added(1, 5)


def test_with_replaced_dup_fid_other_slot_raises():
    """Replacing slot 0 (fid=1) with fid=3 clashes with slot 1 (fid=3)."""
    fc = FeatureConditioning.from_dict({1: 2, 3: 4})
    with pytest.raises(FeatureGcgError):
        fc.with_replaced(0, 3, 7)


def test_with_replaced_same_slot_same_fid_allowed():
    """Replacing a slot with the same fid (different bucket) is valid."""
    fc = FeatureConditioning.from_dict({1: 2, 3: 4})
    fc2 = fc.with_replaced(0, 1, 99)
    assert fc2.entries == ((1, 99), (3, 4))


# ─────────────────────────────────────────────────────────────────────────────
# with_removed
# ─────────────────────────────────────────────────────────────────────────────


def test_with_removed_drops_slot():
    fc = FeatureConditioning.from_dict({1: 2, 3: 4, 5: 6})
    fc2 = fc.with_removed(1)  # slot 1 = (3, 4) in sorted order
    assert fc2.entries == ((1, 2), (5, 6))


def test_with_removed_first_slot():
    fc = FeatureConditioning.from_dict({1: 2, 3: 4})
    fc2 = fc.with_removed(0)
    assert fc2.entries == ((3, 4),)


def test_with_removed_last_slot():
    fc = FeatureConditioning.from_dict({1: 2, 3: 4})
    fc2 = fc.with_removed(1)
    assert fc2.entries == ((1, 2),)


def test_with_removed_to_empty():
    fc = FeatureConditioning.from_dict({1: 2})
    fc2 = fc.with_removed(0)
    assert fc2.entries == ()


# ─────────────────────────────────────────────────────────────────────────────
# with_replaced
# ─────────────────────────────────────────────────────────────────────────────


def test_with_replaced_changes_bucket():
    fc = FeatureConditioning.from_dict({1: 2, 3: 4})
    fc2 = fc.with_replaced(0, 1, 99)
    assert fc2.entries == ((1, 99), (3, 4))


def test_with_replaced_changes_fid_and_resorts():
    fc = FeatureConditioning.from_dict({1: 2, 5: 6})
    # Replace slot 1 (fid=5) with fid=3 → entries should re-sort
    fc2 = fc.with_replaced(1, 3, 9)
    assert fc2.entries == ((1, 2), (3, 9))


# ─────────────────────────────────────────────────────────────────────────────
# digest
# ─────────────────────────────────────────────────────────────────────────────


def test_digest_returns_string():
    fc = FeatureConditioning.from_dict({1: 2})
    assert isinstance(fc.digest(), str)
    assert len(fc.digest()) == 64  # SHA256 hex is 64 chars


def test_digest_permutation_invariant():
    fc1 = FeatureConditioning.from_dict({1: 2, 3: 4})
    fc2 = FeatureConditioning.from_dict({3: 4, 1: 2})
    assert fc1.digest() == fc2.digest()


def test_digest_different_content_different_digest():
    fc1 = FeatureConditioning.from_dict({1: 2})
    fc2 = FeatureConditioning.from_dict({1: 3})
    assert fc1.digest() != fc2.digest()


def test_digest_different_fid_different_digest():
    fc1 = FeatureConditioning.from_dict({1: 2})
    fc2 = FeatureConditioning.from_dict({2: 2})
    assert fc1.digest() != fc2.digest()


def test_digest_empty_is_deterministic():
    fc = FeatureConditioning(entries=())
    d1 = fc.digest()
    d2 = fc.digest()
    assert d1 == d2


def test_digest_extra_entry_differs():
    fc1 = FeatureConditioning.from_dict({1: 2})
    fc2 = FeatureConditioning.from_dict({1: 2, 3: 4})
    assert fc1.digest() != fc2.digest()


# ─────────────────────────────────────────────────────────────────────────────
# feature_ids / __len__
# ─────────────────────────────────────────────────────────────────────────────


def test_feature_ids_sorted():
    fc = FeatureConditioning.from_dict({3: 1, 1: 2})
    assert fc.feature_ids() == (1, 3)


def test_feature_ids_empty():
    fc = FeatureConditioning(entries=())
    assert fc.feature_ids() == ()


def test_len_nonempty():
    fc = FeatureConditioning.from_dict({1: 2, 3: 4})
    assert len(fc) == 2


def test_len_empty():
    fc = FeatureConditioning(entries=())
    assert len(fc) == 0


def test_len_one():
    fc = FeatureConditioning.from_dict({5: 7})
    assert len(fc) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Range validation
# ─────────────────────────────────────────────────────────────────────────────


def test_negative_fid_raises():
    with pytest.raises(BeartypeCallHintParamViolation, match="entries"):
        FeatureConditioning.from_dict({-1: 2})


def test_negative_bucket_raises():
    with pytest.raises(BeartypeCallHintParamViolation, match="entries"):
        FeatureConditioning.from_dict({1: -1})


def test_negative_fid_in_direct_construction_raises():
    with pytest.raises(BeartypeCallHintParamViolation, match="entries"):
        FeatureConditioning(entries=((-1, 2),))


def test_with_added_negative_fid_raises():
    fc = FeatureConditioning(entries=())
    with pytest.raises(BeartypeCallHintParamViolation, match="fid"):
        fc.with_added(-1, 2)


def test_with_added_negative_bucket_raises():
    fc = FeatureConditioning(entries=())
    with pytest.raises(BeartypeCallHintParamViolation, match="bucket"):
        fc.with_added(1, -1)


def test_with_replaced_negative_fid_raises():
    fc = FeatureConditioning.from_dict({1: 2})
    with pytest.raises(BeartypeCallHintParamViolation, match="fid"):
        fc.with_replaced(0, -1, 3)


def test_with_replaced_negative_bucket_raises():
    fc = FeatureConditioning.from_dict({1: 2})
    with pytest.raises(BeartypeCallHintParamViolation, match="bucket"):
        fc.with_replaced(0, 1, -1)


# ─────────────────────────────────────────────────────────────────────────────
# Immutability
# ─────────────────────────────────────────────────────────────────────────────


def test_frozen_raises_on_assignment():
    fc = FeatureConditioning.from_dict({1: 2})
    with pytest.raises(dataclasses.FrozenInstanceError):
        fc.entries = ()  # type: ignore[misc]  # ty: ignore[invalid-assignment]


def test_with_added_does_not_mutate_original():
    fc = FeatureConditioning.from_dict({1: 2})
    _ = fc.with_added(3, 4)
    assert fc.entries == ((1, 2),)


def test_with_removed_does_not_mutate_original():
    fc = FeatureConditioning.from_dict({1: 2, 3: 4})
    _ = fc.with_removed(0)
    assert fc.entries == ((1, 2), (3, 4))


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis property tests
# ─────────────────────────────────────────────────────────────────────────────


@_PROFILE
@given(bag=_valid_bag_strategy())
def test_property_entries_always_sorted_and_dup_free(bag: dict[int, int]) -> None:
    """For any valid bag, entries are always sorted by fid and contain no duplicates."""
    fc = FeatureConditioning.from_dict(bag)
    fids = [fid for fid, _ in fc.entries]
    assert fids == sorted(fids), "entries not in sorted order"
    assert len(fids) == len(set(fids)), "duplicate feature_ids found"


@_PROFILE
@given(bag=_valid_bag_strategy())
def test_property_digest_permutation_invariant(bag: dict[int, int]) -> None:
    """digest is invariant to construction order."""
    if not bag:
        return
    items = list(bag.items())
    reversed_bag = dict(reversed(items))
    fc1 = FeatureConditioning.from_dict(bag)
    fc2 = FeatureConditioning.from_dict(reversed_bag)
    assert fc1.digest() == fc2.digest()
