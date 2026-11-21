"""Unit tests for negative-mode partition modifiers (Task 28).

The four ``modify_neg_ours_*`` functions in
``steering.eval.axbench_base.encoder_input`` express "this concept
should NOT be present" by editing both the active partition (to remove
the concept fids) and the inactive partition (to add or restructure the
list of features pinned to the inactive bucket).
"""

from __future__ import annotations

import random

import pytest

from steering.eval.axbench_base.encoder_input import (
    Partition,
    modify_neg_ours_add_to_negatives,
    modify_neg_ours_remove,
    modify_neg_ours_replace_all_negatives,
    modify_neg_ours_replace_random_negatives,
)

_INACTIVE_BUCKET = 7


def _active() -> Partition:
    return {1: 2, 2: 3, 3: 4, 4: 5}


def _inactive() -> Partition:
    return {10: 7, 11: 7, 12: 7}


# ---------------------------------------------------------------------------
# modify_neg_ours_remove
# ---------------------------------------------------------------------------


def test_modify_neg_remove_drops_top_k_from_active() -> None:
    new_active, _ = modify_neg_ours_remove(
        _active(), _inactive(), top_k_concept_fids=[1, 3],
    )
    assert new_active == {2: 3, 4: 5}


def test_modify_neg_remove_leaves_inactive() -> None:
    _, new_inactive = modify_neg_ours_remove(
        _active(), _inactive(), top_k_concept_fids=[1, 3],
    )
    assert new_inactive == _inactive()


def test_modify_neg_remove_top_k_not_in_active() -> None:
    """Removing fids not present in active is a no-op (no KeyError)."""
    new_active, _ = modify_neg_ours_remove(
        _active(), _inactive(), top_k_concept_fids=[999, 1000],
    )
    assert new_active == _active()


def test_modify_neg_remove_empty_top_k() -> None:
    new_active, new_inactive = modify_neg_ours_remove(
        _active(), _inactive(), top_k_concept_fids=[],
    )
    assert new_active == _active()
    assert new_inactive == _inactive()


def test_modify_neg_remove_does_not_mutate_inputs() -> None:
    active = _active()
    inactive = _inactive()
    modify_neg_ours_remove(active, inactive, top_k_concept_fids=[1, 2])
    assert active == _active()
    assert inactive == _inactive()


def test_modify_neg_remove_ignores_extra_kwargs() -> None:
    new_active, _ = modify_neg_ours_remove(
        _active(),
        _inactive(),
        top_k_concept_fids=[1],
        coefficient=99,
        rng=random.Random(0),
        inactive_bucket=42,
    )
    assert 1 not in new_active


# ---------------------------------------------------------------------------
# modify_neg_ours_add_to_negatives
# ---------------------------------------------------------------------------


def test_modify_neg_add_to_negatives_moves_top_k() -> None:
    new_active, new_inactive = modify_neg_ours_add_to_negatives(
        _active(),
        _inactive(),
        top_k_concept_fids=[1, 2],
        inactive_bucket=_INACTIVE_BUCKET,
    )
    # Removed from active.
    assert 1 not in new_active
    assert 2 not in new_active
    assert new_active == {3: 4, 4: 5}
    # Added to inactive.
    assert new_inactive[1] == _INACTIVE_BUCKET
    assert new_inactive[2] == _INACTIVE_BUCKET


def test_modify_neg_add_to_negatives_preserves_existing_inactive() -> None:
    _, new_inactive = modify_neg_ours_add_to_negatives(
        _active(),
        _inactive(),
        top_k_concept_fids=[1],
        inactive_bucket=_INACTIVE_BUCKET,
    )
    # Original inactive entries still there.
    for fid in _inactive():
        assert new_inactive[fid] == _INACTIVE_BUCKET


def test_modify_neg_add_to_negatives_uses_inactive_bucket() -> None:
    """The bucket parameter is the one written to inactive — not hardcoded."""
    _, new_inactive = modify_neg_ours_add_to_negatives(
        _active(),
        _inactive(),
        top_k_concept_fids=[1],
        inactive_bucket=99,
    )
    assert new_inactive[1] == 99


def test_modify_neg_add_to_negatives_top_k_already_in_inactive() -> None:
    """Top-K already in inactive: bucket gets overwritten with inactive_bucket."""
    inactive = {10: 7, 11: 7, 100: 4}  # 100 will be a top-K fid
    _, new_inactive = modify_neg_ours_add_to_negatives(
        _active(), inactive, top_k_concept_fids=[100],
        inactive_bucket=_INACTIVE_BUCKET,
    )
    assert new_inactive[100] == _INACTIVE_BUCKET


def test_modify_neg_add_to_negatives_does_not_mutate_inputs() -> None:
    active = _active()
    inactive = _inactive()
    modify_neg_ours_add_to_negatives(
        active, inactive, top_k_concept_fids=[1], inactive_bucket=_INACTIVE_BUCKET,
    )
    assert active == _active()
    assert inactive == _inactive()


# ---------------------------------------------------------------------------
# modify_neg_ours_replace_random_negatives
# ---------------------------------------------------------------------------


def test_modify_neg_replace_random_negatives_drops_k_existing() -> None:
    """K random existing inactives are dropped; top-K added."""
    rng = random.Random(0)
    active = _active()
    inactive = dict.fromkeys(range(20, 30), 5)  # 10 inactives
    new_active, new_inactive = modify_neg_ours_replace_random_negatives(
        active,
        inactive,
        top_k_concept_fids=[100, 101, 102],
        inactive_bucket=_INACTIVE_BUCKET,
        rng=rng,
    )
    # 10 - 3 dropped + 3 top-K added = 10 entries (assuming no overlap with
    # original inactive fids, which is guaranteed since 100+ > 30).
    assert len(new_inactive) == 10
    assert new_inactive[100] == _INACTIVE_BUCKET
    assert new_inactive[101] == _INACTIVE_BUCKET
    assert new_inactive[102] == _INACTIVE_BUCKET
    # active is unaffected (top-K not in active).
    assert new_active == _active()


def test_modify_neg_replace_random_inactive_short_of_k(caplog) -> None:
    """When |inactive| < K, drop fewer (warning logged), still add top-K."""
    rng = random.Random(0)
    inactive = {10: 7, 11: 7}  # 2 entries
    with caplog.at_level("WARNING"):
        _, new_inactive = modify_neg_ours_replace_random_negatives(
            _active(),
            inactive,
            top_k_concept_fids=[100, 101, 102, 103, 104],  # K=5
            inactive_bucket=_INACTIVE_BUCKET,
            rng=rng,
        )
    # All originals dropped, all top-K added (no overlap).
    assert new_inactive == dict.fromkeys([100, 101, 102, 103, 104], _INACTIVE_BUCKET)


def test_modify_neg_replace_random_warning_interpolates_integers() -> None:
    """Regression test for commit 9e7934a: loguru warnings in
    ``modify_neg_ours_replace_random_negatives`` were emitted with
    printf-style ``%d`` placeholders that loguru does not interpolate.
    Assert the actual integers reach the log output now that the call
    uses f-strings.
    """
    from io import StringIO

    from loguru import logger

    sink = StringIO()
    handler_id = logger.add(
        sink, format="{level}|{message}", level="WARNING",
    )
    try:
        rng = random.Random(0)
        inactive = {10: 7, 11: 7}  # size 2
        modify_neg_ours_replace_random_negatives(
            _active(),
            inactive,
            top_k_concept_fids=[100, 101, 102, 103, 104],  # size 5
            inactive_bucket=_INACTIVE_BUCKET,
            rng=rng,
        )
    finally:
        logger.remove(handler_id)
    out = sink.getvalue()
    assert "top-K size 5" in out, out
    assert "non-top-K inactive count 2" in out, out
    assert "inactive size = 2" in out, out
    assert "%d" not in out, out


def test_modify_neg_replace_random_negatives_strips_top_k_from_active() -> None:
    """Top-K fids are also removed from active (mirrors add_to_negatives)."""
    rng = random.Random(0)
    active = {1: 2, 2: 3, 100: 9}  # 100 will be a top-K fid
    new_active, _ = modify_neg_ours_replace_random_negatives(
        active,
        _inactive(),
        top_k_concept_fids=[100, 101],
        inactive_bucket=_INACTIVE_BUCKET,
        rng=rng,
    )
    assert 100 not in new_active
    assert new_active == {1: 2, 2: 3}


def test_modify_neg_replace_random_negatives_determinism() -> None:
    inactive = dict.fromkeys(range(20, 40), 5)  # 20 entries
    new_a = modify_neg_ours_replace_random_negatives(
        _active(),
        inactive,
        top_k_concept_fids=[100, 101, 102],
        inactive_bucket=_INACTIVE_BUCKET,
        rng=random.Random(42),
    )
    new_b = modify_neg_ours_replace_random_negatives(
        _active(),
        inactive,
        top_k_concept_fids=[100, 101, 102],
        inactive_bucket=_INACTIVE_BUCKET,
        rng=random.Random(42),
    )
    assert new_a == new_b


def test_modify_neg_replace_random_negatives_empty_top_k() -> None:
    """K = 0: no drops, no adds, identity."""
    rng = random.Random(0)
    new_active, new_inactive = modify_neg_ours_replace_random_negatives(
        _active(),
        _inactive(),
        top_k_concept_fids=[],
        inactive_bucket=_INACTIVE_BUCKET,
        rng=rng,
    )
    assert new_active == _active()
    assert new_inactive == _inactive()


def test_modify_neg_replace_random_negatives_does_not_mutate_inputs() -> None:
    active = _active()
    inactive = _inactive()
    modify_neg_ours_replace_random_negatives(
        active,
        inactive,
        top_k_concept_fids=[100],
        inactive_bucket=_INACTIVE_BUCKET,
        rng=random.Random(0),
    )
    assert active == _active()
    assert inactive == _inactive()


# ---------------------------------------------------------------------------
# modify_neg_ours_replace_all_negatives
# ---------------------------------------------------------------------------


def test_modify_neg_replace_all_negatives_strips_all() -> None:
    """All existing inactives deleted; only top-K survive."""
    new_active, new_inactive = modify_neg_ours_replace_all_negatives(
        _active(),
        _inactive(),
        top_k_concept_fids=[100, 101],
        inactive_bucket=_INACTIVE_BUCKET,
    )
    assert new_inactive == {100: _INACTIVE_BUCKET, 101: _INACTIVE_BUCKET}
    # active stripped of top-K (here top-K not in active, so passthrough).
    assert new_active == _active()


def test_modify_neg_replace_all_negatives_strips_top_k_from_active() -> None:
    new_active, _ = modify_neg_ours_replace_all_negatives(
        {1: 2, 2: 3, 100: 9},
        {10: 7},
        top_k_concept_fids=[100],
        inactive_bucket=_INACTIVE_BUCKET,
    )
    assert 100 not in new_active
    assert new_active == {1: 2, 2: 3}


def test_modify_neg_replace_all_negatives_empty_top_k() -> None:
    """K = 0: inactive becomes empty, active strips nothing."""
    new_active, new_inactive = modify_neg_ours_replace_all_negatives(
        _active(),
        _inactive(),
        top_k_concept_fids=[],
        inactive_bucket=_INACTIVE_BUCKET,
    )
    assert new_inactive == {}
    assert new_active == _active()


def test_modify_neg_replace_all_negatives_does_not_mutate_inputs() -> None:
    active = _active()
    inactive = _inactive()
    modify_neg_ours_replace_all_negatives(
        active, inactive, top_k_concept_fids=[100], inactive_bucket=_INACTIVE_BUCKET,
    )
    assert active == _active()
    assert inactive == _inactive()


def test_modify_neg_replace_all_negatives_ignores_extra_kwargs() -> None:
    new_active, new_inactive = modify_neg_ours_replace_all_negatives(
        _active(),
        _inactive(),
        top_k_concept_fids=[100],
        inactive_bucket=_INACTIVE_BUCKET,
        coefficient=99,  # ignored
        rng=random.Random(0),  # ignored
    )
    assert new_inactive == {100: _INACTIVE_BUCKET}
    assert new_active == _active()


# ---------------------------------------------------------------------------
# Cross-modifier: collision behavior on a fid that is in both partitions
# ---------------------------------------------------------------------------


def test_collision_concept_fid_in_both_partitions_goes_to_target_partition() -> None:
    """If a top-K fid is initially in both active and inactive, the negative-mode
    modifier should remove it from active. (The collision filter in the orchestrator
    further enforces mode-aware preferences; here we just verify the modifier's
    contribution.)
    """
    active = {1: 2, 100: 9}
    inactive = {10: 7, 100: 7}  # 100 in both
    new_active, new_inactive = modify_neg_ours_add_to_negatives(
        active,
        inactive,
        top_k_concept_fids=[100],
        inactive_bucket=_INACTIVE_BUCKET,
    )
    # active no longer has 100.
    assert 100 not in new_active
    # inactive still has 100, now at the canonical inactive bucket.
    assert new_inactive[100] == _INACTIVE_BUCKET


# ---------------------------------------------------------------------------
# Property: no duplicate fids after any negative-mode modifier
# ---------------------------------------------------------------------------


_NEGATIVE_MODIFIERS = [
    (
        modify_neg_ours_remove,
        {"top_k_concept_fids": [100, 101]},
    ),
    (
        modify_neg_ours_add_to_negatives,
        {"top_k_concept_fids": [100, 101], "inactive_bucket": _INACTIVE_BUCKET},
    ),
    (
        modify_neg_ours_replace_random_negatives,
        {"top_k_concept_fids": [100, 101], "inactive_bucket": _INACTIVE_BUCKET},
    ),
    (
        modify_neg_ours_replace_all_negatives,
        {"top_k_concept_fids": [100, 101], "inactive_bucket": _INACTIVE_BUCKET},
    ),
]


@pytest.mark.parametrize("modifier,extra_kwargs", _NEGATIVE_MODIFIERS)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_no_duplicate_fids_in_active_after_any_negative_modifier(
    modifier, extra_kwargs, seed,
) -> None:
    rng = random.Random(seed)
    active = {i: (i % 5) for i in range(15)}
    inactive = dict.fromkeys(range(20, 25), 7)
    kwargs = dict(extra_kwargs)
    kwargs.setdefault("rng", rng)
    new_active, new_inactive = modifier(active, inactive, **kwargs)
    assert len(new_active) == len(set(new_active.keys()))
    assert len(new_inactive) == len(set(new_inactive.keys()))


# ---------------------------------------------------------------------------
# Items 9-11: partition-modifier dedup bugs + disjoint invariant
# ---------------------------------------------------------------------------


class TestNegativeModifierDisjointInvariant:
    """After every negative-mode modifier, the two partitions must be
    DISJOINT before the orchestrator's collision filter runs. Negative
    mode resolves cross-partition collisions in favor of ``inactive``.
    """

    def test_remove_resolves_pre_collision_in_favor_of_inactive(self) -> None:
        """modify_neg_ours_remove: pre-existing cross-partition collision
        is resolved by stripping the fid from active (negative convention).
        """
        new_active, new_inactive = modify_neg_ours_remove(
            active={5: 9, 1: 2},
            inactive={5: 7, 10: 7},
            top_k_concept_fids=[],
        )
        assert new_active == {1: 2}
        assert new_inactive == {5: 7, 10: 7}
        assert set(new_active).isdisjoint(new_inactive)

    def test_add_to_negatives_disjoint_even_with_pre_collision(self) -> None:
        new_active, new_inactive = modify_neg_ours_add_to_negatives(
            active={5: 9, 100: 3, 1: 2},
            inactive={5: 7, 10: 7},
            top_k_concept_fids=[100],
            inactive_bucket=_INACTIVE_BUCKET,
        )
        # 100 moved from active to inactive at _INACTIVE_BUCKET.
        # 5 in both partitions (pre-existing collision) -> removed
        # from active by negative-convention.
        assert 100 not in new_active
        assert new_inactive[100] == _INACTIVE_BUCKET
        assert 5 not in new_active
        assert new_inactive[5] == 7  # original bucket preserved
        assert set(new_active).isdisjoint(new_inactive)

    def test_replace_random_negatives_excludes_top_k_from_drop_candidates(
        self,
    ) -> None:
        """Top-K fids already in inactive must NOT be drop candidates.

        Before the fix, a top-K fid that happened to already be in
        inactive could be dropped (and then re-added), reducing the
        effective drop count to K-1. With the exclusion, the dropped
        set is strictly from the non-top-K subset of inactive.
        """
        rng = random.Random(0)
        inactive = {10: 5, 11: 5, 12: 5, 100: 5, 101: 5}  # 100/101 top-K
        new_active, new_inactive = modify_neg_ours_replace_random_negatives(
            active=_active(),
            inactive=inactive,
            top_k_concept_fids=[100, 101],
            inactive_bucket=_INACTIVE_BUCKET,
            rng=rng,
        )
        # 3 non-top-K candidates (10, 11, 12), 2 dropped,
        # 1 survives + 2 top-K reinstated = 3 inactive.
        assert len(new_inactive) == 3
        assert new_inactive[100] == _INACTIVE_BUCKET
        assert new_inactive[101] == _INACTIVE_BUCKET
        # Exactly one of the non-top-K original inactives survived.
        non_top_k = [fid for fid in new_inactive if fid not in {100, 101}]
        assert len(non_top_k) == 1
        assert non_top_k[0] in {10, 11, 12}

    def test_replace_random_negatives_disjoint_with_top_k_in_active(self) -> None:
        """Top-K fids in active are stripped; partitions are disjoint."""
        rng = random.Random(0)
        active = {1: 2, 100: 9, 101: 9}  # 100 and 101 are top-K
        inactive = {20: 5, 21: 5}
        new_active, new_inactive = modify_neg_ours_replace_random_negatives(
            active=active,
            inactive=inactive,
            top_k_concept_fids=[100, 101],
            inactive_bucket=_INACTIVE_BUCKET,
            rng=rng,
        )
        assert 100 not in new_active
        assert 101 not in new_active
        assert new_inactive[100] == _INACTIVE_BUCKET
        assert new_inactive[101] == _INACTIVE_BUCKET
        assert set(new_active).isdisjoint(new_inactive)

    def test_replace_all_negatives_disjoint_even_with_pre_collision(self) -> None:
        """modify_neg_ours_replace_all_negatives: pre-existing active/inactive
        overlap is resolved by stripping from active.
        """
        new_active, new_inactive = modify_neg_ours_replace_all_negatives(
            active={1: 2, 5: 9, 100: 3},
            inactive={5: 7, 10: 7},
            top_k_concept_fids=[100],
            inactive_bucket=_INACTIVE_BUCKET,
        )
        assert new_inactive == {100: _INACTIVE_BUCKET}
        assert 100 not in new_active
        # 5 stays in active since inactive is REPLACED (inactive = {100}),
        # so there's no collision on 5 anymore (old inactive dropped).
        assert set(new_active).isdisjoint(new_inactive)

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_replace_random_negatives_produces_disjoint_under_any_seed(
        self, seed: int,
    ) -> None:
        rng = random.Random(seed)
        active = {i: (i % 5) for i in range(10)}
        inactive = dict.fromkeys(range(8, 15), 7)  # 8, 9 in both partitions
        new_active, new_inactive = modify_neg_ours_replace_random_negatives(
            active=active,
            inactive=inactive,
            top_k_concept_fids=[8, 9, 100],
            inactive_bucket=_INACTIVE_BUCKET,
            rng=rng,
        )
        assert set(new_active).isdisjoint(new_inactive)
        for fid in [8, 9, 100]:
            assert new_inactive[fid] == _INACTIVE_BUCKET
