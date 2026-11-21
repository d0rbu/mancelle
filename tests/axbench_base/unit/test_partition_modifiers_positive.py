"""Unit tests for positive-mode partition modifiers (Task 27).

Each modifier in ``steering.eval.axbench_base.encoder_input`` is a pure
function that takes ``(active, inactive)`` partitions plus a small set
of keyword arguments and returns a NEW ``(active, inactive)`` tuple.
Inputs must never be mutated; outputs must obey dict semantics (a
feature id appears at most once per partition).
"""

from __future__ import annotations

import random

import pytest

from steering.eval.axbench_base.encoder_input import (
    Partition,
    modify_ours_extract_and_add,
    modify_ours_replace_all,
    modify_ours_replace_random,
)


def _active() -> Partition:
    return {1: 2, 2: 3, 3: 4, 4: 5}


def _inactive() -> Partition:
    return {10: 7, 11: 7, 12: 7}


# ---------------------------------------------------------------------------
# modify_ours_extract_and_add
# ---------------------------------------------------------------------------


def test_modify_ours_extract_and_add_injects_top_k() -> None:
    active = _active()
    inactive = _inactive()
    new_active, new_inactive = modify_ours_extract_and_add(
        active, inactive, top_k_concept_fids=[100, 101, 102], coefficient=4,
    )
    assert new_active == {1: 2, 2: 3, 3: 4, 4: 5, 100: 4, 101: 4, 102: 4}
    assert new_inactive == _inactive()


def test_modify_ours_extract_and_add_updates_existing_bucket() -> None:
    """Top-K fid already in active gets its bucket overwritten (no duplicate)."""
    active = _active()
    new_active, _ = modify_ours_extract_and_add(
        active, {}, top_k_concept_fids=[2, 100], coefficient=6,
    )
    assert new_active[2] == 6  # was 3, now 6
    assert new_active[100] == 6
    assert new_active[1] == 2  # untouched
    assert len(new_active) == 5


def test_modify_ours_extract_and_add_empty_active() -> None:
    new_active, new_inactive = modify_ours_extract_and_add(
        {}, _inactive(), top_k_concept_fids=[1, 2], coefficient=3,
    )
    assert new_active == {1: 3, 2: 3}
    assert new_inactive == _inactive()


def test_modify_ours_extract_and_add_empty_top_k() -> None:
    """Empty top-K leaves active unchanged."""
    active = _active()
    new_active, _ = modify_ours_extract_and_add(
        active, {}, top_k_concept_fids=[], coefficient=3,
    )
    assert new_active == _active()


def test_modify_ours_extract_and_add_does_not_mutate_inputs() -> None:
    active = _active()
    inactive = _inactive()
    modify_ours_extract_and_add(
        active, inactive, top_k_concept_fids=[100], coefficient=1,
    )
    assert active == _active()
    assert inactive == _inactive()


def test_modify_ours_extract_and_add_uses_coefficient_as_bucket() -> None:
    """``coefficient`` is the bucket index injected for each top-K fid."""
    new_active, _ = modify_ours_extract_and_add(
        {}, {}, top_k_concept_fids=[5, 6], coefficient=0,
    )
    assert new_active == {5: 0, 6: 0}


# ---------------------------------------------------------------------------
# modify_ours_replace_random
# ---------------------------------------------------------------------------


def test_modify_ours_replace_random_drops_k() -> None:
    """K random items dropped, then top-K added."""
    rng = random.Random(0)
    active = _active()  # 4 items
    new_active, _ = modify_ours_replace_random(
        active,
        {},
        top_k_concept_fids=[100, 101],
        coefficient=3,
        rng=rng,
    )
    # 4 - 2 dropped + 2 added = 4 active items.
    assert len(new_active) == 4
    # The two top-K are always present.
    assert new_active[100] == 3
    assert new_active[101] == 3


def test_modify_ours_replace_random_k_exceeds_len_active(caplog) -> None:
    """When K > |active|, drop all of active, still add top-K, log warning."""
    rng = random.Random(0)
    active = {1: 2, 2: 3}
    with caplog.at_level("WARNING"):
        new_active, _ = modify_ours_replace_random(
            active,
            {},
            top_k_concept_fids=[100, 101, 102, 103, 104],
            coefficient=4,
            rng=rng,
        )
    # All originals dropped, all top-K added.
    assert new_active == {100: 4, 101: 4, 102: 4, 103: 4, 104: 4}


def test_modify_ours_replace_random_warning_interpolates_integers() -> None:
    """Regression test for commit 9e7934a: loguru does NOT interpolate
    ``%d`` positional args the way stdlib logging does. The original
    code emitted literal ``%d`` in the log output, hiding the real
    counts. Capture the loguru sink and assert the actual integers
    appear in the warning body.
    """
    from io import StringIO

    from loguru import logger

    sink = StringIO()
    handler_id = logger.add(
        sink, format="{level}|{message}", level="WARNING",
    )
    try:
        rng = random.Random(0)
        active = {1: 2, 2: 3}  # size 2
        modify_ours_replace_random(
            active,
            {},
            top_k_concept_fids=[100, 101, 102, 103, 104],  # size 5
            coefficient=4,
            rng=rng,
        )
    finally:
        logger.remove(handler_id)
    out = sink.getvalue()
    assert "top-K size 5" in out, out  # k=5
    assert "drop-candidate count 2" in out, out  # |active|
    assert "active size = 2" in out, out
    # And critically, no literal ``%d`` leaked through:
    assert "%d" not in out, out


def test_modify_ours_replace_random_exact_k_equals_len_active() -> None:
    """K == |active|: drop all, add top-K."""
    rng = random.Random(0)
    active = {1: 2, 2: 3}
    new_active, _ = modify_ours_replace_random(
        active,
        {},
        top_k_concept_fids=[100, 101],
        coefficient=4,
        rng=rng,
    )
    assert new_active == {100: 4, 101: 4}


def test_modify_ours_replace_random_determinism() -> None:
    """Same seed produces identical dropped fid sets."""
    active1 = dict.fromkeys(range(20), 1)
    active2 = dict.fromkeys(range(20), 1)
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    new1, _ = modify_ours_replace_random(
        active1, {}, top_k_concept_fids=[100, 101, 102], coefficient=2, rng=rng1,
    )
    new2, _ = modify_ours_replace_random(
        active2, {}, top_k_concept_fids=[100, 101, 102], coefficient=2, rng=rng2,
    )
    assert new1 == new2


def test_modify_ours_replace_random_different_seeds_drop_different_sets() -> None:
    active = dict.fromkeys(range(20), 1)
    new_a, _ = modify_ours_replace_random(
        active, {}, top_k_concept_fids=[100], coefficient=2, rng=random.Random(1),
    )
    new_b, _ = modify_ours_replace_random(
        active, {}, top_k_concept_fids=[100], coefficient=2, rng=random.Random(2),
    )
    # The kept-from-original sets should generally differ.
    keep_a = set(new_a.keys()) - {100}
    keep_b = set(new_b.keys()) - {100}
    assert keep_a != keep_b


def test_modify_ours_replace_random_does_not_mutate_inputs() -> None:
    active = _active()
    inactive = _inactive()
    modify_ours_replace_random(
        active,
        inactive,
        top_k_concept_fids=[100],
        coefficient=2,
        rng=random.Random(0),
    )
    assert active == _active()
    assert inactive == _inactive()


def test_modify_ours_replace_random_top_k_already_in_active() -> None:
    """Top-K fid already in active: bucket gets updated, no double-counting.

    The dropped set may overlap the top-K set; the post-condition is that
    every top-K fid is present at ``coefficient`` (added after the drop)
    and that the resulting size is bounded by ``len(active)`` (we never
    grow past it because the top-K were already counted in the original).
    """
    rng = random.Random(0)
    active = {1: 2, 2: 3, 3: 4, 4: 5}
    new_active, _ = modify_ours_replace_random(
        active, {}, top_k_concept_fids=[1, 2], coefficient=9, rng=rng,
    )
    assert new_active[1] == 9
    assert new_active[2] == 9
    # No fid appears twice (dict invariant) and we never exceed |active|.
    assert len(new_active) <= 4


def test_modify_ours_replace_random_empty_top_k() -> None:
    rng = random.Random(0)
    active = _active()
    new_active, _ = modify_ours_replace_random(
        active, {}, top_k_concept_fids=[], coefficient=3, rng=rng,
    )
    # K=0 means drop 0 + add 0 = passthrough.
    assert new_active == _active()


# ---------------------------------------------------------------------------
# modify_ours_replace_all
# ---------------------------------------------------------------------------


def test_modify_ours_replace_all_strips_and_sets() -> None:
    active = _active()
    inactive = _inactive()
    new_active, new_inactive = modify_ours_replace_all(
        active, inactive, top_k_concept_fids=[100, 101], coefficient=5,
    )
    assert new_active == {100: 5, 101: 5}
    assert new_inactive == _inactive()


def test_modify_ours_replace_all_empty_top_k_yields_empty_active() -> None:
    new_active, _ = modify_ours_replace_all(
        _active(), {}, top_k_concept_fids=[], coefficient=3,
    )
    assert new_active == {}


def test_modify_ours_replace_all_does_not_mutate_inputs() -> None:
    active = _active()
    inactive = _inactive()
    modify_ours_replace_all(
        active, inactive, top_k_concept_fids=[100], coefficient=2,
    )
    assert active == _active()
    assert inactive == _inactive()


# ---------------------------------------------------------------------------
# Property: no duplicate fids after any positive modifier
# ---------------------------------------------------------------------------


_POSITIVE_MODIFIERS = [
    (
        modify_ours_extract_and_add,
        {"top_k_concept_fids": [100, 101], "coefficient": 4},
    ),
    (
        modify_ours_replace_random,
        {"top_k_concept_fids": [100, 101], "coefficient": 4},
    ),
    (
        modify_ours_replace_all,
        {"top_k_concept_fids": [100, 101], "coefficient": 4},
    ),
]


@pytest.mark.parametrize("modifier,extra_kwargs", _POSITIVE_MODIFIERS)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_no_duplicate_fids_in_active_after_any_positive_modifier(
    modifier, extra_kwargs, seed,
) -> None:
    rng = random.Random(seed)
    active = {i: (i % 5) for i in range(15)}
    inactive = dict.fromkeys(range(20, 25), 7)
    kwargs = dict(extra_kwargs)
    kwargs.setdefault("rng", rng)
    new_active, new_inactive = modifier(active, inactive, **kwargs)
    # Dict-of-dicts can't have duplicate keys, but explicitly assert
    # dict semantics (no overlap between keys() and itself).
    assert len(new_active) == len(set(new_active.keys()))
    assert len(new_inactive) == len(set(new_inactive.keys()))


# ---------------------------------------------------------------------------
# Items 9-11: partition-modifier dedup bugs + disjoint invariant
# ---------------------------------------------------------------------------


class TestPositiveModifierDisjointInvariant:
    """After every positive-mode modifier, the two partitions must be
    DISJOINT before the orchestrator's collision filter runs.
    ``_assert_partition_invariants`` fires inside each modifier; these
    tests exercise the "would be disjoint anyway" case so the assertion
    is covered.
    """

    def test_extract_and_add_removes_top_k_from_inactive(self) -> None:
        """If a top-K fid lives in inactive, extract_and_add removes it
        from inactive so the returned partitions are disjoint."""
        new_active, new_inactive = modify_ours_extract_and_add(
            active={1: 2, 2: 3},
            inactive={100: 7, 101: 7, 200: 7},
            top_k_concept_fids=[100, 101],
            coefficient=4,
        )
        assert new_active == {1: 2, 2: 3, 100: 4, 101: 4}
        assert new_inactive == {200: 7}
        assert set(new_active).isdisjoint(new_inactive)

    def test_replace_random_excludes_top_k_from_drop_candidates(self) -> None:
        """Top-K fids already in active must NOT be drop candidates.

        Before the fix, a top-K fid could be dropped and then re-added,
        reducing the effective drop count. With the exclusion, the
        dropped set is strictly from the non-top-K subset of active.
        """
        rng = random.Random(0)
        active = {1: 2, 2: 3, 3: 4, 100: 9, 101: 9}  # 100 and 101 are top-K
        new_active, _ = modify_ours_replace_random(
            active,
            {},
            top_k_concept_fids=[100, 101],
            coefficient=7,
            rng=rng,
        )
        # Expect: we drop 2 non-top-K fids (candidates {1,2,3}), then
        # re-add top-K at coefficient 7. Total size = 3 + 2 = 5, but
        # 3 in active minus 2 dropped + 2 top-K re-added = 3. No—
        # len(active)=5, 2 drops -> 3 non-top-K-ever fids + 2 re-added
        # top-K (still at 100 and 101 but bucket now 7) = 5 entries
        # minus 2 dropped = 3 entries, plus 2 top-K reinstated = 5.
        # Actually: 5 entries - 2 drops = 3 entries (including the 2
        # top-K that survived, since they weren't drop candidates) +
        # 2 top-K re-added (no-op overwrite) = 3 entries total? No:
        # the 2 top-K are in the 3 surviving entries. So total = 3.
        # Wait — we DROPPED 2 out of the 3 non-top-K candidates (1,2,3).
        # Surviving active = 1 non-top-K + 2 top-K = 3 entries.
        # Then we re-overwrite top-K to bucket 7: still 3 entries.
        assert len(new_active) == 3
        assert new_active[100] == 7
        assert new_active[101] == 7
        # Exactly one non-top-K fid survived.
        non_top_k = [fid for fid in new_active if fid not in {100, 101}]
        assert len(non_top_k) == 1
        assert non_top_k[0] in {1, 2, 3}

    def test_replace_random_removes_top_k_from_inactive(self) -> None:
        """A top-K fid in inactive is removed from inactive by the
        modifier, so the returned partitions are disjoint.
        """
        rng = random.Random(0)
        new_active, new_inactive = modify_ours_replace_random(
            active={1: 2, 2: 3},
            inactive={100: 7, 101: 7, 200: 7},
            top_k_concept_fids=[100, 101],
            coefficient=4,
            rng=rng,
        )
        assert 100 in new_active
        assert 101 in new_active
        assert 100 not in new_inactive
        assert 101 not in new_inactive
        assert 200 in new_inactive
        assert set(new_active).isdisjoint(new_inactive)

    def test_replace_all_removes_top_k_from_inactive(self) -> None:
        """modify_ours_replace_all: top-K fids stripped from inactive."""
        new_active, new_inactive = modify_ours_replace_all(
            active={1: 2, 2: 3},
            inactive={100: 7, 200: 7},
            top_k_concept_fids=[100, 101],
            coefficient=5,
        )
        assert new_active == {100: 5, 101: 5}
        assert new_inactive == {200: 7}
        assert set(new_active).isdisjoint(new_inactive)

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_replace_random_produces_disjoint_partitions_under_any_seed(
        self, seed: int,
    ) -> None:
        """Fuzz the replace_random modifier with overlapping inputs."""
        rng = random.Random(seed)
        active = {i: (i % 5) for i in range(10)}
        inactive = dict.fromkeys(range(8, 15), 7)  # 8, 9 in both
        new_active, new_inactive = modify_ours_replace_random(
            active,
            inactive,
            top_k_concept_fids=[8, 9, 100],  # 8 and 9 in both
            coefficient=3,
            rng=rng,
        )
        assert set(new_active).isdisjoint(new_inactive)
        # Top-K fids end up in active at the coefficient bucket.
        for fid in [8, 9, 100]:
            assert new_active[fid] == 3
