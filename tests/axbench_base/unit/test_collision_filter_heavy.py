"""Heavy collision_filter tests (Items 9-11, Batch R1 review fix).

The ``collision_filter`` helper in
``steering.eval.axbench_base.encoder_input`` resolves fid overlap
between the ``active`` and ``inactive`` partitions. The fast unit
tests in ``test_encoder_input.py`` cover the small-input paths; this
file hits the pathological big-overlap scenarios that the user
specifically asked for (items 9-11 of the review):

  * 100% overlap
  * 50% overlap
  * 0% overlap (identity)
  * Hundreds of fids
  * Bucket value mismatches on collision (positive-mode keeps the
    active bucket; negative-mode keeps the inactive bucket)
  * Parametrized over mode, overlap ratio, and scale

Every case asserts the filter's post-condition:

    set(new_active) & set(new_inactive) == set()

which is the invariant the Stage 4 generation pipeline relies on.
"""

from __future__ import annotations

import pytest

from steering.eval.axbench_base.encoder_input import (
    Partition,
    collision_filter,
)
from steering.eval.axbench_base.types import SteeringMode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_partitions_with_overlap_ratio(
    *,
    n_fids_each: int,
    overlap_ratio: float,
    active_bucket: int,
    inactive_bucket: int,
) -> tuple[Partition, Partition]:
    """Build two partitions of ``n_fids_each`` with a given overlap ratio.

    ``overlap_ratio`` must lie in ``[0.0, 1.0]``:

      * 0.0 → disjoint keyspaces (0 shared fids).
      * 0.5 → exactly ``n_fids_each // 2`` shared fids.
      * 1.0 → identical keyspaces (all ``n_fids_each`` shared).

    Active fids use ``active_bucket``, inactive fids use
    ``inactive_bucket``. The shared fids end up in BOTH partitions at
    their respective buckets, so the caller can verify that the filter
    keeps the right side on collision.
    """
    if not 0.0 <= overlap_ratio <= 1.0:
        raise ValueError(f"overlap_ratio must be in [0, 1], got {overlap_ratio}")
    n_overlap = int(round(n_fids_each * overlap_ratio))
    # Shared fids: 0 .. n_overlap-1. Active-only: n_overlap .. n_fids_each-1.
    # Inactive-only: n_fids_each .. 2*n_fids_each - n_overlap - 1.
    active_fids = list(range(n_fids_each))
    # Inactive: first n_overlap are the shared ones; the rest are from
    # a disjoint range starting at ``n_fids_each``.
    inactive_only_start = n_fids_each
    n_inactive_only = n_fids_each - n_overlap
    inactive_fids = (
        list(range(n_overlap))
        + list(range(inactive_only_start, inactive_only_start + n_inactive_only))
    )
    active: Partition = dict.fromkeys(active_fids, active_bucket)
    inactive: Partition = dict.fromkeys(inactive_fids, inactive_bucket)
    return active, inactive


# ---------------------------------------------------------------------------
# Heavy parametrized tests (at least 10 scenarios as spec'd)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode", [SteeringMode.POSITIVE, SteeringMode.NEGATIVE],
)
@pytest.mark.parametrize(
    "n_fids_each,overlap_ratio",
    [
        (100, 0.0),   # zero overlap
        (100, 0.25),  # partial
        (100, 0.50),  # half
        (100, 0.75),  # most of them
        (100, 1.0),   # total
        (250, 0.30),  # bigger scale, partial
        (250, 0.60),  # bigger scale, more-than-half
        (500, 0.10),  # 500 fids, 10% overlap
        (500, 0.50),  # 500 fids, 50% overlap
        (500, 0.90),  # 500 fids, 90% overlap
    ],
)
def test_collision_filter_heavy_parametrized(
    mode: SteeringMode, n_fids_each: int, overlap_ratio: float,
) -> None:
    """The filter produces disjoint partitions for every mode and ratio."""
    active, inactive = _make_partitions_with_overlap_ratio(
        n_fids_each=n_fids_each,
        overlap_ratio=overlap_ratio,
        active_bucket=9,  # distinctive bucket so we can see who wins
        inactive_bucket=7,
    )

    new_active, new_inactive = collision_filter(active, inactive, mode=mode)

    # Post-condition 1: disjoint.
    assert set(new_active).isdisjoint(new_inactive), (
        f"mode={mode} n={n_fids_each} r={overlap_ratio}: "
        f"non-disjoint after filter."
    )

    # Post-condition 2: bucket survivors.
    n_overlap = int(round(n_fids_each * overlap_ratio))
    shared_fids = set(range(n_overlap))
    if mode == SteeringMode.POSITIVE:
        # Active wins: shared fids should all be in new_active at bucket 9,
        # and absent from new_inactive.
        for fid in shared_fids:
            assert fid in new_active
            assert new_active[fid] == 9
            assert fid not in new_inactive
    else:
        # Negative wins: shared fids should all be in new_inactive at
        # bucket 7, and absent from new_active.
        for fid in shared_fids:
            assert fid in new_inactive
            assert new_inactive[fid] == 7
            assert fid not in new_active


# ---------------------------------------------------------------------------
# Edge-case heavy scenarios
# ---------------------------------------------------------------------------


def test_collision_filter_heavy_zero_overlap_identity() -> None:
    """0% overlap -> filter is an identity (returns copies of inputs)."""
    active, inactive = _make_partitions_with_overlap_ratio(
        n_fids_each=200, overlap_ratio=0.0,
        active_bucket=9, inactive_bucket=7,
    )
    new_active, new_inactive = collision_filter(active, inactive, mode=SteeringMode.POSITIVE)
    assert new_active == active
    assert new_inactive == inactive
    # Returned dicts are new objects (safe to mutate).
    assert new_active is not active
    assert new_inactive is not inactive


def test_collision_filter_heavy_full_overlap_positive_mode() -> None:
    """100% overlap, positive mode -> inactive emptied entirely."""
    active, inactive = _make_partitions_with_overlap_ratio(
        n_fids_each=300, overlap_ratio=1.0,
        active_bucket=3, inactive_bucket=7,
    )
    new_active, new_inactive = collision_filter(active, inactive, mode=SteeringMode.POSITIVE)
    assert new_active == active  # all 300 fids survive in active
    assert new_inactive == {}  # entire inactive wiped


def test_collision_filter_heavy_full_overlap_negative_mode() -> None:
    """100% overlap, negative mode -> active emptied entirely."""
    active, inactive = _make_partitions_with_overlap_ratio(
        n_fids_each=300, overlap_ratio=1.0,
        active_bucket=3, inactive_bucket=7,
    )
    new_active, new_inactive = collision_filter(active, inactive, mode=SteeringMode.NEGATIVE)
    assert new_active == {}  # entire active wiped
    assert new_inactive == inactive  # all 300 fids survive in inactive


def test_collision_filter_heavy_bucket_mismatch_positive() -> None:
    """Positive mode keeps the ACTIVE bucket on collision."""
    # Intentional mismatch: shared fid 50 at bucket 9 in active, bucket 2
    # in inactive. Positive mode should keep the 9.
    active = dict.fromkeys(range(100), 9)
    inactive = dict.fromkeys(range(50, 150), 2)  # 50..99 shared
    new_active, new_inactive = collision_filter(active, inactive, mode=SteeringMode.POSITIVE)
    assert set(new_active).isdisjoint(new_inactive)
    # Shared fids kept active bucket 9.
    for fid in range(50, 100):
        assert new_active[fid] == 9
        assert fid not in new_inactive
    # Non-shared inactive kept at bucket 2.
    for fid in range(100, 150):
        assert new_inactive[fid] == 2


def test_collision_filter_heavy_bucket_mismatch_negative() -> None:
    """Negative mode keeps the INACTIVE bucket on collision."""
    active = dict.fromkeys(range(100), 9)
    inactive = dict.fromkeys(range(50, 150), 2)  # 50..99 shared
    new_active, new_inactive = collision_filter(active, inactive, mode=SteeringMode.NEGATIVE)
    assert set(new_active).isdisjoint(new_inactive)
    # Non-shared active kept at bucket 9.
    for fid in range(50):
        assert new_active[fid] == 9
    # Shared fids kept inactive bucket 2.
    for fid in range(50, 100):
        assert new_inactive[fid] == 2
        assert fid not in new_active


def test_collision_filter_heavy_large_overlap_preserves_size() -> None:
    """Big inputs: post-filter cardinality is |total unique fids|."""
    active, inactive = _make_partitions_with_overlap_ratio(
        n_fids_each=500, overlap_ratio=0.4,
        active_bucket=9, inactive_bucket=7,
    )
    # Unique fids: active[0..499] + inactive[0..199 overlap, 500..799 new]
    # = 500 + 300 = 800 unique. After collision_filter(positive):
    # new_active keeps all 500, new_inactive only keeps 300 (the
    # non-overlap part).
    new_active, new_inactive = collision_filter(active, inactive, mode=SteeringMode.POSITIVE)
    assert len(new_active) + len(new_inactive) == 800
    assert set(new_active).isdisjoint(new_inactive)


@pytest.mark.parametrize(
    "mode", [SteeringMode.POSITIVE, SteeringMode.NEGATIVE],
)
def test_collision_filter_heavy_input_not_mutated(mode: SteeringMode) -> None:
    """The filter must not mutate either input dict."""
    active, inactive = _make_partitions_with_overlap_ratio(
        n_fids_each=200, overlap_ratio=0.5,
        active_bucket=9, inactive_bucket=7,
    )
    active_snapshot = dict(active)
    inactive_snapshot = dict(inactive)
    collision_filter(active, inactive, mode=mode)
    assert active == active_snapshot
    assert inactive == inactive_snapshot


def test_collision_filter_heavy_empty_inputs() -> None:
    """Empty input should return empty outputs for both modes."""
    out_pos = collision_filter({}, {}, mode=SteeringMode.POSITIVE)
    assert out_pos == ({}, {})
    out_neg = collision_filter({}, {}, mode=SteeringMode.NEGATIVE)
    assert out_neg == ({}, {})


def test_collision_filter_heavy_one_side_empty_positive() -> None:
    """Positive mode: empty inactive is a no-op; active preserved."""
    active = dict.fromkeys(range(300), 9)
    new_active, new_inactive = collision_filter(active, {}, mode=SteeringMode.POSITIVE)
    assert new_active == active
    assert new_inactive == {}


def test_collision_filter_heavy_one_side_empty_negative() -> None:
    """Negative mode: empty active is a no-op; inactive preserved."""
    inactive = dict.fromkeys(range(300), 7)
    new_active, new_inactive = collision_filter({}, inactive, mode=SteeringMode.NEGATIVE)
    assert new_active == {}
    assert new_inactive == inactive
