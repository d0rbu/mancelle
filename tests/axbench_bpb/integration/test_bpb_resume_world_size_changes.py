"""Test BPB resume semantics when world_size changes between runs.

The cache contract (per ``bpb_cache.py:reconcile_cache``):
- Each rank R has its own SQLite cache at ``bpb_cache_path(R)``.
- ``BpbCache(path)`` opens a single SQLite file at ``path`` — no cross-rank
  lookup in ``has`` or ``put``.
- ``reconcile_cache`` syncs that single SQLite with parquet shards in
  ``rank_dir`` (also per-rank); it never reads another rank's directory.

Implication for resume across world_size:
- When ws_resume != ws_initial, work units are re-partitioned. A rank
  that newly owns a work unit (under the new partition) won't see the
  other rank's done state — that work must be redone.
- The test asserts THE ACTUAL BEHAVIOR, not an idealized version.

If a future change adds cross-rank cache reading, this test pins the
current contract; the test should be updated to match the new contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from steering.eval.axbench_bpb.bpb_cache import BpbCache
from steering.eval.axbench_bpb.bpb_scoring import is_mine
from steering.eval.axbench_bpb.types import WorkUnit


def _stage2_partial_run_target(
    rank: int,
    world_size: int,
    run_root: str,
    work_units_serialized: list,
    process_indices: list[int],
) -> dict:
    """Process only the work units in ``process_indices`` whose hash partition
    is owned by THIS rank under ``world_size``. Stores results in the rank's
    own SQLite cache.

    Returns a dict with rank and count of units cached by this rank.
    """
    cache_path = Path(run_root) / f"rank_{rank}_cache.sqlite"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = 0
    with BpbCache(cache_path) as cache:
        for idx in process_indices:
            if idx >= len(work_units_serialized):
                continue
            wu_tup = work_units_serialized[idx]
            wu_obj = WorkUnit(
                method_name=wu_tup[0],
                concept_id=wu_tup[1],
                sample_id=wu_tup[2],
                coefficient_str=wu_tup[3],
            )
            if is_mine(wu_obj, rank=rank, world_size=world_size):
                cache.put(
                    method_name=wu_tup[0],
                    concept_id=wu_tup[1],
                    sample_id=wu_tup[2],
                    coefficient_str=wu_tup[3],
                    target="decoder_only",
                    bpb=1.0,
                    n_output_tokens=10,
                    n_output_bytes=20,
                    truncated=False,
                )
                cached += 1
    return {"rank": rank, "cached": cached}


def _stage2_resume_count_target(
    rank: int,
    world_size: int,
    run_root: str,
    work_units_serialized: list,
) -> dict:
    """Resume run: under the new world_size, count how many work units this
    rank owns AND have NOT been previously cached in this rank's SQLite.

    Returns a dict with rank, owned unit count, and remaining (uncached) count.
    """
    cache_path = Path(run_root) / f"rank_{rank}_cache.sqlite"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    not_in_cache = 0
    owned = 0
    with BpbCache(cache_path) as cache:
        for wu_tup in work_units_serialized:
            wu_obj = WorkUnit(
                method_name=wu_tup[0],
                concept_id=wu_tup[1],
                sample_id=wu_tup[2],
                coefficient_str=wu_tup[3],
            )
            if is_mine(wu_obj, rank=rank, world_size=world_size):
                owned += 1
                if not cache.has(
                    method_name=wu_tup[0],
                    concept_id=wu_tup[1],
                    sample_id=wu_tup[2],
                    coefficient_str=wu_tup[3],
                    target="decoder_only",
                ):
                    not_in_cache += 1
    return {"rank": rank, "owned_now": owned, "remaining": not_in_cache}


def _expected_remaining_after_partial_run(
    ws_initial: int, ws_resume: int,
    work_units: list, process_first_half_indices: list[int],
) -> int:
    """Compute the EXACT expected total_remaining after partial run at
    ws_initial then resume at ws_resume.

    A unit is 'preserved' (not remaining) iff it was processed during
    the initial run AND its is_mine assignment is the same rank under
    both ws_initial and ws_resume. This is the actual contract of the
    per-rank cache (each rank only reads its own SQLite); a unit
    cached by rank R under ws_initial is invisible to rank R' != R
    under ws_resume.
    """
    initial_owners: dict[int, int] = {}
    for idx in process_first_half_indices:
        if idx >= len(work_units):
            continue
        wu_tup = work_units[idx]
        wu_obj = WorkUnit(
            method_name=wu_tup[0], concept_id=wu_tup[1],
            sample_id=wu_tup[2], coefficient_str=wu_tup[3],
        )
        for r in range(ws_initial):
            if is_mine(wu_obj, rank=r, world_size=ws_initial):
                initial_owners[idx] = r
                break

    remaining = 0
    for idx, wu_tup in enumerate(work_units):
        wu_obj = WorkUnit(
            method_name=wu_tup[0], concept_id=wu_tup[1],
            sample_id=wu_tup[2], coefficient_str=wu_tup[3],
        )
        resume_owner = next(
            r for r in range(ws_resume)
            if is_mine(wu_obj, rank=r, world_size=ws_resume)
        )
        if idx in initial_owners and initial_owners[idx] == resume_owner:
            continue  # preserved by same-rank cache
        remaining += 1
    return remaining


@pytest.mark.parametrize("ws_initial,ws_resume", [
    (1, 2),
    (1, 4),  # NEW: drop ws=1 → ws=4
    (2, 2),  # control: same ws — full preservation
    (2, 4),
    (4, 2),
    (4, 4),  # NEW: control at larger ws
    (4, 1),
    (8, 2),  # NEW: drop from large to small
])
def test_resume_world_size_change_partial_done_set(
    gloo_multiprocess, tmp_path, ws_initial, ws_resume,
):
    """After partial run at ws_initial, resume at ws_resume sees the
    EXACT expected number of remaining units, computed from the
    documented per-rank cache contract.

    A regression in BpbCache.has or the partition function (is_mine)
    surfaces as an exact-equality mismatch. Note: this test does NOT
    exercise reconcile_cache (the workers only call cache.put / cache.has,
    never reconcile from parquet shards); reconcile_cache regressions
    must be caught by ``tests/axbench_bpb/functional/test_bpb_cache.py``.
    """
    work_units = [
        (f"m{m}", c, str(s), str(coef))
        for m in range(2)
        for c in range(3)
        for s in range(4)
        for coef in [1.0, 5.0]
    ]
    n_total = len(work_units)
    process_first_half = list(range(n_total // 2))

    expected_remaining = _expected_remaining_after_partial_run(
        ws_initial, ws_resume, work_units, process_first_half,
    )

    # Initial run: process first half
    gloo_multiprocess(
        ws_initial, _stage2_partial_run_target, str(tmp_path),
        work_units, process_first_half,
    )

    # Resume at new world_size: count remaining
    results = gloo_multiprocess(
        ws_resume, _stage2_resume_count_target, str(tmp_path), work_units,
    )

    total_owned_now = sum(r["owned_now"] for r in results)
    total_remaining = sum(r["remaining"] for r in results)

    # Sanity: ranks under ws_resume own ALL work_units exactly once.
    assert total_owned_now == n_total, (
        f"At ws_resume={ws_resume}, ranks should own all {n_total} units; "
        f"got {total_owned_now}"
    )

    # EXACT EQUALITY — no loose bounds.
    assert total_remaining == expected_remaining, (
        f"Resume mismatch at ws_initial={ws_initial} -> ws_resume={ws_resume}: "
        f"expected exactly {expected_remaining} remaining, got {total_remaining}. "
        f"This indicates a regression in BpbCache.has or is_mine."
    )
