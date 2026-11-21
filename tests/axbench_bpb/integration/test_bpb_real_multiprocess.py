"""Real-multiprocess (gloo backend) tests for AxBench-BPB DDP correctness.

These tests use the gloo_multiprocess fixture from PR-1 to spawn actual
processes, initialise a real distributed process group, and verify that
each BPB stage's distributed semantics match the documented contract.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch.distributed as dist

from steering.eval.axbench_bpb.bpb_scoring import is_mine
from steering.eval.axbench_bpb.types import WorkUnit
from tests.runtime_attribute_helpers import set_runtime_attribute
from tests.statistics_doubles import fixed_statistics_double


def _stage0_target(rank: int, world_size: int, run_root: str):
    """Worker that runs _stage_prepare_context_cache and reports the
    number of times build_context_cache was invoked, plus whether the
    manifest exists at the end.

    Parallel Stage 1 contract: every rank calls build_context_cache once
    (with its own ddp_rank/ddp_world_size kwargs); only rank 0 calls
    write_context_cache_manifest after the barrier."""
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))

    import scripts.eval.axbench_bpb_eval as eval_mod
    from scripts.eval.axbench_bpb_eval import _stage_prepare_context_cache
    from tests.axbench_bpb._mini_runtime import build_mini_runtime_state

    build_call_count = [0]
    manifest_write_count = [0]

    def fake_build_context_cache(**kwargs):
        build_call_count[0] += 1

    def fake_write_manifest(**kwargs):
        manifest_write_count[0] += 1
        op = kwargs["output_paths"]
        manifest_path = op.context_cache_manifest()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({}), encoding="utf-8")

    rs = build_mini_runtime_state(
        run_root=Path(run_root), rank=rank, world_size=world_size,
    )

    # Patch all heavy callers in the dispatcher module so no real I/O
    # happens. load_statistics + build_boundaries_from_stats run on
    # every rank now (cheap, CPU-only); collect_unique_prefixes +
    # write_context_cache_manifest are rank-0-only after the barrier.
    fake_stats = fixed_statistics_double()
    fake_boundaries = MagicMock()

    original_build = eval_mod.build_context_cache
    original_load_stats = eval_mod.load_statistics
    original_build_boundaries = eval_mod.build_boundaries_from_stats
    original_assert_tok_checksum = eval_mod.assert_tokenizer_checksum_matches
    original_assert_bos_lengths = eval_mod.assert_bos_lengths_match
    original_collect_prefixes = eval_mod.collect_unique_prefixes
    original_write_manifest = eval_mod.write_context_cache_manifest
    set_runtime_attribute(
        eval_mod,
        "build_context_cache",
        fake_build_context_cache,
    )
    set_runtime_attribute(
        eval_mod,
        "load_statistics",
        lambda *_a, **_kw: fake_stats,
    )
    set_runtime_attribute(
        eval_mod,
        "build_boundaries_from_stats",
        lambda *_a, **_kw: fake_boundaries,
    )
    set_runtime_attribute(
        eval_mod,
        "assert_tokenizer_checksum_matches",
        lambda **_kw: None,
    )
    set_runtime_attribute(
        eval_mod,
        "assert_bos_lengths_match",
        lambda **_kw: None,
    )
    set_runtime_attribute(
        eval_mod,
        "collect_unique_prefixes",
        lambda *_a, **_kw: {},
    )
    set_runtime_attribute(
        eval_mod,
        "write_context_cache_manifest",
        fake_write_manifest,
    )
    try:
        _stage_prepare_context_cache(rs)
    finally:
        eval_mod.build_context_cache = original_build
        eval_mod.load_statistics = original_load_stats
        eval_mod.build_boundaries_from_stats = original_build_boundaries
        eval_mod.assert_tokenizer_checksum_matches = original_assert_tok_checksum
        eval_mod.assert_bos_lengths_match = original_assert_bos_lengths
        eval_mod.collect_unique_prefixes = original_collect_prefixes
        eval_mod.write_context_cache_manifest = original_write_manifest

    manifest_exists = rs.output_paths().context_cache_manifest().exists()
    return {
        "rank": rank,
        "build_calls": build_call_count[0],
        "manifest_writes": manifest_write_count[0],
        "manifest_exists": manifest_exists,
    }


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_stage0_every_rank_calls_build_context_cache(
    gloo_multiprocess, tmp_path, world_size,
):
    """Parallel Stage 1: every rank invokes build_context_cache once for
    its sharded slice. Only rank 0 writes the manifest after the barrier."""
    results = gloo_multiprocess(world_size, _stage0_target, str(tmp_path))
    by_rank = {r["rank"]: r for r in results}
    for r in range(world_size):
        assert by_rank[r]["build_calls"] == 1, (
            f"Rank {r} should call build_context_cache exactly once "
            f"(sharded extraction); got {by_rank[r]['build_calls']}"
        )
    assert by_rank[0]["manifest_writes"] == 1, (
        f"Rank 0 must write the manifest exactly once; "
        f"got {by_rank[0]['manifest_writes']}"
    )
    for r in range(1, world_size):
        assert by_rank[r]["manifest_writes"] == 0, (
            f"Only rank 0 writes the manifest; rank {r} wrote "
            f"{by_rank[r]['manifest_writes']} times"
        )


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_stage0_manifest_visible_on_all_ranks_after_barrier(
    gloo_multiprocess, tmp_path, world_size,
):
    """After Stage 0 completes (with barrier), all ranks see the manifest."""
    results = gloo_multiprocess(world_size, _stage0_target, str(tmp_path))
    for r in results:
        assert r["manifest_exists"], (
            f"Rank {r['rank']} did not see the manifest after Stage 0"
        )


def _stage2_target(rank: int, world_size: int, work_units_serialized: list):
    """Worker that returns which work_units it owns per is_mine.

    work_units_serialized is a list of (method_name, concept_id, sample_id,
    coefficient_str) tuples; we reconstruct WorkUnit instances inside the
    worker to keep the spawn payload pickle-friendly.
    """
    owned = []
    for tup in work_units_serialized:
        wu = WorkUnit(
            method_name=tup[0],
            concept_id=tup[1],
            sample_id=tup[2],
            coefficient_str=tup[3],
        )
        if is_mine(wu, rank=rank, world_size=world_size):
            owned.append(tup)
    return {"rank": rank, "owned": owned}


@pytest.mark.parametrize("world_size", [2, 3, 4, 5, 8])
def test_stage2_hash_partition_partitions_complete_and_disjoint(
    gloo_multiprocess, world_size,
):
    """Across all ranks, every work_unit is owned by exactly one rank.

    Verifies the SHA256-based hash partition produces a complete + disjoint
    cover of the work-unit space under real gloo distributed execution.
    """
    work_units = [
        (f"method{m}", c, s, str(coef))
        for m in range(3)
        for c in range(5)
        for s in range(4)
        for coef in [1.0, 5.0, 10.0]
    ]  # 3 * 5 * 4 * 3 = 180 units total

    results = gloo_multiprocess(world_size, _stage2_target, work_units)

    all_owned: list[tuple] = []
    for r in results:
        all_owned.extend(r["owned"])

    # Coverage: every input work_unit is owned by some rank.
    assert sorted(all_owned) == sorted(work_units), (
        f"Hash partition missed or duplicated work units at world_size={world_size}: "
        f"got {len(all_owned)} owned, expected {len(work_units)}"
    )
    # Disjoint: no work unit is owned by more than one rank.
    counts = Counter(all_owned)
    duplicates = [k for k, v in counts.items() if v > 1]
    assert not duplicates, (
        f"Hash partition assigned {len(duplicates)} units to multiple ranks "
        f"at world_size={world_size}: {duplicates[:3]}..."
    )


def test_stage2_hash_partition_world_size_invariant():
    """Re-partitioning under different world_size still produces a complete,
    valid partition (each work_unit has exactly one owner in [0, ws)).

    This is a pure unit test (no gloo); it stress-tests is_mine across
    a range of world_sizes including world_size=1 (degenerate).
    """
    work_units = [
        WorkUnit(
            method_name=f"m{i}",
            concept_id=i % 5,
            sample_id=str(i % 3),
            coefficient_str=str(i),
        )
        for i in range(100)
    ]
    for ws in [1, 2, 4, 8]:
        # Find each unit's owner via brute-force over all candidate ranks.
        owners: list[int] = []
        for wu in work_units:
            owning_ranks = [r for r in range(ws) if is_mine(wu, rank=r, world_size=ws)]
            assert len(owning_ranks) == 1, (
                f"WorkUnit {wu} has {len(owning_ranks)} owners at world_size={ws}: "
                f"{owning_ranks}"
            )
            owners.append(owning_ranks[0])
        assert len(owners) == 100
        assert all(0 <= o < ws for o in owners), (
            f"Owner rank out of range [0, {ws}) at world_size={ws}: {owners}"
        )


def test_is_mine_uses_documented_sha256_payload_recipe():
    """Pin the documented hash recipe: SHA256 of '{method}|{concept}|{sample}|{coef}',
    truncated to first 8 hex chars, mod world_size.

    A regression to MD5, a different separator, or different field ordering
    silently breaks cache compatibility across runs/world_sizes — this test
    pins the exact recipe.
    """
    import hashlib

    from steering.eval.axbench_bpb.bpb_scoring import is_mine
    from steering.eval.axbench_bpb.types import WorkUnit

    test_cases = [
        ("m0", 0, "0", "1.0"),
        ("method_alpha", 42, "sample_x", "0.001"),
        ("ours_v1", 99, "long_sample_id", "100.0"),
    ]
    for method, concept, sample, coef in test_cases:
        wu = WorkUnit(method_name=method, concept_id=concept,
                      sample_id=sample, coefficient_str=coef)
        expected_payload = f"{method}|{concept}|{sample}|{coef}"
        expected_hash = int(
            hashlib.sha256(expected_payload.encode("utf-8")).hexdigest()[:8],
            16,
        )
        for ws in [1, 2, 3, 4, 8]:
            expected_owner = expected_hash % ws
            for rank in range(ws):
                actual = is_mine(wu, rank=rank, world_size=ws)
                expected = (rank == expected_owner)
                assert actual == expected, (
                    f"is_mine misassigned: wu={wu}, ws={ws}, rank={rank}, "
                    f"expected_owner={expected_owner}, got is_mine={actual}. "
                    f"Recipe under test: SHA256('{expected_payload}')[:8] = "
                    f"{expected_hash:08x} % {ws} = {expected_owner}"
                )


def test_is_mine_partition_distinguishes_coefficients():
    """coefficient_str must be in the hash payload, otherwise all
    coefficient sweeps for a given (method, concept, sample) map to the
    same rank — defeating parallelism for the most common workload shape.
    """
    from steering.eval.axbench_bpb.bpb_scoring import is_mine
    from steering.eval.axbench_bpb.types import WorkUnit

    # Six coefficients that span enough of hash-space to virtually
    # guarantee at least 2 distinct owners at world_size=4.
    coefficients = ["0.001", "1.0", "5.0", "10.0", "50.0", "100.0"]
    units = [
        WorkUnit(method_name="m", concept_id=1, sample_id="s",
                 coefficient_str=c)
        for c in coefficients
    ]
    owners_at_ws4 = []
    for u in units:
        owning_ranks = [r for r in range(4) if is_mine(u, rank=r, world_size=4)]
        assert len(owning_ranks) == 1, (
            f"WorkUnit {u} has {len(owning_ranks)} owners at ws=4: {owning_ranks}"
        )
        owners_at_ws4.append(owning_ranks[0])

    distinct_owners = set(owners_at_ws4)
    assert len(distinct_owners) >= 2, (
        f"All 6 distinct coefficients hit the same rank at ws=4: {owners_at_ws4}. "
        "This indicates coefficient_str was dropped from the is_mine hash payload — "
        "every coefficient sweep would serialize to one rank."
    )


def test_is_mine_partition_distinguishes_methods():
    """method_name must be in the hash payload (defense against analogous
    coefficient_str regression for methods)."""
    from steering.eval.axbench_bpb.bpb_scoring import is_mine
    from steering.eval.axbench_bpb.types import WorkUnit

    methods = [f"method_{i}" for i in range(8)]
    units = [
        WorkUnit(method_name=m, concept_id=1, sample_id="s",
                 coefficient_str="1.0")
        for m in methods
    ]
    owners_at_ws4 = [
        next(r for r in range(4) if is_mine(u, rank=r, world_size=4))
        for u in units
    ]
    assert len(set(owners_at_ws4)) >= 2, (
        f"All 8 distinct methods hit the same rank at ws=4: {owners_at_ws4}. "
        "method_name may have been dropped from the is_mine hash payload."
    )


def _stage3_target_with_artificial_slow_rank(
    rank: int, world_size: int, slow_rank: int, run_root: str,
):
    """Artificially slow down `slow_rank` before the production
    `_maybe_barrier` call. Asserts the production helper actually blocks
    rank 0 from proceeding past the barrier until the slow rank arrives.
    """
    from scripts.eval.axbench_bpb_eval import _maybe_barrier

    started_path = Path(run_root) / f"rank_{rank}_started.txt"
    started_path.write_text(str(time.monotonic()))

    if rank == slow_rank:
        time.sleep(2.0)

    # Use the PRODUCTION helper, not raw dist.barrier()
    _maybe_barrier(world_size)

    barrier_done_path = Path(run_root) / f"rank_{rank}_barrier_done.txt"
    barrier_done_path.write_text(str(time.monotonic()))

    if rank == 0:
        agg_started_path = Path(run_root) / "agg_started.txt"
        agg_started_path.write_text(str(time.monotonic()))


def test_stage3_barrier_blocks_until_slow_rank_arrives(
    gloo_multiprocess, tmp_path,
):
    """Rank 0's agg step must start AFTER all ranks finish Stage 2.

    Slow rank 2 sleeps 2 seconds before the barrier. Without a working
    barrier, rank 0 would race ahead and start aggregation immediately.
    With the barrier, rank 0 must wait for rank 2.
    """
    gloo_multiprocess(
        4, _stage3_target_with_artificial_slow_rank,
        2, str(tmp_path),  # slow_rank=2
    )

    # Read timestamps written by each rank
    rank_0_done = float((tmp_path / "rank_0_barrier_done.txt").read_text())
    rank_2_done = float((tmp_path / "rank_2_barrier_done.txt").read_text())
    agg_start = float((tmp_path / "agg_started.txt").read_text())

    # Rank 0 should have completed barrier no earlier than rank 2 (slow).
    # Allow 50ms tolerance for clock skew across processes.
    assert rank_0_done >= rank_2_done - 0.05, (
        f"Rank 0 finished barrier at {rank_0_done} but rank 2 only "
        f"reached barrier at {rank_2_done}; barrier did NOT block rank 0"
    )
    # Aggregation start must come after rank 2's barrier completion.
    assert agg_start >= rank_2_done - 0.05, (
        f"Aggregation started at {agg_start} but rank 2 barrier completed "
        f"at {rank_2_done}; rank 0 raced ahead of the barrier"
    )


def _stage_target_with_failing_rank(
    rank: int, world_size: int, failing_rank: int,
):
    """A rank-failure simulation: `failing_rank` raises mid-stage; other
    ranks attempt to barrier and would block waiting for the failed
    rank if the fixture's timeout mechanism didn't surface the error.
    """
    if rank == failing_rank:
        raise RuntimeError(f"intentional rank-{failing_rank} failure")
    # Other ranks barrier — would block forever without timeout
    dist.barrier()
    return rank


def test_rank_failure_propagates_does_not_deadlock(gloo_multiprocess):
    """If a rank fails mid-stage, the test process surfaces the failure
    via the gloo_multiprocess fixture's timeout / exception relay rather
    than blocking the test forever.

    Without the fixture's `_RESULT_TIMEOUT_S` mechanism (added in PR-1
    review hardening), this test would hang because ranks 0/1/3 would
    block on `dist.barrier()` waiting for rank 2 which raised before
    arriving at the barrier.

    Note: The fixture surfaces the first error collected from any rank
    (format: "gloo_multiprocess: rank N raised: <traceback>"). The root
    cause may appear as rank 2's original ``RuntimeError`` in the message,
    or as a gloo TCP disconnect error on a surviving rank whose barrier was
    interrupted when rank 2 exited — both confirm the fixture intercepted
    the failure rather than hanging indefinitely.
    """
    with pytest.raises(RuntimeError, match="gloo_multiprocess: rank \\d+ raised:"):
        gloo_multiprocess(4, _stage_target_with_failing_rank, 2)


def _stage_target_silent_hang(rank: int, world_size: int, hanging_rank: int):
    """A rank-hang simulation: `hanging_rank` sleeps silently mid-stage.
    Other ranks reach `dist.barrier()` and would block forever.
    """
    if rank == hanging_rank:
        time.sleep(300)  # Hang silently — no exception
    dist.barrier()
    return rank


def test_silent_rank_hang_surfaces_via_timeout(monkeypatch):
    """A silently-hung rank surfaces via _RESULT_TIMEOUT_S (PR-1
    hardening), not by deadlocking the test forever.

    Distinct from test_rank_failure_propagates_does_not_deadlock which
    exercises the exception-relay path; this test exercises the
    queue-timeout path that fires when no worker reports an error.
    """
    import tests.axbench_base._multiprocess_helpers as mp_helpers
    monkeypatch.setattr(mp_helpers, "_RESULT_TIMEOUT_S", 5)

    from tests.axbench_base._multiprocess_helpers import run_gloo_multiprocess

    with pytest.raises(RuntimeError, match="did not report results within"):
        run_gloo_multiprocess(4, _stage_target_silent_hang, 2)


def _stage0_target_with_rank_0_failure(
    rank: int, world_size: int, run_root: str,
):
    """Simulate rank 0 raising during ``build_context_cache``.

    Stubs the upstream ``load_statistics`` + ``build_boundaries_from_stats``
    helpers and the rank-0 tokenizer guards so rank 0 actually REACHES
    ``build_context_cache``. Under the parallel-Stage-1 contract, ranks
    1+ also reach ``build_context_cache`` (they now own a shard of the
    extraction). The fake raises on every rank; the ``gloo_multiprocess``
    fixture must surface the failure rather than deadlocking at the
    barrier.
    """
    from unittest.mock import MagicMock

    import scripts.eval.axbench_bpb_eval as eval_mod
    from scripts.eval.axbench_bpb_eval import _stage_prepare_context_cache
    from tests.axbench_bpb._mini_runtime import build_mini_runtime_state

    rs = build_mini_runtime_state(
        run_root=Path(run_root), rank=rank, world_size=world_size,
    )

    fake_stats = MagicMock(per_feature_stats=MagicMock())

    def fake_build_context_cache(**kwargs):
        # Now invoked on every rank (sharded extraction). Always raises.
        raise RuntimeError("intentional rank-0 Stage 0 failure")

    original_build = eval_mod.build_context_cache
    original_load_stats = eval_mod.load_statistics
    original_build_boundaries = eval_mod.build_boundaries_from_stats
    original_assert_tok = eval_mod.assert_tokenizer_checksum_matches
    original_assert_bos = eval_mod.assert_bos_lengths_match
    set_runtime_attribute(
        eval_mod,
        "build_context_cache",
        fake_build_context_cache,
    )
    set_runtime_attribute(
        eval_mod,
        "load_statistics",
        lambda path: fake_stats,
    )
    set_runtime_attribute(
        eval_mod,
        "build_boundaries_from_stats",
        lambda *a, **kw: MagicMock(),
    )
    set_runtime_attribute(
        eval_mod,
        "assert_tokenizer_checksum_matches",
        lambda **_kw: None,
    )
    set_runtime_attribute(
        eval_mod,
        "assert_bos_lengths_match",
        lambda **_kw: None,
    )
    try:
        _stage_prepare_context_cache(rs)
    finally:
        eval_mod.build_context_cache = original_build
        eval_mod.load_statistics = original_load_stats
        eval_mod.build_boundaries_from_stats = original_build_boundaries
        eval_mod.assert_tokenizer_checksum_matches = original_assert_tok
        eval_mod.assert_bos_lengths_match = original_assert_bos


def test_stage0_rank_0_failure_surfaces_does_not_deadlock(gloo_multiprocess):
    """If rank 0 raises during Stage 0's build_context_cache, ranks 1+
    block at the post-stage barrier. The fixture must surface the
    failure (via TCP-disconnect detection or the _RESULT_TIMEOUT_S
    timeout) rather than hanging the test forever.
    """
    with pytest.raises(RuntimeError, match="gloo_multiprocess: rank \\d+ raised:"):
        gloo_multiprocess(4, _stage0_target_with_rank_0_failure, "/tmp")


# ---------- Post-review hardening: between-barrier failure scenarios.
# These test the precise deadlock class commit 32cc48db claims to narrow:
# rank 0 raises BETWEEN barrier 1 (post-extraction) and barrier 2
# (post-manifest+guards), while ranks 1+ are stranded at barrier 2.
# Under gloo, the fixture's TCP-disconnect detection or _RESULT_TIMEOUT_S
# surfaces the failure rather than hanging. Under NCCL the survivors
# would block for the watchdog timeout (10 min default) — separately
# documented as a known limitation. ----------


def _stage0_target_rank_0_fails_in_manifest_write(
    rank: int, world_size: int, run_root: str,
):
    """Simulate rank 0 raising during write_context_cache_manifest
    (i.e., between barrier 1 and barrier 2). Other ranks pass barrier 1
    and block on barrier 2; the rank-0 failure must surface without
    deadlocking the test."""
    from unittest.mock import MagicMock

    import scripts.eval.axbench_bpb_eval as eval_mod
    from scripts.eval.axbench_bpb_eval import _stage_prepare_context_cache
    from tests.axbench_bpb._mini_runtime import build_mini_runtime_state

    rs = build_mini_runtime_state(
        run_root=Path(run_root), rank=rank, world_size=world_size,
    )
    fake_stats = MagicMock(per_feature_stats=MagicMock())

    def fake_build_ok(**kwargs):
        # Succeeds on every rank; manifest write is the failure point.
        return None

    def fake_write_manifest_raises(**kwargs):
        raise RuntimeError("intentional rank-0 manifest write failure")

    original_build = eval_mod.build_context_cache
    original_load_stats = eval_mod.load_statistics
    original_build_boundaries = eval_mod.build_boundaries_from_stats
    original_assert_tok = eval_mod.assert_tokenizer_checksum_matches
    original_assert_bos = eval_mod.assert_bos_lengths_match
    original_collect = eval_mod.collect_unique_prefixes
    original_write = eval_mod.write_context_cache_manifest
    set_runtime_attribute(eval_mod, "build_context_cache", fake_build_ok)
    set_runtime_attribute(
        eval_mod,
        "load_statistics",
        lambda path: fake_stats,
    )
    set_runtime_attribute(
        eval_mod,
        "build_boundaries_from_stats",
        lambda *a, **kw: MagicMock(),
    )
    set_runtime_attribute(
        eval_mod,
        "assert_tokenizer_checksum_matches",
        lambda **_kw: None,
    )
    set_runtime_attribute(
        eval_mod,
        "assert_bos_lengths_match",
        lambda **_kw: None,
    )
    set_runtime_attribute(
        eval_mod,
        "collect_unique_prefixes",
        lambda *_a, **_kw: {},
    )
    set_runtime_attribute(
        eval_mod,
        "write_context_cache_manifest",
        fake_write_manifest_raises,
    )
    try:
        _stage_prepare_context_cache(rs)
    finally:
        eval_mod.build_context_cache = original_build
        eval_mod.load_statistics = original_load_stats
        eval_mod.build_boundaries_from_stats = original_build_boundaries
        eval_mod.assert_tokenizer_checksum_matches = original_assert_tok
        eval_mod.assert_bos_lengths_match = original_assert_bos
        eval_mod.collect_unique_prefixes = original_collect
        eval_mod.write_context_cache_manifest = original_write


def test_stage0_rank_0_manifest_write_failure_does_not_deadlock(
    gloo_multiprocess,
):
    """Rank 0 raises in write_context_cache_manifest after barrier 1.

    What this test verifies (precisely): under the gloo backend, when
    rank 0 raises ANY exception during the rank-0-only block between
    barriers 1 and 2, the gloo_multiprocess fixture surfaces the
    failure (via TCP disconnect or _RESULT_TIMEOUT_S) rather than
    hanging the test indefinitely.

    What this test does NOT verify: the production barrier-1
    ordering. Removing ``_maybe_barrier(ws)`` (barrier 1) entirely
    would still pass this test because the fixture surfaces the
    rank-0 exception either way. Barrier-1 ordering is pinned by
    ``tests/axbench_bpb/integration/test_ddp_aggregation_barrier.py::
    test_stage0_emits_barriers_around_rank0_manifest`` (recorder-based
    unit-level assertion on the call sequence
    build → barrier → manifest → barrier).
    """
    with pytest.raises(RuntimeError, match="gloo_multiprocess: rank \\d+ raised:"):
        gloo_multiprocess(4, _stage0_target_rank_0_fails_in_manifest_write, "/tmp")


def _stage0_target_rank_0_fails_in_bos_guard(
    rank: int, world_size: int, run_root: str,
):
    """Simulate rank 0 raising in assert_bos_lengths_match — the
    precise scenario from the 2026-05-13 incident, except now executed
    AFTER barrier 1 instead of BEFORE."""
    from unittest.mock import MagicMock

    import scripts.eval.axbench_bpb_eval as eval_mod
    from scripts.eval.axbench_bpb_eval import _stage_prepare_context_cache
    from steering.eval.axbench_bpb.types import AxBenchBpbError
    from tests.axbench_bpb._mini_runtime import build_mini_runtime_state

    rs = build_mini_runtime_state(
        run_root=Path(run_root), rank=rank, world_size=world_size,
    )
    fake_stats = MagicMock(per_feature_stats=MagicMock())

    def fake_assert_bos_raises(**kwargs):
        raise AxBenchBpbError("simulated BOS-length mismatch on rank 0")

    original_build = eval_mod.build_context_cache
    original_load_stats = eval_mod.load_statistics
    original_build_boundaries = eval_mod.build_boundaries_from_stats
    original_assert_tok = eval_mod.assert_tokenizer_checksum_matches
    original_assert_bos = eval_mod.assert_bos_lengths_match
    original_collect = eval_mod.collect_unique_prefixes
    original_write = eval_mod.write_context_cache_manifest
    set_runtime_attribute(
        eval_mod,
        "build_context_cache",
        lambda **_kw: None,
    )
    set_runtime_attribute(
        eval_mod,
        "load_statistics",
        lambda path: fake_stats,
    )
    set_runtime_attribute(
        eval_mod,
        "build_boundaries_from_stats",
        lambda *a, **kw: MagicMock(),
    )
    set_runtime_attribute(
        eval_mod,
        "assert_tokenizer_checksum_matches",
        lambda **_kw: None,
    )
    set_runtime_attribute(
        eval_mod,
        "assert_bos_lengths_match",
        fake_assert_bos_raises,
    )
    set_runtime_attribute(
        eval_mod,
        "collect_unique_prefixes",
        lambda *_a, **_kw: {},
    )
    set_runtime_attribute(
        eval_mod,
        "write_context_cache_manifest",
        lambda **_kw: None,
    )
    try:
        _stage_prepare_context_cache(rs)
    finally:
        eval_mod.build_context_cache = original_build
        eval_mod.load_statistics = original_load_stats
        eval_mod.build_boundaries_from_stats = original_build_boundaries
        eval_mod.assert_tokenizer_checksum_matches = original_assert_tok
        eval_mod.assert_bos_lengths_match = original_assert_bos
        eval_mod.collect_unique_prefixes = original_collect
        eval_mod.write_context_cache_manifest = original_write


def test_stage0_rank_0_bos_guard_failure_does_not_deadlock(gloo_multiprocess):
    """Regression test for commit 32cc48db. Rank 0 raises in
    assert_bos_lengths_match after the manifest write — the same
    failure shape as the 2026-05-13 incident, now executed
    post-manifest.

    What this test verifies (precisely): the gloo fixture surfaces
    the rank-0 exception rather than hanging the test, regardless of
    the production barrier topology.

    What this test does NOT verify: the production barrier-1
    ordering. See the equivalent caveat in
    ``test_stage0_rank_0_manifest_write_failure_does_not_deadlock``;
    barrier ordering is pinned by the recorder-based unit test in
    ``test_ddp_aggregation_barrier.py``.

    Note: under NCCL in production, ranks 1+ would still block at
    barrier 2 for the NCCL watchdog timeout (default 10 min) because
    destroy_process_group on rank 0 doesn't broadcast an abort. Set
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 to fail-fast in production.
    """
    with pytest.raises(RuntimeError, match="gloo_multiprocess: rank \\d+ raised:"):
        gloo_multiprocess(4, _stage0_target_rank_0_fails_in_bos_guard, "/tmp")
