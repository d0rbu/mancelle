"""Smoke tests for the gloo_multiprocess fixture."""

from __future__ import annotations

import time

import pytest
import torch.distributed as dist


def _all_gather_target(rank: int, world_size: int, payload):
    """Worker that gathers `payload` across ranks and returns the full list."""
    gathered: list = [None] * world_size
    dist.all_gather_object(gathered, (rank, payload[rank]))
    return gathered


def _barrier_target(rank: int, world_size: int):
    """Worker that just barriers and returns the rank that ran."""
    dist.barrier()
    return rank


def _raise_on_rank_1(rank: int, world_size: int):
    if rank == 1:
        raise RuntimeError(f"intentional failure on rank {rank}")


def _hang_target(rank: int, world_size: int):
    """Worker that sleeps long enough for the parent's deadline to fire."""
    time.sleep(60)


def test_gloo_multiprocess_all_gather_object_two_ranks(gloo_multiprocess):
    """Spawning 2 ranks via gloo_multiprocess and running all_gather_object
    returns each rank's payload in rank order."""
    payloads = ["alpha", "beta"]
    results = gloo_multiprocess(2, _all_gather_target, payloads)
    assert len(results) == 2
    for rank_result in results:
        assert len(rank_result) == 2
    all_pairs = {pair for rank_result in results for pair in rank_result}
    assert all_pairs == {(0, "alpha"), (1, "beta")}


def test_gloo_multiprocess_all_gather_object_four_ranks(gloo_multiprocess):
    payloads = ["a", "b", "c", "d"]
    results = gloo_multiprocess(4, _all_gather_target, payloads)
    assert len(results) == 4
    all_pairs = {pair for rank_result in results for pair in rank_result}
    assert all_pairs == {(0, "a"), (1, "b"), (2, "c"), (3, "d")}


def test_gloo_multiprocess_barrier_smoke(gloo_multiprocess):
    """All ranks pass the barrier and return their rank id."""
    results = gloo_multiprocess(3, _barrier_target)
    assert sorted(results) == [0, 1, 2]


def test_gloo_multiprocess_propagates_worker_exception(gloo_multiprocess):
    """An exception in any worker surfaces to the parent."""
    with pytest.raises(RuntimeError, match="intentional failure on rank 1"):
        gloo_multiprocess(2, _raise_on_rank_1)


def test_gloo_multiprocess_worker_hang_raises_clear_error(
    gloo_multiprocess, monkeypatch,
):
    """A hung worker (never calls put) raises a clear timeout error
    rather than blocking the test forever.

    Uses a 5s deadline (rather than the previous 3s) so the test does
    not falsely pass on slow CI runners where ``mp.spawn`` startup
    alone can approach 3s — in that case the timeout would fire
    before the worker actually hung, yielding a green test that
    would not catch a regression where the worker DID return on
    time. Asserting the elapsed wall-time exceeded ``hung_for_at_least_s``
    confirms we hit the timeout because the worker hung, not because
    spawn itself was slow.
    """
    import time

    import tests.axbench_base._multiprocess_helpers as h
    monkeypatch.setattr(h, "_RESULT_TIMEOUT_S", 5)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="did not report results within"):
        gloo_multiprocess(2, _hang_target)
    elapsed = time.monotonic() - started

    # Lower bound: at minimum the deadline must have fired; if the test
    # returned in <2s, we are not actually testing the hang path.
    assert elapsed >= 2.0, (
        f"hang test returned in {elapsed:.2f}s — deadline did not fire, "
        f"so the test is not actually exercising the hang path"
    )
