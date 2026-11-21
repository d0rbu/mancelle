"""broadcast_bitmap_from_rank_zero: rank-once work with aligned failure.

Mirrors tests/scripts/test_encdec_sft_ddp_barrier.py's spawn+file-init gloo
harness. The success test proves the compute ran EXACTLY once (rank 0) and
every rank received an equal bitmap; the failure tests prove a rank-0
exception — or a result violating the bitmap contract — aborts both ranks
with AlignedRankFailure (no hang).
"""

from __future__ import annotations

import pytest
import torch as t
import torch.distributed as dist

import steering._common.ddp as ddp_module
from steering._common.ddp import (
    AlignedRankFailure,
    broadcast_bitmap_from_rank_zero,
)
from tests.gloo_two_rank_helpers import run_two_ranks


def test_single_process_calls_compute_directly() -> None:
    calls: list[int] = []

    def compute() -> t.Tensor:
        calls.append(1)
        bitmap = t.zeros(64, dtype=t.bool)
        bitmap[[3, 17]] = True
        return bitmap

    result = broadcast_bitmap_from_rank_zero(compute, numel=64, device="cpu", stage="test stage")
    assert calls == [1]
    assert result.dtype is t.bool
    assert result.nonzero().squeeze(1).tolist() == [3, 17]


def test_single_process_propagates_compute_error_unwrapped() -> None:
    def compute() -> t.Tensor:
        raise ValueError("scan exploded")

    with pytest.raises(ValueError, match="scan exploded"):
        broadcast_bitmap_from_rank_zero(compute, numel=64, device="cpu", stage="test stage")


def test_single_process_rejects_a_noncontract_result() -> None:
    with pytest.raises(ValueError, match=r"bool\[64\]"):
        broadcast_bitmap_from_rank_zero(
            lambda: t.zeros(10, dtype=t.bool),
            numel=64,
            device="cpu",
            stage="test stage",
        )
    with pytest.raises(ValueError, match=r"bool\[64\]"):
        broadcast_bitmap_from_rank_zero(
            lambda: t.zeros(64, dtype=t.uint8),
            numel=64,
            device="cpu",
            stage="test stage",
        )


def _gloo_success_worker(rank, world_size, init_file, result_queue) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        calls: list[int] = []

        def compute() -> t.Tensor:
            calls.append(1)
            bitmap = t.zeros(64, dtype=t.bool)
            bitmap[[3, 17, 42]] = True
            return bitmap

        result = broadcast_bitmap_from_rank_zero(
            compute, numel=64, device="cpu", stage="sft_percentile_id_scan"
        )
        result_queue.put((rank, len(calls), result.nonzero().squeeze(1).tolist()))
    except BaseException as exc:  # noqa: BLE001 - report child failure
        result_queue.put((rank, -1, f"{type(exc).__name__}: {exc}"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _gloo_rank0_failure_worker(rank, world_size, init_file, result_queue) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )

        def compute() -> t.Tensor:
            raise OSError("cannot read shard shard_00003")

        try:
            broadcast_bitmap_from_rank_zero(
                compute, numel=64, device="cpu", stage="sft_percentile_id_scan"
            )
        except AlignedRankFailure as exc:
            result_queue.put((rank, str(exc)))
        else:
            result_queue.put((rank, "survived"))
    except BaseException as exc:  # noqa: BLE001 - report child failure
        result_queue.put((rank, f"{type(exc).__name__}: {exc}"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed Gloo is unavailable",
)
def test_two_ranks_compute_runs_once_and_result_is_broadcast() -> None:
    results = run_two_ranks(_gloo_success_worker, "rank_zero_success")
    assert results[0] == (1, [3, 17, 42]), results
    assert results[1] == (0, [3, 17, 42]), results


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed Gloo is unavailable",
)
def test_two_ranks_rank0_failure_aborts_both_without_hang() -> None:
    results = run_two_ranks(_gloo_rank0_failure_worker, "rank_zero_failure")
    for rank in (0, 1):
        message = results[rank][0]
        assert "sft_percentile_id_scan" in message, message
        assert "rank 0" in message, message
        assert "cannot read shard shard_00003" in message, message


def _gloo_wrong_shape_result_worker(rank, world_size, init_file, result_queue) -> None:
    """A rank-0 result violating the bitmap contract must become an ALIGNED
    failure BEFORE the fixed-size broadcast, never a peer stuck inside it."""
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        try:
            broadcast_bitmap_from_rank_zero(
                lambda: t.zeros(10, dtype=t.bool),
                numel=64,
                device="cpu",
                stage="sft_percentile_id_scan",
            )
        except AlignedRankFailure as exc:
            result_queue.put((rank, str(exc)))
        else:
            result_queue.put((rank, "survived"))
    except BaseException as exc:  # noqa: BLE001 - report child failure
        result_queue.put((rank, f"{type(exc).__name__}: {exc}"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed Gloo is unavailable",
)
def test_two_ranks_wrong_shape_result_fails_aligned() -> None:
    results = run_two_ranks(_gloo_wrong_shape_result_worker, "rank_zero_wrong_shape")
    for rank in (0, 1):
        message = results[rank][0]
        assert "sft_percentile_id_scan" in message, message
        assert "rank 0" in message, message


class _WireAllocFailsTorchProxy:
    """Delegates to torch, except the wire-buffer allocation fails."""

    def __getattr__(self, name):
        if name == "empty":

            def failing_empty(*args, **kwargs):
                raise RuntimeError("cannot allocate wire buffer")

            return failing_empty
        return getattr(t, name)


def _gloo_rank1_wire_alloc_failure_worker(rank, world_size, init_file, result_queue) -> None:
    """A NON-zero rank whose wire-buffer preparation fails must abort BOTH
    ranks aligned — never leave rank 0 blocked in the broadcast. The wire is
    prepared inside the guarded region BEFORE the aligned sync, so this
    failure participates in the failure all-reduce like any other."""
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        if rank == 1:
            # Direct module-attribute patch: monkeypatch is unavailable inside
            # a spawned gloo worker; the child process isolates the mutation.
            ddp_module.t = _WireAllocFailsTorchProxy()  # ty: ignore[invalid-assignment]
        try:
            broadcast_bitmap_from_rank_zero(
                lambda: t.zeros(64, dtype=t.bool),
                numel=64,
                device="cpu",
                stage="sft_percentile_id_scan",
            )
        except AlignedRankFailure as exc:
            result_queue.put((rank, str(exc)))
        else:
            result_queue.put((rank, "survived"))
    except BaseException as exc:  # noqa: BLE001 - report child failure
        result_queue.put((rank, f"{type(exc).__name__}: {exc}"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed Gloo is unavailable",
)
def test_two_ranks_wire_allocation_failure_fails_aligned() -> None:
    results = run_two_ranks(_gloo_rank1_wire_alloc_failure_worker, "rank_one_wire_alloc")
    for rank in (0, 1):
        message = results[rank][0]
        assert "sft_percentile_id_scan" in message, message
        assert "rank 1" in message, message
        assert "cannot allocate wire buffer" in message, message
