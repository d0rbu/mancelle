"""Aligned batch-construction failures shared by both SFT trainers."""

from __future__ import annotations

import multiprocessing
import tempfile
from pathlib import Path

import pytest
import torch.distributed as dist

from steering._common.ddp import AlignedRankFailure, rank_synchronized_stage


def _failing_batches():
    raise ValueError("bad local batch")
    yield None


def test_batch_stage_single_process_returns_batch() -> None:
    batch = object()
    with rank_synchronized_stage("test batch construction", device="cpu"):
        result = next(iter([batch]))
    assert result is batch


def test_batch_stage_single_process_preserves_local_error() -> None:
    with pytest.raises(ValueError, match="bad local batch"):
        with rank_synchronized_stage("test batch construction", device="cpu"):
            next(_failing_batches())


def _gloo_batch_failure_worker(
    rank: int,
    world_size: int,
    init_file: str,
    result_queue,
) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        batches = _failing_batches() if rank == 1 else iter([object()])
        try:
            with rank_synchronized_stage(
                "conditioned SFT training batch construction",
                device="cpu",
            ):
                next(batches)
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
def test_one_rank_batch_failure_aborts_every_rank() -> None:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    with tempfile.TemporaryDirectory() as temp_dir:
        init_file = str(Path(temp_dir) / "gloo_batch_failure")
        processes = [
            context.Process(
                target=_gloo_batch_failure_worker,
                args=(rank, 2, init_file, result_queue),
            )
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        results = [result_queue.get(timeout=60) for _ in processes]
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)

    assert [process.exitcode for process in processes] == [0, 0]
    messages = dict(results)
    assert set(messages) == {0, 1}
    assert all(
        "conditioned SFT training batch construction" in message for message in messages.values()
    )
    assert all("rank 1" in message for message in messages.values())
    assert all("bad local batch" in message for message in messages.values())
