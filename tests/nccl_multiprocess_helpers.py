"""Single-node spawned NCCL harness for opt-in production-backend tests."""

from __future__ import annotations

import os
import queue as queue_module
import socket
import time
import traceback
from collections.abc import Callable
from typing import Any

import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp

_RESULT_TIMEOUT_S = 900


def _find_free_port() -> int:
    """Return an ephemeral localhost port for the short-lived test rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _worker_main(
    rank: int,
    world_size: int,
    master_port: int,
    target: Callable[..., Any],
    args: tuple[Any, ...],
    result_queue: Any,
) -> None:
    """Initialize one production-backend rank and return its result or traceback."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    result: tuple[str, Any]
    try:
        t.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        result = ("ok", target(rank, world_size, *args))
    except BaseException:  # noqa: BLE001 - the parent needs the complete child failure
        result = ("error", traceback.format_exc())
    finally:
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except BaseException:  # noqa: BLE001 - teardown failure is a test failure
            result = ("error", traceback.format_exc())
    result_queue.put((rank, result))


def _join_or_kill_process(process: Any) -> None:
    """Reap one worker, escalating past a compile worker that ignores SIGTERM."""

    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)


def run_nccl_multiprocess(
    world_size: int,
    target: Callable[..., Any],
    *args: Any,
) -> list[Any]:
    """Run ``target(rank, world_size, *args)`` in local NCCL workers.

    This helper is intentionally single-node: rank ``r`` owns ``cuda:r``. It is
    for opt-in two-device correctness gates, not multi-node performance runs.
    Worker failures retain their full tracebacks, and one common deadline plus
    unconditional child cleanup prevents a collective hang from blocking pytest.
    """
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if not callable(target):
        raise TypeError(f"target must be callable, got {type(target).__name__}")
    if not t.cuda.is_available() or t.cuda.device_count() < world_size:
        raise RuntimeError(
            f"NCCL test requires {world_size} local CUDA devices, found {t.cuda.device_count()}"
        )
    if not dist.is_nccl_available():
        raise RuntimeError("PyTorch NCCL backend is unavailable")

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    master_port = _find_free_port()
    processes = [
        context.Process(
            target=_worker_main,
            args=(rank, world_size, master_port, target, args, result_queue),
        )
        for rank in range(world_size)
    ]
    results_by_rank: dict[int, Any] = {}
    errors: list[tuple[int, str]] = []
    reported_ranks: set[int] = set()
    started_processes: list[Any] = []
    deadline = time.monotonic() + _RESULT_TIMEOUT_S
    try:
        for process in processes:
            process.start()
            started_processes.append(process)
        for _ in range(world_size):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    "nccl_multiprocess: workers did not report within "
                    f"{_RESULT_TIMEOUT_S}s; received "
                    f"{len(reported_ranks)}/{world_size}"
                )
            try:
                rank, (status, payload) = result_queue.get(timeout=remaining)
            except queue_module.Empty:
                raise RuntimeError(
                    "nccl_multiprocess: workers did not report within "
                    f"{_RESULT_TIMEOUT_S}s; received "
                    f"{len(reported_ranks)}/{world_size}"
                ) from None
            if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < world_size:
                raise RuntimeError(
                    f"nccl_multiprocess: worker reported invalid rank {rank!r}"
                )
            if rank in reported_ranks:
                raise RuntimeError(
                    f"nccl_multiprocess: worker rank {rank} reported more than once"
                )
            reported_ranks.add(rank)
            if status == "ok":
                results_by_rank[rank] = payload
            elif status == "error":
                errors.append((rank, payload))
            else:
                raise RuntimeError(
                    f"nccl_multiprocess: rank {rank} reported invalid status {status!r}"
                )
    finally:
        for process in started_processes:
            _join_or_kill_process(process)

    if errors:
        rank, child_traceback = min(errors, key=lambda item: item[0])
        raise RuntimeError(f"nccl_multiprocess: rank {rank} raised:\n{child_traceback}")
    exit_codes = [process.exitcode for process in processes]
    if exit_codes != [0] * world_size:
        raise RuntimeError(
            f"nccl_multiprocess: worker exit codes were {exit_codes}, expected {[0] * world_size}"
        )
    missing_ranks = sorted(set(range(world_size)) - set(results_by_rank))
    if missing_ranks:
        raise RuntimeError(f"nccl_multiprocess: missing results for ranks {missing_ranks}")
    return [results_by_rank[rank] for rank in range(world_size)]


__all__ = ["run_nccl_multiprocess"]
