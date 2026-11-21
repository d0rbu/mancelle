"""Real-multiprocess test helpers using gloo backend.

Spawns N processes via torch.multiprocessing, initializes a gloo
process group, runs a target callable in each, and collects results
back to the parent. Worker exceptions are surfaced to the parent.

Used by Stage 0c multi-GPU tests (PR-4) and BPB DDP audit tests (PR-3).
CPU-only — does not require any GPU in CI.
"""

from __future__ import annotations

import os
import queue as _queue_mod
import socket
import time
import traceback
from collections.abc import Callable
from typing import Any

import torch.distributed as dist
import torch.multiprocessing as mp

# Per-call upper bound on how long the parent waits for ALL workers to
# post their (ok, result) or (err, ...) tuples to the result queue. A
# worker that hangs (never posts) would otherwise block the test forever;
# the timeout surfaces a clear error instead. 60s is generous for normal
# CI; tests that intentionally hang a worker monkey-patch this lower.
_RESULT_TIMEOUT_S = 60


def _find_free_port() -> int:
    """Find an unused TCP port for the gloo MASTER_PORT."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker_main(
    rank: int,
    world_size: int,
    master_port: int,
    target: Callable[..., Any],
    args: tuple,
    result_queue: Any,
) -> None:
    """Worker process entrypoint. Initialises gloo, runs `target`, posts
    `(rank, result_or_exception)` to `result_queue`, then destroys the
    process group."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    try:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        try:
            result = target(rank, world_size, *args)
            result_queue.put((rank, ("ok", result)))
        finally:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
    except BaseException as exc:
        tb = traceback.format_exc()
        result_queue.put((rank, ("err", repr(exc), tb)))


def run_gloo_multiprocess(
    world_size: int,
    target: Callable[..., Any],
    *args: Any,
) -> list[Any]:
    """Spawn ``world_size`` workers, run ``target(rank, world_size, *args)``
    in each, return a list of results indexed by rank.

    Raises a RuntimeError that includes the first worker's traceback
    when any worker fails. Raises a RuntimeError with a clear timeout
    message if any worker fails to post a result within
    ``_RESULT_TIMEOUT_S`` seconds (avoids the test blocking forever on
    a hung worker).
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    master_port = _find_free_port()
    procs: list[Any] = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_worker_main,
            args=(rank, world_size, master_port, target, args, queue),
        )
        p.start()
        procs.append(p)

    results_by_rank: dict[int, Any] = {}
    errors: list[tuple[int, str, str]] = []
    deadline = time.monotonic() + _RESULT_TIMEOUT_S
    for _ in range(world_size):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            for p in procs:
                if p.is_alive():
                    p.terminate()
            raise RuntimeError(
                f"gloo_multiprocess: worker(s) did not report results within "
                f"{_RESULT_TIMEOUT_S}s; received "
                f"{len(results_by_rank)}/{world_size}. "
                "This usually indicates a worker hang in the user-supplied target."
            )
        try:
            rank, result = queue.get(timeout=remaining)
        except _queue_mod.Empty:
            # Most common cause of an Empty after the per-call deadline
            # is a hung worker that never posted a result. Surface a
            # clear timeout error rather than re-iterating and blocking
            # on the next get() call.
            for p in procs:
                if p.is_alive():
                    p.terminate()
            raise RuntimeError(
                f"gloo_multiprocess: worker(s) did not report results within "
                f"{_RESULT_TIMEOUT_S}s; received "
                f"{len(results_by_rank)}/{world_size}. "
                "This usually indicates a worker hang in the user-supplied target."
            ) from None
        except Exception as exc:
            for p in procs:
                if p.is_alive():
                    p.terminate()
            raise RuntimeError(
                f"gloo_multiprocess: result queue read failed: {exc}"
            ) from exc
        if result[0] == "ok":
            results_by_rank[rank] = result[1]
        else:
            errors.append((rank, result[1], result[2]))

    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)

    if errors:
        rank, _repr, tb = errors[0]
        raise RuntimeError(
            f"gloo_multiprocess: rank {rank} raised:\n{tb}"
        )

    return [results_by_rank[r] for r in range(world_size)]


__all__ = ["run_gloo_multiprocess"]
