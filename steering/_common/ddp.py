"""Shared aligned DDP rendezvous helpers for training scripts."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import timedelta
from typing import Any

import torch as t
import torch.distributed as dist
from loguru import logger
from torch.distributed.constants import default_pg_nccl_timeout

from nanochat.common import (
    DeviceType,
    _resolve_nccl_timeout_seconds,
    compute_init,
    get_dist_info,
    is_ddp_initialized,
    is_ddp_requested,
)


class AlignedRankFailure(RuntimeError):
    """A peer rank failed an aligned stage, so every rank aborts together."""


def compute_init_for_checkpoint_resume(
    device_type: str,
    *,
    resume_step: int,
) -> tuple[bool, int, int, int, t.device]:
    """Give resumed CUDA DDP a fresh store namespace after process exec."""
    if resume_step <= 0 or device_type != DeviceType.CUDA or not is_ddp_requested():
        return compute_init(device_type)

    assert t.cuda.is_available(), (
        "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
    )
    t.manual_seed(42)
    t.cuda.manual_seed(42)
    t.set_float32_matmul_precision("high")
    ddp, rank, local_rank, world_size = get_dist_info()
    assert ddp
    device = t.device("cuda", local_rank)
    t.cuda.set_device(device)

    timeout_seconds = _resolve_nccl_timeout_seconds()
    rendezvous_timeout = (
        default_pg_nccl_timeout
        if timeout_seconds is None
        else timedelta(seconds=timeout_seconds)
    )
    store, _, _ = next(dist.rendezvous("env://", timeout=rendezvous_timeout))
    store.set_timeout(rendezvous_timeout)
    generation = os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")
    store = dist.PrefixStore(
        f"checkpoint_resume_step_{resume_step}_restart_{generation}",
        store,
    )
    if timeout_seconds is None:
        dist.init_process_group(
            backend="nccl",
            store=store,
            rank=rank,
            world_size=world_size,
            device_id=device,
        )
    else:
        dist.init_process_group(
            backend="nccl",
            store=store,
            rank=rank,
            world_size=world_size,
            device_id=device,
            timeout=rendezvous_timeout,
        )
    dist.barrier()
    if rank == 0:
        logger.info(f"Distributed world size: {world_size}")
    return ddp, rank, local_rank, world_size, device


def ddp_barrier() -> None:
    """A collective barrier that no-ops in single-process / CPU runs.

    Passed as ``barrier_fn`` to
    ``steering.conditional.freq_blacklist.generate_or_load_blacklist`` so the
    pure library stays ``torch.distributed``-free (and unit-testable) while
    the scripts inject the real rendezvous.
    """
    if is_ddp_initialized():
        dist.barrier()


def _sync_aligned_failure(
    local_error: BaseException | None,
    *,
    device: str | t.device,
    stage: str,
    dist_module: Any = dist,
) -> None:
    """Propagate the lowest-rank failure from an aligned stage to every rank.

    All ranks must call this function in the same order. The lowest failing
    rank broadcasts the originating exception text so peers fail promptly
    instead of entering a later optimizer or evaluation collective.
    """
    if not dist_module.is_initialized() or dist_module.get_world_size() <= 1:
        if local_error is not None:
            raise local_error
        return

    world_size = dist_module.get_world_size()
    rank = dist_module.get_rank()
    marker = t.tensor(
        rank if local_error is not None else world_size,
        dtype=t.int32,
        device=device,
    )
    dist_module.all_reduce(marker, op=dist_module.ReduceOp.MIN)
    first_failing = int(marker.item())
    if first_failing == world_size:
        return

    payload: list[Any] = [
        f"{type(local_error).__name__}: {local_error}" if rank == first_failing else None
    ]
    dist_module.broadcast_object_list(
        payload,
        src=first_failing,
        device=t.device(device),
    )
    detail = payload[0]
    if local_error is not None:
        raise AlignedRankFailure(
            f"aligned stage {stage!r} failed on rank {rank} "
            f"(first failing rank {first_failing}): {detail}. Every rank aborts "
            "before entering another collective."
        ) from local_error
    raise AlignedRankFailure(
        f"aligned stage {stage!r} failed on rank {first_failing}: {detail}. "
        "This rank passed the stage and is aborting in lockstep."
    )


def _require_bitmap(result: t.Tensor, *, numel: int, stage: str) -> t.Tensor:
    """The broadcast payload contract: a 1-D bool tensor of exactly ``numel``."""
    if isinstance(result, t.Tensor) and result.dtype is t.bool and result.shape == (numel,):
        return result
    described = (
        f"{result.dtype}[{tuple(result.shape)}]"
        if isinstance(result, t.Tensor)
        else type(result).__name__
    )
    raise ValueError(
        f"aligned stage {stage!r} rank-zero compute must return a "
        f"bool[{numel}] bitmap, got {described}"
    )


def broadcast_bitmap_from_rank_zero(
    compute: Callable[[], t.Tensor],
    *,
    numel: int,
    device: str | t.device,
    stage: str,
    dist_module: Any = dist,
) -> t.Tensor:
    """Run ``compute`` on rank 0 only and broadcast its bool bitmap to every rank.

    Deadlock-safe by construction: every rank prepares its side of the
    fixed-size uint8 wire buffer INSIDE a guarded region (rank 0 also runs
    ``compute`` and validates the ``bool[numel]`` contract there), then all
    ranks run :func:`_sync_aligned_failure` — so any failure raises
    :class:`AlignedRankFailure` on every rank instead of leaving a peer
    blocked in the result broadcast, which is the only step after the sync
    and whose payload shape every rank already knows from ``numel``.
    ``Exception`` (not ``BaseException``) is caught, deliberately matching
    :func:`rank_synchronized_stage`. Single-process / uninitialized
    ``torch.distributed`` calls ``compute`` directly (with the same result
    validation) and re-raises its exceptions unwrapped.
    """
    if not dist_module.is_initialized() or dist_module.get_world_size() <= 1:
        return _require_bitmap(compute(), numel=numel, stage=stage)
    rank = dist_module.get_rank()
    local_error: BaseException | None = None
    wire: t.Tensor | None = None
    try:
        if rank == 0:
            bitmap = _require_bitmap(compute(), numel=numel, stage=stage)
            wire = bitmap.to(t.uint8).to(t.device(device))
        else:
            wire = t.empty(numel, dtype=t.uint8, device=t.device(device))
    except Exception as exc:  # noqa: BLE001 - synchronized and re-raised below
        local_error = exc
    _sync_aligned_failure(local_error, device=device, stage=stage, dist_module=dist_module)
    assert wire is not None  # success path: prepared above on every rank
    dist_module.broadcast(wire, src=0)
    return wire.to("cpu").to(t.bool)


@contextmanager
def rank_synchronized_stage(
    stage: str,
    *,
    device: str | t.device,
    dist_module: Any = dist,
) -> Iterator[None]:
    """Run an aligned stage and make any local exception an all-rank failure."""
    local_error: BaseException | None = None
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - synchronized and re-raised below
        local_error = exc
    _sync_aligned_failure(
        local_error,
        device=device,
        stage=stage,
        dist_module=dist_module,
    )


def sync_stop_and_progress(
    *,
    last_step: bool,
    progress: float,
    device: str | t.device,
) -> tuple[bool, float]:
    """Synchronize an SFT loop boundary so every rank uses the same LR."""
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return last_step, progress
    state = t.tensor(
        [1.0 if last_step else 0.0, progress],
        dtype=t.float64,
        device=device,
    )
    dist.all_reduce(state, op=dist.ReduceOp.MAX)
    return bool(state[0].item()), min(1.0, max(0.0, float(state[1].item())))
