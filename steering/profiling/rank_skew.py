"""Cross-rank duration skew reporting for checkpoint I/O.

The NCCL watchdog kills a job when the SLOWEST participant of a collective
lags the fastest by more than the timeout — during the 2026-08-28 hstor013
incidents that skew came from per-rank checkpoint reads/writes and was
invisible until the watchdog fired.  ``log_rank_duration_skew`` gathers one
duration from every rank and logs min/median/max plus the slowest rank on
rank 0, warning when the spread approaches watchdog territory.

The gather is itself a collective and therefore a synchronization point; for
checkpoint save/load that is intentional — the very next training collective
would synchronize the ranks anyway, so this only moves the wait to where it
is measured.
"""

from __future__ import annotations

import statistics

import torch.distributed as dist
from loguru import logger

RANK_SKEW_WARN_S = 300.0


def log_rank_duration_skew(
    label: str,
    seconds: float,
    *,
    rank: int,
    world_size: int,
    skew_warn_s: float = RANK_SKEW_WARN_S,
) -> dict[str, float]:
    """Gather ``seconds`` across ranks; log the spread on rank 0.

    Returns the spread statistics on every rank (each rank computes them from
    the same gathered list).  With ``world_size <= 1`` or no initialized
    process group, logs the solo duration and returns degenerate stats.
    """
    if world_size <= 1 or not dist.is_available() or not dist.is_initialized():
        logger.info(f"{label}: {seconds:.1f}s (single rank)")
        return {
            "min_s": seconds,
            "median_s": seconds,
            "max_s": seconds,
            "skew_s": 0.0,
            "slowest_rank": rank,
        }

    durations: list[float | None] = [None] * world_size
    dist.all_gather_object(durations, float(seconds))
    gathered = [float(d) for d in durations if d is not None]
    slowest_rank = max(range(len(gathered)), key=gathered.__getitem__)
    stats = {
        "min_s": min(gathered),
        "median_s": statistics.median(gathered),
        "max_s": max(gathered),
        "skew_s": max(gathered) - min(gathered),
        "slowest_rank": slowest_rank,
    }
    if rank == 0:
        line = (
            f"{label}: rank duration min/median/max = "
            f"{stats['min_s']:.1f}/{stats['median_s']:.1f}/{stats['max_s']:.1f}s "
            f"(slowest rank {slowest_rank}, skew {stats['skew_s']:.1f}s)"
        )
        if stats["skew_s"] > skew_warn_s:
            logger.warning(
                f"{line} — skew above {skew_warn_s:.0f}s risks the NCCL "
                "watchdog on the next collective (storage degradation likely)"
            )
        else:
            logger.info(line)
    return stats
