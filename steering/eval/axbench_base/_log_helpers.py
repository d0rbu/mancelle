"""Rank-aware logging helpers for AxBench-Base instrumentation.

The standard rank-aware loguru sink is configured in
``nanochat/common.py`` and gates non-zero ranks at WARNING. When
Stage 0c (or any axbench_base entry point) is invoked off the
standard CLI path that sink may not be installed, in which case the
new heartbeat / progress ``logger.info`` calls would flood stderr
from every rank. The :func:`is_log_rank` helper provides a defensive
guard so callers can short-circuit log emission on non-rank-0
processes regardless of the sink configuration.

:class:`StepwiseTimer` extracts the manual ``time.monotonic()`` /
``int(... * 1000)`` dance previously crowding the body of
``_load_feature_extraction_pair`` into a small reusable helper.
"""

from __future__ import annotations

import os
import time
from typing import Any

from loguru import logger


def is_log_rank() -> bool:
    """Return True iff this process should emit user-facing INFO logs.

    True when ``RANK`` is unset (single-process run) or parses as 0.
    Malformed values (empty string, non-integer) return False — better
    silent than spam from a process whose rank we cannot prove is 0.
    """
    raw = os.environ.get("RANK")
    if raw is None:
        return True
    try:
        return int(raw) == 0
    except ValueError:
        return False


class StepwiseTimer:
    """Step-wise timing helper that emits one log line per completed step.

    Usage mirrors the manual ``_t_start`` / ``_t_resolve`` / ... pattern
    previously open-coded in :func:`_load_feature_extraction_pair`::

        timer = StepwiseTimer()
        logger.info("[runtime_state] resolving Neuronpedia source")
        parse_neuronpedia_source(...)
        timer.step("[runtime_state] resolving SAE release/sae_id")
        resolve_sae(...)
        timer.step("[runtime_state] loading TransformerLens base model + SAE",
                   device=cfg.device)
        ...
        logger.info("[runtime_state] feature-extraction pair ready "
                    "[total_elapsed_ms={ms}]", ms=timer.total_ms())

    Each ``step()`` call appends ``[elapsed_ms=<ms>]`` to the supplied
    message and emits at INFO level on rank 0 (per :func:`is_log_rank`).
    Internal mark advances on every call, even when emission is
    suppressed, so :meth:`total_ms` always reports true wall time.
    """

    def __init__(self) -> None:
        self._start = time.monotonic()
        self._mark = self._start

    def step(self, message: str, **kwargs: Any) -> None:
        """Log ``message + " [elapsed_ms=X]"`` for the just-completed step.

        ``kwargs`` are forwarded to :func:`loguru.logger.info` for
        format-string substitution. The reserved key
        ``_step_elapsed_ms`` is used internally to interpolate the
        elapsed time and must not be supplied by the caller.
        """
        now = time.monotonic()
        elapsed_ms = int((now - self._mark) * 1000)
        self._mark = now
        if is_log_rank():
            logger.info(
                message + " [elapsed_ms={_step_elapsed_ms}]",
                _step_elapsed_ms=elapsed_ms,
                **kwargs,
            )

    def total_ms(self) -> int:
        """Wall-clock since construction, in integer milliseconds."""
        return int((time.monotonic() - self._start) * 1000)


__all__ = ["StepwiseTimer", "is_log_rank"]
