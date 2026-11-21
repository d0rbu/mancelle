"""Progress + ETA reporting for long-running BPB-Adv stages.

Extracted from ``gcg_bags`` when the score stage needed the same thing. Both
surfaces run for HOURS on a real clustered run -- the feature-GCG derives at
~7 min per fair group over 100 groups, the score stage at ~1.9 h for 160k attack
rows -- and both used to log one line at the start and nothing again until they
finished. A slow run and a hung run were indistinguishable from the log.

The reuse/work split is the part worth keeping in one place. Both stages skip
work they have already done (the derives via committed-bag reuse, the scorer via
its per-scoring-identity memo), and those skips cost ~0. Folding them into the
rate makes the run look arbitrarily fast and projects a finish that never
arrives -- exactly when the ETA matters most.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from loguru import logger


class StageProgress:
    """Per-unit progress + ETA for a long-running stage.

    Used by the feature-GCG derives (100 fair groups at ~7 min each, plus ~693
    roofline bags) and by the score stage (160k rows in ~1.9 h). Both used to log
    one line at the start and nothing again until they finished, so a slow run
    and a hung run were indistinguishable and neither could be sized in advance.

    Reuse is counted separately from work. Both stages skip work they have
    already done -- the derives via ``committed_bag_for_reuse``, the scorer via
    its per-scoring-identity memo -- and those skips cost ~0. Folding them into
    the rate makes the run look arbitrarily fast and projects a finish that never
    arrives, which is worse than no ETA because it looks authoritative.

    ``log_every`` throttles OUTPUT only; every unit is always counted, so the
    rate and ETA are unaffected by how often they are printed. A derive wants
    every unit (they are minutes apart); a 160k-row scorer does not.
    """

    def __init__(
        self,
        label: str,
        total: int,
        unit: str = "group",
        *,
        log_every: int = 1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        # `clock` is injectable so the rate/ETA arithmetic is testable without
        # sleeping: a real-time test completes in ~0s, which makes every ETA
        # format as "0s" and leaves the projection itself unverified.
        self._label = label
        self._total = total
        self._unit = unit
        self._clock = clock
        self._log_every = max(int(log_every), 1)
        self._started = clock()
        self._done = 0
        self._reused = 0
        self._worked = 0

    def record(self, *, reused: bool) -> None:
        self._done += 1
        if reused:
            self._reused += 1
        else:
            self._worked += 1
        if self._done % self._log_every != 0 and self._done != self._total:
            return
        elapsed = self._clock() - self._started
        parts = [
            f"{self._label} progress: {self._done}/{self._total} "
            f"{self._unit}s ({100.0 * self._done / max(self._total, 1):.0f}%)",
            f"elapsed={format_duration(elapsed)}",
        ]
        if self._reused:
            parts.append(f"reused={self._reused}")
        # Rate and ETA come from WORKED units only; a resumed run's skipped bags
        # cost ~0 and would otherwise project a finish time that never arrives.
        if self._worked:
            per_unit = elapsed / self._worked
            remaining = max(self._total - self._done, 0)
            parts.append(f"{per_unit / 60.0:.1f} min/{self._unit}")
            parts.append(f"eta={format_duration(per_unit * remaining)}")
        logger.info(" ".join(parts))

    def summary(self) -> str:
        elapsed = self._clock() - self._started
        return (
            f"{self._label} complete: {self._done} {self._unit}s in "
            f"{format_duration(elapsed)} ({self._worked} derived, "
            f"{self._reused} reused)"
        )


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 90.0:
        return f"{seconds:.0f}s"
    if seconds < 5400.0:
        return f"{seconds / 60.0:.1f}m"
    return f"{seconds / 3600.0:.1f}h"
