"""GPU idle-gate for the perf test suite.

Polls ``nvidia-smi`` until a GPU is found with utilization <= max_util_pct
and free memory >= min_free_gib GiB, then returns its index.  Each poll's
state is logged so the wait is visible in test output.

Public API
----------
wait_for_idle_local_gpu(*, max_util_pct=5, min_free_gib=20, poll_s=30, timeout_s=None) -> int
    Block until an idle GPU is available and return its index.

Internal (patchable for tests)
-------------------------------
_query_nvidia_smi() -> str
    Runs nvidia-smi and returns the raw CSV text.
"""

from __future__ import annotations

import subprocess
import time

from loguru import logger


def _query_nvidia_smi() -> str:
    """Run nvidia-smi and return the raw CSV output string.

    Separated from the polling loop so tests can monkeypatch it without
    touching subprocess.
    """
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def wait_for_idle_local_gpu(
    *,
    max_util_pct: int = 5,
    min_free_gib: int = 20,
    poll_s: int = 30,
    timeout_s: float | None = None,
) -> int:
    """Block until a local GPU is idle and return its index.

    Parses ``nvidia-smi --query-gpu=index,utilization.gpu,memory.free
    --format=csv,noheader,nounits`` and returns the first GPU index whose
    utilization is <= max_util_pct and whose free memory is >= min_free_gib GiB.
    Polls every poll_s seconds.

    Parameters
    ----------
    max_util_pct:
        Maximum GPU utilization percentage to consider idle (inclusive).
    min_free_gib:
        Minimum free GPU memory in GiB to consider available.
    poll_s:
        Seconds to sleep between polls.
    timeout_s:
        Maximum total seconds to wait.  ``None`` = wait forever — for an
        UNATTENDED run prefer a finite value, since a persistently busy or
        unparseable (``[N/A]`` utilization) GPU would otherwise block indefinitely
        (each poll now logs WHY it is waiting / skipping, so the stall is visible).

    Returns
    -------
    int
        Index of the first idle GPU.

    Raises
    ------
    TimeoutError
        When timeout_s is set and no idle GPU is found within the deadline.
    """
    start = time.monotonic()

    while True:
        output = _query_nvidia_smi()
        for line in output.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            try:
                idx, util, free_mib = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                # A non-integer field (e.g. utilization '[N/A]' during a driver
                # hiccup / MIG mode) — do NOT silently skip forever.  Log it so a
                # persistent unparseable GPU is VISIBLE rather than a silent stall,
                # and fall through to the deadline check below so a finite
                # ``timeout_s`` still fires (this path cannot defeat the timeout).
                logger.warning(
                    "nvidia-smi line {!r} has a non-integer field; skipping this "
                    "GPU for this poll (cannot evaluate idleness).",
                    line,
                )
                continue
            free_gib = free_mib / 1024.0
            logger.info(
                "GPU {}: util={}%, free={:.1f} GiB  (threshold: util<={}%, free>={}GiB)",
                idx,
                util,
                free_gib,
                max_util_pct,
                min_free_gib,
            )
            if util <= max_util_pct and free_gib >= min_free_gib:
                logger.info("GPU {} is idle — proceeding with profile run.", idx)
                return idx

        if timeout_s is not None and (time.monotonic() - start) >= timeout_s:
            raise TimeoutError(
                f"No idle GPU found within {timeout_s}s "
                f"(need util<={max_util_pct}%, free>={min_free_gib}GiB)"
            )

        logger.info(
            "No idle GPU found; retrying in {}s (timeout={}s).",
            poll_s,
            timeout_s,
        )
        time.sleep(poll_s)
