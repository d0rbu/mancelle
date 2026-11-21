import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from steering.refined_types import PositiveInt
from steering.staging.prefetcher import StagedPrefetcher

_SERIAL_DEPTH = PositiveInt.parse(1)
_PARALLEL_DEPTH = PositiveInt.parse(4)


def _drain_all(
    depth: PositiveInt,
    paths: list[str],
    load_fn: Callable[[str, str], Any],
    scratch: Path,
) -> None:
    """Drive the prefetcher with the same prime + get-before-submit cadence the
    extraction loop uses, for any depth."""
    n = len(paths)
    with StagedPrefetcher(scratch_dir=scratch, depth=depth, load_fn=load_fn) as pf:
        for j in range(min(depth, n)):
            pf.submit(paths[j])
        for i in range(n):
            pf.get()
            nxt = i + depth
            if nxt < n:
                pf.submit(paths[nxt])


def test_deeper_prefetch_overlaps_slow_reads(tmp_path):
    """A deep prefetch overlaps slow reads, so depth=4 finishes much faster than the
    serial depth=1 baseline. Asserted as a RELATIVE speedup (not an absolute wall-clock
    deadline) so it isn't flaky under CI load."""
    src = tmp_path / "src"
    src.mkdir()
    for i in range(4):
        (src / f"s{i}.bin").write_bytes(b"x")
    paths = [str(src / f"s{i}.bin") for i in range(4)]

    def slow_load(source_path, local_path):
        time.sleep(0.05)  # simulate a slow network read
        return source_path

    t0 = time.perf_counter()
    _drain_all(
        _SERIAL_DEPTH, paths, slow_load, tmp_path / "sc1"
    )  # serial: ~4 * 0.05 = 0.20s
    t_serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    _drain_all(
        _PARALLEL_DEPTH, paths, slow_load, tmp_path / "sc4"
    )  # overlapped: ~0.05s
    t_parallel = time.perf_counter() - t0

    # depth=4 overlaps all four reads -> well under 60% of the serial time. Relative to
    # the same machine's serial run, so it holds regardless of absolute speed/contention.
    assert t_parallel < t_serial * 0.6, (t_parallel, t_serial)
