"""Shared two-rank gloo spawn harness for aligned-collective tests.

One home for the spawn + ``file://``-init + collect + kill-stragglers pattern
used by the DDP broadcast and percentile-preflight suites (mirroring
``tests/scripts/test_encdec_sft_ddp_barrier.py``'s child cleanup). Workers are
module-level functions ``worker(rank, world_size, init_file, result_queue,
*extra_args)`` that put ``(rank, *payload)`` on the queue exactly once —
BEFORE tearing down their process group, so a blocked teardown surfaces as a
queue timeout here rather than a silent hang.
"""

from __future__ import annotations

import multiprocessing
import tempfile
from pathlib import Path


def run_two_ranks(target, tag, extra_args=(), join_timeout_s=120.0):
    """Run ``target`` on two spawned ranks; return ``{rank: payload_tuple}``."""
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    with tempfile.TemporaryDirectory() as temp_dir:
        init_file = str(Path(temp_dir) / tag)
        processes = [
            context.Process(target=target, args=(rank, 2, init_file, result_queue, *extra_args))
            for rank in range(2)
        ]
        try:
            for process in processes:
                process.start()
            results = {}
            for _ in range(2):
                rank, *payload = result_queue.get(timeout=join_timeout_s)
                results[rank] = tuple(payload)
            for process in processes:
                process.join(timeout=join_timeout_s)
        finally:
            for process in processes:
                if process.is_alive():
                    process.kill()
                    process.join()
        for process in processes:
            assert process.exitcode == 0
    return results
