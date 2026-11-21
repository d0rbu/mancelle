"""CUDA allocator cleanup helpers for long AxBench loops."""

from __future__ import annotations

import torch as t


def release_cuda_iteration_memory() -> None:
    """Release cached CUDA blocks after one loop unit.

    AxBench Stage 0c/1b/1c QC runs many independent SAE forwards in one
    process. The returned activations are local values, but long runs can
    still accumulate cached allocator blocks between concepts/candidates.
    This helper is intentionally output-inert: it changes only CUDA memory
    lifetime, not tensors used by callers.

    This used to call ``gc.collect()`` first. It no longer does. A ``py-spy``
    profile of a live Stage 1b run measured that collection at **65% of the
    serialized result-handler thread** (581s of 894s): a full generational pass
    walks a heap holding a 4B-parameter model, and the handler runs
    single-threaded, so it sat directly on the throughput ceiling. Dropping it
    moved the busiest thread from 99.8% of all sampled CPU to 69% -- the stage
    stopped being serialized behind the collector.

    ``empty_cache()`` is kept: it measured ~0.7% (6.4s of 894s) and is the part
    that actually bounds the allocator's cached blocks.

    CPython reclaims the activation tensors here by refcount as soon as the
    callers' ``del`` statements run; the cycle collector was only needed for
    reference *cycles*, which this path does not create. If a future change
    does introduce cycles and RSS grows over a long run, reinstate a collection
    at a coarse cadence -- per concept termination, which is what the original
    "between concepts/candidates" wording describes -- rather than per QC batch.
    """
    if t.cuda.is_available():
        t.cuda.empty_cache()
