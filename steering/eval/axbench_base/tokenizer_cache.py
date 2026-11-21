"""Process-local cache for the nanochat tokenizer used by AxBench stages.

The nanochat loader performs filesystem work and reconstructs the tokenizer
encoding on every call.  Stage 1b and Stage 1c need the tokenizer for many
row-boundary checks, so those calls must share one process-local instance.

This cache is intentionally owned by ``steering`` rather than the vendored
``nanochat`` package.  ``NANOCHAT_BASE_DIR`` is a process-level configuration
for the evaluation runner; callers that change it in tests must clear this
cache before loading the tokenizer for the new base directory.
"""

from __future__ import annotations

import functools
import threading
from typing import Any

from nanochat.tokenizer import get_tokenizer as _nanochat_get_tokenizer


@functools.lru_cache(maxsize=1)
def _get_nanochat_tokenizer_cached() -> Any:
    """Load the tokenizer once for this process.

    ``lru_cache`` does not cache exceptions, so a failed first load can be
    retried after the underlying artifact or environment has been repaired.
    """

    return _nanochat_get_tokenizer()


_TOKENIZER_CACHE_LOCK = threading.RLock()


def get_nanochat_tokenizer() -> Any:
    """Return the process-local nanochat tokenizer instance.

    The lock makes a cold concurrent lookup single-flight.  Stage 1 row
    assembly is caller-thread-only, but keeping this helper correct for
    concurrent first use prevents duplicate expensive tokenizer construction
    if another steering path accesses it during startup.
    """

    with _TOKENIZER_CACHE_LOCK:
        return _get_nanochat_tokenizer_cached()
