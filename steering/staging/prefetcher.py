# steering/staging/prefetcher.py
"""N-deep parquet prefetch with local-disk staging.

Each job copies the source parquet to local scratch (sequential bulk copy via
os.sendfile), runs load_fn(source_path, local_path) (e.g. read texts into RAM),
then deletes the local copy. get() returns results in submitted (FIFO) order; the
caller keeps <= depth in flight. Backed by the shared BoundedFifoExecutor (the one
bounded-FIFO future queue), so only the staging + overflow policy lives here.
Generic over load_fn to avoid a circular import with extract_features (which
supplies prefetch_shard_data as load_fn).
"""
from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from beartype import beartype

from steering.refined_types import PositiveInt
from steering.staging._queue import BoundedFifoExecutor
from steering.staging.relocate import prepare_scratch_dir, sweep_dir


class StagedPrefetcher:
    """N-deep staged prefetcher.

    Not thread-safe across callers: submit()/get() must be driven from one thread
    (the extraction main loop). Worker threads only run the staged copy+load and
    never touch the executor's queue.
    """

    @beartype
    def __init__(
        self,
        *,
        scratch_dir: str | Path,
        depth: PositiveInt,
        load_fn: Callable[[str, str], Any],
    ) -> None:
        # The decorated constructor validates depth before the body runs, so a
        # bad depth cannot spawn a thread pool or touch the scratch directory.
        self._exec = BoundedFifoExecutor(depth=depth, thread_name_prefix="staged-prefetch")
        self._depth = depth
        self._load_fn = load_fn
        self._scratch_dir = prepare_scratch_dir(scratch_dir)  # mkdir + sweep crash leftovers

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        # Drain remaining prefetches, swallowing failures so we don't mask the
        # exception that triggered teardown; then shut the pool down and sweep.
        while self._exec.in_flight:
            try:
                self._exec.get()
            except BaseException:  # noqa: BLE001 — don't mask the original exception on exit
                pass
        self._exec.__exit__(None, None, None)  # queue now empty -> just shuts the pool down
        sweep_dir(self._scratch_dir, swallow_errors=True)  # teardown: never mask the original exc
        return False

    def submit(self, source_path: str) -> None:
        # Invariant: <= depth in flight. The main loop submits `depth` initially then
        # get()-before-submit(), so this raises only on a contract violation — it must
        # NOT silently drop a prefetched result the consumer is waiting to get()
        # (which the executor's fire-and-forget drain-oldest overflow would do).
        if self._exec.in_flight >= self._depth:
            raise RuntimeError(f"StagedPrefetcher: submit beyond depth={self._depth}; call get() first")
        self._exec.submit(self._stage_and_load, source_path)

    def get(self) -> Any:
        return self._exec.get()

    def _stage_and_load(self, source_path: str) -> Any:
        # Uniquify the local name so two in-flight sources sharing a basename
        # (possible when depth>1) cannot collide in scratch.
        local = self._scratch_dir / f"{uuid.uuid4().hex[:8]}_{Path(source_path).name}"
        try:
            shutil.copyfile(source_path, local)  # one sequential bulk network read
            return self._load_fn(source_path, str(local))
        finally:
            # cleans the local copy on success AND a partial/failed copy
            try:
                os.remove(local)
            except OSError:
                pass
