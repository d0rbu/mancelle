# steering/staging/saver.py
"""Background save->relocate pipeline for extraction output shards.

Each job: save_features(shard_dict) -> LOCAL scratch (atomic, same-FS), then
relocate() -> network output_dir. Bounded to `depth` in flight (via the shared
BoundedFifoExecutor), so peak local = depth * shard_size. When enabled, the free-space
floor is checked before scratch serialization and again before destination relocation.
submit(data, out_stem) matches the BackgroundSaver protocol process_shard expects.
"""
from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any, Self

from beartype import beartype

from steering.feature_io import save_features
from steering.refined_types import PositiveInt
from steering.staging._queue import BoundedFifoExecutor
from steering.staging.relocate import (
    prepare_scratch_dir,
    raise_destination_space_error,
    relocate,
    sweep_dir,
    wait_for_free_space,
)


class StagedSaver:
    @beartype
    def __init__(
        self,
        *,
        scratch_dir: str | Path,
        final_dir: str | Path,
        depth: PositiveInt,
        zstd_level: int | None = None,
        min_free_gb: float = 0.0,
    ) -> None:
        self._final_dir = Path(final_dir)
        try:
            self._final_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise_destination_space_error(self._final_dir, e)
            raise
        self._scratch_dir = prepare_scratch_dir(scratch_dir)  # mkdir + sweep crash leftovers
        self._zstd_level = zstd_level
        self._min_free_gb = min_free_gb
        self._exec = BoundedFifoExecutor(depth=depth, thread_name_prefix="staged-saver")

    @property
    def in_flight(self) -> int:
        return self._exec.in_flight

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        try:
            return self._exec.__exit__(exc_type, exc_val, exc_tb)
        finally:
            # teardown: swallow sweep errors so they never mask the drain exception
            sweep_dir(self._scratch_dir, swallow_errors=True)

    def drain(self) -> None:
        """Block until all in-flight save+relocate jobs finish, surfacing the first
        exception, WITHOUT closing the saver. Call this at the end of the extraction
        loop while the profiler is still measuring, so the GPU-idle save tail lands
        inside the window (see run_extraction_loop)."""
        self._exec.drain()

    def submit(self, data: dict[str, Any], out_stem: str | Path) -> None:
        self._exec.submit(self._save_and_move, data, str(out_stem))

    def _save_and_move(self, data: dict[str, Any], out_stem: str) -> None:
        if self._min_free_gb > 0.0:
            wait_for_free_space(self._scratch_dir, min_free_gb=self._min_free_gb)
        scratch_stem = str(self._scratch_dir / Path(out_stem).name)
        local_path = save_features(data, scratch_stem, zstd_level=self._zstd_level)
        final_path = self._final_dir / local_path.name  # authoritative final dir
        if self._min_free_gb > 0.0:
            wait_for_free_space(
                self._final_dir, min_free_gb=self._min_free_gb, resource="destination"
            )
        relocate(local_path, final_path)
