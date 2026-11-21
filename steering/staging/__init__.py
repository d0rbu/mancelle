# steering/staging/__init__.py
"""Staging package: EXDEV-safe shard relocation and free-space guards.

Note: `relocate` is intentionally NOT imported here as a function — doing so would
shadow `steering.staging.relocate` (the *module*) in the package attribute namespace,
breaking `monkeypatch.setattr("steering.staging.relocate._same_filesystem", ...)`.
Import the function directly: `from steering.staging.relocate import relocate`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from steering.staging.relocate import (
    DestinationSpaceError,
    assert_startup_free_space,
    coordinate_free_space_check,
    free_gb,
    sweep_dir,
    wait_for_free_space,
)


@runtime_checkable
class BackgroundSaver(Protocol):
    """Duck-type accepted by process_shard(background_saver=...)."""

    def submit(self, data: dict[str, Any], path: str | Path, /) -> object: ...

    @property
    def in_flight(self) -> int: ...


__all__ = [
    "BackgroundSaver",
    "DestinationSpaceError",
    "assert_startup_free_space",
    "coordinate_free_space_check",
    "free_gb",
    "sweep_dir",
    "wait_for_free_space",
]
