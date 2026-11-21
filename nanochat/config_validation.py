"""Hydra config validation helpers for scripts with complex CLI semantics."""

from __future__ import annotations

from omegaconf import DictConfig, ListConfig


def validate_sweep_mode(cfg: DictConfig) -> None:
    """Validate the polymorphic sweep ``mode`` field.

    Valid values:
    - ``None``       — run all configs
    - ``"dry_run"``  — list configs without executing
    - ``<int>``      — run single config at this index
    - ``[s, e]``     — run configs in range [s, e)

    Used by: run_encdec_sweep.py, run_encdec_sweep_fast.py
    """
    mode = cfg.mode
    if mode is None:
        return  # run all
    if isinstance(mode, str):
        if mode != "dry_run":
            raise ValueError(
                f"mode string must be 'dry_run', got '{mode}'"
            )
        return
    if isinstance(mode, int):
        return  # index
    if isinstance(mode, (list, ListConfig)):
        if len(mode) != 2:
            raise ValueError(
                f"mode list must have exactly 2 elements [start, end), got {len(mode)}"
            )
        return
    raise ValueError(
        f"mode must be null, 'dry_run', an int (index), or a [start, end) list — got {type(mode).__name__}: {mode}"
    )
