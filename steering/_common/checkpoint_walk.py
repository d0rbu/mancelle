"""Soft variant of ``nanochat.checkpoint_manager.find_last_step``.

Returns ``None`` for missing-dir, file-instead-of-dir, empty-dir, or
dir-with-no-``model_*.pt`` cases — where the strict ``find_last_step``
would raise ``FileNotFoundError``. Used by ``_recover_model_config`` and
``_recover_parent_user_config`` in :mod:`nanochat.checkpoint_manager` so
an empty parent checkpoint directory falls through to the documented
Strategy-2 reconstruction instead of aborting the ancestor walk.

The helper is **self-contained**: it does not delegate to
``find_last_step`` because :mod:`nanochat.checkpoint_manager` imports
*this* module, and routing back via ``find_last_step`` would create a
circular import. The inlined glob logic is byte-equivalent to
``find_last_step``'s implementation; if the canonical filename pattern
ever changes, both call sites must be updated together.

Errors other than ``FileNotFoundError`` (e.g. ``PermissionError``,
``OSError``) propagate so the caller never silently treats a permission
issue as "no checkpoint here, skip recovery".
"""

from __future__ import annotations

import glob
import os


def find_last_step_or_none(checkpoint_dir: str | os.PathLike[str]) -> int | None:
    """Return the latest step in ``checkpoint_dir``, or ``None``.

    Behaviour
    ---------
    * If ``checkpoint_dir`` does not exist or is not a directory:
      return ``None``.
    * If ``checkpoint_dir`` exists but contains no ``model_*.pt``
      files: return ``None`` (instead of raising
      ``FileNotFoundError``).
    * Otherwise: return the integer step parsed from the highest
      ``model_<step>.pt`` filename.

    The contract intentionally mirrors the ``parent_step is not None``
    pattern in ``_recover_*`` so ``parent_step = find_last_step_or_none(parent_dir)``
    becomes a one-line replacement for the existing
    ``parent_step = find_last_step(parent_dir) if os.path.isdir(parent_dir) else None``
    that crashes on empty dirs.

    Notes
    -----
    * This module is self-contained (no nanochat imports) to avoid a
      circular dependency: ``nanochat.checkpoint_manager`` imports this
      function, so this module must not import from ``nanochat``.
    * Permission errors and ``OSError`` other than ``FileNotFoundError``
      propagate (silent corruption avoidance).
    """
    path = os.fspath(checkpoint_dir)
    if not os.path.isdir(path):
        return None
    checkpoint_files = glob.glob(os.path.join(path, "model_*.pt"))
    if not checkpoint_files:
        return None
    return int(
        max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files)
    )
