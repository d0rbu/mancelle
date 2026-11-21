"""Compatibility utilities for nnsight/nnterp import side-effects.

nnsight patches ``torch.Tensor.backward`` globally at import time (via
``DEFAULT_PATCHER.__enter__()`` in ``nnsight/__init__.py``).  This adds a
try/except wrapper around every ``.backward()`` call to support nnsight's
``BackwardsTracer``, which is unnecessary during normal training and adds
measurable overhead to every backward pass.

Call ``unpatch_nnsight()`` after importing any module that transitively
imports nnsight/nnterp to restore the original ``Tensor.backward``.
"""

from __future__ import annotations

import torch
from loguru import logger


def unpatch_nnsight() -> bool:
    """Restore ``torch.Tensor.backward`` to its original implementation.

    Safe to call even if nnsight was never imported (returns False).
    Safe to call multiple times (idempotent after the first unpatch).

    Returns:
        True if a patch was found and removed, False otherwise.
    """
    try:
        import nnsight
    except ImportError:
        return False

    patcher = getattr(nnsight, "DEFAULT_PATCHER", None)
    if patcher is None:
        return False

    # Find the Tensor.backward patch
    for patch in patcher.patches:
        if (
            getattr(patch, "parent", None) is torch.Tensor
            and getattr(patch, "key", None) == "backward"
        ):
            # Check if backward is currently the nnsight wrapper
            if torch.Tensor.backward is not patch.replacement:
                return False  # already restored

            # Restore the original method directly.
            torch.Tensor.backward = patch.orig
            logger.info(
                "Restored torch.Tensor.backward (removed nnsight wrapper)"
            )
            return True

    return False
