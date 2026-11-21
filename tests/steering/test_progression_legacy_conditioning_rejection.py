"""Rejection coverage for ``_validate_no_legacy_conditioning_modes``
(scripts/train/training_progression.py).

Progression encoder-decoder eval is always full-context. The legacy
no-context switches (``run_unconditioned``, ``require_context_features``,
``context_mode``) must be rejected with an operator-actionable error rather
than silently ignored — a silently-ignored ``run_unconditioned: true`` would
make an operator believe an unconditioned baseline ran when none did.

The conditioning block is built with OmegaConf to mirror how the value
arrives in production (a Hydra ``DictConfig``).
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from scripts.train.training_progression import (
    _validate_no_legacy_conditioning_modes,
)


def _cfg(conditioning: dict | None):
    """Build a Hydra-style cfg with the given conditioning block."""
    return OmegaConf.create({"conditioning": conditioning})


def test_run_unconditioned_key_is_rejected() -> None:
    cfg = _cfg({"run_unconditioned": True})
    with pytest.raises(ValueError, match="run_unconditioned is no longer supported"):
        _validate_no_legacy_conditioning_modes(cfg)


def test_run_unconditioned_false_is_still_rejected() -> None:
    """The KEY presence is what's rejected, not its truthiness — a `false`
    value must not slip through as "harmless"."""
    cfg = _cfg({"run_unconditioned": False})
    with pytest.raises(ValueError, match="run_unconditioned is no longer supported"):
        _validate_no_legacy_conditioning_modes(cfg)


def test_require_context_features_key_is_rejected() -> None:
    cfg = _cfg({"require_context_features": True})
    with pytest.raises(
        ValueError, match="require_context_features is no longer supported"
    ):
        _validate_no_legacy_conditioning_modes(cfg)


def test_context_mode_key_is_rejected() -> None:
    cfg = _cfg({"context_mode": "prompt_only"})
    with pytest.raises(ValueError, match="context_mode is no longer supported"):
        _validate_no_legacy_conditioning_modes(cfg)


def test_clean_conditioning_block_passes() -> None:
    """A conditioning block carrying none of the legacy keys is accepted."""
    cfg = _cfg(
        {
            "identity": {
                "model_id": "m",
                "sae_id": "s",
                "layer": 0,
                "pooling": "mean",
            },
            "context_cache_dir": "/tmp/cache",
        }
    )
    # Must not raise.
    _validate_no_legacy_conditioning_modes(cfg)


def test_missing_conditioning_block_passes() -> None:
    """No conditioning block at all is not a legacy-mode violation."""
    cfg = OmegaConf.create({})
    _validate_no_legacy_conditioning_modes(cfg)


def test_null_conditioning_block_passes() -> None:
    """A null conditioning value resolves to an empty dict and is accepted."""
    cfg = _cfg(None)
    _validate_no_legacy_conditioning_modes(cfg)
