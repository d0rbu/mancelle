"""CLI dispatch for scripts.utils.hf_push (Hydra-composed)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from omegaconf import DictConfig, OmegaConf

from scripts.utils.hf_push import main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(**overrides) -> DictConfig:
    """Build a minimal DictConfig matching what Hydra composes from default.yaml."""
    base = {
        "category": "base_checkpoints",
        "bucket_owner": "acme",
        "bucket_name": "bkt",
        "dry_run": False,
        "delete": False,
        "no_default_include": False,
        "no_default_exclude": False,
        "categories": {
            "base_checkpoints": {
                "local_root": "/tmp/test_base_checkpoints",
                "remote_subpath": "checkpoints/base",
                "default_include": [],
                "default_exclude": ["**/optim_*.pt"],
            },
        },
    }
    base.update(overrides)
    return OmegaConf.create(base)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_main_is_hydra_wrapped():
    """``hydra.main`` decorates ``main`` so the underlying function
    is accessible via ``__wrapped__``. This also validates importability."""
    assert hasattr(main, "__wrapped__")


def test_push_invokes_sync_push(tmp_path: Path):
    cfg = _make_cfg(dry_run=True, delete=False)
    with patch("scripts.utils.hf_push.sync_push") as mock_push:
        main.__wrapped__(cfg)
    assert mock_push.called
    _, kwargs = mock_push.call_args
    assert kwargs["category_name"] == "base_checkpoints"
    assert kwargs["dry_run"] is True
    assert kwargs["delete"] is False  # default


def test_push_delete_flag_propagates():
    cfg = _make_cfg(dry_run=True, delete=True)
    with patch("scripts.utils.hf_push.sync_push") as mock_push:
        main.__wrapped__(cfg)
    _, kwargs = mock_push.call_args
    assert kwargs["delete"] is True


def test_push_user_include_exclude_lists():
    cfg = _make_cfg(dry_run=True, include=["*.safetensors"], exclude=["scratch/**"])
    with patch("scripts.utils.hf_push.sync_push") as mock_push:
        main.__wrapped__(cfg)
    _, kwargs = mock_push.call_args
    assert kwargs["user_include"] == ["*.safetensors"]
    assert kwargs["user_exclude"] == ["scratch/**"]
