"""CLI dispatch for scripts.utils.hf_pull (Hydra-composed)."""
from __future__ import annotations

import inspect
from unittest.mock import patch

from omegaconf import DictConfig, OmegaConf

import scripts.utils.hf_pull as hf_pull_mod
from scripts.utils.hf_pull import main


def _make_cfg(**overrides) -> DictConfig:
    """Build a minimal DictConfig matching what Hydra composes from default.yaml."""
    base = {
        "category": "features",
        "bucket_owner": "acme",
        "bucket_name": "bkt",
        "dry_run": False,
        "no_default_include": False,
        "no_default_exclude": False,
        "categories": {
            "features": {
                "local_root": "/tmp/test_features",
                "remote_subpath": "features",
                "default_include": [],
                "default_exclude": [],
            },
        },
    }
    base.update(overrides)
    return OmegaConf.create(base)


def test_main_is_hydra_wrapped():
    """``hydra.main`` decorates ``main`` so the underlying function
    is accessible via ``__wrapped__``."""
    assert hasattr(main, "__wrapped__")


def test_pull_invokes_sync_pull():
    cfg = _make_cfg(dry_run=True)
    with patch("scripts.utils.hf_pull.sync_pull") as mock_pull:
        main.__wrapped__(cfg)
    assert mock_pull.called
    _, kwargs = mock_pull.call_args
    assert kwargs["category_name"] == "features"
    assert kwargs["dry_run"] is True


def test_pull_user_include_extends_filters():
    cfg = _make_cfg(include=["s128/l8/**"])
    with patch("scripts.utils.hf_pull.sync_pull") as mock_pull:
        main.__wrapped__(cfg)
    _, kwargs = mock_pull.call_args
    assert kwargs["user_include"] == ["s128/l8/**"]


def test_pull_does_not_accept_delete():
    """Pull must not expose a `delete` flag — there's no safe symmetric semantics for it."""
    src = inspect.getsource(hf_pull_mod)
    # Allow the word "delete" in the docstring/comment that explains the asymmetry,
    # but not as a kwarg passed to sync_pull or any function call.
    assert "delete=" not in src
