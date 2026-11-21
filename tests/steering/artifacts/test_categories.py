"""Category lookup + remote-URI resolution."""
from __future__ import annotations

import pytest

from steering.artifacts.categories import ResolvedCategory, resolve_category
from steering.artifacts.config import ArtifactsConfig, CategoryConfig

_CFG = ArtifactsConfig(
    bucket_owner="acme",
    bucket_name="bkt",
    categories={
        "base_checkpoints": CategoryConfig(
            local_root="/cache/base_checkpoints",
            remote_subpath="checkpoints/base",
            default_exclude=["**/optim_*.pt"],
        ),
        "features": CategoryConfig(
            local_root="out/data/features",
            remote_subpath="features",
        ),
    },
)


def test_resolve_known_category():
    rc = resolve_category("base_checkpoints", _CFG)
    assert isinstance(rc, ResolvedCategory)
    assert rc.name == "base_checkpoints"
    assert rc.local_root.as_posix().endswith("/cache/base_checkpoints")
    assert rc.remote_uri == "hf://buckets/acme/bkt/checkpoints/base"
    assert rc.config.default_exclude == ["**/optim_*.pt"]


def test_resolve_unknown_category_lists_options():
    with pytest.raises(KeyError, match="unknown category 'nope'.*base_checkpoints.*features"):
        resolve_category("nope", _CFG)
