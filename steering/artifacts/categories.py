"""Resolve a category name → typed local/remote endpoints."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from steering.artifacts.config import ArtifactsConfig, CategoryConfig
from steering.artifacts.paths import BucketURI


@dataclass(frozen=True)
class ResolvedCategory:
    """A category name turned into concrete local + remote roots."""

    name: str
    local_root: Path
    remote_uri: str  # hf://buckets/<owner>/<bucket>/<remote_subpath>
    config: CategoryConfig


def resolve_category(name: str, cfg: ArtifactsConfig) -> ResolvedCategory:
    if name not in cfg.categories:
        known = ", ".join(sorted(cfg.categories))
        raise KeyError(f"unknown category {name!r}; known: {known}")
    cat = cfg.categories[name]
    local_root = Path(os.path.expandvars(os.path.expanduser(cat.local_root)))
    remote_uri = BucketURI(
        owner=cfg.bucket_owner,
        bucket=cfg.bucket_name,
        path=cat.remote_subpath,
    )
    return ResolvedCategory(
        name=name,
        local_root=local_root,
        remote_uri=remote_uri,
        config=cat,
    )
