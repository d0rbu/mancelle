"""Hydra-backed config dataclasses for the HF artifacts bucket sync."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class CategoryConfig:
    """One logical grouping of files synced to the same bucket subpath."""

    local_root: str
    remote_subpath: str
    default_include: list[str] = field(default_factory=list)
    default_exclude: list[str] = field(default_factory=list)

@dataclass(frozen=True)
class ArtifactsConfig:
    """Top-level config: which bucket and which categories live in it."""

    bucket_owner: str
    bucket_name: str
    categories: dict[str, CategoryConfig]

    @classmethod
    def from_config(cls, raw: DictConfig) -> ArtifactsConfig:
        """Build a typed artifacts config from the composed Hydra mapping."""
        container = cast(
            dict[str, Any],
            OmegaConf.to_container(raw, resolve=True),
        )

        try:
            cats = {
                name: CategoryConfig(**spec)
                for name, spec in container["categories"].items()
            }
        except KeyError as exc:
            raise KeyError(f"ArtifactsConfig.from_config: missing required key {exc}") from exc
        except TypeError as exc:
            raise TypeError(
                f"ArtifactsConfig.from_config: malformed category spec — {exc}"
            ) from exc
        return cls(
            bucket_owner=container["bucket_owner"],
            bucket_name=container["bucket_name"],
            categories=cats,
        )
