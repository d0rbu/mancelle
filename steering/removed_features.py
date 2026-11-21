"""Guards for the intentionally removed legacy clustering pipeline.

The discovery-based pipeline used the singular ``feature_clustering`` schema;
it remains removed. The preselection-only static-file feature uses the distinct
plural ``feature_clusters`` schema. These narrow guards preserve fail-loud
handling for old configs and artifacts on both supported and unsupported
surfaces.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import OmegaConf

from errors import RemovedFeatureError

_MISSING = object()


def _plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _get(container: Any, key: str, default: Any = _MISSING) -> Any:
    container = _plain(container)
    if isinstance(container, Mapping):
        return container.get(key, default)
    return getattr(container, key, default)


def _has_key(container: Any, key: str) -> bool:
    container = _plain(container)
    if isinstance(container, Mapping):
        return key in container
    return hasattr(container, key)


def _removed_message(surface: str) -> str:
    return (
        f"{surface}: the feature clustering pipeline was removed. Remove "
        "`feature_clustering` config, regenerate unclustered artifacts, or use "
        "an unclustered checkpoint."
    )


def reject_feature_clustering_config(cfg: Any, surface: str) -> None:
    """Reject any current-run ``feature_clustering`` config block."""
    if cfg is not None and _has_key(cfg, "feature_clustering"):
        raise RemovedFeatureError(_removed_message(surface))


def reject_feature_clustering_metadata(meta: Any, surface: str) -> None:
    """Reject legacy metadata that records clustered features.

    Older unclustered checkpoints may contain ``feature_clustering: null`` or a
    disabled ``user_config.feature_clustering: {enabled: false}``; those pass.
    Anything that records actual clustered semantics raises.
    """
    meta = _plain(meta)
    if meta is None:
        return
    if not isinstance(meta, Mapping):
        return

    if _get(meta, "feature_clustering_canonical", None) is not None:
        raise RemovedFeatureError(_removed_message(surface))
    if _get(meta, "feature_clustering", None) is not None:
        raise RemovedFeatureError(_removed_message(surface))

    user_config = _get(meta, "user_config", None)
    user_config = _plain(user_config)
    if not isinstance(user_config, Mapping):
        return

    if _get(user_config, "feature_clustering_canonical", None) is not None:
        raise RemovedFeatureError(_removed_message(surface))

    if not _has_key(user_config, "feature_clustering"):
        return
    clustering_cfg = _plain(_get(user_config, "feature_clustering"))
    if clustering_cfg is None:
        return
    if not isinstance(clustering_cfg, Mapping):
        raise RemovedFeatureError(_removed_message(surface))

    enabled = clustering_cfg.get("enabled", None)
    if enabled not in (None, False):
        raise RemovedFeatureError(_removed_message(surface))

    allowed_disabled_keys = {"enabled"}
    for key, value in clustering_cfg.items():
        if key not in allowed_disabled_keys and value is not None:
            raise RemovedFeatureError(_removed_message(surface))


def reject_autointerp_clustering_manifest(manifest: Any, surface: str) -> None:
    """Reject stale Autointerp manifests that encoded removed target grouping."""
    manifest = _plain(manifest)
    if not isinstance(manifest, Mapping):
        return

    clustering_enabled = manifest.get("clustering_enabled", None)
    if clustering_enabled not in (None, False):
        raise RemovedFeatureError(_removed_message(surface))
    if (
        manifest.get("feature_clustering") is not None
        or manifest.get("feature_clustering_canonical") is not None
    ):
        raise RemovedFeatureError(_removed_message(surface))

    targets = manifest.get("targets", [])
    targets = _plain(targets)
    if not isinstance(targets, list):
        return
    for raw_target in targets:
        target = _plain(raw_target)
        if not isinstance(target, Mapping):
            continue
        feature_ids = target.get("feature_ids", [])
        kind = target.get("kind")
        if (
            (kind is not None and kind != "feature")
            or str(target.get("target_key", "")).startswith("cluster:")
            or (isinstance(feature_ids, list) and len(feature_ids) != 1)
        ):
            raise RemovedFeatureError(_removed_message(surface))
