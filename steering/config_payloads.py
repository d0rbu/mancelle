"""Validated conversions from Hydra configs to external-service payloads."""

from __future__ import annotations

from typing import Any, cast

from omegaconf import DictConfig, OmegaConf


def resolved_config_dict(cfg: DictConfig) -> dict[str, Any]:
    """Resolve a Hydra config and validate its string-keyed mapping shape.

    ``OmegaConf.to_container`` can also return lists, scalars, and mappings
    with non-string keys. External logging APIs such as W&B accept only the
    string-keyed mapping case used by this repository, so reject any other
    shape at that boundary.
    """
    payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Resolved Hydra config must be a mapping, got {type(payload).__name__}")
    non_string_keys = [key for key in payload if not isinstance(key, str)]
    if non_string_keys:
        raise TypeError(f"Resolved Hydra config keys must be strings, got {non_string_keys[0]!r}")
    return cast(dict[str, Any], payload)
