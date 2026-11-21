"""Feature configuration parsing for conditioned encoder-decoder evaluation.

Provides :class:`FeatureConfig` for describing a single conditioning setup and
:func:`parse_feature_configs` for resolving Hydra CLI input into a list of
configs.  Accepts four formats:

1. ``None`` / empty list → no static feature configs.
2. Path to a ``.yaml`` file → flat ``{feature_id: bucket_index}`` dict.
3. Path to a directory → recursive discovery of ``features.yaml`` files.
4. List of dicts (inline ``{name, features}``) or file/dir paths (mixed).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import yaml
from beartype import beartype
from beartype.vale import Is

from steering.conditional.checkpoint_conditioning import checkpoint_negative_selection

_UNSUPPORTED_CONFIG_KEYS = {"last_segment_features", "feature_conditions"}


def _is_signed_feature_mapping(value: object) -> bool:
    return isinstance(value, dict) and all(
        isinstance(feature_id, int)
        and not isinstance(feature_id, bool)
        and feature_id >= 0
        and isinstance(score, int)
        and not isinstance(score, bool)
        for feature_id, score in value.items()
    )


_SignedFeatureMapping = Annotated[dict[int, int], Is[_is_signed_feature_mapping]]


@beartype
@dataclass(frozen=True)
class FeatureConfig:
    """A single feature-conditioning configuration for evaluation.

    Attributes:
        name: Sanitized identifier safe for filesystem paths and JSON keys.
            Pure label — has NO dispatch effect at eval time. ``features``
            is the optional target/delta signal applied after prompt/context
            features are loaded by the eval wrapper. Naming has no dispatch
            effect.
        features: SAE feature id → integer value. Existing static-target
            consumers interpret values as bucket indices; harmfulness
            self-conditioning interprets them as sparse signed scores in
            ``{-1, +1}``. Each consumer validates its own value domain.
        strength: Harmfulness self-conditioning interpolation weight in
            ``[0, 1]``. Other consumers continue to use ``features`` directly,
            so the backward-compatible default is ``1.0``.
        source: Originating YAML path, if loaded from disk. None for inline
            configs.
        display_name: Human-readable label for chart legends and reports.
            Defaults to ``name`` if unset.
    """

    name: str
    features: _SignedFeatureMapping
    source: str | None = None
    display_name: str | None = None
    strength: float | int = 1.0

    def __post_init__(self):
        if (
            isinstance(self.strength, bool)
            or not isinstance(self.strength, (int, float))
            or not math.isfinite(float(self.strength))
            or not 0.0 <= float(self.strength) <= 1.0
        ):
            raise ValueError(
                f"FeatureConfig.strength must be finite and in [0, 1], got {self.strength!r}"
            )
        object.__setattr__(self, "strength", float(self.strength))
        if self.display_name is None:
            object.__setattr__(self, "display_name", self.name)


def sanitize_config_name(name: str) -> str:
    """Make a config name safe for JSON keys and filenames."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "features"


def uniquify_config_names(
    configs: Iterable[FeatureConfig],
) -> list[FeatureConfig]:
    """Return configs with deterministic unique names."""

    counts: dict[str, int] = {}
    result: list[FeatureConfig] = []
    for cfg in configs:
        base = sanitize_config_name(cfg.name)
        counts[base] = counts.get(base, 0) + 1
        name = base if counts[base] == 1 else f"{base}_{counts[base]}"
        result.append(
            FeatureConfig(
                name=name,
                features=cfg.features,
                strength=cfg.strength,
                source=cfg.source,
                display_name=cfg.display_name,
            )
        )
    return result


def _coerce_feature_dict(raw, *, source: str) -> dict[int, int]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Expected a feature dictionary in {source}, got {type(raw).__name__}"
        )
    result: dict[int, int] = {}
    for key, value in raw.items():
        if value is None:
            continue
        try:
            fid = int(key)
            bucket = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Feature ids and buckets in {source} must be integers; "
                f"got {key!r}: {value!r}"
            ) from exc
        result[fid] = bucket
    return result


def _reject_unsupported_config_keys(data: Mapping, *, source: str) -> None:
    unsupported = sorted(str(key) for key in data if str(key) in _UNSUPPORTED_CONFIG_KEYS)
    if not unsupported:
        return
    keys = ", ".join(f"`{key}`" for key in unsupported)
    raise ValueError(
        f"Unsupported feature config key(s) in {source}: {keys}. "
        "Use `features` for target/delta feature configs."
    )


def _is_flat_feature_mapping(data: dict) -> bool:
    if not data:
        return True
    structured_keys = {"name", "features", "strength"}
    return not any(str(key) in structured_keys for key in data)


def _load_yaml_config(path: Path) -> FeatureConfig:
    if path.is_dir():
        raise ValueError(
            f"Directories are not accepted for progression conditioned_configs: {path}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Feature config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        features: dict[int, int] = {}
        raw_name = path.stem
        strength = 1.0
    elif isinstance(data, dict) and _is_flat_feature_mapping(data):
        _reject_unsupported_config_keys(data, source=str(path))
        features = _coerce_feature_dict(data, source=str(path))
        raw_name = path.stem
        strength = 1.0
    elif isinstance(data, dict):
        _reject_unsupported_config_keys(data, source=str(path))
        raw_features = data.get("features")
        features = _coerce_feature_dict(raw_features, source=str(path))
        raw_name = str(data.get("name") or path.stem)
        strength = data.get("strength", 1.0)
    else:
        raise ValueError(
            f"Expected YAML mapping in feature config {path}, got {type(data).__name__}"
        )

    return FeatureConfig(
        name=sanitize_config_name(raw_name),
        features=features,
        strength=strength,
        source=str(path),
        display_name=raw_name,
    )


def _parse_inline_config(entry: Mapping, *, index: int) -> FeatureConfig:
    _reject_unsupported_config_keys(entry, source=f"conditioned_configs[{index}]")
    features_file = entry.get("features_file")
    raw_features = entry.get("features")
    if features_file is not None and raw_features is not None:
        raise ValueError(
            "conditioned_configs entry cannot set both `features_file` and an "
            "inline `features` dict — pick one."
        )

    if features_file is not None:
        path = Path(str(features_file))
        loaded = _load_yaml_config(path)
        name = entry.get("name")
        if name is None:
            resolved_name = loaded.name
            resolved_display = loaded.display_name
        else:
            raw = str(name)
            resolved_name = sanitize_config_name(raw)
            resolved_display = raw
        return FeatureConfig(
            name=resolved_name,
            features=loaded.features,
            strength=entry.get("strength", loaded.strength),
            source=loaded.source,
            display_name=resolved_display,
        )

    raw_name = str(entry.get("name") or f"custom_{index}")
    return FeatureConfig(
        name=sanitize_config_name(raw_name),
        features=_coerce_feature_dict(raw_features, source=raw_name),
        strength=entry.get("strength", 1.0),
        source=None,
        display_name=raw_name,
    )


def validate_inactive_bucket_feature_configs(
    configs: Iterable[object],
    *,
    num_buckets: int,
    checkpoint_meta: Mapping | None,
    checkpoint_label: str = "checkpoint",
) -> None:
    """Reject inactive-bucket configs on checkpoints without negative_selection.

    Mixed positive+negative eval YAMLs reserve bucket ``num_buckets - 1`` for
    inactive features. That encoding is only valid for checkpoints trained with
    ``negative_selection`` enabled, where positives use the reduced
    ``effective_buckets = num_buckets - 1`` space.
    """
    offending: list[str] = []
    for cfg in configs:
        features = getattr(cfg, "features", None)
        if not features:
            continue
        if num_buckets < 2:
            return
        inactive_bucket = num_buckets - 1
        if any(int(bucket) == inactive_bucket for bucket in features.values()):
            offending.append(str(getattr(cfg, "name", "feature config")))

    if not offending:
        return
    if checkpoint_negative_selection(checkpoint_meta) is not None:
        return

    preview = ", ".join(repr(name) for name in offending[:3])
    if len(offending) > 3:
        preview += f", ... ({len(offending)} total)"
    raise ValueError(
        "Inactive-bucket feature configs require a checkpoint trained with "
        f"negative_selection. Found bucket {inactive_bucket} in {preview}, but "
        f"{checkpoint_label} metadata has user_config.negative_selection=null."
    )


def require_nonempty_feature_configs(
    configs: Iterable[FeatureConfig],
    *,
    surface: str,
) -> None:
    """Reject explicit empty static target-delta configs.

    ``feature_configs: []`` means "no static target-delta variants" and is
    valid for evals that only run target_real or BPB. An explicit
    ``FeatureConfig(features={})`` is different: it asks the eval to emit a
    context-only baseline under an operator-chosen name. Encoder-decoder eval
    surfaces should not accept that as a renamed unconditioned/plain baseline.
    """

    empty_names = [cfg.name for cfg in configs if not cfg.features]
    if not empty_names:
        return
    raise ValueError(
        f"{surface} feature_configs must contain non-empty target/delta "
        "features. Empty static configs are intentionally unsupported because "
        "they create context-only baseline rows instead of evaluating a "
        f"conditioning signal. Empty config(s): {empty_names}"
    )


def ensure_core_eval_has_work(
    configs: list[FeatureConfig],
    *,
    is_encdec: bool,
    has_target_real: bool,
    surface: str,
) -> list[FeatureConfig]:
    """Resolve an empty ``feature_configs`` list for CORE/chat eval surfaces.

    An empty ``feature_configs`` must never silently produce zero work:

    - Decoder-only models have no encoder, so the no-intervention
      (unconditioned) pass is the meaningful baseline. An empty list becomes a
      single ``unconditional`` config so exactly one pass runs. This is the
      decoder-only baseline the encoder-decoder cleanup intentionally keeps.
    - Encoder-decoder models condition prompt tokens via a context cache and
      route work through ``target_real`` configs. An empty list with no
      ``target_real`` work has nothing to run and is a hard error rather than a
      silent no-op; an empty list *with* ``target_real`` work is left empty
      because the target_real path provides the work.

    Non-empty ``configs`` are returned unchanged.
    """
    if configs:
        return configs
    if not is_encdec:
        return [FeatureConfig(name="unconditional", features={})]
    if has_target_real:
        return []
    raise ValueError(
        f"{surface}: encoder-decoder eval has nothing to run — feature_configs "
        "is empty and no target_real configs were provided. Provide target_real "
        "configs (backed by a full context cache) or explicit static "
        "feature_configs."
    )


def resolve_next_segment_conditioning_offset(
    cfg,
    meta: Mapping | None = None,
) -> bool:
    """Resolve explicit/auto next-segment conditioning configuration.

    ``next_segment_conditioning_offset`` accepts:
    - ``true`` / ``false``: explicit eval behavior.
    - ``auto``: read ``meta["user_config"]["next_segment_conditioning_offset"]``.

    When metadata is unavailable, ``auto`` resolves to ``False``. Malformed
    explicit values raise instead of flowing through Python truthiness.
    """
    raw = cfg.get("next_segment_conditioning_offset", False)
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return False
    if not isinstance(raw, str):
        raise TypeError(
            "next_segment_conditioning_offset must be a bool, 'auto', or a "
            f"boolean string; got {type(raw).__name__}"
        )

    normalized = raw.strip().lower()
    if normalized == "auto":
        if meta is None:
            return False
        user_config = meta.get("user_config") or {}
        metadata_raw = user_config.get("next_segment_conditioning_offset", False)
        return _parse_bool(metadata_raw, source="checkpoint metadata")
    return _parse_bool(raw, source="eval config")


def _parse_bool(value, *, source: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise TypeError(
        "next_segment_conditioning_offset in "
        f"{source} must be a bool or boolean string; got {value!r}"
    )


def _load_features_from_yaml(path: str) -> dict[int, int]:
    """Load a ``{feature_id: bucket_index}`` dict from a YAML file.

    The YAML file should contain a flat mapping of integer keys to integer
    values, as produced by ``scripts/extract_common_features.py``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file contents are not a dict.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Feature config file not found: {path}")
    with open(p) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a {{feature_id: bucket_index}} dict in {path}, "
            f"got {type(data).__name__}"
        )
    return {int(k): int(v) for k, v in data.items()}


def _discover_feature_yamls(directory: str) -> list[FeatureConfig]:
    """Recursively discover ``features.yaml`` files under a directory.

    Each file is loaded as a feature config. The config name is derived from
    the directory's own basename plus the file's path relative to the directory,
    excluding the filename itself. For example, calling with directory
    ``out/data/common_features/safety_sft/`` containing
    ``f0.5/m0.5/features.yaml`` produces config name ``safety_sft/f0.5/m0.5``.

    This prefix ensures that configs from different phase directories (e.g.
    ``safety_sft/`` vs ``capabilities_sft/``) produce distinct names.

    Args:
        directory: Root directory to search under.

    Returns:
        List of FeatureConfig instances sorted by name.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If no ``features.yaml`` files are found.
    """
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Feature config directory not found: {directory}")

    yamls = sorted(root.rglob("features.yaml"))
    if not yamls:
        raise ValueError(f"No features.yaml files found under {directory}")

    dir_prefix = root.name
    configs: list[FeatureConfig] = []
    for yaml_path in yamls:
        features = _load_features_from_yaml(str(yaml_path))
        # Name = directory basename / relative path (excluding filename)
        rel = yaml_path.relative_to(root).parent
        if str(rel) != ".":
            name = f"{dir_prefix}/{rel}"
        else:
            name = f"{dir_prefix}/{yaml_path.stem}"
        configs.append(FeatureConfig(name=name, features=features))

    return configs


_COMMON_FEATURES_DIR = "out/data/common_features"


def _name_from_file_path(p: Path) -> str:
    """Derive a unique config name from a feature YAML file path.

    When the path lives under ``out/data/common_features/``, uses the relative
    path (excluding the filename) to produce a name consistent with directory
    discovery — e.g. ``safety_sft/delta_base/.../m0.5``.

    Falls back to the filename stem when the path is outside the common
    features tree (backward compatible).
    """
    resolved = p.resolve()
    try:
        rel = resolved.relative_to(Path(_COMMON_FEATURES_DIR).resolve())
        # Use parent dir path (drop the filename), like _discover_feature_yamls
        parent_rel = rel.parent
        if str(parent_rel) != ".":
            return str(parent_rel)
    except ValueError:
        pass
    return p.stem


def parse_feature_configs(raw_configs) -> list[FeatureConfig]:
    """Parse feature configs from Hydra CLI input.

    Accepts four formats:

    1. ``None`` or empty list — no static feature configs.
    2. A string path to a ``.yaml`` file — loads a ``{feature_id: bucket}``
       dict; config name is the filename stem. The YAML must be a flat
       ``{feature_id: bucket}`` mapping (no ``name`` or ``features`` keys).
       For the structured format that supports ``name`` + ``features`` keys,
       use :func:`parse_strict_feature_configs` instead.
    3. A string path to a directory — recursively discovers all
       ``features.yaml`` files; config names from relative paths
       (e.g. ``f0.5/m0.5``).
    4. A list of dicts with ``name`` and ``features`` keys, or string paths
       (files or directories can be mixed).

    When features are specified inline (format 4), ``features`` must be present.
    Without an explicit ``name``, the config is named ``"custom"``. Resolved names
    must be unique, so more than one unnamed inline config (or any explicit name
    collision) raises rather than silently overwriting earlier configs.

    Returns:
        List of :class:`FeatureConfig` instances.
    """
    # Format 1: null / empty
    if raw_configs is None or (
        isinstance(raw_configs, (list, tuple)) and len(raw_configs) == 0
    ):
        return []

    # Format 2/3: single string path
    if isinstance(raw_configs, str):
        p = Path(raw_configs)
        if p.is_dir():
            return _discover_feature_yamls(raw_configs)
        features = _load_features_from_yaml(raw_configs)
        name = _name_from_file_path(p)
        return [FeatureConfig(name=name, features=features)]

    # Format 4: list of entries (strings, dicts, or mix)
    configs: list[FeatureConfig] = []
    for entry in raw_configs:
        if isinstance(entry, str):
            p = Path(entry)
            if p.is_dir():
                configs.extend(_discover_feature_yamls(entry))
            else:
                features = _load_features_from_yaml(entry)
                name = _name_from_file_path(p)
                configs.append(FeatureConfig(name=name, features=features))
            continue
        if not isinstance(entry, Mapping):
            raise TypeError(
                "feature_configs entries must be YAML paths, directories, or dicts; "
                f"got {type(entry).__name__}"
            )
        _reject_unsupported_config_keys(entry, source="feature_configs")
        if "features" not in entry:
            raise ValueError("feature_configs dict entries must set `features`")
        features_raw = entry.get("features")
        if features_raw is None:
            features = {}
        else:
            features = {int(k): int(v) for k, v in features_raw.items()}
        raw_name = str(entry.get("name") or "custom")
        name = sanitize_config_name(raw_name)
        configs.append(
            FeatureConfig(
                name=name,
                features=features,
                strength=entry.get("strength", 1.0),
            )
        )

    name_indices: dict[str, list[int]] = {}
    for config_index, config in enumerate(configs):
        name_indices.setdefault(config.name, []).append(config_index)
    collisions = {
        name: indices for name, indices in name_indices.items() if len(indices) > 1
    }
    if collisions:
        details = ", ".join(
            f"{name!r} (entries {', '.join(str(index) for index in indices)})"
            for name, indices in collisions.items()
        )
        raise ValueError(
            "feature_configs contains duplicate resolved name(s): "
            f"{details}. Each config must resolve to a unique name because names "
            "identify evaluation output paths."
        )
    return configs


def parse_strict_feature_configs(raw_configs) -> list[FeatureConfig]:
    """Parse explicit target-segment feature configs.

    ``None`` means "no static target-delta variants", and directories are
    rejected. This is deliberate: progression runs should only evaluate YAMLs
    the operator explicitly names.
    """
    if raw_configs is None:
        return []
    if isinstance(raw_configs, str):
        return uniquify_config_names([_load_yaml_config(Path(raw_configs))])
    if isinstance(raw_configs, Mapping):
        return uniquify_config_names([_parse_inline_config(raw_configs, index=0)])
    if isinstance(raw_configs, (list, tuple)):
        configs: list[FeatureConfig] = []
        for i, entry in enumerate(raw_configs):
            if entry is None:
                continue
            if isinstance(entry, str):
                configs.append(_load_yaml_config(Path(entry)))
            elif isinstance(entry, Mapping):
                configs.append(_parse_inline_config(entry, index=i))
            else:
                raise TypeError(
                    "conditioned_configs entries must be YAML paths or dicts; "
                    f"got {type(entry).__name__}"
                )
        return uniquify_config_names(configs)
    raise TypeError(
        "conditioned_configs must be null, a YAML path, a dict, or a list; "
        f"got {type(raw_configs).__name__}"
    )
