"""YAML root-config compose smoke test for ``configs/``.

Slice 18 audit map.md §7.2: there is no test that composes EVERY root YAML.
Such a test would have caught the seven configs missing ``_self_`` (P1 #6)
and the malformed ``configs/extraction/experiment.yaml`` (P1 #15).

This test parametrises over every YAML in ``configs/`` and asserts:
1. ``OmegaConf.load(path)`` succeeds — guards basic YAML syntax.
2. If the file is in a known-root location AND is intended as a Hydra
   ``@hydra.main`` ``--config-name`` target, ``defaults`` (when present)
   is a list (not a dict) AND lists ``_self_`` either explicitly OR has
   no defaults at all (per Hydra ≥1.2 contract).

Group / overlay configs (those starting with ``# @package`` or living
inside a ``configs/<root>/<group>/<name>.yaml`` group dir) are checked
only for syntax — they do NOT need a ``defaults: [_self_]`` block.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig, ListConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_ROOT = REPO_ROOT / "configs"


def _all_yaml_configs() -> list[Path]:
    return sorted(p for p in CONFIGS_ROOT.glob("**/*.yaml") if p.is_file())


YAML_FILES = _all_yaml_configs()


@pytest.mark.parametrize("path", YAML_FILES, ids=lambda p: str(p.relative_to(CONFIGS_ROOT)))
def test_yaml_loads(path: Path) -> None:
    """Every YAML in ``configs/`` must parse cleanly."""
    cfg = OmegaConf.load(path)
    assert cfg is not None, f"{path} loaded as None"


@pytest.mark.parametrize("path", YAML_FILES, ids=lambda p: str(p.relative_to(CONFIGS_ROOT)))
def test_yaml_defaults_block_well_formed(path: Path) -> None:
    """If a YAML has a ``defaults:`` block, it must be a list (Hydra
    contract). The legacy ``defaults`` as a dict (e.g. the deleted
    ``experiment.yaml``) is rejected by Hydra and must not exist."""
    text = path.read_text()
    if "\ndefaults:" not in text and not text.startswith("defaults:"):
        return  # No defaults block; nothing to check.

    cfg = OmegaConf.load(path)
    assert isinstance(cfg, DictConfig)
    if "defaults" not in cfg:
        return  # `defaults:` was inside a sub-key, not top-level.
    defaults = cfg.defaults
    assert isinstance(defaults, ListConfig), (
        f"{path.relative_to(CONFIGS_ROOT)}: ``defaults`` must be a list "
        f"(Hydra config defaults syntax), got {type(defaults).__name__}. "
        f"This was the failure mode of the deleted experiment.yaml."
    )


# Configs under these subdirs are GROUP configs (chosen via
# ``defaults: - {group}: {name}`` from a parent), not Hydra @hydra.main
# entry points. They typically use ``# @package`` headers and do not need
# their own ``_self_``.
GROUP_DIRS_BY_ROOT: dict[str, set[str]] = {
    "eval/axbench_base": {"methods", "stages", "tiers"},
    "eval/axbench_bpb": {"tiers"},
    "eval/feature_filter": {"judge"},
    "eval/profile_progression": {"path"},
}


def _is_group_config(path: Path) -> bool:
    """True if path is inside a Hydra config group sub-directory (which
    is composed by parent rather than dispatched directly)."""
    rel = path.relative_to(CONFIGS_ROOT)
    parts = rel.parts
    if len(parts) < 3:
        return False
    parent_rel = "/".join(parts[:-2])
    group_name = parts[-2]
    return group_name in GROUP_DIRS_BY_ROOT.get(parent_rel, set())


@pytest.mark.parametrize("path", YAML_FILES, ids=lambda p: str(p.relative_to(CONFIGS_ROOT)))
def test_root_yaml_with_defaults_lists_self(path: Path) -> None:
    """Root configs with a ``defaults:`` list referencing a parent must
    explicitly list ``_self_``. Slice 18 audit P1 #6 fixed seven configs
    that previously relied on Hydra's implicit ``_self_``-last fallback.

    Group configs (``# @package`` headers, group sub-dirs) are skipped.
    """
    text = path.read_text()
    if _is_group_config(path):
        return
    # `# @package _global_` overlays do not need _self_.
    first_line = text.lstrip().splitlines()[0] if text.strip() else ""
    if first_line.startswith("# @package"):
        return

    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig) or "defaults" not in cfg:
        return  # No defaults block; not in scope.
    defaults = cfg.defaults
    if not isinstance(defaults, ListConfig) or len(defaults) == 0:
        return  # Empty defaults; nothing to enforce.
    # Each defaults entry is either a string (e.g., "_self_") or a dict
    # (e.g., {optional local@_global_: extract_features}).
    has_self = any(
        (isinstance(d, str) and d == "_self_")
        or (isinstance(d, DictConfig) and "_self_" in d)
        for d in defaults
    )
    # If defaults is just [_self_] alone, that's also fine — slice 4 root
    # configs use this form.
    if len(defaults) == 1 and isinstance(defaults[0], str) and defaults[0] == "_self_":
        return
    assert has_self, (
        f"{path.relative_to(CONFIGS_ROOT)}: ``defaults`` references parents "
        f"({list(defaults)!r}) without ``_self_``; Hydra ≥1.2 emits a "
        f"deprecation warning and this configuration's overrides may not "
        f"land where the author intended (slice 18 audit P1 #6)."
    )
