"""Slice 14 audit X2 smoke tests for ``scripts/eval/plot_harmfulness_*``.

The three plot scripts each have:

1. A top-level module body whose imports must succeed (no
   side-effects beyond the explicit ``matplotlib.use("Agg")``).
2. A Hydra-decorated ``main(cfg)`` and an extracted ``_main(cfg)`` so
   tests can call into the core logic without a full CLI launch.

These tests verify (a) imports stay clean and (b) the matching Hydra
YAML config loads via ``hydra.compose`` and exposes the keys ``_main``
reads (``base_dir``, ``output_dir``, ``model_tag`` / ``num_harmful``
where applicable).

We intentionally do NOT call ``_main(cfg)`` itself — that requires
on-disk harmfulness JSONL records and would be a slow integration test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
PLOT_CONFIG_DIR = REPO_ROOT / "configs" / "plot"


@pytest.fixture(autouse=True)
def _silence_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the Agg backend before any plot-script import (the modules
    call ``matplotlib.use("Agg")`` themselves but tests run earlier).
    """
    monkeypatch.setenv("MPLBACKEND", "Agg")


def _require_keys(cfg: DictConfig, keys: list[str]) -> None:
    """Helper: assert that every name in ``keys`` resolves to something."""
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    for k in keys:
        assert k in container, f"Hydra config missing key {k!r}; got {list(container)}"


def test_plot_harmfulness_progression_imports() -> None:
    """Module body imports must not raise."""
    import scripts.eval.plot_harmfulness_progression as mod  # noqa: F401

    assert callable(mod._main)
    assert callable(mod.main)


def test_plot_harmfulness_progression_config_loads() -> None:
    """Hydra config compose returns a DictConfig with the keys ``_main`` reads."""
    with initialize_config_dir(
        config_dir=str(PLOT_CONFIG_DIR), version_base=None
    ):
        cfg = compose(config_name="harmfulness_progression")
    _require_keys(cfg, ["model_tag", "base_dir", "output_dir", "num_harmful"])


def test_plot_harmfulness_reversed_progression_imports() -> None:
    """Module body imports must not raise."""
    import scripts.eval.plot_harmfulness_reversed_progression as mod  # noqa: F401

    assert callable(mod._main)
    assert callable(mod.main)


def test_plot_harmfulness_reversed_progression_config_loads() -> None:
    """Hydra config compose returns a DictConfig with the keys ``_main`` reads."""
    with initialize_config_dir(
        config_dir=str(PLOT_CONFIG_DIR), version_base=None
    ):
        cfg = compose(config_name="harmfulness_reversed_progression")
    _require_keys(cfg, ["base_dir", "output_dir"])


def test_plot_harmfulness_standalone_progression_imports() -> None:
    """Module body imports must not raise."""
    import scripts.eval.plot_harmfulness_standalone_progression as mod  # noqa: F401

    assert callable(mod._main)
    assert callable(mod.main)


def test_plot_harmfulness_standalone_progression_config_loads() -> None:
    """Hydra config compose returns a DictConfig with the keys ``_main`` reads."""
    with initialize_config_dir(
        config_dir=str(PLOT_CONFIG_DIR), version_base=None
    ):
        cfg = compose(config_name="harmfulness_standalone_progression")
    _require_keys(cfg, ["base_dir", "output_dir"])
