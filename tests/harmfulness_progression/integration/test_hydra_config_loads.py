"""Hydra configs at configs/plot/*.yaml load correctly with all required fields."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# parents[0]=integration/, parents[1]=harmfulness_progression/, parents[2]=tests/,
# .parent = repo root
CONFIG_DIR = Path(__file__).resolve().parents[2].parent / "configs" / "plot"


def _compose(name: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=name)


def test_harmfulness_progression_config_loads() -> None:
    cfg = _compose("harmfulness_progression")
    assert cfg.model_tag == "d12_baseline"
    assert cfg.base_dir == "nanochat_cache"
    assert cfg.output_dir is None
    assert int(cfg.num_harmful) == 700


def test_harmfulness_standalone_progression_config_loads() -> None:
    cfg = _compose("harmfulness_standalone_progression")
    assert cfg.base_dir == "nanochat_cache"
    assert cfg.output_dir == "fig/harmfulness_standalone_progression"
    assert int(cfg.num_harmful) == 700
    assert int(cfg.min_step) == 53000
    assert cfg.d13_progression_path == "out/progression/d13_base/results.json"
    assert cfg.d12_progression_path == "out/progression/d12_baseline/results.json"
    assert int(cfg.last_pretrain_step) == 54600
    assert cfg.model_tag == "d12_baseline"


def test_harmfulness_reversed_progression_config_loads() -> None:
    cfg = _compose("harmfulness_reversed_progression")
    assert cfg.model_tag == "d12_baseline"
    assert cfg.base_dir == "nanochat_cache"
    assert cfg.output_dir is None
    assert int(cfg.num_harmful) == 50


def test_plot_config_num_harmful_matches_reversed_eval_config() -> None:
    """The reverse plot config's num_harmful must match the harm_num_harmful
    in the reversed training_progression config. A mismatch (e.g., bumping
    one without the other) silently produces blank reverse-plot charts."""
    plot_cfg = _compose("harmfulness_reversed_progression")
    GlobalHydra.instance().clear()
    eval_dir = Path(__file__).resolve().parents[2].parent / "configs" / "eval"
    with initialize_config_dir(config_dir=str(eval_dir), version_base=None):
        eval_cond = compose(config_name="training_progression_d12_baseline_reversed")
    GlobalHydra.instance().clear()

    assert int(plot_cfg.num_harmful) == int(eval_cond.harm_num_harmful), (
        f"plot config num_harmful ({plot_cfg.num_harmful}) must match "
        f"reversed cond eval harm_num_harmful ({eval_cond.harm_num_harmful}); "
        "a mismatch silently drops all records from the reverse-plot."
    )
