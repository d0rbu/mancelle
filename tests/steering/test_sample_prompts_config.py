"""Pin: cfg.sample_prompts list propagates from Hydra YAML to runtime."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def test_default_sample_prompts_list_is_seven():
    config_dir = Path(__file__).resolve().parents[2] / "configs" / "training"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="conditional_encdec_train")
    assert "sample_prompts" in cfg, (
        "conditional_encdec_train.yaml must declare sample_prompts"
    )
    prompts = list(cfg.sample_prompts)
    assert len(prompts) == 7, f"expected 7 default prompts, got {len(prompts)}"
    assert all(isinstance(p, str) for p in prompts)
    assert all(len(p) > 0 for p in prompts)
    assert prompts[0] == "The capital of France is"


def test_sample_prompts_list_overridable_via_hydra():
    config_dir = Path(__file__).resolve().parents[2] / "configs" / "training"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="conditional_encdec_train",
            overrides=["sample_prompts=['hello world','foo bar']"],
        )
    assert list(cfg.sample_prompts) == ["hello world", "foo bar"]


def test_sample_prompts_resolves_to_list_of_strings_after_omegaconf_to_container():
    config_dir = Path(__file__).resolve().parents[2] / "configs" / "training"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="conditional_encdec_train")
    resolved = OmegaConf.to_container(cfg.sample_prompts, resolve=True)
    assert isinstance(resolved, list)
    assert all(isinstance(p, str) for p in resolved)
