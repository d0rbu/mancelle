"""Regression tests for training progression eval config cadence."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra.errors import MissingConfigException

from nanochat.checkpoint_manager import _SEGMENT_DIRS

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_CONFIG_DIR = REPO_ROOT / "configs" / "eval"

SPARSE_4K_CADENCE_CONFIGS = [
    "training_progression_d12_base",
    "training_progression_d12_fp",
    "training_progression_d12_fp10",
    "training_progression_d13_base",
]
ALL_PROGRESSION_CONFIGS = sorted(
    path.stem for path in EVAL_CONFIG_DIR.glob("training_progression*.yaml")
)


def _pretrain_steps(cfg) -> list[int]:
    for phase in cfg.phases:
        if phase.phase == "pretrain":
            return [int(step) for step in phase.steps]
    raise AssertionError("pretrain phase missing")


def _phase(cfg, name: str):
    for phase in cfg.phases:
        if phase.phase == name:
            return phase
    raise AssertionError(f"{name} phase missing")


def _conditioned_config_names(cfg) -> list[str]:
    return [c.name for c in cfg.conditioning.conditioned_configs]


def test_issue_1338_judge_defaults_are_provider_qualified(hydra_compose) -> None:
    progression = hydra_compose(
        str(EVAL_CONFIG_DIR),
        "training_progression",
    )
    standalone = hydra_compose(
        str(EVAL_CONFIG_DIR),
        "harmfulness_eval",
    )
    assert progression.harm_judge_model == "openai/gpt-5.6-luna"
    assert standalone.judge_model == "openai/gpt-5.6-luna"


def test_issue_1338_judge_defaults_remain_explicitly_overridable(
    hydra_compose,
) -> None:
    progression = hydra_compose(
        str(EVAL_CONFIG_DIR),
        "training_progression",
        ["harm_judge_model=openai/test-override"],
    )
    standalone = hydra_compose(
        str(EVAL_CONFIG_DIR),
        "harmfulness_eval",
        ["judge_model=openai/test-override"],
    )
    assert progression.harm_judge_model == "openai/test-override"
    assert standalone.judge_model == "openai/test-override"


@pytest.mark.parametrize("profile_path", ["A.yaml", "C.yaml"])
def test_issue_1338_profile_progression_paths_use_same_judge_default(
    profile_path: str,
) -> None:
    text = (EVAL_CONFIG_DIR / "profile_progression" / "path" / profile_path).read_text(
        encoding="utf-8"
    )
    assert 'harm_judge_model: "openai/gpt-5.6-luna"' in text
    assert "gpt-5-nano" not in text


@pytest.mark.parametrize("config_name", ALL_PROGRESSION_CONFIGS)
def test_progression_configs_use_current_checkpoint_routing_schema(
    hydra_compose,
    config_name: str,
) -> None:
    cfg = hydra_compose(str(EVAL_CONFIG_DIR), config_name)

    assert cfg.model_family in {"auto", "decoder", "encoder_decoder"}
    assert isinstance(cfg.checkpoint_root, str) and cfg.checkpoint_root
    assert isinstance(cfg.checkpoint_prefix, str)
    assert "source" not in cfg


@pytest.mark.parametrize("config_name", SPARSE_4K_CADENCE_CONFIGS)
def test_progression_configs_use_sparse_4k_pretrain_cadence(
    hydra_compose,
    config_name: str,
) -> None:
    cfg = hydra_compose(str(EVAL_CONFIG_DIR), config_name)

    assert _pretrain_steps(cfg) == [
        4000,
        8000,
        12000,
        16000,
        20000,
        24000,
        28000,
        32000,
        36000,
        40000,
        44000,
        48000,
        52000,
        54600,
    ]


def test_d12_baseline_conditioned_progression_config_shape(hydra_compose) -> None:
    cfg = hydra_compose(str(EVAL_CONFIG_DIR), "training_progression_d12_baseline")

    assert _pretrain_steps(cfg) == [52000, 53000, 54000, 54600]
    assert cfg.model_tag == "d12_baseline"
    assert cfg.model_family == "encoder_decoder"
    assert cfg.checkpoint_root == "conditional_encdec_checkpoints"
    assert cfg.evals == "core,chat,harmfulness"
    assert cfg.results_file == "out/progression/d12_baseline/results.json"
    assert cfg.fig_dir == "fig/progression/d12_baseline"

    assert "run_unconditioned" not in cfg.conditioning
    assert "context_mode" not in cfg.conditioning
    assert _conditioned_config_names(cfg) == [
        "Safety delta-base",
        "Capability delta-base",
    ]

    safety = _phase(cfg, "safety_sft")
    cap = _phase(cfg, "capabilities_sft")
    assert safety.subdir == _SEGMENT_DIRS["safety_cond"]
    assert cap.subdir == (f"{_SEGMENT_DIRS['safety_cond']}/{_SEGMENT_DIRS['capabilities_cond']}")
    assert list(safety.steps) == [886]
    assert list(cap.steps) == [1106]


def test_d12_baseline_reversed_conditioned_config_shape(hydra_compose) -> None:
    cfg = hydra_compose(str(EVAL_CONFIG_DIR), "training_progression_d12_baseline_reversed")

    assert _pretrain_steps(cfg) == [52000, 53000, 54000, 54600]
    assert cfg.model_tag == "d12_baseline"
    assert cfg.model_family == "encoder_decoder"
    assert cfg.evals == "core,chat,harmfulness"
    assert int(cfg.num_seeds) == 3
    assert int(cfg.num_gpus) == 1
    assert cfg.results_file == "out/progression/d12_baseline_reversed/results.json"
    assert cfg.fig_dir == "fig/progression/d12_baseline_reversed"

    assert "run_unconditioned" not in cfg.conditioning
    assert "context_mode" not in cfg.conditioning
    assert _conditioned_config_names(cfg) == [
        "Safety delta-base",
        "Capability delta-base",
    ]

    cap = _phase(cfg, "capabilities_sft")
    safety = _phase(cfg, "safety_sft")
    assert cap.subdir == _SEGMENT_DIRS["capabilities_cond"]
    assert safety.subdir == (f"{_SEGMENT_DIRS['capabilities_cond']}/{_SEGMENT_DIRS['safety_cond']}")
    assert list(cap.steps) == [1104]
    assert list(safety.steps) == [886]


@pytest.mark.parametrize(
    "config_name",
    [
        "training_progression_d12_baseline_uncond",
        "training_progression_d12_baseline_reversed_uncond",
    ],
)
def test_d12_baseline_unconditioned_progression_configs_are_removed(
    hydra_compose,
    config_name: str,
) -> None:
    with pytest.raises(MissingConfigException):
        hydra_compose(str(EVAL_CONFIG_DIR), config_name)
