"""The calibrated presets retain the intended model and execution scopes."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from scripts.eval.axbench_adv.pressures import Pressure, parse_pressures, validate_magnitudes
from scripts.eval.axbench_adv.run import build_config
from steering.eval.axbench_adv.execution_scope import parse_method_filter
from steering.eval.axbench_adv.native_counter import _point_selection
from steering.eval.axbench_adv.types import ALL_METHODS

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs/eval/axbench_adv"


def _compose(name: str, overrides: list[str] | None = None) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        return compose(config_name=name, overrides=overrides or [], return_hydra_config=True)


def test_encdec_preset_selects_all_three_project_methods_without_hf_generation() -> None:
    overrides = [
        f"source.{field}=/source/{field}"
        for field in (
            "prepared_dir",
            "main_parquet",
            "aux_parquet",
            "evaluated_jsonl",
            "project_parquet",
            "target_meta",
            "target_receipt",
            "feature_statistics",
        )
    ]
    overrides.extend(
        f"source.nl_attachment.paths.{field}=/attachment/{field}"
        for field in ("project_parquet", "project_manifest", "evaluated_jsonl")
    )
    overrides.extend(
        [
            "project_counter_direction_root=/fits",
            "output_dir=/run/output",
            "judge_cache=/run/judge.sqlite",
            "reflection_cache=/run/reflection.sqlite",
        ]
    )
    cfg = _compose("encdec400m_sft_latent_c50", overrides)
    environ = dict(cfg.hydra.job.env_set)
    del cfg.hydra
    config = build_config(cfg)
    methods = parse_method_filter(ALL_METHODS, environ=environ)
    assert set(methods) == {"SelfConditioning", "SelfConditioningMasked", "NLSteering"}
    assert set(parse_pressures(config.pressures, environ=environ)) == {
        Pressure.DIFFMEAN_COUNTER,
        Pressure.LINEAR_PROBE_COUNTER,
    }
    assert set(config.factor_overrides) == set(methods)
    assert set(config.factor_overrides.values()) == {6.0}
    assert config.source_expectations.project_target_step == 1013
    assert config.source_expectations.max_concepts == 50
    assert config.project_counter_layer == 9
    assert config.nl_attachment_expectations is not None
    assert config.nl_attachment_expectations.max_concepts == 100
    assert validate_magnitudes(config.counter_magnitudes) == (
        256,
        1024,
        4096,
        8192,
        16384,
        32768,
        65536,
    )


def test_d14_presets_share_grid_and_bind_actual_sft_geometry() -> None:
    prompt = _compose("native_counter_d14_sft_latent_c50")
    methods = _compose("native_methods_d14_sft_latent_c50")
    grid = validate_magnitudes(prompt.magnitudes)
    assert grid == validate_magnitudes(methods.counter_magnitudes)
    assert grid == (64, 256, 512, 1024, 2048, 4096)
    assert prompt.step == prompt.expected.step == methods.step == 1001
    assert (prompt.expected.n_layer, prompt.expected.n_embd, prompt.layer) == (14, 896, 11)
    assert prompt.max_concepts == 100 and prompt.concept_count == 50
    assert _point_selection(prompt, grid) == (("DiffMean", "LinearProbe"), grid)
    assert set(methods.methods) == {"DiffMean", "LinearProbe", "LoRA"}
    assert set(parse_pressures(methods.pressures, environ={})) == {
        Pressure.DIFFMEAN_COUNTER,
        Pressure.LINEAR_PROBE_COUNTER,
    }
    assert methods.zero_control_source == "archived"
    assert not OmegaConf.is_missing(prompt.expected, "model_sha256")
    assert OmegaConf.is_missing(prompt, "checkpoint_dir")
    assert OmegaConf.is_missing(methods, "final_root")
