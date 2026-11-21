"""Unit tests for BPB-Adv config fields."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from steering.eval.axbench_base.profiler import ProfileConfig
from steering.eval.axbench_bpb.types import SteeringMode
from steering.eval.axbench_bpb_adv.attack_rows import (
    BpbAttackPrefix,
    BpbAttackWorkSpec,
    prepare_bpb_attack_rows,
)
from steering.eval.axbench_bpb_adv.config import (
    AdditiveDirectionConfig,
    AxBenchBpbAdvConfig,
    FasterGcgConfig,
    GeneralValBpbConfig,
    OursGcgConfig,
    RoundtripGenerationConfig,
    ScatterBaselineX,
    SteeringMetric,
    validate_all_stage_representative_target_contract,
    validate_config,
)
from steering.eval.axbench_bpb_adv.config_loader import (
    _parse_adversarial_method_config,
    cfg_to_dataclass,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.roundtrip_steering import RoundtripEvalConfig
from steering.sae_resolver import L0Tier

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "eval" / "axbench_bpb_adv"


def _valid_config(**overrides: object) -> AxBenchBpbAdvConfig:
    base = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


# --------------------------------------------------------------------------- #
# Defaults                                                                     #
# --------------------------------------------------------------------------- #
def test_new_config_fields_default_values() -> None:
    cfg = AxBenchBpbAdvConfig()
    assert not hasattr(cfg, "pairing_policy")
    assert cfg.pair_subsample == 1
    assert cfg.selection_splice_batch_size is None
    assert cfg.selection_manifest_path is None
    assert cfg.steering_metric == "roundtrip"
    assert cfg.roundtrip_generation == RoundtripGenerationConfig()
    assert cfg.roundtrip_generation.samples_per_candidate == 4
    assert cfg.roundtrip_steering.enabled is True
    assert cfg.scatter_baseline_x == "general_val_bpb"
    assert cfg.plot_steering_strength_families is False
    assert cfg.feature_extraction_model_id is None


def test_clustered_robustness_pins_source_contract_selection_manifest() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name=(
                "axbench_bpb_adv_gemma3_4bpt_l22_f65k_clustered_robustness"
            )
        )

    assert cfg.selection_manifest_path.endswith(
        "/c1024/sall/sel/"
        "e9e395739f1ac4c9498f0935e2e48a407ae9eb13b0134904d70f7ec64a3d5be3/"
        "manifest.json"
    )


def test_feature_extraction_model_id_round_trips_through_yaml_and_validation() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="axbench_bpb_adv_eval",
            overrides=[
                "source_run_dir=/tmp/source",
                "feature_dir=/tmp/features",
                "feature_statistics_path=/tmp/stats.safetensors",
                "feature_extraction_model_id=google/gemma-3-4b-pt",
                "roundtrip_steering.probe.model_name=google/gemma-3-4b-pt",
            ],
        )

    parsed = cfg_to_dataclass(cfg)
    validate_config(parsed)

    assert parsed.feature_extraction_model_id == "google/gemma-3-4b-pt"
    assert parsed.roundtrip_steering.probe.model_name == "google/gemma-3-4b-pt"


@pytest.mark.parametrize("model_id", ["", "   "])
def test_feature_extraction_model_id_rejects_empty_strings(model_id: str) -> None:
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match="feature_extraction_model_id",
    ):
        AxBenchBpbAdvConfig(
            feature_extraction_model_id=model_id,
            steering_metric="bpb_class_margin",
        )


@pytest.mark.parametrize("model_id", [None, "google/gemma-3-4b-pt"])
def test_feature_extraction_model_id_accepts_null_or_non_empty_string(
    model_id: str | None,
) -> None:
    validate_config(
        _valid_config(
            feature_extraction_model_id=model_id,
            steering_metric="bpb_class_margin",
        )
    )


def test_research_eval_config_uses_feature_diffmean_and_gcg_methods() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="axbench_bpb_adv_eval",
            overrides=[
                "source_run_dir=/tmp/source",
                "feature_dir=/tmp/features",
                "feature_statistics_path=/tmp/stats.safetensors",
            ],
        )
    parsed = cfg_to_dataclass(cfg)
    base_methods = set(parsed.base_methods or [])
    assert "feature_diffmean_encdec" in base_methods
    assert "neg_feature_diffmean_encdec" in base_methods
    assert "ours_gcg_encdec" in base_methods
    assert "neg_ours_gcg_encdec" in base_methods
    assert "ours_replace_all_encdec" not in base_methods
    assert "neg_ours_replace_all_negatives_encdec" not in base_methods


def test_profile_block_round_trips_through_cfg_to_dataclass() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="axbench_bpb_adv_eval",
            overrides=[
                "source_run_dir=/tmp/source",
                "feature_dir=/tmp/features",
                "feature_statistics_path=/tmp/stats.safetensors",
                "+profile.enabled=true",
                "+profile.output_dir=/tmp/profile",
                "+profile.gpu_poll_interval_ms=25",
            ],
        )
    parsed = cfg_to_dataclass(cfg)
    assert isinstance(parsed.profile, ProfileConfig)
    assert parsed.profile.enabled is True
    assert parsed.profile.output_dir == "/tmp/profile"
    assert parsed.profile.gpu_poll_interval_ms == 25


def test_invalid_profile_domain_is_wrapped_by_config_loader() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="axbench_bpb_adv_eval",
            overrides=[
                "source_run_dir=/tmp/source",
                "feature_dir=/tmp/features",
                "feature_statistics_path=/tmp/stats.safetensors",
                "+profile.gpu_poll_interval_ms=0",
            ],
        )
    with pytest.raises(AxBenchBpbAdvError, match="invalid profile config") as exc_info:
        cfg_to_dataclass(cfg)
    assert isinstance(
        exc_info.value.__cause__,
        BeartypeCallHintParamViolation,
    )


def test_pairing_policy_override_is_rejected_as_unknown_config() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="axbench_bpb_adv_eval",
            overrides=[
                "source_run_dir=/tmp/source",
                "feature_dir=/tmp/features",
                "feature_statistics_path=/tmp/stats.safetensors",
                "+pairing_policy=warn",
            ],
        )
    with pytest.raises(AxBenchBpbAdvError, match="pairing_policy"):
        cfg_to_dataclass(cfg)


def test_timing_probe_config_has_no_representative_selector_field() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="axbench_bpb_adv_timing_probe",
            overrides=[
                "source_run_dir=/tmp/source",
                "feature_dir=/tmp/features",
                "feature_statistics_path=/tmp/stats.safetensors",
            ],
        )
    parsed = cfg_to_dataclass(cfg)
    assert not hasattr(parsed, "representative_coefficient_selection")
    assert parsed.base_methods == ["diffmean_decoder_only"]
    assert parsed.representative_target_methods == ["feature_diffmean_encdec"]
    assert parsed.target_models == ["decoder_only", "encdec"]


def test_generate_samples_config_has_no_representative_selector_field() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="generate_samples",
            overrides=[
                "source_run_dir=/tmp/source",
                "feature_dir=/tmp/features",
                "feature_statistics_path=/tmp/stats.safetensors",
            ],
        )
    assert "representative_coefficient_selection" not in cfg


def test_cfg_to_dataclass_rejects_removed_representative_selector_key() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="axbench_bpb_adv_eval",
            overrides=[
                "source_run_dir=/tmp/source",
                "feature_dir=/tmp/features",
                "feature_statistics_path=/tmp/stats.safetensors",
                "+representative_coefficient_selection=max_control_steerability",
            ],
        )
    with pytest.raises(AxBenchBpbAdvError, match="representative_coefficient_selection"):
        cfg_to_dataclass(cfg)


def test_representative_target_methods_validate_success_path() -> None:
    validate_config(
        _valid_config(
            base_methods=["diffmean_decoder_only"],
            representative_target_methods=["feature_diffmean_encdec"],
            target_models=["decoder_only", "encdec"],
        )
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"representative_target_methods": ["feature_diffmean_roofline_encdec"]},
            "supports only fair",
        ),
        (
            {
                "representative_target_methods": [
                    "feature_diffmean_encdec",
                    "feature_diffmean_encdec",
                ]
            },
            "duplicate",
        ),
        (
            {
                "representative_target_methods": ["feature_diffmean_encdec"],
                "target_models": ["decoder_only"],
            },
            "requires target_models",
        ),
        (
            {
                "base_methods": ["feature_diffmean_encdec"],
                "representative_target_methods": ["feature_diffmean_encdec"],
            },
            "must not duplicate base_methods",
        ),
    ],
)
def test_representative_target_methods_validate_guardrails(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(AxBenchBpbAdvError, match=message):
        validate_config(_valid_config(**overrides))


def test_steering_metric_is_a_strenum_with_roundtrip_default() -> None:
    assert SteeringMetric.ROUNDTRIP == "roundtrip"
    assert SteeringMetric.BPB_CLASS_MARGIN == "bpb_class_margin"
    assert SteeringMetric.coerce("roundtrip") is SteeringMetric.ROUNDTRIP
    assert SteeringMetric.coerce("bpb_class_margin") is SteeringMetric.BPB_CLASS_MARGIN
    with pytest.raises(AxBenchBpbAdvError, match="steering_metric"):
        SteeringMetric.coerce("loss_diff")


def test_roundtrip_metric_requires_enabled_roundtrip_config() -> None:
    cfg = _valid_config()
    cfg.roundtrip_steering = cfg.roundtrip_steering.__class__(enabled=False)
    with pytest.raises(AxBenchBpbAdvError, match="roundtrip_steering.enabled"):
        validate_config(cfg)


def test_bpb_metric_allows_disabled_roundtrip_config() -> None:
    cfg = _valid_config(steering_metric="bpb_class_margin")
    cfg.roundtrip_steering = cfg.roundtrip_steering.__class__(enabled=False)
    validate_config(cfg)


def test_feature_diffmean_extraction_batch_size_validates() -> None:
    validate_config(_valid_config(feature_diffmean_extraction_batch_size=1))
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match="feature_diffmean_extraction_batch_size",
    ):
        AxBenchBpbAdvConfig(feature_diffmean_extraction_batch_size=0)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_selection_splice_batch_size_requires_positive_strict_int(value: object) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="selection_splice_batch_size"):
        AxBenchBpbAdvConfig(
            selection_splice_batch_size=cast(Any, value),
            steering_metric="bpb_class_margin",
        )


def test_selection_splice_batch_size_accepts_positive_int() -> None:
    validate_config(
        _valid_config(
            selection_splice_batch_size=1,
            steering_metric="bpb_class_margin",
        )
    )


@pytest.mark.parametrize("value", ["", "   ", 7])
def test_selection_manifest_path_requires_non_empty_string(value: object) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="selection_manifest_path"):
        AxBenchBpbAdvConfig(
            selection_manifest_path=cast(Any, value),
            steering_metric="bpb_class_margin",
        )


def test_selection_manifest_path_accepts_non_empty_string() -> None:
    validate_config(
        _valid_config(
            selection_manifest_path="/tmp/selection-manifest.json",
            steering_metric="bpb_class_margin",
        )
    )


def test_roundtrip_metric_requires_probe_model_to_match_effective_extraction_model() -> None:
    cfg = _valid_config()
    cfg.roundtrip_steering = RoundtripEvalConfig(
        enabled=True,
        probe=cfg.roundtrip_steering.probe.__class__(model_name="gemma-2-9b"),
    )
    with pytest.raises(AxBenchBpbAdvError, match="probe model_name"):
        validate_config(cfg)


def test_roundtrip_metric_accepts_probe_matching_feature_extraction_model_id() -> None:
    cfg = _valid_config(feature_extraction_model_id="google/gemma-3-4b-pt")
    cfg.roundtrip_steering = RoundtripEvalConfig(
        enabled=True,
        probe=cfg.roundtrip_steering.probe.__class__(
            model_name="google/gemma-3-4b-pt"
        ),
    )

    validate_config(cfg)


def test_roundtrip_metric_rejects_probe_matching_only_neuronpedia_model_id() -> None:
    cfg = _valid_config(feature_extraction_model_id="google/gemma-3-4b-pt")
    assert cfg.roundtrip_steering.probe.model_name == cfg.neuronpedia_model_id

    with pytest.raises(
        AxBenchBpbAdvError,
        match="probe model_name must match feature_extraction_model_id",
    ):
        validate_config(cfg)


@pytest.mark.parametrize("raw_l0", [42, "42"])
def test_roundtrip_metric_rejects_numeric_probe_l0(raw_l0: int | str) -> None:
    cfg = _valid_config()
    cfg.roundtrip_steering = RoundtripEvalConfig(
        enabled=True,
        probe=cfg.roundtrip_steering.probe.__class__(l0=cast(Any, raw_l0)),
    )
    with pytest.raises(AxBenchBpbAdvError, match="numeric.*probe.l0"):
        validate_config(cfg)


@pytest.mark.parametrize(
    ("probe_l0", "neuronpedia_source"),
    [
        (L0Tier.small, "12-gemmascope-2-res-65k-small"),
        (L0Tier.medium, "8-gemmascope-2-res-16k-medium"),
        (
            L0Tier.medium_seed_1,
            "12-gemmascope-2-res-262k-medium_seed_1",
        ),
        (L0Tier.big, "8-gemmascope-2-res-16k-big"),
        (None, "8-gemmascope-res-16k"),
    ],
)
def test_roundtrip_metric_accepts_probe_l0_tiers_and_legacy_null(
    probe_l0: L0Tier | None,
    neuronpedia_source: str,
) -> None:
    cfg = _valid_config(neuronpedia_source=neuronpedia_source)
    cfg.roundtrip_steering = RoundtripEvalConfig(
        enabled=True,
        probe=cfg.roundtrip_steering.probe.__class__(l0=probe_l0),
    )

    validate_config(cfg)


def test_roundtrip_metric_rejects_probe_l0_disagreeing_with_source_tier() -> None:
    cfg = _valid_config(neuronpedia_source="8-gemmascope-2-res-16k-medium")
    cfg.roundtrip_steering = RoundtripEvalConfig(
        enabled=True,
        probe=cfg.roundtrip_steering.probe.__class__(l0=L0Tier.big),
    )

    with pytest.raises(AxBenchBpbAdvError) as exc_info:
        validate_config(cfg)

    message = str(exc_info.value)
    assert "probe='big'" in message
    assert "source='medium'" in message


def test_roundtrip_metric_rejects_tiered_probe_for_v1_source() -> None:
    cfg = _valid_config(neuronpedia_source="8-gemmascope-res-16k")
    cfg.roundtrip_steering = RoundtripEvalConfig(
        enabled=True,
        probe=cfg.roundtrip_steering.probe.__class__(l0=L0Tier.medium),
    )

    with pytest.raises(AxBenchBpbAdvError) as exc_info:
        validate_config(cfg)

    message = str(exc_info.value)
    assert "probe='medium'" in message
    assert "source=None" in message


def test_roundtrip_metric_wraps_unparseable_source_for_tiered_probe() -> None:
    cfg = _valid_config(neuronpedia_source="not-a-neuronpedia-source")
    cfg.roundtrip_steering = RoundtripEvalConfig(
        enabled=True,
        probe=cfg.roundtrip_steering.probe.__class__(l0=L0Tier.medium),
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match=(
            "could not validate roundtrip probe l0 tier against "
            "neuronpedia_source='not-a-neuronpedia-source'"
        ),
    ):
        validate_config(cfg)


def test_roundtrip_l0_tier_is_reachable_through_composed_config() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="axbench_bpb_adv_eval",
            overrides=[
                "source_run_dir=/tmp/source",
                "feature_dir=/tmp/features",
                "feature_statistics_path=/tmp/stats.safetensors",
                "neuronpedia_source=8-gemmascope-2-res-16k-medium",
                "roundtrip_steering.probe.l0=medium",
            ],
        )

    parsed = cfg_to_dataclass(cfg)
    validate_config(parsed)

    assert parsed.roundtrip_steering.probe.l0 is L0Tier.medium
    assert parsed.neuronpedia_source == "8-gemmascope-2-res-16k-medium"

def test_roundtrip_generation_validation_rejects_bad_values() -> None:
    bad_constructors = [
        lambda: RoundtripGenerationConfig(samples_per_candidate=0),
        lambda: RoundtripGenerationConfig(temperature=math.nan),
        lambda: RoundtripGenerationConfig(temperature=-0.1),
        lambda: RoundtripGenerationConfig(top_k_sampling=0),
    ]
    for construct in bad_constructors:
        with pytest.raises(BeartypeCallHintParamViolation):
            construct()


def test_faster_gcg_universal_minibatch_size_default_is_sixteen() -> None:
    # The universal per-concept optimizer deals a minibatch of pairs per step;
    # the default population is 16 (design DECISION C). Existing GCG knobs keep
    # their defaults.
    cfg = FasterGcgConfig()
    assert cfg.universal_minibatch_size == 16


def test_faster_gcg_max_steps_default_is_256() -> None:
    # The universal per-concept trigger is shared across a whole concept's
    # prefixes, so it gets a larger optimization budget than the old per-prefix
    # bespoke trigger (default bumped 64 -> 256; Phase 6).
    cfg = FasterGcgConfig()
    assert cfg.max_steps == 256


def test_faster_gcg_candidate_eval_batch_defaults_to_global_batch() -> None:
    cfg = FasterGcgConfig()
    assert cfg.candidate_eval_device_batch_size is None


def test_removed_candidate_eval_mode_is_rejected() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="candidate_eval_mode"):
        _parse_adversarial_method_config(
            {"method": "faster_gcg", "candidate_eval_mode": "full_sequence"}
        )


@pytest.mark.parametrize("value", [1, 16, 32, None])
def test_candidate_eval_device_batch_size_accepts_positive_or_null(
    value: int | None,
) -> None:
    assert (
        FasterGcgConfig(candidate_eval_device_batch_size=value).candidate_eval_device_batch_size
        == value
    )


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "32"])
def test_candidate_eval_device_batch_size_rejects_invalid_values(value: Any) -> None:
    with pytest.raises(
        BeartypeCallHintParamViolation, match="candidate_eval_device_batch_size"
    ):
        FasterGcgConfig(candidate_eval_device_batch_size=value)


def test_faster_gcg_settings_round_trip_through_cfg_to_dataclass() -> None:
    """The two universal-optimizer knobs survive the real hydra ->
    ``cfg_to_dataclass`` parse: the committed eval yaml pins ``max_steps: 256``
    and ``universal_minibatch_size: 16`` (the new dataclass defaults), and they
    round-trip to those exact values. Pins L1 -- the dataclass default change must
    propagate to the shipped config, not just the in-memory default."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="axbench_bpb_adv_eval",
            overrides=[
                "source_run_dir=/tmp/source",
                "feature_dir=/tmp/features",
                "feature_statistics_path=/tmp/stats.safetensors",
            ],
        )
    dataclass_cfg = cfg_to_dataclass(cfg)
    faster_gcg = next(
        method
        for method in dataclass_cfg.adversarial_methods
        if isinstance(method, FasterGcgConfig)
    )
    assert faster_gcg.max_steps == 256
    assert faster_gcg.universal_minibatch_size == 16
    assert faster_gcg.candidate_eval_device_batch_size is None
    assert dataclass_cfg.plot_steering_strength_families is False


def test_faster_gcg_candidate_eval_batch_hydra_override_round_trips() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="axbench_bpb_adv_eval",
            overrides=[
                "source_run_dir=/tmp/source",
                "feature_dir=/tmp/features",
                "feature_statistics_path=/tmp/stats.safetensors",
                "adversarial_methods.0.candidate_eval_device_batch_size=32",
            ],
        )
    parsed = cfg_to_dataclass(cfg)
    faster_gcg = next(
        method
        for method in parsed.adversarial_methods
        if isinstance(method, FasterGcgConfig)
    )
    assert faster_gcg.candidate_eval_device_batch_size == 32


def test_strength_sweep_config_enables_plot_family() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="axbench_bpb_adv_strength_sweep",
            overrides=[
                "source_run_dir=/tmp/source",
                "feature_dir=/tmp/features",
                "feature_statistics_path=/tmp/stats.safetensors",
            ],
        )
    dataclass_cfg = cfg_to_dataclass(cfg)
    validate_config(dataclass_cfg)

    assert dataclass_cfg.plot_steering_strength_families is True
    assert dataclass_cfg.general_val_bpb.enabled is True
    assert dataclass_cfg.general_val_bpb.n_documents == 20
    assert len(dataclass_cfg.adversarial_methods) == 1
    [additive] = dataclass_cfg.adversarial_methods
    assert isinstance(additive, AdditiveDirectionConfig)
    assert tuple(additive.magnitudes) == (0.0,)
    assert tuple(additive.families) == ("diffmean", "linear_probe", "ours")


def test_ours_gcg_block_coerces_to_dataclass_through_cfg_to_dataclass() -> None:
    """A nested ``ours_gcg`` yaml block round-trips to an ``OursGcgConfig``
    dataclass (not a bare dict) via the real hydra -> ``cfg_to_dataclass`` parse.

    Regression: the loader coerced ``context_cache`` / ``general_val_bpb`` but
    OMITTED ``ours_gcg``, so ``cfg.ours_gcg`` stayed a plain dict and the gcg
    coefficient-grid fn raised ``'dict' object has no attribute 'reg_lambda_grid'``
    at ``prepare_attack_rows`` the first time a real config ran a feature-GCG method.
    """
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="axbench_bpb_adv_eval",
            overrides=[
                "source_run_dir=/tmp/source",
                "feature_dir=/tmp/features",
                "feature_statistics_path=/tmp/stats.safetensors",
                "+ours_gcg.reg_lambda_grid=[0.25,0.5]",
                "+ours_gcg.max_steps=32",
                "+ours_gcg.top_k_pos=7",
            ],
        )
    parsed = cfg_to_dataclass(cfg)
    assert isinstance(parsed.ours_gcg, OursGcgConfig)
    # Hydra's list container is normalized to a tuple; element values are not coerced.
    assert parsed.ours_gcg.reg_lambda_grid == (0.25, 0.5)
    assert parsed.ours_gcg.max_steps == 32
    assert parsed.ours_gcg.top_k_pos == 7


def test_faster_gcg_universal_minibatch_size_must_be_positive() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="universal_minibatch_size"):
        FasterGcgConfig(universal_minibatch_size=0)


@pytest.mark.parametrize("value", [0, 1, 2, 16])
def test_pair_subsample_accepts_non_negative(value: int) -> None:
    validate_config(_valid_config(pair_subsample=value))


def test_pair_subsample_rejects_negative() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="pair_subsample"):
        AxBenchBpbAdvConfig(pair_subsample=-1)


@pytest.mark.parametrize("value", ["baseline_loss", "min", "general_val_bpb"])
def test_scatter_baseline_x_accepts_valid_values(value: str) -> None:
    cfg = _valid_config(scatter_baseline_x=value)
    if value == "general_val_bpb":
        cfg.general_val_bpb = GeneralValBpbConfig(enabled=True)
        # general_val + explicit roofline precomputed-bag methods is rejected at
        # config time; scope to a non-roofline method.
        cfg.base_methods = ["diffmean_encdec"]
    validate_config(cfg)


def test_scatter_baseline_x_rejects_unknown_value() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="scatter_baseline_x"):
        validate_config(_valid_config(scatter_baseline_x="rich"))


def test_scatter_baseline_x_is_a_strenum_with_coerce() -> None:
    """`scatter_baseline_x` is a real StrEnum (single source of truth for the
    allowed set), matching the sibling Stage / ContextSignalMode / AdversarialMethod
    enums in the same module."""
    assert ScatterBaselineX.BASELINE_LOSS == "baseline_loss"
    assert ScatterBaselineX.MIN == "min"
    assert ScatterBaselineX.GENERAL_VAL_BPB == "general_val_bpb"
    assert ScatterBaselineX.coerce("baseline_loss") is ScatterBaselineX.BASELINE_LOSS
    assert ScatterBaselineX.coerce("min") is ScatterBaselineX.MIN
    assert ScatterBaselineX.coerce("general_val_bpb") is ScatterBaselineX.GENERAL_VAL_BPB
    assert ScatterBaselineX.coerce(ScatterBaselineX.MIN) is ScatterBaselineX.MIN
    with pytest.raises(AxBenchBpbAdvError, match="scatter_baseline"):
        ScatterBaselineX.coerce("rich")


# --------------------------------------------------------------------------- #
# plot_methods (plot-stage method whitelist; gates plotting, not scoring)      #
# --------------------------------------------------------------------------- #
def test_plot_methods_defaults_to_none() -> None:
    # None => plot every method present in the aggregates (mirrors base_methods).
    assert AxBenchBpbAdvConfig().plot_methods is None


def test_plot_methods_none_and_subset_are_valid() -> None:
    validate_config(_valid_config(plot_methods=None))
    validate_config(
        _valid_config(plot_methods=["diffmean_decoder_only", "linear_probe_decoder_only"])
    )


def test_plot_methods_empty_list_rejected() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="plot_methods"):
        validate_config(_valid_config(plot_methods=[]))


def test_plot_methods_duplicates_rejected() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="plot_methods"):
        validate_config(_valid_config(plot_methods=["diffmean_decoder_only"] * 2))


def test_plot_methods_unknown_method_name_rejected_at_config_time() -> None:
    # An obvious typo (not a registered method) fails fast at config validation
    # -- before the expensive scoring run for stage=all -- mirroring how
    # base_methods is registry-validated. (The PLOT stage additionally validates
    # against the actual aggregates, catching valid-but-absent-from-this-run.)
    with pytest.raises(AxBenchBpbAdvError, match="plot_methods"):
        validate_config(_valid_config(plot_methods=["not_a_real_method_xyz"]))


@pytest.mark.parametrize(
    "base_methods",
    [
        None,
        ["diffmean_decoder_only", "neg_diffmean_decoder_only"],
    ],
)
def test_all_stage_decoder_only_run_fails_before_expensive_work(
    base_methods: list[str] | None,
) -> None:
    cfg = _valid_config(
        stage="all",
        target_models=["decoder_only"],
        base_methods=base_methods,
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="stage='all'.*feature_diffmean_encdec.*neg_feature_diffmean_encdec",
    ):
        validate_all_stage_representative_target_contract(cfg)


def test_all_stage_requires_only_the_fair_target_for_selected_mode() -> None:
    cfg = _valid_config(
        stage="all",
        target_models=["decoder_only"],
        base_methods=["diffmean_decoder_only"],
    )

    with pytest.raises(AxBenchBpbAdvError, match="feature_diffmean_encdec") as exc_info:
        validate_all_stage_representative_target_contract(cfg)
    assert "neg_feature_diffmean_encdec" not in str(exc_info.value)


def test_all_stage_accepts_fair_targets_from_representative_target_methods() -> None:
    cfg = _valid_config(
        stage="all",
        target_models=["decoder_only", "encdec"],
        base_methods=["diffmean_decoder_only", "neg_diffmean_decoder_only"],
        representative_target_methods=[
            "feature_diffmean_encdec",
            "neg_feature_diffmean_encdec",
        ],
    )

    validate_all_stage_representative_target_contract(cfg)


@pytest.mark.parametrize("stage", ["score", "aggregate", "plot"])
def test_non_all_stages_are_artifact_or_stage_specific(
    stage: str,
) -> None:
    cfg = _valid_config(
        stage=stage,
        target_models=["decoder_only"],
        base_methods=["diffmean_decoder_only"],
    )

    validate_all_stage_representative_target_contract(cfg)


def test_plot_methods_cannot_waive_all_stage_target_requirement() -> None:
    cfg = _valid_config(
        stage="all",
        target_models=["decoder_only"],
        base_methods=["diffmean_decoder_only"],
        plot_methods=["diffmean_decoder_only"],
    )

    with pytest.raises(AxBenchBpbAdvError, match="feature_diffmean_encdec"):
        validate_all_stage_representative_target_contract(cfg)


def test_additive_magnitudes_reject_non_finite() -> None:
    # NaN would poison additive_counter_coefficient; the refined collection
    # domain rejects it at construction time.
    for bad in (math.nan, math.inf):
        with pytest.raises(BeartypeCallHintParamViolation, match="magnitudes"):
            AdditiveDirectionConfig(magnitudes=(0.0, bad))


def test_faster_gcg_temperature_rejects_non_finite() -> None:
    # `temperature <= 0` lets NaN/inf through (`nan <= 0` and `inf <= 0` are both
    # False), the same footgun the additive-magnitude guard was added for. A NaN
    # temperature poisons the candidate softmax (exp(-(score-min)/nan)); reject it.
    for bad in (math.nan, math.inf):
        with pytest.raises(BeartypeCallHintParamViolation, match="temperature"):
            FasterGcgConfig(temperature=bad)


def test_faster_gcg_distance_weight_rejects_non_finite() -> None:
    # `distance_weight < 0` lets NaN/inf through (both `nan < 0` and `inf < 0` are
    # False); an inf weight makes every same-token penalty infinite.
    for bad in (math.nan, math.inf):
        with pytest.raises(BeartypeCallHintParamViolation, match="distance_weight"):
            FasterGcgConfig(distance_weight=bad)


def test_aggregation_reuses_the_config_scatter_baseline_enum() -> None:
    """The allowed-set is NOT duplicated: aggregation imports the enum rather
    than maintaining its own choices tuple."""
    import steering.eval.axbench_bpb_adv.aggregation as aggregation

    assert aggregation.ScatterBaselineX is ScatterBaselineX
    assert not hasattr(aggregation, "_SCATTER_BASELINE_CHOICES")


# --------------------------------------------------------------------------- #
# Generalized additive counter-steering config                                #
# --------------------------------------------------------------------------- #
def test_additive_counter_direction_family_default_is_diffmean() -> None:
    assert AdditiveDirectionConfig().counter_direction_family == "diffmean"


@pytest.mark.parametrize("family", ["diffmean", "linear_probe"])
def test_additive_counter_direction_family_accepts_direction_families(
    family: str,
) -> None:
    cfg = _valid_config()
    cfg.adversarial_methods = [
        AdditiveDirectionConfig(counter_direction_family=cast(Any, family))
    ]
    validate_config(cfg)


def test_additive_counter_direction_family_rejects_non_direction() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="counter_direction_family"):
        AdditiveDirectionConfig(counter_direction_family=cast(Any, "lora"))


def test_additive_families_accepts_lora_and_ours() -> None:
    cfg = _valid_config()
    cfg.adversarial_methods = [
        AdditiveDirectionConfig(families=("diffmean", "linear_probe", "lora", "ours"))
    ]
    validate_config(cfg)


def test_additive_no_longer_requires_decoder_only_target() -> None:
    """additive now attacks encoder-decoder baselines too, so a pure-encdec run
    with additive is valid (the old decoder_only requirement is gone)."""
    cfg = _valid_config(target_models=["encdec"])
    cfg.adversarial_methods = [AdditiveDirectionConfig(families=("ours",))]
    validate_config(cfg)


def test_general_val_short_prefix_allowed_for_encdec_only_gcg_methods() -> None:
    """The decoder trigger-length guard depends on actual decoder rows, not the
    default target_models list. If base_methods whitelists only encdec methods,
    faster_gcg general_val scores clean encdec docs and needs no last-k decoder
    substitution."""

    cfg = _valid_config(
        base_methods=["diffmean_encdec"],
        general_val_bpb=GeneralValBpbConfig(
            enabled=True,
            prefix_tokens=2,
            output_tokens=1,
        ),
        adversarial_methods=[FasterGcgConfig(optimized_token_counts=(0, 8))],
    )

    validate_config(cfg)


@pytest.mark.parametrize(
    "method_name",
    ["ours_gcg_roofline_encdec", "feature_diffmean_roofline_encdec"],
)
def test_general_val_accepts_explicit_roofline_precomputed_bag_methods(
    method_name: str,
) -> None:
    cfg = _valid_config(
        base_methods=[method_name],
        general_val_bpb=GeneralValBpbConfig(enabled=True),
    )

    validate_config(cfg)


@pytest.mark.parametrize(
    "base_methods",
    [
        None,  # no whitelist: general_val is allowed (roofline gate is explicit-only)
        ["diffmean_decoder_only"],
        ["diffmean_encdec", "diffmean_decoder_only"],
    ],
)
def test_general_val_short_prefix_rejected_when_decoder_gcg_rows_can_exist(
    base_methods: list[str] | None,
) -> None:
    cfg = _valid_config(
        base_methods=base_methods,
        general_val_bpb=GeneralValBpbConfig(
            enabled=True,
            prefix_tokens=2,
            output_tokens=1,
        ),
        adversarial_methods=[FasterGcgConfig(optimized_token_counts=(0, 8))],
    )

    with pytest.raises(AxBenchBpbAdvError, match="prefix_tokens"):
        validate_config(cfg)


# --------------------------------------------------------------------------- #
# Additive control rows                                                        #
# --------------------------------------------------------------------------- #
def _prefix():
    return BpbAttackPrefix(
        core_prefix_id="core-0",
        concept_rich_sample_ids=["rich-0"],
        concept_avoiding_sample_ids=["avoid-0"],
        concept_id=1,
        mode=SteeringMode.POSITIVE,
        prefix_text="prefix",
        prefix_hash="hash",
        prefix_tokens=[0, 10, 11, 12, 13],
        concept_rich_output_tokens=[[20, 21]],
        concept_avoiding_output_tokens=[[30, 31]],
    )


def test_additive_control_rows_emit_one_row_per_magnitude() -> None:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
        adversarial_methods=[AdditiveDirectionConfig(magnitudes=(0.0, 2.0))],
    )

    rows = prepare_bpb_attack_rows(
        prefixes=[_prefix()],
        work_specs=[
            BpbAttackWorkSpec(
                method_name="diffmean_decoder_only",
                mode=SteeringMode.POSITIVE,
                target_model="decoder_only",
                coefficient_str="1.0",
                baseline_family="diffmean",
            )
        ],
        cfg=cfg,
    ).to_pylist()

    assert {row["adversarial_strength"] for row in rows} == {0.0, 2.0}


def test_yaml_default_metric_is_roundtrip_and_stats_path_falls_back() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="axbench_bpb_adv_eval",
            overrides=[
                "source_run_dir=/tmp/source",
                "output_root=/tmp/out",
                "feature_dir=/tmp/features",
                "feature_statistics_path=/tmp/stats.safetensors",
            ],
        )

    parsed = cfg_to_dataclass(cfg)

    assert parsed.steering_metric == "roundtrip"
    assert "max_new_tokens" not in cfg.roundtrip_generation
    assert cfg.roundtrip_generation.samples_per_candidate == 4
    assert "empty_response_policy" not in cfg.roundtrip_steering
    assert parsed.roundtrip_steering.enabled is True
    assert parsed.roundtrip_steering.feature_statistics_path == "/tmp/stats.safetensors"


def test_cfg_to_dataclass_rejects_roundtrip_empty_response_policy_override() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="axbench_bpb_adv_eval",
            overrides=[
                "source_run_dir=/tmp/source",
                "output_root=/tmp/out",
                "feature_dir=/tmp/features",
                "feature_statistics_path=/tmp/stats.safetensors",
                "+roundtrip_steering.empty_response_policy=skip",
            ],
        )

    with pytest.raises(AxBenchBpbAdvError, match="empty_response_policy"):
        cfg_to_dataclass(cfg)


def test_cfg_to_dataclass_accepts_legacy_bpb_metric_override() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="axbench_bpb_adv_eval",
            overrides=[
                "source_run_dir=/tmp/source",
                "output_root=/tmp/out",
                "feature_dir=/tmp/features",
                "feature_statistics_path=/tmp/stats.safetensors",
                "steering_metric=bpb_class_margin",
                "roundtrip_steering.enabled=false",
            ],
        )

    parsed = cfg_to_dataclass(cfg)

    assert parsed.steering_metric == "bpb_class_margin"
    assert parsed.roundtrip_steering.enabled is False
    validate_config(parsed)
