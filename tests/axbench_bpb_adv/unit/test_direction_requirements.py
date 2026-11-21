from __future__ import annotations

import pytest

from steering.eval.axbench_base.direction_artifacts import DirectionArtifactRequest
from steering.eval.axbench_bpb_adv.config import (
    AdditiveDirectionConfig,
    AxBenchBpbAdvConfig,
    FasterGcgConfig,
)
from steering.eval.axbench_bpb_adv.direction_requirements import (
    requirements_from_candidate_rows,
    requirements_from_config,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError


def _request(
    family: str,
    target: str,
    mode: str,
    concept_id: int,
    *,
    layer: int = 6,
) -> DirectionArtifactRequest:
    return DirectionArtifactRequest(
        family=family,
        target_model=target,
        mode=mode,
        concept_id=concept_id,
        layer=layer,
    )


def test_selected_scalar_methods_require_both_targets_modes_and_concepts() -> None:
    cfg = AxBenchBpbAdvConfig(
        target_models=["decoder_only", "encdec"],
        base_methods=[
            "diffmean_decoder_only",
            "neg_diffmean_encdec",
            "linear_probe_encdec",
            "neg_linear_probe_decoder_only",
        ],
        adversarial_methods=[FasterGcgConfig()],
    )

    assert set(requirements_from_config(cfg, concept_ids=[7, 2])) == {
        _request(family, target, mode, concept_id)
        for family, target, mode in (
            ("diffmean", "decoder_only", "positive"),
            ("diffmean", "encdec", "negative"),
            ("linear_probe", "encdec", "positive"),
            ("linear_probe", "decoder_only", "negative"),
        )
        for concept_id in (2, 7)
    }


@pytest.mark.parametrize("magnitudes", [(0.0,), (0,)])
def test_directionless_controls_and_zero_additive_strength_are_legal(
    magnitudes: tuple[float, ...],
) -> None:
    cfg = AxBenchBpbAdvConfig(
        target_models=["decoder_only", "encdec"],
        base_methods=["control_decoder_only", "neg_control_encdec"],
        adversarial_methods=[
            AdditiveDirectionConfig(
                families=("diffmean",),
                magnitudes=magnitudes,
                counter_direction_family="linear_probe",
            )
        ],
    )
    assert requirements_from_config(cfg, concept_ids=[1]) == []


def test_nonzero_additive_controls_require_configured_counter_family() -> None:
    cfg = AxBenchBpbAdvConfig(
        target_models=["decoder_only", "encdec"],
        base_methods=["control_decoder_only", "neg_control_encdec"],
        adversarial_methods=[
            AdditiveDirectionConfig(
                families=("ours",),
                magnitudes=(0.0, 0.5),
                counter_direction_family="linear_probe",
            )
        ],
    )
    assert set(requirements_from_config(cfg, concept_ids=[3])) == {
        _request("linear_probe", "decoder_only", "positive", 3),
        _request("linear_probe", "encdec", "negative", 3),
    }


def test_nonzero_additive_only_crosses_enabled_attack_families() -> None:
    cfg = AxBenchBpbAdvConfig(
        target_models=["decoder_only", "encdec"],
        base_methods=["lora_decoder_only", "ours_extract_and_add_encdec"],
        adversarial_methods=[
            AdditiveDirectionConfig(
                families=("ours",),
                magnitudes=(1.0,),
                counter_direction_family="diffmean",
            )
        ],
    )
    assert requirements_from_config(cfg, concept_ids=[1]) == [
        _request("diffmean", "encdec", "positive", 1)
    ]


def test_representative_target_control_never_requires_additive_direction() -> None:
    cfg = AxBenchBpbAdvConfig(
        target_models=["encdec"],
        base_methods=[],
        representative_target_methods=["feature_diffmean_encdec"],
        adversarial_methods=[
            AdditiveDirectionConfig(
                families=("ours",),
                magnitudes=(0.0, 1.0),
            )
        ],
    )
    assert requirements_from_config(cfg, concept_ids=[1]) == []


def test_candidate_requirements_include_base_and_nonzero_additive_counter() -> None:
    rows = [
        {
            "method_name": "neg_linear_probe_encdec",
            "concept_id": 9,
            "target_model": "encdec",
            "mode": "negative",
            "adversarial_method": "additive_direction",
            "adversarial_strength": 0.75,
        },
        {
            "method_name": "control_decoder_only",
            "concept_id": 4,
            "target_model": "decoder_only",
            "mode": "positive",
            "adversarial_method": "additive_direction",
            "adversarial_strength": 0.0,
        },
    ]
    assert set(
        requirements_from_candidate_rows(
            rows,
            layer=11,
            counter_direction_family="diffmean",
        )
    ) == {
        _request("linear_probe", "encdec", "negative", 9, layer=11),
        _request("diffmean", "encdec", "negative", 9, layer=11),
    }


def test_candidate_method_coordinate_swap_fails_before_model_loading() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="coordinates disagree"):
        requirements_from_candidate_rows(
            [
                {
                    "method_name": "diffmean_decoder_only",
                    "concept_id": 1,
                    "target_model": "encdec",
                    "mode": "positive",
                    "adversarial_method": "none",
                    "adversarial_strength": 0.0,
                }
            ],
            layer=6,
            counter_direction_family="diffmean",
        )


# ---------------------------------------------------------------------------
# #1249 follow-up: exclusions must reach the preflight, not just row emission
# ---------------------------------------------------------------------------
#
# The launch that motivated this: `exclude_concept_modes: ["37:positive"]`
# filtered attack rows correctly, but `requirements_from_config` -- the
# validate_source preflight, which derives coordinates from CONFIG x concepts
# before any row exists -- still demanded diffmean/encdec/positive/37 and
# failed the run 90 seconds in. The invariant: the preflight must never demand
# a direction for a row that the emitter will never produce.


def _excluding_cfg(exclusions: list[str]) -> AxBenchBpbAdvConfig:
    cfg = AxBenchBpbAdvConfig(
        target_models=["decoder_only", "encdec"],
        base_methods=[
            "diffmean_decoder_only",
            "feature_diffmean_encdec",
            "neg_feature_diffmean_encdec",
        ],
        adversarial_methods=[
            AdditiveDirectionConfig(
                families=("diffmean", "ours"),
                magnitudes=(0.0, 1.0),
            )
        ],
    )
    cfg.exclude_concept_modes = exclusions
    return cfg


def test_excluded_concept_mode_is_absent_from_preflight_requirements() -> None:
    cfg = _excluding_cfg(["37:positive"])
    requested = {
        (r.family, r.target_model, r.mode, r.concept_id)
        for r in requirements_from_config(cfg, concept_ids=[7, 37])
    }
    assert not any(
        mode == "positive" and concept_id == 37
        for _f, _t, mode, concept_id in requested
    ), "the preflight demanded a coordinate whose rows can never exist"
    # the OTHER mode of the same concept, and other concepts, stay required
    assert any(
        concept_id == 37 and mode == "negative"
        for _f, _t, mode, concept_id in requested
    )
    assert any(concept_id == 7 for _f, _t, _m, concept_id in requested)


def test_exclusion_scopes_to_mode_within_the_preflight() -> None:
    """Excluding (37, positive) must not silently drop 37's negative coords."""
    baseline = {
        (r.family, r.target_model, r.mode, r.concept_id)
        for r in requirements_from_config(_excluding_cfg([]), concept_ids=[37])
    }
    excluded = {
        (r.family, r.target_model, r.mode, r.concept_id)
        for r in requirements_from_config(
            _excluding_cfg(["37:positive"]), concept_ids=[37]
        )
    }
    dropped = baseline - excluded
    assert dropped, "the exclusion must drop something"
    assert all(mode == "positive" for _f, _t, mode, _c in dropped), (
        f"only positive-mode coordinates may be dropped, got {sorted(dropped)}"
    )
