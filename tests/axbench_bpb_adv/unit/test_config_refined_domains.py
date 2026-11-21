"""Property contracts for AxBench-BPB-Adv refined configuration domains."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st
from omegaconf import OmegaConf

from steering.eval.axbench_bpb_adv.config import (
    AdditiveDirectionConfig,
    AxBenchBpbAdvConfig,
    FasterGcgConfig,
    GeneralValBpbConfig,
    GepaConfig,
    OursGcgConfig,
    RoundtripGenerationConfig,
    validate_config,
)
from steering.eval.axbench_bpb_adv.config_loader import cfg_to_dataclass
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.refined_types import (
    UniqueNonEmptyNonNegativeFiniteNumberTuple,
    UniqueNonEmptyNonNegativeIntTuple,
    UniqueNonEmptyPositiveIntTuple,
)

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True)
INGRESS_SETTINGS = settings(max_examples=300, derandomize=True)
SCALARS = st.one_of(
    st.integers(min_value=-10_000, max_value=10_000),
    st.floats(allow_nan=True, allow_infinity=True, width=64),
    st.booleans(),
    st.none(),
    st.text(max_size=16),
)
TUPLES = st.lists(SCALARS, max_size=12).map(tuple)
Constructor = Callable[..., object]

POSITIVE_INT_FIELDS: tuple[tuple[Constructor, str, bool], ...] = (
    (FasterGcgConfig, "candidate_batch_size", False),
    (FasterGcgConfig, "top_k", False),
    (FasterGcgConfig, "universal_minibatch_size", False),
    (GepaConfig, "k_editable_tokens", False),
    (OursGcgConfig, "top_k_pos", True),
    (OursGcgConfig, "top_k_neg", True),
    (OursGcgConfig, "minibatch_size", False),
    (OursGcgConfig, "mean_g_steer_batch", False),
    (OursGcgConfig, "max_forward_batch", False),
    (RoundtripGenerationConfig, "samples_per_candidate", False),
    (RoundtripGenerationConfig, "top_k_sampling", False),
    (AxBenchBpbAdvConfig, "device_batch_size", False),
    (AxBenchBpbAdvConfig, "feature_diffmean_extraction_batch_size", True),
    (AxBenchBpbAdvConfig, "selection_splice_batch_size", True),
    (AxBenchBpbAdvConfig, "extraction_max_tokens", False),
)
NON_NEGATIVE_INT_FIELDS: tuple[tuple[Constructor, str], ...] = (
    (FasterGcgConfig, "max_steps"),
    (GeneralValBpbConfig, "max_interventions"),
    (GeneralValBpbConfig, "seed"),
    (AxBenchBpbAdvConfig, "pair_subsample"),
    (OursGcgConfig, "max_steps"),
)
STRICT_INT_FIELDS: tuple[tuple[Constructor, str], ...] = (
    (FasterGcgConfig, "master_seed"),
    (GepaConfig, "master_seed"),
    (OursGcgConfig, "split_seed"),
    (OursGcgConfig, "master_seed"),
    (RoundtripGenerationConfig, "master_seed"),
    (GeneralValBpbConfig, "n_documents"),
    (GeneralValBpbConfig, "prefix_tokens"),
    (GeneralValBpbConfig, "output_tokens"),
)


def _strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _construct(constructor: Constructor, field: str, value: object) -> object:
    return constructor(**{field: cast(Any, value)})


def _loader_base() -> dict[str, str]:
    return {
        "source_run_dir": "/tmp/source",
        "output_root": "/tmp/out",
        "feature_dir": "/tmp/features",
        "feature_statistics_path": "/tmp/stats",
    }


def _relationship_config(**overrides: Any) -> AxBenchBpbAdvConfig:
    values: dict[str, Any] = {
        **_loader_base(),
        "steering_metric": "bpb_class_margin",
        "scatter_baseline_x": "baseline_loss",
        "adversarial_methods": [AdditiveDirectionConfig()],
    }
    values.update(overrides)
    return AxBenchBpbAdvConfig(**values)


@PROPERTY_SETTINGS
@given(case=st.sampled_from(POSITIVE_INT_FIELDS), value=SCALARS)
def test_positive_integer_fields_match_independent_oracle(
    case: tuple[Constructor, str, bool],
    value: object,
) -> None:
    constructor, field, optional = case
    valid = (optional and value is None) or (
        _strict_int(value) and value > 0  # ty: ignore[unsupported-operator]
    )
    if valid:
        result = _construct(constructor, field, value)
        assert getattr(result, field) == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            _construct(constructor, field, value)


@PROPERTY_SETTINGS
@given(case=st.sampled_from(NON_NEGATIVE_INT_FIELDS), value=SCALARS)
def test_non_negative_integer_fields_match_independent_oracle(
    case: tuple[Constructor, str],
    value: object,
) -> None:
    constructor, field = case
    valid = _strict_int(value) and value >= 0  # ty: ignore[unsupported-operator]
    if valid:
        assert getattr(_construct(constructor, field, value), field) == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            _construct(constructor, field, value)


@PROPERTY_SETTINGS
@given(case=st.sampled_from(STRICT_INT_FIELDS), value=SCALARS)
def test_strict_integer_fields_match_independent_oracle(
    case: tuple[Constructor, str],
    value: object,
) -> None:
    constructor, field = case
    if _strict_int(value):
        assert getattr(_construct(constructor, field, value), field) == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            _construct(constructor, field, value)


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_positive_finite_temperature_matches_independent_oracle(value: object) -> None:
    valid = _finite_number(value) and value > 0  # ty: ignore[unsupported-operator]
    if valid:
        assert FasterGcgConfig(temperature=cast(Any, value)).temperature == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="temperature"):
            FasterGcgConfig(temperature=cast(Any, value))


@PROPERTY_SETTINGS
@given(
    case=st.sampled_from(
        (
            (FasterGcgConfig, "distance_weight"),
            (RoundtripGenerationConfig, "temperature"),
        )
    ),
    value=SCALARS,
)
def test_non_negative_finite_fields_match_independent_oracle(
    case: tuple[Constructor, str],
    value: object,
) -> None:
    constructor, field = case
    valid = _finite_number(value) and value >= 0  # ty: ignore[unsupported-operator]
    if valid:
        assert getattr(_construct(constructor, field, value), field) == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            _construct(constructor, field, value)


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_open_split_ratio_matches_independent_oracle(value: object) -> None:
    valid = (
        _finite_number(value)
        and value > 0  # ty: ignore[unsupported-operator]
        and value < 1  # ty: ignore[unsupported-operator]
    )
    if valid:
        assert OursGcgConfig(split_ratio=cast(Any, value)).split_ratio == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="split_ratio"):
            OursGcgConfig(split_ratio=cast(Any, value))


@PROPERTY_SETTINGS
@given(values=TUPLES)
def test_faster_token_grid_matches_independent_oracle(values: tuple[object, ...]) -> None:
    valid = (
        bool(values)
        and all(_strict_int(value) and value >= 0 for value in values)  # ty: ignore[unsupported-operator]
        and len(values) == len(set(values))
    )
    if valid:
        result = FasterGcgConfig(optimized_token_counts=cast(Any, values))
        assert isinstance(result.optimized_token_counts, UniqueNonEmptyNonNegativeIntTuple)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="optimized_token_counts"):
            FasterGcgConfig(optimized_token_counts=cast(Any, values))


@PROPERTY_SETTINGS
@given(values=TUPLES)
def test_gepa_budget_grid_matches_independent_oracle(values: tuple[object, ...]) -> None:
    valid = (
        bool(values)
        and all(_strict_int(value) and value > 0 for value in values)  # ty: ignore[unsupported-operator]
        and len(values) == len(set(values))
    )
    if valid:
        result = GepaConfig(metric_call_budgets=cast(Any, values))
        assert isinstance(result.metric_call_budgets, UniqueNonEmptyPositiveIntTuple)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="metric_call_budgets"):
            GepaConfig(metric_call_budgets=cast(Any, values))


@PROPERTY_SETTINGS
@given(
    constructor=st.sampled_from((AdditiveDirectionConfig, OursGcgConfig)),
    values=TUPLES,
)
def test_non_negative_finite_grids_match_independent_oracle(
    constructor: Constructor,
    values: tuple[object, ...],
) -> None:
    field = "magnitudes" if constructor is AdditiveDirectionConfig else "reg_lambda_grid"
    valid = (
        bool(values)
        and all(_finite_number(value) and value >= 0 for value in values)  # ty: ignore[unsupported-operator]
        and len(values) == len(set(values))
    )
    if valid:
        result = _construct(constructor, field, values)
        assert isinstance(
            getattr(result, field),
            UniqueNonEmptyNonNegativeFiniteNumberTuple,
        )
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            _construct(constructor, field, values)


@PROPERTY_SETTINGS
@given(values=st.lists(st.text(max_size=16), max_size=8).map(tuple))
def test_additive_families_match_closed_unique_nonempty_oracle(
    values: tuple[str, ...],
) -> None:
    allowed = {"diffmean", "linear_probe", "lora", "ours"}
    valid = bool(values) and len(values) == len(set(values)) and set(values) <= allowed
    if valid:
        assert AdditiveDirectionConfig(families=cast(Any, values)).families == values
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="families"):
            AdditiveDirectionConfig(families=cast(Any, values))


@PROPERTY_SETTINGS
@given(values=st.lists(SCALARS, max_size=8))
def test_target_models_match_closed_unique_nonempty_oracle(
    values: list[object],
) -> None:
    allowed = {"decoder_only", "encdec"}
    valid = (
        bool(values)
        and all(isinstance(value, str) and value in allowed for value in values)
        and len(values) == len(set(cast(list[Any], values)))
    )
    if valid:
        assert AxBenchBpbAdvConfig(target_models=cast(Any, values)).target_models == values
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="target_models"):
            AxBenchBpbAdvConfig(target_models=cast(Any, values))


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_optional_nonblank_paths_match_independent_oracle(value: object) -> None:
    valid = value is None or (isinstance(value, str) and bool(value.strip()))
    for constructor, field in (
        (GepaConfig, "reflection_model"),
        (AxBenchBpbAdvConfig, "feature_extraction_model_id"),
        (AxBenchBpbAdvConfig, "selection_manifest_path"),
    ):
        if valid:
            assert getattr(_construct(constructor, field, value), field) == value
        else:
            with pytest.raises(BeartypeCallHintParamViolation, match=field):
                _construct(constructor, field, value)


@PROPERTY_SETTINGS
@given(value=st.text(max_size=24))
def test_tagged_literals_match_closed_sets(value: str) -> None:
    for constructor, field, allowed in (
        (FasterGcgConfig, "method", {"faster_gcg"}),
        (GepaConfig, "method", {"gepa"}),
        (AdditiveDirectionConfig, "method", {"additive_direction"}),
        (
            AdditiveDirectionConfig,
            "counter_direction_family",
            {"diffmean", "linear_probe"},
        ),
        (
            AxBenchBpbAdvConfig,
            "steering_metric",
            {"roundtrip", "bpb_class_margin"},
        ),
        (
            AxBenchBpbAdvConfig,
            "scatter_baseline_x",
            {"baseline_loss", "min", "general_val_bpb"},
        ),
    ):
        if value in allowed:
            assert getattr(_construct(constructor, field, value), field) == value
        else:
            with pytest.raises(BeartypeCallHintParamViolation, match=field):
                _construct(constructor, field, value)


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("n_documents", "prefix_tokens", "output_tokens")),
    value=st.integers(min_value=-100, max_value=100),
    enabled=st.booleans(),
)
def test_general_val_conditional_relationship_is_preserved(
    field: str,
    value: int,
    enabled: bool,
) -> None:
    general_val = GeneralValBpbConfig(enabled=enabled, **{field: value})
    config = _relationship_config(general_val_bpb=general_val)
    if enabled and value < 1:
        with pytest.raises(AxBenchBpbAdvError, match=f"general_val_bpb.{field}"):
            validate_config(config)
    else:
        validate_config(config)


@PROPERTY_SETTINGS
@given(prefix_tokens=st.integers(min_value=1, max_value=256), token_count=st.integers(0, 512))
def test_faster_gcg_prefix_relationship_is_preserved(
    prefix_tokens: int,
    token_count: int,
) -> None:
    config = _relationship_config(
        target_models=["decoder_only"],
        adversarial_methods=[FasterGcgConfig(optimized_token_counts=(token_count,))],
        general_val_bpb=GeneralValBpbConfig(prefix_tokens=prefix_tokens),
    )
    if token_count > prefix_tokens:
        with pytest.raises(AxBenchBpbAdvError, match="prefix_tokens"):
            validate_config(config)
    else:
        validate_config(config)


@INGRESS_SETTINGS
@given(values=st.lists(st.integers(min_value=0, max_value=100), min_size=1, unique=True))
def test_hydra_ingress_normalizes_faster_grid_container_only(values: list[int]) -> None:
    config = cfg_to_dataclass(
        OmegaConf.create(
            {
                **_loader_base(),
                "adversarial_methods": [
                    {"method": "faster_gcg", "optimized_token_counts": values}
                ],
            }
        )
    )
    [method] = config.adversarial_methods
    assert isinstance(method, FasterGcgConfig)
    assert method.optimized_token_counts == tuple(values)


@INGRESS_SETTINGS
@given(values=st.lists(st.integers(min_value=1, max_value=100), min_size=1, unique=True))
def test_hydra_ingress_normalizes_gepa_grid_container_only(values: list[int]) -> None:
    config = cfg_to_dataclass(
        OmegaConf.create(
            {
                **_loader_base(),
                "adversarial_methods": [
                    {
                        "method": "gepa",
                        "metric_call_budgets": values,
                        "reflection_model": "model",
                    }
                ],
            }
        )
    )
    [method] = config.adversarial_methods
    assert isinstance(method, GepaConfig)
    assert method.metric_call_budgets == tuple(values)


@INGRESS_SETTINGS
@given(
    values=st.lists(
        st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=1,
        unique=True,
    )
)
def test_hydra_ingress_normalizes_additive_grid_container_only(values: list[float]) -> None:
    config = cfg_to_dataclass(
        OmegaConf.create(
            {
                **_loader_base(),
                "adversarial_methods": [
                    {"method": "additive_direction", "magnitudes": values}
                ],
            }
        )
    )
    [method] = config.adversarial_methods
    assert isinstance(method, AdditiveDirectionConfig)
    assert method.magnitudes == tuple(values)


@INGRESS_SETTINGS
@given(
    values=st.lists(
        st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=1,
        unique=True,
    )
)
def test_hydra_ingress_normalizes_ours_grid_container_only(values: list[float]) -> None:
    config = cfg_to_dataclass(
        OmegaConf.create(
            {
                **_loader_base(),
                "ours_gcg": {"reg_lambda_grid": values},
            }
        )
    )
    assert config.ours_gcg.reg_lambda_grid == tuple(values)


@pytest.mark.parametrize(
    ("block", "payload", "match"),
    [
        (
            "adversarial_methods",
            [{"method": "faster_gcg", "optimized_token_counts": [0, True]}],
            "invalid faster_gcg",
        ),
        (
            "ours_gcg",
            {"split_ratio": 1.0},
            "invalid ours_gcg",
        ),
        (
            "roundtrip_generation",
            {"samples_per_candidate": 1.5},
            "invalid roundtrip_generation",
        ),
        (
            "general_val_bpb",
            {"max_interventions": -1},
            "invalid general_val_bpb",
        ),
    ],
)
def test_hydra_ingress_wraps_refined_domain_failures(
    block: str,
    payload: object,
    match: str,
) -> None:
    with pytest.raises(AxBenchBpbAdvError, match=match) as exc_info:
        cfg_to_dataclass(OmegaConf.create({**_loader_base(), block: payload}))
    assert isinstance(exc_info.value.__cause__, BeartypeCallHintParamViolation)


def test_hydra_ingress_wraps_top_level_refined_domain_failure() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="invalid axbench-bpb-adv") as exc_info:
        cfg_to_dataclass(
            OmegaConf.create({**_loader_base(), "device_batch_size": True})
        )
    assert isinstance(exc_info.value.__cause__, BeartypeCallHintParamViolation)
