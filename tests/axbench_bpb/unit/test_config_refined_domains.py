"""Property contracts for AxBench-BPB refined configuration domains."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast
from unittest.mock import patch

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st
from omegaconf import OmegaConf

from scripts.eval.axbench_bpb_eval import _cfg_to_dataclass
from steering.eval.axbench_bpb.config import (
    AggregationConfig,
    AxBenchBpbConfig,
    ContextCacheConfig,
    FeatureDiffMeanConfig,
    validate_bpb_config,
)
from steering.eval.axbench_bpb.types import AxBenchBpbError
from steering.eval.axbench_bpb_adv.config_loader import (
    cfg_to_dataclass as adv_cfg_to_dataclass,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.refined_types import AtLeastTwoInt

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True)
INGRESS_SETTINGS = settings(max_examples=300, derandomize=True)
SCALARS = st.one_of(
    st.integers(min_value=-10_000, max_value=10_000),
    st.floats(allow_nan=True, allow_infinity=True, width=64),
    st.booleans(),
    st.none(),
    st.text(max_size=20),
)
Constructor = Callable[..., object]

POSITIVE_INT_FIELDS: tuple[tuple[Constructor, str, bool], ...] = (
    (ContextCacheConfig, "extraction_batch_size", False),
    (ContextCacheConfig, "segment_size", False),
    (AggregationConfig, "min_rows_per_side", False),
    (FeatureDiffMeanConfig, "top_k_pos", True),
    (FeatureDiffMeanConfig, "top_k_neg", True),
    (AxBenchBpbConfig, "device_batch_size", False),
)
BOOL_FIELDS: tuple[tuple[Constructor, str], ...] = (
    (ContextCacheConfig, "force_rebuild"),
    (AxBenchBpbConfig, "next_segment_conditioning_offset"),
    (AxBenchBpbConfig, "filter_unaccepted_samples"),
    (AxBenchBpbConfig, "offload_between_stages"),
)
STAGE_AND_METRIC_VALUES = (
    "prepare_context_cache",
    "bpb_score",
    "bpb_aggregate",
    "all",
    "macro_mean_best_primary_margin",
    "macro_mean_best_margin_intervened",
    "macro_mean_dose_auc_margin",
    "macro_mean_truncation_rate",
)


def _strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _construct(constructor: Constructor, field: str, value: object) -> object:
    return constructor(**{field: cast(Any, value)})


def _loader_base() -> dict[str, Any]:
    return {
        "source_run_dir": "/tmp/source",
        "output_root": "/tmp/out",
        "feature_dir": "/tmp/features",
        "feature_statistics_path": "/tmp/stats.safetensors",
        "decoder_only": {},
        "encdec": {},
    }


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
        assert getattr(_construct(constructor, field, value), field) == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            _construct(constructor, field, value)


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_context_bucket_count_matches_at_least_two_oracle(value: object) -> None:
    valid = _strict_int(value) and value >= 2  # ty: ignore[unsupported-operator]
    if valid:
        config = ContextCacheConfig(num_buckets=cast(Any, value))
        assert isinstance(config.num_buckets, AtLeastTwoInt)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="num_buckets"):
            ContextCacheConfig(num_buckets=cast(Any, value))


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_non_negative_layer_matches_independent_oracle(value: object) -> None:
    valid = _strict_int(value) and value >= 0  # ty: ignore[unsupported-operator]
    if valid:
        config = AxBenchBpbConfig(diffmean_and_probe_layer=cast(Any, value))
        assert config.diffmean_and_probe_layer == value
    else:
        with pytest.raises(
            BeartypeCallHintParamViolation,
            match="diffmean_and_probe_layer",
        ):
            AxBenchBpbConfig(diffmean_and_probe_layer=cast(Any, value))


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("invalid_warning_threshold", "invalid_drop_threshold")),
    value=SCALARS,
)
def test_unit_interval_thresholds_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    valid = (
        _finite_number(value)
        and value >= 0  # ty: ignore[unsupported-operator]
        and value <= 1  # ty: ignore[unsupported-operator]
    )
    if valid:
        assert getattr(_construct(AggregationConfig, field, value), field) == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            _construct(AggregationConfig, field, value)


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_optional_feature_model_id_matches_nonblank_oracle(value: object) -> None:
    valid = value is None or (isinstance(value, str) and bool(value.strip()))
    if valid:
        config = AxBenchBpbConfig(feature_extraction_model_id=cast(Any, value))
        assert config.feature_extraction_model_id == value
    else:
        with pytest.raises(
            BeartypeCallHintParamViolation,
            match="feature_extraction_model_id",
        ):
            AxBenchBpbConfig(feature_extraction_model_id=cast(Any, value))


@PROPERTY_SETTINGS
@given(case=st.sampled_from(BOOL_FIELDS), value=SCALARS)
def test_boolean_fields_reject_integer_lookalikes(
    case: tuple[Constructor, str],
    value: object,
) -> None:
    constructor, field = case
    if isinstance(value, bool):
        assert getattr(_construct(constructor, field, value), field) is value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            _construct(constructor, field, value)


@PROPERTY_SETTINGS
@given(
    value=st.one_of(
        st.sampled_from(STAGE_AND_METRIC_VALUES),
        st.text(max_size=40),
    )
)
def test_stage_and_primary_metric_match_closed_sets(value: str) -> None:
    for constructor, field, allowed in (
        (
            AxBenchBpbConfig,
            "stage",
            {"prepare_context_cache", "bpb_score", "bpb_aggregate", "all"},
        ),
        (
            AggregationConfig,
            "primary_metric",
            {
                "macro_mean_best_primary_margin",
                "macro_mean_best_margin_intervened",
                "macro_mean_dose_auc_margin",
                "macro_mean_truncation_rate",
            },
        ),
    ):
        if value in allowed:
            assert getattr(_construct(constructor, field, value), field) == value
        else:
            with pytest.raises(BeartypeCallHintParamViolation, match=field):
                _construct(constructor, field, value)


@PROPERTY_SETTINGS
@given(
    warning=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
    drop=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
)
def test_threshold_ordering_relationship_is_preserved(
    warning: float,
    drop: float,
) -> None:
    config = AxBenchBpbConfig(
        **_loader_base(),
        enabled_methods=["control_decoder_only"],
        aggregation=AggregationConfig(
            invalid_warning_threshold=warning,
            invalid_drop_threshold=drop,
        ),
    )
    with patch(
        "steering.eval.axbench_bpb.config.validate_stage_0_3_artifacts",
    ):
        if warning > drop:
            with pytest.raises(AxBenchBpbError, match="threshold"):
                validate_bpb_config(config)
        else:
            validate_bpb_config(config)


@INGRESS_SETTINGS
@given(
    extraction_batch_size=st.integers(min_value=1, max_value=10_000),
    segment_size=st.integers(min_value=1, max_value=10_000),
    num_buckets=st.integers(min_value=2, max_value=100),
)
def test_hydra_ingress_preserves_valid_context_domains(
    extraction_batch_size: int,
    segment_size: int,
    num_buckets: int,
) -> None:
    config = _cfg_to_dataclass(
        OmegaConf.create(
            {
                **_loader_base(),
                "context_cache": {
                    "extraction_batch_size": extraction_batch_size,
                    "segment_size": segment_size,
                    "num_buckets": num_buckets,
                },
            }
        )
    )
    assert config.context_cache == ContextCacheConfig(
        extraction_batch_size=extraction_batch_size,
        segment_size=segment_size,
        num_buckets=num_buckets,
    )


@pytest.mark.parametrize(
    ("block", "payload", "match"),
    [
        ("context_cache", {"num_buckets": 1}, "invalid context_cache"),
        ("aggregation", {"min_rows_per_side": True}, "invalid aggregation"),
        ("feature_diffmean", {"top_k_pos": 1.5}, "invalid feature_diffmean"),
    ],
)
def test_hydra_ingress_wraps_nested_refined_domain_failures(
    block: str,
    payload: object,
    match: str,
) -> None:
    with pytest.raises(AxBenchBpbError, match=match) as exc_info:
        _cfg_to_dataclass(OmegaConf.create({**_loader_base(), block: payload}))
    assert isinstance(exc_info.value.__cause__, BeartypeCallHintParamViolation)


def test_hydra_ingress_wraps_top_level_refined_domain_failure() -> None:
    with pytest.raises(AxBenchBpbError, match="invalid axbench-bpb") as exc_info:
        _cfg_to_dataclass(OmegaConf.create({**_loader_base(), "device_batch_size": False}))
    assert isinstance(exc_info.value.__cause__, BeartypeCallHintParamViolation)


def test_bpb_adv_loader_wraps_shared_context_domain_failure() -> None:
    payload = {
        **_loader_base(),
        "steering_metric": "bpb_class_margin",
        "scatter_baseline_x": "baseline_loss",
        "context_cache": {"segment_size": 0},
    }
    with pytest.raises(AxBenchBpbAdvError, match="invalid context_cache") as exc_info:
        adv_cfg_to_dataclass(OmegaConf.create(payload))
    assert isinstance(exc_info.value.__cause__, BeartypeCallHintParamViolation)
