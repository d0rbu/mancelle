"""Property contracts for AxBench-Base refined configuration domains."""

from __future__ import annotations

import math
from typing import Any, cast

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.eval.axbench_base.config import (
    AxBenchBaseConfig,
    JudgeConfig,
    LayerHooksConfig,
    SynthConfig,
)
from steering.eval.axbench_base.profiler import ProfileConfig
from steering.refined_types import (
    NonNegativeInt,
    OpenUnitInterval,
    PositiveInt,
    UniqueNonEmptyNonNegativeIntTuple,
    UnitInterval,
    UnitIntervalOpenLower,
    UnitIntervalOpenUpper,
)

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True)
SCALARS = st.one_of(
    st.integers(min_value=-10_000, max_value=10_000),
    st.floats(allow_nan=True, allow_infinity=True, width=64),
    st.booleans(),
    st.none(),
    st.text(max_size=16),
)

POSITIVE_INT_FIELDS = (
    "max_concepts",
    "seed_prefix_max_nanochat_tokens",
    "seed_continuation_nanochat_tokens",
    "seed_extraction_candidate_shard_size",
    "seed_extraction_gpu_batch_size",
    "seed_extraction_cpu_batch_size",
    "qc_batch_size",
    "verification_cross_prefix_batch_size",
    "stage0c_multi_gpu_block_size",
    "max_prefix_chars",
    "n_positives_per_concept",
    "positive_max_attempts_per_seed",
)
NON_NEGATIVE_INT_FIELDS = (
    "min_source_mismatch_features",
    "min_prefix_chars",
    "max_retries",
)
OPTIONAL_POSITIVE_INT_FIELDS = (
    "verification_zero_hit_bail_after",
    "positive_zero_hit_bail_after",
)
OPTIONAL_NON_NEGATIVE_INT_FIELDS = (
    "concept_examples_qc_target_surviving",
    "concept_description_fetch_limit",
    "verification_candidate_scan_limit",
    "verification_candidate_attempt_limit",
)
UNIT_INTERVAL_FIELDS = (
    "max_not_published_description_rate",
    "max_missing_description_rate",
    "max_failed_description_rate",
    "verification_percentile_threshold",
)
OPEN_LOWER_UNIT_INTERVAL_FIELDS = (
    "accept_threshold",
    "positive_prefix_headroom_ceiling",
)
OPEN_UPPER_UNIT_INTERVAL_FIELDS = (
    "max_concept_drop_fraction",
    "max_dropped_concept_fraction",
)


def _base_config() -> AxBenchBaseConfig:
    return AxBenchBaseConfig(
        feature_dir="out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base",
        feature_statistics_path="stats.safetensors",
        decoder_only={"source": "base"},
        encdec={"source": "encdec"},
        device="cpu",
        output_root=None,
    )


def _strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@PROPERTY_SETTINGS
@given(field=st.sampled_from(POSITIVE_INT_FIELDS), value=SCALARS)
def test_positive_integer_fields_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    if _strict_int(value) and value > 0:  # ty: ignore[unsupported-operator]
        config = _base_config().replace(**{field: value})
        assert isinstance(getattr(config, field), PositiveInt)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            _base_config().replace(**{field: cast(Any, value)})


@PROPERTY_SETTINGS
@given(field=st.sampled_from(NON_NEGATIVE_INT_FIELDS), value=SCALARS)
def test_non_negative_integer_fields_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    if _strict_int(value) and value >= 0:  # ty: ignore[unsupported-operator]
        config = _base_config().replace(**{field: value})
        assert isinstance(getattr(config, field), NonNegativeInt)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            _base_config().replace(**{field: cast(Any, value)})


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(OPTIONAL_POSITIVE_INT_FIELDS),
    value=SCALARS,
)
def test_optional_positive_integer_fields_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    valid = value is None or (_strict_int(value) and value > 0)  # ty: ignore[unsupported-operator]
    if valid:
        _base_config().replace(**{field: value})
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            _base_config().replace(**{field: cast(Any, value)})


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(OPTIONAL_NON_NEGATIVE_INT_FIELDS),
    value=SCALARS,
)
def test_optional_non_negative_integer_fields_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    valid = value is None or (_strict_int(value) and value >= 0)  # ty: ignore[unsupported-operator]
    if valid:
        _base_config().replace(**{field: value})
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            _base_config().replace(**{field: cast(Any, value)})


@PROPERTY_SETTINGS
@given(field=st.sampled_from(UNIT_INTERVAL_FIELDS), value=SCALARS)
def test_closed_unit_interval_fields_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    valid = (
        _finite_number(value)
        and value >= 0  # ty: ignore[unsupported-operator]
        and value <= 1  # ty: ignore[unsupported-operator]
    )
    if valid:
        config = _base_config().replace(**{field: value})
        assert isinstance(getattr(config, field), UnitInterval)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            _base_config().replace(**{field: cast(Any, value)})


@PROPERTY_SETTINGS
@given(field=st.sampled_from(OPEN_LOWER_UNIT_INTERVAL_FIELDS), value=SCALARS)
def test_open_lower_unit_interval_fields_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    valid = (
        _finite_number(value)
        and value > 0  # ty: ignore[unsupported-operator]
        and value <= 1  # ty: ignore[unsupported-operator]
    )
    if valid:
        config = _base_config().replace(**{field: value})
        assert isinstance(getattr(config, field), UnitIntervalOpenLower)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            _base_config().replace(**{field: cast(Any, value)})


@PROPERTY_SETTINGS
@given(field=st.sampled_from(OPEN_UPPER_UNIT_INTERVAL_FIELDS), value=SCALARS)
def test_open_upper_unit_interval_fields_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    valid = (
        _finite_number(value)
        and value >= 0  # ty: ignore[unsupported-operator]
        and value < 1  # ty: ignore[unsupported-operator]
    )
    if valid:
        config = _base_config().replace(**{field: value})
        assert isinstance(getattr(config, field), UnitIntervalOpenUpper)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            _base_config().replace(**{field: cast(Any, value)})


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_open_unit_interval_seed_ratio_matches_independent_oracle(value: object) -> None:
    valid = (
        _finite_number(value)
        and value > 0  # ty: ignore[unsupported-operator]
        and value < 1  # ty: ignore[unsupported-operator]
    )
    if valid:
        config = _base_config().replace(seed_pool_train_ratio=value)
        assert isinstance(config.seed_pool_train_ratio, OpenUnitInterval)
    else:
        with pytest.raises(
            BeartypeCallHintParamViolation,
            match="seed_pool_train_ratio",
        ):
            _base_config().replace(seed_pool_train_ratio=cast(Any, value))


@PROPERTY_SETTINGS
@given(values=st.lists(SCALARS, max_size=20))
def test_validation_shards_match_nonempty_unique_nonnegative_oracle(
    values: list[object],
) -> None:
    valid = (
        bool(values)
        and all(_strict_int(value) and value >= 0 for value in values)  # ty: ignore[unsupported-operator]
        and len(values) == len(set(cast(list[Any], values)))
    )
    if valid:
        config = _base_config().replace(validation_shards=cast(Any, values))
        assert isinstance(
            tuple(config.validation_shards),
            UniqueNonEmptyNonNegativeIntTuple,
        )
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="validation_shards"):
            _base_config().replace(validation_shards=cast(Any, values))


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_nested_config_domains_match_independent_oracles(value: object) -> None:
    strict_non_negative = _strict_int(value) and value >= 0  # ty: ignore[unsupported-operator]
    if strict_non_negative:
        JudgeConfig(parse_max_retries=cast(Any, value))
        SynthConfig(transport_max_retries=cast(Any, value))
        LayerHooksConfig(diffmean_and_probe_layer=cast(Any, value))
    else:
        for constructor, field in (
            (JudgeConfig, "parse_max_retries"),
            (SynthConfig, "transport_max_retries"),
            (LayerHooksConfig, "diffmean_and_probe_layer"),
        ):
            with pytest.raises(BeartypeCallHintParamViolation, match=field):
                constructor(**{field: cast(Any, value)})

    non_negative_finite = _finite_number(value) and value >= 0  # ty: ignore[unsupported-operator]
    if non_negative_finite:
        SynthConfig(transport_retry_budget_seconds=cast(Any, value))
        ProfileConfig(gpu_idle_threshold=cast(Any, value))
    else:
        with pytest.raises(
            BeartypeCallHintParamViolation,
            match="transport_retry_budget_seconds",
        ):
            SynthConfig(transport_retry_budget_seconds=cast(Any, value))
        with pytest.raises(
            BeartypeCallHintParamViolation,
            match="gpu_idle_threshold",
        ):
            ProfileConfig(gpu_idle_threshold=cast(Any, value))
