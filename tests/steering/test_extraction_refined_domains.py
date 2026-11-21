"""Property contracts for extraction runtime and staging scalar domains."""

from typing import Any

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.extract_features import (
    ActivationType,
    AutoBatchSearchConfig,
    ExtractionRuntimeConfig,
    ExtractionStagingConfig,
    PoolingMode,
    ProcessShardParams,
    SegmentMode,
    find_optimal_batch_sizes,
)
from steering.refined_types import (
    NonBlankStr,
    NonNegativeFiniteNumber,
    NonNegativeInt,
    PositiveFiniteNumber,
    PositiveInt,
    UnitInterval,
    ZstdCompressionLevel,
)

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True)


def _staging_config(
    *,
    output_depth: int = 0,
    input_depth: int = 0,
    min_free_gb: int | float = 0.0,
    est_shard_gb: int | float = 0.001,
) -> ExtractionStagingConfig:
    return ExtractionStagingConfig(
        scratch_dir="/tmp/extraction-staging",
        output_depth=NonNegativeInt.parse(output_depth),
        input_depth=NonNegativeInt.parse(input_depth),
        min_free_gb=NonNegativeFiniteNumber.parse(min_free_gb),
        est_shard_gb=PositiveFiniteNumber.parse(est_shard_gb),
    )


def _process_params(**overrides: Any) -> ProcessShardParams:
    values: dict[str, Any] = {
        "model": object(),
        "sae": object(),
        "layer": NonNegativeInt.parse(0),
        "output_dir": "/tmp/features",
        "context_window": PositiveInt.parse(128),
        "pooling": PoolingMode.max_pool,
        "gpu_batch_size": PositiveInt.parse(2),
        "cpu_batch_size": PositiveInt.parse(4),
        "gc_mem_threshold": UnitInterval.parse(0.9),
        "nnterp_model": None,
        "native_model": None,
        "segment_mode": SegmentMode.fixed_tokens,
        "segment_size": PositiveInt.parse(16),
        "hook_name": None,
        "activation_type": ActivationType.residual,
        "dataset_phase": None,
        "zstd_level": ZstdCompressionLevel.parse(12),
        "context_independent_features": False,
        "ignore_bos_features": False,
        "run_l0_tier": None,
    }
    values.update(overrides)
    return ProcessShardParams(**values)


def _runtime_config(**overrides: Any) -> ExtractionRuntimeConfig:
    values: dict[str, Any] = {
        "rank": NonNegativeInt.parse(0),
        "world_size": PositiveInt.parse(1),
        "local_rank": NonNegativeInt.parse(0),
        "model_name": NonBlankStr.parse("gemma-2-2b"),
        "layer": NonNegativeInt.parse(8),
        "context_window": PositiveInt.parse(1024),
        "segment_size": PositiveInt.parse(128),
        "gc_mem_threshold": UnitInterval.parse(0.9),
        "zstd_level": ZstdCompressionLevel.parse(12),
        "num_shards": NonNegativeInt.parse(1),
        "start_shard": NonNegativeInt.parse(0),
        "gpu_batch_size": PositiveInt.parse(16),
        "cpu_batch_size": PositiveInt.parse(64),
        "min_gpu_batch_size": PositiveInt.parse(1),
        "min_cpu_batch_size": PositiveInt.parse(1),
        "max_gpu_batch_size": PositiveInt.parse(512),
        "max_cpu_batch_size": PositiveInt.parse(512),
        "staging": _staging_config(),
    }
    values.update(overrides)
    return ExtractionRuntimeConfig(**values)


@PROPERTY_SETTINGS
@given(
    max_gpu=st.integers(min_value=1),
    max_cpu=st.integers(min_value=1),
    min_gpu=st.integers(min_value=1),
    min_cpu=st.integers(min_value=1),
)
def test_auto_batch_search_accepts_complete_positive_domain(
    max_gpu: int,
    max_cpu: int,
    min_gpu: int,
    min_cpu: int,
) -> None:
    config = AutoBatchSearchConfig(
        max_gpu_batch_size=PositiveInt.parse(max_gpu),
        max_cpu_batch_size=PositiveInt.parse(max_cpu),
        min_gpu_batch_size=PositiveInt.parse(min_gpu),
        min_cpu_batch_size=PositiveInt.parse(min_cpu),
    )
    assert tuple(config.__dict__.values()) == (max_gpu, max_cpu, min_gpu, min_cpu)


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(
        (
            "max_gpu_batch_size",
            "max_cpu_batch_size",
            "min_gpu_batch_size",
            "min_cpu_batch_size",
        )
    ),
    bad=st.one_of(
        st.integers(max_value=0),
        st.booleans(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.text(),
        st.none(),
    ),
)
def test_auto_batch_search_rejects_complete_invalid_domain(
    field: str,
    bad: Any,
) -> None:
    values: dict[str, Any] = {
        "max_gpu_batch_size": PositiveInt.parse(8),
        "max_cpu_batch_size": PositiveInt.parse(8),
        "min_gpu_batch_size": PositiveInt.parse(1),
        "min_cpu_batch_size": PositiveInt.parse(1),
    }
    values[field] = bad
    with pytest.raises(BeartypeCallHintParamViolation):
        AutoBatchSearchConfig(**values)


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("max_gpu_batch_size", "min_cpu_batch_size")),
    bad=st.one_of(st.integers(max_value=0), st.booleans()),
)
def test_find_optimal_batch_sizes_rejects_bad_bounds_before_model_access(
    field: str,
    bad: Any,
) -> None:
    kwargs: dict[str, Any] = {
        "model": None,
        "sae": None,
        "layer": 0,
        "context_window": 128,
        "max_gpu_batch_size": 8,
        "max_cpu_batch_size": 8,
        "min_gpu_batch_size": 1,
        "min_cpu_batch_size": 1,
    }
    kwargs[field] = bad
    with pytest.raises(BeartypeCallHintParamViolation):
        find_optimal_batch_sizes(**kwargs)


@PROPERTY_SETTINGS
@given(
    output_depth=st.integers(min_value=0),
    input_depth=st.integers(min_value=0),
    min_free_gb=st.floats(
        min_value=0.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    est_shard_gb=st.floats(
        min_value=1e-12,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_staging_config_accepts_complete_valid_domains(
    output_depth: int,
    input_depth: int,
    min_free_gb: float,
    est_shard_gb: float,
) -> None:
    config = _staging_config(
        output_depth=output_depth,
        input_depth=input_depth,
        min_free_gb=min_free_gb,
        est_shard_gb=est_shard_gb,
    )
    assert config.output_depth == output_depth
    assert config.input_depth == input_depth
    assert config.min_free_gb == min_free_gb
    assert config.est_shard_gb == est_shard_gb


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("output_depth", "input_depth")),
    bad=st.one_of(
        st.integers(max_value=-1),
        st.booleans(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.text(),
        st.none(),
    ),
)
def test_staging_config_rejects_complete_invalid_depth_domain(
    field: str,
    bad: Any,
) -> None:
    values: dict[str, Any] = {
        "scratch_dir": "/tmp/extraction-staging",
        "output_depth": NonNegativeInt.parse(0),
        "input_depth": NonNegativeInt.parse(0),
        "min_free_gb": NonNegativeFiniteNumber.parse(0.0),
        "est_shard_gb": PositiveFiniteNumber.parse(0.001),
    }
    values[field] = bad
    with pytest.raises(BeartypeCallHintParamViolation):
        ExtractionStagingConfig(**values)


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("min_free_gb", "est_shard_gb")),
    bad=st.one_of(
        st.integers(max_value=0),
        st.booleans(),
        st.floats(max_value=0, allow_nan=False, allow_infinity=False),
        st.sampled_from((float("nan"), float("inf"), float("-inf"))),
        st.text(),
        st.none(),
    ),
)
def test_staging_config_rejects_invalid_capacity_domains(
    field: str,
    bad: Any,
) -> None:
    if field == "min_free_gb" and bad == 0:
        bad = -1
    values: dict[str, Any] = {
        "scratch_dir": "/tmp/extraction-staging",
        "output_depth": NonNegativeInt.parse(0),
        "input_depth": NonNegativeInt.parse(0),
        "min_free_gb": NonNegativeFiniteNumber.parse(0.0),
        "est_shard_gb": PositiveFiniteNumber.parse(0.001),
    }
    values[field] = bad
    with pytest.raises(BeartypeCallHintParamViolation):
        ExtractionStagingConfig(**values)


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(
        ("layer", "context_window", "gpu_batch_size", "cpu_batch_size", "segment_size")
    ),
    bad=st.one_of(st.integers(max_value=0), st.booleans()),
)
def test_process_params_reject_invalid_integer_domains(
    field: str,
    bad: Any,
) -> None:
    if field == "layer" and bad == 0:
        bad = -1
    with pytest.raises(BeartypeCallHintParamViolation):
        _process_params(**{field: bad})


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("gpu_batch_size", "cpu_batch_size")),
    value=st.one_of(st.just("auto"), st.integers(min_value=1)),
)
def test_runtime_config_accepts_auto_or_positive_batch_sizes(
    field: str,
    value: str | int,
) -> None:
    typed_value = value if value == "auto" else PositiveInt.parse(value)
    config = _runtime_config(**{field: typed_value})
    assert getattr(config, field) == value


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("gpu_batch_size", "cpu_batch_size")),
    bad=st.one_of(
        st.integers(max_value=0),
        st.booleans(),
        st.text().filter(lambda value: value != "auto"),
        st.none(),
    ),
)
def test_runtime_config_rejects_invalid_batch_size_domain(
    field: str,
    bad: Any,
) -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        _runtime_config(**{field: bad})


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(
        (
            "rank",
            "world_size",
            "local_rank",
            "layer",
            "context_window",
            "num_shards",
            "start_shard",
            "min_gpu_batch_size",
            "min_cpu_batch_size",
            "max_gpu_batch_size",
            "max_cpu_batch_size",
        )
    ),
    bad=st.one_of(
        st.integers(max_value=0),
        st.booleans(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.text(),
        st.none(),
    ),
)
def test_runtime_config_rejects_invalid_integer_domains(
    field: str,
    bad: Any,
) -> None:
    non_negative_fields = {"rank", "local_rank", "layer", "num_shards", "start_shard"}
    if field in non_negative_fields and bad == 0:
        bad = -1
    with pytest.raises(BeartypeCallHintParamViolation):
        _runtime_config(**{field: bad})
