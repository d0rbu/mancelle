"""Property contracts for feature-filter refined configuration domains."""

from __future__ import annotations

import math
from typing import Any, cast

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st
from omegaconf import OmegaConf

from steering.eval.feature_filter.cli import _build_cfg
from steering.eval.feature_filter.config import (
    FeatureFilterConfig,
    JudgeConfig,
    NeuronpediaInlineConfig,
)
from steering.eval.feature_filter.interp_judge import InterpJudgeConfig
from steering.eval.feature_filter.types import ConfigValidationError
from steering.eval.neuronpedia_config import NeuronpediaConfig

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True)
INGRESS_SETTINGS = settings(max_examples=300, derandomize=True)
SCALARS = st.one_of(
    st.integers(min_value=-10_000, max_value=10_000),
    st.floats(allow_nan=True, allow_infinity=True, width=64),
    st.booleans(),
    st.none(),
    st.text(max_size=20),
)


def _strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _filter_config(**overrides: Any) -> FeatureFilterConfig:
    payload: dict[str, Any] = {
        "model_id": "gemma-2-2b",
        "sae_source": "8-gemmascope-res-16k",
        "tokenizer_id": "google/gemma-2-2b",
        "output_dir": "/tmp/feature-filter-property",
    }
    payload.update(overrides)
    return FeatureFilterConfig(**payload)


def _loader_payload() -> dict[str, Any]:
    return {
        "model_id": "gemma-2-2b",
        "sae_source": "8-gemmascope-res-16k",
        "tokenizer_id": "google/gemma-2-2b",
        "output_dir": "/tmp/feature-filter-property",
        "neuronpedia": {},
        "judge": {},
    }


def _construct_positive_int(field: str, value: object) -> object:
    typed = cast(Any, value)
    if field in {
        "n_sae_features",
        "max_concurrent_fetches",
        "max_concurrent_judges",
        "max_judge_calls",
    }:
        return _filter_config(**{field: typed})
    if field == "neuronpedia.max_concurrent":
        return NeuronpediaInlineConfig(max_concurrent=typed)
    if field == "client.max_concurrent":
        return NeuronpediaConfig(max_concurrent=typed)
    if field == "client.requests_per_minute":
        return NeuronpediaConfig(requests_per_minute=typed)
    raise AssertionError(f"unknown positive-int field {field}")


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(
        (
            "n_sae_features",
            "max_concurrent_fetches",
            "max_concurrent_judges",
            "max_judge_calls",
            "neuronpedia.max_concurrent",
            "client.max_concurrent",
            "client.requests_per_minute",
        )
    ),
    value=SCALARS,
)
def test_positive_integer_domains_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    optional = field in {"max_judge_calls", "client.requests_per_minute"}
    valid = (optional and value is None) or (
        _strict_int(value) and value > 0  # ty: ignore[unsupported-operator]
    )
    if valid:
        _construct_positive_int(field, value)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field.rsplit(".", maxsplit=1)[-1]):
            _construct_positive_int(field, value)


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(
        (
            "judge.parse_max_retries",
            "interp.parse_max_retries",
            "neuronpedia.max_retries",
            "client.max_retries",
        )
    ),
    value=SCALARS,
)
def test_non_negative_retry_domains_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    typed = cast(Any, value)
    constructor = {
        "judge.parse_max_retries": lambda: JudgeConfig(parse_max_retries=typed),
        "interp.parse_max_retries": lambda: InterpJudgeConfig(parse_max_retries=typed),
        "neuronpedia.max_retries": lambda: NeuronpediaInlineConfig(max_retries=typed),
        "client.max_retries": lambda: NeuronpediaConfig(max_retries=typed),
    }[field]
    valid = _strict_int(value) and value >= 0  # ty: ignore[unsupported-operator]
    if valid:
        constructor()
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field.rsplit(".", maxsplit=1)[-1]):
            constructor()


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(
        ("hist_sum_tolerance", "neuronpedia.timeout_seconds", "client.timeout_seconds")
    ),
    value=SCALARS,
)
def test_positive_finite_domains_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    typed = cast(Any, value)
    constructor = {
        "hist_sum_tolerance": lambda: _filter_config(hist_sum_tolerance=typed),
        "neuronpedia.timeout_seconds": lambda: NeuronpediaInlineConfig(timeout_seconds=typed),
        "client.timeout_seconds": lambda: NeuronpediaConfig(timeout_seconds=typed),
    }[field]
    valid = _finite_number(value) and value > 0  # ty: ignore[unsupported-operator]
    if valid:
        constructor()
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field.rsplit(".", maxsplit=1)[-1]):
            constructor()


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("error_abort_fraction", "shutdown_grace_seconds")),
    value=SCALARS,
)
def test_numeric_policy_domains_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    typed = cast(Any, value)
    if field == "error_abort_fraction":
        valid = (
            _finite_number(value)
            and value >= 0  # ty: ignore[unsupported-operator]
            and value <= 1  # ty: ignore[unsupported-operator]
        )
    else:
        valid = _finite_number(value) and value >= 0  # ty: ignore[unsupported-operator]
    if valid:
        _filter_config(**{field: typed})
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            _filter_config(**{field: typed})


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("samples_per_bin", "interp_min_score", "consistency_min_score")),
    value=SCALARS,
)
def test_five_point_integer_domains_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    valid = _strict_int(value) and (
        1 <= value <= 5  # ty: ignore[unsupported-operator]
        if field == "samples_per_bin"
        else 0 <= value <= 5  # ty: ignore[unsupported-operator]
    )
    if valid:
        _filter_config(**{field: cast(Any, value)})
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            _filter_config(**{field: cast(Any, value)})


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("model_id", "sae_source")),
    value=SCALARS,
)
def test_path_component_domains_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    valid = (
        isinstance(value, str)
        and bool(value.strip())
        and "/" not in value
        and "\\" not in value
        and ".." not in value
    )
    if valid:
        _filter_config(**{field: value})
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            _filter_config(**{field: cast(Any, value)})


@PROPERTY_SETTINGS
@given(values=st.lists(SCALARS, max_size=10))
def test_feature_id_collection_domain_matches_independent_oracle(
    values: list[object],
) -> None:
    valid = (
        bool(values)
        and all(_strict_int(value) and value >= 0 for value in values)  # ty: ignore[unsupported-operator]
        and len(values) == len(set(values))
    )
    if valid:
        _filter_config(feature_ids=cast(Any, values))
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="feature_ids"):
            _filter_config(feature_ids=cast(Any, values))


@PROPERTY_SETTINGS
@given(
    base=st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
    maximum=st.floats(
        min_value=1e-6,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_backoff_relationship_is_preserved(base: float, maximum: float) -> None:
    inline = NeuronpediaInlineConfig(
        backoff_base_seconds=base,
        backoff_max_seconds=maximum,
    )
    if maximum < base:
        with pytest.raises(ConfigValidationError, match="backoff_max_seconds"):
            inline.validate()
        with pytest.raises(ValueError, match="backoff_max_seconds"):
            NeuronpediaConfig(
                backoff_base_seconds=base,
                backoff_max_seconds=maximum,
            )
    else:
        inline.validate()
        NeuronpediaConfig(
            backoff_base_seconds=base,
            backoff_max_seconds=maximum,
        )


@INGRESS_SETTINGS
@given(
    max_concurrent=st.integers(min_value=1, max_value=1_000),
    max_retries=st.integers(min_value=0, max_value=100),
    samples_per_bin=st.integers(min_value=1, max_value=5),
    error_abort_fraction=st.floats(
        min_value=0,
        max_value=1,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_hydra_ingress_preserves_valid_nested_and_top_domains(
    max_concurrent: int,
    max_retries: int,
    samples_per_bin: int,
    error_abort_fraction: float,
) -> None:
    payload = _loader_payload()
    payload["neuronpedia"] = {
        "max_concurrent": max_concurrent,
        "max_retries": max_retries,
    }
    payload["judge"] = {"parse_max_retries": max_retries}
    payload["samples_per_bin"] = samples_per_bin
    payload["error_abort_fraction"] = error_abort_fraction
    config = _build_cfg(OmegaConf.create(payload))
    assert config.neuronpedia.max_concurrent == max_concurrent
    assert config.neuronpedia.max_retries == max_retries
    assert config.judge.parse_max_retries == max_retries
    assert config.samples_per_bin == samples_per_bin
    assert config.error_abort_fraction == error_abort_fraction


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"neuronpedia": {"timeout_seconds": 0}}, "invalid neuronpedia"),
        ({"judge": {"parse_max_retries": -1}}, "invalid judge"),
        ({"samples_per_bin": 0}, "invalid feature_filter"),
    ],
)
def test_hydra_ingress_wraps_refined_domain_failures(
    payload: dict[str, Any],
    match: str,
) -> None:
    merged = _loader_payload()
    merged.update(payload)
    with pytest.raises(ConfigValidationError, match=match) as exc_info:
        _build_cfg(OmegaConf.create(merged))
    assert isinstance(exc_info.value.__cause__, BeartypeCallHintParamViolation)
