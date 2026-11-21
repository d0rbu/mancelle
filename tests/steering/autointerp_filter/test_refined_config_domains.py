"""Property coverage for beartype-enforced autointerp scalar domains."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st
from omegaconf import OmegaConf

from steering.eval.autointerp_filter.cli import build_config
from steering.eval.autointerp_filter.config import (
    AutoInterpFilterConfig,
    IntruderJudgeConfig,
    NeuronpediaDescriptionConfig,
    SourceIndexConfig,
    TokenExamplesConfig,
    validate_config,
)
from steering.eval.autointerp_filter.runner import _evaluate_trial_wave
from steering.eval.autointerp_filter.targets import resolve_targets
from steering.eval.autointerp_filter.types import AutoInterpConfigError

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True)

_VALID_POSITIVE_INT = st.integers(min_value=1)
_VALID_NON_NEGATIVE_INT = st.integers(min_value=0)
_VALID_UNIT_INTERVAL = st.one_of(
    st.integers(min_value=0, max_value=1),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
_VALID_POSITIVE_FINITE = st.one_of(
    st.integers(min_value=1, max_value=10**100),
    st.floats(min_value=1e-300, max_value=1e300, allow_nan=False, allow_infinity=False),
)
_INVALID_POSITIVE_INT = st.one_of(
    st.booleans(),
    st.integers(max_value=0),
    st.floats(allow_nan=True, allow_infinity=True),
    st.none(),
    st.text(),
)
_INVALID_NON_NULL_POSITIVE_INT = st.one_of(
    st.booleans(),
    st.integers(max_value=0),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(),
)
_INVALID_NON_NEGATIVE_INT = st.one_of(
    st.booleans(),
    st.integers(max_value=-1),
    st.floats(allow_nan=True, allow_infinity=True),
    st.none(),
    st.text(),
)
_INVALID_UNIT_INTERVAL = st.one_of(
    st.booleans(),
    st.integers(max_value=-1),
    st.integers(min_value=2),
    st.floats(max_value=-1e-300, allow_nan=False, allow_infinity=True),
    st.floats(min_value=1.0000000000000002, allow_nan=False, allow_infinity=True),
    st.just(float("nan")),
    st.none(),
    st.text(),
)
_INVALID_NON_NULL_UNIT_INTERVAL = st.one_of(
    st.booleans(),
    st.integers(max_value=-1),
    st.integers(min_value=2),
    st.floats(max_value=-1e-300, allow_nan=False, allow_infinity=True),
    st.floats(min_value=1.0000000000000002, allow_nan=False, allow_infinity=True),
    st.just(float("nan")),
    st.text(),
)
_INVALID_POSITIVE_FINITE = st.one_of(
    st.booleans(),
    st.integers(max_value=0),
    st.just(10**1000),
    st.floats(max_value=0.0, allow_nan=False, allow_infinity=True),
    st.just(float("nan")),
    st.just(float("inf")),
    st.none(),
    st.text(),
)
_INVALID_FEATURE_ID_LIST = st.one_of(
    st.just([]),
    _INVALID_NON_NEGATIVE_INT.map(lambda value: [value]),
    _VALID_NON_NEGATIVE_INT.map(lambda value: [value, value]),
)


def _kwargs(field_name: str, value: object) -> dict[str, Any]:
    return {field_name: value}


class _UnusedJudge:
    def evaluate(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("empty trial waves must not invoke the judge")


class _UnusedRenderer:
    def render(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("empty trial waves must not invoke the renderer")


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(("max_concurrent", "max_prompt_chars")),
    value=_VALID_POSITIVE_INT,
)
def test_judge_accepts_every_valid_positive_integer(
    field_name: str,
    value: int,
) -> None:
    cfg = IntruderJudgeConfig(**_kwargs(field_name, value))
    assert getattr(cfg, field_name) == value


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(("max_concurrent", "max_prompt_chars")),
    value=_INVALID_POSITIVE_INT,
)
def test_judge_rejects_every_invalid_positive_integer(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
        IntruderJudgeConfig(**_kwargs(field_name, value))


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(("requests_per_minute", "parse_max_retries")),
    value=_VALID_NON_NEGATIVE_INT,
)
def test_judge_accepts_every_valid_non_negative_integer(
    field_name: str,
    value: int,
) -> None:
    cfg = IntruderJudgeConfig(**_kwargs(field_name, value))
    assert getattr(cfg, field_name) == value


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(("requests_per_minute", "parse_max_retries")),
    value=_INVALID_NON_NEGATIVE_INT,
)
def test_judge_rejects_every_invalid_non_negative_integer(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
        IntruderJudgeConfig(**_kwargs(field_name, value))


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(
        (
            "example_ctx_len",
            "max_positive_source_docs_per_target",
            "max_intruder_source_docs",
            "max_windows_per_source_doc",
            "max_positive_windows_per_target",
            "max_non_activating_windows_per_target",
            "materialization_batch_size",
        )
    ),
    value=_INVALID_POSITIVE_INT,
)
def test_token_examples_rejects_invalid_positive_integer_fields(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
        TokenExamplesConfig(**_kwargs(field_name, value))


@PROPERTY_SETTINGS
@given(value=_INVALID_NON_NULL_POSITIVE_INT)
def test_token_examples_rejects_invalid_context_windows(value: object) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="context_window"):
        TokenExamplesConfig(context_window=cast(Any, value))


@PROPERTY_SETTINGS
@given(value=_VALID_UNIT_INTERVAL)
def test_token_examples_accepts_complete_activation_ratio_domain(value: int | float) -> None:
    cfg = TokenExamplesConfig(activation_threshold_ratio=cast(Any, value))
    assert cfg.activation_threshold_ratio == value


@PROPERTY_SETTINGS
@given(value=_INVALID_UNIT_INTERVAL)
def test_token_examples_rejects_invalid_activation_ratios(value: object) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="activation_threshold_ratio"):
        TokenExamplesConfig(activation_threshold_ratio=cast(Any, value))


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(("chunk_size", "scan_workers")),
    value=_INVALID_POSITIVE_INT,
)
def test_source_index_rejects_invalid_positive_integer_fields(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
        SourceIndexConfig(**_kwargs(field_name, value))


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(
        ("timeout_seconds", "backoff_base_seconds", "backoff_max_seconds")
    ),
    value=_VALID_POSITIVE_FINITE,
)
def test_neuronpedia_accepts_complete_positive_finite_domain(
    field_name: str,
    value: int | float,
) -> None:
    cfg = NeuronpediaDescriptionConfig(**_kwargs(field_name, value))
    assert getattr(cfg, field_name) == value


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(
        ("timeout_seconds", "backoff_base_seconds", "backoff_max_seconds")
    ),
    value=_INVALID_POSITIVE_FINITE,
)
def test_neuronpedia_rejects_invalid_positive_finite_fields(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
        NeuronpediaDescriptionConfig(**_kwargs(field_name, value))


@PROPERTY_SETTINGS
@given(value=_INVALID_NON_NEGATIVE_INT)
def test_neuronpedia_rejects_invalid_retry_counts(value: object) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="max_retries"):
        NeuronpediaDescriptionConfig(max_retries=cast(Any, value))


def test_neuronpedia_accepts_zero_retries() -> None:
    assert NeuronpediaDescriptionConfig(max_retries=0).max_retries == 0


@pytest.mark.parametrize(
    ("surface", "field_name"),
    [
        ("judge", "model"),
        ("token_examples", "sae_release"),
    ],
)
def test_config_required_strings_reject_empty_values(
    surface: str,
    field_name: str,
) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
        if surface == "judge":
            IntruderJudgeConfig(model="")
        else:
            TokenExamplesConfig(sae_release="")


@PROPERTY_SETTINGS
@given(feature_ids=_INVALID_FEATURE_ID_LIST)
def test_root_config_rejects_every_invalid_feature_id_domain(
    feature_ids: list[object],
) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="feature_ids"):
        AutoInterpFilterConfig(
            feature_dir="features",
            output_dir="output",
            feature_ids=cast(Any, feature_ids),
        )


@PROPERTY_SETTINGS
@given(feature_ids=_INVALID_FEATURE_ID_LIST)
def test_hydra_ingress_translates_invalid_feature_id_domains(
    feature_ids: list[object],
) -> None:
    raw = OmegaConf.create(
        {
            "feature_dir": "features",
            "output_dir": "output",
            "feature_ids": feature_ids,
        }
    )
    with pytest.raises(AutoInterpConfigError, match="feature_ids"):
        build_config(raw)


def test_config_consumers_revalidate_mutated_feature_id_lists() -> None:
    feature_ids = [1]
    cfg = AutoInterpFilterConfig(
        feature_dir="features",
        output_dir="output",
        feature_ids=feature_ids,
    )
    feature_ids.append(-1)
    with pytest.raises(BeartypeCallHintParamViolation, match="feature_ids"):
        validate_config(cfg)
    with pytest.raises(BeartypeCallHintParamViolation, match="feature_ids"):
        resolve_targets(
            cfg,
            n_sae_features=10**9 + 1,
            feature_dir=Path("features"),
        )


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(
        (
            "n_trials_per_target",
            "min_trials_for_score",
            "n_positive_examples",
            "n_quantiles",
        )
    ),
    value=_INVALID_POSITIVE_INT,
)
def test_root_config_rejects_invalid_positive_integer_fields(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
        AutoInterpFilterConfig(
            feature_dir="features",
            output_dir="output",
            **_kwargs(field_name, value),
        )


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(
        ("n_sae_features", "max_targets", "max_judge_calls", "top_k")
    ),
    value=_INVALID_NON_NULL_POSITIVE_INT,
)
def test_root_config_rejects_invalid_optional_positive_integer_fields(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
        AutoInterpFilterConfig(
            feature_dir="features",
            output_dir="output",
            **_kwargs(field_name, value),
        )


@PROPERTY_SETTINGS
@given(value=_INVALID_NON_NULL_UNIT_INTERVAL)
def test_root_config_rejects_invalid_score_thresholds(value: object) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="score_threshold"):
        AutoInterpFilterConfig(
            feature_dir="features",
            output_dir="output",
            score_threshold=cast(Any, value),
        )


@PROPERTY_SETTINGS
@given(value=_INVALID_NON_NEGATIVE_INT)
def test_root_config_rejects_invalid_audit_limits(value: object) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="audit_sample_limit"):
        AutoInterpFilterConfig(
            feature_dir="features",
            output_dir="output",
            audit_sample_limit=cast(Any, value),
        )


@PROPERTY_SETTINGS
@given(value=_INVALID_POSITIVE_INT)
def test_hydra_ingress_translates_nested_beartype_failures(value: object) -> None:
    raw = OmegaConf.create(
        {
            "feature_dir": "features",
            "output_dir": "output",
            "judge": {"max_concurrent": value},
        }
    )
    with pytest.raises(AutoInterpConfigError, match="max_concurrent"):
        build_config(raw)


@PROPERTY_SETTINGS
@given(value=_INVALID_POSITIVE_INT)
def test_trial_wave_beartype_boundary_rejects_invalid_worker_counts(value: object) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="max_concurrent"):
        _evaluate_trial_wave(
            cast(Any, _UnusedJudge()),
            renderer=cast(Any, _UnusedRenderer()),
            work=[],
            max_concurrent=cast(Any, value),
            on_success=cast(Any, lambda *_args: None),
        )
