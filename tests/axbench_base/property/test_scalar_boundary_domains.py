"""Property oracles for scalar-only AxBench construction boundaries."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, TypeGuard, cast

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.eval.axbench_base.config import LLMClientConfig
from steering.eval.axbench_base.llm_judge_client import (
    OPENROUTER_API_KEY_ENV,
    OpenRouterLLMClient,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.general_val_roofline import (
    GeneralValSignalBudget,
    GeneralValWarmStartCaps,
)
from steering.feature_paths import DatasetPhase, PoolingMode, build_feature_dir
from steering.sae_resolver import ActivationType, SAEWidth

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True, deadline=None)
SCALARS = st.one_of(
    st.integers(min_value=-10_000, max_value=10_000),
    st.floats(allow_nan=True, allow_infinity=True, width=64),
    st.booleans(),
    st.none(),
    st.text(max_size=16),
)


def _strict_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _feature_dir_kwargs() -> dict[str, Any]:
    return {
        "root": Path("/tmp/features"),
        "segment_size": 128,
        "layer": 8,
        "activation_type": ActivationType.residual,
        "pooling": PoolingMode.last_token,
        "width": SAEWidth.W_16K,
        "model": "gemma-2-2b",
        "dataset_phase": DatasetPhase.BASE,
    }


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("k_pos", "k_neg", "positive_bucket", "negative_bucket")),
    value=SCALARS,
)
def test_signal_budget_scalar_domains_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    fields: dict[str, Any] = {
        "k_pos": 1,
        "k_neg": 0,
        "positive_bucket": 0,
        "negative_bucket": 1,
    }
    fields[field] = value
    valid = _strict_int(value) and value >= 0
    if valid and fields["positive_bucket"] != fields["negative_bucket"]:
        assert getattr(GeneralValSignalBudget(**fields), field) == value
    elif valid:
        with pytest.raises(AxBenchBpbAdvError, match="must differ"):
            GeneralValSignalBudget(**fields)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            GeneralValSignalBudget(**fields)


@PROPERTY_SETTINGS
@given(field=st.sampled_from(("max_k_pos", "max_k_neg")), value=SCALARS)
def test_warm_start_cap_domains_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    fields: dict[str, Any] = {"max_k_pos": 1, "max_k_neg": 1}
    fields[field] = value
    valid = _strict_int(value) and value > 0
    if valid:
        assert getattr(GeneralValWarmStartCaps(**fields), field) == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            GeneralValWarmStartCaps(**fields)


@PROPERTY_SETTINGS
@given(field=st.sampled_from(("segment_size", "layer", "model")), value=SCALARS)
def test_feature_directory_scalar_domains_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    kwargs = _feature_dir_kwargs()
    kwargs[field] = value
    valid = (
        (_strict_int(value) and value > 0)
        if field == "segment_size"
        else (_strict_int(value) and value >= 0)
        if field == "layer"
        else isinstance(value, str) and bool(value.strip())
    )
    if valid:
        assert isinstance(build_feature_dir(**kwargs), Path)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            build_feature_dir(**kwargs)


@PROPERTY_SETTINGS
@given(
    value=st.one_of(
        st.integers(max_value=0),
        st.floats(allow_nan=True, allow_infinity=True, width=64),
        st.booleans(),
        st.text(max_size=16),
    )
)
def test_openrouter_max_tokens_rejects_complete_invalid_domain(value: object) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="max_tokens"):
        OpenRouterLLMClient(
            LLMClientConfig(),
            max_tokens=cast(Any, value),
        )


def test_scalar_only_boundaries_preserve_global_python_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "test-key")
    random.seed(31_415)
    before = random.getstate()

    GeneralValSignalBudget(1, 0, 0, 1)
    GeneralValWarmStartCaps(1, 1)
    build_feature_dir(**_feature_dir_kwargs())
    OpenRouterLLMClient(LLMClientConfig(), max_tokens=1)

    assert random.getstate() == before
