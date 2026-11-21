"""Property coverage for refined encoder-decoder scalar boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

from scripts.train.conditional_encdec_selfcond import (
    resolve_grad_accum_steps,
    resolve_train_prefetch_batches,
)
from scripts.train.conditional_encdec_sft import SFTProfileSettings
from scripts.train.run_encdec_sft_pipeline import run_pipeline, run_sft
from steering.conditional.encoder_decoder.config import (
    FeaturePredLossMode,
    resolve_encoder_depth,
    resolve_encoder_dim,
    resolve_encoder_heads,
)
from steering.conditional.encoder_decoder.dataloader import resolve_train_shard_lookahead
from steering.conditional.encoder_decoder.sae_embedding_init import (
    EncoderEmbeddingInit,
    project_sae_decoder_matrix,
)
from steering.conditional.encoder_decoder.selfcond_training import (
    pool_fp_logits_per_segment,
)
from steering.refined_types import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True)
_POSITIVE_ONE = PositiveInt.parse(1)
_POSITIVE_FOUR = PositiveInt.parse(4)
_POSITIVE_SIXTY_FOUR = PositiveInt.parse(64)

_INVALID_POSITIVE_INT = st.one_of(st.booleans(), st.integers(max_value=0))
_INVALID_NON_NEGATIVE_INT = st.one_of(st.booleans(), st.integers(max_value=-1))
_INVALID_POSITIVE_FLOAT = st.one_of(
    st.floats(max_value=0.0, allow_nan=False, allow_infinity=True),
    st.just(float("nan")),
)
_INVALID_NON_NEGATIVE_FLOAT = st.one_of(
    st.floats(max_value=-0.0, exclude_max=True, allow_nan=False, allow_infinity=True),
    st.just(float("nan")),
)


@PROPERTY_SETTINGS
@given(_INVALID_POSITIVE_INT)
def test_encoder_dim_rejects_every_invalid_absolute_dimension(value: object) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="n_embd"):
        resolve_encoder_dim(
            cast(PositiveInt | PositiveFloat, value),
            _POSITIVE_SIXTY_FOUR,
        )


@PROPERTY_SETTINGS
@given(_INVALID_POSITIVE_FLOAT)
def test_encoder_dim_rejects_every_invalid_ratio(value: object) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="n_embd"):
        resolve_encoder_dim(
            cast(PositiveInt | PositiveFloat, value),
            _POSITIVE_SIXTY_FOUR,
        )


@PROPERTY_SETTINGS
@given(st.one_of(_INVALID_NON_NEGATIVE_INT, _INVALID_NON_NEGATIVE_FLOAT))
def test_encoder_depth_rejects_every_invalid_absolute_or_ratio(value: object) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="depth"):
        resolve_encoder_depth(
            cast(NonNegativeInt | NonNegativeFloat, value),
            _POSITIVE_FOUR,
        )


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("n_head", "n_kv_head", "enc_dim", "head_dim")),
    value=_INVALID_POSITIVE_INT,
)
def test_encoder_heads_reject_every_invalid_unary_field(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, Any] = {
        "n_head": _POSITIVE_FOUR,
        "n_kv_head": _POSITIVE_ONE,
        "enc_dim": _POSITIVE_SIXTY_FOUR,
        "head_dim": PositiveInt.parse(16),
    }
    kwargs[field] = value
    with pytest.raises(BeartypeCallHintParamViolation, match=field):
        resolve_encoder_heads(**kwargs)


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("device_batch_size", "sequence_len", "ddp_world_size")),
    value=_INVALID_POSITIVE_INT,
)
def test_grad_accum_rejects_every_invalid_positive_factor(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, Any] = {
        "device_batch_size": _POSITIVE_ONE,
        "sequence_len": _POSITIVE_SIXTY_FOUR,
        "ddp_world_size": _POSITIVE_ONE,
        "total_batch_size": 64,
    }
    kwargs[field] = value
    with pytest.raises(BeartypeCallHintParamViolation, match=field):
        resolve_grad_accum_steps(**kwargs)


@PROPERTY_SETTINGS
@given(
    device_batch_size=st.integers(min_value=1, max_value=32),
    sequence_len=st.integers(min_value=1, max_value=4096),
    world_size=st.integers(min_value=1, max_value=64),
    steps=st.integers(min_value=1, max_value=128),
)
def test_grad_accum_formula_holds_across_the_complete_positive_domain_sample(
    device_batch_size: int,
    sequence_len: int,
    world_size: int,
    steps: int,
) -> None:
    micro_tokens = device_batch_size * sequence_len * world_size
    assert (
        resolve_grad_accum_steps(
            PositiveInt.parse(device_batch_size),
            PositiveInt.parse(sequence_len),
            PositiveInt.parse(world_size),
            micro_tokens * steps,
        )
        == steps
    )


@PROPERTY_SETTINGS
@given(
    st.one_of(
        st.booleans(),
        st.integers(max_value=-1),
        st.floats(allow_nan=True, allow_infinity=True),
        st.text(),
    )
)
def test_prefetch_parser_rejects_every_non_nonnegative_integer(value: object) -> None:
    with pytest.raises(TypeError):
        resolve_train_prefetch_batches(value, _POSITIVE_ONE)


@PROPERTY_SETTINGS
@given(
    st.one_of(
        st.booleans(),
        st.integers(max_value=-1),
        st.floats(allow_nan=True, allow_infinity=True),
        st.text(),
    )
)
def test_shard_lookahead_parser_rejects_every_non_nonnegative_integer(value: object) -> None:
    with pytest.raises(TypeError, match="Could not parse.*NonNegativeInt"):
        resolve_train_shard_lookahead(value)


@PROPERTY_SETTINGS
@given(_INVALID_POSITIVE_INT)
def test_projection_rejects_every_invalid_encoder_dimension(value: object) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="encoder_dim"):
        project_sae_decoder_matrix(
            t.ones(2, 2),
            cast(PositiveInt, value),
            EncoderEmbeddingInit.RANDOM,
            seed=0,
        )


@PROPERTY_SETTINGS
@given(_INVALID_POSITIVE_INT)
def test_segment_pooling_rejects_every_invalid_segment_count(value: object) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="n_seg"):
        pool_fp_logits_per_segment(
            t.zeros(1, 1, 1),
            t.zeros(1, 1, dtype=t.int64),
            cast(PositiveInt, value),
            FeaturePredLossMode.MAX_POOL,
        )


@PROPERTY_SETTINGS
@given(_INVALID_POSITIVE_INT)
def test_sft_pipeline_rejects_every_invalid_gpu_count(value: object) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="num_gpus"):
        run_pipeline(
            model_tag="property-test",
            num_gpus=cast(PositiveInt, value),
        )


@PROPERTY_SETTINGS
@given(_INVALID_POSITIVE_INT)
def test_sft_phase_rejects_every_invalid_gpu_count(value: object) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="num_gpus"):
        run_sft(
            dataset="safety",
            model_tag="property-test",
            sft_overrides=[],
            num_gpus=cast(PositiveInt, value),
        )


@PROPERTY_SETTINGS
@given(_INVALID_POSITIVE_INT)
def test_profile_settings_reject_every_invalid_trace_length(value: object) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="trace_steps"):
        SFTProfileSettings(
            output_dir=Path("profile"),
            trace_step=None,
            trace_steps=cast(PositiveInt, value),
            timing_enabled=False,
            timing_sync_cuda=False,
        )


@PROPERTY_SETTINGS
@given(_INVALID_NON_NEGATIVE_INT)
def test_profile_settings_reject_every_invalid_trace_start(value: object) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="trace_step"):
        SFTProfileSettings(
            output_dir=Path("profile"),
            trace_step=cast(NonNegativeInt, value),
            trace_steps=_POSITIVE_ONE,
            timing_enabled=False,
            timing_sync_cuda=False,
        )
