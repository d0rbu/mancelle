"""Property oracles for scalar-only training runtime boundaries."""

from __future__ import annotations

import math
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypeGuard, cast

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import nn

from steering.conditional.encoder_decoder.dataloader import (
    _CanonicalLabelDropoutStream,
    _encode_rng_snapshot,
)
from steering.conditional.encoder_decoder.precomputed_cuda_staging import (
    PrecomputedCudaStagingLoader,
)
from steering.conditional.splice_provenance import SpliceProvenanceIndex
from steering.eval.axbench_base.baselines.lora.module import LoRAAdapter
from steering.feature_shard_splice import PROVENANCE_DIRNAME
from steering.sft.clipping import resolve_grad_clip_max_norm

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True, deadline=None)
SCALARS = st.one_of(
    st.integers(min_value=-100, max_value=100),
    st.floats(allow_nan=True, allow_infinity=True, width=64),
    st.booleans(),
    st.none(),
    st.text(max_size=16),
)
INVALID_POSITIVE_INTS = st.one_of(
    st.integers(min_value=-100, max_value=0),
    st.floats(allow_nan=True, allow_infinity=True, width=64),
    st.booleans(),
    st.text(max_size=16),
)


def _strict_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_finite_number(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _dropout_snapshot() -> dict[str, Any]:
    return _encode_rng_snapshot(t.get_rng_state(), t.device("cpu"))


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_grad_clip_domain_matches_independent_oracle(value: object) -> None:
    if value is None:
        assert resolve_grad_clip_max_norm(None) is None
    elif _positive_finite_number(value):
        assert resolve_grad_clip_max_norm(value) == float(value)
    else:
        with pytest.raises(ValueError, match="grad_clip_max_norm"):
            resolve_grad_clip_max_norm(value)


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_lora_rank_domain_matches_independent_oracle(value: object) -> None:
    base = nn.Linear(2, 2)
    if _strict_int(value) and value > 0:
        assert LoRAAdapter(base, rank=value, alpha=1).rank == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="rank"):
            LoRAAdapter(base, rank=cast(Any, value), alpha=1)


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_lora_alpha_domain_matches_independent_oracle(value: object) -> None:
    base = nn.Linear(2, 2)
    if _positive_finite_number(value):
        assert LoRAAdapter(base, rank=1, alpha=cast(Any, value)).alpha == float(value)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="alpha"):
            LoRAAdapter(base, rank=1, alpha=cast(Any, value))


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_splice_cache_capacity_domain_matches_independent_oracle(value: object) -> None:
    if _strict_int(value) and value > 0:
        with TemporaryDirectory() as root:
            Path(root, PROVENANCE_DIRNAME).mkdir()
            index = SpliceProvenanceIndex(root, max_resident_shards=value)
            assert index.max_resident_shards == value
    else:
        with pytest.raises(
            BeartypeCallHintParamViolation,
            match="max_resident_shards",
        ):
            SpliceProvenanceIndex("unused", max_resident_shards=cast(Any, value))


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_splice_cache_compact_domain_matches_independent_oracle(value: object) -> None:
    if isinstance(value, bool):
        with TemporaryDirectory() as root:
            Path(root, PROVENANCE_DIRNAME).mkdir()
            SpliceProvenanceIndex(root, compact=value)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="compact"):
            SpliceProvenanceIndex("unused", compact=cast(Any, value))


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_dropout_draw_width_domain_matches_independent_oracle(value: object) -> None:
    if _strict_int(value) and value > 0:
        stream = _CanonicalLabelDropoutStream(
            _dropout_snapshot(),
            device="cpu",
            draw_batch_size=value,
        )
        assert stream.draw_batch_size == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="draw_batch_size"):
            _CanonicalLabelDropoutStream(
                _dropout_snapshot(),
                device="cpu",
                draw_batch_size=cast(Any, value),
            )


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_dropout_draw_offset_preserves_relational_contract(value: object) -> None:
    if _strict_int(value) and 0 <= value < 10:
        stream = _CanonicalLabelDropoutStream(
            _dropout_snapshot(),
            device="cpu",
            draw_batch_size=10,
            draw_offset=value,
        )
        assert stream.offset == value
    elif _strict_int(value) and value >= 10:
        with pytest.raises(ValueError, match="draw_offset"):
            _CanonicalLabelDropoutStream(
                _dropout_snapshot(),
                device="cpu",
                draw_batch_size=10,
                draw_offset=value,
            )
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="draw_offset"):
            _CanonicalLabelDropoutStream(
                _dropout_snapshot(),
                device="cpu",
                draw_batch_size=10,
                draw_offset=cast(Any, value),
            )


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(
        (
            "host_queue_bytes",
            "pinned_slots",
            "expected_batch_size",
            "expected_context_len",
            "expected_pad_f",
            "expected_n_seg",
        )
    ),
    value=INVALID_POSITIVE_INTS,
)
def test_staging_loader_rejects_every_invalid_positive_scalar(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, Any] = {
        "device": "cuda:0",
        "host_queue_bytes": 1,
        "pinned_slots": 1,
        "expected_batch_size": 1,
        "expected_context_len": 1,
        "expected_pad_f": 1,
        "expected_n_seg": 1,
    }
    kwargs[field] = value
    with pytest.raises(BeartypeCallHintParamViolation, match=field):
        PrecomputedCudaStagingLoader(iter(()), **kwargs)


def test_scalar_only_boundaries_preserve_global_python_rng() -> None:
    random.seed(31_415)
    before = random.getstate()

    resolve_grad_clip_max_norm(1)
    LoRAAdapter(nn.Linear(2, 2), rank=1, alpha=1)
    _CanonicalLabelDropoutStream(
        _dropout_snapshot(),
        device="cpu",
        draw_batch_size=1,
    )
    with TemporaryDirectory() as root:
        Path(root, PROVENANCE_DIRNAME).mkdir()
        SpliceProvenanceIndex(root)

    assert random.getstate() == before
