"""Shared base-shard document alignment contract (GA-006/GA-012)."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch as t

from errors import GradientAttributionError
from scripts.preprocess.precompute_gradient_attribution import (
    _doc_seg_offsets as gradient_doc_seg_offsets,
)
from scripts.preprocess.precompute_gradient_attribution import (
    _verify_doc_alignment as gradient_verify_doc_alignment,
)
from steering.feature_shard_splice import AlignmentError
from steering.feature_shard_splice import _doc_seg_offsets as splice_doc_seg_offsets
from steering.feature_shard_splice import (
    _verify_doc_alignment as splice_verify_doc_alignment,
)


@pytest.fixture(
    params=[
        pytest.param(
            (gradient_doc_seg_offsets, GradientAttributionError, "tensor"),
            id="gradient-producer",
        ),
        pytest.param(
            (splice_doc_seg_offsets, AlignmentError, "list"),
            id="feature-splice",
        ),
    ]
)
def offsets_adapter(request):
    return request.param


def test_canonical_doc_blocks_return_adapter_specific_boundaries(offsets_adapter):
    adapter, _, container = offsets_adapter
    got = adapter(t.tensor([0, 0, 1, 2, 2], dtype=t.int32), 3)
    if container == "tensor":
        assert isinstance(got, t.Tensor)
        assert got.dtype == t.int32
        assert got.tolist() == [0, 2, 3, 5]
    else:
        assert isinstance(got, list)
        assert got == [0, 2, 3, 5]


@pytest.mark.parametrize(
    ("doc_ids", "n_docs"),
    [
        (t.tensor([1, 1, 0, 0], dtype=t.int32), 2),  # reversed equal blocks
        (t.tensor([0, 1, 0, 1], dtype=t.int32), 2),  # interleaved
        (t.tensor([0, 0, 2, 2], dtype=t.int32), 3),  # skipped expected doc
        (t.tensor([0, 2], dtype=t.int32), 2),  # out of range
        (t.tensor([-1, 0], dtype=t.int32), 2),  # negative
        (t.tensor([[0, 0], [1, 1]], dtype=t.int32), 2),  # not one-dimensional
        (t.tensor([0.0, 1.0], dtype=t.float32), 2),  # non-integral dtype
        (t.tensor([False, True], dtype=t.bool), 2),  # bool is not an ID dtype
        (t.tensor([], dtype=t.int32), 1),  # expected doc has no block
    ],
)
def test_malformed_doc_id_geometry_raises_adapter_error(
    offsets_adapter, doc_ids, n_docs
):
    adapter, error_type, _ = offsets_adapter
    with pytest.raises(error_type):
        adapter(doc_ids, n_docs)


def _alignment_data() -> dict:
    return {
        "doc_token_counts": t.tensor([8], dtype=t.int32),
        "segment_positions": t.tensor([0, 4], dtype=t.int32),
        "segment_indices": t.tensor([0, 1], dtype=t.int32),
    }


def _call_gradient(*, data: dict, n_full: int, segments: list[tuple[int, int]]):
    gradient_verify_doc_alignment(
        shard_name="shard_00000",
        doc_idx=0,
        n_full_tokens=n_full,
        segments=segments,
        data=data,
        seg_lo=0,
        seg_hi=2,
    )


def _call_splice(*, data: dict, n_full: int, segments: list[tuple[int, int]]):
    splice_verify_doc_alignment(
        shard_name="shard_00000",
        doc_idx=0,
        n_full=n_full,
        segments=segments,
        base_shard=data,
        seg_lo=0,
        seg_hi=2,
    )


@pytest.fixture(
    params=[
        pytest.param((_call_gradient, GradientAttributionError), id="gradient-producer"),
        pytest.param((_call_splice, AlignmentError), id="feature-splice"),
    ]
)
def verify_adapter(request) -> tuple[Callable, type[Exception]]:
    return request.param


def test_canonical_per_doc_alignment_is_accepted(verify_adapter):
    adapter, _ = verify_adapter
    adapter(data=_alignment_data(), n_full=8, segments=[(0, 4), (4, 8)])


@pytest.mark.parametrize("drift", ["token_count", "segment_count", "position", "ordinal"])
def test_per_doc_drift_raises_adapter_error(verify_adapter, drift):
    adapter, error_type = verify_adapter
    data = _alignment_data()
    n_full = 8
    segments = [(0, 4), (4, 8)]
    if drift == "token_count":
        n_full = 9
    elif drift == "segment_count":
        segments = [(0, 8)]
    elif drift == "position":
        data["segment_positions"] = t.tensor([1, 4], dtype=t.int32)
    else:
        data["segment_indices"] = t.tensor([1, 0], dtype=t.int32)

    with pytest.raises(error_type):
        adapter(data=data, n_full=n_full, segments=segments)
