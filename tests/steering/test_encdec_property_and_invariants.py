"""Slice-10 audit: property + invariant tests pinning the contracts the
model depends on.

Covers plan T-6, T-7, T-8.

* T-6: ``seg_map`` is non-decreasing along T after ``_build_row_segments``.
* T-7: ``resolve_encoder_dim`` returns a positive multiple-of-128, monotonic
  in the input ratio, and bounded above by ``2 * decoder_dim`` rounded up.
* T-8: ``setup_optimizer`` parameter-group total equals
  ``len(list(model.parameters()))`` across multiple random configs.

See slice-10 map §7.5.
"""

from __future__ import annotations

from typing import cast

import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

from steering.conditional.encoder_decoder.config import (
    resolve_encoder_dim as _resolve_encoder_dim,
)
from steering.refined_types import PositiveFloat, PositiveInt


def resolve_encoder_dim(n_embd: int | float, decoder_dim: int) -> PositiveInt:
    """Pass raw property examples through the production beartype boundary."""
    return _resolve_encoder_dim(
        cast(PositiveInt | PositiveFloat, n_embd),
        cast(PositiveInt, decoder_dim),
    )

# ---------------------------------------------------------------------------
# T-6: seg_map non-decreasing
# ---------------------------------------------------------------------------


def test_seg_map_non_decreasing_after_build_row_segments():
    """Construct a synthetic packed row and verify seg_map[t] is monotone.

    NOTE: ``SegFeatures.token_start`` / ``token_end`` are **doc-relative**
    indices (offsets into the document, not the packed row), so each doc's
    segments use 0-based offsets even though the doc itself starts mid-row.
    """
    from steering.conditional.encoder_decoder.dataloader import (
        SegFeatures,
        _build_row_segments,
    )

    T = 64
    # Two docs in one packed row, each with 2 segments. Doc A occupies
    # row positions 0..20, doc B occupies 20..64.
    sf_a0 = SegFeatures(
        feature_ids=torch.tensor([1, 2], dtype=torch.long),
        buckets=torch.tensor([1, 2], dtype=torch.long),
        percentiles=torch.tensor([0.5, 0.6], dtype=torch.float32),
        char_start=0, char_end=10,
        token_start=0, token_end=10,  # doc A tokens 0..10
    )
    sf_a1 = SegFeatures(
        feature_ids=torch.tensor([3], dtype=torch.long),
        buckets=torch.tensor([1], dtype=torch.long),
        percentiles=torch.tensor([0.7], dtype=torch.float32),
        char_start=10, char_end=20,
        token_start=10, token_end=20,  # doc A tokens 10..20
    )
    sf_b0 = SegFeatures(
        feature_ids=torch.tensor([4, 5], dtype=torch.long),
        buckets=torch.tensor([2, 3], dtype=torch.long),
        percentiles=torch.tensor([0.8, 0.9], dtype=torch.float32),
        char_start=0, char_end=15,
        token_start=0, token_end=15,  # doc B tokens 0..15 (relative to doc B!)
    )
    sf_b1 = SegFeatures(
        feature_ids=torch.tensor([6], dtype=torch.long),
        buckets=torch.tensor([0], dtype=torch.long),
        percentiles=torch.tensor([0.4], dtype=torch.float32),
        char_start=15, char_end=30,
        token_start=15, token_end=44,  # doc B tokens 15..44
    )
    row_docs = [
        (0, [sf_a0, sf_a1], list(range(20))),
        (20, [sf_b0, sf_b1], list(range(44))),
    ]
    out = _build_row_segments(row_docs, T, tokenizer=None)  # ty: ignore[invalid-argument-type]
    seg_map = out[3]
    assert seg_map.shape == (T,), f"unexpected seg_map shape: {seg_map.shape}"
    diff = seg_map[1:] - seg_map[:-1]
    assert (diff >= 0).all(), (
        f"seg_map must be non-decreasing along T; got delta with negative "
        f"entries at positions {torch.nonzero(diff < 0).flatten().tolist()}; "
        f"seg_map={seg_map.tolist()}"
    )
    # And we should see all 4 distinct segment indices appear.
    distinct = set(seg_map.tolist())
    assert distinct == {0, 1, 2, 3}, f"expected 4 segments, got {distinct}"


def test_seg_map_dtype_int64():
    """Pin the dtype contract: seg_map is int64 (consumers index with it)."""
    from steering.conditional.encoder_decoder.dataloader import (
        SegFeatures,
        _build_row_segments,
    )

    sf = SegFeatures(
        feature_ids=torch.tensor([0], dtype=torch.long),
        buckets=torch.tensor([0], dtype=torch.long),
        percentiles=torch.tensor([0.5], dtype=torch.float32),
        char_start=0, char_end=10,
        # token_start/end are doc-relative
        token_start=0, token_end=10,
    )
    out = _build_row_segments([(0, [sf], list(range(10)))], 10, tokenizer=None)
    seg_map = out[3]
    assert seg_map.dtype == torch.int64


# ---------------------------------------------------------------------------
# T-7: resolve_encoder_dim contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decoder_dim", [256, 512, 768, 1024, 1536])
@pytest.mark.parametrize("ratio", [0.25, 0.5, 0.75, 1.0, 1.5])
def test_resolve_encoder_dim_multiple_of_128(decoder_dim, ratio):
    """Float ratio is rounded to the nearest multiple of 128, ≥ 128."""
    out = resolve_encoder_dim(ratio, decoder_dim)
    assert out >= 128, f"got {out} for ratio={ratio}, decoder_dim={decoder_dim}"
    assert out % 128 == 0, f"got {out}, not a multiple of 128"


@pytest.mark.parametrize("decoder_dim", [512, 768])
def test_resolve_encoder_dim_monotonic_in_ratio(decoder_dim):
    """Larger ratio → ≥ resolved dim. Strict for big jumps; non-strict OK
    when both rounding bins land on the same multiple of 128."""
    ratios = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    resolved = [resolve_encoder_dim(r, decoder_dim) for r in ratios]
    for i in range(len(resolved) - 1):
        assert resolved[i] <= resolved[i + 1], f"non-monotonic: {resolved}"


@pytest.mark.parametrize("decoder_dim", [256, 512, 1024])
def test_resolve_encoder_dim_bounded_above_by_2x_decoder(decoder_dim):
    """For ratio ≤ 2.0, the resolved dim is ≤ ceil(2 * decoder_dim / 128) * 128."""
    upper = ((2 * decoder_dim + 127) // 128) * 128
    out = resolve_encoder_dim(2.0, decoder_dim)
    assert out <= upper


def test_resolve_encoder_dim_int_passthrough():
    """Int values are passed through unmodified (no 128 rounding)."""
    assert resolve_encoder_dim(7, 768) == 7
    assert resolve_encoder_dim(257, 768) == 257


@pytest.mark.parametrize("bad", [0.0, -0.5, 0, -3])
def test_resolve_encoder_dim_rejects_non_positive(bad):
    """Zero or negative inputs always raise, regardless of int vs float."""
    with pytest.raises(BeartypeCallHintParamViolation, match="n_embd"):
        resolve_encoder_dim(bad, decoder_dim=512)


# ---------------------------------------------------------------------------
# T-8: param-group total = model parameter count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_layer,n_embd",
    [
        (2, 64),
        (3, 128),
        (4, 64),
    ],
)
def test_param_groups_total_equals_model_param_count(n_layer, n_embd):
    """Across multiple random configs, the per-group param count sums to the
    total number of parameters in the model.

    The model itself asserts this internally via ``len(list(self.parameters()))``.
    Here we re-verify it explicitly so a regression on group enumeration is
    caught at the test level (not just via training-time assertion).
    """
    from tests.encdec_test_helpers import build_tiny_model

    model = build_tiny_model(n_sae_features=64, n_layer=n_layer, n_embd=n_embd)
    optimizer = model.setup_optimizer()
    grouped_params = [p for g in optimizer.param_groups for p in g["params"]]
    model_params = list(model.parameters())
    assert len(grouped_params) == len(model_params)
    # Param ids should also be a 1:1 set match.
    ids_grouped = sorted(id(p) for p in grouped_params)
    ids_model = sorted(id(p) for p in model_params)
    assert ids_grouped == ids_model
