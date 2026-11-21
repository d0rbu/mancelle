"""Compact feature-prediction target identity and scatter goldens."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.dataloader import (
    _build_fp_targets_from_flat_features,
    build_fp_targets_from_shard,
)
from steering.conditional.encoder_decoder.feature_prediction import (
    FeaturePredictionTargetLayout,
)
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import FeatureVocab
from steering.packed_batch_io import _BatchIndex, _build_batch_index
from steering.refined_types import PositiveInt
from tests.encdec_test_helpers import _make_test_shard

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True)


@pytest.mark.parametrize("rows", [[1, 0], [1, 0, 1]])
@pytest.mark.parametrize("negative_offsets", [False, True])
def test_public_target_gather_reordered_rows_and_strided_offsets(rows, negative_offsets):
    shard = _make_test_shard(n_rows=2, segs_per_row=1, feats_per_seg=2, n_features=8)
    shard.feature_ids = t.tensor([0, 1, 3, 7], dtype=t.int64)
    shard.feature_pcts = t.tensor([51, 102, 204, 255], dtype=t.uint8)
    backing = t.tensor([-4, 99, -2, 99, 0, 99] if negative_offsets else [0, 99, 2, 99, 4, 99])
    shard.seg_feat_offsets = backing[::2]
    idx = _build_batch_index(shard, rows, pad_n_seg=3)
    mask = FeatureMask(FeatureMaskMode.whitelist, (0, 1, 3, 7))
    layout = _layout(8, mask=mask, clusters=_clusters(8, ((1, 3),), mask))

    actual = build_fp_targets_from_shard(shard, idx, fp_target_layout=layout)

    # Literal original-row goldens: IDs 1 and 3 share output column 1.
    raw_rows = t.tensor([[51, 102, 0], [0, 204, 255]], dtype=t.float32) * (1.0 / 255.0)
    expected = t.zeros(len(rows), 3, 3)
    expected[:, 0] = raw_rows[rows]
    assert t.equal(actual, expected)
    assert t.equal(shard.seg_feat_offsets, backing[::2])
    assert shard.seg_feat_offsets.stride() == (2,)


@pytest.mark.parametrize(
    "offsets,error,message",
    [
        ([0, -1, 4], RuntimeError, "repeats can not be negative"),
        ([0, 2, 5], IndexError, "index 4 is out of bounds for dimension 0 with size 4"),
    ],
)
def test_public_target_gather_retains_offset_errors(offsets, error, message):
    shard = _make_test_shard(n_rows=2, segs_per_row=1, feats_per_seg=2, n_features=8)
    shard.seg_feat_offsets = t.tensor(offsets)
    idx = _build_batch_index(shard, [0, 1])

    with pytest.raises(error) as caught:
        build_fp_targets_from_shard(shard, idx, fp_target_layout=_layout(8))

    assert str(caught.value) == message


def _clusters(
    n_sae_features: int,
    groups: tuple[tuple[int, ...], ...],
    mask: FeatureMask | None,
) -> FeatureClusterConsolidation:
    definition = FeatureClusterDefinition(clusters=groups)
    resolved = ResolvedFeatureClusters(
        source_path=Path("<compact-fp-unit>"),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition, n_sae_features=n_sae_features
        ),
    )
    return FeatureClusterConsolidation.build(resolved, mask)


def _layout(
    n_sae_features: int,
    *,
    mask: FeatureMask | None = None,
    clusters: FeatureClusterConsolidation | None = None,
) -> FeaturePredictionTargetLayout:
    vocab = effective_feature_vocab(
        clusters if clusters is not None else mask,
        n_sae_features,
    )
    return FeaturePredictionTargetLayout.build(
        n_sae_features=PositiveInt.parse(n_sae_features),
        feature_mask=mask,
        feature_clusters=clusters,
        feature_vocab=vocab,
    )


def _index(*, batch_n_seg: int = 3, total_segs: int = 3) -> _BatchIndex:
    return _BatchIndex(
        B=1,
        T=3,
        inputs=t.zeros(1, 3, dtype=t.int64),
        targets=t.zeros(1, 3, dtype=t.int64),
        batch_n_seg=batch_n_seg,
        total_segs=total_segs,
        seg_counts=t.tensor([total_segs], dtype=t.int64),
        b_flat=t.zeros(total_segs, dtype=t.int64),
        local_flat=t.arange(total_segs, dtype=t.int64),
        global_flat=t.arange(total_segs, dtype=t.int64),
        seg_map=t.arange(3, dtype=t.int64).unsqueeze(0),
        seg_doc_id=t.zeros(1, batch_n_seg, dtype=t.int64),
    )


@PROPERTY_SETTINGS
@given(st.integers(min_value=1))
def test_full_layout_accepts_complete_positive_width_domain(width: int) -> None:
    layout = FeaturePredictionTargetLayout.build(
        n_sae_features=PositiveInt.parse(width),
        feature_mask=None,
        feature_clusters=None,
        feature_vocab=None,
    )
    assert layout.n_sae_features == width
    assert layout.output_width == width


@PROPERTY_SETTINGS
@given(st.one_of(st.integers(max_value=0), st.booleans()))
def test_layout_rejects_complete_invalid_width_domain(width: int) -> None:
    runtime_width: Any = width
    with pytest.raises(BeartypeCallHintParamViolation, match="n_sae_features"):
        FeaturePredictionTargetLayout.build(
            n_sae_features=runtime_width,
            feature_mask=None,
            feature_clusters=None,
            feature_vocab=None,
        )


def test_full_identity_and_all_allowing_mask_normalize_to_full_layout():
    full = _layout(6)
    all_allowed = FeatureMask(FeatureMaskMode.whitelist, tuple(range(6)))
    normalized = _layout(6, mask=all_allowed)

    assert not full.is_compact
    assert not normalized.is_compact
    assert full.output_width == normalized.output_width == 6
    assert full.original_to_output_row is None
    assert normalized.original_to_output_row is None


@pytest.mark.parametrize(
    "mask,expected_ids",
    [
        (FeatureMask(FeatureMaskMode.blacklist, (1, 4)), (0, 2, 3, 5)),
        (FeatureMask(FeatureMaskMode.whitelist, (1, 4)), (1, 4)),
    ],
)
def test_mask_only_layout_maps_original_ids_to_compact_rows(mask, expected_ids):
    layout = _layout(6, mask=mask)
    assert layout.feature_vocab is not None
    assert layout.feature_vocab.original_ids == expected_ids
    assert layout.output_width == len(expected_ids)
    mapping = layout.original_to_output_row
    assert mapping is not None
    for row, original_id in enumerate(expected_ids):
        assert int(mapping[original_id]) == row
    for original_id in set(range(6)) - set(expected_ids):
        assert int(mapping[original_id]) == -1


def test_mask_induced_representative_shift_and_sibling_mapping():
    mask = FeatureMask(FeatureMaskMode.blacklist, (1, 6))
    clusters = _clusters(8, ((1, 3, 5), (2, 6)), mask)
    layout = _layout(8, mask=mask, clusters=clusters)

    assert clusters.rep_of[[3, 5]].tolist() == [3, 3]
    assert clusters.rep_of[2].item() == 2
    assert layout.feature_vocab is not None
    assert layout.feature_vocab.original_ids == (0, 2, 3, 4, 7)
    mapping = layout.original_to_output_row
    assert mapping is not None
    assert mapping[[3, 5]].tolist() == [2, 2]
    assert int(mapping[1]) == -1
    assert int(mapping[6]) == -1


def test_entire_cluster_masked_and_unclustered_allowed_ids_survive():
    mask = FeatureMask(FeatureMaskMode.whitelist, (0, 4, 7))
    clusters = _clusters(8, ((1, 3, 5), (2, 6)), mask)
    layout = _layout(8, mask=mask, clusters=clusters)
    assert layout.feature_vocab is not None
    assert layout.feature_vocab.original_ids == (0, 4, 7)
    assert not layout.has_active_clusters


def test_layout_rejects_mismatched_mask_cluster_vocab_and_width():
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 3, 5))
    other_mask = FeatureMask(FeatureMaskMode.whitelist, (1, 4, 5))
    clusters = _clusters(8, ((1, 3),), mask)
    vocab = effective_feature_vocab(clusters, 8)
    assert vocab is not None

    with pytest.raises(ValueError, match="different FeatureMask"):
        FeaturePredictionTargetLayout.build(
            n_sae_features=PositiveInt.parse(8),
            feature_mask=other_mask,
            feature_clusters=clusters,
            feature_vocab=vocab,
        )
    with pytest.raises(ValueError, match="does not equal"):
        FeaturePredictionTargetLayout.build(
            n_sae_features=PositiveInt.parse(8),
            feature_mask=mask,
            feature_clusters=clusters,
            feature_vocab=FeatureVocab(8, (3, 5)),
        )
    with pytest.raises(ValueError, match="n_sae_features"):
        FeaturePredictionTargetLayout.build(
            n_sae_features=PositiveInt.parse(7),
            feature_mask=mask,
            feature_clusters=clusters,
            feature_vocab=vocab,
        )


def test_mask_only_compact_target_is_exact_legacy_column_slice():
    mask = FeatureMask(FeatureMaskMode.whitelist, (0, 2, 4, 6))
    layout = _layout(8, mask=mask)
    idx = _index()
    ids = t.tensor([0, 2, 7, 4, 6], dtype=t.int64)
    pcts = t.tensor([1.0, 0.2, 0.9, 128 / 255, 64 / 255], dtype=t.float32)
    sizes = t.tensor([3, 0, 2], dtype=t.int64)

    legacy = _build_fp_targets_from_flat_features(
        idx, ids, pcts, sizes, 8, feature_mask=mask
    )
    compact = _build_fp_targets_from_flat_features(
        idx, ids, pcts, sizes, fp_target_layout=layout
    )
    assert layout.feature_vocab is not None
    assert t.equal(
        compact,
        legacy[..., t.tensor(layout.feature_vocab.original_ids)],
    )
    assert compact.dtype is t.float32
    assert t.count_nonzero(compact[:, 1]) == 0


def test_cluster_target_max_reduces_distinct_siblings_without_full_intermediate():
    clusters = _clusters(8, ((1, 3, 5),), None)
    layout = _layout(8, clusters=clusters)
    idx = _index(batch_n_seg=2, total_segs=2)
    ids = t.tensor([1, 3, 5, 7], dtype=t.int64)
    pcts = t.tensor([0.2, 0.9, 0.4, 128 / 255], dtype=t.float32)
    target = _build_fp_targets_from_flat_features(
        idx,
        ids,
        pcts,
        t.tensor([3, 1]),
        fp_target_layout=layout,
    )

    assert target.shape == (1, 2, 6)
    assert layout.feature_vocab is not None
    representative_row = layout.feature_vocab.original_ids.index(1)
    feature_7_row = layout.feature_vocab.original_ids.index(7)
    assert target[0, 0, representative_row].item() == pytest.approx(0.9)
    assert target[0, 1, feature_7_row].item() == pytest.approx(128 / 255)
    assert t.count_nonzero(target[0, 1]).item() == 1


def test_compact_target_rejects_negative_and_out_of_range_original_ids():
    layout = _layout(
        5, mask=FeatureMask(FeatureMaskMode.whitelist, (0, 1, 2))
    )
    idx = _index(batch_n_seg=1, total_segs=1)
    for bad in (-1, 5):
        with pytest.raises(AssertionError):
            _build_fp_targets_from_flat_features(
                idx,
                t.tensor([bad]),
                t.tensor([0.5]),
                t.tensor([1]),
                fp_target_layout=layout,
            )
