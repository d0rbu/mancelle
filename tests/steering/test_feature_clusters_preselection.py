"""Producer-level tests for static feature clustering in preselection.

The ordinary tests serialize full selection shards. Splice ordering and
representative-level delta semantics live in
``test_clustered_splice_counterfactual_delta.py``; this module retains the
identity-after-mask parity and shared materialization/support contracts.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch as t

from errors import FeatureClustersError
from scripts.preprocess.precompute_selected_features import (
    _materialize_splice_segment_selection,
    _validate_clustered_selection_outputs,
    assert_feature_clusters_precompute_supported,
    select_features_for_shard,
)
from steering.conditional.delta import (
    DeltaMode,
    DeltaSamplingWeights,
    FeatureDeltaConfig,
    compute_doc_counterfactual_deltas,
)
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    build_clustered_selection_shard_integrity,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_pipeline import (
    DeltaBundle,
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
)
from steering.conditional.percentile_table import (
    magnitude_to_percentile,
    percentile_to_uint8,
    uint8_to_percentile,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeBundle,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
    ThresholdSelection,
)
from steering.conditional.splice_counterfactual import (
    resolve_packed_row_fingerprint_strict,
)
from steering.conditional.thresholding import ThresholdType
from steering.overlay_io import overlay_shard_path
from steering.packed_batch_io import (
    PackedRowShard,
    SelectionShard,
    load_selection_shard,
    save_packed_row_shard,
    save_selection_shard,
)
from tests.reference_feature_clustering import (
    HistoricalFeatureConsolidation,
    HistoricalPreselectionAdapter,
)

N_SAE = 32


def _clusters(
    tmp_path: Path,
    *,
    mask: FeatureMask | None = None,
    groups: list[list[int]] | None = None,
    name: str = "clusters.json",
) -> FeatureClusterConsolidation:
    source = tmp_path / name
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "clusters": groups or [[1, 3, 5], [7, 8]],
            }
        ),
        encoding="utf-8",
    )
    resolved = resolve_feature_clusters(
        {"feature_clusters": {"path": str(source)}},
        N_SAE,
    )
    assert resolved is not None
    return FeatureClusterConsolidation.build(resolved, mask)


def _base_shard(tmp_path: Path, *, stem: str = "shard_00000") -> tuple[PackedRowShard, Path]:
    shard = PackedRowShard(
        tokens=t.arange(2 * 9, dtype=t.int64).reshape(2, 9).add(1).to(t.uint16),
        row_seg_offsets=t.tensor([0, 1, 2], dtype=t.int32),
        seg_token_start=t.tensor([0, 0], dtype=t.uint16),
        seg_token_end=t.tensor([8, 8], dtype=t.uint16),
        seg_doc_id=t.tensor([0, 0], dtype=t.uint8),
        seg_feat_offsets=t.tensor([0, 4, 7], dtype=t.int32),
        feature_ids=t.tensor([1, 3, 5, 10, 7, 8, 9], dtype=t.uint16),
        feature_pcts=t.tensor([20, 100, 240, 150, 80, 200, 120], dtype=t.uint8),
    )
    base_dir = tmp_path / "base"
    base_dir.mkdir(exist_ok=True)
    path = save_packed_row_shard(shard, base_dir / stem)
    return shard, path


def _threshold_pipeline(
    mask: FeatureMask | None,
    *,
    threshold_type: ThresholdType = ThresholdType.feature_percentile,
    threshold: float = 0.0,
) -> FeaturePipelineConfig:
    return FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(threshold_type, threshold),
            precomputed_thresholds=t.zeros(N_SAE, dtype=t.float32),
        ),
        boundaries=t.zeros(N_SAE, 3, dtype=t.float64),
        negative_bundle=None,
        effective_buckets=4,
        feature_mask=mask,
    )


def _percentile_table() -> t.Tensor:
    # p1..p99 are exactly 1..99 for every feature. This gives transparent
    # fallback expectations when a counterfactual segment lacks stored pcts.
    return t.arange(1, 100, dtype=t.float64).unsqueeze(0).expand(N_SAE, -1).clone()


def _sampling_pipeline(
    *,
    mask: FeatureMask | None = None,
    negative: bool = False,
    delta_weights: DeltaSamplingWeights | None = DeltaSamplingWeights.percentile_delta,
    seed: int = 17,
) -> FeaturePipelineConfig:
    table = _percentile_table()
    selection = SamplingSelection(
        distribution=Distribution.normal,
        mean=2.0,
        std=0.01,
        delta_weights=delta_weights,
        min_k=2,
        max_k=3,
        mode=SamplingMode.top_k_percentile,
    )
    negative_bundle = None
    if negative:
        negative_bundle = NegativeBundle(
            selection=NegativeSamplingSelection(
                distribution=Distribution.normal,
                mean=2.0,
                std=0.01,
                min_k=2,
                max_k=3,
            ),
            n_sae_features=N_SAE,
            rng=t.Generator().manual_seed(seed + 1),
            feature_mask=mask,
        )
    return FeaturePipelineConfig(
        selection_config=SamplingBundle(
            selection=selection,
            percentile_table=table,
            rng=t.Generator().manual_seed(seed),
        ),
        boundaries=t.zeros(N_SAE, 3, dtype=t.float64),
        negative_bundle=negative_bundle,
        effective_buckets=3 if negative else 4,
        delta_bundle=DeltaBundle(
            config=FeatureDeltaConfig(
                mode=DeltaMode.splice_counterfactual,
                neg_delta_ratio=1.0,
            ),
            percentile_table=table,
        ),
        feature_mask=mask,
    )


def _identity_splice_pipeline(
    mask: FeatureMask,
    *,
    negative: bool,
    seed: int,
) -> FeaturePipelineConfig:
    """Production-shaped three-positive-bucket splice sampling pipeline."""
    table = _percentile_table()
    selection = SamplingSelection(
        distribution=Distribution.normal,
        mean=2.0,
        std=0.01,
        delta_weights=DeltaSamplingWeights.percentile_delta,
        min_k=2,
        max_k=3,
        mode=SamplingMode.top_k_percentile,
    )
    negative_bundle = None
    if negative:
        negative_bundle = NegativeBundle(
            selection=NegativeSamplingSelection(
                distribution=Distribution.normal,
                mean=2.0,
                std=0.01,
                min_k=2,
                max_k=3,
            ),
            n_sae_features=N_SAE,
            rng=t.Generator().manual_seed(seed + 1),
            feature_mask=mask,
        )
    boundaries = t.empty((N_SAE, 2), dtype=t.float64)
    boundaries[:, 0] = 33.0
    boundaries[:, 1] = 66.0
    return FeaturePipelineConfig(
        selection_config=SamplingBundle(
            selection=selection,
            percentile_table=table,
            rng=t.Generator().manual_seed(seed),
        ),
        boundaries=boundaries,
        negative_bundle=negative_bundle,
        effective_buckets=3,
        delta_bundle=DeltaBundle(
            config=FeatureDeltaConfig(
                mode=DeltaMode.splice_counterfactual,
                neg_delta_ratio=1.0,
            ),
            percentile_table=table,
        ),
        feature_mask=mask,
    )


def _selection_tensors(selection: SelectionShard) -> tuple[t.Tensor, ...]:
    return (
        selection.sel_offsets,
        selection.sel_fids,
        selection.sel_buckets,
        selection.sel_pcts,
        selection.neg_offsets,
        selection.neg_fids,
    )


def test_ordinary_preselection_applies_mask_before_choosing_representative(
    tmp_path: Path,
):
    _shard, shard_path = _base_shard(tmp_path)
    mask = FeatureMask(FeatureMaskMode.blacklist, (1,))
    clusters = _clusters(tmp_path, mask=mask)
    out_dir = tmp_path / "selection"
    out_dir.mkdir()

    select_features_for_shard(
        shard_path,
        _threshold_pipeline(mask),
        out_dir,
        "shard_00000",
        feature_clusters=clusters,
    )
    selection = load_selection_shard(overlay_shard_path(out_dir, "shard_00000"))

    assert selection.sel_offsets.tolist() == [0, 2, 4]
    assert selection.sel_fids.tolist() == [3, 10, 7, 9]
    assert 1 not in selection.sel_fids.tolist()
    assert 5 not in selection.sel_fids.tolist()
    assert 8 not in selection.sel_fids.tolist()


def test_ordinary_preselection_uses_max_member_percentile_for_rep_and_bucket(
    tmp_path: Path,
):
    _shard, shard_path = _base_shard(tmp_path)
    mask = FeatureMask(FeatureMaskMode.blacklist, (1,))
    clusters = _clusters(tmp_path, mask=mask)
    out_dir = tmp_path / "selection"
    out_dir.mkdir()
    pipeline = _threshold_pipeline(mask)

    select_features_for_shard(
        shard_path,
        pipeline,
        out_dir,
        "shard_00000",
        feature_clusters=clusters,
    )
    selection = load_selection_shard(overlay_shard_path(out_dir, "shard_00000"))
    by_id = {
        int(fid): (int(pct), int(bucket))
        for fid, pct, bucket in zip(
            selection.sel_fids,
            selection.sel_pcts,
            selection.sel_buckets,
            strict=True,
        )
    }

    assert by_id[3] == (240, 3)
    assert by_id[7] == (200, 3)
    assert by_id[10] == (150, 2)
    assert by_id[9] == (120, 1)


def test_whitelist_single_survivor_remains_its_original_id(tmp_path: Path):
    _shard, shard_path = _base_shard(tmp_path)
    mask = FeatureMask(FeatureMaskMode.whitelist, (5, 9))
    clusters = _clusters(tmp_path, mask=mask)
    out_dir = tmp_path / "selection"
    out_dir.mkdir()

    select_features_for_shard(
        shard_path,
        _threshold_pipeline(mask),
        out_dir,
        "shard_00000",
        feature_clusters=clusters,
    )
    selection = load_selection_shard(overlay_shard_path(out_dir, "shard_00000"))
    assert selection.sel_fids.tolist() == [5, 9]
    assert selection.sel_pcts.tolist() == [240, 120]


def test_cluster_with_no_mask_survivors_emits_nothing_for_that_cluster(tmp_path: Path):
    _shard, shard_path = _base_shard(tmp_path)
    mask = FeatureMask(FeatureMaskMode.blacklist, (1, 3, 5))
    clusters = _clusters(tmp_path, mask=mask)
    out_dir = tmp_path / "selection"
    out_dir.mkdir()

    select_features_for_shard(
        shard_path,
        _threshold_pipeline(mask),
        out_dir,
        "shard_00000",
        feature_clusters=clusters,
    )
    selection = load_selection_shard(overlay_shard_path(out_dir, "shard_00000"))
    assert selection.sel_fids.tolist() == [10, 7, 9]


def test_segment_topk_operates_after_consolidation(tmp_path: Path):
    _shard, shard_path = _base_shard(tmp_path)
    mask = FeatureMask(FeatureMaskMode.blacklist, (1,))
    clusters = _clusters(tmp_path, mask=mask)
    out_dir = tmp_path / "selection"
    out_dir.mkdir()

    select_features_for_shard(
        shard_path,
        _threshold_pipeline(
            mask,
            threshold_type=ThresholdType.feature_percentile_segment_topk,
            threshold=1.01,
        ),
        out_dir,
        "shard_00000",
        feature_clusters=clusters,
    )
    selection = load_selection_shard(overlay_shard_path(out_dir, "shard_00000"))
    assert selection.sel_offsets.tolist() == [0, 1, 2]
    assert selection.sel_fids.tolist() == [3, 7]
    assert selection.sel_pcts.tolist() == [240, 200]


def test_disabled_cluster_argument_is_exactly_equal_to_legacy_omission(tmp_path: Path):
    _shard, shard_path = _base_shard(tmp_path)
    pipeline_a = _threshold_pipeline(None)
    pipeline_b = _threshold_pipeline(None)
    omitted_dir = tmp_path / "omitted"
    explicit_dir = tmp_path / "explicit"
    omitted_dir.mkdir()
    explicit_dir.mkdir()

    select_features_for_shard(
        shard_path,
        pipeline_a,
        omitted_dir,
        "shard_00000",
    )
    select_features_for_shard(
        shard_path,
        pipeline_b,
        explicit_dir,
        "shard_00000",
        feature_clusters=None,
    )
    omitted = load_selection_shard(overlay_shard_path(omitted_dir, "shard_00000"))
    explicit = load_selection_shard(overlay_shard_path(explicit_dir, "shard_00000"))

    for old_tensor, new_tensor in zip(
        _selection_tensors(omitted),
        _selection_tensors(explicit),
        strict=True,
    ):
        assert old_tensor.dtype == new_tensor.dtype
        assert t.equal(old_tensor, new_tensor)


@pytest.mark.parametrize("selection_mode", ["threshold", "sampling"])
@pytest.mark.parametrize("mask_mode", ["blacklist", "whitelist"])
def test_active_clusters_that_become_identity_after_mask_are_bit_exact_to_unclustered(
    tmp_path: Path,
    selection_mode: str,
    mask_mode: str,
) -> None:
    """An active descriptor may be numerically inert without being identity-null.

    Schema-v1 clusters cannot literally contain singleton groups.  The supported
    identity-equivalent case is therefore an active file whose groups each have
    at most one survivor after the already-resolved manual mask.  It must retain
    distinct artifact identity, but preselection bytes *and the continuation of
    both RNG streams* must exactly match the unclustered implementation.
    """
    _shard, shard_path = _base_shard(tmp_path)
    if mask_mode == "blacklist":
        mask = FeatureMask(FeatureMaskMode.blacklist, (1, 7))
    else:
        mask = FeatureMask(
            FeatureMaskMode.whitelist,
            (3, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30),
        )
    clusters = _clusters(
        tmp_path,
        mask=mask,
        groups=[[1, 3], [7, 8]],
        name=f"identity_after_{mask_mode}.json",
    )
    assert not bool(clusters.consolidated_away.any())
    assert clusters.is_identity_after_mask is True
    assert t.equal(clusters.rep_of, t.arange(N_SAE, dtype=t.int64))
    assert clusters.effective_negative_mask() == mask

    if selection_mode == "threshold":
        unclustered_pipeline = _threshold_pipeline(mask)
        clustered_pipeline = _threshold_pipeline(mask)
    else:
        unclustered_pipeline = _sampling_pipeline(
            mask=mask,
            negative=True,
            seed=714,
        )
        clustered_pipeline = _sampling_pipeline(
            mask=mask,
            negative=True,
            seed=714,
        )
        # This test targets the ordinary packed-row producer.  Splice ordering
        # has its own differential coverage below.
        unclustered_pipeline.delta_bundle = None
        clustered_pipeline.delta_bundle = None

    unclustered_dir = tmp_path / f"unclustered_{selection_mode}_{mask_mode}"
    clustered_dir = tmp_path / f"clustered_{selection_mode}_{mask_mode}"
    unclustered_dir.mkdir()
    clustered_dir.mkdir()
    select_features_for_shard(
        shard_path,
        unclustered_pipeline,
        unclustered_dir,
        "shard_00000",
    )
    select_features_for_shard(
        shard_path,
        clustered_pipeline,
        clustered_dir,
        "shard_00000",
        feature_clusters=clusters,
    )

    unclustered_path = overlay_shard_path(unclustered_dir, "shard_00000")
    clustered_path = overlay_shard_path(clustered_dir, "shard_00000")
    assert clustered_path.read_bytes() == unclustered_path.read_bytes()
    unclustered = load_selection_shard(unclustered_path)
    clustered = load_selection_shard(clustered_path)
    for actual, expected in zip(
        _selection_tensors(clustered),
        _selection_tensors(unclustered),
        strict=True,
    ):
        assert actual.dtype == expected.dtype
        assert t.equal(actual, expected)

    if selection_mode == "sampling":
        assert isinstance(unclustered_pipeline.selection_config, SamplingBundle)
        assert isinstance(clustered_pipeline.selection_config, SamplingBundle)
        assert t.equal(
            clustered_pipeline.selection_config.rng.get_state(),
            unclustered_pipeline.selection_config.rng.get_state(),
        )
        assert unclustered_pipeline.negative_bundle is not None
        assert clustered_pipeline.negative_bundle is not None
        assert t.equal(
            clustered_pipeline.negative_bundle.rng.get_state(),
            unclustered_pipeline.negative_bundle.rng.get_state(),
        )


def test_identity_after_mask_rejects_unsorted_packed_row_before_output(
    tmp_path: Path,
) -> None:
    shard, _sorted_path = _base_shard(tmp_path)
    shard.feature_ids[:4] = t.tensor([5, 1, 3, 10], dtype=t.uint16)
    unsorted_path = save_packed_row_shard(
        shard,
        tmp_path / "base" / "shard_unsorted",
    )
    mask = FeatureMask(FeatureMaskMode.blacklist, (1, 7))
    clusters = _clusters(
        tmp_path,
        mask=mask,
        groups=[[1, 3], [7, 8]],
        name="identity_unsorted.json",
    )
    assert clusters.is_identity_after_mask is True
    output_dir = tmp_path / "unsorted_selection"
    output_dir.mkdir()
    output_path = overlay_shard_path(output_dir, "shard_unsorted")

    with pytest.raises(
        FeatureClustersError,
        match=r"shard_unsorted.*segment 0",
    ):
        select_features_for_shard(
            unsorted_path,
            _threshold_pipeline(mask),
            output_dir,
            "shard_unsorted",
            feature_clusters=clusters,
        )

    assert not output_path.exists()


@pytest.mark.parametrize(
    ("survivor_mags", "expected_buckets", "expected_pcts"),
    [
        pytest.param((33.0, 66.0), [2, 1], [170, 85], id="exact-boundaries"),
        pytest.param((40.0, 80.0), [2, 1], [206, 103], id="interpolated"),
    ],
)
@pytest.mark.parametrize("stored_percentiles", [False, True])
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("counterfactual", [False, True])
@pytest.mark.parametrize("mask_mode", ["blacklist", "whitelist"])
def test_active_clusters_that_become_identity_after_mask_are_bit_exact_for_splice(
    tmp_path: Path,
    survivor_mags: tuple[float, float],
    expected_buckets: list[int],
    expected_pcts: list[int],
    stored_percentiles: bool,
    negative: bool,
    counterfactual: bool,
    mask_mode: str,
) -> None:
    """An inert cluster descriptor must take the complete legacy splice path.

    This uses the real counterfactual-delta implementation and production
    percentile encode/decode contract. Exact-boundary inputs catch bucket
    drift caused by quantization plus ``floor(p * B)``; interpolated inputs
    catch serialized-percentile drift even when the bucket happens to agree.
    Positive and negative sampling continuation states are part of the exact
    contract, not merely the selected ID sets.
    """
    if mask_mode == "blacklist":
        mask = FeatureMask(FeatureMaskMode.blacklist, (1, 7))
    else:
        mask = FeatureMask(
            FeatureMaskMode.whitelist,
            (3, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 26, 28, 30),
        )
    clusters = _clusters(
        tmp_path,
        mask=mask,
        groups=[[1, 3], [7, 8]],
        name=f"splice_identity_after_{mask_mode}.json",
    )
    assert not bool(clusters.consolidated_away.any())
    assert t.equal(clusters.rep_of, t.arange(N_SAE, dtype=t.int64))
    assert clusters.effective_negative_mask() is mask

    raw_fids = t.tensor([1, 3, 7, 8], dtype=t.int64)
    raw_mags = t.tensor(
        [20.0, survivor_mags[0], 50.0, survivor_mags[1]],
        dtype=t.float32,
    )
    ref_fids = t.tensor([1, 3, 7, 8, 20, 25], dtype=t.int64)
    ref_mags = t.tensor([5.0, 5.0, 5.0, 5.0, 90.0, 80.0], dtype=t.float32)
    table = _percentile_table()

    def stored_pcts(fids: t.Tensor, mags: t.Tensor) -> t.Tensor | None:
        if not stored_percentiles:
            return None
        ranks = magnitude_to_percentile(fids, mags, table)
        return uint8_to_percentile(percentile_to_uint8(ranks))

    raw_pcts = stored_pcts(raw_fids, raw_mags)
    ref_pcts = stored_pcts(ref_fids, ref_mags)
    if counterfactual:
        delta_result = compute_doc_counterfactual_deltas(
            [raw_fids],
            [raw_mags],
            [raw_pcts],
            [(ref_fids, ref_mags, ref_pcts)],
            N_SAE,
            table,
        )[0]
        assert delta_result is not None
        assert delta_result[0].tolist() == raw_fids.tolist()
        if raw_pcts is None:
            assert delta_result[-1] is None
        else:
            assert delta_result[-1] is not None
            assert t.equal(delta_result[-1], raw_pcts)
    else:
        delta_result = None

    unclustered_pipeline = _identity_splice_pipeline(
        mask,
        negative=negative,
        seed=714,
    )
    clustered_pipeline = _identity_splice_pipeline(
        mask,
        negative=negative,
        seed=714,
    )
    unclustered = _materialize_splice_segment_selection(
        raw_fids=raw_fids,
        raw_mags=raw_mags,
        raw_pcts=raw_pcts,
        delta_result=delta_result,
        pipeline_config=unclustered_pipeline,
    )
    clustered = _materialize_splice_segment_selection(
        raw_fids=raw_fids,
        raw_mags=raw_mags,
        raw_pcts=raw_pcts,
        delta_result=delta_result,
        pipeline_config=clustered_pipeline,
        feature_clusters=clusters,
    )

    assert unclustered[0] == [8, 3]
    assert unclustered[1] == expected_buckets
    assert unclustered[2] == expected_pcts
    if negative and counterfactual:
        assert unclustered[3] == [20, 25]
    elif negative:
        assert len(unclustered[3]) == 2
        assert set(unclustered[3]).isdisjoint(raw_fids.tolist())
        allowed = mask.allowed_universe_mask(N_SAE)
        assert all(bool(allowed[feature_id]) for feature_id in unclustered[3])
    else:
        assert unclustered[3] == []
    assert isinstance(unclustered_pipeline.selection_config, SamplingBundle)
    assert isinstance(clustered_pipeline.selection_config, SamplingBundle)
    assert t.equal(
        clustered_pipeline.selection_config.rng.get_state(),
        unclustered_pipeline.selection_config.rng.get_state(),
    )
    if negative:
        assert unclustered_pipeline.negative_bundle is not None
        assert clustered_pipeline.negative_bundle is not None
        assert t.equal(
            clustered_pipeline.negative_bundle.rng.get_state(),
            unclustered_pipeline.negative_bundle.rng.get_state(),
        )
    assert clustered == unclustered


def test_identity_after_mask_does_not_bypass_splice_sampling_contract(
    tmp_path: Path,
) -> None:
    mask = FeatureMask(FeatureMaskMode.blacklist, (1, 7))
    clusters = _clusters(
        tmp_path,
        mask=mask,
        groups=[[1, 3], [7, 8]],
        name="unsupported_identity_after_mask.json",
    )
    assert clusters.is_identity_after_mask is True
    pipeline = _threshold_pipeline(mask)
    with pytest.raises(
        FeatureClustersError,
        match="clustered splice materialization requires SamplingBundle",
    ):
        _materialize_splice_segment_selection(
            raw_fids=t.tensor([3, 8], dtype=t.int64),
            raw_mags=t.tensor([1.0, 2.0], dtype=t.float32),
            raw_pcts=t.tensor([0.4, 0.8], dtype=t.float32),
            delta_result=None,
            pipeline_config=pipeline,
            feature_clusters=clusters,
        )


@pytest.mark.parametrize("selection_mode", ["threshold", "sampling"])
@pytest.mark.parametrize("mask_mode", ["none", "blacklist", "whitelist"])
def test_serialized_selection_shard_is_bit_exact_to_removed_reference(
    tmp_path: Path,
    selection_mode: str,
    mask_mode: str,
):
    """Differential end-to-end gate for the valid historical shared subset."""
    _shard, shard_path = _base_shard(tmp_path)
    if mask_mode == "none":
        mask = None
    elif mask_mode == "blacklist":
        mask = FeatureMask(FeatureMaskMode.blacklist, (1, 9))
    else:
        mask = FeatureMask(
            FeatureMaskMode.whitelist,
            (1, 3, 5, 7, 8, 9, 10, 12, 14),
        )
    current = _clusters(tmp_path, mask=mask)
    historical = HistoricalPreselectionAdapter(
        HistoricalFeatureConsolidation.build(
            current.resolved.definition.clusters,
            mask,
            N_SAE,
        ),
        mask,
    )

    if selection_mode == "threshold":
        current_pipeline = _threshold_pipeline(mask)
        historical_pipeline = _threshold_pipeline(mask)
    else:
        current_pipeline = _sampling_pipeline(mask=mask, negative=True, seed=121)
        historical_pipeline = _sampling_pipeline(mask=mask, negative=True, seed=121)
        current_pipeline.delta_bundle = None
        historical_pipeline.delta_bundle = None

    current_dir = tmp_path / "current_selection"
    historical_dir = tmp_path / "historical_selection"
    current_dir.mkdir()
    historical_dir.mkdir()
    select_features_for_shard(
        shard_path,
        current_pipeline,
        current_dir,
        "shard_00000",
        feature_clusters=current,
    )
    select_features_for_shard(
        shard_path,
        historical_pipeline,
        historical_dir,
        "shard_00000",
        feature_clusters=cast(Any, historical),
    )

    current_path = overlay_shard_path(current_dir, "shard_00000")
    historical_path = overlay_shard_path(historical_dir, "shard_00000")
    assert current_path.read_bytes() == historical_path.read_bytes()
    current_selection = load_selection_shard(current_path)
    historical_selection = load_selection_shard(historical_path)
    for actual, expected in zip(
        _selection_tensors(current_selection),
        _selection_tensors(historical_selection),
        strict=True,
    ):
        assert actual.dtype == expected.dtype
        assert t.equal(actual, expected)


@pytest.mark.parametrize(
    "threshold_type",
    [
        ThresholdType.segment_topk,
        ThresholdType.segment_percentile,
    ],
)
def test_support_gate_rejects_raw_magnitude_or_segment_thresholds(
    tmp_path: Path,
    threshold_type: ThresholdType,
):
    clusters = _clusters(tmp_path)
    pipeline = _threshold_pipeline(None, threshold_type=threshold_type, threshold=1.0)
    with pytest.raises(FeatureClustersError, match="raw-magnitude threshold"):
        assert_feature_clusters_precompute_supported(pipeline, None, clusters)


@pytest.mark.parametrize(
    "threshold_type",
    [
        ThresholdType.feature_percentile,
        ThresholdType.feature_percentile_segment_topk,
    ],
)
def test_support_gate_accepts_reviewed_ordinary_thresholds(
    tmp_path: Path,
    threshold_type: ThresholdType,
):
    clusters = _clusters(tmp_path)
    assert_feature_clusters_precompute_supported(
        _threshold_pipeline(None, threshold_type=threshold_type),
        None,
        clusters,
    )


def test_support_gate_accepts_sampling_for_ordinary_and_splice(tmp_path: Path):
    clusters = _clusters(tmp_path)
    splice_pipeline = _sampling_pipeline()
    assert_feature_clusters_precompute_supported(splice_pipeline, None, clusters)
    splice_pipeline.delta_bundle = None
    assert_feature_clusters_precompute_supported(splice_pipeline, None, clusters)


def test_producer_completion_scan_rejects_consolidated_away_positive(tmp_path: Path):
    shard, shard_path = _base_shard(tmp_path)
    clusters = _clusters(tmp_path)
    out_dir = tmp_path / "selection"
    out_dir.mkdir()
    corrupt = SelectionShard(
        sel_offsets=t.tensor([0, 1, 1], dtype=t.int32),
        sel_fids=t.tensor([5], dtype=t.uint16),
        sel_buckets=t.tensor([1], dtype=t.uint8),
        sel_pcts=t.tensor([100], dtype=t.uint8),
        neg_offsets=t.tensor([0, 0, 0], dtype=t.int32),
        neg_fids=t.tensor([], dtype=t.uint16),
    )
    selection_path = save_selection_shard(corrupt, out_dir / "shard_00000")
    packed_proof = resolve_packed_row_fingerprint_strict(shard_path.parent)
    assert shard.total_segments == 2
    with pytest.raises(FeatureClustersError, match="consolidated-away feature ID 5"):
        _validate_clustered_selection_outputs(
            output_dir=out_dir,
            base_shard_paths=[shard_path],
            packed_resolution=packed_proof,
            expected_selection_integrity=(
                build_clustered_selection_shard_integrity([selection_path])
            ),
            expected_packed_rows_hash=packed_proof.aggregate_sha256,
            feature_clusters=clusters,
            effective_buckets=4,
        )


def test_producer_completion_scan_requires_exact_stem_coverage(tmp_path: Path):
    _shard, shard_path = _base_shard(tmp_path)
    clusters = _clusters(tmp_path)
    out_dir = tmp_path / "selection"
    out_dir.mkdir()
    packed_proof = resolve_packed_row_fingerprint_strict(shard_path.parent)
    with pytest.raises(FeatureClustersError, match="missing=.*shard_00000"):
        _validate_clustered_selection_outputs(
            output_dir=out_dir,
            base_shard_paths=[shard_path],
            packed_resolution=packed_proof,
            expected_selection_integrity={},
            expected_packed_rows_hash=packed_proof.aggregate_sha256,
            feature_clusters=clusters,
            effective_buckets=4,
        )


def test_producer_completion_scan_returns_exact_positive_plus_negative_width(
    tmp_path: Path,
):
    _shard, shard_path = _base_shard(tmp_path)
    clusters = _clusters(tmp_path)
    out_dir = tmp_path / "selection"
    out_dir.mkdir()
    selection = SelectionShard(
        sel_offsets=t.tensor([0, 2, 3], dtype=t.int32),
        sel_fids=t.tensor([1, 2, 7], dtype=t.uint16),
        sel_buckets=t.tensor([1, 2, 3], dtype=t.uint8),
        sel_pcts=t.tensor([10, 20, 30], dtype=t.uint8),
        neg_offsets=t.tensor([0, 1, 3], dtype=t.int32),
        neg_fids=t.tensor([7, 1, 2], dtype=t.uint16),
    )
    selection_path = save_selection_shard(selection, out_dir / "shard_00000")
    packed_proof = resolve_packed_row_fingerprint_strict(shard_path.parent)

    assert (
        _validate_clustered_selection_outputs(
            output_dir=out_dir,
            base_shard_paths=[shard_path],
            packed_resolution=packed_proof,
            expected_selection_integrity=(
                build_clustered_selection_shard_integrity([selection_path])
            ),
            expected_packed_rows_hash=packed_proof.aggregate_sha256,
            feature_clusters=clusters,
            effective_buckets=4,
        )
        == 3
    )


def test_cluster_module_has_no_delta_or_splice_import_dependency():
    module_path = (
        Path(__file__).parents[2]
        / "steering"
        / "conditional"
        / "feature_clusters.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)
    assert not any("delta" in module for module in imported_modules)
    assert not any("splice" in module for module in imported_modules)
