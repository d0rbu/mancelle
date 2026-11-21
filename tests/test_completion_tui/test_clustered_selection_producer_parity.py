from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch as t
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import scripts.completion_tui.data_loader as data_loader_module
from scripts.completion_tui.data_loader import (
    TUIFeatureSamplingConfig,
    _clustered_threshold_select,
    clustered_sample_segment_features,
)
from scripts.completion_tui.state import display_to_internal
from scripts.preprocess.precompute_selected_features import (
    select_features_for_shard,
)
from steering.conditional.discretization import percentile_to_bucket
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    resolve_feature_clusters,
)
from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_pipeline import (
    SamplingBundle,
    build_feature_pipeline,
)
from steering.conditional.percentile_table import normalized_percentile_to_uint8
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingMode,
    SamplingSelection,
    ThresholdSelection,
)
from steering.conditional.thresholding import ThresholdType, select_from_packed_percentiles
from steering.overlay_io import overlay_shard_path
from steering.packed_batch_io import (
    PackedRowShard,
    load_selection_shard,
    save_packed_row_shard,
)
from steering.stats_types import PerFeatureStat

N_SAE = 64
SEED = 1234
CLUSTERS = [[3, 5, 9], [10, 11], [20, 21, 22, 23], [40, 41]]
SEGMENT_IDS = [
    [3, 5, 9, 1],
    [10, 11, 15, 21],
    [],
    [40, 41],
    [5],
    [20, 21, 22, 23, 30],
]
SEGMENT_PCTS = [
    [20, 100, 240, 230],
    [250, 120, 180, 200],
    [],
    [245, 40],
    [225],
    [10, 230, 150, 220, 255],
]


def _stats() -> dict[str, PerFeatureStat]:
    percentiles = {f"p{k}": k / 100 for k in range(1, 100)}
    return {
        str(feature_id): PerFeatureStat(
            count=100,
            pct_segments_active=0.5,
            mean_mag=0.5,
            mean_mag_incl_zeros=0.25,
            std_mag=0.1,
            min_mag=0.0,
            max_mag=1.0,
            median_mag=0.5,
            percentiles=dict(percentiles),
        )
        for feature_id in range(N_SAE)
    }


def _consolidation(
    tmp_path: Path,
    mask: FeatureMask | None,
) -> FeatureClusterConsolidation:
    path = tmp_path / "clusters.json"
    path.write_text(
        json.dumps({"schema_version": 1, "clusters": CLUSTERS}),
        encoding="utf-8",
    )
    resolved = resolve_feature_clusters({"feature_clusters": {"path": str(path)}}, N_SAE)
    assert resolved is not None
    return FeatureClusterConsolidation.build(resolved, mask)


def _base_shard(tmp_path: Path) -> Path:
    flat_ids = [feature_id for segment in SEGMENT_IDS for feature_id in segment]
    flat_pcts = [percentile for segment in SEGMENT_PCTS for percentile in segment]
    offsets = [0]
    for segment in SEGMENT_IDS:
        offsets.append(offsets[-1] + len(segment))
    n_segments = len(SEGMENT_IDS)
    context_len = n_segments * 2
    shard = PackedRowShard(
        tokens=t.arange(context_len + 1, dtype=t.int64).add(1).unsqueeze(0).to(t.uint16),
        row_seg_offsets=t.tensor([0, n_segments], dtype=t.int32),
        seg_token_start=t.arange(0, context_len, 2, dtype=t.int64).to(t.uint16),
        seg_token_end=t.arange(2, context_len + 1, 2, dtype=t.int64).to(t.uint16),
        seg_doc_id=t.zeros(n_segments, dtype=t.uint8),
        seg_feat_offsets=t.tensor(offsets, dtype=t.int32),
        feature_ids=t.tensor(flat_ids, dtype=t.uint16),
        feature_pcts=t.tensor(flat_pcts, dtype=t.uint8),
    )
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    return save_packed_row_shard(shard, base_dir / "shard_00000")


def _mask(kind: str) -> FeatureMask | None:
    if kind == "none":
        return None
    if kind == "blacklist":
        return FeatureMask(FeatureMaskMode.blacklist, (5, 21, 40, 41))
    assert kind == "whitelist"
    return FeatureMask(FeatureMaskMode.whitelist, tuple(range(32)))


def _selection(kind: str) -> SamplingSelection | ThresholdSelection:
    if kind == "weighted":
        return SamplingSelection(
            distribution=Distribution.geometric,
            mean=2.0,
            min_k=1,
            max_k=4,
            mode=SamplingMode.weighted,
        )
    if kind == "top_k_percentile":
        return SamplingSelection(
            distribution=Distribution.geometric,
            mean=2.0,
            min_k=1,
            max_k=4,
            mode=SamplingMode.top_k_percentile,
        )
    if kind == "feature_percentile":
        return ThresholdSelection(ThresholdType.feature_percentile, 0.90)
    assert kind == "feature_percentile_segment_topk"
    return ThresholdSelection(ThresholdType.feature_percentile_segment_topk, 2.90)


def _negative(enabled: bool) -> NegativeSamplingSelection | None:
    if not enabled:
        return None
    return NegativeSamplingSelection(
        distribution=Distribution.geometric,
        mean=3.0,
        min_k=0,
        max_k=5,
    )


@pytest.mark.parametrize("mask_kind", ["none", "whitelist", "blacklist"])
@pytest.mark.parametrize(
    "selection_kind",
    [
        "weighted",
        "top_k_percentile",
        "feature_percentile",
        "feature_percentile_segment_topk",
    ],
)
@pytest.mark.parametrize("with_negatives", [False, True])
def test_tui_clustered_selection_matches_real_producer_exactly(
    tmp_path: Path,
    mask_kind: str,
    selection_kind: str,
    with_negatives: bool,
):
    mask = _mask(mask_kind)
    selection = _selection(selection_kind)
    negative = _negative(with_negatives)
    consolidation = _consolidation(tmp_path, mask)
    base_path = _base_shard(tmp_path)
    producer_pipeline = build_feature_pipeline(
        feature_selection=selection,
        negative_selection=negative,
        per_feature_stats=_stats(),
        n_sae_features=N_SAE,
        num_buckets=4,
        seed=SEED,
        ddp_rank=0,
        feature_mask=mask,
    )
    output_dir = tmp_path / "selection"
    output_dir.mkdir()
    select_features_for_shard(
        base_path,
        producer_pipeline,
        output_dir,
        "shard_00000",
        feature_clusters=consolidation,
    )
    producer = load_selection_shard(overlay_shard_path(output_dir, "shard_00000"))

    percentile_table = (
        producer_pipeline.selection_config.percentile_table
        if isinstance(producer_pipeline.selection_config, SamplingBundle)
        else None
    )
    tui_cfg = TUIFeatureSamplingConfig(
        boundaries=producer_pipeline.boundaries,
        n_sae_features=N_SAE,
        feature_selection=selection,
        negative_selection=negative,
        percentile_table=percentile_table,
        feature_mask=mask,
        consolidation=consolidation,
        effective_buckets=producer_pipeline.effective_buckets,
    )
    tui_rng = t.Generator().manual_seed(SEED)

    for segment_index, (ids, pct_bytes) in enumerate(zip(SEGMENT_IDS, SEGMENT_PCTS, strict=True)):
        sf = SegmentFeatures(
            segment_idx=segment_index,
            position=segment_index * 2,
            feature_ids=t.tensor(ids, dtype=t.int32),
            feature_mags=t.zeros(len(ids), dtype=t.float32),
            feature_percentiles=t.tensor(pct_bytes, dtype=t.uint8),
            char_start=segment_index * 10,
            char_end=(segment_index + 1) * 10,
        )
        result = clustered_sample_segment_features(
            sf,
            tui_cfg,
            4,
            with_negatives,
            rng=tui_rng,
        )

        masked_ids = sf.feature_ids.to(t.int64)
        assert sf.feature_percentiles is not None
        masked_pcts = sf.feature_percentiles.float() / 255.0
        if mask is not None and masked_ids.numel() > 0:
            masked_ids, masked_pcts = mask.apply_to_positive(masked_ids, masked_pcts)
        if masked_ids.numel() > 0:
            reps, rep_pcts = consolidation.consolidate_segment(masked_ids, masked_pcts)
        else:
            reps = t.tensor([], dtype=t.int64)
            rep_pcts = t.tensor([], dtype=t.float32)

        positive_rows = result.features[: int(reps.numel())]
        negative_rows = result.features[int(reps.numel()) :]
        tui_percentile_bytes = normalized_percentile_to_uint8(rep_pcts)
        tui_buckets = percentile_to_bucket(rep_pcts, producer_pipeline.effective_buckets)
        pct_by_rep = dict(zip(reps.tolist(), tui_percentile_bytes.tolist(), strict=True))
        bucket_by_rep = dict(zip(reps.tolist(), tui_buckets.tolist(), strict=True))
        tui_positive_triples_list = sorted(
            (
                row.feature_id,
                int(bucket_by_rep[row.feature_id]),
                int(pct_by_rep[row.feature_id]),
            )
            for row in positive_rows
            if row.enabled
        )

        sel_start = int(producer.sel_offsets[segment_index])
        sel_end = int(producer.sel_offsets[segment_index + 1])
        producer_positive_triples_list = sorted(
            (int(feature_id), int(bucket), int(percentile))
            for feature_id, bucket, percentile in zip(
                producer.sel_fids[sel_start:sel_end].tolist(),
                producer.sel_buckets[sel_start:sel_end].tolist(),
                producer.sel_pcts[sel_start:sel_end].tolist(),
                strict=True,
            )
        )
        # Duplicate representatives are unreachable: consolidation emits
        # unique reps and selection shards enforce unique IDs per row. Keep a
        # multiset comparison anyway so a future duplicate cannot hide in a set.
        assert tui_positive_triples_list == producer_positive_triples_list
        assert set(tui_positive_triples_list) == set(producer_positive_triples_list)
        assert {row.feature_id for row in positive_rows if row.enabled} == set(
            producer.sel_fids[sel_start:sel_end].tolist()
        )
        for row in positive_rows:
            assert display_to_internal(row.magnitude, 4, with_negatives) == int(
                bucket_by_rep[row.feature_id]
            )

        neg_start = int(producer.neg_offsets[segment_index])
        neg_end = int(producer.neg_offsets[segment_index + 1])
        assert [row.feature_id for row in negative_rows] == producer.neg_fids[
            neg_start:neg_end
        ].tolist()

    if isinstance(producer_pipeline.selection_config, SamplingBundle):
        assert t.equal(tui_rng.get_state(), producer_pipeline.selection_config.rng.get_state())
    elif producer_pipeline.negative_bundle is not None:
        assert t.equal(tui_rng.get_state(), producer_pipeline.negative_bundle.rng.get_state())


def test_clustered_threshold_select_delegates_to_shared_selector(monkeypatch):
    """The TUI threshold arm calls the ONE shared packed-percentile selector.

    Since the #767 refactor, `steering.conditional.thresholding.
    select_from_packed_percentiles` is the single implementation consumed by
    both the training preselection producer and this live path — parity is by
    construction, so this test pins the delegation (exact arguments, returned
    object identity) instead of comparing two implementations.
    """
    calls: dict[str, tuple] = {}
    sentinel = t.tensor([7], dtype=t.int64)

    def _spy(feature_ids, percentiles, threshold_type, threshold):
        calls["args"] = (feature_ids, percentiles, threshold_type, threshold)
        return sentinel

    monkeypatch.setattr(data_loader_module, "select_from_packed_percentiles", _spy)
    fids = t.tensor([7, 9], dtype=t.int64)
    pcts = t.tensor([0.9, 0.1], dtype=t.float32)
    selection = ThresholdSelection(ThresholdType.feature_percentile, 0.5)
    out = _clustered_threshold_select(fids, pcts, selection)
    assert out is sentinel
    spied_fids, spied_pcts, spied_type, spied_threshold = calls["args"]
    assert spied_fids is fids
    assert spied_pcts is pcts
    assert spied_type is ThresholdType.feature_percentile
    assert spied_threshold == 0.5


@settings(max_examples=150, deadline=None)
@given(
    pct_bytes=st.lists(st.integers(min_value=0, max_value=255), min_size=0, max_size=20),
    threshold_type=st.sampled_from(
        [
            ThresholdType.feature_percentile,
            ThresholdType.feature_percentile_segment_topk,
        ]
    ),
    threshold_pct=st.sampled_from([0.0, 0.25, 0.5, 0.9, 1.0]),
    k=st.integers(min_value=0, max_value=25),
)
def test_clustered_threshold_select_equals_shared_selector_unmocked(
    pct_bytes: list[int],
    threshold_type: ThresholdType,
    threshold_pct: float,
    k: int,
):
    if threshold_type is ThresholdType.feature_percentile_segment_topk:
        assume(k > 0)
        assume(0.0 < threshold_pct < 1.0)
    fids = t.arange(len(pct_bytes), dtype=t.int64)
    pcts = t.tensor(pct_bytes, dtype=t.float32) / 255.0
    threshold = (
        threshold_pct if threshold_type is ThresholdType.feature_percentile else k + threshold_pct
    )
    selection = ThresholdSelection(threshold_type, threshold)
    expected = select_from_packed_percentiles(fids, pcts, threshold_type, threshold)
    actual = _clustered_threshold_select(fids, pcts, selection)
    assert t.equal(actual, expected)


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_mask_consolidate_bucket_matches_plain_python_oracle(
    data: st.DataObject,
    tmp_path: Path,
):
    width = data.draw(st.integers(min_value=4, max_value=32), label="width")
    cluster = sorted(
        data.draw(
            st.sets(st.integers(min_value=0, max_value=width - 1), min_size=2),
            label="cluster",
        )
    )
    active = sorted(
        data.draw(
            st.sets(st.integers(min_value=0, max_value=width - 1)),
            label="active",
        )
    )
    pct_bytes = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=255),
            min_size=len(active),
            max_size=len(active),
        ),
        label="percentiles",
    )
    mode = data.draw(st.sampled_from(["none", "blacklist", "whitelist"]), label="mode")
    if mode == "none":
        mask = None
        allowed = set(range(width))
    else:
        mask_ids = data.draw(
            st.sets(st.integers(min_value=0, max_value=width - 1), min_size=1),
            label="mask_ids",
        )
        mask = FeatureMask(FeatureMaskMode(mode), tuple(mask_ids))
        allowed = set(range(width)) - mask_ids if mode == "blacklist" else mask_ids

    path = tmp_path / "oracle_clusters.json"
    path.write_text(
        json.dumps({"schema_version": 1, "clusters": [cluster]}),
        encoding="utf-8",
    )
    resolved = resolve_feature_clusters({"feature_clusters": {"path": str(path)}}, width)
    assert resolved is not None
    consolidation = FeatureClusterConsolidation.build(resolved, mask)
    fids = t.tensor(active, dtype=t.int64)
    pcts = t.tensor(pct_bytes, dtype=t.float32) / 255.0
    if mask is not None and fids.numel() > 0:
        fids, pcts = mask.apply_to_positive(fids, pcts)
    actual_reps, actual_pcts = consolidation.consolidate_segment(fids, pcts)
    actual_buckets = percentile_to_bucket(actual_pcts, 4)

    allowed_cluster = [feature_id for feature_id in cluster if feature_id in allowed]
    representative = min(allowed_cluster) if allowed_cluster else None
    grouped: dict[int, list[float]] = {}
    for feature_id, percentile in zip(active, pct_bytes, strict=True):
        if feature_id not in allowed:
            continue
        rep = representative if representative is not None and feature_id in cluster else feature_id
        grouped.setdefault(rep, []).append(percentile / 255.0)
    expected_reps = sorted(grouped)
    expected_pcts = [max(grouped[rep]) for rep in expected_reps]
    expected_buckets = [min(int(percentile * 4), 3) for percentile in expected_pcts]

    assert actual_reps.tolist() == expected_reps
    assert actual_pcts.tolist() == pytest.approx(expected_pcts)
    assert actual_buckets.tolist() == expected_buckets
