"""Unit and differential gates for splice multi-segment selection.

These tests deliberately separate three contracts:

1. legacy mode stays byte/RNG identical;
2. the tensor-native CSR builder is an exact representation transform;
3. batching may change random draws, but never eligibility, cardinality bounds,
   ranking semantics, dtypes, CSR order, or determinism for a fixed mode.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t

import steering.conditional.sampling as sampling
from errors import FeatureClustersError
from scripts.preprocess.precompute_selected_features import (
    _materialize_ordinary_splice_selection_batch,
    _materialize_splice_segment_selection,
    _OrdinarySpliceSelectionInput,
    _require_cpu_vector,
    _selected_pcts_uint8,
    _tensor_selection_batch,
    _TensorSelectionBatch,
    _TensorSelectionShardBuilder,
)
from steering.conditional.delta import DeltaMode, FeatureDeltaConfig
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_pipeline import (
    DeltaBundle,
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
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
from steering.conditional.splice_provenance import SpliceProvenanceError
from steering.conditional.splice_selection_batching import (
    LEGACY_SPLICE_SELECTION_BATCH_SIZE,
    SPLICE_SELECTION_BATCHING_ALGORITHM,
    SPLICE_SELECTION_BATCHING_SCHEMA_VERSION,
    canonical_splice_selection_batching,
    resolve_splice_selection_batch_size,
)
from steering.conditional.thresholding import ThresholdType
from steering.eval.cache_identity import resolve_feature_selection_hash
from steering.eval.feature_selection_hash import (
    compute_canonical_feature_selection_hash,
)
from steering.packed_batch_io import SelectionShard, compute_selection_hash

N_SAE = 128

_TensorSelectionRow = tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]
_ListSelectionRow = tuple[list[int], list[int], list[int], list[int]]


def _selection_config(
    *,
    mode: SamplingMode = SamplingMode.top_k_percentile,
    min_k: int = 0,
    max_k: int = 9,
) -> SamplingSelection:
    return SamplingSelection(
        distribution=Distribution.geometric,
        mean=4.0,
        std=4.0,
        min_k=min_k,
        max_k=max_k,
        mode=mode,
    )


def _pipeline(
    *,
    seed: int,
    mode: SamplingMode = SamplingMode.top_k_percentile,
    negative: bool = True,
    mask: FeatureMask | None = None,
    min_k: int = 0,
    max_k: int = 9,
) -> FeaturePipelineConfig:
    table = t.arange(1, 100, dtype=t.float64).repeat(N_SAE, 1)
    negative_bundle = None
    if negative:
        negative_bundle = NegativeBundle(
            selection=NegativeSamplingSelection(
                distribution=Distribution.geometric,
                mean=3.0,
                std=3.0,
                min_k=0,
                max_k=8,
            ),
            n_sae_features=N_SAE,
            rng=t.Generator().manual_seed(seed + 1),
            feature_mask=mask,
        )
    return FeaturePipelineConfig(
        selection_config=SamplingBundle(
            selection=_selection_config(
                mode=mode,
                min_k=min_k,
                max_k=max_k,
            ),
            percentile_table=table,
            rng=t.Generator().manual_seed(seed),
        ),
        boundaries=t.tensor([0.2, 0.5, 0.8], dtype=t.float64).repeat(N_SAE, 1),
        negative_bundle=negative_bundle,
        effective_buckets=4,
        delta_bundle=DeltaBundle(
            config=FeatureDeltaConfig(
                mode=DeltaMode.splice_counterfactual,
                neg_delta_ratio=1.0,
            ),
            percentile_table=table,
        ),
        feature_mask=mask,
    )


def _threshold_pipeline(
    *,
    seed: int,
    negative: bool,
    mask: FeatureMask | None = None,
) -> FeaturePipelineConfig:
    table = t.arange(1, 100, dtype=t.float64).repeat(N_SAE, 1)
    negative_bundle = None
    if negative:
        negative_bundle = NegativeBundle(
            selection=NegativeSamplingSelection(
                distribution=Distribution.geometric,
                mean=3.0,
                std=3.0,
                min_k=0,
                max_k=8,
            ),
            n_sae_features=N_SAE,
            rng=t.Generator().manual_seed(seed),
            feature_mask=mask,
        )
    return FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.segment_topk, 3),
            precomputed_thresholds=None,
        ),
        boundaries=t.tensor([0.2, 0.5, 0.8], dtype=t.float64).repeat(N_SAE, 1),
        negative_bundle=negative_bundle,
        effective_buckets=4,
        delta_bundle=DeltaBundle(
            config=FeatureDeltaConfig(
                mode=DeltaMode.splice_counterfactual,
                neg_delta_ratio=1.0,
            ),
            percentile_table=table,
        ),
        feature_mask=mask,
    )


def _inputs(
    count: int,
    *,
    with_percentiles: bool,
) -> list[_OrdinarySpliceSelectionInput]:
    rows: list[_OrdinarySpliceSelectionInput] = []
    for row in range(count):
        size = row % 7
        fids = (t.arange(size, dtype=t.int64) * 11 + row * 3) % N_SAE
        fids = t.unique(fids, sorted=True)
        mags = t.arange(1, fids.numel() + 1, dtype=t.float32) / 10.0
        pcts = (
            t.linspace(0.05, 0.95, fids.numel(), dtype=t.float32)
            if with_percentiles and fids.numel() > 0
            else (t.empty(0, dtype=t.float32) if with_percentiles else None)
        )
        rows.append(
            _OrdinarySpliceSelectionInput(
                segment_index=row,
                raw_fids=fids,
                raw_mags=mags,
                raw_pcts=pcts,
            )
        )
    return rows


def _batch_rows(
    batch: _TensorSelectionBatch,
) -> list[_TensorSelectionRow]:
    result: list[_TensorSelectionRow] = []
    sel_offset = 0
    neg_offset = 0
    for sel_count_t, neg_count_t in zip(
        batch.sel_counts,
        batch.neg_counts,
        strict=True,
    ):
        sel_count = int(sel_count_t.item())
        neg_count = int(neg_count_t.item())
        result.append(
            (
                batch.sel_fids[sel_offset : sel_offset + sel_count].to(t.int64),
                batch.sel_buckets[sel_offset : sel_offset + sel_count].to(t.int64),
                batch.sel_pcts[sel_offset : sel_offset + sel_count].to(t.int64),
                batch.neg_fids[neg_offset : neg_offset + neg_count].to(t.int64),
            )
        )
        sel_offset += sel_count
        neg_offset += neg_count
    assert sel_offset == batch.sel_fids.numel()
    assert neg_offset == batch.neg_fids.numel()
    return result


def _require_tensor_row(
    row: _ListSelectionRow | _TensorSelectionRow,
) -> _TensorSelectionRow:
    assert all(isinstance(value, t.Tensor) for value in row)
    return cast(_TensorSelectionRow, row)


def _require_list_row(
    row: _ListSelectionRow | _TensorSelectionRow,
) -> _ListSelectionRow:
    assert all(isinstance(value, list) for value in row)
    return cast(_ListSelectionRow, row)


def _selection_tensors(shard: SelectionShard) -> tuple[t.Tensor, ...]:
    return (
        shard.sel_offsets,
        shard.sel_fids,
        shard.sel_buckets,
        shard.sel_pcts,
        shard.neg_offsets,
        shard.neg_fids,
    )


class TestBatchingIdentity:
    def test_resolver_preserves_legacy_none_and_one(self) -> None:
        assert resolve_splice_selection_batch_size(None) == 1
        assert resolve_splice_selection_batch_size(1) == 1
        assert LEGACY_SPLICE_SELECTION_BATCH_SIZE == 1

    @pytest.mark.parametrize("bad", [True, False, 0, -1, 1.0, "8", [], {}])
    def test_resolver_rejects_ambiguous_or_invalid_values(self, bad: object) -> None:
        with pytest.raises(ValueError, match="selection_splice_batch_size"):
            resolve_splice_selection_batch_size(bad)

    def test_canonical_descriptor_is_exact_and_versioned(self) -> None:
        assert canonical_splice_selection_batching(
            {"mode": "splice_counterfactual", "neg_delta_ratio": 1.0},
            64,
        ) == {
            "schema_version": SPLICE_SELECTION_BATCHING_SCHEMA_VERSION,
            "algorithm": SPLICE_SELECTION_BATCHING_ALGORITHM,
            "batch_size": 64,
        }

    @pytest.mark.parametrize("batch_size", [None, 1])
    def test_legacy_hash_payload_is_exactly_unchanged(self, batch_size: object) -> None:
        args: dict[str, Any] = {
            "positive_selection": {"distribution": "geometric", "mean": 4.0},
            "negative_selection": None,
            "delta_config": {
                "mode": "splice_counterfactual",
                "neg_delta_ratio": 1.0,
            },
            "num_buckets": 2,
            "seed": 42,
            "ddp_rank": 0,
        }
        before = compute_selection_hash(**args)
        after = compute_selection_hash(**args, splice_batch_size=batch_size)
        assert after == before

    def test_enabled_batching_changes_only_splice_identity(self) -> None:
        base: dict[str, Any] = {
            "positive_selection": {"distribution": "geometric", "mean": 4.0},
            "negative_selection": None,
            "num_buckets": 2,
            "seed": 42,
        }
        splice = {"mode": "splice_counterfactual", "neg_delta_ratio": 1.0}
        assert compute_selection_hash(
            **base,
            delta_config=splice,
            splice_batch_size=64,
        ) != compute_selection_hash(**base, delta_config=splice)
        assert compute_selection_hash(
            **base,
            delta_config=None,
            splice_batch_size=64,
        ) == compute_selection_hash(**base, delta_config=None)
        assert compute_selection_hash(
            **base,
            delta_config={"mode": "prev_segment", "neg_delta_ratio": 1.0},
            splice_batch_size=64,
        ) == compute_selection_hash(
            **base,
            delta_config={"mode": "prev_segment", "neg_delta_ratio": 1.0},
        )

    def test_each_batch_size_has_distinct_identity(self) -> None:
        hashes = {
            compute_selection_hash(
                positive_selection={"distribution": "geometric", "mean": 4.0},
                negative_selection=None,
                delta_config={"mode": "splice_counterfactual"},
                num_buckets=2,
                seed=42,
                splice_batch_size=size,
            )
            for size in (2, 8, 64, 256)
        }
        assert len(hashes) == 4

    def test_canonical_eval_wrapper_matches_direct_hash(self) -> None:
        positive = {"distribution": "geometric", "mean": 4.0}
        negative = {"distribution": "normal", "mean": 2.0, "std": 1.0}
        stats_hash = "a" * 64
        delta = {"mode": "splice_counterfactual", "neg_delta_ratio": 0.5}
        wrapped = compute_canonical_feature_selection_hash(
            positive_selection=positive,
            negative_selection=negative,
            num_buckets=2,
            stats_source_sha256=stats_hash,
            feature_delta=delta,
            splice_batch_size=32,
        )
        direct = compute_selection_hash(
            positive_selection=positive,
            negative_selection=negative,
            delta_config={**delta, "stats_source_sha256": stats_hash},
            num_buckets=2,
            seed=0,
            ddp_rank=0,
            splice_batch_size=32,
        )
        assert wrapped == direct

    def test_eval_cache_identity_reads_the_exact_semantic_batching_key(self) -> None:
        conditioning = {
            "feature_selection": {
                "distribution": "geometric",
                "mean": 4.0,
            },
            "negative_selection": None,
            "num_buckets": 2,
            "stats_source_sha256": "b" * 64,
            "feature_delta": {
                "mode": "splice_counterfactual",
                "neg_delta_ratio": 1.0,
            },
            "selection_splice_batch_size": 64,
        }
        actual = resolve_feature_selection_hash({}, conditioning)
        expected = compute_canonical_feature_selection_hash(
            positive_selection=conditioning["feature_selection"],
            negative_selection=None,
            num_buckets=2,
            stats_source_sha256="b" * 64,
            feature_delta=conditioning["feature_delta"],
            splice_batch_size=64,
        )
        legacy = resolve_feature_selection_hash(
            {},
            {
                key: value
                for key, value in conditioning.items()
                if key != "selection_splice_batch_size"
            },
        )
        assert actual == expected
        assert actual != legacy

    def test_descriptor_json_is_canonical_and_stable(self) -> None:
        descriptor = canonical_splice_selection_batching(
            {"neg_delta_ratio": 1.0, "mode": "splice_counterfactual"},
            64,
        )
        encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
        assert encoded == (
            '{"algorithm":"ordinary_execution_runs_v1","batch_size":64,"schema_version":1}'
        )
        assert hashlib.sha256(encoded.encode()).hexdigest() == (
            "8a86145ebcc05375413a8a6e4c672a1b93cd5e4edd1d380d32878994b0d3f688"
        )


class TestTensorSelectionBuilder:
    def test_empty_percentile_materialization_is_canonical_uint8(self) -> None:
        actual = _selected_pcts_uint8(
            selected_fids=t.tensor([], dtype=t.int64),
            selected_mags=t.tensor([], dtype=t.float32),
            boundaries=t.empty((N_SAE, 3)),
        )
        assert actual.dtype == t.uint8
        assert actual.shape == (0,)

    def test_vector_validator_rejects_non_vector_input(self) -> None:
        with pytest.raises(SpliceProvenanceError, match="one-dimensional CPU"):
            _require_cpu_vector("probe", t.zeros((1, 1)))

    @pytest.mark.parametrize("total_segments", [-1, True, 1.0])
    def test_builder_requires_a_strict_non_negative_segment_count(
        self,
        total_segments: object,
    ) -> None:
        with pytest.raises(ValueError, match="non-negative int"):
            _TensorSelectionShardBuilder(cast(int, total_segments))

    def test_zero_segment_shard_builds_canonical_empty_csr(self) -> None:
        shard = _TensorSelectionShardBuilder(0).build()
        assert t.equal(shard.sel_offsets, t.tensor([0], dtype=t.int32))
        assert t.equal(shard.neg_offsets, t.tensor([0], dtype=t.int32))
        assert shard.sel_fids.dtype == t.uint16 and shard.sel_fids.numel() == 0
        assert shard.sel_buckets.dtype == t.uint8 and shard.sel_buckets.numel() == 0
        assert shard.sel_pcts.dtype == t.uint8 and shard.sel_pcts.numel() == 0
        assert shard.neg_fids.dtype == t.uint16 and shard.neg_fids.numel() == 0

    def test_out_of_order_batches_reconstruct_exact_global_csr(self) -> None:
        first = _tensor_selection_batch(
            [3, 0],
            [
                (
                    t.tensor([30, 31]),
                    t.tensor([1, 2]),
                    t.tensor([130, 131]),
                    t.tensor([90]),
                ),
                (
                    t.tensor([1]),
                    t.tensor([3]),
                    t.tensor([101]),
                    t.tensor([], dtype=t.int64),
                ),
            ],
        )
        second = _tensor_selection_batch(
            [2, 1],
            [
                (
                    t.tensor([], dtype=t.int64),
                    t.tensor([], dtype=t.int64),
                    t.tensor([], dtype=t.int64),
                    t.tensor([70, 71]),
                ),
                (
                    t.tensor([10, 11, 12]),
                    t.tensor([0, 1, 2]),
                    t.tensor([110, 111, 112]),
                    t.tensor([80]),
                ),
            ],
        )
        builder = _TensorSelectionShardBuilder(4)
        builder.add(first)
        builder.add(second)
        shard = builder.build()
        assert t.equal(shard.sel_offsets, t.tensor([0, 1, 4, 4, 6], dtype=t.int32))
        assert t.equal(
            shard.sel_fids,
            t.tensor([1, 10, 11, 12, 30, 31], dtype=t.uint16),
        )
        assert t.equal(
            shard.sel_buckets,
            t.tensor([3, 0, 1, 2, 1, 2], dtype=t.uint8),
        )
        assert t.equal(
            shard.sel_pcts,
            t.tensor([101, 110, 111, 112, 130, 131], dtype=t.uint8),
        )
        assert t.equal(shard.neg_offsets, t.tensor([0, 0, 1, 3, 4], dtype=t.int32))
        assert t.equal(shard.neg_fids, t.tensor([80, 70, 71, 90], dtype=t.uint16))

    def test_all_empty_rows_have_complete_zero_csr(self) -> None:
        empty = t.tensor([], dtype=t.int64)
        batch = _tensor_selection_batch(
            [2, 0, 1],
            [(empty, empty, empty, empty) for _ in range(3)],
        )
        builder = _TensorSelectionShardBuilder(3)
        builder.add(batch)
        shard = builder.build()
        assert t.equal(shard.sel_offsets, t.zeros(4, dtype=t.int32))
        assert t.equal(shard.neg_offsets, t.zeros(4, dtype=t.int32))
        assert shard.sel_fids.dtype == t.uint16 and shard.sel_fids.numel() == 0
        assert shard.sel_buckets.dtype == t.uint8 and shard.sel_buckets.numel() == 0
        assert shard.sel_pcts.dtype == t.uint8 and shard.sel_pcts.numel() == 0
        assert shard.neg_fids.dtype == t.uint16 and shard.neg_fids.numel() == 0

    def test_uint16_boundary_values_survive_bulk_narrowing(self) -> None:
        empty = t.tensor([], dtype=t.int64)
        batch = _tensor_selection_batch(
            [0],
            [
                (
                    t.tensor([0, 65_535]),
                    t.tensor([0, 255]),
                    t.tensor([0, 255]),
                    t.tensor([65_535, 0]),
                )
            ],
        )
        builder = _TensorSelectionShardBuilder(1)
        builder.add(batch)
        shard = builder.build()
        assert t.equal(shard.sel_fids, t.tensor([0, 65_535], dtype=t.uint16))
        assert t.equal(shard.neg_fids, t.tensor([65_535, 0], dtype=t.uint16))
        assert not t.equal(shard.sel_fids, empty)

    @pytest.mark.parametrize(
        "row,match",
        [
            (
                (
                    t.tensor([-1]),
                    t.tensor([0]),
                    t.tensor([0]),
                    t.tensor([], dtype=t.int64),
                ),
                "selected feature ID",
            ),
            (
                (
                    t.tensor([65_536]),
                    t.tensor([0]),
                    t.tensor([0]),
                    t.tensor([], dtype=t.int64),
                ),
                "selected feature ID",
            ),
            (
                (
                    t.tensor([1]),
                    t.tensor([-1]),
                    t.tensor([0]),
                    t.tensor([], dtype=t.int64),
                ),
                "bucket",
            ),
            (
                (
                    t.tensor([1]),
                    t.tensor([0]),
                    t.tensor([256]),
                    t.tensor([], dtype=t.int64),
                ),
                "percentile",
            ),
            (
                (
                    t.tensor([], dtype=t.int64),
                    t.tensor([], dtype=t.int64),
                    t.tensor([], dtype=t.int64),
                    t.tensor([65_536]),
                ),
                "negative feature ID",
            ),
        ],
    )
    def test_storage_range_violations_fail_before_narrowing(
        self,
        row,
        match: str,
    ) -> None:
        with pytest.raises(SpliceProvenanceError, match=match):
            _tensor_selection_batch([0], [row])

    def test_misaligned_positive_payload_fails_loud(self) -> None:
        with pytest.raises(SpliceProvenanceError, match="misaligned positive"):
            _tensor_selection_batch(
                [0],
                [
                    (
                        t.tensor([1, 2]),
                        t.tensor([3]),
                        t.tensor([4, 5]),
                        t.tensor([], dtype=t.int64),
                    )
                ],
            )

    def test_duplicate_batch_indices_fail_loud(self) -> None:
        empty = t.tensor([], dtype=t.int64)
        with pytest.raises(SpliceProvenanceError, match="unique segment index"):
            _tensor_selection_batch(
                [1, 1],
                [(empty, empty, empty, empty), (empty, empty, empty, empty)],
            )

    @pytest.mark.parametrize("indices", [[], [True]])
    def test_empty_or_non_integer_segment_indices_fail_loud(self, indices) -> None:
        rows = (
            []
            if not indices
            else [
                (
                    t.tensor([], dtype=t.int64),
                    t.tensor([], dtype=t.int64),
                    t.tensor([], dtype=t.int64),
                    t.tensor([], dtype=t.int64),
                )
            ]
        )
        with pytest.raises(SpliceProvenanceError, match="unique segment index"):
            _tensor_selection_batch(indices, rows)

    @pytest.mark.parametrize("bad_dtype", [t.float32, t.bool, t.complex64])
    def test_non_integer_payload_dtype_fails_before_narrowing(
        self,
        bad_dtype: t.dtype,
    ) -> None:
        bad = t.tensor([1], dtype=bad_dtype)
        with pytest.raises(SpliceProvenanceError, match="integer dtype"):
            _tensor_selection_batch(
                [0],
                [
                    (
                        bad,
                        t.tensor([0]),
                        t.tensor([0]),
                        t.tensor([], dtype=t.int64),
                    )
                ],
            )

    def test_builder_rejects_malformed_direct_count_tensors(self) -> None:
        malformed = _TensorSelectionBatch(
            segment_indices=(0, 1),
            sel_counts=t.tensor([0], dtype=t.int32),
            sel_fids=t.tensor([], dtype=t.uint16),
            sel_buckets=t.tensor([], dtype=t.uint8),
            sel_pcts=t.tensor([], dtype=t.uint8),
            neg_counts=t.tensor([0, -1], dtype=t.int32),
            neg_fids=t.tensor([], dtype=t.uint16),
        )
        with pytest.raises(SpliceProvenanceError, match="selected counts has 1 rows"):
            _TensorSelectionShardBuilder(2).add(malformed)
        malformed = replace(
            malformed,
            sel_counts=t.tensor([0, 0], dtype=t.int32),
        )
        with pytest.raises(SpliceProvenanceError, match="must be non-negative"):
            _TensorSelectionShardBuilder(2).add(malformed)

    @pytest.mark.parametrize(
        "batch,match",
        [
            (
                _TensorSelectionBatch(
                    segment_indices=(0,),
                    sel_counts=t.tensor([1]),
                    sel_fids=t.tensor([], dtype=t.int64),
                    sel_buckets=t.tensor([], dtype=t.int64),
                    sel_pcts=t.tensor([], dtype=t.int64),
                    neg_counts=t.tensor([0]),
                    neg_fids=t.tensor([], dtype=t.int64),
                ),
                "positive counts",
            ),
            (
                _TensorSelectionBatch(
                    segment_indices=(0,),
                    sel_counts=t.tensor([0]),
                    sel_fids=t.tensor([], dtype=t.int64),
                    sel_buckets=t.tensor([], dtype=t.int64),
                    sel_pcts=t.tensor([], dtype=t.int64),
                    neg_counts=t.tensor([1]),
                    neg_fids=t.tensor([], dtype=t.int64),
                ),
                "negative counts",
            ),
            (
                _TensorSelectionBatch(
                    segment_indices=(0,),
                    sel_counts=t.tensor([1]),
                    sel_fids=t.tensor([65_536]),
                    sel_buckets=t.tensor([0]),
                    sel_pcts=t.tensor([0]),
                    neg_counts=t.tensor([0]),
                    neg_fids=t.tensor([], dtype=t.int64),
                ),
                "selected feature ID",
            ),
        ],
    )
    def test_builder_rejects_direct_payload_misalignment_or_narrowing_wrap(
        self,
        batch: _TensorSelectionBatch,
        match: str,
    ) -> None:
        with pytest.raises(SpliceProvenanceError, match=match):
            _TensorSelectionShardBuilder(1).add(batch)

    @pytest.mark.parametrize(
        "counts,payload_size,match",
        [
            ([0], 1, "zero-count"),
            ([1], 2, "counts sum"),
        ],
    )
    def test_row_target_builder_rejects_count_payload_mismatch(
        self,
        counts: list[int],
        payload_size: int,
        match: str,
    ) -> None:
        with pytest.raises(SpliceProvenanceError, match=match):
            _TensorSelectionShardBuilder._row_targets(
                payload_size=payload_size,
                local_counts=t.tensor(counts),
                global_offsets=t.tensor([0, counts[0]]),
                segment_indices=(0,),
            )

    @pytest.mark.parametrize("payload_kind", ["selected", "negative"])
    def test_builder_rejects_int32_csr_overflow_before_allocation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        payload_kind: str,
    ) -> None:
        empty = t.tensor([], dtype=t.int64)
        selected = t.tensor([1]) if payload_kind == "selected" else empty
        negative = t.tensor([2]) if payload_kind == "negative" else empty
        batch = _tensor_selection_batch(
            [0],
            [(selected, selected.clone(), selected.clone(), negative)],
        )
        builder = _TensorSelectionShardBuilder(1)
        builder.add(batch)
        original_iinfo = t.iinfo

        def tiny_int32(dtype: t.dtype):
            if dtype == t.int32:
                return SimpleNamespace(max=0)
            return original_iinfo(dtype)

        monkeypatch.setattr(t, "iinfo", tiny_int32)
        with pytest.raises(SpliceProvenanceError, match=f"{payload_kind}-feature CSR"):
            builder.build()

    def test_build_revalidates_payload_alignment_after_add(self) -> None:
        batch = _tensor_selection_batch(
            [0],
            [(t.tensor([1]), t.tensor([2]), t.tensor([3]), t.tensor([4]))],
        )
        builder = _TensorSelectionShardBuilder(1)
        builder.add(batch)
        batch.sel_buckets.resize_(0)
        with pytest.raises(SpliceProvenanceError, match="positive payload"):
            builder.build()

    @pytest.mark.parametrize("indices", [(), (0, 0), (True,)])
    def test_builder_rejects_malformed_direct_indices(self, indices) -> None:
        malformed = _TensorSelectionBatch(
            segment_indices=indices,
            sel_counts=t.zeros(len(indices), dtype=t.int32),
            sel_fids=t.tensor([], dtype=t.uint16),
            sel_buckets=t.tensor([], dtype=t.uint8),
            sel_pcts=t.tensor([], dtype=t.uint8),
            neg_counts=t.zeros(len(indices), dtype=t.int32),
            neg_fids=t.tensor([], dtype=t.uint16),
        )
        with pytest.raises(SpliceProvenanceError, match="unique integer"):
            _TensorSelectionShardBuilder(2).add(malformed)

    def test_builder_rejects_duplicate_across_batches(self) -> None:
        empty = t.tensor([], dtype=t.int64)
        row = (empty, empty, empty, empty)
        builder = _TensorSelectionShardBuilder(1)
        builder.add(_tensor_selection_batch([0], [row]))
        with pytest.raises(SpliceProvenanceError, match="more than once"):
            builder.add(_tensor_selection_batch([0], [row]))

    def test_rejected_add_is_transactional(self) -> None:
        empty = t.tensor([], dtype=t.int64)
        row = (empty, empty, empty, empty)
        builder = _TensorSelectionShardBuilder(2)
        builder.add(_tensor_selection_batch([0], [row]))
        with pytest.raises(SpliceProvenanceError, match="more than once"):
            builder.add(_tensor_selection_batch([1, 0], [row, row]))
        # Index 1 from the rejected batch must not have been partially committed.
        builder.add(_tensor_selection_batch([1], [row]))
        shard = builder.build()
        assert t.equal(shard.sel_offsets, t.zeros(3, dtype=t.int32))
        assert t.equal(shard.neg_offsets, t.zeros(3, dtype=t.int32))

    @pytest.mark.parametrize("index", [-1, 2])
    def test_builder_rejects_out_of_domain_index(self, index: int) -> None:
        empty = t.tensor([], dtype=t.int64)
        builder = _TensorSelectionShardBuilder(2)
        with pytest.raises(SpliceProvenanceError, match="outside"):
            builder.add(_tensor_selection_batch([index], [(empty, empty, empty, empty)]))

    def test_builder_rejects_missing_rows(self) -> None:
        empty = t.tensor([], dtype=t.int64)
        builder = _TensorSelectionShardBuilder(2)
        builder.add(_tensor_selection_batch([0], [(empty, empty, empty, empty)]))
        with pytest.raises(SpliceProvenanceError, match=r"missing=\[1\]"):
            builder.build()

    def test_randomized_builder_matches_python_reference(self) -> None:
        setup = t.Generator().manual_seed(73_119)
        for trial in range(100):
            n_rows = trial % 19
            rows: list[tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]] = []
            for _ in range(n_rows):
                n_sel = int(t.randint(0, 9, (1,), generator=setup).item())
                n_neg = int(t.randint(0, 8, (1,), generator=setup).item())
                fids = t.randint(0, 65_536, (n_sel,), generator=setup)
                buckets = t.randint(0, 256, (n_sel,), generator=setup)
                pcts = t.randint(0, 256, (n_sel,), generator=setup)
                neg = t.randint(0, 65_536, (n_neg,), generator=setup)
                rows.append((fids, buckets, pcts, neg))
            order = t.randperm(n_rows, generator=setup).tolist()
            builder = _TensorSelectionShardBuilder(n_rows)
            cursor = 0
            while cursor < n_rows:
                width = min(1 + (cursor + trial) % 5, n_rows - cursor)
                indices = order[cursor : cursor + width]
                builder.add(_tensor_selection_batch(indices, [rows[i] for i in indices]))
                cursor += width
            shard = builder.build()

            sel_counts = [int(row[0].numel()) for row in rows]
            neg_counts = [int(row[3].numel()) for row in rows]
            expected_sel_offsets = [0]
            expected_neg_offsets = [0]
            for count in sel_counts:
                expected_sel_offsets.append(expected_sel_offsets[-1] + count)
            for count in neg_counts:
                expected_neg_offsets.append(expected_neg_offsets[-1] + count)
            assert shard.sel_offsets.tolist() == expected_sel_offsets
            assert shard.neg_offsets.tolist() == expected_neg_offsets
            for actual, field, dtype in (
                (shard.sel_fids, 0, t.uint16),
                (shard.sel_buckets, 1, t.uint8),
                (shard.sel_pcts, 2, t.uint8),
                (shard.neg_fids, 3, t.uint16),
            ):
                expected = (
                    t.cat([row[field] for row in rows]).to(dtype)
                    if rows and any(row[field].numel() for row in rows)
                    else t.empty(0, dtype=dtype)
                )
                assert t.equal(actual, expected), (trial, field)


class TestOrdinaryBatchDifferential:
    def test_empty_ordinary_batch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _materialize_ordinary_splice_selection_batch(
                [],
                pipeline_config=_pipeline(seed=8),
            )

    def test_clustered_threshold_config_is_rejected_before_selection(self) -> None:
        cluster_probe = cast(
            FeatureClusterConsolidation,
            SimpleNamespace(is_identity_after_mask=False),
        )
        with pytest.raises(FeatureClustersError, match="requires SamplingBundle"):
            _materialize_ordinary_splice_selection_batch(
                _inputs(1, with_percentiles=True),
                pipeline_config=_threshold_pipeline(seed=8, negative=False),
                feature_clusters=cluster_probe,
            )

    def test_unknown_selection_config_is_rejected(self) -> None:
        malformed = replace(
            _pipeline(seed=8),
            selection_config=cast(Any, object()),
        )
        with pytest.raises(TypeError, match="Unsupported selection config"):
            _materialize_ordinary_splice_selection_batch(
                _inputs(1, with_percentiles=True),
                pipeline_config=malformed,
            )

    def test_real_cluster_consolidation_computes_missing_percentiles(
        self,
        tmp_path,
    ) -> None:
        path = tmp_path / "clusters.json"
        path.write_text(
            json.dumps({"schema_version": 1, "clusters": [[1, 7]]}),
            encoding="utf-8",
        )
        resolved = resolve_feature_clusters(
            {"feature_clusters": {"path": str(path)}},
            N_SAE,
        )
        assert resolved is not None
        clusters = FeatureClusterConsolidation.build(resolved, None)
        item = _OrdinarySpliceSelectionInput(
            segment_index=0,
            raw_fids=t.tensor([1, 7, 9]),
            raw_mags=t.tensor([0.2, 0.8, 0.4]),
            raw_pcts=None,
        )
        batch = _materialize_ordinary_splice_selection_batch(
            [item],
            pipeline_config=_pipeline(seed=8, negative=False),
            feature_clusters=clusters,
        )
        assert 7 not in batch.sel_fids.to(t.int64).tolist()

    def test_duplicate_segment_indices_fail_before_consuming_rng(self) -> None:
        inputs = _inputs(2, with_percentiles=True)
        inputs[1] = replace(inputs[1], segment_index=inputs[0].segment_index)
        pipeline = _pipeline(seed=612)
        assert isinstance(pipeline.selection_config, SamplingBundle)
        positive_before = pipeline.selection_config.rng.get_state().clone()
        assert pipeline.negative_bundle is not None
        negative_before = pipeline.negative_bundle.rng.get_state().clone()
        with pytest.raises(SpliceProvenanceError, match="unique integer"):
            _materialize_ordinary_splice_selection_batch(
                inputs,
                pipeline_config=pipeline,
            )
        assert t.equal(pipeline.selection_config.rng.get_state(), positive_before)
        assert t.equal(pipeline.negative_bundle.rng.get_state(), negative_before)

    @pytest.mark.parametrize("mode", list(SamplingMode))
    @pytest.mark.parametrize("with_percentiles", [False, True])
    @pytest.mark.parametrize("masked", [False, True])
    def test_batch_size_one_is_output_and_rng_exact_legacy(
        self,
        mode: SamplingMode,
        with_percentiles: bool,
        masked: bool,
    ) -> None:
        mask = (
            FeatureMask(FeatureMaskMode.whitelist, tuple(range(112)))
            if masked
            else None
        )
        item = _inputs(1, with_percentiles=with_percentiles)[0]
        # Force a non-empty row for this exactness test.
        item = replace(
            item,
            raw_fids=t.tensor([1, 7, 15, 31, 63, 95]),
            raw_mags=t.tensor([0.1, 0.2, 0.4, 0.8, 1.6, 3.2]),
            raw_pcts=(t.tensor([0.05, 0.15, 0.35, 0.55, 0.75, 0.95]) if with_percentiles else None),
        )
        legacy_pipeline = _pipeline(seed=912, mode=mode, mask=mask)
        batched_pipeline = _pipeline(seed=912, mode=mode, mask=mask)
        legacy = _require_tensor_row(
            _materialize_splice_segment_selection(
                raw_fids=item.raw_fids,
                raw_mags=item.raw_mags,
                raw_pcts=item.raw_pcts,
                delta_result=None,
                pipeline_config=legacy_pipeline,
                tensor_output=True,
            )
        )
        batch = _materialize_ordinary_splice_selection_batch(
            [item],
            pipeline_config=batched_pipeline,
        )
        actual = _batch_rows(batch)[0]
        for got, expected in zip(actual, legacy, strict=True):
            assert t.equal(got, expected)
        assert isinstance(legacy_pipeline.selection_config, SamplingBundle)
        assert isinstance(batched_pipeline.selection_config, SamplingBundle)
        assert t.equal(
            legacy_pipeline.selection_config.rng.get_state(),
            batched_pipeline.selection_config.rng.get_state(),
        )
        assert legacy_pipeline.negative_bundle is not None
        assert batched_pipeline.negative_bundle is not None
        assert t.equal(
            legacy_pipeline.negative_bundle.rng.get_state(),
            batched_pipeline.negative_bundle.rng.get_state(),
        )

    def test_fixed_cardinality_unique_ranking_is_exact_despite_rng_regrouping(self) -> None:
        items = []
        for row in range(12):
            fids = t.tensor([row, row + 16, row + 32, row + 48])
            pcts = t.tensor([0.1, 0.4, 0.7, 0.9]) + row * 1e-5
            items.append(
                _OrdinarySpliceSelectionInput(
                    row,
                    fids,
                    pcts.clone(),
                    pcts,
                )
            )
        legacy_pipeline = _pipeline(
            seed=441,
            negative=False,
            min_k=2,
            max_k=3,
        )
        batched_pipeline = _pipeline(
            seed=441,
            negative=False,
            min_k=2,
            max_k=3,
        )
        legacy_rows = []
        for item in items:
            row = _require_tensor_row(
                _materialize_splice_segment_selection(
                    raw_fids=item.raw_fids,
                    raw_mags=item.raw_mags,
                    raw_pcts=item.raw_pcts,
                    delta_result=None,
                    pipeline_config=legacy_pipeline,
                    tensor_output=True,
                )
            )
            legacy_rows.append(row)
        batch = _materialize_ordinary_splice_selection_batch(
            items,
            pipeline_config=batched_pipeline,
        )
        for actual, expected in zip(_batch_rows(batch), legacy_rows, strict=True):
            for got, want in zip(actual, expected, strict=True):
                assert t.equal(got, want)
        assert isinstance(legacy_pipeline.selection_config, SamplingBundle)
        assert isinstance(batched_pipeline.selection_config, SamplingBundle)
        # The grouping reallocates draws among operations but consumes the
        # same total count here (fixed k, equal row widths). Pin the final state
        # too: representation refactoring must not introduce hidden draws.
        assert t.equal(
            legacy_pipeline.selection_config.rng.get_state(),
            batched_pipeline.selection_config.rng.get_state(),
        )

    @pytest.mark.parametrize("mode", list(SamplingMode))
    def test_positive_rows_are_exact_when_cardinality_rng_is_factored_out(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mode: SamplingMode,
    ) -> None:
        """Prove tensor concatenation/ranking does not cause output drift.

        The real mode draws all row cardinalities before row-local ranking,
        which deliberately reassigns RNG values relative to one-row dispatch.
        Replacing only that draw with a fixed cardinality makes serial and
        multi-row execution consume the same row-local random stream. Exact
        output and final-state equality then isolate allowed drift to RNG
        scheduling rather than score, mask, ranking, or CSR semantics.
        """

        def fixed_k(
            distribution,
            mean,
            std,
            rng,
            n,
            min_k=0,
            max_k=None,
        ):
            del distribution, mean, std, rng, min_k, max_k
            return t.full((n,), 2, dtype=t.int64)

        monkeypatch.setattr(sampling, "sample_k_batched", fixed_k)
        items = _inputs(19, with_percentiles=True)
        serial_pipeline = _pipeline(seed=2_117, mode=mode, negative=False)
        batch_pipeline = _pipeline(seed=2_117, mode=mode, negative=False)
        expected = []
        for item in items:
            row = _require_tensor_row(
                _materialize_splice_segment_selection(
                    raw_fids=item.raw_fids,
                    raw_mags=item.raw_mags,
                    raw_pcts=item.raw_pcts,
                    delta_result=None,
                    pipeline_config=serial_pipeline,
                    tensor_output=True,
                )
            )
            expected.append(row)
        actual = _batch_rows(
            _materialize_ordinary_splice_selection_batch(
                items,
                pipeline_config=batch_pipeline,
            )
        )
        for got_row, expected_row in zip(actual, expected, strict=True):
            for got, want in zip(got_row, expected_row, strict=True):
                assert t.equal(got, want)
        assert isinstance(serial_pipeline.selection_config, SamplingBundle)
        assert isinstance(batch_pipeline.selection_config, SamplingBundle)
        assert t.equal(
            serial_pipeline.selection_config.rng.get_state(),
            batch_pipeline.selection_config.rng.get_state(),
        )

    def test_multisegment_threshold_selection_is_exact_serial_without_rng(self) -> None:
        mask = FeatureMask(FeatureMaskMode.blacklist, (5, 17, 65))
        items = _inputs(37, with_percentiles=False)
        serial_pipeline = _threshold_pipeline(
            seed=991,
            negative=False,
            mask=mask,
        )
        batch_pipeline = _threshold_pipeline(
            seed=991,
            negative=False,
            mask=mask,
        )
        expected = []
        for item in items:
            row = _require_tensor_row(
                _materialize_splice_segment_selection(
                    raw_fids=item.raw_fids,
                    raw_mags=item.raw_mags,
                    raw_pcts=item.raw_pcts,
                    delta_result=None,
                    pipeline_config=serial_pipeline,
                    tensor_output=True,
                )
            )
            expected.append(row)
        actual = _batch_rows(
            _materialize_ordinary_splice_selection_batch(
                items,
                pipeline_config=batch_pipeline,
            )
        )
        for got_row, expected_row in zip(actual, expected, strict=True):
            for got, want in zip(got_row, expected_row, strict=True):
                assert t.equal(got, want)

    def test_one_row_threshold_negative_output_and_rng_are_exact_serial(self) -> None:
        mask = FeatureMask(FeatureMaskMode.whitelist, tuple(range(112)))
        item = replace(
            _inputs(1, with_percentiles=False)[0],
            raw_fids=t.tensor([1, 7, 15, 31, 63, 95]),
            raw_mags=t.tensor([0.1, 0.2, 0.4, 0.8, 1.6, 3.2]),
        )
        serial_pipeline = _threshold_pipeline(seed=811, negative=True, mask=mask)
        batch_pipeline = _threshold_pipeline(seed=811, negative=True, mask=mask)
        expected = _require_tensor_row(
            _materialize_splice_segment_selection(
                raw_fids=item.raw_fids,
                raw_mags=item.raw_mags,
                raw_pcts=None,
                delta_result=None,
                pipeline_config=serial_pipeline,
                tensor_output=True,
            )
        )
        actual = _batch_rows(
            _materialize_ordinary_splice_selection_batch(
                [item],
                pipeline_config=batch_pipeline,
            )
        )[0]
        for got, want in zip(actual, expected, strict=True):
            assert t.equal(got, want)
        assert serial_pipeline.negative_bundle is not None
        assert batch_pipeline.negative_bundle is not None
        assert t.equal(
            serial_pipeline.negative_bundle.rng.get_state(),
            batch_pipeline.negative_bundle.rng.get_state(),
        )

    @pytest.mark.parametrize("mode", list(SamplingMode))
    @pytest.mark.parametrize("with_percentiles", [False, True])
    def test_multisegment_semantic_invariants(
        self,
        mode: SamplingMode,
        with_percentiles: bool,
    ) -> None:
        mask = FeatureMask(FeatureMaskMode.blacklist, (2, 5, 9, 17, 33, 65))
        items = _inputs(37, with_percentiles=with_percentiles)
        pipeline = _pipeline(seed=71_003, mode=mode, mask=mask)
        first = _materialize_ordinary_splice_selection_batch(
            items,
            pipeline_config=pipeline,
        )
        replay = _materialize_ordinary_splice_selection_batch(
            items,
            pipeline_config=_pipeline(seed=71_003, mode=mode, mask=mask),
        )
        assert first.segment_indices == replay.segment_indices
        for got, expected in zip(
            (
                first.sel_counts,
                first.sel_fids,
                first.sel_buckets,
                first.sel_pcts,
                first.neg_counts,
                first.neg_fids,
            ),
            (
                replay.sel_counts,
                replay.sel_fids,
                replay.sel_buckets,
                replay.sel_pcts,
                replay.neg_counts,
                replay.neg_fids,
            ),
            strict=True,
        ):
            assert t.equal(got, expected)

        allowed = mask.allowed_universe_mask(N_SAE)
        for item, (selected, buckets, pcts, negative) in zip(
            items,
            _batch_rows(first),
            strict=True,
        ):
            raw = set(item.raw_fids.tolist())
            assert set(selected.tolist()).issubset(raw)
            assert set(negative.tolist()).isdisjoint(raw)
            assert selected.unique().numel() == selected.numel()
            assert negative.unique().numel() == negative.numel()
            assert selected.numel() <= min(8, item.raw_fids.numel())
            assert negative.numel() <= 7
            assert buckets.numel() == pcts.numel() == selected.numel()
            assert bool((buckets >= 0).all()) and bool((buckets < 4).all())
            assert bool(allowed[selected].all()) if selected.numel() else True
            assert bool(allowed[negative].all()) if negative.numel() else True

    def test_mixed_percentile_representation_fails_instead_of_dropping_overlay(self) -> None:
        items = _inputs(2, with_percentiles=True)
        items[1] = replace(items[1], raw_pcts=None)
        with pytest.raises(SpliceProvenanceError, match="mixed segments"):
            _materialize_ordinary_splice_selection_batch(
                items,
                pipeline_config=_pipeline(seed=8),
            )

    @pytest.mark.parametrize("field", ["raw_mags", "raw_pcts"])
    def test_per_row_tensor_misalignment_fails_before_concatenation(
        self,
        field: str,
    ) -> None:
        item = replace(
            _inputs(1, with_percentiles=True)[0],
            raw_fids=t.tensor([1, 2, 3]),
            raw_mags=t.tensor([0.1, 0.2, 0.3]),
            raw_pcts=t.tensor([0.1, 0.2, 0.3]),
        )
        if field == "raw_mags":
            item = replace(item, raw_mags=t.tensor([0.1, 0.2]))
        else:
            item = replace(item, raw_pcts=t.tensor([0.1, 0.2]))
        with pytest.raises(SpliceProvenanceError, match="misaligned raw tensors"):
            _materialize_ordinary_splice_selection_batch(
                [item],
                pipeline_config=_pipeline(seed=8),
            )

    def test_non_integer_raw_feature_ids_fail_before_selection(self) -> None:
        item = replace(
            _inputs(1, with_percentiles=False)[0],
            raw_fids=t.tensor([1.5]),
            raw_mags=t.tensor([0.2]),
        )
        with pytest.raises(SpliceProvenanceError, match="integer dtype"):
            _materialize_ordinary_splice_selection_batch(
                [item],
                pipeline_config=_pipeline(seed=8),
            )

    def test_tensor_output_is_exactly_the_legacy_list_values(self) -> None:
        item = replace(
            _inputs(1, with_percentiles=True)[0],
            raw_fids=t.tensor([1, 3, 7, 15, 31]),
            raw_mags=t.tensor([0.2, 0.3, 0.7, 1.5, 3.1]),
            raw_pcts=t.tensor([0.1, 0.2, 0.4, 0.7, 0.9]),
        )
        list_pipeline = _pipeline(seed=1234)
        tensor_pipeline = _pipeline(seed=1234)
        list_result = _require_list_row(
            _materialize_splice_segment_selection(
                raw_fids=item.raw_fids,
                raw_mags=item.raw_mags,
                raw_pcts=item.raw_pcts,
                delta_result=None,
                pipeline_config=list_pipeline,
            )
        )
        tensor_result = _require_tensor_row(
            _materialize_splice_segment_selection(
                raw_fids=item.raw_fids,
                raw_mags=item.raw_mags,
                raw_pcts=item.raw_pcts,
                delta_result=None,
                pipeline_config=tensor_pipeline,
                tensor_output=True,
            )
        )
        for values, tensor in zip(list_result, tensor_result, strict=True):
            assert values == tensor.tolist()
