"""Exact fast-vs-reference tests for conditioned-SFT segment selection."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pytest
import torch as t
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.conditional.encoder_decoder import sft_selection
from steering.conditional.encoder_decoder.dataloader import SegFeatures
from steering.conditional.encoder_decoder.sft_selection import (
    SegmentSelectionContext,
    select_conversation_segments,
    select_conversation_segments_reference,
)
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
)
from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.missing_percentile import (
    LOWEST_MAGNITUDE_BUCKET,
    MissingPercentilePolicy,
    MissingPercentileResolver,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeBundle,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
)
from steering.conditional.thresholding import ThresholdConfig, ThresholdType

N_SAE = 2048
CASES = (
    "threshold_binary",
    "threshold_masked_fp",
    "threshold_lowest_fp",
    "threshold_fp_lowest_negatives",
    "sampling_weighted_shared_fp_masked",
    "sampling_weighted_shared_fp_unmasked",
    "sampling_topk_binary",
    "sampling_lowest_own_table_negatives",
    "clustered_threshold",
    "clustered_sampling_shared_fp_negatives",
)


def _table(*, uncovered: tuple[int, ...] = (), offset: float = 0.0) -> t.Tensor:
    table = (
        t.linspace(0.5 + offset, 64.5 + offset, 99, dtype=t.float64)
        .unsqueeze(0)
        .expand(N_SAE, -1)
        .clone()
    )
    if uncovered:
        table[list(uncovered)] = 0.0
    return table


def _boundaries(*, binary: bool = False) -> t.Tensor:
    if binary:
        return t.zeros(N_SAE, 0, dtype=t.float64)
    return t.tensor([4.0, 12.0, 28.0], dtype=t.float64).expand(N_SAE, -1).contiguous()


def _sampling(
    *,
    seed: int,
    table: t.Tensor,
    mode: SamplingMode = SamplingMode.weighted,
) -> SamplingBundle:
    return SamplingBundle(
        selection=SamplingSelection(
            distribution=Distribution.geometric,
            mean=3.0,
            std=3.0,
            min_k=1,
            max_k=6,
            mode=mode,
        ),
        percentile_table=table,
        rng=t.Generator().manual_seed(seed),
    )


def _negative(*, rng: t.Generator, feature_mask: FeatureMask | None) -> NegativeBundle:
    return NegativeBundle(
        selection=NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=2.0,
            std=2.0,
            min_k=1,
            max_k=5,
        ),
        n_sae_features=N_SAE,
        rng=rng,
        feature_mask=feature_mask,
    )


def _consolidation(feature_mask: FeatureMask | None = None) -> FeatureClusterConsolidation:
    definition = FeatureClusterDefinition(clusters=((1, 2, 3), (10, 11), (20, 21, 22), (100, 101)))
    resolved = ResolvedFeatureClusters(
        source_path=Path("/tmp/sft-selection-clusters.yaml"),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition,
            n_sae_features=N_SAE,
        ),
    )
    return FeatureClusterConsolidation.build(resolved, feature_mask)


def _context(case: str, *, seed: int) -> SegmentSelectionContext:
    error = MissingPercentileResolver.error()
    uncovered = tuple(range(0, N_SAE, 17))
    covered = t.ones(N_SAE, dtype=t.bool)
    covered[list(uncovered)] = False
    lowest = MissingPercentileResolver(MissingPercentilePolicy.LOWEST, covered)

    if case == "threshold_binary":
        return SegmentSelectionContext(
            threshold_cfg=ThresholdConfig(ThresholdType.segment_topk, 7),
            precomputed_thresholds=None,
            sampling_bundle=None,
            negative_bundle=None,
            boundaries=_boundaries(binary=True),
            fp_percentile_table=None,
            cluster_threshold_percentile_table=None,
            feature_mask=None,
            consolidation=None,
            missing_percentile=error,
        )
    if case == "threshold_masked_fp":
        mask = FeatureMask(FeatureMaskMode.blacklist, tuple(range(0, N_SAE, 11)))
        return SegmentSelectionContext(
            threshold_cfg=ThresholdConfig(ThresholdType.feature_percentile, 0.5),
            precomputed_thresholds=t.full((N_SAE,), 8.0, dtype=t.float64),
            sampling_bundle=None,
            negative_bundle=None,
            boundaries=_boundaries(),
            fp_percentile_table=_table(),
            cluster_threshold_percentile_table=None,
            feature_mask=mask,
            consolidation=None,
            missing_percentile=error,
        )
    if case == "threshold_lowest_fp":
        return SegmentSelectionContext(
            threshold_cfg=ThresholdConfig(ThresholdType.segment_topk, 5),
            precomputed_thresholds=None,
            sampling_bundle=None,
            negative_bundle=None,
            boundaries=_boundaries(),
            fp_percentile_table=_table(uncovered=uncovered),
            cluster_threshold_percentile_table=None,
            feature_mask=None,
            consolidation=None,
            missing_percentile=lowest,
        )
    if case == "threshold_fp_lowest_negatives":
        # The production shape (PR-1 scenario 1): threshold selection, an FP
        # table, uncovered ids under ``lowest``, and negatives drawn from a
        # generator that is NOT shared with any sampling bundle.
        return SegmentSelectionContext(
            threshold_cfg=ThresholdConfig(
                ThresholdType.feature_percentile_segment_topk,
                4.5,
            ),
            precomputed_thresholds=t.full((N_SAE,), 6.0, dtype=t.float64),
            sampling_bundle=None,
            negative_bundle=_negative(
                rng=t.Generator().manual_seed(seed),
                feature_mask=None,
            ),
            boundaries=_boundaries(),
            fp_percentile_table=_table(uncovered=uncovered),
            cluster_threshold_percentile_table=None,
            feature_mask=None,
            consolidation=None,
            missing_percentile=lowest,
        )
    if case == "sampling_weighted_shared_fp_masked":
        table = _table()
        mask = FeatureMask(FeatureMaskMode.blacklist, tuple(range(0, N_SAE, 13)))
        sampling = _sampling(seed=seed, table=table)
        return SegmentSelectionContext(
            threshold_cfg=None,
            precomputed_thresholds=None,
            sampling_bundle=sampling,
            negative_bundle=None,
            boundaries=_boundaries(),
            fp_percentile_table=table,
            cluster_threshold_percentile_table=None,
            feature_mask=mask,
            consolidation=None,
            missing_percentile=error,
        )
    if case == "sampling_weighted_shared_fp_unmasked":
        # The one shape where the batched FP rank slice reaches
        # ``sample_features`` as ``override_weights`` UNTOUCHED: with a mask,
        # ``apply_to_positive`` indexes it into a fresh tensor first, so this
        # is the only config in which a raw ``split`` view crosses into the
        # sampler.
        table = _table()
        sampling = _sampling(seed=seed, table=table)
        return SegmentSelectionContext(
            threshold_cfg=None,
            precomputed_thresholds=None,
            sampling_bundle=sampling,
            negative_bundle=None,
            boundaries=_boundaries(),
            fp_percentile_table=table,
            cluster_threshold_percentile_table=None,
            feature_mask=None,
            consolidation=None,
            missing_percentile=error,
        )
    if case == "sampling_topk_binary":
        sampling = _sampling(
            seed=seed,
            table=_table(),
            mode=SamplingMode.top_k_percentile,
        )
        return SegmentSelectionContext(
            threshold_cfg=None,
            precomputed_thresholds=None,
            sampling_bundle=sampling,
            negative_bundle=None,
            boundaries=_boundaries(binary=True),
            fp_percentile_table=None,
            cluster_threshold_percentile_table=None,
            feature_mask=None,
            consolidation=None,
            missing_percentile=error,
        )
    if case == "sampling_lowest_own_table_negatives":
        sampling = _sampling(seed=seed, table=_table(uncovered=uncovered))
        negative = _negative(rng=sampling.rng, feature_mask=None)
        return SegmentSelectionContext(
            threshold_cfg=None,
            precomputed_thresholds=None,
            sampling_bundle=sampling,
            negative_bundle=negative,
            boundaries=_boundaries(),
            fp_percentile_table=_table(offset=0.25),
            cluster_threshold_percentile_table=None,
            feature_mask=None,
            consolidation=None,
            missing_percentile=lowest,
        )
    if case == "clustered_threshold":
        return SegmentSelectionContext(
            threshold_cfg=ThresholdConfig(
                ThresholdType.feature_percentile_segment_topk,
                3.5,
            ),
            precomputed_thresholds=None,
            sampling_bundle=None,
            negative_bundle=None,
            boundaries=_boundaries(),
            fp_percentile_table=None,
            cluster_threshold_percentile_table=_table(),
            feature_mask=None,
            consolidation=_consolidation(),
            missing_percentile=error,
        )
    if case == "clustered_sampling_shared_fp_negatives":
        table = _table()
        mask = FeatureMask(FeatureMaskMode.blacklist, tuple(range(1000, N_SAE, 19)))
        consolidation = _consolidation(mask)
        sampling = _sampling(seed=seed, table=table)
        negative = _negative(
            rng=sampling.rng,
            feature_mask=consolidation.effective_negative_mask(),
        )
        return SegmentSelectionContext(
            threshold_cfg=None,
            precomputed_thresholds=None,
            sampling_bundle=sampling,
            negative_bundle=negative,
            boundaries=_boundaries(),
            fp_percentile_table=table,
            cluster_threshold_percentile_table=None,
            feature_mask=mask,
            consolidation=consolidation,
            missing_percentile=error,
        )
    raise AssertionError(case)


@st.composite
def _conversations(draw: st.DrawFn) -> list[SegmentFeatures]:
    sizes = draw(
        st.lists(
            st.integers(min_value=0, max_value=2000),
            min_size=1,
            max_size=10,
        )
    )
    seed = draw(st.integers(min_value=0, max_value=10_000))
    # Real shards store each segment's ids ascending; unsorted ids still reach
    # the selector through cluster consolidation, so cover both.
    sort_ids = draw(st.booleans())
    segments: list[SegmentFeatures] = []
    for segment_idx, size in enumerate(sizes):
        # A permutation prefix: ids are distinct within the segment, which is
        # what ``_assert_unique_sft_fp_source_ids`` requires of real data.
        gen = t.Generator().manual_seed(seed * 31 + segment_idx)
        fids = t.randperm(N_SAE, generator=gen)[:size].to(t.int32)
        if sort_ids:
            fids = fids.sort().values
        mags = t.rand(size, generator=gen, dtype=t.float32) * 60.0 + 0.25
        segments.append(
            SegmentFeatures(
                segment_idx=segment_idx,
                position=segment_idx * 128,
                feature_ids=fids,
                feature_mags=mags,
                char_start=segment_idx * 8,
                char_end=(segment_idx + 1) * 8,
            )
        )
    return segments


def _assert_seg_features_equal(
    actual: list[SegFeatures],
    expected: list[SegFeatures],
) -> None:
    assert len(actual) == len(expected)
    for actual_seg, expected_seg in zip(actual, expected, strict=True):
        for field in (
            "feature_ids",
            "buckets",
            "percentiles",
            "all_feature_ids",
            "all_feature_pcts",
        ):
            actual_value = getattr(actual_seg, field)
            expected_value = getattr(expected_seg, field)
            if actual_value is None:
                assert expected_value is None
            else:
                assert isinstance(expected_value, t.Tensor)
                assert t.equal(actual_value, expected_value), field
        for field in (
            "char_start",
            "char_end",
            "token_start",
            "token_end",
            "prediction_target_only",
        ):
            assert getattr(actual_seg, field) == getattr(expected_seg, field)
        assert actual_seg.feature_ids.dtype is t.int64
        assert actual_seg.buckets.dtype is t.int64
        assert actual_seg.percentiles.dtype is t.float32
        if actual_seg.all_feature_ids is not None:
            assert actual_seg.all_feature_ids.dtype is t.int64
            assert actual_seg.all_feature_pcts is not None
            assert actual_seg.all_feature_pcts.dtype is t.float32


def _assert_rng_states_equal(
    actual: SegmentSelectionContext,
    expected: SegmentSelectionContext,
) -> None:
    for field in ("sampling_bundle", "negative_bundle"):
        actual_bundle = getattr(actual, field)
        expected_bundle = getattr(expected, field)
        if actual_bundle is None:
            assert expected_bundle is None
        else:
            assert expected_bundle is not None
            assert t.equal(actual_bundle.rng.get_state(), expected_bundle.rng.get_state())
    if actual.sampling_bundle is not None and actual.negative_bundle is not None:
        assert expected.sampling_bundle is not None
        assert expected.negative_bundle is not None
        assert (actual.sampling_bundle.rng is actual.negative_bundle.rng) == (
            expected.sampling_bundle.rng is expected.negative_bundle.rng
        )


@pytest.mark.parametrize("case", CASES)
@settings(max_examples=60, deadline=None)
@given(raw_segments=_conversations(), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_vectorized_matches_reference_hypothesis(
    case: str,
    raw_segments: list[SegmentFeatures],
    seed: int,
) -> None:
    fast_ctx = _context(case, seed=seed)
    reference_ctx = _context(case, seed=seed)
    with t.random.fork_rng(devices=[]):
        default_before = t.random.get_rng_state().clone()
        actual = select_conversation_segments(raw_segments, ctx=fast_ctx, example_idx=37)
        assert t.equal(t.random.get_rng_state(), default_before)
        expected = select_conversation_segments_reference(
            raw_segments,
            ctx=reference_ctx,
            example_idx=37,
        )
        assert t.equal(t.random.get_rng_state(), default_before)
    _assert_seg_features_equal(actual, expected)
    _assert_rng_states_equal(fast_ctx, reference_ctx)


@pytest.mark.parametrize("sizes", ([0, 3, 0, 5], [7], [0, 0, 0]))
def test_empty_interleaving_single_and_all_empty(sizes: list[int]) -> None:
    raw = [
        SegmentFeatures(
            segment_idx=i,
            position=i * 128,
            feature_ids=t.arange(size, dtype=t.int32),
            feature_mags=t.arange(size, dtype=t.float32) + 1.0,
            char_start=i * 8,
            char_end=(i + 1) * 8,
        )
        for i, size in enumerate(sizes)
    ]
    fast_ctx = _context("sampling_weighted_shared_fp_masked", seed=41)
    reference_ctx = _context("sampling_weighted_shared_fp_masked", seed=41)
    actual = select_conversation_segments(raw, ctx=fast_ctx, example_idx=1)
    expected = select_conversation_segments_reference(raw, ctx=reference_ctx, example_idx=1)
    _assert_seg_features_equal(actual, expected)
    _assert_rng_states_equal(fast_ctx, reference_ctx)
    for source, output in zip(raw, actual, strict=True):
        if source.feature_ids.numel() == 0:
            assert output.all_feature_ids is None
            assert output.all_feature_pcts is None
            assert output.buckets.dtype is t.int64 and output.buckets.numel() == 0


def test_all_masked_segments_match_reference() -> None:
    raw = [
        SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([1, 2, 3], dtype=t.int32),
            feature_mags=t.tensor([4.0, 8.0, 12.0], dtype=t.float32),
            char_start=0,
            char_end=8,
        )
    ]
    mask = FeatureMask(FeatureMaskMode.whitelist, (N_SAE - 1,))

    def make_ctx() -> SegmentSelectionContext:
        return SegmentSelectionContext(
            threshold_cfg=ThresholdConfig(ThresholdType.segment_topk, 3),
            precomputed_thresholds=None,
            sampling_bundle=None,
            negative_bundle=None,
            boundaries=_boundaries(),
            fp_percentile_table=_table(),
            cluster_threshold_percentile_table=None,
            feature_mask=mask,
            consolidation=None,
            missing_percentile=MissingPercentileResolver.error(),
        )

    actual = select_conversation_segments(raw, ctx=make_ctx(), example_idx=2)
    expected = select_conversation_segments_reference(raw, ctx=make_ctx(), example_idx=2)
    _assert_seg_features_equal(actual, expected)
    assert actual[0].feature_ids.numel() == 0
    assert actual[0].buckets.dtype is t.int64
    assert actual[0].all_feature_ids is not None
    assert actual[0].all_feature_ids.tolist() == [1, 2, 3]


def test_uncovered_lowest_pins_fp_and_bucket() -> None:
    uncovered_id = 17
    covered = t.ones(N_SAE, dtype=t.bool)
    covered[uncovered_id] = False
    ctx = SegmentSelectionContext(
        threshold_cfg=ThresholdConfig(ThresholdType.segment_topk, 1),
        precomputed_thresholds=None,
        sampling_bundle=None,
        negative_bundle=None,
        boundaries=_boundaries(),
        fp_percentile_table=_table(uncovered=(uncovered_id,)),
        cluster_threshold_percentile_table=None,
        feature_mask=None,
        consolidation=None,
        missing_percentile=MissingPercentileResolver(
            MissingPercentilePolicy.LOWEST,
            covered,
        ),
    )
    raw = [
        SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([uncovered_id], dtype=t.int32),
            feature_mags=t.tensor([50.0], dtype=t.float32),
            char_start=0,
            char_end=8,
        )
    ]
    result = select_conversation_segments(raw, ctx=ctx, example_idx=3)
    assert result[0].feature_ids.tolist() == [uncovered_id]
    assert result[0].buckets.tolist() == [LOWEST_MAGNITUDE_BUCKET]
    assert result[0].all_feature_pcts is not None
    assert result[0].all_feature_pcts.tolist() == pytest.approx([0.005])


def test_missing_percentile_error_type_message_and_rng_match() -> None:
    table = _table(uncovered=(17, 34))
    raw = [
        SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([17], dtype=t.int32),
            feature_mags=t.tensor([2.0], dtype=t.float32),
            char_start=0,
            char_end=8,
        ),
        SegmentFeatures(
            segment_idx=1,
            position=128,
            feature_ids=t.tensor([34], dtype=t.int32),
            feature_mags=t.tensor([3.0], dtype=t.float32),
            char_start=8,
            char_end=16,
        ),
    ]

    def make_ctx() -> SegmentSelectionContext:
        sampling = _sampling(seed=99, table=table)
        return SegmentSelectionContext(
            threshold_cfg=None,
            precomputed_thresholds=None,
            sampling_bundle=sampling,
            negative_bundle=None,
            boundaries=_boundaries(),
            fp_percentile_table=table,
            cluster_threshold_percentile_table=None,
            feature_mask=None,
            consolidation=None,
            missing_percentile=MissingPercentileResolver.error(),
        )

    fast_ctx = make_ctx()
    reference_ctx = make_ctx()
    pattern = r"Percentile statistics are missing for 1 distinct feature ID\(s\)"
    with pytest.raises(ValueError, match=pattern) as fast_exc:
        select_conversation_segments(raw, ctx=fast_ctx, example_idx=4)
    with pytest.raises(ValueError, match=pattern) as reference_exc:
        select_conversation_segments_reference(raw, ctx=reference_ctx, example_idx=4)
    assert type(fast_exc.value) is type(reference_exc.value)
    assert str(fast_exc.value) == str(reference_exc.value)
    _assert_rng_states_equal(fast_ctx, reference_ctx)


def test_out_of_range_error_type_and_message_match(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Discretize error parity with the batched prefix LIVE.

    An FP table is supplied so the batched rank phase actually runs and
    succeeds; the boundaries tensor is deliberately narrower than the table so
    the id is valid for the percentile lookup but out of range for
    ``discretize_activations``. That puts the raise in the per-segment stage
    AFTER the prefix — the plan's "exceptions raised in the per-segment
    selection/RNG stage propagate directly" arm. Pointing this at a context
    whose prefix is inert would compare the verbatim loop with itself.
    """
    boundaries_rows = 64
    out_of_range_id = 100
    assert out_of_range_id >= boundaries_rows
    raw = [
        SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([out_of_range_id], dtype=t.int32),
            feature_mags=t.tensor([3.0], dtype=t.float32),
            char_start=0,
            char_end=8,
        )
    ]

    def make_ctx() -> SegmentSelectionContext:
        return SegmentSelectionContext(
            threshold_cfg=ThresholdConfig(ThresholdType.segment_topk, 1),
            precomputed_thresholds=None,
            sampling_bundle=None,
            negative_bundle=None,
            boundaries=t.tensor([4.0, 12.0, 28.0], dtype=t.float64)
            .expand(boundaries_rows, -1)
            .contiguous(),
            fp_percentile_table=_table(),
            cluster_threshold_percentile_table=None,
            feature_mask=None,
            consolidation=None,
            missing_percentile=MissingPercentileResolver.error(),
        )

    fast_ctx = make_ctx()
    reference_ctx = make_ctx()
    message = (
        f"Feature IDs [{out_of_range_id}] exceed boundaries size {boundaries_rows}. "
        "Boundaries tensor must cover all feature IDs."
    )
    with caplog.at_level(logging.WARNING):
        with pytest.raises(ValueError, match=rf"^{re.escape(message)}$") as fast_exc:
            select_conversation_segments(raw, ctx=fast_ctx, example_idx=5)
    with pytest.raises(ValueError, match=rf"^{re.escape(message)}$") as reference_exc:
        select_conversation_segments_reference(raw, ctx=reference_ctx, example_idx=5)
    assert type(fast_exc.value) is type(reference_exc.value)
    assert str(fast_exc.value) == str(reference_exc.value)
    # Non-vacuity: the batched prefix ran to completion and the raise came from
    # the per-segment stage directly, NOT from the reference rerun.
    assert not any(
        "fell back to the reference segment loop" in record.message for record in caplog.records
    )


def test_batched_prefix_fallback_uses_reference_before_rng(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw = [
        SegmentFeatures(
            segment_idx=i,
            position=i * 128,
            feature_ids=t.arange(i + 2, dtype=t.int32) + i * 20,
            feature_mags=t.arange(i + 2, dtype=t.float32) + 1.0,
            char_start=i * 8,
            char_end=(i + 1) * 8,
        )
        for i in range(2)
    ]
    expected_ctx = _context("sampling_weighted_shared_fp_masked", seed=123)
    expected = select_conversation_segments_reference(
        raw,
        ctx=expected_ctx,
        example_idx=6,
    )
    fast_ctx = _context("sampling_weighted_shared_fp_masked", seed=123)
    real_ranks = MissingPercentileResolver.ranks
    real_reference = sft_selection.select_conversation_segments_reference
    reference_calls: list[int] = []

    def fail_only_batched(
        self: MissingPercentileResolver,
        feature_ids: t.Tensor,
        feature_mags: t.Tensor,
        percentile_table: t.Tensor,
    ) -> t.Tensor:
        if len(feature_ids) > 3:
            raise RuntimeError("forced pure-prefix failure")
        return real_ranks(self, feature_ids, feature_mags, percentile_table)

    def record_reference(*args: Any, **kwargs: Any) -> list[SegFeatures]:
        reference_calls.append(1)
        return real_reference(*args, **kwargs)

    monkeypatch.setattr(MissingPercentileResolver, "ranks", fail_only_batched)
    monkeypatch.setattr(
        sft_selection,
        "select_conversation_segments_reference",
        record_reference,
    )
    with caplog.at_level(logging.WARNING):
        actual = select_conversation_segments(raw, ctx=fast_ctx, example_idx=6)
    assert reference_calls == [1]
    _assert_seg_features_equal(actual, expected)
    _assert_rng_states_equal(fast_ctx, expected_ctx)
    # The fallback is a silent-correctness path: it must announce itself, or a
    # production run could degrade to the slow loop on every conversation with
    # nothing in the logs to say so.
    assert any(
        "fell back to the reference segment loop" in record.message
        and "RuntimeError: forced pure-prefix failure" in record.message
        for record in caplog.records
    ), [record.message for record in caplog.records]


def _ranks_call_sizes(
    monkeypatch: pytest.MonkeyPatch,
    fn: Any,
    raw: list[SegmentFeatures],
    ctx: SegmentSelectionContext,
) -> list[int]:
    """Record the population size of every ``ranks`` call ``fn`` makes."""
    sizes: list[int] = []
    real_ranks = MissingPercentileResolver.ranks

    def record(
        self: MissingPercentileResolver,
        feature_ids: t.Tensor,
        feature_mags: t.Tensor,
        percentile_table: t.Tensor,
    ) -> t.Tensor:
        sizes.append(int(feature_ids.numel()))
        return real_ranks(self, feature_ids, feature_mags, percentile_table)

    with monkeypatch.context() as patched:
        patched.setattr(MissingPercentileResolver, "ranks", record)
        fn(raw, ctx=ctx, example_idx=8)
    return sizes


@pytest.mark.parametrize(
    "case",
    ("threshold_masked_fp", "clustered_threshold", "sampling_lowest_own_table_negatives"),
)
def test_pure_percentile_phases_are_batched_once(
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-vacuity: the fast path really collapses the per-segment rank calls.

    Without this the equality tests would still pass against a fast path that
    silently degraded to the reference for every conversation.
    """
    raw = [
        SegmentFeatures(
            segment_idx=i,
            position=i * 128,
            feature_ids=t.arange(40, dtype=t.int32) + i * 40,
            feature_mags=t.arange(40, dtype=t.float32) + 1.0,
            char_start=i * 8,
            char_end=(i + 1) * 8,
        )
        for i in range(4)
    ]
    fast_sizes = _ranks_call_sizes(
        monkeypatch,
        select_conversation_segments,
        raw,
        _context(case, seed=17),
    )
    reference_sizes = _ranks_call_sizes(
        monkeypatch,
        select_conversation_segments_reference,
        raw,
        _context(case, seed=17),
    )
    assert reference_sizes
    assert len(fast_sizes) < len(reference_sizes)
    # Same total population, redistributed into whole-conversation calls.
    assert sum(fast_sizes) == sum(reference_sizes)
    assert max(fast_sizes) == 160


def test_error_policy_sampling_keeps_override_weights_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The frozen ``sample_features`` argument shape, which goldens cannot see.

    Under the ``error`` policy with a sampling table that is not the FP table,
    today's code passes ``override_weights=None`` and lets ``sample_features``
    compute the ranking itself. Precomputing it in the batched prefix would be
    numerically identical, so no digest would move — but it is a frozen call
    shape, so pin it directly on both paths.
    """
    raw = [
        SegmentFeatures(
            segment_idx=i,
            position=i * 128,
            feature_ids=t.arange(12, dtype=t.int32) + i * 12,
            feature_mags=t.arange(12, dtype=t.float32) + 1.0,
            char_start=i * 8,
            char_end=(i + 1) * 8,
        )
        for i in range(3)
    ]

    def make_ctx() -> SegmentSelectionContext:
        return SegmentSelectionContext(
            threshold_cfg=None,
            precomputed_thresholds=None,
            sampling_bundle=_sampling(seed=5, table=_table(offset=1.0)),
            negative_bundle=None,
            boundaries=_boundaries(),
            fp_percentile_table=_table(),
            cluster_threshold_percentile_table=None,
            feature_mask=None,
            consolidation=None,
            missing_percentile=MissingPercentileResolver.error(),
        )

    seen: dict[str, list[bool]] = {"fast": [], "reference": []}
    real_sample_features = sft_selection.sample_features

    def record(label: str) -> Any:
        def _inner(*args: Any, **kwargs: Any) -> tuple[t.Tensor, t.Tensor]:
            seen[label].append(kwargs.get("override_weights") is not None)
            return real_sample_features(*args, **kwargs)

        return _inner

    for label, fn in (
        ("fast", select_conversation_segments),
        ("reference", select_conversation_segments_reference),
    ):
        with monkeypatch.context() as patched:
            patched.setattr(sft_selection, "sample_features", record(label))
            fn(raw, ctx=make_ctx(), example_idx=9)

    assert seen["reference"] == [False, False, False]
    assert seen["fast"] == seen["reference"]


def test_batched_rank_results_remain_float64(monkeypatch: pytest.MonkeyPatch) -> None:
    dtypes: list[t.dtype] = []
    real_ranks = MissingPercentileResolver.ranks

    def record_ranks(
        self: MissingPercentileResolver,
        feature_ids: t.Tensor,
        feature_mags: t.Tensor,
        percentile_table: t.Tensor,
    ) -> t.Tensor:
        result = real_ranks(self, feature_ids, feature_mags, percentile_table)
        dtypes.append(result.dtype)
        return result

    monkeypatch.setattr(MissingPercentileResolver, "ranks", record_ranks)
    raw = [
        SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([1, 2, 3], dtype=t.int32),
            feature_mags=t.tensor([1.0, 2.0, 3.0], dtype=t.float32),
            char_start=0,
            char_end=8,
        )
    ]
    select_conversation_segments(
        raw,
        ctx=_context("clustered_threshold", seed=7),
        example_idx=7,
    )
    assert dtypes and all(dtype is t.float64 for dtype in dtypes)
