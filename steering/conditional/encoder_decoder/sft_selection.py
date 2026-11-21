"""Per-conversation feature selection for conditioned SFT."""

from __future__ import annotations

from dataclasses import dataclass

import torch as t
from loguru import logger

from steering.conditional.discretization import (
    discretize_activations,
    percentile_to_bucket,
)
from steering.conditional.encoder_decoder.dataloader import SegFeatures
from steering.conditional.feature_clusters import FeatureClusterConsolidation
from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.feature_mask import FeatureMask
from steering.conditional.feature_pipeline import append_negative_features
from steering.conditional.missing_percentile import MissingPercentileResolver
from steering.conditional.sampling import (
    NegativeBundle,
    SamplingBundle,
    sample_features,
    sample_inactive_features_batched,
)
from steering.conditional.thresholding import (
    ThresholdConfig,
    ThresholdType,
    apply_threshold,
    select_from_packed_percentiles,
)


@dataclass(frozen=True)
class SegmentSelectionContext:
    threshold_cfg: ThresholdConfig | None
    precomputed_thresholds: t.Tensor | None
    sampling_bundle: SamplingBundle | None
    negative_bundle: NegativeBundle | None
    boundaries: t.Tensor
    fp_percentile_table: t.Tensor | None
    cluster_threshold_percentile_table: t.Tensor | None
    feature_mask: FeatureMask | None
    consolidation: FeatureClusterConsolidation | None
    missing_percentile: MissingPercentileResolver


@dataclass(frozen=True)
class _SegmentInputs:
    """One segment's pure, pre-RNG inputs to the selection loop.

    Produced either segment by segment (:func:`_segment_inputs`, the reference
    order) or for the whole conversation at once (:func:`_batched_inputs`).
    The two are exactly equal: every phase here is elementwise, so concatenating
    and splitting cannot change a value.
    """

    all_fids: t.Tensor | None
    all_pcts: t.Tensor | None
    selection_fids: t.Tensor
    selection_mags: t.Tensor
    selection_weights: t.Tensor | None
    precomputed_ranks: t.Tensor | None

    def ranks(
        self,
        resolver: MissingPercentileResolver,
        percentile_table: t.Tensor,
    ) -> t.Tensor:
        """The batched rank slice when the prefix produced one, else the live call.

        ``precomputed_ranks`` is ``None`` for every segment the batched prefix
        skipped (and for every segment on the unbatched path), which resolves to
        exactly the call the reference makes at this point.
        """
        if self.precomputed_ranks is not None:
            return self.precomputed_ranks
        return resolver.ranks(self.selection_fids, self.selection_mags, percentile_table)


def _assert_unique_sft_fp_source_ids(
    feature_ids: t.Tensor,
    *,
    example_idx: int,
    segment_idx: int,
) -> None:
    """Certify the per-conversation substitute for packed-row uniqueness."""
    if feature_ids.numel() <= 1:
        return
    values, counts = t.unique(feature_ids.to(t.int64), return_counts=True)
    duplicate_values = values[counts > 1]
    if duplicate_values.numel() == 0:
        return
    duplicate = int(duplicate_values[0].item())
    raise ValueError(
        f"conditioned SFT example {example_idx} segment {segment_idx} contains "
        f"duplicate original SAE feature ID {duplicate}"
    )


def _mask_selection(
    feature_mask: FeatureMask | None,
    feature_ids: t.Tensor,
    feature_mags: t.Tensor,
    weights: t.Tensor | None,
) -> tuple[t.Tensor, t.Tensor, t.Tensor | None]:
    """``apply_to_positive`` in both of its frozen argument shapes."""
    if feature_mask is None:
        return feature_ids, feature_mags, weights
    if weights is None:
        masked_ids, masked_mags = feature_mask.apply_to_positive(feature_ids, feature_mags)
        return masked_ids, masked_mags, None
    masked_ids, masked_mags, masked_weights = feature_mask.apply_to_positive(
        feature_ids,
        feature_mags,
        weights,
    )
    return masked_ids, masked_mags, masked_weights


def _segment_inputs(
    seg: SegmentFeatures,
    *,
    ctx: SegmentSelectionContext,
    example_idx: int,
) -> _SegmentInputs:
    """One segment's pure prefix, in the reference's statement order."""
    all_fids: t.Tensor | None = None
    all_pcts: t.Tensor | None = None
    sampling_weights: t.Tensor | None = None
    if ctx.fp_percentile_table is not None and len(seg.feature_ids) > 0:
        full_fids = seg.feature_ids.to(t.int64)
        _assert_unique_sft_fp_source_ids(
            full_fids,
            example_idx=example_idx,
            segment_idx=seg.segment_idx,
        )
        # The resolved percentile rank is float64 in [0.5, 99.5]; normalise to
        # [0, 1] and clamp for numerical safety.  When stochastic sampling uses
        # the same table, pass these scores back into sample_features so we do
        # this lookup once.
        full_pct_scores = ctx.missing_percentile.ranks(
            full_fids,
            seg.feature_mags,
            ctx.fp_percentile_table,
        )
        all_fids = full_fids
        all_pcts = (full_pct_scores / 100.0).clamp(0.0, 1.0).to(t.float32)
        if (
            ctx.sampling_bundle is not None
            and ctx.sampling_bundle.percentile_table is ctx.fp_percentile_table
        ):
            sampling_weights = full_pct_scores

    selection_fids, selection_mags, selection_weights = _mask_selection(
        ctx.feature_mask,
        seg.feature_ids,
        seg.feature_mags,
        sampling_weights,
    )
    return _SegmentInputs(
        all_fids=all_fids,
        all_pcts=all_pcts,
        selection_fids=selection_fids,
        selection_mags=selection_mags,
        selection_weights=selection_weights,
        precomputed_ranks=None,
    )


def _batched_inputs(
    raw_segments: list[SegmentFeatures],
    *,
    ctx: SegmentSelectionContext,
    example_idx: int,
) -> list[_SegmentInputs]:
    """The whole conversation's pure prefix, in two batched percentile calls.

    Every phase is elementwise, so one call over the concatenated segments and a
    ``split`` back is exactly the per-segment result — with the rank population
    redistributed from one call per segment into one call per conversation.
    """
    sizes = [len(seg.feature_ids) for seg in raw_segments]
    all_fids_parts: list[t.Tensor | None] = [None] * len(raw_segments)
    all_pcts_parts: list[t.Tensor | None] = [None] * len(raw_segments)
    sampling_weight_parts: list[t.Tensor | None] = [None] * len(raw_segments)

    if ctx.fp_percentile_table is not None:
        full_fids_parts: list[t.Tensor] = []
        for seg in raw_segments:
            full_fids = seg.feature_ids.to(t.int64)
            if len(seg.feature_ids) > 0:
                _assert_unique_sft_fp_source_ids(
                    full_fids,
                    example_idx=example_idx,
                    segment_idx=seg.segment_idx,
                )
            full_fids_parts.append(full_fids)
        # Documented assumption: concatenating first can mask a per-segment
        # ids/mags length mismatch whose totals happen to coincide (e.g. seg A
        # short by k, seg B long by k). The reference would raise
        # ``magnitude_to_percentile``'s "feature_mags shape != feature_ids
        # shape" here; the batched call sees matched totals and the discrepancy
        # would instead surface later, inside the per-segment selection. This is
        # structurally unreachable through ``FeatureIndex``: every segment's ids
        # and mags come from ``split_with_sizes`` over the same offsets, so they
        # are equal by construction. Recorded rather than re-checked — a
        # per-segment length guard here would be defensive machinery for a state
        # the loader cannot produce.
        full_pct_scores = ctx.missing_percentile.ranks(
            t.cat(full_fids_parts),
            t.cat([seg.feature_mags for seg in raw_segments]),
            ctx.fp_percentile_table,
        )
        for segment_idx, (size, full_fids, scores) in enumerate(
            zip(sizes, full_fids_parts, full_pct_scores.split(sizes), strict=True)
        ):
            if size == 0:
                continue
            all_fids_parts[segment_idx] = full_fids
            all_pcts_parts[segment_idx] = (scores / 100.0).clamp(0.0, 1.0).to(t.float32)
            if (
                ctx.sampling_bundle is not None
                and ctx.sampling_bundle.percentile_table is ctx.fp_percentile_table
            ):
                sampling_weight_parts[segment_idx] = scores

    masked = [
        _mask_selection(
            ctx.feature_mask,
            seg.feature_ids,
            seg.feature_mags,
            sampling_weight_parts[segment_idx],
        )
        for segment_idx, seg in enumerate(raw_segments)
    ]

    # The reference resolves selection-percentile currency in exactly three
    # places; batch precisely those segments and nothing else.  The unclustered
    # sampling arm is gated on ``substitutes`` because without a substitution
    # policy the reference passes ``override_weights=None`` and lets
    # ``sample_features`` compute the ranking itself — an argument shape that is
    # frozen.
    rank_segment_indices = [
        segment_idx
        for segment_idx, (selection_fids, _mags, selection_weights) in enumerate(masked)
        if selection_fids.numel() > 0
        and selection_weights is None
        and (
            ctx.consolidation is not None
            or (ctx.sampling_bundle is not None and ctx.missing_percentile.substitutes)
        )
    ]
    rank_parts: list[t.Tensor | None] = [None] * len(raw_segments)
    if rank_segment_indices:
        if ctx.consolidation is not None and ctx.sampling_bundle is None:
            assert ctx.cluster_threshold_percentile_table is not None
            percentile_table = ctx.cluster_threshold_percentile_table
        else:
            assert ctx.sampling_bundle is not None
            percentile_table = ctx.sampling_bundle.percentile_table
        rank_sizes = [len(masked[i][0]) for i in rank_segment_indices]
        ranks = ctx.missing_percentile.ranks(
            t.cat([masked[i][0] for i in rank_segment_indices]),
            t.cat([masked[i][1] for i in rank_segment_indices]),
            percentile_table,
        )
        for segment_idx, segment_ranks in zip(
            rank_segment_indices,
            ranks.split(rank_sizes),
            strict=True,
        ):
            rank_parts[segment_idx] = segment_ranks

    return [
        _SegmentInputs(
            all_fids=all_fids_parts[segment_idx],
            all_pcts=all_pcts_parts[segment_idx],
            selection_fids=selection_fids,
            selection_mags=selection_mags,
            selection_weights=selection_weights,
            precomputed_ranks=rank_parts[segment_idx],
        )
        for segment_idx, (selection_fids, selection_mags, selection_weights) in enumerate(masked)
    ]


def _select_segments(
    raw_segments: list[SegmentFeatures],
    *,
    ctx: SegmentSelectionContext,
    example_idx: int,
    batched: list[_SegmentInputs] | None,
) -> list[SegFeatures]:
    """The per-segment selection loop, over batched or per-segment pure inputs.

    ``batched=None`` computes each segment's pure prefix immediately before that
    segment's selection, which is the original per-segment implementation
    statement for statement — including the order in which its errors are
    raised.  That is why it is the authoritative arbiter for the fast path.
    """
    threshold_cfg = ctx.threshold_cfg
    precomputed_thresholds = ctx.precomputed_thresholds
    sampling_bundle = ctx.sampling_bundle
    negative_bundle = ctx.negative_bundle
    boundaries = ctx.boundaries
    cluster_threshold_percentile_table = ctx.cluster_threshold_percentile_table
    feature_mask = ctx.feature_mask
    consolidation = ctx.consolidation
    missing_percentile_resolver = ctx.missing_percentile
    seg_features_list: list[SegFeatures] = []
    for segment_idx, seg in enumerate(raw_segments):
        prepared = (
            batched[segment_idx]
            if batched is not None
            else _segment_inputs(seg, ctx=ctx, example_idx=example_idx)
        )
        selection_fids = prepared.selection_fids
        selection_mags = prepared.selection_mags
        selection_weights = prepared.selection_weights
        if consolidation is not None:
            assert boundaries is not None
            if sampling_bundle is not None:
                selection_percentiles = (
                    selection_weights / 100.0
                    if selection_weights is not None
                    else prepared.ranks(
                        missing_percentile_resolver,
                        sampling_bundle.percentile_table,
                    )
                    / 100.0
                ).to(t.float32)
            else:
                assert cluster_threshold_percentile_table is not None
                selection_percentiles = (
                    prepared.ranks(
                        missing_percentile_resolver,
                        cluster_threshold_percentile_table,
                    )
                    / 100.0
                ).to(t.float32)
            representatives, consolidated_percentiles = consolidation.consolidate_segment(
                selection_fids,
                selection_percentiles,
            )
            if sampling_bundle is not None:
                fids, fmags = sample_features(
                    representatives,
                    consolidated_percentiles,
                    sampling_bundle.percentile_table,
                    sampling_bundle.selection,
                    sampling_bundle.rng,
                    override_weights=consolidated_percentiles,
                )
            else:
                assert threshold_cfg is not None
                if threshold_cfg.threshold_type in (
                    ThresholdType.feature_percentile,
                    ThresholdType.feature_percentile_segment_topk,
                ):
                    fids = select_from_packed_percentiles(
                        representatives,
                        consolidated_percentiles,
                        threshold_cfg.threshold_type,
                        threshold_cfg.threshold,
                    )
                    selected_idx = t.searchsorted(representatives, fids)
                    fmags = consolidated_percentiles[selected_idx]
                else:
                    fids, fmags = apply_threshold(
                        threshold_cfg,
                        representatives,
                        consolidated_percentiles,
                    )
            consolidation.assert_representatives_only(
                fids,
                surface="conditional_encdec_sft live selection",
            )
            # Negatives are drawn from everything ACTIVE in this segment's
            # candidate pool, not from the selected positives — the same
            # exclusion set the pretrain writers use (post-mask,
            # post-consolidation ids).
            negative_exclusion_ids = representatives
        elif sampling_bundle is not None:
            if selection_weights is None and missing_percentile_resolver.substitutes:
                # ``sample_features`` would otherwise call the percentile
                # primitive itself, bypassing the policy. The override it
                # accepts is exactly the ranking score it would have computed
                # (float64 percentile ranks), for BOTH weighted and
                # top_k_percentile.
                selection_weights = prepared.ranks(
                    missing_percentile_resolver,
                    sampling_bundle.percentile_table,
                )
            fids, fmags = sample_features(
                selection_fids,
                selection_mags,
                sampling_bundle.percentile_table,
                sampling_bundle.selection,
                sampling_bundle.rng,
                override_weights=selection_weights,
            )
            negative_exclusion_ids = selection_fids
        else:
            assert threshold_cfg is not None
            fids, fmags = apply_threshold(
                threshold_cfg,
                selection_fids,
                selection_mags,
                precomputed=precomputed_thresholds,
            )
            negative_exclusion_ids = selection_fids
        if feature_mask is not None:
            feature_mask.assert_no_masked(fids)
        assert boundaries is not None
        if consolidation is not None:
            buckets = percentile_to_bucket(
                fmags,
                boundaries.shape[1] + 1,
            )
        else:
            # ``discretize_activations`` searchsorts a positive magnitude
            # against an uncovered feature's all-zero boundaries row and returns
            # the TOP bucket; pin it. The clustered branch above needs no
            # equivalent: its buckets come from percentiles already pinned
            # upstream, and its ids are representatives rather than queried ids,
            # so pinning on them would misattribute a covered sibling's
            # percentile.
            buckets = missing_percentile_resolver.pin_buckets(
                fids,
                discretize_activations(fids, fmags, boundaries)
                if len(fids) > 0
                else t.tensor([], dtype=t.int64),
            )
        pcts = t.zeros_like(buckets, dtype=t.float32)
        if negative_bundle is not None:
            # Negative (inactive) features, reusing pretraining's sampler and
            # merge verbatim: same ``sample_inactive_features_batched``, same
            # ``append_negative_features`` the live pretrain dataloader calls,
            # and the same reserved inactive row ``effective_buckets``.
            # ``boundaries.shape[1] + 1`` IS ``pipeline.effective_buckets`` —
            # the boundaries tensor was built from it — and the clustered
            # positive path above already relies on that identity.
            neg_fids = sample_inactive_features_batched(
                [negative_exclusion_ids.to(t.int64)],
                negative_bundle.n_sae_features,
                negative_bundle.selection,
                negative_bundle.rng,
                feature_mask=negative_bundle.feature_mask,
            )[0]
            fids, buckets, pcts = append_negative_features(
                fids,
                buckets,
                pcts,
                neg_fids,
                effective_buckets=boundaries.shape[1] + 1,
            )
            # Re-assert on the COMBINED id set: a negative reaching a masked or
            # non-representative id is exactly the failure mode a cluster-blind
            # negative pool produces.
            if feature_mask is not None:
                feature_mask.assert_no_masked(fids)
            if consolidation is not None:
                consolidation.assert_representatives_only(
                    fids,
                    surface="conditional_encdec_sft live selection with negatives",
                )
        # Full-conversation shards keep source-text character coordinates,
        # matching the decoded nanochat training text written by SFT conversion.
        #
        # ``all_feature_ids`` / ``all_feature_pcts`` deliberately stay the
        # PRE-selection full set: they build fp_targets, which predict what is
        # actually active, and a sampled INACTIVE feature must never enter them.
        seg_features_list.append(
            SegFeatures(
                feature_ids=fids.to(t.int64),
                buckets=buckets,
                percentiles=pcts,
                char_start=seg.char_start,
                char_end=seg.char_end,
                all_feature_ids=prepared.all_fids,
                all_feature_pcts=prepared.all_pcts,
            )
        )
    return seg_features_list


def select_conversation_segments_reference(
    raw_segments: list[SegmentFeatures],
    *,
    ctx: SegmentSelectionContext,
    example_idx: int,
) -> list[SegFeatures]:
    """Select every segment with its pure prefix computed segment by segment."""
    return _select_segments(raw_segments, ctx=ctx, example_idx=example_idx, batched=None)


def select_conversation_segments(
    raw_segments: list[SegmentFeatures],
    *,
    ctx: SegmentSelectionContext,
    example_idx: int,
) -> list[SegFeatures]:
    """Select every segment, batching the conversation's pure percentile phases.

    Any failure in the batched prefix hands the whole conversation back to the
    reference, whose outcome — result or raise — is authoritative. No RNG has
    been consumed at that point, so the rerun starts from the identical
    generator state and reproduces the error content and ordering exactly.
    """
    if not any(len(seg.feature_ids) > 0 for seg in raw_segments):
        # ``t.cat`` of an all-empty conversation raises; the per-segment guards
        # already define its outputs.
        return select_conversation_segments_reference(
            raw_segments,
            ctx=ctx,
            example_idx=example_idx,
        )
    try:
        batched = _batched_inputs(raw_segments, ctx=ctx, example_idx=example_idx)
    except Exception as exc:  # noqa: BLE001 - reference path is authoritative
        logger.warning(
            f"conditioned SFT example {example_idx} batched percentile selection "
            f"fell back to the reference segment loop: {type(exc).__name__}: {exc}"
        )
        return select_conversation_segments_reference(
            raw_segments,
            ctx=ctx,
            example_idx=example_idx,
        )
    return _select_segments(raw_segments, ctx=ctx, example_idx=example_idx, batched=batched)
