"""Frozen-origin differential coverage for the live document loader.

The reference below is a source copy of
``_document_batches_with_features`` from origin/main at 0881ab37, immediately
before audit PR10's mechanical phase extraction. Keep it inline and independent
of the refactored production entry point: this test is intended to detect any
change to document payloads, cursor snapshots, or generator advancement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
import torch as t

from nanochat.dataset import DatasetSplit
from steering.conditional.encoder_decoder import dataloader as dl_mod
from steering.conditional.encoder_decoder.dataloader import (
    SegFeatures,
    encoder_decoder_data_loader_with_state,
)
from steering.conditional.feature_index import FeatureIndex
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
)
from steering.conditional.sampling import ThresholdSelection
from steering.conditional.thresholding import ThresholdType
from steering.feature_io import TensorKey
from steering.overlay_io import (
    OverlayKind,
    overlay_dir,
    overlay_shard_path,
    save_overlay_shard,
    write_manifest,
)
from tests.steering._splice_cf_fixtures import build_derived_env
from tests.steering.test_live_dataloader import (
    _N_SAE_FEATURES,
    _SEGMENT_SIZE,
    _build_delta_sampling_pipeline,
    _build_negative_bundle,
    _build_sampling_pipeline,
    _build_threshold_pipeline,
    _compute_token_boundaries_for_seg,
    _make_synthetic_data,
)
from tests.steering.test_live_dataloader_splice_cf import (
    _sampling_pipeline as _build_splice_sampling_pipeline,
)

_FROZEN_DOCUMENT_BATCHES_WITH_FEATURES_SOURCE = r'''
def _frozen_document_batches_with_features(
    split: DatasetSplit,
    resume_state_dict: dict | None,
    tokenizer_batch_size: int,
    tokenizer,
    feature_index: FeatureIndex,
    pipeline: FeaturePipelineConfig,
    segment_size: int,
    data_dir: str | None = None,
    max_shards: int | None = None,
    context_len: int | None = None,
):
    """Infinite iterator over tokenized documents with prepared encoder inputs.

    Handles DDP sharding (row groups striped across ranks) and approximate
    resume via pq_idx/rg_idx state dict.

    Args:
        split: Train or val split.
        resume_state_dict: Optional dict with pq_idx/rg_idx/epoch for resume.
        tokenizer_batch_size: Number of documents per yielded batch.
        tokenizer: Training tokenizer instance.
        feature_index: FeatureIndex for looking up segment features.
        pipeline: Feature pipeline config (selection, boundaries, negatives).
        segment_size: Extraction segment size in characters.
        data_dir: Optional override for dataset directory.
        max_shards: Optional limit on number of shards.
        context_len: When set and precomputed token boundaries exist, truncates
            documents to boundary coverage via truncate_to_boundary_coverage().
            Typically set to T+1 (row_capacity in the packing loop).

    Yields:
        ``(doc_list, cursor_state)`` where ``cursor_state`` contains
        ``pq_idx``, ``rg_idx``, ``epoch``, ``ddp_world_size``, and the
        cursor-aligned feature-pipeline RNG snapshot. ``doc_list`` is a list
        of (text_tokens, seg_features) tuples.
        seg_features is a list of SegFeatures per segment, or None if no features exist.
    """
    assert split in DatasetSplit, f"split must be one of {list(DatasetSplit)}, got {split}"

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    _validate_positive_world_size(ddp_rank, ddp_world_size)
    _validate_resume_world_size(
        resume_state_dict,
        ddp_world_size,
        context="live dataloader resume",
    )
    if resume_state_dict is not None:
        if "feature_pipeline_rng_state" not in resume_state_dict:
            logger.warning(
                "Live dataloader: legacy resume state has no cursor-aligned "
                "feature_pipeline_rng_state; continuing with the configured seed. "
                "Live resume remains approximate."
            )
        else:
            _restore_feature_pipeline_rng_state(
                pipeline,
                resume_state_dict["feature_pipeline_rng_state"],
                context="live dataloader resume",
            )

    splice_provenance_index: SpliceProvenanceIndex | None = None
    if (
        pipeline.delta_bundle is not None
        and pipeline.delta_bundle.config.mode is DeltaMode.splice_counterfactual
    ):
        cf_feature_dir = feature_index.feature_dir
        if cf_feature_dir is None:
            raise SpliceProvenanceError(
                "feature_delta.mode=splice_counterfactual requires a FeatureIndex "
                "with a feature_dir (a spliced derived directory)."
            )
        splice_provenance_index = SpliceProvenanceIndex(cf_feature_dir)

    parquet_paths = list_parquet_files(data_dir)
    assert len(parquet_paths) != 0, "No dataset parquet files found"
    # When max_shards is set, slice BEFORE splitting train/val so the val shard
    # is the last of the limited set (not the global last parquet).
    if max_shards is not None:
        parquet_paths = parquet_paths[:max_shards]
    parquet_paths = parquet_paths[:-1] if split == DatasetSplit.TRAIN else parquet_paths[-1:]
    if max_shards is not None and split == DatasetSplit.TRAIN:
        logger.info(
            f"Limiting training to first {len(parquet_paths)} shards (max_shards={max_shards})"
        )

    bos_token = tokenizer.get_bos_token_id()

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict else 1
    first_pass = True
    epoch = resume_epoch

    while True:
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            with t.profiler.record_function("live/open_parquet"):
                pf = pq.ParquetFile(filepath)
            shard_stem = Path(filepath).stem
            resumed_mid_shard = (
                first_pass and resume_state_dict is not None and pq_idx == resume_pq_idx
            )

            if splice_provenance_index is not None:
                n_sidecar = splice_provenance_index.shard_n_docs(shard_stem)
                shard_expected_cf_total = splice_provenance_index.shard_n_splice_positions(
                    shard_stem
                )
                n_rows = pf.metadata.num_rows
                if n_sidecar != n_rows:
                    raise SpliceProvenanceError(
                        f"{shard_stem}: sidecar n_docs={n_sidecar} != parquet "
                        f"rows={n_rows}. The data_dir does not pair with this "
                        f"derived feature_dir — pass data_dir=<feature_dir>/_parquets."
                    )
            else:
                shard_expected_cf_total = None
            shard_expected_cf_seen = 0
            shard_scored_cf_seen = 0

            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None
            else:
                rg_idx = ddp_rank

            while rg_idx < pf.num_row_groups:
                # Compute global doc offset for this row group (accounts for all
                # prior row groups, including ones skipped by DDP striping)
                global_doc_offset = sum(pf.metadata.row_group(i).num_rows for i in range(rg_idx))

                with t.profiler.record_function("live/read_parquet_row_group"):
                    rg = pf.read_row_group(rg_idx)
                    texts = rg.column("text").to_pylist()

                doc_list: list[tuple[list[int], list[SegFeatures] | None]] = []

                # ============================================================
                # Phase 1: Per-doc tokenization, feature lookup, delta filtering
                # Collects all filtered segments across all docs for row-group-
                # level batched percentile computation.
                # ============================================================

                # Per-doc results from phase 1
                doc_tokens_list: list[list[int]] = []
                doc_segments_list: list[list] = []  # raw SegmentFeatures per doc
                doc_filtered_list: list[
                    list[
                        tuple[
                            t.Tensor,
                            t.Tensor,
                            t.Tensor | None,
                            t.Tensor | None,
                            t.Tensor | None,
                            t.Tensor | None,
                        ]
                    ]
                ] = []
                doc_decoded_pcts: list[list[t.Tensor | None]] = []  # decoded percentiles per doc

                # Row-group-level accumulators for batched percentile call
                rg_all_fids_parts: list[t.Tensor] = []
                rg_all_mags_parts: list[t.Tensor] = []
                rg_all_pcts_parts: list[t.Tensor | None] = []
                rg_seg_sizes: list[int] = []
                rg_override_weights: list[t.Tensor | None] = []

                use_sampling = isinstance(pipeline.selection_config, SamplingBundle)
                has_precomputed = feature_index.has_precomputed_percentiles

                with t.profiler.record_function("live/tokenize"):
                    all_token_lists = tokenizer.encode(texts, prepend=bos_token)

                for local_idx, text in enumerate(texts):  # noqa: B007
                    doc_idx = global_doc_offset + local_idx
                    text_tokens = all_token_lists[local_idx]

                    with t.profiler.record_function("live/feature_lookup"):
                        segments = feature_index.lookup(shard_stem, doc_idx)
                    # Re-check after lookup so the shard is loaded and checksum validated
                    has_precomputed = feature_index.has_precomputed_percentiles
                    # lookup() raises KeyError (naming shard+doc) if the row is absent
                    # — corruption. No None to guard against (issue #521).

                    # Decode precomputed percentiles for all segments
                    decoded_pcts_list: list[t.Tensor | None] = []
                    for seg in segments:
                        if has_precomputed and seg.feature_percentiles is not None:
                            decoded_pcts_list.append(uint8_to_percentile(seg.feature_percentiles))
                        else:
                            decoded_pcts_list.append(None)

                    # Truncate to precomputed boundary coverage when available
                    has_boundaries = (
                        feature_index.has_precomputed_token_boundaries
                        and segments[0].token_start is not None
                    )
                    if has_boundaries and context_len is not None:
                        # Build lightweight SegFeatures for truncation (only needs
                        # char/token boundary fields)
                        raw_seg_features = [
                            SegFeatures(
                                feature_ids=seg.feature_ids,
                                buckets=t.zeros(len(seg.feature_ids), dtype=t.int64),
                                percentiles=t.zeros(len(seg.feature_ids), dtype=t.float32),
                                char_start=seg.char_start,
                                char_end=seg.char_end,
                                token_start=seg.token_start,
                                token_end=seg.token_end,
                            )
                            for seg in segments
                        ]
                        text_tokens = truncate_to_boundary_coverage(
                            text_tokens,
                            raw_seg_features,
                            context_len=context_len,
                            segment_size=segment_size,
                            doc_id=f"{shard_stem}:{doc_idx}",
                        )

                    doc_tokens_list.append(text_tokens)
                    doc_segments_list.append(segments)
                    doc_decoded_pcts.append(decoded_pcts_list)

                    # Delta filtering (per-doc, batched within doc).
                    # batch_results is None (no delta), or a per-segment list
                    # aligned with `segments` where a None slot means "no delta
                    # for this segment — apply absolute-percentile conditioning".
                    # prev_segment/cum_max never produce None slots;
                    # splice_counterfactual produces None only for non-splice segments.
                    has_delta = pipeline.delta_bundle is not None

                    batch_results: list | None = None
                    if (
                        has_delta
                        and pipeline.delta_bundle.config.mode is DeltaMode.splice_counterfactual
                    ):
                        assert splice_provenance_index is not None
                        n_sae = pipeline.delta_bundle.percentile_table.shape[0]
                        for s_idx, seg in enumerate(segments):
                            if len(seg.feature_ids) == 0:
                                raise SpliceProvenanceError(
                                    f"{shard_stem} doc {doc_idx} segment {s_idx}: "
                                    "zero feature ids under "
                                    "feature_delta.mode=splice_counterfactual — "
                                    "corrupt feature shard."
                                )
                        seg_refs, expected_cf_positions = _resolve_counterfactual_refs(
                            splice_provenance_index,
                            feature_index,
                            shard_stem,
                            doc_idx,
                            segments,
                            has_precomputed,
                        )
                        batch_results = compute_doc_counterfactual_deltas(
                            [seg.feature_ids for seg in segments],
                            [seg.feature_mags for seg in segments],
                            decoded_pcts_list,
                            seg_refs,
                            n_sae,
                            pipeline.delta_bundle.percentile_table,
                        )
                        shard_expected_cf_seen += len(expected_cf_positions)
                        shard_scored_cf_seen += _validate_counterfactual_delta_results(
                            shard_stem=shard_stem,
                            doc_idx=doc_idx,
                            expected_positions=expected_cf_positions,
                            seg_refs=seg_refs,
                            batch_results=batch_results,
                        )
                    elif has_delta:
                        assert pipeline.delta_bundle is not None
                        n_sae = pipeline.delta_bundle.percentile_table.shape[0]
                        batch_results = compute_doc_deltas(
                            [seg.feature_ids for seg in segments],
                            [seg.feature_mags for seg in segments],
                            decoded_pcts_list,
                            n_sae,
                            pipeline.delta_bundle.config.mode,
                            pipeline.delta_bundle.percentile_table,
                        )

                    filtered_segments: list[
                        tuple[
                            t.Tensor,
                            t.Tensor,
                            t.Tensor | None,
                            t.Tensor | None,
                            t.Tensor | None,
                            t.Tensor | None,
                        ]
                    ] = []
                    for s_idx, seg in enumerate(segments):
                        r = batch_results[s_idx] if batch_results is not None else None
                        if r is None:
                            # (4a) Non-delta positive filter: keep only unmasked features.
                            f_ids = seg.feature_ids
                            f_mags_seg = seg.feature_mags
                            f_pcts_seg = decoded_pcts_list[s_idx]
                            if pipeline.feature_mask is not None:
                                if f_pcts_seg is not None:
                                    f_ids, f_mags_seg, f_pcts_seg = (
                                        pipeline.feature_mask.apply_to_positive(
                                            f_ids, f_mags_seg, f_pcts_seg
                                        )
                                    )
                                else:
                                    f_ids, f_mags_seg = pipeline.feature_mask.apply_to_positive(
                                        f_ids, f_mags_seg
                                    )
                            filtered_segments.append((f_ids, f_mags_seg, None, None, None, None))
                            if use_sampling:
                                rg_all_fids_parts.append(f_ids)
                                rg_all_mags_parts.append(f_mags_seg)
                                rg_seg_sizes.append(len(f_ids))
                                rg_all_pcts_parts.append(f_pcts_seg)
                                rg_override_weights.append(None)
                        else:
                            f_fids, f_mags, d_vals, neg_fids, neg_w, ref_mask, f_pcts = r
                            # (4b) Delta positive filter: keep only unmasked positive features.
                            # neg_fids/neg_w/ref_mask are NOT co-indexed with f_fids — delegate
                            # to sample_delta_negatives (Task 9) via feature_mask forwarding.
                            if pipeline.feature_mask is not None:
                                k = pipeline.feature_mask.keep_mask(f_fids)
                                f_fids = f_fids[k]
                                f_mags = f_mags[k]
                                if d_vals is not None:
                                    d_vals = d_vals[k]
                                if f_pcts is not None:
                                    f_pcts = f_pcts[k]
                            filtered_segments.append(
                                (f_fids, f_mags, d_vals, neg_fids, neg_w, ref_mask)
                            )
                            if use_sampling:
                                rg_all_fids_parts.append(f_fids)
                                rg_all_mags_parts.append(f_mags)
                                rg_seg_sizes.append(len(f_fids))
                                rg_all_pcts_parts.append(f_pcts)
                                ow = None
                                if d_vals is not None and len(d_vals) > 0:
                                    dw = pipeline.selection_config.selection.delta_weights
                                    if dw is DeltaSamplingWeights.percentile_delta:
                                        ow = d_vals
                                rg_override_weights.append(ow)
                    doc_filtered_list.append(filtered_segments)

                # ============================================================
                # Phase 2: Row-group-level batched positive sampling
                # One magnitude_to_percentile call for ALL segments in the
                # row group (~38K features), then per-segment multinomial.
                # ============================================================
                if use_sampling:
                    rg_all_fids = (
                        t.cat(rg_all_fids_parts)
                        if rg_all_fids_parts
                        else t.tensor([], dtype=t.int64)
                    )
                    rg_all_mags = (
                        t.cat(rg_all_mags_parts)
                        if rg_all_mags_parts
                        else t.tensor([], dtype=t.float32)
                    )

                    # Build precomputed percentiles tensor if all segments have them
                    rg_precomputed_pcts: t.Tensor | None = None
                    if (
                        has_precomputed
                        and rg_all_pcts_parts
                        and all(p is not None for p in rg_all_pcts_parts)
                    ):
                        rg_precomputed_pcts = t.cat([p for p in rg_all_pcts_parts if p is not None])

                    rg_sampled_results = sample_features_batched(
                        rg_all_fids,
                        rg_all_mags,
                        rg_seg_sizes,
                        pipeline.selection_config.percentile_table,
                        pipeline.selection_config.selection,
                        pipeline.selection_config.rng,
                        override_weights_list=rg_override_weights,
                        precomputed_percentiles=rg_precomputed_pcts,
                    )
                # ============================================================
                # Phase 3 & 4: Per-doc discretization + negative sampling
                # Split the row-group-level sampled results back to per-doc,
                # then discretize and add negatives.
                # ============================================================
                rg_seg_offset = 0  # tracks position in rg_sampled_results

                for local_idx in range(len(texts)):
                    text_tokens = doc_tokens_list[local_idx]
                    segments = doc_segments_list[local_idx]
                    filtered_segments = doc_filtered_list[local_idx]
                    decoded_pcts_list = doc_decoded_pcts[local_idx]

                    n_doc_segs = len(segments)

                    # Get this doc's sampled results from the row-group batch
                    if use_sampling:
                        sampled_results = rg_sampled_results[
                            rg_seg_offset : rg_seg_offset + n_doc_segs
                        ]
                        rg_seg_offset += n_doc_segs
                    elif isinstance(pipeline.selection_config, ThresholdSelectionConfig):
                        threshold_cfg = ThresholdConfig(
                            pipeline.selection_config.selection.threshold_type,
                            pipeline.selection_config.selection.threshold,
                        )
                        sampled_results = []
                        for filtered_fids, filtered_fmags, _dv, _nd, _nw, _rf in filtered_segments:
                            fids, fmags = apply_threshold(
                                threshold_cfg,
                                filtered_fids,
                                filtered_fmags,
                                precomputed=pipeline.selection_config.precomputed_thresholds,
                            )
                            sampled_results.append((fids, fmags))
                    else:
                        raise TypeError(
                            f"Unsupported selection config: {type(pipeline.selection_config).__name__}"
                        )

                    # Batched discretization for this doc
                    disc_fids_parts = []
                    disc_mags_parts = []
                    disc_sizes = []
                    for fids, fmags in sampled_results:
                        disc_fids_parts.append(
                            fids if len(fids) > 0 else t.tensor([], dtype=t.int64)
                        )
                        disc_mags_parts.append(
                            fmags if len(fmags) > 0 else t.tensor([], dtype=t.float32)
                        )
                        disc_sizes.append(len(fids))
                    all_disc_fids = (
                        t.cat(disc_fids_parts)
                        if any(s > 0 for s in disc_sizes)
                        else t.tensor([], dtype=t.int64)
                    )
                    all_disc_mags = (
                        t.cat(disc_mags_parts)
                        if any(s > 0 for s in disc_sizes)
                        else t.tensor([], dtype=t.float32)
                    )

                    if len(all_disc_fids) > 0:
                        all_buckets = discretize_activations(
                            all_disc_fids, all_disc_mags, pipeline.boundaries
                        )
                        all_pcts = compute_percentiles(
                            all_disc_fids, all_disc_mags, pipeline.boundaries
                        )
                    else:
                        all_buckets = t.tensor([], dtype=t.int64)
                        all_pcts = t.tensor([], dtype=t.float32)

                    disc_offset = 0
                    per_seg_sampled: list[tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]] = []
                    for i, sz in enumerate(disc_sizes):
                        fids = disc_fids_parts[i]
                        if sz > 0:
                            buckets = all_buckets[disc_offset : disc_offset + sz]
                            pcts = all_pcts[disc_offset : disc_offset + sz]
                        else:
                            buckets = t.tensor([], dtype=t.int64)
                            pcts = t.tensor([], dtype=t.float32)
                        per_seg_sampled.append((fids, disc_mags_parts[i], buckets, pcts))
                        disc_offset += sz

                    # Negative sampling: batch non-delta segments, keep delta per-segment
                    # Pass 1: collect non-delta segments for batched negative sampling
                    nondelta_indices: list[int] = []
                    nondelta_active_ids: list[t.Tensor] = []
                    delta_neg_results: dict[int, t.Tensor] = {}

                    if pipeline.negative_bundle is not None:
                        neg_mask = negative_sampling_mask(pipeline)
                        for seg_i, seg in enumerate(segments):
                            _, _, _, seg_neg_delta_fids, seg_neg_delta_weights, seg_ref_mask = (
                                filtered_segments[seg_i]
                            )
                            if (
                                has_delta
                                and seg_neg_delta_fids is not None
                                and seg_ref_mask is not None
                            ):
                                # Delta path: per-segment (stateful, needs delta context)
                                assert pipeline.delta_bundle is not None
                                assert seg_neg_delta_weights is not None
                                fids = per_seg_sampled[seg_i][0]
                                neg_k = sample_k(
                                    pipeline.negative_bundle.selection.distribution,
                                    pipeline.negative_bundle.selection.mean,
                                    pipeline.negative_bundle.selection.std,
                                    pipeline.negative_bundle.rng,
                                    min_k=pipeline.negative_bundle.selection.min_k,
                                    max_k=pipeline.negative_bundle.selection.max_k,
                                )
                                active_ids = seg.feature_ids
                                # Clamp to the allowed-universe pool consistent with the
                                # non-delta path. None means full width.
                                _allowed = (
                                    neg_mask.allowed_universe_mask(
                                        pipeline.negative_bundle.n_sae_features
                                    )
                                    if neg_mask is not None
                                    else None
                                )
                                if _allowed is not None:
                                    if len(active_ids) > 0:
                                        _active_bits = _allowed.clone()
                                        _active_bits[active_ids.long()] = False
                                        _allowed_pool = int(_active_bits.sum().item())
                                    else:
                                        _allowed_pool = int(_allowed.sum().item())
                                    neg_k = max(0, min(neg_k, _allowed_pool))
                                else:
                                    neg_k = max(
                                        0,
                                        min(
                                            neg_k,
                                            pipeline.negative_bundle.n_sae_features
                                            - len(active_ids),
                                        ),
                                    )
                                delta_neg_results[seg_i] = sample_delta_negatives(
                                    k=neg_k,
                                    neg_delta_ratio=pipeline.delta_bundle.config.neg_delta_ratio,
                                    neg_delta_fids=seg_neg_delta_fids,
                                    neg_delta_weights=seg_neg_delta_weights,
                                    raw_feature_ids=active_ids,
                                    ref_mask=seg_ref_mask,
                                    selected_positive_ids=fids,
                                    n_sae_features=pipeline.negative_bundle.n_sae_features,
                                    rng=pipeline.negative_bundle.rng,
                                    feature_mask=neg_mask,
                                )
                            else:
                                # Non-delta path: collect for batched call.
                                nondelta_indices.append(seg_i)
                                nondelta_active_ids.append(seg.feature_ids)

                        # One batched call for all non-delta segments
                        if nondelta_indices:
                            batched_neg = sample_inactive_features_batched(
                                nondelta_active_ids,
                                pipeline.negative_bundle.n_sae_features,
                                pipeline.negative_bundle.selection,
                                pipeline.negative_bundle.rng,
                                feature_mask=neg_mask,
                            )
                            nondelta_neg_results = dict(
                                zip(nondelta_indices, batched_neg)  # noqa: B905
                            )
                        else:
                            nondelta_neg_results = {}

                    # Pass 2: merge negative results and build SegFeatures
                    seg_features: list[SegFeatures] = []
                    for seg_i, seg in enumerate(segments):
                        fids, fmags, buckets, pcts = per_seg_sampled[seg_i]

                        if pipeline.negative_bundle is not None:
                            neg_fids = delta_neg_results.get(seg_i)
                            if neg_fids is None:
                                neg_fids = nondelta_neg_results.get(seg_i)
                            if neg_fids is not None and len(neg_fids) > 0:
                                neg_buckets = t.full(
                                    (len(neg_fids),),
                                    pipeline.effective_buckets,
                                    dtype=t.int64,
                                )
                                fids = t.cat([fids.to(t.int64), neg_fids])
                                buckets = t.cat([buckets, neg_buckets])
                                pcts = t.cat([pcts, t.zeros(len(neg_fids), dtype=t.float32)])

                        # Defense-in-depth (H-assert): assert finalized per-segment
                        # ids (positives + negatives combined) contain no masked features.
                        # Cheap: boolean membership over fids (tens–hundreds of ids).
                        # Strictly mask-gated (I5: no-op when feature_mask is None).
                        if pipeline.feature_mask is not None:
                            pipeline.feature_mask.assert_no_masked(fids.to(t.int64))

                        # Store pre-selection features for fp_targets.
                        # Both must be set together (same length) or both None.
                        all_pcts_i = decoded_pcts_list[seg_i]
                        if all_pcts_i is not None:
                            all_fids_i = seg.feature_ids.to(t.int64)
                        else:
                            all_fids_i = None

                        seg_features.append(
                            SegFeatures(
                                feature_ids=fids.to(t.int64),
                                buckets=buckets,
                                percentiles=pcts,
                                char_start=seg.char_start,
                                char_end=seg.char_end,
                                token_start=seg.token_start,
                                token_end=seg.token_end,
                                all_feature_ids=all_fids_i,
                                all_feature_pcts=all_pcts_i,
                            )
                        )

                    doc_list.append((text_tokens, seg_features))

                # Postcondition: all sampled results consumed
                if use_sampling:
                    assert rg_seg_offset == len(rg_sampled_results), (
                        f"rg_seg_offset={rg_seg_offset} != len(rg_sampled_results)="
                        f"{len(rg_sampled_results)} — segment count mismatch between "
                        f"accumulation and consumption phases"
                    )

                # This snapshot is aligned to the row-group cursor: all feature
                # selection draws for this row group have completed, but no later
                # row group has started. The outer best-fit buffer is not serialized,
                # so restoring it improves continuity without making live resume exact.
                cursor_state = {
                    "pq_idx": pq_idx,
                    "rg_idx": rg_idx,
                    "epoch": epoch,
                    "ddp_world_size": ddp_world_size,
                    "feature_pipeline_rng_state": (_capture_feature_pipeline_rng_state(pipeline)),
                }

                # Yield in tokenizer_batch_size chunks
                for i in range(0, len(doc_list), tokenizer_batch_size):
                    yield (
                        doc_list[i : i + tokenizer_batch_size],
                        cursor_state,
                    )

                rg_idx += ddp_world_size
            if splice_provenance_index is not None:
                if shard_expected_cf_seen != shard_scored_cf_seen:
                    raise SpliceProvenanceError(
                        f"{shard_stem}: scored {shard_scored_cf_seen} "
                        f"counterfactual segment(s) but expected "
                        f"{shard_expected_cf_seen} in row groups processed by "
                        f"rank {ddp_rank}."
                    )
                assert shard_expected_cf_total is not None
                if (
                    ddp_world_size == 1
                    and not resumed_mid_shard
                    and shard_scored_cf_seen != shard_expected_cf_total
                ):
                    raise SpliceProvenanceError(
                        f"{shard_stem}: scored {shard_scored_cf_seen} "
                        f"counterfactual segment(s) but the sidecar records "
                        f"{shard_expected_cf_total} splice position(s)."
                    )
                if ddp_world_size == 1 and not resumed_mid_shard:
                    logger.info(
                        "{}: splice_counterfactual scored {}/{} reference "
                        "counterfactual segment(s)",
                        shard_stem,
                        shard_scored_cf_seen,
                        shard_expected_cf_total,
                    )
                else:
                    logger.info(
                        "{}: splice_counterfactual scored {} rank-local "
                        "reference counterfactual segment(s); sidecar shard "
                        "total={} (rank {}, world_size={})",
                        shard_stem,
                        shard_scored_cf_seen,
                        shard_expected_cf_total,
                        ddp_rank,
                        ddp_world_size,
                    )
            pq_idx += 1
        first_pass = False
        epoch += 1
'''

_FROZEN_NAMESPACE = vars(dl_mod).copy()
exec(
    compile(
        _FROZEN_DOCUMENT_BATCHES_WITH_FEATURES_SOURCE,
        "<frozen origin/main _document_batches_with_features>",
        "exec",
    ),
    _FROZEN_NAMESPACE,
)
_frozen_document_batches_with_features = _FROZEN_NAMESPACE[
    "_frozen_document_batches_with_features"
]


def _rewrite_row_groups(data_dir: str, layout: str) -> None:
    for parquet_path in sorted(Path(data_dir).glob("*.parquet")):
        table = pq.read_table(parquet_path)
        row_group_size = len(table) if layout == "single" else 2
        pq.write_table(table, parquet_path, row_group_size=row_group_size)


def _add_splice_token_boundaries(env: dict[str, Any]) -> None:
    checksum = "pr10_splice_token_boundaries"
    feature_index = FeatureIndex(env["feature_dir"])
    token_dir = overlay_dir(Path(env["feature_dir"]), OverlayKind.SEG_BOUNDARY, checksum)
    write_manifest(token_dir, checksum, OverlayKind.SEG_BOUNDARY)

    parquet_by_stem = {
        path.stem: path for path in Path(env["data_dir"]).glob("*.parquet")
    }
    tokenizer = env["tokenizer"]
    bos_token = tokenizer.get_bos_token_id()
    for shard_stem in feature_index.shard_stems:
        texts = pq.read_table(parquet_by_stem[shard_stem]).column("text").to_pylist()
        token_starts: list[int] = []
        token_ends: list[int] = []
        for doc_idx in feature_index.doc_ids_for_shard(shard_stem):
            segments = feature_index.lookup(shard_stem, doc_idx)
            text_tokens = tokenizer.encode(texts[doc_idx], prepend=bos_token)
            starts, ends = _compute_token_boundaries_for_seg(
                tokenizer,
                text_tokens,
                [segment.char_start for segment in segments],
                [segment.char_end for segment in segments],
            )
            token_starts.extend(starts)
            token_ends.extend(ends)
        save_overlay_shard(
            {
                TensorKey.SEGMENT_TOKEN_STARTS: t.tensor(token_starts, dtype=t.int32),
                TensorKey.SEGMENT_TOKEN_ENDS: t.tensor(token_ends, dtype=t.int32),
            },
            overlay_shard_path(token_dir, shard_stem),
        )

    env["tokenizer_checksum"] = checksum
    validating_index = FeatureIndex(env["feature_dir"], tokenizer_checksum=checksum)
    for shard_stem in validating_index.shard_stems:
        first_doc = validating_index.doc_ids_for_shard(shard_stem)[0]
        validating_index.lookup(shard_stem, first_doc)


@pytest.fixture(scope="module", params=("single", "multi"))
def parity_env(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory):
    layout = str(request.param)
    env = _make_synthetic_data(tmp_path_factory.mktemp(f"pr10-{layout}"))
    _rewrite_row_groups(env["data_dir"], layout)
    return layout, env


@pytest.fixture(scope="module", params=("single", "multi"))
def splice_parity_env(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
):
    layout = str(request.param)
    env = build_derived_env(
        tmp_path_factory.mktemp(f"pr10-splice-{layout}"),
        cut_fraction=0.6,
    )
    _add_splice_token_boundaries(env)
    _rewrite_row_groups(env["data_dir"], layout)
    return layout, env


def _build_no_features_pipeline() -> FeaturePipelineConfig:
    base = _build_threshold_pipeline()
    return FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.feature_percentile, 0.99),
            precomputed_thresholds=t.full(
                (_N_SAE_FEATURES,),
                float("inf"),
                dtype=t.float64,
            ),
        ),
        boundaries=base.boundaries,
        negative_bundle=None,
        effective_buckets=base.effective_buckets,
    )


def _build_case_pipeline(case: str) -> FeaturePipelineConfig:
    if case == "no_features":
        return _build_no_features_pipeline()
    if case in {"threshold", "label_dropout"}:
        return _build_threshold_pipeline(threshold=3)
    if case == "sampling_rng":
        return _build_sampling_pipeline(negative_bundle=_build_negative_bundle())
    if case == "delta":
        return _build_delta_sampling_pipeline(with_negatives=True)
    raise AssertionError(f"unknown parity case {case!r}")


def _make_feature_index(env: dict[str, Any], *, with_boundaries: bool) -> FeatureIndex:
    return FeatureIndex(
        env["feature_dir"],
        tokenizer_checksum=env["tokenizer_checksum"] if with_boundaries else None,
        percentile_checksum=env.get("percentile_checksum"),
    )


def _train_parquet(env: dict[str, Any]) -> Path:
    return sorted(Path(env["data_dir"]).glob("*.parquet"))[0]


def _take_document_stream(
    implementation,
    *,
    env: dict[str, Any],
    feature_index: FeatureIndex,
    pipeline: FeaturePipelineConfig,
    segment_size: int,
) -> list[tuple[list[tuple[list[int], list[SegFeatures] | None]], dict[str, Any]]]:
    row_groups = pq.ParquetFile(_train_parquet(env)).num_row_groups
    generator = implementation(
        split=DatasetSplit.TRAIN,
        resume_state_dict=None,
        tokenizer_batch_size=128,
        tokenizer=env["tokenizer"],
        feature_index=feature_index,
        pipeline=pipeline,
        segment_size=segment_size,
        data_dir=env["data_dir"],
        context_len=9,
    )
    return [next(generator) for _ in range(row_groups + 1)]


def _assert_tensor_bytes_equal(actual: t.Tensor, expected: t.Tensor) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.stride() == expected.stride()
    assert actual.device == expected.device
    assert actual.detach().cpu().contiguous().numpy().tobytes() == (
        expected.detach().cpu().contiguous().numpy().tobytes()
    )


def _assert_optional_tensor_bytes_equal(
    actual: t.Tensor | None,
    expected: t.Tensor | None,
) -> None:
    assert (actual is None) == (expected is None)
    if actual is not None and expected is not None:
        _assert_tensor_bytes_equal(actual, expected)


def _assert_seg_features_equal(actual: SegFeatures, expected: SegFeatures) -> None:
    _assert_tensor_bytes_equal(actual.feature_ids, expected.feature_ids)
    _assert_tensor_bytes_equal(actual.buckets, expected.buckets)
    _assert_tensor_bytes_equal(actual.percentiles, expected.percentiles)
    assert actual.char_start == expected.char_start
    assert actual.char_end == expected.char_end
    assert actual.token_start == expected.token_start
    assert actual.token_end == expected.token_end
    _assert_optional_tensor_bytes_equal(actual.all_feature_ids, expected.all_feature_ids)
    _assert_optional_tensor_bytes_equal(actual.all_feature_pcts, expected.all_feature_pcts)
    assert actual.prediction_target_only == expected.prediction_target_only


def _assert_document_stream_equal(
    actual_stream: list[
        tuple[list[tuple[list[int], list[SegFeatures] | None]], dict[str, Any]]
    ],
    expected_stream: list[
        tuple[list[tuple[list[int], list[SegFeatures] | None]], dict[str, Any]]
    ],
) -> None:
    assert len(actual_stream) == len(expected_stream)
    for (actual_docs, actual_cursor), (expected_docs, expected_cursor) in zip(
        actual_stream,
        expected_stream,
        strict=True,
    ):
        assert actual_cursor == expected_cursor
        assert len(actual_docs) == len(expected_docs)
        for (actual_tokens, actual_segments), (
            expected_tokens,
            expected_segments,
        ) in zip(actual_docs, expected_docs, strict=True):
            assert actual_tokens == expected_tokens
            assert (actual_segments is None) == (expected_segments is None)
            if actual_segments is not None and expected_segments is not None:
                assert len(actual_segments) == len(expected_segments)
                for actual_segment, expected_segment in zip(
                    actual_segments,
                    expected_segments,
                    strict=True,
                ):
                    _assert_seg_features_equal(actual_segment, expected_segment)


def _assert_layout_and_boundary_coverage(
    stream: list[
        tuple[list[tuple[list[int], list[SegFeatures] | None]], dict[str, Any]]
    ],
    *,
    layout: str,
    with_boundaries: bool,
) -> None:
    assert stream[0][1]["epoch"] == 1
    assert stream[-1][1]["epoch"] == 2
    assert all(item[1]["ddp_world_size"] == 1 for item in stream)
    if layout == "multi":
        assert len({item[1]["rg_idx"] for item in stream[:-1]}) > 1

    segments = [
        segment
        for docs, _cursor in stream
        for _tokens, doc_segments in docs
        for segment in (doc_segments or [])
    ]
    assert segments
    if with_boundaries:
        assert all(
            segment.token_start is not None and segment.token_end is not None
            for segment in segments
        )
    else:
        assert all(
            segment.token_start is None and segment.token_end is None
            for segment in segments
        )


def _clone_train_batch(batch: tuple) -> tuple:
    return tuple(value.clone() if isinstance(value, t.Tensor) else value for value in batch)


def _assert_train_batches_equal(actual: list[tuple], expected: list[tuple]) -> None:
    assert len(actual) == len(expected)
    for actual_batch, expected_batch in zip(actual, expected, strict=True):
        assert len(actual_batch) == len(expected_batch)
        for actual_value, expected_value in zip(
            actual_batch,
            expected_batch,
            strict=True,
        ):
            if isinstance(actual_value, t.Tensor):
                assert isinstance(expected_value, t.Tensor)
                _assert_tensor_bytes_equal(actual_value, expected_value)
            else:
                assert actual_value == expected_value


def _take_label_dropout_batches(
    env: dict[str, Any],
    *,
    with_boundaries: bool,
) -> tuple[list[tuple], list[dict[str, Any]], t.Tensor]:
    pipeline = _build_sampling_pipeline(negative_bundle=_build_negative_bundle())
    loader = encoder_decoder_data_loader_with_state(
        tokenizer=env["tokenizer"],
        B=1,
        T=12,
        split=DatasetSplit.TRAIN,
        feature_index=_make_feature_index(env, with_boundaries=with_boundaries),
        pipeline=pipeline,
        segment_size=_SEGMENT_SIZE,
        decoder_segment_size=None,
        device="cpu",
        tokenizer_batch_size=2,
        buffer_size=4,
        data_dir=env["data_dir"],
        prefetch_batches=0,
        label_dropout=0.5,
    )
    batches = [_clone_train_batch(next(loader)) for _ in range(3)]
    return (
        batches,
        dl_mod._capture_feature_pipeline_rng_state(pipeline),
        t.get_rng_state().clone(),
    )


@pytest.mark.parametrize(
    "case",
    ("no_features", "threshold", "sampling_rng", "label_dropout", "delta"),
)
@pytest.mark.parametrize("with_boundaries", (False, True), ids=("boundaries_off", "boundaries_on"))
def test_document_batches_match_frozen_origin(
    parity_env,
    case: str,
    with_boundaries: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, env = parity_env
    expected_pipeline = _build_case_pipeline(case)
    actual_pipeline = _build_case_pipeline(case)

    expected = _take_document_stream(
        _frozen_document_batches_with_features,
        env=env,
        feature_index=_make_feature_index(env, with_boundaries=with_boundaries),
        pipeline=expected_pipeline,
        segment_size=_SEGMENT_SIZE,
    )
    actual = _take_document_stream(
        dl_mod._document_batches_with_features,
        env=env,
        feature_index=_make_feature_index(env, with_boundaries=with_boundaries),
        pipeline=actual_pipeline,
        segment_size=_SEGMENT_SIZE,
    )

    _assert_document_stream_equal(actual, expected)
    assert dl_mod._capture_feature_pipeline_rng_state(actual_pipeline) == (
        dl_mod._capture_feature_pipeline_rng_state(expected_pipeline)
    )
    _assert_layout_and_boundary_coverage(
        actual,
        layout=layout,
        with_boundaries=with_boundaries,
    )
    if case == "no_features":
        assert all(
            segment.feature_ids.numel() == 0
            for docs, _cursor in actual
            for _tokens, segments in docs
            for segment in (segments or [])
        )

    if case == "label_dropout":
        original_default_rng = t.get_rng_state()
        try:
            t.manual_seed(78123)
            production_batches, production_pipeline_rng, production_default_rng = (
                _take_label_dropout_batches(env, with_boundaries=with_boundaries)
            )
            t.manual_seed(78123)
            monkeypatch.setattr(
                dl_mod,
                "_document_batches_with_features",
                _frozen_document_batches_with_features,
            )
            frozen_batches, frozen_pipeline_rng, frozen_default_rng = (
                _take_label_dropout_batches(env, with_boundaries=with_boundaries)
            )
            _assert_train_batches_equal(production_batches, frozen_batches)
            assert production_pipeline_rng == frozen_pipeline_rng
            _assert_tensor_bytes_equal(production_default_rng, frozen_default_rng)
        finally:
            t.set_rng_state(original_default_rng)


@pytest.mark.parametrize("with_boundaries", (False, True), ids=("boundaries_off", "boundaries_on"))
def test_splice_document_batches_match_frozen_origin(
    splice_parity_env,
    with_boundaries: bool,
) -> None:
    layout, env = splice_parity_env
    expected_pipeline = _build_splice_sampling_pipeline(
        mode=dl_mod.DeltaMode.splice_counterfactual,
        with_negatives=True,
    )
    actual_pipeline = _build_splice_sampling_pipeline(
        mode=dl_mod.DeltaMode.splice_counterfactual,
        with_negatives=True,
    )
    expected = _take_document_stream(
        _frozen_document_batches_with_features,
        env=env,
        feature_index=_make_feature_index(env, with_boundaries=with_boundaries),
        pipeline=expected_pipeline,
        segment_size=env["seg_size"],
    )
    actual = _take_document_stream(
        dl_mod._document_batches_with_features,
        env=env,
        feature_index=_make_feature_index(env, with_boundaries=with_boundaries),
        pipeline=actual_pipeline,
        segment_size=env["seg_size"],
    )

    _assert_document_stream_equal(actual, expected)
    assert dl_mod._capture_feature_pipeline_rng_state(actual_pipeline) == (
        dl_mod._capture_feature_pipeline_rng_state(expected_pipeline)
    )
    _assert_layout_and_boundary_coverage(
        actual,
        layout=layout,
        with_boundaries=with_boundaries,
    )
