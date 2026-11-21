"""Build eval context cache entries from extracted SAE feature shards."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch as t
from beartype import beartype

from errors import FeatureClustersError
from scripts.preprocess.precompute_token_boundaries import compute_doc_token_boundaries_from_tokens
from steering.conditional.cluster_segment_selection import (
    select_clustered_segment_from_magnitudes,
)
from steering.conditional.discretization import discretize_activations
from steering.conditional.encoder_decoder.full_sequence_routing import (
    build_full_sequence_segment_routing,
)
from steering.conditional.encoder_decoder.sft_selection import select_conversation_segments
from steering.conditional.feature_clusters import FeatureClusterConsolidation
from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
)
from steering.conditional.sampling import SamplingBundle, sample_features, sample_inactive_features
from steering.conditional.thresholding import (
    ThresholdConfig,
    apply_threshold,
)
from steering.eval.context_cache_io import ContextCacheEntry
from steering.eval.context_enumeration import EvalContextRecord, validate_turn_context_records
from steering.eval.segment_mode_guard import require_fixed_segment_mode
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.eval.turn_context_contract import TurnContextContract
from steering.feature_io import MetadataKey, TensorKey, load_features
from steering.refined_types import NonNegativeInt, PositiveInt
from steering.segmentation import SegmentMode, segment_mode_from_metadata, token_to_char_offsets
from steering.shard_validation import validate_shard_data


def _validate_context_conversion_contract(
    *,
    pipeline: FeaturePipelineConfig,
    n_sae_features: PositiveInt,
    num_buckets: PositiveInt,
    feature_clusters: FeatureClusterConsolidation | None,
    source: str,
) -> None:
    """Validate the shared shard-backed/in-memory conversion contract."""

    if pipeline.boundaries.shape[0] < n_sae_features:
        raise ValueError(
            f"{source}: pipeline boundaries cover {pipeline.boundaries.shape[0]} "
            f"features, but n_sae_features={n_sae_features}"
        )
    if pipeline.effective_buckets > num_buckets:
        raise ValueError(
            f"{source}: pipeline effective_buckets={pipeline.effective_buckets} "
            f"exceeds num_buckets={num_buckets}"
        )
    if feature_clusters is None:
        return
    if feature_clusters.n_sae_features != n_sae_features:
        raise FeatureClustersError(
            f"{source}: context-cache cluster SAE width does not match n_sae_features"
        )
    if feature_clusters.feature_mask != pipeline.feature_mask:
        raise FeatureClustersError(
            f"{source}: context-cache clusters were derived from a different FeatureMask"
        )
    if not isinstance(pipeline.selection_config, SamplingBundle):
        raise FeatureClustersError(
            f"{source}: clustered threshold context selection is unsupported; "
            "use the audited sampling policy"
        )


@beartype
def build_cache_entries_from_extracted_feature_shards(
    *,
    context_records: Sequence[EvalContextRecord],
    feature_shard_paths: Sequence[str | Path],
    pipeline: FeaturePipelineConfig,
    n_sae_features: PositiveInt,
    num_buckets: PositiveInt,
    tokenizer,
    source_bos_char_len: NonNegativeInt,
    feature_clusters: FeatureClusterConsolidation | None = None,
) -> list[ContextCacheEntry]:
    """Convert raw extracted feature shards into strict eval cache entries.

    The feature shards must be generated from the enumerated contexts in the
    same order: document ``i`` across the concatenated feature shards maps to
    ``context_records[i]``.

    Unclustered selection retains the standard magnitude-space path; clustered
    sampling uses the pipeline's per-feature percentile table directly.

    ``@beartype`` owns the unary SAE-width, bucket-count, and BOS-offset
    contracts. The body retains only relationships involving pipeline shapes,
    cluster identity, and extracted shard contents.

    Raises:
        BeartypeCallHintParamViolation: A scalar argument violates its refined
            type, including a non-positive width/count or negative BOS offset.
    """

    _validate_context_conversion_contract(
        pipeline=pipeline,
        n_sae_features=n_sae_features,
        num_buckets=num_buckets,
        feature_clusters=feature_clusters,
        source="context-cache conversion",
    )

    entries: list[ContextCacheEntry | None] = [None] * len(context_records)
    doc_offset = 0
    for shard_path in sorted(Path(p) for p in feature_shard_paths):
        data = load_features(shard_path)
        require_fixed_segment_mode(
            segment_mode_from_metadata(data, source=shard_path),
            surface="eval context-cache conversion",
            source=f"feature shard {shard_path}",
        )
        n_docs = int(data.get("n_docs", len(data[TensorKey.DOC_TOKEN_COUNTS])))
        for local_doc_id in range(n_docs):
            context_idx = doc_offset + local_doc_id
            if context_idx >= len(context_records):
                raise ValueError(
                    f"Feature shards contain more docs than context_records; "
                    f"first extra doc index={context_idx}"
                )
            record = context_records[context_idx]
            entries[context_idx] = _entry_from_shard_doc(
                record=record,
                data=data,
                local_doc_id=local_doc_id,
                pipeline=pipeline,
                n_sae_features=n_sae_features,
                num_buckets=num_buckets,
                tokenizer=tokenizer,
                source_bos_char_len=source_bos_char_len,
                feature_clusters=feature_clusters,
                source=str(shard_path),
            )
        doc_offset += n_docs

    if doc_offset != len(context_records):
        raise ValueError(
            f"Feature shards contain {doc_offset} docs but context_records has "
            f"{len(context_records)}"
        )
    return [entry for entry in entries if entry is not None]


def _entry_from_shard_doc(
    *,
    record: EvalContextRecord,
    data: dict,
    local_doc_id: int,
    pipeline: FeaturePipelineConfig,
    n_sae_features: PositiveInt,
    num_buckets: PositiveInt,
    tokenizer,
    source_bos_char_len: NonNegativeInt,
    source: str,
    feature_clusters: FeatureClusterConsolidation | None = None,
) -> ContextCacheEntry:
    doc_ids = data[TensorKey.DOC_IDS]
    seg_rows = t.nonzero(doc_ids == local_doc_id, as_tuple=False).flatten()
    if len(seg_rows) == 0:
        raise ValueError(f"{source}: doc {local_doc_id} has no extracted segments")

    seg_indices = data[TensorKey.SEGMENT_INDICES][seg_rows]
    order = t.argsort(seg_indices)
    seg_rows = seg_rows[order]

    raw_char_starts = [int(data[TensorKey.SEGMENT_CHAR_STARTS][row].item()) for row in seg_rows]
    raw_char_ends = [int(data[TensorKey.SEGMENT_CHAR_ENDS][row].item()) for row in seg_rows]

    offsets = data[TensorKey.OFFSETS]
    all_feature_ids = data[TensorKey.FEATURE_IDS]
    all_feature_mags = data[TensorKey.FEATURE_MAGS]
    segment_feature_ids: list[t.Tensor] = []
    segment_feature_magnitudes: list[t.Tensor] = []
    for row in seg_rows:
        start = int(offsets[row].item())
        end = int(offsets[row + 1].item())
        segment_feature_ids.append(all_feature_ids[start:end].to(t.int64))
        segment_feature_magnitudes.append(all_feature_mags[start:end].to(t.float32))

    return build_context_entry_from_extracted_segments(
        token_ids=record.token_ids,
        segment_feature_ids=segment_feature_ids,
        segment_feature_magnitudes=segment_feature_magnitudes,
        segment_char_starts=raw_char_starts,
        segment_char_ends=raw_char_ends,
        pipeline=pipeline,
        n_sae_features=n_sae_features,
        num_buckets=num_buckets,
        tokenizer=tokenizer,
        source_bos_char_len=source_bos_char_len,
        feature_clusters=feature_clusters,
        source=f"{source}: doc {local_doc_id}",
    )


@beartype
def build_context_entry_from_selected_segments(
    *,
    token_ids: Sequence[int],
    segment_feature_ids: Sequence[t.Tensor],
    segment_buckets: Sequence[t.Tensor],
    segment_char_starts: Sequence[int],
    segment_char_ends: Sequence[int],
    tokenizer,
    source_bos_char_len: NonNegativeInt,
    source: str,
) -> ContextCacheEntry:
    """Route already-selected source segments onto one exact nanochat entry.

    Token ownership comes from :func:`build_full_sequence_segment_routing`, the
    primitive conditioned SFT trains under, so a source window that no target
    token overlaps best -- a Gemma token straddling a chat boundary can leave a
    window like that -- owns nothing here exactly as it owns nothing in
    training.  The fixed-window builder below keeps the persistent-cache
    converter, which requires every window to own a token, because that is the
    contract its caches were built and validated under.
    """

    if len(segment_feature_ids) == 0:
        raise ValueError(f"{source}: selected context has no segments")
    for pos, (fids, buckets) in enumerate(zip(segment_feature_ids, segment_buckets, strict=True)):
        if fids.ndim != 1 or buckets.ndim != 1:
            raise ValueError(f"{source}: selected segment {pos} ids/buckets must be 1-D")
        if fids.numel() != buckets.numel():
            raise ValueError(
                f"{source}: selected segment {pos} ids/buckets lengths differ "
                f"({fids.numel()} vs {buckets.numel()})"
            )

    ids = [int(token_id) for token_id in token_ids]
    try:
        routing = build_full_sequence_segment_routing(
            ids=ids,
            tokenizer=tokenizer,
            seg_features=(
                list(segment_feature_ids),
                list(segment_buckets),
                [int(value) for value in segment_char_starts],
                [int(value) for value in segment_char_ends],
            ),
            source_bos_char_len=int(source_bos_char_len),
        )
    except ValueError as exc:
        raise ValueError(f"{source}: segment routing failed: {exc}") from exc
    token_offsets = token_to_char_offsets(tokenizer, ids)

    segments: list[ContextFeatureSegment] = []
    assignments = [-1] * len(ids)
    for slot in range(1, len(routing.per_seg_fids)):
        positions = (routing.seg_map == slot).nonzero().flatten().tolist()
        if not positions:
            continue
        char_start = int(token_offsets[positions[0]])
        char_end = int(token_offsets[positions[-1] + 1])
        if char_end <= char_start:
            raise ValueError(f"{source}: segment slot {slot} owns only zero-width tokens")
        for position in positions:
            assignments[position] = len(segments)
        segments.append(
            ContextFeatureSegment(
                feature_ids=tuple(int(fid) for fid in routing.per_seg_fids[slot].tolist()),
                buckets=tuple(int(bucket) for bucket in routing.per_seg_buckets[slot].tolist()),
                char_start=char_start,
                char_end=char_end,
            )
        )

    if -1 in assignments:
        raise ValueError(f"{source}: turn context has tokens without a feature window")
    return ContextCacheEntry.from_sequences(
        token_ids,
        segments,
        segment_mode=SegmentMode.turn_delimited_fixed_tokens,
        token_assignments=assignments,
    )


@beartype
def build_context_entry_from_extracted_segments(
    *,
    token_ids: Sequence[int],
    segment_feature_ids: Sequence[t.Tensor],
    segment_feature_magnitudes: Sequence[t.Tensor],
    segment_char_starts: Sequence[int],
    segment_char_ends: Sequence[int],
    pipeline: FeaturePipelineConfig,
    n_sae_features: PositiveInt,
    num_buckets: PositiveInt,
    tokenizer,
    source_bos_char_len: NonNegativeInt,
    feature_clusters: FeatureClusterConsolidation | None = None,
    source: str = "extracted context",
) -> ContextCacheEntry:
    """Convert already-extracted Gemma segments into one exact nanochat entry."""

    lengths = {
        len(segment_feature_ids),
        len(segment_feature_magnitudes),
        len(segment_char_starts),
        len(segment_char_ends),
    }
    if len(lengths) != 1:
        raise ValueError(
            f"{source}: extracted segment ids, magnitudes, char starts, and "
            "char ends must have equal lengths"
        )
    n_segments = len(segment_feature_ids)
    if n_segments == 0:
        raise ValueError(f"{source}: extracted context has no segments")
    _validate_context_conversion_contract(
        pipeline=pipeline,
        n_sae_features=n_sae_features,
        num_buckets=num_buckets,
        feature_clusters=feature_clusters,
        source=source,
    )

    try:
        token_starts, token_ends = compute_doc_token_boundaries_from_tokens(
            [int(token_id) for token_id in token_ids],
            [int(value) for value in segment_char_starts],
            [int(value) for value in segment_char_ends],
            tokenizer,
            source_bos_char_len=source_bos_char_len,
        )
    except ValueError as exc:
        raise ValueError(f"{source}: token-boundary alignment failed: {exc}") from exc
    token_offsets = token_to_char_offsets(
        tokenizer,
        [int(token_id) for token_id in token_ids],
    )

    segments: list[ContextFeatureSegment] = []
    for pos, (raw_fids, raw_mags) in enumerate(
        zip(segment_feature_ids, segment_feature_magnitudes, strict=True)
    ):
        fids, buckets = _select_and_bucket_segment(
            raw_fids.to(t.int64),
            raw_mags.to(t.float32),
            pipeline=pipeline,
            n_sae_features=n_sae_features,
            num_buckets=num_buckets,
            feature_clusters=feature_clusters,
            source=f"{source}.segment[{pos}]",
        )

        tok_start = int(token_starts[pos].item())
        tok_end = int(token_ends[pos].item())
        if tok_end <= tok_start:
            raise ValueError(
                f"{source}: segment {pos} maps to an empty token interval [{tok_start}, {tok_end})"
            )
        if tok_start < 0 or tok_end > len(token_ids):
            raise ValueError(
                f"{source}: segment {pos} token interval "
                f"[{tok_start}, {tok_end}) outside context length "
                f"{len(token_ids)}"
            )
        segments.append(
            ContextFeatureSegment(
                feature_ids=tuple(int(fid) for fid in fids.tolist()),
                buckets=tuple(int(bucket) for bucket in buckets.tolist()),
                char_start=int(token_offsets[tok_start]),
                char_end=int(token_offsets[tok_end]),
            )
        )

    return ContextCacheEntry.from_sequences(token_ids, segments)


def _validate_buckets_and_dups(
    fids: t.Tensor, buckets: t.Tensor, num_buckets: PositiveInt, source: str
) -> None:
    """Shared output validation for the positive+negative (fids, buckets)."""
    if buckets.numel() and int(buckets.max().item()) >= num_buckets:
        raise ValueError(f"{source}: bucket exceeds num_buckets")
    if buckets.numel() and int(buckets.min().item()) < 0:
        raise ValueError(f"{source}: bucket is negative")
    if len({int(fid) for fid in fids.tolist()}) != int(fids.numel()):
        raise ValueError(f"{source}: duplicate selected feature ids")


def _select_and_bucket_segment(
    raw_fids: t.Tensor,
    raw_mags: t.Tensor,
    *,
    pipeline: FeaturePipelineConfig,
    n_sae_features: PositiveInt,
    num_buckets: PositiveInt,
    source: str,
    feature_clusters: FeatureClusterConsolidation | None = None,
) -> tuple[t.Tensor, t.Tensor]:
    if raw_fids.numel() != raw_mags.numel():
        raise ValueError(
            f"{source}: raw feature_ids and feature_mags lengths differ "
            f"({raw_fids.numel()} vs {raw_mags.numel()})"
        )
    if raw_fids.numel() and int(raw_fids.max().item()) >= n_sae_features:
        raise ValueError(f"{source}: raw feature id exceeds n_sae_features")
    if raw_fids.numel() and int(raw_fids.min().item()) < 0:
        raise ValueError(f"{source}: raw feature id is negative")

    if feature_clusters is not None:
        # D-dedup (PR #739 Track D): delegates to the merged A5 primitive
        # (steering/conditional/cluster_segment_selection.py) instead of a
        # second hand-rolled clustered selection sequence. This is a NAMED
        # BUG-FIX, not a pure dedup -- the empirical differential in
        # tests/steering/test_d_dedup_clustered_selection_differential.py
        # (Cells A-G) found the pre-existing inline body had drifted from
        # what training's real preselection
        # (scripts/preprocess/precompute_selected_features.py:1244) actually
        # does in four independent ways (raw-float vs uint8 percentiles;
        # t.multinomial vs Gumbel-max weighted sampling; percentile computed
        # after vs before masking; adaptive vs fixed negative-oversample
        # factor) plus a fifth: negatives were previously drawn even for an
        # empty positive candidate set. All five resolve to A5's behavior,
        # which is training currency.
        #
        # Preserved locally (A5 cannot make these checks itself): the
        # SamplingBundle-only scope boundary -- A5 also supports
        # ThresholdSelectionConfig, but clustered *threshold* context
        # selection stays a deliberately unsupported surface here (see
        # test_clustered_threshold_context_selection_is_explicitly_
        # unsupported); the raw-magnitude finiteness check -- A5 never sees
        # raw magnitudes, only pre-encoded uint8 percentiles, so this is the
        # only remaining place NaN/Inf raw activations can be caught; and the
        # 1-D shape check -- without it, a malformed input would fail deep
        # inside magnitude_to_percentile with a confusing shape-mismatch
        # RuntimeError instead of a clean FeatureClustersError, since that
        # call happens before A5 gets a chance to validate. The local
        # duplicate-raw-id check was DELETED: A5's own _validate_inputs
        # raises the same FeatureClustersError for it, making the local copy
        # redundant.
        if not isinstance(pipeline.selection_config, SamplingBundle):
            raise FeatureClustersError(
                "clustered threshold context selection is unsupported; use the "
                "audited sampling policy"
            )
        if raw_fids.ndim != 1 or raw_mags.ndim != 1:
            raise FeatureClustersError(
                f"{source}: clustered raw IDs/magnitudes must be one-dimensional"
            )
        if not raw_mags.is_floating_point() or not bool(t.isfinite(raw_mags).all()):
            raise FeatureClustersError(
                f"{source}: clustered raw magnitudes must be finite floating-point values"
            )

        assert isinstance(pipeline.selection_config, SamplingBundle)
        result = select_clustered_segment_from_magnitudes(
            raw_fids,
            raw_mags,
            percentile_table=pipeline.selection_config.percentile_table,
            clusters=feature_clusters,
            pipeline=pipeline,
            source=source,
        )
        negative_buckets = t.full(
            (result.negative_ids.numel(),),
            pipeline.effective_buckets,
            dtype=t.int64,
        )
        fids = t.cat([result.positive_ids, result.negative_ids])
        buckets = t.cat([result.positive_buckets, negative_buckets])
        # A5's own _assert_result already re-validated (inside
        # select_clustered_segment, before returning `result`) that
        # positive_ids/negative_ids are representative-only, mutually
        # disjoint, and duplicate-free -- re-checking here would be dead
        # code duplicating that postcondition.
        _validate_buckets_and_dups(fids, buckets, num_buckets, source)
        return fids.to(t.int64), buckets.to(t.int64)

    # R3 fix: apply the feature_mask to positive candidates. This builder
    # previously ignored pipeline.feature_mask, leaking masked features into the
    # cache; negatives respect the mask universe below. No-mask ⇒ byte-identical.
    pos_fids, pos_mags = raw_fids, raw_mags
    if pipeline.feature_mask is not None:
        pos_fids, pos_mags = pipeline.feature_mask.apply_to_positive(raw_fids, raw_mags)

    if isinstance(pipeline.selection_config, ThresholdSelectionConfig):
        selection = pipeline.selection_config.selection
        threshold_cfg = ThresholdConfig(selection.threshold_type, selection.threshold)
        fids, mags = apply_threshold(
            threshold_cfg,
            pos_fids,
            pos_mags,
            precomputed=pipeline.selection_config.precomputed_thresholds,
        )
    elif isinstance(pipeline.selection_config, SamplingBundle):
        fids, mags = sample_features(
            pos_fids,
            pos_mags,
            pipeline.selection_config.percentile_table,
            pipeline.selection_config.selection,
            pipeline.selection_config.rng,
        )
    else:
        raise TypeError(
            f"Unsupported feature selection config: {type(pipeline.selection_config).__name__}"
        )

    if fids.numel() > 0:
        buckets = discretize_activations(fids, mags, pipeline.boundaries)
    else:
        buckets = t.tensor([], dtype=t.int64)

    if pipeline.negative_bundle is not None:
        negative_ids = sample_inactive_features(
            raw_fids,
            n_sae_features,
            pipeline.negative_bundle.selection,
            pipeline.negative_bundle.rng,
            feature_mask=pipeline.feature_mask,
        )
        if negative_ids.numel() > 0:
            negative_buckets = t.full(
                (negative_ids.numel(),),
                pipeline.effective_buckets,
                dtype=t.int64,
            )
            fids = t.cat([fids.to(t.int64), negative_ids.to(t.int64)])
            buckets = t.cat([buckets.to(t.int64), negative_buckets])

    # R3 defense-in-depth: no masked feature reached the (positive+negative) output.
    if pipeline.feature_mask is not None:
        pipeline.feature_mask.assert_no_masked(fids.to(t.int64))

    _validate_buckets_and_dups(fids, buckets, num_buckets, source)
    return fids.to(t.int64), buckets.to(t.int64)


def validate_turn_source_shard(data: dict, contract: TurnContextContract, *, source: str) -> None:
    """Validate every available source setting and the raw CSR geometry before selection."""
    lineage = contract.lineage
    expected = {
        MetadataKey.SEGMENT_MODE: SegmentMode.turn_delimited_fixed_tokens.value,
        MetadataKey.SEGMENT_SIZE: lineage.source_segment_size,
        MetadataKey.CONTEXT_WINDOW: contract.semantic["source_context_window"],
        MetadataKey.LAYER: lineage.layer,
        MetadataKey.POOLING: contract.semantic["pooling"],
        MetadataKey.CONTEXT_INDEPENDENT_FEATURES: True,
        MetadataKey.IGNORE_BOS_FEATURES: True,
    }
    if isinstance(lineage.l0, str):
        expected[MetadataKey.L0_TIER] = lineage.l0
    for field, value in expected.items():
        if data.get(field) != value:
            raise ValueError(f"{source}: turn shard {field}={data.get(field)!r}, expected {value!r}")
    result = validate_shard_data(data)
    if not result.valid:
        raise ValueError(f"{source}: invalid turn source shard: {result.violations}")
    doc_ids = data[TensorKey.DOC_IDS]
    n_docs = int(data[MetadataKey.N_DOCS])
    if bool(((doc_ids < 0) | (doc_ids >= n_docs)).any()):
        raise ValueError(f"{source}: source segment doc_id outside [0, {n_docs})")
    feature_ids = data[TensorKey.FEATURE_IDS]
    if feature_ids.numel() and bool(((feature_ids < 0) | (feature_ids >= lineage.n_sae_features)).any()):
        raise ValueError(f"{source}: raw feature ID outside checkpoint SAE width")
    mags = data[TensorKey.FEATURE_MAGS]
    if not bool(t.isfinite(mags).all()):
        raise ValueError(f"{source}: non-finite raw feature magnitudes")


def turn_source_segments(data: dict, local_doc_id: int) -> list[SegmentFeatures]:
    """Read one raw conversation in source segment order, without snapping its spans."""
    rows = t.nonzero(data[TensorKey.DOC_IDS] == local_doc_id, as_tuple=False).flatten()
    rows = rows[t.argsort(data[TensorKey.SEGMENT_INDICES][rows])]
    if not rows.numel() or data[TensorKey.SEGMENT_INDICES][rows].tolist() != list(range(len(rows))):
        raise ValueError(f"turn source doc {local_doc_id} has missing/duplicate segment indices")
    offsets = data[TensorKey.OFFSETS]
    segments = []
    for row in rows.tolist():
        start, end = int(offsets[row]), int(offsets[row + 1])
        segments.append(SegmentFeatures(
            segment_idx=int(data[TensorKey.SEGMENT_INDICES][row]),
            position=int(data[TensorKey.SEGMENT_POSITIONS][row]),
            feature_ids=data[TensorKey.FEATURE_IDS][start:end],
            feature_mags=data[TensorKey.FEATURE_MAGS][start:end],
            char_start=int(data[TensorKey.SEGMENT_CHAR_STARTS][row]),
            char_end=int(data[TensorKey.SEGMENT_CHAR_ENDS][row]),
        ))
    return segments


def build_turn_cache_entries(
    *, context_records: Sequence[EvalContextRecord], feature_shard_paths: Sequence[str | Path],
    turn_contract: TurnContextContract, tokenizer,
) -> list[ContextCacheEntry]:
    """Convert raw turn-delimited shards with the producer checkpoint's own selection.

    The contract owns SAE width, bucket count and source BOS geometry, and the
    checkpoint owns selection, so this entry point admits no pipeline, cluster
    or geometry argument that could disagree with the producer.
    """
    validate_turn_context_records(context_records, tokenizer)
    if len({record.key for record in context_records}) != len(context_records):
        raise ValueError("turn conversion requires deduplicated enumeration rows")
    selection_context = turn_contract.build_selection_context()
    entries: list[ContextCacheEntry] = []
    for path in sorted(Path(path) for path in feature_shard_paths):
        data = load_features(path)
        validate_turn_source_shard(data, turn_contract, source=str(path))
        visited_segments = 0
        for local_doc in range(int(data[MetadataKey.N_DOCS])):
            index = len(entries)
            if index >= len(context_records):
                raise ValueError("turn feature shards contain more docs than context_records")
            raw = turn_source_segments(data, local_doc)
            visited_segments += len(raw)
            selected = select_conversation_segments(
                raw, ctx=selection_context, example_idx=index,
            )
            segments = tuple(ContextFeatureSegment(
                feature_ids=tuple(int(fid) for fid in segment.feature_ids.tolist()),
                buckets=tuple(int(bucket) for bucket in segment.buckets.tolist()),
                char_start=segment.char_start, char_end=segment.char_end,
            ) for segment in selected)
            entries.append(ContextCacheEntry.from_sequences(
                context_records[index].token_ids, segments,
                segment_mode=SegmentMode.turn_delimited_fixed_tokens,
            ))
        if visited_segments != int(data[MetadataKey.N_SEGMENTS]):
            raise ValueError(f"{path}: document traversal did not account for every source segment")
    if len(entries) != len(context_records):
        raise ValueError(f"turn feature shards contain {len(entries)} docs but context_records has {len(context_records)}")
    return entries
