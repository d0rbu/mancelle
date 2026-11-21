"""Validate eval context caches against exact evaluation token contexts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch as t

from errors import FeatureVocabError
from steering.conditional.checkpoint_feature_contract import (
    assert_feature_vocab_width,
)
from steering.conditional.encoder_decoder.eval_adapter import build_multi_segment_eval_plan
from steering.conditional.encoder_decoder.full_sequence_routing import (
    build_full_sequence_segment_routing,
    source_aligned_char_offsets,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.context_cache_build import (
    build_turn_cache_entries,
    turn_source_segments,
    validate_turn_source_shard,
)
from steering.eval.context_cache_io import (
    ContextCacheEntry,
    SafetensorsContextFeatureProvider,
)
from steering.eval.context_cache_io import (
    ContextCacheValidationError as _ContextCacheValidationError,
)
from steering.eval.context_enumeration import (
    EvalContextRecord,
    TargetRealRole,
    dedupe_context_records,
    turn_supervision_interval,
    validate_turn_context_records,
)
from steering.eval.feature_config import FeatureConfig
from steering.eval.pooling_op import PoolingOp
from steering.eval.target_conditioning import (
    LockstepTargetRealConfig,
    MultiTargetSegments,
    PooledTargetRealConfig,
    SingleTargetSegment,
    TargetConditioningMode,
    validate_feature_dict,
)
from steering.eval.target_real_dispatch import build_turn_target_for_real_config
from steering.eval.target_segments import pool_segments_to_dict
from steering.eval.turn_context_contract import TurnContextContract, validate_turn_context_contract
from steering.eval.turn_context_routing import prepare_turn_context
from steering.feature_io import MetadataKey, TensorKey, load_features
from steering.refined_types import PositiveInt
from steering.segmentation import (
    assign_tokens_to_segments,
    strict_segment_bounds_for_assignment,
    token_to_char_offsets,
    truncate_to_training_window,
    turn_segments_for_document,
)


@dataclass(frozen=True)
class ContextRoutingCheck:
    """Per-context routing summary for observed context and final target tokens."""

    key: str
    suite: str
    task: str | None
    token_count: int
    final_start: int
    context_segment_count: int
    context_feature_count: int
    context_assigned_token_count: int
    final_assigned_token_count: int
    target_conditioning_mode: TargetConditioningMode
    encoder_slot_count: int
    target_slot_index: int
    shifted_context_slot_count: int


@dataclass(frozen=True)
class ContextCacheValidationSummary:
    """Aggregate validation result for a context cache and eval context set."""

    expected_contexts: int
    unique_contexts: int
    cached_contexts: int
    extra_cached_contexts: int
    total_context_segments: int
    total_context_features: int
    target_feature_configs: int
    routing_checks: tuple[ContextRoutingCheck, ...]
    turn_target_checks: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "expected_contexts": self.expected_contexts,
            "unique_contexts": self.unique_contexts,
            "cached_contexts": self.cached_contexts,
            "extra_cached_contexts": self.extra_cached_contexts,
            "total_context_segments": self.total_context_segments,
            "total_context_features": self.total_context_features,
            "target_feature_configs": self.target_feature_configs,
            "routing_checks": [
                {
                    "key": check.key,
                    "suite": check.suite,
                    "task": check.task,
                    "token_count": check.token_count,
                    "final_start": check.final_start,
                    "context_segment_count": check.context_segment_count,
                    "context_feature_count": check.context_feature_count,
                    "context_assigned_token_count": check.context_assigned_token_count,
                    "final_assigned_token_count": check.final_assigned_token_count,
                    "target_conditioning_mode": check.target_conditioning_mode.value,
                    "encoder_slot_count": check.encoder_slot_count,
                    "target_slot_index": check.target_slot_index,
                    "shifted_context_slot_count": check.shifted_context_slot_count,
                }
                for check in self.routing_checks
            ],
        }
        # Fixed validation leaves no turn checks; keep its summary bytes unchanged.
        if self.turn_target_checks:
            summary["turn_target_checks"] = list(self.turn_target_checks)
        return summary


def infer_final_segment_start(record: EvalContextRecord) -> int:
    """Infer where the final conditioned target/generated segment starts."""

    token_count = len(record.token_ids)
    if token_count == 0:
        raise _ContextCacheValidationError(
            f"{record.suite}/{record.task}: eval context has zero tokens"
        )

    metadata = record.metadata
    if "target_start" in metadata:
        final_start = int(metadata["target_start"])
    elif "answer_position" in metadata:
        final_start = int(metadata["answer_position"])
    else:
        final_start = token_count - 1

    if final_start < 0 or final_start >= token_count:
        raise _ContextCacheValidationError(
            f"{record.suite}/{record.task}: final segment start {final_start} "
            f"is outside token range [0, {token_count})"
        )
    return final_start


def compare_context_record_sets(
    expected_records: list[EvalContextRecord],
    observed_records: list[EvalContextRecord],
    *,
    expected_label: str = "eval enumeration",
    observed_label: str = "contexts_file",
) -> None:
    """Require two context record sets to contain exactly the same token keys."""

    expected_keys = {record.key for record in dedupe_context_records(expected_records)}
    observed_keys = {record.key for record in dedupe_context_records(observed_records)}
    missing = sorted(expected_keys - observed_keys)
    extra = sorted(observed_keys - expected_keys)
    if missing or extra:
        details = []
        if missing:
            details.append(
                f"{observed_label} is missing {len(missing)} context(s) from "
                f"{expected_label}; first={missing[0]}"
            )
        if extra:
            details.append(
                f"{observed_label} contains {len(extra)} context(s) not in "
                f"{expected_label}; first={extra[0]}"
            )
        raise _ContextCacheValidationError("; ".join(details))


def validate_context_cache_records(
    *,
    records: list[EvalContextRecord],
    provider: SafetensorsContextFeatureProvider,
    tokenizer,
    target_feature_configs: list[FeatureConfig] | None = None,
    feature_vocab: FeatureVocab | None = None,
    require_nonempty_context_features: bool = True,
    require_nonempty_context_segments: bool = False,
    fail_on_extra_cached_contexts: bool = False,
    target_conditioning_mode: TargetConditioningMode | str = TargetConditioningMode.CURRENT,
) -> ContextCacheValidationSummary:
    """Validate cache rows and token routing for exact eval context records."""

    if provider.turn_contract is not None:
        raise _ContextCacheValidationError("turn cache validation requires an explicit checkpoint contract")
    if not records:
        raise _ContextCacheValidationError("No eval context records were provided")

    provider_feature_vocab = provider.expected_feature_vocab
    if (
        feature_vocab is not None
        and provider_feature_vocab is not None
        and feature_vocab != provider_feature_vocab
    ):
        raise FeatureVocabError(
            "explicit context-cache feature vocabulary disagrees with the "
            "provider's authoritative vocabulary: explicit "
            f"V={feature_vocab.size}, n_sae={feature_vocab.n_sae_features}, "
            f"fingerprint={feature_vocab.fingerprint()}; provider "
            f"V={provider_feature_vocab.size}, "
            f"n_sae={provider_feature_vocab.n_sae_features}, "
            f"fingerprint={provider_feature_vocab.fingerprint()}"
        )
    validation_feature_vocab = (
        provider_feature_vocab if feature_vocab is None else feature_vocab
    )
    target_conditioning_mode = _parse_target_conditioning_mode(target_conditioning_mode)
    target_feature_configs = list(target_feature_configs or [])
    n_sae_features = int(provider.manifest["n_sae_features"])
    num_buckets = int(provider.manifest["num_buckets"])
    assert_feature_vocab_width(
        validation_feature_vocab,
        n_sae_features,
        name="context cache feature vocabulary",
    )
    for cfg in target_feature_configs:
        validate_feature_dict(
            cfg.features,
            n_sae_features,
            num_buckets,
            name=f"target_feature_configs[{cfg.name}]",
            feature_vocab=validation_feature_vocab,
        )

    unique_records = dedupe_context_records(records)
    extra_cached_contexts = len(provider) - len(unique_records)
    if fail_on_extra_cached_contexts and extra_cached_contexts > 0:
        raise _ContextCacheValidationError(
            f"Cache contains {extra_cached_contexts} extra context row(s) not "
            "required by the selected eval configuration"
        )

    checks: list[ContextRoutingCheck] = []
    total_segments = 0
    total_features = 0
    missing: list[EvalContextRecord] = []
    empty_contexts: list[EvalContextRecord] = []
    empty_segments: list[tuple[EvalContextRecord, int]] = []

    for record in unique_records:
        segments = provider.get(record.token_ids)
        if segments is None:
            missing.append(record)
            continue
        if len(segments) == 0:
            empty_contexts.append(record)
            continue

        final_start = infer_final_segment_start(record)
        for seg_idx, segment in enumerate(segments):
            _validate_context_segment_features(
                record=record,
                segment=segment,
                seg_idx=seg_idx,
                n_sae_features=n_sae_features,
                num_buckets=num_buckets,
                feature_vocab=validation_feature_vocab,
            )
        feature_count = sum(len(segment.feature_ids) for segment in segments)
        if require_nonempty_context_features and feature_count == 0:
            empty_contexts.append(record)
        if require_nonempty_context_segments:
            for seg_idx, segment in enumerate(segments):
                if len(segment.feature_ids) == 0:
                    empty_segments.append((record, seg_idx))

        routing = _validate_record_routing(
            record=record,
            segments=segments,
            final_start=final_start,
            tokenizer=tokenizer,
            target_conditioning_mode=target_conditioning_mode,
        )
        total_segments += len(segments)
        total_features += feature_count
        checks.append(
            ContextRoutingCheck(
                key=record.key,
                suite=record.suite,
                task=record.task,
                token_count=len(record.token_ids),
                final_start=final_start,
                context_segment_count=len(segments),
                context_feature_count=feature_count,
                context_assigned_token_count=routing["context_assigned"],
                final_assigned_token_count=routing["final_assigned"],
                target_conditioning_mode=target_conditioning_mode,
                encoder_slot_count=routing["encoder_slot_count"],
                target_slot_index=routing["target_slot_index"],
                shifted_context_slot_count=routing["shifted_context_slots"],
            )
        )

    if missing:
        first = missing[0]
        raise _ContextCacheValidationError(
            f"Cache is missing {len(missing)} required eval context row(s); "
            f"first missing suite={first.suite!r} task={first.task!r} key={first.key}"
        )
    if empty_contexts:
        first = empty_contexts[0]
        raise _ContextCacheValidationError(
            f"{len(empty_contexts)} eval context row(s) have no real context "
            f"features; first suite={first.suite!r} task={first.task!r} key={first.key}"
        )
    if empty_segments:
        first_record, first_seg_idx = empty_segments[0]
        raise _ContextCacheValidationError(
            f"{len(empty_segments)} context segment(s) have no features; first "
            f"suite={first_record.suite!r} task={first_record.task!r} "
            f"segment={first_seg_idx} key={first_record.key}"
        )

    return ContextCacheValidationSummary(
        expected_contexts=len(records),
        unique_contexts=len(unique_records),
        cached_contexts=len(provider),
        extra_cached_contexts=max(0, extra_cached_contexts),
        total_context_segments=total_segments,
        total_context_features=total_features,
        target_feature_configs=len(target_feature_configs),
        routing_checks=tuple(checks),
    )


def _validate_record_routing(
    *,
    record: EvalContextRecord,
    segments,
    final_start: int,
    tokenizer,
    target_conditioning_mode: TargetConditioningMode | str = TargetConditioningMode.CURRENT,
) -> dict[str, int]:
    if len(segments) == 0:
        raise _ContextCacheValidationError(
            f"{record.suite}/{record.task}: cache row has no context segments"
        )

    token_ids = list(record.token_ids)
    assignments = _assign_context_tokens_to_segments(tokenizer, token_ids, segments)
    target_conditioning_mode = _parse_target_conditioning_mode(target_conditioning_mode)
    if target_conditioning_mode is TargetConditioningMode.NEXT:
        encoder_slot_count = len(segments)
        final_idx = len(segments) - 1
    else:
        encoder_slot_count = len(segments) + 1
        final_idx = len(segments)

    seg_map = assignments.clone()
    seg_map[final_start:] = final_idx

    observed_prefix = seg_map[:final_start]
    if target_conditioning_mode is TargetConditioningMode.CURRENT and bool(
        (observed_prefix >= final_idx).any().item()
    ):  # pragma: no cover
        raise _ContextCacheValidationError(
            f"{record.suite}/{record.task}: observed prefix token was assigned "
            "to the final target segment"
        )
    final_suffix = seg_map[final_start:]
    if not bool((final_suffix == final_idx).all().item()):  # pragma: no cover
        raise _ContextCacheValidationError(
            f"{record.suite}/{record.task}: final target/generated tokens were "
            "not all assigned to the final segment"
        )

    return {
        "context_assigned": int(observed_prefix.numel()),
        "final_assigned": int(final_suffix.numel()),
        "encoder_slot_count": encoder_slot_count,
        "target_slot_index": final_idx,
        "shifted_context_slots": (
            max(len(segments) - 1, 0)
            if target_conditioning_mode is TargetConditioningMode.NEXT
            else 0
        ),
    }


def _parse_target_conditioning_mode(
    mode: TargetConditioningMode | str,
) -> TargetConditioningMode:
    try:
        return TargetConditioningMode.parse(mode)
    except ValueError as exc:
        raise _ContextCacheValidationError(str(exc)) from exc


def _validate_context_segment_features(
    *,
    record: EvalContextRecord,
    segment,
    seg_idx: int,
    n_sae_features: int,
    num_buckets: int,
    feature_vocab: FeatureVocab | None,
) -> None:
    if len(segment.feature_ids) != len(segment.buckets):
        raise _ContextCacheValidationError(
            f"{record.suite}/{record.task}: context segment {seg_idx} has "
            f"{len(segment.feature_ids)} feature ids but "
            f"{len(segment.buckets)} buckets"
        )
    if len(set(segment.feature_ids)) != len(segment.feature_ids):
        raise _ContextCacheValidationError(
            f"{record.suite}/{record.task}: context segment {seg_idx} has "
            "duplicate feature ids"
        )
    validate_feature_dict(
        {
            int(fid): int(bucket)
            for fid, bucket in zip(segment.feature_ids, segment.buckets, strict=True)
        },
        n_sae_features,
        num_buckets,
        name=f"{record.suite}/{record.task} context segment {seg_idx}",
        feature_vocab=feature_vocab,
    )


def _assign_context_tokens_to_segments(tokenizer, token_ids: list[int], segments) -> t.Tensor:
    offsets = token_to_char_offsets(tokenizer, token_ids)
    token_starts = t.tensor(offsets[:-1], dtype=t.long)
    token_ends = t.tensor(offsets[1:], dtype=t.long)

    seg_starts = t.tensor([segment.char_start for segment in segments], dtype=t.long)
    seg_ends = t.tensor([segment.char_end for segment in segments], dtype=t.long)
    try:
        seg_starts, seg_ends = strict_segment_bounds_for_assignment(
            seg_char_starts=seg_starts,
            seg_char_ends=seg_ends,
            token_starts=token_starts,
            token_ends=token_ends,
        )
        assignments = assign_tokens_to_segments(
            token_starts,
            token_ends,
            seg_starts,
            seg_ends,
        )
    except ValueError as exc:
        raise _ContextCacheValidationError(
            "context token/segment character alignment failed: "
            f"{exc}"
        ) from exc
    return t.cummax(assignments, dim=0).values


def validate_turn_source_geometry(
    *, records: list[EvalContextRecord], feature_shard_paths: list[Path],
    contract: TurnContextContract, extraction_tokenizer, tokenizer,
) -> None:
    """Reproduce the unchanged extraction geometry without a Gemma model forward."""
    validate_turn_context_records(records, tokenizer)
    # Full-reference rows are extraction/target provenance, not decoder inputs.
    # Their paired completion prefix is the sequence admitted by the evaluator
    # and must fit the checkpoint context window; the reference may be longer.
    if any(
        len(record.token_ids) > contract.model_max_seq_len
        and record.metadata.get("target_real_role") != TargetRealRole.FULL
        for record in records
    ):
        raise _ContextCacheValidationError("turn context exceeds checkpoint model_max_seq_len")
    index = 0
    for path in sorted(feature_shard_paths):
        data = load_features(path)
        validate_turn_source_shard(data, contract, source=str(path))
        visited_segments = 0
        for local_doc in range(int(data[MetadataKey.N_DOCS])):
            if index >= len(records):
                raise _ContextCacheValidationError("turn source shards contain extra documents")
            record = records[index]
            kept_ids, full_count = truncate_to_training_window(
                record.context_text, extraction_tokenizer,
                contract.semantic["source_context_window"], None,
            )
            if int(data[TensorKey.DOC_TOKEN_COUNTS][local_doc]) != full_count:
                raise _ContextCacheValidationError(f"{path}: doc {local_doc} source token count disagrees")
            if record.turn_char_starts is None:
                raise ValueError("turn source geometry requires turn_char_starts")
            positions, starts, ends, _ = turn_segments_for_document(
                kept_ids, extraction_tokenizer, record.context_text, record.turn_char_starts,
                PositiveInt.parse(contract.lineage.source_segment_size),
            )
            raw = turn_source_segments(data, local_doc)
            visited_segments += len(raw)
            expected = [(position[0], start, end) for position, start, end in zip(positions, starts, ends, strict=True)]
            observed = [(segment.position, segment.char_start, segment.char_end) for segment in raw]
            if expected != observed:
                raise _ContextCacheValidationError(f"{path}: doc {local_doc} reconstructed turn geometry disagrees")
            # Also enforce SFT's coverage/end-marker tolerance before conversion writes a cache.
            routing = build_full_sequence_segment_routing(
                ids=record.token_ids, tokenizer=tokenizer,
                seg_features=([segment.feature_ids for segment in raw],
                              [t.zeros_like(segment.feature_ids) for segment in raw], starts, ends),
                source_bos_char_len=contract.semantic["source_bos_char_len"],
            )
            if bool((routing.seg_map <= 0).any()):
                raise _ContextCacheValidationError(f"{record.key}: source extraction leaves unrouted decoder tokens")
            index += 1
        if visited_segments != int(data[MetadataKey.N_SEGMENTS]):
            raise _ContextCacheValidationError(f"{path}: document traversal did not account for every source segment")
    if index != len(records):
        raise _ContextCacheValidationError("turn source shards are missing documents")


def validate_turn_context_cache_records(
    *, records: list[EvalContextRecord], provider: SafetensorsContextFeatureProvider,
    tokenizer, turn_contract: TurnContextContract, extraction_tokenizer,
    feature_shard_paths: list[Path],
    require_nonempty_context_features: bool = True,
    require_nonempty_context_segments: bool = False,
    target_conditioning_mode: TargetConditioningMode | str = TargetConditioningMode.CURRENT,
) -> ContextCacheValidationSummary:
    """Replay a schema-5 turn cache against its own sources and checkpoint.

    The contract owns the feature vocabulary. Reference rows additionally
    exercise target-real policies; harmfulness rows are prompt-only and still
    receive the complete source, selection, vocabulary, and routing checks.
    """
    target_conditioning_mode = _parse_target_conditioning_mode(target_conditioning_mode)
    if provider.turn_contract is None:
        raise _ContextCacheValidationError("turn validation cannot consume a fixed cache")
    validate_turn_context_contract(provider.turn_contract.semantic, turn_contract)
    validate_turn_source_geometry(records=records, feature_shard_paths=feature_shard_paths,
                                  contract=turn_contract, extraction_tokenizer=extraction_tokenizer,
                                  tokenizer=tokenizer)
    expected_entries = build_turn_cache_entries(
        context_records=records, feature_shard_paths=feature_shard_paths,
        turn_contract=turn_contract, tokenizer=tokenizer,
    )
    if len(provider) != len(records):
        raise _ContextCacheValidationError("turn cache must contain exactly the enumerated rows")
    segments_by_tokens = {}
    routing_by_tokens = {}
    checks = []
    target_checks = []
    total_features = 0
    total_segments = 0
    for record, expected_entry in zip(records, expected_entries, strict=True):
        segments = provider.get(record.token_ids)
        if segments is None or segments != expected_entry.segments:
            raise _ContextCacheValidationError(f"{record.key}: cached selection/raw spans disagree with source replay")
        feature_count = sum(len(segment.feature_ids) for segment in segments)
        if require_nonempty_context_features and feature_count == 0:
            raise _ContextCacheValidationError(f"{record.key}: no selected context features")
        for index, segment in enumerate(segments):
            _validate_context_segment_features(record=record, segment=segment, seg_idx=index,
                n_sae_features=turn_contract.lineage.n_sae_features, num_buckets=turn_contract.lineage.num_buckets,
                feature_vocab=turn_contract.feature_vocab)
            if require_nonempty_context_segments and not segment.feature_ids:
                raise _ContextCacheValidationError(f"{record.key}: empty selected context segment {index}")
        routing = build_full_sequence_segment_routing(
            ids=record.token_ids, tokenizer=tokenizer,
            seg_features=([t.tensor(segment.feature_ids, dtype=t.int64) for segment in segments],
                          [t.tensor(segment.buckets, dtype=t.int64) for segment in segments],
                          [segment.char_start for segment in segments], [segment.char_end for segment in segments]),
            source_bos_char_len=turn_contract.semantic["source_bos_char_len"],
        )
        if bool((routing.seg_map <= 0).any()):
            raise _ContextCacheValidationError(f"{record.key}: invalid routed source slot")
        segments_by_tokens[record.token_ids] = segments
        routing_by_tokens[record.token_ids] = routing.seg_map
        final_start = infer_final_segment_start(record)
        checks.append(ContextRoutingCheck(
            key=record.key, suite=record.suite, task=record.task, token_count=len(record.token_ids),
            final_start=final_start, context_segment_count=len(segments), context_feature_count=feature_count,
            context_assigned_token_count=final_start, final_assigned_token_count=len(record.token_ids) - final_start,
            target_conditioning_mode=target_conditioning_mode, encoder_slot_count=len(routing.per_seg_fids),
            target_slot_index=int(routing.seg_map[final_start]), shifted_context_slot_count=0,
        ))
        total_features += feature_count
        total_segments += len(segments)
    target_feature_configs = 0
    for record in records:
        if record.metadata.get("target_real_role") != TargetRealRole.FULL:
            continue
        body_start = record.metadata["target_start"]
        prefix_tokens = record.token_ids[:body_start]
        marker_position = body_start - 1
        body_end = record.metadata["target_char_end"]
        _, end_position = turn_supervision_interval(record.token_ids, tokenizer)
        source_offsets = source_aligned_char_offsets(tokenizer, record.token_ids,
            source_bos_char_len=turn_contract.semantic["source_bos_char_len"])
        start_char, end_char = source_offsets[marker_position], source_offsets[end_position]
        full_segments = segments_by_tokens[record.token_ids]
        full_routing = routing_by_tokens[record.token_ids]
        first_target = int(full_routing[marker_position]) - 1
        last_target = int(full_routing[end_position - 1]) - 1
        keep = int(full_routing[:marker_position].max())
        if any(segment.char_end <= start_char for segment in full_segments[keep:first_target]):
            raise _ContextCacheValidationError(
                f"{record.key}: unused turn source slot before target ends at or before the assistant marker")
        target_segments = full_segments[first_target:last_target + 1]
        targets = [build_turn_target_for_real_config(
            cfg=config, entry=ContextCacheEntry(record.token_ids, full_segments),
            target_char_start=record.metadata["target_char_start"], target_char_end=body_end,
            full_token_ids=record.token_ids, target_token_loss_mask=[0] * len(record.token_ids),
            tokenizer=tokenizer, context_token_ids=prefix_tokens, turn_contract=turn_contract,
        ) for config in (PooledTargetRealConfig("validate", Path("."), PoolingOp.BUCKET_MAX),
                         LockstepTargetRealConfig("validate", Path(".")))]
        pooled, lockstep = targets
        if not isinstance(pooled, SingleTargetSegment) or not isinstance(lockstep, MultiTargetSegments):
            raise ValueError("turn validation requires pooled and lockstep target objects")
        assignments = lockstep.token_assignments
        expected_assignments = full_routing[marker_position:end_position] - 1 - first_target
        expected_slots = [dict(zip(segment.feature_ids, segment.buckets, strict=True)) for segment in target_segments]
        if lockstep.segments != expected_slots or not t.equal(assignments, expected_assignments):
            raise _ContextCacheValidationError(f"{record.key}: lockstep target disagrees with full-sequence routing")
        routed = sorted({int(slot) - 1 for slot in full_routing[marker_position:end_position].tolist()})
        routed_segments = tuple(full_segments[slot] for slot in routed)
        if pooled.features != pool_segments_to_dict(routed_segments, op=PoolingOp.BUCKET_MAX):
            raise _ContextCacheValidationError(f"{record.key}: pooled target disagrees with full-sequence routing")
        prefix_segments = segments_by_tokens[prefix_tokens]
        prompt = prepare_turn_context(prefix_tokens, prefix_segments, tokenizer,
            source_bos_char_len=turn_contract.semantic["source_bos_char_len"])
        prompt_conditions = [dict(zip(segment.feature_ids, segment.buckets, strict=True)) for segment in prompt.segments]
        plans = []
        for conditions, next_offset in (([pooled.features], False), ([pooled.features], True), (lockstep.segments, False)):
            plan = build_multi_segment_eval_plan(prefix_segment_conditions=prompt_conditions,
                target_segment_conditions=conditions, next_segment_conditioning_offset=next_offset)
            plans.append({"encoder_slots": plan.encoder_segment_conditions, "generation_slot": plan.generation_slot})
        target_feature_configs = len(plans)
        target_checks.append({"key": record.key, "task": record.task, "target_start_pos": marker_position,
                              "answer_char_start": record.metadata["target_char_start"], "answer_char_end": body_end,
                              "conditioning_char_start": start_char, "conditioning_char_end": end_char,
                              "lockstep_assignments": assignments.tolist(), "lockstep_segments": lockstep.segments,
                              "pooled_features": pooled.features, "plans": plans})
    return ContextCacheValidationSummary(
        expected_contexts=len(records), unique_contexts=len(records), cached_contexts=len(provider),
        extra_cached_contexts=0, total_context_segments=total_segments, total_context_features=total_features,
        target_feature_configs=target_feature_configs, routing_checks=tuple(checks),
        turn_target_checks=tuple(target_checks),
    )
