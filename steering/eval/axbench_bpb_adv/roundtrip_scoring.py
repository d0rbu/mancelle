"""Round-trip steering scoring for AxBench-BPB-Adv candidates."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import pyarrow as pa
import torch as t
from loguru import logger

from steering.conditional.feature_gcg.edit_primitives import token_sequence_digest
from steering.eval.axbench_base.config import GenerationConfig, LoraConfig
from steering.eval.axbench_base.generation_helpers import (
    _generate_samples_with_nanochat_tokens,
)
from steering.eval.axbench_base.method_registry import get_method_by_name
from steering.eval.axbench_base.model_detection import detect_target_model_metadata
from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.eval.axbench_base.synth.response_parser import get_extraction_tokenizer
from steering.eval.axbench_base.types import Concept
from steering.eval.axbench_bpb.bpb_scoring import _coefficient_from_str
from steering.eval.axbench_bpb.intervention import apply_intervention
from steering.eval.axbench_bpb.runtime_state import RuntimeState
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle
from steering.eval.axbench_bpb.types import Intervention, LoraParams, SteeringMode
from steering.eval.axbench_bpb_adv.artifact_identity import (
    ClusteredCandidateSnapshot,
    require_clustered_candidate_snapshot,
)
from steering.eval.axbench_bpb_adv.clustered_stage_source import (
    ClusteredStageSource,
    require_clustered_stage_source,
)
from steering.eval.axbench_bpb_adv.config import (
    AxBenchBpbAdvConfig,
    get_additive_direction_config,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.evaluation_population import (
    read_evaluation_population,
    require_table_in_evaluation_population,
)
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.roundtrip_checkpoint import (
    CheckpointFamily,
    RoundtripCheckpointStore,
    build_roundtrip_checkpoint_fingerprint,
    candidates_payload_sha256,
    current_runtime_fingerprint,
    rows_bits_equal,
    warn_on_roundtrip_runtime_mismatch,
)
from steering.eval.axbench_bpb_adv.schemas import (
    CANDIDATES_SCHEMA,
    ROUNDTRIP_SCORE_SCHEMA,
)
from steering.eval.axbench_bpb_adv.scoring import (
    ModelBackedDualCompletionScorer,
    _bpb_config_from_adv_config,
    _build_token_bytes,
    _concepts_from_entries,
    _hashable,
    _is_control_candidate,
    _preflight_direction_scales_for_rows,
    _read_concepts,
    _resolve_context_segments,
    _token_list,
    _validate_candidate_digests,
    build_bpb_adv_score_table,
    load_context_segments_by_prefix_hash,
    require_precomputed_bag_candidates_cover_scored_prefixes,
)
from steering.eval.axbench_bpb_adv.universal_gcg_checkpoint import (
    assert_target_checkpoints_unchanged,
    resolve_and_pin_target_checkpoints,
)
from steering.eval.roundtrip_budgeting import (
    RoundtripGenerationBudgetError,
    reference_segment_generation_budget,
)
from steering.eval.roundtrip_steering import (
    RoundtripFeatureScores,
    RoundtripProbeConfig,
    load_percentile_table_for_roundtrip,
    load_probe_model_for_roundtrip,
    probe_text_against_features,
    resolve_probe_device,
)
from steering.eval.target_conditioning import (
    ContextFeatureSegment,
    SingleEntryContextProvider,
)
from steering.feature_paths import parse_feature_dir
from steering.refined_types import PositiveInt
from steering.sae_resolver import (
    SAE_WIDTH_TO_NUM_FEATURES,
    L0Tier,
    SAEWidth,
    parse_neuronpedia_source,
)
from steering.segmentation import resolve_bos_token_id, token_to_char_offsets
from steering.stats_io import load_statistics

if TYPE_CHECKING:
    from steering.eval.cluster_feature_contract import ClusterFeatureContract

_ROUNDTRIP_IDENTITY_EXCLUDED = frozenset({"candidate_id", "attack_row_id"})
_CONTROL_PROVIDER_CONTENT_KEY_INDEX = 2


@dataclass(frozen=True)
class RoundtripGenerationInputs:
    prompt_tokens: list[int]
    prompt_text: str
    context_provider: SingleEntryContextProvider | None


@dataclass(frozen=True)
class RoundtripGeneratedCompletion:
    text: str
    token_count: int
    stop_reason: str


class RoundtripGenerateBatchFn(Protocol):
    def __call__(
        self,
        *,
        candidate_row: Mapping[str, Any],
        intervention: Intervention,
        inputs: RoundtripGenerationInputs,
        generation_cfg: GenerationConfig,
        sample_seeds: Sequence[tuple[int, int]],
    ) -> list[RoundtripGeneratedCompletion]: ...


class RoundtripProbeFn(Protocol):
    def __call__(
        self,
        *,
        prompt_text: str,
        response_text: str,
        feature_ids: Sequence[int],
        probe_model: Any,
        percentile_table: t.Tensor | None,
        feature_chunk_size: int,
    ) -> RoundtripFeatureScores: ...


def _ensure_candidates_schema(candidates: pa.Table) -> None:
    if not candidates.schema.equals(CANDIDATES_SCHEMA):
        raise AxBenchBpbAdvError("expected candidates with CANDIDATES_SCHEMA")


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in sorted(value.items())}
    return value


def _roundtrip_generation_identity_key(
    candidate_row: Mapping[str, Any],
    *,
    prompt_tokens: Sequence[int],
    sample_index: int,
    generation_cfg: GenerationConfig,
) -> tuple[Any, ...]:
    batch_identity = _roundtrip_generation_batch_identity_key(
        candidate_row,
        prompt_tokens=prompt_tokens,
        generation_cfg=generation_cfg,
    )
    generation_identity = batch_identity[-1]
    return (
        *batch_identity[:-1],
        (
            generation_identity[0],
            int(sample_index),
            *generation_identity[1:],
        ),
    )


def _roundtrip_generation_batch_identity_key(
    candidate_row: Mapping[str, Any],
    *,
    prompt_tokens: Sequence[int],
    generation_cfg: GenerationConfig,
) -> tuple[Any, ...]:
    """Return the steered generation identity without ``sample_index``."""
    candidate_identity = tuple(
        _hashable(candidate_row[name])
        for name in CANDIDATES_SCHEMA.names
        if name not in _ROUNDTRIP_IDENTITY_EXCLUDED
    )
    generation_identity = (
        tuple(int(token) for token in prompt_tokens),
        int(generation_cfg.max_new_tokens),
        float(generation_cfg.temperature),
        int(generation_cfg.top_k_sampling),
    )
    return (*candidate_identity, generation_identity)


def _float64_bits(value: float | None) -> str | None:
    """Return a stable, bit-exact binary64 identity for ``value``."""

    if value is None:
        return None
    # Sign-of-zero is not a semantic distinction for coefficient math.
    value = 0.0 if value == 0.0 else value
    return struct.pack(">d", float(value)).hex()


def _canonical_resolved_provider_content(
    inputs: RoundtripGenerationInputs,
) -> tuple[Any, ...]:
    """Return exactly the resolved context content consumed by generation."""

    if inputs.context_provider is None:
        return ("decoder_only",)
    segments = inputs.context_provider.get(inputs.prompt_tokens)
    if segments is None:
        raise AxBenchBpbAdvError(
            "roundtrip context provider does not contain its generation prompt tokens"
        )
    return tuple(
        (
            tuple(int(feature_id) for feature_id in segment.feature_ids),
            tuple(int(bucket) for bucket in segment.buckets),
            int(segment.char_start),
            int(segment.char_end),
        )
        for segment in segments
    )


def _control_generation_identity_key(
    candidate_row: Mapping[str, Any],
    *,
    control_intervention: Intervention,
    control_inputs: RoundtripGenerationInputs,
    sample_index: int,
    generation_cfg: GenerationConfig,
) -> tuple[Any, ...]:
    """Identify exactly one semantic control-generation draw.

    Raw candidate provenance is deliberately excluded except where it identifies
    an active counter-direction tensor that the resolved ``Intervention`` cannot
    name by itself. In particular, an active counter retains bit-exact adversarial
    strength provenance, so strengths cannot share controls; a resolved no-op uses
    an explicit marker and can never collide with a countered intervention.
    Direction coefficients and adversarial strengths use their binary64 bits,
    rather than numeric equality, as identity. The steered-side key is separate
    and deliberately unchanged. Encoder-decoder identity uses the canonical
    resolved provider segments consumed by generation, not the candidate's raw
    context mode or source segments.
    """

    batch_identity = _control_generation_batch_identity_key(
        candidate_row,
        control_intervention=control_intervention,
        control_inputs=control_inputs,
        generation_cfg=generation_cfg,
    )
    return (*batch_identity[:6], int(sample_index), *batch_identity[6:])


def _control_generation_batch_identity_key(
    candidate_row: Mapping[str, Any],
    *,
    control_intervention: Intervention,
    control_inputs: RoundtripGenerationInputs,
    generation_cfg: GenerationConfig,
) -> tuple[Any, ...]:
    """Return the semantic control identity without ``sample_index``."""
    feature_conditions = (
        None
        if control_intervention.feature_conditions is None
        else tuple(
            sorted(
                (int(feature_id), int(bucket))
                for feature_id, bucket in control_intervention.feature_conditions.items()
            )
        )
    )
    prompt_prefix_tokens = (
        None
        if control_intervention.prompt_prefix_tokens is None
        else tuple(int(token) for token in control_intervention.prompt_prefix_tokens)
    )
    counter_provenance: tuple[Any, ...]
    if control_intervention.counter_direction is None:
        counter_provenance = ("no_counter",)
    else:
        counter_provenance = (
            str(candidate_row["adversarial_method"]),
            _float64_bits(float(candidate_row["adversarial_strength"])),
            str(candidate_row["mode"]),
            int(candidate_row["concept_id"]),
        )
    return (
        str(candidate_row["target_model"]),
        tuple(int(token) for token in control_inputs.prompt_tokens),
        _canonical_resolved_provider_content(control_inputs),
        int(generation_cfg.max_new_tokens),
        float(generation_cfg.temperature),
        int(generation_cfg.top_k_sampling),
        str(control_intervention.kind),
        feature_conditions,
        control_intervention.direction_layer,
        _float64_bits(control_intervention.direction_coefficient),
        (
            None
            if control_intervention.lora_adapter_path is None
            else str(control_intervention.lora_adapter_path)
        ),
        prompt_prefix_tokens,
        control_intervention.counter_direction_layer,
        _float64_bits(control_intervention.counter_direction_coefficient),
        counter_provenance,
    )


def _roundtrip_batch_seed(
    *,
    identity_key_without_sample_index: tuple[Any, ...],
    master_seed: int,
) -> int:
    """Derive one RNG seed for all rows in a candidate-side batch."""
    payload = json.dumps(
        _jsonable(identity_key_without_sample_index),
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(f"{int(master_seed)}|{payload}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**31 - 1)


RoundtripGenerationCache = dict[
    tuple[Any, ...],
    tuple[RoundtripGenerationInputs, RoundtripGeneratedCompletion, int],
]


def _resolve_candidate_side_generations(
    *,
    side: str,
    candidate_row: Mapping[str, Any],
    candidate_id: str,
    intervention: Intervention,
    inputs: RoundtripGenerationInputs,
    generation_cfg: GenerationConfig,
    samples_per_candidate: int,
    master_seed: int,
    generation_cache: RoundtripGenerationCache,
    generate_completions: RoundtripGenerateBatchFn,
) -> list[tuple[RoundtripGenerationInputs, RoundtripGeneratedCompletion, int]]:
    if side == "control":
        batch_identity = _control_generation_batch_identity_key(
            candidate_row,
            control_intervention=intervention,
            control_inputs=inputs,
            generation_cfg=generation_cfg,
        )
        identities = [
            _control_generation_identity_key(
                candidate_row,
                control_intervention=intervention,
                control_inputs=inputs,
                sample_index=sample_index,
                generation_cfg=generation_cfg,
            )
            for sample_index in range(samples_per_candidate)
        ]
    else:
        batch_identity = _roundtrip_generation_batch_identity_key(
            candidate_row,
            prompt_tokens=inputs.prompt_tokens,
            generation_cfg=generation_cfg,
        )
        identities = [
            _roundtrip_generation_identity_key(
                candidate_row,
                prompt_tokens=inputs.prompt_tokens,
                sample_index=sample_index,
                generation_cfg=generation_cfg,
            )
            for sample_index in range(samples_per_candidate)
        ]
    batch_seed = _roundtrip_batch_seed(
        identity_key_without_sample_index=batch_identity,
        master_seed=master_seed,
    )
    sample_seeds = [
        (sample_index, batch_seed) for sample_index in range(samples_per_candidate)
    ]
    cached_indices = [
        sample_index
        for sample_index, identity in enumerate(identities)
        if identity in generation_cache
    ]
    missing_indices = [
        sample_index
        for sample_index, identity in enumerate(identities)
        if identity not in generation_cache
    ]
    for _sample_index in range(samples_per_candidate):
        record_counter(
            "axbench_bpb_adv.roundtrip.sample_requests",
            1,
            side=side,
        )
    if cached_indices and missing_indices:
        raise AxBenchBpbAdvError(
            f"candidate_id={candidate_id!r} has mixed {side} generation cache "
            f"state: cached sample indices {cached_indices}; missing sample "
            f"indices {missing_indices}"
        )
    if cached_indices:
        for _sample_index in cached_indices:
            record_counter(
                "axbench_bpb_adv.roundtrip.sample_cache_hits",
                1,
                side=side,
            )
        if side == "control":
            current_provider_content = _canonical_resolved_provider_content(inputs)
            for identity in identities:
                try:
                    keyed_provider_content = identity[
                        _CONTROL_PROVIDER_CONTENT_KEY_INDEX
                    ]
                except IndexError as exc:
                    raise AxBenchBpbAdvError(
                        "cached control generation key has no resolved context-provider "
                        "component"
                    ) from exc
                if current_provider_content != keyed_provider_content:
                    raise AxBenchBpbAdvError(
                        f"candidate_id={candidate_id!r} resolved context-provider "
                        "content does not match the cached control generation key"
                    )
        return [
            (inputs, generation_cache[identity][1], generation_cache[identity][2])
            for identity in identities
        ]

    generated = generate_completions(
        candidate_row=candidate_row,
        intervention=intervention,
        inputs=inputs,
        generation_cfg=generation_cfg,
        sample_seeds=sample_seeds,
    )
    if len(generated) != samples_per_candidate:
        raise AxBenchBpbAdvError(
            f"candidate_id={candidate_id!r} {side} batch generation returned "
            f"{len(generated)} completions for {samples_per_candidate} sample seeds"
        )
    if side == "control":
        record_counter(
            "axbench_bpb_adv.roundtrip.control_unique_batches",
            1,
        )
        record_counter(
            "axbench_bpb_adv.roundtrip.control_unique_completions",
            len(sample_seeds),
        )
    for identity, completion, (_sample_index, seed) in zip(
        identities,
        generated,
        sample_seeds,
        strict=True,
    ):
        generation_cache[identity] = (inputs, completion, seed)
    return [generation_cache[identity] for identity in identities]


def _shift_context_segments(
    segments: tuple[ContextFeatureSegment, ...],
    *,
    char_offset: int,
) -> tuple[ContextFeatureSegment, ...]:
    """Shift attacked-prompt-relative segments into full prompt coordinates."""

    return tuple(
        replace(
            segment,
            char_start=int(segment.char_start) + int(char_offset),
            char_end=int(segment.char_end) + int(char_offset),
        )
        for segment in segments
    )


def _roundtrip_probe_prompt_text(
    *,
    attacked_tokens: Sequence[int],
    prompt_prefix_tokens: Sequence[int],
    tokenizer: Any,
) -> str:
    bos_id = resolve_bos_token_id(tokenizer)
    attacked_payload_tokens = [int(token) for token in attacked_tokens]
    if (
        bos_id is not None
        and attacked_payload_tokens
        and attacked_payload_tokens[0] == int(bos_id)
    ):
        attacked_payload_tokens = attacked_payload_tokens[1:]
    probe_prompt_tokens = [
        *[int(token) for token in prompt_prefix_tokens],
        *attacked_payload_tokens,
    ]
    return str(tokenizer.decode(probe_prompt_tokens))


def _roundtrip_context_attacked_char_start(
    *,
    prompt_tokens: Sequence[int],
    prompt_prefix_token_count: int,
    tokenizer: Any,
) -> int:
    if not prompt_tokens:
        return 0
    offsets = token_to_char_offsets(tokenizer, [int(token) for token in prompt_tokens])
    attacked_start_token = int(prompt_prefix_token_count)
    return int(offsets[attacked_start_token])


def _validate_roundtrip_context_text_alignment(
    *,
    attacked_tokens: Sequence[int],
    context_text: str,
    attacked_char_start: int,
    tokenizer: Any,
) -> None:
    attacked_context_text = str(tokenizer.decode([int(token) for token in attacked_tokens]))
    if context_text[int(attacked_char_start) :] != attacked_context_text:
        raise AxBenchBpbAdvError(
            "cannot align encoder-decoder roundtrip context segments: decoded "
            "roundtrip context text does not preserve the decoded attacked prompt "
            "text at the context-token boundary"
        )


def _roundtrip_generation_inputs(
    *,
    candidate_row: Mapping[str, Any],
    context_segments_by_prefix_hash: Mapping[str, tuple[ContextFeatureSegment, ...]],
    tokenizer: Any,
    intervention: Intervention,
    cluster_contract: ClusterFeatureContract | None = None,
) -> RoundtripGenerationInputs:
    _validate_candidate_digests(candidate_row)
    attacked_tokens = _token_list(
        candidate_row["attacked_prompt_tokens"],
        field_name="attacked_prompt_tokens",
    )
    prompt_prefix_tokens = [
        int(token) for token in (intervention.prompt_prefix_tokens or [])
    ]
    prompt_tokens = [*prompt_prefix_tokens, *attacked_tokens]
    prompt_text = _roundtrip_probe_prompt_text(
        attacked_tokens=attacked_tokens,
        prompt_prefix_tokens=prompt_prefix_tokens,
        tokenizer=tokenizer,
    )
    target_model = str(candidate_row["target_model"])
    if target_model == "decoder_only":
        return RoundtripGenerationInputs(
            prompt_tokens=prompt_tokens,
            prompt_text=prompt_text,
            context_provider=None,
        )
    if target_model != "encdec":
        raise AxBenchBpbAdvError(f"unsupported target_model={target_model!r}")

    context_segments = _resolve_context_segments(
        candidate_row=candidate_row,
        context_segments_by_prefix_hash=context_segments_by_prefix_hash,
        tokenizer=tokenizer,
    )
    if cluster_contract is not None:
        for segment_index, segment in enumerate(context_segments):
            ids = t.tensor(segment.feature_ids, dtype=t.int64)
            try:
                cluster_contract.consolidation.assert_representatives_only(
                    ids,
                    surface=f"roundtrip context segment {segment_index}",
                )
                cluster_contract.feature_mask.assert_no_masked(ids)
            except Exception as exc:
                raise AxBenchBpbAdvError(
                    "roundtrip generation context must be exact clustered-cache "
                    f"content before provider construction: {exc}"
                ) from exc
    attacked_char_start = _roundtrip_context_attacked_char_start(
        prompt_tokens=prompt_tokens,
        prompt_prefix_token_count=len(prompt_prefix_tokens),
        tokenizer=tokenizer,
    )
    _validate_roundtrip_context_text_alignment(
        attacked_tokens=attacked_tokens,
        context_text=str(tokenizer.decode(prompt_tokens)),
        attacked_char_start=attacked_char_start,
        tokenizer=tokenizer,
    )
    if prompt_prefix_tokens and attacked_char_start <= 0:
        raise AxBenchBpbAdvError(
            "prompting intervention decoded to an empty prompt-prefix span; "
            "cannot build encoder-decoder roundtrip context segments"
        )
    if attacked_char_start > 0:
        # Context segments are char spans in the attacked prompt. Prompting
        # interventions and decoded BOS markers prepend text during generation,
        # so the full prompt gets an empty no-feature prefix segment and the
        # original spans shift right.
        context_segments = (
            ContextFeatureSegment(
                feature_ids=(),
                buckets=(),
                char_start=0,
                char_end=attacked_char_start,
            ),
            *_shift_context_segments(
                context_segments,
                char_offset=attacked_char_start,
            ),
        )
    provider = SingleEntryContextProvider(prompt_tokens, context_segments)
    return RoundtripGenerationInputs(
        prompt_tokens=prompt_tokens,
        prompt_text=prompt_text,
        context_provider=provider,
    )


def _generation_config(cfg: AxBenchBpbAdvConfig) -> GenerationConfig:
    rt = cfg.roundtrip_generation
    return GenerationConfig(
        master_seed=rt.master_seed,
        temperature=rt.temperature,
        top_k_sampling=rt.top_k_sampling,
    )


def _reference_output_tokens_for_mode(candidate_row: Mapping[str, Any]) -> list[int]:
    """Reference continuation for the feature direction this candidate targets."""

    mode = SteeringMode(str(candidate_row["mode"]))
    if mode is SteeringMode.POSITIVE:
        field = "concept_rich_output_tokens"
    elif mode is SteeringMode.NEGATIVE:
        field = "concept_avoiding_output_tokens"
    else:
        raise AxBenchBpbAdvError(f"unknown steering mode {mode!r}")
    tokens = _token_list(candidate_row[field], field_name=field)
    if not tokens:
        raise AxBenchBpbAdvError(
            f"candidate {candidate_row['candidate_id']!r} has empty {field}; "
            "cannot set roundtrip generation length"
        )
    return tokens


def _generation_config_for_candidate(
    *,
    base_generation_cfg: GenerationConfig,
    reference_segment_token_count: int,
) -> GenerationConfig:
    return replace(
        base_generation_cfg,
        max_new_tokens=int(reference_segment_token_count),
    )


def _target_feature_id(
    *,
    candidate_row: Mapping[str, Any],
    concepts_by_id: Mapping[int, Concept],
) -> int:
    concept_id = int(candidate_row["concept_id"])
    concept = concepts_by_id.get(concept_id)
    if concept is None:
        raise AxBenchBpbAdvError(
            f"source concepts do not contain concept_id={concept_id}"
        )
    return int(concept.feature_index)


def _validate_roundtrip_probe_matches_concepts(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    concepts_by_id: Mapping[int, Concept],
    probe_config: RoundtripProbeConfig,
) -> None:
    checked: set[int] = set()
    for candidate_row in candidate_rows:
        concept_id = int(candidate_row["concept_id"])
        if concept_id in checked:
            continue
        checked.add(concept_id)
        concept = concepts_by_id.get(concept_id)
        if concept is None:
            raise AxBenchBpbAdvError(
                f"source concepts do not contain concept_id={concept_id}"
            )
        try:
            release, activation_type, layer, width, l0_tier = (
                parse_neuronpedia_source(concept.sae_source)
            )
        except ValueError as exc:
            raise AxBenchBpbAdvError(
                f"concept_id={concept_id} has invalid sae_source="
                f"{concept.sae_source!r}: {exc}"
            ) from exc
        mismatches: list[str] = []
        if int(concept.layer) != int(probe_config.layer):
            mismatches.append(
                f"concept.layer={int(concept.layer)} != "
                f"roundtrip probe layer={int(probe_config.layer)}"
            )
        if int(layer) != int(probe_config.layer):
            mismatches.append(
                f"sae_source layer={int(layer)} != "
                f"roundtrip probe layer={int(probe_config.layer)}"
            )
        if release.value != str(probe_config.sae_release):
            mismatches.append(
                f"sae_source release={release.value!r} != "
                f"roundtrip probe sae_release={probe_config.sae_release!r}"
            )
        if activation_type.value != str(probe_config.activation_type):
            mismatches.append(
                f"sae_source activation={activation_type.value!r} != "
                f"roundtrip probe activation_type={probe_config.activation_type!r}"
            )
        if width.value != str(probe_config.sae_width):
            mismatches.append(
                f"sae_source width={width.value!r} != "
                f"roundtrip probe sae_width={probe_config.sae_width!r}"
            )
        configured_l0 = probe_config.l0
        if l0_tier != configured_l0:
            mismatches.append(
                "sae_source l0 tier identity="
                f"{getattr(l0_tier, 'value', l0_tier)!r} != roundtrip probe "
                f"l0={getattr(configured_l0, 'value', configured_l0)!r}"
            )
        if mismatches:
            raise AxBenchBpbAdvError(
                f"roundtrip probe SAE does not match concept_id={concept_id} "
                f"({concept.sae_source!r}): " + "; ".join(mismatches)
            )


def _validate_roundtrip_stats_match_probe(
    *,
    stats_path: str | Path,
    probe_config: RoundtripProbeConfig,
) -> None:
    stats = load_statistics(stats_path)
    try:
        expected_width = SAEWidth(str(probe_config.sae_width))
    except ValueError as exc:
        raise AxBenchBpbAdvError(
            f"invalid roundtrip probe sae_width={probe_config.sae_width!r}"
        ) from exc
    expected_features = SAE_WIDTH_TO_NUM_FEATURES[expected_width]
    mismatches: list[str] = []
    if int(stats.metadata.layer) != int(probe_config.layer):
        mismatches.append(
            f"stats layer={int(stats.metadata.layer)} != "
            f"roundtrip probe layer={int(probe_config.layer)}"
        )
    if int(stats.metadata.n_sae_features) != int(expected_features):
        mismatches.append(
            f"stats n_sae_features={int(stats.metadata.n_sae_features)} != "
            f"roundtrip probe width {expected_width.value!r} "
            f"({int(expected_features)} features)"
        )
    configured_tier = (
        probe_config.l0 if isinstance(probe_config.l0, L0Tier) else None
    )
    if stats.l0_tier != configured_tier:
        mismatches.append(
            "stats l0 tier="
            f"{getattr(stats.l0_tier, 'value', stats.l0_tier)!r} != "
            "roundtrip probe l0 tier="
            f"{getattr(configured_tier, 'value', configured_tier)!r}"
        )
    if mismatches:
        raise AxBenchBpbAdvError(
            f"roundtrip feature statistics at {stats_path!r} do not match "
            "roundtrip probe: " + "; ".join(mismatches)
        )


@dataclass(frozen=True)
class _PreparedRoundtripCandidate:
    candidate_index: int
    candidate_row: Mapping[str, Any]
    candidate_id: str
    intervention: Intervention
    control_intervention: Intervention
    inputs: RoundtripGenerationInputs
    control_inputs: RoundtripGenerationInputs
    generation_cfg: GenerationConfig
    reference_segment_token_count: int
    target_feature_id: int
    loss_concept_rich: float
    loss_concept_avoiding: float
    restored_rows: list[dict[str, Any]] | None


def _seed_restored_generation_caches(
    *,
    prepared: _PreparedRoundtripCandidate,
    master_seed: int,
    generation_cache: RoundtripGenerationCache,
    control_generation_cache: RoundtripGenerationCache,
) -> None:
    restored_rows = prepared.restored_rows
    if restored_rows is None:
        return
    for sample_index, row in enumerate(restored_rows):
        prompt_tokens = [int(token) for token in row["roundtrip_prompt_tokens"]]
        control_prompt_tokens = [
            int(token) for token in row["control_roundtrip_prompt_tokens"]
        ]
        expected_prompt_digest = token_sequence_digest(prepared.inputs.prompt_tokens)
        expected_control_digest = token_sequence_digest(
            prepared.control_inputs.prompt_tokens
        )
        prompt_mismatches = []
        if prompt_tokens != prepared.inputs.prompt_tokens:
            prompt_mismatches.append("roundtrip_prompt_tokens")
        if str(row["roundtrip_prompt_token_digest"]) != expected_prompt_digest:
            prompt_mismatches.append("roundtrip_prompt_token_digest")
        if str(row["roundtrip_prompt_text"]) != prepared.inputs.prompt_text:
            prompt_mismatches.append("roundtrip_prompt_text")
        if control_prompt_tokens != prepared.control_inputs.prompt_tokens:
            prompt_mismatches.append("control_roundtrip_prompt_tokens")
        if str(row["control_roundtrip_prompt_token_digest"]) != expected_control_digest:
            prompt_mismatches.append("control_roundtrip_prompt_token_digest")
        if str(row["control_roundtrip_prompt_text"]) != prepared.control_inputs.prompt_text:
            prompt_mismatches.append("control_roundtrip_prompt_text")
        if prompt_mismatches:
            raise AxBenchBpbAdvError(
                f"candidate_id={prepared.candidate_id!r} restored generation prompt "
                f"does not match freshly built inputs: {prompt_mismatches}"
            )
        steered_identity = _roundtrip_generation_identity_key(
            prepared.candidate_row,
            prompt_tokens=prepared.inputs.prompt_tokens,
            sample_index=sample_index,
            generation_cfg=prepared.generation_cfg,
        )
        control_identity = _control_generation_identity_key(
            prepared.candidate_row,
            control_intervention=prepared.control_intervention,
            control_inputs=prepared.control_inputs,
            sample_index=sample_index,
            generation_cfg=prepared.generation_cfg,
        )
        seed = _roundtrip_batch_seed(
            identity_key_without_sample_index=_roundtrip_generation_batch_identity_key(
                prepared.candidate_row,
                prompt_tokens=prepared.inputs.prompt_tokens,
                generation_cfg=prepared.generation_cfg,
            ),
            master_seed=master_seed,
        )
        control_seed = _roundtrip_batch_seed(
            identity_key_without_sample_index=_control_generation_batch_identity_key(
                prepared.candidate_row,
                control_intervention=prepared.control_intervention,
                control_inputs=prepared.control_inputs,
                generation_cfg=prepared.generation_cfg,
            ),
            master_seed=master_seed,
        )
        if int(row["generation_seed"]) != seed:
            raise AxBenchBpbAdvError(
                f"candidate_id={prepared.candidate_id!r} restored generation_seed "
                f"differs at sample_index={sample_index}"
            )
        if int(row["control_generation_seed"]) != control_seed:
            raise AxBenchBpbAdvError(
                f"candidate_id={prepared.candidate_id!r} restored control_generation_seed "
                f"differs at sample_index={sample_index}"
            )
        completions = (
            (
                generation_cache,
                steered_identity,
                prepared.inputs,
                RoundtripGeneratedCompletion(
                    text=str(row["generated_text"]),
                    token_count=int(row["n_generated_tokens"]),
                    stop_reason=str(row["stop_reason"]),
                ),
                seed,
            ),
            (
                control_generation_cache,
                control_identity,
                prepared.control_inputs,
                RoundtripGeneratedCompletion(
                    text=str(row["control_generated_text"]),
                    token_count=int(row["control_n_generated_tokens"]),
                    stop_reason=str(row["control_stop_reason"]),
                ),
                control_seed,
            ),
        )
        for cache, identity, inputs, completion, resolved_seed in completions:
            existing = cache.get(identity)
            restored = (inputs, completion, resolved_seed)
            if existing is not None and not rows_bits_equal(
                [{"completion": existing[1], "seed": existing[2]}],
                [{"completion": restored[1], "seed": restored[2]}],
            ):
                raise AxBenchBpbAdvError(
                    f"candidate_id={prepared.candidate_id!r} restored completion "
                    "conflicts with another restored semantic cache entry"
                )
            cache[identity] = restored


def build_roundtrip_generation_rows(
    *,
    candidates: pa.Table,
    context_segments_by_prefix_hash: Mapping[str, tuple[ContextFeatureSegment, ...]],
    tokenizer: Any,
    extraction_tokenizer: Any,
    concepts_by_id: Mapping[int, Concept],
    generation_cfg: GenerationConfig,
    reference_segment_size_by_target: Mapping[str, int],
    samples_per_candidate: int,
    master_seed: int,
    baseline_losses_by_candidate_id: Mapping[str, tuple[float, float]],
    build_intervention_for_row: Callable[[Mapping[str, Any]], Intervention],
    build_control_intervention_for_row: Callable[[Mapping[str, Any]], Intervention],
    generate_completions: RoundtripGenerateBatchFn,
    stats: dict[str, int] | None = None,
    log_progress_every: int = 0,
    cluster_contract: ClusterFeatureContract | None = None,
    checkpoint_store: RoundtripCheckpointStore | None = None,
    runtime_fingerprint: str | None = None,
) -> list[dict[str, Any]]:
    _ensure_candidates_schema(candidates)
    if samples_per_candidate < 1:
        raise AxBenchBpbAdvError("samples_per_candidate must be >= 1")
    if (checkpoint_store is None) != (runtime_fingerprint is None):
        raise AxBenchBpbAdvError(
            "roundtrip checkpoint_store and runtime_fingerprint must be supplied together"
        )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    generation_cache: RoundtripGenerationCache = {}
    control_generation_cache: RoundtripGenerationCache = {}
    candidate_rows = candidates.to_pylist()
    n_candidates = len(candidate_rows)
    restored_by_id = (
        {} if checkpoint_store is None else checkpoint_store.load_generation_rows()
    )
    unknown_restored = set(restored_by_id) - {
        str(row["candidate_id"]) for row in candidate_rows
    }
    if unknown_restored:
        raise AxBenchBpbAdvError(
            f"restored generation rows contain unknown candidate ids {sorted(unknown_restored)}"
        )
    prepared_candidates: list[_PreparedRoundtripCandidate] = []
    for candidate_index, candidate_row in enumerate(candidate_rows):
        candidate_id = str(candidate_row["candidate_id"])
        if candidate_id in seen:
            raise AxBenchBpbAdvError(f"duplicate candidate_id={candidate_id!r}")
        seen.add(candidate_id)
        try:
            loss_concept_rich, loss_concept_avoiding = baseline_losses_by_candidate_id[
                candidate_id
            ]
        except KeyError as exc:
            raise AxBenchBpbAdvError(
                f"missing BPB baseline losses for candidate_id={candidate_id!r}"
            ) from exc
        with profile_region(
            "axbench_bpb_adv.roundtrip.build_interventions",
            candidate_index=candidate_index,
            candidate_id=candidate_id,
            target_model=str(candidate_row["target_model"]),
        ):
            intervention = build_intervention_for_row(candidate_row)
            control_intervention = build_control_intervention_for_row(candidate_row)
        target_model = str(candidate_row["target_model"])
        try:
            raw_reference_segment_size = reference_segment_size_by_target[target_model]
        except KeyError as exc:
            raise AxBenchBpbAdvError(
                f"missing reference segment size for target_model={target_model!r}"
            ) from exc
        try:
            reference_segment_size = PositiveInt.parse(raw_reference_segment_size)
        except TypeError as exc:
            raise AxBenchBpbAdvError(
                "reference_segment_size must be a strict positive integer for "
                f"target_model={target_model!r}, got {raw_reference_segment_size!r}"
            ) from exc
        reference_tokens = _reference_output_tokens_for_mode(candidate_row)
        source_prompt_tokens = _token_list(
            candidate_row["source_prompt_tokens"],
            field_name="source_prompt_tokens",
        )
        try:
            reference_budget = reference_segment_generation_budget(
                prefix_text=str(candidate_row["prefix_text"]),
                prefix_tokens=source_prompt_tokens,
                reference_output_tokens=reference_tokens,
                target_tokenizer=tokenizer,
                extraction_tokenizer=extraction_tokenizer,
                reference_segment_size=reference_segment_size,
            )
        except RoundtripGenerationBudgetError as exc:
            raise AxBenchBpbAdvError(
                f"candidate {candidate_id!r} cannot set roundtrip generation "
                f"length: {exc}"
            ) from exc
        reference_segment_token_count = int(reference_budget.max_new_tokens)
        cluster_kwargs = (
            {} if cluster_contract is None else {"cluster_contract": cluster_contract}
        )
        with profile_region(
            "axbench_bpb_adv.roundtrip.build_generation_inputs",
            candidate_index=candidate_index,
            candidate_id=candidate_id,
            target_model=target_model,
        ):
            inputs = _roundtrip_generation_inputs(
                candidate_row=candidate_row,
                context_segments_by_prefix_hash=context_segments_by_prefix_hash,
                tokenizer=tokenizer,
                intervention=intervention,
                **cluster_kwargs,
            )
            control_inputs = _roundtrip_generation_inputs(
                candidate_row=candidate_row,
                context_segments_by_prefix_hash=context_segments_by_prefix_hash,
                tokenizer=tokenizer,
                intervention=control_intervention,
                **cluster_kwargs,
            )
        candidate_generation_cfg = _generation_config_for_candidate(
            base_generation_cfg=generation_cfg,
            reference_segment_token_count=reference_segment_token_count,
        )
        target_feature_id = _target_feature_id(
            candidate_row=candidate_row,
            concepts_by_id=concepts_by_id,
        )
        prepared_candidates.append(
            _PreparedRoundtripCandidate(
                candidate_index=candidate_index,
                candidate_row=candidate_row,
                candidate_id=candidate_id,
                intervention=intervention,
                control_intervention=control_intervention,
                inputs=inputs,
                control_inputs=control_inputs,
                generation_cfg=candidate_generation_cfg,
                reference_segment_token_count=reference_segment_token_count,
                target_feature_id=target_feature_id,
                loss_concept_rich=float(loss_concept_rich),
                loss_concept_avoiding=float(loss_concept_avoiding),
                restored_rows=restored_by_id.get(candidate_id),
            )
        )

    # Seed every restored semantic completion before any fresh generation. This
    # preserves control sharing regardless of candidate order while still
    # rebuilding the cheap inputs/interventions and checking their prompt bytes.
    for prepared in prepared_candidates:
        _seed_restored_generation_caches(
            prepared=prepared,
            master_seed=master_seed,
            generation_cache=generation_cache,
            control_generation_cache=control_generation_cache,
        )

    for prepared in prepared_candidates:
        candidate_index = prepared.candidate_index
        candidate_row = prepared.candidate_row
        candidate_id = prepared.candidate_id
        generated_samples = _resolve_candidate_side_generations(
            side="steered",
            candidate_row=candidate_row,
            candidate_id=candidate_id,
            intervention=prepared.intervention,
            inputs=prepared.inputs,
            generation_cfg=prepared.generation_cfg,
            samples_per_candidate=samples_per_candidate,
            master_seed=master_seed,
            generation_cache=generation_cache,
            generate_completions=generate_completions,
        )
        control_generated_samples = _resolve_candidate_side_generations(
            side="control",
            candidate_row=candidate_row,
            candidate_id=candidate_id,
            intervention=prepared.control_intervention,
            inputs=prepared.control_inputs,
            generation_cfg=prepared.generation_cfg,
            samples_per_candidate=samples_per_candidate,
            master_seed=master_seed,
            generation_cache=control_generation_cache,
            generate_completions=generate_completions,
        )
        candidate_output_rows: list[dict[str, Any]] = []
        for sample_index in range(samples_per_candidate):
            inputs, generated, seed = generated_samples[sample_index]
            control_inputs, control_generated, control_seed = (
                control_generated_samples[sample_index]
            )
            candidate_output_rows.append(
                {
                    "candidate_id": candidate_id,
                    "core_prefix_id": str(candidate_row["core_prefix_id"]),
                    "concept_id": int(candidate_row["concept_id"]),
                    "mode": str(candidate_row["mode"]),
                    "method_name": str(candidate_row["method_name"]),
                    "target_model": str(candidate_row["target_model"]),
                    "coefficient_str": str(candidate_row["coefficient_str"]),
                    "adversarial_method": str(candidate_row["adversarial_method"]),
                    "adversarial_strength": float(
                        candidate_row["adversarial_strength"]
                    ),
                    "strength_unit": str(candidate_row["strength_unit"]),
                    "strength_index": int(candidate_row["strength_index"]),
                    "context_signal_mode": str(candidate_row["context_signal_mode"]),
                    "source_prompt_tokens": _token_list(
                        candidate_row["source_prompt_tokens"],
                        field_name="source_prompt_tokens",
                    ),
                    "source_prompt_token_digest": str(
                        candidate_row["source_prompt_token_digest"]
                    ),
                    "attacked_prompt_tokens": _token_list(
                        candidate_row["attacked_prompt_tokens"],
                        field_name="attacked_prompt_tokens",
                    ),
                    "attacked_prompt_token_digest": str(
                        candidate_row["attacked_prompt_token_digest"]
                    ),
                    "roundtrip_prompt_tokens": list(inputs.prompt_tokens),
                    "roundtrip_prompt_token_digest": token_sequence_digest(
                        inputs.prompt_tokens
                    ),
                    "roundtrip_prompt_text": inputs.prompt_text,
                    "control_roundtrip_prompt_tokens": list(
                        control_inputs.prompt_tokens
                    ),
                    "control_roundtrip_prompt_token_digest": token_sequence_digest(
                        control_inputs.prompt_tokens
                    ),
                    "control_roundtrip_prompt_text": control_inputs.prompt_text,
                    "editable_token_start": int(candidate_row["editable_token_start"]),
                    "editable_token_count": int(candidate_row["editable_token_count"]),
                    "replacement_tokens": _token_list(
                        candidate_row["replacement_tokens"],
                        field_name="replacement_tokens",
                    ),
                    "loss_concept_rich": prepared.loss_concept_rich,
                    "loss_concept_avoiding": prepared.loss_concept_avoiding,
                    "sample_index": int(sample_index),
                    "reference_segment_token_count": int(
                        prepared.reference_segment_token_count
                    ),
                    "generation_seed": int(seed),
                    "generated_text": str(generated.text),
                    "n_generated_tokens": int(generated.token_count),
                    "stop_reason": str(generated.stop_reason),
                    "control_generation_seed": int(control_seed),
                    "control_generated_text": str(control_generated.text),
                    "control_n_generated_tokens": int(control_generated.token_count),
                    "control_stop_reason": str(control_generated.stop_reason),
                    "is_control": _is_control_candidate(candidate_row),
                    "target_feature_id": prepared.target_feature_id,
                }
            )
        if prepared.restored_rows is not None:
            if not rows_bits_equal(candidate_output_rows, prepared.restored_rows):
                raise AxBenchBpbAdvError(
                    f"candidate_id={candidate_id!r} restored generation rows do not "
                    "reconstruct byte-equivalent row payloads"
                )
            candidate_output_rows = deepcopy(prepared.restored_rows)
        if checkpoint_store is not None:
            checkpoint_store.commit_generation_rows(
                candidate_id,
                candidate_output_rows,
                cast(str, runtime_fingerprint),
            )
        rows.extend(candidate_output_rows)
        if (
            log_progress_every > 0
            and (candidate_index + 1) % log_progress_every == 0
        ):
            logger.info(
                "bpb-adv roundtrip generation: {}/{} candidates processed "
                "({} unique steered, {} unique control completions)",
                candidate_index + 1,
                n_candidates,
                len(generation_cache),
                len(control_generation_cache),
            )
    generated_unique = len(generation_cache)
    cache_hits = len(rows) - generated_unique
    control_generated_unique = len(control_generation_cache)
    control_cache_hits = len(rows) - control_generated_unique
    if stats is not None:
        stats["generated_unique"] = generated_unique
        stats["generation_cache_hits"] = cache_hits
        stats["generated_tokens"] = sum(int(row["n_generated_tokens"]) for row in rows)
        stats["control_generated_unique"] = control_generated_unique
        stats["control_generation_cache_hits"] = control_cache_hits
        stats["control_generated_tokens"] = sum(
            int(row["control_n_generated_tokens"]) for row in rows
        )
        if checkpoint_store is not None:
            stats["checkpoint_restored_generation_candidates"] = len(restored_by_id)
            stats["checkpoint_fresh_generation_candidates"] = n_candidates - len(
                restored_by_id
            )
    if cache_hits:
        logger.info(
            "bpb-adv roundtrip generation: {} candidate samples, {} unique "
            "sampled completions, {} reused (deduplicated)",
            len(rows),
            generated_unique,
            cache_hits,
        )
    if control_cache_hits:
        logger.info(
            "bpb-adv roundtrip control generation: {} candidate samples, {} "
            "unique sampled completions, {} reused (deduplicated)",
            len(rows),
            control_generated_unique,
            control_cache_hits,
        )
    return rows


def score_roundtrip_generation_rows(
    *,
    generation_rows: Sequence[Mapping[str, Any]],
    probe_model: Any,
    percentile_table: t.Tensor | None,
    feature_chunk_size: int,
    probe_text_fn: RoundtripProbeFn = probe_text_against_features,
    stats: dict[str, int] | None = None,
    checkpoint_store: RoundtripCheckpointStore | None = None,
    runtime_fingerprint: str | None = None,
) -> pa.Table:
    if (checkpoint_store is None) != (runtime_fingerprint is None):
        raise AxBenchBpbAdvError(
            "roundtrip checkpoint_store and runtime_fingerprint must be supplied together"
        )
    rows: list[dict[str, Any]] = []
    # The probe model and percentile table are fixed for this whole call, so
    # prompt/response/feature identity fully determines a probe result.
    probe_cache: dict[tuple[str, str, tuple[int, ...]], RoundtripFeatureScores] = {}

    def _probe_once(
        *,
        side: str,
        prompt_text: str,
        response_text: str,
        feature_ids: tuple[int, ...],
    ) -> RoundtripFeatureScores:
        key = (prompt_text, response_text, feature_ids)
        cached = probe_cache.get(key)
        if cached is not None:
            record_counter(
                "axbench_bpb_adv.roundtrip.probe_cache_hits",
                1,
                side=side,
            )
            return cached
        scores = probe_text_fn(
            prompt_text=prompt_text,
            response_text=response_text,
            feature_ids=feature_ids,
            probe_model=probe_model,
            percentile_table=percentile_table,
            feature_chunk_size=feature_chunk_size,
        )
        probe_cache[key] = scores
        record_counter(
            "axbench_bpb_adv.roundtrip.probe_unique_texts",
            1,
            side=side,
        )
        return scores

    candidate_groups: list[tuple[str, list[Mapping[str, Any]]]] = []
    seen_candidates: set[str] = set()
    for generation_row in generation_rows:
        candidate_id = str(generation_row["candidate_id"])
        if not candidate_groups or candidate_groups[-1][0] != candidate_id:
            if candidate_id in seen_candidates:
                raise AxBenchBpbAdvError(
                    f"generation rows contain non-contiguous candidate_id={candidate_id!r}"
                )
            seen_candidates.add(candidate_id)
            candidate_groups.append((candidate_id, []))
        candidate_groups[-1][1].append(generation_row)
    restored_by_id = (
        {} if checkpoint_store is None else checkpoint_store.load_probe_rows()
    )
    unknown_restored = set(restored_by_id) - seen_candidates
    if unknown_restored:
        raise AxBenchBpbAdvError(
            f"restored probe rows contain unknown candidate ids {sorted(unknown_restored)}"
        )

    def _seed_probe_result(
        key: tuple[str, str, tuple[int, ...]], scores: RoundtripFeatureScores
    ) -> None:
        existing = probe_cache.get(key)
        if existing is not None and not rows_bits_equal(
            [vars(existing)], [vars(scores)]
        ):
            raise AxBenchBpbAdvError(
                "restored probe score conflicts with another restored semantic cache entry"
            )
        probe_cache[key] = scores

    # Probe rows retain every field needed to reconstruct both score objects, so
    # seed all restored results before fresh candidates. This makes restored/fresh
    # semantic twins share work in either candidate ordering.
    for restored_rows in restored_by_id.values():
        for restored_row in restored_rows:
            feature_ids = (int(restored_row["target_feature_id"]),)
            _seed_probe_result(
                (
                    str(restored_row["roundtrip_prompt_text"]),
                    str(restored_row["generated_text"]),
                    feature_ids,
                ),
                RoundtripFeatureScores(
                    n_features=int(restored_row["n_roundtrip_features"]),
                    pre_act_mean=float(restored_row["roundtrip_pre_act"]),
                    post_act_mean=float(restored_row["roundtrip_post_act"]),
                    percentile_mean=float(restored_row["roundtrip_percentile"]),
                ),
            )
            _seed_probe_result(
                (
                    str(restored_row["control_roundtrip_prompt_text"]),
                    str(restored_row["control_generated_text"]),
                    feature_ids,
                ),
                RoundtripFeatureScores(
                    n_features=int(restored_row["control_n_roundtrip_features"]),
                    pre_act_mean=float(restored_row["control_roundtrip_pre_act"]),
                    post_act_mean=float(restored_row["control_roundtrip_post_act"]),
                    percentile_mean=float(restored_row["control_roundtrip_percentile"]),
                ),
            )

    n_empty_steered = sum(
        str(row["generated_text"]) == "" for row in generation_rows
    )
    n_empty_control = sum(
        str(row["control_generated_text"]) == "" for row in generation_rows
    )
    n_empty_skipped = sum(
        str(row["generated_text"]) == ""
        or str(row["control_generated_text"]) == ""
        for row in generation_rows
    )
    for candidate_id, candidate_generation_rows in candidate_groups:
        restored_rows = restored_by_id.get(candidate_id)
        if restored_rows is not None:
            generation_by_index = {
                int(row["sample_index"]): row for row in candidate_generation_rows
            }
            for restored_row in restored_rows:
                source = generation_by_index.get(int(restored_row["sample_index"]))
                if source is None or not rows_bits_equal(
                    [{name: restored_row[name] for name in source}], [source]
                ):
                    raise AxBenchBpbAdvError(
                        f"candidate_id={candidate_id!r} restored probe rows do not "
                        "match current generation rows"
                    )
            rows.extend(deepcopy(restored_rows))
            continue

        candidate_score_rows: list[dict[str, Any]] = []
        for generation_row in candidate_generation_rows:
            response_text = str(generation_row["generated_text"])
            control_response_text = str(generation_row["control_generated_text"])
        # A steered or control completion can legitimately come back empty when
        # steering suppresses all output (the model emits nothing). The SAE
        # reactivation probe has no text to score, so such a sample cannot
        # contribute a roundtrip measurement: skip it rather than aborting the
        # whole (multi-hour) run. The per-side and overall counts are recorded
        # so the dropped fraction is observable (see the warning below + the
        # ``empty_*`` stats). We deliberately do NOT synthesize a
        # zero-reactivation score — the percentile floor for "no activation" is
        # distribution-dependent and could mislead.
            if response_text == "" or control_response_text == "":
                continue
            target_feature_id = int(generation_row["target_feature_id"])
            record_counter(
                "axbench_bpb_adv.roundtrip.probe_requests",
                2,
                candidate_id=candidate_id,
            )
            feature_ids = (target_feature_id,)
            scores = _probe_once(
                side="steered",
                prompt_text=str(generation_row["roundtrip_prompt_text"]),
                response_text=response_text,
                feature_ids=feature_ids,
            )
            control_scores = _probe_once(
                side="control",
                prompt_text=str(generation_row["control_roundtrip_prompt_text"]),
                response_text=control_response_text,
                feature_ids=feature_ids,
            )
            row = dict(generation_row)
            row.update(
                {
                    "n_roundtrip_features": int(scores.n_features),
                    "roundtrip_pre_act": float(scores.pre_act_mean),
                    "roundtrip_post_act": float(scores.post_act_mean),
                    "roundtrip_percentile": float(scores.percentile_mean),
                    "control_n_roundtrip_features": int(control_scores.n_features),
                    "control_roundtrip_pre_act": float(control_scores.pre_act_mean),
                    "control_roundtrip_post_act": float(control_scores.post_act_mean),
                    "control_roundtrip_percentile": float(
                        control_scores.percentile_mean
                    ),
                }
            )
            candidate_score_rows.append(row)
        if checkpoint_store is not None:
            checkpoint_store.commit_probe_rows(
                candidate_id,
                candidate_score_rows,
                cast(str, runtime_fingerprint),
            )
        rows.extend(candidate_score_rows)
    total = len(generation_rows)
    if n_empty_skipped:
        logger.warning(
            "roundtrip scoring: probed {} sample(s); skipped {} of {} ({:.1f}%) "
            "for an empty steered or control completion (steering suppressed all "
            "output) — excluded from the roundtrip metric "
            "(empty_steered={}, empty_control={}).",
            len(rows),
            n_empty_skipped,
            total,
            (100.0 * n_empty_skipped / total) if total else 0.0,
            n_empty_steered,
            n_empty_control,
        )
    if stats is not None:
        stats["probe_rows"] = len(rows)
        stats["empty_skipped"] = n_empty_skipped
        stats["empty_steered"] = n_empty_steered
        stats["empty_control"] = n_empty_control
        if checkpoint_store is not None:
            stats["checkpoint_restored_probe_candidates"] = len(restored_by_id)
            stats["checkpoint_fresh_probe_candidates"] = len(candidate_groups) - len(
                restored_by_id
            )
    return pa.Table.from_pylist(rows, schema=ROUNDTRIP_SCORE_SCHEMA)


def _build_intervention_factory(
    *,
    scorer: ModelBackedDualCompletionScorer,
) -> Callable[[Mapping[str, Any]], Intervention]:
    def _build(candidate_row: Mapping[str, Any]) -> Intervention:
        method = get_method_by_name(str(candidate_row["method_name"]))
        concept = scorer.concepts_by_id.get(int(candidate_row["concept_id"]))
        if concept is None:
            raise AxBenchBpbAdvError(
                f"source concepts do not contain concept_id={candidate_row['concept_id']}"
            )
        coefficient = _coefficient_from_str(str(candidate_row["coefficient_str"]))
        return scorer._steered_intervention(
            method=method,
            concept=concept,
            coefficient=coefficient,
            candidate_row=candidate_row,
        )

    return _build


def _build_control_intervention_factory(
    *,
    scorer: ModelBackedDualCompletionScorer,
) -> Callable[[Mapping[str, Any]], Intervention]:
    def _build(candidate_row: Mapping[str, Any]) -> Intervention:
        method = get_method_by_name(str(candidate_row["method_name"]))
        concept = scorer.concepts_by_id.get(int(candidate_row["concept_id"]))
        if concept is None:
            raise AxBenchBpbAdvError(
                f"source concepts do not contain concept_id={candidate_row['concept_id']}"
            )
        return scorer._control_intervention(
            method=method,
            concept=concept,
            candidate_row=candidate_row,
        )

    return _build


def _baseline_losses_by_candidate_id(scores: pa.Table) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for row in scores.to_pylist():
        out[str(row["candidate_id"])] = (
            float(row["loss_concept_rich"]),
            float(row["loss_concept_avoiding"]),
        )
    return out


def _build_model_generate_batch_fn(
    *,
    target_bundles: Mapping[str, TargetModelBundle],
    device: str,
    lora: LoraParams | None,
) -> RoundtripGenerateBatchFn:
    def _generate(
        *,
        candidate_row: Mapping[str, Any],
        intervention: Intervention,
        inputs: RoundtripGenerationInputs,
        generation_cfg: GenerationConfig,
        sample_seeds: Sequence[tuple[int, int]],
    ) -> list[RoundtripGeneratedCompletion]:
        target_model = str(candidate_row["target_model"])
        target_bundle = target_bundles.get(target_model)
        if target_bundle is None:
            raise AxBenchBpbAdvError(
                f"no target bundle loaded for target_model={target_model!r}"
            )
        encoder_input = None
        if target_model == "encdec":
            encoder_input = (
                dict(intervention.feature_conditions)
                if intervention.feature_conditions is not None
                else {}
            )
            if target_bundle.cluster_contract is not None:
                ids = t.tensor(sorted(encoder_input), dtype=t.int64)
                try:
                    target_bundle.cluster_contract.consolidation.assert_representatives_only(
                        ids,
                        surface="roundtrip generation encoder input",
                    )
                    target_bundle.cluster_contract.feature_mask.assert_no_masked(ids)
                except Exception as exc:
                    raise AxBenchBpbAdvError(
                        "roundtrip generation reached encoder packing with an "
                        f"invalid clustered bag: {exc}"
                    ) from exc
        with profile_region(
            "axbench_bpb_adv.roundtrip.generate_completions",
            candidate_id=str(candidate_row.get("candidate_id", "")),
            target_model=target_model,
            prompt_tokens=len(inputs.prompt_tokens),
            max_new_tokens=generation_cfg.max_new_tokens,
            has_encoder_input=bool(encoder_input),
            n_samples=len(sample_seeds),
        ):
            batch_seeds = {int(seed) for _sample_index, seed in sample_seeds}
            if len(batch_seeds) != 1:
                raise AxBenchBpbAdvError(
                    "roundtrip generation requires one shared batch seed; "
                    f"observed {sorted(batch_seeds)}"
                )
            with apply_intervention(target_bundle.model, intervention, lora=lora):
                results = _generate_samples_with_nanochat_tokens(
                    model=target_bundle.model,
                    tokenizer=target_bundle.tokenizer,
                    device=device,
                    nanochat_tokens=list(inputs.prompt_tokens),
                    sampling_seed=next(iter(batch_seeds)),
                    n_samples=len(sample_seeds),
                    gen_cfg=generation_cfg,
                    encoder_input=encoder_input,
                    context_provider=inputs.context_provider,
                )
        return [
            RoundtripGeneratedCompletion(
                text=result.text,
                token_count=result.token_count,
                stop_reason=result.stop_reason,
            )
            for result in results
        ]

    return _generate


def _roundtrip_probe_stats_path(cfg: AxBenchBpbAdvConfig) -> str:
    """Resolve feature statistics for the SAE reactivation probe only.

    Direction steering reads each direction artifact's own ``max_activation``
    metadata and cannot observe this path (#1228). The roundtrip probe override
    wins when set; otherwise the probe uses the main feature-statistics path.
    """
    return (
        cfg.roundtrip_steering.feature_statistics_path or cfg.feature_statistics_path
    )


def run_model_backed_roundtrip_score_stage(
    *,
    cfg: AxBenchBpbAdvConfig,
    paths: AxBenchBpbAdvPaths,
    candidates: pa.Table,
    stats: dict[str, int] | None = None,
    cluster_contract: ClusterFeatureContract | None = None,
    stage_source: ClusteredStageSource | None = None,
    candidate_snapshot: ClusteredCandidateSnapshot | None = None,
) -> pa.Table:
    resolved_stage_source = require_clustered_stage_source(
        stage_source=stage_source,
        contract=cluster_contract,
        surface="roundtrip score stage",
    )
    source_validation = (
        None
        if resolved_stage_source is None
        else resolved_stage_source.validation
    )
    resolved_candidate_snapshot = require_clustered_candidate_snapshot(
        candidate_snapshot,
        paths=paths,
        contract=cluster_contract,
        source_identity=(
            None
            if resolved_stage_source is None
            else resolved_stage_source.source_identity
        ),
        surface="roundtrip score stage",
        table=candidates,
    )
    if cfg.run_tag is None:
        raise AxBenchBpbAdvError("cfg.run_tag must be set before scoring")
    if candidates.num_rows == 0:
        raise AxBenchBpbAdvError("no candidates to score")
    # Require the method-independent held-out pair population BEFORE
    # anything downstream consumes the candidates, so baseline BPB and generated
    # round-trip rows see the SAME exact set as class-margin scoring.
    record_counter("axbench_bpb_adv.roundtrip.input_candidate_rows", candidates.num_rows)
    with profile_region(
        "axbench_bpb_adv.roundtrip.population_guard",
        rows=candidates.num_rows,
    ):
        candidates = require_table_in_evaluation_population(
            candidates,
            population=read_evaluation_population(
                paths.evaluation_population(),
                **(
                    {}
                    if cluster_contract is None
                    else {
                        "expected_feature_contract_fingerprint": (
                            cluster_contract.feature_contract_fingerprint
                        )
                    }
                ),
            ),
            label="roundtrip score-stage candidates",
            stats=stats,
        )
    # Bag-backed rows get an additional family artifact guard: after the universal
    # population check, a missing per-family scored prefix is a derive bug.
    with profile_region(
        "axbench_bpb_adv.roundtrip.precomputed_bag_coverage_guard",
        rows=candidates.num_rows,
    ):
        candidates = require_precomputed_bag_candidates_cover_scored_prefixes(
            candidates,
            paths=paths,
            stats=stats,
            **(
                {}
                if cluster_contract is None
                else {
                    "feature_contract_fingerprint": (
                        cluster_contract.feature_contract_fingerprint
                    )
                }
            ),
        )
    candidate_rows = candidates.to_pylist()
    if not cfg.roundtrip_steering.enabled:
        raise AxBenchBpbAdvError("roundtrip scoring requires roundtrip_steering.enabled")
    probe_stats_path = _roundtrip_probe_stats_path(cfg)
    if not probe_stats_path:
        raise AxBenchBpbAdvError("roundtrip scoring requires feature_statistics_path")

    generation_cfg = _generation_config(cfg)
    bpb_cfg = _bpb_config_from_adv_config(cfg)
    targets = {str(row["target_model"]) for row in candidate_rows}
    unknown_targets = targets - {"decoder_only", "encdec"}
    if unknown_targets:
        raise AxBenchBpbAdvError(
            f"unsupported target_model values: {sorted(unknown_targets)}"
        )
    with profile_region(
        "axbench_bpb_adv.roundtrip.checkpoint_identity",
        n_targets=len(targets),
    ):
        resolved_targets = resolve_and_pin_target_checkpoints(bpb_cfg, targets)
        parsed_feature_dir = parse_feature_dir(Path(bpb_cfg.feature_dir))
        reference_segment_size_by_target = {
            target: int(parsed_feature_dir.segment_size) for target in sorted(targets)
        }
        if resolved_candidate_snapshot is None:
            candidates_sha256 = candidates_payload_sha256(
                candidates=candidates,
                paths=paths,
            )
        else:
            candidates_sha256 = candidates_payload_sha256(
                candidates=candidates,
                paths=paths,
                snapshot_payload=resolved_candidate_snapshot.payload,
                snapshot_sha256=resolved_candidate_snapshot.sha256,
            )
        checkpoint_fingerprint = build_roundtrip_checkpoint_fingerprint(
            cfg=cfg,
            paths=paths,
            candidates_sha256=candidates_sha256,
            candidate_rows=candidate_rows,
            cluster_feature_contract_fingerprint=(
                None
                if cluster_contract is None
                else cluster_contract.feature_contract_fingerprint
            ),
            resolved_targets=resolved_targets,
            reference_segment_size_by_target=reference_segment_size_by_target,
            probe_statistics_path=probe_stats_path,
        )
        checkpoint_store = RoundtripCheckpointStore.prepare(
            checkpoint_dir=paths.roundtrip_checkpoint_dir(),
            fingerprint=checkpoint_fingerprint,
            expected_candidate_rows=candidate_rows,
            samples_per_candidate=cfg.roundtrip_generation.samples_per_candidate,
        )
    runtime_kwargs: dict[str, Any] = {
        "cfg": bpb_cfg,
        "ddp_rank": 0,
        "ddp_world_size": 1,
    }
    if cluster_contract is not None:
        runtime_kwargs["cluster_contract"] = cluster_contract
    runtime_state = RuntimeState(**runtime_kwargs)
    _runtime_payload, generation_runtime_fingerprint = current_runtime_fingerprint(
        runtime_state.target_device()
    )
    warn_on_roundtrip_runtime_mismatch(
        checkpoint_store,
        CheckpointFamily.GENERATION,
        generation_runtime_fingerprint,
    )
    logger.info(
        "roundtrip checkpoint summary: expected={} restored_generation={} "
        "restored_probe={} pending={}",
        checkpoint_store.expected_count,
        checkpoint_store.restored_generation_count,
        checkpoint_store.restored_probe_count,
        checkpoint_store.pending_count,
    )
    checkpoint_counts = {
        "checkpoint_expected_candidates": checkpoint_store.expected_count,
        "checkpoint_restored_generation_candidates": (
            checkpoint_store.restored_generation_count
        ),
        "checkpoint_restored_probe_candidates": checkpoint_store.restored_probe_count,
        "checkpoint_pending_candidates": checkpoint_store.pending_count,
    }
    for name, value in checkpoint_counts.items():
        record_counter(f"axbench_bpb_adv.roundtrip.{name}", value)
        if stats is not None:
            stats[name] = value
    source_paths = SourcePaths(Path(cfg.source_run_dir))
    with profile_region("axbench_bpb_adv.roundtrip.concept_probe_preflight"):
        concepts_by_id = (
            _read_concepts(source_paths)
            if source_validation is None
            else _concepts_from_entries(
                source_validation.concept_entries()
            )
        )
        _validate_roundtrip_probe_matches_concepts(
            candidate_rows=candidate_rows,
            concepts_by_id=concepts_by_id,
            probe_config=cfg.roundtrip_steering.probe,
        )
    generation_rows: list[dict[str, Any]]
    try:
        with profile_region(
            "axbench_bpb_adv.roundtrip.target_runtime_setup",
            n_targets=len(targets),
        ):
            target_bundles: dict[str, TargetModelBundle] = {}
            if "decoder_only" in targets:
                target_bundles["decoder_only"] = runtime_state.decoder_only()
            if "encdec" in targets:
                target_bundles["encdec"] = runtime_state.encdec()
        unloaded_targets = targets - set(target_bundles)
        if unloaded_targets:
            raise AxBenchBpbAdvError(
                f"unsupported target_model values: {sorted(unloaded_targets)}"
            )
        assert_target_checkpoints_unchanged(resolved_targets)
        tokenizer = next(iter(target_bundles.values())).tokenizer
        with profile_region("axbench_bpb_adv.roundtrip.tokenizer_setup"):
            extraction_tokenizer = get_extraction_tokenizer(cfg.extraction_tokenizer_id)
            token_bytes = _build_token_bytes(tokenizer)
        with profile_region("axbench_bpb_adv.roundtrip.context_segments_load"):
            context_segments_by_hash = (
                load_context_segments_by_prefix_hash(
                    paths,
                    **(
                        {}
                        if cluster_contract is None
                        else {"cluster_contract": cluster_contract}
                    ),
                )
                if "encdec" in targets
                else {}
            )
        record_counter(
            "axbench_bpb_adv.roundtrip.context_prefixes_loaded",
            len(context_segments_by_hash),
        )
        detect_kwargs = (
            {}
            if cluster_contract is None
            else {"cluster_contract": cluster_contract}
        )
        target_metas = {
            target: detect_target_model_metadata(
                bundle.model,
                bpb_cfg,
                **detect_kwargs,
            )
            for target, bundle in target_bundles.items()
        }
        loaded_reference_segment_size_by_target = {
            target: int(meta.segment_size) for target, meta in target_metas.items()
        }
        if loaded_reference_segment_size_by_target != reference_segment_size_by_target:
            raise AxBenchBpbAdvError(
                "roundtrip target reference segment sizes differ from the checkpoint "
                f"manifest identity: expected {reference_segment_size_by_target!r}, "
                f"observed {loaded_reference_segment_size_by_target!r}"
            )
        lora = (
            LoraParams(
                cfg=LoraConfig(**cfg.lora_adapter),
                layer=cfg.diffmean_and_probe_layer,
            )
            if cfg.lora_adapter
            else None
        )
        additive_cfg = get_additive_direction_config(cfg)
        with profile_region("axbench_bpb_adv.roundtrip.direction_scale_preflight"):
            _preflight_direction_scales_for_rows(
                candidate_rows=candidate_rows,
                concepts_by_id=concepts_by_id,
                source_paths=source_paths,
                expected_layer=cfg.diffmean_and_probe_layer,
                counter_direction_family=(
                    additive_cfg.counter_direction_family
                    if additive_cfg is not None
                    else "diffmean"
                ),
                source_validation=source_validation,
                stage="roundtrip",
            )
        scorer_kwargs: dict[str, Any] = (
            {}
            if cluster_contract is None
            else {
                "cluster_contract": cluster_contract,
                "source_validation": source_validation,
            }
        )
        scorer = ModelBackedDualCompletionScorer(
            target_bundles=target_bundles,
            target_model_metas=target_metas,
            concepts_by_id=concepts_by_id,
            source_paths=source_paths,
            tokenizer=tokenizer,
            device=t.device(runtime_state.target_device()),
            diffmean_layer=cfg.diffmean_and_probe_layer,
            inactive_sampler=(
                runtime_state.inactive_sampler() if "encdec" in targets else None
            ),
            apply_max_act_scale=True,
            lora=lora,
            counter_direction_family=(
                additive_cfg.counter_direction_family
                if additive_cfg is not None
                else "diffmean"
            ),
            # GCG Phase-2: the run artifact paths _build_ours_intervention needs to
            # locate the precomputed conditioning bags. Without this the four
            # ours_gcg_* methods crash the DEFAULT roundtrip score stage (the fix for
            # the Task-10 CRITICAL). Mirrors run_model_backed_bpb_adv_score_stage.
            adv_paths=paths,
            **scorer_kwargs,
        )
        with profile_region(
            "axbench_bpb_adv.roundtrip.baseline_bpb_score_table",
            rows=candidates.num_rows,
        ):
            bpb_scores = build_bpb_adv_score_table(
                candidates=candidates,
                context_segments_by_prefix_hash=context_segments_by_hash,
                tokenizer=tokenizer,
                token_bytes=token_bytes,
                max_seq_len_by_target={
                    target: bundle.max_seq_len for target, bundle in target_bundles.items()
                },
                score_pair=scorer,
                stats=stats,
            )
        baseline_losses_by_candidate_id = _baseline_losses_by_candidate_id(bpb_scores)
        del bpb_scores
        with profile_region(
            "axbench_bpb_adv.roundtrip.generation_rows",
            rows=candidates.num_rows,
        ):
            generation_rows = build_roundtrip_generation_rows(
                candidates=candidates,
                context_segments_by_prefix_hash=context_segments_by_hash,
                tokenizer=tokenizer,
                extraction_tokenizer=extraction_tokenizer,
                concepts_by_id=concepts_by_id,
                generation_cfg=generation_cfg,
                reference_segment_size_by_target=reference_segment_size_by_target,
                samples_per_candidate=cfg.roundtrip_generation.samples_per_candidate,
                master_seed=cfg.roundtrip_generation.master_seed,
                baseline_losses_by_candidate_id=baseline_losses_by_candidate_id,
                build_intervention_for_row=_build_intervention_factory(scorer=scorer),
                build_control_intervention_for_row=_build_control_intervention_factory(
                    scorer=scorer
                ),
                generate_completions=_build_model_generate_batch_fn(
                    target_bundles=target_bundles,
                    device=runtime_state.target_device(),
                    lora=lora,
                ),
                stats=stats,
                log_progress_every=cfg.log_progress_every,
                cluster_contract=cluster_contract,
                checkpoint_store=checkpoint_store,
                runtime_fingerprint=generation_runtime_fingerprint,
            )
    finally:
        runtime_state.offload_models()
        if t.cuda.is_available():
            t.cuda.empty_cache()

    with profile_region("axbench_bpb_adv.roundtrip.probe_load"):
        resolved_probe_device = resolve_probe_device(
            cfg.roundtrip_steering.probe.device
        )
        _probe_runtime_payload, probe_runtime_fingerprint = current_runtime_fingerprint(
            resolved_probe_device
        )
        warn_on_roundtrip_runtime_mismatch(
            checkpoint_store,
            CheckpointFamily.PROBE,
            probe_runtime_fingerprint,
        )
        _validate_roundtrip_stats_match_probe(
            stats_path=probe_stats_path,
            probe_config=cfg.roundtrip_steering.probe,
        )
        percentile_table, _checksum = load_percentile_table_for_roundtrip(
            probe_stats_path
        )
        probe_model = load_probe_model_for_roundtrip(cfg.roundtrip_steering.probe)
    try:
        with profile_region(
            "axbench_bpb_adv.roundtrip.probe_score_rows",
            rows=len(generation_rows),
        ):
            return score_roundtrip_generation_rows(
                generation_rows=generation_rows,
                probe_model=probe_model,
                percentile_table=percentile_table,
                feature_chunk_size=cfg.roundtrip_steering.feature_chunk_size,
                stats=stats,
                checkpoint_store=checkpoint_store,
                runtime_fingerprint=probe_runtime_fingerprint,
            )
    finally:
        del probe_model
        if t.cuda.is_available():
            t.cuda.empty_cache()
