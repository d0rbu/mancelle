"""Prepare AxBench-BPB-Adv attack rows from Stage 0-3 source artifacts."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal

import pyarrow as pa
from beartype import beartype
from beartype.vale import Is
from loguru import logger

from steering._common.strict_json import canonical_json_str_fallback
from steering.conditional.feature_gcg.edit_primitives import (
    EditableSpan,
    token_sequence_digest,
)
from steering.eval.axbench_base.method_registry import (
    ALL_METHODS,
    Method,
    filter_methods,
)
from steering.eval.axbench_base.types import AxBenchBaseError, TargetModelMetadata
from steering.eval.axbench_bpb.representative_selection import (
    FAIR_FEATURE_DIFFMEAN_METHOD_BY_MODE,
)
from steering.eval.axbench_bpb.sample_loader import make_prefix_hash, make_sample_id
from steering.eval.axbench_bpb.tokenization import encode_output, encode_prefix
from steering.eval.axbench_bpb.types import SteeringMode
from steering.eval.axbench_bpb_adv.config import (
    AdditiveDirectionConfig,
    AdversarialMethod,
    AdversarialMethodConfig,
    AxBenchBpbAdvConfig,
    ContextSignalMode,
    FasterGcgConfig,
    GepaConfig,
    adversarial_method_configs_by_name,
    method_config_name,
)
from steering.eval.axbench_bpb_adv.editable_tokens import (
    select_final_k_editable_span,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.schemas import ATTACK_ROWS_SCHEMA
from steering.refined_types import (
    NonEmptyNonBlankStrListLike,
    NonEmptyNonNegativeIntListLike,
    NonEmptyNonNegativeIntListListLike,
    NonEmptyStr,
    NonEmptyStrLike,
    NonNegativeInt,
    NonNegativeIntLike,
    NonNegativeIntListLike,
)


@dataclass(frozen=True, init=False)
class BpbAttackWorkSpec:
    method_name: NonEmptyStr
    mode: SteeringMode
    target_model: NonEmptyStr
    coefficient_str: NonEmptyStr
    baseline_family: NonEmptyStr
    target_only: bool

    @beartype
    def __init__(
        self,
        method_name: NonEmptyStrLike,
        mode: SteeringMode,
        target_model: NonEmptyStrLike,
        coefficient_str: NonEmptyStrLike,
        baseline_family: NonEmptyStrLike,
        target_only: bool = False,
    ) -> None:
        object.__setattr__(self, "method_name", NonEmptyStr.parse(method_name))
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "target_model", NonEmptyStr.parse(target_model))
        object.__setattr__(
            self,
            "coefficient_str",
            NonEmptyStr.parse(coefficient_str),
        )
        object.__setattr__(
            self,
            "baseline_family",
            NonEmptyStr.parse(baseline_family),
        )
        object.__setattr__(self, "target_only", target_only)


@dataclass(frozen=True, init=False)
class BpbAttackSample:
    sample_id: NonEmptyStr
    concept_id: NonNegativeInt
    mode: SteeringMode
    content_class: Literal["concept_rich", "concept_avoiding"]
    prefix_text: str
    prefix_hash: NonEmptyStr
    prefix_tokens: NonEmptyNonNegativeIntListLike
    output_tokens: NonNegativeIntListLike

    @beartype
    def __init__(
        self,
        sample_id: NonEmptyStrLike,
        concept_id: NonNegativeIntLike,
        mode: SteeringMode,
        content_class: Literal["concept_rich", "concept_avoiding"],
        prefix_text: str,
        prefix_hash: NonEmptyStrLike,
        prefix_tokens: NonEmptyNonNegativeIntListLike,
        output_tokens: NonNegativeIntListLike,
    ) -> None:
        object.__setattr__(self, "sample_id", NonEmptyStr.parse(sample_id))
        object.__setattr__(self, "concept_id", NonNegativeInt.parse(concept_id))
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "content_class", content_class)
        object.__setattr__(self, "prefix_text", prefix_text)
        object.__setattr__(self, "prefix_hash", NonEmptyStr.parse(prefix_hash))
        object.__setattr__(self, "prefix_tokens", prefix_tokens)
        object.__setattr__(self, "output_tokens", output_tokens)


@dataclass(frozen=True, init=False)
class BpbAttackPrefix:
    """One core prefix carrying ALL AxBench class-margin completions.

    A core prefix is the (concept_id, mode, prefix_text) triple keyed by
    ``core_prefix_id``. It owns EVERY ``concept_rich`` and ``concept_avoiding``
    completion of the SAME prefix (Gap B: the synth pipeline emits N rich and M
    avoid samples per prefix). Each (rich_i, avoid_j) pair is later teacher-forced
    under the same steered intervention to compute the signed ``primary_margin_y``;
    ``prepare_bpb_attack_rows`` explodes the Cartesian product into one attack row
    per pair (optionally subsampled). The per-class fields are therefore LISTS,
    index-aligned (``concept_rich_sample_ids[i]`` labels
    ``concept_rich_output_tokens[i]``).
    """

    core_prefix_id: NonEmptyStr
    concept_id: NonNegativeInt
    mode: SteeringMode
    prefix_text: str
    prefix_hash: NonEmptyStr
    prefix_tokens: NonEmptyNonNegativeIntListLike
    concept_rich_sample_ids: NonEmptyNonBlankStrListLike
    concept_avoiding_sample_ids: NonEmptyNonBlankStrListLike
    concept_rich_output_tokens: NonEmptyNonNegativeIntListListLike
    concept_avoiding_output_tokens: NonEmptyNonNegativeIntListListLike

    @beartype
    def __init__(
        self,
        core_prefix_id: NonEmptyStrLike,
        concept_id: NonNegativeIntLike,
        mode: SteeringMode,
        prefix_text: str,
        prefix_hash: NonEmptyStrLike,
        prefix_tokens: NonEmptyNonNegativeIntListLike,
        concept_rich_sample_ids: NonEmptyNonBlankStrListLike,
        concept_avoiding_sample_ids: NonEmptyNonBlankStrListLike,
        concept_rich_output_tokens: NonEmptyNonNegativeIntListListLike,
        concept_avoiding_output_tokens: NonEmptyNonNegativeIntListListLike,
    ) -> None:
        object.__setattr__(self, "core_prefix_id", NonEmptyStr.parse(core_prefix_id))
        object.__setattr__(self, "concept_id", NonNegativeInt.parse(concept_id))
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "prefix_text", prefix_text)
        object.__setattr__(self, "prefix_hash", NonEmptyStr.parse(prefix_hash))
        object.__setattr__(self, "prefix_tokens", prefix_tokens)
        object.__setattr__(
            self,
            "concept_rich_sample_ids",
            concept_rich_sample_ids,
        )
        object.__setattr__(
            self,
            "concept_avoiding_sample_ids",
            concept_avoiding_sample_ids,
        )
        object.__setattr__(
            self,
            "concept_rich_output_tokens",
            concept_rich_output_tokens,
        )
        object.__setattr__(
            self,
            "concept_avoiding_output_tokens",
            concept_avoiding_output_tokens,
        )
        if len(self.concept_rich_sample_ids) != len(self.concept_rich_output_tokens):
            raise AxBenchBpbAdvError(
                "concept_rich_sample_ids and concept_rich_output_tokens must have equal lengths"
            )
        if len(self.concept_avoiding_sample_ids) != len(self.concept_avoiding_output_tokens):
            raise AxBenchBpbAdvError(
                "concept_avoiding_sample_ids and "
                "concept_avoiding_output_tokens must have equal lengths"
            )


def _is_attack_sample_sequence(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and all(isinstance(item, BpbAttackSample) for item in value)
    )


def _is_non_empty_attack_prefix_sequence(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and bool(value)
        and all(isinstance(item, BpbAttackPrefix) for item in value)
    )


def _is_attack_prefix_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, BpbAttackPrefix) for item in value)


def _is_non_empty_attack_work_spec_sequence(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and bool(value)
        and all(isinstance(item, BpbAttackWorkSpec) for item in value)
    )


if TYPE_CHECKING:
    _BpbAttackPrefixListLike = list[BpbAttackPrefix]
    _BpbAttackSampleSequenceLike = Sequence[BpbAttackSample]
    _NonEmptyBpbAttackPrefixSequenceLike = Sequence[BpbAttackPrefix]
    _NonEmptyBpbAttackWorkSpecSequenceLike = Sequence[BpbAttackWorkSpec]
else:
    _BpbAttackPrefixListLike = Annotated[
        list,
        Is[_is_attack_prefix_list],
    ]
    _BpbAttackSampleSequenceLike = Annotated[
        Sequence,
        Is[_is_attack_sample_sequence],
    ]
    _NonEmptyBpbAttackPrefixSequenceLike = Annotated[
        Sequence,
        Is[_is_non_empty_attack_prefix_sequence],
    ]
    _NonEmptyBpbAttackWorkSpecSequenceLike = Annotated[
        Sequence,
        Is[_is_non_empty_attack_work_spec_sequence],
    ]


def _dedup_completions_by_digest(
    samples: _BpbAttackSampleSequenceLike,
) -> list[BpbAttackSample]:
    """Collapse completions with byte-identical ``output_tokens`` to one each.

    Distinct completions that emit identical output tokens (the LLM produced the
    same text twice on one side) carry the same ``token_sequence_digest``.
    Because downstream ids key on (core_prefix_id, rich_output_digest,
    avoid_output_digest, ...) WITHOUT a per-pair index, two such completions
    would explode into Cartesian pairs that collapse onto a single id and trip
    the ``duplicate candidate_id`` guard. Averaging over redundant completions
    adds no information, so we keep the FIRST occurrence (stable order) and drop
    later duplicates. Identical ``sample_id``s collapse too, since a repeated
    sample necessarily repeats its output digest.
    """

    seen: set[str] = set()
    kept: list[BpbAttackSample] = []
    for sample in samples:
        digest = token_sequence_digest(sample.output_tokens)
        if digest in seen:
            continue
        seen.add(digest)
        kept.append(sample)
    return kept


@beartype
def core_prefix_id(
    *,
    concept_id: NonNegativeIntLike,
    mode: SteeringMode,
    prefix_text: str,
) -> str:
    """16-hex-char SHA256 derivative identifying a core prefix.

    The two completions of a prefix have distinct ``sample_id``s (they differ
    by ``content_class`` and ``output_text``), so attack/candidate/score rows
    key on this prefix-level id instead. CONTRACT: changing the field order,
    separator, encoding, or truncation length changes aggregation identity.
    """

    payload = f"{concept_id}|{mode.value}|{prefix_text}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


@beartype
def pair_samples_into_prefixes(
    samples: _BpbAttackSampleSequenceLike,
) -> _BpbAttackPrefixListLike:
    """Group samples by (concept_id, mode, prefix_text) into core prefixes.

    Each group MUST contain at least one ``concept_rich`` and at least one
    ``concept_avoiding`` sample, ALL sharing identical ``prefix_tokens`` (Gap B:
    N rich x M avoid completions of one prefix are all carried). A group missing
    an entire side is always a source error.

    The same-prefix-tokens invariant is enforced with an ``assert`` across ALL
    rich+avoid completions because a completion that disagrees on prefix tokens
    is a synth-pipeline bug, not user input.
    """

    grouped: dict[tuple[int, SteeringMode, str], dict[str, list[BpbAttackSample]]] = {}
    for sample in samples:
        key = (sample.concept_id, sample.mode, sample.prefix_text)
        grouped.setdefault(key, {}).setdefault(sample.content_class, []).append(sample)

    prefixes: list[BpbAttackPrefix] = []
    for key, by_class in grouped.items():
        concept_id, mode, prefix_text = key
        rich = by_class.get("concept_rich", [])
        avoid = by_class.get("concept_avoiding", [])
        problems: list[str] = []
        if len(rich) < 1:
            problems.append("concept_rich x0")
        if len(avoid) < 1:
            problems.append("concept_avoiding x0")
        if problems:
            raise AxBenchBpbAdvError(
                f"core prefix (concept_id={concept_id}, mode={mode.value!r}, "
                f"prefix_text={prefix_text!r}) requires at least one concept_rich "
                f"and one concept_avoiding sample, got {', '.join(problems)}"
            )

        prefix_tokens = list(rich[0].prefix_tokens)
        for sample in (*rich, *avoid):
            assert sample.prefix_tokens == prefix_tokens, (
                f"core prefix (concept_id={concept_id}, mode={mode.value!r}, "
                f"prefix_text={prefix_text!r}) has a completion whose "
                f"prefix_tokens disagree: {sample.prefix_tokens} != {prefix_tokens}"
            )
        # Collapse redundant identical completions per side BEFORE the Cartesian
        # explosion so every (rich_i, avoid_j) pair carries a distinct
        # (rich_digest, avoid_digest) tuple -> distinct attack/candidate ids.
        rich = _dedup_completions_by_digest(rich)
        avoid = _dedup_completions_by_digest(avoid)
        prefixes.append(
            BpbAttackPrefix(
                core_prefix_id=core_prefix_id(
                    concept_id=concept_id, mode=mode, prefix_text=prefix_text
                ),
                concept_id=concept_id,
                mode=mode,
                prefix_text=prefix_text,
                prefix_hash=rich[0].prefix_hash,
                prefix_tokens=prefix_tokens,
                concept_rich_sample_ids=[sample.sample_id for sample in rich],
                concept_avoiding_sample_ids=[sample.sample_id for sample in avoid],
                concept_rich_output_tokens=[list(sample.output_tokens) for sample in rich],
                concept_avoiding_output_tokens=[list(sample.output_tokens) for sample in avoid],
            )
        )

    return prefixes


def _attack_row_id(payload: Mapping[str, Any]) -> str:
    fields = {
        "core_prefix_id": payload["core_prefix_id"],
        "concept_rich_output_digest": payload["concept_rich_output_digest"],
        "concept_avoiding_output_digest": payload["concept_avoiding_output_digest"],
        "method_name": payload["method_name"],
        "target_model": payload["target_model"],
        "coefficient_str": payload["coefficient_str"],
        "adversarial_method": payload["adversarial_method"],
        "adversarial_strength": payload["adversarial_strength"],
        "strength_unit": payload["strength_unit"],
        "strength_index": payload["strength_index"],
        "context_signal_mode": payload["context_signal_mode"],
        "source_prompt_token_digest": payload["source_prompt_token_digest"],
    }
    return hashlib.sha256(canonical_json_str_fallback(fields).encode("utf-8")).hexdigest()


def bpb_attack_sample_from_synth_row(
    row: dict[str, Any],
    *,
    mode: SteeringMode | str,
    tokenizer: Any,
) -> BpbAttackSample:
    resolved_mode = SteeringMode(mode)
    prefix_text = row["prefix_text"]
    output_text = row["output_text"]
    content_class = row["content_class"]
    try:
        concept_id = NonNegativeInt.parse(row["concept_id"])
    except TypeError as exc:
        raise AxBenchBpbAdvError("synth row concept_id must be a non-negative integer") from exc
    if not isinstance(prefix_text, str):
        raise AxBenchBpbAdvError("synth row prefix_text must be a string")
    if not isinstance(output_text, str):
        raise AxBenchBpbAdvError("synth row output_text must be a string")
    if not isinstance(content_class, str):
        raise AxBenchBpbAdvError("synth row content_class must be a string")
    if content_class not in {"concept_rich", "concept_avoiding"}:
        raise AxBenchBpbAdvError(
            "synth row content_class must be 'concept_rich' or "
            f"'concept_avoiding', got {content_class!r}"
        )
    return BpbAttackSample(
        sample_id=make_sample_id(
            concept_id=concept_id,
            mode=resolved_mode,
            content_class=content_class,
            prefix_text=prefix_text,
            output_text=output_text,
        ),
        concept_id=concept_id,
        mode=resolved_mode,
        content_class=content_class,
        prefix_text=prefix_text,
        prefix_hash=make_prefix_hash(prefix_text),
        prefix_tokens=encode_prefix(prefix_text, tokenizer),
        output_tokens=encode_output(output_text, tokenizer),
    )


def _known_method_names() -> set[str]:
    return {method.name for method in ALL_METHODS}


def _validate_base_methods(base_methods: list[str] | None) -> None:
    if base_methods is None:
        return
    known = _known_method_names()
    unknown = sorted(set(base_methods) - known)
    if unknown:
        raise AxBenchBpbAdvError(
            f"unknown BPB-Adv base method(s): {unknown}. "
            f"Available methods: {sorted(known)}"
        )


def _selected_methods_for_targets(
    *,
    base_methods: list[str] | None,
    target_set: set[str],
    has_negative_sampling: bool,
) -> list[Method]:
    _validate_base_methods(base_methods)
    try:
        methods = filter_methods(
            list(ALL_METHODS),
            enabled_methods=base_methods,
            disabled_methods=[],
            target_model_has_negative_sampling=has_negative_sampling,
        )
    except AxBenchBaseError as exc:
        raise AxBenchBpbAdvError(str(exc)) from exc
    if base_methods is not None:
        missing = sorted(set(base_methods) - {method.name for method in methods})
        if missing:
            raise AxBenchBpbAdvError(
                "requested BPB-Adv base method(s) are unavailable with the "
                f"current source/model metadata: {missing}"
            )
        wrong_target = sorted(
            method.name for method in methods if method.target_model not in target_set
        )
        if wrong_target:
            raise AxBenchBpbAdvError(
                "requested BPB-Adv base method(s) target models not listed in "
                f"target_models={sorted(target_set)}: {wrong_target}"
            )
        return methods
    return [method for method in methods if method.target_model in target_set]


def work_specs_from_method_registry(
    *,
    base_methods: list[str] | None,
    target_models: Iterable[str],
    target_model_metas: Mapping[str, TargetModelMetadata],
    cfg: AxBenchBpbAdvConfig,
) -> list[BpbAttackWorkSpec]:
    """Build method/coefficient work directly from AxBench-Base metadata."""

    target_set = {str(target) for target in target_models}
    if not target_set:
        raise AxBenchBpbAdvError("target_models must contain at least one target")
    unknown_targets = target_set - set(target_model_metas)
    if unknown_targets:
        raise AxBenchBpbAdvError(
            f"missing target model metadata for {sorted(unknown_targets)}"
        )
    has_negative_sampling = any(
        meta.negative_selection is not None for meta in target_model_metas.values()
    )
    methods = _selected_methods_for_targets(
        base_methods=base_methods,
        target_set=target_set,
        has_negative_sampling=has_negative_sampling,
    )
    representative_target_methods = list(cfg.representative_target_methods)
    requested_target_methods = (
        _selected_methods_for_targets(
            base_methods=representative_target_methods,
            target_set=target_set,
            has_negative_sampling=has_negative_sampling,
        )
        if representative_target_methods
        else []
    )
    active_method_names = {method.name for method in methods}
    target_only_methods = [
        method for method in requested_target_methods if method.name not in active_method_names
    ]
    method_specs: list[tuple[Method, bool]] = [(method, False) for method in methods]
    method_specs.extend((method, True) for method in target_only_methods)

    out: list[BpbAttackWorkSpec] = []
    for method, target_only in method_specs:
        meta = target_model_metas[method.target_model]
        coefficients = method.coefficient_grid_fn(
            meta,
            cfg,
        )
        for coefficient in coefficients:
            out.append(
                BpbAttackWorkSpec(
                    method_name=method.name,
                    mode=method.mode,
                    target_model=method.target_model,
                    coefficient_str=str(coefficient),
                    baseline_family=str(method.baseline.value),
                    target_only=target_only,
                )
            )
    if not out:
        raise AxBenchBpbAdvError(
            "no BPB-Adv method/coefficient specs survived method and target filters"
        )
    return out


def _prompt_attack_specs(
    *,
    method_config: AdversarialMethodConfig,
) -> list[tuple[float, str, int, int]]:
    adversarial_method = method_config_name(method_config)
    if adversarial_method == AdversarialMethod.FASTER_GCG:
        if not isinstance(method_config, FasterGcgConfig):
            raise AxBenchBpbAdvError("faster_gcg config has the wrong type")
        return [
            (float(k), "optimized_tokens", idx, int(k))
            for idx, k in enumerate(method_config.optimized_token_counts)
        ]
    if adversarial_method == AdversarialMethod.GEPA:
        if not isinstance(method_config, GepaConfig):
            raise AxBenchBpbAdvError("gepa config has the wrong type")
        return [
            (float(budget), "metric_calls", idx, method_config.k_editable_tokens)
            for idx, budget in enumerate(method_config.metric_call_budgets)
        ]
    raise AxBenchBpbAdvError(f"{adversarial_method.value} is not a prompt-token method")


def _additive_specs(
    method_config: AdditiveDirectionConfig,
) -> list[tuple[float, str, int, int]]:
    return [
        (float(magnitude), "counter_direction_magnitude", idx, 0)
        for idx, magnitude in enumerate(method_config.magnitudes)
    ]


def _synthetic_control_spec(
    specs: Sequence[tuple[float, str, int, int]],
) -> tuple[float, str, int, int]:
    """A canonical no-adversary control spec (strength 0, empty editable span).

    GEPA's editable-token count is config-validated ``>= 1``, so its grid can
    never contain a no-adversary control the way faster_gcg (``k == 0``) or
    additive (magnitude ``0.0``) can. When the fair feature-DiffMean target is
    crossed with GEPA we synthesize one -- tagged with GEPA's strength unit -- so
    the target keeps its no-adversary reference instead of aborting a legitimate
    GEPA + feature-DiffMean run. The row scores as a pure no-op: strength 0 with
    an empty editable span routes to the control-candidate builder, never the
    GEPA optimizer, and its margin is independent of ``adversarial_method``.
    """
    unit = specs[0][1] if specs else "metric_calls"
    return (0.0, unit, 0, 0)


def _has_control_spec(specs: Sequence[tuple[float, str, int, int]]) -> bool:
    return any(
        float(strength) == 0.0 and int(editable_count) == 0
        for strength, _unit, _index, editable_count in specs
    )


def _control_specs_for_target(
    specs: Sequence[tuple[float, str, int, int]],
    *,
    synthesize: bool,
) -> list[tuple[float, str, int, int]]:
    """One no-adversary control spec for a control-only (target-only) method.

    Uses the grid's own control when present; otherwise synthesizes one iff the
    adversarial method structurally cannot express it (``synthesize=True`` for
    GEPA). faster_gcg / additive can configure an explicit ``0``, so they RAISE
    LOUDLY rather than silently papering over a missing control.
    """
    if not specs:
        raise AxBenchBpbAdvError(
            "representative target method has no adversarial strength specs"
        )
    for strength, unit, index, editable_count in specs:
        if float(strength) == 0.0 and int(editable_count) == 0:
            return [(float(strength), unit, int(index), 0)]
    if synthesize:
        return [_synthetic_control_spec(specs)]
    raise AxBenchBpbAdvError(
        "representative target method requires an explicit no-adversary "
        "control strength"
    )


def _grid_with_control_spec(
    specs: Sequence[tuple[float, str, int, int]],
    *,
    synthesize: bool,
) -> list[tuple[float, str, int, int]]:
    """Full adversarial grid PLUS a guaranteed no-adversary control spec.

    When the grid already contains a control it is returned unchanged (identity
    for the faster_gcg-with-0 / additive-with-0.0 paths). Otherwise a control is
    synthesized iff ``synthesize`` (GEPA only); the control sorts first by
    ``strength_index`` (strength 0) and the real grid shifts up one ordinal so
    ``strength_index`` stays a unique, strength-ordered key per series. Methods
    that can configure an explicit ``0`` RAISE LOUDLY instead.
    """
    if not specs:
        raise AxBenchBpbAdvError(
            "representative target method has no adversarial strength specs"
        )
    if _has_control_spec(specs):
        return list(specs)
    if synthesize:
        shifted = [
            (strength, unit, int(index) + 1, editable_count)
            for strength, unit, index, editable_count in specs
        ]
        return [_synthetic_control_spec(specs), *shifted]
    raise AxBenchBpbAdvError(
        "representative target method requires an explicit no-adversary "
        "control strength"
    )


@beartype
@dataclass(frozen=True)
class _PrefixPair:
    """One (rich_i, avoid_j) completion pair drawn from a core prefix."""

    rich_sample_id: NonEmptyStrLike
    avoid_sample_id: NonEmptyStrLike
    rich_output_tokens: NonNegativeIntListLike
    avoid_output_tokens: NonNegativeIntListLike


def _all_prefix_pairs(prefix: BpbAttackPrefix) -> list[_PrefixPair]:
    """Full Cartesian product of the prefix's rich x avoid completions.

    Iteration order is rich-major then avoid (``itertools.product`` order) so the
    pair index used for deterministic subsampling is stable across runs.
    """

    pairs: list[_PrefixPair] = []
    for rich_index, rich_output in enumerate(prefix.concept_rich_output_tokens):
        rich_sample_id = prefix.concept_rich_sample_ids[rich_index]
        rich_tokens = list(rich_output)
        for avoid_index, avoid_output in enumerate(
            prefix.concept_avoiding_output_tokens
        ):
            pairs.append(
                _PrefixPair(
                    rich_sample_id=rich_sample_id,
                    avoid_sample_id=prefix.concept_avoiding_sample_ids[avoid_index],
                    rich_output_tokens=rich_tokens,
                    avoid_output_tokens=list(avoid_output),
                )
            )
    return pairs


@beartype
def _select_prefix_pairs(
    prefix: BpbAttackPrefix,
    *,
    pair_subsample: NonNegativeIntLike,
) -> list[_PrefixPair]:
    """Cartesian pairs for one prefix, optionally subsampled deterministically.

    ``pair_subsample <= 0`` keeps every pair. When the Cartesian count exceeds
    ``pair_subsample``, a reproducible subset of that size is selected with a
    seed derived solely from ``core_prefix_id`` (so distinct prefixes draw
    independent subsets and re-runs are byte-identical). The surviving pairs are
    returned in their original Cartesian order to keep row emission stable.
    """

    pairs = _all_prefix_pairs(prefix)
    if pair_subsample == 0 or len(pairs) <= pair_subsample:
        return pairs
    seed = int.from_bytes(
        hashlib.sha256(prefix.core_prefix_id.encode("utf-8")).digest()[:8],
        byteorder="little",
        signed=False,
    )
    rng = random.Random(seed)
    chosen_indices = sorted(rng.sample(range(len(pairs)), pair_subsample))
    return [pairs[index] for index in chosen_indices]


def _row_base(
    *,
    prefix: BpbAttackPrefix,
    pair: _PrefixPair,
    spec: BpbAttackWorkSpec,
    adversarial_method: AdversarialMethod,
    strength: float,
    strength_unit: str,
    strength_index: int,
    context_signal_mode: ContextSignalMode,
    editable_span: EditableSpan,
) -> dict[str, Any]:
    source_tokens = list(prefix.prefix_tokens)
    original_editable = source_tokens[editable_span.start:editable_span.stop]
    rich_output = list(pair.rich_output_tokens)
    avoid_output = list(pair.avoid_output_tokens)
    row = {
        "attack_row_id": "",
        "core_prefix_id": prefix.core_prefix_id,
        "concept_rich_sample_id": pair.rich_sample_id,
        "concept_avoiding_sample_id": pair.avoid_sample_id,
        "concept_id": prefix.concept_id,
        "mode": prefix.mode.value,
        "method_name": spec.method_name,
        "target_model": spec.target_model,
        "coefficient_str": spec.coefficient_str,
        "adversarial_method": adversarial_method.value,
        "adversarial_strength": float(strength),
        "strength_unit": strength_unit,
        "strength_index": int(strength_index),
        "context_signal_mode": context_signal_mode.value,
        "prefix_text": prefix.prefix_text,
        "prefix_hash": prefix.prefix_hash,
        "source_prompt_tokens": source_tokens,
        "source_prompt_token_digest": token_sequence_digest(source_tokens),
        "concept_rich_output_tokens": rich_output,
        "concept_avoiding_output_tokens": avoid_output,
        "concept_rich_output_digest": token_sequence_digest(rich_output),
        "concept_avoiding_output_digest": token_sequence_digest(avoid_output),
        "editable_token_start": int(editable_span.start),
        "editable_token_count": int(editable_span.count),
        "original_editable_tokens": list(original_editable),
    }
    row["attack_row_id"] = _attack_row_id(row)
    return row


def _spec_supports_additive(
    spec: BpbAttackWorkSpec,
    additive_cfg: AdditiveDirectionConfig,
) -> bool:
    # additive composes a counter-direction on top of ANY baseline (decoder-only
    # direction/lora OR encoder-decoder `ours` conditioning). The `control`
    # (no-steering) baselines are ALWAYS counter-steered when additive is
    # configured — they are the pure-control reference (magnitude 0 = unmodified
    # model) — regardless of which steering families the user chose to attack.
    return (
        spec.baseline_family == "control"
        or spec.baseline_family in set(additive_cfg.families)
    )


def _is_fair_feature_diffmean_target_spec(spec: BpbAttackWorkSpec) -> bool:
    mode = spec.mode.value
    return spec.method_name == FAIR_FEATURE_DIFFMEAN_METHOD_BY_MODE.get(mode)


@beartype
def prepare_bpb_attack_rows(
    *,
    prefixes: _NonEmptyBpbAttackPrefixSequenceLike,
    work_specs: _NonEmptyBpbAttackWorkSpecSequenceLike,
    cfg: AxBenchBpbAdvConfig,
) -> pa.Table:
    """Build deterministic attack rows from core prefixes and method grids.

    Each prefix carries ALL of its rich and avoid class-margin completions (Gap
    B). The Cartesian product of those completions is exploded into one attack
    row per (rich_i, avoid_j) pair -- optionally deterministically subsampled to
    ``cfg.pair_subsample`` pairs per prefix -- so each row still owns exactly one
    rich and one avoid completion for downstream dual-completion scoring under one
    steered intervention. Distinct pairs differ in their output digests, so every
    pair yields a distinct ``attack_row_id``.
    """

    pair_subsample = cfg.pair_subsample
    # #1249 follow-up: drop excluded (concept, mode) prefixes BEFORE any method
    # grid sees them, so no row -- steered, control, or counter-steered -- is
    # emitted for an excluded pair and the direction preflight never requests
    # its coordinates. Loud by construction: per-exclusion drop counts are
    # logged, and an exclusion matching nothing raises (a typo would otherwise
    # silently exclude nothing).
    if cfg.exclude_concept_modes:
        excluded = {
            (int(e.split(":")[0]), e.split(":")[1])
            for e in cfg.exclude_concept_modes
        }
        drop_counts: dict[tuple[int, str], int] = dict.fromkeys(excluded, 0)
        kept: list[BpbAttackPrefix] = []
        for prefix in prefixes:
            key = (int(prefix.concept_id), str(prefix.mode))
            if key in excluded:
                drop_counts[key] += 1
            else:
                kept.append(prefix)
        unmatched = sorted(key for key, n in drop_counts.items() if n == 0)
        if unmatched:
            raise AxBenchBpbAdvError(
                f"exclude_concept_modes entries matched no prefixes: {unmatched}; "
                "a typo here would silently exclude nothing"
            )
        for (concept_id, mode), count in sorted(drop_counts.items()):
            logger.info(
                f"attack rows: EXCLUDED concept {concept_id} mode={mode} "
                f"({count} prefixes dropped) per cfg.exclude_concept_modes"
            )
        prefixes = kept
        if not prefixes:
            raise AxBenchBpbAdvError(
                "exclude_concept_modes removed every prefix; nothing to score"
            )
    method_configs = adversarial_method_configs_by_name(cfg)
    specs_by_mode: dict[SteeringMode, list[BpbAttackWorkSpec]] = {}
    for spec in work_specs:
        specs_by_mode.setdefault(spec.mode, []).append(spec)

    rows: list[dict[str, Any]] = []
    for prefix in prefixes:
        prefix_pairs = _select_prefix_pairs(prefix, pair_subsample=pair_subsample)
        for spec in specs_by_mode.get(prefix.mode, []):
            for method, method_config in method_configs.items():
                if method in {AdversarialMethod.FASTER_GCG, AdversarialMethod.GEPA}:
                    if not isinstance(method_config, FasterGcgConfig | GepaConfig):
                        raise AxBenchBpbAdvError(
                            f"{method.value} config has the wrong type"
                        )
                    prompt_specs = _prompt_attack_specs(method_config=method_config)
                    # GEPA cannot express a no-adversary control in its grid
                    # (editable count >= 1); faster_gcg can (k == 0), so it stays
                    # strict and raises when the control is missing.
                    synthesize_control = method is AdversarialMethod.GEPA
                    if spec.target_only:
                        prompt_specs = _control_specs_for_target(
                            prompt_specs, synthesize=synthesize_control
                        )
                    elif _is_fair_feature_diffmean_target_spec(spec):
                        prompt_specs = _grid_with_control_spec(
                            prompt_specs, synthesize=synthesize_control
                        )
                    for strength, unit, index, editable_count in prompt_specs:
                        try:
                            span = select_final_k_editable_span(
                                prefix.prefix_tokens,
                                k=editable_count,
                            )
                        except AxBenchBpbAdvError as exc:
                            raise AxBenchBpbAdvError(
                                "cannot build prompt-token adversarial row for "
                                f"core_prefix_id={prefix.core_prefix_id!r}, "
                                f"adversarial_method={method.value!r}, "
                                f"strength={strength!r}: {exc}"
                            ) from exc
                        for pair in prefix_pairs:
                            rows.append(
                                _row_base(
                                    prefix=prefix,
                                    pair=pair,
                                    spec=spec,
                                    adversarial_method=method,
                                    strength=strength,
                                    strength_unit=unit,
                                    strength_index=index,
                                    context_signal_mode=ContextSignalMode.FIXED_CONTEXT,
                                    editable_span=span,
                                )
                            )
                elif method == AdversarialMethod.ADDITIVE_DIRECTION:
                    if not isinstance(method_config, AdditiveDirectionConfig):
                        raise AxBenchBpbAdvError(
                            "additive_direction config has the wrong type"
                        )
                    supports_additive = _spec_supports_additive(spec, method_config)
                    use_target_control_only = (
                        spec.target_only
                        or (
                            _is_fair_feature_diffmean_target_spec(spec)
                            and not supports_additive
                        )
                    )
                    if not use_target_control_only and not supports_additive:
                        continue
                    span = EditableSpan(start=len(prefix.prefix_tokens), count=0)
                    # additive can configure an explicit magnitude 0.0 control,
                    # so it stays strict (synthesize=False) and raises loudly when
                    # that control is missing.
                    additive_specs = _additive_specs(method_config)
                    if use_target_control_only:
                        additive_specs = _control_specs_for_target(
                            additive_specs, synthesize=False
                        )
                    elif _is_fair_feature_diffmean_target_spec(spec):
                        additive_specs = _grid_with_control_spec(
                            additive_specs, synthesize=False
                        )
                    for strength, unit, index, _editable_count in additive_specs:
                        for pair in prefix_pairs:
                            rows.append(
                                _row_base(
                                    prefix=prefix,
                                    pair=pair,
                                    spec=spec,
                                    adversarial_method=method,
                                    strength=strength,
                                    strength_unit=unit,
                                    strength_index=index,
                                    context_signal_mode=ContextSignalMode.NOT_APPLICABLE,
                                    editable_span=span,
                                )
                            )
                else:
                    raise AxBenchBpbAdvError(
                        f"unsupported adversarial method {method.value!r}"
                    )
    return pa.Table.from_pylist(rows, schema=ATTACK_ROWS_SCHEMA)
