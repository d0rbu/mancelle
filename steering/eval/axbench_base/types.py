"""Shared types for the axbench_base evaluation pipeline."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from beartype import beartype

from steering.refined_types import (
    NonBlankStrLike,
    NonNegativeIntLike,
    StrictIntLike,
)

if TYPE_CHECKING:
    from steering.conditional.encoder_decoder.eval_adapter import ContextProvider
    from steering.eval.cluster_checkpoint_pin import CheckpointFilesPin
    from steering.eval.cluster_feature_contract import ClusterFeatureContract


class AxBenchBaseError(Exception):
    """Base exception for all axbench_base pipeline errors."""


# Canonical synth-corpus content_class labels for the axbench_base pipeline (#559).
# Stage 1 row builders write these into the ``content_class`` column; the
# axbench_base Stage 2/3 consumers filter on them via this constant.
# WARNING: the value is a corpus-schema contract, NOT a free-to-rename constant.
# The sibling axbench_bpb / axbench_bpb_adv packages (e.g. source_validation.py,
# bpb_aggregation.py, scoring.py, attack_rows.py) read the SAME synth corpus but
# still hardcode the "concept_rich"/"concept_avoiding" string literals — they are
# not yet wired to this constant — so renaming a VALUE here MUST be applied there
# in lockstep or it silently desyncs bpb/bpb_adv source validation + aggregation.
CONCEPT_RICH_LABEL: str = "concept_rich"
CONCEPT_AVOIDING_LABEL: str = "concept_avoiding"

# Raw SAE percentile domain shared by live scoring and persisted checkpoints.
MIN_RAW_PERCENTILE: float = 0.5
MAX_RAW_PERCENTILE: float = 99.5

# Stable prefix for trusted Stage 0b document continuations reused as natural
# Stage 1b avoiding rows. Writers and checkpoint validators must classify this
# provenance identically.
DOC_CONTINUATION_GEN_SOURCE_PREFIX: str = "corpus:doc_continuation@"

# Stage 0c's local and API algorithms intentionally occupy distinct checkpoint
# schemas. Both the single-rank and DDP checkpoint implementations consume
# these values, so keep them here rather than duplicating numeric literals in
# modules whose import graph is cyclic through ``_stage0c_internals``.
STAGE0C_LOCAL_CHECKPOINT_SCHEMA_VERSION: int = 8
STAGE0C_API_CHECKPOINT_SCHEMA_VERSION: int = 9


class QcOutcome(StrEnum):
    """Closed set of concept-examples QC outcomes.

    Lives here (not in ``concept_examples_qc``) so
    ``RejectedConceptReason.from_qc_outcome`` can reference it without
    creating an import cycle — ``concept_examples_qc`` imports from
    ``types``, not the reverse. Re-exported from ``concept_examples_qc``
    for ergonomic backwards compatibility. The string values are part of
    the persisted QC-report JSON schema; renaming a value is a breaking
    change.

    A :class:`StrEnum` so members compare equal to their underlying
    string value (``QcOutcome.ACCEPTED == "accepted"`` is ``True``).
    This lets every existing bare-string comparison
    (``if row.outcome == "accepted"``) keep working while giving callers
    the option of the strongly-typed member form. JSON serialisation
    emits the bare string value because :class:`StrEnum` subclasses
    :class:`str`, so the persisted QC-report schema is byte-identical
    to the prior :class:`typing.Literal` formulation.
    """

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FETCH_ERROR = "fetch_error"
    JUDGE_ERROR = "judge_error"
    EMPTY_EXAMPLES = "empty_examples"


class SteeringMode(StrEnum):
    """Direction of a steering evaluation method.

    A :class:`StrEnum` so ``SteeringMode.POSITIVE == "positive"`` is
    ``True``. This lets every existing bare-string comparison in the
    pipeline (``if method.mode == "positive"``, ``if mode == "negative"``)
    keep working while giving callers the option of the strongly-typed
    member form. The parquet columns that carry ``mode`` stay declared
    as ``pa.string()``; serialising a :class:`SteeringMode` member to
    parquet yields the same string literal as the old code path because
    ``StrEnum`` subclasses :class:`str`.

    - :attr:`POSITIVE`: push a neutral prefix TOWARDS a concept.
    - :attr:`NEGATIVE`: push a concept-rich prefix AWAY from a concept.
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"


class Stage(StrEnum):
    """Identifiers for each stage in the axbench_base pipeline.

    Stage IDs are short strings (``"0a"``, ``"3b"``, ``"all"``) that
    historically appeared as magic literals throughout the DAG and CLI.
    Using a :class:`StrEnum` here:

    - preserves every existing string comparison (``stage == "0a"``)
      because ``Stage.PREPARE_CONCEPTS == "0a"`` is ``True`` under
      :class:`StrEnum`;
    - lets Hydra configs continue to write ``stage: 0a`` / ``stage: all``
      in YAML — OmegaConf deserializes them as plain strings that
      compare equal to the corresponding :class:`Stage` member and the
      CLI coerces them on entry;
    - makes dict keys keyed on :class:`Stage` accept string lookups,
      since :class:`StrEnum` preserves ``__hash__`` equality with the
      underlying string;
    - provides a single source of truth for the CLI dispatch table
      and the ordered stage list used by ``_resolve_stage_list``.

    The ``ALL`` pseudo-stage is the "run every stage in order" sentinel
    consumed by the CLI's ``_resolve_stage_list``. It is intentionally
    NOT a key in ``STAGE_DEPENDENCIES`` — the dispatcher expands it to
    the full ordered stage list before the dep check runs.

    Member values are stable CLI identifiers; renaming a value is a
    breaking change (it invalidates existing Hydra configs, shell
    scripts, and operator runbooks). Member NAMES are developer-facing
    and may be refactored freely.
    """

    PREPARE_CONCEPTS = "0a"              # prepare_concept_list
    PREPARE_SEED_POOLS = "0b"            # prepare_seed_pools
    PREPARE_NEGATIVE_PREFIXES = "0c"     # prepare_negative_prefixes
    SYNTH_POSITIVE_MODE = "1b"           # synthesize_positive_mode
    SYNTH_NEGATIVE_MODE = "1c"           # synthesize_negative_mode
    CONCEPT_FEATURE_MAPPING = "2"        # concept_feature_mapping_*
    TRAIN_DIFFMEAN = "3a"                # train_diffmean_all
    TRAIN_LINEAR_PROBE = "3b"            # train_linear_probe_all
    TRAIN_PROMPTING = "3c"               # write_prompting_descriptions
    TRAIN_LORA = "3d"                    # train_lora_all
    ENUMERATE_STAGE4_CONTEXTS = "3e"      # enumerate_stage4_contexts
    GENERATE = "4"                       # generate_all
    JUDGE = "5"                          # judge_all
    AGGREGATE = "6"                      # aggregate
    ALL = "all"                          # pseudo-stage: run every stage in order

    @classmethod
    def coerce(cls, stage: Stage | str | int) -> Stage:
        """Turn a caller-supplied stage identifier into a :class:`Stage`.

        Accepts a :class:`Stage` member, a bare-string identifier
        (``"0a"``, ``"all"``), or a bare-int identifier (``2``, ``4``,
        ``5``, ``6``). The int path exists because Hydra and OmegaConf
        type-infer numeric CLI overrides as Python ints: the operator
        writes ``stage=2`` on the command line, Hydra parses ``2`` as
        an int, and the original version rejected it because
        ``Stage(2)`` doesn't match ``Stage("2")`` under StrEnum
        equality. Normalise to str before dispatching to the enum so
        both forms resolve identically.

        Unknown values raise :class:`AxBenchBaseError` with a message
        listing every known stage identifier so the operator can spot
        typos.
        """
        # Bool is an int subclass; exclude it so ``Stage.coerce(True)``
        # doesn't silently resolve to a stage whose value is "1".
        if isinstance(stage, int) and not isinstance(stage, bool):
            normalised: Stage | str | bool = str(stage)
        else:
            normalised = stage
        try:
            return cls(normalised)
        except ValueError as exc:
            known = sorted(s.value for s in cls)
            raise AxBenchBaseError(
                f"Unknown stage {stage!r}. Known stages: {known}"
            ) from exc


class RejectedConceptReason(StrEnum):
    """Stable reason values written to ``rejected_concepts.json``.

    ``StrEnum`` keeps the artifact schema as plain strings while avoiding
    untyped literals in Stage 0c's implementation and tests.
    """

    LLM_JUDGE_BELOW_MIN_SCORE = "llm_judge_below_min_score"
    LLM_JUDGE_FETCH_ERROR = "llm_judge_fetch_error"
    LLM_JUDGE_ERROR = "llm_judge_error"
    LLM_JUDGE_EMPTY_EXAMPLES = "llm_judge_empty_examples"
    NOT_ENOUGH_VERIFIED_PREFIXES = "not_enough_verified_prefixes"

    @classmethod
    def from_qc_outcome(cls, outcome: QcOutcome) -> RejectedConceptReason:
        """Map a concept-examples QC outcome to a rejection reason.

        :attr:`QcOutcome.ACCEPTED` is intentionally not a valid input —
        callers must gate on ``outcome != QcOutcome.ACCEPTED`` before
        calling this helper, so a stray accepted-outcome row that
        reached the rejection path surfaces as
        :class:`AxBenchBaseError` instead of being silently coerced to
        a fixed reason.
        """
        if outcome == QcOutcome.REJECTED:
            return cls.LLM_JUDGE_BELOW_MIN_SCORE
        if outcome == QcOutcome.FETCH_ERROR:
            return cls.LLM_JUDGE_FETCH_ERROR
        if outcome == QcOutcome.JUDGE_ERROR:
            return cls.LLM_JUDGE_ERROR
        if outcome == QcOutcome.EMPTY_EXAMPLES:
            return cls.LLM_JUDGE_EMPTY_EXAMPLES
        raise AxBenchBaseError(
            f"Unexpected concept examples QC outcome: {outcome!r}"
        )


@beartype
@dataclass(frozen=True)
class CandidateConcept:
    """A pre-LLM-QC candidate emitted by Stage 0a.

    ``candidate_concepts.json`` intentionally has no ``concept_id``:
    Stage 0c assigns the temporary post-LLM-QC id used for verification
    RNG and then writes final contiguous ``Concept.concept_id`` values
    only for candidates that pass negative-prefix verification.
    """

    feature_index: NonNegativeIntLike
    layer: StrictIntLike
    sae_source: NonBlankStrLike
    description: NonBlankStrLike
    corpus_count: NonNegativeIntLike
    description_source: NonBlankStrLike


@beartype
@dataclass(frozen=True)
class Concept:
    """A concept entry for axbench_base evaluation.

    One SAE feature + its human-readable description, plus the
    bookkeeping fields the pipeline needs across stages.

    Fields are serialized to/from ``concepts.json`` via ``dataclasses.asdict``
    and ``Concept(**item)`` respectively. ``load_and_validate_concepts`` in
    ``steering.eval.axbench_base.validate`` is the canonical stage-boundary
    reader.
    """
    concept_id: NonNegativeIntLike
    feature_index: NonNegativeIntLike
    layer: StrictIntLike
    sae_source: NonBlankStrLike
    description: NonBlankStrLike
    corpus_count: NonNegativeIntLike
    description_source: NonBlankStrLike


@dataclass(frozen=True)
class TargetModelMetadata:
    """Runtime metadata about a loaded target model.

    Produced by ``detect_target_model_metadata`` and consumed by
    coefficient grid factories, partition modifiers, and the Stage 4
    generation dispatch.

    Fields:
        num_buckets: the target model's encoder feature-bucket count
            (``None`` for decoder-only targets that have no encoder).
        n_sae_features: size of the SAE vocabulary the encoder was
            trained against (``None`` for decoder-only targets).
        negative_selection: the training-time negative-feature-selection
            config dict, as stored in the checkpoint's ``user_config``.
            ``None`` if negative sampling was not configured, or for
            decoder-only targets.
        feature_selection: training-time feature-selection config dict
            (threshold-based or sampling-based). ``None`` if not set.
        feature_mask: training-time feature blacklist/whitelist config
            dict, as stored in the checkpoint's ``user_config``.
            ``None`` if no mask was configured, or for decoder-only
            targets. A mask-trained encoder-decoder target MUST carry
            this so Stage-4 extraction excludes masked features and the
            extraction cache cannot collide with an unmasked run.
        per_feature_stats: lazily-populated corpus statistics keyed by
            feature index (as string). ``None`` until an extraction
            helper loads the stats file. Downstream code should treat
            a ``None`` here as "not yet loaded".
        device: compute device the pipeline is running on, copied from
            ``cfg.device`` at detection time so downstream consumers
            don't need to plumb the full config through.
        sae_layer: residual-stream layer the SAE was trained against,
            parsed from ``cfg.feature_dir`` (the ``l{layer}``
            component). Replaces the previously hardcoded
            ``_SAE_LAYER = 8`` constant in ``extraction.py`` /
            ``synth/qc.py``. Threaded through every extraction /
            QC call site so that swapping to a different-layer SAE
            is a config-only change.
        segment_size: per-segment token count used at feature-
            extraction time, parsed from ``cfg.feature_dir`` (the
            ``s{segment_size}`` component). Carried alongside
            ``sae_layer`` for downstream consumers that need to
            reconstruct the training-time pooling convention.
    """

    num_buckets: int | None
    n_sae_features: int | None
    negative_selection: dict[str, Any] | None
    feature_selection: dict[str, Any] | None
    per_feature_stats: dict[str, Any] | None
    device: str
    sae_layer: int
    segment_size: int
    feature_mask: dict[str, Any] | None = None


@dataclass(frozen=True)
class SeedRow:
    """Schema-v2 seed: exact BOS-free source slices plus prefix SAE features."""

    shard_idx: int
    doc_idx: int
    source_nanochat_token_count: int
    prefix_source_token_start: int
    prefix_source_token_end: int
    prefix_text: str
    prefix_content_nanochat_tokens: list[int]
    sae_feature_ids: list[int]
    sae_feature_mags: list[float]
    continuation_text: str
    continuation_content_nanochat_tokens: list[int]

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (self.shard_idx, self.doc_idx)
        ) or self.shard_idx < 0 or self.doc_idx < 0:
            raise AxBenchBaseError(
                "SeedRow provenance indices must be non-negative, got "
                f"shard_idx={self.shard_idx}, doc_idx={self.doc_idx}."
            )
        if not isinstance(self.prefix_text, str) or not isinstance(
            self.continuation_text, str
        ):
            raise AxBenchBaseError("SeedRow text fields must be strings")
        if not self.prefix_content_nanochat_tokens:
            raise AxBenchBaseError("SeedRow prefix content tokens must be non-empty")
        if not self.continuation_content_nanochat_tokens:
            raise AxBenchBaseError("SeedRow has empty continuation content tokens")
        all_tokens = (
            self.prefix_content_nanochat_tokens
            + self.continuation_content_nanochat_tokens
        )
        if any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in all_tokens
        ):
            raise AxBenchBaseError(
                "SeedRow nanochat content token IDs must be non-negative integers"
            )
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (
                self.source_nanochat_token_count,
                self.prefix_source_token_start,
                self.prefix_source_token_end,
            )
        ):
            raise AxBenchBaseError(
                "SeedRow source counts and bounds must be integers"
            )
        continuation_length = len(self.continuation_content_nanochat_tokens)
        if self.source_nanochat_token_count < 2 * continuation_length:
            raise AxBenchBaseError(
                "SeedRow source token count must be at least twice the "
                f"continuation length, got {self.source_nanochat_token_count} "
                f"and {continuation_length}."
            )
        expected_prefix_end = self.source_nanochat_token_count - continuation_length
        if self.prefix_source_token_end != expected_prefix_end:
            raise AxBenchBaseError(
                "SeedRow prefix end must equal source count minus continuation "
                f"length ({expected_prefix_end}), got {self.prefix_source_token_end}."
            )
        if not 0 <= self.prefix_source_token_start < self.prefix_source_token_end:
            raise AxBenchBaseError(
                "SeedRow prefix bounds must satisfy 0 <= start < end, got "
                f"[{self.prefix_source_token_start}, {self.prefix_source_token_end})."
            )
        expected_prefix_length = (
            self.prefix_source_token_end - self.prefix_source_token_start
        )
        if len(self.prefix_content_nanochat_tokens) != expected_prefix_length:
            raise AxBenchBaseError(
                "SeedRow prefix token length does not match its source bounds: "
                f"len={len(self.prefix_content_nanochat_tokens)}, "
                f"bounds_length={expected_prefix_length}."
            )
        if any(
            not isinstance(feature_id, int) or isinstance(feature_id, bool)
            for feature_id in self.sae_feature_ids
        ):
            raise AxBenchBaseError("SeedRow SAE feature IDs must be integers")
        if self.sae_feature_ids != sorted(set(self.sae_feature_ids)):
            raise AxBenchBaseError("SeedRow SAE feature IDs must be sorted and unique")
        if any(feature_id < 0 for feature_id in self.sae_feature_ids):
            raise AxBenchBaseError("SeedRow SAE feature IDs must be non-negative")
        if len(self.sae_feature_ids) != len(self.sae_feature_mags):
            raise AxBenchBaseError("SeedRow SAE feature ID/magnitude lengths must match")
        if any(
            not isinstance(mag, int | float)
            or isinstance(mag, bool)
            or not math.isfinite(mag)
            or mag <= 0.0
            for mag in self.sae_feature_mags
        ):
            raise AxBenchBaseError(
                "SeedRow SAE feature magnitudes must be finite and strictly positive"
            )


@dataclass(frozen=True)
class TargetBundle:
    """A fully-loaded target model with all resources a generation stage needs.

    Replaces the parallel ``target_models``, ``target_model_metas``,
    ``engines``, and ``tokenizers`` dicts that Stage 4's
    :func:`~steering.eval.axbench_base.generation.generate_all` previously
    accepted as separate keyword arguments. Each bundle groups a single
    target's live runtime resources under one typed object.

    The ``name`` distinguishes between targets in the method registry
    (e.g., ``"decoder_only"`` vs ``"encdec"``). The ``model``,
    ``meta``, and ``tokenizer`` are the live runtime resources from
    :class:`~steering.eval.axbench_base.runtime_state.RuntimeState`.

    For encoder-decoder targets, ``context_provider`` must supply real
    prompt/context features per generation call (via
    :func:`~steering.eval.inference_engine.build_engine_for_features`).
    """

    name: str
    model: Any
    meta: TargetModelMetadata
    tokenizer: Any
    context_provider: ContextProvider | None
    cluster_contract: ClusterFeatureContract | None = None
    cluster_pre_forward_guard: Callable[[], None] | None = None
    checkpoint_files_pin: CheckpointFilesPin | None = None

    def ensure_ready_for_forward(self) -> None:
        """Recheck clustered encoder-decoder bytes at the first forward seam."""

        # Decoder-only bundles legitimately carry the eval-side cluster
        # contract for scalar-artifact validation. This guard re-reads
        # checkpoint bytes and is meaningful only for encdec checkpoints.
        if self.cluster_contract is None or self.name != "encdec":
            return
        if self.cluster_pre_forward_guard is None:
            raise AxBenchBaseError(
                "clustered encoder-decoder target bundle is missing its "
                "validated pre-forward checkpoint guard"
            )
        self.cluster_pre_forward_guard()
