"""Configuration for the AxBench-BPB adversarial robustness eval."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Any, Literal

from beartype import beartype
from beartype.vale import Is

from steering.conditional.feature_gcg.config import (
    DEFAULT_MAX_FORWARD_BATCH,
    DEFAULT_MAX_GRAD_BATCH,
    DEFAULT_OBJECTIVE_ALPHA,
)
from steering.context_defaults import DEFAULT_CONTEXT_WINDOW
from steering.eval.axbench_base.config import effective_extraction_model_id
from steering.eval.axbench_base.method_registry import ALL_METHODS
from steering.eval.axbench_base.profiler import ProfileConfig
from steering.eval.axbench_bpb.config import ContextCacheConfig
from steering.eval.axbench_bpb.representative_selection import (
    FAIR_FEATURE_DIFFMEAN_METHOD_BY_MODE,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.gcg_family import ALPHA_AXIS_GCG_METHODS
from steering.eval.cluster_config import (
    ClusterEvalConfigError,
    parse_cluster_eval_paths,
    validate_clustered_method_allowlist,
)
from steering.eval.roundtrip_steering import RoundtripEvalConfig
from steering.refined_types import (
    NonBlankStrLike,
    NonNegativeFiniteNumberLike,
    NonNegativeIntLike,
    OpenUnitIntervalLike,
    PositiveFiniteNumberLike,
    PositiveIntLike,
    StrictIntLike,
    UniqueNonEmptyNonBlankStrTuple,
    UniqueNonEmptyNonNegativeFiniteNumberTupleLike,
    UniqueNonEmptyNonNegativeIntTupleLike,
    UniqueNonEmptyPositiveIntTupleLike,
)
from steering.sae_resolver import L0Tier, parse_neuronpedia_source


class Stage(StrEnum):
    VALIDATE_SOURCE = "validate_source"
    PREPARE_CONTEXT_CACHE = "prepare_context_cache"
    PREPARE_ATTACK_ROWS = "prepare_attack_rows"
    BUILD_CANDIDATES = "build_candidates"
    SCORE = "score"
    # Optional pass (gated on general_val_bpb.enabled): score a held-out generic
    # validation set under each steering+adversarial intervention, to measure how
    # much steering degrades GENERAL language-modelling ability (not just the
    # concept margin). Runs after SCORE and before AGGREGATE so its per-
    # intervention BPB table can be joined onto the macro rows.
    SCORE_GENERAL_VAL = "score_general_val"
    AGGREGATE = "aggregate"
    PLOT = "plot"
    ALL = "all"

    @classmethod
    def coerce(cls, value: Stage | str) -> Stage:
        try:
            return cls(value)
        except ValueError as exc:
            raise AxBenchBpbAdvError(
                f"Unknown axbench-bpb-adv stage {value!r}; expected one of "
                f"{', '.join(stage.value for stage in cls)}"
            ) from exc


class ContextSignalMode(StrEnum):
    FIXED_CONTEXT = "fixed_context"
    NOT_APPLICABLE = "not_applicable"

    @classmethod
    def coerce(cls, value: ContextSignalMode | str) -> ContextSignalMode:
        try:
            return cls(value)
        except ValueError as exc:
            raise AxBenchBpbAdvError(
                f"Unknown context signal mode {value!r}; expected one of "
                f"{', '.join(mode.value for mode in cls)}"
            ) from exc


class AdversarialMethod(StrEnum):
    FASTER_GCG = "faster_gcg"
    GEPA = "gepa"
    ADDITIVE_DIRECTION = "additive_direction"

    @classmethod
    def coerce(cls, value: AdversarialMethod | str) -> AdversarialMethod:
        try:
            return cls(value)
        except ValueError as exc:
            raise AxBenchBpbAdvError(
                f"Unknown adversarial method {value!r}; expected one of "
                f"{', '.join(method.value for method in cls)}"
            ) from exc


class SteeringMetric(StrEnum):
    """Y-axis steering metric used by the score/aggregate/plot stages."""

    ROUNDTRIP = "roundtrip"
    BPB_CLASS_MARGIN = "bpb_class_margin"

    @classmethod
    def coerce(cls, value: SteeringMetric | str) -> SteeringMetric:
        try:
            return cls(value)
        except ValueError as exc:
            raise AxBenchBpbAdvError(
                f"Unknown steering_metric {value!r}; expected one of "
                f"{', '.join(metric.value for metric in cls)}"
            ) from exc


class ScatterBaselineX(StrEnum):
    """The x-axis baseline for the steerability-vs-baseline scatter.

    * ``baseline_loss``: the model's NATURAL (unsteered) completion BPB, which is
      mode-dependent (positive -> concept-avoiding, negative -> concept-rich).
    * ``min``: ``min(mean_loss_concept_rich, mean_loss_concept_avoiding)`` (symmetric).
    * ``general_val_bpb``: the model's BPB on a held-out GENERIC validation set
      (pretraining-style text) scored under the SAME steering intervention as the
      point. A less noisy, concept-agnostic measure of how much steering degrades
      general language-modelling ability than the single concept-completion loss.
      Requires ``general_val_bpb.enabled`` and the ``score_general_val`` stage;
      points without a joined ``general_val_bpb`` value are skipped.

    In the class-margin schema there is no neutral/steered axis; the two losses are
    the two completion classes, so ``min`` is the smaller of the two class BPBs.
    """

    BASELINE_LOSS = "baseline_loss"
    MIN = "min"
    GENERAL_VAL_BPB = "general_val_bpb"

    @classmethod
    def coerce(cls, value: ScatterBaselineX | str) -> ScatterBaselineX:
        try:
            return cls(value)
        except ValueError as exc:
            raise AxBenchBpbAdvError(
                f"Unknown scatter_baseline_x {value!r}; expected one of "
                f"{', '.join(choice.value for choice in cls)}"
            ) from exc


@beartype
@dataclass(frozen=True)
class FasterGcgConfig:
    method: Literal["faster_gcg"] = AdversarialMethod.FASTER_GCG.value
    optimized_token_counts: UniqueNonEmptyNonNegativeIntTupleLike = field(
        default_factory=lambda: (0, 1, 2, 4, 8, 16, 32)
    )
    # Optimization budget for the universal per-concept trigger. The trigger is
    # SHARED across all of a concept's prefixes (not a per-prefix bespoke
    # suffix), so it gets a larger budget than the legacy per-prefix attack
    # (Phase 6: 64 -> 256). For very small k the trigger's candidate space is
    # tiny and `visited` saturates well before 256 steps -- the optimizer then
    # terminates cleanly on empty no-op steps.
    max_steps: NonNegativeIntLike = 256
    candidate_batch_size: PositiveIntLike = 64
    top_k: PositiveIntLike = 32
    distance_weight: NonNegativeFiniteNumberLike = 10.0
    temperature: PositiveFiniteNumberLike = 0.1
    master_seed: StrictIntLike = 0
    # Universal per-concept optimizer: pairs dealt without replacement into each
    # GCG step's minibatch (design DECISION C). >= n_pairs => the whole group
    # every step.
    universal_minibatch_size: PositiveIntLike = 16
    # Scoped shared-prefix candidate-eval microbatch. ``None`` inherits the top-level
    # device_batch_size without changing gradient, GEPA, scoring, or general-val
    # batch composition.
    candidate_eval_device_batch_size: PositiveIntLike | None = None


@beartype
@dataclass(frozen=True)
class GepaConfig:
    method: Literal["gepa"] = AdversarialMethod.GEPA.value
    metric_call_budgets: UniqueNonEmptyPositiveIntTupleLike = field(
        default_factory=lambda: (8, 16, 32, 64)
    )
    k_editable_tokens: PositiveIntLike = 16
    reflection_model: NonBlankStrLike | None = None
    master_seed: StrictIntLike = 0


AdditiveDirectionFamily = Literal["diffmean", "linear_probe", "lora", "ours"]
AdditiveDirectionFamiliesLike = Annotated[
    tuple[AdditiveDirectionFamily, ...],
    Is[
        lambda value: isinstance(value, UniqueNonEmptyNonBlankStrTuple)
        and all(
            family in {"diffmean", "linear_probe", "lora", "ours"}
            for family in value
        )
    ],
]
CounterDirectionFamily = Literal["diffmean", "linear_probe"]

TargetModel = Literal["decoder_only", "encdec"]
TargetModelsLike = Annotated[
    list[TargetModel],
    Is[
        lambda value: isinstance(
            tuple(value),
            UniqueNonEmptyNonBlankStrTuple,
        )
        and all(target in {"decoder_only", "encdec"} for target in value)
    ],
]


@beartype
@dataclass(frozen=True)
class AdditiveDirectionConfig:
    method: Literal["additive_direction"] = AdversarialMethod.ADDITIVE_DIRECTION.value
    families: AdditiveDirectionFamiliesLike = field(
        default_factory=lambda: ("diffmean", "linear_probe")
    )
    magnitudes: UniqueNonEmptyNonNegativeFiniteNumberTupleLike = field(
        default_factory=lambda: (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    )
    # The SPECIFIED counter-steering vector: the concept's direction of this
    # family (loaded per concept) is pushed, opposing the concept, on top of the
    # full baseline steering — for EVERY attacked baseline (diffmean, probe,
    # lora, ours/encdec). Must be a residual-direction family.
    counter_direction_family: CounterDirectionFamily = "diffmean"


AdversarialMethodConfig = FasterGcgConfig | GepaConfig | AdditiveDirectionConfig

# The ONE home of the mean_g_steer batching default (members per batched
# forward in the warm-start / held-out / in-sample g_steer measurements).
# ``OursGcgConfig.mean_g_steer_batch`` and ``gcg_bags.mean_g_steer``'s
# signature default both read THIS constant. Matches the GCG-step
# minibatch size.
DEFAULT_MEAN_G_STEER_BATCH = 16


@beartype
@dataclass(frozen=True)
class OursGcgConfig:
    """Config for the feature-GCG optimizer BPB-Adv methods.

    Controls the feature-bag GCG optimizer (Phase 2) when it is used as a
    steering method within the AxBench-BPB-Adv evaluation.  All fields have
    sensible defaults so that a bare ``OursGcgConfig()`` is a valid starting
    point.

    Fields:
      * ``reg_lambda_grid``: regularisation-lambda sweep for the GCG
        coefficient axis (default ``(0.0,)`` — single unregularised run).
      * ``objective_alpha_grid``: class-margin alpha sweep — the coefficient
        axis of the ``ours_gcg_alpha*`` methods, exactly as ``reg_lambda_grid``
        is for ``ours_gcg*``. alpha weights the unsteered term of the minimized
        margin ``loss_desired - alpha * loss_contrast`` (see
        :data:`DEFAULT_OBJECTIVE_ALPHA`): lower alpha spends less of the search
        on inflating the unsteered loss, so the derived bag steers at a smaller
        cost to base BPB. Default ``(1.0,)`` — the plain margin, whose bags are
        bit-for-bit the ``ours_gcg*`` bags and are REUSED rather than re-derived
        (alpha is stripped from identity and path at 1.0).
      * ``top_k_pos``: number of top-positive features injected per prefix.
        ``None`` → read the geometric mean from the checkpoint's
        ``feature_selection`` metadata (≈10 for d12_l12); else the integer
        value is used verbatim.
      * ``top_k_neg``: same for the negative-feature bag.  ``None`` → read
        the checkpoint's ``negative_selection`` geometric mean; else use
        verbatim.
      * ``split_ratio``: fraction of prefixes/concept reserved for
        optimisation (train split); the rest is held-out evaluation.
      * ``split_seed``: random seed used for the train/eval split.
      * ``max_steps``: GCG step budget per concept (default ``128``).
      * ``minibatch_size``: number of prefix-pairs sampled per GCG step.
      * ``master_seed``: top-level seed for all GCG random choices.
      * ``mean_g_steer_batch``: max members per batched forward in
        ``mean_g_steer`` (warm-start / held-out g_steer measurement). Bounds
        peak GPU memory; default ``16`` (matches the GCG-step minibatch size).
      * ``max_forward_batch``: max members per sub-batched loss forward inside
        the feature-GCG optimizer (``FeatureGcgConfig.max_forward_batch``,
        threaded to the derive's objective). Bounds peak GPU memory; default
        ``8``.
      * ``max_grad_batch``: max members per sub-batched fwd+BWD inside the
        feature-GCG optimizer (``FeatureGcgConfig.max_grad_batch``). Bounds peak
        GPU memory on the GRADIENT path, which retains activations for the
        backward and so is heavier than the loss path ``max_forward_batch``
        bounds; default ``4``. Memory-only: members are independent, so slicing
        cannot change any member's gradient (#1245).
    """

    reg_lambda_grid: UniqueNonEmptyNonNegativeFiniteNumberTupleLike = (0.0,)
    objective_alpha_grid: UniqueNonEmptyNonNegativeFiniteNumberTupleLike = (
        DEFAULT_OBJECTIVE_ALPHA,
    )
    top_k_pos: PositiveIntLike | None = None
    top_k_neg: PositiveIntLike | None = None
    # The fair derive requires non-empty optimize and held-out splits. The
    # held-out prefixes are the scored set, so either endpoint would silently
    # produce no fair results.
    split_ratio: OpenUnitIntervalLike = 0.8
    split_seed: StrictIntLike = 0
    max_steps: NonNegativeIntLike = 128
    minibatch_size: PositiveIntLike = 16
    master_seed: StrictIntLike = 0
    mean_g_steer_batch: PositiveIntLike = DEFAULT_MEAN_G_STEER_BATCH
    max_forward_batch: PositiveIntLike = DEFAULT_MAX_FORWARD_BATCH
    max_grad_batch: PositiveIntLike = DEFAULT_MAX_GRAD_BATCH


@beartype
@dataclass(frozen=True)
class RoundtripGenerationConfig:
    """Generation settings for the round-trip steering metric."""

    samples_per_candidate: PositiveIntLike = 4
    temperature: NonNegativeFiniteNumberLike = 0.7
    top_k_sampling: PositiveIntLike = 50
    master_seed: StrictIntLike = 42


@beartype
@dataclass(frozen=True)
class GeneralValBpbConfig:
    """General-validation BPB under intervention (opt-in via ``enabled``).

    Scores a held-out GENERIC validation set (FineWeb-Edu held-out shard,
    pretraining-style text) under the SAME steering + adversarial intervention
    used for the concept margin, to measure how much steering degrades general
    language-modelling ability. The metric is computed once per unique
    intervention signature -- ``(target_model, method_name, concept_id,
    coefficient_str, adversarial_signature)`` -- and joined onto
    the macro rows (averaged over concepts) at aggregation time.

    Cost note: under the UNIVERSAL Faster-GCG attack ``general_val_bpb`` is
    ATTACK-AWARE and NON-FLAT across the GCG axis. Each ``k``
    substitutes its concept's optimized k-token trigger (from
    ``universal_triggers.parquet``) into the last-k tokens of every DECODER-ONLY
    validation doc, so the signature forks on ``faster_gcg:k{k}`` and
    each fork is scored separately (k=0 is the clean baseline). Additive
    counter-direction magnitude (a residual hook) also moves the steered
    intervention itself. So the work is ``(#GCG k>0 forks per concept)
    + (#steering settings) + (#additive magnitudes > 0)`` distinct interventions,
    each scored over ``n_documents`` generic samples. (Encdec general_val stays
    on the clean docs -- the attacked docs are decoder-only in v1.)

    Fields:
      * ``enabled``: master switch. Default True computes the concept-agnostic
        general-val BPB (the default steerability-frontier x-axis needs it, see
        ``scatter_baseline_x``); set False to skip ``score_general_val`` and
        keep the pre-feature behaviour (the agg column stays null), but then
        ``scatter_baseline_x`` must be moved off ``general_val_bpb`` too.
      * ``n_documents``: number of generic validation documents to score per
        intervention.
      * ``prefix_tokens`` / ``output_tokens``: each document is sliced (in
        nanochat decoder tokens) into a ``prefix_tokens``-long context prefix and
        an ``output_tokens``-long scored continuation. The prefix must be long
        enough to yield >= 1 Gemma feature segment for the encdec encoder.
      * ``max_interventions``: cap on the number of distinct interventions scored
        (0 = no cap). When exceeded, a deterministic ``seed``-derived subset is
        kept and the dropped count is logged (no silent truncation).
      * ``seed``: deterministic seed for document selection and the
        ``max_interventions`` subsample.
    """

    enabled: bool = True
    n_documents: StrictIntLike = 64
    prefix_tokens: StrictIntLike = 256
    output_tokens: StrictIntLike = 128
    max_interventions: NonNegativeIntLike = 0
    seed: NonNegativeIntLike = 0

    def validate(self) -> None:
        """Validate fields whose meaning is local to general-validation BPB."""
        if self.enabled:
            if self.n_documents < 1:
                raise AxBenchBpbAdvError("general_val_bpb.n_documents must be >= 1")
            if self.prefix_tokens < 1:
                raise AxBenchBpbAdvError("general_val_bpb.prefix_tokens must be >= 1")
            if self.output_tokens < 1:
                raise AxBenchBpbAdvError("general_val_bpb.output_tokens must be >= 1")


@beartype
@dataclass
class AxBenchBpbAdvConfig:
    source_run_dir: str = ""
    output_root: str = ""
    run_tag: str | None = None
    device: str = "cuda"
    stage: Stage | Literal[
        "validate_source",
        "prepare_context_cache",
        "prepare_attack_rows",
        "build_candidates",
        "score",
        "score_general_val",
        "aggregate",
        "plot",
        "all",
    ] = Stage.ALL.value

    base_methods: list[str] | None = None
    # Methods used only to supply representative-coefficient target rows. These
    # are scored at the no-adversary control strength for each adversarial family,
    # but are not optimized over the prompt-token attack grid.
    representative_target_methods: list[str] = field(default_factory=list)
    # #1249 follow-up: (concept, mode) pairs excluded from attack-row emission,
    # as "<concept_id>:<mode>" strings, e.g. ["37:positive"]. The one legitimate
    # use is a concept whose direction the #1228 fail-loud scale guard refuses
    # (concept 37 positive: every fit projection is negative, max_activation
    # -1098), where counter-steering that (concept, mode) is undefined. An
    # exclusion that matches no prefix RAISES at attack-row time (typo guard),
    # every drop is logged with counts, and a non-empty list participates in
    # run-tag identity -- excluding concepts changes the scored population, so
    # two runs differing in exclusions must never share an auto-generated tag.
    exclude_concept_modes: list[str] = field(default_factory=list)
    # Restrict the DiffMean / LinearProbe scalar steering-coefficient axis to an
    # explicit subset (``None`` = the full default grid, so unset changes
    # nothing and keeps existing run tags byte-for-byte).
    #
    # For adversarial runs each coefficient costs a full attack optimization,
    # while the robustness plot keeps only ONE representative coefficient per
    # family and discards the rest -- so the default 10-value grid multiplies
    # attack cost ~10x for output that is thrown away. Pinning a superset of the
    # representatives leaves the selected coefficients, and therefore the
    # figure, unchanged. Set it ONLY when the representatives are already known
    # (e.g. from the companion run being compared against): a subset that
    # excludes a family's representative silently moves that family's curve to
    # whichever remaining coefficient is closest to the baseline.
    steering_coefficient_grid: UniqueNonEmptyNonNegativeFiniteNumberTupleLike | None = None
    target_models: TargetModelsLike = field(
        default_factory=lambda: ["decoder_only", "encdec"]
    )

    adversarial_methods: list[AdversarialMethodConfig] = field(
        default_factory=lambda: [FasterGcgConfig(), AdditiveDirectionConfig()]
    )

    feature_dir: str = ""
    feature_statistics_path: str = ""
    # Clustered evaluation uses path-only schemas. Both fields remain absent
    # from hashes/profiler payloads when null (see cluster_config serializers).
    feature_mask: dict[str, Any] | None = None
    feature_clusters: dict[str, Any] | None = None
    force_cluster_shard_revalidation: bool = False
    # Producer-side provenance needed to identify the exact clustered
    # selection family. The checkpoint records neither the splice batching
    # choice nor which of several semantically distinct manifests below one
    # feature root the evaluator should consume.
    selection_splice_batch_size: PositiveIntLike | None = None
    selection_manifest_path: NonBlankStrLike | None = None
    extraction_tokenizer_id: str = "google/gemma-2-2b"
    neuronpedia_model_id: str = "gemma-2-2b"
    feature_extraction_model_id: NonBlankStrLike | None = None
    neuronpedia_source: str = "8-gemmascope-res-16k"
    extraction_max_tokens: PositiveIntLike = DEFAULT_CONTEXT_WINDOW
    extraction_threshold: float = 0.0
    filter_unaccepted_samples: bool = True
    next_segment_conditioning_offset: bool = False
    diffmean_and_probe_layer: NonNegativeIntLike = 6
    device_batch_size: PositiveIntLike = 16
    feature_diffmean_extraction_batch_size: PositiveIntLike | None = None

    context_cache: ContextCacheConfig = field(default_factory=ContextCacheConfig)
    general_val_bpb: GeneralValBpbConfig = field(default_factory=GeneralValBpbConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)

    pair_subsample: NonNegativeIntLike = 1
    steering_metric: SteeringMetric | Literal["roundtrip", "bpb_class_margin"] = (
        SteeringMetric.ROUNDTRIP.value
    )
    roundtrip_generation: RoundtripGenerationConfig = field(
        default_factory=RoundtripGenerationConfig
    )
    roundtrip_steering: RoundtripEvalConfig = field(
        default_factory=lambda: RoundtripEvalConfig(enabled=True)
    )
    scatter_baseline_x: ScatterBaselineX | Literal[
        "baseline_loss",
        "min",
        "general_val_bpb",
    ] = ScatterBaselineX.GENERAL_VAL_BPB.value
    # Plot-stage method whitelist: when set, ONLY these method names are plotted
    # (scatter + line plots + legend), filtered straight from the existing
    # aggregates at plot time -- no re-scoring. ``None`` => plot every method in
    # the aggregates. Distinct from ``base_methods``, which gates SCORING.
    plot_methods: list[str] | None = None
    # Optional production plot: per-method steering-metric-vs-base-strength
    # families plus the steerability frontier against general_val_bpb.
    plot_steering_strength_families: bool = False
    ours_gcg: OursGcgConfig = field(default_factory=OursGcgConfig)

    scores_path: str | None = None
    decoder_only_target: dict[str, Any] = field(default_factory=dict)
    encdec_target: dict[str, Any] = field(default_factory=dict)
    lora_adapter: dict[str, Any] = field(default_factory=dict)
    log_progress_every: int = 1000
    offload_between_stages: bool = True


def _require_non_empty(name: str, values: Sequence[Any]) -> None:
    if not values:
        raise AxBenchBpbAdvError(f"{name} must contain at least one value")


def _require_unique(name: str, values: Sequence[Any]) -> None:
    seen: set[str] = set()
    duplicates: list[Any] = []
    for value in values:
        key = f"number:{float(value)!r}" if isinstance(value, int | float) else repr(value)
        if key in seen:
            duplicates.append(value)
        seen.add(key)
    if duplicates:
        raise AxBenchBpbAdvError(
            f"{name} must not contain duplicate values, got duplicates {duplicates!r}"
        )


def method_config_name(config: AdversarialMethodConfig) -> AdversarialMethod:
    return AdversarialMethod.coerce(config.method)


def adversarial_method_configs_by_name(
    cfg: AxBenchBpbAdvConfig,
) -> dict[AdversarialMethod, AdversarialMethodConfig]:
    out: dict[AdversarialMethod, AdversarialMethodConfig] = {}
    for method_config in cfg.adversarial_methods:
        method = method_config_name(method_config)
        if method in out:
            raise AxBenchBpbAdvError(
                f"adversarial_methods contains duplicate method {method.value!r}"
            )
        out[method] = method_config
    return out


def get_faster_gcg_config(cfg: AxBenchBpbAdvConfig) -> FasterGcgConfig | None:
    method_config = adversarial_method_configs_by_name(cfg).get(AdversarialMethod.FASTER_GCG)
    if method_config is None:
        return None
    if not isinstance(method_config, FasterGcgConfig):
        raise AxBenchBpbAdvError("faster_gcg config has the wrong dataclass type")
    return method_config


def get_gepa_config(cfg: AxBenchBpbAdvConfig) -> GepaConfig | None:
    method_config = adversarial_method_configs_by_name(cfg).get(AdversarialMethod.GEPA)
    if method_config is None:
        return None
    if not isinstance(method_config, GepaConfig):
        raise AxBenchBpbAdvError("gepa config has the wrong dataclass type")
    return method_config


def get_additive_direction_config(
    cfg: AxBenchBpbAdvConfig,
) -> AdditiveDirectionConfig | None:
    method_config = adversarial_method_configs_by_name(cfg).get(
        AdversarialMethod.ADDITIVE_DIRECTION
    )
    if method_config is None:
        return None
    if not isinstance(method_config, AdditiveDirectionConfig):
        raise AxBenchBpbAdvError("additive_direction config has the wrong dataclass type")
    return method_config


def _has_enabled_decoder_only_base_method(cfg: AxBenchBpbAdvConfig) -> bool:
    """Whether this config can produce decoder-only attack rows.

    ``target_models`` alone is not enough: ``base_methods`` may whitelist only
    encdec methods while leaving the default target list unchanged.
    """

    if "decoder_only" not in cfg.target_models:
        return False
    enabled = None if cfg.base_methods is None else set(cfg.base_methods)
    return any(
        method.target_model == "decoder_only" and (enabled is None or method.name in enabled)
        for method in ALL_METHODS
    )


def _validate_required_inputs(cfg: AxBenchBpbAdvConfig) -> None:
    if not cfg.source_run_dir:
        raise AxBenchBpbAdvError("source_run_dir is required")
    if not cfg.output_root:
        raise AxBenchBpbAdvError("output_root is required")
    if not cfg.feature_dir:
        raise AxBenchBpbAdvError("feature_dir is required")
    if not cfg.feature_statistics_path:
        raise AxBenchBpbAdvError("feature_statistics_path is required")


def _validate_cluster_inputs(cfg: AxBenchBpbAdvConfig) -> None:
    if cfg.force_cluster_shard_revalidation and cfg.feature_clusters is None:
        raise AxBenchBpbAdvError("force_cluster_shard_revalidation=true requires feature_clusters")
    if cfg.feature_mask is None and cfg.feature_clusters is None:
        return
    try:
        parse_cluster_eval_paths(
            feature_mask=cfg.feature_mask,
            feature_clusters=cfg.feature_clusters,
        )
        validate_clustered_method_allowlist(
            cfg.base_methods,
            surface="axbench_bpb_adv",
        )
    except ClusterEvalConfigError as exc:
        raise AxBenchBpbAdvError(str(exc)) from exc
    if cfg.context_cache.force_rebuild:
        raise AxBenchBpbAdvError(
            "clustered context_cache.force_rebuild=true is forbidden because "
            "clustered cache evidence is immutable within a run_tag; use a "
            "fresh run_tag instead"
        )


def _validate_exclude_concept_modes(cfg: AxBenchBpbAdvConfig) -> None:
    _require_unique("exclude_concept_modes", cfg.exclude_concept_modes)
    for entry in cfg.exclude_concept_modes:
        parts = str(entry).split(":")
        if len(parts) != 2 or not parts[0].isdigit() or parts[1] not in (
            "positive",
            "negative",
        ):
            raise AxBenchBpbAdvError(
                "exclude_concept_modes entries must be "
                f"'<concept_id>:<positive|negative>', got {entry!r}"
            )


def _validate_representative_target_methods(cfg: AxBenchBpbAdvConfig) -> None:
    _require_unique(
        "representative_target_methods",
        cfg.representative_target_methods,
    )
    if not cfg.representative_target_methods:
        return
    allowed_target_methods = set(FAIR_FEATURE_DIFFMEAN_METHOD_BY_MODE.values())
    unknown_target_methods = sorted(set(cfg.representative_target_methods) - allowed_target_methods)
    if unknown_target_methods:
        raise AxBenchBpbAdvError(
            "representative_target_methods supports only fair "
            f"Feature-DiffMean target method(s) {sorted(allowed_target_methods)}, "
            f"got {unknown_target_methods}"
        )
    if "encdec" not in set(cfg.target_models):
        raise AxBenchBpbAdvError(
            "representative_target_methods requires target_models to include "
            "'encdec' so the fair Feature-DiffMean target rows can be scored"
        )
    if cfg.base_methods is None:
        return
    overlap = sorted(set(cfg.base_methods) & set(cfg.representative_target_methods))
    if overlap:
        raise AxBenchBpbAdvError(
            "representative_target_methods must not duplicate base_methods; "
            f"duplicate method(s): {overlap}"
        )


def _validate_plot_methods(cfg: AxBenchBpbAdvConfig) -> None:
    if cfg.plot_methods is None:
        return
    _require_non_empty("plot_methods", cfg.plot_methods)
    _require_unique("plot_methods", cfg.plot_methods)
    # Reject typos before an expensive all-stage scoring run. The plot stage
    # separately checks which valid methods are actually present in the run.
    known_methods = {method.name for method in ALL_METHODS}
    unknown_plot_methods = sorted(set(cfg.plot_methods) - known_methods)
    if unknown_plot_methods:
        raise AxBenchBpbAdvError(
            f"plot_methods contains unknown method name(s): "
            f"{unknown_plot_methods}. Available methods: {sorted(known_methods)}"
        )


def _validate_alpha_family_reg_lambda(cfg: AxBenchBpbAdvConfig) -> None:
    """Keep the alpha grid and the alpha methods consistent, in BOTH directions.

    Both errors are the same shape -- the derive sweeps a grid the score stage
    never resolves to -- and both are otherwise silent until a run has already
    paid for GPU-days of GCG optimization:

    * **alpha methods selected, ``reg_lambda_grid != (0.0,)``.** The
      ``ours_gcg_alpha*`` methods report ONE scalar coefficient and it is the
      alpha, so nothing is left to carry a reg_lambda; ``intervention.py``
      resolves their bags at the unregularised ``0.0`` while the derive sweeps
      the full ``(reg_lambda, alpha)`` product. Surfaces late as a missing-bag
      error. Sweep the lambda axis via ``ours_gcg_encdec`` in a separate run.
    * **non-default alpha grid, NO alpha method selected.** Every non-unit alpha
      derives a bag that no scored method resolves to (``ours_gcg_encdec`` reads
      alpha = 1 regardless). Never surfaces at all -- the run simply does extra
      work and discards it.

    Checking both here turns each into an immediate config error naming the fix.
    """
    if cfg.base_methods is None:
        return
    active_alpha_methods = sorted(set(cfg.base_methods) & ALPHA_AXIS_GCG_METHODS)
    alpha_grid = tuple(float(value) for value in cfg.ours_gcg.objective_alpha_grid)
    if not active_alpha_methods:
        # The symmetric error: a non-default alpha grid with NO alpha-axis
        # method derives a bag per alpha and then scores none of them --
        # `ours_gcg_encdec` resolves at alpha = 1 regardless. Silent, and each
        # wasted bag is a full GCG optimization, so this is GPU-days of work
        # producing artifacts nothing will ever read.
        if alpha_grid != (DEFAULT_OBJECTIVE_ALPHA,):
            raise AxBenchBpbAdvError(
                f"objective_alpha_grid is {list(alpha_grid)} but base_methods "
                f"selects no alpha-axis method (one of "
                f"{sorted(ALPHA_AXIS_GCG_METHODS)}). Every non-default alpha "
                "would derive a bag that no scored method resolves to, wasting "
                "a full GCG optimization each. Add ours_gcg_alpha_encdec (and/or "
                "its neg_ sibling), or leave objective_alpha_grid at "
                f"[{DEFAULT_OBJECTIVE_ALPHA}]."
            )
        return
    lambda_grid = tuple(float(value) for value in cfg.ours_gcg.reg_lambda_grid)
    if lambda_grid != (0.0,):
        raise AxBenchBpbAdvError(
            f"base_methods selects alpha-axis method(s) {active_alpha_methods}, "
            f"whose coefficient is objective_alpha, but reg_lambda_grid is "
            f"{list(lambda_grid)}. Those methods resolve their bags at "
            "reg_lambda=0.0, so any other grid derives bags the score stage "
            "will never load. Set reg_lambda_grid to [0.0] and sweep "
            "objective_alpha_grid, or sweep lambda via ours_gcg_encdec in a "
            "separate run."
        )


def _validate_method_selection(
    cfg: AxBenchBpbAdvConfig,
) -> dict[AdversarialMethod, AdversarialMethodConfig]:
    _require_non_empty("adversarial_methods", cfg.adversarial_methods)
    methods_by_name = adversarial_method_configs_by_name(cfg)
    if cfg.base_methods is not None:
        _require_non_empty("base_methods", cfg.base_methods)
        _require_unique("base_methods", cfg.base_methods)
    _validate_exclude_concept_modes(cfg)
    _validate_alpha_family_reg_lambda(cfg)
    _validate_representative_target_methods(cfg)
    _validate_plot_methods(cfg)
    return methods_by_name


def validate_all_stage_representative_target_contract(
    cfg: AxBenchBpbAdvConfig,
) -> None:
    """Reject fresh all-stage runs that cannot produce plot selector targets.

    The representative coefficient selector intentionally has one fixed target
    contract: each plotted steering mode needs its fair encoder-decoder
    Feature-DiffMean control rows. This check belongs at the evaluation
    entrypoint, rather than in :func:`validate_config`, because the sample
    generator reuses the default ``Stage.ALL`` enum while consuming an already
    completed aggregate artifact.

    Direct ``stage=plot`` replay is deliberately excluded. Its macro table is
    the source of truth and is validated immediately before plot cleanup by the
    aggregation layer.
    """

    if Stage.coerce(cfg.stage) is not Stage.ALL:
        return

    methods_by_name = {method.name: method for method in ALL_METHODS}
    target_models = {str(target_model) for target_model in cfg.target_models}
    if cfg.base_methods is None:
        selected_methods = [
            method for method in ALL_METHODS if method.target_model in target_models
        ]
    else:
        unknown = sorted(set(cfg.base_methods) - set(methods_by_name))
        if unknown:
            raise AxBenchBpbAdvError(
                f"unknown BPB-Adv base method(s): {unknown}. "
                f"Available methods: {sorted(methods_by_name)}"
            )
        selected_methods = [methods_by_name[name] for name in cfg.base_methods]
        wrong_target = sorted(
            method.name
            for method in selected_methods
            if method.target_model not in target_models
        )
        if wrong_target:
            raise AxBenchBpbAdvError(
                "requested BPB-Adv base method(s) target models not listed in "
                f"target_models={sorted(target_models)}: {wrong_target}"
            )

    selected_names = {method.name for method in selected_methods}
    selected_names.update(str(name) for name in cfg.representative_target_methods)
    selected_modes = {
        str(method.mode.value)
        for method in ALL_METHODS
        if method.name in selected_names
    }
    missing_by_mode = {
        mode: FAIR_FEATURE_DIFFMEAN_METHOD_BY_MODE[mode]
        for mode in sorted(selected_modes)
        if mode in FAIR_FEATURE_DIFFMEAN_METHOD_BY_MODE
        and FAIR_FEATURE_DIFFMEAN_METHOD_BY_MODE[mode] not in selected_names
    }
    if not missing_by_mode:
        return

    missing_methods = sorted(set(missing_by_mode.values()))
    modes = sorted(missing_by_mode)
    raise AxBenchBpbAdvError(
        "stage='all' cannot reach representative-coefficient plotting: selected "
        f"method(s) cover mode(s) {modes}, but the required fair Feature-DiffMean "
        f"target method(s) {missing_methods} are not selected. Add the target "
        "method(s) to base_methods or representative_target_methods and include "
        "target_models='encdec', or run only through stage='aggregate'. "
        "The selector has no decoder-only fallback; direct stage='plot' replay "
        "checks the actual macro artifact instead."
    )


def _validate_general_val_relationships(
    cfg: AxBenchBpbAdvConfig,
    scatter_x: ScatterBaselineX,
) -> GeneralValBpbConfig:
    gv = cfg.general_val_bpb
    if not isinstance(gv, GeneralValBpbConfig):
        raise AxBenchBpbAdvError("general_val_bpb must use GeneralValBpbConfig")
    gv.validate()
    if scatter_x is ScatterBaselineX.GENERAL_VAL_BPB and not gv.enabled:
        raise AxBenchBpbAdvError(
            "scatter_baseline_x='general_val_bpb' requires general_val_bpb.enabled=true"
        )
    if cfg.plot_steering_strength_families and not gv.enabled:
        # The frontier needs a capability x-coordinate for every family.
        raise AxBenchBpbAdvError(
            "plot_steering_strength_families=true requires "
            "general_val_bpb.enabled=true: the frontier's x-axis is "
            "general_val_bpb, so every plotted family needs a capability x. "
            "Enable general_val_bpb or disable the strength-family plots."
        )
    return gv


def _validate_roundtrip_config(
    cfg: AxBenchBpbAdvConfig,
    steering_metric: SteeringMetric,
) -> None:
    generation = cfg.roundtrip_generation
    if not isinstance(generation, RoundtripGenerationConfig):
        raise AxBenchBpbAdvError("roundtrip_generation must be RoundtripGenerationConfig")
    steering = cfg.roundtrip_steering
    if not isinstance(steering, RoundtripEvalConfig):
        raise AxBenchBpbAdvError("roundtrip_steering must be RoundtripEvalConfig")
    if steering_metric is SteeringMetric.ROUNDTRIP and not steering.enabled:
        raise AxBenchBpbAdvError(
            "roundtrip_steering.enabled=false is incompatible with steering_metric='roundtrip'"
        )
    if steering_metric is not SteeringMetric.ROUNDTRIP:
        return
    # Probe the same model family that produced the SAE feature indices.
    extraction_model_id = effective_extraction_model_id(cfg)
    if steering.probe.model_name != extraction_model_id:
        raise AxBenchBpbAdvError(
            "axbench-bpb-adv roundtrip probe model_name must match "
            "feature_extraction_model_id (or neuronpedia_model_id when "
            "unset) so Concept.feature_index is probed in the exact SAE "
            "model family"
        )
    _validate_roundtrip_probe_l0(cfg, steering)


def _validate_roundtrip_probe_l0(
    cfg: AxBenchBpbAdvConfig,
    steering: RoundtripEvalConfig,
) -> None:
    probe_l0 = steering.probe.l0
    if probe_l0 is not None and not isinstance(probe_l0, L0Tier):
        raise AxBenchBpbAdvError(
            "axbench-bpb-adv roundtrip scoring does not support numeric "
            "roundtrip_steering.probe.l0 because source Concept.sae_source "
            "metadata does not encode numeric GemmaScope v1 L0 variants; "
            "GemmaScope-2 named tiers are supported"
        )
    if not isinstance(probe_l0, L0Tier):
        return
    try:
        *_identity, source_l0 = parse_neuronpedia_source(cfg.neuronpedia_source)
    except ValueError as exc:
        raise AxBenchBpbAdvError(
            "could not validate roundtrip probe l0 tier against "
            f"neuronpedia_source={cfg.neuronpedia_source!r}: {exc}"
        ) from exc
    if source_l0 != probe_l0:
        raise AxBenchBpbAdvError(
            "roundtrip_steering.probe.l0 tier must match neuronpedia_source: "
            f"probe={probe_l0.value!r}, "
            f"source={getattr(source_l0, 'value', source_l0)!r}"
        )


def _validate_faster_gcg_general_val(
    cfg: AxBenchBpbAdvConfig,
    faster_gcg: FasterGcgConfig,
    gv: GeneralValBpbConfig,
) -> None:
    if not (
        gv.enabled
        and _has_enabled_decoder_only_base_method(cfg)
        and faster_gcg.optimized_token_counts
    ):
        return
    # The universal trigger replaces the final k non-BOS prefix tokens.
    largest_k = max(faster_gcg.optimized_token_counts)
    if largest_k > gv.prefix_tokens:
        raise AxBenchBpbAdvError(
            "general_val_bpb.prefix_tokens "
            f"({gv.prefix_tokens}) must be >= the largest "
            "faster_gcg.optimized_token_counts "
            f"({largest_k}): the universal trigger is substituted into the "
            "last-k tokens of each BOS-prefixed decoder-only validation doc, "
            "which needs at least k non-BOS prefix tokens."
        )


def _validate_adversarial_method_configs(
    cfg: AxBenchBpbAdvConfig,
    methods_by_name: dict[AdversarialMethod, AdversarialMethodConfig],
    gv: GeneralValBpbConfig,
) -> None:
    faster_gcg = methods_by_name.get(AdversarialMethod.FASTER_GCG)
    if faster_gcg is not None:
        if not isinstance(faster_gcg, FasterGcgConfig):
            raise AxBenchBpbAdvError("faster_gcg must use FasterGcgConfig")
        _validate_faster_gcg_general_val(cfg, faster_gcg, gv)

    gepa = methods_by_name.get(AdversarialMethod.GEPA)
    if gepa is not None:
        if not isinstance(gepa, GepaConfig):
            raise AxBenchBpbAdvError("gepa must use GepaConfig")
        if gepa.reflection_model is None:
            raise AxBenchBpbAdvError("gepa adversarial method requires reflection_model")

    additive_direction = methods_by_name.get(AdversarialMethod.ADDITIVE_DIRECTION)
    if additive_direction is not None:
        if not isinstance(additive_direction, AdditiveDirectionConfig):
            raise AxBenchBpbAdvError("additive_direction must use AdditiveDirectionConfig")


def validate_config(cfg: AxBenchBpbAdvConfig) -> None:
    _validate_required_inputs(cfg)
    _validate_cluster_inputs(cfg)
    Stage.coerce(cfg.stage)
    steering_metric = SteeringMetric.coerce(cfg.steering_metric)
    methods_by_name = _validate_method_selection(cfg)
    if cfg.next_segment_conditioning_offset:
        raise AxBenchBpbAdvError(
            "axbench-bpb-adv next_segment_conditioning_offset=true is not supported in v1"
        )

    scatter_x = ScatterBaselineX.coerce(cfg.scatter_baseline_x)

    gv = _validate_general_val_relationships(cfg, scatter_x)
    _validate_roundtrip_config(cfg, steering_metric)

    _validate_adversarial_method_configs(cfg, methods_by_name, gv)
