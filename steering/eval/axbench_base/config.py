"""Hydra-compatible config dataclasses for axbench_base.

Note: the config is designed to be constructed from an OmegaConf dict via
`AxBenchBaseConfig(**OmegaConf.to_container(cfg, resolve=True))`, so all
fields are plain Python types (no Hydra-specific annotations).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from dataclasses import replace as _dc_replace
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Literal, Protocol

from beartype import beartype

from steering.context_defaults import DEFAULT_CONTEXT_WINDOW
from steering.eval.axbench_base.percentile_units import (
    PERCENTILE_POINTS_PER_FRACTION,
    percentile_fraction_to_points,
)
from steering.eval.axbench_base.prefix_text import validate_prefix_char_bounds
from steering.eval.axbench_base.profiler import ProfileConfig
from steering.eval.axbench_base.types import (
    MAX_RAW_PERCENTILE,
    AxBenchBaseError,
)
from steering.eval.cluster_config import (
    CLUSTERED_AXBENCH_BASE_SOURCE_METHODS,
    CLUSTERED_AXBENCH_BASE_STAGE4_METHODS,
    ClusterEvalConfigError,
    parse_cluster_eval_paths,
    validate_clustered_method_allowlist,
)
from steering.eval.neuronpedia_config import NeuronpediaConfig
from steering.feature_paths import FeatureDirectoryComponents, parse_feature_dir
from steering.refined_types import (
    NonBlankStrLike,
    NonNegativeFiniteNumberLike,
    NonNegativeIntLike,
    OpenUnitIntervalLike,
    PositiveIntLike,
    UniqueNonEmptyNonNegativeIntListLike,
    UnitIntervalLike,
    UnitIntervalOpenLowerLike,
    UnitIntervalOpenUpperLike,
)
from steering.sae_resolver import parse_neuronpedia_source, resolve_neuronpedia_ids
from steering.sae_resolver_v2_data import GEMMASCOPE_2_HF_MODEL_IDS


class ExtractionModelIdentityConfig(Protocol):
    """Config capability needed to resolve the local extraction model id."""

    @property
    def feature_extraction_model_id(self) -> str | None: ...

    @property
    def neuronpedia_model_id(self) -> str: ...


class TargetMetadataConfig(Protocol):
    """Config capability needed when detecting target-model metadata."""

    @property
    def feature_dir(self) -> str: ...

    @property
    def device(self) -> str: ...


class FeatureStatisticsPathConfig(Protocol):
    """Config capability needed by feature-statistics-backed helpers."""

    @property
    def feature_statistics_path(self) -> str: ...


class VerificationPrefillMode(StrEnum):
    """Stage 0c verification prefill algorithms.

    ``SHARED`` prefills each candidate prefix once and expands its public
    Transformers cache state into the prefix-major seed-row layout before
    decoding. ``REPEATED`` retains the pre-schema-8 implementation, which
    prefills one repeated prompt row per verification seed.

    The selected value is semantic Stage 0c identity, not a transparent
    performance toggle: transformer numerics can depend on prefill batch shape.
    ``StrEnum`` keeps Hydra and JSON representations as the plain strings
    ``"shared"`` and ``"repeated"``.
    """

    SHARED = "shared"
    REPEATED = "repeated"


@dataclass(frozen=True)
class PositiveQcThresholds:
    """Validated positive Stage 1b thresholds in raw percentile-point units."""

    accept_fraction: float
    headroom_ceiling_fraction: float
    required_delta: float
    completion_ceiling: float
    maximum_seed_percentile: float


def derive_positive_qc_thresholds(
    accept_threshold: object,
    headroom_ceiling: object,
) -> PositiveQcThresholds:
    """Validate and convert positive-QC fractions to percentile-point units.

    Raw SAE QC percentiles are ``0.0`` for no activation or lie in
    ``[0.5, MAX_RAW_PERCENTILE]`` for a firing feature. Config values are
    fractions of the
    nominal 100-point percentile space, but the representable completion
    ceiling is capped at ``99.5``.
    """

    required_delta = percentile_fraction_to_points(
        accept_threshold,
        field_name="accept_threshold",
    )
    requested_completion_ceiling = percentile_fraction_to_points(
        headroom_ceiling,
        field_name="positive_prefix_headroom_ceiling",
    )
    accept_fraction = required_delta / PERCENTILE_POINTS_PER_FRACTION
    ceiling_fraction = requested_completion_ceiling / PERCENTILE_POINTS_PER_FRACTION
    if not 0.0 < accept_fraction < ceiling_fraction <= 1.0:
        raise AxBenchBaseError(
            "positive Stage 1b QC requires "
            "0 < accept_threshold < positive_prefix_headroom_ceiling <= 1; "
            f"got accept_threshold={accept_fraction!r}, "
            f"positive_prefix_headroom_ceiling={ceiling_fraction!r}"
        )

    completion_ceiling = min(
        requested_completion_ceiling,
        MAX_RAW_PERCENTILE,
    )
    maximum_seed_percentile = completion_ceiling - required_delta
    if maximum_seed_percentile < 0.0:
        raise AxBenchBaseError(
            "positive Stage 1b QC has no representable prefix-headroom band: "
            f"required_delta={required_delta}, "
            f"representable_completion_ceiling={completion_ceiling}, "
            f"maximum_seed_percentile={maximum_seed_percentile}. "
            f"The raw percentile maximum is {MAX_RAW_PERCENTILE}."
        )
    return PositiveQcThresholds(
        accept_fraction=accept_fraction,
        headroom_ceiling_fraction=ceiling_fraction,
        required_delta=required_delta,
        completion_ceiling=completion_ceiling,
        maximum_seed_percentile=maximum_seed_percentile,
    )


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 64
    eval_seeds_per_concept_positive: int = 10
    eval_seeds_per_concept_negative: int = 10
    master_seed: int = 42
    temperature: float = 0.7
    top_k_sampling: int = 50


@dataclass(frozen=True)
class LLMClientConfig:
    """Shared config for LLM API clients (judge + synth)."""

    provider: str = "openrouter"
    model: str = "openai/gpt-5-nano"
    requests_per_minute: int = 60
    max_concurrent: int = 10
    # Wall-clock budget (seconds) for backing off a *retryable* status
    # (429 / 5xx). Attempt-count budgets are the wrong unit for provider
    # capacity exhaustion: an upstream 429 clears on its own schedule, not
    # after N tries. When positive, retryable statuses keep retrying until
    # this much cumulative backoff has elapsed, even once
    # ``max_retries`` attempts are spent; zero keeps pure attempt-count
    # semantics. Non-retryable failures (4xx, malformed body) are
    # unaffected — waiting does not heal a bad request.
    transport_retry_budget_seconds: float = 0.0


@beartype
@dataclass(frozen=True)
class JudgeConfig(LLMClientConfig):
    # Additional API calls beyond the first on a parse failure. 3 extra
    # attempts (4 total) absorbs the ~17% bare-`Rating: N` emission rate
    # observed against openai/gpt-5-nano plus a safety margin; the old
    # default of 1 was too tight and let format-drift turn into silent
    # invalid-rows under the 1% warning threshold.
    parse_max_retries: NonNegativeIntLike = 3
    invalid_warning_threshold: float = 0.01
    invalid_drop_threshold: float = 0.05


@beartype
@dataclass(frozen=True)
class SynthConfig(LLMClientConfig):
    # Synth calls are naturally batched and Stage 0c's strict parser + local
    # SAE gate make failed provider admissions ordinary retryable proposal
    # misses. Keep judge traffic conservative while allowing synth workloads
    # to use the higher OpenRouter ceiling independently.
    model: str = "inclusionai/ling-2.6-flash"
    requests_per_minute: int = 6_000
    max_concurrent: int = 100
    # Additional HTTP attempts after the first request. This is deliberately
    # independent of AxBenchBaseConfig.max_retries: transport failures back off,
    # while content/QC rejection sampling should resample immediately.
    transport_max_retries: NonNegativeIntLike = 3
    # The default synth model is served by a single upstream provider, whose
    # shared capacity returns transient 429s at a double-digit rate even at
    # concurrency 1 and clears within seconds. Under attempt-count-only
    # semantics the tolerance is 1+2+4 = 7s of backoff, so over a long stage
    # some in-flight call eventually burns all four attempts inside one
    # outage and aborts the whole run. Five minutes of cumulative backoff
    # absorbs those bursts while still failing loudly on a real outage.
    transport_retry_budget_seconds: NonNegativeFiniteNumberLike = 300.0


@dataclass(frozen=True)
class LoraConfig:
    rank: int = 8
    alpha: float = 16.0
    lr: float = 1e-4
    steps: int = 100
    batch_size: int = 8
    min_training_rows: int = 8


@beartype
@dataclass(frozen=True)
class LayerHooksConfig:
    diffmean_and_probe_layer: NonNegativeIntLike = 6


@beartype
@dataclass(frozen=True)
class AxBenchBaseConfig:
    # --- Mandatory fields (environment-specific, no sensible default) ---
    feature_dir: str
    feature_statistics_path: str
    decoder_only: dict[str, Any]
    encdec: dict[str, Any]
    device: str
    output_root: NonBlankStrLike | None

    # --- Defaulted fields ---
    run_tag: str | None = None  # auto-generated by CLI if None
    # #1249: adv-run `ours_gcg_bags` store holding the committed fair
    # feature-DiffMean bags. REQUIRED whenever stage 3a/3b trains the encdec
    # target: those forwards are conditioned by the concept's bag, and there is
    # deliberately no unconditioned fallback. None is valid only for runs that
    # never fit encdec-target directions.
    encdec_fit_bag_store: str | None = None
    validation_shards: UniqueNonEmptyNonNegativeIntListLike = field(
        default_factory=lambda: [369]
    )  # locked-in decision #19
    neuronpedia_source: str = "8-gemmascope-res-16k"  # matches default SAE
    neuronpedia_model_id: str = "gemma-2-2b"  # matches default SAE
    # Optional local model id used only for loading the feature-extraction
    # model + resolving the SAELens SAE. Defaults to neuronpedia_model_id,
    # but Gemma-3 PT needs Neuronpedia's public "gemma-3-4b" namespace while
    # local model/SAE loading uses "gemma-3-4b-pt".
    feature_extraction_model_id: str | None = None
    extraction_max_tokens: int = DEFAULT_CONTEXT_WINDOW  # hard upper bound on tokens fed
    # to base Gemma in Stage 2 / Stage 4 extraction. Lets you swap to a
    # longer-context target model without touching code.
    extraction_threshold: float = 0.0  # spec default
    extraction_tokenizer_id: str = "google/gemma-2-2b"  # matches default SAE vocab
    run_negative_mode: bool = True  # usually want both modes
    concept_feature_mode: Literal["canonical", "delta_ranked"] = "canonical"
    n_rank_store: int = 256  # spec default
    min_corpus_count: int = 10  # spec default
    min_description_length: int = 10  # spec default
    concept_list_seed: int = 42  # standard seed
    max_concepts: PositiveIntLike = 500  # spec default
    # Stage 0c lazy LLM-based concept examples QC (review-fix F / G)
    enable_concept_examples_qc: bool = True  # on by default (fix G)
    n_activation_examples: int = 5  # how many top examples per concept
    activation_example_judge_min_score: int = 1  # 0-2 scale, drop if below
    # Optional Stage 0c live-API cap. None preserves the production
    # contract: examples QC evaluates candidates until Stage 0c has
    # enough final concepts or exhausts the pre-QC pool. Profiling/dev
    # runs may set this to bound how many QC survivors move on to
    # verification.
    concept_examples_qc_target_surviving: NonNegativeIntLike | None = None
    # Optional Stage 0a dev/profiling cap applied before fetching
    # Neuronpedia descriptions. None preserves the production behavior
    # of scanning every count-filtered candidate.
    concept_description_fetch_limit: NonNegativeIntLike | None = None
    # When True, Stage 0a uses ONLY Neuronpedia descriptions already in the
    # on-disk cache and makes no API calls. Uncached features are silently
    # omitted from candidacy. This freezes the description set to the cache,
    # making concept selection fully reproducible (and fast) against a prior
    # run's cached descriptions, and avoiding repeated re-querying of features
    # Neuronpedia does not publish (which are never cached). Defaults to the
    # production behavior of fetching every count-filtered candidate.
    concept_descriptions_cache_only: bool = False
    # Optional Stage 0a concept-candidate whitelist file. Restricts candidate
    # features to an explicit set BEFORE Neuronpedia description fetching, so a
    # curated autointerp feature list bounds the fetch/QC cost. Concept-candidate
    # filtering is distinct from the downstream conditioning.feature_mask. The
    # path accepts accepted_feature_ids.txt or manual_feature_filter.yaml. See
    # concept_whitelist.py. Every requested ID must be present in the selected
    # feature statistics; absent (dead or wrong-SAE/layer) IDs fail Stage 0a.
    concept_feature_whitelist_path: str | None = None
    # Stage 0a fails loud when more than this fraction of the count-filtered
    # candidate features return HTTP 404 / not-published from Neuronpedia. A
    # high not-published rate means the configured ``neuronpedia_source`` /
    # ``neuronpedia_model_id`` does not match the SAE being evaluated (e.g. a
    # 16k source against a 65k SAE), which would otherwise silently truncate
    # and mislabel the concept pool. A correctly-configured run sees ~0%.
    # ``0.0`` disables the guard; raise toward 1.0 for an SAE with genuinely
    # sparse Neuronpedia coverage. ``concept_descriptions_cache_only`` runs
    # skip the check.
    max_not_published_description_rate: UnitIntervalLike = 0.02
    # Companion guard for a source that answers HTTP 200 with *empty*
    # explanations (rather than 404) for indices it does not host: Stage 0a
    # also fails loud when more than this fraction of candidates have no usable
    # description at all (404 + empty). A correctly-configured run has only a
    # small genuine-empty baseline, so this threshold is deliberately loose (a
    # mass mismatch is nearly all of them). ``0.0`` disables it.
    max_missing_description_rate: UnitIntervalLike = 0.5
    # Backstop guard for a source that is wholly nonexistent / typo'd (the live
    # API answers 5xx, not 404, for every index -- so the two guards above stay
    # 0 and miss it) or a rate-limit storm that would silently truncate the
    # pool. Fails loud when more than this fraction of candidates could not be
    # fetched for ANY reason (404 + empty + transient after retries). A correct,
    # reachable source fails ~none, so this is deliberately loose. ``0.0``
    # disables it.
    max_failed_description_rate: UnitIntervalLike = 0.5
    # Absolute floor for all three guards above: a mismatch is only raised when
    # the offending count also reaches this many features. Keeps a small
    # candidate pool (concept_feature_whitelist_path /
    # concept_description_fetch_limit) from tripping on a handful of
    # legitimately-missing features or a brief network blip.
    min_source_mismatch_features: NonNegativeIntLike = 5
    concept_examples_qc_judge: LLMClientConfig = field(default_factory=LLMClientConfig)
    seed_prefix_max_nanochat_tokens: PositiveIntLike = 512
    # count. Prefix is the tokens immediately before the continuation slice
    # (i.e., doc_tokens[:-M]) capped at this length. Variable in practice.
    # Reviewer override: was 64 (spec), then 256. 512 nanochat tokens
    # (~2048 chars of English) gives the base model even more context to
    # latch onto a concept before being asked to continue, still well under
    # the model's sequence_len (typically >= the 1024 project default).
    seed_continuation_nanochat_tokens: PositiveIntLike = 256
    # each source document to store as the natural continuation in each
    # SeedRow. Used by Stage 1b's positive-mode natural tier
    # (``synthesize_positive_mode``) as the per-prefix concept-avoiding
    # completion: the seed's own natural continuation, accepted iff the
    # concept feature does NOT fire on it. Every SeedRow has EXACTLY this
    # many continuation tokens — the last M tokens of the source document.
    seed_pool_train_ratio: OpenUnitIntervalLike = 0.75  # spec default
    seed_pool_seed: int = 42  # standard seed
    seed_extraction_candidate_shard_size: PositiveIntLike = 1024
    seed_extraction_gpu_batch_size: PositiveIntLike = 16
    seed_extraction_cpu_batch_size: PositiveIntLike = 64
    n_verified_prefixes_per_concept_target: int = 30  # spec default
    n_verification_seeds: int = 5  # spec default
    # Fraction of percentile space. Stage 0c resolves 0.7 to 70 raw percentile
    # points and keeps a continuation only when its output percentile is
    # strictly greater than that threshold.
    verification_percentile_threshold: UnitIntervalLike = 0.7  # spec default
    min_consistency_passes: int = 3  # spec default
    verification_candidate_scan_limit: NonNegativeIntLike | None = None
    verification_candidate_attempt_limit: NonNegativeIntLike | None = None
    verification_zero_hit_bail_after: PositiveIntLike | None = 2000
    # Per-candidate-concept Stage 0c short-circuit. When set, abort
    # verification for a candidate as soon as it has produced ZERO
    # verified prefixes after this many candidate-prefix attempts
    # (treated as a normal not_enough_verified_prefixes rejection).
    # ``None`` disables the bailout and preserves the legacy behavior
    # of scanning every candidate prefix in the pool. Applies within
    # the post-scan verification loop; if
    # verification_candidate_attempt_limit is smaller, that limit
    # takes effect first. Default 2000 cuts dead-feature wall-clock
    # without touching candidates that show any positive signal.
    verification_length: int = 64  # spec default
    # OpenRouter model for Stage 0c verification generation. The shipped
    # default asks this API model for evidence-grounded concept-rich
    # continuations, then applies the same local SAE percentile/consistency
    # rejection gate. Set ``None`` explicitly to preserve the legacy local
    # feature-extraction-model sampler exactly.
    verification_api_model: str | None = "inclusionai/ling-2.6-flash"
    # Maximum candidate prefixes proposed in one API verification wave. The
    # candidate pool/attempt limit and zero-hit bailout may make the effective
    # wave smaller. A wave may intentionally exceed the remaining accepted-
    # prefix target to keep provider workers occupied at low hit rates; every
    # proposal is scored, while only the first target passes are retained.
    # This is intentionally distinct from the local
    # Gemma cross-prefix batch size below: API calls consume provider capacity,
    # while local batches consume GPU KV-cache memory and are part of the exact
    # legacy schema-8 compatibility contract.
    verification_api_prefix_wave_size: int = 100
    # Stage 0c performance knobs (PR-2)
    qc_batch_size: PositiveIntLike = 16
    qc_pipeline: bool = True
    verification_prefill_mode: VerificationPrefillMode | Literal["shared", "repeated"] = (
        VerificationPrefillMode.SHARED
    )
    verification_cross_prefix_batch_size: PositiveIntLike = 4
    # Multi-GPU Stage 0c (PR-4): block size B. Each block spans B*world_size
    # candidates; rank R takes B indices within the block before the all_gather
    # rendezvous. Default 16 keeps the loop responsive while amortising the
    # all_gather cost over enough work. Set higher (32-64) to reduce gather
    # frequency on slower interconnects; lower (4-8) to surface a runaway
    # rank earlier when iterating on the multi-GPU code.
    stage0c_multi_gpu_block_size: PositiveIntLike = 16
    # Shared decoded-text bounds for Stage 0c candidate prefixes and Stage 1b
    # positive seed prefixes. Both stages keep the most recent characters;
    # Stage 1b applies this as a view and never mutates Stage 0b seed artifacts.
    min_prefix_chars: NonNegativeIntLike = 100  # spec default
    max_prefix_chars: PositiveIntLike = 2048  # spec default
    # Stage 1b complete accepted rich/avoiding pair target. Candidate
    # exhaustion triggers deterministic replacement, not a smaller roster.
    n_positives_per_concept: PositiveIntLike = 72  # spec default
    # Stage 1b complete-pair replacement policy. This is a TOTAL semantic
    # attempt budget per seed and per generated side (rich and, only when the
    # natural continuation fails strict-zero QC, LLM avoiding). Transport
    # retries remain independently owned by SynthConfig.
    positive_max_attempts_per_seed: PositiveIntLike = 8
    #: Persist Stage 1b prefix-percentile scores across runs. They are a pure
    #: function of the existing cache key, so a warm run is observationally
    #: identical to a cold one -- this only avoids repeating ~8.5 minutes of
    #: pre-provider SAE scoring on every relaunch. Disable to force recompute.
    enable_prefix_percentile_store: bool = True
    # Maximum distinct Stage 1b candidates admitted while a concept still has
    # zero complete accepted pairs. None disables the zero-hit guard.
    positive_zero_hit_bail_after: PositiveIntLike | None = 2000
    output_length: int = 64  # spec default
    # Fraction of the nominal 100-point percentile range. Stage 1b requires the
    # generated continuation to raise the target by at least
    # ``accept_threshold * 100`` raw percentile points.
    accept_threshold: UnitIntervalOpenLowerLike = 0.3
    # Fractional completion ceiling used to reserve feasible prefix headroom.
    # With accept_threshold=0.3 and this default=0.9, positive prefixes must
    # score <= 60 raw percentile points for their assigned concept target.
    positive_prefix_headroom_ceiling: UnitIntervalOpenLowerLike = 0.9
    # Additional content proposals after the first for Stage 0c and Stage 1c.
    # Stage 1b deliberately does not consult this compatibility field; it uses
    # positive_max_attempts_per_seed total attempts and replaces failed seeds.
    max_retries: NonNegativeIntLike = 100
    qc_fallback_on_exhaust: Literal["error", "warn", "best"] = "error"
    # Exclude a Stage 1 concept when the fraction of logical items lacking a
    # complete, accepted rich/avoiding pair is strictly greater than this.
    max_concept_drop_fraction: UnitIntervalOpenUpperLike = 0.5
    # Abort a Stage 1 stage after writing its yield report when the fraction of
    # concepts excluded by the per-concept guard is strictly greater than this.
    max_dropped_concept_fraction: UnitIntervalOpenUpperLike = 0.5
    synth: SynthConfig = field(default_factory=SynthConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    neuronpedia: NeuronpediaConfig = field(default_factory=NeuronpediaConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    layer_hooks: LayerHooksConfig = field(default_factory=LayerHooksConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    conditioning: Mapping[str, Any] | None = None
    stage: str = "all"
    enabled_methods: list[str] | None = None
    #: Producer-side ``selection_splice_batch_size`` used by
    #: ``precompute_selected_features`` when it wrote this checkpoint's
    #: ``sel/<hash>/`` overlay. It participates in the selection hash but is not
    #: recorded in checkpoint metadata, so a splice-trained checkpoint whose
    #: producer batched (> 1) must state it here or manifest discovery finds
    #: nothing. None reproduces the unbatched producer identity.
    selection_splice_batch_size: int | None = None
    #: Explicit ``sel/<hash>/manifest.json`` to pin, bypassing discovery. Needed
    #: when one feature root holds several packed-row splits (train / val /
    #: val-subset) that share a selection hash: discovery finds all of them and
    #: refuses to guess, since their manifests are semantically distinct.
    selection_manifest_path: str | None = None
    force_cluster_shard_revalidation: bool = False
    disabled_methods: list[str] = field(default_factory=list)

    def replace(self, **kwargs) -> AxBenchBaseConfig:
        return _dc_replace(self, **kwargs)


def effective_extraction_model_id(cfg: ExtractionModelIdentityConfig) -> str:
    """The local model id used for SAE/model loading and tokenization.

    Resolves the #667 API/local split: ``feature_extraction_model_id`` when
    set, otherwise ``neuronpedia_model_id`` (the Neuronpedia API id). This is
    the single source of truth for that rule; loaders (``runtime_state``,
    Stage 0c) call it directly. Callers that compare the result against a
    *bare* id -- the ``feature_dir`` ``m_{model}`` segment or a
    :data:`GEMMASCOPE_2_HF_MODEL_IDS` key -- must additionally normalize it to
    the last path component (``.rsplit("/", 1)[-1]``) so an org-prefixed form
    like ``google/gemma-2-2b`` compares equal to the bare ``gemma-2-2b``.
    """
    return cfg.feature_extraction_model_id or cfg.neuronpedia_model_id


def bare_model_id(model_id: str) -> str:
    """The last path component of a model / tokenizer id, so an org- or
    path-prefixed form (``google/gemma-2-2b``, ``/models/gemma-3-4b-pt/``)
    compares equal to the bare ``gemma-2-2b`` written into ``feature_dir``'s
    ``m_{model}`` segment and used as a :data:`GEMMASCOPE_2_HF_MODEL_IDS` key.
    ``PurePosixPath.name`` handles a trailing separator (``.../`` -> the
    basename, not an empty string).
    """
    return PurePosixPath(model_id).name


def _validate_clustered_conditioning(cfg: AxBenchBaseConfig) -> None:
    conditioning = cfg.conditioning
    # Named BUG-FIX deviation from the oracle (review finding 4, MEDIUM):
    # the oracle's own version of this check requires isinstance(dict)
    # unconditionally, before even looking at feature_clusters -- a
    # previously-accepted non-dict Mapping (e.g. a UserDict) conditioning
    # with NO feature_clusters (the UNCLUSTERED case, which pre-C2 never
    # validated cfg.conditioning's type at all) would then start raising,
    # a behavior change on the unclustered path. isinstance(Mapping) here
    # instead matches clustered_source_enabled's own acceptance criterion
    # (cluster_artifacts.py:226: isinstance(conditioning, Mapping) +
    # .get(...)) and A4's parse_path_feature_mask/parse_path_feature_
    # clusters (both isinstance(raw, Mapping) too) -- consistent across
    # the whole clustered-validation stack, not just this one block.
    feature_clusters = None if conditioning is None else conditioning.get("feature_clusters")
    if cfg.force_cluster_shard_revalidation and feature_clusters is None:
        raise AxBenchBaseError(
            "force_cluster_shard_revalidation=true requires clustered conditioning"
        )
    if feature_clusters is None:
        return

    assert isinstance(conditioning, Mapping)
    try:
        parse_cluster_eval_paths(
            feature_mask=conditioning.get("feature_mask"),
            feature_clusters=feature_clusters,
        )
        validate_clustered_method_allowlist(
            cfg.enabled_methods,
            surface="axbench_base",
        )
    except ClusterEvalConfigError as exc:
        raise AxBenchBaseError(str(exc)) from exc
    if cfg.concept_feature_mode != "canonical":
        raise AxBenchBaseError(
            "clustered AxBench-Base requires "
            "concept_feature_mode='canonical'; delta_ranked is unsupported"
        )
    clustered_supported_stages = {
        "0a",
        "0b",
        "0c",
        "1b",
        "1c",
        "2",
        "3a",
        "3b",
        "3e",
        "4",
    }
    if str(cfg.stage) not in clustered_supported_stages:
        raise AxBenchBaseError(
            "clustered AxBench-Base supports source construction plus exact "
            "Stage 3e context enumeration and Stage 4 generation, one stage "
            f"at a time: {sorted(clustered_supported_stages)}; got "
            f"stage={cfg.stage!r}. Prompting/LoRA training and later judge/"
            "aggregation stages remain unsupported."
        )
    stage_methods = (
        CLUSTERED_AXBENCH_BASE_STAGE4_METHODS
        if str(cfg.stage) in {"3e", "4"}
        else CLUSTERED_AXBENCH_BASE_SOURCE_METHODS
    )
    assert cfg.enabled_methods is not None
    stage_method_offenders = sorted(set(cfg.enabled_methods) - set(stage_methods))
    if stage_method_offenders:
        raise AxBenchBaseError(
            f"clustered AxBench-Base stage {cfg.stage!s} does not support "
            f"enabled_methods={stage_method_offenders}; allowed methods are "
            f"{list(stage_methods)}"
        )


def _validate_positive_synthesis(cfg: AxBenchBaseConfig) -> None:
    validate_prefix_char_bounds(
        cfg.min_prefix_chars,
        cfg.max_prefix_chars,
    )
    derive_positive_qc_thresholds(
        cfg.accept_threshold,
        cfg.positive_prefix_headroom_ceiling,
    )


def _validated_extraction_model(cfg: AxBenchBaseConfig) -> str:
    # The extraction tokenizer MUST be the extraction model's own tokenizer:
    # the SAE was trained in that model's tokenization, so a mismatch silently
    # corrupts Stage 2/4 feature extraction. For gemma-scope-2 (v2) models the
    # canonical HF tokenizer is known, so enforce it. The extraction model is
    # ``feature_extraction_model_id or neuronpedia_model_id`` (mirrors
    # runtime_state). v1 models are absent from the map and are not checked.
    # Model / tokenizer ids are compared by bare name so an org- or
    # path-prefixed spelling (google/..., /models/...) is accepted.
    if not (cfg.feature_extraction_model_id or cfg.neuronpedia_model_id):
        raise AxBenchBaseError(
            "neuronpedia_model_id (or feature_extraction_model_id) must be a "
            "non-empty model id; got neuronpedia_model_id="
            f"{cfg.neuronpedia_model_id!r}, feature_extraction_model_id="
            f"{cfg.feature_extraction_model_id!r}."
        )
    # The model ids that feed the SAE resolver (resolve_sae) and the Neuronpedia
    # API MUST be bare: both are keyed by the bare name, so an org-/path-prefixed
    # id passes the identity checks below (which compare bare names) yet crashes
    # at Stage 0c SAE load / mislabels at API-query time. Fail loud at config
    # time instead. (``extraction_tokenizer_id`` may stay org-prefixed -- it is a
    # full HF tokenizer id, not an SAE-resolver / Neuronpedia key.)
    for model_field, model_value in (
        ("feature_extraction_model_id", cfg.feature_extraction_model_id),
        ("neuronpedia_model_id", cfg.neuronpedia_model_id),
    ):
        if model_value is not None and bare_model_id(model_value) != model_value:
            raise AxBenchBaseError(
                f"{model_field}={model_value!r} must be a BARE model id "
                f"('{bare_model_id(model_value)}'), not org-/path-prefixed. The "
                "SAE resolver and the Neuronpedia API are keyed by the bare "
                "name, so an org-prefixed id passes the identity checks (which "
                "compare bare) but fails at SAE load / query time."
            )
    extraction_model = bare_model_id(effective_extraction_model_id(cfg))
    expected_tokenizer = GEMMASCOPE_2_HF_MODEL_IDS.get(extraction_model)
    if expected_tokenizer is not None and (
        not cfg.extraction_tokenizer_id
        or bare_model_id(cfg.extraction_tokenizer_id) != bare_model_id(expected_tokenizer)
    ):
        raise AxBenchBaseError(
            f"extraction_tokenizer_id={cfg.extraction_tokenizer_id!r} does not "
            f"match the gemma-scope-2 extraction model {extraction_model!r}, "
            f"whose tokenizer is {expected_tokenizer!r}. They MUST be the same "
            "tokenizer (the SAE was trained in that model's tokenization); set "
            f"extraction_tokenizer_id={expected_tokenizer!r}."
        )
    return extraction_model


def _validated_feature_dir(
    cfg: AxBenchBaseConfig,
    extraction_model: str,
) -> FeatureDirectoryComponents:
    # The SAE's model identity is encoded in feature_dir (the ``m_{model}``
    # segment); it MUST match the extraction model. A mismatch means the run
    # points at one model's SAE shards while configured for another -- e.g. a
    # gemma-3 SAE left at the gemma-2-2b defaults -- which would query the wrong
    # Neuronpedia source and mislabel every concept (issue #657). This catches
    # the all-defaults case (a v1/default model id absent from the tokenizer
    # map). feature_dir MUST be canonical: Stage 0a parses the layer, segment
    # size, and SAE model from it, so a path that does not parse is itself a
    # config error. Fail loud here rather than deferring the (later, less clear)
    # failure to Stage 0a. (An unset ``feature_dir=???`` never reaches this: the
    # mandatory-field check in ``cfg_to_dataclass`` rejects it first.)
    try:
        parsed_feature_dir = parse_feature_dir(cfg.feature_dir)
    except (ValueError, TypeError) as exc:
        # ValueError: a present-but-malformed path (wrong component count/shape).
        # TypeError: a non-str feature_dir (e.g. None) -- Path(None) raises it;
        # the field is typed str but not enforced at runtime, so surface a clean
        # config error rather than a raw TypeError traceback.
        raise AxBenchBaseError(
            f"feature_dir={cfg.feature_dir!r} is not a canonical feature path "
            "(s{seg}/l{layer}/a_{act}/p_{pool}/f{width}/m_{model}/{phase}). "
            "Stage 0a parses the layer, segment size, and SAE model from it, so "
            f"it must be a string following that convention. Parse error: {exc}"
        ) from exc
    sae_model = bare_model_id(parsed_feature_dir.model)
    if extraction_model != sae_model:
        raise AxBenchBaseError(
            f"Model mismatch: feature_dir is for SAE model {sae_model!r} "
            f"(from feature_dir={cfg.feature_dir!r}), but the configured "
            f"extraction model is {extraction_model!r} "
            "(feature_extraction_model_id or neuronpedia_model_id). These MUST "
            "match -- a gemma-3 SAE evaluated with the gemma-2-2b defaults would "
            "query the wrong Neuronpedia source and mislabel every concept "
            f"(issue #657). Set feature_extraction_model_id={sae_model!r} (and "
            "the matching neuronpedia_model_id / neuronpedia_source / "
            "extraction_tokenizer_id)."
        )
    return parsed_feature_dir


def _validate_neuronpedia_identity(
    cfg: AxBenchBaseConfig,
    parsed_feature_dir: FeatureDirectoryComponents,
) -> None:
    # neuronpedia_source AND neuronpedia_model_id together name the exact
    # Neuronpedia site queried at runtime. Both must agree with the SAE the run
    # actually evaluates (feature_dir). The most dangerous #657 misconfig is a
    # *same-model but wrong-layer/width/model* source: Neuronpedia answers valid
    # HTTP 200s for the wrong site, so NONE of the runtime rate guards (404 /
    # empty / failed) fire, and every concept is silently mislabeled with
    # another site's descriptions. Only this config-time cross-check catches it.
    #
    # neuronpedia_source encodes layer + activation + width (e.g.
    # ``22-gemmascope-2-res-65k`` -> layer 22, res, width 65k); these MUST match
    # feature_dir's ``l{layer}`` / ``a_{act}`` / ``f{width}``. The canonical
    # Neuronpedia model id for the SAE is then DERIVED from feature_dir's model
    # via the SAELens registry (``resolve_neuronpedia_ids``, the single source of
    # truth -- it encodes the #667 API/local split, e.g. a ``gemma-3-4b-pt`` SAE
    # -> API model ``gemma-3-4b``) and neuronpedia_model_id MUST equal it, so a
    # wrong-but-existing sibling namespace (``gemma-3-4b-it``) is rejected. An
    # unparseable / unpublished source is itself a misconfig and fails loud (the
    # pipeline is gemma-scope-based) rather than being skipped.
    try:
        source_release, source_activation, source_layer, source_width, source_l0 = (
            parse_neuronpedia_source(cfg.neuronpedia_source)
        )
    except (ValueError, AttributeError, TypeError) as exc:
        raise AxBenchBaseError(
            f"neuronpedia_source={cfg.neuronpedia_source!r} is not a recognized "
            "gemma-scope source ({layer}-gemmascope[-2]-{act}-{width}), so it "
            "cannot be checked against the SAE. Set a valid source matching "
            f"feature_dir (layer {parsed_feature_dir.layer}, "
            f"{parsed_feature_dir.activation_type.value}, width "
            f"{parsed_feature_dir.width.value}), e.g. "
            f"'{parsed_feature_dir.layer}-gemmascope-2-res-"
            f"{parsed_feature_dir.width.value}'. Parse error: {exc}"
        ) from exc
    if source_layer != parsed_feature_dir.layer:
        raise AxBenchBaseError(
            f"Layer mismatch: neuronpedia_source={cfg.neuronpedia_source!r} is "
            f"layer {source_layer}, but feature_dir is layer "
            f"{parsed_feature_dir.layer} (from feature_dir={cfg.feature_dir!r}). "
            "A wrong-layer source returns valid Neuronpedia descriptions for the "
            "WRONG layer, tripping no runtime guard and mislabeling every concept "
            "(issue #657). Set neuronpedia_source to the feature_dir's layer."
        )
    if source_width != parsed_feature_dir.width:
        raise AxBenchBaseError(
            f"Width mismatch: neuronpedia_source={cfg.neuronpedia_source!r} is "
            f"width {source_width.value}, but feature_dir is width "
            f"{parsed_feature_dir.width.value} (from "
            f"feature_dir={cfg.feature_dir!r}). A wrong-width source (e.g. a 16k "
            "source against a 65k SAE) mislabels every concept it can answer "
            "(issue #657). Set neuronpedia_source to the feature_dir's SAE width."
        )
    if source_activation != parsed_feature_dir.activation_type:
        raise AxBenchBaseError(
            f"Activation mismatch: neuronpedia_source={cfg.neuronpedia_source!r} "
            f"is activation {source_activation.value}, but feature_dir is "
            f"{parsed_feature_dir.activation_type.value} (from "
            f"feature_dir={cfg.feature_dir!r}). Set neuronpedia_source to the "
            "feature_dir's activation type (issue #657)."
        )
    sae_model = bare_model_id(parsed_feature_dir.model)
    # Derive the canonical (model_id, source) for THIS SAE from feature_dir's
    # ground truth (model/layer/width/activation) + the source's release/L0
    # tier, then require neuronpedia_model_id to equal the derived model id.
    try:
        canonical_model_id, _ = resolve_neuronpedia_ids(
            # Bare, like every sibling check: resolve_neuronpedia_ids ->
            # _resolve_gemmascope_2 is keyed by the bare model name, so an
            # org-prefixed feature_dir model would otherwise fail with a
            # misleading "unpublished" error.
            sae_model,
            source_release,
            parsed_feature_dir.activation_type,
            parsed_feature_dir.layer,
            parsed_feature_dir.width,
            source_l0,
        )
    except ValueError as exc:
        raise AxBenchBaseError(
            "Neuronpedia does not publish a source for the SAE in "
            f"feature_dir={cfg.feature_dir!r} (model {sae_model!r}, layer "
            f"{parsed_feature_dir.layer}, width "
            f"{parsed_feature_dir.width.value}) per the SAELens registry, so "
            f"neuronpedia_source={cfg.neuronpedia_source!r} cannot be correct. "
            f"Resolver error: {exc}"
        ) from exc
    if cfg.neuronpedia_model_id is None:
        raise AxBenchBaseError(
            "neuronpedia_model_id must be set: it is the Neuronpedia API model "
            "namespace queried in Stage 0a. Got null/None. Set it to the SAE's "
            f"Neuronpedia model, {canonical_model_id!r}."
        )
    if bare_model_id(cfg.neuronpedia_model_id) != bare_model_id(canonical_model_id):
        raise AxBenchBaseError(
            f"Neuronpedia model mismatch: neuronpedia_model_id="
            f"{cfg.neuronpedia_model_id!r}, but the SAE in "
            f"feature_dir={cfg.feature_dir!r} (model "
            f"{parsed_feature_dir.model!r}) is published on Neuronpedia under "
            f"model {canonical_model_id!r}. Querying the wrong model namespace "
            "returns valid HTTP 200s for a DIFFERENT model's features, tripping "
            "no runtime guard and mislabeling every concept (issue #657). Set "
            f"neuronpedia_model_id={canonical_model_id!r}."
        )


def _validate_seed_token_relationship(cfg: AxBenchBaseConfig) -> None:
    if cfg.seed_prefix_max_nanochat_tokens < cfg.seed_continuation_nanochat_tokens:
        raise AxBenchBaseError(
            f"seed_prefix_max_nanochat_tokens "
            f"({cfg.seed_prefix_max_nanochat_tokens}) must be >= "
            f"seed_continuation_nanochat_tokens "
            f"({cfg.seed_continuation_nanochat_tokens}) so every SeedRow "
            f"has at least as many prefix tokens as continuation tokens."
        )


def _validate_verification_api(cfg: AxBenchBaseConfig) -> None:
    if cfg.verification_api_model is None:
        return
    if (
        not isinstance(cfg.verification_api_model, str)
        or not cfg.verification_api_model.strip()
        or cfg.verification_api_model != cfg.verification_api_model.strip()
    ):
        raise AxBenchBaseError(
            "verification_api_model must be a non-empty, whitespace-"
            "trimmed OpenRouter model string or null, got "
            f"{cfg.verification_api_model!r}"
        )
    if cfg.synth.provider != "openrouter":
        raise AxBenchBaseError(
            "verification_api_model currently requires "
            "synth.provider='openrouter', got "
            f"{cfg.synth.provider!r}"
        )
    if cfg.synth.max_concurrent < 1:
        raise AxBenchBaseError(
            "verification_api_model requires synth.max_concurrent >= 1, "
            f"got {cfg.synth.max_concurrent}"
        )
    if cfg.synth.requests_per_minute < 0:
        raise AxBenchBaseError(
            "verification_api_model requires "
            "synth.requests_per_minute >= 0 (0 disables client-side rate "
            f"limiting), got {cfg.synth.requests_per_minute}"
        )
    if cfg.n_activation_examples < 1:
        raise AxBenchBaseError(
            "verification_api_model requires n_activation_examples >= 1 "
            "to derive a concept definition from Neuronpedia activation "
            f"evidence, got {cfg.n_activation_examples}"
        )
    if cfg.verification_api_prefix_wave_size < 1:
        raise AxBenchBaseError(
            "verification_api_prefix_wave_size must be >= 1 when "
            "verification_api_model is set, got "
            f"{cfg.verification_api_prefix_wave_size}"
        )


def _validate_distributed_device(cfg: AxBenchBaseConfig) -> None:
    # Reject explicit GPU index under multi-GPU launch.
    #
    # Heavy-model loaders read ``cfg.device`` directly. With
    # ``cfg.device="cuda:0"`` under torchrun, every rank would load onto
    # GPU 0 — silent multi-GPU breakage that wastes one GPU per rank
    # and bottlenecks on the shared device.
    #
    # ``compute_init`` calls ``torch.cuda.set_device(local_rank)`` so
    # every rank owns its own GPU. Callers MUST pass ``cfg.device="cuda"``
    # (no index) so that ``.to("cuda")`` resolves to the rank's
    # currently-set CUDA device.
    if os.environ.get("WORLD_SIZE", "1") != "1" and cfg.device.startswith("cuda:"):
        raise AxBenchBaseError(
            f"cfg.device={cfg.device!r} with explicit GPU index is "
            "incompatible with multi-GPU launch (WORLD_SIZE>1). Each "
            "rank must auto-select its own GPU via compute_init's "
            "torch.cuda.set_device. Set cfg.device='cuda' (no index) "
            "and let the launcher assign devices."
        )


def validate_config(cfg: AxBenchBaseConfig) -> None:
    """Fail fast on unacceptable config values. Raises AxBenchBaseError."""
    _validate_clustered_conditioning(cfg)
    _validate_positive_synthesis(cfg)
    extraction_model = _validated_extraction_model(cfg)
    parsed_feature_dir = _validated_feature_dir(cfg, extraction_model)
    _validate_neuronpedia_identity(cfg, parsed_feature_dir)
    _validate_seed_token_relationship(cfg)
    _validate_verification_api(cfg)
    _validate_distributed_device(cfg)
