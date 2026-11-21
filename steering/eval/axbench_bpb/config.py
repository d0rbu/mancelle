"""AxBenchBpbConfig + validation. validate_bpb_config runs at the top of
every CLI invocation before any heavy resource load."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from beartype import beartype

from steering.context_defaults import DEFAULT_CONTEXT_WINDOW
from steering.eval.axbench_base.direction_artifacts import (
    DirectionArtifactRequest,
    direction_artifact_path,
    load_direction_artifact,
)
from steering.eval.axbench_base.method_registry import (
    ALL_METHODS,
    BaselineKind,
    filter_methods,
)
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.eval.axbench_bpb.source_paths import (
    SourcePaths,
    _load_final_source_concept_ids,
    validate_stage_0_3_artifacts,
)
from steering.eval.axbench_bpb.types import (
    AxBenchBpbError,
    LeaderboardMetric,
    Stage,
)
from steering.refined_types import (
    AtLeastTwoIntLike,
    NonBlankStrLike,
    NonNegativeIntLike,
    PositiveIntLike,
    UnitIntervalLike,
)


@beartype
@dataclass
class ContextCacheConfig:
    extraction_batch_size: PositiveIntLike = 32
    force_rebuild: bool = False
    segment_size: PositiveIntLike = 256
    # 2 buckets matches the d12_s32 / d12_baseline encoder.magnitude_embed
    # shape (2, 384) — the canonical conditional-encdec discretization for
    # this codebase. Override on the CLI when scoring a model trained with
    # a different bucket count.
    num_buckets: AtLeastTwoIntLike = 2


@beartype
@dataclass
class AggregationConfig:
    min_rows_per_side: PositiveIntLike = 10
    invalid_warning_threshold: UnitIntervalLike = 0.01
    invalid_drop_threshold: UnitIntervalLike = 0.05
    primary_metric: (
        LeaderboardMetric
        | Literal[
            "macro_mean_best_primary_margin",
            "macro_mean_best_margin_intervened",
            "macro_mean_dose_auc_margin",
            "macro_mean_truncation_rate",
        ]
    ) = LeaderboardMetric.MACRO_MEAN_BEST_PRIMARY_MARGIN


@beartype
@dataclass
class FeatureDiffMeanConfig:
    top_k_pos: PositiveIntLike | None = None
    top_k_neg: PositiveIntLike | None = None


@beartype
@dataclass
class AxBenchBpbConfig:
    # --- Anchoring ---
    source_run_dir: str = ""  # REQUIRED (validated)
    output_root: str = ""  # REQUIRED (validated)
    run_tag: str | None = None

    # --- Feature directory ---
    feature_dir: str = ""  # REQUIRED
    feature_statistics_path: str = ""  # REQUIRED
    extraction_tokenizer_id: str = "google/gemma-2-2b"
    neuronpedia_model_id: str = "gemma-2-2b"
    # Optional local model id for loading the extraction model + SAE.  Some
    # Neuronpedia namespaces do not equal their HuggingFace/SAELens model id
    # (for example gemma-3-4b vs gemma-3-4b-pt).  None preserves the legacy
    # neuronpedia_model_id-owned loading path exactly.
    feature_extraction_model_id: NonBlankStrLike | None = None
    neuronpedia_source: str = "8-gemmascope-res-16k"
    extraction_max_tokens: int = DEFAULT_CONTEXT_WINDOW
    extraction_threshold: float = 0.0

    # --- Target models ---
    decoder_only: dict[str, Any] = field(default_factory=dict)
    encdec: dict[str, Any] = field(default_factory=dict)
    device: str = "cuda"
    next_segment_conditioning_offset: bool = False

    # --- Direction-hook layer (used by diffmean / linear_probe interventions) ---
    diffmean_and_probe_layer: NonNegativeIntLike = 8

    # --- Stage / method filters ---
    stage: (
        Stage
        | Literal[
            "prepare_context_cache",
            "bpb_score",
            "bpb_aggregate",
            "all",
        ]
    ) = Stage.ALL.value
    enabled_methods: list[str] | None = None
    disabled_methods: list[str] = field(default_factory=list)

    # --- Sample selection ---
    filter_unaccepted_samples: bool = True

    # --- Batching ---
    device_batch_size: PositiveIntLike = 16

    # --- Context cache ---
    context_cache: ContextCacheConfig = field(default_factory=ContextCacheConfig)

    # --- Aggregation ---
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    feature_diffmean: FeatureDiffMeanConfig = field(default_factory=FeatureDiffMeanConfig)

    # --- LoRA hyperparameters (used by lora-kind interventions) ---
    lora: dict[str, Any] = field(default_factory=dict)

    # --- Progress / logging ---
    log_progress_every: int = 1000

    # --- Offload ---
    offload_between_stages: bool = True


def _validate_direction_requirements(
    cfg: AxBenchBpbConfig,
    source_paths: SourcePaths,
) -> None:
    """Resolve enabled scalar methods and validate their artifacts up front."""

    known_names = {method.name for method in ALL_METHODS}
    unknown_enabled = sorted(set(cfg.enabled_methods or ()) - known_names)
    if unknown_enabled:
        raise AxBenchBpbError(
            f"unknown enabled method(s): {unknown_enabled}. "
            f"Available method names: {sorted(known_names)}"
        )
    try:
        methods = filter_methods(
            list(ALL_METHODS),
            enabled_methods=cfg.enabled_methods,
            disabled_methods=cfg.disabled_methods,
            # Only ours methods depend on this flag; scalar direction methods do
            # not, so True resolves the exact direction requirement set without
            # constructing the encoder-decoder model first.
            target_model_has_negative_sampling=True,
        )
    except AxBenchBaseError as exc:
        raise AxBenchBpbError(str(exc)) from exc
    concept_ids = _load_final_source_concept_ids(source_paths.concepts_json())
    for method in methods:
        if method.baseline not in (
            BaselineKind.DIFFMEAN,
            BaselineKind.LINEAR_PROBE,
        ):
            continue
        for concept_id in concept_ids:
            request = DirectionArtifactRequest(
                family=method.baseline.value,
                target_model=method.target_model,
                mode=method.mode.value,
                concept_id=concept_id,
                layer=cfg.diffmean_and_probe_layer,
            )
            load_direction_artifact(
                direction_artifact_path(source_paths.baselines_root(), request),
                request=request,
                error_type=AxBenchBpbError,
            )


def validate_bpb_config(cfg: AxBenchBpbConfig) -> None:
    """Raise AxBenchBpbError on any invariant violation. Runs before any
    heavy resource load.

    SIDE EFFECT (P0.G, slice 17 audit 2026-05-05): this function MUTATES
    ``cfg.aggregation.primary_metric`` in place, coercing the Hydra-supplied
    plain string into the ``LeaderboardMetric`` enum so downstream code
    sees the typed value. The mutation is intentional — re-binding is the
    cleanest way to share a single coerced enum across stages without
    threading the coerced value through every call site — but operators
    extending validation must be aware of it. ``cfg.aggregation`` is an
    ``AggregationConfig`` dataclass instance after ``_cfg_to_dataclass``,
    NOT an OmegaConf node, so the mutation does not violate Hydra's
    read-only discipline.
    """
    if not cfg.source_run_dir:
        raise AxBenchBpbError("source_run_dir is required")
    if not cfg.output_root:
        raise AxBenchBpbError("output_root is required")
    if not cfg.feature_dir:
        raise AxBenchBpbError("feature_dir is required")
    if not cfg.feature_statistics_path:
        raise AxBenchBpbError("feature_statistics_path is required")

    # Stage must be valid
    Stage.coerce(cfg.stage)  # raises AxBenchBpbError with 'Unknown stage'

    # primary_metric must be a known LeaderboardMetric. Hydra passes plain
    # strings; coerce in-place so downstream code sees the enum.
    cfg.aggregation.primary_metric = LeaderboardMetric.coerce(cfg.aggregation.primary_metric)

    if cfg.next_segment_conditioning_offset:
        raise AxBenchBpbError(
            "axbench-bpb next_segment_conditioning_offset=true is not wired into "
            "teacher-forced encoder routing; leave it false until that path is "
            "implemented and tested."
        )
    if not (
        0.0
        <= cfg.aggregation.invalid_warning_threshold
        <= cfg.aggregation.invalid_drop_threshold
        <= 1.0
    ):
        raise AxBenchBpbError(
            f"Invalid threshold ordering: "
            f"0 <= warning={cfg.aggregation.invalid_warning_threshold} "
            f"<= drop={cfg.aggregation.invalid_drop_threshold} <= 1 required"
        )

    # Validate source run dir contains Stage 0-3 artifacts
    source_paths = SourcePaths(source_run_dir=Path(cfg.source_run_dir))
    validate_stage_0_3_artifacts(
        source_paths,
        filter_unaccepted_samples=cfg.filter_unaccepted_samples,
    )
    _validate_direction_requirements(cfg, source_paths)
