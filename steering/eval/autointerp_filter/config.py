"""Config dataclasses and validation for token-level autointerp filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from beartype import beartype

from steering.eval.autointerp_filter.path_safety import paths_alias
from steering.eval.autointerp_filter.prompts import (
    DEFAULT_INTRUDER_FEW_SHOT_SET,
    DEFAULT_INTRUDER_PROMPT_VARIANT,
    resolve_intruder_prompt_spec,
)
from steering.eval.autointerp_filter.types import AutoInterpConfigError
from steering.refined_types import (
    NonEmptyStrLike,
    NonNegativeIntLike,
    PositiveFiniteNumber,
    PositiveInt,
    PositiveIntLike,
    UniqueNonEmptyNonNegativeIntListLike,
    UnitInterval,
)


@beartype
@dataclass(frozen=True)
class IntruderJudgeConfig:
    model: NonEmptyStrLike = "openai/gpt-5-nano"
    requests_per_minute: NonNegativeIntLike = 60
    max_concurrent: PositiveIntLike = 8
    max_prompt_chars: PositiveIntLike = 32_000
    parse_max_retries: NonNegativeIntLike = 2
    cot: bool = False
    prompt_variant: str = DEFAULT_INTRUDER_PROMPT_VARIANT
    few_shot_set: str = DEFAULT_INTRUDER_FEW_SHOT_SET

    def validate(self) -> None:
        """Validate judge-local transport and prompt settings."""
        resolve_intruder_prompt_spec(
            prompt_variant=self.prompt_variant,
            few_shot_set=self.few_shot_set,
        )


@beartype
@dataclass(frozen=True)
class TokenExamplesConfig:
    device: str = "cpu"
    model: str | None = None
    sae_release: NonEmptyStrLike = "gemmascope"
    sae_l0: int | str | None = None
    activation_backend: str = "auto"
    model_dtype: str = "bfloat16"
    context_window: PositiveInt | None = None
    example_ctx_len: PositiveIntLike = 32
    activation_threshold_ratio: UnitInterval = UnitInterval.parse(0.3)
    max_positive_source_docs_per_target: PositiveIntLike = 128
    max_intruder_source_docs: PositiveIntLike = 5000
    max_windows_per_source_doc: PositiveIntLike = 8
    max_positive_windows_per_target: PositiveIntLike = 512
    max_non_activating_windows_per_target: PositiveIntLike = 512
    materialization_batch_size: PositiveIntLike = 128

    def validate(self) -> None:
        """Validate token-example model, window, and materialization settings."""
        self._validate_model_settings()

    def _validate_model_settings(self) -> None:
        if self.activation_backend not in {
            "auto",
            "native",
            "transformerlens",
        }:
            raise AutoInterpConfigError(
                "token_examples.activation_backend must be one of auto, native, transformerlens"
            )
        if self.model_dtype not in {"float32", "bfloat16"}:
            raise AutoInterpConfigError("token_examples.model_dtype must be float32 or bfloat16")


@beartype
@dataclass(frozen=True)
class SourceIndexConfig:
    enabled: bool = False
    path: str = "${output_dir}/source_doc_index.sqlite"
    manifest_path: str = "${output_dir}/source_doc_index.manifest.json"
    shared_cache_dir: str | None = None
    force_rebuild: bool = False
    chunk_size: PositiveIntLike = 5_000_000
    # Scheduling knob: number of worker processes for the per-shard scan.
    # 1 = serial (stage-2 path). EXCLUDED from source-index cache identity — one
    # index is reusable across any worker count; it cannot change any output byte.
    scan_workers: PositiveIntLike = 1

    def validate(self) -> None:
        """Validate source-index-local sizing and path identity."""
        if not self.enabled:
            return
        try:
            source_paths_alias = paths_alias(
                Path(self.path),
                Path(self.manifest_path),
            )
        except OSError as exc:
            raise AutoInterpConfigError(
                f"cannot verify source_index.path and source_index.manifest_path identity: {exc}"
            ) from exc
        if source_paths_alias:
            raise AutoInterpConfigError(
                "source_index.path and source_index.manifest_path must identify distinct files"
            )


@beartype
@dataclass(frozen=True)
class NeuronpediaDescriptionConfig:
    enabled: bool = False
    model_id: str | None = None
    sae_source: str | None = None
    timeout_seconds: PositiveFiniteNumber = PositiveFiniteNumber.parse(30.0)
    max_retries: NonNegativeIntLike = 3
    backoff_base_seconds: PositiveFiniteNumber = PositiveFiniteNumber.parse(1.0)
    backoff_max_seconds: PositiveFiniteNumber = PositiveFiniteNumber.parse(30.0)

    def validate(self) -> None:
        """Validate Neuronpedia-local retry and identity settings."""
        if self.backoff_max_seconds < self.backoff_base_seconds:
            raise AutoInterpConfigError(
                "neuronpedia.backoff_max_seconds must be >= neuronpedia.backoff_base_seconds"
            )
        if not self.enabled:
            return
        if self.model_id is None:
            raise AutoInterpConfigError(
                "neuronpedia.model_id is required when neuronpedia.enabled=true"
            )
        if self.sae_source is None:
            raise AutoInterpConfigError(
                "neuronpedia.sae_source is required when neuronpedia.enabled=true"
            )


@beartype
@dataclass(frozen=True)
class AutoInterpFilterConfig:
    feature_dir: str
    output_dir: str
    data_dir: str | None = None
    n_sae_features: PositiveInt | None = None
    feature_ids: UniqueNonEmptyNonNegativeIntListLike | None = None
    feature_mask: dict[str, Any] | None = None
    document_blacklist_path: str | None = None
    n_trials_per_target: PositiveIntLike = 100
    min_trials_for_score: PositiveIntLike = 10
    n_positive_examples: PositiveIntLike = 3
    n_quantiles: PositiveIntLike = 5
    seed: int = 42
    max_targets: PositiveInt | None = None
    max_judge_calls: PositiveInt | None = None
    score_threshold: UnitInterval | None = UnitInterval.parse(0.6)
    top_k: PositiveInt | None = None
    fail_on_empty: bool = True
    audit_sample_limit: NonNegativeIntLike = 20
    token_example_cache_path: str = "${output_dir}/token_example_cache.sqlite"
    judge_cache_path: str = "${output_dir}/intruder_trial_cache.sqlite"
    target_scores_path: str = "${output_dir}/target_scores.jsonl"
    ranked_yaml_path: str = "${output_dir}/autointerp_ranked.yaml"
    accepted_features_path: str = "${output_dir}/accepted_feature_ids.txt"
    manual_feature_filter_path: str = "${output_dir}/manual_feature_filter.yaml"
    manual_feature_filter_meta_path: str = "${output_dir}/manual_feature_filter.meta.json"
    report_output_path: str = "${output_dir}/report.json"
    trial_audit_path: str = "${output_dir}/trial_audit.jsonl"
    trial_audit_yaml_path: str = "${output_dir}/trial_audit.yaml"
    runner_manifest_path: str = "${output_dir}/runner_manifest.json"
    lockfile_path: str = "${output_dir}/.lockfile"
    judge: IntruderJudgeConfig = field(default_factory=IntruderJudgeConfig)
    token_examples: TokenExamplesConfig = field(default_factory=TokenExamplesConfig)
    source_index: SourceIndexConfig = field(default_factory=SourceIndexConfig)
    neuronpedia: NeuronpediaDescriptionConfig = field(default_factory=NeuronpediaDescriptionConfig)


def _validate_optional_path_token(value: str, field_name: str) -> None:
    if "${" in value:
        raise AutoInterpConfigError(
            f"{field_name} contains unresolved interpolation token: {value!r}"
        )


@beartype
def _validate_target_selection(
    feature_dir: str,
    feature_ids: UniqueNonEmptyNonNegativeIntListLike | None,
    n_sae_features: PositiveIntLike | None,
) -> None:
    if not feature_dir:
        raise AutoInterpConfigError("feature_dir is required")
    for fid in feature_ids or []:
        if n_sae_features is not None and fid >= n_sae_features:
            raise AutoInterpConfigError(
                f"feature_id {fid} out of range [0, {n_sae_features})"
            )


def _validate_path_tokens(cfg: AutoInterpFilterConfig) -> None:
    for field_name in (
        "output_dir",
        "token_example_cache_path",
        "judge_cache_path",
        "target_scores_path",
        "ranked_yaml_path",
        "accepted_features_path",
        "manual_feature_filter_path",
        "manual_feature_filter_meta_path",
        "report_output_path",
        "trial_audit_path",
        "trial_audit_yaml_path",
        "runner_manifest_path",
        "lockfile_path",
    ):
        _validate_optional_path_token(
            str(getattr(cfg, field_name)),
            field_name,
        )
    for field_name in ("path", "manifest_path", "shared_cache_dir"):
        value = getattr(cfg.source_index, field_name)
        if value is not None:
            _validate_optional_path_token(
                str(value),
                f"source_index.{field_name}",
            )
    if cfg.document_blacklist_path is not None:
        _validate_optional_path_token(
            cfg.document_blacklist_path,
            "document_blacklist_path",
        )


def _ensure_output_directory(cfg: AutoInterpFilterConfig) -> None:
    out = Path(cfg.output_dir)
    if out.exists() and not out.is_dir():
        raise AutoInterpConfigError(f"output_dir exists but is not a directory: {cfg.output_dir}")
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AutoInterpConfigError(f"output_dir {cfg.output_dir} unwritable: {exc}") from exc


def validate_intruder_judge_config(cfg: IntruderJudgeConfig) -> None:
    cfg.validate()


def validate_config(cfg: AutoInterpFilterConfig) -> None:
    """Fail fast on invalid config before any API call."""
    _validate_target_selection(cfg.feature_dir, cfg.feature_ids, cfg.n_sae_features)
    validate_intruder_judge_config(cfg.judge)
    cfg.token_examples.validate()
    cfg.source_index.validate()
    cfg.neuronpedia.validate()
    _validate_path_tokens(cfg)
    _ensure_output_directory(cfg)
