"""Hydra config dataclasses + validators."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from beartype import beartype

from steering.eval.feature_filter.types import ConfigValidationError
from steering.refined_types import (
    FivePointScoreLike,
    NonBlankPathComponentLike,
    NonNegativeFiniteNumberLike,
    NonNegativeIntLike,
    OneToFiveIntLike,
    PositiveFiniteNumberLike,
    PositiveIntLike,
    UniqueNonEmptyNonNegativeIntListLike,
    UnitIntervalLike,
)

__all__ = [
    "FeatureFilterConfig",
    "JudgeConfig",
    "NeuronpediaInlineConfig",
    "validate_config",
]


@beartype
@dataclass(frozen=True)
class NeuronpediaInlineConfig:
    timeout_seconds: PositiveFiniteNumberLike = 30.0
    max_retries: NonNegativeIntLike = 5
    backoff_base_seconds: PositiveFiniteNumberLike = 1.0
    backoff_max_seconds: PositiveFiniteNumberLike = 60.0
    max_concurrent: PositiveIntLike = 14

    def validate(self) -> None:
        """Validate the relationship between the retry delays."""
        if self.backoff_max_seconds < self.backoff_base_seconds:
            raise ConfigValidationError(
                "neuronpedia.backoff_max_seconds must be >= neuronpedia.backoff_base_seconds"
            )


@beartype
@dataclass(frozen=True)
class JudgeConfig:
    model: str = "openai/gpt-5-nano"
    requests_per_minute: int = 60
    max_prompt_chars: int = 16000
    max_examples_in_prompt: int = 7
    example_window_chars: int = 240
    parse_max_retries: NonNegativeIntLike = 3


@beartype
@dataclass(frozen=True)
class FeatureFilterConfig:
    model_id: NonBlankPathComponentLike
    sae_source: NonBlankPathComponentLike
    tokenizer_id: str
    output_dir: str
    feature_ids: UniqueNonEmptyNonNegativeIntListLike | None = None
    n_sae_features: PositiveIntLike = 16384
    samples_per_bin: OneToFiveIntLike = 3
    interp_min_score: FivePointScoreLike = 1
    consistency_min_score: FivePointScoreLike = 3
    max_concurrent_fetches: PositiveIntLike = 14
    max_concurrent_judges: PositiveIntLike = 8
    max_judge_calls: PositiveIntLike | None = None
    error_abort_fraction: UnitIntervalLike = 0.5
    shutdown_grace_seconds: NonNegativeFiniteNumberLike = 30.0
    hist_sum_tolerance: PositiveFiniteNumberLike = 1e-3
    neuronpedia_cache_dir: str = "${output_dir}/np_cache"
    judge_cache_path: str = "${output_dir}/judge_cache.sqlite"
    features_output_path: str = "${output_dir}/feature_scores.jsonl"
    accepted_features_path: str = "${output_dir}/accepted_feature_ids.txt"
    report_output_path: str = "${output_dir}/filter_report.json"
    runner_manifest_path: str = "${output_dir}/runner_manifest.json"
    lockfile_path: str = "${output_dir}/.lockfile"
    neuronpedia: NeuronpediaInlineConfig = field(default_factory=NeuronpediaInlineConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    def validate(self) -> None:
        """Validate the complete filter config in stable fail-fast order."""
        _validate_feature_selection(self)
        self.neuronpedia.validate()
        _validate_paths(self)
        _ensure_output_directory(self)


def _validate_feature_selection(cfg: FeatureFilterConfig) -> None:
    if cfg.feature_ids is None:
        return
    for fid in cfg.feature_ids:
        if fid >= cfg.n_sae_features:
            raise ConfigValidationError(f"feature_id {fid} out of range [0, {cfg.n_sae_features})")


def _validate_paths(cfg: FeatureFilterConfig) -> None:
    path_fields = (
        "output_dir",
        "neuronpedia_cache_dir",
        "judge_cache_path",
        "features_output_path",
        "accepted_features_path",
        "report_output_path",
        "runner_manifest_path",
        "lockfile_path",
    )
    for field_name in path_fields:
        value = getattr(cfg, field_name)
        if "${" in value:
            raise ConfigValidationError(
                f"{field_name} contains unresolved interpolation token: "
                f"{value!r}. This likely means the config was instantiated "
                "without Hydra/OmegaConf."
            )


def _ensure_output_directory(cfg: FeatureFilterConfig) -> None:
    out = Path(cfg.output_dir)
    if out.exists() and not out.is_dir():
        raise ConfigValidationError(f"output_dir exists but is not a directory: {cfg.output_dir}")
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigValidationError(f"output_dir {cfg.output_dir} unwritable: {exc}") from exc


def validate_config(cfg: FeatureFilterConfig) -> None:
    cfg.validate()
