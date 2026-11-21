"""Dataclasses + exception hierarchy for the feature_filter package."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from errors import FeatureFilterError as _FeatureFilterError

__all__ = [
    # exceptions
    "ConfigValidationError",
    "ConcurrentRunnerError",
    "CostCapExceeded",
    "InvalidHistogramError",
    "NoNonzeroActivationsError",
    "PromptTooLongError",
    "ResumeMismatchError",
    "SchemaVersionError",
    # dataclasses
    "ActivatingExample",
    "BinSample",
    "BinSampleRecord",
    "FeatureFilterReport",
    "FeatureInterpScore",
    "OutcomeType",
    "SCHEMA_VERSION",
]


class ConfigValidationError(_FeatureFilterError):
    """Config failed validation at load time (before any API call)."""


class ConcurrentRunnerError(_FeatureFilterError):
    """Another runner already holds the output_dir lockfile."""


class CostCapExceeded(_FeatureFilterError):
    """`max_judge_calls` budget exhausted; run halted before next OpenRouter call."""


class InvalidHistogramError(_FeatureFilterError):
    """Neuronpedia payload has malformed histogram or bin fields."""


class NoNonzeroActivationsError(_FeatureFilterError):
    """Feature has no activating examples in any magnitude bin (all bins empty or sentinel-only)."""


class PromptTooLongError(_FeatureFilterError):
    """Constructed judge prompt exceeds `max_prompt_chars`."""


class ResumeMismatchError(_FeatureFilterError):
    """Existing runner_manifest.json disagrees with current config on (model, source, schema)."""


class SchemaVersionError(_FeatureFilterError):
    """Persisted artifact has an unsupported `schema_version`."""


SCHEMA_VERSION: int = 3

OutcomeType = Literal[
    "accepted",
    "rejected",
    "fetch_error",
    "judge_error",
    "no_nonzero_activations",
    "prompt_error",
    "not_published",
]


@dataclass(frozen=True)
class ActivatingExample:
    """A single activating example for an SAE feature."""

    tokens: list[str]
    text: str
    max_value: float
    max_value_token_index: int
    peak_char_offset: int

    def example_text(self, window_chars: int = 120) -> str:
        """Return a snippet of self.text windowed around the peak token."""
        if window_chars <= 0 or len(self.text) <= 2 * window_chars:
            return self.text
        start = max(0, self.peak_char_offset - window_chars)
        end = min(len(self.text), self.peak_char_offset + window_chars)
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(self.text) else ""
        return prefix + self.text[start:end] + suffix


@dataclass(frozen=True)
class BinSample:
    """In-memory: one activating example tagged with its magnitude band."""

    bin_index: int
    band_label: str
    is_absent: bool
    max_value: float
    example: ActivatingExample

    def __post_init__(self) -> None:
        if self.is_absent and self.max_value != 0.0:
            raise ValueError(
                f"BinSample.is_absent=True requires max_value==0.0, "
                f"got {self.max_value}"
            )
        if self.is_absent and self.bin_index != 0:
            raise ValueError(
                f"BinSample.is_absent=True requires bin_index==0, "
                f"got {self.bin_index}"
            )
        label_says_absent = self.band_label.startswith("band 1/5 (absent")
        if self.is_absent != label_says_absent:
            raise ValueError(
                f"BinSample.is_absent={self.is_absent} inconsistent with "
                f"band_label={self.band_label!r}"
            )


@dataclass(frozen=True)
class BinSampleRecord:
    """Persisted JSONL form of a BinSample."""

    bin_index: int
    band_label: str
    is_absent: bool
    max_value: float
    snippet: str


@dataclass(frozen=True)
class FeatureInterpScore:
    """Per-feature output row. None scores are never coerced to 0."""

    feature_id: int
    description: str | None
    outcome: OutcomeType
    interpretability: int | None
    consistency: int | None
    samples: list[BinSampleRecord]
    error_message: str | None
    schema_version: int


@dataclass(frozen=True)
class FeatureFilterReport:
    """Aggregate run report."""

    model_id: str
    sae_source: str
    samples_per_bin: int
    thresholds: dict[str, int]
    n_total: int
    n_accepted: int
    n_rejected: int
    n_fetch_errors: int
    n_judge_errors: int
    n_no_nonzero: int
    n_prompt_errors: int
    n_not_published: int
    interpretability_histogram: dict[int, int]
    consistency_histogram: dict[int, int]
    per_feature_path: str
    accepted_feature_ids_path: str
    schema_version: int
