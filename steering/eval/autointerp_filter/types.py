"""Dataclasses and exceptions for token-level intruder autointerp filtering."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Any, Protocol, runtime_checkable

from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation
from beartype.vale import Is

from errors import AutoInterpFilterError
from steering.refined_types import (
    NonEmptyNonNegativeIntTuple,
    NonEmptyNonNegativeIntTupleLike,
    NonEmptyStr,
    NonEmptyStrLike,
    NonNegativeFiniteNumber,
    NonNegativeFiniteNumberLike,
    NonNegativeFiniteNumberTuple,
    NonNegativeFiniteNumberTupleLike,
    NonNegativeInt,
    NonNegativeIntLike,
    NonNegativeIntTuple,
    NonNegativeIntTupleLike,
    PositiveInt,
    PositiveIntLike,
    SingletonNonNegativeIntTuple,
    SingletonNonNegativeIntTupleLike,
    StrTuple,
    StrTupleLike,
    UnitInterval,
    UnitIntervalLike,
)

SCHEMA_VERSION: int = 4


class TargetKind(StrEnum):
    feature = "feature"


class TargetOutcome(StrEnum):
    accepted = "accepted"
    rejected = "rejected"
    insufficient_examples = "insufficient_examples"


class AutoInterpConfigError(AutoInterpFilterError):
    """Config failed validation before scoring started."""


class AutoInterpDataError(AutoInterpFilterError):
    """Local feature shards, source text, or SAE activations are inconsistent."""


class AutoInterpCacheError(AutoInterpFilterError):
    """Persisted cache schema or data is invalid."""


class AutoInterpBatchValidationError(AutoInterpFilterError):
    """Fixed-block materialization failed a correctness or support gate."""


class AutoInterpCostCapExceeded(AutoInterpFilterError):
    """The configured maximum number of uncached judge calls was reached."""


class IntruderPromptError(AutoInterpFilterError):
    """A rendered intruder prompt is invalid or too long."""


class IntruderResponseParseError(AutoInterpFilterError):
    """A judge response did not contain a valid intruder index."""


@dataclass(frozen=True, init=False)
class SourceDocRef:
    shard_stem: NonEmptyStr
    shard_index: NonNegativeInt
    doc_id: NonNegativeInt
    segment_token_start: NonNegativeInt
    segment_token_end: NonNegativeInt

    @beartype
    def __init__(
        self,
        shard_stem: NonEmptyStrLike,
        shard_index: NonNegativeIntLike,
        doc_id: NonNegativeIntLike,
        segment_token_start: NonNegativeIntLike,
        segment_token_end: NonNegativeIntLike,
    ) -> None:
        object.__setattr__(self, "shard_stem", NonEmptyStr.parse(shard_stem))
        object.__setattr__(self, "shard_index", NonNegativeInt.parse(shard_index))
        object.__setattr__(self, "doc_id", NonNegativeInt.parse(doc_id))
        object.__setattr__(
            self,
            "segment_token_start",
            NonNegativeInt.parse(segment_token_start),
        )
        object.__setattr__(
            self,
            "segment_token_end",
            NonNegativeInt.parse(segment_token_end),
        )
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.segment_token_end <= self.segment_token_start:
            raise AutoInterpDataError(
                "source document segment_token_end must exceed segment_token_start: "
                f"{self.segment_token_start}:{self.segment_token_end}"
            )

    def stable_key(self) -> str:
        focus = f"tok{self.segment_token_start}-{self.segment_token_end}"
        return f"{self.shard_stem}:shard{self.shard_index}:doc{self.doc_id}:{focus}"


@dataclass(frozen=True, init=False)
class ScoringTarget:
    """A singleton feature target to score."""

    kind: TargetKind
    feature_ids: SingletonNonNegativeIntTuple
    target_key: NonEmptyStr
    display_name: NonEmptyStr

    @beartype
    def __init__(
        self,
        kind: TargetKind,
        feature_ids: SingletonNonNegativeIntTupleLike,
        target_key: NonEmptyStrLike,
        display_name: NonEmptyStrLike,
    ) -> None:
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self,
            "feature_ids",
            SingletonNonNegativeIntTuple.parse(feature_ids),
        )
        object.__setattr__(self, "target_key", NonEmptyStr.parse(target_key))
        object.__setattr__(self, "display_name", NonEmptyStr.parse(display_name))


@dataclass(frozen=True, init=False)
class TokenWindowRef:
    """Reference to one source-token window and its target activations."""

    shard_stem: NonEmptyStr
    shard_index: NonNegativeInt
    doc_id: NonNegativeInt
    source_text_hash: NonEmptyStr
    token_start: NonNegativeInt
    token_end: NonNegativeInt
    token_ids: NonNegativeIntTuple
    token_strings: StrTuple
    target_feature_ids: NonEmptyNonNegativeIntTuple
    activations: NonNegativeFiniteNumberTuple
    max_activation: NonNegativeFiniteNumber
    quantile: NonNegativeInt | None
    active: bool

    @beartype
    def __init__(
        self,
        shard_stem: NonEmptyStrLike,
        shard_index: NonNegativeIntLike,
        doc_id: NonNegativeIntLike,
        source_text_hash: NonEmptyStrLike,
        token_start: NonNegativeIntLike,
        token_end: NonNegativeIntLike,
        token_ids: NonNegativeIntTupleLike,
        token_strings: StrTupleLike,
        target_feature_ids: NonEmptyNonNegativeIntTupleLike,
        activations: NonNegativeFiniteNumberTupleLike,
        max_activation: NonNegativeFiniteNumberLike,
        quantile: NonNegativeIntLike | None,
        active: bool,
    ) -> None:
        object.__setattr__(self, "shard_stem", NonEmptyStr.parse(shard_stem))
        object.__setattr__(self, "shard_index", NonNegativeInt.parse(shard_index))
        object.__setattr__(self, "doc_id", NonNegativeInt.parse(doc_id))
        object.__setattr__(
            self,
            "source_text_hash",
            NonEmptyStr.parse(source_text_hash),
        )
        object.__setattr__(self, "token_start", NonNegativeInt.parse(token_start))
        object.__setattr__(self, "token_end", NonNegativeInt.parse(token_end))
        object.__setattr__(self, "token_ids", NonNegativeIntTuple.parse(token_ids))
        object.__setattr__(self, "token_strings", StrTuple.parse(token_strings))
        object.__setattr__(
            self,
            "target_feature_ids",
            NonEmptyNonNegativeIntTuple.parse(target_feature_ids),
        )
        object.__setattr__(
            self,
            "activations",
            NonNegativeFiniteNumberTuple.parse(activations),
        )
        object.__setattr__(
            self,
            "max_activation",
            NonNegativeFiniteNumber.parse(max_activation),
        )
        object.__setattr__(
            self,
            "quantile",
            None if quantile is None else NonNegativeInt.parse(quantile),
        )
        object.__setattr__(self, "active", active)
        self.__post_init__()

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.token_strings):
            raise AutoInterpDataError(
                "token_ids/token_strings length mismatch: "
                f"{len(self.token_ids)} vs {len(self.token_strings)}"
            )
        if len(self.token_ids) != len(self.activations):
            raise AutoInterpDataError(
                "token_ids/activations length mismatch: "
                f"{len(self.token_ids)} vs {len(self.activations)}"
            )
        if self.token_end < self.token_start:
            raise AutoInterpDataError(
                f"token_end {self.token_end} < token_start {self.token_start}"
            )
        if self.token_end - self.token_start != len(self.token_ids):
            raise AutoInterpDataError(
                "token window span length does not match token_ids length: "
                f"{self.token_start}-{self.token_end} vs {len(self.token_ids)} tokens"
            )

    def stable_key(self) -> str:
        payload = {
            "method": "token_window_v1",
            "shard": self.shard_stem,
            "shard_index": self.shard_index,
            "doc_id": self.doc_id,
            "source_text_hash": self.source_text_hash,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "token_ids": list(self.token_ids),
            "target_feature_ids": list(self.target_feature_ids),
            "activations": [float(v) for v in self.activations],
            "active": self.active,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def to_json(self) -> dict:
        return {
            "shard_stem": self.shard_stem,
            "shard_index": self.shard_index,
            "doc_id": self.doc_id,
            "source_text_hash": self.source_text_hash,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "token_ids": list(self.token_ids),
            "token_strings": list(self.token_strings),
            "target_feature_ids": list(self.target_feature_ids),
            "activations": list(self.activations),
            "max_activation": self.max_activation,
            "quantile": self.quantile,
            "active": self.active,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> TokenWindowRef:
        if not isinstance(payload, dict):
            raise AutoInterpCacheError("token window cache row is not a JSON object")
        required = {
            "shard_stem",
            "shard_index",
            "doc_id",
            "source_text_hash",
            "token_start",
            "token_end",
            "token_ids",
            "token_strings",
            "target_feature_ids",
            "activations",
            "max_activation",
            "quantile",
            "active",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise AutoInterpCacheError(f"token window cache row missing fields: {missing}")
        try:
            return cls(
                shard_stem=payload["shard_stem"],
                shard_index=payload["shard_index"],
                doc_id=payload["doc_id"],
                source_text_hash=payload["source_text_hash"],
                token_start=payload["token_start"],
                token_end=payload["token_end"],
                token_ids=tuple(payload["token_ids"]),
                token_strings=tuple(payload["token_strings"]),
                target_feature_ids=tuple(payload["target_feature_ids"]),
                activations=tuple(payload["activations"]),
                max_activation=payload["max_activation"],
                quantile=payload["quantile"],
                active=payload["active"],
            )
        except (
            AutoInterpDataError,
            BeartypeCallHintParamViolation,
            OverflowError,
            TypeError,
            ValueError,
        ) as exc:
            raise AutoInterpCacheError(
                f"token window cache row contains invalid data: {exc}"
            ) from exc


if TYPE_CHECKING:
    _TokenWindowRefTupleLike = tuple[TokenWindowRef, ...]
else:
    _TokenWindowRefTupleLike = Annotated[
        tuple,
        Is[
            lambda value: (
                isinstance(value, tuple) and all(isinstance(item, TokenWindowRef) for item in value)
            )
        ],
    ]


@dataclass(frozen=True, init=False)
class TokenExampleSet:
    """Cached token windows for one scoring target."""

    positives: tuple[TokenWindowRef, ...]
    non_activating: tuple[TokenWindowRef, ...]

    @beartype
    def __init__(
        self,
        positives: _TokenWindowRefTupleLike,
        non_activating: _TokenWindowRefTupleLike,
    ) -> None:
        object.__setattr__(self, "positives", positives)
        object.__setattr__(self, "non_activating", non_activating)

    def to_json(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "positives": [ref.to_json() for ref in self.positives],
            "non_activating": [ref.to_json() for ref in self.non_activating],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> TokenExampleSet:
        if not isinstance(payload, dict):
            raise AutoInterpCacheError("token example cache payload is not a JSON object")
        try:
            payload_schema_version = int(payload.get("schema_version", -1))
        except (OverflowError, TypeError, ValueError) as exc:
            raise AutoInterpCacheError(
                "token example cache payload has invalid schema_version="
                f"{payload.get('schema_version')!r}"
            ) from exc
        if payload_schema_version != SCHEMA_VERSION:
            raise AutoInterpCacheError(
                "token example cache payload schema_version="
                f"{payload.get('schema_version')!r} != {SCHEMA_VERSION}"
            )
        positives = payload.get("positives", [])
        non_activating = payload.get("non_activating", [])
        if not isinstance(positives, list) or not isinstance(non_activating, list):
            raise AutoInterpCacheError(
                "token example cache payload positives and non_activating must be lists"
            )
        try:
            return cls(
                positives=tuple(TokenWindowRef.from_json(row) for row in positives),
                non_activating=tuple(TokenWindowRef.from_json(row) for row in non_activating),
            )
        except BeartypeCallHintParamViolation as exc:
            raise AutoInterpCacheError(
                f"token example cache payload contains invalid data: {exc}"
            ) from exc


@dataclass(frozen=True, init=False)
class RenderedExample:
    """Prompt-ready example text plus token-level provenance."""

    text: NonEmptyStr
    ref: TokenWindowRef
    activating: bool
    highlighted_token_indices: NonNegativeIntTuple

    @beartype
    def __init__(
        self,
        text: NonEmptyStrLike,
        ref: TokenWindowRef,
        activating: bool,
        highlighted_token_indices: NonNegativeIntTupleLike,
    ) -> None:
        object.__setattr__(self, "text", NonEmptyStr.parse(text))
        object.__setattr__(self, "ref", ref)
        object.__setattr__(self, "activating", activating)
        object.__setattr__(
            self,
            "highlighted_token_indices",
            NonNegativeIntTuple.parse(highlighted_token_indices),
        )


if TYPE_CHECKING:
    _RenderedExampleTupleLike = tuple[RenderedExample, ...]
else:
    _RenderedExampleTupleLike = Annotated[
        tuple,
        Is[
            lambda value: (
                isinstance(value, tuple)
                and all(isinstance(item, RenderedExample) for item in value)
            )
        ],
    ]


@dataclass(frozen=True, init=False)
class IntruderTrial:
    """One token-level intruder-detection trial."""

    target: ScoringTarget
    trial_index: NonNegativeInt
    positive_refs: tuple[TokenWindowRef, ...]
    intruder_ref: TokenWindowRef
    intruder_index: NonNegativeInt
    quantile: NonNegativeInt | None
    intruder_highlight_indices: NonNegativeIntTuple
    activation_threshold_ratio: UnitInterval
    trial_key: NonEmptyStr

    @beartype
    def __init__(
        self,
        target: ScoringTarget,
        trial_index: NonNegativeIntLike,
        positive_refs: _TokenWindowRefTupleLike,
        intruder_ref: TokenWindowRef,
        intruder_index: NonNegativeIntLike,
        quantile: NonNegativeIntLike | None,
        intruder_highlight_indices: NonNegativeIntTupleLike,
        activation_threshold_ratio: UnitIntervalLike,
        trial_key: NonEmptyStrLike,
    ) -> None:
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "trial_index", NonNegativeInt.parse(trial_index))
        object.__setattr__(self, "positive_refs", positive_refs)
        object.__setattr__(self, "intruder_ref", intruder_ref)
        object.__setattr__(
            self,
            "intruder_index",
            NonNegativeInt.parse(intruder_index),
        )
        object.__setattr__(
            self,
            "quantile",
            None if quantile is None else NonNegativeInt.parse(quantile),
        )
        object.__setattr__(
            self,
            "intruder_highlight_indices",
            NonNegativeIntTuple.parse(intruder_highlight_indices),
        )
        object.__setattr__(
            self,
            "activation_threshold_ratio",
            UnitInterval.parse(activation_threshold_ratio),
        )
        object.__setattr__(self, "trial_key", NonEmptyStr.parse(trial_key))


@dataclass(frozen=True, init=False)
class IntruderTrialResult:
    """Parsed judge result for one trial."""

    trial_key: NonEmptyStr
    prediction: NonNegativeInt | None
    correct_index: NonNegativeInt
    correct: bool
    response: str | None
    cached: bool
    error_message: str | None = None
    rendered_examples: tuple[RenderedExample, ...] = ()
    final_user_prompt: str | None = None

    @beartype
    def __init__(
        self,
        trial_key: NonEmptyStrLike,
        prediction: NonNegativeIntLike | None,
        correct_index: NonNegativeIntLike,
        correct: bool,
        response: str | None,
        cached: bool,
        error_message: str | None = None,
        rendered_examples: _RenderedExampleTupleLike = (),
        final_user_prompt: str | None = None,
    ) -> None:
        object.__setattr__(self, "trial_key", NonEmptyStr.parse(trial_key))
        object.__setattr__(
            self,
            "prediction",
            None if prediction is None else NonNegativeInt.parse(prediction),
        )
        object.__setattr__(
            self,
            "correct_index",
            NonNegativeInt.parse(correct_index),
        )
        object.__setattr__(self, "correct", correct)
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "cached", cached)
        object.__setattr__(self, "error_message", error_message)
        object.__setattr__(self, "rendered_examples", rendered_examples)
        object.__setattr__(self, "final_user_prompt", final_user_prompt)


def _is_descriptions(value: object) -> bool:
    return isinstance(value, dict) and all(
        isinstance(feature_id, NonNegativeInt)
        and (description is None or isinstance(description, str))
        for feature_id, description in value.items()
    )


if TYPE_CHECKING:
    _DescriptionsLike = dict[int, str | None]
else:
    _DescriptionsLike = Annotated[dict, Is[_is_descriptions]]


@dataclass(frozen=True, init=False)
class TargetScore:
    """Per-target persisted score row."""

    target_key: NonEmptyStr
    display_name: NonEmptyStr
    kind: TargetKind
    feature_ids: NonEmptyNonNegativeIntTuple
    score: UnitInterval | None
    n_correct: NonNegativeInt
    n_trials: NonNegativeInt
    outcome: TargetOutcome
    descriptions: dict[int, str | None]
    error_message: str | None = None
    schema_version: PositiveInt = PositiveInt.parse(SCHEMA_VERSION)

    @beartype
    def __init__(
        self,
        target_key: NonEmptyStrLike,
        display_name: NonEmptyStrLike,
        kind: TargetKind,
        feature_ids: NonEmptyNonNegativeIntTupleLike,
        score: UnitIntervalLike | None,
        n_correct: NonNegativeIntLike,
        n_trials: NonNegativeIntLike,
        outcome: TargetOutcome,
        descriptions: _DescriptionsLike | None = None,
        error_message: str | None = None,
        schema_version: PositiveIntLike = SCHEMA_VERSION,
    ) -> None:
        object.__setattr__(self, "target_key", NonEmptyStr.parse(target_key))
        object.__setattr__(self, "display_name", NonEmptyStr.parse(display_name))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self,
            "feature_ids",
            NonEmptyNonNegativeIntTuple.parse(feature_ids),
        )
        object.__setattr__(
            self,
            "score",
            None if score is None else UnitInterval.parse(score),
        )
        object.__setattr__(self, "n_correct", NonNegativeInt.parse(n_correct))
        object.__setattr__(self, "n_trials", NonNegativeInt.parse(n_trials))
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(
            self,
            "descriptions",
            {} if descriptions is None else descriptions,
        )
        object.__setattr__(self, "error_message", error_message)
        object.__setattr__(
            self,
            "schema_version",
            PositiveInt.parse(schema_version),
        )
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.n_correct > self.n_trials:
            raise AutoInterpDataError(
                f"target score n_correct {self.n_correct} exceeds n_trials {self.n_trials}"
            )


@dataclass(frozen=True, init=False)
class AutoInterpFilterReport:
    """Aggregate run report."""

    n_targets_total: NonNegativeInt
    n_targets_scored: NonNegativeInt
    n_targets_accepted: NonNegativeInt
    n_features_accepted: NonNegativeInt
    score_threshold: UnitInterval | None
    top_k: PositiveInt | None
    target_scores_path: NonEmptyStr
    ranked_yaml_path: NonEmptyStr
    accepted_features_path: NonEmptyStr
    manual_feature_filter_path: NonEmptyStr
    manual_feature_filter_meta_path: NonEmptyStr
    trial_audit_path: NonEmptyStr
    trial_audit_yaml_path: NonEmptyStr
    schema_version: PositiveInt = PositiveInt.parse(SCHEMA_VERSION)

    @beartype
    def __init__(
        self,
        n_targets_total: NonNegativeIntLike,
        n_targets_scored: NonNegativeIntLike,
        n_targets_accepted: NonNegativeIntLike,
        n_features_accepted: NonNegativeIntLike,
        score_threshold: UnitIntervalLike | None,
        top_k: PositiveIntLike | None,
        target_scores_path: NonEmptyStrLike,
        ranked_yaml_path: NonEmptyStrLike,
        accepted_features_path: NonEmptyStrLike,
        manual_feature_filter_path: NonEmptyStrLike,
        manual_feature_filter_meta_path: NonEmptyStrLike,
        trial_audit_path: NonEmptyStrLike,
        trial_audit_yaml_path: NonEmptyStrLike,
        schema_version: PositiveIntLike = SCHEMA_VERSION,
    ) -> None:
        object.__setattr__(
            self,
            "n_targets_total",
            NonNegativeInt.parse(n_targets_total),
        )
        object.__setattr__(
            self,
            "n_targets_scored",
            NonNegativeInt.parse(n_targets_scored),
        )
        object.__setattr__(
            self,
            "n_targets_accepted",
            NonNegativeInt.parse(n_targets_accepted),
        )
        object.__setattr__(
            self,
            "n_features_accepted",
            NonNegativeInt.parse(n_features_accepted),
        )
        object.__setattr__(
            self,
            "score_threshold",
            None if score_threshold is None else UnitInterval.parse(score_threshold),
        )
        object.__setattr__(
            self,
            "top_k",
            None if top_k is None else PositiveInt.parse(top_k),
        )
        object.__setattr__(
            self,
            "target_scores_path",
            NonEmptyStr.parse(target_scores_path),
        )
        object.__setattr__(
            self,
            "ranked_yaml_path",
            NonEmptyStr.parse(ranked_yaml_path),
        )
        object.__setattr__(
            self,
            "accepted_features_path",
            NonEmptyStr.parse(accepted_features_path),
        )
        object.__setattr__(
            self,
            "manual_feature_filter_path",
            NonEmptyStr.parse(manual_feature_filter_path),
        )
        object.__setattr__(
            self,
            "manual_feature_filter_meta_path",
            NonEmptyStr.parse(manual_feature_filter_meta_path),
        )
        object.__setattr__(
            self,
            "trial_audit_path",
            NonEmptyStr.parse(trial_audit_path),
        )
        object.__setattr__(
            self,
            "trial_audit_yaml_path",
            NonEmptyStr.parse(trial_audit_yaml_path),
        )
        object.__setattr__(
            self,
            "schema_version",
            PositiveInt.parse(schema_version),
        )
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.n_targets_scored > self.n_targets_total:
            raise AutoInterpDataError("autointerp report n_targets_scored exceeds n_targets_total")
        if self.n_targets_accepted > self.n_targets_scored:
            raise AutoInterpDataError(
                "autointerp report n_targets_accepted exceeds n_targets_scored"
            )


class TokenExampleProvider(Protocol):
    """Provide cached or freshly built token windows for scoring targets."""

    def prepare(self, targets: list[ScoringTarget]) -> None: ...

    def examples_for_target(self, target: ScoringTarget) -> TokenExampleSet: ...

    def cache_context(self) -> dict: ...


@runtime_checkable
class ExampleRenderer(Protocol):
    """Render token-window refs into prompt examples."""

    def render(
        self,
        ref: TokenWindowRef,
        *,
        activating: bool,
        highlighted_token_indices: NonEmptyNonNegativeIntTupleLike,
    ) -> RenderedExample: ...
