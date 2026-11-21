# steering/eval/axbench_bpb/types.py
"""Core types for axbench-bpb. More types added across later phases."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Literal

from steering.eval.axbench_base.types import SteeringMode  # re-export

if TYPE_CHECKING:
    from pathlib import Path

    import torch

    from steering.eval.axbench_base.config import LoraConfig
    from steering.eval.target_conditioning import ContextFeatureSegment


class AxBenchBpbError(Exception):
    """Single user-visible exception for the axbench-bpb package."""


class Stage(StrEnum):
    PREPARE_CONTEXT_CACHE = "prepare_context_cache"
    BPB_SCORE = "bpb_score"
    BPB_AGGREGATE = "bpb_aggregate"
    ALL = "all"

    @classmethod
    def coerce(cls, value: str) -> Stage:
        try:
            return cls(value)
        except ValueError as exc:
            raise AxBenchBpbError(
                f"Unknown stage {value!r}; valid: {', '.join(m.value for m in cls)}"
            ) from exc


class LeaderboardMetric(StrEnum):
    """Macro-level metric used to sort the leaderboard CSV.

    Values must match column names produced by ``compute_macro_rollup`` in
    ``bpb_aggregation.py``; if a new macro column is added there, it should
    also be added here.
    """

    MACRO_MEAN_BEST_PRIMARY_MARGIN = "macro_mean_best_primary_margin"
    MACRO_MEAN_BEST_MARGIN_INTERVENED = "macro_mean_best_margin_intervened"
    MACRO_MEAN_DOSE_AUC_MARGIN = "macro_mean_dose_auc_margin"
    MACRO_MEAN_TRUNCATION_RATE = "macro_mean_truncation_rate"

    @classmethod
    def coerce(cls, value: str | LeaderboardMetric) -> LeaderboardMetric:
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as exc:
            raise AxBenchBpbError(
                f"Unknown primary_metric {value!r}; valid: "
                f"{', '.join(m.value for m in cls)}"
            ) from exc


class InterventionKind(StrEnum):
    """Intervention family for axbench-bpb teacher-force scoring."""

    NONE = "none"
    OURS = "ours"
    DIFFMEAN = "diffmean"
    LINEAR_PROBE = "linear_probe"
    LORA = "lora"
    PROMPTING = "prompting"

    @classmethod
    def coerce(cls, value: str | InterventionKind) -> InterventionKind:
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(
                f"Unknown intervention kind {value!r}; valid: "
                f"{', '.join(kind.value for kind in cls)}"
            ) from exc


class TargetModel(StrEnum):
    """Model family scored by the AxBench BPB pipeline."""

    DECODER_ONLY = "decoder_only"
    ENCDEC = "encdec"

    @classmethod
    def coerce(cls, value: str | TargetModel) -> TargetModel:
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(
                f"Unknown target model {value!r}; valid: "
                f"{', '.join(target.value for target in cls)}"
            ) from exc


@dataclass(frozen=True)
class ScoringInputs:
    sample_id: str
    concept_id: int
    mode: SteeringMode
    content_class: str                  # "concept_rich" | "concept_avoiding"
    prefix_text: str
    prefix_hash: str
    prefix_tokens: list[int]
    output_tokens: list[int]
    prefix_context_segments: tuple[ContextFeatureSegment, ...]
    n_prefix: int
    n_output: int
    output_byte_weights: torch.Tensor
    truncated: bool


@dataclass(frozen=True)
class WorkUnit:
    method_name: str
    concept_id: int
    sample_id: str
    coefficient_str: str


@dataclass(frozen=True)
class BpbScoreRow:
    method_name: str
    concept_id: int
    mode: str                            # "positive" | "negative"
    content_class: str
    target: str                          # "decoder_only" | "encdec"
    coefficient_str: str
    sample_id: str
    bpb_intervened: float
    n_output_tokens: int
    n_output_bytes: int
    truncated: bool


@dataclass(frozen=True)
class LoraParams:
    """The co-traveling LoRA attach parameters for a ``lora``-kind run.

    Bundles the two values that ``apply_intervention`` needs to attach a
    per-concept LoRA adapter onto the target model — the training-time
    hyperparameters (``cfg``) and the residual-stream layer to attach at
    (``layer``). These two are all-or-nothing: an attach needs both or
    neither (the other intervention kinds need neither, so they pass
    ``None``). Bundling them into one frozen value enshrines that
    invariant by construction — you cannot build a ``LoraParams`` with
    only one of the pair — so callers thread a single ``LoraParams |
    None`` rather than a loose ``(lora_cfg, lora_layer)`` pair guarded by
    a both-or-neither runtime check.

    The adapter file path itself lives on the ``Intervention`` (spec §10
    carries it there); ``LoraParams`` carries only what is NOT stored on
    the Intervention.
    """

    cfg: LoraConfig
    layer: int


@dataclass(frozen=True)
class Intervention:
    """Intervention specification for axbench-bpb teacher-force scoring.

    Exactly one intervention family is active per instance, indicated by
    ``kind``. Fields outside the active family are ``None``. Use the named
    factory classmethods (``none``, ``ours``, ``from_direction``, ``lora``,
    ``prompting``) for construction rather than calling the dataclass
    constructor directly — factories enforce the per-family invariants.

    Consumed by ``apply_intervention`` (see ``intervention.py``).
    """

    # Mutable spec-mandated fields (dict/list/Tensor) preclude consistent
    # hashing; disable __hash__ explicitly rather than auto-generating one
    # that silently blows up for populated-field instances.
    __hash__: ClassVar[None] = None

    kind: InterventionKind | str
    target_model: TargetModel | str
    feature_conditions: dict[int, int] | None = None
    # Excluded from __eq__: tensor equality returns a Tensor (not a bool),
    # breaking dataclass-generated comparison. Callers should compare
    # directions via `torch.equal(a.direction, b.direction)` explicitly.
    direction: torch.Tensor | None = field(default=None, compare=False)
    direction_layer: int | None = None
    direction_coefficient: float | None = None
    lora_adapter_path: Path | None = None
    prompt_prefix_tokens: list[int] | None = None
    # Orthogonal ADDITIVE counter-direction add-on, independent of ``kind``.
    # When ``counter_direction`` is set, ``apply_intervention`` installs a
    # residual hook (``counter_direction_coefficient * counter_direction`` at
    # ``counter_direction_layer``) IN ADDITION to whatever ``kind`` does — for
    # both decoder-only and encoder-decoder targets. Used by additive
    # counter-steering to push along a specified concept direction on top of the
    # full baseline steering. (Excluded from __eq__: tensor equality is a Tensor.)
    counter_direction: torch.Tensor | None = field(default=None, compare=False)
    counter_direction_layer: int | None = None
    counter_direction_coefficient: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", InterventionKind.coerce(self.kind))
        object.__setattr__(self, "target_model", TargetModel.coerce(self.target_model))

    def with_counter_direction(
        self, *, direction: torch.Tensor, layer: int, coefficient: float,
    ) -> Intervention:
        """Return a copy carrying an additive counter-direction add-on.

        Orthogonal to ``kind``: the base intervention (direction hook / LoRA
        adapter / encdec conditioning) is preserved; ``apply_intervention``
        additionally installs ``coefficient * direction`` as a residual hook at
        ``layer``. ``coefficient == 0`` is a no-op (the control / magnitude-0
        case).
        """

        return replace(
            self,
            counter_direction=direction,
            counter_direction_layer=int(layer),
            counter_direction_coefficient=float(coefficient),
        )

    @classmethod
    def none(cls) -> Intervention:
        """No intervention on the decoder-only model."""

        return cls(
            kind=InterventionKind.NONE,
            target_model=TargetModel.DECODER_ONLY,
        )

    @classmethod
    def ours(cls, *, feature_conditions: dict[int, int]) -> Intervention:
        return cls(
            kind=InterventionKind.OURS,
            target_model=TargetModel.ENCDEC,
            feature_conditions=dict(feature_conditions),
        )

    @classmethod
    def from_direction(
        cls, *, kind: Literal["diffmean", "linear_probe"] | InterventionKind,
        target_model: TargetModel | str,
        direction: torch.Tensor, direction_layer: int, coefficient: float,
    ) -> Intervention:
        coerced = InterventionKind.coerce(kind)
        if coerced not in (InterventionKind.DIFFMEAN, InterventionKind.LINEAR_PROBE):
            raise ValueError(
                "direction interventions must use kind='diffmean' or "
                f"kind='linear_probe', got {coerced.value!r}"
            )
        return cls(
            kind=coerced, target_model=target_model,
            direction=direction, direction_layer=direction_layer,
            direction_coefficient=coefficient,
        )

    @classmethod
    def lora(cls, *, adapter_path: Path) -> Intervention:
        return cls(
            kind=InterventionKind.LORA,
            target_model=TargetModel.DECODER_ONLY,
            lora_adapter_path=adapter_path,
        )

    @classmethod
    def prompting(
        cls, *, target_model: TargetModel | str,
        prompt_prefix_tokens: list[int],
    ) -> Intervention:
        return cls(
            kind=InterventionKind.PROMPTING, target_model=target_model,
            prompt_prefix_tokens=list(prompt_prefix_tokens),
        )


__all__ = [
    "AxBenchBpbError",
    "BpbScoreRow",
    "Intervention",
    "InterventionKind",
    "LeaderboardMetric",
    "LoraParams",
    "ScoringInputs",
    "Stage",
    "SteeringMode",
    "TargetModel",
    "WorkUnit",
]
