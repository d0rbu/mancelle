"""Shared steered-intervention builder for AxBench-BPB-Adv.

A single source of truth for the steered (+ optional additive counter-direction)
intervention, so every consumer -- the dual-completion scorer
(``scoring.ModelBackedDualCompletionScorer``), the general-validation stage
(``general_val``), and the forthcoming completion-generation feature -- provably
builds the EXACT same ``Intervention`` the eval scores.

The additive counter-steering composes a SPECIFIED concept counter-direction as a
residual hook ON TOP of the full baseline steering (the baseline is applied
separately, at its own coefficient). ``additive_counter_coefficient`` gives the
signed counter coefficient; ``_load_concept_direction`` loads the concept's
counter-direction for the baseline's own model; ``apply_additive_counter`` glues
the two onto an already-built base intervention; ``build_steered_intervention``
is the one-call convenience (``build_intervention`` + ``apply_additive_counter``)
for callers that have no separate base-only path.

This module imports only from ``axbench_bpb.intervention`` / ``axbench_bpb.types``
/ ``axbench_bpb_adv.config`` (and never from ``scoring`` / ``general_val``) so it
can be imported by both without a cycle.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import torch as t

from steering.eval.axbench_base.direction_artifacts import (
    DirectionArtifactRequest,
    load_direction_artifact,
)
from steering.eval.axbench_base.direction_scale import (
    normalize_direction,
    require_direction_max_activation,
)
from steering.eval.axbench_base.method_registry import BaselineKind
from steering.eval.axbench_base.types import Concept
from steering.eval.axbench_bpb.intervention import build_intervention
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.types import Intervention, SteeringMode
from steering.eval.axbench_bpb_adv.config import AdversarialMethod
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError

if TYPE_CHECKING:
    from steering.eval.axbench_bpb_adv.source_validation import (
        BpbAdvSourceValidation,
    )
    from steering.eval.cluster_feature_contract import ClusterFeatureContract


@dataclass(frozen=True)
class AdditiveCounterDirection:
    """Resolved additive counter-direction payload reusable across interventions."""

    direction: t.Tensor
    layer: int
    coefficient: float

    def __post_init__(self) -> None:
        if self.direction.numel() == 0:
            raise AxBenchBpbAdvError("resolved additive counter direction is empty")
        if not bool(t.isfinite(self.direction).all().item()):
            raise AxBenchBpbAdvError("resolved additive counter direction is non-finite")
        if float(self.coefficient) == 0.0:
            raise AxBenchBpbAdvError(
                "resolved additive counter direction has zero coefficient; use "
                "AdditiveCounterNoOp for explicit no-op cases"
            )


class AdditiveCounterNoOpReason(StrEnum):
    NOT_ADDITIVE_DIRECTION = "not_additive_direction"
    ZERO_COEFFICIENT = "zero_coefficient"


@dataclass(frozen=True)
class AdditiveCounterNoOp:
    """Explicit additive-counter no-op resolution."""

    reason: AdditiveCounterNoOpReason


AdditiveCounterResolution = AdditiveCounterDirection | AdditiveCounterNoOp


def additive_counter_coefficient(
    *,
    mode: SteeringMode | str,
    magnitude: float,
) -> float:
    """The standalone counter-direction coefficient for additive counter-steering.

    additive composes a SPECIFIED concept counter-direction as a residual hook on
    top of the FULL baseline steering (the baseline is applied separately, at its
    own coefficient). The counter pushes along the concept axis OPPOSING the
    STEERING (degrading it): positive mode steers TOWARD the concept, so the
    counter pushes away (``-magnitude``); negative mode steers AWAY, so the counter
    pushes back toward the concept (``+magnitude``). ``magnitude == 0`` is the
    pure-baseline control (no counter).

    This function returns the raw ``±magnitude`` sign rule;
    ``apply_additive_counter`` then scales it by the counter direction artifact's
    own ``max_activation``. When a diffmean baseline is countered with that same
    diffmean artifact, the net shared-axis magnitude is
    ``(|base| - magnitude) * max_activation``. With different direction families,
    each residual axis is calibrated by its own artifact value. With scaling off,
    the legacy ``|base| - magnitude`` algebra is unchanged.
    """

    if SteeringMode(mode) == SteeringMode.POSITIVE:
        return -float(magnitude)
    return float(magnitude)


def _load_concept_direction(
    *,
    family: str,
    mode: str,
    concept: Concept,
    source_paths: SourcePaths,
    expected_layer: int,
    target: str = "decoder_only",
    source_validation: BpbAdvSourceValidation | None = None,
) -> tuple[t.Tensor, int, dict[str, str]]:
    """Load the concept's residual direction of ``family`` for additive's counter.

    Returns ``(direction, expected_layer, metadata)``. ``family`` must be a residual-direction
    family (``diffmean`` / ``linear_probe``); directions are stored per (target,
    mode, concept) and are validated against ``expected_layer`` when the file
    records layer metadata.
    """

    if family == "diffmean":
        path = source_paths.diffmean_direction(target, mode, concept.concept_id)
    elif family == "linear_probe":
        path = source_paths.linear_probe_direction(target, mode, concept.concept_id)
    else:
        raise AxBenchBpbAdvError(
            "additive counter_direction_family supports only diffmean and "
            f"linear_probe, got {family!r}"
        )
    request = DirectionArtifactRequest(
        family=family,
        target_model=target,
        mode=mode,
        concept_id=concept.concept_id,
        layer=expected_layer,
    )
    if source_validation is None:
        loaded = load_direction_artifact(
            path,
            request=request,
            error_type=AxBenchBpbAdvError,
        )
    else:
        payload = source_validation.direction_artifact_bytes(
            family=family,
            target=target,
            mode=mode,
            concept_id=concept.concept_id,
        )
        loaded = load_direction_artifact(
            payload,
            artifact_path=path,
            request=request,
            error_type=AxBenchBpbAdvError,
        )
    return loaded.direction, int(expected_layer), loaded.metadata_dict()


def apply_additive_counter(
    base_intervention: Intervention,
    *,
    adversarial_method: str,
    adversarial_strength: float,
    method: Any,
    concept: Concept,
    source_paths: SourcePaths,
    diffmean_layer: int,
    target: str,
    counter_direction_family: str,
    apply_max_act_scale: bool,
    source_validation: BpbAdvSourceValidation | None = None,
) -> Intervention:
    """Attach the additive counter-direction add-on to ``base_intervention``.

    The SINGLE source of truth for the additive counter logic shared by the dual
    scorer, the general-val stage, and completion generation. Non-additive rows
    and the zero-coefficient (magnitude-0 control) case return ``base_intervention``
    UNCHANGED -- byte-identical to the plain baseline. Otherwise the SPECIFIED
    concept counter-direction (``counter_direction_family``) is loaded for the
    BASELINE's own model (``target``) -- so it lives in the same residual basis as
    the model the hook is installed on (the encdec direction for an encdec
    baseline, NOT the decoder-only one) -- and composed via
    ``Intervention.with_counter_direction``.
    """

    counter = resolve_additive_counter(
        adversarial_method=adversarial_method,
        adversarial_strength=adversarial_strength,
        method=method,
        concept=concept,
        source_paths=source_paths,
        diffmean_layer=diffmean_layer,
        target=target,
        counter_direction_family=counter_direction_family,
        apply_max_act_scale=apply_max_act_scale,
        source_validation=source_validation,
    )
    return attach_additive_counter(base_intervention, counter)


def resolve_additive_counter(
    *,
    adversarial_method: str,
    adversarial_strength: float,
    method: Any,
    concept: Concept,
    source_paths: SourcePaths,
    diffmean_layer: int,
    target: str,
    counter_direction_family: str,
    apply_max_act_scale: bool,
    source_validation: BpbAdvSourceValidation | None = None,
) -> AdditiveCounterResolution:
    """Resolve the additive counter-direction once for a combo.

    Non-additive rows and magnitude-0 controls return an explicit
    ``AdditiveCounterNoOp`` so callers can preserve byte-identical plain
    interventions without treating bare ``None`` as meaningful state. A
    direction result is independent of the base intervention payload and can be
    attached to many per-document roofline bags without re-opening the same
    safetensors direction file.
    """

    if adversarial_method != AdversarialMethod.ADDITIVE_DIRECTION.value:
        return AdditiveCounterNoOp(AdditiveCounterNoOpReason.NOT_ADDITIVE_DIRECTION)
    counter_coefficient = additive_counter_coefficient(
        mode=method.mode,
        magnitude=float(adversarial_strength),
    )
    # magnitude 0 is the pure-baseline control: skip the counter hook entirely
    # so the control intervention is byte-identical to the plain baseline.
    if counter_coefficient == 0.0:
        return AdditiveCounterNoOp(AdditiveCounterNoOpReason.ZERO_COEFFICIENT)
    # The == 0 control check above runs BEFORE any artifact load or scaling, so
    # the magnitude-0 control stays byte-identical.
    # additive counter-steering: push the SPECIFIED concept counter-direction,
    # opposing the concept, as a residual hook ON TOP of the full baseline
    # steering. Works for ANY baseline (diffmean/probe/lora/ours). The counter
    # direction is loaded for the BASELINE's own model (``target``) so it lives in
    # the same residual basis as the model the hook is installed on.
    counter_direction, counter_layer, counter_metadata = _load_concept_direction(
        family=counter_direction_family,
        mode=method.mode.value,
        concept=concept,
        source_paths=source_paths,
        expected_layer=diffmean_layer,
        target=target,
        source_validation=source_validation,
    )
    if apply_max_act_scale:
        if not counter_metadata:
            raise AxBenchBpbAdvError(
                "additive counter direction metadata absent where scaled binding "
                "is required; re-run stage 3a/3b under a fresh run tag"
            )
        counter_direction = normalize_direction(
            counter_direction, error_cls=AxBenchBpbAdvError
        )
        counter_scale = require_direction_max_activation(
            counter_metadata,
            direction=counter_direction,
            expected_concept_id=concept.concept_id,
            expected_mode=method.mode.value,
            expected_target_model=target,
            expected_layer=int(diffmean_layer),
            surface=(
                f"additive counter {counter_direction_family!r} direction "
                f"concept={concept.concept_id}"
            ),
            error_cls=AxBenchBpbAdvError,
        )
        # Zero-direction provenance, decided on the VALIDATED VALUE (not the
        # multiplied product): require_... established value==0.0 iff the
        # direction is all-zero, and the pre-#1228 zero-vector hook was a
        # no-op — preserve that as an explicit no-op resolution.
        if counter_scale == 0.0:
            return AdditiveCounterNoOp(AdditiveCounterNoOpReason.ZERO_COEFFICIENT)
        counter_coefficient *= counter_scale
        # Nonzero magnitude × positive scale can still underflow binary64 to
        # exactly 0.0 (denormal magnitude × subnormal scale). That is NOT the
        # zero-direction pair; a silent no-op would disguise a degenerate
        # configuration as the magnitude-0 control (PR #1237 final-gate
        # review). Fail loud instead.
        if counter_coefficient == 0.0:
            raise AxBenchBpbAdvError(
                "additive counter coefficient underflowed to 0.0 from nonzero "
                f"magnitude and positive max_activation ({counter_scale!r}) "
                f"for concept={concept.concept_id}; this degenerate "
                "configuration is indistinguishable from the magnitude-0 "
                "control. Inspect the adversarial strength grid and the "
                "direction artifact."
            )
    return AdditiveCounterDirection(
        direction=counter_direction,
        layer=counter_layer,
        coefficient=float(counter_coefficient),
    )


def attach_additive_counter(
    base_intervention: Intervention,
    counter: AdditiveCounterResolution,
) -> Intervention:
    """Attach a pre-resolved additive counter payload to ``base_intervention``."""

    if not isinstance(base_intervention, Intervention):
        raise AxBenchBpbAdvError(
            "attach_additive_counter requires an Intervention base, got "
            f"{type(base_intervention).__name__}"
        )
    if (
        base_intervention.counter_direction is not None
        or base_intervention.counter_direction_layer is not None
        or base_intervention.counter_direction_coefficient is not None
    ):
        raise AxBenchBpbAdvError(
            "base intervention already carries an additive counter-direction; "
            "refusing to attach or no-op over ambiguous counter state"
        )
    if isinstance(counter, AdditiveCounterNoOp):
        return base_intervention
    if not isinstance(counter, AdditiveCounterDirection):
        raise AxBenchBpbAdvError(
            "attach_additive_counter requires AdditiveCounterDirection or "
            "AdditiveCounterNoOp; bare None is not a valid counter resolution"
        )
    return base_intervention.with_counter_direction(
        direction=counter.direction,
        layer=counter.layer,
        coefficient=counter.coefficient,
    )


def build_steered_intervention(
    *,
    method: Any,
    concept: Concept,
    coefficient: int | float | None,
    source_paths: SourcePaths,
    diffmean_layer: int,
    target_model_meta: Any,
    inactive_sampler: Any | None,
    tokenizer: Any,
    rng: random.Random,
    adversarial_method: str,
    adversarial_strength: float,
    counter_direction_family: str,
    target: str,
    apply_max_act_scale: bool,
    model: Any | None = None,
    adv_paths: Any | None = None,
    core_prefix_id: str | None = None,
    cluster_contract: ClusterFeatureContract | None = None,
    source_validation: BpbAdvSourceValidation | None = None,
) -> Intervention:
    """Build the steered (+ optional additive counter) intervention in one call.

    The convenience entry point for callers (e.g. completion generation) that need
    base+counter together and have no separate base-only path. Equivalent to
    ``apply_additive_counter(build_intervention(...), ...)``: it builds the
    baseline steered intervention via ``build_intervention`` (seeded by the
    caller's ``rng``) then attaches the additive counter via
    ``apply_additive_counter``. ``target`` is the baseline's own model (the
    residual basis the counter is loaded for); it is normally
    ``method.target_model``.

    ``model`` is forwarded to ``build_intervention`` unchanged and is REQUIRED
    for every ``baseline == "ours"`` path so the loaded checkpoint remains the
    authoritative feature vocabulary. It remains optional for non-``ours``
    methods. ``adv_paths``/``core_prefix_id`` identify persisted fair/roofline
    bags, while ``cluster_contract`` enforces representative-only clustered
    conditioning at the shared intervention boundary.
    """

    direction_payload = None
    if (
        source_validation is not None
        and method.baseline in (BaselineKind.DIFFMEAN, BaselineKind.LINEAR_PROBE)
    ):
        direction_payload = source_validation.direction_artifact_bytes(
            family=method.baseline.value,
            target=target,
            mode=method.mode.value,
            concept_id=concept.concept_id,
        )
    base = build_intervention(
        method=method,
        concept=concept,
        coefficient=coefficient,
        source_paths=source_paths,
        diffmean_layer=diffmean_layer,
        target_model_meta=target_model_meta,
        inactive_sampler=inactive_sampler,
        tokenizer_nanochat=tokenizer,
        rng=rng,
        apply_max_act_scale=apply_max_act_scale,
        model=model,
        adv_paths=adv_paths,
        core_prefix_id=core_prefix_id,
        cluster_contract=cluster_contract,
        direction_payload=direction_payload,
    )
    return apply_additive_counter(
        base,
        adversarial_method=adversarial_method,
        adversarial_strength=adversarial_strength,
        method=method,
        concept=concept,
        source_paths=source_paths,
        diffmean_layer=diffmean_layer,
        target=target,
        counter_direction_family=counter_direction_family,
        apply_max_act_scale=apply_max_act_scale,
        source_validation=source_validation,
    )


__all__ = [
    "AdditiveCounterDirection",
    "AdditiveCounterNoOp",
    "AdditiveCounterNoOpReason",
    "AdditiveCounterResolution",
    "_load_concept_direction",
    "additive_counter_coefficient",
    "apply_additive_counter",
    "attach_additive_counter",
    "build_steered_intervention",
    "resolve_additive_counter",
]
