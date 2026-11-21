"""Resolve exact BPB-Adv direction coordinates before model construction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from steering.eval.axbench_base.direction_artifacts import DirectionArtifactRequest
from steering.eval.axbench_base.method_registry import ALL_METHODS, BaselineKind, Method
from steering.eval.axbench_bpb_adv.config import (
    AdversarialMethod,
    AxBenchBpbAdvConfig,
    get_additive_direction_config,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError


def _selected_methods(cfg: AxBenchBpbAdvConfig) -> list[Method]:
    by_name = {method.name: method for method in ALL_METHODS}
    target_models = set(cfg.target_models)
    selected_names = (
        {
            name
            for name, method in by_name.items()
            if method.target_model in target_models
        }
        if cfg.base_methods is None
        else set(cfg.base_methods)
    )
    selected_names.update(cfg.representative_target_methods)
    unknown = sorted(selected_names - set(by_name))
    if unknown:
        raise AxBenchBpbAdvError(f"unknown BPB-Adv base method(s): {unknown}")
    wrong_target = sorted(
        name for name in selected_names if by_name[name].target_model not in target_models
    )
    if wrong_target:
        raise AxBenchBpbAdvError(
            "selected BPB-Adv method target is absent from target_models: "
            f"{wrong_target}"
        )
    return [method for method in ALL_METHODS if method.name in selected_names]


def requirements_from_config(
    cfg: AxBenchBpbAdvConfig,
    *,
    concept_ids: Iterable[int],
) -> list[DirectionArtifactRequest]:
    """Directions selected by base methods plus non-zero additive attacks."""

    concepts = sorted(set(concept_ids))
    methods = _selected_methods(cfg)
    # #1249 follow-up: an excluded (concept, mode) emits NO attack rows (see
    # prepare_bpb_attack_rows), so the preflight must not demand its direction
    # coordinates either -- the requirement would reference rows that can never
    # exist. Both loops below honour the same exclusion the row emitter does.
    excluded = {
        (int(entry.split(":")[0]), entry.split(":")[1])
        for entry in cfg.exclude_concept_modes
    }
    requirements: set[DirectionArtifactRequest] = set()
    for method in methods:
        if method.baseline in {BaselineKind.DIFFMEAN, BaselineKind.LINEAR_PROBE}:
            for concept_id in concepts:
                if (concept_id, method.mode.value) in excluded:
                    continue
                requirements.add(
                    DirectionArtifactRequest(
                        family=method.baseline.value,
                        target_model=method.target_model,
                        mode=method.mode.value,
                        concept_id=concept_id,
                        layer=cfg.diffmean_and_probe_layer,
                    )
                )

    additive = get_additive_direction_config(cfg)
    if additive is not None and any(float(value) != 0.0 for value in additive.magnitudes):
        supported_families = set(additive.families)
        # Representative-target methods receive only their explicit zero-strength
        # control row, so they never consume a counter direction.
        representative_names = set(cfg.representative_target_methods)
        for method in methods:
            if method.name in representative_names:
                continue
            if not (
                method.baseline is BaselineKind.CONTROL
                or method.baseline.value in supported_families
            ):
                continue
            for concept_id in concepts:
                if (concept_id, method.mode.value) in excluded:
                    continue
                requirements.add(
                    DirectionArtifactRequest(
                        family=additive.counter_direction_family,
                        target_model=method.target_model,
                        mode=method.mode.value,
                        concept_id=concept_id,
                        layer=cfg.diffmean_and_probe_layer,
                    )
                )
    return sorted(
        requirements,
        key=lambda item: (
            item.family,
            item.target_model,
            item.mode,
            item.concept_id,
        ),
    )


def requirements_from_candidate_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    layer: int,
    counter_direction_family: str,
    method_whitelist: set[str] | None = None,
) -> list[DirectionArtifactRequest]:
    """Directions consumed by an already-materialized candidate population."""

    by_name = {method.name: method for method in ALL_METHODS}
    requirements: set[DirectionArtifactRequest] = set()
    for row in rows:
        method_name = str(row["method_name"])
        if method_whitelist is not None and method_name not in method_whitelist:
            continue
        try:
            method = by_name[method_name]
        except KeyError as exc:
            raise AxBenchBpbAdvError(
                f"candidate row names unknown method {method_name!r}"
            ) from exc
        raw_concept_id = row["concept_id"]
        if type(raw_concept_id) is not int or raw_concept_id < 0:
            raise AxBenchBpbAdvError(
                f"candidate row has invalid concept_id={raw_concept_id!r}"
            )
        concept_id = raw_concept_id
        target_model = str(row["target_model"])
        mode = str(row["mode"])
        if method.target_model != target_model or method.mode.value != mode:
            raise AxBenchBpbAdvError(
                "candidate method coordinates disagree with the method registry: "
                f"method={method_name!r}, target={target_model!r}, mode={mode!r}"
            )
        if method.baseline in {BaselineKind.DIFFMEAN, BaselineKind.LINEAR_PROBE}:
            requirements.add(
                DirectionArtifactRequest(
                    family=method.baseline.value,
                    target_model=target_model,
                    mode=mode,
                    concept_id=concept_id,
                    layer=layer,
                )
            )
        raw_strength = row["adversarial_strength"]
        if isinstance(raw_strength, bool) or not isinstance(raw_strength, (int, float)):
            raise AxBenchBpbAdvError(
                f"candidate row has invalid adversarial_strength={raw_strength!r}"
            )
        if (
            str(row["adversarial_method"])
            == AdversarialMethod.ADDITIVE_DIRECTION.value
            and float(raw_strength) != 0.0
        ):
            requirements.add(
                DirectionArtifactRequest(
                    family=counter_direction_family,
                    target_model=target_model,
                    mode=mode,
                    concept_id=concept_id,
                    layer=layer,
                )
            )
    return sorted(
        requirements,
        key=lambda item: (
            item.family,
            item.target_model,
            item.mode,
            item.concept_id,
        ),
    )


__all__ = ["requirements_from_candidate_rows", "requirements_from_config"]
