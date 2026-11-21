"""Hydra ``DictConfig`` -> ``AxBenchBpbAdvConfig`` parser (library home).

The single source of truth for parsing the AxBench-BPB-Adv base config out of a
Hydra ``DictConfig``. It lives in the library (not the eval CLI script) so library
modules -- e.g. ``sample_generation`` -- and tests can import the actual definition
without a library-to-script dependency.

Imports only the config dataclasses + the shared error type, so it can be imported
from anywhere in the package without a cycle.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from beartype.roar import BeartypeCallHintParamViolation
from omegaconf import DictConfig, OmegaConf

from steering.eval.axbench_base.profiler import ProfileConfig
from steering.eval.axbench_bpb.config import ContextCacheConfig
from steering.eval.axbench_bpb_adv.config import (
    AdditiveDirectionConfig,
    AdversarialMethod,
    AdversarialMethodConfig,
    AxBenchBpbAdvConfig,
    FasterGcgConfig,
    GeneralValBpbConfig,
    GepaConfig,
    OursGcgConfig,
    RoundtripGenerationConfig,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.roundtrip_steering import (
    ROUNDTRIP_BLOCK_KEY,
    parse_roundtrip_eval_config,
)


def _normalize_tuple_fields(
    payload: dict[str, Any],
    *field_names: str,
) -> dict[str, Any]:
    """Normalize Hydra sequence containers without coercing their elements."""
    for field_name in field_names:
        value = payload.get(field_name)
        if isinstance(value, list):
            payload[field_name] = tuple(value)
    return payload


def _parse_adversarial_method_config(value: Any) -> AdversarialMethodConfig:
    if isinstance(value, dict):
        payload = {str(k): v for k, v in value.items()}
    else:
        raise AxBenchBpbAdvError(
            "adversarial_methods entries must be tagged objects with a "
            f"'method' field, got {type(value).__name__}"
        )
    method_raw = payload.get("method")
    if method_raw is None:
        raise AxBenchBpbAdvError("adversarial_methods entry is missing 'method'")
    method = AdversarialMethod.coerce(str(method_raw))
    config_cls = {
        AdversarialMethod.FASTER_GCG: FasterGcgConfig,
        AdversarialMethod.GEPA: GepaConfig,
        AdversarialMethod.ADDITIVE_DIRECTION: AdditiveDirectionConfig,
    }[method]
    tuple_fields = {
        AdversarialMethod.FASTER_GCG: ("optimized_token_counts",),
        AdversarialMethod.GEPA: ("metric_call_budgets",),
        AdversarialMethod.ADDITIVE_DIRECTION: ("families", "magnitudes"),
    }[method]
    _normalize_tuple_fields(payload, *tuple_fields)
    try:
        return config_cls(**payload)
    except (BeartypeCallHintParamViolation, TypeError) as exc:
        raise AxBenchBpbAdvError(
            f"invalid {method.value} adversarial method config: {exc}"
        ) from exc


def _parse_adversarial_methods(values: Any) -> list[AdversarialMethodConfig]:
    if not isinstance(values, list):
        raise AxBenchBpbAdvError("adversarial_methods must be a list")
    return [_parse_adversarial_method_config(value) for value in values]


def _default_adversarial_method_payloads() -> list[dict[str, Any]]:
    return [
        dataclasses.asdict(method_config)
        for method_config in AxBenchBpbAdvConfig().adversarial_methods
    ]


def _nested_config_mapping(name: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise AxBenchBpbAdvError(f"{name} must be a config object")
    return {str(k): v for k, v in value.items()}


def _parse_roundtrip_generation(value: Any) -> RoundtripGenerationConfig:
    payload = _nested_config_mapping("roundtrip_generation", value)
    try:
        return RoundtripGenerationConfig(**payload)
    except (BeartypeCallHintParamViolation, TypeError) as exc:
        raise AxBenchBpbAdvError(
            f"invalid roundtrip_generation config: {exc}"
        ) from exc


def _parse_roundtrip_steering(
    value: Any,
    *,
    fallback_feature_statistics_path: Any,
):
    payload = _nested_config_mapping("roundtrip_steering", value)
    if "empty_response_policy" in payload:
        raise AxBenchBpbAdvError(
            "roundtrip_steering.empty_response_policy is no longer supported; "
            "empty round-trip responses always raise"
        )
    payload.setdefault("enabled", True)
    if payload.get("feature_statistics_path") is None and fallback_feature_statistics_path:
        payload["feature_statistics_path"] = str(fallback_feature_statistics_path)
    try:
        return parse_roundtrip_eval_config({ROUNDTRIP_BLOCK_KEY: payload})
    except (TypeError, ValueError) as exc:
        raise AxBenchBpbAdvError(f"invalid roundtrip_steering config: {exc}") from exc


def cfg_to_dataclass(cfg: DictConfig) -> AxBenchBpbAdvConfig:
    raw = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(raw, dict)
    resolved = {str(k): v for k, v in raw.items()}
    resolved.pop("hydra", None)
    try:
        context_cache = ContextCacheConfig(
            **_nested_config_mapping("context_cache", resolved.pop("context_cache", {}))
        )
    except (BeartypeCallHintParamViolation, TypeError) as exc:
        raise AxBenchBpbAdvError(
            f"invalid context_cache config: {exc}"
        ) from exc
    try:
        general_val_bpb = GeneralValBpbConfig(
            **_nested_config_mapping(
                "general_val_bpb", resolved.pop("general_val_bpb", {})
            )
        )
    except (BeartypeCallHintParamViolation, TypeError) as exc:
        raise AxBenchBpbAdvError(
            f"invalid general_val_bpb config: {exc}"
        ) from exc
    try:
        profile = ProfileConfig(
            **_nested_config_mapping("profile", resolved.pop("profile", {}))
        )
    except (BeartypeCallHintParamViolation, TypeError) as exc:
        raise AxBenchBpbAdvError(
            f"invalid profile config: {exc}"
        ) from exc
    ours_payload = _nested_config_mapping("ours_gcg", resolved.pop("ours_gcg", {}))
    _normalize_tuple_fields(ours_payload, "reg_lambda_grid", "objective_alpha_grid")
    try:
        ours_gcg = OursGcgConfig(**ours_payload)
    except (BeartypeCallHintParamViolation, TypeError) as exc:
        raise AxBenchBpbAdvError(
            f"invalid ours_gcg config: {exc}"
        ) from exc
    adversarial_methods = _parse_adversarial_methods(
        resolved.pop("adversarial_methods", _default_adversarial_method_payloads())
    )
    roundtrip_generation = _parse_roundtrip_generation(
        resolved.pop("roundtrip_generation", {})
    )
    fallback_feature_statistics_path = resolved.get("feature_statistics_path")
    roundtrip_steering = _parse_roundtrip_steering(
        resolved.pop("roundtrip_steering", {}),
        fallback_feature_statistics_path=fallback_feature_statistics_path,
    )
    _normalize_tuple_fields(resolved, "steering_coefficient_grid")
    valid_fields = {field.name for field in dataclasses.fields(AxBenchBpbAdvConfig)}
    extras = set(resolved) - valid_fields
    if extras:
        raise AxBenchBpbAdvError(
            f"Unknown axbench-bpb-adv config keys: {sorted(extras)}. "
            f"Valid keys: {sorted(valid_fields)} plus nested config blocks."
        )
    try:
        return AxBenchBpbAdvConfig(
            context_cache=context_cache,
            general_val_bpb=general_val_bpb,
            profile=profile,
            ours_gcg=ours_gcg,
            adversarial_methods=adversarial_methods,
            roundtrip_generation=roundtrip_generation,
            roundtrip_steering=roundtrip_steering,
            **resolved,
        )
    except (BeartypeCallHintParamViolation, TypeError) as exc:
        raise AxBenchBpbAdvError(
            f"invalid axbench-bpb-adv config: {exc}"
        ) from exc


__all__ = [
    "_default_adversarial_method_payloads",
    "_nested_config_mapping",
    "_normalize_tuple_fields",
    "_parse_adversarial_method_config",
    "_parse_adversarial_methods",
    "_parse_roundtrip_generation",
    "_parse_roundtrip_steering",
    "cfg_to_dataclass",
]
