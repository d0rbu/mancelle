"""Progression results.json variant types and JSON loader/dumper."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SuiteResultKind(StrEnum):
    """Canonical kind discriminators for SuiteResultVariant tagged union."""

    STATIC_YAML = "static_yaml"
    TARGET_REAL_LOCKSTEP = "target_real_lockstep"
    TARGET_REAL_POOLED = "target_real_pooled"
    SELF_CONDITIONED = "self_conditioned"


@dataclass(frozen=True)
class StaticYamlSuiteResult:
    name: str
    target_conditioning_mode: str = "current"
    seeds: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return SuiteResultKind.STATIC_YAML.value


@dataclass(frozen=True)
class TargetRealLockstepSuiteResult:
    name: str
    cache_dir: str
    target_conditioning_mode: str = "current"
    seeds: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return SuiteResultKind.TARGET_REAL_LOCKSTEP.value


@dataclass(frozen=True)
class TargetRealPooledSuiteResult:
    name: str
    cache_dir: str
    pooling_op: str
    target_conditioning_mode: str = "current"
    seeds: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return SuiteResultKind.TARGET_REAL_POOLED.value


@dataclass(frozen=True)
class SelfConditionedSuiteResult:
    name: str
    cache_dir: str
    seeds: dict[str, dict[str, Any]] = field(default_factory=dict)
    @property
    def kind(self) -> SuiteResultKind:
        return SuiteResultKind.SELF_CONDITIONED


type SuiteResultVariant = (
    StaticYamlSuiteResult
    | TargetRealLockstepSuiteResult
    | TargetRealPooledSuiteResult
    | SelfConditionedSuiteResult
)


def dump_suite_to_dict(variant: SuiteResultVariant) -> dict[str, Any]:
    """Serialize a variant to its results.json shape."""
    match variant:
        case SelfConditionedSuiteResult(cache_dir=cd, seeds=seeds, kind=kind):
            return {"kind": kind.value, "cache_dir": cd, "seeds": dict(seeds)}
        case StaticYamlSuiteResult(name=_, target_conditioning_mode=tcm, seeds=seeds):
            return {
                "kind": "static_yaml",
                "target_conditioning_mode": tcm,
                "seeds": dict(seeds),
            }
        case TargetRealLockstepSuiteResult(name=_, target_conditioning_mode=tcm, cache_dir=cd, seeds=seeds):
            return {
                "kind": "target_real_lockstep",
                "target_conditioning_mode": tcm,
                "cache_dir": cd,
                "seeds": dict(seeds),
            }
        case TargetRealPooledSuiteResult(name=_, target_conditioning_mode=tcm, cache_dir=cd, pooling_op=op, seeds=seeds):
            return {
                "kind": "target_real_pooled",
                "target_conditioning_mode": tcm,
                "cache_dir": cd,
                "pooling_op": op,
                "seeds": dict(seeds),
            }


def _require_str_field(data: dict, key: str, kind_str: str) -> str:
    """Return str(data[key]) or raise a ValueError with an actionable message."""
    if key not in data:
        raise ValueError(
            f"target_real result yaml (kind={kind_str!r}) missing required field "
            f"{key!r}; got keys: {sorted(data.keys())}"
        )
    return str(data[key])


def load_suite_from_dict(data: dict[str, Any], *, name: str) -> SuiteResultVariant:
    """Deserialize a variant from results.json."""
    kind = data.get("kind")
    seeds = dict(data.get("seeds", {}))
    if kind == SuiteResultKind.SELF_CONDITIONED:
        return SelfConditionedSuiteResult(name=name, cache_dir=_require_str_field(data, "cache_dir", kind),
                                         seeds=seeds)
    if kind is None:
        raise ValueError(
            "progression result entry missing required 'kind' field; expected "
            "static_yaml | target_real_lockstep | target_real_pooled"
        )
    if kind == "static_yaml":
        return StaticYamlSuiteResult(
            name=name,
            target_conditioning_mode=_require_str_field(data, "target_conditioning_mode", kind),
            seeds=seeds,
        )
    if kind == "target_real_lockstep":
        return TargetRealLockstepSuiteResult(
            name=name,
            target_conditioning_mode=_require_str_field(data, "target_conditioning_mode", kind),
            cache_dir=_require_str_field(data, "cache_dir", kind),
            seeds=seeds,
        )
    if kind == "target_real_pooled":
        return TargetRealPooledSuiteResult(
            name=name,
            target_conditioning_mode=_require_str_field(data, "target_conditioning_mode", kind),
            cache_dir=_require_str_field(data, "cache_dir", kind),
            pooling_op=_require_str_field(data, "pooling_op", kind),
            seeds=seeds,
        )
    raise ValueError(f"unknown kind {kind!r}; expected static_yaml | target_real_lockstep | target_real_pooled")
