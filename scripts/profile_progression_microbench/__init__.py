"""Microbench registry for progressive-eval profiling."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path

MicrobenchResult = dict | list[dict]
MicrobenchFn = Callable[[Path, str, int], MicrobenchResult]

_REGISTRY: dict[str, MicrobenchFn] = {}
_BUILTINS = {
    "encdec_build_conditioning": (
        "scripts.profile_progression_microbench.encdec_build_conditioning"
    ),
}


def register(name: str) -> Callable[[MicrobenchFn], MicrobenchFn]:
    def _decorator(fn: MicrobenchFn) -> MicrobenchFn:
        if name in _REGISTRY and getattr(fn, "__module__", None) != "__main__":
            raise ValueError(f"microbench {name!r} already registered")
        _REGISTRY[name] = fn
        return fn

    return _decorator


def get(name: str) -> MicrobenchFn:
    if name not in _REGISTRY and name in _BUILTINS:
        importlib.import_module(_BUILTINS[name])
    if name not in _REGISTRY:
        raise KeyError(
            f"microbench {name!r} not registered; known: {registered_names()}"
        )
    return _REGISTRY[name]


def registered_names() -> list[str]:
    return sorted(set(_REGISTRY) | set(_BUILTINS))
