"""Closed configuration and identity for AxBench-Adv's opt-in GCG pressure."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from typing import Any

from beartype.roar import BeartypeException

from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_bpb_adv.config import FasterGcgConfig


class GcgObjective(StrEnum):
    RESPONSE_MARGIN = "response_margin"
    ACTIVATION_L2 = "activation_l2"


@dataclass(frozen=True)
class AxBenchAdvGcgConfig:
    """Search settings; disabled runs never serialize this configuration."""

    objective: GcgObjective = GcgObjective.RESPONSE_MARGIN
    alpha: float = 0.01
    device_batch_size: int = 32
    search: FasterGcgConfig = field(default_factory=FasterGcgConfig)

    def __post_init__(self) -> None:
        try:
            objective = GcgObjective(self.objective)
        except ValueError as exc:
            raise AxBenchAdvError(f"unknown GCG objective {self.objective!r}") from exc
        object.__setattr__(self, "objective", objective)
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)):
            raise AxBenchAdvError("GCG alpha must be finite and nonnegative")
        alpha = float(self.alpha)
        if not math.isfinite(alpha) or alpha < 0.0:
            raise AxBenchAdvError("GCG alpha must be finite and nonnegative")
        object.__setattr__(self, "alpha", alpha)
        if type(self.device_batch_size) is not int or self.device_batch_size <= 0:
            raise AxBenchAdvError("GCG device_batch_size must be positive")
        if not isinstance(self.search, FasterGcgConfig):
            raise AxBenchAdvError("GCG search must be a FasterGcgConfig")

    def resolved_search(self, *, master_seed: int) -> FasterGcgConfig:
        """Bind the campaign seed while retaining the shared Faster-GCG surface."""

        return replace(self.search, master_seed=master_seed)

    def effective_identity(self) -> dict[str, Any]:
        """Return cache identity; alpha is scientifically irrelevant to activation-L2."""

        identity: dict[str, Any] = {
            "objective": self.objective.value,
            "device_batch_size": self.device_batch_size,
            "search": asdict(self.search),
        }
        if self.objective is GcgObjective.RESPONSE_MARGIN:
            identity["alpha"] = self.alpha
        else:
            identity["direction_factor_policy"] = "selected_method_factor"
        return identity


def parse_gcg_config(raw: Any) -> AxBenchAdvGcgConfig:
    """Construct the nested typed config from a strict plain mapping."""

    if isinstance(raw, AxBenchAdvGcgConfig):
        return raw
    if not isinstance(raw, dict) or set(raw) != {
        "objective",
        "alpha",
        "device_batch_size",
        "search",
    }:
        raise AxBenchAdvError(
            "gcg config must contain exactly objective, alpha, device_batch_size, search"
        )
    search = raw["search"]
    if not isinstance(search, dict):
        raise AxBenchAdvError("gcg.search must be a mapping")
    search = dict(search)
    if isinstance(search.get("optimized_token_counts"), list):
        search["optimized_token_counts"] = tuple(search["optimized_token_counts"])
    try:
        parsed_search = FasterGcgConfig(**search)
    except (BeartypeException, TypeError, ValueError) as exc:
        raise AxBenchAdvError(f"invalid gcg.search config: {exc}") from exc
    return AxBenchAdvGcgConfig(
        objective=raw["objective"],
        alpha=raw["alpha"],
        device_batch_size=raw["device_batch_size"],
        search=parsed_search,
    )


__all__ = ["AxBenchAdvGcgConfig", "GcgObjective", "parse_gcg_config"]
