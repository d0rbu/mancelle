"""Closed configuration for the opt-in continuous AxBench-Adv suffix."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.gcg_config import GcgObjective


@dataclass(frozen=True)
class AxBenchAdvSoftPromptConfig:
    """Adam settings; the run supplies the campaign seed separately."""

    objective: GcgObjective = GcgObjective.RESPONSE_MARGIN
    alpha: float = 0.01
    device_batch_size: int = 32
    optimized_token_counts: tuple[int, ...] = (0, 1, 4, 16)
    learning_rate: float = 0.01
    max_steps: int = 64
    universal_minibatch_size: int = 16

    def __post_init__(self) -> None:
        try:
            objective = GcgObjective(self.objective)
        except (TypeError, ValueError) as exc:
            raise AxBenchAdvError(f"unknown soft-prompt objective {self.objective!r}") from exc
        object.__setattr__(self, "objective", objective)
        for name in ("alpha", "learning_rate"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
                or (name == "learning_rate" and value == 0)
            ):
                raise AxBenchAdvError(
                    f"soft_prompt.{name} must be finite and "
                    f"{'positive' if name == 'learning_rate' else 'nonnegative'}"
                )
            object.__setattr__(self, name, float(value))
        for name in ("device_batch_size", "max_steps", "universal_minibatch_size"):
            value = getattr(self, name)
            minimum = 0 if name == "max_steps" else 1
            if type(value) is not int or value < minimum:
                raise AxBenchAdvError(f"soft_prompt.{name} must be an integer >= {minimum}")
        counts = self.optimized_token_counts
        if (
            not isinstance(counts, (tuple, list))
            or not counts
            or any(type(value) is not int or value < 0 for value in counts)
            or counts[0] != 0
            or list(counts) != sorted(set(counts))
        ):
            raise AxBenchAdvError(
                "soft_prompt.optimized_token_counts must be unique, ascending, and start at zero"
            )
        object.__setattr__(self, "optimized_token_counts", tuple(counts))

    def effective_identity(self, *, master_seed: int) -> dict[str, Any]:
        """Bind derived algorithm provenance without exposing extra tuning knobs."""

        if type(master_seed) is not int or not 0 <= master_seed < 1 << 63:
            raise AxBenchAdvError("soft-prompt master_seed must be an integer in [0, 2^63 - 1]")
        identity = asdict(self)
        identity["objective"] = self.objective.value
        identity["optimized_token_counts"] = list(self.optimized_token_counts)
        identity["master_seed"] = master_seed
        identity["optimizer"] = {
            "name": "Adam",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "amsgrad": False,
            "foreach": False,
            "fused": False,
            "dtype": "torch.float32",
        }
        identity["native_cast_policy"] = "fp32_master_to_native_embedding_dtype"
        if self.objective is GcgObjective.ACTIVATION_L2:
            del identity["alpha"]
            identity["direction_factor_policy"] = "selected_method_factor"
        return identity


def parse_soft_prompt_config(raw: Any) -> AxBenchAdvSoftPromptConfig:
    if isinstance(raw, AxBenchAdvSoftPromptConfig):
        return raw
    if not isinstance(raw, dict) or set(raw) != {
        "objective",
        "alpha",
        "device_batch_size",
        "optimized_token_counts",
        "learning_rate",
        "max_steps",
        "universal_minibatch_size",
    }:
        raise AxBenchAdvError(
            "soft_prompt config must contain exactly objective, alpha, device_batch_size, "
            "optimized_token_counts, learning_rate, max_steps, universal_minibatch_size"
        )
    return AxBenchAdvSoftPromptConfig(**raw)


__all__ = ["AxBenchAdvSoftPromptConfig", "parse_soft_prompt_config"]
