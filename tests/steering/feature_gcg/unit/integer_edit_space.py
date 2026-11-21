from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from steering.conditional.feature_gcg.edit_space import (
    GroupResult,
    ValidatedGroup,
)


@dataclass
class IntGroup:
    key_str: str
    members: list[int]
    is_control: bool


class IntegerEditSpace:
    def validate(self, group: IntGroup) -> ValidatedGroup:
        return ValidatedGroup(key_str=group.key_str, members=list(group.members),
                              payload={"is_control": group.is_control})

    def is_trivial(self, validated: ValidatedGroup) -> bool:
        return bool(validated.payload["is_control"])

    def finalize_trivial(self, validated: ValidatedGroup) -> GroupResult:
        return GroupResult(payload={"control": True})

    def init_target(self, validated: ValidatedGroup) -> int:
        return int(validated.members[0])

    def digest(self, target: int) -> str:
        return str(int(target))

    def aggregate_gradient(self, per_member: Any) -> None:
        for _ in per_member:  # consume so the strict zip in run_gcg is exercised
            pass
        return None

    def rank_candidates(self, agg_grad: Any, current: int, cfg: Any) -> list[int]:
        return [current - 1, current + 1]

    def sample_candidates(self, proposals: list[int], current: int, *, batch_size: int,
                          temperature: float, rng: random.Random, visited: set[str]) -> list[int]:
        out: list[int] = []
        for value in proposals:
            d = self.digest(value)
            if d in visited:
                continue
            visited.add(d)
            out.append(value)
            if len(out) >= batch_size:
                break
        return out

    def model_item(self, member: int, target: int) -> tuple[int, int]:
        return (member, target)

    def group_event_fields(self, validated: ValidatedGroup) -> dict[str, Any]:
        return {"key_str": validated.key_str, "n_members": len(validated.members)}

    def step_event_fields(self, validated: ValidatedGroup) -> dict[str, Any]:
        return {"key_str": validated.key_str, "n_members": len(validated.members)}

    def finalize(self, validated: ValidatedGroup, target: int, *, objective: float,
                 accepted_steps: int, visited_count: int) -> GroupResult:
        return GroupResult(payload={
            "target": target, "objective": objective,
            "accepted_steps": accepted_steps, "visited_count": visited_count})
