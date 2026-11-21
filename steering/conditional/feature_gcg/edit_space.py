from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import torch as t

E = TypeVar("E")


@dataclass
class ValidatedGroup:
    key_str: str
    members: list[Any]
    payload: Any = None


@dataclass
class GroupResult:
    payload: Any


class EditSpace(Protocol[E]):
    def validate(self, group: Any) -> ValidatedGroup: ...
    def is_trivial(self, validated: ValidatedGroup) -> bool: ...
    def finalize_trivial(self, validated: ValidatedGroup) -> GroupResult: ...
    def init_target(self, validated: ValidatedGroup) -> E: ...
    def digest(self, target: E) -> str: ...
    def aggregate_gradient(self, per_member: Iterable[tuple[t.Tensor, Any]]) -> Any: ...
    def rank_candidates(self, agg_grad: Any, current: E, cfg: Any) -> Any: ...
    def sample_candidates(
        self, proposals: Any, current: E, *,
        batch_size: int, temperature: float, rng: random.Random, visited: set[str],
    ) -> list[E]: ...
    def model_item(self, member: Any, target: E) -> Any: ...
    def group_event_fields(self, validated: ValidatedGroup) -> dict[str, Any]: ...
    def step_event_fields(self, validated: ValidatedGroup) -> dict[str, Any]: ...
    def finalize(
        self, validated: ValidatedGroup, target: E, *,
        objective: float, accepted_steps: int, visited_count: int,
    ) -> GroupResult: ...
