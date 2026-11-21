"""Generic BOS-safe prompt-token editing primitives for the GCG loop."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from steering.conditional.feature_gcg.errors import FeatureGcgError


@dataclass(frozen=True)
class EditableSpan:
    start: int
    count: int

    @property
    def stop(self) -> int:
        return self.start + self.count

    def positions(self) -> range:
        return range(self.start, self.stop)


@dataclass(frozen=True)
class TokenReplacement:
    position: int
    token_id: int
    score: float


def token_sequence_digest(token_ids: Sequence[int]) -> str:
    h = hashlib.sha256()
    for token_id in token_ids:
        h.update(int(token_id).to_bytes(8, byteorder="little", signed=True))
    return h.hexdigest()


def replace_single_token(
    prompt_tokens: Sequence[int],
    *,
    position: int,
    token_id: int,
) -> list[int]:
    if position <= 0:
        raise FeatureGcgError(f"position {position} would edit BOS")
    if position >= len(prompt_tokens):
        raise FeatureGcgError(
            f"position {position} out of range for prompt length {len(prompt_tokens)}"
        )
    out = [int(existing) for existing in prompt_tokens]
    out[position] = int(token_id)
    return out
