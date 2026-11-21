"""Logical-turn recovery for BOS-inclusive legacy nanochat renders."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise

from beartype import beartype

from nanochat.tokenizer import LogicalTurnSpan
from steering.segmentation import token_to_char_offsets


class TurnRole(StrEnum):
    """Logical roles emitted by the nanochat conversation renderer."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class ChatSpecialTokens:
    """Special-token IDs needed to recover logical turns without decoding."""

    bos: int
    user_start: int
    assistant_start: int
    output_start: int
    output_end: int

    @classmethod
    def from_tokenizer(cls, tokenizer) -> ChatSpecialTokens:
        """Read the required chat special-token IDs from ``tokenizer``."""
        return cls(
            bos=tokenizer.get_bos_token_id(),
            user_start=tokenizer.encode_special("<user_start>"),
            assistant_start=tokenizer.encode_special("<assistant_start>"),
            output_start=tokenizer.encode_special("<output_start>"),
            output_end=tokenizer.encode_special("<output_end>"),
        )


class TurnSpanError(ValueError):
    """A legacy render or its character offsets violate the turn contract."""


@beartype
def legacy_turn_spans(
    ids: Sequence[int],
    specials: ChatSpecialTokens,
) -> tuple[LogicalTurnSpan, ...]:
    """Recover BOS-free logical turns from one BOS-inclusive legacy render."""
    if not ids:
        raise TurnSpanError("legacy render must contain a BOS token")
    if ids[0] != specials.bos:
        raise TurnSpanError(f"legacy render must start with BOS token {specials.bos}, got {ids[0]}")
    if len(ids) < 2 or ids[1] != specials.user_start:
        raise TurnSpanError("legacy render must have <user_start> immediately after BOS")

    targets = ids[1:]
    spans: list[LogicalTurnSpan] = []
    role: TurnRole | None = None
    start = 0

    def close_span(end: int) -> None:
        if role is None or end <= start:
            return
        span = LogicalTurnSpan(role=role.value, start=start, end=end)
        if spans and spans[-1].role == span.role and spans[-1].end == span.start:
            # Merge the <output_end><output_start> adjacency accepted by the
            # legacy renderer into one logical tool turn.
            previous = spans[-1]
            spans[-1] = LogicalTurnSpan(
                role=span.role,
                start=previous.start,
                end=span.end,
            )
        else:
            spans.append(span)

    for index, token_id in enumerate(targets):
        new_role = None
        if token_id == specials.user_start:
            new_role = TurnRole.USER
        elif token_id == specials.assistant_start:
            new_role = TurnRole.ASSISTANT
        elif token_id == specials.output_start:
            new_role = TurnRole.TOOL
        elif index > 0 and targets[index - 1] == specials.output_end:
            new_role = TurnRole.ASSISTANT

        if new_role is not None:
            close_span(index)
            role = new_role
            start = index

    close_span(len(targets))

    if spans[0].start != 0:
        raise TurnSpanError(
            f"logical turn spans must start at token 0, got {spans[0].start}"
        )
    return tuple(spans)


@beartype
def turn_char_starts(
    spans: Sequence[LogicalTurnSpan],
    char_offsets: Sequence[int],
) -> tuple[int, ...]:
    """Return BOS-free text-coordinate starts for validated logical spans."""
    if not char_offsets:
        raise TurnSpanError("char_offsets must be non-empty")
    if char_offsets[0] != 0:
        raise TurnSpanError("char_offsets[0] must be 0")
    if any(span.start >= len(char_offsets) for span in spans):
        raise TurnSpanError("logical turn start is outside char_offsets")
    if not spans:
        raise TurnSpanError("logical turn spans must not be empty")
    if spans[0].start != 0:
        raise TurnSpanError(
            f"logical turn spans must start at token 0, got {spans[0].start}"
        )

    starts = tuple(char_offsets[span.start] for span in spans)
    if any(right <= left for left, right in pairwise(starts)):
        raise TurnSpanError("logical turn character starts must be strictly increasing")
    return starts


@beartype
def rendered_text_and_turn_char_starts(
    ids: Sequence[int],
    tokenizer,
) -> tuple[str, tuple[int, ...]]:
    """Return the exact BOS-free render text and its logical-turn starts.

    This is the shared producer/consumer bridge for legacy nanochat renders.
    ``ids`` must be the original BOS-inclusive renderer output; character
    offsets are computed over exactly ``ids[1:]``, which is also the sequence
    decoded into the feature-extraction text.  Keeping both results behind one
    validation boundary prevents an inference caller from deriving turns from
    a token slice different from the one sent to Gemma extraction.
    """

    spans = legacy_turn_spans(ids, ChatSpecialTokens.from_tokenizer(tokenizer))
    source_ids = [int(token_id) for token_id in ids[1:]]
    offsets = token_to_char_offsets(tokenizer, source_ids)
    if len(offsets) != len(source_ids) + 1:
        raise TurnSpanError(
            "token_to_char_offsets must return one boundary per BOS-free token "
            f"plus the final end; got {len(offsets)} offsets for "
            f"{len(source_ids)} tokens"
        )
    if any(right < left for left, right in pairwise(offsets)):
        raise TurnSpanError("BOS-free render character offsets must be non-decreasing")

    text = tokenizer.decode(source_ids)
    if offsets[-1] != len(text):
        raise TurnSpanError(
            "BOS-free render character offsets do not cover the exact decoded "
            f"text: final offset={offsets[-1]}, decoded length={len(text)}"
        )
    return text, turn_char_starts(spans, offsets)


__all__ = [
    "ChatSpecialTokens",
    "TurnRole",
    "TurnSpanError",
    "legacy_turn_spans",
    "rendered_text_and_turn_char_starts",
    "turn_char_starts",
]
