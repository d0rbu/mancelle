"""Display-mode transcript rendering: raw tokens vs the familiar chat view.

The settled transcript is rendered by an exact token-level mapper
(``pretty_view``) so segment colors always cover the same tokens they cover in
raw mode — including the ``› ``/``• `` markers standing in for the turn-start
specials — while in-flight streaming text uses a cheap literal substitution
(``stream_display_text``) that the next settled render supersedes.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, replace
from typing import Any

from scripts.chat_tui.session import TranscriptSpan
from steering.segmentation import token_to_char_offsets

USER_MARKER = "› "
ASSISTANT_MARKER = "• "


@dataclass(frozen=True)
class SpecialDisplay:
    """Special-token ids and their decoded literals for one tokenizer."""

    bos_id: int
    user_start_id: int
    user_end_id: int
    assistant_start_id: int
    assistant_end_id: int
    literals: tuple[tuple[str, str], ...]
    """(decoded literal, replacement) pairs for in-flight text substitution."""


def build_special_display(tokenizer: Any) -> SpecialDisplay:
    bos_id = int(tokenizer.get_bos_token_id())
    user_start_id = int(tokenizer.encode_special("<user_start>"))
    user_end_id = int(tokenizer.encode_special("<user_end>"))
    assistant_start_id = int(tokenizer.encode_special("<assistant_start>"))
    assistant_end_id = int(tokenizer.encode_special("<assistant_end>"))
    literals = tuple(
        (literal, replacement)
        for literal, replacement in (
            (tokenizer.decode([user_start_id]), f"\n\n{USER_MARKER}"),
            (tokenizer.decode([assistant_start_id]), f"\n\n{ASSISTANT_MARKER}"),
            (tokenizer.decode([user_end_id]), ""),
            (tokenizer.decode([assistant_end_id]), ""),
            (tokenizer.decode([bos_id]), ""),
        )
        if literal
    )
    return SpecialDisplay(
        bos_id=bos_id,
        user_start_id=user_start_id,
        user_end_id=user_end_id,
        assistant_start_id=assistant_start_id,
        assistant_end_id=assistant_end_id,
        literals=literals,
    )


def _pretty_piece(
    token_id: int,
    raw_piece: str,
    special: SpecialDisplay,
    at_start: bool,
) -> str:
    if token_id == special.bos_id:
        return ""
    if token_id == special.user_start_id:
        return USER_MARKER if at_start else f"\n\n{USER_MARKER}"
    if token_id == special.assistant_start_id:
        return ASSISTANT_MARKER if at_start else f"\n\n{ASSISTANT_MARKER}"
    if token_id in (special.user_end_id, special.assistant_end_id):
        return ""
    return raw_piece


def pretty_view(
    tokenizer: Any,
    tokens: tuple[int, ...],
    spans: tuple[TranscriptSpan, ...],
    special: SpecialDisplay,
) -> tuple[str, tuple[TranscriptSpan, ...]]:
    """Render the settled transcript in chat form with exactly remapped spans.

    Raw char offsets are derived from the tokens themselves, so every span
    boundary (token-aligned by construction) maps to the boundary of the same
    token's replacement text; markers therefore inherit the color of the token
    they stand in for, and omitted tokens collapse their span coverage.
    """

    if not tokens:
        if spans:
            raise ValueError("transcript spans exist for an empty conversation")
        return "", ()
    raw_offsets = token_to_char_offsets(tokenizer, list(tokens))
    raw_text = tokenizer.decode(list(tokens))
    if raw_offsets[-1] != len(raw_text):
        raise ValueError(
            f"token char offsets end at {raw_offsets[-1]} but the decoded "
            f"transcript has {len(raw_text)} chars"
        )

    pieces: list[str] = []
    pretty_offsets = [0] * (len(tokens) + 1)
    length = 0
    for index, token_id in enumerate(tokens):
        pretty_offsets[index] = length
        raw_piece = raw_text[raw_offsets[index] : raw_offsets[index + 1]]
        piece = _pretty_piece(int(token_id), raw_piece, special, at_start=length == 0)
        pieces.append(piece)
        length += len(piece)
    pretty_offsets[len(tokens)] = length

    def map_offset(raw_offset: int) -> int:
        boundary = bisect_left(raw_offsets, raw_offset)
        if boundary == len(raw_offsets) or raw_offsets[boundary] != raw_offset:
            raise ValueError(f"span offset {raw_offset} is not a token boundary of the transcript")
        # Zero-width raw tokens (e.g. a BOS the tokenizer decodes to nothing)
        # share a raw boundary; bisect_left lands on the FIRST such boundary,
        # which maps span starts before the omitted token's (empty) output and
        # keeps end offsets minimal — both sides stay token-aligned.
        return pretty_offsets[boundary]

    mapped = tuple(
        replace(
            span,
            char_start=map_offset(span.char_start),
            char_end=map_offset(span.char_end),
        )
        for span in spans
    )
    return "".join(pieces), mapped


def stream_display_text(raw_text: str, special: SpecialDisplay) -> str:
    """Cheap in-flight chat view: literal marker substitution on raw text.

    Display-only; the exact ``pretty_view`` render at operation settle
    supersedes it, so a (practically impossible) generated collision with a
    special literal self-heals.
    """

    display = raw_text
    for literal, replacement in special.literals:
        display = display.replace(literal, replacement)
    return display.lstrip("\n")
