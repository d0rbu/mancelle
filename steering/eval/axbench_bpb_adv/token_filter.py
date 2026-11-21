"""Allowed-token filtering for prompt-token adversaries."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable

from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError


def _tokenizer_vocab_size(tokenizer) -> int:
    if hasattr(tokenizer, "get_vocab_size"):
        return int(tokenizer.get_vocab_size())
    if hasattr(tokenizer, "vocab_size"):
        return int(tokenizer.vocab_size)
    raise AxBenchBpbAdvError(
        f"tokenizer {type(tokenizer).__name__} has no get_vocab_size() or vocab_size"
    )


def _maybe_special_id(tokenizer, attr_name: str) -> int | None:
    value = getattr(tokenizer, attr_name, None)
    if callable(value):
        value = value()
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def default_special_token_ids(tokenizer) -> set[int]:
    ids: set[int] = set()
    for attr in (
        "get_bos_token_id",
        "bos_token_id",
        "get_eos_token_id",
        "eos_token_id",
        "get_pad_token_id",
        "pad_token_id",
    ):
        token_id = _maybe_special_id(tokenizer, attr)
        if token_id is not None:
            ids.add(token_id)
    return ids


def decodes_to_control_only(text: str) -> bool:
    if text == "":
        return True
    return all(unicodedata.category(char).startswith("C") for char in text)


def build_allowed_token_ids(
    tokenizer,
    *,
    extra_disallowed_ids: Iterable[int] = (),
) -> tuple[int, ...]:
    vocab_size = _tokenizer_vocab_size(tokenizer)
    if vocab_size <= 0:
        raise AxBenchBpbAdvError(f"tokenizer vocab size must be positive, got {vocab_size}")
    disallowed = default_special_token_ids(tokenizer)
    disallowed.update(int(token_id) for token_id in extra_disallowed_ids)
    allowed: list[int] = []
    for token_id in range(vocab_size):
        if token_id in disallowed:
            continue
        try:
            decoded = tokenizer.decode([token_id])
        except Exception:
            continue
        if decodes_to_control_only(str(decoded)):
            continue
        allowed.append(token_id)
    if not allowed:
        raise AxBenchBpbAdvError("allowed token filter removed every token")
    return tuple(allowed)
