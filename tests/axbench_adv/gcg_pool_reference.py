"""Frozen canonical-pool and token-ID normalization references.

Pool construction is from e4c3507ce9ace1f6d1298332f65dc1c2534065ba;
ID normalization is from 5d786892d0db69b957683a58e7b2b7dbeb58a695.
The project pool loop is from 52d49aa6c00f70a05e1ded97f1eadc1c65ed8853.
Only names/signatures and the pool's module-helper bindings differ.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

import torch as t


def original_ids(encoded: object, *, label: str) -> tuple[int, ...]:
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if isinstance(encoded, t.Tensor):
        encoded = encoded.detach().cpu().tolist()
    if isinstance(encoded, list) and encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError(f"{label} unexpectedly returned a batch")
        encoded = encoded[0]
    if not isinstance(encoded, list) or not all(type(value) is int for value in encoded):
        raise ValueError(f"{label} returned invalid input_ids")
    return tuple(encoded)


def scalar_allowed_token_ids(worker: Any, runtime: Any, embedding_rows: int) -> tuple[int, ...]:
    _decode = worker._decode
    _encode = worker._encode
    tokenizer = runtime.tokenizer
    excluded = {int(value) for value in getattr(tokenizer, "all_special_ids", ())}
    added = getattr(tokenizer, "added_tokens_decoder", {})
    if hasattr(added, "keys"):
        excluded.update(int(value) for value in added.keys())
    allowed: list[int] = []
    for token_id in range(embedding_rows):
        if token_id in excluded:
            continue
        try:
            text = _decode(tokenizer, (token_id,))
            if (
                not text
                or "\x00" in text
                or unicodedata.normalize("NFC", text) != text
                or any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in text)
                or _encode(tokenizer, text, add_special_tokens=False) != (token_id,)
            ):
                continue
        except (UnicodeError, ValueError, TypeError, RuntimeError):
            continue
        allowed.append(token_id)
    if not allowed:
        raise ValueError("tokenizer has no canonical non-special GCG token ids")
    return tuple(allowed)


def original_project_allowed_token_ids(
    tokenizer: Any, special_tokens: Sequence[str]
) -> tuple[int, ...]:
    special_ids = {
        int(tokenizer.get_bos_token_id()),
        *(int(tokenizer.encode_special(name)) for name in special_tokens),
    }
    allowed: list[int] = []
    for token_id in range(tokenizer.get_vocab_size()):
        if token_id in special_ids:
            continue
        try:
            text = tokenizer.decode([token_id])
        except (UnicodeError, RuntimeError, ValueError):
            continue
        if not text or "\x00" in text:
            continue
        if any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in text):
            continue
        try:
            text.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            continue
        if tokenizer.encode(text) == [token_id]:
            allowed.append(token_id)
    return tuple(allowed)
