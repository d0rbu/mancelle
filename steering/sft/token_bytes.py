"""Project-owned Hugging Face token-byte accounting for BPB evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from loguru import logger
from tokenizers.decoders import ByteLevel

EXPECTED_DECODE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    UnicodeDecodeError,
    UnicodeError,
    RuntimeError,
)


def compute_hf_token_bytes(
    tokenizer_adapter: Any,
    *,
    additional_zero_ids: Iterable[int] = (),
) -> list[int]:
    """Return UTF-8 byte lengths, zeroing tokenizer and project controls."""
    tokenizer = tokenizer_adapter.tokenizer
    zero_ids = set(tokenizer.all_special_ids)
    zero_ids.update(int(token_id) for token_id in additional_zero_ids)
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    byte_level_vocab = isinstance(getattr(backend_tokenizer, "decoder", None), ByteLevel)
    added_ids = set(tokenizer.get_added_vocab().values()) if byte_level_vocab else set()
    out: list[int] = []
    for token_id in range(tokenizer_adapter.get_vocab_size()):
        if token_id in zero_ids:
            out.append(0)
            continue
        if byte_level_vocab and token_id not in added_ids:
            # ByteLevel maps each source byte to one Unicode code point before
            # BPE. The vocabulary piece length therefore remains exact even
            # when decoding that piece alone would emit U+FFFD.
            piece = tokenizer.convert_ids_to_tokens(token_id)
            if not isinstance(piece, str):
                raise RuntimeError(f"ByteLevel token id={token_id} did not resolve to text")
            out.append(len(piece))
            continue
        try:
            text = tokenizer_adapter.decode([token_id])
        except EXPECTED_DECODE_EXCEPTIONS as exc:
            logger.warning(
                f"compute_hf_token_bytes: decode failed for id={token_id}: "
                f"{type(exc).__name__}: {exc}; recording 0 bytes"
            )
            out.append(0)
            continue
        out.append(len(text.encode("utf-8")) if text else 0)
    return out
