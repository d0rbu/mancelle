"""Tokenization helpers for axbench-bpb. Wraps existing primitives so
the bpb-specific error hierarchy is preserved."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from steering.eval.axbench_bpb.types import AxBenchBpbError
from steering.segmentation import compute_tokenizer_checksum, resolve_bos_token_id
from steering.stats_io import find_statistics_path, load_tokenizer_checksum


def encode_prefix(prefix_text: str, tokenizer: Any) -> list[int]:
    """Encode prefix content and prepend its model-owned BOS."""
    bos_id = tokenizer.get_bos_token_id()
    # Prefer `encode(..., prepend=bos)` if the tokenizer supports it;
    # fall back to explicit prepend. The SimpleTokenizer test fixture
    # does not support `prepend=`, so we always go via the explicit path.
    tokens = tokenizer.encode(prefix_text)
    return [bos_id, *tokens]


def encode_output(output_text: str, tokenizer: Any) -> list[int]:
    """Encode output text without BOS."""
    return list(tokenizer.encode(output_text))


def truncate_prefix_left(
    *,
    prefix: list[int],
    n_output: int,
    max_seq_len: int,
) -> tuple[list[int], bool]:
    """Drop leftmost prefix tokens so prefix + output fits in max_seq_len.

    Output tokens are NEVER truncated. If max_seq_len cannot accommodate
    even the output alone, raise AxBenchBpbError.

    Returns (trimmed_prefix, truncated).
    """
    if n_output > max_seq_len:
        raise AxBenchBpbError(
            f"output length {n_output} exceeds max_seq_len {max_seq_len}; "
            f"cannot fit output even with empty prefix"
        )
    if len(prefix) + n_output <= max_seq_len:
        return prefix, False
    allowed_prefix = max_seq_len - n_output
    return prefix[-allowed_prefix:] if allowed_prefix > 0 else [], True


def assert_bos_lengths_match(
    *, tokenizer_extraction: Any, tokenizer_nanochat: Any
) -> None:
    """Ensure both tokenizers' BOS tokens decode to equal-length strings.

    The whole multi-segment char-offset alignment relies on this invariant
    (documented in inline_steering_eval._tokenize_and_split). Gemma's
    <bos> and nanochat's <bos> both decode to 5 chars in production; if
    either ever changes, every char_start / char_end computation silently
    misaligns. Fail loud at startup.

    Uses :func:`steering.segmentation.resolve_bos_token_id` to discover
    the BOS token id across both contracts in use: HF tokenizers expose
    ``bos_token_id`` as an attribute (Gemma's extraction tokenizer) and
    nanochat's RustBPE / TransformersTokenizer expose a
    ``get_bos_token_id()`` method. Without the resolver, calling the
    method form on an HF tokenizer raises AttributeError — which under
    DDP cleanup deadlocks rank 0's destroy_process_group while ranks
    1+ spin on the post-extraction barrier (observed 2026-05-13).
    """
    bos_ext_id = resolve_bos_token_id(tokenizer_extraction)
    bos_nc_id = resolve_bos_token_id(tokenizer_nanochat)
    if bos_ext_id is None or bos_nc_id is None:
        raise AxBenchBpbError(
            f"Could not resolve BOS token id for one of the tokenizers: "
            f"extraction={type(tokenizer_extraction).__name__} (bos_id={bos_ext_id}), "
            f"nanochat={type(tokenizer_nanochat).__name__} (bos_id={bos_nc_id}). "
            f"Both tokenizers must expose either `bos_token_id` attribute "
            f"or `get_bos_token_id()` method."
        )
    bos_ext = tokenizer_extraction.decode([bos_ext_id])
    bos_nc = tokenizer_nanochat.decode([bos_nc_id])
    if len(bos_ext) != len(bos_nc):
        raise AxBenchBpbError(
            f"BOS decoded-length mismatch: extraction tokenizer decodes BOS "
            f"to {bos_ext!r} (len={len(bos_ext)}) but nanochat tokenizer "
            f"decodes BOS to {bos_nc!r} (len={len(bos_nc)}). The multi-segment "
            f"char-offset alignment assumes these are equal; see inline_steering_eval."
        )


def assert_tokenizer_checksum_matches(
    *, tokenizer: Any, feature_dir: Path | str
) -> None:
    """Ensure the nanochat tokenizer's checksum matches the feature_dir's
    recorded checksum. Mirrors the startup check in
    scripts/conditional_encdec_train.py.

    P1.B fix (slice 17 audit 2026-05-05): resolves the canonical
    ``feature_statistics.safetensors`` path inside ``feature_dir`` via
    ``steering.stats_io.find_statistics_path``. If the resolution fails
    (no stats file under ``feature_dir``), raise ``AxBenchBpbError`` —
    the wire-up site should ensure the stats file exists. If the stats
    file exists but carries no recorded checksum, log a warning and
    skip the check (matching the reference training script's behaviour:
    older stats files predate the checksum metadata).
    """
    actual = compute_tokenizer_checksum(tokenizer)
    stats_path = find_statistics_path(feature_dir)
    if stats_path is None:
        raise AxBenchBpbError(
            f"feature_dir {feature_dir} has no feature_statistics.safetensors; "
            f"cannot verify tokenizer checksum. Either re-run extraction or "
            f"point feature_dir at a directory that contains the stats file."
        )
    expected = load_tokenizer_checksum(stats_path)
    if expected is None:
        logger.warning(
            f"Tokenizer checksum check skipped: {stats_path} has no recorded "
            f"checksum (older stats file predating the checksum metadata). "
            f"Re-run feature extraction to populate the checksum."
        )
        return
    if actual != expected:
        raise AxBenchBpbError(
            f"Tokenizer checksum mismatch: nanochat tokenizer has checksum "
            f"{actual!r} but feature_dir {feature_dir} was built with checksum "
            f"{expected!r}. The feature shards are incompatible with this "
            f"tokenizer — re-run feature extraction or load a matching tokenizer."
        )
