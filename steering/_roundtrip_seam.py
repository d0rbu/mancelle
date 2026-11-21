"""Round-trip seam machinery for the post-extraction feature-shard splice.

Weight-free (no model weights / no GPU). Decides whether a recombined token
sequence re-tokenizes to itself under the SAE tokenizer — the property that
keeps every derived document a real, extraction-equivalent document (so
gradient-attribution + source-cross-ref validation continue to work on the
derived directory). See
``docs/superpowers/specs/2026-06-30-post-extraction-feature-shard-splicer-design.md``.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger
from transformers import PreTrainedTokenizerBase

from steering.segmentation import resolve_bos_token_id


class Gemma3MarkerError(RuntimeError):
    """Too many gemma-3 U+2581 source-doc normalizations in one shard — the
    per-shard cap was exceeded, which signals systematic source-data corruption
    rather than the expected rare marker doc."""

    def __init__(self, doc_index: int) -> None:
        super().__init__(
            f"gemma-3 U+2581 normalization cap exceeded at source doc {doc_index}"
        )
        self.doc_index = doc_index


def _input_ids_tuple(encoding: Any) -> tuple[int, ...]:
    """Normalize a tokenizer ``__call__`` result to a flat tuple of ints."""
    try:
        ids = encoding["input_ids"]
    except Exception as exc:  # noqa: BLE001 — surface a clean contract error
        raise TypeError(
            "tokenizer encoding must expose an 'input_ids' field"
        ) from exc
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    try:
        return tuple(int(x) for x in ids)
    except Exception as exc:  # noqa: BLE001
        raise TypeError(
            f"tokenizer input_ids must be an integer sequence, got {ids!r}"
        ) from exc


def to_text(ids: tuple[int, ...], tokenizer: Any) -> str:
    """Detokenize to BOS-free text (matching the BOS-free source parquet).

    Strips a single leading BOS token before decoding when the tokenizer has a
    discoverable BOS id and ``ids`` starts with it.
    """
    if not ids:
        return ""
    bos_id = resolve_bos_token_id(tokenizer)
    body = ids[1:] if (bos_id is not None and ids[0] == bos_id) else ids
    if (
        isinstance(tokenizer, PreTrainedTokenizerBase)
        and getattr(tokenizer.decode, "__func__", None) is PreTrainedTokenizerBase.decode
    ):
        # HF converts the array to Python IDs without scanning a numeric list.
        return tokenizer.decode(np.asarray(body, dtype=object))
    return tokenizer.decode(list(body))


def _reencodes_to(text: str, ids: tuple[int, ...], tokenizer: Any) -> bool:
    """True iff re-tokenizing ``text`` reproduces ``ids`` exactly."""
    return _input_ids_tuple(tokenizer(text, padding=False)) == tuple(ids)


def is_roundtrip_stable(ids: tuple[int, ...], tokenizer: Any) -> bool:
    """True iff ``tokenizer(to_text(ids)) == ids`` — re-tokenizing the decoded
    text reproduces the intended token ids exactly."""
    return _reencodes_to(to_text(ids, tokenizer), ids, tokenizer)


# ---------------------------------------------------------------------------
# Windowed seam checker
# ---------------------------------------------------------------------------
#
# A tail-swap at boundary ``k`` only introduces TWO new seams:
# ``A[k] | B[k+1]`` and ``B[k] | A[k+1]``. Source documents are full-document
# round-trip-validated once, and every previously committed seam was already
# checked, so a bounded local check around each new seam is sufficient — provided
# the window is wide enough to capture any cross-seam merge. The window side is
# the tokenizer's maximum decoded UTF-8 byte length plus slack (HF) or a fixed
# conservative constant (non-HF). The whole class is exactly equivalent to the
# full-document oracle (proven by the differential tests).

_HF_WINDOW_GUARD_TOKENS = 8
_NON_HF_WINDOW_SIDE_TOKENS = 64
_WINDOW_SIDE_TOKENS_CACHE: dict[int, tuple[weakref.ReferenceType, int]] = {}


def _looks_like_hf_tokenizer(tokenizer: Any) -> bool:
    return callable(getattr(tokenizer, "get_vocab", None)) or hasattr(
        tokenizer, "vocab_size"
    )


def _infer_window_side_tokens(tokenizer: Any) -> int:
    """Per-side token window for the seam check (uncached).

    Non-HF tokenizers use a fixed conservative constant. HF-like tokenizers infer
    a window from the maximum decoded UTF-8 byte length across the vocabulary plus
    a guard, so any byte-fallback / merge run at a seam is captured.
    """
    if not _looks_like_hf_tokenizer(tokenizer):
        return _NON_HF_WINDOW_SIDE_TOKENS
    try:
        n_vocab = len(tokenizer)
    except Exception:  # noqa: BLE001 — fall back to the explicit vocab_size attr
        try:
            n_vocab = int(tokenizer.vocab_size)
        except Exception as exc:  # noqa: BLE001
            raise TypeError(
                "HF-like tokenizer must expose len(tokenizer) or integer vocab_size"
            ) from exc
    if n_vocab <= 0:
        raise ValueError(f"tokenizer vocabulary size must be positive, got {n_vocab}")
    max_bytes = 0
    for token_id in range(n_vocab):
        piece = tokenizer.decode([token_id])
        max_bytes = max(max_bytes, len(piece.encode("utf-8")))
    return max(1, max_bytes + _HF_WINDOW_GUARD_TOKENS)


def _cached_window_side_tokens(tokenizer: Any) -> int:
    """``id()``-keyed cache, validated with a weakref so a reused id never returns a
    stale window. ``id()`` is only unique among *live* objects: once a tokenizer is
    garbage-collected a new object can take its address, and an unguarded cache
    would then hand the new tokenizer the old one's window. The weakref check
    (``ref() is tokenizer``) detects that collision and recomputes. Tokenizers that
    cannot be weakly referenced are simply not cached (still correct)."""
    key = id(tokenizer)
    cached = _WINDOW_SIDE_TOKENS_CACHE.get(key)
    if cached is not None and cached[0]() is tokenizer:
        return cached[1]
    value = _infer_window_side_tokens(tokenizer)
    try:
        _WINDOW_SIDE_TOKENS_CACHE[key] = (weakref.ref(tokenizer), value)
    except TypeError:
        _WINDOW_SIDE_TOKENS_CACHE.pop(key, None)  # not weakly referenceable
    return value


@dataclass
class _SeamCheckerStats:
    tokenizer_calls: int = 0
    local_checks: int = 0
    local_accepts: int = 0
    local_rejects: int = 0


class WindowedSeamChecker:
    """Production seam checker: prove new seams locally; fail-fast on tokenizer
    contract violations.  Exactly equivalent to the full-document oracle."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.stats = _SeamCheckerStats()

    @property
    def window_side_tokens(self) -> int:
        return _cached_window_side_tokens(self.tokenizer)

    # -- windowed local check ----------------------------------------------
    def _encode_window(self, text: str, *, starts_at_doc_start: bool) -> tuple[int, ...]:
        if starts_at_doc_start:
            encoding = self.tokenizer(text, padding=False)
        else:
            encoding = self.tokenizer(text, padding=False, add_special_tokens=False)
        return _input_ids_tuple(encoding)

    def _encode_expected_window(self, expected_ids: tuple[int, ...], *, starts_at_doc_start: bool) -> tuple[int, ...]:
        text = (
            to_text(expected_ids, self.tokenizer)
            if starts_at_doc_start
            else self.tokenizer.decode(list(expected_ids))
        )
        self.stats.tokenizer_calls += 1
        return self._encode_window(text, starts_at_doc_start=starts_at_doc_start)

    @staticmethod
    def _first_mismatch(left: tuple[int, ...], right: tuple[int, ...]) -> int | None:
        n = min(len(left), len(right))
        for idx in range(n):
            if left[idx] != right[idx]:
                return idx
        if len(left) != len(right):
            return n
        return None

    def stitch_is_stable(self, prefix_ids: tuple[int, ...], suffix_ids: tuple[int, ...]) -> bool:
        """Windowed round-trip verdict for the single junction ``prefix_ids | suffix_ids``.

        Takes up to ``window_side_tokens`` from the tail of ``prefix_ids`` and head of
        ``suffix_ids`` (shorter sequences are used in full). The window is doubled on each
        side to leave room for conservative edge trimming when a far edge cuts a multi-byte
        character; the core (``side_tokens`` per side) is never trimmed.

        ``starts_at_doc_start`` is true when the full ``prefix_ids`` fits in the window
        (i.e. the window spans back to the start of the left document), so BOS-aware
        encoding is used on that path.
        """
        side_tokens = self.window_side_tokens
        outer = side_tokens * 2

        left_start = max(0, len(prefix_ids) - outer)
        window_left = prefix_ids[left_start:]
        window_right = suffix_ids[:outer]

        starts_at_doc_start = left_start == 0
        seam_offset = len(window_left)
        window_ids = window_left + window_right

        if not window_ids:
            raise ValueError("seam window unexpectedly produced no token ids")

        self.stats.local_checks += 1
        expected_ids = window_ids
        encoded_ids = self._encode_expected_window(
            expected_ids, starts_at_doc_start=starts_at_doc_start
        )
        if encoded_ids == expected_ids:
            self.stats.local_accepts += 1
            return True

        # The diagnostic window may itself cut a byte-fallback character at a far
        # edge; trim conservatively (never inside the required seam core) and retry.
        left_count = seam_offset
        right_count = len(expected_ids) - seam_offset
        max_left_trim = left_count - min(side_tokens, left_count)
        max_right_trim = right_count - min(side_tokens, right_count)
        left_trim = right_trim = 0
        while left_trim < max_left_trim or right_trim < max_right_trim:
            first_mismatch = self._first_mismatch(expected_ids, encoded_ids)
            if first_mismatch is None:
                raise AssertionError("unreachable: unequal windows had no mismatch")
            left_extra = max_left_trim - left_trim
            right_extra = max_right_trim - right_trim
            if left_extra and first_mismatch <= left_extra:
                left_trim += 1
            elif right_extra and first_mismatch >= len(expected_ids) - right_extra:
                right_trim += 1
            else:
                break
            end = len(window_ids) - right_trim if right_trim else len(window_ids)
            expected_ids = window_ids[left_trim:end]
            encoded_ids = self._encode_expected_window(
                expected_ids,
                starts_at_doc_start=starts_at_doc_start and left_trim == 0,
            )
            if encoded_ids == expected_ids:
                self.stats.local_accepts += 1
                return True
        self.stats.local_rejects += 1
        return False




# ---------------------------------------------------------------------------
# Gemma-3 U+2581 source-data normalization
# ---------------------------------------------------------------------------
#
# Gemma-3 parquet text may contain literal SentencePiece space markers (U+2581)
# that decode as normal spaces. The source doc's own tokenization is then not
# idempotent. When (and only when) the source differs from to_text(ids) solely by
# U+2581 -> space, the re-tokenized length is identical, and every mismatched run
# has the same joined tokenizer pieces containing U+2581, we substitute the exact
# re-tokenization of the decoded text before recombination. Any other shape is
# left untouched and surfaces through the normal round-trip guards.

_GEMMA3_MARKER = "▁"
_GEMMA3_MARKER_MAX_DOCS_PER_SHARD = 64
_GEMMA3_MARKER_MAX_MISMATCHES_PER_DOC = 64
_GEMMA3_MARKER_MAX_MISMATCHES_PER_SHARD = 512


@dataclass(frozen=True)
class _Gemma3Normalization:
    ids: list[int]
    token_mismatches: int


def _tokenizer_name(tokenizer: Any) -> str:
    return str(getattr(tokenizer, "name_or_path", None) or tokenizer.__class__.__name__)


def _is_gemma3_tokenizer(tokenizer: Any) -> bool:
    # Name-substring gate is a DELIBERATE CONVENTION: EXTRACTION_MODELS names are
    # canonical identifiers and the most reliable signal. Behavior-based detection
    # (e.g. inspecting the vocabulary for U+2581 markers) was considered and
    # rejected as less predictable — future tokenizers could share the same behavior
    # without the same marker semantics. Don't "fix" this to a behavior-based check.
    name = _tokenizer_name(tokenizer).lower()
    return "gemma-3" in name or "gemma3" in name


def _token_pieces(tokenizer: Any, ids: tuple[int, ...]) -> tuple[str, ...]:
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(convert):
        raise TypeError(
            "gemma-3 U+2581 normalization requires tokenizer.convert_ids_to_tokens"
        )
    pieces = convert(list(ids))
    if isinstance(pieces, str):
        pieces = [pieces]
    if len(pieces) != len(ids):
        raise ValueError(
            f"convert_ids_to_tokens returned {len(pieces)} pieces for {len(ids)} ids"
        )
    return tuple(str(piece) for piece in pieces)


def _mismatch_runs(left: tuple[int, ...], right: tuple[int, ...]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, (lhs, rhs) in enumerate(zip(left, right, strict=True)):
        if lhs != rhs:
            if start is None:
                start = idx
        elif start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(left)))
    return runs


def _source_matches_marker_case(source_text: str, decoded_text: str, *, truncated: bool) -> bool:
    if _GEMMA3_MARKER not in source_text:
        return False
    normalized_source = source_text.replace(_GEMMA3_MARKER, " ")
    if truncated:
        return normalized_source.startswith(decoded_text)
    return normalized_source == decoded_text


def _maybe_normalize_marker_ids(
    kept_ids: list[int], tokenizer: Any, *, source_text: str, truncated: bool,
) -> _Gemma3Normalization | None:
    if not _is_gemma3_tokenizer(tokenizer):
        return None
    original_ids = tuple(kept_ids)
    decoded_text = to_text(original_ids, tokenizer)
    if not _source_matches_marker_case(source_text, decoded_text, truncated=truncated):
        return None
    reencoded_ids = _input_ids_tuple(tokenizer(decoded_text, padding=False))
    if reencoded_ids == original_ids:
        return None
    if len(reencoded_ids) != len(original_ids):
        return None
    runs = _mismatch_runs(original_ids, reencoded_ids)
    mismatch_count = sum(end - start for start, end in runs)
    if mismatch_count > _GEMMA3_MARKER_MAX_MISMATCHES_PER_DOC:
        return None
    original_pieces = _token_pieces(tokenizer, original_ids)
    reencoded_pieces = _token_pieces(tokenizer, reencoded_ids)
    for start, end in runs:
        original_joined = "".join(original_pieces[start:end])
        if original_joined != "".join(reencoded_pieces[start:end]):
            return None
        if _GEMMA3_MARKER not in original_joined:
            return None
    return _Gemma3Normalization(list(reencoded_ids), mismatch_count)


def normalize_gemma3_space_marker_doc(
    kept_ids: list[int],
    tokenizer: Any,
    *,
    source_text: str,
    doc_index: int,
    truncated: bool,
    normalized_docs_seen: int,
    token_mismatches_seen: int,
) -> tuple[list[int], int, int]:
    """Repair the narrow gemma-3 U+2581 case; otherwise a no-op.

    Returns ``(ids, normalized_docs_seen, token_mismatches_seen)``. Raises
    :class:`Gemma3MarkerError` when the per-shard doc / mismatch caps overflow.
    """
    if not (_is_gemma3_tokenizer(tokenizer) and _GEMMA3_MARKER in source_text):
        return kept_ids, normalized_docs_seen, token_mismatches_seen
    normalization = _maybe_normalize_marker_ids(
        kept_ids, tokenizer, source_text=source_text, truncated=truncated
    )
    if normalization is None:
        return kept_ids, normalized_docs_seen, token_mismatches_seen
    next_docs = normalized_docs_seen + 1
    next_mismatches = token_mismatches_seen + normalization.token_mismatches
    if (
        next_docs > _GEMMA3_MARKER_MAX_DOCS_PER_SHARD
        or next_mismatches > _GEMMA3_MARKER_MAX_MISMATCHES_PER_SHARD
    ):
        raise Gemma3MarkerError(doc_index)
    logger.error(
        "gemma-3 tokenizer normalized literal U+2581 source doc before recombination: "
        "doc_index={}, token_mismatches={}, docs_seen={}/{}, mismatches_seen={}/{}",
        doc_index, normalization.token_mismatches, next_docs,
        _GEMMA3_MARKER_MAX_DOCS_PER_SHARD, next_mismatches,
        _GEMMA3_MARKER_MAX_MISMATCHES_PER_SHARD,
    )
    return normalization.ids, next_docs, next_mismatches
