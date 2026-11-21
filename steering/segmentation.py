"""Token and turn-delimited segmentation strategies for feature extraction.

Splits a token sequence into segments for per-segment feature pooling.
Each segment is a (start, end) index pair into the token list.

This module also hosts the canonical cross-tokenizer end-marker window
constant (``CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS``) and a two-layer
alignment helper used to reconcile a consumer tokenizer's view of a
prefix with extraction-time segment boundaries:

* :func:`align_char_ranges_to_decoded_prefix` -- Layer-1 char-only
  primitive operating on parallel ``(char_starts, char_ends,
  decoded_len)`` lists; returns ``(new_starts, new_ends, kept_indices)``
  for callers that need to filter their own parallel arrays in lockstep.
* :func:`align_segments_to_decoded_prefix` -- Layer-2 dataclass wrapper
  over ``tuple[ContextFeatureSegment, ...]``; thin shim that decodes the
  prefix with the supplied tokenizer, extracts char ranges, delegates to
  the primitive, and rebuilds the dataclass tuple.

Production consumers include
:mod:`steering.eval.axbench_bpb.seg_map` (axbench-bpb teacher-forced
scoring) and, after the centralisation in PR
``feat-centralise-segmentation-alignment``,
:mod:`steering.conditional.encoder_decoder.full_sequence_routing`,
:mod:`steering.eval.target_segments`,
:mod:`steering.eval.multiseg_generator`, and
:mod:`steering.conditional.encoder_decoder.eval_adapter`. All sites
share the same canonical counter and window bound.
"""

import codecs
import dataclasses
import hashlib
import inspect
import itertools
import json
import os
import time
import types
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
import tiktoken
import torch as t
from beartype import beartype
from beartype.vale import Is
from loguru import logger
from tokenizers import Tokenizer as NativeTokenizer
from tokenizers.decoders import Decoder
from tokenizers.decoders import Sequence as NativeDecoderSequence
from transformers import GemmaTokenizer, PreTrainedTokenizerBase, TokenizersBackend

from steering.eval.axbench_base.profiler import record_counter
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.feature_io import MetadataKey
from steering.refined_types import (
    NonEmptyStr,
    NonEmptyStrLike,
    NonNegativeInt,
    NonNegativeIntLike,
    NonNegativeIntListLike,
    NonNegativeIntTuple,
    PositiveInt,
    PositiveIntLike,
)
from steering.segmentation_primitives import SegmentMode as SegmentMode

# The pure primitives live in the dependency-light leaf module; they are
# re-exported here so existing callers keep importing from steering.segmentation.
from steering.segmentation_primitives import (
    _segment_fixed,
    segment_turn_delimited,
)
from steering.segmentation_primitives import turn_token_starts as turn_token_starts


@beartype
def segment_mode_from_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    key: str = MetadataKey.SEGMENT_MODE,
    source: object,
) -> SegmentMode:
    """Parse segment mode metadata; an absent key means legacy fixed tokens."""
    raw = (metadata or {}).get(key, SegmentMode.fixed_tokens.value)
    try:
        return SegmentMode(str(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source} has unknown {key}={raw!r}; must be one of "
            f"{sorted(mode.value for mode in SegmentMode)}"
        ) from exc


# Canonical char-window for cross-tokenizer end-marker residue at a segment
# boundary. Used on BOTH sides of the Gemma <-> RustBPE bridge:
#
#   - ``scripts/precompute_token_boundaries.py`` (extraction side):
#       tiny-trailing-segment repair — a trailing source segment narrower
#       than this receives the last nanochat token if no token naturally
#       lands in it (end-of-turn marker residue).
#
#   - ``steering/conditional/encoder_decoder/full_sequence_routing.py``
#     (consumer side):
#       partial-coverage absorption — if the extraction tokenizer covered
#       ``<=`` this many fewer chars than the training tokenizer, assume
#       the gap is an ``<end_of_turn>`` / ``<|im_end|>`` marker and extend
#       the last segment; anything larger is a real data issue and raises.
#
# The value was unified on 20 chars (Wave 3D task 2): this matches the decoded
# length of the longest end-of-turn markers we currently support
# (`<end_of_turn>\n`, `<|im_end|>\n`, `<|assistant|>\n`, etc.). Keeping the two
# sides in lockstep eliminates a silent dead-band where the extractor would
# reject a tiny trailing segment that the consumer would silently absorb — or
# vice versa.
CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS: Literal[20] = 20
_DEFAULT_ALIGNMENT_COUNTER_NAME: NonEmptyStr = NonEmptyStr.parse(
    "segmentation.char_end_extended"
)

# Sentencepiece renders its word-boundary marker (U+2581, the "lower one eighth
# block") as a space on decode, so a source text that literally contains that
# character (e.g. SmolTalk rows discussing BPE) never round-trips byte-for-byte
# through a sentencepiece tokenizer. The substitution is width-preserving, so
# every char offset stays aligned; the round-trip check therefore compares
# both sides with the marker mapped to a space. Nothing else is normalized.
SENTENCEPIECE_WORD_BOUNDARY_MARKER: Literal["▁"] = "▁"


@beartype
def normalize_roundtrip_text(text: str) -> str:
    """Map the sentencepiece word-boundary marker to the space it decodes as.

    Applied unconditionally, whatever tokenizer produced the decode: a
    byte-level BPE decoder cannot turn a literal marker into a space, so the
    tolerance is inert there and only sentencepiece decodes ever exercise it.
    """
    return text.replace(SENTENCEPIECE_WORD_BOUNDARY_MARKER, " ")


@beartype
def assert_turn_roundtrip(
    decoded: str,
    text: str,
    *,
    bos_char_len: NonNegativeIntLike,
    boundaries: Sequence[int],
    window: NonNegativeIntLike = CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS,
) -> int | None:
    """Validate a BOS-inclusive decode against its BOS-free source prefix.

    A mismatch is accepted only in the final ``window`` decoded characters and
    strictly after every retained logical-turn boundary. Returns the decoded
    coordinate of the first tolerated mismatch, or ``None`` for an exact
    prefix round-trip. Both sides are compared through
    :func:`normalize_roundtrip_text`: a literal sentencepiece word-boundary
    marker in the source decodes as a same-width space and is not a mismatch.
    """
    if bos_char_len > len(decoded):
        raise ValueError(
            f"BOS character length {bos_char_len} exceeds decoded length {len(decoded)} "
            f"for document {text[:100]!r}"
        )

    decoded_text = normalize_roundtrip_text(decoded[bos_char_len:])
    source_prefix = normalize_roundtrip_text(text[: len(decoded_text)])
    mismatch = len(os.path.commonprefix([decoded_text, source_prefix]))
    if mismatch == len(decoded_text):
        return None

    difference_index = bos_char_len + mismatch
    delta = len(decoded) - difference_index
    if delta > window or any(boundary >= difference_index for boundary in boundaries):
        raise ValueError(
            "Turn-delimited tokenizer round-trip mismatch for document "
            f"{text[:100]!r} at decoded character index {difference_index}; "
            f"decoded_len={len(decoded)}, tail_delta={delta}, window={window}, "
            f"retained_boundaries={list(boundaries)}"
        )
    return difference_index


def _is_aligned_char_ranges(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 3
        and all(
            isinstance(items, list) and isinstance(tuple(items), NonNegativeIntTuple)
            for items in value
        )
    )


if TYPE_CHECKING:
    _AlignedCharRangesLike = tuple[list[int], list[int], list[int]]
else:
    # Avoid Beartype's randomized O(1) sampling for parameterized containers;
    # the predicate checks every returned offset and retained index.
    _AlignedCharRangesLike = Annotated[tuple, Is[_is_aligned_char_ranges]]


@beartype
def align_char_ranges_to_decoded_prefix(
    *,
    char_starts: NonNegativeIntListLike,
    char_ends: NonNegativeIntListLike,
    decoded_len: NonNegativeIntLike,
    end_marker_window: NonNegativeIntLike = CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS,
    counter_name: NonEmptyStrLike = _DEFAULT_ALIGNMENT_COUNTER_NAME,
) -> _AlignedCharRangesLike:
    """Adjust parallel char ranges so coverage matches ``decoded_len``.

    Layer-1 primitive used by:
      * :func:`align_segments_to_decoded_prefix` (Layer-2 wrapper over
        ``tuple[ContextFeatureSegment, ...]``).
      * :func:`steering.conditional.encoder_decoder.full_sequence_routing.build_full_sequence_segment_routing`
        (parallel-list shape with fids/mags lockstep filtering).
      * :func:`steering.eval.target_segments.filter_segments_to_target`
        (rebased-tuple shape; reverse direction is a no-op in production).
      * :func:`steering.eval.multiseg_generator.generate_for_document`
        per-target build inside the function body (SegmentFeatures tuple
        shape). The sibling ``build_multiseg_inputs`` does NOT call this
        helper — cross-tokenizer noise on that path is absorbed upstream by
        the overlay-precomputed token boundaries (see
        ``scripts.preprocess.precompute_token_boundaries``).

    Behaviour
    ---------

    Let ``coverage_end = char_ends[-1]`` (or ``0`` for empty inputs).

    * **Equal / empty** (``decoded_len == coverage_end`` or both empty):
        returns inputs unchanged, ``kept_indices = list(range(n))``, no
        counter emitted.
    * **Forward** (``decoded_len > coverage_end``): if ``delta =
        decoded_len - coverage_end <= end_marker_window``, extends
        ``char_ends[-1]`` to ``decoded_len`` and emits ``counter_name``
        with ``direction="forward"``, ``delta=delta``; otherwise raises
        ``ValueError``.
    * **Reverse** (``decoded_len < coverage_end``): drops indices where
        ``char_starts[i] >= decoded_len`` (fully past the decoded
        prefix), then clips the last surviving ``char_ends[-1]`` to
        ``decoded_len`` if needed. If the original ``overall_delta =
        coverage_end - decoded_len > end_marker_window``, raises
        ``ValueError``. If every index would be filtered, raises
        ``ValueError``. Emits ``counter_name`` with
        ``direction="reverse"``, ``delta=overall_delta`` — even in the
        no-clip case (filtering happened, observability captures it).

    Invariants
    ----------

    * **Filter is tail-only**: in the reverse-direction filter,
      ``kept_indices`` is always a contiguous prefix of ``range(n)`` (i.e.,
      if index ``k`` is filtered out, every index ``j > k`` is also filtered
      out). This holds because segments are assumed to tile contiguously with
      monotonically-non-decreasing ``char_starts``; once ``char_starts[k] >=
      decoded_len``, every ``char_starts[j > k] >= char_starts[k] >=
      decoded_len`` too. **This invariant is LOAD-BEARING for consumers**
      that pack parallel encoder slots from pre-alignment segments and then
      read seg_map values pointing into those slots (notably
      ``steering.conditional.encoder_decoder.eval_adapter._align_row_context_segments``,
      called from ``_build_conditioning_impl`` before ``_pack_encoder_inputs``):
      the post-alignment seg_map values stay in ``[0, len(kept_indices))``,
      which is a prefix of ``[0, N)``, so encoder slot ``k`` is semantically
      identical to the unaligned segment ``k``.
    * **char_starts are never mutated**: only ``char_ends[-1]`` may change
      (forward-extend or reverse-clip). Earlier segment boundaries are
      preserved verbatim.

    Returns:
        ``(new_char_starts, new_char_ends, kept_indices)``. The first two
        are NEW lists (caller may mutate). ``kept_indices`` is the list
        of original indices that survived filtering (in original order;
        ``list(range(n))`` when no filtering happened).

    Raises:
        BeartypeCallHintParamViolation: An argument violates its annotated
            contract, including a negative ``end_marker_window`` or empty
            ``counter_name``.
        ValueError: forward delta > window, reverse overall_delta > window,
            or reverse filter removes every range.
    """
    # Length mismatch MUST be checked before the empty short-circuit:
    # otherwise ``char_starts=[1, 2], char_ends=[]`` falls into the
    # "either empty" branch and silently returns an inconsistent
    # ``([1, 2], [], [0, 1])`` triple. Length mismatch is always a
    # programming error and must surface loudly.
    if len(char_starts) != len(char_ends):
        raise ValueError(
            f"char_starts/char_ends length mismatch: "
            f"{len(char_starts)} vs {len(char_ends)}"
        )
    # Both-empty is the only remaining "empty" case; lengths equal, no
    # work to do.
    if not char_starts:
        return list(char_starts), list(char_ends), list(range(len(char_starts)))

    n = len(char_starts)
    coverage_end = char_ends[-1]

    if decoded_len == coverage_end:
        return list(char_starts), list(char_ends), list(range(n))

    if decoded_len > coverage_end:
        delta = decoded_len - coverage_end
        if delta > end_marker_window:
            raise ValueError(
                "Forward tokenizer-mismatch delta exceeds end-marker window: "
                f"decoded_len={decoded_len}, coverage_end={coverage_end}, "
                f"delta={delta}, window={end_marker_window}. The consumer "
                "decoded materially more characters than the source reported; "
                "this is a data issue, not cross-tokenizer noise. Re-extract "
                "features or check the chat template."
            )
        record_counter(
            counter_name,
            1,
            delta=delta,
            direction="forward",
        )
        logger.debug(
            f"align_char_ranges_to_decoded_prefix: extending last range "
            f"char_end {coverage_end} -> {decoded_len} (delta={delta}, "
            f"direction=forward) for tokenizer mismatch"
        )
        new_ends = list(char_ends)
        new_ends[-1] = decoded_len
        return list(char_starts), new_ends, list(range(n))

    # decoded_len < coverage_end → reverse direction: bound check, then
    # filter + clip.
    overall_delta = coverage_end - decoded_len
    if overall_delta > end_marker_window:
        raise ValueError(
            "Reverse tokenizer-mismatch delta exceeds end-marker window: "
            f"decoded_len={decoded_len}, coverage_end={coverage_end}, "
            f"delta={overall_delta}, window={end_marker_window}. The consumer "
            "decoded materially fewer characters than the source reported; "
            "this is a data issue, not cross-tokenizer noise."
        )

    kept_indices = [i for i in range(n) if char_starts[i] < decoded_len]
    if not kept_indices:
        raise ValueError(
            "Reverse tokenizer-mismatch: every range starts past the "
            f"consumer-decoded prefix length. decoded_len={decoded_len}, "
            f"coverage_end={coverage_end}, delta={overall_delta}, "
            f"window={end_marker_window}. No range overlaps the decoded prefix."
        )

    new_starts = [char_starts[i] for i in kept_indices]
    new_ends = [char_ends[i] for i in kept_indices]
    last_end = new_ends[-1]

    if last_end > decoded_len:
        new_ends[-1] = decoded_len
        logger.debug(
            f"align_char_ranges_to_decoded_prefix: clipping last range "
            f"char_end {last_end} -> {decoded_len} (overall_delta={overall_delta}, "
            f"direction=reverse) for tokenizer mismatch"
        )
    else:
        logger.debug(
            f"align_char_ranges_to_decoded_prefix: filtered "
            f"{n - len(kept_indices)} range(s) past decoded prefix; "
            f"no clipping needed (overall_delta={overall_delta}, direction=reverse)"
        )

    # Emit overall_delta (not clip_delta) so the no-clip-reverse path is
    # observable too; dashboards aggregating on delta see the cross-tokenizer
    # mismatch magnitude, not the post-filter clip residue.
    record_counter(
        counter_name,
        1,
        delta=overall_delta,
        direction="reverse",
    )
    return new_starts, new_ends, kept_indices


@beartype
def align_segments_to_decoded_prefix(
    *,
    tokenizer: Any,
    prefix_tokens: list[int],
    segments: tuple[ContextFeatureSegment, ...],
    end_marker_window: NonNegativeIntLike = CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS,
    counter_name: NonEmptyStrLike = _DEFAULT_ALIGNMENT_COUNTER_NAME,
) -> tuple[ContextFeatureSegment, ...]:
    """Align ``segments`` to the consumer tokenizer's decoded prefix length.

    Layer-2 wrapper. Decodes ``prefix_tokens`` with ``tokenizer``,
    extracts char ranges from the dataclass tuple, delegates to
    :func:`align_char_ranges_to_decoded_prefix`, and rebuilds the
    surviving tuple via ``dataclasses.replace``.

    ``end_marker_window`` and ``counter_name`` carry the same refined scalar
    contracts as the Layer-1 primitive and are enforced by ``beartype`` before
    the empty-input short circuit. See the primitive for full semantics — in
    particular, the **filter-is-tail-only** and **char_starts-never-mutated**
    invariants documented under the primitive's ``Invariants`` section apply
    transitively to this wrapper.

    ``counter_name`` controls the observability counter emitted on every
    non-equal alignment. Defaults to ``segmentation.char_end_extended`` for
    backward compatibility; production consumers pass a site-scoped name so
    dashboards aggregating on counter name can attribute events to specific
    call sites. See ``docs/reference/architecture.md`` for the 5 site-scoped
    names in current use.

    Raises:
        BeartypeCallHintParamViolation: An argument violates its annotated
            contract, including a negative ``end_marker_window`` or empty
            ``counter_name``.
        ValueError: The delegated coverage relationship is invalid.
    """
    if not segments or not prefix_tokens:
        return segments

    offsets = token_to_char_offsets(tokenizer, prefix_tokens)
    decoded_len = offsets[-1] if offsets else 0

    new_starts, new_ends, kept_indices = align_char_ranges_to_decoded_prefix(
        char_starts=[s.char_start for s in segments],
        char_ends=[s.char_end for s in segments],
        decoded_len=decoded_len,
        end_marker_window=end_marker_window,
        counter_name=counter_name,
    )

    return tuple(
        dataclasses.replace(
            segments[orig_idx],
            char_start=new_starts[new_idx],
            char_end=new_ends[new_idx],
        )
        for new_idx, orig_idx in enumerate(kept_indices)
    )


def assign_tokens_to_segments(
    token_char_starts: t.Tensor,
    token_char_ends: t.Tensor,
    seg_char_starts: t.Tensor,
    seg_char_ends: t.Tensor,
) -> t.Tensor:
    """Assign each token to the segment with maximum character overlap.

    Uses O(n log m) searchsorted on token midpoints when segments tile
    contiguously (the common case for extraction segments). Falls back to
    O(n × m) overlap matrix for non-contiguous segments.

    Raises ValueError if any token has zero overlap with all segments.

    Args:
        token_char_starts: (n_tokens,) int — character start of each token.
        token_char_ends: (n_tokens,) int — character end of each token.
        seg_char_starts: (n_segs,) int — character start of each segment.
        seg_char_ends: (n_segs,) int — character end of each segment.

    Returns:
        (n_tokens,) int64 tensor of segment indices.
    """
    if len(token_char_starts) == 0:
        return t.tensor([], dtype=t.int64)

    n_segs = len(seg_char_starts)

    # Zero-width tokens (start == end) arise from byte-merge artifacts
    # in the nanochat tokenizer (U+FFFD replacement characters).  They
    # have no character extent and therefore zero overlap with any
    # segment.  Handle them separately: assign each to the segment
    # whose range contains its position (or the last segment if at the
    # boundary).
    zero_width = token_char_starts == token_char_ends

    # Check if segments tile contiguously: each seg_start[i+1] == seg_end[i]
    is_tiling = n_segs == 1 or (seg_char_starts[1:] == seg_char_ends[:-1]).all()

    if is_tiling:
        # B-2 (slice-02 audit, P1): when the contiguous-tiling fast path
        # short-circuits, ``t.searchsorted`` requires a sorted needle to
        # produce correct results.  Production callers (extraction → CSR
        # shard reads) always emit non-decreasing seg_char_ends; this guard
        # makes the contract explicit so a future caller that skips the
        # extraction pipeline raises loudly instead of silently misbehaving.
        if n_segs > 1 and not bool((seg_char_ends[1:] >= seg_char_ends[:-1]).all()):
            bad = int(
                (seg_char_ends[1:] < seg_char_ends[:-1])
                .nonzero(as_tuple=True)[0][0]
                .item()
            )
            raise ValueError(
                "seg_char_ends must be monotonically non-decreasing for the "
                "contiguous-tiling fast path; segment "
                f"{bad} ends at {int(seg_char_ends[bad].item())} but segment "
                f"{bad + 1} ends at {int(seg_char_ends[bad + 1].item())}"
            )
        # Fast path: O(n log m) searchsorted on token midpoints.
        # The segment containing a token's midpoint has the maximum overlap.
        # Use side="left" so that when a midpoint falls exactly on a segment
        # boundary, the token is assigned to the earlier segment (matching
        # argmax tie-breaking behavior where the first segment wins).
        midpoints = (token_char_starts + token_char_ends + 1) // 2  # ceil division
        assignments = t.searchsorted(seg_char_ends, midpoints, side="left")
        assignments = assignments.clamp(max=n_segs - 1)

        # Validate: every non-zero-width token must fall within the segment range.
        # Zero-width tokens are exempt — they have no extent so "before start"
        # and "after end" checks based on character ranges don't apply.
        before_start = token_char_ends <= seg_char_starts[0]
        after_end = token_char_starts >= seg_char_ends[-1]
        unassigned = (before_start | after_end) & ~zero_width
    else:
        # Slow path: O(n × m) overlap matrix for non-contiguous segments
        lo = t.max(token_char_starts.unsqueeze(1), seg_char_starts.unsqueeze(0))
        hi = t.min(token_char_ends.unsqueeze(1), seg_char_ends.unsqueeze(0))
        overlap = (hi - lo).clamp(min=0)
        max_overlap = overlap.max(dim=1).values
        unassigned = (max_overlap == 0) & ~zero_width
        assignments = overlap.argmax(dim=1).to(t.int64)

    # For zero-width tokens, searchsorted may have assigned them correctly
    # (fast path) or argmax may have defaulted to 0 (slow path).
    # Re-assign explicitly: use the segment containing their position.
    if zero_width.any():
        zw_positions = token_char_starts[zero_width]
        zw_assignments = t.searchsorted(seg_char_ends, zw_positions, side="left")
        zw_assignments = zw_assignments.clamp(max=n_segs - 1)
        assignments[zero_width] = zw_assignments

    if unassigned.any():
        bad_indices = unassigned.nonzero(as_tuple=True)[0]
        first_bad = int(bad_indices[0].item())
        raise ValueError(
            f"{int(unassigned.sum().item())} token(s) have zero overlap with all segments. "
            f"First unassigned token {first_bad}: chars [{token_char_starts[first_bad]}, {token_char_ends[first_bad]}), "
            f"segment char range: [{seg_char_starts[0]}, {seg_char_ends[-1]})"
        )

    return assignments


def strict_segment_bounds_for_assignment(
    *,
    seg_char_starts: t.Tensor,
    seg_char_ends: t.Tensor,
    token_starts: t.Tensor,
    token_ends: t.Tensor,
    allow_zero_width_segment_idx: int | None = None,
) -> tuple[t.Tensor, t.Tensor]:
    """Validate segment char ranges before token assignment.

    Segment boundaries come from source-model extraction and token boundaries
    come from the downstream tokenizer. The only implicit adjustment allowed is
    folding one exact leading BOS token span into the first segment. Gaps,
    overlaps, non-positive spans, and other leading tokens raise loudly.
    """

    seg_char_starts = t.as_tensor(seg_char_starts, dtype=t.int64)
    seg_char_ends = t.as_tensor(seg_char_ends, dtype=t.int64)
    token_starts = t.as_tensor(token_starts, dtype=t.int64)
    token_ends = t.as_tensor(token_ends, dtype=t.int64)

    if len(seg_char_starts) == 0:
        raise ValueError("segments must contain at least one segment")
    if len(seg_char_starts) != len(seg_char_ends):
        raise ValueError(
            "segment char starts/ends length mismatch: "
            f"{len(seg_char_starts)} vs {len(seg_char_ends)}"
        )

    non_positive = seg_char_ends <= seg_char_starts
    if allow_zero_width_segment_idx is not None:
        allowed = t.zeros_like(non_positive)
        if 0 <= allow_zero_width_segment_idx < len(allowed):
            allowed[allow_zero_width_segment_idx] = True
        non_positive = non_positive & ~(
            allowed & (seg_char_ends == seg_char_starts)
        )
    if non_positive.any():
        bad = int(non_positive.nonzero(as_tuple=True)[0][0].item())
        raise ValueError(
            f"Segment {bad} has non-positive char span "
            f"[{seg_char_starts[bad]}, {seg_char_ends[bad]})"
        )

    if len(seg_char_starts) > 1 and not (
        seg_char_starts[1:] == seg_char_ends[:-1]
    ).all():
        bad = int((seg_char_starts[1:] != seg_char_ends[:-1]).nonzero(as_tuple=True)[0][0].item())
        raise ValueError(
            f"Segment char ranges must tile contiguously; segment {bad} ends at "
            f"{seg_char_ends[bad]} but segment {bad + 1} starts at "
            f"{seg_char_starts[bad + 1]}"
        )

    if len(token_starts) == 0:
        return seg_char_starts, seg_char_ends
    if len(token_starts) != len(token_ends):
        raise ValueError(
            "token char starts/ends length mismatch: "
            f"{len(token_starts)} vs {len(token_ends)}"
        )

    first_seg_start = seg_char_starts[0]
    leading = token_ends <= first_seg_start
    if leading.any():
        n_leading = int(leading.sum().item())
        is_exact_bos = (
            n_leading == 1
            and bool(leading[0].item())
            and int(token_starts[0].item()) == 0
            and int(token_ends[0].item()) == int(first_seg_start.item())
        )
        if not is_exact_bos:
            first_bad = int(leading.nonzero(as_tuple=True)[0][0].item())
            raise ValueError(
                f"{n_leading} token(s) precede the first segment before token/char "
                f"assignment. First leading token {first_bad}: chars "
                f"[{token_starts[first_bad]}, {token_ends[first_bad]}), "
                f"first segment starts at {first_seg_start}"
            )
        seg_char_starts = seg_char_starts.clone()
        seg_char_starts[0] = token_starts[0]

    return seg_char_starts, seg_char_ends


# Module-level counters for profiling path distribution
_path_stats: dict[str, list[float]] = {"fast": [], "slow": [], "small_n": []}


def get_path_stats() -> dict[str, dict]:
    """Return per-path call counts and mean times (ms). Resets counters."""
    stats = {}
    for path, times in _path_stats.items():
        if times:
            stats[path] = {
                "count": len(times),
                "mean_ms": sum(times) / len(times),
                "total_ms": sum(times),
            }
        else:
            stats[path] = {"count": 0, "mean_ms": 0.0, "total_ms": 0.0}
    for v in _path_stats.values():
        v.clear()
    return stats


_HF_OFFSET_DECODE = PreTrainedTokenizerBase.decode


def _decode_offset_ids(decode, token_ids):
    # HF converts this object array to the same fresh list without its numeric scan.
    if (
        type(decode) is types.MethodType
        and decode.__func__ is _HF_OFFSET_DECODE
        and type(token_ids) is list
        and len(token_ids) > 16
        and all(type(tid) is int for tid in token_ids)
    ):
        return decode(np.asarray(token_ids, dtype=object))
    return decode(token_ids)


_GEMMA_OFFSET_METHODS = (
    PreTrainedTokenizerBase.decode,
    TokenizersBackend._decode,
    TokenizersBackend.convert_ids_to_tokens,
)
_GEMMA_OFFSET_DEFAULTS = ((False,), (False, None), (False,))
_GEMMA_OFFSET_BACKEND = TokenizersBackend.backend_tokenizer
_GEMMA_OFFSET_DECODER_CONFIG = {
    "type": "Sequence",
    "decoders": [
        {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
        {"type": "ByteFallback"},
        {"type": "Fuse"},
    ],
}


def _gemma_prefix_lengths(tokenizer, token_ids):
    # Generic objects take the original route before any extra method/state lookup.
    if type(tokenizer) is not GemmaTokenizer or type(token_ids) is not list or not token_ids:
        return None
    if type(tokenizer).__getattribute__ is not object.__getattribute__:
        return None
    methods = []
    for name, original, expected in zip(  # noqa: B905 - fixed three-method schema
        ("decode", "_decode", "convert_ids_to_tokens"),
        _GEMMA_OFFSET_METHODS,
        _GEMMA_OFFSET_DEFAULTS,
    ):
        if inspect.getattr_static(tokenizer, name, None) is not original:
            return None
        defaults = original.__defaults__
        if (
            type(defaults) is not tuple
            or len(defaults) != len(expected)
            or any(
                value is not wanted
                for value, wanted in zip(defaults, expected)  # noqa: B905 - lengths checked above
            )
        ):
            return None
        methods.append(original.__get__(tokenizer, type(tokenizer)))
    if inspect.getattr_static(tokenizer, "clean_up_tokenization_spaces", None) is not False or not all(
        type(tid) is int and 0 <= tid <= 0xFFFFFFFF for tid in token_ids
    ):
        return None
    if inspect.getattr_static(tokenizer, "backend_tokenizer", None) is not _GEMMA_OFFSET_BACKEND:
        return None
    backend = inspect.getattr_static(tokenizer, "_tokenizer", None)
    if type(backend) is not NativeTokenizer or tokenizer.backend_tokenizer is not backend:
        return None
    decoder = backend.decoder
    if type(decoder) is not NativeDecoderSequence:
        return None
    try:
        if json.loads(Decoder.__getstate__(decoder)) != _GEMMA_OFFSET_DECODER_CONFIG:
            return None
        pieces = methods[2](token_ids, skip_special_tokens=False)
    except Exception:
        return None
    if type(pieces) is not list or not all(type(piece) is str for piece in pieces):
        return None
    # Native ByteFallback can replace the entire pending run after a later invalid byte.
    # Let that decoder own UTF-8 behavior; ordinary pieces end the run.
    committed = pending_length = 0
    pending, lengths = [], []
    for piece in pieces:
        if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
            pending.append(piece)
            pending_length = len(Decoder.decode(decoder, pending))
        else:
            committed += pending_length + len(piece)
            pending, pending_length = [], 0
        lengths.append(committed + pending_length)
    return lengths


def _token_to_char_offsets_progressive(tokenizer, token_ids: list[int]) -> list[int]:
    """O(n²) progressive prefix decode with merge-aware offset correction.

    When byte-fallback tokens complete a multi-byte character, the decoded
    prefix shrinks (e.g., 3 replacement chars → 1 real char).  Instead of
    clamping with max() (which inflates offsets[-1]), we detect the merge and
    retroactively fix the preceding offsets for the byte-fallback group.

    Guarantees:
        offsets[0] == 0
        offsets[-1] == len(tokenizer.decode(token_ids))
        offsets[i+1] >= offsets[i]  (non-decreasing)

    Zero-width tokens (offsets[i] == offsets[i+1]) occur for intermediate
    byte-fallback tokens within a multi-byte group.  Callers that need
    non-zero width must expand these upstream (e.g. _boundaries_from_offsets).
    """
    native_lengths = _gemma_prefix_lengths(tokenizer, token_ids)
    offsets = [0]
    for i in range(len(token_ids)):
        if native_lengths is None:
            prefix = _decode_offset_ids(tokenizer.decode, token_ids[: i + 1])
            new_len = len(prefix)
        else:
            new_len = native_lengths[i]

        if new_len < offsets[-1]:
            # Merge detected: byte-fallback group completed a multi-byte char.
            # The preceding offsets were inflated by replacement characters.
            # Walk back and reset them to the base offset (just before the
            # merged character).  A completed multi-byte merge replaces N
            # replacement chars with 1 real char, so the char position just
            # before the merged char is at new_len - 1.
            base = max(0, new_len - 1)
            for j in range(len(offsets) - 1, 0, -1):
                if offsets[j] <= base:
                    break
                offsets[j] = base

        offsets.append(new_len)
    return offsets


_UTF8_INCREMENTAL_DECODER_FACTORY = codecs.getincrementaldecoder("utf-8")


def _tiktoken_token_bytes(tokenizer, token_ids: list[int]) -> list[bytes] | None:
    """Per-token byte payloads when tokenizer is tiktoken-backed, else None.

    isinstance (not hasattr): MagicMock tokenizers auto-create truthy ``enc``
    attributes, and HuggingFace/SentencePiece wrappers have context-dependent
    decode semantics the byte path must never claim. For a tiktoken Encoding,
    ``decode(ids) == b"".join(decode_tokens_bytes(ids)).decode("utf-8",
    errors="replace")`` by construction (pinned by
    tests/steering/test_token_offsets_tiktoken_bytes.py).
    """
    enc = getattr(tokenizer, "enc", None)
    if not isinstance(enc, tiktoken.Encoding):
        return None
    return enc.decode_tokens_bytes(token_ids)


class _IncrementalReplaceLength:
    """Incremental ``len(b.decode("utf-8", errors="replace"))`` over a growing stream.

    ``feed`` returns the exact one-shot replace-decode length of all bytes fed so
    far, in O(len(chunk)) per call: CPython's own incremental UTF-8 codec emits
    the finalized characters, and the buffered trailing incomplete sequence
    (``getstate()[0]``, at most 3 bytes, always a valid prefix of a multi-byte
    character) is decoded separately to account for the replacement character(s)
    a one-shot decode would render for it. Differentially fuzzed against one-shot
    decoding in test_token_offsets_tiktoken_bytes.py.
    """

    def __init__(self) -> None:
        self._decoder = _UTF8_INCREMENTAL_DECODER_FACTORY(errors="replace")
        self._emitted = 0

    def feed(self, data: bytes) -> int:
        self._emitted += len(self._decoder.decode(data, final=False))
        pending = self._decoder.getstate()[0]
        if pending:
            return self._emitted + len(pending.decode("utf-8", errors="replace"))
        return self._emitted


def _token_to_char_offsets_progressive_bytes(token_bytes: list[bytes]) -> list[int]:
    """Byte-level twin of ``_token_to_char_offsets_progressive`` for tiktoken
    tokenizers, where ``decode(prefix)`` is exactly ``concat(prefix
    bytes).decode("utf-8", errors="replace")``. Each O(prefix) full decode
    becomes an O(token-bytes) incremental step; the merge walk-back block is
    copied verbatim from the reference (it cannot fire under UTF-8 replace
    semantics — a truncated valid prefix already renders as a single
    replacement char, so lengths never shrink — but keeping it preserves
    structural equivalence rather than relying on that proof).
    """
    offsets = [0]
    lengths = _IncrementalReplaceLength()
    for data in token_bytes:
        new_len = lengths.feed(data)

        if new_len < offsets[-1]:
            base = max(0, new_len - 1)
            for j in range(len(offsets) - 1, 0, -1):
                if offsets[j] <= base:
                    break
                offsets[j] = base

        offsets.append(new_len)
    return offsets


def _token_to_char_offsets_chunked_bytes(
    token_bytes: list[bytes],
    per_token_texts: list[str],
) -> list[int]:
    """Byte-level twin of ``_token_to_char_offsets_chunked``: identical chunking,
    identical optimistic per-token check, identical bail-to-progressive dispatch —
    only the per-chunk-boundary prefix decode is replaced by the incremental
    length engine (O(chunk bytes) instead of O(prefix bytes) per boundary).

    Reachability note (mutation-campaign finding, #1269): for a pure
    byte-concat decoder a chunk is clean iff no cross-token UTF-8 effect lands
    in it, so "whole-doc fast check failed but every chunk clean" cannot
    happen — every call routed here by ``token_to_char_offsets`` bails to
    progressive, and the clean-chunk fill below only decides the result for
    direct (unit-level) callers. It is kept for structural parity with the
    generic twin, whose fill IS reachable (SentencePiece decode can expand as
    well as shrink), and is pinned by a direct all-clean unit test.
    """
    n = len(token_bytes)
    chunk_size = _HYBRID_CHUNK_SIZE
    per_tok_lens = [len(s) for s in per_token_texts]

    offsets = [0] * (n + 1)
    chunk_start = 0
    lengths = _IncrementalReplaceLength()
    actual_prefix_len = 0

    while chunk_start < n:
        chunk_end = min(chunk_start + chunk_size, n)

        optimistic_end = offsets[chunk_start]
        for i in range(chunk_start, chunk_end):
            optimistic_end += per_tok_lens[i]

        for i in range(chunk_start, chunk_end):
            actual_prefix_len = lengths.feed(token_bytes[i])

        if actual_prefix_len != optimistic_end:
            return _token_to_char_offsets_progressive_bytes(token_bytes)

        pos = offsets[chunk_start]
        for i in range(chunk_start, chunk_end):
            pos += per_tok_lens[i]
            offsets[i + 1] = pos

        chunk_start = chunk_end

    return offsets


_SINGLETON_DECODE_CACHE_MAX_ENTRIES = 16384


def token_to_char_offsets(
    tokenizer, token_ids: list[int], *, _singleton_cache: dict[int, str] | None = None,
) -> list[int]:
    """Convert token boundaries to character offsets.

    Uses an O(n) fast path when per-token decodes concatenate to the full
    decoded text (true for ASCII-heavy byte-level BPE). For inputs where the
    fast path fails on the whole-sequence check, hands off to a chunked
    fast path that processes the document chunk-by-chunk; the first chunk
    that diverges triggers a fall-back to the O(n²) progressive prefix
    decode for the whole document. The progressive path handles
    context-dependent decoding (multi-byte UTF-8 merges, SentencePiece
    whitespace, byte fallback) correctly; an earlier per-chunk slow
    fallback was removed because its merge-correction walk-back could
    cross chunk boundaries and produce inconsistent offsets — see
    ``_token_to_char_offsets_chunked`` for the post-mortem.

    Returns a list of length len(token_ids) + 1 where offsets[i] is the character
    position where token i starts, and offsets[-1] is the total character count.

    Tiktoken-backed tokenizers use their exact per-token byte payloads and an
    incremental UTF-8 replacement decoder; the differential coverage is pinned
    in ``tests/steering/test_token_offsets_tiktoken_bytes.py``.

    The private singleton cache belongs to one derived-shard build with an
    unchanged tokenizer and unchanged default decode settings throughout.
    """
    if not token_ids:
        return [0]

    token_bytes = _tiktoken_token_bytes(tokenizer, token_ids)
    n = len(token_ids)
    if n <= 16:
        t0 = time.perf_counter()
        if token_bytes is not None:
            result = _token_to_char_offsets_progressive_bytes(token_bytes)
        else:
            result = _token_to_char_offsets_progressive(tokenizer, token_ids)
        _path_stats["small_n"].append((time.perf_counter() - t0) * 1e3)
        return result

    # O(n): decode each token individually + full sequence once
    t0 = time.perf_counter()
    if token_bytes is not None:
        per_token_texts = [data.decode("utf-8", errors="replace") for data in token_bytes]
        full_text = b"".join(token_bytes).decode("utf-8", errors="replace")
    else:
        # Preserve first-appearance order, including the first invalid ID.
        unique_ids = list(dict.fromkeys(token_ids))
        if (
            isinstance(tokenizer, PreTrainedTokenizerBase)
            and getattr(tokenizer.decode, "__func__", None) is PreTrainedTokenizerBase.decode
            and getattr(tokenizer.batch_decode, "__func__", None)
            is PreTrainedTokenizerBase.batch_decode
        ):
            if (
                _singleton_cache is not None
                and len(unique_ids) <= _SINGLETON_DECODE_CACHE_MAX_ENTRIES
                and getattr(tokenizer._decode, "__func__", None) is TokenizersBackend._decode
            ):
                singleton_texts = _singleton_cache
                missing_ids = [tid for tid in unique_ids if tid not in singleton_texts]
                if len(singleton_texts) + len(missing_ids) > _SINGLETON_DECODE_CACHE_MAX_ENTRIES:
                    singleton_texts.clear()
                    missing_ids = unique_ids
            else:
                singleton_texts = {}
                missing_ids = unique_ids
            if missing_ids:
                # Object arrays retain Python integer IDs and public decode defaults.
                rows = np.asarray(missing_ids, dtype=object).reshape(-1, 1)
                singleton_texts.update(zip(missing_ids, tokenizer.batch_decode(rows), strict=True))
        else:
            singleton_texts = {tid: tokenizer.decode([tid]) for tid in unique_ids}
        per_token_texts = [singleton_texts[tid] for tid in token_ids]
        full_text = _decode_offset_ids(tokenizer.decode, token_ids)

    # String comparison catches all context-dependent decode effects:
    # - Multi-byte UTF-8 merges (replacement chars vs actual chars)
    # - SentencePiece leading space stripping vs preservation
    # - Byte fallback hex strings vs merged characters
    if "".join(per_token_texts) == full_text:
        offsets = [0, *itertools.accumulate(len(text) for text in per_token_texts)]
        _path_stats["fast"].append((time.perf_counter() - t0) * 1e3)
        return offsets

    # Hybrid path: chunk-by-chunk fast check. Each clean chunk uses
    # cumulative per-token lengths (O(n) total). The first chunk that
    # diverges triggers a fall-back to the full progressive O(n²) path
    # for the whole document — see ``_token_to_char_offsets_chunked``
    # for why we do not try to slow-process individual chunks any more.
    if token_bytes is not None:
        result = _token_to_char_offsets_chunked_bytes(token_bytes, per_token_texts)
    else:
        result = _token_to_char_offsets_chunked(
            tokenizer, token_ids, per_token_texts,
        )
    _path_stats["slow"].append((time.perf_counter() - t0) * 1e3)
    return result


# Chunk size for the hybrid path. Each chunk costs one prefix decode
# call for the boundary check. Smaller chunks localize divergent regions
# more precisely (cheaper to detect divergence) but add more boundary
# checks. 32 is a good balance: ~30 boundary checks for a 1000-token doc.
# When any chunk diverges, the function bails to the full progressive
# path for the whole document, so this size only affects the cost of
# detecting divergence, not the cost of recovering from it.
_HYBRID_CHUNK_SIZE = 32


def _token_to_char_offsets_chunked(
    tokenizer,
    token_ids: list[int],
    per_token_texts: list[str],
) -> list[int]:
    """Hybrid path: chunk the sequence and fast-check each chunk.

    For each chunk, decodes the prefix up to the chunk boundary and checks
    if the actual length matches the running optimistic offset from per-token
    lengths. Chunks that match use fast per-token offsets. The first chunk
    that diverges triggers a fall-back to the full progressive O(n²) path
    for the whole document — guaranteed correct, no chunk-boundary corner
    cases. Most pretraining text has no divergent chunks so the fast O(n)
    path runs to completion.

    The previous slow per-chunk fallback was buggy: a walk-back triggered
    inside a divergent chunk could cross into a previously fast-processed
    chunk and clamp its offsets to a low ``base`` value, producing a long
    flat region whose offsets were inconsistent with what the progressive
    path would produce. This manifested as zero-width source segments in
    feature shards — observed in production on shard_00355/doc_id=15373,
    a Wikipedia article containing an Old Italic inscription whose 32+
    consecutive byte-fallback Gemma tokens collapsed onto a single char
    position. By UTF-8 design, 32 bytes always span at least 8 characters,
    so a 32-token zero-width range is impossible — proving the chunked
    path was wrong rather than the data being pathological.
    """
    n = len(token_ids)
    chunk_size = _HYBRID_CHUNK_SIZE
    per_tok_lens = [len(s) for s in per_token_texts]

    offsets = [0] * (n + 1)
    chunk_start = 0

    while chunk_start < n:
        chunk_end = min(chunk_start + chunk_size, n)

        # Optimistic: assume per-token lengths are correct for this chunk
        optimistic_end = offsets[chunk_start]
        for i in range(chunk_start, chunk_end):
            optimistic_end += per_tok_lens[i]

        # Validate: does the actual prefix decode length match?
        actual_prefix_len = len(_decode_offset_ids(tokenizer.decode, token_ids[:chunk_end]))

        if actual_prefix_len != optimistic_end:
            # Any divergence ⇒ fall back to the full progressive path for
            # the whole document. See the docstring for why we don't try
            # to slow-process just this chunk.
            return _token_to_char_offsets_progressive(tokenizer, token_ids)

        # Fast: per-token offsets are correct for this chunk
        pos = offsets[chunk_start]
        for i in range(chunk_start, chunk_end):
            pos += per_tok_lens[i]
            offsets[i + 1] = pos

        chunk_start = chunk_end

    return offsets


@beartype
def segment_tokens(
    token_ids: list[int],
    mode: SegmentMode = SegmentMode.fixed_tokens,
    segment_size: PositiveIntLike = 128,
    token_offset: NonNegativeIntLike = 0,
    *,
    turn_starts: Sequence[int] | None = None,
) -> list[tuple[int, int]]:
    """Segment a list of token IDs into (start, end) index pairs.

    Args:
        token_ids: List of token IDs (already truncated to tail).
        mode: Segmentation strategy.
        segment_size: Number of tokens per segment (for fixed_tokens mode).
        token_offset: Start segmentation at this token index instead of 0.
        turn_starts: Token starts for logical turns in turn-delimited mode.

    Returns:
        List of (start, end) tuples where start is inclusive and end is exclusive.

    Raises:
        BeartypeCallHintParamViolation: An argument violates its annotated
            contract, including a non-positive ``segment_size`` or negative
            ``token_offset``.
        AssertionError: ``token_ids`` is empty or ``token_offset`` is not less
            than the token count.
    """
    n = len(token_ids)
    assert n > 0, "segment_tokens called with empty token list"
    if mode == SegmentMode.turn_delimited_fixed_tokens and token_offset != 0:
        raise ValueError("turn_delimited_fixed_tokens segmentation requires token_offset == 0")
    assert token_offset < n, f"token_offset={token_offset} out of range for {n} tokens"

    if mode == SegmentMode.fixed_tokens:
        if turn_starts is not None:
            raise ValueError("fixed_tokens segmentation does not accept turn_starts")
        return _segment_fixed(
            n,
            PositiveInt.parse(segment_size),
            NonNegativeInt.parse(token_offset),
        )

    if turn_starts is None:
        raise ValueError("turn_delimited_fixed_tokens segmentation requires turn_starts")
    return segment_turn_delimited(turn_starts, n, segment_size)


def truncate_to_training_window(
    text: str, tokenizer, context_window: int, sae_max_context: int | None
) -> tuple[list[int], int]:
    """Tokenize + head-truncate exactly as feature extraction does.

    Returns (kept_ids, n_full_tokens). kept_ids is BOS-inclusive when the
    tokenizer prepends BOS; empty text -> ([], 0) without consulting the
    tokenizer. limit = min(context_window, sae_max_context); a None
    sae_max_context degrades to context_window.
    """
    if not text:
        return ([], 0)
    encoding = tokenizer(text, padding=False)
    ids = encoding["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    n_full = len(ids)
    limit = context_window if sae_max_context is None else min(context_window, sae_max_context)
    if n_full <= limit:
        return (ids, n_full)
    return (ids[:limit], n_full)


def batch_truncate_to_training_window(
    texts: list[str], tokenizer, context_window: int, sae_max_context: int | None,
    *, chunk_size: int = 2048,
) -> list[tuple[list[int], int]]:
    """Batched, BIT-IDENTICAL equivalent of per-text truncate_to_training_window."""
    limit = context_window if sae_max_context is None else min(context_window, sae_max_context)
    out: list[tuple[list[int], int]] = []
    for start in range(0, len(texts), chunk_size):
        chunk = texts[start : start + chunk_size]
        nonempty = [txt for txt in chunk if txt]
        encoded = tokenizer(nonempty, padding=False)["input_ids"] if nonempty else []
        enc_iter = iter(encoded)
        for txt in chunk:
            if not txt:
                out.append(([], 0))
                continue
            ids = next(enc_iter)
            n_full = len(ids)
            out.append((ids if n_full <= limit else ids[:limit], n_full))
    return out


@beartype
def segment_document(
    kept_ids: list[int],
    tokenizer,
    segment_mode: SegmentMode,
    segment_size: PositiveInt,
    *,
    turn_starts: Sequence[int] | None = None,
    char_offsets: Sequence[int] | None = None,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Return segments and character bounds for one non-empty token sequence.

    The exact per-doc body of extract_features._precompute_batch_cpu_work.
    Assumes kept_ids is non-empty (segment_tokens asserts otherwise); callers
    that may pass empty docs must guard first. ``turn_starts`` is forwarded to
    ``segment_tokens`` for turn-delimited segmentation. ``char_offsets`` may
    supply the caller's already-computed token-to-character boundary table.
    """
    if char_offsets is not None and len(char_offsets) != len(kept_ids) + 1:
        raise ValueError(
            "char_offsets must contain one boundary per token plus the final "
            f"end boundary; got {len(char_offsets)} offsets for {len(kept_ids)} tokens"
        )
    segments = segment_tokens(
        kept_ids,
        segment_mode,
        segment_size,
        turn_starts=turn_starts,
    )
    resolved_char_offsets = (
        token_to_char_offsets(tokenizer, kept_ids)
        if char_offsets is None
        else char_offsets
    )
    char_starts = [resolved_char_offsets[s] for s, _ in segments]
    char_ends = [resolved_char_offsets[e] for _, e in segments]
    return segments, char_starts, char_ends


@beartype
def turn_segments_for_document(
    kept_ids: list[int],
    tokenizer,
    text: str,
    turn_char_starts: Sequence[int],
    segment_size: PositiveInt,
) -> tuple[list[tuple[int, int]], list[int], list[int], int | None]:
    """Map source turn boundaries to one document's extraction geometry."""
    bos_token_id = resolve_bos_token_id(tokenizer)
    if not kept_ids or kept_ids[0] != bos_token_id:
        actual = kept_ids[0] if kept_ids else None
        raise ValueError(
            "turn-delimited boundary mapping assumes a leading BOS token; "
            f"expected {bos_token_id}, got {actual} for document {text[:100]!r}"
        )
    char_offsets = token_to_char_offsets(tokenizer, kept_ids)
    decoded = tokenizer.decode(kept_ids)
    bos_char_len = char_offsets[1]
    try:
        turn_starts = turn_token_starts(
            char_offsets,
            turn_char_starts,
            bos_char_len=bos_char_len,
        )
    except ValueError as exc:
        raise ValueError(f"{exc} for document {text[:100]!r}") from exc
    retained_boundaries = [
        boundary + bos_char_len
        for boundary in turn_char_starts[1 : len(turn_starts)]
    ]
    difference_index = assert_turn_roundtrip(
        decoded,
        text,
        bos_char_len=bos_char_len,
        boundaries=retained_boundaries,
    )
    segments, char_starts, char_ends = segment_document(
        kept_ids,
        tokenizer,
        SegmentMode.turn_delimited_fixed_tokens,
        segment_size,
        turn_starts=turn_starts,
        char_offsets=char_offsets,
    )
    tail_drift = None if difference_index is None else len(decoded) - difference_index
    return segments, char_starts, char_ends, tail_drift


# Canonical test strings for tokenizer checksum — cover ASCII, UTF-8, punctuation, numbers
_CHECKSUM_TEST_STRINGS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "0123456789",
    "café résumé naïve",
    "日本語テスト",
    "emoji: 🎉🔥",
    '!@#$%^&*()_+-=[]{}|;:\'",.<>?/',
    "a" * 256,
    "\n\t\r",
    "混合 mixed テキスト text 123",
]


def resolve_bos_token_id(tokenizer) -> int | None:
    """Return the tokenizer's BOS id if discoverable.

    Probes the two contracts seen in the codebase: HF-style ``.bos_token_id``
    attribute (Gemma/RustBPE-via-HF) and the ``get_bos_token_id()`` method
    (``tests/cross_tokenizer_helpers.SimpleTokenizer``). Returns ``None`` if
    neither is available — the checksum still hashes the canonical strings
    so it remains useful, just without the BOS-id dimension.

    Both probe paths catch ``Exception`` symmetrically: HF tokenizers
    occasionally expose ``bos_token_id`` as a property (lazy I/O on
    config files); if such a property raises, we fall through to the
    method probe rather than propagating. ``int(...)`` failures (e.g.
    a non-coercible value) are also caught — better to return None
    and let the caller raise its own domain error than crash deep
    inside this helper.
    """
    try:
        bos = getattr(tokenizer, "bos_token_id", None)
        if bos is not None:
            return int(bos)
    except Exception:
        pass
    getter = getattr(tokenizer, "get_bos_token_id", None)
    if callable(getter):
        try:
            value = getter()
        except Exception:
            return None
        if value is not None:
            try:
                return int(value)
            except Exception:
                return None
    return None


def special_token_ids(tokenizer, bos_id: int | None = None) -> frozenset[int]:
    """All special-token ids that text can encode to under
    ``add_special_tokens=False`` — the shared special-id detection set.

    The union of the named ``bos``/``eos``/``pad`` attrs, the tokenizer's
    ``all_special_ids``, and — when the caller supplies it — ``bos_id`` (which the
    SAE-forward pooling prepends and which a method-probed BOS may omit from
    ``all_special_ids``).  Never narrows on a minimal tokenizer — the union only
    ever grows the set.  ``bos_id`` is optional: the BOS-prepending SAE-forward
    sites pass their resolved BOS to catch a method-probed one; a caller that
    only inspects text for specials (e.g. the LLM-output check in
    ``response_parser``) can omit it.

    Shared by ``extraction.warn_on_special_id_pollution`` and ``feature_diffmean``
    (both WARN on, but tolerate, an in-data special-token string), and by
    ``continuation_protocol.parse_stage1_continuation_response`` (which RAISES on
    a special in LLM output), so their detection sets cannot drift apart.
    (``_stage0c_internals._tokenizer_stop_token_ids`` is a deliberately separate
    function: it is the generation-STOP set, a distinct concern.)
    """
    ids: set[int] = set()
    if bos_id is not None:
        ids.add(int(bos_id))
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int):
            ids.add(value)
    for value in getattr(tokenizer, "all_special_ids", None) or ():
        if isinstance(value, int):
            ids.add(value)
    return frozenset(ids)


def compute_tokenizer_checksum(tokenizer) -> str:
    """Compute a SHA-256 checksum capturing a tokenizer's vocabulary and merge rules.

    Encodes a fixed canonical set of test strings and hashes the resulting token
    IDs. Two tokenizers produce the same checksum iff they encode all test strings
    identically AND share the same BOS id.

    The BOS id is mixed in so a tokenizer swap that only flips the BOS id
    (e.g. Gemma-IT vs Gemma-base) flips the checksum — slice-02 audit B-12.
    Tokenizers without a discoverable BOS id contribute a sentinel ``"none"``
    field.
    """
    h = hashlib.sha256()
    for text in _CHECKSUM_TEST_STRINGS:
        token_ids = tokenizer.encode(text)
        h.update(",".join(str(tid) for tid in token_ids).encode("utf-8"))
        h.update(b"|")
    bos = resolve_bos_token_id(tokenizer)
    h.update(b"bos=")
    h.update((str(bos) if bos is not None else "none").encode("utf-8"))
    return h.hexdigest()


def compute_context_token_segment_assignments(
    *,
    tokenizer: Any,
    context_tokens: list[int],
    context_segments: "tuple[ContextFeatureSegment, ...]",
) -> t.Tensor:
    """Assign each context token to its segment slot via char-level overlap.

    Computes character offsets for ``context_tokens`` against ``context_segments``
    using :func:`token_to_char_offsets`, validates segment bounds via
    :func:`strict_segment_bounds_for_assignment`, performs the assignment via
    :func:`assign_tokens_to_segments`, and applies cummax monotonicity so that
    once a token has been assigned to segment ``k`` no later token may be
    assigned to a segment ``< k``.

    This is the shared core of
    :func:`steering.eval.axbench_bpb.seg_map.build_multi_segment_seg_map`
    and
    :meth:`steering.conditional.encoder_decoder.eval_adapter.TargetAwareEncoderDecoderEvalWrapper._fill_context_assignments`.
    Note: in the encdec eval adapter, alignment was lifted to a separate
    ``_align_row_context_segments`` method that runs BEFORE this function
    is called (see ``_build_conditioning_impl``).

    Args:
        tokenizer: Decoder-only-style tokenizer with ``decode(token_ids)``.
        context_tokens: Token IDs to assign.
        context_segments: ``ContextFeatureSegment`` tuple with ``char_start``
            / ``char_end`` per segment.

    Returns:
        ``(len(context_tokens),)`` int64 tensor of segment indices on CPU. If
        either ``context_tokens`` or ``context_segments`` is empty, returns a
        zero tensor of shape ``(len(context_tokens),)`` (empty when there are
        no tokens). Callers that need a different device should ``.to(...)``
        the result.
    """
    if not context_tokens or not context_segments:
        return t.zeros(len(context_tokens), dtype=t.long)

    offsets = token_to_char_offsets(tokenizer, context_tokens)
    token_starts = t.tensor(offsets[:-1], dtype=t.long)
    token_ends = t.tensor(offsets[1:], dtype=t.long)

    seg_char_starts = t.tensor(
        [s.char_start for s in context_segments], dtype=t.long
    )
    seg_char_ends = t.tensor(
        [s.char_end for s in context_segments], dtype=t.long
    )
    seg_char_starts, seg_char_ends = strict_segment_bounds_for_assignment(
        seg_char_starts=seg_char_starts,
        seg_char_ends=seg_char_ends,
        token_starts=token_starts,
        token_ends=token_ends,
    )
    assignments = assign_tokens_to_segments(
        token_starts,
        token_ends,
        seg_char_starts,
        seg_char_ends,
    )
    return t.cummax(assignments, dim=0).values
