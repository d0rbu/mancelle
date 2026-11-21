"""Post-extraction feature-shard splice core (issue #590).

Per base shard, re-tokenize the source parquet (CPU, weight-free), verify the
re-tokenization reproduces the base shard's segmentation, attach each base
segment's stored ``(feature_ids, feature_mags)`` verbatim, recombine by cutting a
seeded fraction of splice points and stitching each cut's tail onto a deranged
partner's (gated at every stage by round-trip stability), and build a derived CSR
shard + decoded text + provenance. The derived directory is a fully-valid,
extraction-equivalent feature directory.

See ``docs/superpowers/specs/2026-06-30-post-extraction-feature-shard-splicer-design.md``.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import deque
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch as t
from loguru import logger

from steering._roundtrip_seam import (
    WindowedSeamChecker,
    _input_ids_tuple,
    is_roundtrip_stable,
    normalize_gemma3_space_marker_doc,
    to_text,
)
from steering.feature_io import load_features
from steering.feature_shard_pairing import OppositePercentilePairing
from steering.parquet_io import read_parquet_texts
from steering.refined_types import PositiveInt
from steering.segmentation import (
    SegmentMode,
    batch_truncate_to_training_window,
    resolve_bos_token_id,
    segment_mode_from_metadata,
    segment_tokens,
    token_to_char_offsets,
)
from steering.shard_alignment import (
    AlignmentViolation,
    canonical_doc_blocks,
    verify_document_alignment,
)

if TYPE_CHECKING:
    from scripts.preprocess._splice_profile import SpliceShardProfile

SegmentRef = tuple[int, int]  # (src_doc, seg_idx)

PROVENANCE_DIRNAME = "_splice_provenance"
PARQUET_DIRNAME = "_parquets"

# Checksum-like keys pinned to base shard bytes; never valid on a freshly built derived shard.
_DERIVED_METADATA_DISCARD = frozenset({"percentile_checksum", "tokenizer_checksum"})

def _profile_detail(profile: SpliceShardProfile | None, key: str):
    if profile is None:
        return nullcontext()
    return profile.time_detail(key)


class AlignmentError(RuntimeError):
    """The source parquet re-tokenization does not reproduce the base shard's
    recorded segmentation — tokenizer/source-data drift, or the wrong parquet."""


class EmptyBaseSegmentError(ValueError):
    """A base shard segment carries zero features (violates issue #522). Base
    extracted shards must be non-empty; this signals a corrupt base shard."""


class PoolExhaustedError(RuntimeError):
    """The stability redraw pool ran out of replacement splice points."""


class UnsplittableChainError(RuntimeError):
    """No boundary in an overflowing chain yields a stable split."""


class SpliceInternalError(RuntimeError):
    """A never-happens internal invariant was violated."""


class TokenizerContractError(RuntimeError):
    """The tokenizer advertises a BOS id but does not prepend it by default.

    The splice pipeline requires that tokenizer(text)["input_ids"][0] == bos_id
    whenever bos_id = resolve_bos_token_id(tokenizer) is not None. A tokenizer
    that breaks this contract will produce derived documents whose first token is not
    the BOS, invalidating the BOS-led document assumptions throughout recombination.
    """


@dataclass(frozen=True)
class SourceSegment:
    """One base segment: its token tuple (from re-tokenization) plus its stored
    features (copied verbatim from the base shard CSR)."""

    tokens: tuple[int, ...]
    feature_ids: t.Tensor
    feature_mags: t.Tensor


# ---------------------------------------------------------------------------
# Seeded selection and derangement primitives (Task 3, issue #590)
# ---------------------------------------------------------------------------


def recombination_seed(global_seed: int, shard_index: int, cut_fraction: float) -> int:
    """Per-shard engine seed; cut_fraction folds in so different fractions give
    independent selections. repr() of the float is stable across platforms."""
    material = f"{global_seed}:{shard_index}:{cut_fraction!r}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _enumerate_splice_points(source_segments: list[list[SourceSegment]]) -> list[tuple[int, int]]:
    """(doc, boundary) for boundary in 0..n_segs-2, lexicographic order."""
    return [
        (d, b)
        for d in range(len(source_segments))
        for b in range(len(source_segments[d]) - 1)
    ]


def _draw_cut_selection(rng: random.Random, n_points: int, target_k: int) -> tuple[list[int], deque[int]]:
    """One shuffle; selected = first target_k, pool = the rest (redraw order)."""
    order = list(range(n_points))
    rng.shuffle(order)
    return order[:target_k], deque(order[target_k:])


def _draw_derangement(rng: random.Random, k: int) -> list[int]:
    """Uniform derangement by rejection sampling (uniform permutations filtered)."""
    if k < 2:
        raise SpliceInternalError("derangement requires k >= 2")
    while True:
        sigma = list(range(k))
        rng.shuffle(sigma)
        if all(sigma[i] != i for i in range(k)):
            return sigma


# ---------------------------------------------------------------------------
# Stitch, chain assembly, loop detection (Task 4, issue #590)
# ---------------------------------------------------------------------------


def _build_successors(
    source_segments: list[list[SourceSegment]], cuts: list[tuple[int, int]], sigma: list[int],
) -> dict[SegmentRef, SegmentRef | None]:
    """Source adjacencies, minus cut edges, plus stitches X(cuts[i]) -> Y(cuts[sigma[i]])."""
    succ: dict[SegmentRef, SegmentRef | None] = {}
    for d, segs in enumerate(source_segments):
        for j in range(len(segs)):
            succ[(d, j)] = (d, j + 1) if j + 1 < len(segs) else None
    for i, (d, b) in enumerate(cuts):
        dp, bp = cuts[sigma[i]]
        succ[(d, b)] = (dp, bp + 1)
    return succ


def _read_chains(
    source_segments: list[list[SourceSegment]], succ: dict[SegmentRef, SegmentRef | None],
) -> tuple[list[list[SegmentRef]], list[list[SegmentRef]]]:
    """Chains walked from each root (d, 0); leftovers are pure cycles (loops).

    Why loops are exactly the unvisited part: non-root refs have exactly one
    predecessor (their source edge or the unique stitch targeting them — sigma
    is a bijection); root refs (d, 0) have none (stitches target ordinal >= 1).
    Combined with no two refs sharing a successor (in-degree <= 1), i.e. the
    successor function is injective, the successor graph is an injective
    function where every ref maps to at most one successor. Root-walks from
    (d,0) are simple paths ending at source-doc tails (where successor is None),
    and all unvisited segments form a disjoint union of cycles. A cycle can
    never contain any (d,0) because roots are exclusively served as chain starts
    (once visited in a root-walk, the chain proceeds forward; the only edge
    leading into a root is from its own document's source-doc adjacency, which
    is never cut, and stitches only rewrite cut edges, leaving (d,0)'s incoming
    edge intact).
    """
    visited: set[SegmentRef] = set()
    chains: list[list[SegmentRef]] = []
    for d in range(len(source_segments)):
        cur: SegmentRef | None = (d, 0)
        chain: list[SegmentRef] = []
        while cur is not None:
            if cur in visited:
                raise SpliceInternalError(f"chain re-entered {cur}")
            visited.add(cur)
            chain.append(cur)
            cur = succ[cur]
        chains.append(chain)
    loops: list[list[SegmentRef]] = []
    seen: set[SegmentRef] = set(visited)
    for start in sorted(set(succ) - visited):
        if start in seen:
            continue
        cyc: list[SegmentRef] = []
        cur = start
        while cur not in seen:
            seen.add(cur)
            cyc.append(cur)
            nxt = succ[cur]
            if nxt is None:
                raise SpliceInternalError(f"loop walk hit chain end at {cur}")
            cur = nxt
        loops.append(cyc)
    return chains, loops


# ---------------------------------------------------------------------------
# Batched oracle, piece IDs, loop-break, overflow split (Task 5, issue #590)
# ---------------------------------------------------------------------------


def _batch_roundtrip_stable(docs_ids: list[tuple[int, ...]], tokenizer: Any) -> list[bool]:
    """Batched is_roundtrip_stable — MUST agree with it element-for-element.

    Handles all HF and non-HF tokenizer output shapes:
    - List of lists (standard HF batched output)
    - Tuple of tuples (some tokenizers wrap in tuples)
    - List/tuple of ndarray-like objects exposing .tolist()
    - Flat list (non-HF tokenizers returning one flat list for a single text)

    Each row is normalized via _input_ids_tuple (same coercion as the single-doc oracle)
    to produce a plain tuple of ints for comparison.

    Raises :class:`SpliceInternalError` on an empty ``docs_ids`` list — an empty batch
    is a caller logic error: :func:`load_source_segments` always has >= 1 doc, and
    :func:`_first_unstable_piece` is only reached when target_k >= 2 (guaranteeing at
    least one non-identity piece).
    """
    if not docs_ids:
        raise SpliceInternalError("_batch_roundtrip_stable: empty docs_ids")
    texts = [to_text(ids, tokenizer) for ids in docs_ids]
    encoded = tokenizer(texts, padding=False)["input_ids"]
    # Coerce the outer container to Python if it is a tensor/ndarray (handles
    # batch ndarray output where .tolist() produces a list of lists).
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    # Detect nesting: if the first element has no __iter__ attribute it is an
    # integer scalar — the whole result is a single flat row (non-HF flat-list
    # regime for single-text input). Wrap to make the structure uniformly nested.
    if encoded and not hasattr(encoded[0], "__iter__"):
        encoded = [encoded]
    # Normalize each row via _input_ids_tuple — the same coercion the single-doc
    # oracle uses — so tuple/ndarray/tensor row types are handled uniformly.
    rows = [_input_ids_tuple({"input_ids": row}) for row in encoded]
    return [row == ids for row, ids in zip(rows, docs_ids, strict=True)]


def _piece_ids(
    piece: list[SegmentRef],
    new_started: bool,
    seg_tokens: list[list[tuple[int, ...]]],
    bos_id: int,
) -> tuple[int, ...]:
    """Concatenated token ids for a piece, with BOS prepended iff new-started."""
    ids = tuple(tid for ref in piece for tid in seg_tokens[ref[0]][ref[1]])
    return ((bos_id,) + ids) if new_started else ids


def _break_loop(
    cyc: list[SegmentRef],
    seg_tokens: list[list[tuple[int, ...]]],
    tokenizer: Any,
    bos_id: int,
) -> list[SegmentRef] | None:
    """Try each stitch-entry candidate in the cycle (ascending ref order).

    A ref ``r = cyc[i]`` is a stitch candidate when its cycle-predecessor edge is a
    stitch: ``cyc[i-1] != (r[0], r[1] - 1)`` (predecessor is not the natural adjacency).
    Candidate detection uses cycle adjacency, not the global successor map.  For each
    candidate, rotate the cycle to start there and check whether
    ``is_roundtrip_stable((bos_id,) + concat_ids)`` passes.  Return the first
    passing rotation as a list of refs; return ``None`` if none pass.
    """
    n = len(cyc)
    # Collect stitch candidates: refs whose cycle-predecessor edge is a stitch.
    candidates: list[tuple[int, SegmentRef]] = []  # (cycle index, ref)
    for i in range(n):
        ref = cyc[i]
        pred = cyc[(i - 1) % n]
        # Edge pred->ref is a stitch iff pred is not the natural predecessor of ref.
        if pred != (ref[0], ref[1] - 1):
            candidates.append((i, ref))
    # Sort by ascending ref (doc, seg) order.
    candidates.sort(key=lambda x: x[1])
    for start_idx, _ in candidates:
        rotation = cyc[start_idx:] + cyc[:start_idx]
        concat_ids = tuple(tid for ref in rotation for tid in seg_tokens[ref[0]][ref[1]])
        if is_roundtrip_stable((bos_id,) + concat_ids, tokenizer):
            return rotation
    return None


def _split_overflow(
    piece: list[SegmentRef],
    new_started: bool,
    seg_tokens: list[list[tuple[int, ...]]],
    tokenizer: Any,
    bos_id: int,
    limit: int,
) -> list[tuple[list[SegmentRef], bool]]:
    """Split an overflowing piece into sub-pieces each fitting within ``limit`` tokens.

    Token count of a piece includes a synthetic BOS (+1) iff the piece is new-started.
    While the current piece exceeds ``limit``: find the maximal boundary ``b`` (number
    of kept whole segments) such that the kept piece's token count is ≤ ``limit``; slide
    ``b`` earlier while the double gate fails — the kept piece and (BOS + remainder) must
    both pass ``is_roundtrip_stable``.  ``b`` reaching 0 raises
    :class:`UnsplittableChainError`.  Emits ``(kept, current_new_started)`` then
    continues with ``(remainder, True)``.  A piece exactly at the limit is not split.
    """
    result: list[tuple[list[SegmentRef], bool]] = []
    current_piece = piece
    current_ns = new_started

    while True:
        ids = _piece_ids(current_piece, current_ns, seg_tokens, bos_id)
        if len(ids) <= limit:
            result.append((current_piece, current_ns))
            break

        # Find maximal b in [1, len-1] such that kept-token-count ≤ limit.
        n = len(current_piece)
        bos_overhead = 1 if current_ns else 0
        # Cumulative token counts (with BOS overhead for the kept piece)
        b = 0
        running = bos_overhead
        for i, ref in enumerate(current_piece):
            seg_len = len(seg_tokens[ref[0]][ref[1]])
            if running + seg_len > limit:
                break
            running += seg_len
            b = i + 1  # b segments kept

        # b must be at least 1 and at most n-1 (leave something for remainder)
        b = min(b, n - 1)

        # Slide b earlier until the double gate passes.
        while b > 0:
            kept = current_piece[:b]
            remainder = current_piece[b:]
            kept_ids = _piece_ids(kept, current_ns, seg_tokens, bos_id)
            rem_raw = tuple(tid for ref in remainder for tid in seg_tokens[ref[0]][ref[1]])
            rem_ids = (bos_id,) + rem_raw
            if is_roundtrip_stable(kept_ids, tokenizer) and is_roundtrip_stable(rem_ids, tokenizer):
                break
            b -= 1

        if b == 0:
            first_ref = current_piece[0]
            sizes = [len(seg_tokens[r[0]][r[1]]) for r in current_piece]
            n_piece_segs = len(current_piece)
            n_piece_tokens = sum(sizes) + (1 if current_ns else 0)
            # Fatal by design: recovery-by-redraw was considered and rejected.
            # An overflowing chain with no stable split boundary signals a
            # degenerate tokenizer/corpus interaction — strictness over yield.
            raise UnsplittableChainError(
                f"no stable split boundary found in piece starting at {first_ref} "
                f"(doc={first_ref[0]}, new_started={current_ns}, "
                f"piece_segments={n_piece_segs}, piece_tokens={n_piece_tokens}, "
                f"limit={limit}, bos_overhead={bos_overhead}, "
                f"boundaries_tried=1..{n_piece_segs - 1}, "
                f"segment_sizes={sizes})"
            )

        result.append((kept, current_ns))
        current_piece = remainder
        current_ns = True

    return result


def _doc_seg_offsets(doc_ids: t.Tensor, n_docs: int) -> list[int]:
    """Per-doc cumulative segment-count boundaries ``[0, ..., n_segments]``.

    Base shards are doc-ordered (``doc_ids`` monotonic non-decreasing, segments
    0..k-1 within each doc), so doc ``d`` owns global rows ``[bounds[d], bounds[d+1])``.

    Adapts the shared pure alignment core to the list/error contract used by
    feature-shard splicing.
    """
    result = canonical_doc_blocks(doc_ids, n_docs)
    if isinstance(result, AlignmentViolation):
        raise AlignmentError(
            f"invalid base-shard doc_ids ({result.code}): {result.detail}"
        )
    return list(result.boundaries)


def _verify_doc_alignment(
    *,
    shard_name: str,
    doc_idx: int,
    n_full: int,
    segments: list[tuple[int, int]],
    base_shard: dict,
    seg_lo: int,
    seg_hi: int,
) -> None:
    """Fail loud if the re-tokenized doc disagrees with the base shard.

    Adapts the shared pure alignment core to ``AlignmentError``.
    """
    violation = verify_document_alignment(
        doc_idx=doc_idx,
        n_full_tokens=n_full,
        segments=segments,
        stored_doc_token_counts=base_shard["doc_token_counts"],
        stored_segment_positions=base_shard["segment_positions"],
        stored_segment_indices=base_shard["segment_indices"],
        seg_lo=seg_lo,
        seg_hi=seg_hi,
    )
    if violation is not None:
        raise AlignmentError(
            f"{shard_name}: {violation.detail}."
        )


@dataclass(frozen=True)
class SourceLoadStats:
    """Counts from one :func:`load_source_segments` call."""

    n_trimmed_docs: int
    n_trimmed_tokens: int
    n_marker_normalized_docs: int


# ---------------------------------------------------------------------------
# Byte-fallback trim helpers
# ---------------------------------------------------------------------------

_BYTE_FALLBACK_PIECE_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def _is_byte_fallback_token(tokenizer: Any, token_id: int) -> bool:
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if convert is None:
        return False
    piece = convert(int(token_id))
    return isinstance(piece, str) and _BYTE_FALLBACK_PIECE_RE.fullmatch(piece) is not None


def _trim_truncated_doc_tail(
    kept_ids: list[int], tokenizer: Any, *, shard_name: str, doc_idx: int
) -> int:
    """Return the number of trailing tokens popped before the doc round-trips."""
    ids = list(kept_ids)
    n_popped = 0
    while not is_roundtrip_stable(tuple(ids), tokenizer):
        if not ids or not _is_byte_fallback_token(tokenizer, ids[-1]):
            raise SourceDocNotStableError(
                f"{shard_name} doc {doc_idx}: not round-trip stable after popping "
                f"{n_popped} trailing byte-fallback token(s); refusing to trim further."
            )
        ids.pop()
        n_popped += 1
    return n_popped


def load_source_segments(
    base_shard: dict,
    texts: list[str],
    tokenizer: Any,
    *,
    segment_size: int,
    context_window: int,
    sae_max_context: int | None,
    shard_name: str,
    profile: SpliceShardProfile | None = None,
) -> tuple[list[list[SourceSegment]], SourceLoadStats]:
    """Re-tokenize each source doc, verify alignment with the base shard, and
    return ``[doc][seg]`` segments carrying the base shard's features verbatim.

    For TRUNCATED docs that fail :func:`~steering._roundtrip_seam.is_roundtrip_stable`,
    trailing byte-fallback tokens (piece matches ``<0xNN>``) are popped until the
    doc round-trips.  Any other instability raises :class:`SourceDocNotStableError`.
    Non-truncated unstable docs always raise.

    Raises :class:`AlignmentError` on any drift, :class:`EmptyBaseSegmentError`
    on an empty base segment, and :class:`SourceDocNotStableError` when a doc
    cannot be stabilised.

    Returns ``(segments, stats)`` where ``stats`` counts trimmed docs / tokens
    and gemma-3 marker-normalised docs.
    """
    base_mode = segment_mode_from_metadata(base_shard, source=shard_name)
    if base_mode is not SegmentMode.fixed_tokens:
        raise AlignmentError(
            f"feature-shard splicing supports fixed_tokens base shards only (segment_mode={base_mode.value!r})"
        )
    refined_segment_size = PositiveInt.parse(segment_size)
    n_docs = int(base_shard["n_docs"])
    if len(texts) != n_docs:
        raise AlignmentError(
            f"{shard_name}: parquet has {len(texts)} docs but the base shard "
            f"records n_docs={n_docs} — mismatched parquet."
        )
    offsets = base_shard["offsets"]
    feature_ids = base_shard["feature_ids"]
    feature_mags = base_shard["feature_mags"]
    seg_bounds = _doc_seg_offsets(base_shard["doc_ids"], n_docs)

    with _profile_detail(profile, "source.truncate_tokenize"):
        truncated = batch_truncate_to_training_window(
            list(texts), tokenizer, context_window, sae_max_context
        )
    norm_docs = 0
    norm_mismatches = 0
    n_trimmed_docs = 0
    n_trimmed_tokens = 0
    out: list[list[SourceSegment]] = []
    # Per-doc accumulators for the batch stability check (built during first-pass loop).
    _all_ids: list[tuple[int, ...]] = []
    _all_was_truncated: list[bool] = []

    with _profile_detail(profile, "source.align_attach"):
        for d in range(n_docs):
            kept_ids, n_full = truncated[d]
            was_truncated = n_full > len(kept_ids)
            kept_ids, norm_docs, norm_mismatches = normalize_gemma3_space_marker_doc(
                kept_ids, tokenizer, source_text=texts[d], doc_index=d,
                truncated=was_truncated, normalized_docs_seen=norm_docs,
                token_mismatches_seen=norm_mismatches,
            )
            if not kept_ids:
                raise AlignmentError(f"{shard_name} doc {d}: 0 tokens after truncation.")
            # Alignment uses UNTRIMMED ids — the base shard was extracted from them.
            segments = segment_tokens(
                kept_ids,
                SegmentMode.fixed_tokens,
                refined_segment_size,
            )
            seg_lo, seg_hi = seg_bounds[d], seg_bounds[d + 1]
            _verify_doc_alignment(
                shard_name=shard_name, doc_idx=d, n_full=n_full, segments=segments,
                base_shard=base_shard, seg_lo=seg_lo, seg_hi=seg_hi,
            )
            doc_segs: list[SourceSegment] = []
            for j, (s, e) in enumerate(segments):
                gi = seg_lo + j
                lo, hi = int(offsets[gi]), int(offsets[gi + 1])
                if hi <= lo:
                    raise EmptyBaseSegmentError(
                        f"{shard_name} doc {d} seg {j}: base segment has zero features "
                        f"(offsets[{gi}]==offsets[{gi + 1}]={lo}); base shards must be "
                        f"non-empty (issue #522)."
                    )
                doc_segs.append(
                    SourceSegment(
                        tokens=tuple(kept_ids[s:e]),
                        feature_ids=feature_ids[lo:hi],
                        feature_mags=feature_mags[lo:hi],
                    )
                )
            out.append(doc_segs)
            _all_ids.append(tuple(kept_ids))
            _all_was_truncated.append(was_truncated)

    # Batch round-trip stability check — one tokenizer call for all docs rather than
    # N serial calls. Docs that fail enter the existing per-doc repair path (truncated
    # → _trim_truncated_doc_tail; non-truncated → raise SourceDocNotStableError).
    with _profile_detail(profile, "source.batch_roundtrip"):
        _verdicts = _batch_roundtrip_stable(_all_ids, tokenizer)
    with _profile_detail(profile, "source.trim_unstable"):
        for d, _ok in enumerate(_verdicts):
            if _ok:
                continue
            # Doc d is not round-trip stable — repair path (unchanged semantics).
            _was_truncated = _all_was_truncated[d]
            if not _was_truncated:
                raise SourceDocNotStableError(
                    f"{shard_name} doc {d}: not round-trip stable and was not "
                    f"truncated; refusing to trim."
                )
            _kept_ids = list(_all_ids[d])
            n_popped = _trim_truncated_doc_tail(
                _kept_ids, tokenizer, shard_name=shard_name, doc_idx=d
            )
            last_seg = out[d][-1]
            if n_popped >= len(last_seg.tokens):
                raise SourceDocNotStableError(
                    f"{shard_name} doc {d}: byte-fallback trim ({n_popped} token(s)) "
                    f"would empty or cross the last segment boundary "
                    f"(last segment has {len(last_seg.tokens)} token(s))."
                )
            out[d][-1] = replace(last_seg, tokens=last_seg.tokens[:-n_popped])
            n_trimmed_docs += 1
            n_trimmed_tokens += n_popped

    if n_trimmed_docs > 0:
        logger.info(
            "{}: trimmed {} doc(s), {} trailing byte-fallback token(s)",
            shard_name, n_trimmed_docs, n_trimmed_tokens,
        )
    stats = SourceLoadStats(
        n_trimmed_docs=n_trimmed_docs,
        n_trimmed_tokens=n_trimmed_tokens,
        n_marker_normalized_docs=norm_docs,
    )
    return out, stats


# ---------------------------------------------------------------------------
# Recombination: batch select -> derange -> stitch engine (Task 6, issue #590)
# ---------------------------------------------------------------------------


class SourceDocNotStableError(RuntimeError):
    """A source document's own tokenization is not idempotent (re-tokenizing its
    decoded text changes the ids), so it cannot safely participate in the splice.
    Raised by :func:`load_source_segments` when byte-fallback trimming cannot
    stabilise a truncated doc (or a non-truncated doc is unstable)."""

    def __init__(self, msg_or_idx: str | int) -> None:
        if isinstance(msg_or_idx, int):
            super().__init__(
                f"source doc {msg_or_idx} is not round-trip stable (degenerate input)"
            )
            self.doc_index: int | None = msg_or_idx
        else:
            super().__init__(str(msg_or_idx))
            self.doc_index = None


@dataclass(frozen=True)
class DerivedDoc:
    """One derived document: an ordered tuple of source segment refs plus a flag.

    ``new_started`` is True for documents the engine forms fresh — a broken loop
    or an overflow remainder — whose reconstruction prepends a synthetic BOS. Root
    chains keep ``new_started=False``: their first segment is a source doc's
    BOS-led segment 0, so the real BOS is already present.
    """

    segment_refs: tuple[SegmentRef, ...]
    new_started: bool


@dataclass(frozen=True)
class ShardSpliceResult:
    """One shard's recombination outcome plus its engine counters."""

    docs: tuple[DerivedDoc, ...]
    n_cuts_final: int
    n_redrawn: int
    n_loops: int
    n_splits: int


def _cut_boundaries_by_doc(cuts: list[tuple[int, int]]) -> dict[int, list[int]]:
    """Per source-doc sorted list of cut boundaries under the current cut set."""
    by_doc: dict[int, list[int]] = {}
    for d, b in cuts:
        by_doc.setdefault(d, []).append(b)
    for boundaries in by_doc.values():
        boundaries.sort()
    return by_doc


def _seam_context(
    slot: int,
    cuts: list[tuple[int, int]],
    sigma: list[int],
    seg_tokens: list[list[tuple[int, ...]]],
    cut_bounds: dict[int, list[int]],
    bos_id: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Run-local, BOS-led ``(prefix, suffix)`` ids for the stitch at ``slot``.

    ``prefix`` = tokens from the start of X's cut-free run through X = ``cuts[slot]``;
    ``suffix`` = tokens from Y through the end of Y's run, where Y is the stitch
    target ``(cuts[sigma[slot]].doc, cuts[sigma[slot]].boundary + 1)``. Runs are the
    maximal cut-free source intervals under the CURRENT cut set.

    The prefix is made BOS-led so :meth:`WindowedSeamChecker.stitch_is_stable`'s
    doc-start round-trip is well-formed: a run that starts at segment 0 already
    carries the real BOS; a mid-document run (``run_start > 0``) gets a synthetic
    BOS prepended. This is a local approximation (the run's true left context in the
    recombined doc is another stitch); Phase E's full-document oracle is the final
    authority. The BOS never reaches the seam being tested, so it cannot mask a
    boundary merge, and keeping the prefix run-local avoids false merges from the
    already-cut region to X's left.
    """
    d, b = cuts[slot]
    run_start = 0
    for c in cut_bounds[d]:
        if c < b:
            run_start = c + 1
        else:
            break
    prefix = tuple(tid for s in range(run_start, b + 1) for tid in seg_tokens[d][s])
    if run_start > 0:  # mid-doc run: not the real BOS-led doc start
        prefix = (bos_id, *prefix)

    dp, bp = cuts[sigma[slot]]
    y0 = bp + 1
    run_end = len(seg_tokens[dp]) - 1
    for c in cut_bounds[dp]:
        if c >= y0:
            run_end = c
            break
    suffix = tuple(tid for s in range(y0, run_end + 1) for tid in seg_tokens[dp][s])
    return prefix, suffix


def _prefilter_seams(
    cuts: list[tuple[int, int]],
    sigma: list[int],
    seg_tokens: list[list[tuple[int, ...]]],
    checker: WindowedSeamChecker,
    replace_fn: Callable[[int], None],
    bos_id: int,
    memo: dict[tuple[tuple[int, ...], tuple[int, ...]], bool] | None = None,
) -> None:
    """Windowed pre-filter over all k stitch seams (run-local, BOS-led context).

    On an unstable seam at slot ``i``, redraw via ``replace_fn(i)`` (pool order) and
    re-check ONLY the <=2 slots the replacement touches: slot ``i`` (whose prefix
    the moved cut supplies) and the unique slot ``j`` with ``sigma[j] == i`` (whose
    suffix it supplies). Loop until every seam passes; ``replace_fn`` raises on pool
    exhaustion. The full-document oracle (Phase E) is the final authority for the
    run-length side effects this local re-check intentionally defers.

    ``memo`` is an optional dict keyed on ``(prefix_ids, suffix_ids)`` that persists
    across outer fixpoint iterations (created in :func:`recombine_segments` and passed
    here). Because the key IS the full checked input, verdicts are provably identical
    to a fresh call — a hit saves a tokenizer call while producing the same decision.

    Memo purity contract: the verdict stored under a key depends ONLY on that
    (prefix, suffix) pair given a fixed checker.  The memo is owned exclusively by
    one :func:`recombine_segments` call; it is never shared across calls, so stale
    entries are impossible.
    """
    k = len(cuts)
    inverse = [0] * k
    for j in range(k):
        inverse[sigma[j]] = j  # inverse[i] = the slot j with sigma[j] == i
    pending = deque(range(k))
    queued = [True] * k
    cut_bounds = _cut_boundaries_by_doc(cuts)  # stable until a redraw mutates cuts
    while pending:
        i = pending.popleft()
        queued[i] = False
        prefix, suffix = _seam_context(i, cuts, sigma, seg_tokens, cut_bounds, bos_id)
        memo_key = (prefix, suffix)
        if memo is not None and memo_key in memo:
            verdict = memo[memo_key]
        else:
            verdict = checker.stitch_is_stable(prefix, suffix)
            if memo is not None:
                # O(1) type assertion: key components must be tuples (not lists/arrays).
                # Elementwise checks are omitted for performance — outer isinstance suffices
                # since _seam_context always returns tuple literals.
                assert isinstance(memo_key[0], tuple) and isinstance(memo_key[1], tuple), (
                    f"memo key components must be tuples of ints; "
                    f"got {type(memo_key[0])!r} and {type(memo_key[1])!r}"
                )
                memo[memo_key] = verdict
        if verdict:
            continue
        replace_fn(i)  # redraw slot i from the pool (raises PoolExhaustedError)
        cut_bounds = _cut_boundaries_by_doc(cuts)  # recompute after redraw mutates cuts
        for touched in (i, inverse[i]):
            if not queued[touched]:
                queued[touched] = True
                pending.append(touched)


def _lowest_cut_slot_in_cycle(cyc: list[SegmentRef], cuts: list[tuple[int, int]]) -> int:
    """The lowest-numbered cut slot whose cut point lies in the cycle."""
    cyc_set = set(cyc)
    involved = [i for i in range(len(cuts)) if cuts[i] in cyc_set]
    if not involved:
        raise SpliceInternalError(f"cycle {cyc} contains no cut point")
    return min(involved)


def _resolve_loops(
    loops: list[list[SegmentRef]],
    cuts: list[tuple[int, int]],
    seg_tokens: list[list[tuple[int, ...]]],
    tokenizer: Any,
    bos_id: int,
) -> tuple[list[tuple[list[SegmentRef], bool]], int | None]:
    """Break every loop into a fresh ``new_started`` document (oracle-gated).

    Returns ``(loop_docs, culprit)``. On the first loop that cannot be broken
    stably, returns ``([], slot)`` where ``slot`` is that loop's lowest-numbered
    involved cut slot (the caller redraws it and restarts). Otherwise returns the
    broken loop docs ordered by lowest involved cut ref, with ``culprit=None``.
    """
    resolved: list[tuple[SegmentRef, list[SegmentRef]]] = []
    for cyc in loops:
        broken = _break_loop(cyc, seg_tokens, tokenizer, bos_id)
        if broken is None:
            return [], _lowest_cut_slot_in_cycle(cyc, cuts)
        cyc_set = set(cyc)
        involved_refs = [cuts[i] for i in range(len(cuts)) if cuts[i] in cyc_set]
        resolved.append((min(involved_refs), broken))
    resolved.sort(key=lambda kv: kv[0])
    loop_docs = [(refs, True) for _key, refs in resolved]
    return loop_docs, None


def _split_all(
    chains: list[list[SegmentRef]],
    loop_docs: list[tuple[list[SegmentRef], bool]],
    seg_tokens: list[list[tuple[int, ...]]],
    tokenizer: Any,
    bos_id: int,
    limit: int,
) -> list[tuple[list[SegmentRef], bool]]:
    """Overflow-split every piece; remainders follow their kept piece inline.

    Order: root chains in root order (each ``new_started=False``), then loop docs
    (each ``new_started=True``). Every emitted piece fits within ``limit``; a split
    raises :class:`UnsplittableChainError` on a dead end.
    """
    pieces: list[tuple[list[SegmentRef], bool]] = []
    for chain in chains:
        pieces.extend(_split_overflow(chain, False, seg_tokens, tokenizer, bos_id, limit))
    for refs, new_started in loop_docs:
        pieces.extend(
            _split_overflow(refs, new_started, seg_tokens, tokenizer, bos_id, limit)
        )
    return pieces


def _first_unstable_piece(
    pieces: list[tuple[list[SegmentRef], bool]],
    source_segments: list[list[SourceSegment]],
    seg_tokens: list[list[tuple[int, ...]]],
    tokenizer: Any,
    bos_id: int,
) -> tuple[list[SegmentRef], bool] | None:
    """First piece that fails the batched full-document oracle, else ``None``.

    Pieces identical to a full source doc (an unstitched root chain, ``new_started``
    False) are skipped — their stability was proven in :func:`load_source_segments`.
    """
    identity = {
        tuple((d, j) for j in range(len(doc)))
        for d, doc in enumerate(source_segments)
    }
    checked: list[tuple[list[SegmentRef], bool]] = []
    ids_list: list[tuple[int, ...]] = []
    for refs, new_started in pieces:
        if (not new_started) and tuple(refs) in identity:
            continue
        ids_list.append(_piece_ids(refs, new_started, seg_tokens, bos_id))
        checked.append((refs, new_started))
    verdicts = _batch_roundtrip_stable(ids_list, tokenizer)
    for piece, ok in zip(checked, verdicts, strict=True):
        if not ok:
            return piece
    return None


def _first_cut_slot_in_piece(
    bad: tuple[list[SegmentRef], bool], cuts: list[tuple[int, int]]
) -> int:
    """Map a failing piece's earliest stitched junction back to its cut slot.

    A junction ``(a, b)`` is stitched when ``b != (a.doc, a.ord + 1)`` (not the
    natural source adjacency). The earliest such ``a`` is a live cut point; return
    the slot ``i`` with ``cuts[i] == a``. Raises :class:`SpliceInternalError` when
    the piece holds no stitch (source-pure pieces are skipped in Phase E; split /
    loop pieces are locally gated, so a genuinely-failing piece must weave).
    """
    refs, _new_started = bad
    for m in range(len(refs) - 1):
        a, b = refs[m], refs[m + 1]
        if b != (a[0], a[1] + 1):
            try:
                return cuts.index(a)
            except ValueError as exc:
                raise SpliceInternalError(
                    f"stitched junction prefix {a} is not a live cut"
                ) from exc
    raise SpliceInternalError(f"failing piece {refs} contains no stitched junction")


def recombine_segments(
    source_segments: list[list[SourceSegment]],
    tokenizer: Any,
    *,
    cut_fraction: float,
    shard_seed: int,
    shard_name: str,
    limit: int,
    profile: SpliceShardProfile | None = None,
    pairing: OppositePercentilePairing | None = None,
) -> ShardSpliceResult:
    """Recombine a shard's segments by cutting a seeded fraction of splice points
    and stitching each cut's tail onto a deranged partner's, gated at every stage
    by round-trip stability.

    ``target_k = round(cut_fraction * n_splice_points)``; ``target_k < 2`` is
    identity (a derangement needs >= 2 cuts). One seeded shuffle selects the cuts
    and a redraw pool; a seeded derangement pairs them. The outer fixpoint pre-
    filters the k seams (redrawing from the pool on any unstable seam), reads root
    chains + loops, turns loops into fresh BOS-led docs, overflow-splits every
    piece, then runs the batched full-document oracle — redrawing the earliest
    stitched cut of any piece that still fails — until every piece is stable.

    Output order is deterministic: root chains in root order (each followed by its
    overflow remainders), then loop docs by lowest involved cut ref (each followed
    by its remainders).

    ``pairing`` optionally improves the initial derangement by bounded percentile
    L1 scoring. Cut selection and all safety/redraw behavior stay unchanged.
    """
    with _profile_detail(profile, "recombine.prepare"):
        seg_tokens: list[list[tuple[int, ...]]] = [
            [seg.tokens for seg in doc] for doc in source_segments
        ]
        all_sp = _enumerate_splice_points(source_segments)
        target_k = round(cut_fraction * len(all_sp))
    if target_k < 2:  # identity: a derangement is impossible below k == 2
        docs = tuple(
            DerivedDoc(tuple((d, j) for j in range(len(doc))), False)
            for d, doc in enumerate(source_segments)
        )
        return ShardSpliceResult(docs, 0, 0, 0, 0)

    bos_id = resolve_bos_token_id(tokenizer)
    if bos_id is None:
        raise SpliceInternalError(
            f"{shard_name}: tokenizer has no BOS; cannot form new documents"
        )

    rng = random.Random(shard_seed)
    with _profile_detail(profile, "recombine.select_cuts"):
        sel_idx, pool = _draw_cut_selection(rng, len(all_sp), target_k)
    with _profile_detail(profile, "recombine.draw_derangement"):
        sigma = _draw_derangement(rng, target_k)
    cuts = [all_sp[i] for i in sel_idx]
    if pairing is not None:
        with _profile_detail(profile, "recombine.opposite_pairing"):
            sigma = pairing.improve_pairing(source_segments, cuts, sigma)
    checker = WindowedSeamChecker(tokenizer)
    n_redrawn = 0

    def _replace(slot: int) -> None:  # the ONLY redraw path (pool order, no rng)
        nonlocal n_redrawn
        if not pool:
            raise PoolExhaustedError(
                f"{shard_name}: redraw pool exhausted after {n_redrawn} redraws"
            )
        cuts[slot] = all_sp[pool.popleft()]
        n_redrawn += 1

    # Cross-iteration memo: (prefix_ids, suffix_ids) -> verdict bool.
    #
    # Purity: verdict depends ONLY on the key given a fixed checker; the checker
    # does not accumulate state between calls.  Lifetime: this dict is owned by
    # this recombine_segments activation and is never shared with any other call
    # (created fresh here, not returned, not captured by any closure that outlives
    # this call).  So any hit always returns the same verdict a fresh call would.
    #
    # Hits save tokenizer calls with no behavioural change.  Cut redraws invalidate
    # the affected slots' context (not their keys), so stale entries are impossible:
    # a redrawn cut changes the cut point, which changes the prefix/suffix ids,
    # which changes the key — the old key is never consulted again for the new seam.
    _seam_memo: dict[tuple[tuple[int, ...], tuple[int, ...]], bool] = {}

    while True:  # outer fixpoint
        with _profile_detail(profile, "recombine.prefilter_seams"):
            _prefilter_seams(
                cuts, sigma, seg_tokens, checker, _replace, bos_id, _seam_memo
            )
        with _profile_detail(profile, "recombine.read_graph"):
            succ = _build_successors(source_segments, cuts, sigma)
            chains, loops = _read_chains(source_segments, succ)
        with _profile_detail(profile, "recombine.resolve_loops"):
            loop_docs, culprit = _resolve_loops(loops, cuts, seg_tokens, tokenizer, bos_id)
        if culprit is not None:
            _replace(culprit)
            continue
        with _profile_detail(profile, "recombine.split_overflow"):
            pieces = _split_all(chains, loop_docs, seg_tokens, tokenizer, bos_id, limit)
        with _profile_detail(profile, "recombine.full_doc_oracle"):
            bad = _first_unstable_piece(
                pieces, source_segments, seg_tokens, tokenizer, bos_id
            )
        if bad is None:
            break
        _replace(_first_cut_slot_in_piece(bad, cuts))

    docs = tuple(DerivedDoc(tuple(refs), new_started) for refs, new_started in pieces)
    n_splits = len(pieces) - len(chains) - len(loop_docs)
    return ShardSpliceResult(docs, len(cuts), n_redrawn, len(loop_docs), n_splits)


# ---------------------------------------------------------------------------
# Derived shard build (CSR rebuild + char reconstruction)
# ---------------------------------------------------------------------------


class DerivedSegmentationError(RuntimeError):
    """A derived document's segment token counts violated the tiling invariant.

    Every interior segment must carry exactly ``segment_size`` tokens
    (``segment_size + 1`` for seg-0 of a ``new_started`` doc, which includes
    the synthetic BOS).  The last segment may be shorter.  A violation signals
    that a partial segment landed mid-document — a recombination bug."""


class DerivedDocTooLongError(RuntimeError):
    """A derived document exceeds the context window.

    Purely defensive: the overflow-split step in :func:`recombine_segments`
    guarantees every piece fits within ``min(context_window, sae_max_context)``,
    so this error should never fire in normal operation.  It is retained as a
    loud last-resort guard for programmatic misuse of :func:`build_derived_shard`."""


def build_derived_shard(
    result: ShardSpliceResult,
    source_segments: list[list[SourceSegment]],
    base_meta: dict,
    tokenizer: Any,
    *,
    segment_size: int,
    profile: SpliceShardProfile | None = None,
) -> tuple[dict, list[str]]:
    """Build the derived CSR shard dict + per-doc decoded texts.

    For each :class:`DerivedDoc`, concatenates its source segment tokens via
    :func:`_piece_ids` (prepending a synthetic BOS iff ``new_started``), computes
    token boundaries from the known per-segment lengths (no re-segmentation), derives
    char boundaries from :func:`~steering.segmentation.token_to_char_offsets`, and
    copies each source segment's features verbatim.

    Token-boundary invariant (TILING ASSERT): every interior segment carries exactly
    ``segment_size`` tokens — ``segment_size + 1`` for seg-0 of a ``new_started``
    doc, which absorbs the synthetic BOS.  The last segment is allowed to be shorter
    (≤ ``segment_size``; ≤ ``segment_size + 1`` for a single-segment ``new_started``
    doc).  Violation → :class:`DerivedSegmentationError`.

    A :class:`DerivedDocTooLongError` fires when a doc exceeds ``context_window``
    (defensive only; the engine's overflow-split step prevents this in practice).

    Returns ``(shard_dict, texts)`` in derived-doc order so the derived parquet's
    row order aligns with ``doc_ids``.
    """
    with _profile_detail(profile, "derived.prepare"):
        bos_id_opt = resolve_bos_token_id(tokenizer)
        has_new_started = any(doc.new_started for doc in result.docs)
        if has_new_started and bos_id_opt is None:
            raise SpliceInternalError(
                "build_derived_shard: tokenizer has no BOS id; cannot form new-started docs"
            )
        # Safe sentinel for the new_started=False path (never used, but _piece_ids needs int).
        bos_id: int = bos_id_opt if bos_id_opt is not None else 0

        context_window = int(base_meta["context_window"])
        seg_tokens: list[list[tuple[int, ...]]] = [
            [seg.tokens for seg in doc] for doc in source_segments
        ]
        all_fids: list[t.Tensor] = []
        all_fmags: list[t.Tensor] = []
        offsets: list[int] = [0]
        doc_ids: list[int] = []
        seg_indices: list[int] = []
        seg_positions: list[int] = []
        char_starts: list[int] = []
        char_ends: list[int] = []
        doc_token_counts: list[int] = []
        texts: list[str] = []
        singleton_cache: dict[int, str] = {}
        gi = 0

    for doc_i, doc in enumerate(result.docs):
        refs = list(doc.segment_refs)
        n_segs = len(refs)

        with _profile_detail(profile, "derived.piece_boundaries"):
            spliced_ids = _piece_ids(refs, doc.new_started, seg_tokens, bos_id)

            # Defensive: engine guarantees every piece fits within limit ≤ context_window.
            if len(spliced_ids) > context_window:
                raise DerivedDocTooLongError(
                    f"derived doc {doc_i} has {len(spliced_ids)} tokens "
                    f"> context_window={context_window} (engine invariant broken)."
                )

            # Compute token boundaries from known per-segment lengths.
            # For new_started docs, seg-0 absorbs one synthetic BOS token (+1).
            tok_boundaries: list[int] = [0]
            for k, ref in enumerate(refs):
                seg_len = len(seg_tokens[ref[0]][ref[1]])
                extra = 1 if (doc.new_started and k == 0) else 0
                tok_boundaries.append(tok_boundaries[-1] + seg_len + extra)

            # STRUCTURAL TILING ASSERT: interior segments must be full-size.
            seg_lens = [tok_boundaries[k + 1] - tok_boundaries[k] for k in range(n_segs)]
            for k in range(n_segs - 1):  # non-last segments
                expected = segment_size + 1 if (doc.new_started and k == 0) else segment_size
                if seg_lens[k] != expected:
                    raise DerivedSegmentationError(
                        f"derived doc {doc_i}: segment {k} has {seg_lens[k]} tokens "
                        f"(expected {expected}); violation of the tiling invariant "
                        f"— a partial segment landed mid-document. "
                        f"Per-segment lengths: {seg_lens}."
                    )
            # Last segment: ≤ segment_size; ≤ segment_size+1 for single-seg new_started.
            max_last = (segment_size + 1) if (doc.new_started and n_segs == 1) else segment_size
            if seg_lens[-1] > max_last:
                raise DerivedSegmentationError(
                    f"derived doc {doc_i}: last segment has {seg_lens[-1]} tokens "
                    f"(max allowed {max_last}). Per-segment lengths: {seg_lens}."
                )

        with _profile_detail(profile, "derived.char_text"):
            # Char offsets from the full spliced token sequence.
            char_offsets = token_to_char_offsets(
                tokenizer, list(spliced_ids), _singleton_cache=singleton_cache,
            )
            texts.append(to_text(spliced_ids, tokenizer))
            doc_token_counts.append(len(spliced_ids))

        with _profile_detail(profile, "derived.feature_refs"):
            for k, ref in enumerate(refs):
                src = source_segments[ref[0]][ref[1]]
                fids = src.feature_ids
                fmags = src.feature_mags
                if int(fids.numel()) == 0:
                    raise EmptyBaseSegmentError(
                        f"derived doc {doc_i} seg {k}: empty feature set (issue #522)."
                    )
                all_fids.append(fids)
                all_fmags.append(fmags)
                offsets.append(offsets[-1] + int(fids.numel()))
                doc_ids.append(doc_i)
                seg_indices.append(k)
                seg_positions.append(tok_boundaries[k])
                char_starts.append(char_offsets[tok_boundaries[k]])
                char_ends.append(char_offsets[tok_boundaries[k + 1]])
                gi += 1

    with _profile_detail(profile, "derived.tensor_assembly"):
        seg_mode = base_meta["segment_mode"]
        if isinstance(seg_mode, str):
            seg_mode = SegmentMode(seg_mode)
        # Start from ALL scalar (non-tensor) entries in base_meta so that future
        # scalar metadata (e.g. l0_tier) is propagated automatically without requiring
        # explicit handling in this function. Exclude checksum keys which describe the
        # base shard's overlay bytes and are invalid on a freshly built derived shard.
        # Computed keys below then override whatever base_meta supplies for those fields.
        shard: dict = {
            k: v for k, v in base_meta.items()
            if not isinstance(v, t.Tensor) and k not in _DERIVED_METADATA_DISCARD
        }
        shard.update({
            "offsets": t.tensor(offsets, dtype=t.int64),
            "feature_ids": (
                t.cat(all_fids).to(t.int32) if all_fids else t.zeros(0, dtype=t.int32)
            ),
            "feature_mags": (
                t.cat(all_fmags).to(t.float32) if all_fmags else t.zeros(0, dtype=t.float32)
            ),
            "doc_ids": t.tensor(doc_ids, dtype=t.int32),
            "segment_indices": t.tensor(seg_indices, dtype=t.int32),
            "segment_positions": t.tensor(seg_positions, dtype=t.int32),
            "segment_char_starts": t.tensor(char_starts, dtype=t.int32),
            "segment_char_ends": t.tensor(char_ends, dtype=t.int32),
            "doc_token_counts": t.tensor(doc_token_counts, dtype=t.int32),
            "n_docs": len(result.docs),
            "n_segments": gi,
            "layer": int(base_meta["layer"]),
            "context_window": int(base_meta["context_window"]),
            "pooling": str(base_meta["pooling"]),
            "segment_mode": seg_mode,
            "segment_size": segment_size,
            "context_independent_features": bool(base_meta["context_independent_features"]),
            "ignore_bos_features": bool(base_meta["ignore_bos_features"]),
        })
    return shard, texts


# ---------------------------------------------------------------------------
# Provenance manifest
# ---------------------------------------------------------------------------


def build_provenance(
    result: ShardSpliceResult,
    source_segments: list[list[SourceSegment]],
    *,
    params: dict,
    shard_index: int,
    shard_name: str,
    shard_seed: int,
    profile: SpliceShardProfile | None = None,
) -> dict:
    """Build the provenance sidecar dict (schema_version 2).

    Each derived segment maps back to ``[base_shard_idx, doc_idx, seg_ordinal]``
    — the same triple the packed-row provenance uses — so derived segments can
    join base-aligned overlays. ``params`` records the run knobs for audit.

    Schema v2 (issue #596) additionally records, per derived doc, one
    ``splices`` entry for every splice position ``p`` (a segment that is not
    its predecessor's natural continuation): the would-have-followed base
    identity ``whf_base = [shard, pred_doc, pred_ord + 1]`` and its location in
    the derived shard ``whf_derived = [derived_doc, position]``. Docs without
    junctions carry ``splices: []``. ``params["n_splice_positions"]`` totals
    the records. The existence invariant (the whf segment is always a real,
    placed base segment) is asserted; a violation raises
    :class:`SpliceInternalError` (never-happens: stitch predecessors are cuts
    with ``boundary <= n_segs - 2``, and chains + loops place every segment).

    ``shard_index`` / ``shard_name`` / ``shard_seed`` are supplied by the caller
    (the recombination result no longer carries them). Per-doc counters are
    recomputed from the segment refs.
    """
    with _profile_detail(profile, "provenance.placement"):
        # Where every source segment landed in the derived shard (bijection).
        placement: dict[SegmentRef, tuple[int, int]] = {}
        for i, doc in enumerate(result.docs):
            for p, ref in enumerate(doc.segment_refs):
                placement[ref] = (i, p)
    n_splice_positions = 0
    docs_meta: list[dict] = []
    with _profile_detail(profile, "provenance.docs"):
        for i, doc in enumerate(result.docs):
            refs = doc.segment_refs
            starts: list[int] = []
            acc = 1 if doc.new_started else 0  # a fresh doc's real tokens follow the BOS
            for ref in refs:
                starts.append(acc)
                acc += len(source_segments[ref[0]][ref[1]].tokens)
            n_switches = sum(
                1 for x in range(1, len(refs)) if refs[x][0] != refs[x - 1][0]
            )
            splices: list[dict] = []
            for p in range(1, len(refs)):
                pred_doc, pred_ord = refs[p - 1]
                if refs[p] == (pred_doc, pred_ord + 1):
                    continue  # natural adjacency — not a splice position
                whf = (pred_doc, pred_ord + 1)
                if whf[1] >= len(source_segments[pred_doc]) or whf not in placement:
                    raise SpliceInternalError(
                        f"{shard_name} derived doc {i} pos {p}: would-have-followed "
                        f"segment {whf} does not exist or was never placed — "
                        f"existence invariant broken."
                    )
                derived_doc, derived_pos = placement[whf]
                splices.append({
                    "pos": p,
                    "whf_base": [shard_index, whf[0], whf[1]],
                    "whf_derived": [derived_doc, derived_pos],
                })
            n_splice_positions += len(splices)
            docs_meta.append({
                "doc_id": i,
                "seed_doc_id": refs[0][0] if refs else -1,
                "n_segments": len(refs),
                "n_switches": n_switches,
                "provenance": [[shard_index, ref[0], ref[1]] for ref in refs],
                "segment_token_starts": starts,
                "splices": splices,
            })
    params_out = dict(params)
    params_out["n_splice_positions"] = n_splice_positions
    return {
        "schema_version": 2,
        "shard_name": shard_name,
        "shard_index": shard_index,
        "shard_seed": shard_seed,
        "params": params_out,
        "n_docs": len(result.docs),
        "docs": docs_meta,
    }


# ---------------------------------------------------------------------------
# Per-shard entry point
# ---------------------------------------------------------------------------


def splice_feature_shard(
    base_shard_path: str | Path,
    parquet_path: str | Path,
    tokenizer: Any,
    *,
    model: str,
    cut_fraction: float,
    seed: int,
    shard_index: int,
    shard_name: str,
    segment_size: int,
    context_window: int,
    sae_max_context: int | None,
    profile: SpliceShardProfile | None = None,
    pairing: OppositePercentilePairing | None = None,
) -> tuple[dict, list[str], dict]:
    """Splice one base shard: load it + its parquet, recombine, build the derived
    shard + texts + provenance. Pure (no writes). ``shard_index`` doubles as the
    ``base_shard_idx`` in provenance (within-shard recombination)."""
    owns_total = profile.start_total() if profile is not None else False
    try:
        limit = context_window if sae_max_context is None else min(context_window, sae_max_context)
        if limit % segment_size != 0:
            raise ValueError(
                f"feature-shard splice requires min(context_window, sae_max_context)="
                f"{limit} to be a multiple of segment_size={segment_size}, so no "
                f"recombined doc can exceed the truncation limit; got "
                f"{limit} % {segment_size} = {limit % segment_size}."
            )
        # BOS-prepend contract probe: if the tokenizer advertises a BOS id, verify that
        # tokenizing a single text actually prepends it. This fires BEFORE any I/O so a
        # contract violation is caught loudly on the first shard rather than silently
        # corrupting derived documents. BOS-less tokenizers (bos_id is None) are legal.
        bos_id_probe = resolve_bos_token_id(tokenizer)
        if bos_id_probe is not None:
            probe_ids = _input_ids_tuple(tokenizer("a", padding=False))
            if not probe_ids or probe_ids[0] != bos_id_probe:
                tok_name = str(getattr(tokenizer, "name_or_path", None) or tokenizer.__class__.__name__)
                observed = probe_ids[0] if probe_ids else "<empty>"
                raise TokenizerContractError(
                    f"Tokenizer {tok_name!r} advertises bos_id={bos_id_probe} but "
                    f"tokenizer('a')['input_ids'][0] = {observed!r}. "
                    f"The splice pipeline requires BOS to be prepended by default."
                )
        with profile.time_phase("load_features") if profile is not None else nullcontext():
            base_shard = load_features(base_shard_path)
        base_seg_size = int(base_shard["segment_size"])
        if base_seg_size != segment_size:
            raise ValueError(
                f"{shard_name}: base shard segment_size={base_seg_size} != requested "
                f"segment_size={segment_size}."
            )
        with profile.time_phase("read_parquet") if profile is not None else nullcontext():
            texts = read_parquet_texts(parquet_path)
        with profile.time_phase("load_source_segments") if profile is not None else nullcontext():
            source_segments, load_stats = load_source_segments(
                base_shard, texts, tokenizer, segment_size=segment_size,
                context_window=context_window, sae_max_context=sae_max_context,
                shard_name=shard_name, profile=profile,
            )
        # recombination_seed folds cut_fraction in so different fractions give independent
        # selections even when the global seed and shard index match.
        shard_seed = recombination_seed(seed, shard_index, cut_fraction)
        with profile.time_phase("recombine") if profile is not None else nullcontext():
            result = recombine_segments(
                source_segments, tokenizer, cut_fraction=cut_fraction, shard_seed=shard_seed,
                shard_name=shard_name, limit=limit, profile=profile, pairing=pairing,
            )
        # The engine's overflow split guarantees every piece fits within ``limit``.
        with profile.time_phase("build_derived_shard") if profile is not None else nullcontext():
            derived, derived_texts = build_derived_shard(
                result, source_segments, base_shard, tokenizer, segment_size=segment_size,
                profile=profile,
            )
        logger.info(
            "{}: {} source docs -> {} derived docs "
            "[cuts={} redrawn={} loops={} splits={} "
            "trimmed_docs={} trimmed_tokens={} marker_norm={}]",
            shard_name,
            len(source_segments), len(result.docs),
            result.n_cuts_final, result.n_redrawn, result.n_loops, result.n_splits,
            load_stats.n_trimmed_docs, load_stats.n_trimmed_tokens,
            load_stats.n_marker_normalized_docs,
        )
        params = {
            "model": model,
            "segment_size": segment_size,
            "context_window": context_window,
            "sae_max_context": sae_max_context,
            "cut_fraction": cut_fraction,
            "seed": seed,
            "base_feature_dir": str(Path(base_shard_path).parent),
            "data_dir": str(Path(parquet_path).parent),
            # Engine counters from ShardSpliceResult:
            "n_cuts_final": result.n_cuts_final,
            "n_redrawn": result.n_redrawn,
            "n_loops": result.n_loops,
            "n_splits": result.n_splits,
            # Load-phase counters from SourceLoadStats:
            "n_trimmed_docs": load_stats.n_trimmed_docs,
            "n_trimmed_tokens": load_stats.n_trimmed_tokens,
            "n_marker_normalized_docs": load_stats.n_marker_normalized_docs,
        }
        if pairing is not None:
            params.update(pairing.provenance_params())
        with profile.time_phase("build_provenance") if profile is not None else nullcontext():
            provenance = build_provenance(
                result, source_segments, params=params, shard_index=shard_index,
                shard_name=shard_name, shard_seed=shard_seed, profile=profile,
            )
        return derived, derived_texts, provenance
    finally:
        if profile is not None and owns_total:
            profile.finish_total()
