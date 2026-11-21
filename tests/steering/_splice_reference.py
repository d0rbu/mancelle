"""Independent reference implementation of ``recombine_segments`` for differential testing.

Shares exactly two gate primitives with production:
  - ``WindowedSeamChecker.stitch_is_stable`` -- seam pre-filter
  - ``is_roundtrip_stable``                  -- every full-doc oracle call

Independently re-implements ALL orchestration with naive data structures:
splice-point enumeration, RNG contract replay, cut/stitch bookkeeping via explicit
edge dicts, root-walk chain reading, brute-force cycle detection, loop-break
candidate selection and rotation, overflow split boundary scan-and-slide, piece
ordering, and counter attribution.

Nothing ``_``-prefixed from ``steering.feature_shard_splice`` is imported or called.
The gate primitives are already differentially proven against the full-doc oracle
elsewhere; this reference exists to catch ORCHESTRATION bugs in the engine.
"""

from __future__ import annotations

import random
from collections import deque

from steering._roundtrip_seam import WindowedSeamChecker, is_roundtrip_stable
from steering.feature_shard_splice import (
    PoolExhaustedError,
    SegmentRef,
    SourceSegment,
    SpliceInternalError,
    UnsplittableChainError,
)
from steering.segmentation import resolve_bos_token_id

# ---------------------------------------------------------------------------
# Local helpers (naive, no production internals)
# ---------------------------------------------------------------------------


def _ref_enumerate_splice_points(source_segments: list[list[SourceSegment]]) -> list[tuple[int, int]]:
    """All (doc, boundary) pairs in lexicographic order: boundary in 0..n_segs-2."""
    return [
        (d, b)
        for d in range(len(source_segments))
        for b in range(len(source_segments[d]) - 1)
    ]


def _ref_seg_tokens(source_segments: list[list[SourceSegment]]) -> list[list[tuple[int, ...]]]:
    """Extract token tuples from source segments: ``[doc][seg] -> tokens``."""
    return [[seg.tokens for seg in doc] for doc in source_segments]


def _ref_piece_ids(
    piece: list[SegmentRef],
    new_started: bool,
    seg_tokens: list[list[tuple[int, ...]]],
    bos_id: int,
) -> tuple[int, ...]:
    """Concatenated token ids for a piece, BOS prepended iff new_started.

    Independently reimplements ``_piece_ids`` -- does NOT call the private helper.
    """
    ids = tuple(tid for ref in piece for tid in seg_tokens[ref[0]][ref[1]])
    return ((bos_id,) + ids) if new_started else ids


def _ref_cut_bounds(cuts: list[tuple[int, int]]) -> dict[int, list[int]]:
    """Per-doc sorted list of cut boundaries under the current cut set."""
    by_doc: dict[int, list[int]] = {}
    for d, b in cuts:
        by_doc.setdefault(d, []).append(b)
    for lst in by_doc.values():
        lst.sort()
    return by_doc


def _ref_seam_context(
    slot: int,
    cuts: list[tuple[int, int]],
    sigma: list[int],
    seg_tokens: list[list[tuple[int, ...]]],
    bos_id: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Run-local, BOS-led (prefix, suffix) ids for the stitch at ``slot``.

    Replicates the production seam-context derivation exactly:

    prefix = tokens from X's run start through X = cuts[slot], BOS-led when
    run_start > 0 (mid-doc run).  Runs are maximal cut-free source intervals
    under the CURRENT cuts list.

    suffix = tokens from Y through Y's run end, where
    Y = (cuts[sigma[slot]].doc, cuts[sigma[slot]].boundary + 1).

    Called fresh each pre-filter iteration so it always sees the current cuts.
    """
    cut_bounds = _ref_cut_bounds(cuts)

    # --- prefix: X's run through cuts[slot] ---
    d, b = cuts[slot]
    run_start = 0
    for c in cut_bounds.get(d, []):
        if c < b:
            run_start = c + 1  # this cut is earlier in the same doc: advance run start
        else:
            break  # first cut >= b: run extends back to run_start
    prefix = tuple(tid for s in range(run_start, b + 1) for tid in seg_tokens[d][s])
    if run_start > 0:
        # Mid-doc run: the real BOS is not present; prepend a synthetic one so
        # WindowedSeamChecker.stitch_is_stable sees a BOS-led context on the left.
        prefix = (bos_id, *prefix)

    # --- suffix: Y's run from Y onward ---
    dp, bp = cuts[sigma[slot]]
    y0 = bp + 1  # first segment after the partner's cut boundary
    run_end = len(seg_tokens[dp]) - 1  # default: end of document
    for c in cut_bounds.get(dp, []):
        if c >= y0:
            run_end = c  # next cut in the same doc cuts the run here
            break
    suffix = tuple(tid for s in range(y0, run_end + 1) for tid in seg_tokens[dp][s])
    return prefix, suffix


def _ref_prefilter(
    cuts: list[tuple[int, int]],
    sigma: list[int],
    seg_tokens: list[list[tuple[int, ...]]],
    checker: WindowedSeamChecker,
    replace_fn,
    bos_id: int,
) -> None:
    """Windowed pre-filter over all k stitch seams.

    On an unstable seam at slot i: call replace_fn(i) then re-queue both i
    (prefix context changed) and inverse[i] (suffix context changed).  Loops until
    every seam passes; replace_fn raises PoolExhaustedError on exhaustion.
    """
    k = len(cuts)
    # inverse[i] = j such that sigma[j] == i (slot j uses slot i as stitch target).
    inverse = [0] * k
    for j in range(k):
        inverse[sigma[j]] = j
    pending: deque[int] = deque(range(k))
    queued = [True] * k
    while pending:
        i = pending.popleft()
        queued[i] = False
        # Recompute context fresh each time (cuts may have changed via replace_fn).
        prefix, suffix = _ref_seam_context(i, cuts, sigma, seg_tokens, bos_id)
        if checker.stitch_is_stable(prefix, suffix):
            continue
        replace_fn(i)  # redraw: modifies cuts[i] in-place via pool.popleft()
        for touched in (i, inverse[i]):
            if not queued[touched]:
                queued[touched] = True
                pending.append(touched)


def _ref_build_successors(
    source_segments: list[list[SourceSegment]],
    cuts: list[tuple[int, int]],
    sigma: list[int],
) -> dict[SegmentRef, SegmentRef | None]:
    """Source adjacencies minus cut edges, plus stitch edges X(cuts[i]) -> Y(cuts[sigma[i]]).

    Independently reimplements ``_build_successors``.
    """
    succ: dict[SegmentRef, SegmentRef | None] = {}
    # Default: natural source adjacency.
    for d, segs in enumerate(source_segments):
        for j in range(len(segs)):
            succ[(d, j)] = (d, j + 1) if j + 1 < len(segs) else None
    # Overwrite cut edges with stitch edges.
    for i, (d, b) in enumerate(cuts):
        dp, bp = cuts[sigma[i]]
        succ[(d, b)] = (dp, bp + 1)  # stitch: X -> Y (tail of partner's cut)
    return succ


def _ref_read_chains_loops(
    source_segments: list[list[SourceSegment]],
    succ: dict[SegmentRef, SegmentRef | None],
) -> tuple[list[list[SegmentRef]], list[list[SegmentRef]]]:
    """Walk each root (d, 0) to build chains; all unvisited refs form pure cycles.

    Independently reimplements ``_read_chains``.

    Why unvisited = pure cycles: the successor function is injective (each ref has
    at most one successor), stitches target ordinal >= 1 so (d, 0) is never a stitch
    target, and every (d, 0) is visited by its root-walk. Therefore unvisited refs
    have no predecessor from any root, and the injective successor graph restricted
    to them is a disjoint union of cycles.
    """
    visited: set[SegmentRef] = set()
    chains: list[list[SegmentRef]] = []
    for d in range(len(source_segments)):
        cur: SegmentRef | None = (d, 0)
        chain: list[SegmentRef] = []
        while cur is not None:
            visited.add(cur)
            chain.append(cur)
            cur = succ[cur]
        chains.append(chain)

    # Detect cycles: walk from each lexicographically-smallest unvisited node.
    loops: list[list[SegmentRef]] = []
    seen: set[SegmentRef] = set(visited)
    for start in sorted(set(succ) - visited):
        if start in seen:
            continue  # already absorbed into a previously found cycle
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


def _ref_break_loop(
    cyc: list[SegmentRef],
    seg_tokens: list[list[tuple[int, ...]]],
    tokenizer,
    bos_id: int,
) -> list[SegmentRef] | None:
    """Try each stitch-entry candidate in ascending ref order; return first stable rotation.

    A ref r = cyc[i] is a stitch candidate when its cycle-predecessor edge is a
    stitch (pred != (r[0], r[1]-1)).  Rotate the cycle to start at r and check
    ``is_roundtrip_stable((bos_id,) + concat_ids)``.  Returns None if none pass.

    Independently reimplements ``_break_loop``.
    """
    n = len(cyc)
    candidates: list[tuple[int, SegmentRef]] = []
    for i in range(n):
        ref = cyc[i]
        pred = cyc[(i - 1) % n]
        # Edge pred->ref is a stitch iff pred is not the natural predecessor of ref.
        if pred != (ref[0], ref[1] - 1):
            candidates.append((i, ref))
    candidates.sort(key=lambda x: x[1])  # ascending by (doc, seg)
    for start_idx, _ in candidates:
        rotation = cyc[start_idx:] + cyc[:start_idx]
        concat_ids = tuple(
            tid for ref in rotation for tid in seg_tokens[ref[0]][ref[1]]
        )
        if is_roundtrip_stable((bos_id,) + concat_ids, tokenizer):
            return rotation
    return None


def _ref_lowest_cut_slot_in_cycle(
    cyc: list[SegmentRef], cuts: list[tuple[int, int]]
) -> int:
    """Lowest-numbered cut slot whose cut point lies in the cycle.

    Independently reimplements ``_lowest_cut_slot_in_cycle``.
    """
    cyc_set = set(cyc)
    involved = [i for i in range(len(cuts)) if cuts[i] in cyc_set]
    if not involved:
        raise SpliceInternalError(f"cycle {cyc} contains no cut point")
    return min(involved)


def _ref_resolve_loops(
    loops: list[list[SegmentRef]],
    cuts: list[tuple[int, int]],
    seg_tokens: list[list[tuple[int, ...]]],
    tokenizer,
    bos_id: int,
) -> tuple[list[tuple[list[SegmentRef], bool]], int | None]:
    """Break every loop into a fresh new_started doc; return (loop_docs, culprit).

    On the first unbreakable loop, returns ([], culprit_slot) where culprit is that
    loop's lowest-numbered cut slot.  Otherwise returns all broken loop docs sorted
    by their lowest involved cut ref, with culprit=None.

    Independently reimplements ``_resolve_loops``.
    """
    resolved: list[tuple[SegmentRef, list[SegmentRef]]] = []
    for cyc in loops:
        broken = _ref_break_loop(cyc, seg_tokens, tokenizer, bos_id)
        if broken is None:
            return [], _ref_lowest_cut_slot_in_cycle(cyc, cuts)
        cyc_set = set(cyc)
        involved_refs = [cuts[i] for i in range(len(cuts)) if cuts[i] in cyc_set]
        resolved.append((min(involved_refs), broken))
    resolved.sort(key=lambda kv: kv[0])
    loop_docs: list[tuple[list[SegmentRef], bool]] = [
        (refs, True) for _sort_key, refs in resolved
    ]
    return loop_docs, None


def _ref_split_overflow(
    piece: list[SegmentRef],
    new_started: bool,
    seg_tokens: list[list[tuple[int, ...]]],
    tokenizer,
    bos_id: int,
    limit: int,
) -> list[tuple[list[SegmentRef], bool]]:
    """Split an overflowing piece into sub-pieces each fitting within ``limit`` tokens.

    Algorithm (mirrors production ``_split_overflow``):
      1. While piece > limit: find maximal b in [1, n-1] s.t. kept-token-count <= limit.
      2. Slide b earlier while the double gate fails:
           is_roundtrip_stable(kept_ids) AND is_roundtrip_stable((bos,)+rem_raw)
      3. b == 0 -> UnsplittableChainError.
      4. Emit (kept, current_new_started); continue with (remainder, True).

    Independently reimplements ``_split_overflow``.
    """
    result: list[tuple[list[SegmentRef], bool]] = []
    current_piece = piece
    current_ns = new_started

    while True:
        ids = _ref_piece_ids(current_piece, current_ns, seg_tokens, bos_id)
        if len(ids) <= limit:
            result.append((current_piece, current_ns))
            break

        n = len(current_piece)
        bos_overhead = 1 if current_ns else 0
        # Find the maximal b such that kept-token-count (with BOS overhead) <= limit.
        b = 0
        running = bos_overhead
        for i, ref in enumerate(current_piece):
            seg_len = len(seg_tokens[ref[0]][ref[1]])
            if running + seg_len > limit:
                break
            running += seg_len
            b = i + 1
        b = min(b, n - 1)  # must leave at least one ref for the remainder

        # Slide b earlier until the double gate passes (kept stable AND bos+rem stable).
        while b > 0:
            kept = current_piece[:b]
            remainder = current_piece[b:]
            kept_ids = _ref_piece_ids(kept, current_ns, seg_tokens, bos_id)
            rem_raw = tuple(
                tid for ref in remainder for tid in seg_tokens[ref[0]][ref[1]]
            )
            rem_ids = (bos_id,) + rem_raw
            if (
                is_roundtrip_stable(kept_ids, tokenizer)
                and is_roundtrip_stable(rem_ids, tokenizer)
            ):
                break
            b -= 1

        if b == 0:
            raise UnsplittableChainError(
                f"no stable split boundary found in piece starting at {current_piece[0]}"
            )

        result.append((current_piece[:b], current_ns))
        current_piece = current_piece[b:]
        current_ns = True  # remainder always gets a fresh BOS

    return result


def _ref_first_unstable_piece(
    pieces: list[tuple[list[SegmentRef], bool]],
    source_segments: list[list[SourceSegment]],
    seg_tokens: list[list[tuple[int, ...]]],
    tokenizer,
    bos_id: int,
) -> tuple[list[SegmentRef], bool] | None:
    """Return the first piece failing the full-doc oracle, or None if all pass.

    Skips identity pieces (not new_started AND refs == the full source doc) whose
    stability was proven in load_source_segments.

    Independently reimplements ``_first_unstable_piece`` (without batching).
    """
    identity = {
        tuple((d, j) for j in range(len(doc)))
        for d, doc in enumerate(source_segments)
    }
    for refs, new_started in pieces:
        if (not new_started) and tuple(refs) in identity:
            continue  # provably stable; skip the oracle call
        ids = _ref_piece_ids(refs, new_started, seg_tokens, bos_id)
        if not is_roundtrip_stable(ids, tokenizer):
            return (refs, new_started)
    return None


def _ref_first_cut_slot(
    bad: tuple[list[SegmentRef], bool],
    cuts: list[tuple[int, int]],
) -> int:
    """Map the failing piece's earliest stitched junction to its cut slot.

    A junction (a, b) is stitched when b != (a[0], a[1]+1).  The earliest such
    a is a live cut point.  Returns the slot i with cuts[i] == a.

    Independently reimplements ``_first_cut_slot_in_piece``.
    """
    refs, _ns = bad
    for m in range(len(refs) - 1):
        a, b = refs[m], refs[m + 1]
        if b != (a[0], a[1] + 1):  # stitched junction
            try:
                return cuts.index(a)
            except ValueError as exc:
                raise SpliceInternalError(
                    f"stitched junction prefix {a} is not a live cut"
                ) from exc
    raise SpliceInternalError(
        f"failing piece {refs} contains no stitched junction"
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def reference_recombine(
    source_segments: list[list[SourceSegment]],
    tokenizer,
    *,
    cut_fraction: float,
    shard_seed: int,
    limit: int,
) -> tuple[list[tuple[tuple[SegmentRef, ...], bool]], dict[str, int]]:
    """Independent reference implementation of ``recombine_segments``.

    Returns ``(docs, counters)`` where:
      ``docs``     = list of ``(segment_refs, new_started)`` tuples in output order
      ``counters`` = ``{"n_cuts_final": ..., "n_redrawn": ..., "n_loops": ..., "n_splits": ...}``

    Output order: root chains in root order (each followed by its overflow remainders
    inline), then loop docs by lowest involved cut ref (each followed by its remainders).

    RNG contract (pinned, matches production exactly):
      - target_k < 2 -> identity short-circuit before any RNG draw.
      - bos_id = resolve_bos_token_id(tokenizer); raise on None.
      - rng = random.Random(shard_seed).
      - One selection shuffle: order = list(range(n_sp)); rng.shuffle(order);
          cuts = [all_sp[i] for i in order[:target_k]]; pool = deque(order[target_k:]).
      - Derangement by rejection: fresh sigma = list(range(k)); rng.shuffle(sigma)
          per attempt until all(sigma[i] != i for i in range(k)).
      - Redraws: pool.popleft() only, no rng.

    Gate primitives shared with production:
      - WindowedSeamChecker.stitch_is_stable (seam pre-filter)
      - is_roundtrip_stable (all full-doc oracle calls)
    """
    seg_tokens = _ref_seg_tokens(source_segments)
    all_sp = _ref_enumerate_splice_points(source_segments)
    target_k = round(cut_fraction * len(all_sp))

    # Identity short-circuit: derangement requires k >= 2.
    if target_k < 2:
        docs_out: list[tuple[tuple[SegmentRef, ...], bool]] = [
            (tuple((d, j) for j in range(len(doc))), False)
            for d, doc in enumerate(source_segments)
        ]
        return docs_out, {"n_cuts_final": 0, "n_redrawn": 0, "n_loops": 0, "n_splits": 0}

    bos_id = resolve_bos_token_id(tokenizer)
    if bos_id is None:
        raise SpliceInternalError("tokenizer has no BOS; cannot form new documents")

    # One selection shuffle.
    rng = random.Random(shard_seed)
    order = list(range(len(all_sp)))
    rng.shuffle(order)
    sel_idx = order[:target_k]
    pool: deque[int] = deque(order[target_k:])
    cuts: list[tuple[int, int]] = [all_sp[i] for i in sel_idx]

    # Derangement by rejection sampling (uniform random derangement).
    sigma: list[int] = list(range(target_k))
    rng.shuffle(sigma)
    while any(sigma[i] == i for i in range(target_k)):
        sigma = list(range(target_k))
        rng.shuffle(sigma)

    checker = WindowedSeamChecker(tokenizer)
    n_redrawn = 0

    def _replace(slot: int) -> None:
        """Redraw slot from pool (no RNG); raise PoolExhaustedError if empty."""
        nonlocal n_redrawn
        if not pool:
            raise PoolExhaustedError(
                f"redraw pool exhausted after {n_redrawn} redraws"
            )
        cuts[slot] = all_sp[pool.popleft()]
        n_redrawn += 1

    loop_docs: list[tuple[list[SegmentRef], bool]] = []
    chains: list[list[SegmentRef]] = []
    pieces: list[tuple[list[SegmentRef], bool]] = []

    while True:  # outer fixpoint
        # Phase A: windowed pre-filter over all k stitch seams.
        _ref_prefilter(cuts, sigma, seg_tokens, checker, _replace, bos_id)

        # Phase B: build successor graph and read chains/loops.
        succ = _ref_build_successors(source_segments, cuts, sigma)
        chains, loops = _ref_read_chains_loops(source_segments, succ)

        # Phase C: break loops into new_started docs (oracle-gated).
        loop_docs, culprit = _ref_resolve_loops(loops, cuts, seg_tokens, tokenizer, bos_id)
        if culprit is not None:
            _replace(culprit)
            continue  # restart fixpoint with the redrawn cut

        # Phase D: overflow-split every piece; remainders are new_started.
        pieces = []
        for chain in chains:
            pieces.extend(
                _ref_split_overflow(chain, False, seg_tokens, tokenizer, bos_id, limit)
            )
        for refs, new_started in loop_docs:
            pieces.extend(
                _ref_split_overflow(refs, new_started, seg_tokens, tokenizer, bos_id, limit)
            )

        # Phase E: full-doc oracle over all non-identity pieces.
        bad = _ref_first_unstable_piece(
            pieces, source_segments, seg_tokens, tokenizer, bos_id
        )
        if bad is None:
            break  # every piece is stable -> fixpoint reached
        _replace(_ref_first_cut_slot(bad, cuts))

    n_loops = len(loop_docs)
    n_splits = len(pieces) - len(chains) - n_loops
    docs_result: list[tuple[tuple[SegmentRef, ...], bool]] = [
        (tuple(refs), ns) for refs, ns in pieces
    ]
    counters: dict[str, int] = {
        "n_cuts_final": len(cuts),
        "n_redrawn": n_redrawn,
        "n_loops": n_loops,
        "n_splits": n_splits,
    }
    return docs_result, counters
