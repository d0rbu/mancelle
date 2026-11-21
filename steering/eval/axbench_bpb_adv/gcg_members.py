"""Convert BpbAttackPrefix → ClassMarginMember list for the feature_gcg optimizer.

Each ``ClassMarginMember`` holds a ``(prefix, rich_i, avoid_j)`` triple
teacher-forced under the same steered intervention; ``build_class_margin_members``
produces one member per Cartesian ``(rich_i, avoid_j)`` pair so the optimizer can
minibatch them freely.

Shift convention matched to teacher_force.py score_sample (lines 742-774)
--------------------------------------------------------------------------
For a (prefix_tokens, output_tokens) pair:

    all_tokens  = prefix_tokens + output_tokens
    dec_input   = all_tokens[:-1]          # (1, T-1), T = n_prefix + n_output
    dec_target  = all_tokens[1:]           # (1, T-1), with:
                    positions 0..n_prefix-2  → -1   (CE ignored; prefix→prefix preds)
                    positions n_prefix-1..T-2 → real token IDs (output scored)

This replicates the ``build_output_only_mask(n_full_prefix, n_output)`` mask
(teacher_force.py:81: ``mask[n_full_prefix - 1:] = True``), encoding it directly
in ``dec_target`` so the objective's ``_masked_mean_scalar(dec_target != -1)``
selects the same positions the eval scores.

seg_map convention:
    positions 0..n_prefix-1  → context_slot (prefix tokens in dec_input)
    positions n_prefix..T-2  → generation_slot (output tokens in dec_input)

After ``seg_map[:, :-1]`` (teacher_force.py:774), output tokens ``[0..n_output-2]``
occupy positions ``n_prefix..T-2``.  When ``n_output == 1`` the range is empty;
the gradient from the lone scored position (n_prefix-1, context-conditioned) w.r.t.
the generation-slot bag is zero — correct by construction: that decoder position
attends context, not generation conditioning.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from loguru import logger

from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.gradient import FeatureGradMember
from steering.conditional.feature_gcg.objective import ClassMarginMember
from steering.eval.axbench_bpb.seg_map import build_multi_segment_seg_map
from steering.eval.axbench_bpb.vocab_restriction import restrict_segment_conditions_to_vocab
from steering.eval.axbench_bpb_adv.attack_rows import BpbAttackPrefix
from steering.eval.target_conditioning import ContextFeatureSegment


def _build_member(
    prefix_tokens: list[int],
    output_tokens: list[int],
    *,
    segment_conditions: Sequence[FeatureConditioning],
    prefix_segments: tuple[ContextFeatureSegment, ...],
    tokenizer: Any,
    generation_slot: int,
    device: torch.device,
) -> FeatureGradMember:
    """Build one FeatureGradMember for a single (prefix, output) pair.

    Matches the shift and masking convention from teacher_force.py score_sample
    lines 742-774 and build_output_only_mask line 81, and — crucially — builds
    the seg_map with the SAME per-segment router the score stage uses
    (:func:`build_multi_segment_seg_map`), so a multi-segment prefix's tokens are
    assigned to their OWN context slots ``[0, len(prefix_segments))`` (via
    char-overlap against the context-cache segment spans) rather than collapsed
    into one slot. Output positions all map to ``generation_slot``.
    """
    n_prefix = len(prefix_tokens)
    all_tokens = prefix_tokens + output_tokens

    # Shift-by-1 (teacher_force.py L744-L745)
    dec_input = torch.tensor([all_tokens[:-1]], dtype=torch.long, device=device)
    dec_target = torch.tensor([all_tokens[1:]], dtype=torch.long, device=device)

    # Mask prefix positions in dec_target with -1 (model ignore_index=-1).
    # build_output_only_mask sets mask[n_full_prefix-1:]=True, i.e. the first
    # n_prefix-1 positions are NOT scored → set to -1 here.
    # (teacher_force.py build_output_only_mask L81: mask[n_full_prefix - 1:] = True)
    if n_prefix > 1:
        dec_target[0, : n_prefix - 1] = -1

    # Per-segment seg_map, IDENTICAL to the score-stage router
    # (teacher_force._build_encdec_seg_map_and_encoder -> build_multi_segment_seg_map):
    # prefix positions 0..n_prefix-1 -> their context segment slot via char-overlap,
    # output positions n_prefix.. -> generation_slot. The full seg_map has width
    # n_prefix + n_output; trim the last column ([:, :-1]) exactly as scoring does
    # (teacher_force.py L774) so it aligns with dec_input = all_tokens[:-1].
    full_seg_map = build_multi_segment_seg_map(
        tokenizer=tokenizer,
        prefix_tokens=prefix_tokens,
        prefix_segments=prefix_segments,
        n_output=len(output_tokens),
        generation_slot=generation_slot,
    )
    seg_map = full_seg_map[:, :-1].to(device)
    t_len = len(all_tokens) - 1  # == n_prefix + n_output - 1
    if seg_map.shape[1] != t_len:
        raise FeatureGcgError(
            f"build_class_margin_members: seg_map width {seg_map.shape[1]} != "
            f"expected {t_len} (n_prefix={n_prefix}, n_output={len(output_tokens)}); "
            "the per-segment router must produce a (1, n_prefix + n_output) seg_map "
            "whose [:, :-1] trim aligns with the shifted dec_input."
        )

    return FeatureGradMember(
        dec_input=dec_input,
        dec_target=dec_target,
        segment_conditions=list(segment_conditions),
        seg_map=seg_map,
        generation_slot=generation_slot,
    )


def build_class_margin_members(
    prefix: BpbAttackPrefix,
    *,
    prefix_segments: tuple[ContextFeatureSegment, ...],
    tokenizer: Any,
    max_seq_len: int,
    device: torch.device,
    allowed_ids: torch.Tensor | None = None,
) -> list[ClassMarginMember]:
    """Convert a BpbAttackPrefix into ClassMarginMember list (full Cartesian product).

    One ``ClassMarginMember`` is produced per ``(rich_i, avoid_j)`` pair.

    Per-segment context conditioning (Phase 3)
    ------------------------------------------
    ``prefix_segments`` are the prefix's REAL context-cache segments (the
    ``ContextFeatureSegment`` tuple the score stage loads for this prefix).  Each
    member's ``segment_conditions`` is ``len(prefix_segments) + 1`` bags:

      * one FIXED context bag per segment, built from that segment's extracted
        ``{feature_id: bucket}`` — EXACTLY the conditioning
        ``teacher_force._build_encdec_seg_map_and_encoder`` feeds the context
        slots at score time (``prefix_segment_conditions``);
      * one empty generation bag at index ``len(prefix_segments)`` — the ONLY bag
        the GCG optimizer substitutes into (``generation_slot``).

    The seg_map is built by the SAME router the score stage uses
    (:func:`build_multi_segment_seg_map`, via :func:`_build_member`), so a
    multi-segment prefix's tokens are assigned to their own context slots and the
    derive optimizes under the identical layout scoring applies the bag under.
    The construction otherwise matches the ``teacher_force.py score_sample`` shift
    convention (lines 742-774) so the optimizer scores the SAME output tokens the
    eval scores.

    This mirrors the NO-prompt-prefix branch of
    ``teacher_force._build_encdec_seg_map_and_encoder`` only (``generation_slot =
    len(prefix_segments)``, no reserved empty slot 0).  That is the sole reachable
    branch here: feature-GCG scoring routes through ``Intervention.ours`` which
    carries NO ``prompt_prefix_tokens`` (only the ``prompting`` baseline family
    sets them), so ``n_prompt_surviving`` is always 0.  Were feature-GCG ever
    combined with a prompt prefix, the score stage would take the shifted branch
    (``generation_slot = 1 + len(prefix_segments)``, empty slot 0) and this
    layout would diverge — a mutual exclusivity the method taxonomy currently
    guarantees.

    Args:
        prefix: BpbAttackPrefix carrying prefix_tokens, concept_rich_output_tokens,
            and concept_avoiding_output_tokens.
        prefix_segments: The prefix's context-cache segments (>= 1).  Their count
            sets ``num_segments = len(prefix_segments) + 1`` and their extracted
            features become the FIXED context bags.
        tokenizer: The decoder (nanochat) tokenizer ``build_multi_segment_seg_map``
            uses to align ``prefix.prefix_tokens`` to the segment char-spans.
        max_seq_len: The encdec decoder's ``sequence_len`` — the SAME window the
            score stage left-truncates against (``encdec_dec_sequence_len``).
            The derive does not mirror score-time left-truncation, so a
            ``(prefix, completion)`` whose combined length exceeds this window
            RAISES rather than optimizing under an untruncated layout the score
            stage would truncate (a silent derive/score layout divergence).
        device: Torch device for tensor construction.
        allowed_ids: The checkpoint's compacted vocabulary ``V`` (original SAE
            feature ids, int64, sorted ascending), or ``None`` for an
            uncompacted checkpoint (see
            ``steering.eval.axbench_bpb.vocab_restriction.feature_vocab_allowed_ids``).
            Restricts each FIXED per-segment context bag to ``V`` — mirroring
            ``teacher_force._build_encdec_seg_map_and_encoder``'s context-cache
            filter (``teacher_force.py:306-340``) — via
            ``vocab_restriction.restrict_segment_conditions_to_vocab``: an id
            outside ``V`` is dropped, with ONE aggregated ``logger.warning``
            naming the total dropped count across every segment (not one per
            segment), and a segment fully emptied by the filter raises
            ``AxBenchBpbError``.  The empty generation bag (the only bag the
            GCG optimizer substitutes into) is NEVER restricted here — the
            method's own bag is already restricted upstream of wherever it is
            threaded in.  ``None`` is a strict no-op, byte-identical to before
            this parameter existed.

    Returns:
        List of ``ClassMarginMember`` — one per ``(rich_i, avoid_j)`` Cartesian pair
        (always non-empty: ``pair_samples_into_prefixes`` guarantees each prefix
        carries >= 1 rich and >= 1 avoid completion).

    Raises:
        FeatureGcgError: ``prefix_segments`` is empty (the context cache must
            produce >= 1 segment), or a ``(prefix, completion)`` pair would
            exceed ``max_seq_len`` (score-time left-truncation is not mirrored —
            deferred). ``BpbAttackPrefix`` construction already guarantees at
            least one rich and one avoiding completion.
        AxBenchBpbError: a context segment had >= 1 feature id and every one of
            them fell outside ``allowed_ids`` (see above).
    """
    n_context = len(prefix_segments)
    if n_context < 1:
        raise FeatureGcgError(
            "build_class_margin_members: prefix (core_prefix_id="
            f"{prefix.core_prefix_id!r}) has zero context segments; the context "
            "cache must produce >= 1 segment per prefix. A missing/empty segment "
            "set is a real pipeline bug, not a skippable case."
        )

    rich_completions = prefix.concept_rich_output_tokens
    avoid_completions = prefix.concept_avoiding_output_tokens

    # Score-time left-truncation guard. score_sample left-truncates full_prefix
    # to fit (len(prefix) + n_output) <= max_seq_len before teacher-forcing
    # (teacher_force.py truncate_prefix_left). This derive builds members on the
    # UNtruncated prefix, so a (prefix, completion) whose combined length exceeds
    # max_seq_len would be optimized under a longer layout than the score stage
    # applies the bag under — and, worse, truncation drops/re-projects context
    # segments, desyncing the fixed context bags below. Rather than silently
    # diverge, RAISE: real corpora keep prefix+completion well under the window
    # (measured max 769 vs sequence_len 1024 on clean10_l12), so this fires only
    # on out-of-distribution data that genuinely needs truncation-mirroring
    # (deferred). Checking the longest completion covers every pair.
    n_prefix = len(prefix.prefix_tokens)
    longest_completion = max(
        len(c) for c in (*rich_completions, *avoid_completions)
    )
    if n_prefix + longest_completion > max_seq_len:
        raise FeatureGcgError(
            "build_class_margin_members: prefix (core_prefix_id="
            f"{prefix.core_prefix_id!r}) + its longest completion "
            f"({n_prefix} + {longest_completion} = {n_prefix + longest_completion} "
            f"tokens) exceeds max_seq_len={max_seq_len}. The score stage would "
            "left-truncate this prefix, but the derive does not mirror "
            "score-time left-truncation, so the optimized bag would not match "
            "the scored layout. Multi-segment truncation-mirroring is deferred; "
            "this raises rather than silently optimizing a divergent member."
        )

    # segment_conditions: N FIXED context bags (the extracted features per segment,
    # mirroring teacher_force's prefix_segment_conditions) + 1 empty generation bag.
    # generation_slot is the slot AFTER the context segments (== n_context), so the
    # encoder has n_context + 1 slots [0..n_context]; the optimizer substitutes the
    # shared bag ONLY into segment_conditions[generation_slot]. Context bags are
    # FIXED (never optimized), exactly as scoring conditions the context segments.
    generation_slot = n_context
    raw_context_conditions = [
        {int(f): int(b) for f, b in zip(seg.feature_ids, seg.buckets, strict=True)}
        for seg in prefix_segments
    ]
    # allowed_ids restricts each FIXED context bag to the checkpoint's compacted
    # vocabulary V, mirroring teacher_force._build_encdec_seg_map_and_encoder's
    # context-cache filter (teacher_force.py:306-340) at the encoder-input
    # boundary: an id outside V is dropped, with ONE aggregated warning below
    # (not one per segment), and a segment fully emptied BY the filter raises.
    # allowed_ids=None (an uncompacted checkpoint) is a strict no-op -- the
    # returned list is the SAME object, byte-identical to before this parameter
    # existed.
    restricted_context_conditions, total_dropped = restrict_segment_conditions_to_vocab(
        raw_context_conditions, allowed_ids,
    )
    if total_dropped > 0:
        logger.warning(
            "build_class_margin_members: dropped "
            f"{total_dropped} context feature id(s) outside the checkpoint's "
            f"compacted vocabulary V, across {len(prefix_segments)} prefix "
            f"segment(s) (core_prefix_id={prefix.core_prefix_id!r})."
        )
    context_conditions = [
        FeatureConditioning.from_dict(bag) for bag in restricted_context_conditions
    ]
    segment_conditions = [*context_conditions, FeatureConditioning(entries=())]

    rich_members = [
        _build_member(
            prefix.prefix_tokens,
            rich_tokens,
            segment_conditions=segment_conditions,
            prefix_segments=prefix_segments,
            tokenizer=tokenizer,
            generation_slot=generation_slot,
            device=device,
        )
        for rich_tokens in rich_completions
    ]
    avoid_members = [
        _build_member(
            prefix.prefix_tokens,
            avoid_tokens,
            segment_conditions=segment_conditions,
            prefix_segments=prefix_segments,
            tokenizer=tokenizer,
            generation_slot=generation_slot,
            device=device,
        )
        for avoid_tokens in avoid_completions
    ]

    return [
        ClassMarginMember(rich=r, avoid=a)
        for r in rich_members
        for a in avoid_members
    ]
