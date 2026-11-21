"""Raw-source prompt preparation without importing routing into its cache types."""

from collections.abc import Sequence

import torch as t
from phantom.sized import NonEmptyStr

from steering.conditional.encoder_decoder.full_sequence_routing import (
    build_full_sequence_segment_routing,
    source_aligned_char_offsets,
)
from steering.eval.target_conditioning import ContextFeatureSegment, PreparedTurnContext
from steering.segmentation import align_char_ranges_to_decoded_prefix
from steering.sft.turn_spans import ChatSpecialTokens, legacy_turn_spans

_PROMPT_ADMISSION_ALIGNMENT_COUNTER: NonEmptyStr = NonEmptyStr.parse(
    "segmentation.char_end_extended.chatcore_prompt_admission"
)


def turn_target_start_position(context_token_ids: Sequence[int], tokenizer) -> int:
    """Validate the independently rendered prefix and return its assistant marker."""
    specials = ChatSpecialTokens.from_tokenizer(tokenizer)
    spans = legacy_turn_spans(context_token_ids, specials)
    p0 = len(context_token_ids) - 1
    if context_token_ids[-1] != specials.assistant_start or spans[-1].start != p0 - 1:
        raise ValueError("turn target-real requires a marker-only terminal <assistant_start> turn")
    return p0


def prepare_turn_context(
    token_ids: Sequence[int], segments: tuple[ContextFeatureSegment, ...], tokenizer,
    *, source_bos_char_len: int,
) -> PreparedTurnContext:
    """Keep the raw source slots referenced by inputs before the assistant marker.

    Gemma merges ``><`` at nanochat turn boundaries: the marker's initial ``<``
    belongs to the preceding source window, so character equality cannot decide
    ownership. Route the whole prefix exactly as conditioned SFT routes the
    marker (``scripts/train/conditional_encdec_sft.py:3393-3402``), then retain
    every ordinal through the last slot used before it. The marker never shares
    that slot in real data: the merge puts it in the window after the straddle,
    so its own slot is ``keep`` or ``keep + 1`` in the probe below. Drop only
    marker-exclusive and unused tail slots; keep internal unused ordinals and
    the marker token in the exact cache lookup key.

    The stage-6 real-data probe found 2,203 of 2,420 prefixes dropping one
    inside-marker slot (marker_slot == keep), and 217 dropping an unrouted
    straddle plus an inside-marker slot (marker_slot == keep + 1). For example,
    p0=62, marker_start=261, keep=2, marker_slot=3 with raw windows
    (0,140), (140,256), (256,262), (262,278): no decoder token uses slot 2.
    After bounded source/decoder alignment, every dropped surviving slot must
    end after marker_start. Rejecting an unused window ending at or before it
    is an additional ChatCORE admission restriction, not an SFT invariant:
    the SFT routing helper accepts that geometry. The 2,203-row example has p0=116,
    marker_start=460, keep=marker_slot=4 and final windows (118,238),
    (238,354), (354,461), (461,477).
    """
    p0 = turn_target_start_position(token_ids, tokenizer)
    offsets = source_aligned_char_offsets(tokenizer, token_ids, source_bos_char_len=source_bos_char_len)
    marker_start = offsets[p0]
    routing = build_full_sequence_segment_routing(
        ids=token_ids, tokenizer=tokenizer,
        seg_features=([t.tensor(seg.feature_ids, dtype=t.int64) for seg in segments],
                      [t.tensor(seg.buckets, dtype=t.int64) for seg in segments],
                      [seg.char_start for seg in segments], [seg.char_end for seg in segments]),
        source_bos_char_len=source_bos_char_len,
    )
    # ``per_seg_fids`` is built ``max(assignments) + 2`` long, so a slot index
    # at or past its length is unreachable; only unrouted (zero) can occur.
    if bool((routing.seg_map <= 0).any()):
        raise ValueError("turn prefix contains unrouted source slots")
    assignments = routing.seg_map - 1
    # Monotone cummax assignments with no unrouted position put the marker at
    # or after ``assignments[p0 - 1] == keep - 1``, never before it.
    keep = int(assignments[:p0].max().item()) + 1
    if bool(((assignments[:p0] < 0) | (assignments[:p0] >= keep)).any()):
        raise ValueError("turn prompt routing is outside its retained source slots")
    # The routing helper validates raw contiguity before bounded alignment.
    # Apply the admission restriction only to aligned survivors: extraction
    # may leave tolerated residue after the marker, including whole tail slots.
    _, aligned_ends, _ = align_char_ranges_to_decoded_prefix(
        char_starts=[seg.char_start for seg in segments],
        char_ends=[seg.char_end for seg in segments], decoded_len=offsets[-1],
        counter_name=_PROMPT_ADMISSION_ALIGNMENT_COUNTER,
    )
    if any(end <= marker_start for end in aligned_ends[keep:]):
        raise ValueError("dropped turn prefix slot ends at or before the assistant marker")
    return PreparedTurnContext(tuple(token_ids), segments[:keep], assignments[:p0], p0)
