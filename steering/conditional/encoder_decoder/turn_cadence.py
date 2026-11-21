"""Measured generated-assistant cadence for turn-delimited SFT post-training."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Final

from beartype import beartype

from errors import SelfCondTrainingError
from scripts.train.conditional_encdec_sft import prepare_sft_example
from steering.conditional.encoder_decoder.dataloader import SegFeatures
from steering.conditional.encoder_decoder.full_sequence_routing import (
    EMPTY_FEATURES,
    source_aligned_char_offsets,
)
from steering.conditional.feature_index import SegmentFeatures
from steering.refined_types import PositiveInt
from steering.sft.loss_mask import LossMaskMode
from steering.sft.turn_spans import ChatSpecialTokens, TurnRole, legacy_turn_spans

# Measuring every conversation of a million-row mixture would cost every rank a
# long render+route pass at each startup. An even-spaced sample of this many
# conversations bounds that cost and is identical on every rank and resume;
# it does not guarantee the same rounded mean as measuring the entire corpus.
TURN_CADENCE_MAX_CONVERSATIONS: Final = 4096
# The cadence pass is the first stage to route a broad sample of the corpus, so
# it is where a mis-paired conversation surfaces. It prepares each conversation
# exactly as the conditioned-SFT generator does (row truncation, then the
# coverage-gap pre-check, then segment clipping; see
# ``scripts/train/conditional_encdec_sft.py``) and reports the SAME remedy: a
# document this pass rejects is a document a full SFT epoch would die on, so
# tolerating it here would only defer the failure to a longer, costlier run.
_MEASUREMENT_LABEL: Final = "turn-delimited SFT self-conditioning cadence measurement"
_REEXTRACT_REMEDY: Final = (
    "Re-extract full-sequence SFT features with the same rendered task mixture "
    "and tokenization settings before training."
)


@beartype
@dataclass(frozen=True)
class TurnCadenceMeasurement:
    """The measured cadence and the window population it was measured over."""

    cadence: PositiveInt
    windows: PositiveInt
    documents: PositiveInt
    mean: Fraction
    minimum: PositiveInt
    maximum: PositiveInt


@beartype
def _assistant_positions(ids: Sequence[int], tokenizer: Any) -> frozenset[int]:
    """Return the render positions that fall inside a logical assistant turn.

    ``legacy_turn_spans`` indexes ``ids[1:]``; the leading BOS belongs to no
    turn, so render position ``p`` maps to span index ``p - 1``.
    """
    spans = legacy_turn_spans(ids, ChatSpecialTokens.from_tokenizer(tokenizer))
    return frozenset(
        index + 1
        for span in spans
        if TurnRole(span.role) is TurnRole.ASSISTANT
        for index in range(span.start, span.end)
    )


@beartype
def measure_turn_self_cond_segment_size_rustbpe(
    *,
    segments_by_document: Mapping[int, list[SegmentFeatures]],
    rendered: Callable[[int], tuple[Sequence[int], Sequence[int]]],
    tokenizer: Any,
    segment_size: int,
    row_capacity: int,
    loss_mask_mode: LossMaskMode,
    min_loss_tokens: int,
    feature_source_bos_char_len: int,
    feature_dir: str,
    parquet_dir: str,
) -> TurnCadenceMeasurement:
    """Measure the RustBPE width of a full assistant window from the artifacts.

    Turn-delimited extraction caps every segment at ``segment_size`` SOURCE
    (Gemma) tokens, so a segment whose source-token count equals
    ``segment_size`` is a FULL window; source-token counts are the differences
    of consecutive ``SegmentFeatures.position`` values, so a document's
    trailing segment has no successor to measure against and is excluded rather
    than guessed. Each conversation is rendered exactly as the trainer renders
    it and its segments are routed onto those RustBPE positions with
    ``build_full_sequence_segment_routing``, so the width returned is the
    decoder-side extent the source-side window occupies as a CONDITIONED
    INPUT: the generator attributes only ``len(ids) - 1`` positions to a
    conversation's windows and leaves its final position unconditioned
    (``seg_map`` 0) whether or not it is the row's last input, so it is
    excluded here too. K therefore counts one position fewer than the
    inference-side context extent of a window that owns a closing marker.

    Every sampled document is prepared exactly as the conditioned-SFT generator
    prepares it: ``truncate_conv_and_mask`` cuts the render and its loss mask
    to ``row_capacity`` (the packing row, ``dec_config.sequence_len + 1``), a
    row with fewer than ``min_loss_tokens`` supervised targets under
    ``loss_mask_mode`` is skipped the way the generator skips it, then comes the
    generator's coverage-gap check (``compute_full_coverage_gap`` against
    ``CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS``), and a TRUNCATED row's segments
    are clipped to the retained text the way ``_truncate_seg_features`` clips
    them (an untruncated row's over-coverage reaches routing, which clips it
    within the marker window and rejects it beyond).
    A gap rejected here would therefore also fail a full SFT epoch over the same
    artifacts, and a window the generator never sees (past the row) is never
    measured. The offending conversation is named, which the routing primitive
    cannot do; any other per-document routing ``ValueError`` is re-raised the
    same way.

    Assistant membership comes from ``legacy_turn_spans`` and NOT from the
    render loss mask: ``nanochat.tokenizer.render_conversation`` marks
    ``<assistant_start>`` with 0 and python-output tokens inside an assistant
    turn with 0, so that mask is a supervision mask, not a role mask.

    K is ``round(mean(...))`` over the RustBPE width of every full assistant
    window in the sampled documents, taken exactly over a ``Fraction`` (ties
    resolve to even, as Python's ``round`` does). If the sample has no full
    assistant window, the cadence cannot be measured; this raises
    ``SelfCondTrainingError`` rather than choosing a fallback.

    When the store holds more than ``TURN_CADENCE_MAX_CONVERSATIONS`` documents, exactly
    ``TURN_CADENCE_MAX_CONVERSATIONS`` of them are measured, chosen at even spacing over the
    sorted document ids (``ids[((2k + 1) * L) // (2 * N)]``), so the pass is
    bounded and identical on every rank and every resume. ``documents`` counts
    the sampled conversations that contributed at least one window.
    """
    if row_capacity < 1:
        raise ValueError(f"row_capacity must be positive, got {row_capacity}")
    document_ids = sorted(segments_by_document)
    if len(document_ids) > TURN_CADENCE_MAX_CONVERSATIONS:
        total = len(document_ids)
        document_ids = [
            document_ids[((2 * k + 1) * total) // (2 * TURN_CADENCE_MAX_CONVERSATIONS)]
            for k in range(TURN_CADENCE_MAX_CONVERSATIONS)
        ]
    widths: list[int] = []
    documents = 0
    for document_id in document_ids:
        segments = segments_by_document[document_id]
        rendered_ids, rendered_mask = rendered(document_id)
        try:
            prepared = prepare_sft_example(
                ids=rendered_ids,
                loss_mask=rendered_mask,
                row_capacity=row_capacity,
                loss_mask_mode=loss_mask_mode,
                min_loss_tokens=min_loss_tokens,
                select_segments=lambda segments=segments: [
                    SegFeatures(
                        EMPTY_FEATURES,
                        EMPTY_FEATURES,
                        EMPTY_FEATURES,
                        segment.char_start,
                        segment.char_end,
                    )
                    for segment in segments
                ],
                tokenizer=tokenizer,
                feature_source_bos_char_len=feature_source_bos_char_len,
                example_idx=document_id,
            )
            if prepared.routing is None:
                continue
            assistant = _assistant_positions(prepared.ids, tokenizer)
        except ValueError as error:
            raise SelfCondTrainingError(
                f"{_MEASUREMENT_LABEL} document {document_id}: {error} "
                f"(feature_dir={feature_dir!r} parquet_dir={parquet_dir!r}). {_REEXTRACT_REMEDY}"
            ) from error
        routing = prepared.routing
        # Routing's own alignment drops windows past the decoded text and clips
        # the last survivor, so only windows that START inside it can be full;
        # the last of those is the clipped or trailing one and is not measured.
        decoded_len = source_aligned_char_offsets(
            tokenizer, prepared.ids, source_bos_char_len=feature_source_bos_char_len
        )[-1]
        measurable = sum(
            segment.char_start < decoded_len and not segment.prediction_target_only
            for segment in prepared.segments
        )
        measured = len(widths)
        for ordinal in range(measurable - 1):
            source_tokens = segments[ordinal + 1].position - segments[ordinal].position
            if source_tokens != segment_size:
                continue
            # The generator conditions only ``len(ids) - 1`` positions of each
            # conversation (its final position keeps ``seg_map`` 0), so a window
            # that owns the closing marker is one position narrower than the
            # full routing.
            positions = (routing.seg_map == ordinal + 1).nonzero().flatten().tolist()
            if positions and all(position in assistant for position in positions):
                widths.append(len(positions))
        documents += len(widths) > measured
    if not widths:
        raise SelfCondTrainingError(
            "turn-delimited SFT self-conditioning cannot measure the "
            "generated-assistant cadence: no full assistant window "
            f"({segment_size} source tokens, entirely inside an assistant turn) "
            f"exists in the TRAIN artifacts feature_dir={feature_dir!r} "
            f"parquet_dir={parquet_dir!r}."
        )
    mean = Fraction(sum(widths), len(widths))
    return TurnCadenceMeasurement(
        cadence=PositiveInt.parse(round(mean)),
        windows=PositiveInt.parse(len(widths)),
        documents=PositiveInt.parse(documents),
        mean=mean,
        minimum=PositiveInt.parse(min(widths)),
        maximum=PositiveInt.parse(max(widths)),
    )


__all__ = ["TurnCadenceMeasurement", "measure_turn_self_cond_segment_size_rustbpe"]
