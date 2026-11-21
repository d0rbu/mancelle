"""The generated-assistant cadence is MEASURED from turn-delimited artifacts.

Every fixture here is hand-computable in character space: the tokenizer decodes
one character per token, so render position ``p >= 1`` covers source characters
``[p - 1, p)`` and a segment's RustBPE width equals its character width. The
SOURCE (Gemma) token count of a segment is instead declared per segment, which
is what makes ``segment_size`` and the measured cadence independent quantities
the way the real dual-tokenizer artifacts make them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import pytest
import torch as t
from transformers import AutoTokenizer

import steering.conditional.encoder_decoder.turn_cadence as turn_cadence_module
from errors import SelfCondTrainingError
from steering.conditional.encoder_decoder.turn_cadence import (
    TurnCadenceMeasurement,
    measure_turn_self_cond_segment_size_rustbpe,
)
from steering.conditional.feature_index import SegmentFeatures
from steering.refined_types import PositiveInt
from steering.segmentation import resolve_bos_token_id, turn_segments_for_document
from steering.sft.loss_mask import LossMaskMode
from steering.sft.turn_spans import rendered_text_and_turn_char_starts
from tests.conftest import get_tokenizer_or_skip

_BOS = 0
_SPECIALS = {
    "<user_start>": 1,
    "<user_end>": 2,
    "<assistant_start>": 3,
    "<assistant_end>": 4,
    "<output_start>": 5,
    "<output_end>": 6,
}
_CONTENT_BASE = 10
_SEGMENT_SIZE = 2
_FEATURE_DIR = "/features/s2/safety_sft/turn_delimited"
_PARQUET_DIR = "/data/sft_data/safety"


class _CharTokenizer:
    """One id decodes to one character; chat specials have their own ids."""

    def get_bos_token_id(self) -> int:
        return _BOS

    def encode_special(self, name: str) -> int:
        return _SPECIALS[name]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(max(32, int(token_id))) for token_id in ids)


@dataclass(frozen=True)
class _Doc:
    """One rendered conversation plus its declared segment geometry."""

    turns: tuple[tuple[str, int], ...]
    # (char_start, char_end, source_tokens) per segment, tiling the render.
    plan: tuple[tuple[int, int, int], ...]


def _render(turns: Sequence[tuple[str, int]]) -> tuple[list[int], list[int]]:
    """Ids plus the completion-only mask: assistant content supervises, markers do not."""
    ids, mask = [_BOS], [0]
    for role, length in turns:
        ids.append(_SPECIALS[f"<{role}_start>"])
        mask.append(0)
        ids.extend(_CONTENT_BASE + offset for offset in range(length))
        mask.extend([int(role == "assistant")] * length)
    return ids, mask


def _segments(plan: Sequence[tuple[int, int, int]]) -> list[SegmentFeatures]:
    segments: list[SegmentFeatures] = []
    position = 0
    for segment_idx, (char_start, char_end, source_tokens) in enumerate(plan):
        segments.append(
            SegmentFeatures(
                segment_idx=segment_idx,
                position=position,
                feature_ids=t.tensor([segment_idx], dtype=t.int32),
                feature_mags=t.tensor([1.0], dtype=t.float32),
                char_start=char_start,
                char_end=char_end,
            )
        )
        position += source_tokens
    return segments


def _measure(
    docs: Sequence[_Doc],
    *,
    segment_size: int = _SEGMENT_SIZE,
    document_ids: Sequence[int] | None = None,
    tiles: bool = True,
    row_capacity: int = 1 << 20,
    min_loss_tokens: int = 0,
    loss_mask_mode: LossMaskMode = LossMaskMode.COMPLETION_ONLY,
) -> TurnCadenceMeasurement:
    keys = list(range(len(docs))) if document_ids is None else list(document_ids)
    pairs = list(zip(keys, docs, strict=True))
    renders = {key: _render(doc.turns) for key, doc in pairs}
    for key, doc in pairs:
        # ``tiles=False`` is how a fixture declares a deliberately mis-paired
        # artifact row: coverage that stops short of the rendered conversation.
        if not tiles:
            continue
        assert doc.plan[0][0] == 0
        assert doc.plan[-1][1] == len(renders[key][0]) - 1, "plan must tile the whole render"
    return measure_turn_self_cond_segment_size_rustbpe(
        segments_by_document={key: _segments(doc.plan) for key, doc in pairs},
        rendered=renders.__getitem__,
        tokenizer=_CharTokenizer(),
        segment_size=segment_size,
        row_capacity=row_capacity,
        loss_mask_mode=loss_mask_mode,
        min_loss_tokens=min_loss_tokens,
        feature_source_bos_char_len=0,
        feature_dir=_FEATURE_DIR,
        parquet_dir=_PARQUET_DIR,
    )


# A user turn of 4 content tokens spans chars [0, 5); the assistant turn that
# follows spans [5, 16). Doc 0's segment 1 and doc 1's segment 1 are the only
# full assistant windows: widths 4 and 6, so K = round(mean) = 5.
_WIDTH_4_DOC = _Doc(
    turns=(("user", 4), ("assistant", 10)),
    plan=((0, 5, 2), (5, 9, 2), (9, 12, 1), (12, 16, 2)),
)
_WIDTH_6_DOC = _Doc(
    turns=(("user", 4), ("assistant", 10)),
    plan=((0, 5, 1), (5, 11, 2), (11, 16, 3)),
)


# kills: returning segment_size (2) or any single window's width instead of the
# mean; counting the user-turn full window (doc 0 segment 0, source_tokens ==
# segment_size), the partial assistant window (doc 0 segment 2), or either
# document's trailing segment, each of which moves K off 5.
def test_cadence_is_the_mean_width_of_full_assistant_windows() -> None:
    measurement = _measure([_WIDTH_4_DOC, _WIDTH_6_DOC])

    assert int(measurement.cadence) == 5
    assert int(measurement.cadence) != _SEGMENT_SIZE
    assert int(measurement.windows) == 2
    assert measurement.mean == Fraction(5)
    assert int(measurement.minimum) == 4
    assert int(measurement.maximum) == 6


# kills: replacing `round` with truncation/floor/ceil, and computing the mean in
# binary floating point (5.5 and 6.5 are exact there, but a Fraction states the
# tie rule outright: Python's round resolves ties to even).
@pytest.mark.parametrize(
    ("widths", "expected"),
    [((5, 6), 6), ((6, 7), 6), ((4, 5), 4), ((5, 5), 5)],
)
def test_the_mean_is_rounded_with_ties_to_even(
    widths: tuple[int, int],
    expected: int,
) -> None:
    docs = [
        _Doc(
            turns=(("user", 4), ("assistant", 12)),
            plan=((0, 5, 1), (5, 5 + width, 2), (5 + width, 18, 3)),
        )
        for width in widths
    ]

    measurement = _measure(docs)

    assert measurement.mean == Fraction(sum(widths), 2)
    assert int(measurement.cadence) == expected


# kills: treating a window whose source-token count merely REACHES segment_size
# (or any non-equal count) as full -- the only assistant window here is short.
def test_a_partial_assistant_window_is_not_measured() -> None:
    partial = _Doc(
        turns=(("user", 4), ("assistant", 10)),
        plan=((0, 5, 2), (5, 9, 1), (9, 16, 2)),
    )

    with pytest.raises(
        SelfCondTrainingError,
        match=r"no full assistant window.*2 source tokens",
    ):
        _measure([partial])


# kills: dropping the assistant-membership test, or testing only the window's
# first/last position -- window [3, 7) is full and straddles the user turn.
def test_a_full_window_outside_or_straddling_an_assistant_turn_is_not_measured() -> None:
    user_only = _Doc(
        turns=(("user", 8), ("assistant", 6)),
        plan=((0, 3, 2), (3, 7, 2), (7, 16, 2)),
    )

    with pytest.raises(SelfCondTrainingError, match=r"no full assistant window"):
        _measure([user_only])


# kills: measuring a document's trailing segment against a guessed source-token
# count -- the only assistant window here is the last segment of the row.
def test_the_trailing_segment_is_never_a_window() -> None:
    trailing = _Doc(
        turns=(("user", 4), ("assistant", 10)),
        plan=((0, 5, 1), (5, 11, 1), (11, 16, 2)),
    )

    with pytest.raises(SelfCondTrainingError, match=r"no full assistant window"):
        _measure([trailing])


# kills: falling back to segment_size, to a Gemma/RustBPE ratio, or to any
# default when the artifacts yield nothing, and dropping either directory from
# the message an operator needs to find the mis-paired family.
def test_an_unmeasurable_store_names_both_artifact_directories() -> None:
    single_segment = _Doc(
        turns=(("user", 4), ("assistant", 10)),
        plan=((0, 16, 2),),
    )

    with pytest.raises(
        SelfCondTrainingError,
        match=(
            rf"cannot measure the generated-assistant cadence.*"
            rf"feature_dir='{_FEATURE_DIR}'.*parquet_dir='{_PARQUET_DIR}'"
        ),
    ):
        _measure([single_segment])


# kills: dropping the cap (all ten documents measured -> mean 4.8 -> K = 5),
# selecting the first N documents instead of even spacing (ids 0-3 -> widths
# 4,6,4,6 -> K = 5), or an off-by-one in ((2k + 1) * L) // (2 * N) (ids 1,3,6,8
# are the only width-6 documents, so any other choice pulls K below 6).
def test_the_sample_cap_selects_evenly_spaced_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    assert turn_cadence_module.TURN_CADENCE_MAX_CONVERSATIONS == 4096
    docs = [_WIDTH_6_DOC if index in (1, 3, 6, 8) else _WIDTH_4_DOC for index in range(10)]

    full = _measure(docs)
    monkeypatch.setattr(turn_cadence_module, "TURN_CADENCE_MAX_CONVERSATIONS", 4)
    capped = _measure(docs)

    assert int(full.cadence) == 5
    assert int(full.documents) == 10
    assert int(capped.cadence) == 6
    assert int(capped.documents) == 4
    assert int(capped.windows) == 4
    with pytest.raises(ValueError, match="row_capacity must be positive"):
        _measure(docs, row_capacity=0)


# A 4-content-token user turn followed by a 30-content-token assistant turn
# renders 37 tokens, so the feature-source text is 36 characters long. Coverage
# that stops at 15 leaves a 21-character tail -- one past the cross-tokenizer
# end-marker window the conditioned-SFT generator itself allows.
_SHORT_COVERAGE_DOC = _Doc(
    turns=(("user", 4), ("assistant", 30)),
    plan=((0, 5, 1), (5, 11, 2), (11, 15, 3)),
)
# The same render covered to 16: a 20-character tail, exactly at the window, so
# alignment extends the trailing segment and the pass measures normally.
_TOLERATED_TAIL_DOC = _Doc(
    turns=(("user", 4), ("assistant", 30)),
    plan=((0, 5, 1), (5, 11, 2), (11, 16, 3)),
)


# kills: dropping training's coverage-gap pre-check, which lets one mis-paired
# conversation in a corpus abort the whole pass inside
# align_char_ranges_to_decoded_prefix naming no document; and reporting the loop
# ordinal (1) instead of the store's document id (9), which is what an operator
# needs to find or re-extract the conversation.
def test_a_document_whose_coverage_stops_short_names_itself_and_its_gap() -> None:
    with pytest.raises(
        SelfCondTrainingError,
        match=(
            rf"cadence measurement document 9: .*feature coverage ends 21 chars "
            rf"before the rendered conversation.*feature_dir='{_FEATURE_DIR}' "
            rf"parquet_dir='{_PARQUET_DIR}'.*Re-extract full-sequence SFT features"
        ),
    ):
        _measure(
            [_WIDTH_4_DOC, _SHORT_COVERAGE_DOC],
            document_ids=(5, 9),
            tiles=False,
        )


# kills: pre-checking the gap against any window narrower than the one the
# conditioned-SFT generator uses -- a 20-character tail is inside the
# cross-tokenizer end-marker window and must still be measured, not rejected.
def test_a_document_with_a_tail_inside_the_window_is_measured_normally() -> None:
    measurement = _measure([_TOLERATED_TAIL_DOC], tiles=False)

    assert int(measurement.cadence) == 6
    assert int(measurement.windows) == 1
    assert int(measurement.documents) == 1


# kills: leaving every other per-document routing ValueError unwrapped, which
# reports the malformed geometry with no way to tell WHICH conversation carries
# it. The overlap here passes the coverage-gap pre-check and fails in routing.
def test_a_document_whose_segments_overlap_is_named_with_the_routing_error() -> None:
    overlapping = _Doc(
        turns=(("user", 4), ("assistant", 30)),
        plan=((0, 6, 1), (5, 11, 2), (11, 16, 3)),
    )

    with pytest.raises(
        SelfCondTrainingError,
        match=(
            r"cadence measurement document 4: .*"
            r"Overlap between segment 0 \(end=6\) and segment 1 \(start=5\)"
        ),
    ):
        _measure([overlapping], document_ids=(4,), tiles=False)


# A render longer than the packing row. The conditioned-SFT generator truncates
# the render to ``row_capacity`` BEFORE its coverage check, so coverage that
# stops short of the FULL render but covers the retained prefix is normal
# head truncation, not a mis-paired artifact.
# kills: checking coverage against the untruncated render (the 21-char gap
# below then aborts, contradicting the generator, which trains on this row).
def test_a_render_longer_than_the_row_is_measured_over_the_retained_prefix() -> None:
    measurement = _measure([_SHORT_COVERAGE_DOC], tiles=False, row_capacity=16)

    assert int(measurement.cadence) == 6
    assert int(measurement.windows) == 1
    assert int(measurement.documents) == 1


# Windows the generator never sees must not shape K: after truncation to 15
# tokens (14 source chars) the 8-wide window at [11, 19) is clipped, exactly as
# ``_truncate_seg_features`` clips it, and only the 6-wide window survives.
# kills: measuring the untruncated render (widths 6 and 8 -> K = 7) or
# counting a clipped window at its clipped width.
def test_windows_clipped_by_row_truncation_are_not_measured() -> None:
    long_doc = _Doc(
        turns=(("user", 4), ("assistant", 30)),
        plan=((0, 5, 1), (5, 11, 2), (11, 19, 2), (19, 36, 3)),
    )

    full = _measure([long_doc])
    truncated = _measure([long_doc], row_capacity=15)

    assert int(full.cadence) == 7
    assert int(full.windows) == 2
    assert int(truncated.cadence) == 6
    assert int(truncated.windows) == 1


# kills: skipping the coverage pre-check for single-segment documents (the
# generator applies it to every example, so this document would still abort
# a full SFT epoch).
def test_a_single_segment_document_with_a_coverage_gap_is_rejected() -> None:
    single = _Doc(turns=(("user", 4), ("assistant", 30)), plan=((0, 15, 2),))

    with pytest.raises(
        SelfCondTrainingError,
        match=r"cadence measurement document 3: .*feature coverage ends 21 chars",
    ):
        _measure([single], document_ids=(3,), tiles=False)


# kills: reporting the sampled count as the measured population -- a sampled
# document that yields no full assistant window must not inflate ``documents``.
def test_documents_counts_only_conversations_that_contributed_a_window() -> None:
    windowless = _Doc(
        turns=(("user", 8), ("assistant", 6)),
        plan=((0, 3, 2), (3, 7, 2), (7, 16, 2)),
    )

    measurement = _measure([_WIDTH_4_DOC, windowless])

    assert int(measurement.cadence) == 4
    assert int(measurement.windows) == 1
    assert int(measurement.documents) == 1


# The generator skips a row with fewer than ``min_loss_tokens`` supervised
# targets (counted on the TRUNCATED mask) before it checks coverage. A long
# user turn whose assistant answer falls past the row therefore never reaches
# the coverage check in training and must not abort the measurement either.
# kills: checking coverage (or counting windows) before the supervision skip.
def test_a_row_the_generator_skips_for_supervision_is_not_measured() -> None:
    # 4 supervised assistant tokens; its features stop 26 chars short.
    user_heavy = _Doc(
        turns=(("user", 30), ("assistant", 4)),
        plan=((0, 10, 2),),
    )

    measurement = _measure([_WIDTH_4_DOC, user_heavy], tiles=False, min_loss_tokens=5)

    assert int(measurement.cadence) == 4
    assert int(measurement.documents) == 1
    with pytest.raises(SelfCondTrainingError, match=r"feature coverage ends 26 chars"):
        _measure([_WIDTH_4_DOC, user_heavy], tiles=False, min_loss_tokens=4)


# The generator clips segments only when it truncated the row. An untruncated
# render whose features run more than 20 chars PAST it is the mirror image of
# a coverage gap; the generator dies on it inside routing, so the measurement
# must reject it too -- and name the document.
# kills: clipping unconditionally (the over-coverage is silently trimmed away
# and the pass accepts a row training will die on).
def test_an_untruncated_row_with_over_coverage_is_rejected_and_named() -> None:
    over_covered = _Doc(
        turns=(("user", 4), ("assistant", 10)),
        plan=((0, 5, 1), (5, 11, 2), (11, 40, 3)),
    )

    with pytest.raises(
        SelfCondTrainingError,
        match=r"cadence measurement document 6: .*delta=24",
    ):
        _measure([over_covered], document_ids=(6,), tiles=False)


# kills: leaving the coverage call outside the wrapper -- an empty feature row
# would surface as a bare ValueError naming no document.
def test_an_empty_feature_row_is_named() -> None:
    empty = _Doc(turns=(("user", 4), ("assistant", 10)), plan=())

    with pytest.raises(
        SelfCondTrainingError,
        match=r"cadence measurement document 2: .*feature row has no segments",
    ):
        _measure([empty], document_ids=(2,), tiles=False)


# Under FULL_SEQUENCE every input position supervises (``len(ids) - 1``), under
# COMPLETION_ONLY only the assistant content does (10 here); the generator
# computes both the same way. ``_WIDTH_4_DOC`` renders 17 tokens.
# The FULL_SEQUENCE arm is otherwise unexercised in this suite.
# kills: swapping the two formulas (the thresholds below sit between them).
@pytest.mark.parametrize(
    ("mode", "kept", "skipped"),
    [
        pytest.param(LossMaskMode.FULL_SEQUENCE, 16, 17, id="full-sequence"),
        pytest.param(LossMaskMode.COMPLETION_ONLY, 10, 11, id="completion-only"),
    ],
)
def test_supervision_is_counted_the_generator_way_for_each_mask_mode(
    mode: LossMaskMode, kept: int, skipped: int
) -> None:
    assert int(_measure([_WIDTH_4_DOC], loss_mask_mode=mode, min_loss_tokens=kept).cadence) == 4
    with pytest.raises(SelfCondTrainingError, match=r"no full assistant window \(2 source tokens"):
        _measure([_WIDTH_4_DOC], loss_mask_mode=mode, min_loss_tokens=skipped)


# Features that overrun an UNTRUNCATED render (16 decoded chars) by up to the
# marker window are tolerated by routing, which drops the windows past the
# text and clips the last survivor. In the first plan that survivor (11, 17)
# -> (11, 16) still has a successor in the store, so a loop bounded by the
# store would count it as a full window at its clipped width (K = 5); in the
# second the next window starts exactly at the decoded length, which routing
# drops (`>=`), so the (11, 16) window is the last measurable one. The
# generator never routes either as a full window.
# kills: bounding the measurement loop by the unclipped segment list, and an
# off-by-one (`<=`) in the measurable bound.
@pytest.mark.parametrize(
    "plan",
    [
        pytest.param(((0, 5, 1), (5, 11, 2), (11, 17, 2), (17, 20, 2)), id="clipped-survivor"),
        pytest.param(((0, 5, 1), (5, 11, 2), (11, 16, 2), (16, 20, 2)), id="start-at-decoded-len"),
    ],
)
def test_a_window_clipped_by_routings_alignment_is_not_measured(
    plan: tuple[tuple[int, int, int], ...],
) -> None:
    over_covered = _Doc(turns=(("user", 4), ("assistant", 10)), plan=plan)

    measurement = _measure([over_covered], tiles=False)

    assert int(measurement.windows) == 1
    assert int(measurement.cadence) == 6
    assert int(measurement.minimum) == 6


# With the real tokenizers a trailing Gemma "><" straddle leaves the last
# source window empty and routes <assistant_end> into the preceding full
# window. That window has 50 routed positions in the full routing but the
# generator copies only 49 as inputs (the conversation's final position is a
# target only); the values below are the generator's counts, measured against
# make_sft_data_generator on identical feature rows.
# kills: counting the full routing (K = 50 / 242).
@pytest.mark.parametrize(
    ("repeats", "segment_size", "expected_cadence"),
    [pytest.param(24, 32, 49, id="seg32"), pytest.param(120, 128, 241, id="seg128")],
)
def test_the_final_position_is_never_counted_with_real_tokenizers(
    repeats: int, segment_size: int, expected_cadence: int
) -> None:
    nanochat_tokenizer = get_tokenizer_or_skip()
    try:
        gemma = AutoTokenizer.from_pretrained("google/gemma-3-4b-pt", local_files_only=True)
    except OSError as exc:
        pytest.skip(f"gemma-3-4b-pt tokenizer is not in the local HF cache: {exc}")
    ids, mask = nanochat_tokenizer.render_conversation(
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Hello " * repeats},
            ]
        }
    )
    text, turn_starts = rendered_text_and_turn_char_starts(ids, nanochat_tokenizer)
    gemma_bos = resolve_bos_token_id(gemma)
    kept = [gemma_bos, *gemma.encode(text, add_special_tokens=False)]
    token_spans, starts, ends, _drift = turn_segments_for_document(
        kept, gemma, text, turn_starts, PositiveInt.parse(segment_size)
    )
    positions = [0]
    for token_start, token_end in token_spans[:-1]:
        positions.append(positions[-1] + (token_end - token_start))
    segments = [
        SegmentFeatures(
            segment_idx=index,
            position=position,
            feature_ids=t.tensor([index], dtype=t.int32),
            feature_mags=t.tensor([1.0], dtype=t.float32),
            char_start=int(start),
            char_end=int(end),
        )
        for index, (position, start, end) in enumerate(zip(positions, starts, ends, strict=True))
    ]

    measurement = measure_turn_self_cond_segment_size_rustbpe(
        segments_by_document={0: segments},
        rendered=lambda _document_id: (ids, mask),
        tokenizer=nanochat_tokenizer,
        segment_size=segment_size,
        row_capacity=1 << 20,
        loss_mask_mode=LossMaskMode.COMPLETION_ONLY,
        min_loss_tokens=1,
        feature_source_bos_char_len=len(gemma.decode([gemma_bos])),
        feature_dir=_FEATURE_DIR,
        parquet_dir=_PARQUET_DIR,
    )

    assert int(measurement.windows) == 1
    assert int(measurement.cadence) == expected_cadence


# Single-segment rows are routed too, so a malformed one is rejected and named
# exactly where a full SFT epoch would die on it.
# kills: skipping rows with fewer than two segments before routing.
def test_a_single_segment_document_with_malformed_geometry_is_named() -> None:
    off_by_five = _Doc(turns=(("user", 4), ("assistant", 10)), plan=((5, 16, 2),))

    with pytest.raises(
        SelfCondTrainingError,
        match=r"cadence measurement document 8: .*First segment must start at 0",
    ):
        _measure([off_by_five], document_ids=(8,), tiles=False)


# kills: leaving the turn-span scan outside the wrapper -- a render that does
# not open with a user turn would surface as a bare TurnSpanError naming no
# document.
def test_a_render_with_a_malformed_turn_structure_is_named() -> None:
    assistant_first = _Doc(turns=(("assistant", 10),), plan=((0, 11, 2), (11, 12, 1)))

    with pytest.raises(
        SelfCondTrainingError,
        match=r"cadence measurement document 9: legacy render must have <user_start> immediately after BOS",
    ):
        _measure([assistant_first], document_ids=(9,), tiles=False)
