"""Parity tests for transient and shard-backed context conversion."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
from transformers import AutoTokenizer

import steering.eval.context_cache_build as context_build
from errors import FeatureClustersError
from scripts.preprocess.precompute_token_boundaries import compute_doc_token_boundaries_from_tokens
from steering.conditional.encoder_decoder.eval_adapter import (
    TargetAwareEncoderDecoderEvalWrapper,
)
from steering.conditional.encoder_decoder.full_sequence_routing import (
    build_full_sequence_segment_routing,
    context_token_assignments,
    turn_context_token_assignments,
)
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
)
from steering.conditional.feature_mask import FeatureMask
from steering.conditional.feature_pipeline import FeaturePipelineConfig
from steering.conditional.sampling import (
    Distribution,
    NegativeBundle,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
)
from steering.eval.context_enumeration import EvalContextRecord
from steering.extract_features import _precompute_batch_cpu_work
from steering.feature_io import TensorKey
from steering.refined_types import NonNegativeInt, PositiveInt
from steering.segmentation import (
    SegmentMode,
    compute_context_token_segment_assignments,
    resolve_bos_token_id,
    token_to_char_offsets,
    turn_segments_for_document,
)
from steering.sft.turn_spans import (
    ChatSpecialTokens,
    legacy_turn_spans,
    rendered_text_and_turn_char_starts,
)
from tests.conftest import get_tokenizer_or_skip
from tests.nanochat.test_structured_conversation_renderer import _CharacterTokenizer


class _DifferentSourceTokenizer:
    """BOS-prefixed, variable-width tokenizer for cross-tokenizer geometry tests."""

    bos_token_id = 0
    _structural = (
        "<user_start>",
        "<user_end>",
        "<assistant_start>",
        "<assistant_end>",
        "<python_start>",
        "<python_end>",
        "<output_start>",
        "<output_end>",
    )

    def __init__(self) -> None:
        self._piece_to_id: dict[str, int] = {}
        self._id_to_piece: dict[int, str] = {}

    def _id_for_piece(self, piece: str) -> int:
        token_id = self._piece_to_id.get(piece)
        if token_id is None:
            token_id = 100 + len(self._piece_to_id)
            self._piece_to_id[piece] = token_id
            self._id_to_piece[token_id] = piece
        return token_id

    def encode(self, text: str) -> list[int]:
        pieces: list[str] = []
        cursor = 0
        width = 2
        while cursor < len(text):
            structural = next(
                (token for token in self._structural if text.startswith(token, cursor)),
                None,
            )
            if structural is not None:
                pieces.append(structural)
                cursor += len(structural)
                continue
            next_structural = min(
                (
                    position
                    for token in self._structural
                    if (position := text.find(token, cursor)) >= 0
                ),
                default=len(text),
            )
            end = min(cursor + width, next_structural)
            pieces.append(text[cursor:end])
            cursor = end
            width = 3 if width == 2 else 2
        return [self.bos_token_id, *(self._id_for_piece(piece) for piece in pieces)]

    def __call__(self, text: str, *, padding: bool = False):
        assert padding is False
        return {"input_ids": self.encode(text)}

    def decode(self, token_ids) -> str:
        return "".join(
            "<g>" if int(token_id) == self.bos_token_id else self._id_to_piece[int(token_id)]
            for token_id in token_ids
        )


def _pipeline(*, feature_mask: FeatureMask | None = None) -> FeaturePipelineConfig:
    positive_rng = t.Generator(device="cpu")
    positive_rng.manual_seed(11)
    negative_rng = t.Generator(device="cpu")
    negative_rng.manual_seed(12)
    return FeaturePipelineConfig(
        selection_config=SamplingBundle(
            selection=SamplingSelection(
                distribution=Distribution.geometric,
                mean=1.0,
                std=1.0,
                min_k=1,
                max_k=2,
                mode=SamplingMode.top_k_percentile,
            ),
            percentile_table=t.linspace(0.01, 1.0, 99, dtype=t.float64).repeat(8, 1),
            rng=positive_rng,
        ),
        boundaries=t.empty((8, 0), dtype=t.float64),
        negative_bundle=NegativeBundle(
            selection=NegativeSamplingSelection(
                distribution=Distribution.geometric,
                mean=1.0,
                std=1.0,
                min_k=1,
                max_k=2,
            ),
            n_sae_features=PositiveInt.parse(8),
            rng=negative_rng,
        ),
        effective_buckets=1,
        feature_mask=feature_mask,
    )


def _consolidation(
    *,
    n_sae_features: int,
    feature_mask: FeatureMask | None = None,
) -> FeatureClusterConsolidation:
    definition = FeatureClusterDefinition(clusters=((0, 1),))
    resolved = ResolvedFeatureClusters(
        source_path=Path("fixture-clusters.yaml"),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition,
            n_sae_features=n_sae_features,
        ),
    )
    return FeatureClusterConsolidation.build(resolved, feature_mask)


def _patch_alignment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        context_build,
        "compute_doc_token_boundaries_from_tokens",
        lambda *_args, **_kwargs: (t.tensor([0, 2]), t.tensor([2, 4])),
    )
    monkeypatch.setattr(
        context_build,
        "token_to_char_offsets",
        lambda *_args, **_kwargs: [0, 1, 2, 3, 4],
    )


def _turn_render_and_source_ranges():
    tokenizer = _CharacterTokenizer()
    rendered, _mask = tokenizer.render_conversation(
        {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "known"},
                {"role": "user", "content": "last"},
                {"role": "assistant", "content": ""},
            ]
        }
    )
    prompt = rendered[:-1]
    assert prompt[-1] == tokenizer.encode_special("<assistant_start>")
    text, turn_starts = rendered_text_and_turn_char_starts(prompt, tokenizer)
    source_bos_char_len = 11
    source_starts = [0, *(source_bos_char_len + start for start in turn_starts[1:])]
    source_ends = [*source_starts[1:], source_bos_char_len + len(text)]
    return (
        tokenizer,
        prompt,
        turn_starts,
        source_bos_char_len,
        source_starts,
        source_ends,
    )


def _build_selected_turn_entry(*, alignment_bos_char_len: int):
    (
        tokenizer,
        prompt,
        _turn_starts,
        _source_bos_char_len,
        source_starts,
        source_ends,
    ) = _turn_render_and_source_ranges()
    n_segments = len(source_starts)
    entry = context_build.build_context_entry_from_selected_segments(
        token_ids=prompt,
        segment_feature_ids=[t.tensor([10 + index]) for index in range(n_segments)],
        segment_buckets=[t.tensor([index % 2]) for index in range(n_segments)],
        segment_char_starts=source_starts,
        segment_char_ends=source_ends,
        tokenizer=tokenizer,
        source_bos_char_len=NonNegativeInt.parse(alignment_bos_char_len),
        source="turn conversion fixture",
    )
    return tokenizer, prompt, entry


def _prepare_turn_entry(tokenizer, prompt, entry):
    surface = cast(
        Any,
        SimpleNamespace(
            tokenizer=tokenizer,
            _turn_specials=ChatSpecialTokens.from_tokenizer(tokenizer),
            context_provider=entry,
            self_cond_segment_mode=entry.segment_mode,
        ),
    )
    return TargetAwareEncoderDecoderEvalWrapper._prepare_turn_self_cond_prompt_context(
        surface,
        prompt,
        entry.segments,
    )


def test_selected_turn_conversion_preserves_exact_geometry_across_bos_shift() -> None:
    (
        tokenizer,
        prompt,
        turn_starts,
        source_bos_char_len,
        _source_starts,
        _source_ends,
    ) = _turn_render_and_source_ranges()
    converted_tokenizer, converted_prompt, entry = _build_selected_turn_entry(
        alignment_bos_char_len=source_bos_char_len,
    )
    assert isinstance(converted_tokenizer, tokenizer.__class__)
    assert converted_prompt == prompt

    offsets = token_to_char_offsets(converted_tokenizer, converted_prompt)
    spans = legacy_turn_spans(
        converted_prompt,
        ChatSpecialTokens.from_tokenizer(converted_tokenizer),
    )
    expected_starts = [0, *(offsets[span.start + 1] for span in spans[1:])]
    expected_ends = [*expected_starts[1:], offsets[-1]]
    assert tuple(expected_starts[1:]) == tuple(offsets[1] + start for start in turn_starts[1:])
    assert [segment.char_start for segment in entry.segments] == expected_starts
    assert [segment.char_end for segment in entry.segments] == expected_ends
    assert entry.segments[-1].char_start == offsets[-2]
    assert entry.segments[-1].char_end == offsets[-1]

    real_tokens, real_segments = _prepare_turn_entry(
        converted_tokenizer,
        converted_prompt,
        entry,
    )
    assert real_tokens == converted_prompt[:-1]
    assert real_segments == entry.segments[:-1]
    assert [segment.char_start for segment in real_segments] == expected_starts[:-1]


def test_turn_runtime_rejects_selected_geometry_from_wrong_bos_shift() -> None:
    tokenizer, prompt, _turn_starts, _source_bos, _starts, _ends = _turn_render_and_source_ranges()
    converted_tokenizer, converted_prompt, entry = _build_selected_turn_entry(
        alignment_bos_char_len=20,
    )
    assert isinstance(converted_tokenizer, tokenizer.__class__)
    assert converted_prompt == prompt

    with pytest.raises(ValueError, match="exact marker-only"):
        _prepare_turn_entry(converted_tokenizer, converted_prompt, entry)


def test_complete_turn_extraction_chain_matches_runtime_geometry() -> None:
    target_tokenizer = _CharacterTokenizer()
    rendered, _mask = target_tokenizer.render_conversation(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "A deliberately long first request: abcdefghijklmnopqrstuvwxyz.",
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "checking"},
                        {"type": "python", "text": "6 * 7"},
                        {"type": "python_output", "text": "42"},
                        {"type": "text", "text": "done"},
                    ],
                },
                {
                    "role": "user",
                    "content": "A second long request: 0123456789 and 世界.",
                },
                {"role": "assistant", "content": ""},
            ]
        }
    )
    prompt = rendered[:-1]
    assert prompt[-1] == target_tokenizer.encode_special("<assistant_start>")
    text, logical_turn_char_starts = rendered_text_and_turn_char_starts(
        prompt,
        target_tokenizer,
    )

    source_tokenizer = _DifferentSourceTokenizer()
    source_ids = source_tokenizer.encode(text)
    source_segments, n_tokens, source_char_starts, source_char_ends = _precompute_batch_cpu_work(
        [(source_ids, len(source_ids))],
        source_tokenizer,
        SegmentMode.turn_delimited_fixed_tokens,
        10,
        texts=[text],
        turn_char_starts=[logical_turn_char_starts],
    )[0]
    assert n_tokens == len(source_ids)
    assert len(source_segments) > len(logical_turn_char_starts)

    entry = context_build.build_context_entry_from_selected_segments(
        token_ids=prompt,
        segment_feature_ids=[t.tensor([index]) for index in range(len(source_segments))],
        segment_buckets=[t.tensor([index % 2]) for index in range(len(source_segments))],
        segment_char_starts=source_char_starts,
        segment_char_ends=source_char_ends,
        tokenizer=target_tokenizer,
        source_bos_char_len=NonNegativeInt.parse(len(source_tokenizer.decode([0]))),
        source="complete turn extraction chain",
    )

    target_offsets = token_to_char_offsets(target_tokenizer, prompt)
    logical_spans = legacy_turn_spans(
        prompt,
        ChatSpecialTokens.from_tokenizer(target_tokenizer),
    )
    expected_logical_starts = {target_offsets[span.start + 1] for span in logical_spans[1:]}
    converted_starts = {segment.char_start for segment in entry.segments}
    assert expected_logical_starts <= converted_starts
    assert any(
        segment.char_start not in expected_logical_starts | {0} for segment in entry.segments[:-1]
    )
    assert entry.segments[-1].char_start == target_offsets[-2]
    assert entry.segments[-1].char_end == target_offsets[-1]

    real_tokens, real_segments = _prepare_turn_entry(
        target_tokenizer,
        prompt,
        entry,
    )
    assert real_tokens == prompt[:-1]
    assert real_segments == entry.segments[:-1]


def test_in_memory_conversion_matches_shard_converter_with_identical_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_alignment(monkeypatch)
    record = EvalContextRecord(
        suite="unit",
        task="task",
        seed=1,
        example_index=0,
        row_index=0,
        token_ids=(0, 2, 3, 4),
        context_text="abc",
        metadata={},
    )
    data = {
        TensorKey.DOC_IDS: t.tensor([0, 0]),
        TensorKey.SEGMENT_INDICES: t.tensor([0, 1]),
        TensorKey.SEGMENT_CHAR_STARTS: t.tensor([0, 2]),
        TensorKey.SEGMENT_CHAR_ENDS: t.tensor([2, 4]),
        TensorKey.OFFSETS: t.tensor([0, 3, 6]),
        TensorKey.FEATURE_IDS: t.tensor([1, 2, 3, 4, 5, 6]),
        TensorKey.FEATURE_MAGS: t.tensor([0.9, 0.5, 0.2, 0.8, 0.4, 0.1]),
    }
    shard_entry = context_build._entry_from_shard_doc(
        record=record,
        data=data,
        local_doc_id=0,
        pipeline=_pipeline(),
        n_sae_features=PositiveInt.parse(8),
        num_buckets=PositiveInt.parse(2),
        tokenizer=object(),
        source_bos_char_len=NonNegativeInt.parse(0),
        source="fixture shard",
    )
    in_memory_entry = context_build.build_context_entry_from_extracted_segments(
        token_ids=record.token_ids,
        segment_feature_ids=(t.tensor([1, 2, 3]), t.tensor([4, 5, 6])),
        segment_feature_magnitudes=(t.tensor([0.9, 0.5, 0.2]), t.tensor([0.8, 0.4, 0.1])),
        segment_char_starts=(0, 2),
        segment_char_ends=(2, 4),
        pipeline=_pipeline(),
        n_sae_features=PositiveInt.parse(8),
        num_buckets=PositiveInt.parse(2),
        tokenizer=object(),
        source_bos_char_len=NonNegativeInt.parse(0),
        source="in-memory fixture",
    )

    assert in_memory_entry == shard_entry
    assert len(in_memory_entry.segments) == 2
    for segment in in_memory_entry.segments:
        assert len(segment.feature_ids) == 2
        assert segment.buckets == (0, 1)
        assert len(set(segment.feature_ids)) == 2


def test_fixed_alignment_failure_precedes_selection_and_preserves_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _pipeline()
    assert isinstance(pipeline.selection_config, SamplingBundle)
    assert pipeline.negative_bundle is not None
    positive_before = pipeline.selection_config.rng.get_state().clone()
    negative_before = pipeline.negative_bundle.rng.get_state().clone()

    def fail_alignment(*_args, **_kwargs):
        raise ValueError("frozen alignment failure")

    monkeypatch.setattr(
        context_build,
        "compute_doc_token_boundaries_from_tokens",
        fail_alignment,
    )
    monkeypatch.setattr(
        context_build,
        "_select_and_bucket_segment",
        lambda *_args, **_kwargs: pytest.fail("selection ran before alignment"),
    )

    with pytest.raises(ValueError, match="token-boundary alignment failed.*frozen"):
        context_build.build_context_entry_from_extracted_segments(
            token_ids=(0, 2),
            segment_feature_ids=(t.tensor([1, 2]),),
            segment_feature_magnitudes=(t.tensor([0.9, 0.2]),),
            segment_char_starts=(0,),
            segment_char_ends=(2,),
            pipeline=pipeline,
            n_sae_features=PositiveInt.parse(8),
            num_buckets=PositiveInt.parse(2),
            tokenizer=object(),
            source_bos_char_len=NonNegativeInt.parse(0),
        )

    assert t.equal(pipeline.selection_config.rng.get_state(), positive_before)
    assert t.equal(pipeline.negative_bundle.rng.get_state(), negative_before)


def test_fixed_selection_failure_occurs_after_complete_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def align(*_args, **_kwargs):
        events.append("alignment")
        return t.tensor([0]), t.tensor([2])

    def offsets(*_args, **_kwargs):
        events.append("offsets")
        return [0, 1, 2]

    def fail_selection(*_args, **_kwargs):
        events.append("selection")
        raise RuntimeError("frozen selection failure")

    monkeypatch.setattr(context_build, "compute_doc_token_boundaries_from_tokens", align)
    monkeypatch.setattr(context_build, "token_to_char_offsets", offsets)
    monkeypatch.setattr(context_build, "_select_and_bucket_segment", fail_selection)

    with pytest.raises(RuntimeError, match="frozen selection failure"):
        context_build.build_context_entry_from_extracted_segments(
            token_ids=(0, 2),
            segment_feature_ids=(t.tensor([1]),),
            segment_feature_magnitudes=(t.tensor([0.9]),),
            segment_char_starts=(0,),
            segment_char_ends=(2,),
            pipeline=_pipeline(),
            n_sae_features=PositiveInt.parse(8),
            num_buckets=PositiveInt.parse(2),
            tokenizer=object(),
            source_bos_char_len=NonNegativeInt.parse(0),
        )

    assert events == ["alignment", "offsets", "selection"]


def test_in_memory_conversion_rejects_parallel_length_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_alignment(monkeypatch)
    with pytest.raises(ValueError, match="must have equal lengths"):
        context_build.build_context_entry_from_extracted_segments(
            token_ids=(0, 2, 3),
            segment_feature_ids=(t.tensor([1]),),
            segment_feature_magnitudes=(t.tensor([0.5]), t.tensor([0.4])),
            segment_char_starts=(0,),
            segment_char_ends=(2,),
            pipeline=_pipeline(),
            n_sae_features=PositiveInt.parse(8),
            num_buckets=PositiveInt.parse(2),
            tokenizer=object(),
            source_bos_char_len=NonNegativeInt.parse(0),
        )


@pytest.mark.parametrize("mismatch", ["sae-width", "feature-mask"])
def test_in_memory_conversion_rejects_mismatched_cluster_contract(
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    _patch_alignment(monkeypatch)
    pipeline_mask = FeatureMask("whitelist", (0, 1, 2, 3))
    cluster_mask = (
        FeatureMask("whitelist", (0, 1, 2)) if mismatch == "feature-mask" else pipeline_mask
    )
    clusters = _consolidation(
        n_sae_features=9 if mismatch == "sae-width" else 8,
        feature_mask=cluster_mask,
    )

    with pytest.raises(
        FeatureClustersError,
        match="SAE width|different FeatureMask",
    ):
        context_build.build_context_entry_from_extracted_segments(
            token_ids=(0, 2),
            segment_feature_ids=(t.tensor([0, 1]),),
            segment_feature_magnitudes=(t.tensor([0.5, 0.4]),),
            segment_char_starts=(0,),
            segment_char_ends=(2,),
            pipeline=_pipeline(feature_mask=pipeline_mask),
            n_sae_features=PositiveInt.parse(8),
            num_buckets=PositiveInt.parse(2),
            tokenizer=object(),
            source_bos_char_len=NonNegativeInt.parse(0),
            feature_clusters=clusters,
        )


def _selected_kwargs(**overrides):
    """Baseline arguments for the selected-segments builder over a real render."""

    (
        tokenizer,
        prompt,
        _turn_starts,
        source_bos_char_len,
        source_starts,
        source_ends,
    ) = _turn_render_and_source_ranges()
    n_segments = len(source_starts)
    return {
        "token_ids": prompt,
        "segment_feature_ids": [t.tensor([10 + index]) for index in range(n_segments)],
        "segment_buckets": [t.tensor([index % 2]) for index in range(n_segments)],
        "segment_char_starts": source_starts,
        "segment_char_ends": source_ends,
        "tokenizer": tokenizer,
        "source_bos_char_len": NonNegativeInt.parse(source_bos_char_len),
        "source": "probe",
        **overrides,
    }


def test_selected_segment_conversion_accepts_the_unmutated_baseline() -> None:
    # Non-vacuity partner for the rejection matrix below: the same kwargs with
    # no perturbation must build an entry, so each raise is caused by the
    # perturbation and not by the fixture.
    kwargs = _selected_kwargs()
    entry = context_build.build_context_entry_from_selected_segments(**kwargs)

    assert len(entry.segments) == 4
    # kills: a routing-based builder that disagrees with the fixed-cache
    # converter on ORDINARY geometry (no straddle), which both must own alike.
    token_starts, token_ends = compute_doc_token_boundaries_from_tokens(
        list(kwargs["token_ids"]),
        list(kwargs["segment_char_starts"]),
        list(kwargs["segment_char_ends"]),
        kwargs["tokenizer"],
        int(kwargs["source_bos_char_len"]),
    )
    converter_owners: list[tuple[int, ...] | None] = [None] * len(kwargs["token_ids"])
    for window, (first, last) in enumerate(
        zip(token_starts.tolist(), token_ends.tolist(), strict=True)
    ):
        for index in range(first, last):
            converter_owners[index] = (10 + window,)
    assert (
        _token_owners_from_entry(entry, kwargs["tokenizer"], kwargs["token_ids"])
        == converter_owners
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param(
            {"segment_buckets": [t.tensor([10])]},
            r"zip\(\)",
            id="ragged-parallel-arrays",
        ),
        pytest.param(
            {
                "segment_feature_ids": [],
                "segment_buckets": [],
                "segment_char_starts": [],
                "segment_char_ends": [],
            },
            r"probe: selected context has no segments",
            id="empty-segment-list",
        ),
        pytest.param(
            {"segment_char_ends": _selected_kwargs()["segment_char_ends"][:-1] + [5]},
            r"probe: segment routing failed: Segment 3 has non-positive length",
            id="routing-failure-wrapping",
        ),
        pytest.param(
            {"segment_feature_ids": [t.tensor([[10]])] * 4},
            r"probe: selected segment 0 ids/buckets must be 1-D",
            id="non-1d-ids",
        ),
        pytest.param(
            {"segment_buckets": [t.tensor([0, 1])] * 4},
            r"probe: selected segment 0 ids/buckets lengths differ \(1 vs 2\)",
            id="ids-buckets-width-mismatch",
        ),
    ],
)
def test_selected_segment_conversion_rejects_malformed_inputs(
    overrides: dict,
    message: str,
) -> None:
    # kills: turning any of the selected-path guards into `if False:`. Each
    # case perturbs exactly one input of the accepted baseline above.
    with pytest.raises(ValueError, match=message):
        context_build.build_context_entry_from_selected_segments(**_selected_kwargs(**overrides))


def _token_owners_from_entry(entry, tokenizer, token_ids) -> list[tuple[int, ...] | None]:
    """Feature ids owning each target token, read back from the entry's char ranges."""
    offsets = token_to_char_offsets(tokenizer, token_ids)
    owners: list[tuple[int, ...] | None] = []
    for index in range(len(token_ids)):
        start, end = offsets[index], offsets[index + 1]
        owner = [
            segment.feature_ids
            for segment in entry.segments
            if segment.char_start <= start and end <= segment.char_end
        ]
        assert len(owner) <= 1, f"token {index} is claimed by {len(owner)} segments"
        owners.append(owner[0] if owner else None)
    return owners


def _token_owners_from_training_routing(
    token_ids, tokenizer, fids, buckets, starts, ends, source_bos_char_len
) -> list[tuple[int, ...] | None]:
    routing = build_full_sequence_segment_routing(
        ids=list(token_ids),
        tokenizer=tokenizer,
        seg_features=(list(fids), list(buckets), list(starts), list(ends)),
        source_bos_char_len=source_bos_char_len,
    )
    return [
        None if slot == 0 else tuple(routing.per_seg_fids[slot].tolist())
        for slot in routing.seg_map.tolist()
    ]


def _selected_entry(token_ids, tokenizer, starts, ends, source_bos_char_len):
    fids = [t.tensor([10 + index]) for index in range(len(starts))]
    buckets = [t.tensor([0]) for _ in starts]
    entry = context_build.build_context_entry_from_selected_segments(
        token_ids=token_ids,
        segment_feature_ids=fids,
        segment_buckets=buckets,
        segment_char_starts=starts,
        segment_char_ends=ends,
        tokenizer=tokenizer,
        source_bos_char_len=NonNegativeInt.parse(source_bos_char_len),
        source="straddle",
    )
    expected = _token_owners_from_training_routing(
        token_ids, tokenizer, fids, buckets, starts, ends, source_bos_char_len
    )
    return entry, expected


# A source window covering only the last character of ``<user_end>`` and the
# first of ``<assistant_start>`` is what a Gemma token straddling that chat
# boundary leaves behind. Under max-overlap routing neither structural token
# prefers it, so it owns nothing in training and must own nothing here.
# kills: reinstating the fixed-cache converter on the selected path (it raises
# "Segments with no tokens: [3]") or dropping the empty-slot skip.
def test_selected_turn_conversion_drops_a_window_that_owns_no_target_token() -> None:
    tokenizer, prompt, _turn_starts, bos, starts, ends = _turn_render_and_source_ranges()
    boundary = starts[-1]
    straddle_starts = [*starts[:-1], boundary - 1, boundary + 1]
    straddle_ends = [*ends[:-2], boundary - 1, boundary + 1, ends[-1]]
    with pytest.raises(ValueError, match=r"Segments with no tokens: \[3\]"):
        compute_doc_token_boundaries_from_tokens(
            list(prompt), straddle_starts, straddle_ends, tokenizer, bos
        )

    entry, expected = _selected_entry(prompt, tokenizer, straddle_starts, straddle_ends, bos)

    assert _token_owners_from_entry(entry, tokenizer, prompt) == expected
    assert (13,) not in {segment.feature_ids for segment in entry.segments}
    assert len(entry.segments) == len(straddle_starts) - 1
    real_tokens, real_segments = _prepare_turn_entry(tokenizer, prompt, entry)
    assert real_tokens == prompt[:-1]
    assert real_segments == entry.segments[:-1]


# The fit-row shape: a complete conversation whose final source window is the
# lone closing character of ``<assistant_end>``. The fixed-cache converter
# silently reassigns that token to the tiny window; training leaves it with the
# window that overlaps it most.
# kills: the same reinstatement, which here changes ownership without raising.
def test_selected_conversion_keeps_training_ownership_for_a_trailing_straddle_window() -> None:
    tokenizer = _CharacterTokenizer()
    ids, _mask = tokenizer.render_conversation(
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]}
    )
    text, turn_starts = rendered_text_and_turn_char_starts(ids, tokenizer)
    bos = 11
    starts = [0, bos + turn_starts[1], bos + len(text) - 1]
    ends = [*starts[1:], bos + len(text)]
    token_starts, token_ends = compute_doc_token_boundaries_from_tokens(
        list(ids), starts, ends, tokenizer, bos
    )
    assert int(token_starts[-1]) == len(ids) - 1, "fixed converter moved <assistant_end>"

    entry, expected = _selected_entry(ids, tokenizer, starts, ends, bos)

    assert _token_owners_from_entry(entry, tokenizer, ids) == expected
    assert expected[-1] == (11,)
    assert entry.segments[-1].char_end == token_to_char_offsets(tokenizer, ids)[-1]


def _real_tokenizers():
    nanochat_tokenizer = get_tokenizer_or_skip()
    try:
        gemma = AutoTokenizer.from_pretrained("google/gemma-3-4b-pt", local_files_only=True)
    except OSError as exc:
        pytest.skip(f"gemma-3-4b-pt tokenizer is not in the local HF cache: {exc}")
    return nanochat_tokenizer, gemma


def _real_render(tokenizer, turns, *, open_assistant: bool):
    specials = {
        name: tokenizer.encode_special(name)
        for name in ("<user_start>", "<user_end>", "<assistant_start>", "<assistant_end>")
    }
    ids = [tokenizer.get_bos_token_id()]
    for role, content in turns:
        ids.append(specials[f"<{role}_start>"])
        ids.extend(tokenizer.encode(content))
        ids.append(specials[f"<{role}_end>"])
    if open_assistant:
        ids.append(specials["<assistant_start>"])
    return ids


# The exact geometry that aborted the first 3B AxBench run: with the real
# tokenizers, 22 repetitions of "Hello " leave the Gemma "><" straddle token as
# its own 32-token window. Runs only where both real tokenizers are cached.
@pytest.mark.parametrize(
    ("turns", "open_assistant", "segment_size", "straddles"),
    [
        pytest.param([("user", "Hello " * 22)], True, 32, True, id="prompt-seg32"),
        pytest.param(
            [("user", "hi"), ("assistant", "Hello " * 24)], False, 32, True, id="fit-row-seg32"
        ),
        pytest.param([("user", "Hello " * 118)], True, 128, True, id="prompt-seg128"),
        pytest.param([("user", "Tell me a joke")], True, 32, False, id="control"),
    ],
)
def test_selected_turn_conversion_matches_training_routing_with_real_tokenizers(
    turns, open_assistant, segment_size, straddles
) -> None:
    nanochat_tokenizer, gemma = _real_tokenizers()
    ids = _real_render(nanochat_tokenizer, turns, open_assistant=open_assistant)
    text, turn_starts = rendered_text_and_turn_char_starts(ids, nanochat_tokenizer)
    gemma_bos = resolve_bos_token_id(gemma)
    kept = [gemma_bos, *gemma.encode(text, add_special_tokens=False)]
    _segments, starts, ends, _drift = turn_segments_for_document(
        kept, gemma, text, turn_starts, PositiveInt.parse(segment_size)
    )
    bos = len(gemma.decode([gemma_bos]))
    try:
        token_starts, token_ends = compute_doc_token_boundaries_from_tokens(
            list(ids), list(starts), list(ends), nanochat_tokenizer, bos
        )
    except ValueError as exc:
        assert straddles and "Segments with no tokens" in str(exc)
        converter_owners = None
    else:
        converter_owners: list[tuple[int, ...] | None] = [None] * len(ids)
        for window, (first, last) in enumerate(
            zip(token_starts.tolist(), token_ends.tolist(), strict=True)
        ):
            for index in range(first, last):
                converter_owners[index] = (10 + window,)

    entry, expected = _selected_entry(ids, nanochat_tokenizer, list(starts), list(ends), bos)

    assert _token_owners_from_entry(entry, nanochat_tokenizer, ids) == expected
    assert len(entry.segments) == len(starts) - (1 if straddles else 0)
    if converter_owners is not None:
        # The fit-row shape does not raise in the fixed-cache converter; it
        # silently disagrees with training about the final token instead.
        assert (converter_owners != expected) is straddles
    if open_assistant:
        real_tokens, real_segments = _prepare_turn_entry(nanochat_tokenizer, ids, entry)
        assert real_tokens == ids[:-1]
        assert real_segments == entry.segments[:-1]


class _ZeroWidthTokenizer(_CharacterTokenizer):
    """The character double plus one id that decodes to nothing (a UTF-8 byte piece).

    Fidelity anchor: the real-tokenizer emoji cases below; this double only
    makes which assignment rule a consumer dispatches to observable.
    """

    ZERO_WIDTH = 900

    def decode(self, token_ids) -> str:
        return super().decode([token_id for token_id in token_ids if token_id != self.ZERO_WIDTH])


def _assignment_owners(assignments, entry) -> list[tuple[int, ...]]:
    return [entry.segments[index].feature_ids for index in assignments.tolist()]


# Two zero-width tokens sit exactly on the boundary between the last user
# window and the marker window. Training routes them with the FOLLOWING window
# (their expanded interval overlaps it); the fixed-cache assignment keeps them
# with the preceding one. Character ranges cannot tell the two apart, so the
# turn consumer must derive ownership with the training primitive.
# kills: assigning turn-mode tokens with compute_context_token_segment_assignments.
def test_turn_assignments_follow_training_for_zero_width_boundary_tokens() -> None:
    tokenizer = _ZeroWidthTokenizer()
    _t, prompt, _turn_starts, bos, starts, ends = _turn_render_and_source_ranges()
    marker = prompt[-1]
    prompt = [*prompt[:-1], tokenizer.ZERO_WIDTH, tokenizer.ZERO_WIDTH, marker]
    entry, expected = _selected_entry(prompt, tokenizer, starts, ends, bos)
    boundary = len(prompt) - 3

    turn_owners = _assignment_owners(
        turn_context_token_assignments(
            tokenizer=tokenizer, context_tokens=prompt, context_segments=entry.segments
        ),
        entry,
    )
    fixed_owners = _assignment_owners(
        compute_context_token_segment_assignments(
            tokenizer=tokenizer, context_tokens=prompt, context_segments=entry.segments
        ),
        entry,
    )

    assert turn_owners == expected
    assert turn_owners[boundary : boundary + 2] == [expected[-1], expected[-1]]
    assert fixed_owners[boundary : boundary + 2] == [expected[boundary - 1]] * 2
    assert fixed_owners != turn_owners


# Real tokenizers: emoji become multi-byte RustBPE pieces, some of zero width.
@pytest.mark.parametrize(
    ("content", "segment_size"),
    [
        pytest.param("a " + "\U0001f642" * 30, 32, id="emoji30-seg32"),
        pytest.param("a " + "\U0001f642" * 200, 128, id="emoji200-seg128"),
    ],
)
def test_turn_assignments_match_training_for_real_zero_width_pieces(content, segment_size) -> None:
    nanochat_tokenizer, gemma = _real_tokenizers()
    ids = _real_render(nanochat_tokenizer, [("user", content)], open_assistant=True)
    text, turn_starts = rendered_text_and_turn_char_starts(ids, nanochat_tokenizer)
    gemma_bos = resolve_bos_token_id(gemma)
    kept = [gemma_bos, *gemma.encode(text, add_special_tokens=False)]
    _segments, starts, ends, _drift = turn_segments_for_document(
        kept, gemma, text, turn_starts, PositiveInt.parse(segment_size)
    )
    bos = len(gemma.decode([gemma_bos]))
    offsets = token_to_char_offsets(nanochat_tokenizer, ids)
    assert any(offsets[i] == offsets[i + 1] for i in range(len(ids))), (
        "fixture must contain zero-width pieces"
    )

    entry, expected = _selected_entry(ids, nanochat_tokenizer, list(starts), list(ends), bos)

    turn_owners = _assignment_owners(
        turn_context_token_assignments(
            tokenizer=nanochat_tokenizer, context_tokens=ids, context_segments=entry.segments
        ),
        entry,
    )
    fixed_owners = _assignment_owners(
        compute_context_token_segment_assignments(
            tokenizer=nanochat_tokenizer, context_tokens=ids, context_segments=entry.segments
        ),
        entry,
    )
    assert turn_owners == expected
    assert fixed_owners != expected


# kills: returning ``seg_map - 1`` unguarded -- a trailing zero-width piece
# that no window covers would come back as -1 and silently index the last
# encoder slot downstream.
def test_turn_assignments_reject_a_token_no_window_covers() -> None:
    tokenizer = _ZeroWidthTokenizer()
    _t, prompt, _turn_starts, bos, starts, ends = _turn_render_and_source_ranges()
    entry, _expected = _selected_entry(prompt, tokenizer, starts, ends, bos)

    with pytest.raises(
        ValueError, match=rf"leave token {len(prompt)} unrouted \(1 of {len(prompt) + 1}"
    ):
        turn_context_token_assignments(
            tokenizer=tokenizer,
            context_tokens=[*prompt, tokenizer.ZERO_WIDTH],
            context_segments=entry.segments,
        )


# kills: reverting the wrapper's turn branch to the fixed-cache assignment (the
# zero-width boundary pieces then land in the preceding window).
def test_wrapper_turn_branch_dispatches_to_the_training_assignment() -> None:
    tokenizer = _ZeroWidthTokenizer()
    _t, prompt, _turn_starts, bos, starts, ends = _turn_render_and_source_ranges()
    marker = prompt[-1]
    prompt = [*prompt[:-1], tokenizer.ZERO_WIDTH, tokenizer.ZERO_WIDTH, marker]
    entry, expected = _selected_entry(prompt, tokenizer, starts, ends, bos)
    fill = TargetAwareEncoderDecoderEvalWrapper._fill_context_assignments.__get__(
        SimpleNamespace(
            tokenizer=tokenizer, is_turn_self_conditioned=True,
            self_cond_segment_mode=entry.segment_mode, context_provider=entry,
        ),
        TargetAwareEncoderDecoderEvalWrapper,
    )
    row_seg_map = t.full((len(prompt),), -1, dtype=t.long)

    fill(row_seg_map, prompt, entry.segments)

    assert _assignment_owners(row_seg_map, entry) == expected
    assert _assignment_owners(row_seg_map, entry) != _assignment_owners(
        compute_context_token_segment_assignments(
            tokenizer=tokenizer, context_tokens=prompt, context_segments=entry.segments
        ),
        entry,
    )


# A prior assistant turn with a python tool call: "<output_end>,hello" makes a
# Gemma token ">," straddle the unmarked tool-to-assistant boundary, so training
# routes the assistant's first token with the tool window and no window starts
# at that logical boundary. The prompt is valid and must be accepted with
# training-identical ownership.
# kills: requiring every interior logical-turn start to be a window start.
@pytest.mark.parametrize("segment_size", [32, 128])
def test_turn_prefill_accepts_a_straddled_tool_boundary_with_real_tokenizers(
    segment_size: int,
) -> None:
    nanochat_tokenizer, gemma = _real_tokenizers()
    rendered, _mask = nanochat_tokenizer.render_conversation(
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "python", "text": "print(42)"},
                        {"type": "python_output", "text": "42"},
                        {"type": "text", "text": ",hello"},
                    ],
                },
                {"role": "user", "content": "thanks"},
                {"role": "assistant", "content": ""},
            ]
        }
    )
    prompt = rendered[:-1]
    assert prompt[-1] == nanochat_tokenizer.encode_special("<assistant_start>")
    text, turn_starts = rendered_text_and_turn_char_starts(prompt, nanochat_tokenizer)
    gemma_bos = resolve_bos_token_id(gemma)
    kept = [gemma_bos, *gemma.encode(text, add_special_tokens=False)]
    _segments, starts, ends, _drift = turn_segments_for_document(
        kept, gemma, text, turn_starts, PositiveInt.parse(segment_size)
    )
    bos = len(gemma.decode([gemma_bos]))
    entry, expected = _selected_entry(prompt, nanochat_tokenizer, list(starts), list(ends), bos)
    offsets = token_to_char_offsets(nanochat_tokenizer, prompt)
    spans = legacy_turn_spans(prompt, ChatSpecialTokens.from_tokenizer(nanochat_tokenizer))
    logical_starts = {int(offsets[span.start + 1]) for span in spans[1:]}
    assert not logical_starts <= {segment.char_start for segment in entry.segments}, (
        "fixture must contain a logical-turn start that is not a window start"
    )

    real_tokens, real_segments = _prepare_turn_entry(nanochat_tokenizer, prompt, entry)

    assert real_tokens == prompt[:-1]
    assert real_segments == entry.segments[:-1]
    assert (
        _assignment_owners(
            turn_context_token_assignments(
                tokenizer=nanochat_tokenizer, context_tokens=prompt, context_segments=entry.segments
            ),
            entry,
        )
        == expected
    )


@pytest.mark.parametrize("mode", list(SegmentMode))
def test_shared_context_dispatch_preserves_each_modes_ownership(mode: SegmentMode) -> None:
    tokenizer = _ZeroWidthTokenizer()
    _t, prompt, _turn_starts, bos, starts, ends = _turn_render_and_source_ranges()
    prompt = [*prompt[:-1], tokenizer.ZERO_WIDTH, tokenizer.ZERO_WIDTH, prompt[-1]]
    entry, _expected = _selected_entry(prompt, tokenizer, starts, ends, bos)
    reference = (
        turn_context_token_assignments
        if mode is SegmentMode.turn_delimited_fixed_tokens
        else compute_context_token_segment_assignments
    )
    expected = reference(tokenizer=tokenizer, context_tokens=prompt, context_segments=entry.segments)
    actual = context_token_assignments(
        segment_mode=mode, tokenizer=tokenizer, context_tokens=prompt, context_segments=entry.segments
    )
    assert t.equal(actual, expected)
    if mode is SegmentMode.turn_delimited_fixed_tokens:
        persisted = context_token_assignments(
            segment_mode=mode, tokenizer=object(), context_tokens=prompt,
            context_segments=entry.segments, token_assignments=entry.token_assignments,
        )
        assert t.equal(persisted, expected)


@pytest.mark.parametrize("mutation", ["short", "negative", "outside", "backwards", "bool"])
def test_context_entry_rejects_invalid_persisted_ownership(mutation: str) -> None:
    tokenizer, prompt, _turn_starts, bos, starts, ends = _turn_render_and_source_ranges()
    entry, _expected = _selected_entry(prompt, tokenizer, starts, ends, bos)
    owners = list(entry.token_assignments)
    if mutation == "short":
        owners.pop()
    elif mutation == "negative":
        owners[0] = -1
    elif mutation == "outside":
        owners[-1] = len(entry.segments)
    elif mutation == "backwards":
        owners[0] = 1
    else:
        owners[0] = True
    with pytest.raises(ValueError, match="ownership"):
        replace(entry, token_assignments=tuple(owners))
