"""The conditioned-SFT generator's steered FP prediction-group split.

``seg_doc_id`` is the feature-prediction *prediction-group* id: consecutive
segments may predict each other only inside one group. A group boundary sits at
every packed conversation boundary and, for a **steered** conversation, also at
its first assistant-turn segment — the steered response's features are
deliberately perturbed, so the head must not learn to predict them from the
unperturbed user turn.

These tests pin the EXACT ``seg_doc_id`` row the generator emits for every
truncation shape, plus the byte-identity of everything non-steered, using the
``_CharacterTokenizer`` from ``tests/nanochat/test_structured_conversation_renderer``:
unlike ``SFTMockTokenizer`` it has real ``encode_special`` ids (so
``legacy_turn_spans`` works) and decodes every special token to its literal
string (so ``token_to_char_offsets`` is exact and segment ``char_start``s can be
placed on deliberately chosen token boundaries).
"""

from __future__ import annotations

import ast
import copy
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch as t

import scripts.train.conditional_encdec_sft as sft_script
from errors import DataLoadError
from scripts.preprocess.convert_sft_to_parquet import prepare_messages_for_full_nanochat_render
from scripts.train.conditional_encdec_sft import SFTState, make_sft_data_generator
from steering.conditional.encoder_decoder.full_sequence_routing import (
    source_aligned_char_offsets,
)
from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.percentile_table import build_percentile_table
from steering.conditional.thresholding import ThresholdConfig, ThresholdType
from steering.segmentation import SegmentMode
from steering.sft.turn_spans import ChatSpecialTokens, legacy_turn_spans
from steering.sft_schema import SFTSplit
from tasks.common import Task
from tasks.steered import SteeredConversations, SteeredTaskMixture
from tests.encdec_test_helpers import N_SAE_FEATURES, NUM_BUCKETS
from tests.nanochat.test_structured_conversation_renderer import _CharacterTokenizer
from tests.steering.test_encdec_sft_conditioned_integration import patch_load_sft_features
from tests.steering.test_encdec_sft_feature_prediction import _make_per_feature_stats

# ``_CharacterTokenizer``'s own BOS decodes to exactly 5 characters, so at
# ``BOS_CHARS = 5`` ``source_aligned_char_offsets`` returns its raw decode
# offsets unchanged (delta 0) and a segment ``char_start`` equals the character
# position of the token it starts on.  Every helper below takes ``bos_chars``
# so a test can run the whole fixture at a NON-ZERO delta, where forgetting
# ``feature_source_bos_char_len`` stops being a no-op.
BOS_CHARS = 5


# ---------------------------------------------------------------------------
# Fixtures: tasks, geometry, synthetic features, generator
# ---------------------------------------------------------------------------


class _PlainTask(Task):
    """Single-turn conversations of a fixed rendered size."""

    def __init__(self, n: int, user_chars: int, asst_chars: int) -> None:
        super().__init__()
        self._n = n
        self._user_chars = user_chars
        self._asst_chars = asst_chars

    def num_examples(self) -> int:
        return self._n

    def get_example(self, index: int) -> dict:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"b{index}".ljust(self._user_chars, "x")[: self._user_chars],
                },
                {
                    "role": "assistant",
                    "content": f"c{index}".ljust(self._asst_chars, "y")[: self._asst_chars],
                },
            ]
        }


def _write_steered_jsonl(path: Path, *, rows: int, user_chars: int, asst_chars: int) -> str:
    path.write_text(
        "".join(
            json.dumps(
                [
                    {"role": "user", "content": f"s{index}".ljust(user_chars, "p")[:user_chars]},
                    {
                        "role": "assistant",
                        "content": f"t{index}".ljust(asst_chars, "q")[:asst_chars],
                    },
                ]
            )
            + "\n"
            for index in range(rows)
        ),
        encoding="utf-8",
    )
    return str(path)


def _steered_mixture(
    tmp_path: Path,
    *,
    base_user: int,
    base_asst: int,
    steered_user: int = 12,
    steered_asst: int = 16,
    base_rows: int = 3,
    steered_rows: int = 1,
) -> SteeredTaskMixture:
    jsonl = _write_steered_jsonl(
        tmp_path / "steered.jsonl",
        rows=steered_rows,
        user_chars=steered_user,
        asst_chars=steered_asst,
    )
    return SteeredTaskMixture(
        [_PlainTask(base_rows, base_user, base_asst)], SteeredConversations(jsonl)
    )


def _geometry(
    tokenizer, conversation: dict, *, bos_chars: int = BOS_CHARS
) -> tuple[list[int], list[int], int]:
    """Return ``(ids, source char offsets, assistant-turn token index)``.

    The generator renders the converter's slice, so this must too: measuring the
    untrimmed conversation would fabricate coverage the real pipeline never
    produces.
    """
    messages, skip_reason = prepare_messages_for_full_nanochat_render(conversation)
    assert messages is not None, f"fixture conversation would be skipped ({skip_reason})"
    ids, _mask = tokenizer.render_conversation({"messages": messages})
    offsets = source_aligned_char_offsets(tokenizer, ids, source_bos_char_len=bos_chars)
    spans = legacy_turn_spans(ids, ChatSpecialTokens.from_tokenizer(tokenizer))
    # Target-space span start ``j`` is BOS-inclusive token index ``j + 1``.
    return ids, offsets, spans[1].start + 1


def _build_features(
    task, tokenizer, seg_token_indices, *, bos_chars: int = BOS_CHARS
) -> dict[int, list[SegmentFeatures]]:
    """Synthetic features whose segments start on chosen TOKEN boundaries.

    ``seg_token_indices(position, ids, assistant_token)`` picks the token index
    each segment starts on, so a test can place a boundary exactly at the
    assistant turn, one token before it, or well inside either turn.
    """
    return _build_features_at_chars(
        task,
        tokenizer,
        lambda position, ids, offsets, assistant_token: [
            offsets[index] for index in seg_token_indices(position, ids, assistant_token)
        ],
        bos_chars=bos_chars,
    )


def _build_features_at_chars(
    task, tokenizer, seg_char_starts, *, bos_chars: int = BOS_CHARS
) -> dict[int, list[SegmentFeatures]]:
    """Synthetic features at arbitrary SOURCE-CHARACTER starts.

    ``seg_char_starts(position, ids, offsets, assistant_token)`` returns the
    starts directly, so a segment can begin part-way INTO a multi-character
    special token — the only way a synthetic fixture can reproduce what real
    cross-tokenizer extraction does, and therefore the only way to build a
    conversation whose trailing in-range segment routing drops.
    """
    features: dict[int, list[SegmentFeatures]] = {}
    cursor = 0
    for position in range(len(task)):
        ids, offsets, assistant_token = _geometry(tokenizer, task[position], bos_chars=bos_chars)
        starts = seg_char_starts(position, ids, offsets, assistant_token)
        ends = [*starts[1:], offsets[-1]]
        segments: list[SegmentFeatures] = []
        for index, (char_start, char_end) in enumerate(zip(starts, ends, strict=True)):
            feature_ids = t.tensor(
                [(cursor + offset) % N_SAE_FEATURES for offset in range(3)], dtype=t.int32
            )
            cursor = (cursor + 3) % N_SAE_FEATURES
            segments.append(
                SegmentFeatures(
                    segment_idx=index,
                    position=index,
                    feature_ids=feature_ids,
                    feature_mags=t.tensor([3.0, 2.0, 1.0], dtype=t.float32),
                    char_start=char_start,
                    char_end=char_end,
                )
            )
        features[position] = segments
    return features


def _four_segments(position: int, ids: list[int], assistant_token: int) -> list[int]:
    """Two user-turn segments then two assistant-turn segments.

    Short base conversations cannot host four, so they get the minimal split
    that still puts one segment in each turn.
    """
    if len(ids) < assistant_token + 9:
        return [0, assistant_token]
    return [0, assistant_token - 3, assistant_token, assistant_token + 8]


def _make_generator(
    task,
    tokenizer,
    *,
    B: int,
    T: int,
    steered_positions: frozenset[int] | None,
    buffer_size: int = 4,
    min_loss_tokens: int = 1,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    bos_chars: int = BOS_CHARS,
    sft_state: SFTState | None = None,
    single_pass: bool = False,
):
    boundaries = (
        t.linspace(0.5, float(NUM_BUCKETS) - 0.5, NUM_BUCKETS - 1)
        .unsqueeze(0)
        .expand(N_SAE_FEATURES, -1)
        .contiguous()
    )
    return make_sft_data_generator(
        task=task,
        tokenizer=tokenizer,
        B=B,
        T=T,
        device="cpu",
        sft_state=SFTState() if sft_state is None else sft_state,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        buffer_size=buffer_size,
        feature_dir="/dummy/features",
        parquet_dir="/dummy/parquet",
        threshold_cfg=ThresholdConfig(threshold_type=ThresholdType.segment_topk, threshold=1000),
        boundaries=boundaries,
        n_sae_features=N_SAE_FEATURES,
        fp_percentile_table=build_percentile_table(
            _make_per_feature_stats(N_SAE_FEATURES), N_SAE_FEATURES
        ),
        return_seg_doc_id=True,
        min_loss_tokens=min_loss_tokens,
        alignment_probe_samples=0,
        feature_source_bos_char_len=bos_chars,
        expected_split=SFTSplit.TRAIN,
        steered_positions=steered_positions,
        single_pass=single_pass,
    )


def _first_seg_doc_id(task, tokenizer, features, **generator_kwargs) -> list[int]:
    with patch_load_sft_features(features, segment_mode=SegmentMode.turn_delimited_fixed_tokens):
        batch = next(_make_generator(task, tokenizer, **generator_kwargs))
    return [int(value) for value in batch[-1][0].tolist()]


@pytest.fixture
def tokenizer() -> _CharacterTokenizer:
    return _CharacterTokenizer()


# ---------------------------------------------------------------------------
# The exact seg_doc_id row for every truncation shape
# ---------------------------------------------------------------------------


class TestSteeredSegDocIdRows:
    """Row-exact pins.

    The fixture mixture is ``[base(2/3 chars) x 3] + one steered(12/16)``, whose
    steered row lands at combined position 2 and is by far the longest
    conversation, so best-fit packing always puts it in row 0 alone. Its four
    feature segments start at tokens ``0``, ``assistant - 3``, ``assistant``,
    ``assistant + 8`` — two in the user turn, two in the assistant turn.
    """

    @pytest.fixture
    def case(self, tmp_path: Path, tokenizer: _CharacterTokenizer):
        task = _steered_mixture(tmp_path, base_user=2, base_asst=3)
        assert task.steered_positions == frozenset({2})
        return task, _build_features(task, tokenizer, _four_segments)

    @pytest.mark.parametrize(
        ("T", "expected", "why"),
        [
            (32, [-1, 0, 0, 1, 1], "untruncated: 2 user segments then 2 assistant segments"),
            (22, [-1, 0, 0, 1, 1], "cut inside the assistant turn: extra = 2nd assistant chunk"),
            (14, [-1, 0, 0, 1], "cut at the end of the user turn: extra = 1st assistant segment"),
            (11, [-1, 0, 0], "cut inside the user turn: no assistant segment survives"),
        ],
    )
    def test_seg_doc_id_row_is_exact(
        self,
        case,
        tokenizer: _CharacterTokenizer,
        T: int,
        expected: list[int],
        why: str,
    ) -> None:
        task, features = case
        assert (
            _first_seg_doc_id(
                task,
                tokenizer,
                features,
                B=1,
                T=T,
                steered_positions=frozenset(task.steered_positions),
            )
            == expected
        ), why

    def test_the_truncated_extra_segment_carries_the_assistant_group(
        self, case, tokenizer: _CharacterTokenizer
    ) -> None:
        """Case (b)'s last slot is a prediction-target-only segment.

        It has features but NO decoder positions, so ``seg_map`` never points at
        it — that is what makes it the extra segment rather than a fourth
        routed one, and it still belongs to the assistant group.
        """
        task, features = case
        with patch_load_sft_features(
            features, segment_mode=SegmentMode.turn_delimited_fixed_tokens
        ):
            batch = next(
                _make_generator(
                    task,
                    tokenizer,
                    B=1,
                    T=22,
                    steered_positions=frozenset(task.steered_positions),
                )
            )
        _inputs, _targets, _fids, _mags, enc_mask, seg_map, _fp, seg_doc_id = batch
        assert [int(value) for value in seg_doc_id[0].tolist()] == [-1, 0, 0, 1, 1]
        # Four real encoder segments, but only three of them route decoder tokens.
        assert int(enc_mask[0].any(dim=-1).sum()) == 4
        assert int(seg_map[0].max()) == 3

    def test_a_steered_conversation_packed_first_advances_the_group_for_what_follows(
        self, case, tokenizer: _CharacterTokenizer
    ) -> None:
        """The conversation after a steered one must open group 2, never re-use 1.

        The steered conversation (12/16 chars) is by far the longest, so at
        ``T=64`` best-fit packs it FIRST and three base conversations follow it
        in the same row. Re-using the steered assistant group for the next
        conversation would make ``(steered response) -> (next conversation's
        first segment)`` a valid FP source pair: a cross-conversation leak
        caused by the very feature that exists to add a boundary. Every other
        row-exact pin here packs the steered conversation last (or after a
        plain one), so this is the only test that exercises the advance.
        """
        task, features = case
        assert _first_seg_doc_id(
            task,
            tokenizer,
            features,
            B=1,
            T=64,
            steered_positions=frozenset(task.steered_positions),
        ) == [-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

    def test_a_segment_on_the_token_before_the_assistant_turn_stays_in_the_user_group(
        self, tmp_path: Path, tokenizer: _CharacterTokenizer
    ) -> None:
        """``spans[1].start + 1`` is the BOS-INCLUSIVE index of the assistant turn.

        Without the ``+ 1`` the boundary lands on the user turn's LAST token, so
        a segment starting there is wrongly promoted into the assistant group.
        ``_four_segments`` puts its second segment three tokens earlier, which
        is far enough away to survive the off-by-one — this fixture is not.
        """
        task = _steered_mixture(tmp_path, base_user=2, base_asst=3)

        def segments(position: int, ids: list[int], assistant_token: int) -> list[int]:
            if len(ids) < assistant_token + 9:
                return [0, assistant_token]
            return [0, assistant_token - 1, assistant_token, assistant_token + 8]

        features = _build_features(task, tokenizer, segments)
        assert _first_seg_doc_id(
            task,
            tokenizer,
            features,
            B=1,
            T=32,
            steered_positions=frozenset(task.steered_positions),
        ) == [-1, 0, 0, 1, 1]

    @pytest.mark.parametrize("bos_chars", [5, 15])
    def test_the_turn_boundary_is_read_in_the_feature_source_coordinate_system(
        self, tmp_path: Path, tokenizer: _CharacterTokenizer, bos_chars: int
    ) -> None:
        """The boundary must be read in FEATURE-SOURCE coordinates, not nanochat's.

        ``_CharacterTokenizer``'s BOS decodes to 5 characters, so at
        ``bos_chars=5`` the shift ``source_bos_char_len - nanochat_bos_len`` is
        zero and dropping ``feature_source_bos_char_len`` at the generator's
        boundary lookup changes nothing. At ``bos_chars=15`` the source system
        is shifted ``+10``: a generator that forgets the parameter reads the
        boundary 15 characters early, and the second USER segment (10
        characters before the turn in this fixture) lands in the assistant
        group.
        """
        task = _steered_mixture(tmp_path, base_user=2, base_asst=3)
        features = _build_features(task, tokenizer, _four_segments, bos_chars=bos_chars)
        assert _first_seg_doc_id(
            task,
            tokenizer,
            features,
            B=1,
            T=32,
            steered_positions=frozenset(task.steered_positions),
            bos_chars=bos_chars,
        ) == [-1, 0, 0, 1, 1]

    def test_a_row_packing_a_plain_then_a_steered_conversation_uses_three_groups(
        self, tmp_path: Path, tokenizer: _CharacterTokenizer
    ) -> None:
        """Conversation boundary AND steered turn boundary in the same row.

        The base conversations are longer here, so best-fit packs one of them
        first and the steered conversation second; the row therefore carries
        group 0 (the plain conversation), group 1 (the steered user turn) and
        group 2 (the steered response).
        """
        task = _steered_mixture(tmp_path, base_user=20, base_asst=24)
        assert task.steered_positions == frozenset({2})

        def segments(position: int, ids: list[int], assistant_token: int) -> list[int]:
            if position in task.steered_positions:
                return [0, assistant_token - 3, assistant_token, assistant_token + 8]
            return [0, assistant_token]

        features = _build_features(task, tokenizer, segments)
        assert _first_seg_doc_id(
            task,
            tokenizer,
            features,
            B=1,
            T=81,
            steered_positions=frozenset(task.steered_positions),
        ) == [-1, 0, 0, 1, 1, 2, 2]

    def test_a_trailing_segment_routing_drops_never_reaches_the_group_ids(
        self, tmp_path: Path, tokenizer: _CharacterTokenizer
    ) -> None:
        """Group ids are read over the segments actually APPENDED, not the source list.

        Routing legitimately drops trailing in-range segments that no decoder
        token reaches. Here the steered conversation's third segment begins one
        character before the end of the 15-character ``<assistant_end>`` token,
        so that token's midpoint still belongs to segment 1 and segment 2
        receives nothing. The two segments that DO get appended are both
        user-turn segments, so this conversation stays a single group and the
        next conversation opens group 1.

        Without the ``[: n_local_segs - 1]`` alignment slice the dropped
        segment's assistant group would be read back as ``seg_groups[-1]``, the
        per-row ``group_id`` would advance one too far, and every following
        conversation in the row would be renumbered.

        The slice is only observable when the conversation has NO
        prediction-target-only extra segment: with one, ``seg_groups[-1]`` is
        the extra segment's group either way. Hence an untruncated fixture.
        """
        task = _steered_mixture(tmp_path, base_user=2, base_asst=3)
        steered = frozenset(task.steered_positions)

        def starts(
            position: int, ids: list[int], offsets: list[int], assistant_token: int
        ) -> list[int]:
            if position not in steered:
                return [offsets[0], offsets[assistant_token]]
            return [offsets[0], offsets[assistant_token - 3], offsets[-1] - 1]

        features = _build_features_at_chars(task, tokenizer, starts)
        assert len(features[next(iter(steered))]) == 3
        with patch_load_sft_features(
            features, segment_mode=SegmentMode.turn_delimited_fixed_tokens
        ):
            batch = next(_make_generator(task, tokenizer, B=1, T=64, steered_positions=steered))
        _inputs, _targets, _fids, _mags, enc_mask, _seg_map, _fp, seg_doc_id = batch
        # Non-vacuity: the steered conversation really did lose a segment. Four
        # conversations pack into the row; three of them contribute two encoder
        # segments and the steered one contributes two of its three.
        assert int(enc_mask[0].any(dim=-1).sum()) == 8
        assert [int(value) for value in seg_doc_id[0].tolist()] == [-1, 0, 0, 1, 1, 2, 2, 3, 3]


# ---------------------------------------------------------------------------
# Non-steered output is untouched
# ---------------------------------------------------------------------------


class TestNonSteeredIdentity:
    @pytest.fixture
    def case(self, tmp_path: Path, tokenizer: _CharacterTokenizer):
        task = _steered_mixture(tmp_path, base_user=20, base_asst=24)

        def segments(position: int, ids: list[int], assistant_token: int) -> list[int]:
            if position in task.steered_positions:
                return [0, assistant_token - 3, assistant_token, assistant_token + 8]
            return [0, assistant_token]

        return task, _build_features(task, tokenizer, segments)

    def _batches(self, task, tokenizer, features, *, steered_positions, count: int = 3):
        with patch_load_sft_features(
            features, segment_mode=SegmentMode.turn_delimited_fixed_tokens
        ):
            generator = _make_generator(
                task, tokenizer, B=2, T=81, steered_positions=steered_positions
            )
            return [next(generator) for _ in range(count)]

    def test_an_empty_steered_set_is_byte_identical_to_no_steered_set(
        self, case, tokenizer: _CharacterTokenizer
    ) -> None:
        """The machinery is a strict no-op when no row is steered.

        ``frozenset()`` still builds ``ChatSpecialTokens``, still runs the
        membership test on every conversation and still takes the
        prediction-group branch in the row loop — it simply never splits.
        """
        task, features = case
        plain = self._batches(task, tokenizer, features, steered_positions=None)
        empty = self._batches(task, tokenizer, features, steered_positions=frozenset())
        for plain_batch, empty_batch in zip(plain, empty, strict=True):
            assert len(plain_batch) == len(empty_batch) == 8
            for plain_tensor, empty_tensor in zip(plain_batch, empty_batch, strict=True):
                assert t.equal(plain_tensor, empty_tensor)

    def test_a_real_steered_set_changes_seg_doc_id_and_nothing_else(
        self, case, tokenizer: _CharacterTokenizer
    ) -> None:
        """Non-vacuity for the test above, and the scope guard for the feature.

        Packing, tokens, targets, encoder inputs, ``seg_map`` and ``fp_targets``
        are all independent of steering: only the prediction-group ids move.
        """
        task, features = case
        plain = self._batches(task, tokenizer, features, steered_positions=None)
        steered = self._batches(
            task, tokenizer, features, steered_positions=frozenset(task.steered_positions)
        )
        differed = False
        for plain_batch, steered_batch in zip(plain, steered, strict=True):
            for index, (plain_tensor, steered_tensor) in enumerate(
                zip(plain_batch, steered_batch, strict=True)
            ):
                if index == 7:
                    differed = differed or not t.equal(plain_tensor, steered_tensor)
                else:
                    assert t.equal(plain_tensor, steered_tensor), f"tensor {index} moved"
        assert differed, "the steered set must change seg_doc_id somewhere"


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


class TestSteeredGuards:
    @pytest.mark.parametrize(
        ("messages", "roles"),
        [
            (
                [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                    {"role": "user", "content": "again"},
                    {"role": "assistant", "content": "second"},
                ],
                "('user', 'assistant', 'user', 'assistant')",
            ),
            (
                [
                    {"role": "user", "content": "question"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "before"},
                            {"type": "python", "text": "1 + 1"},
                            {"type": "python_output", "text": "2"},
                            {"type": "text", "text": "after"},
                        ],
                    },
                ],
                "('user', 'assistant', 'tool', 'assistant')",
            ),
        ],
        ids=["two-turn-pairs", "assistant-with-a-python-tool-part"],
    )
    def test_a_steered_conversation_that_is_not_one_user_then_one_assistant_raises(
        self,
        tokenizer: _CharacterTokenizer,
        messages: list[dict[str, Any]],
        roles: str,
    ) -> None:
        """The split has no meaning without exactly two logical turns.

        A steered JSONL cannot express these shapes, but ``steered_positions``
        is an index set over the LIVE task — a drifted task row must fail loud
        rather than silently splitting at the wrong turn.
        """

        class _Task(Task):
            def num_examples(self) -> int:
                return 2

            def get_example(self, index: int) -> dict:
                return {"messages": messages}

        task = _Task()
        features = _build_features(task, tokenizer, lambda position, ids, assistant: [0, assistant])
        with (
            patch_load_sft_features(features, segment_mode=SegmentMode.turn_delimited_fixed_tokens),
            pytest.raises(
                DataLoadError,
                match=(
                    "steered conversation original_index=[01] must render exactly one user "
                    "turn followed by one assistant turn; got roles " + re.escape(roles)
                ),
            ),
        ):
            next(_make_generator(task, tokenizer, B=1, T=200, steered_positions=frozenset({0, 1})))

    def test_steered_positions_against_a_fixed_mode_store_raises(
        self, tmp_path: Path, tokenizer: _CharacterTokenizer
    ) -> None:
        """Fixed-token segments do not respect turn boundaries at all.

        A segment can then straddle the user/assistant boundary, which makes the
        split unrepresentable — so the store's mode is checked as soon as it is
        loaded, before a single batch is built.
        """
        task = _steered_mixture(tmp_path, base_user=2, base_asst=3)
        features = _build_features(task, tokenizer, _four_segments)
        with (
            patch_load_sft_features(features, segment_mode=SegmentMode.fixed_tokens),
            pytest.raises(
                DataLoadError,
                match=(
                    "steered conversations require turn-delimited SFT features "
                    r"\(store segment_mode=fixed_tokens\)"
                ),
            ),
        ):
            next(
                _make_generator(
                    task,
                    tokenizer,
                    B=1,
                    T=32,
                    steered_positions=frozenset(task.steered_positions),
                )
            )

    def test_a_group_invariant_violation_names_the_example(
        self, tmp_path: Path, tokenizer: _CharacterTokenizer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The invariant is unreachable through real artifacts — pin the wrapper.

        Segments arrive strictly ascending and a validated steered render has
        exactly two turns, so ``steered_prediction_groups`` cannot legitimately
        raise here. If that ever changes, the generator must say WHICH example
        broke, and ``original_index`` is not otherwise in scope in the row
        builder — that recovery is what this pins.
        """
        task = _steered_mixture(tmp_path, base_user=2, base_asst=3)
        features = _build_features(task, tokenizer, _four_segments)

        def _boom(*_args: Any, **_kwargs: Any) -> list[int]:
            raise RuntimeError("segment starts (0, 1) map badly")

        monkeypatch.setattr(sft_script, "steered_prediction_groups", _boom)
        with (
            patch_load_sft_features(features, segment_mode=SegmentMode.turn_delimited_fixed_tokens),
            pytest.raises(
                RuntimeError,
                match=re.escape("conditioned SFT example 2: segment starts (0, 1) map badly"),
            ),
        ):
            next(
                _make_generator(
                    task,
                    tokenizer,
                    B=1,
                    T=32,
                    steered_positions=frozenset(task.steered_positions),
                )
            )

    def test_a_fixed_mode_store_is_accepted_without_steered_positions(
        self, tmp_path: Path, tokenizer: _CharacterTokenizer
    ) -> None:
        """Non-vacuity: the raise above is about steering, not about the fixture."""
        task = _steered_mixture(tmp_path, base_user=2, base_asst=3)
        features = _build_features(task, tokenizer, _four_segments)
        with patch_load_sft_features(features, segment_mode=SegmentMode.fixed_tokens):
            batch = next(_make_generator(task, tokenizer, B=1, T=32, steered_positions=None))
        assert [int(value) for value in batch[-1][0].tolist()] == [-1, 0, 0, 0, 0]


# ---------------------------------------------------------------------------
# Counters and the per-rank coverage log
# ---------------------------------------------------------------------------


class TestSteeredLogging:
    @pytest.mark.parametrize(
        ("rank", "min_loss_tokens", "nonzero_level", "expected_buffer", "expected_steered"),
        [
            (0, 1, None, True, True),
            (2, 1, None, False, False),
            (2, 40, None, True, False),
            (2, 1, "INFO", True, True),
        ],
    )
    def test_first_pass_messages_use_production_rank_aware_console_sink(
        self,
        tmp_path: Path,
        rank: int,
        min_loss_tokens: int,
        nonzero_level: str | None,
        expected_buffer: bool,
        expected_steered: bool,
    ) -> None:
        """A fresh process loads the real sink before any pytest log interception.

        No process group is needed: these summaries have no collectives, and
        the production console decides visibility from RANK at import time.
        """
        script = """
import sys
from pathlib import Path
from scripts.train.conditional_encdec_sft import SFTState
from steering.segmentation import SegmentMode
from tests.nanochat.test_structured_conversation_renderer import _CharacterTokenizer
from tests.steering.test_encdec_sft_conditioned_integration import patch_load_sft_features
from tests.steering.test_encdec_sft_steered_groups import (
    _build_features, _four_segments, _make_generator, _steered_mixture,
)

tokenizer = _CharacterTokenizer()
task = _steered_mixture(Path(sys.argv[1]), base_user=40, base_asst=40,
                        base_rows=9, steered_rows=3)
features = _build_features(task, tokenizer, _four_segments)
state = SFTState()
with patch_load_sft_features(features, segment_mode=SegmentMode.turn_delimited_fixed_tokens):
    generator = _make_generator(
        task, tokenizer, B=1, T=60, buffer_size=1, ddp_rank=int(sys.argv[2]),
        ddp_world_size=4, steered_positions=frozenset(task.steered_positions),
        sft_state=state, single_pass=True, min_loss_tokens=int(sys.argv[3]),
    )
    for _ in range(4):
        next(generator)
        if state.exhausted:
            break
assert state.exhausted
"""
        env: dict[str, str] = {
            **os.environ,
            "RANK": str(rank),
            "NANOCHAT_LOG_LEVEL": "INFO",
        }
        env.pop("NANOCHAT_LOG_LEVEL_NONZERO_RANK", None)
        if nonzero_level is not None:
            env["NANOCHAT_LOG_LEVEL_NONZERO_RANK"] = nonzero_level
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), str(rank), str(min_loss_tokens)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stderr.count("SFT buffer first pass") == int(expected_buffer)
        assert result.stderr.count("SFT steered rows first pass") == int(expected_steered)
        if rank and expected_buffer:
            assert f"rank {rank}" in result.stderr

    def test_the_first_pass_summary_counts_rows_without_an_assistant_segment(
        self, tmp_path: Path, tokenizer: _CharacterTokenizer, caplog
    ) -> None:
        """Truncation case (d): the steered response is cut away entirely.

        The summary fires on the render that completes the first pass, and
        packing trails rendering by at most ``buffer_size`` conversations — so
        the dataset here is deliberately much larger than the buffer, exactly
        as in production.
        """
        task = _steered_mixture(tmp_path, base_user=2, base_asst=3, base_rows=19)
        assert task.steered_positions == frozenset({10})
        features = _build_features(task, tokenizer, _four_segments)
        with patch_load_sft_features(
            features, segment_mode=SegmentMode.turn_delimited_fixed_tokens
        ):
            generator = _make_generator(
                task,
                tokenizer,
                B=1,
                T=11,
                buffer_size=2,
                steered_positions=frozenset(task.steered_positions),
            )
            for _ in range(len(task) + 5):
                next(generator)
        assert (
            "SFT steered rows first pass (20 conversations rendered on this rank): "
            "1 seen, 0 skipped by min_loss_tokens, "
            "1 packed with no assistant-turn segment"
        ) in caplog.text

    def test_the_first_pass_summary_counts_rows_dropped_by_min_loss_tokens(
        self, tmp_path: Path, tokenizer: _CharacterTokenizer, caplog
    ) -> None:
        """A steered row too short after truncation is counted, never silent."""
        task = _steered_mixture(tmp_path, base_user=40, base_asst=40)
        features = _build_features(task, tokenizer, _four_segments)
        with patch_load_sft_features(
            features, segment_mode=SegmentMode.turn_delimited_fixed_tokens
        ):
            next(
                _make_generator(
                    task,
                    tokenizer,
                    B=1,
                    T=60,
                    min_loss_tokens=40,
                    steered_positions=frozenset(task.steered_positions),
                )
            )
        assert "1 seen, 1 skipped by min_loss_tokens" in caplog.text

    def test_no_steered_summary_without_steered_positions(
        self, tmp_path: Path, tokenizer: _CharacterTokenizer, caplog
    ) -> None:
        """Non-vacuity: a plain run must not gain a steered log line."""
        task = _steered_mixture(tmp_path, base_user=2, base_asst=3)
        features = _build_features(task, tokenizer, _four_segments)
        with patch_load_sft_features(
            features, segment_mode=SegmentMode.turn_delimited_fixed_tokens
        ):
            next(_make_generator(task, tokenizer, B=1, T=32, steered_positions=None))
        assert "SFT steered rows first pass" not in caplog.text
        assert "Steered SFT coverage" not in caplog.text

    @pytest.mark.parametrize(
        ("ddp_world_size", "ddp_rank", "expected"),
        # 12 documents with the 3 steered rows at positions {2, 6, 10}:
        #   world 2: rank 0 strides [0, 2, 4, 6, 8, 10] -> ALL three steered
        #            rank 1 strides [1, 3, 5, 7, 9, 11] -> none
        #   world 4: rank 2 strides [2, 6, 10]          -> ALL three steered
        #            every other rank                   -> none
        # Even DATASET-level spacing does not survive DDP striding, and the
        # unevenness gets WORSE as the world grows. That is accepted; the log
        # is how it stays visible, so it is pinned as informative (the counts
        # differ per rank and per world size), not merely as present.
        [
            (2, 0, "rank 0/2 serves 3/6 steered documents"),
            (2, 1, "rank 1/2 serves 0/6 steered documents"),
            (4, 0, "rank 0/4 serves 0/3 steered documents"),
            (4, 1, "rank 1/4 serves 0/3 steered documents"),
            (4, 2, "rank 2/4 serves 3/3 steered documents"),
            (4, 3, "rank 3/4 serves 0/3 steered documents"),
        ],
    )
    def test_each_rank_logs_the_steered_share_of_its_own_stride(
        self,
        tmp_path: Path,
        tokenizer: _CharacterTokenizer,
        caplog,
        ddp_world_size: int,
        ddp_rank: int,
        expected: str,
    ) -> None:
        task = _steered_mixture(tmp_path, base_user=2, base_asst=3, base_rows=9, steered_rows=3)
        assert task.steered_positions == frozenset({2, 6, 10})
        features = _build_features(task, tokenizer, _four_segments)
        with patch_load_sft_features(
            features, segment_mode=SegmentMode.turn_delimited_fixed_tokens
        ):
            next(
                _make_generator(
                    task,
                    tokenizer,
                    B=1,
                    T=32,
                    ddp_rank=ddp_rank,
                    ddp_world_size=ddp_world_size,
                    steered_positions=frozenset(task.steered_positions),
                )
            )
        assert f"Steered SFT coverage: {expected}" in caplog.text

    def test_the_first_pass_line_names_the_render_count_that_triggered_it(
        self, tmp_path: Path, tokenizer: _CharacterTokenizer, caplog
    ) -> None:
        """The first-pass count agrees with the rank's startup coverage share."""
        task = _steered_mixture(tmp_path, base_user=2, base_asst=3)
        features = _build_features(task, tokenizer, _four_segments)
        with patch_load_sft_features(
            features, segment_mode=SegmentMode.turn_delimited_fixed_tokens
        ):
            next(
                _make_generator(
                    task,
                    tokenizer,
                    B=1,
                    T=32,
                    ddp_rank=0,
                    ddp_world_size=2,
                    steered_positions=frozenset(task.steered_positions),
                )
            )
        assert "Steered SFT coverage: rank 0/2 serves 1/2 steered documents" in caplog.text
        assert (
            "SFT steered rows first pass (2 conversations rendered on this rank): 1 seen"
        ) in caplog.text

    @pytest.mark.parametrize("ddp_rank", range(4))
    @pytest.mark.parametrize("base_rows", [9, 10])
    @pytest.mark.parametrize("min_loss_tokens", [1, 40])
    @pytest.mark.parametrize("steered_logging", [False, True])
    def test_four_rank_first_pass_summaries_fire_before_local_exhaustion(
        self, tmp_path, tokenizer, caplog, ddp_rank, base_rows, min_loss_tokens, steered_logging
    ):
        task = _steered_mixture(
            tmp_path, base_user=40, base_asst=40, base_rows=base_rows, steered_rows=3
        )
        features = _build_features(task, tokenizer, _four_segments)
        state = SFTState()
        with patch_load_sft_features(
            features, segment_mode=SegmentMode.turn_delimited_fixed_tokens
        ):
            generator = _make_generator(
                task,
                tokenizer,
                B=1,
                T=60,
                buffer_size=1,
                ddp_rank=ddp_rank,
                ddp_world_size=4,
                steered_positions=frozenset(task.steered_positions) if steered_logging else None,
                sft_state=state,
                single_pass=True,
                min_loss_tokens=min_loss_tokens,
            )
            for _ in range(len(task) + 1):
                next(generator)
                if state.exhausted:
                    break
        assert state.exhausted
        rank_indices = list(range(ddp_rank, len(task), 4))
        seen = len(set(rank_indices) & task.steered_positions)
        skipped = seen if min_loss_tokens == 40 else 0
        summaries = [record.message for record in caplog.records if "first pass" in record.message]
        assert len(summaries) == (2 if steered_logging else 1)
        assert summaries[0].startswith(
            f"SFT buffer first pass: skipped {skipped}/{len(rank_indices)} conversations"
        )
        if steered_logging:
            assert summaries[1].startswith(
                f"SFT steered rows first pass ({len(rank_indices)} conversations rendered on this rank): "
                f"{seen} seen, {skipped} skipped by min_loss_tokens, "
            )

    def test_ddp_first_pass_summary_emits_once_even_after_multiple_wraps(
        self, tmp_path, tokenizer, caplog
    ):
        task = _steered_mixture(tmp_path, base_user=2, base_asst=3, base_rows=9, steered_rows=3)
        features = _build_features(task, tokenizer, _four_segments)
        with patch_load_sft_features(
            features, segment_mode=SegmentMode.turn_delimited_fixed_tokens
        ):
            generator = _make_generator(
                task,
                tokenizer,
                B=1,
                T=32,
                buffer_size=1,
                ddp_rank=2,
                ddp_world_size=4,
                steered_positions=frozenset(task.steered_positions),
            )
            for _ in range(12):
                next(generator)
        summaries = [record.message for record in caplog.records if "first pass" in record.message]
        assert len(summaries) == 2
        assert "(3 conversations rendered on this rank): 3 seen" in summaries[1]

    @pytest.mark.parametrize("ddp_rank", range(4))
    @pytest.mark.parametrize("buffer_size", [1, 8])
    def test_sparse_conversion_summary_observes_exact_render_boundary(
        self, tmp_path, tokenizer, caplog, monkeypatch, ddp_rank, buffer_size
    ):
        """Observe real renders at emission, including a final skipped row.

        The task size, attempted conversion prefix, surviving artifact count,
        and rank share deliberately differ. An oversized buffer also ensures
        the summary cannot accidentally depend on packing or buffer refill.
        """
        task = _steered_mixture(tmp_path, base_user=40, base_asst=40, base_rows=9, steered_rows=3)
        covered = [1, 2, 5, 6, 8, 9, 10]
        all_features = _build_features(task, tokenizer, _four_segments)
        features = {index: all_features[index] for index in covered}
        expected_indices = covered[ddp_rank::4]
        rendered = []
        original_get_example = task.get_example

        def record_example(index):
            rendered.append(index)
            return original_get_example(index)

        monkeypatch.setattr(task, "get_example", record_example)
        observations = []

        def observe_summary(message):
            if "first pass" in message.record["message"]:
                observations.append((message.record["message"], list(rendered)))

        sink = sft_script.logger.add(observe_summary)
        state = SFTState()
        try:
            with patch_load_sft_features(
                features,
                conversion_total_conversations=11,
                segment_mode=SegmentMode.turn_delimited_fixed_tokens,
            ):
                generator = _make_generator(
                    task,
                    tokenizer,
                    B=1,
                    T=60,
                    buffer_size=buffer_size,
                    ddp_rank=ddp_rank,
                    ddp_world_size=4,
                    steered_positions=frozenset(task.steered_positions),
                    sft_state=state,
                    single_pass=True,
                    min_loss_tokens=40,
                )
                for _ in range(len(expected_indices) + 1):
                    next(generator)
                    if state.exhausted:
                        break
                assert state.exhausted
                assert rendered == expected_indices
                # Exhausted ranks keep producing padding in distributed eval.
                for _ in range(2):
                    assert (next(generator)[1] == -1).all()
        finally:
            sft_script.logger.remove(sink)

        skipped = len(set(expected_indices) & task.steered_positions)
        assert len(observations) == 2
        assert all(indices == expected_indices for _, indices in observations)
        assert observations[0][0].startswith(
            f"SFT buffer first pass: skipped {skipped}/{len(expected_indices)} conversations"
        )
        assert observations[1][0].endswith(
            f"{skipped} seen, {skipped} skipped by min_loss_tokens, "
            "0 packed with no assistant-turn segment"
        )
        summary_record = next(
            record
            for record in caplog.records
            if record.message.startswith("SFT buffer first pass")
        )
        assert summary_record.levelname == ("WARNING" if skipped else "INFO")

    @pytest.mark.parametrize("ddp_rank", [0, 7])
    def test_single_pass_with_empty_rank_shares_rejects_before_summary(
        self, tmp_path, tokenizer, caplog, ddp_rank
    ):
        task = _steered_mixture(tmp_path, base_user=2, base_asst=3)
        features = _build_features(task, tokenizer, _four_segments)
        with patch_load_sft_features(
            features, segment_mode=SegmentMode.turn_delimited_fixed_tokens
        ):
            generator = _make_generator(
                task,
                tokenizer,
                B=1,
                T=32,
                ddp_rank=ddp_rank,
                ddp_world_size=8,
                steered_positions=frozenset(task.steered_positions),
                single_pass=True,
            )
            with pytest.raises(DataLoadError, match="every rank needs at least one"):
                next(generator)
        assert "first pass" not in caplog.text

    @pytest.mark.parametrize("batches_before_resume", [1, 8])
    def test_resume_restarts_render_window_including_ordered_buffer_replay(
        self, tmp_path, tokenizer, caplog, monkeypatch, batches_before_resume
    ):
        """A restarted summary measures renders, without changing resumed batches."""
        task = _steered_mixture(tmp_path, base_user=2, base_asst=3, base_rows=9, steered_rows=3)
        features = _build_features(task, tokenizer, _four_segments)
        state = SFTState()
        with patch_load_sft_features(
            features, segment_mode=SegmentMode.turn_delimited_fixed_tokens
        ):
            continuous = _make_generator(
                task,
                tokenizer,
                B=1,
                T=32,
                buffer_size=4,
                ddp_rank=0,
                ddp_world_size=2,
                steered_positions=frozenset(task.steered_positions),
                sft_state=state,
            )
            for _ in range(batches_before_resume):
                next(continuous)
            assert state.dataloader_state is not None
            snapshot = copy.deepcopy(state.dataloader_state)
            assert snapshot["buffered_cursor_positions"]
            assert ("first pass" in caplog.text) == (batches_before_resume == 8)
            expected_batches = [next(continuous) for _ in range(8)]
            caplog.clear()

            rendered = []
            original_get_example = task.get_example

            def record_example(index):
                rendered.append(index)
                return original_get_example(index)

            monkeypatch.setattr(task, "get_example", record_example)
            resumed_state = SFTState(consumed=snapshot["consumed"], dataloader_state=snapshot)
            resumed = _make_generator(
                task,
                tokenizer,
                B=1,
                T=32,
                buffer_size=4,
                ddp_rank=0,
                ddp_world_size=2,
                steered_positions=frozenset(task.steered_positions),
                sft_state=resumed_state,
            )
            for expected in expected_batches:
                actual = next(resumed)
                for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
                    assert t.equal(actual_tensor, expected_tensor)

        replay = [position % len(task) for position in snapshot["buffered_cursor_positions"]]
        assert rendered[: len(replay)] == replay
        first_window = rendered[:6]
        seen = sum(index in task.steered_positions for index in first_window)
        summaries = [record.message for record in caplog.records if "first pass" in record.message]
        assert len(summaries) == 2
        assert summaries[0].startswith("SFT buffer first pass: skipped 0/6 conversations")
        assert summaries[1].startswith(
            f"SFT steered rows first pass (6 conversations rendered on this rank): {seen} seen, "
        )


class TestSteeredResume:
    """A resumed generator must re-derive the split for every replayed row."""

    def test_a_resumed_steered_generator_replays_the_split(
        self, tmp_path: Path, tokenizer: _CharacterTokenizer
    ) -> None:
        """``turn_src_starts`` is recomputed on replay, not restored.

        The checkpoint records only ``buffered_cursor_positions``; resume walks
        them back through ``refill_buffer``, which re-renders each conversation
        and rebuilds its turn starts. Nothing else pins that: an optimisation
        that rematerialised buffered entries without re-rendering would silently
        drop the split for every replayed steered conversation and no golden
        would move, because the goldens are non-steered.
        """
        task = _steered_mixture(tmp_path, base_user=2, base_asst=3, base_rows=7)
        steered = frozenset(task.steered_positions)
        features = _build_features(task, tokenizer, _four_segments)
        kwargs: dict[str, Any] = {"B": 1, "T": 32, "steered_positions": steered}
        with patch_load_sft_features(
            features, segment_mode=SegmentMode.turn_delimited_fixed_tokens
        ):
            state = SFTState()
            generator = _make_generator(task, tokenizer, sft_state=state, **kwargs)
            next(generator)
            assert state.dataloader_state is not None
            replayed = {
                position % len(task)
                for position in state.dataloader_state["buffered_cursor_positions"]
            }

            def _from_snapshot(**overrides: Any):
                resume_state = SFTState(
                    consumed=state.consumed,
                    dataloader_state=copy.deepcopy(state.dataloader_state),
                )
                return _make_generator(
                    task, tokenizer, sft_state=resume_state, **{**kwargs, **overrides}
                )

            resumed = _from_snapshot()
            unsteered = _from_snapshot(steered_positions=None)
            reference = _make_generator(task, tokenizer, sft_state=SFTState(), **kwargs)
            next(reference)

            split_seen = False
            for _ in range(2):
                actual, expected, plain = next(resumed), next(reference), next(unsteered)
                for actual_value, expected_value in zip(actual, expected, strict=True):
                    assert t.equal(actual_value, expected_value)
                split_seen = split_seen or not t.equal(actual[-1], plain[-1])

        # Non-vacuity: the steered row is among the REPLAYED positions, and the
        # replayed window really does carry a split (the same window without
        # ``steered_positions`` produces different group ids).
        assert steered <= replayed, f"steered row not replayed; buffer held {sorted(replayed)}"
        assert split_seen, "the replayed batches must show the steered split"


class TestTrainerWiringHelpers:
    """The two ``main()``-level pieces, as the functions ``main()`` calls.

    ``main()`` itself needs a full checkpoint + artifact stack to reach these
    lines, so the behaviour lives in named helpers that can be driven directly.
    """

    def test_a_steered_mixture_yields_its_positions_and_logs_the_share(
        self, tmp_path: Path, caplog
    ) -> None:
        task = _steered_mixture(tmp_path, base_user=2, base_asst=3)

        positions = sft_script._steered_positions_for_task(task)

        assert positions == task.steered_positions
        assert "Steered mixture: 1 steered rows in 4 (25.00%)" in caplog.text

    def test_a_plain_mixture_yields_none_and_logs_nothing(self, caplog) -> None:
        assert sft_script._steered_positions_for_task(_PlainTask(3, 2, 3)) is None
        assert "Steered mixture" not in caplog.text

    def test_a_steered_mixture_on_fixed_mode_features_is_rejected(self) -> None:
        with pytest.raises(
            DataLoadError,
            match=(
                "steered mixture 'standard_steered' requires turn-delimited SFT features"
                ".*segment_mode='fixed_tokens'"
            ),
        ):
            sft_script._assert_steered_segment_mode(
                frozenset({1}), SegmentMode.fixed_tokens, "standard_steered"
            )

    @pytest.mark.parametrize(
        ("positions", "segment_mode"),
        [
            (None, SegmentMode.fixed_tokens),
            (frozenset({1}), SegmentMode.turn_delimited_fixed_tokens),
            (None, SegmentMode.turn_delimited_fixed_tokens),
        ],
    )
    def test_every_other_combination_is_accepted(
        self, positions: frozenset[int] | None, segment_mode: SegmentMode
    ) -> None:
        """Non-vacuity: only steered-on-fixed raises, not the guard's mere presence."""
        assert sft_script._assert_steered_segment_mode(positions, segment_mode, "standard") is None

    def test_main_wires_both_helpers_at_their_own_call_sites(self) -> None:
        """No test executes ``main()`` this deep, so pin the wiring by source.

        Same backstop as ``test_main_threads_resuming_flag_to_segment_mode_guard``
        in ``tests/nanochat/test_encdec_sft_resume.py``: the helpers' behaviour
        is covered above, this catches a refactor that stops calling them.
        """
        source = inspect.getsource(sft_script.main)
        assert "steered_positions = _steered_positions_for_task(task)" in source
        assert (
            "_assert_steered_segment_mode(steered_positions, sft_segment_mode, str(cfg.dataset))"
            in source
        )
        assert "steered_positions=steered_positions," in source
        # ORDER, not just presence: the guard exists to fail BEFORE the feature
        # store is walked and the generator built. Moving its call below the
        # generator construction keeps every substring above present while
        # turning a startup rejection into a first-batch one.
        assert source.index("steered_positions = _steered_positions_for_task(task)") < source.index(
            "_assert_steered_segment_mode("
        )
        assert source.index("_assert_steered_segment_mode(") < source.index(
            "steered_positions=steered_positions,"
        )


# ---------------------------------------------------------------------------
# Scope guard
# ---------------------------------------------------------------------------


def test_the_sft_generator_never_writes_packed_row_shards() -> None:
    """``seg_doc_id`` is uint8 in packed pretraining shards, int64 here.

    A steered run's group ids stay small, but the reason the widened id space is
    safe is that this producer never reaches the uint8 storage path at all.
    Scoped to real identifiers (imports, names, attributes) so that neither a
    comment naming the symbol nor a docstring can trip it.
    """
    module = ast.parse(Path(sft_script.__file__).read_text(encoding="utf-8"))
    referenced = {
        name
        for node in ast.walk(module)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
        for name in (alias.name.split(".")[-1], alias.asname)
        if name is not None
    } | {
        node.id if isinstance(node, ast.Name) else node.attr
        for node in ast.walk(module)
        if isinstance(node, ast.Name | ast.Attribute)
    }
    assert "save_packed_row_shard" not in referenced
    assert "PackedRowShard" not in referenced
