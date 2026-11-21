"""Independent contracts for structured rows, prefix clipping, and segmentation."""

from __future__ import annotations

import pyarrow as pa
import pytest

from nanochat.tokenizer import LogicalTurnSpan
from steering.sft import structured_rollout
from steering.sft.structured_rollout import (
    STRUCTURED_ROLLOUT_ARROW_SCHEMA,
    EnvironmentFilterReason,
    RewardMaskMode,
    SegmentationMode,
    StructuredRolloutError,
    StructuredRolloutRow,
    assert_rollout_artifact_format,
    build_conversation_environment,
    clip_structured_row,
    normalize_complete_prefix,
    stateless_seed,
    structured_row_digest,
    structured_rows_to_table,
    structured_table_to_rows,
)


def _decode(token_ids) -> str:
    return ",".join(str(token_id) for token_id in token_ids)


def _row(
    *,
    target_ids: tuple[int, ...] = (10, 11, 12, 13),
    target_mask: tuple[bool, ...] = (False, False, True, True),
    turns: tuple[LogicalTurnSpan, ...] = (
        LogicalTurnSpan("user", 0, 2),
        LogicalTurnSpan("assistant", 2, 4),
    ),
) -> StructuredRolloutRow:
    return StructuredRolloutRow(
        target_token_ids=target_ids,
        target_mask=target_mask,
        logical_turns=turns,
        text=_decode(target_ids),
        original_index=7,
    )


def test_structured_row_role_domain_is_exactly_user_assistant_tool() -> None:
    assert structured_rollout._ALLOWED_ROLES == frozenset(
        {"user", "assistant", "tool"}
    )


def test_validate_structured_row_rejects_unknown_role() -> None:
    with pytest.raises(StructuredRolloutError, match="role must be one of"):
        structured_row_digest(
            _row(
                target_ids=(10, 11),
                target_mask=(True, True),
                turns=(LogicalTurnSpan("system", 0, 2),),
            )
        )


def test_complete_prefix_retains_completed_transactions_and_rolls_back_trailing_user() -> None:
    conversation = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a0"},
            {"role": "user", "content": "u1"},
        ]
    }
    result = normalize_complete_prefix(conversation)
    assert result.skip_reason is None
    assert result.rolled_back_messages == 1
    assert result.conversation == {"messages": conversation["messages"][:3]}
    assert result.conversation is not conversation


def test_complete_prefix_ignores_inert_task_provenance() -> None:
    messages = [
        {"role": "user", "content": "spell"},
        {"role": "assistant", "content": "word"},
    ]
    result = normalize_complete_prefix(
        {"messages": messages, "letters": ["w"], "subject": "spelling"}
    )
    assert result.conversation == {"messages": messages}

    with pytest.raises(StructuredRolloutError, match="alternate top-level"):
        normalize_complete_prefix({"messages": messages, "tools": []})


@pytest.mark.parametrize(
    "tail",
    [
        [
            {"role": "user", "content": "u1"},
            {
                "role": "assistant",
                "content": [{"type": "tool_call", "name": "x", "arguments": {}}],
            },
        ],
        [
            {"role": "user", "content": "u1"},
            {
                "role": "assistant",
                "content": [{"type": "tool_call", "name": "x", "arguments": {}}],
            },
            {"role": "tool", "name": "x", "content": "r"},
        ],
        [
            {"role": "user", "content": "u1"},
            {
                "role": "assistant",
                "content": [
                    {"type": "python", "text": "1+1"},
                    {"type": "python_output", "text": "2"},
                ],
            },
        ],
    ],
)
def test_complete_prefix_rolls_back_each_trailing_incomplete_tool_transaction(
    tail: list[dict],
) -> None:
    prefix = [
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "a0"},
    ]
    result = normalize_complete_prefix({"messages": [*prefix, *tail]})
    assert result.conversation == {"messages": prefix}
    assert result.rolled_back_messages == len(tail)


def test_no_completed_transaction_has_named_discard_and_middle_malformation_fails() -> None:
    discarded = normalize_complete_prefix(
        {"messages": [{"role": "user", "content": "unfinished"}]}
    )
    assert discarded.conversation is None
    assert discarded.skip_reason == "no_completed_transaction"

    with pytest.raises(StructuredRolloutError, match="middle"):
        normalize_complete_prefix(
            {
                "messages": [
                    {"role": "user", "content": "u"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "python", "text": "1"},
                            {"type": "python_output", "text": "1"},
                        ],
                    },
                    {"role": "user", "content": "not repairable"},
                ]
            }
        )


def test_arrow_schema_is_exact_for_normal_and_empty_tables() -> None:
    row = _row()
    table = structured_rows_to_table([row])
    empty = structured_rows_to_table([])
    assert table.schema == STRUCTURED_ROLLOUT_ARROW_SCHEMA
    assert empty.schema == STRUCTURED_ROLLOUT_ARROW_SCHEMA
    assert empty.num_rows == 0
    assert structured_table_to_rows(table, decode=_decode) == (row,)
    assert structured_table_to_rows(empty) == ()


def test_schema_and_row_corruption_fail_closed() -> None:
    with pytest.raises(StructuredRolloutError, match=r"len\(target_token_ids\)"):
        structured_row_digest(
            StructuredRolloutRow(
                target_token_ids=(1, 2),
                target_mask=(True,),
                logical_turns=(LogicalTurnSpan("assistant", 0, 2),),
                text="1,2",
                original_index=0,
            )
        )
    legacy = pa.table(
        {
            "target_token_ids": [[1]],
            "target_mask": [[True]],
            "logical_turns": [[{"role": "assistant", "start": 0, "end": 1}]],
            "text": ["1"],
            "original_index": [0],
            "assistant_loss_mask": [[1]],
        }
    )
    with pytest.raises(StructuredRolloutError, match="removed legacy"):
        structured_table_to_rows(legacy)

    with pytest.raises(StructuredRolloutError, match="start/end"):
        structured_row_digest(
            StructuredRolloutRow(
                target_token_ids=(1,),
                target_mask=(True,),
                logical_turns=(LogicalTurnSpan("assistant", True, 1),),
                text="1",
                original_index=0,
            )
        )
    with pytest.raises(StructuredRolloutError, match="original_index"):
        structured_row_digest(
            StructuredRolloutRow(
                target_token_ids=(1,),
                target_mask=(True,),
                logical_turns=(LogicalTurnSpan("assistant", 0, 1),),
                text="1",
                original_index=True,
            )
        )


def test_context_clip_cuts_spans_mechanically_without_turn_repair() -> None:
    row = _row(
        target_ids=(1, 2, 3, 4, 5),
        target_mask=(False, False, True, True, True),
        turns=(
            LogicalTurnSpan("user", 0, 2),
            LogicalTurnSpan("assistant", 2, 5),
        ),
    )
    clipped = clip_structured_row(row, context_length=4, decode=_decode)
    assert clipped.target_token_ids == (1, 2, 3, 4)
    assert clipped.target_mask == (False, False, True, True)
    assert clipped.logical_turns[-1] == LogicalTurnSpan("assistant", 2, 4)


def test_input_aligned_segmentation_has_the_intentional_one_token_boundary() -> None:
    result = build_conversation_environment(
        _row(),
        bos_token_id=1,
        context_length=4,
        segment_length=2,
        min_reward_tokens=1,
    )
    assert result.filter_reason is None
    assert result.environment is not None
    environment = result.environment
    assert environment.inputs == (1, 10, 11, 12)
    assert environment.targets == (10, 11, 12, 13)
    assert environment.seg_map == (0, 0, 0, 1)
    assert [(segment.start, segment.end) for segment in environment.segments] == [
        (0, 2),
        (2, 4),
    ]
    assert environment.reward_mask == (False, False, True, True)


def test_turn_delimited_513_token_turn_is_independently_chunked() -> None:
    target_ids = tuple(range(521))
    row = _row(
        target_ids=target_ids,
        target_mask=(False,) * 513 + (True,) * 8,
        turns=(
            LogicalTurnSpan("user", 0, 513),
            LogicalTurnSpan("assistant", 513, 521),
        ),
    )
    result = build_conversation_environment(
        row,
        bos_token_id=999,
        context_length=521,
        segment_length=256,
        segmentation_mode=SegmentationMode.TURN_DELIMITED_FIXED_LENGTH,
    )
    assert result.environment is not None
    assert [
        segment.end - segment.start for segment in result.environment.segments
    ] == [256, 256, 1, 8]
    assert [segment.role for segment in result.environment.segments] == [
        "user",
        "user",
        "user",
        "assistant",
    ]


def test_reward_mask_is_direct_and_full_sequence_is_explicit() -> None:
    assistant = build_conversation_environment(
        _row(),
        bos_token_id=1,
        context_length=4,
        segment_length=2,
        min_reward_tokens=1,
    ).environment
    full = build_conversation_environment(
        _row(),
        bos_token_id=1,
        context_length=4,
        segment_length=2,
        reward_mask_mode=RewardMaskMode.FULL_SEQUENCE,
        min_reward_tokens=1,
    ).environment
    assert assistant is not None and full is not None
    assert assistant.reward_mask == _row().target_mask
    assert full.reward_mask == (True, True, True, True)


def test_post_truncation_guards_are_named() -> None:
    one_turn = build_conversation_environment(
        _row(),
        bos_token_id=1,
        context_length=2,
        segment_length=1,
        min_reward_tokens=1,
    )
    assert one_turn.filter_reason is EnvironmentFilterReason.AT_MOST_ONE_TURN

    one_segment = build_conversation_environment(
        _row(),
        bos_token_id=1,
        context_length=4,
        segment_length=4,
        min_reward_tokens=1,
    )
    assert one_segment.filter_reason is EnvironmentFilterReason.AT_MOST_ONE_SEGMENT

    insufficient = build_conversation_environment(
        _row(),
        bos_token_id=1,
        context_length=4,
        segment_length=2,
        min_reward_tokens=3,
    )
    assert (
        insufficient.filter_reason
        is EnvironmentFilterReason.INSUFFICIENT_REWARD_TARGETS
    )


def test_format_guard_and_stateless_seed_are_strict_and_order_invariant() -> None:
    assert_rollout_artifact_format(
        {"format": "structured_nanochat_rollout_v1"},
        structured=True,
    )
    with pytest.raises(StructuredRolloutError, match="separate artifact namespace"):
        assert_rollout_artifact_format(
            {"format": "full_nanochat_text"},
            structured=True,
        )
    assert stateless_seed({"b": 2, "a": 1}) == stateless_seed({"a": 1, "b": 2})
    assert 0 <= stateless_seed({"namespace": "train", "step": 3}) < 1 << 63
