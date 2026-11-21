"""Authoritative structured-SFT rows and conversation environments.

The stored RustBPE token IDs are load-bearing. Decoded ``text`` exists only
for cross-tokenizer character alignment and is never re-tokenized to recover
the decoder stream.
"""

from __future__ import annotations

import copy
import hashlib
import json
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

import pyarrow as pa

from nanochat.tokenizer import LogicalTurnSpan, StructuredConversationRender
from steering.refined_types import PositiveInt
from steering.segmentation_primitives import _segment_fixed, segment_turn_delimited
from steering.sft.turn_spans import TurnRole

STRUCTURED_ROLLOUT_FORMAT = "structured_nanochat_rollout_v1"
STRUCTURED_ROLLOUT_SCHEMA_VERSION = 1
STRUCTURED_RENDERER_VERSION = 1
STRUCTURED_ROW_FIELDS = (
    "target_token_ids",
    "target_mask",
    "logical_turns",
    "text",
    "original_index",
)
STRUCTURED_ROLLOUT_ARROW_SCHEMA = pa.schema(
    [
        pa.field("target_token_ids", pa.list_(pa.int32()), nullable=False),
        pa.field("target_mask", pa.list_(pa.bool_()), nullable=False),
        pa.field(
            "logical_turns",
            pa.list_(
                pa.struct(
                    [
                        pa.field("role", pa.string(), nullable=False),
                        pa.field("start", pa.int32(), nullable=False),
                        pa.field("end", pa.int32(), nullable=False),
                    ]
                )
            ),
            nullable=False,
        ),
        pa.field("text", pa.string(), nullable=False),
        pa.field("original_index", pa.int64(), nullable=False),
    ]
)

_ALLOWED_ROLES = frozenset(role.value for role in TurnRole)
_INT32_MAX = (1 << 31) - 1


class StructuredRolloutError(ValueError):
    """A structured source, row, or environment violates its contract."""


class SegmentationMode(StrEnum):
    """Supported target-stream segmentation policies."""

    FIXED_LENGTH = "fixed_length"
    TURN_DELIMITED_FIXED_LENGTH = "turn_delimited_fixed_length"


class RewardMaskMode(StrEnum):
    """Targets included in detached teacher-forced reward."""

    ASSISTANT_ONLY = "assistant_only"
    FULL_SEQUENCE = "full_sequence"


class EnvironmentFilterReason(StrEnum):
    """Named eligibility failures used by deterministic filtering/refill."""

    NO_TARGETS = "no_targets_after_truncation"
    AT_MOST_ONE_TURN = "at_most_one_surviving_logical_turn"
    AT_MOST_ONE_SEGMENT = "at_most_one_resulting_segment"
    INSUFFICIENT_REWARD_TARGETS = "insufficient_valid_reward_targets"


@dataclass(frozen=True)
class PrefixNormalizationResult:
    """Deep-copied complete message prefix or one named discard reason."""

    conversation: dict[str, list[dict[str, Any]]] | None
    skip_reason: str | None
    rolled_back_messages: int


@dataclass(frozen=True)
class StructuredRolloutRow:
    """One validated row of ``structured_nanochat_rollout_v1``."""

    target_token_ids: tuple[int, ...]
    target_mask: tuple[bool, ...]
    logical_turns: tuple[LogicalTurnSpan, ...]
    text: str
    original_index: int


@dataclass(frozen=True)
class TargetSegment:
    """One half-open target-token segment."""

    index: int
    start: int
    end: int
    role: str | None


@dataclass(frozen=True)
class ConversationEnvironment:
    """Pure, truncated single-conversation rollout environment."""

    original_index: int
    inputs: tuple[int, ...]
    targets: tuple[int, ...]
    reward_mask: tuple[bool, ...]
    logical_turns: tuple[LogicalTurnSpan, ...]
    segments: tuple[TargetSegment, ...]
    seg_map: tuple[int, ...]


@dataclass(frozen=True)
class EnvironmentBuildResult:
    """Eligible environment or one explicit filtering reason."""

    environment: ConversationEnvironment | None
    filter_reason: EnvironmentFilterReason | None


def _require_exact_fields(
    value: object,
    expected: set[str],
    *,
    surface: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StructuredRolloutError(
            f"{surface} must be a mapping, got {type(value).__name__}"
        )
    if not all(isinstance(key, str) for key in value):
        raise StructuredRolloutError(f"{surface} keys must all be strings")
    typed_value = cast(Mapping[str, Any], value)
    fields = set(typed_value)
    if fields != expected:
        raise StructuredRolloutError(
            f"{surface} fields mismatch: missing={sorted(expected - fields)}, "
            f"extra={sorted(fields - expected)}"
        )
    return typed_value


def _require_structured_conversation(value: object) -> Mapping[str, Any]:
    """Accept inert task provenance but reject alternate semantic inputs."""
    if not isinstance(value, Mapping):
        raise StructuredRolloutError(
            f"structured conversation must be a mapping, got {type(value).__name__}"
        )
    if not all(isinstance(key, str) for key in value):
        raise StructuredRolloutError(
            "structured conversation keys must all be strings"
        )
    if "messages" not in value:
        raise StructuredRolloutError(
            "structured conversation requires the top-level 'messages' field"
        )
    typed_value = cast(Mapping[str, Any], value)
    forbidden = {
        "assistant_loss_mask",
        "full_nanochat_text",
        "pre_rendered_text",
        "prompt",
        "python_tools",
        "rendered_text",
        "response",
        "text",
        "tool_choice",
        "tools",
    }
    present = sorted(forbidden.intersection(typed_value))
    if present:
        raise StructuredRolloutError(
            "structured conversation rejects alternate top-level semantic fields; "
            f"got {present}"
        )
    return typed_value


def _assistant_state(content: object, *, surface: str) -> tuple[str, str | None]:
    """Return ``(final|tool_call|incomplete, outstanding_name)``."""
    if isinstance(content, str):
        return "final", None
    if not isinstance(content, (list, tuple)) or not content:
        raise StructuredRolloutError(
            f"{surface} content must be a string or non-empty multipart sequence"
        )
    native_open = False
    needs_continuation = False
    for index, raw_part in enumerate(content):
        if not isinstance(raw_part, Mapping) or not isinstance(raw_part.get("type"), str):
            raise StructuredRolloutError(
                f"{surface} part {index} must be a mapping with a string type"
            )
        typed_part = cast(Mapping[str, Any], raw_part)
        part_type = typed_part["type"]
        if part_type in {"text", "python", "python_output"}:
            part = _require_exact_fields(
                typed_part,
                {"type", "text"},
                surface=f"{surface} {part_type} part {index}",
            )
            if not isinstance(part["text"], str):
                raise StructuredRolloutError(
                    f"{surface} {part_type} part {index} text must be a string"
                )
            if part_type == "text":
                if native_open:
                    raise StructuredRolloutError(
                        f"{surface} native python output must immediately follow its call"
                    )
                needs_continuation = False
            elif part_type == "python":
                if native_open:
                    raise StructuredRolloutError(
                        f"{surface} contains parallel native python calls"
                    )
                native_open = True
            else:
                if not native_open:
                    raise StructuredRolloutError(
                        f"{surface} contains an orphan native python_output"
                    )
                native_open = False
                needs_continuation = True
            continue
        if part_type != "tool_call":
            raise StructuredRolloutError(
                f"{surface} part {index} has unsupported type {part_type!r}"
            )
        part = _require_exact_fields(
            raw_part,
            {"type", "name", "arguments"},
            surface=f"{surface} tool_call part {index}",
        )
        if native_open:
            raise StructuredRolloutError(
                f"{surface} generic call overlaps a native python call"
            )
        if index != len(content) - 1:
            raise StructuredRolloutError(
                f"{surface} generic tool call must end its assistant portion"
            )
        name = part["name"]
        if not isinstance(name, str) or not name:
            raise StructuredRolloutError(
                f"{surface} tool call name must be a non-empty string"
            )
        return "tool_call", name
    if native_open or needs_continuation:
        return "incomplete", None
    return "final", None


def normalize_complete_prefix(conversation: object) -> PrefixNormalizationResult:
    """Validate source grammar and roll back only a trailing incomplete transaction.

    Malformed structure is never repaired. The returned conversation is a deep
    copy containing the longest prefix ending in a complete user-to-assistant
    transaction. An optional leading system message remains attached to that
    prefix for the structured renderer to wire into the first user turn.
    """
    source = _require_structured_conversation(conversation)
    raw_messages = source["messages"]
    if not isinstance(raw_messages, (list, tuple)) or not raw_messages:
        return PrefixNormalizationResult(
            conversation=None,
            skip_reason="no_completed_transaction",
            rolled_back_messages=0,
        )
    messages = list(raw_messages)
    first_semantic = 0
    if isinstance(messages[0], Mapping) and messages[0].get("role") == "system":
        system = _require_exact_fields(
            messages[0],
            {"role", "content"},
            surface="system message 0",
        )
        if not isinstance(system["content"], str):
            raise StructuredRolloutError("system message content must be a string")
        first_semantic = 1

    state = "user"
    outstanding_name: str | None = None
    completed_end = 0
    for index in range(first_semantic, len(messages)):
        raw_message = messages[index]
        if not isinstance(raw_message, Mapping):
            raise StructuredRolloutError(
                f"message {index} must be a mapping, got {type(raw_message).__name__}"
            )
        role = raw_message.get("role")
        if role == "system":
            raise StructuredRolloutError("system messages are allowed only at position zero")
        if state == "user":
            message = _require_exact_fields(
                raw_message,
                {"role", "content"},
                surface=f"user message {index}",
            )
            if role != "user" or not isinstance(message["content"], str):
                raise StructuredRolloutError(
                    f"message {index} must be a user message with string content"
                )
            state = "assistant"
            continue
        if state == "tool":
            message = _require_exact_fields(
                raw_message,
                {"role", "name", "content"},
                surface=f"tool result message {index}",
            )
            if role != "tool":
                raise StructuredRolloutError(
                    f"message {index} must resolve outstanding tool {outstanding_name!r}"
                )
            name = message["name"]
            if not isinstance(name, str) or not isinstance(message["content"], str):
                raise StructuredRolloutError(
                    f"tool result message {index} requires string name and content"
                )
            if name != outstanding_name:
                raise StructuredRolloutError(
                    f"tool result name {name!r} does not match outstanding call "
                    f"{outstanding_name!r}"
                )
            outstanding_name = None
            state = "assistant"
            continue

        message = _require_exact_fields(
            raw_message,
            {"role", "content"},
            surface=f"assistant message {index}",
        )
        if role != "assistant":
            raise StructuredRolloutError(
                f"message {index} must be assistant while grammar expects {state}"
            )
        assistant_state, call_name = _assistant_state(
            message["content"],
            surface=f"assistant message {index}",
        )
        if assistant_state == "incomplete":
            if index != len(messages) - 1:
                raise StructuredRolloutError(
                    f"assistant message {index} is incomplete in the middle of the source"
                )
            break
        if assistant_state == "tool_call":
            outstanding_name = call_name
            state = "tool"
            continue
        completed_end = index + 1
        state = "user"

    if completed_end == 0:
        return PrefixNormalizationResult(
            conversation=None,
            skip_reason="no_completed_transaction",
            rolled_back_messages=len(messages),
        )
    prefix_start = 0
    prefix = copy.deepcopy(messages[prefix_start:completed_end])
    return PrefixNormalizationResult(
        conversation={"messages": prefix},
        skip_reason=None,
        rolled_back_messages=len(messages) - completed_end,
    )


def validate_structured_row(
    row: StructuredRolloutRow,
    *,
    decode: Callable[[Sequence[int]], str] | None = None,
) -> None:
    """Validate every structural field and optional decoded-text equality."""
    if (
        isinstance(row.original_index, bool)
        or not isinstance(row.original_index, int)
        or row.original_index < 0
    ):
        raise StructuredRolloutError("original_index must be a non-negative int64")
    if row.original_index > (1 << 63) - 1:
        raise StructuredRolloutError("original_index exceeds int64")
    if not row.target_token_ids:
        raise StructuredRolloutError("structured row target_token_ids must not be empty")
    if len(row.target_token_ids) != len(row.target_mask):
        raise StructuredRolloutError(
            "len(target_token_ids) must equal len(target_mask)"
        )
    for index, token_id in enumerate(row.target_token_ids):
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id <= _INT32_MAX
        ):
            raise StructuredRolloutError(
                f"target_token_ids[{index}] must be a non-negative int32"
            )
    if not all(isinstance(value, bool) for value in row.target_mask):
        raise StructuredRolloutError("target_mask values must be bool")
    if not isinstance(row.text, str):
        raise StructuredRolloutError("text must be a string")
    if not row.logical_turns:
        raise StructuredRolloutError("logical_turns must not be empty")
    cursor = 0
    for index, turn in enumerate(row.logical_turns):
        if (
            isinstance(turn.start, bool)
            or not isinstance(turn.start, int)
            or isinstance(turn.end, bool)
            or not isinstance(turn.end, int)
            or not 0 <= turn.start <= _INT32_MAX
            or not 0 <= turn.end <= _INT32_MAX
        ):
            raise StructuredRolloutError(
                f"logical_turns[{index}] start/end must be non-negative int32"
            )
        if not isinstance(turn.role, str) or turn.role not in _ALLOWED_ROLES:
            raise StructuredRolloutError(
                f"logical_turns[{index}].role must be one of "
                f"{', '.join(sorted(_ALLOWED_ROLES))}"
            )
        if turn.start != cursor or turn.end <= turn.start:
            raise StructuredRolloutError(
                f"logical_turns[{index}] does not tile from cursor {cursor}: "
                f"[{turn.start}, {turn.end})"
            )
        if turn.end > len(row.target_token_ids):
            raise StructuredRolloutError(
                f"logical_turns[{index}].end exceeds target stream"
            )
        cursor = turn.end
    if cursor != len(row.target_token_ids):
        raise StructuredRolloutError(
            f"logical_turns tile [0, {cursor}), expected [0, {len(row.target_token_ids)})"
        )
    if decode is not None:
        decoded = decode(row.target_token_ids)
        if decoded != row.text:
            raise StructuredRolloutError(
                "row text does not equal tokenizer.decode(target_token_ids)"
            )


def structured_row_from_render(
    rendered: StructuredConversationRender,
    *,
    bos_token_id: int,
    decode: Callable[[Sequence[int]], str],
    original_index: int,
) -> StructuredRolloutRow:
    """Remove BOS and construct one authoritative validated artifact row."""
    if not rendered.token_ids or rendered.token_ids[0] != bos_token_id:
        raise StructuredRolloutError("structured render does not begin with decoder BOS")
    if not rendered.assistant_mask or rendered.assistant_mask[0]:
        raise StructuredRolloutError("structured render BOS mask must be false")
    target_ids = tuple(rendered.token_ids[1:])
    row = StructuredRolloutRow(
        target_token_ids=target_ids,
        target_mask=tuple(rendered.assistant_mask[1:]),
        logical_turns=rendered.logical_turns,
        text=decode(target_ids),
        original_index=original_index,
    )
    validate_structured_row(row, decode=decode)
    return row


def structured_row_digest(row: StructuredRolloutRow) -> str:
    """Return the full token/mask/span/text/index SHA-256 row digest."""
    validate_structured_row(row)
    payload = {
        "target_token_ids": list(row.target_token_ids),
        "target_mask": list(row.target_mask),
        "logical_turns": [
            {"role": turn.role, "start": turn.start, "end": turn.end}
            for turn in row.logical_turns
        ],
        "text": row.text,
        "original_index": row.original_index,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def structured_rows_to_table(rows: Sequence[StructuredRolloutRow]) -> pa.Table:
    """Serialize rows with the explicit schema, including the zero-row case."""
    for row in rows:
        validate_structured_row(row)
    columns = [
        pa.array([list(row.target_token_ids) for row in rows], type=pa.list_(pa.int32())),
        pa.array([list(row.target_mask) for row in rows], type=pa.list_(pa.bool_())),
        pa.array(
            [
                [
                    {"role": turn.role, "start": turn.start, "end": turn.end}
                    for turn in row.logical_turns
                ]
                for row in rows
            ],
            type=STRUCTURED_ROLLOUT_ARROW_SCHEMA.field("logical_turns").type,
        ),
        pa.array([row.text for row in rows], type=pa.string()),
        pa.array([row.original_index for row in rows], type=pa.int64()),
    ]
    return pa.Table.from_arrays(columns, schema=STRUCTURED_ROLLOUT_ARROW_SCHEMA)


def structured_table_to_rows(
    table: pa.Table,
    *,
    decode: Callable[[Sequence[int]], str] | None = None,
) -> tuple[StructuredRolloutRow, ...]:
    """Fail closed on schema drift and deserialize every validated row."""
    if "assistant_loss_mask" in table.column_names:
        raise StructuredRolloutError(
            "assistant_loss_mask is a removed legacy column, not a structured target mask"
        )
    if not table.schema.equals(STRUCTURED_ROLLOUT_ARROW_SCHEMA, check_metadata=False):
        raise StructuredRolloutError(
            "structured rollout Arrow schema mismatch: "
            f"expected={STRUCTURED_ROLLOUT_ARROW_SCHEMA}, got={table.schema}"
        )
    result = []
    for raw in table.to_pylist():
        turns = tuple(
            LogicalTurnSpan(
                role=turn["role"],
                start=turn["start"],
                end=turn["end"],
            )
            for turn in raw["logical_turns"]
        )
        row = StructuredRolloutRow(
            target_token_ids=tuple(raw["target_token_ids"]),
            target_mask=tuple(raw["target_mask"]),
            logical_turns=turns,
            text=raw["text"],
            original_index=raw["original_index"],
        )
        validate_structured_row(row, decode=decode)
        result.append(row)
    return tuple(result)


def assert_rollout_artifact_format(
    metadata: Mapping[str, object],
    *,
    structured: bool,
) -> None:
    """Prevent legacy and structured consumers from crossing namespaces."""
    actual = metadata.get("format")
    expected = STRUCTURED_ROLLOUT_FORMAT if structured else "full_nanochat_text"
    if actual != expected:
        consumer = "structured rollout" if structured else "legacy full_nanochat_text"
        raise StructuredRolloutError(
            f"{consumer} consumer requires format={expected!r}, got {actual!r}; "
            "use the separate artifact namespace produced for that consumer"
        )


def clip_structured_row(
    row: StructuredRolloutRow,
    *,
    context_length: int,
    decode: Callable[[Sequence[int]], str],
) -> StructuredRolloutRow:
    """Mechanically keep one target prefix and clip intersecting spans."""
    validate_structured_row(row)
    if isinstance(context_length, bool) or not isinstance(context_length, int):
        raise StructuredRolloutError("context_length must be an integer")
    if context_length <= 0:
        raise StructuredRolloutError("context_length must be positive")
    retained = min(context_length, len(row.target_token_ids))
    turns = tuple(
        LogicalTurnSpan(turn.role, turn.start, min(turn.end, retained))
        for turn in row.logical_turns
        if turn.start < retained
    )
    clipped = StructuredRolloutRow(
        target_token_ids=row.target_token_ids[:retained],
        target_mask=row.target_mask[:retained],
        logical_turns=turns,
        text=decode(row.target_token_ids[:retained]),
        original_index=row.original_index,
    )
    validate_structured_row(clipped, decode=decode)
    return clipped


def _fixed_segments(length: int, segment_length: int) -> tuple[TargetSegment, ...]:
    return tuple(
        TargetSegment(
            index=index,
            start=start,
            end=end,
            role=None,
        )
        for index, (start, end) in enumerate(
            _segment_fixed(length, PositiveInt.parse(segment_length))
        )
    )


def _turn_segments(
    turns: Sequence[LogicalTurnSpan],
    segment_length: int,
) -> tuple[TargetSegment, ...]:
    """Chunk a validated contiguous turn tiling starting at token zero.

    ``build_conversation_environment`` guarantees this contract through
    ``validate_structured_row`` before calling this helper.
    """
    starts = [turn.start for turn in turns]
    segments = segment_turn_delimited(
        starts,
        n=turns[-1].end,
        segment_size=segment_length,
    )
    return tuple(
        TargetSegment(
            index=index,
            start=start,
            end=end,
            role=turns[bisect_right(starts, start) - 1].role,
        )
        for index, (start, end) in enumerate(segments)
    )


def build_conversation_environment(
    row: StructuredRolloutRow,
    *,
    bos_token_id: int,
    context_length: int,
    segment_length: int = 256,
    segmentation_mode: SegmentationMode = SegmentationMode.FIXED_LENGTH,
    reward_mask_mode: RewardMaskMode = RewardMaskMode.ASSISTANT_ONLY,
    min_reward_tokens: int = 8,
) -> EnvironmentBuildResult:
    """Build one input-aligned sequential environment without packing."""
    validate_structured_row(row)
    for name, value in (
        ("context_length", context_length),
        ("segment_length", segment_length),
        ("min_reward_tokens", min_reward_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise StructuredRolloutError(f"{name} must be a positive integer")
    try:
        segmentation_mode = SegmentationMode(segmentation_mode)
        reward_mask_mode = RewardMaskMode(reward_mask_mode)
    except ValueError as exc:
        raise StructuredRolloutError(str(exc)) from exc

    retained = min(context_length, len(row.target_token_ids))
    if retained == 0:
        return EnvironmentBuildResult(None, EnvironmentFilterReason.NO_TARGETS)
    targets = row.target_token_ids[:retained]
    target_mask = row.target_mask[:retained]
    turns = tuple(
        LogicalTurnSpan(turn.role, turn.start, min(turn.end, retained))
        for turn in row.logical_turns
        if turn.start < retained
    )
    if len(turns) <= 1:
        return EnvironmentBuildResult(None, EnvironmentFilterReason.AT_MOST_ONE_TURN)
    segments = (
        _fixed_segments(retained, segment_length)
        if segmentation_mode is SegmentationMode.FIXED_LENGTH
        else _turn_segments(turns, segment_length)
    )
    if len(segments) <= 1:
        return EnvironmentBuildResult(None, EnvironmentFilterReason.AT_MOST_ONE_SEGMENT)
    reward_mask = (
        tuple(target_mask)
        if reward_mask_mode is RewardMaskMode.ASSISTANT_ONLY
        else (True,) * retained
    )
    if sum(reward_mask) < min_reward_tokens:
        return EnvironmentBuildResult(
            None,
            EnvironmentFilterReason.INSUFFICIENT_REWARD_TARGETS,
        )

    target_segment = [0] * retained
    for segment in segments:
        target_segment[segment.start : segment.end] = [segment.index] * (
            segment.end - segment.start
        )
    inputs = (bos_token_id, *targets[:-1])
    seg_map = (0, *target_segment[:-1])
    environment = ConversationEnvironment(
        original_index=row.original_index,
        inputs=tuple(inputs),
        targets=tuple(targets),
        reward_mask=reward_mask,
        logical_turns=turns,
        segments=segments,
        seg_map=tuple(seg_map),
    )
    return EnvironmentBuildResult(environment, None)


def stateless_seed(fields: Mapping[str, object]) -> int:
    """Hash canonical JSON fields into a deterministic nonnegative 63-bit seed."""
    if not isinstance(fields, Mapping) or not all(isinstance(key, str) for key in fields):
        raise StructuredRolloutError("stateless seed fields must be a string-keyed mapping")
    encoded = json.dumps(
        dict(fields),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


def rustbpe_tokenizer_sha256(tokenizer: object) -> str:
    """Digest the complete tiktoken vocabulary, pattern, and special-token map."""
    encoding = getattr(tokenizer, "enc", None)
    mergeable = getattr(encoding, "_mergeable_ranks", None)
    specials = getattr(encoding, "_special_tokens", None)
    pattern = getattr(encoding, "_pat_str", None)
    if not isinstance(mergeable, Mapping) or not isinstance(specials, Mapping) or not isinstance(
        pattern, str
    ):
        test_identity = getattr(tokenizer, "structured_tokenizer_sha256", None)
        if isinstance(test_identity, str) and len(test_identity) == 64:
            return test_identity
        raise StructuredRolloutError(
            "structured conversion requires a RustBPETokenizer with a complete "
            "tiktoken encoding identity"
        )
    if not all(
        isinstance(piece, bytes)
        and isinstance(rank, int)
        and not isinstance(rank, bool)
        for piece, rank in mergeable.items()
    ):
        raise StructuredRolloutError("RustBPE mergeable ranks are malformed")
    if not all(
        isinstance(token, str)
        and isinstance(token_id, int)
        and not isinstance(token_id, bool)
        for token, token_id in specials.items()
    ):
        raise StructuredRolloutError("RustBPE special-token mapping is malformed")
    typed_mergeable = cast(Mapping[bytes, int], mergeable)
    typed_specials = cast(Mapping[str, int], specials)
    payload = {
        "pattern": pattern,
        "mergeable_ranks": [
            [piece.hex(), rank]
            for piece, rank in sorted(
                typed_mergeable.items(),
                key=lambda item: (item[1], item[0]),
            )
        ],
        "special_tokens": sorted(typed_specials.items()),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ConversationEnvironment",
    "EnvironmentBuildResult",
    "EnvironmentFilterReason",
    "PrefixNormalizationResult",
    "RewardMaskMode",
    "STRUCTURED_RENDERER_VERSION",
    "STRUCTURED_ROLLOUT_ARROW_SCHEMA",
    "STRUCTURED_ROLLOUT_FORMAT",
    "STRUCTURED_ROLLOUT_SCHEMA_VERSION",
    "SegmentationMode",
    "StructuredRolloutError",
    "StructuredRolloutRow",
    "TargetSegment",
    "assert_rollout_artifact_format",
    "build_conversation_environment",
    "clip_structured_row",
    "normalize_complete_prefix",
    "rustbpe_tokenizer_sha256",
    "stateless_seed",
    "structured_row_digest",
    "structured_row_from_render",
    "structured_rows_to_table",
    "structured_table_to_rows",
    "validate_structured_row",
]
