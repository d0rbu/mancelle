"""Lazy catalog of standard-SFT validation conversations."""

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from scripts.preprocess.convert_sft_to_parquet import (
    prepare_messages_for_full_nanochat_render,
    render_conversation_to_full_nanochat_text,
)
from steering.feature_paths import resolve_data_dir
from steering.sft_schema import SFTConversationSource
from tasks.mixtures import SFTMixture, build_task_mixtures

RENDER_TOKEN_CAP = 2048


@dataclass(frozen=True)
class ConvoEntry:
    index: int
    n_user_turns: int
    token_count: int
    first_user_snippet: str
    skip_reason: str | None
    messages: tuple[dict, ...]


class ConvoCatalog:
    def __init__(self, tokenizer, mixture: SFTConversationSource) -> None:
        self._tokenizer = tokenizer
        self._mixture = mixture
        self._entries: dict[int, ConvoEntry] = {}

    @classmethod
    def load(cls, tokenizer) -> "ConvoCatalog":
        _, validation_mixture = build_task_mixtures(SFTMixture.STANDARD)
        if validation_mixture is None:
            raise ValueError("standard SFT has no validation mixture")
        return cls(tokenizer, validation_mixture)

    def __len__(self) -> int:
        return len(self._mixture)

    def entry(self, index: int) -> ConvoEntry:
        cached = self._entries.get(index)
        if cached is not None:
            return cached

        conversation = self._mixture[index]
        prepared_messages, skip_reason = prepare_messages_for_full_nanochat_render(conversation)
        messages = () if prepared_messages is None else tuple(prepared_messages)

        if prepared_messages is None:
            token_count = 0
            n_user_turns = 0
        else:
            rendered_ids, _ = self._tokenizer.render_conversation(
                {"messages": list(messages)}, max_tokens=2**31
            )
            token_count = len(rendered_ids)
            n_user_turns = sum(message.get("role") == "user" for message in prepared_messages)

        first_user_content = next(
            (
                message.get("content", "")
                for message in conversation.get("messages", [])
                if message.get("role") == "user"
            ),
            "",
        )
        first_user_snippet = " ".join(str(first_user_content).split())[:80]

        result = ConvoEntry(
            index=index,
            n_user_turns=n_user_turns,
            token_count=token_count,
            first_user_snippet=first_user_snippet,
            skip_reason=skip_reason,
            messages=messages,
        )
        self._entries[index] = result
        return result


def prefix_messages(entry: ConvoEntry, start_user_turn: int) -> list[dict]:
    """Return the role-position prefix before a user turn.

    Turn zero is always empty: a leading system message belongs to the first
    turn's context and cannot be rendered safely as a dangling system-only
    prefix, so replacing turn zero replaces that system context too.
    """

    if entry.skip_reason is not None:
        raise ValueError(f"skipped conversation original_index={entry.index}: {entry.skip_reason}")
    if not 0 <= start_user_turn <= entry.n_user_turns:
        raise ValueError(
            f"start_user_turn={start_user_turn} must be between 0 and {entry.n_user_turns}"
        )
    if start_user_turn == 0:
        return []
    user_positions = [
        index for index, message in enumerate(entry.messages) if message.get("role") == "user"
    ]
    if start_user_turn == entry.n_user_turns:
        return list(entry.messages)
    return list(entry.messages[: user_positions[start_user_turn]])


def prefix_headroom_error(
    entry: ConvoEntry,
    start_user_turn: int,
    tokenizer,
    *,
    sequence_len: int,
    max_new_tokens_per_turn: int,
) -> str | None:
    """Return a selection-time sequence-budget error for one prefix, if any."""

    prefix = prefix_messages(entry, start_user_turn)
    if prefix:
        rendered, _mask = tokenizer.render_conversation({"messages": prefix}, max_tokens=2**31)
        prefix_tokens = len(rendered)
    else:
        prefix_tokens = 0
    reserved_tokens = max(32, max_new_tokens_per_turn)
    required_tokens = prefix_tokens + reserved_tokens
    if required_tokens < sequence_len:
        return None
    return (
        f"prefix uses {prefix_tokens} tokens plus {reserved_tokens} reserved generation "
        f"tokens ({required_tokens} total), but sequence_len is {sequence_len}"
    )


def default_val_parquet_dir() -> Path:
    """Return the production-pinned standard-SFT validation parquet family."""

    return resolve_data_dir(None, "standard_sft_val")


def verify_against_parquet(
    catalog: ConvoCatalog,
    entry: ConvoEntry,
    tokenizer,
    parquet_dir: Path,
) -> None:
    original_index = entry.index
    if not parquet_dir.is_dir():
        raise ValueError(
            f"original_index={original_index}: parquet directory does not exist: {parquet_dir}"
        )

    expected_text, skip_reason = render_conversation_to_full_nanochat_text(
        tokenizer,
        catalog._mixture[original_index],
    )
    if expected_text is None:
        raise ValueError(
            f"original_index={original_index}: skipped conversation cannot be verified "
            f"against parquet ({skip_reason})"
        )

    stored_text: str | None = None
    for shard_path in sorted(parquet_dir.glob("shard_*.parquet")):
        table = pq.read_table(shard_path, columns=["original_index", "text"])
        row_matches = pc.call_function(
            "equal",
            [table.column("original_index"), pa.scalar(original_index)],
        )
        matching = table.filter(row_matches)
        if matching.num_rows:
            value = matching.column("text")[0].as_py()
            if not isinstance(value, str):
                raise ValueError(f"original_index={original_index}: parquet text is not a string")
            stored_text = value
        if stored_text is not None:
            break

    if stored_text is None:
        raise ValueError(f"original_index={original_index}: row is missing from parquet shards")
    if stored_text.encode("utf-8") != expected_text.encode("utf-8"):
        raise ValueError(
            f"original_index={original_index}: parquet text does not match live rendering"
        )


def assert_renderable(token_count: int, *, source: str) -> None:
    if token_count >= RENDER_TOKEN_CAP:
        raise ValueError(
            f"{source} rendered to {token_count} tokens; must be strictly below {RENDER_TOKEN_CAP}"
        )
