"""Convert SFT conversation data to full-sequence parquet for feature extraction.

Stores the exact nanochat-rendered training text (minus decoder BOS) in a
``text`` column, its logical-turn character starts, and ``original_index``.
Feature extraction therefore sees the same full sequence the SFT trainer packs
and supervises.

Conversations whose last message is not from the assistant are truncated to
the last assistant message (common in SmolTalk).  Conversations with no
assistant message or with an empty last assistant response are skipped.

Usage:
    uv run python -m scripts.preprocess.convert_sft_to_parquet mixture=standard split=both
    uv run python -m scripts.preprocess.convert_sft_to_parquet mixture=safety split=val max_conversations=100
"""

import fcntl
import hashlib
import json
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import hydra
import pyarrow as pa
import pyarrow.parquet as pq
from beartype import beartype
from loguru import logger
from omegaconf import DictConfig

from errors import DataLoadError
from nanochat.atomic_io import atomic_json_dump, atomic_write_parquet
from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer
from steering.refined_types import PositiveIntLike
from steering.sft.mixture_fingerprint import (
    MIXTURE_FINGERPRINT_KEY,
    MixtureFingerprint,
    compute_mixture_fingerprint,
    format_mixture_fingerprint_mismatch,
    normalize_mixture_fingerprint,
)
from steering.sft.structured_rollout import (
    STRUCTURED_RENDERER_VERSION,
    STRUCTURED_ROLLOUT_FORMAT,
    STRUCTURED_ROLLOUT_SCHEMA_VERSION,
    StructuredRolloutError,
    StructuredRolloutRow,
    normalize_complete_prefix,
    rustbpe_tokenizer_sha256,
    structured_row_digest,
    structured_row_from_render,
    structured_rows_to_table,
    structured_table_to_rows,
)
from steering.sft.turn_spans import (
    TurnSpanError,
    rendered_text_and_turn_char_starts,
)
from steering.sft_schema import (
    SFTConversationSource,
    SFTConversionMeta,
    SFTSplit,
    SFTSplitSelection,
    legacy_sft_columns,
    sft_split_dir,
)
from tasks.mixtures import (
    STEERED_MIXTURES,
    SFTMixture,
    build_task_mixtures,
    steered_mixture_overrides,
)
from tasks.steered import SteeredTaskMixture

LEGACY_SFT_FORMAT = "full_nanochat_text"


def _is_empty_multipart_response(parts: list[dict]) -> bool:
    """Return True iff every part has empty or whitespace-only ``text``.

    The nanochat conversation renderer preserves tool-use structure, so an
    empty ``python`` part can still render to non-empty tool markers. The
    converter checks raw part text first and skips rows where every part is
    empty.

    An empty ``parts`` list is treated as empty. Missing ``text`` keys are
    treated as empty strings.
    """
    if not parts:
        return True
    return all(not part.get("text", "").strip() for part in parts)


def prepare_messages_for_full_nanochat_render(
    conversation: dict,
) -> tuple[list[dict] | None, str | None]:
    """Return messages to render, or a skip reason.

    The conversion format is the exact nanochat-rendered training sequence.
    This helper owns the shared "which conversation rows survive conversion"
    rule so audits compare against the same producer behavior.

    Returns:
        ``(messages, None)`` when the row should be rendered.
        ``(None, "no_assistant")`` when there is no assistant turn.
        ``(None, "empty_response")`` when the final assistant content is empty.
    """
    messages = conversation.get("messages", [])
    last_asst_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "assistant":
            last_asst_idx = idx
            break
    if last_asst_idx is None:
        return None, "no_assistant"

    rendered_messages = list(messages[: last_asst_idx + 1])
    response_content = rendered_messages[-1].get("content", "")
    if isinstance(response_content, list):
        if _is_empty_multipart_response(response_content):
            return None, "empty_response"
    elif not isinstance(response_content, str) or not response_content.strip():
        return None, "empty_response"

    return rendered_messages, None


def _render_source_ids(tokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
    """Render once and enforce the BOS-inclusive legacy-render contract."""
    ids, _ = tokenizer.render_conversation({"messages": messages})
    bos_id = tokenizer.get_bos_token_id()
    if not ids or ids[0] != bos_id:
        raise TurnSpanError(
            "render_conversation must return a BOS-inclusive token stream; "
            "a render without the tokenizer BOS violates the producer contract"
        )
    return ids, ids[1:]


def render_messages_to_full_nanochat_row(
    tokenizer,
    messages: list[dict],
) -> tuple[str, tuple[int, ...]]:
    """Render once, then decode text and validate its logical turn starts."""
    ids, _source_ids = _render_source_ids(tokenizer, messages)
    return rendered_text_and_turn_char_starts(ids, tokenizer)


def render_messages_to_full_nanochat_text(tokenizer, messages: list[dict]) -> str:
    """Render once and decode text without computing or validating turn spans."""
    _, source_ids = _render_source_ids(tokenizer, messages)
    return tokenizer.decode(source_ids)


def render_conversation_to_full_nanochat_text(
    tokenizer,
    conversation: dict,
) -> tuple[str | None, str | None]:
    """Render one conversation to parquet ``text`` or return a skip reason."""
    messages, skip_reason = prepare_messages_for_full_nanochat_render(conversation)
    if messages is None:
        return None, skip_reason
    return render_messages_to_full_nanochat_text(tokenizer, messages), None


def read_conversion_status(output_dir: Path | str) -> SFTConversionMeta | None:
    """Read validated conversion metadata, or return None if not started.

    Args:
        output_dir: Directory where conversion_meta.json would reside.

    Returns:
        Parsed metadata with mixture, completed_shards, total_shards,
        total_conversations -- or None if the directory or file doesn't exist.
    """
    meta_path = Path(output_dir) / "conversion_meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return SFTConversionMeta.parse(json.load(f))


def _write_metadata(
    output_dir: Path,
    mixture_name: str,
    split: SFTSplit,
    completed_shards: int,
    total_shards: int,
    total_conversations: int,
    mixture_fingerprint: MixtureFingerprint,
    steered_rows: int | None = None,
    *,
    max_conversations: int | None,
    conversations_per_shard: int,
    written_conversations: int,
) -> None:
    """Atomically publish immutable conversion identity and committed row counts."""
    meta = SFTConversionMeta.parse(
        {
            "mixture": mixture_name,
            "split": split.value,
            "format": "full_nanochat_text",
            "completed_shards": completed_shards,
            "total_shards": total_shards,
            "total_conversations": total_conversations,
            "max_conversations": max_conversations,
            "conversations_per_shard": conversations_per_shard,
            "written_conversations": written_conversations,
            "skipped_conversations": (
                min(completed_shards * conversations_per_shard, total_conversations)
                - written_conversations
            ),
            MIXTURE_FINGERPRINT_KEY: mixture_fingerprint,
            **({} if steered_rows is None else {"steered_rows": steered_rows}),
        }
    )
    meta_path = output_dir / "conversion_meta.json"
    atomic_json_dump(meta.to_json_dict(), meta_path)


def _shard_filename(shard_idx: int) -> str:
    """Generate a zero-padded shard filename."""
    return f"shard_{shard_idx:05d}.parquet"


def _validate_existing_full_nanochat_shard(shard_path: Path) -> None:
    """Reject stale SFT parquet shards before resume skips them."""
    schema_names = set(pq.read_schema(shard_path).names)
    legacy_cols = legacy_sft_columns(schema_names)
    missing = {"text", "original_index", "turn_char_starts"} - schema_names
    if not legacy_cols and missing == {"turn_char_starts"}:
        raise DataLoadError(
            f"Existing shard {shard_path} predates the turn-boundary column and "
            "must be regenerated into a fresh output_dir."
        )
    if legacy_cols or missing:
        raise DataLoadError(
            f"Existing shard {shard_path} is not a full_nanochat SFT shard. "
            f"Found columns: {sorted(schema_names)}. Remove the stale output_dir "
            "or choose a new one and regenerate."
        )


@contextmanager
def _conversion_lock(output_dir: Path) -> Iterator[None]:
    """Serialize the entire conversion transaction without leaving a lock file."""
    descriptor = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DataLoadError(
                f"Another SFT conversion is already writing {output_dir}; retry after it exits."
            ) from exc
        yield
    finally:
        os.close(descriptor)


@beartype
def convert_mixture_to_parquet(
    mixture: SFTConversationSource,
    output_dir: Path | str,
    max_conversations: PositiveIntLike | None = None,
    conversations_per_shard: PositiveIntLike = 40000,
    mixture_name: str = "unknown",
    split: SFTSplit = SFTSplit.TRAIN,
) -> None:
    """Convert a TaskMixture of SFT conversations to parquet shards.

    Stores ``text`` as the decoded nanochat-rendered conversation (minus the
    decoder BOS, because the extraction tokenizer adds its own BOS) and
    ``original_index`` plus logical ``turn_char_starts`` over that text. This
    matches the token stream used by nanochat SFT and lets feature extraction
    segment the full rendered sample instead of only the final assistant
    response.

    Conversations are truncated to the last assistant message if needed.
    Conversations with no assistant message or empty responses are skipped;
    ``original_index`` correctly tracks which mixture indices were kept.

    Progress is tracked in ``conversion_meta.json`` so conversion can be
    resumed if interrupted.

    Args:
        mixture: A TaskMixture (or any object exposing ``len()``/``mixture[i]``)
            containing SFT conversations.
        output_dir: Directory to write parquet shards into.
        max_conversations: Cap on total conversations to convert (None = all).
        conversations_per_shard: Number of conversations per parquet file.
        mixture_name: Name for metadata (e.g. "standard", "safety").
        split: Source task split.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with _conversion_lock(output_dir):
        _convert_mixture_to_parquet(
            mixture,
            output_dir,
            max_conversations,
            conversations_per_shard,
            mixture_name,
            split,
        )


def _convert_mixture_to_parquet(
    mixture: SFTConversationSource,
    output_dir: Path,
    max_conversations: PositiveIntLike | None,
    conversations_per_shard: PositiveIntLike,
    mixture_name: str,
    split: SFTSplit,
) -> None:
    """Convert under the directory lock, including all resume reads and writes."""
    nanochat_tokenizer = get_tokenizer()

    total_available = len(mixture)
    total_conversations = (
        min(total_available, max_conversations)
        if max_conversations is not None
        else total_available
    )
    total_shards = math.ceil(total_conversations / conversations_per_shard)
    mixture_fingerprint = compute_mixture_fingerprint(
        mixture,
        mixture_name=mixture_name,
        split=split,
    )
    # A steered conversion records how many of its rows came from the steered
    # JSONL so the artifact carries the mix ratio without a content hash.
    steered_rows = (
        len(mixture.tasks[mixture.steered_task_index])
        if isinstance(mixture, SteeredTaskMixture)
        else None
    )

    # Check for existing progress
    existing_meta = read_conversion_status(output_dir)
    if existing_meta is not None:
        existing_format = existing_meta.format
        if existing_format != "full_nanochat_text":
            raise DataLoadError(
                f"Existing conversion at {output_dir} uses format={existing_format!r}, "
                "but SFT conversion requires 'full_nanochat_text'. Use a separate output_dir."
            )
        existing_split = existing_meta.split
        if existing_split is None and split is not SFTSplit.TRAIN:
            raise DataLoadError(
                f"Existing conversion at {output_dir} has no split metadata, "
                f"but SFT conversion requires split={split.value!r}. Legacy split-absent "
                "metadata is valid only for train conversions. Use a separate output_dir."
            )
        if existing_split is not None and existing_split is not split:
            raise DataLoadError(
                f"Existing conversion at {output_dir} uses split={existing_split.value!r}, "
                f"but SFT conversion requires split={split.value!r}. Use a separate output_dir."
            )
        existing_fingerprint_raw = existing_meta.raw.get(MIXTURE_FINGERPRINT_KEY)
        if existing_fingerprint_raw is None:
            if existing_meta.completed_shards > 0:
                raise DataLoadError(
                    f"Existing conversion at {output_dir} has completed shards but no "
                    "mixture_fingerprint. Refusing to stamp today's live mixture identity "
                    "onto legacy rows that may have been produced from different content. "
                    "Legacy conversions cannot be upgraded in place; re-run with "
                    "output_dir=/path/to/fresh/sft_data and regenerate the conversion."
                )
        else:
            try:
                existing_fingerprint = normalize_mixture_fingerprint(
                    existing_fingerprint_raw,
                    source=f"{output_dir}/conversion_meta.json mixture_fingerprint",
                )
            except ValueError as exc:
                raise DataLoadError(str(exc)) from exc
            if existing_fingerprint != mixture_fingerprint:
                raise DataLoadError(
                    f"Existing conversion at {output_dir} has a mixture fingerprint "
                    "that does not match the live TaskMixture: "
                    + format_mixture_fingerprint_mismatch(
                        "conversion_fingerprint",
                        existing_fingerprint,
                        "live_fingerprint",
                        mixture_fingerprint,
                    )
                    + ". Re-run with output_dir=/path/to/fresh/sft_data and regenerate."
                )
        start_shard = existing_meta.completed_shards
        if not 0 <= start_shard <= total_shards:
            raise DataLoadError("conversion completed_shards is outside the requested shard range")
        for shard_idx in range(start_shard):
            shard_path = output_dir / _shard_filename(shard_idx)
            if not shard_path.exists():
                raise DataLoadError(
                    f"metadata records completed shard {shard_idx} but the file is missing; "
                    "use a fresh output_dir"
                )
    else:
        start_shard = 0

    existing_paths = set(output_dir.glob("*.parquet"))
    for shard_path in sorted(existing_paths):
        _validate_existing_full_nanochat_shard(shard_path)
    if existing_meta is None and existing_paths:
        raise DataLoadError(
            "Existing parquet shards have no conversion_meta.json identity; use a fresh output_dir"
        )
    expected_paths = {output_dir / _shard_filename(index) for index in range(total_shards)}
    if extra_paths := existing_paths - expected_paths:
        raise DataLoadError(
            f"Unexpected conversion shards {sorted(extra_paths)}; use a fresh output_dir"
        )
    if existing_meta is not None:
        expected_identity = {
            "mixture": mixture_name,
            "total_conversations": total_conversations,
            "total_shards": total_shards,
            "max_conversations": max_conversations,
            "conversations_per_shard": conversations_per_shard,
        }
        for field, expected in expected_identity.items():
            # Empty legacy progress has no rows to misattribute and may be
            # initialized. Existing artifacts must never acquire guessed sizes.
            if field not in existing_meta.raw and start_shard == 0 and not existing_paths:
                continue
            actual = existing_meta.raw.get(field)
            if (
                field not in existing_meta.raw
                or type(actual) is not type(expected)
                or actual != expected
            ):
                raise DataLoadError(
                    f"Conversion resume identity mismatch for {field}: recorded {actual!r}, "
                    f"requested {expected!r}. Use a fresh output_dir."
                )

    # Audit every reused shard, including a shard published just before a crash
    # advanced the metadata cursor. Count alone cannot distinguish a legitimate
    # skipped conversation from a missing row or a wrong shard boundary.
    # The fingerprint binds composition only, so re-render payloads as well.
    existing_rows: dict[int, int] = {}
    for shard_idx in range(total_shards):
        shard_path = output_dir / _shard_filename(shard_idx)
        if shard_path not in existing_paths:
            continue
        conv_start = shard_idx * conversations_per_shard
        conv_end = min(conv_start + conversations_per_shard, total_conversations)
        expected_indices: list[int] = []
        expected_payloads: list[tuple[str, tuple[int, ...]]] = []
        for index in range(conv_start, conv_end):
            messages, _ = prepare_messages_for_full_nanochat_render(mixture[index])
            if messages is not None:
                expected_indices.append(index)
                expected_payloads.append(
                    render_messages_to_full_nanochat_row(nanochat_tokenizer, messages)
                )
        table = pq.read_table(shard_path, columns=["original_index", "text", "turn_char_starts"])
        actual_indices = table["original_index"].to_pylist()
        if (
            any(type(index) is not int for index in actual_indices)
            or actual_indices != expected_indices
        ):
            raise DataLoadError(
                f"Conversion resume row coverage mismatch in {shard_path}: expected "
                f"{len(expected_indices)} kept indices from [{conv_start}, {conv_end}), got "
                f"{len(actual_indices)} rows with different indices. Use a fresh output_dir."
            )
        for row, (expected_text, expected_starts) in zip(
            table.to_pylist(), expected_payloads, strict=True
        ):
            actual_starts = row["turn_char_starts"]
            if (
                row["text"] != expected_text
                or not isinstance(actual_starts, list)
                or any(type(start) is not int for start in actual_starts)
                or actual_starts != list(expected_starts)
            ):
                raise DataLoadError(
                    f"Conversion resume row content mismatch in {shard_path} at "
                    f"original_index={row['original_index']}: text or turn boundaries "
                    "differ from the live render. Use a fresh output_dir."
                )
        existing_rows[shard_idx] = len(actual_indices)

    written_conversations = sum(existing_rows[index] for index in range(start_shard))
    total_processed_all = min(start_shard * conversations_per_shard, total_conversations)
    total_skipped_all = total_processed_all - written_conversations
    # Identity precedes the first shard, so interrupted first-shard conversion
    # can be resumed without adopting an unproven orphan file.
    _write_metadata(
        output_dir,
        mixture_name,
        split,
        start_shard,
        total_shards,
        total_conversations,
        mixture_fingerprint,
        steered_rows,
        max_conversations=max_conversations,
        conversations_per_shard=conversations_per_shard,
        written_conversations=written_conversations,
    )
    if existing_meta is not None:
        logger.info(f"Resuming conversion: {start_shard}/{total_shards} shards already completed")

    for shard_idx in range(start_shard, total_shards):
        shard_path = output_dir / _shard_filename(shard_idx)

        # An audited shard ahead of the progress cursor survived a crash.
        if shard_idx in existing_rows:
            written_conversations += existing_rows[shard_idx]
            total_processed_all = min(
                (shard_idx + 1) * conversations_per_shard, total_conversations
            )
            total_skipped_all = total_processed_all - written_conversations
            logger.info(f"Shard {shard_idx} already exists, audited and reused")
            if shard_idx + 1 < total_shards:
                _write_metadata(
                    output_dir,
                    mixture_name,
                    split,
                    shard_idx + 1,
                    total_shards,
                    total_conversations,
                    mixture_fingerprint,
                    steered_rows,
                    max_conversations=max_conversations,
                    conversations_per_shard=conversations_per_shard,
                    written_conversations=written_conversations,
                )
            continue

        conv_start = shard_idx * conversations_per_shard
        conv_end = min(conv_start + conversations_per_shard, total_conversations)

        texts: list[str] = []
        original_indices: list[int] = []
        row_turn_char_starts: list[tuple[int, ...]] = []
        skipped_no_assistant = 0
        skipped_empty_response = 0
        for conv_idx in range(conv_start, conv_end):
            conversation = mixture[conv_idx]
            messages, skip_reason = prepare_messages_for_full_nanochat_render(conversation)
            if skip_reason == "no_assistant":
                skipped_no_assistant += 1
                continue
            if skip_reason == "empty_response":
                logger.debug(f"skipping empty response idx={conv_idx}")
                skipped_empty_response += 1
                continue
            assert messages is not None
            text, starts = render_messages_to_full_nanochat_row(
                nanochat_tokenizer,
                messages,
            )

            texts.append(text)
            original_indices.append(conv_idx)
            row_turn_char_starts.append(starts)

        total_skipped = skipped_no_assistant + skipped_empty_response
        shard_size = conv_end - conv_start
        total_skipped_all += total_skipped
        total_processed_all += shard_size
        if total_skipped:
            logger.warning(
                f"Shard {shard_idx}: skipped {total_skipped} conversations "
                f"({skipped_no_assistant} no assistant message, "
                f"{skipped_empty_response} empty response)"
            )

        # Write parquet shard
        table = pa.table(
            {
                "text": texts,
                "original_index": original_indices,
                "turn_char_starts": pa.array(
                    row_turn_char_starts,
                    type=pa.list_(pa.int32()),
                ),
            }
        )
        written = len(texts)
        atomic_write_parquet(table, shard_path)
        written_conversations += written
        logger.info(
            f"Wrote shard {shard_idx + 1}/{total_shards} ({written} conversations) -> {shard_path}"
        )

        # Publish intermediate progress. The final cursor is committed only
        # after the skip-rate guard accepts the complete conversion below.
        if shard_idx + 1 < total_shards:
            _write_metadata(
                output_dir,
                mixture_name,
                split,
                shard_idx + 1,
                total_shards,
                total_conversations,
                mixture_fingerprint,
                steered_rows,
                max_conversations=max_conversations,
                conversations_per_shard=conversations_per_shard,
                written_conversations=written_conversations,
            )

    if total_processed_all > 0:
        skip_rate = total_skipped_all / total_processed_all
        if skip_rate > 0.5:
            if existing_meta is not None and existing_meta.completed_shards == total_shards:
                # Older converters published completion before this guard.
                # Revoke that stale success only after every shard was audited.
                _write_metadata(
                    output_dir,
                    mixture_name,
                    split,
                    total_shards - 1,
                    total_shards,
                    total_conversations,
                    mixture_fingerprint,
                    steered_rows,
                    max_conversations=max_conversations,
                    conversations_per_shard=conversations_per_shard,
                    written_conversations=sum(
                        existing_rows[index] for index in range(total_shards - 1)
                    ),
                )
            raise DataLoadError(
                f"Skipped {total_skipped_all}/{total_processed_all} conversations "
                f"({skip_rate:.0%}). More than 50% of conversations were invalid — "
                f"the dataset may be corrupted or incompatible."
            )

    _write_metadata(
        output_dir,
        mixture_name,
        split,
        total_shards,
        total_shards,
        total_conversations,
        mixture_fingerprint,
        steered_rows,
        max_conversations=max_conversations,
        conversations_per_shard=conversations_per_shard,
        written_conversations=written_conversations,
    )
    logger.info(
        f"Conversion complete: {written_conversations} conversations written, "
        f"{total_skipped_all} skipped from {total_conversations} input conversations "
        f"in {total_shards} shards"
    )


def _canonical_source_digest(conversation: object) -> str:
    """Hash every semantic source field before prefix normalization."""
    try:
        encoded = json.dumps(
            conversation,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DataLoadError(f"structured source row is not canonical JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _aggregate_digests(digests: list[str]) -> str:
    """Bind ordered digest strings without ambiguous concatenation."""
    return hashlib.sha256("".join(f"{digest}\n" for digest in digests).encode()).hexdigest()


def _record_nonnegative_int(record: dict[str, object], field: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataLoadError(
            f"structured shard record {field} must be a non-negative integer"
        )
    return value


def _render_structured_source_row(
    tokenizer,
    conversation: object,
    *,
    original_index: int,
) -> tuple[StructuredRolloutRow | None, str | None, int, str]:
    """Normalize, render, and validate one source row with its source digest."""
    source_digest = _canonical_source_digest(conversation)
    try:
        normalized = normalize_complete_prefix(conversation)
    except StructuredRolloutError as exc:
        raise DataLoadError(
            f"malformed structured source row original_index={original_index}: {exc}"
        ) from exc
    if normalized.conversation is None:
        return None, normalized.skip_reason, 0, source_digest
    try:
        rendered = tokenizer.render_structured_conversation(normalized.conversation)
        row = structured_row_from_render(
            rendered,
            bos_token_id=tokenizer.get_bos_token_id(),
            decode=tokenizer.decode,
            original_index=original_index,
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        raise DataLoadError(
            f"structured render failed for original_index={original_index}: {exc}"
        ) from exc
    return row, None, rendered.literal_tag_collisions, source_digest


def _build_structured_shard(
    mixture: SFTConversationSource,
    tokenizer,
    *,
    conv_start: int,
    conv_end: int,
) -> tuple[list[StructuredRolloutRow], dict[str, object]]:
    """Render one exact source-index range and return its provenance record."""
    rows: list[StructuredRolloutRow] = []
    source_digests: list[str] = []
    row_digests: list[str] = []
    skip_counters: dict[str, int] = {}
    collision_count = 0
    for conv_idx in range(conv_start, conv_end):
        row, skip_reason, collisions, source_digest = _render_structured_source_row(
            tokenizer,
            mixture[conv_idx],
            original_index=conv_idx,
        )
        source_digests.append(source_digest)
        collision_count += collisions
        if row is None:
            reason = skip_reason or "unknown"
            skip_counters[reason] = skip_counters.get(reason, 0) + 1
            continue
        rows.append(row)
        row_digests.append(structured_row_digest(row))
    record: dict[str, object] = {
        "source_start": conv_start,
        "source_end": conv_end,
        "source_row_count": conv_end - conv_start,
        "written_row_count": len(rows),
        "source_content_sha256": _aggregate_digests(source_digests),
        "row_content_sha256": _aggregate_digests(row_digests),
        "literal_tag_collisions": collision_count,
        "skip_counters": dict(sorted(skip_counters.items())),
    }
    return rows, record


def _read_structured_metadata(output_dir: Path) -> dict[str, object] | None:
    meta_path = output_dir / "conversion_meta.json"
    if not meta_path.exists():
        return None
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataLoadError(f"cannot read structured conversion metadata {meta_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise DataLoadError(f"structured conversion metadata must be a JSON object: {meta_path}")
    return raw


def _validate_structured_resume_identity(
    existing: dict[str, object],
    *,
    output_dir: Path,
    mixture_name: str,
    split: SFTSplit,
    total_conversations: int,
    total_shards: int,
    conversations_per_shard: int,
    mixture_fingerprint: MixtureFingerprint,
    tokenizer_sha256: str,
) -> tuple[int, list[dict[str, object]]]:
    """Validate all immutable producer fields before reopening any shard."""
    expected = {
        "format": STRUCTURED_ROLLOUT_FORMAT,
        "format_version": 1,
        "schema_version": STRUCTURED_ROLLOUT_SCHEMA_VERSION,
        "renderer_version": STRUCTURED_RENDERER_VERSION,
        "mixture": mixture_name,
        "split": split.value,
        "total_conversations": total_conversations,
        "total_shards": total_shards,
        "conversations_per_shard": conversations_per_shard,
        MIXTURE_FINGERPRINT_KEY: mixture_fingerprint,
        "tokenizer_sha256": tokenizer_sha256,
    }
    for field, value in expected.items():
        if existing.get(field) != value:
            raise DataLoadError(
                f"structured conversion identity mismatch at {output_dir}: "
                f"{field} expected {value!r}, got {existing.get(field)!r}. "
                "Use a fresh output_dir."
            )
    completed = existing.get("completed_shards")
    shard_records = existing.get("shards")
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or not 0 <= completed <= total_shards
    ):
        raise DataLoadError("structured conversion completed_shards is invalid")
    if not isinstance(shard_records, list) or len(shard_records) != completed:
        raise DataLoadError(
            "structured conversion metadata must contain exactly one record per completed shard"
        )
    if not all(isinstance(record, dict) for record in shard_records):
        raise DataLoadError("structured conversion shard records must be JSON objects")
    return completed, shard_records


def _audit_structured_shard(
    mixture: SFTConversationSource,
    tokenizer,
    *,
    shard_path: Path,
    expected_record: dict[str, object],
    conv_start: int,
    conv_end: int,
) -> tuple[list[StructuredRolloutRow], dict[str, object]]:
    """Re-render every source row and compare full stored row content."""
    try:
        table = pq.read_table(shard_path)
        stored_rows = list(structured_table_to_rows(table, decode=tokenizer.decode))
    except (OSError, pa.ArrowException, StructuredRolloutError) as exc:
        raise DataLoadError(f"invalid structured rollout shard {shard_path}: {exc}") from exc
    expected_rows, actual_record = _build_structured_shard(
        mixture,
        tokenizer,
        conv_start=conv_start,
        conv_end=conv_end,
    )
    if stored_rows != expected_rows:
        first_mismatch = next(
            (
                index
                for index, (stored, expected) in enumerate(
                    zip(stored_rows, expected_rows, strict=False)
                )
                if stored != expected
            ),
            min(len(stored_rows), len(expected_rows)),
        )
        raise DataLoadError(
            f"structured resume audit found row drift in {shard_path} at row "
            f"{first_mismatch}; source tokens, mask, spans, text, or index changed"
        )
    recorded_content = {
        key: value for key, value in expected_record.items() if key != "shard_index"
    }
    if recorded_content != actual_record:
        raise DataLoadError(
            f"structured resume audit found provenance drift in {shard_path}: "
            f"recorded={recorded_content!r}, live={actual_record!r}"
        )
    return expected_rows, actual_record


def _structured_metadata(
    *,
    mixture_name: str,
    split: SFTSplit,
    total_conversations: int,
    total_shards: int,
    conversations_per_shard: int,
    mixture_fingerprint: MixtureFingerprint,
    tokenizer_sha256: str,
    shard_records: list[dict[str, object]],
) -> dict[str, object]:
    """Build one complete atomic metadata snapshot."""
    skip_counters: dict[str, int] = {}
    collisions = 0
    total_rows = 0
    source_digests: list[str] = []
    row_digests: list[str] = []
    for record in shard_records:
        total_rows += _record_nonnegative_int(record, "written_row_count")
        collisions += _record_nonnegative_int(record, "literal_tag_collisions")
        source_digests.append(str(record["source_content_sha256"]))
        row_digests.append(str(record["row_content_sha256"]))
        raw_skips = record["skip_counters"]
        assert isinstance(raw_skips, dict)
        for reason, count in raw_skips.items():
            skip_counters[str(reason)] = skip_counters.get(str(reason), 0) + int(count)
    return {
        "format": STRUCTURED_ROLLOUT_FORMAT,
        "format_version": 1,
        "schema_version": STRUCTURED_ROLLOUT_SCHEMA_VERSION,
        "renderer_version": STRUCTURED_RENDERER_VERSION,
        "mixture": mixture_name,
        "split": split.value,
        "completed_shards": len(shard_records),
        "total_shards": total_shards,
        "total_conversations": total_conversations,
        "total_rows": total_rows,
        "conversations_per_shard": conversations_per_shard,
        MIXTURE_FINGERPRINT_KEY: mixture_fingerprint,
        "tokenizer_sha256": tokenizer_sha256,
        "source_content_sha256": _aggregate_digests(source_digests),
        "row_content_sha256": _aggregate_digests(row_digests),
        "literal_tag_collisions": collisions,
        "skip_counters": dict(sorted(skip_counters.items())),
        "shards": shard_records,
    }


@beartype
def convert_mixture_to_structured_rollout(
    mixture: SFTConversationSource,
    output_dir: Path | str,
    max_conversations: PositiveIntLike | None = None,
    conversations_per_shard: PositiveIntLike = 40000,
    mixture_name: str = "unknown",
    split: SFTSplit = SFTSplit.TRAIN,
) -> None:
    """Write an atomic, resumable ``structured_nanochat_rollout_v1`` family."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = get_tokenizer()
    if not callable(getattr(tokenizer, "render_structured_conversation", None)):
        raise DataLoadError(
            "structured conversion requires RustBPETokenizer.render_structured_conversation"
        )
    try:
        tokenizer_sha256 = rustbpe_tokenizer_sha256(tokenizer)
    except StructuredRolloutError as exc:
        raise DataLoadError(str(exc)) from exc
    total_available = len(mixture)
    total_conversations = (
        min(total_available, max_conversations)
        if max_conversations is not None
        else total_available
    )
    total_shards = math.ceil(total_conversations / conversations_per_shard)
    mixture_fingerprint = compute_mixture_fingerprint(
        mixture,
        mixture_name=mixture_name,
        split=split,
    )
    existing = _read_structured_metadata(output_dir)
    if existing is None:
        start_shard = 0
        shard_records: list[dict[str, object]] = []
    else:
        start_shard, shard_records = _validate_structured_resume_identity(
            existing,
            output_dir=output_dir,
            mixture_name=mixture_name,
            split=split,
            total_conversations=total_conversations,
            total_shards=total_shards,
            conversations_per_shard=int(conversations_per_shard),
            mixture_fingerprint=mixture_fingerprint,
            tokenizer_sha256=tokenizer_sha256,
        )
        for shard_idx in range(start_shard):
            conv_start = shard_idx * conversations_per_shard
            conv_end = min(conv_start + conversations_per_shard, total_conversations)
            shard_path = output_dir / _shard_filename(shard_idx)
            if not shard_path.is_file():
                raise DataLoadError(
                    f"structured metadata records completed shard {shard_idx}, but "
                    f"{shard_path} is missing"
                )
            _audit_structured_shard(
                mixture,
                tokenizer,
                shard_path=shard_path,
                expected_record=shard_records[shard_idx],
                conv_start=conv_start,
                conv_end=conv_end,
            )
        logger.info(
            f"Resuming structured conversion after auditing {start_shard}/{total_shards} shards"
        )

    for shard_idx in range(start_shard, total_shards):
        conv_start = shard_idx * conversations_per_shard
        conv_end = min(conv_start + conversations_per_shard, total_conversations)
        rows, record = _build_structured_shard(
            mixture,
            tokenizer,
            conv_start=conv_start,
            conv_end=conv_end,
        )
        shard_path = output_dir / _shard_filename(shard_idx)
        if shard_path.exists():
            _audit_structured_shard(
                mixture,
                tokenizer,
                shard_path=shard_path,
                expected_record=record,
                conv_start=conv_start,
                conv_end=conv_end,
            )
        else:
            atomic_write_parquet(structured_rows_to_table(rows), shard_path)
        record = {"shard_index": shard_idx, **record}
        shard_records.append(record)
        atomic_json_dump(
            _structured_metadata(
                mixture_name=mixture_name,
                split=split,
                total_conversations=total_conversations,
                total_shards=total_shards,
                conversations_per_shard=int(conversations_per_shard),
                mixture_fingerprint=mixture_fingerprint,
                tokenizer_sha256=tokenizer_sha256,
                shard_records=shard_records,
            ),
            output_dir / "conversion_meta.json",
        )
        logger.info(
            f"Wrote structured shard {shard_idx + 1}/{total_shards} "
            f"({len(rows)} rows) -> {shard_path}"
        )

    final_metadata = _structured_metadata(
        mixture_name=mixture_name,
        split=split,
        total_conversations=total_conversations,
        total_shards=total_shards,
        conversations_per_shard=int(conversations_per_shard),
        mixture_fingerprint=mixture_fingerprint,
        tokenizer_sha256=tokenizer_sha256,
        shard_records=shard_records,
    )
    attempted = total_conversations
    final_skip_counters = final_metadata["skip_counters"]
    assert isinstance(final_skip_counters, dict)
    skipped = sum(int(count) for count in final_skip_counters.values())
    if attempted > 0 and skipped / attempted > 0.5:
        raise DataLoadError(
            f"Skipped {skipped}/{attempted} structured conversations; more than 50% "
            "of source rows contain no completed transaction"
        )
    atomic_json_dump(final_metadata, output_dir / "conversion_meta.json")
    logger.info(
        f"Structured conversion complete: {final_metadata['total_rows']} rows in "
        f"{total_shards} shards"
    )


@hydra.main(
    config_path="../../configs/data",
    config_name="convert_sft_to_parquet",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """CLI entry point for SFT-to-parquet conversion."""
    mixture_enum = SFTMixture(cfg.mixture)
    selection = SFTSplitSelection.parse(str(cfg.split))
    artifact_format = str(cfg.get("artifact_format", LEGACY_SFT_FORMAT))
    if artifact_format not in {LEGACY_SFT_FORMAT, STRUCTURED_ROLLOUT_FORMAT}:
        raise ValueError(
            f"artifact_format must be {LEGACY_SFT_FORMAT!r} or "
            f"{STRUCTURED_ROLLOUT_FORMAT!r}; got {artifact_format!r}"
        )

    # Determine output directory
    if cfg.output_dir:
        train_output_dir = Path(cfg.output_dir)
    else:
        base_dir = get_base_dir()
        if artifact_format == LEGACY_SFT_FORMAT:
            train_output_dir = Path(base_dir) / "sft_data" / cfg.mixture
        else:
            train_output_dir = (
                Path(base_dir)
                / "sft_data"
                / STRUCTURED_ROLLOUT_FORMAT
                / cfg.mixture
            )
    val_output_dir = sft_split_dir(train_output_dir, SFTSplit.VAL)
    converter = (
        convert_mixture_to_parquet
        if artifact_format == LEGACY_SFT_FORMAT
        else convert_mixture_to_structured_rollout
    )

    if mixture_enum in STEERED_MIXTURES and SFTSplitSelection.VAL in selection:
        raise ValueError(
            f"val family for {mixture_enum.value} is 'standard': "
            "run mixture=standard split=val"
        )

    logger.info(f"Building mixture: {cfg.mixture}")
    # One construction path shared with the trainer, so the steered spacing and
    # the original_index a shard records cannot drift apart.
    train_mixture, val_mixture = build_task_mixtures(
        mixture_enum,
        **steered_mixture_overrides(mixture_enum, cfg.steered_jsonl),
    )

    if SFTSplitSelection.TRAIN in selection:
        logger.info(
            f"Train mixture has {len(train_mixture)} conversations, writing to {train_output_dir}"
        )
        converter(
            mixture=train_mixture,
            output_dir=train_output_dir,
            max_conversations=cfg.max_conversations,
            conversations_per_shard=cfg.conversations_per_shard,
            mixture_name=cfg.mixture,
            split=SFTSplit.TRAIN,
        )

    if SFTSplitSelection.VAL in selection:
        if val_mixture is None:
            logger.info(
                f"Mixture {cfg.mixture!r} has an empty val task list; "
                f"no val conversion directory will be written at {val_output_dir}"
            )
            return
        logger.info(
            f"Val mixture has {len(val_mixture)} conversations, writing to {val_output_dir}"
        )
        converter(
            mixture=val_mixture,
            output_dir=val_output_dir,
            max_conversations=cfg.max_conversations,
            conversations_per_shard=cfg.conversations_per_shard,
            mixture_name=cfg.mixture,
            split=SFTSplit.VAL,
        )


if __name__ == "__main__":
    main()
