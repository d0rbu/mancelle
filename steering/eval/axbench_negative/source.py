"""Original-message sources for naturally elicited negative AxBench examples."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from steering._common.strict_json import reject_nonstandard_json_constant, strict_json_object
from steering.immutable_io import install_bytes_once, stream_hash_file_descriptor_bound

ULTRACHAT_DATASET = "HuggingFaceH4/ultrachat_200k"
ULTRACHAT_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
ULTRACHAT_TEST_FILE = "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet"
ULTRACHAT_TEST_SHA256 = "c18fd6e77395577652bdefbc5a87044ea799e989451ecfca5c6cf977ae5c6f70"
SOURCE_FORMAT = "axbench-negative-original-messages-v1"
SPLITS = frozenset({"train", "validation", "test"})
_RECORD_FIELDS = frozenset({"source_id", "group_id", "split", "source_row", "messages"})


class NegativeSourceError(ValueError):
    """An original-message source is ambiguous, changed, or malformed."""


def canonical_bytes(value: Any) -> bytes:
    """Canonical JSON without changing any string content."""
    return (
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        + "\n"
    ).encode("utf-8")


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return stream_hash_file_descriptor_bound(
        Path(path).resolve(strict=True),
        on_error=NegativeSourceError,
    ).sha256


def _nonblank(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NegativeSourceError(f"{label} must be a nonblank string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise NegativeSourceError(f"{label} must contain valid Unicode") from exc
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise NegativeSourceError(f"{label} must be a nonnegative integer")
    return value


def validate_messages(
    messages: object,
    *,
    ending_user: bool = False,
) -> tuple[dict[str, str], ...]:
    """Copy strict text messages; never repair roles, whitespace, or content."""
    if not isinstance(messages, (list, tuple)) or not messages:
        raise NegativeSourceError("messages must be a nonempty list of text messages")
    result: list[dict[str, str]] = []
    expected = "user"
    for index, raw in enumerate(messages):
        if not isinstance(raw, Mapping):
            raise NegativeSourceError(f"message {index} requires exactly role and content")
        message = cast(Mapping[object, object], raw)
        if set(message) != {"role", "content"}:
            raise NegativeSourceError(f"message {index} requires exactly role and content")
        role = message["role"]
        if not isinstance(role, str):
            raise NegativeSourceError(f"message {index} role must be a string")
        content = _nonblank(message["content"], f"message {index} content")
        if index == 0 and role == "system":
            result.append({"role": "system", "content": content})
            continue
        if role != expected:
            raise NegativeSourceError(f"message {index} role must be {expected!r}, got {role!r}")
        result.append({"role": role, "content": content})
        expected = "assistant" if role == "user" else "user"
    if result[-1]["role"] == "system":
        raise NegativeSourceError("a system message must be followed by a user message")
    if ending_user and result[-1]["role"] != "user":
        raise NegativeSourceError("a candidate must end immediately after a user message")
    return tuple(result)


@dataclass(frozen=True)
class FeatureTarget:
    """One original SAE feature or a canonical mask-allowed member union."""

    feature_id: int
    member_ids: tuple[int, ...]
    description: str

    def __post_init__(self) -> None:
        _nonnegative_int(self.feature_id, "feature_id")
        _nonblank(self.description, "description")
        if not isinstance(self.member_ids, tuple) or not self.member_ids:
            raise NegativeSourceError("member_ids must be a nonempty tuple")
        for member in self.member_ids:
            _nonnegative_int(member, "member_id")
        if tuple(sorted(set(self.member_ids))) != self.member_ids:
            raise NegativeSourceError("member_ids must be sorted and unique")
        if self.feature_id not in self.member_ids:
            raise NegativeSourceError(
                "member_ids must include the original representative feature_id"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "member_ids": list(self.member_ids),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FeatureTarget:
        if set(value) != {"feature_id", "member_ids", "description"}:
            raise NegativeSourceError(
                "feature target requires exactly feature_id/member_ids/description"
            )
        members = value["member_ids"]
        if not isinstance(members, (list, tuple)):
            raise NegativeSourceError("member_ids must be a list")
        return cls(value["feature_id"], tuple(members), value["description"])


@dataclass(frozen=True)
class SourceCandidate:
    """One exact user-ending cut; future source answers are never included."""

    source_id: str
    group_id: str
    split: str
    source_row: int
    cut_message_count: int
    messages: tuple[dict[str, str], ...]
    messages_sha256: str
    candidate_id: str

    def __post_init__(self) -> None:
        messages = validate_messages(self.messages, ending_user=True)
        object.__setattr__(self, "messages", messages)
        self.validate()

    def validate(self) -> None:
        _nonblank(self.source_id, "source_id")
        _nonblank(self.group_id, "group_id")
        _nonnegative_int(self.source_row, "source_row")
        if not isinstance(self.split, str) or self.split not in SPLITS:
            raise NegativeSourceError(f"split must be one of {sorted(SPLITS)}")
        if type(self.cut_message_count) is not int or self.cut_message_count != len(self.messages):
            raise NegativeSourceError(
                "cut_message_count must equal the number of original messages"
            )
        validate_messages(self.messages, ending_user=True)
        if value_sha256(list(self.messages)) != self.messages_sha256:
            raise NegativeSourceError(
                "candidate original messages differ from their recorded digest"
            )
        expected = _candidate_id(self.source_id, self.cut_message_count, self.messages_sha256)
        if self.candidate_id != expected:
            raise NegativeSourceError("candidate_id does not match the original source and cut")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "source_id": self.source_id,
            "group_id": self.group_id,
            "split": self.split,
            "source_row": self.source_row,
            "cut_message_count": self.cut_message_count,
            "messages": [dict(message) for message in self.messages],
            "messages_sha256": self.messages_sha256,
            "candidate_id": self.candidate_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SourceCandidate:
        expected = _RECORD_FIELDS | {"cut_message_count", "messages_sha256", "candidate_id"}
        if set(value) != expected:
            raise NegativeSourceError("candidate fields do not match the source-candidate schema")
        return cls(
            source_id=value["source_id"],
            group_id=value["group_id"],
            split=value["split"],
            source_row=value["source_row"],
            cut_message_count=value["cut_message_count"],
            messages=value["messages"],
            messages_sha256=value["messages_sha256"],
            candidate_id=value["candidate_id"],
        )


def _candidate_id(source_id: str, cut: int, messages_sha256: str) -> str:
    return value_sha256(
        {
            "format": SOURCE_FORMAT,
            "source_id": source_id,
            "cut_message_count": cut,
            "messages_sha256": messages_sha256,
        }
    )


def iter_assistant_prefixes(record: Mapping[str, Any]) -> Iterator[SourceCandidate]:
    """Enumerate legal cuts in source order without inspecting future content."""
    if set(record) != _RECORD_FIELDS:
        raise NegativeSourceError(f"source record requires exactly {sorted(_RECORD_FIELDS)}")
    source_id = _nonblank(record["source_id"], "source_id")
    group_id = _nonblank(record["group_id"], "group_id")
    source_row = _nonnegative_int(record["source_row"], "source_row")
    split = record["split"]
    if not isinstance(split, str) or split not in SPLITS:
        raise NegativeSourceError(f"split must be one of {sorted(SPLITS)}")
    messages = validate_messages(record["messages"])
    for index, message in enumerate(messages):
        if message["role"] == "user":
            prefix = messages[: index + 1]
            digest = value_sha256(list(prefix))
            yield SourceCandidate(
                source_id,
                group_id,
                split,
                source_row,
                index + 1,
                prefix,
                digest,
                _candidate_id(source_id, index + 1, digest),
            )


def load_source_candidates(
    path: Path,
    expected_sha256: str | None = None,
) -> list[SourceCandidate]:
    """Validate strict JSONL, grouping, exact-duplicate splits, and file identity."""
    path = Path(path).resolve(strict=True)
    receipt = stream_hash_file_descriptor_bound(
        path,
        retain_payload=True,
        on_error=NegativeSourceError,
    )
    if receipt.payload is None:
        raise NegativeSourceError("source bytes were not retained for strict parsing")
    if expected_sha256 is not None and receipt.sha256 != expected_sha256:
        raise NegativeSourceError(
            f"source SHA-256 mismatch: expected={expected_sha256}, observed={receipt.sha256}"
        )
    candidates: list[SourceCandidate] = []
    source_ids: set[str] = set()
    group_splits: dict[str, str] = {}
    text_splits: dict[str, str] = {}
    for line_number, line in enumerate(receipt.payload.splitlines(), 1):
        try:
            record = json.loads(
                line,
                object_pairs_hook=strict_json_object,
                parse_constant=reject_nonstandard_json_constant,
            )
            if not isinstance(record, dict):
                raise NegativeSourceError("source record must be a JSON object")
            rows = list(iter_assistant_prefixes(record))
            source_id = record["source_id"]
            if source_id in source_ids:
                raise NegativeSourceError(f"duplicate source_id {source_id!r}")
            source_ids.add(source_id)
            group = record["group_id"]
            split = record["split"]
            if group_splits.setdefault(group, split) != split:
                raise NegativeSourceError(f"source group {group!r} crosses splits")
            hashes = [value_sha256(record["messages"]), *(row.messages_sha256 for row in rows)]
            for text_hash in hashes:
                if text_splits.setdefault(text_hash, split) != split:
                    raise NegativeSourceError(
                        "an exact conversation or prefix duplicate crosses splits"
                    )
            candidates.extend(rows)
        except (UnicodeDecodeError, ValueError) as exc:
            raise NegativeSourceError(f"{path}:{line_number}: {exc}") from exc
    if not candidates:
        raise NegativeSourceError("source contains no eligible user-ending conversation prefixes")
    return candidates


def prepare_ultrachat_source(
    *,
    output_dir: Path,
    row_count: int,
    snapshot_dir: Path | None = None,
    revision: str = ULTRACHAT_REVISION,
    expected_parquet_sha256: str = ULTRACHAT_TEST_SHA256,
) -> dict[str, Any]:
    """Export first N original test rows, with no activity-dependent selection."""
    if type(row_count) is not int or row_count < 1:
        raise NegativeSourceError("row_count must be a positive integer")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise NegativeSourceError("revision must be a full immutable 40-character commit")
    if (
        not isinstance(expected_parquet_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_parquet_sha256) is None
    ):
        raise NegativeSourceError("expected_parquet_sha256 must be an explicit SHA-256")
    parquet_path = (
        Path(snapshot_dir) / ULTRACHAT_TEST_FILE
        if snapshot_dir is not None
        else Path(
            hf_hub_download(
                ULTRACHAT_DATASET, ULTRACHAT_TEST_FILE, repo_type="dataset", revision=revision
            )
        )
    )
    before_sha = file_sha256(parquet_path)
    if before_sha != expected_parquet_sha256:
        raise NegativeSourceError("pinned UltraChat test parquet SHA-256 mismatch")
    parquet = pq.ParquetFile(parquet_path)
    if row_count > parquet.metadata.num_rows:
        raise NegativeSourceError(
            f"row_count={row_count} exceeds test_sft rows={parquet.metadata.num_rows}"
        )
    if not {"messages", "prompt_id"}.issubset(parquet.schema_arrow.names):
        raise NegativeSourceError("UltraChat parquet lacks original messages or prompt_id")
    namespace = f"{ULTRACHAT_DATASET}@{revision}/test_sft"
    records: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for batch in parquet.iter_batches(
        batch_size=min(row_count, 256), columns=["messages", "prompt_id"]
    ):
        for raw in batch.to_pylist():
            index = len(selected)
            if index == row_count:
                break
            prompt_id = _nonblank(raw["prompt_id"], "UltraChat prompt_id")
            source_id = f"{namespace}/row/{index}"
            group_id = value_sha256({"source": namespace, "prompt_id": prompt_id})
            selected.append(
                {
                    "source_row": index,
                    "source_id": source_id,
                    "prompt_id": prompt_id,
                    "group_id": group_id,
                }
            )
            try:
                messages = validate_messages(raw["messages"])
            except NegativeSourceError as exc:
                rejections.append({"source_row": index, "reason": str(exc)})
                continue
            records.append(
                {
                    "source_id": source_id,
                    "group_id": group_id,
                    "split": "test",
                    "source_row": index,
                    "messages": list(messages),
                }
            )
        if len(selected) == row_count:
            break
    if len(selected) != row_count:
        raise NegativeSourceError("UltraChat reader did not produce the declared first N rows")
    if file_sha256(parquet_path) != before_sha:
        raise NegativeSourceError("UltraChat parquet changed during preparation")
    payload = b"".join(canonical_bytes(record) for record in records)
    manifest = {
        "format": SOURCE_FORMAT,
        "complete": True,
        "dataset": ULTRACHAT_DATASET,
        "revision": revision,
        "dataset_split": "test_sft",
        "parquet_file": ULTRACHAT_TEST_FILE,
        "parquet_sha256": before_sha,
        "selection": {"policy": "first_n_original_rows_v1", "start": 0, "stop": row_count},
        "selected_rows": selected,
        "rejected_rows": rejections,
        "source": {
            "path": "source.jsonl",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rows": len(records),
        },
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "source.jsonl"
    install_bytes_once(payload, source_path, surface="negative AxBench original messages")
    load_source_candidates(source_path, manifest["source"]["sha256"])
    install_bytes_once(
        canonical_bytes(manifest),
        output_dir / "source_manifest.json",
        surface="negative AxBench original-message manifest",
    )
    return manifest
