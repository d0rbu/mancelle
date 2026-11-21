"""Immutable question-only data and native rewards for encoder-decoder RLVR.

Preparation owns Hugging Face access.  Training consumes only the JSONL bundle
validated here and an already-open schema-5 context-feature provider.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from errors import ParseError
from steering.eval.chat_generation import render_chat_completion
from steering.eval.context_cache_io import CacheIdentity, SafetensorsContextFeatureProvider
from steering.eval.target_conditioning import token_context_key
from steering.eval.turn_context_contract import (
    TurnContextContract,
    canonical_turn_payload,
    turn_context_contract_hash,
    turn_context_identity_fields,
)
from steering.segmentation import SegmentMode, compute_tokenizer_checksum
from tasks.gsm8k import extract_answer

RLVR_ROW_SCHEMA_VERSION = 1
RLVR_BUNDLE_FORMAT_VERSION = 1
RLVR_GRADER_CONTRACT = "native_task_exact_v1"
RLVR_RENDER_CONTRACT = "chat_generation.render_chat_completion_v1"
RLVR_CACHE_ROLE = "chat_rlvr"

RLVR_DATASET_REVISIONS = {
    "arc": "210d026faf9955653af8916fad021475a3f00453",
    "mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",
    "gsm8k": "740312add88f781978c0658806c59bc2815b9866",
}

RLVR_TASKS = ("ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K")
RLVR_MAX_NEW_TOKENS = {
    "ARC-Easy": 16,
    "ARC-Challenge": 16,
    "MMLU": 16,
    "GSM8K": 512,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTEXT_KEY_RE = re.compile(r"^tokens_sha256:[0-9a-f]{64}$")
_ROW_ID_RE = re.compile(r"^rlvr:[0-9a-f]{64}$")
_ROW_FIELDS = {
    "schema_version",
    "split",
    "task",
    "source",
    "subset",
    "source_split",
    "source_index",
    "revision",
    "row_id",
    "prompt_token_ids",
    "context_key",
    "oracle",
}
_BUNDLE_FIELDS = {
    "format_version",
    "bundle_sha256",
    "files",
    "dataset_revisions",
    "tokenizers",
    "render_contract",
    "grader_contract",
    "context_cache",
    "base_checkpoint",
    "statistics",
    "preparation",
}


class RLVRDataError(ValueError):
    """A prepared RLVR artifact is malformed or incompatible."""


@dataclass(frozen=True)
class RLVRTaskSource:
    """One pinned task object plus its stable source identity."""

    task: str
    source: str
    subset: str
    source_split: str
    revision: str
    task_object: Any
    source_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.task not in RLVR_TASKS:
            raise RLVRDataError(f"unsupported RLVR task {self.task!r}")
        if not self.source or not self.subset or not self.source_split:
            raise RLVRDataError("RLVR source fields must be nonempty")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise RLVRDataError("RLVR dataset revision must be an immutable 40-hex commit SHA")
        if (
            len(self.source_indices) != len(self.task_object)
            or any(type(index) is not int or index < 0 for index in self.source_indices)
            or len(set(self.source_indices)) != len(self.source_indices)
        ):
            raise RLVRDataError(
                "RLVR source_indices must be unique nonnegative original row indices "
                "aligned with the filtered task"
            )


def _canonical(value: Mapping[str, Any]) -> bytes:
    return canonical_turn_payload(value)


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RLVRDataError(f"{field} must be 64 lowercase hexadecimal characters")
    return value


def _validate_oracle(task: str, value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RLVRDataError("oracle must be an object")
    if task == "GSM8K":
        if set(value) != {"kind", "answer"} or value.get("kind") != "gsm8k":
            raise RLVRDataError("GSM8K oracle must contain exactly kind='gsm8k' and answer")
        answer = value.get("answer")
        if not isinstance(answer, str) or not answer:
            raise RLVRDataError("GSM8K oracle answer must be a nonempty string")
        return {"kind": "gsm8k", "answer": answer}

    if set(value) != {"kind", "answer", "labels"} or value.get("kind") != "multiple_choice":
        raise RLVRDataError(
            "multiple-choice oracle must contain exactly kind='multiple_choice', answer, labels"
        )
    answer, labels = value.get("answer"), value.get("labels")
    if (
        not isinstance(answer, str)
        or not isinstance(labels, list)
        or not 3 <= len(labels) <= 5
        or any(not isinstance(label, str) or not label for label in labels)
        or len(set(labels)) != len(labels)
        or answer not in labels
    ):
        raise RLVRDataError("multiple-choice oracle requires 3-5 unique labels and an exact answer")
    return {"kind": "multiple_choice", "answer": answer, "labels": list(labels)}


@dataclass(frozen=True)
class RLVRRow:
    """One exact question prompt and its non-model-visible deterministic oracle."""

    split: str
    task: str
    source: str
    subset: str
    source_split: str
    source_index: int
    revision: str
    row_id: str
    prompt_token_ids: tuple[int, ...]
    context_key: str
    oracle: dict[str, Any]

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation"}:
            raise RLVRDataError("RLVR split must be train or validation")
        if self.task not in RLVR_TASKS:
            raise RLVRDataError(f"unsupported RLVR task {self.task!r}")
        if not all(
            isinstance(value, str) and value
            for value in (
                self.source,
                self.subset,
                self.source_split,
            )
        ):
            raise RLVRDataError("RLVR source identity fields must be nonempty strings")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise RLVRDataError("RLVR revision must be a 40-character lowercase commit SHA")
        if type(self.source_index) is not int or self.source_index < 0:
            raise RLVRDataError("RLVR source_index must be a nonnegative integer")
        if not _ROW_ID_RE.fullmatch(self.row_id):
            raise RLVRDataError("RLVR row_id must be rlvr:<64 lowercase hex>")
        if not self.prompt_token_ids or any(
            type(token_id) is not int or token_id < 0 for token_id in self.prompt_token_ids
        ):
            raise RLVRDataError("prompt_token_ids must be a nonempty tuple of nonnegative ints")
        if not _CONTEXT_KEY_RE.fullmatch(self.context_key):
            raise RLVRDataError("context_key must be tokens_sha256:<64 lowercase hex>")
        if token_context_key(self.prompt_token_ids) != self.context_key:
            raise RLVRDataError("context_key disagrees with exact prompt_token_ids")
        normalized_oracle = _validate_oracle(self.task, self.oracle)
        if normalized_oracle != self.oracle:
            raise RLVRDataError("oracle is not in canonical form")
        expected_id = rlvr_row_id(
            split=self.split,
            task=self.task,
            source=self.source,
            subset=self.subset,
            source_split=self.source_split,
            source_index=self.source_index,
            revision=self.revision,
            prompt_token_ids=self.prompt_token_ids,
        )
        if self.row_id != expected_id:
            raise RLVRDataError("row_id disagrees with the exact prompt/source identity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RLVR_ROW_SCHEMA_VERSION,
            "split": self.split,
            "task": self.task,
            "source": self.source,
            "subset": self.subset,
            "source_split": self.source_split,
            "source_index": self.source_index,
            "revision": self.revision,
            "row_id": self.row_id,
            "prompt_token_ids": list(self.prompt_token_ids),
            "context_key": self.context_key,
            "oracle": self.oracle,
        }

    @classmethod
    def from_dict(cls, value: object) -> RLVRRow:
        if not isinstance(value, dict):
            raise RLVRDataError("RLVR row must be an object")
        row = cast(dict[str, Any], value)
        if set(row) != _ROW_FIELDS:
            got = set(row)
            raise RLVRDataError(
                f"RLVR row fields differ: missing={sorted(_ROW_FIELDS - got)}, "
                f"unexpected={sorted(got - _ROW_FIELDS)}"
            )
        if type(row["schema_version"]) is not int or row["schema_version"] != (
            RLVR_ROW_SCHEMA_VERSION
        ):
            raise RLVRDataError("unsupported RLVR row schema_version")
        token_ids = row["prompt_token_ids"]
        if not isinstance(token_ids, list) or any(
            type(token_id) is not int for token_id in token_ids
        ):
            raise RLVRDataError("prompt_token_ids must be a list of integers")
        return cls(
            split=cast(str, row["split"]),
            task=cast(str, row["task"]),
            source=cast(str, row["source"]),
            subset=cast(str, row["subset"]),
            source_split=cast(str, row["source_split"]),
            source_index=cast(int, row["source_index"]),
            revision=cast(str, row["revision"]),
            row_id=cast(str, row["row_id"]),
            prompt_token_ids=tuple(cast(list[int], token_ids)),
            context_key=cast(str, row["context_key"]),
            oracle=_validate_oracle(cast(str, row["task"]), row["oracle"]),
        )


def rlvr_row_id(
    *,
    split: str,
    task: str,
    source: str,
    subset: str,
    source_split: str,
    source_index: int,
    revision: str,
    prompt_token_ids: Sequence[int],
) -> str:
    """Hash question/source identity only, so oracle changes cannot alter rollout RNG."""

    payload = {
        "version": RLVR_ROW_SCHEMA_VERSION,
        "split": split,
        "task": task,
        "source": source,
        "subset": subset,
        "source_split": source_split,
        "source_index": source_index,
        "revision": revision,
        "prompt_token_ids": [int(token_id) for token_id in prompt_token_ids],
    }
    return "rlvr:" + hashlib.sha256(_canonical(payload)).hexdigest()


def _oracle_from_source(source: RLVRTaskSource, index: int, conversation: dict) -> dict[str, Any]:
    if source.task == "GSM8K":
        reference = source.task_object.ds[index]["answer"]
        try:
            answer = extract_answer(reference)
        except ParseError as exc:
            raise RLVRDataError(
                f"{source.task}/{source.source_split} row {index} has an unparseable reference"
            ) from exc
        return {"kind": "gsm8k", "answer": answer}

    labels = list(conversation.get("letters", ()))
    messages = conversation.get("messages", ())
    if not messages or messages[-1].get("role") != "assistant":
        raise RLVRDataError(f"{source.task}/{source.source_split} row {index} lacks a reference")
    return _validate_oracle(
        source.task,
        {
            "kind": "multiple_choice",
            "answer": messages[-1].get("content"),
            "labels": labels,
        },
    )


def prepare_rlvr_rows(
    sources: Sequence[RLVRTaskSource],
    *,
    split: str,
    tokenizer: Any,
    model_max_seq_len: int = 2048,
    caps: Mapping[str, int] | None = None,
) -> list[RLVRRow]:
    """Render source tasks in order after global leakage filtering and optional caps."""

    if split not in {"train", "validation"}:
        raise RLVRDataError("split must be train or validation")
    if [source.task for source in sources] != list(RLVR_TASKS):
        raise RLVRDataError(f"RLVR sources must be ordered exactly as {list(RLVR_TASKS)}")
    cap_map = dict(caps or {})
    if set(cap_map) - set(RLVR_TASKS):
        raise RLVRDataError(f"unknown qualification caps: {sorted(set(cap_map) - set(RLVR_TASKS))}")
    if any(type(cap) is not int or cap <= 0 for cap in cap_map.values()):
        raise RLVRDataError("qualification caps must be positive integers")

    rows: list[RLVRRow] = []
    seen_ids: set[str] = set()
    for source in sources:
        limit = min(len(source.task_object), cap_map.get(source.task, len(source.task_object)))
        # Qualification caps choose persisted rows only after every retained
        # source row passes rendering, reference, and token-budget admission.
        for index in range(len(source.task_object)):
            conversation = source.task_object[index]
            source_index = source.source_indices[index]
            rendered = render_chat_completion(tokenizer, conversation)
            prompt_ids = tuple(rendered.token_ids)
            answer_budget = RLVR_MAX_NEW_TOKENS[source.task]
            if len(prompt_ids) + answer_budget > model_max_seq_len:
                raise RLVRDataError(
                    f"{source.task}/{source.source_split} row {index} prompt length "
                    f"{len(prompt_ids)} plus answer budget {answer_budget} exceeds "
                    f"model_max_seq_len={model_max_seq_len}; no truncation or resampling is allowed"
                )

            messages = conversation.get("messages", ())
            if not messages:
                raise RLVRDataError(
                    f"{source.task}/{source.source_split} row {index} has no messages"
                )
            perturbed = {
                **conversation,
                "messages": [
                    *messages[:-1],
                    {"role": "assistant", "content": "REFERENCE_MUST_NOT_AFFECT_PROMPT"},
                ],
            }
            if tuple(render_chat_completion(tokenizer, perturbed).token_ids) != prompt_ids:
                raise RLVRDataError("completion rendering leaked the reference assistant answer")

            oracle = _oracle_from_source(source, index, conversation)
            row_id = rlvr_row_id(
                split=split,
                task=source.task,
                source=source.source,
                subset=source.subset,
                source_split=source.source_split,
                source_index=source_index,
                revision=source.revision,
                prompt_token_ids=prompt_ids,
            )
            if row_id in seen_ids:
                raise RLVRDataError(
                    f"duplicate reward-identical row_id {row_id}; retain question duplicates only when "
                    "their source or reward identity differs"
                )
            seen_ids.add(row_id)
            row = RLVRRow(
                split=split,
                task=source.task,
                source=source.source,
                subset=source.subset,
                source_split=source.source_split,
                source_index=source_index,
                revision=source.revision,
                row_id=row_id,
                prompt_token_ids=prompt_ids,
                context_key=token_context_key(prompt_ids),
                oracle=oracle,
            )
            if index < limit:
                rows.append(row)
    return rows


@dataclass(frozen=True)
class RLVRGradeResult:
    """Native parsing outcome exposed for metrics as well as the scalar reward."""

    reward: float
    valid_format: bool
    parsed_answer: str | None


def grade_completion_result(row: RLVRRow, completion: str) -> RLVRGradeResult:
    """Apply the repository task's exact, deliberately narrow answer semantics."""

    if not isinstance(completion, str):
        raise TypeError(f"RLVR completion must be str, got {type(completion).__name__}")
    if row.oracle["kind"] == "gsm8k":
        try:
            parsed = extract_answer(completion)
        except ParseError:
            return RLVRGradeResult(0.0, False, None)
        return RLVRGradeResult(float(parsed == row.oracle["answer"]), True, parsed)

    parsed = completion.strip()
    labels = row.oracle["labels"]
    if parsed not in labels:
        return RLVRGradeResult(0.0, False, parsed)
    return RLVRGradeResult(float(parsed == row.oracle["answer"]), True, parsed)


def jsonl_bytes(rows: Sequence[RLVRRow]) -> bytes:
    return b"".join(_canonical(row.to_dict()) + b"\n" for row in rows)


def load_rlvr_jsonl(
    path: str | Path,
    *,
    expected_split: str | None = None,
    payload: bytes | None = None,
) -> tuple[RLVRRow, ...]:
    """Strictly load one immutable prepared split; blank lines are malformed rows."""

    rows: list[RLVRRow] = []
    seen: set[str] = set()
    if payload is None:
        payload = Path(path).read_bytes()
    with io.TextIOWrapper(io.BytesIO(payload), encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n") or not line.strip():
                raise RLVRDataError(
                    f"{path}:{line_number}: malformed blank or unterminated JSONL row"
                )
            try:
                value = json.loads(line)
                row = RLVRRow.from_dict(value)
            except (json.JSONDecodeError, RLVRDataError, TypeError) as exc:
                raise RLVRDataError(f"{path}:{line_number}: {exc}") from exc
            if expected_split is not None and row.split != expected_split:
                raise RLVRDataError(
                    f"{path}:{line_number}: row split={row.split!r}, expected {expected_split!r}"
                )
            if row.row_id in seen:
                raise RLVRDataError(f"{path}:{line_number}: duplicate row_id {row.row_id}")
            seen.add(row.row_id)
            rows.append(row)
    if not rows:
        raise RLVRDataError(f"{path}: RLVR split is empty")
    return tuple(rows)


@dataclass(frozen=True)
class RLVRBundle:
    path: Path
    train: tuple[RLVRRow, ...]
    validation: tuple[RLVRRow, ...]
    manifest: dict[str, Any]

    @property
    def rows(self) -> tuple[RLVRRow, ...]:
        return self.train + self.validation

    @property
    def cache_dir(self) -> Path:
        return self.path / self.manifest["context_cache"]["path"]

    @property
    def identity(self) -> CacheIdentity:
        return CacheIdentity.from_dict(self.manifest["context_cache"]["identity"])

    @property
    def identity_sha256(self) -> str:
        return self.manifest["bundle_sha256"]

    def open_prompt_cache(
        self,
        *,
        expected_turn_context_contract: TurnContextContract,
    ) -> SafetensorsContextFeatureProvider:
        """Open the exact existing cache implementation under bundle authority."""

        provider = SafetensorsContextFeatureProvider(
            self.cache_dir,
            identity=self.identity,
            expected_source_segment_mode=SegmentMode.turn_delimited_fixed_tokens,
            expected_turn_contract=expected_turn_context_contract,
        )
        return admit_rlvr_prompt_cache(self, provider)


def bundle_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("bundle_sha256", None)
    return hashlib.sha256(_canonical(payload)).hexdigest()


def ordered_row_ids_sha256(row_ids: Sequence[str]) -> str:
    """Hash the ordered row population using the bundle's exact JSON encoding."""

    return hashlib.sha256(json.dumps(list(row_ids), separators=(",", ":")).encode()).hexdigest()


def load_rlvr_bundle(
    path: str | Path,
    *,
    tokenizer: Any,
    expected_base_model_sha256: str,
    expected_turn_context_contract: TurnContextContract,
) -> RLVRBundle:
    """Load JSONL only and fail closed on tokenizer/base/turn-contract drift."""

    root = Path(path)
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RLVRDataError(f"cannot read RLVR manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise RLVRDataError("RLVR manifest must be an object")
    manifest = cast(dict[str, Any], manifest)
    if set(manifest) != _BUNDLE_FIELDS:
        got = set(manifest)
        raise RLVRDataError(
            f"RLVR manifest fields differ: missing={sorted(_BUNDLE_FIELDS - got)}, "
            f"unexpected={sorted(got - _BUNDLE_FIELDS)}"
        )
    if manifest.get("format_version") != RLVR_BUNDLE_FORMAT_VERSION:
        raise RLVRDataError("unsupported RLVR bundle format_version")
    if _require_sha256(manifest.get("bundle_sha256"), "bundle_sha256") != (
        bundle_manifest_sha256(manifest)
    ):
        raise RLVRDataError("RLVR bundle_sha256 disagrees with canonical manifest")
    if manifest.get("dataset_revisions") != RLVR_DATASET_REVISIONS:
        raise RLVRDataError("RLVR dataset revision pins disagree with the code contract")
    if manifest.get("render_contract") != RLVR_RENDER_CONTRACT:
        raise RLVRDataError("RLVR render contract disagrees")
    if manifest.get("grader_contract") != RLVR_GRADER_CONTRACT:
        raise RLVRDataError("RLVR grader contract disagrees")
    tokenizers = manifest.get("tokenizers")
    if not isinstance(tokenizers, dict) or tokenizers.get("decoder_sha256") != (
        compute_tokenizer_checksum(tokenizer)
    ):
        raise RLVRDataError("RLVR decoder tokenizer checksum disagrees")
    base = manifest.get("base_checkpoint")
    if not isinstance(base, dict) or base.get("model_sha256") != _require_sha256(
        expected_base_model_sha256, "expected_base_model_sha256"
    ):
        raise RLVRDataError("RLVR immutable base checkpoint authority disagrees")
    expected_meta_sha256 = expected_turn_context_contract.provenance.get(
        "checkpoint_metadata_sha256"
    )
    if (
        not isinstance(expected_meta_sha256, str)
        or base.get("metadata_sha256") != expected_meta_sha256
    ):
        raise RLVRDataError("RLVR immutable base metadata authority disagrees")
    statistics = manifest.get("statistics")
    if (
        not isinstance(statistics, dict)
        or statistics.get("sha256") != expected_turn_context_contract.lineage.statistics_sha256
    ):
        raise RLVRDataError("RLVR source statistics authority disagrees")
    if tokenizers.get("extraction_sha256") != expected_turn_context_contract.semantic.get(
        "extraction_tokenizer_checksum"
    ):
        raise RLVRDataError("RLVR extraction tokenizer checksum disagrees")
    cache = manifest.get("context_cache")
    expected_identity = {
        **turn_context_identity_fields(expected_turn_context_contract),
        "feature_selection_hash": turn_context_contract_hash(expected_turn_context_contract),
    }
    if (
        not isinstance(cache, dict)
        or cache.get("role") != RLVR_CACHE_ROLE
        or cache.get("schema_version") != 5
        or cache.get("segment_mode") != SegmentMode.turn_delimited_fixed_tokens.value
        or cache.get("turn_context_contract_sha256")
        != turn_context_contract_hash(expected_turn_context_contract)
        or cache.get("identity") != expected_identity
        or not isinstance(cache.get("path"), str)
        or Path(cache["path"]).is_absolute()
        or ".." in Path(cache["path"]).parts
        or set(cache)
        != {
            "role",
            "schema_version",
            "segment_mode",
            "turn_context_contract_sha256",
            "identity",
            "path",
            "ordered_row_ids_sha256",
            "data_sha256",
            "manifest_sha256",
            "shards",
        }
        or _SHA256_RE.fullmatch(str(cache.get("data_sha256"))) is None
        or _SHA256_RE.fullmatch(str(cache.get("manifest_sha256"))) is None
        or not isinstance(cache.get("shards"), list)
        or not cache["shards"]
    ):
        raise RLVRDataError("RLVR prompt-cache contract disagrees")

    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {"train", "validation", "contexts", "parquet"}:
        raise RLVRDataError("RLVR manifest files must contain train, validation, contexts, parquet")
    loaded: dict[str, tuple[RLVRRow, ...]] = {}
    for split in ("train", "validation"):
        info = files[split]
        if not isinstance(info, dict) or set(info) != {"path", "sha256", "rows"}:
            raise RLVRDataError(f"RLVR files.{split} is malformed")
        relative = info["path"]
        if not isinstance(relative, str) or Path(relative).name != relative:
            raise RLVRDataError(f"RLVR files.{split}.path must be a basename")
        file_path = root / relative
        payload = file_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != _require_sha256(
            info["sha256"], f"files.{split}.sha256"
        ):
            raise RLVRDataError(f"RLVR {split} JSONL digest disagrees")
        rows = load_rlvr_jsonl(file_path, expected_split=split, payload=payload)
        if type(info["rows"]) is not int or info["rows"] != len(rows):
            raise RLVRDataError(f"RLVR {split} row count disagrees")
        loaded[split] = rows
    all_ids = [row.row_id for rows in loaded.values() for row in rows]
    if len(set(all_ids)) != len(all_ids):
        raise RLVRDataError("RLVR row IDs collide across train and validation")
    ordered_ids_sha256 = ordered_row_ids_sha256(all_ids)
    if cache["ordered_row_ids_sha256"] != ordered_ids_sha256:
        raise RLVRDataError("RLVR cache population identity disagrees with ordered rows")
    distinct_prompts = len({row.prompt_token_ids for rows in loaded.values() for row in rows})
    for name in ("contexts", "parquet"):
        info = files[name]
        if not isinstance(info, dict) or set(info) != {"path", "sha256", "rows"}:
            raise RLVRDataError(f"RLVR files.{name} is malformed")
        relative = info["path"]
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise RLVRDataError(f"RLVR files.{name}.path must be relative and contained")
        artifact = root / relative
        try:
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        except OSError as exc:
            raise RLVRDataError(f"cannot read RLVR {name} artifact: {exc}") from exc
        if digest != _require_sha256(info["sha256"], f"files.{name}.sha256"):
            raise RLVRDataError(f"RLVR {name} artifact digest disagrees")
        if type(info["rows"]) is not int or info["rows"] != distinct_prompts:
            raise RLVRDataError(f"RLVR {name} artifact row count disagrees")
    return RLVRBundle(root, loaded["train"], loaded["validation"], manifest)


def admit_rlvr_prompt_cache(
    bundle: RLVRBundle,
    provider: SafetensorsContextFeatureProvider,
) -> SafetensorsContextFeatureProvider:
    """Validate bundle-specific authority around the existing provider and return it."""

    manifest = provider.manifest
    cache_contract = bundle.manifest["context_cache"]
    turn_contract = manifest.get("turn_contract")
    try:
        manifest_sha256 = hashlib.sha256(provider.manifest_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RLVRDataError(f"cannot authenticate RLVR cache manifest: {exc}") from exc
    expected_shards = [
        {"file": shard["file"], "sha256": shard["sha256"]} for shard in manifest.get("shards", ())
    ]
    base = bundle.manifest["base_checkpoint"]
    if (
        manifest.get("schema_version") != 5
        or manifest.get("source_segment_mode") != SegmentMode.turn_delimited_fixed_tokens.value
        or not isinstance(turn_contract, dict)
        or hashlib.sha256(_canonical(turn_contract)).hexdigest()
        != cache_contract["turn_context_contract_sha256"]
        or manifest_sha256 != cache_contract["manifest_sha256"]
        or expected_shards != cache_contract["shards"]
        or manifest.get("metadata", {}).get("rlvr_data_sha256") != cache_contract["data_sha256"]
        or manifest.get("metadata", {}).get("rlvr_base_model_sha256") != base["model_sha256"]
        or manifest.get("metadata", {}).get("rlvr_base_metadata_sha256") != base["metadata_sha256"]
    ):
        raise RLVRDataError("RLVR provider is not the bundle's schema-5 turn cache")
    unique_prompts = {row.prompt_token_ids for row in bundle.rows}
    if len(provider) != len(unique_prompts):
        raise RLVRDataError(
            "RLVR prompt cache must contain exactly the bundle's distinct prompt population"
        )
    missing = [row.row_id for row in bundle.rows if provider.get(row.prompt_token_ids) is None]
    if missing:
        raise RLVRDataError(f"RLVR prompt cache is missing prepared row {missing[0]}")
    return provider
