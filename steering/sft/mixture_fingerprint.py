"""Canonical TaskMixture identity for conditioned-SFT artifact alignment.

The load-bearing field is ``index_map_sha256``.  It hashes UTF-8 bytes of the
``TaskMixture.index_map`` encoded as compact JSON (a list of two-integer lists,
with ``separators=(",", ":")``).  This is equivalent in purpose to hashing
``repr(index_map)`` but makes the byte contract explicit and language-neutral.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict

from steering._common.strict_json import string_keyed_mapping
from steering.feature_io import read_shard_header
from steering.sft_schema import SFTSplit

MIXTURE_FINGERPRINT_KEY = "mixture_fingerprint"
SFT_MIXTURE_FINGERPRINT_METADATA_KEY = "sft_mixture_fingerprint"

_FINGERPRINT_FIELDS = frozenset(
    {"mixture", "split", "tasks", "len_mixture", "index_map_sha256"}
)
_TASK_FIELDS = frozenset({"class_name", "length"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

class TaskFingerprint(TypedDict):
    class_name: str
    length: int


class MixtureFingerprint(TypedDict):
    mixture: str
    split: str
    tasks: list[TaskFingerprint]
    len_mixture: int
    index_map_sha256: str


class MixtureFingerprintError(ValueError):
    """A serialized mixture fingerprint is malformed or internally inconsistent."""


def _strict_nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MixtureFingerprintError(
            f"{field} must be a non-negative integer, got {value!r}"
        )
    return value


def normalize_mixture_fingerprint(
    value: object,
    *,
    source: str = "mixture fingerprint",
) -> MixtureFingerprint:
    """Validate an artifact fingerprint and return its canonical Python form."""
    if not isinstance(value, Mapping):
        raise MixtureFingerprintError(
            f"{source} must be an object, got {type(value).__name__}"
        )
    normalized_value = string_keyed_mapping(value)
    if normalized_value is None:
        raise MixtureFingerprintError(f"{source} must contain only string keys")
    value = normalized_value
    fields = set(value)
    if fields != _FINGERPRINT_FIELDS:
        raise MixtureFingerprintError(
            f"{source} fields mismatch: missing={sorted(_FINGERPRINT_FIELDS - fields)}, "
            f"extra={sorted(fields - _FINGERPRINT_FIELDS)}"
        )

    mixture = value["mixture"]
    split = value["split"]
    if not isinstance(mixture, str) or not mixture:
        raise MixtureFingerprintError(
            f"{source}.mixture must be a non-empty string, got {mixture!r}"
        )
    if not isinstance(split, str) or not split:
        raise MixtureFingerprintError(
            f"{source}.split must be a non-empty string, got {split!r}"
        )

    tasks_raw = value["tasks"]
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise MixtureFingerprintError(
            f"{source}.tasks must be a non-empty list, got {tasks_raw!r}"
        )
    tasks: list[TaskFingerprint] = []
    for index, raw_task in enumerate(tasks_raw):
        if not isinstance(raw_task, Mapping):
            raise MixtureFingerprintError(
                f"{source}.tasks[{index}] must be an object, got "
                f"{type(raw_task).__name__}"
            )
        normalized_task = string_keyed_mapping(raw_task)
        if normalized_task is None:
            raise MixtureFingerprintError(
                f"{source}.tasks[{index}] must contain only string keys"
            )
        task_fields = set(normalized_task)
        if task_fields != _TASK_FIELDS:
            raise MixtureFingerprintError(
                f"{source}.tasks[{index}] fields mismatch: "
                f"missing={sorted(_TASK_FIELDS - task_fields)}, "
                f"extra={sorted(task_fields - _TASK_FIELDS)}"
            )
        class_name = normalized_task["class_name"]
        if not isinstance(class_name, str) or not class_name:
            raise MixtureFingerprintError(
                f"{source}.tasks[{index}].class_name must be a non-empty string, "
                f"got {class_name!r}"
            )
        length = _strict_nonnegative_int(
            normalized_task["length"],
            field=f"{source}.tasks[{index}].length",
        )
        tasks.append({"class_name": class_name, "length": length})

    len_mixture = _strict_nonnegative_int(
        value["len_mixture"], field=f"{source}.len_mixture"
    )
    summed_length = sum(task["length"] for task in tasks)
    if summed_length != len_mixture:
        raise MixtureFingerprintError(
            f"{source}.len_mixture={len_mixture} does not equal the sum of task "
            f"lengths ({summed_length})"
        )
    index_map_sha256 = value["index_map_sha256"]
    if not isinstance(index_map_sha256, str) or not _SHA256_RE.fullmatch(
        index_map_sha256
    ):
        raise MixtureFingerprintError(
            f"{source}.index_map_sha256 must be 64 lowercase hex characters, "
            f"got {index_map_sha256!r}"
        )

    return {
        "mixture": mixture,
        "split": split,
        "tasks": tasks,
        "len_mixture": len_mixture,
        "index_map_sha256": index_map_sha256,
    }


def _canonical_index_map_bytes(index_map: list[tuple[int, int]]) -> bytes:
    serializable = [[task_index, local_index] for task_index, local_index in index_map]
    return json.dumps(
        serializable,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")


def compute_mixture_fingerprint(
    task,
    *,
    mixture_name: str,
    split: SFTSplit,
) -> MixtureFingerprint:
    """Derive the conversion/generator fingerprint from one live task object.

    Production passes a ``TaskMixture``.  A plain ``Task`` is accepted as a
    one-task identity so focused generator tests exercise the same guard.
    """
    constituent_tasks_raw = getattr(task, "tasks", None)
    index_map_raw = getattr(task, "index_map", None)
    if constituent_tasks_raw is None or index_map_raw is None:
        constituent_tasks = [task]
        index_map = [(0, index) for index in range(len(task))]
    else:
        constituent_tasks = list(constituent_tasks_raw)
        index_map = []
        for position, pair in enumerate(index_map_raw):
            if (
                not isinstance(pair, (tuple, list))
                or len(pair) != 2
                or isinstance(pair[0], bool)
                or isinstance(pair[1], bool)
                or not isinstance(pair[0], int)
                or not isinstance(pair[1], int)
            ):
                raise MixtureFingerprintError(
                    f"task.index_map[{position}] must be a pair of integers, got {pair!r}"
                )
            index_map.append((pair[0], pair[1]))

    mixture_length = len(task)
    if len(index_map) != mixture_length:
        raise MixtureFingerprintError(
            "TaskMixture fingerprinting requires the full shuffled index_map: "
            f"len(index_map)={len(index_map)} != len(task)={mixture_length}. "
            "Do not pass a sliced TaskMixture to SFT conversion/training."
        )
    tasks: list[TaskFingerprint] = [
        TaskFingerprint(
            class_name=constituent.__class__.__name__,
            length=len(constituent),
        )
        for constituent in constituent_tasks
    ]
    fingerprint: MixtureFingerprint = {
        "mixture": mixture_name,
        "split": split.value,
        "tasks": tasks,
        "len_mixture": mixture_length,
        "index_map_sha256": hashlib.sha256(
            _canonical_index_map_bytes(index_map)
        ).hexdigest(),
    }
    return normalize_mixture_fingerprint(fingerprint, source="live task fingerprint")


def mixture_fingerprint_json(fingerprint: object) -> str:
    """Return the canonical compact JSON carried in feature metadata."""
    normalized = normalize_mixture_fingerprint(fingerprint)
    return json.dumps(
        normalized,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def parse_mixture_fingerprint_json(
    payload: str,
    *,
    source: str,
) -> MixtureFingerprint:
    """Parse and validate the canonical feature-metadata representation."""
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise MixtureFingerprintError(f"{source} is not valid JSON: {exc}") from exc
    return normalize_mixture_fingerprint(value, source=source)


def _uniform_sft_mixture_fingerprint_json(
    shard_paths: Sequence[Path],
    *,
    operation: str,
) -> str | None:
    """Validate and return the SFT provenance shared by feature shards.

    Shard headers are the authoritative feature-side carrier because both
    train and validation extraction write them, while only train feature
    families receive a statistics artifact.
    """
    return _uniform_sft_mixture_fingerprint_metadata_json(
        [
            (
                path,
                read_shard_header(path)["__metadata__"],
            )
            for path in shard_paths
        ],
        operation=operation,
    )


def _uniform_sft_mixture_fingerprint_metadata_json(
    metadata_by_path: Sequence[tuple[Path, Mapping[str, str]]],
    *,
    operation: str,
) -> str | None:
    """Apply the shared SFT rule to caller-supplied, identity-bound metadata."""
    raw_by_path = [
        (
            path,
            metadata.get(SFT_MIXTURE_FINGERPRINT_METADATA_KEY),
        )
        for path, metadata in metadata_by_path
    ]
    present = [(path, raw) for path, raw in raw_by_path if raw is not None]
    if not present:
        return None
    if len(present) != len(raw_by_path):
        missing = [str(path) for path, raw in raw_by_path if raw is None]
        raise MixtureFingerprintError(
            f"{operation} found a half-migrated SFT feature family: "
            f"{len(present)} shard(s) carry "
            f"{SFT_MIXTURE_FINGERPRINT_METADATA_KEY!r}, while missing={missing}. "
            "Re-extract the complete feature directory from one conversion."
        )

    canonical_by_path: list[tuple[Path, str]] = []
    for path, raw in present:
        assert raw is not None
        fingerprint = parse_mixture_fingerprint_json(
            raw,
            source=f"feature shard {path} SFT mixture fingerprint",
        )
        canonical_by_path.append((path, mixture_fingerprint_json(fingerprint)))
    expected_path, expected = canonical_by_path[0]
    mismatched = [
        (str(path), canonical)
        for path, canonical in canonical_by_path[1:]
        if canonical != expected
    ]
    if mismatched:
        raise MixtureFingerprintError(
            f"{operation} found inconsistent SFT mixture fingerprints across "
            f"feature shards: expected from {expected_path}={expected}; "
            f"mismatched={mismatched}. Re-extract one coherent feature family."
        )
    return expected


def mixture_fingerprint_differences(
    left: object,
    right: object,
) -> list[str]:
    """Name the fields distinguishing two valid fingerprints."""
    lhs = normalize_mixture_fingerprint(left, source="left fingerprint")
    rhs = normalize_mixture_fingerprint(right, source="right fingerprint")
    differences: list[str] = []
    for field in ("mixture", "split", "len_mixture", "index_map_sha256"):
        if lhs[field] != rhs[field]:
            differences.append(f"{field}: {lhs[field]!r} != {rhs[field]!r}")
    if lhs["tasks"] != rhs["tasks"]:
        differences.append(f"tasks: {lhs['tasks']!r} != {rhs['tasks']!r}")
    return differences


def format_mixture_fingerprint_mismatch(
    left_label: str,
    left: object,
    right_label: str,
    right: object,
) -> str:
    """Format a fail-loud mismatch with both identities and field-level drift."""
    lhs = normalize_mixture_fingerprint(left, source=left_label)
    rhs = normalize_mixture_fingerprint(right, source=right_label)
    differences = mixture_fingerprint_differences(lhs, rhs)
    return (
        f"differing_fields={differences}; "
        f"{left_label}={mixture_fingerprint_json(lhs)}; "
        f"{right_label}={mixture_fingerprint_json(rhs)}"
    )


def sample_alignment_probe_indices(length: int, sample_count: int) -> list[int]:
    """Return a deterministic endpoint-inclusive spread with no RNG/hash use."""
    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise ValueError(f"length must be a non-negative integer, got {length!r}")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 0
    ):
        raise ValueError(
            "alignment_probe_samples must be a non-negative integer, "
            f"got {sample_count!r}"
        )
    if length == 0 or sample_count == 0:
        return []
    count = min(length, sample_count)
    if count == 1:
        return [0]
    return [index * (length - 1) // (count - 1) for index in range(count)]
