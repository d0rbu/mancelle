"""Strict and canonical JSON primitives for source-index manifest readers and
canonical content-identity digests (for example the document blacklist)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from steering.eval.autointerp_filter.path_safety import (
    PathSafetyError,
    file_stat_signature,
    open_regular_file_no_follow,
    verify_open_regular_file_binding,
)
from steering.eval.autointerp_filter.types import AutoInterpCacheError


class _DuplicateJSONKeyError(ValueError):
    """Internal signal used to reject ambiguous JSON objects."""


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKeyError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def reject_json_constant(value: str) -> None:
    """``json.loads`` parse_constant hook that refuses NaN/Infinity literals."""
    raise ValueError(f"non-finite JSON number {value!r} is not permitted")


def canonical_json(value: object) -> str:
    """Render canonical compact UTF-8 JSON with sorted keys and finite numbers."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: object) -> str:
    """Return SHA-256 of canonical compact UTF-8 JSON for ``value``."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def canonical_json_equal(left: object, right: object) -> bool:
    """Compare JSON values without Python's bool/int/float equality aliases."""
    return canonical_json(left) == canonical_json(right)


def loads_json_object(text: str, *, context: str) -> dict[str, object]:
    """Parse one JSON object strictly, rejecting duplicate keys and non-finite numbers."""
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=reject_json_constant,
        )
    except (json.JSONDecodeError, _DuplicateJSONKeyError, ValueError) as exc:
        raise AutoInterpCacheError(f"{context} contains invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise AutoInterpCacheError(f"{context} must contain a JSON object")
    return payload


def load_json_object(path: Path, *, context: str) -> dict[str, object]:
    """Read and strictly parse one JSON object file."""
    file_descriptor: int | None = None
    handle = None
    try:
        file_descriptor = open_regular_file_no_follow(path)
        before = verify_open_regular_file_binding(file_descriptor, path)
        handle = os.fdopen(file_descriptor, "rb")
        file_descriptor = None
        payload = handle.read(before.st_size + 1)
        after = verify_open_regular_file_binding(handle.fileno(), path)
        if (
            file_stat_signature(before) != file_stat_signature(after)
            or len(payload) != before.st_size
        ):
            raise AutoInterpCacheError(f"{context} {path} changed while it was read")
        text = payload.decode("utf-8")
    except BaseException as primary_error:
        cleanup_error: OSError | None = None
        try:
            if handle is not None:
                handle.close()
            elif file_descriptor is not None:
                os.close(file_descriptor)
        except OSError as exc:
            cleanup_error = exc
        if cleanup_error is not None:
            primary_error.add_note(f"reader cleanup also failed for {path}: {cleanup_error}")
        if isinstance(primary_error, AutoInterpCacheError):
            raise
        if isinstance(primary_error, (OSError, PathSafetyError, UnicodeError)):
            raise AutoInterpCacheError(
                f"cannot read {context} {path}: {primary_error}"
            ) from primary_error
        raise

    try:
        assert handle is not None
        handle.close()
    except OSError as exc:
        raise AutoInterpCacheError(f"cannot read {context} {path}: {exc}") from exc
    return loads_json_object(text, context=f"{context} {path}")
