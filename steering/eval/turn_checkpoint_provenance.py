"""Authenticate a turn-cache producer checkpoint without changing cache identity."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

from steering.sft.segment_mode_identity import ONLINE_SAE_LINEAGE_META_KEY


class CheckpointSeedSource(StrEnum):
    ONLINE_SAE_LINEAGE = f"{ONLINE_SAE_LINEAGE_META_KEY}.selection_seed"


class MissingModelReason(StrEnum):
    NOT_STAGED = "model_file_not_present_next_to_metadata"


def _model_digest(path: Path) -> tuple[str | None, str | None]:
    try:
        handle = path.open("rb")
    except FileNotFoundError:
        return None, MissingModelReason.NOT_STAGED.value
    with handle:
        before = os.fstat(handle.fileno())
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        after = os.fstat(handle.fileno())
    current = path.stat()
    signatures = {(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                  for stat in (before, after, current)}
    if len(signatures) != 1:
        raise ValueError("checkpoint model changed while hashing producer provenance")
    return digest, None


def turn_checkpoint_provenance(
    path: Path, payload: bytes, metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Use the producer's already-read metadata bytes, never a second JSON load.

    Resolved tags are relative to the checkpoint root when available, otherwise
    the leaf name, like the fixed checkpoint pin. Relocation may change the
    staging root but must preserve the logical tag and the original file bytes.
    """
    filename_step = re.fullmatch(r"meta_(\d+)\.json", path.name)
    step = int(filename_step[1]) if filename_step else metadata.get("step")
    if type(step) is not int or step < 0:
        raise ValueError("checkpoint provenance requires a numeric meta_<step>.json or metadata step")
    if "step" in metadata and (type(metadata["step"]) is not int or metadata["step"] != step):
        raise ValueError("checkpoint metadata step disagrees with its filename")
    lineage = metadata.get(ONLINE_SAE_LINEAGE_META_KEY)
    if not isinstance(lineage, Mapping) or type(lineage.get("selection_seed")) is not int:
        raise ValueError("checkpoint provenance requires the online-SAE lineage selection seed")
    resolved_path = path.expanduser().resolve(strict=True)
    tag = resolved_path.parent.name
    for parent in resolved_path.parents:
        if parent.name in {"conditional_encdec_checkpoints", "base_checkpoints"}:
            tag = resolved_path.parent.relative_to(parent).as_posix()
            break
    model_sha, reason = _model_digest(resolved_path.with_name(f"model_{step:06d}.pt"))
    return {
        "checkpoint_meta_path": str(path),
        "checkpoint_metadata_sha256": hashlib.sha256(payload).hexdigest(),
        "checkpoint_resolved_tag": tag,
        "checkpoint_step": step,
        "checkpoint_model_sha256": model_sha,
        "checkpoint_model_sha256_reason": reason,
        "checkpoint_preselection_seed": lineage["selection_seed"],
        "checkpoint_preselection_seed_source": CheckpointSeedSource.ONLINE_SAE_LINEAGE.value,
    }


def authenticate_turn_checkpoint_provenance(
    stored: Mapping[str, Any], expected: Mapping[str, Any],
) -> None:
    """Require every recorded field, including explicit nulls; paths may move."""
    for key, value in expected.items():
        if not key.startswith("checkpoint_") or key == "checkpoint_meta_path":
            continue
        if key not in stored or type(stored[key]) is not type(value) or stored[key] != value:
            raise ValueError(f"turn cache checkpoint provenance disagrees: {key}")
