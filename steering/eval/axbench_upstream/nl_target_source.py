"""Strict producer receipt for AxBench natural-language retrieval inputs."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch as t

from steering._common.strict_json import (
    reject_nonstandard_json_constant,
    strict_json_object,
)
from steering.immutable_io import ImmutableArtifactError, stream_hash_file_descriptor_bound

_ARTIFACT_NAMES = ("description_embeddings", "concept_embeddings")
_RECEIPT_KEYS = frozenset({"schema_version", *_ARTIFACT_NAMES})
_ARTIFACT_KEYS = frozenset({"path", "sha256", "source_texts_sha256", "embedding_space"})
_EMBEDDING_SPACE_KEYS = frozenset({"model", "revision", "encoding", "library_versions"})


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class NLTargetArtifact:
    """One verified embedding snapshot and its producer provenance."""

    path: Path
    sha256: str
    source_texts_sha256: str
    embedding_space: dict[str, Any]

    def manifest_record(self, *, relative_to: Path) -> dict[str, Any]:
        """Render the record with a path relative to the published manifest."""

        return {
            "path": os.path.relpath(self.path, start=relative_to),
            "sha256": self.sha256,
            "source_texts_sha256": self.source_texts_sha256,
            "embedding_space": self.embedding_space,
        }


@dataclass(frozen=True)
class NLTargetSourceReceipt:
    """Descriptor-bound identity of both retrieval embedding snapshots."""

    path: Path
    sha256: str
    description_embeddings: NLTargetArtifact
    concept_embeddings: NLTargetArtifact

    @property
    def expected_hashes(self) -> dict[str, str]:
        return {
            "description_embeddings": self.description_embeddings.sha256,
            "concept_embeddings": self.concept_embeddings.sha256,
        }


def _embedding_space(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _EMBEDDING_SPACE_KEYS:
        raise ValueError(f"{label} embedding_space fields differ")
    record = cast(dict[str, object], value)
    model = record["model"]
    revision = record["revision"]
    encoding = record["encoding"]
    versions = record["library_versions"]
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{label} embedding_space model must be a nonblank string")
    if not _is_lower_hex(revision, 40):
        raise ValueError(f"{label} embedding_space revision must be a 40-character commit")
    if not isinstance(encoding, str) or not encoding.strip():
        raise ValueError(f"{label} embedding_space encoding must be a nonblank string")
    if (
        not isinstance(versions, dict)
        or not versions
        or any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(item, str)
            or not item.strip()
            for key, item in versions.items()
        )
    ):
        raise ValueError(f"{label} embedding_space library_versions must be nonempty strings")
    return cast(dict[str, Any], value)


def _artifact(
    value: object,
    *,
    name: str,
    receipt_dir: Path,
) -> NLTargetArtifact:
    if not isinstance(value, dict) or set(value) != _ARTIFACT_KEYS:
        raise ValueError(f"NL target receipt {name} fields differ")
    record = cast(dict[str, object], value)
    raw_path = record["path"]
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"NL target receipt {name} path must be a nonblank string")
    relative_path = Path(raw_path)
    if relative_path.is_absolute():
        raise ValueError(f"NL target receipt {name} path must be relative")
    sha256 = record["sha256"]
    source_texts_sha256 = record["source_texts_sha256"]
    if not _is_lower_hex(sha256, 64):
        raise ValueError(f"NL target receipt {name} sha256 must be lowercase hexadecimal")
    if not _is_lower_hex(source_texts_sha256, 64):
        raise ValueError(
            f"NL target receipt {name} source_texts_sha256 must be lowercase hexadecimal"
        )
    path = (receipt_dir / relative_path).expanduser().resolve()
    actual = stream_hash_file_descriptor_bound(path).sha256
    if actual != sha256:
        raise ValueError(
            f"NL target receipt {name} SHA-256 mismatch: expected {sha256}, read {actual}"
        )
    return NLTargetArtifact(
        path=path,
        sha256=cast(str, sha256),
        source_texts_sha256=cast(str, source_texts_sha256),
        embedding_space=_embedding_space(record["embedding_space"], label=name),
    )


def _concept_texts_sha256(path: Path, *, expected_sha256: str) -> str:
    """Hash the locally observable concept-text population in NPZ row order."""

    digest = stream_hash_file_descriptor_bound(path, retain_payload=True)
    if digest.sha256 != expected_sha256:
        raise ValueError(
            "NL target receipt concept_embeddings SHA-256 mismatch: "
            f"expected {expected_sha256}, read {digest.sha256}"
        )
    snapshot = cast(bytes, digest.payload)
    with np.load(BytesIO(snapshot), allow_pickle=False) as payload:
        if len(payload.files) != 2 or set(payload.files) != {"texts", "emb"}:
            raise ValueError("concept embedding NPZ must contain exactly ['emb', 'texts']")
        texts = payload["texts"]
    if texts.ndim != 1 or texts.dtype.kind != "U":
        raise ValueError("concept embedding texts must be a rank-1 Unicode array")
    population = [str(text) for text in texts.tolist()]
    encoded = json.dumps(
        population,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_nl_target_source_receipt(path: Path) -> NLTargetSourceReceipt:
    """Load, strictly validate, and byte-bind one combined retrieval receipt."""

    resolved = Path(path).expanduser().resolve()
    try:
        digest = stream_hash_file_descriptor_bound(resolved, retain_payload=True)
        payload = cast(bytes, digest.payload)
        loaded = json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_nonstandard_json_constant,
            object_pairs_hook=strict_json_object,
        )
    except (ImmutableArtifactError, OSError, ValueError) as exc:
        raise ValueError(f"expected a readable strict NL target receipt at {resolved}: {exc}") from exc
    if not isinstance(loaded, dict) or set(loaded) != _RECEIPT_KEYS:
        raise ValueError(f"NL target receipt fields must equal {sorted(_RECEIPT_KEYS)}")
    if type(loaded["schema_version"]) is not int or loaded["schema_version"] != 1:
        raise ValueError("NL target receipt schema_version must equal 1")
    description = _artifact(
        loaded["description_embeddings"],
        name="description_embeddings",
        receipt_dir=resolved.parent,
    )
    concepts = _artifact(
        loaded["concept_embeddings"],
        name="concept_embeddings",
        receipt_dir=resolved.parent,
    )
    if len({resolved, description.path, concepts.path}) != 3:
        raise ValueError("NL target receipt and embedding files must be distinct")
    if description.embedding_space != concepts.embedding_space:
        raise ValueError("description and concept embedding spaces differ")
    actual_concept_texts_sha256 = _concept_texts_sha256(
        concepts.path,
        expected_sha256=concepts.sha256,
    )
    if actual_concept_texts_sha256 != concepts.source_texts_sha256:
        raise ValueError(
            "NL target receipt concept_embeddings source_texts_sha256 does not match "
            "the NPZ text population"
        )
    return NLTargetSourceReceipt(
        path=resolved,
        sha256=digest.sha256,
        description_embeddings=description,
        concept_embeddings=concepts,
    )


def retrieval_targets_sha256(targets: dict[int, t.Tensor]) -> str:
    """Hash strict CPU float32 target bytes in concept and row order."""

    if not targets:
        raise ValueError("NL retrieval target mapping must not be empty")
    digest = hashlib.sha256(b"axbench-upstream-nl-targets-v1\0")
    expected_width: int | None = None
    for concept_id in sorted(targets):
        value = targets[concept_id]
        if type(concept_id) is not int or concept_id < 0:
            raise ValueError("NL retrieval target concept IDs must be nonnegative strict integers")
        if (
            not isinstance(value, t.Tensor)
            or value.dtype is not t.float32
            or value.device.type != "cpu"
            or value.ndim != 1
            or value.numel() == 0
            or not value.is_contiguous()
            or not bool(t.isfinite(value).all())
            or bool((value < -1.0).any())
            or bool((value > 1.0).any())
        ):
            raise ValueError(f"NL retrieval target concept_id={concept_id} has an invalid tensor")
        if expected_width is None:
            expected_width = value.numel()
        elif value.numel() != expected_width:
            raise ValueError("NL retrieval targets must have one common feature-row width")
        digest.update(concept_id.to_bytes(8, "big", signed=False))
        digest.update(value.numel().to_bytes(8, "big", signed=False))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


__all__ = (
    "NLTargetArtifact",
    "NLTargetSourceReceipt",
    "load_nl_target_source_receipt",
    "retrieval_targets_sha256",
)
