"""Strict receipt and semantic-byte tests for NL retrieval inputs."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch as t

from steering.eval.axbench_upstream import nl_target_source
from steering.eval.axbench_upstream.nl_target_source import (
    load_nl_target_source_receipt,
    retrieval_targets_sha256,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha256(texts: list[str]) -> str:
    payload = json.dumps(texts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_receipt(tmp_path: Path) -> tuple[Path, Path, Path]:
    description_path = tmp_path / "descriptions.npz"
    concept_path = tmp_path / "concepts.npz"
    np.savez(
        description_path,
        fids=np.asarray([2, 0], dtype=np.int64),
        emb=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    texts = ["café", "second"]
    np.savez(
        concept_path,
        texts=np.asarray(texts),
        emb=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    embedding_space = {
        "model": "producer/model",
        "revision": "1" * 40,
        "encoding": "prompt=v1;pool=mean;normalize=false",
        "library_versions": {"numpy": "2.3.2", "sentence-transformers": "5.1.0"},
    }
    receipt = {
        "schema_version": 1,
        "description_embeddings": {
            "path": description_path.name,
            "sha256": _sha256(description_path),
            "source_texts_sha256": "2" * 64,
            "embedding_space": embedding_space,
        },
        "concept_embeddings": {
            "path": concept_path.name,
            "sha256": _sha256(concept_path),
            "source_texts_sha256": _text_sha256(texts),
            "embedding_space": embedding_space,
        },
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False), encoding="utf-8")
    return receipt_path, description_path, concept_path


def test_receipt_binds_exact_bytes_relative_paths_and_unicode_population(tmp_path: Path) -> None:
    receipt_path, description_path, concept_path = _write_receipt(tmp_path)

    loaded = load_nl_target_source_receipt(receipt_path)

    assert loaded.path == receipt_path.resolve()
    assert loaded.sha256 == _sha256(receipt_path)
    assert loaded.description_embeddings.path == description_path.resolve()
    assert loaded.concept_embeddings.path == concept_path.resolve()
    assert loaded.expected_hashes == {
        "description_embeddings": _sha256(description_path),
        "concept_embeddings": _sha256(concept_path),
    }
    assert loaded.description_embeddings.manifest_record(relative_to=tmp_path / "out") == {
        "path": "../descriptions.npz",
        "sha256": _sha256(description_path),
        "source_texts_sha256": "2" * 64,
        "embedding_space": loaded.description_embeddings.embedding_space,
    }


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: value.update(extra=True), "fields must equal"),
        (lambda value: value.update(schema_version=True), "schema_version must equal 1"),
        (
            lambda value: value["description_embeddings"].update(extra=True),
            "description_embeddings fields differ",
        ),
        (
            lambda value: value["description_embeddings"].update(path=" "),
            "path must be a nonblank string",
        ),
        (
            lambda value: value["description_embeddings"].update(sha256="A" * 64),
            "sha256 must be lowercase hexadecimal",
        ),
        (
            lambda value: value["description_embeddings"].update(source_texts_sha256="0"),
            "source_texts_sha256 must be lowercase hexadecimal",
        ),
        (
            lambda value: value["description_embeddings"]["embedding_space"].update(extra=True),
            "embedding_space fields differ",
        ),
        (
            lambda value: value["description_embeddings"]["embedding_space"].update(model=""),
            "model must be a nonblank string",
        ),
        (
            lambda value: value["description_embeddings"]["embedding_space"].update(
                revision="not-a-commit"
            ),
            "revision must be a 40-character commit",
        ),
        (
            lambda value: value["description_embeddings"]["embedding_space"].update(
                encoding=""
            ),
            "encoding must be a nonblank string",
        ),
        (
            lambda value: value["description_embeddings"]["embedding_space"].update(
                library_versions={}
            ),
            "library_versions must be nonempty strings",
        ),
    ],
)
def test_receipt_rejects_noncanonical_schema(
    tmp_path: Path,
    mutate: Callable[[dict[str, object]], None],
    match: str,
) -> None:
    receipt_path, _description_path, _concept_path = _write_receipt(tmp_path)
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    mutate(value)
    receipt_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_nl_target_source_receipt(receipt_path)


def test_receipt_rejects_absolute_artifact_path_even_with_valid_hash(tmp_path: Path) -> None:
    receipt_path, description_path, _concept_path = _write_receipt(tmp_path)
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    value["description_embeddings"]["path"] = str(description_path.resolve())
    receipt_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="description_embeddings path must be relative"):
        load_nl_target_source_receipt(receipt_path)


def test_receipt_rejects_duplicate_keys_and_nonstandard_constants(tmp_path: Path) -> None:
    receipt_path, _description_path, _concept_path = _write_receipt(tmp_path)
    original = receipt_path.read_text(encoding="utf-8")
    receipt_path.write_text(
        original.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_nl_target_source_receipt(receipt_path)

    receipt_path.write_text(original.replace('"schema_version": 1', '"schema_version": NaN'))
    with pytest.raises(ValueError, match="non-standard JSON numeric constant"):
        load_nl_target_source_receipt(receipt_path)


def test_receipt_rejects_embedding_space_mismatch_and_aliases(tmp_path: Path) -> None:
    receipt_path, _description_path, concept_path = _write_receipt(tmp_path)
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    value["concept_embeddings"]["embedding_space"]["encoding"] = "different"
    receipt_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="embedding spaces differ"):
        load_nl_target_source_receipt(receipt_path)

    receipt_path, _description_path, concept_path = _write_receipt(tmp_path)
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    value["description_embeddings"]["path"] = concept_path.name
    value["description_embeddings"]["sha256"] = _sha256(concept_path)
    receipt_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt and embedding files must be distinct"):
        load_nl_target_source_receipt(receipt_path)


def test_receipt_rejects_changed_artifact_and_changed_concept_population(tmp_path: Path) -> None:
    receipt_path, description_path, concept_path = _write_receipt(tmp_path)
    description_path.write_bytes(description_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="description_embeddings SHA-256 mismatch"):
        load_nl_target_source_receipt(receipt_path)

    receipt_path, _description_path, concept_path = _write_receipt(tmp_path)
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    value["concept_embeddings"]["source_texts_sha256"] = "3" * 64
    receipt_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the NPZ text population"):
        load_nl_target_source_receipt(receipt_path)


def test_receipt_rejects_malformed_concept_npz_before_population_assertion(tmp_path: Path) -> None:
    receipt_path, _description_path, concept_path = _write_receipt(tmp_path)
    np.savez(
        concept_path,
        texts=np.asarray([1, 2], dtype=np.int64),
        emb=np.asarray([[1.0], [2.0]], dtype=np.float32),
    )
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    value["concept_embeddings"]["sha256"] = _sha256(concept_path)
    receipt_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="rank-1 Unicode"):
        load_nl_target_source_receipt(receipt_path)


def test_receipt_rejects_wrong_concept_npz_members_with_updated_hash(tmp_path: Path) -> None:
    receipt_path, _description_path, concept_path = _write_receipt(tmp_path)
    np.savez(
        concept_path,
        texts=np.asarray(["café", "second"]),
        vectors=np.asarray([[1.0], [2.0]], dtype=np.float32),
    )
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    value["concept_embeddings"]["sha256"] = _sha256(concept_path)
    receipt_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="must contain exactly.*emb.*texts"):
        load_nl_target_source_receipt(receipt_path)


def test_concept_population_hash_uses_same_descriptor_bound_artifact_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path, _description_path, concept_path = _write_receipt(tmp_path)
    bytes_a = concept_path.read_bytes()
    staged_b = tmp_path / "concepts-b.npz"
    np.savez(
        staged_b,
        texts=np.asarray(["replacement"]),
        emb=np.asarray([[0.0, 1.0]], dtype=np.float32),
    )
    bytes_b = staged_b.read_bytes()
    assert _sha256(concept_path) != _sha256(staged_b)
    with np.load(staged_b, allow_pickle=False) as payload:
        assert set(payload.files) == {"emb", "texts"}
        assert payload["texts"].tolist() == ["replacement"]

    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    value["concept_embeddings"]["source_texts_sha256"] = _text_sha256(["replacement"])
    receipt_path.write_text(json.dumps(value), encoding="utf-8")

    real_hash = nl_target_source.stream_hash_file_descriptor_bound
    swapped = False

    def hash_then_replace(path: Path, *, retain_payload: bool = False) -> Any:
        nonlocal swapped
        result = real_hash(path, retain_payload=retain_payload)
        if not swapped and Path(path) == concept_path.resolve():
            os.replace(staged_b, concept_path)
            swapped = True
        return result

    monkeypatch.setattr(
        nl_target_source,
        "stream_hash_file_descriptor_bound",
        hash_then_replace,
    )

    with pytest.raises(ValueError, match="concept_embeddings SHA-256 mismatch"):
        load_nl_target_source_receipt(receipt_path)

    assert swapped
    assert concept_path.read_bytes() == bytes_b
    assert hashlib.sha256(bytes_a).digest() != hashlib.sha256(bytes_b).digest()


def test_target_digest_is_order_canonical_byte_sensitive_and_nonmutating() -> None:
    first = t.tensor([0.0, -0.0, 1.0], dtype=t.float32)
    second = t.tensor([-1.0, 0.5, 0.25], dtype=t.float32)
    first_bytes = first.numpy().tobytes()

    digest = retrieval_targets_sha256({9: second, 2: first})

    assert digest == retrieval_targets_sha256({2: first, 9: second})
    assert digest != retrieval_targets_sha256({2: first.clone(), 9: second.flip(0)})
    changed_signed_zero = first.clone()
    changed_signed_zero[1] = 0.0
    assert digest != retrieval_targets_sha256({2: changed_signed_zero, 9: second})
    assert first.numpy().tobytes() == first_bytes


@pytest.mark.parametrize(
    "targets",
    [
        {},
        {-1: t.ones(1)},
        {True: t.ones(1)},
        {0: t.ones(1, dtype=t.float64)},
        {0: t.ones((1, 1), dtype=t.float32)},
        {0: t.tensor([float("nan")], dtype=t.float32)},
        {0: t.tensor([1.01], dtype=t.float32)},
        {0: t.ones(4, dtype=t.float32)[::2]},
    ],
)
def test_target_digest_rejects_malformed_contract(targets: dict[object, t.Tensor]) -> None:
    with pytest.raises(ValueError, match="must not be empty|concept IDs|invalid tensor"):
        # Cross the typed API boundary deliberately to exercise runtime rejection.
        retrieval_targets_sha256(cast(Any, targets))


def test_target_digest_rejects_different_feature_row_widths() -> None:
    with pytest.raises(ValueError, match="common feature-row width"):
        retrieval_targets_sha256(
            {
                0: t.zeros(2, dtype=t.float32),
                1: t.zeros(3, dtype=t.float32),
            }
        )
