"""Strict schema-v2 seed artifact and runtime-loader tests."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch as t
from hypothesis import given
from hypothesis import strategies as st
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from nanochat.atomic_io import atomic_json_dump
from steering._common.strict_json import canonical_json
from steering.eval.axbench_base import seed_contract as seed_contract_mod
from steering.eval.axbench_base.seed_contract import (
    SEED_SCHEMA_VERSION,
    SEED_TOKEN_SEMANTICS,
    SeedPoolExpectation,
    _stage1b_seed_rows_identity,
    compute_extraction_fingerprint,
    compute_seed_fingerprint,
    load_seed_pool,
    load_seed_pool_with_rows_identity,
    save_seed_pool,
    seed_rows_to_tensors,
    sha256_bytes,
    sha256_file,
)
from steering.eval.axbench_base.types import AxBenchBaseError, SeedRow
from steering.segmentation import compute_tokenizer_checksum


class CharTokenizer:
    """Reversible tokenizer whose literal ``§`` token equals BOS."""

    def __init__(self, bos_id: int = 1) -> None:
        self.bos_id = bos_id

    def get_bos_token_id(self) -> int:
        return self.bos_id

    def encode(self, text: str) -> list[int]:
        return [self.bos_id if char == "§" else 1_000 + ord(char) for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(
            "§" if token == self.bos_id else chr(int(token) - 1_000)
            for token in tokens
        )


def _row(*, shard_idx: int, doc_idx: int, tokenizer: CharTokenizer) -> SeedRow:
    prefix = "a§b"
    continuation = "cd"
    return SeedRow(
        shard_idx=shard_idx,
        doc_idx=doc_idx,
        source_nanochat_token_count=5,
        prefix_source_token_start=0,
        prefix_source_token_end=3,
        prefix_text=prefix,
        prefix_content_nanochat_tokens=tokenizer.encode(prefix),
        sae_feature_ids=[2, 5],
        sae_feature_mags=[1.25, 2.5],
        continuation_text=continuation,
        continuation_content_nanochat_tokens=tokenizer.encode(continuation),
    )


def _seed_config_json() -> str:
    return canonical_json(
        {
            "feature_directory": {"width": "16k"},
            "seed_continuation_nanochat_tokens": 2,
            "seed_prefix_max_nanochat_tokens": 3,
        }
    )


def _commit_pair(
    root: Path,
    tokenizer: CharTokenizer,
) -> tuple[Path, Path, SeedPoolExpectation]:
    train_path = root / "seeds_train.safetensors"
    eval_path = root / "seeds_eval.safetensors"
    config_json = _seed_config_json()
    config_fingerprint = sha256_bytes(config_json.encode("utf-8"))
    nano_checksum = compute_tokenizer_checksum(tokenizer)
    extraction_checksum = "2" * 64
    source_hashes = {"/source/shard_00000.parquet": "3" * 64}
    parameter_hashes = {"forward_model": "4" * 64, "sae": "5" * 64}
    runtime_identity = {"backend": "native", "dtype": "float32"}
    extraction_settings = {"pooling": "max_pool", "segment_size": 128}
    extraction_fingerprint = compute_extraction_fingerprint(
        extraction_settings=extraction_settings,
        runtime_identity=runtime_identity,
        parameter_hashes=parameter_hashes,
    )
    seed_fingerprint = compute_seed_fingerprint(
        candidate_digest="6" * 64,
        config_fingerprint=config_fingerprint,
        extraction_fingerprint=extraction_fingerprint,
        nanochat_tokenizer_checksum=nano_checksum,
        source_parquet_hashes=source_hashes,
    )
    common_metadata = {
        "schema_version": str(SEED_SCHEMA_VERSION),
        "seed_fingerprint": seed_fingerprint,
        "token_semantics": SEED_TOKEN_SEMANTICS,
        "nanochat_tokenizer_checksum": nano_checksum,
        "extraction_tokenizer_checksum": extraction_checksum,
        "config_fingerprint": config_fingerprint,
        "seed_config_json": config_json,
        "source_parquet_hashes_json": canonical_json(source_hashes),
        "parameter_hashes_json": canonical_json(parameter_hashes),
        "runtime_identity_json": canonical_json(runtime_identity),
        "extraction_settings_json": canonical_json(extraction_settings),
    }
    save_seed_pool(
        train_path,
        [_row(shard_idx=0, doc_idx=10, tokenizer=tokenizer)],
        metadata={**common_metadata, "split": "train"},
    )
    save_seed_pool(
        eval_path,
        [_row(shard_idx=0, doc_idx=11, tokenizer=tokenizer)],
        metadata={**common_metadata, "split": "eval"},
    )
    atomic_json_dump(
        {
            "schema_version": SEED_SCHEMA_VERSION,
            "seed_fingerprint": seed_fingerprint,
            "token_semantics": SEED_TOKEN_SEMANTICS,
            "candidate_digest": "6" * 64,
            "candidate_count": 2,
            "skip_counts": {"source_too_short": 1},
            "source_parquet_hashes": source_hashes,
            "config_fingerprint": config_fingerprint,
            "extraction_fingerprint": extraction_fingerprint,
            "nanochat_tokenizer_checksum": nano_checksum,
            "extraction_tokenizer_checksum": extraction_checksum,
            "parameter_hashes": parameter_hashes,
            "runtime_identity": runtime_identity,
            "extraction_settings": extraction_settings,
            "files": {
                train_path.name: {
                    "sha256": sha256_file(train_path),
                    "row_count": 1,
                    "split": "train",
                },
                eval_path.name: {
                    "sha256": sha256_file(eval_path),
                    "row_count": 1,
                    "split": "eval",
                },
            },
        },
        root / "seed_pool_manifest.json",
    )
    expectation = SeedPoolExpectation(
        continuation_length=2,
        prefix_cap=3,
        n_sae_features=16_384,
        nanochat_tokenizer_checksum=nano_checksum,
        config_fingerprint=config_fingerprint,
    )
    return train_path, eval_path, expectation


def _metadata(path: Path) -> dict[str, str]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def _rewrite_seed(
    path: Path,
    *,
    mutate_tensors: Callable[[dict[str, t.Tensor]], None] | None = None,
    mutate_metadata: Callable[[dict[str, str]], None] | None = None,
) -> None:
    tensors = load_file(str(path))
    metadata = _metadata(path)
    if mutate_tensors is not None:
        mutate_tensors(tensors)
    if mutate_metadata is not None:
        mutate_metadata(metadata)
    save_file(tensors, str(path), metadata=metadata)
    manifest_path = path.parent / "seed_pool_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][path.name]["sha256"] = sha256_file(path)
    atomic_json_dump(manifest, manifest_path)


def test_seed_contract_roundtrip_preserves_literal_bos_payload(tmp_path: Path) -> None:
    tokenizer = CharTokenizer()
    train_path, _eval_path, expectation = _commit_pair(tmp_path, tokenizer)
    rows = load_seed_pool(
        train_path,
        expected_split="train",
        expectation=expectation,
        tokenizer=tokenizer,
    )
    assert rows == [_row(shard_idx=0, doc_idx=10, tokenizer=tokenizer)]
    assert rows[0].prefix_content_nanochat_tokens[1] == tokenizer.bos_id


def test_loader_can_derive_expectation_from_v2_metadata(tmp_path: Path) -> None:
    tokenizer = CharTokenizer()
    train_path, _eval_path, _expectation = _commit_pair(tmp_path, tokenizer)
    assert load_seed_pool(train_path, expected_split="train", tokenizer=tokenizer)


def test_wrong_split_config_and_tokenizer_fail_loudly(tmp_path: Path) -> None:
    tokenizer = CharTokenizer()
    train_path, _eval_path, expectation = _commit_pair(tmp_path, tokenizer)
    with pytest.raises(AxBenchBaseError, match="split mismatch"):
        load_seed_pool(train_path, expected_split="eval")
    with pytest.raises(AxBenchBaseError, match="config fingerprint mismatch"):
        load_seed_pool(
            train_path,
            expectation=replace(expectation, config_fingerprint="9" * 64),
            tokenizer=tokenizer,
        )
    other = CharTokenizer(bos_id=9)
    with pytest.raises(AxBenchBaseError, match="tokenizer checksum mismatch"):
        load_seed_pool(
            train_path,
            expectation=replace(
                expectation,
                nanochat_tokenizer_checksum=compute_tokenizer_checksum(other),
            ),
            tokenizer=other,
        )


@pytest.mark.parametrize("missing", ["manifest", "train", "eval"])
def test_torn_seed_artifacts_are_rejected(tmp_path: Path, missing: str) -> None:
    tokenizer = CharTokenizer()
    train_path, eval_path, expectation = _commit_pair(tmp_path, tokenizer)
    target = {
        "manifest": tmp_path / "seed_pool_manifest.json",
        "train": train_path,
        "eval": eval_path,
    }[missing]
    target.unlink()
    with pytest.raises(AxBenchBaseError, match="manifest is missing|pair is torn"):
        load_seed_pool(
            train_path if missing != "train" else eval_path,
            expectation=expectation,
            tokenizer=tokenizer,
        )


def test_bad_file_hash_is_rejected_before_payload_use(tmp_path: Path) -> None:
    tokenizer = CharTokenizer()
    train_path, _eval_path, expectation = _commit_pair(tmp_path, tokenizer)
    with train_path.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(AxBenchBaseError, match="hash mismatch"):
        load_seed_pool(train_path, expectation=expectation, tokenizer=tokenizer)


@pytest.mark.parametrize(
    ("field", "error_match"),
    [
        ("runtime_identity", "extraction fingerprint"),
        ("candidate_digest", "manifest fingerprint"),
    ],
)
def test_manifest_fingerprints_are_recomputed_from_semantic_payload(
    tmp_path: Path,
    field: str,
    error_match: str,
) -> None:
    tokenizer = CharTokenizer()
    train_path, _eval_path, expectation = _commit_pair(tmp_path, tokenizer)
    manifest_path = tmp_path / "seed_pool_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if field == "runtime_identity":
        manifest[field]["dtype"] = "bfloat16"
    else:
        manifest[field] = "8" * 64
    atomic_json_dump(manifest, manifest_path)

    with pytest.raises(AxBenchBaseError, match=error_match):
        load_seed_pool(
            train_path,
            expectation=expectation,
            tokenizer=tokenizer,
        )


def test_legacy_file_and_manifest_schemas_are_rejected(tmp_path: Path) -> None:
    tokenizer = CharTokenizer()
    train_path, _eval_path, expectation = _commit_pair(tmp_path, tokenizer)
    _rewrite_seed(
        train_path,
        mutate_metadata=lambda metadata: metadata.__setitem__("schema_version", "1"),
    )
    with pytest.raises(AxBenchBaseError, match="not schema v2"):
        load_seed_pool(train_path, expectation=expectation, tokenizer=tokenizer)

    _commit_pair(tmp_path, tokenizer)
    manifest_path = tmp_path / "seed_pool_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    atomic_json_dump(manifest, manifest_path)
    with pytest.raises(AxBenchBaseError, match="unsupported schema"):
        load_seed_pool(train_path, expectation=expectation, tokenizer=tokenizer)


def test_malformed_offsets_and_utf8_are_rejected(tmp_path: Path) -> None:
    tokenizer = CharTokenizer()
    train_path, _eval_path, expectation = _commit_pair(tmp_path, tokenizer)

    def bad_offsets(tensors: dict[str, t.Tensor]) -> None:
        tensors["prefix_token_offsets"] = t.tensor([0, 99], dtype=t.int64)

    _rewrite_seed(train_path, mutate_tensors=bad_offsets)
    with pytest.raises(AxBenchBaseError, match="offsets/payload length"):
        load_seed_pool(train_path, expectation=expectation, tokenizer=tokenizer)

    _commit_pair(tmp_path, tokenizer)

    def bad_utf8(tensors: dict[str, t.Tensor]) -> None:
        tensors["prefix_text_bytes"] = t.tensor([255], dtype=t.uint8)
        tensors["prefix_text_offsets"] = t.tensor([0, 1], dtype=t.int64)

    _rewrite_seed(train_path, mutate_tensors=bad_utf8)
    with pytest.raises(AxBenchBaseError, match="Malformed seed row payload"):
        load_seed_pool(train_path, expectation=expectation, tokenizer=tokenizer)


@pytest.mark.parametrize(
    ("ids", "mags", "message"),
    [
        ([2, 2], [1.0, 2.0], "sorted and unique"),
        ([5, 2], [1.0, 2.0], "sorted and unique"),
        ([2, 20_000], [1.0, 2.0], "outside the configured dictionary"),
        ([2, 5], [float("nan"), 2.0], "finite and strictly positive"),
        ([2, 5], [float("inf"), 2.0], "finite and strictly positive"),
        ([2, 5], [0.0, 2.0], "finite and strictly positive"),
        ([2, 5], [-1.0, 2.0], "finite and strictly positive"),
    ],
)
def test_malformed_feature_csr_rows_are_rejected(
    tmp_path: Path,
    ids: list[int],
    mags: list[float],
    message: str,
) -> None:
    tokenizer = CharTokenizer()
    train_path, _eval_path, expectation = _commit_pair(tmp_path, tokenizer)

    def mutate(tensors: dict[str, t.Tensor]) -> None:
        tensors["sae_feature_ids"] = t.tensor(ids, dtype=t.int32)
        tensors["sae_feature_mags"] = t.tensor(mags, dtype=t.float32)

    _rewrite_seed(train_path, mutate_tensors=mutate)
    with pytest.raises(AxBenchBaseError, match=message):
        load_seed_pool(train_path, expectation=expectation, tokenizer=tokenizer)


def test_wrong_m_cap_and_decoded_text_are_rejected(tmp_path: Path) -> None:
    tokenizer = CharTokenizer()
    train_path, _eval_path, expectation = _commit_pair(tmp_path, tokenizer)
    with pytest.raises(AxBenchBaseError, match="expected configured M=3"):
        load_seed_pool(
            train_path,
            expectation=replace(expectation, continuation_length=3),
            tokenizer=tokenizer,
        )
    with pytest.raises(AxBenchBaseError, match="configured cap"):
        load_seed_pool(
            train_path,
            expectation=replace(expectation, prefix_cap=2),
            tokenizer=tokenizer,
        )

    def wrong_text(tensors: dict[str, t.Tensor]) -> None:
        tensors["prefix_text_bytes"] = t.tensor(list(b"zzz"), dtype=t.uint8)
        tensors["prefix_text_offsets"] = t.tensor([0, 3], dtype=t.int64)

    _rewrite_seed(train_path, mutate_tensors=wrong_text)
    with pytest.raises(AxBenchBaseError, match="do not decode exactly"):
        load_seed_pool(train_path, expectation=expectation, tokenizer=tokenizer)


def test_committed_pair_ignores_uniquely_named_partial_outputs(tmp_path: Path) -> None:
    tokenizer = CharTokenizer()
    train_path, _eval_path, expectation = _commit_pair(tmp_path, tokenizer)
    (tmp_path / "seeds_train.safetensors.tmp.123.deadbeef").write_bytes(b"partial")
    assert load_seed_pool(
        train_path,
        expectation=expectation,
        tokenizer=tokenizer,
    )


@given(
    prefix=st.lists(st.integers(min_value=0, max_value=2**31 - 1), min_size=1),
    continuation=st.lists(
        st.integers(min_value=0, max_value=2**31 - 1), min_size=1
    ),
)
def test_seed_tensor_encoding_preserves_arbitrary_source_token_ids(
    prefix: list[int],
    continuation: list[int],
) -> None:
    source_count = max(
        len(prefix) + len(continuation),
        2 * len(continuation),
    )
    prefix_end = source_count - len(continuation)
    prefix_start = prefix_end - len(prefix)
    if prefix_start < 0:
        return
    row = SeedRow(
        shard_idx=0,
        doc_idx=0,
        source_nanochat_token_count=source_count,
        prefix_source_token_start=prefix_start,
        prefix_source_token_end=prefix_end,
        prefix_text="prefix",
        prefix_content_nanochat_tokens=prefix,
        sae_feature_ids=[],
        sae_feature_mags=[],
        continuation_text="continuation",
        continuation_content_nanochat_tokens=continuation,
    )
    tensors = seed_rows_to_tensors([row])
    assert tensors["prefix_content_tokens"].tolist() == prefix
    assert tensors["continuation_content_tokens"].tolist() == continuation


def test_save_rejects_noncanonical_or_mismatched_metadata(tmp_path: Path) -> None:
    tokenizer = CharTokenizer()
    row = _row(shard_idx=0, doc_idx=0, tokenizer=tokenizer)
    metadata: dict[str, Any] = {
        "schema_version": "2",
        "split": "train",
        "seed_fingerprint": "1" * 64,
        "token_semantics": SEED_TOKEN_SEMANTICS,
        "nanochat_tokenizer_checksum": "2" * 64,
        "extraction_tokenizer_checksum": "3" * 64,
        "config_fingerprint": "4" * 64,
        "seed_config_json": "{}",
        "source_parquet_hashes_json": "{}",
        "parameter_hashes_json": "{}",
        "runtime_identity_json": "{}",
        "extraction_settings_json": "{}",
    }
    with pytest.raises(AxBenchBaseError, match="config payload/fingerprint"):
        save_seed_pool(tmp_path / "bad.safetensors", [row], metadata=metadata)


def _bulk_identity(tensors: dict[str, t.Tensor]) -> str:
    return _stage1b_seed_rows_identity(
        tensors,
        row_count=int(tensors["shard_idx"].numel()),
    )


def test_stage1b_bulk_identity_matches_frozen_one_row_digest_and_schema_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = CharTokenizer()
    tensors = seed_rows_to_tensors(
        [_row(shard_idx=0, doc_idx=10, tokenizer=tokenizer)]
    )
    reversed_tensors = dict(reversed(tuple(tensors.items())))
    frame_labels: list[str] = []
    original_frame = seed_contract_mod._hash_stage1b_seed_frame

    def record_frame(digest: Any, label: str, payload: Any) -> None:
        frame_labels.append(label)
        original_frame(digest, label, payload)

    monkeypatch.setattr(
        seed_contract_mod,
        "_hash_stage1b_seed_frame",
        record_frame,
    )
    expected = "dc40dadc82be40915c0d04f34133c32f63e8a8d082de72471c8931f6f7229c00"
    assert _bulk_identity(tensors) == expected
    assert _bulk_identity(reversed_tensors) == expected
    assert frame_labels[:18] == [
        "identity_version",
        "row_count",
        *(name for name, _dtype in seed_contract_mod._SEED_TENSOR_SCHEMA),
    ]


@pytest.mark.parametrize("feature_count", [0, 10_000])
def test_stage1b_bulk_identity_has_constant_frame_and_update_count(
    feature_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = CharTokenizer()
    row = _row(shard_idx=0, doc_idx=10, tokenizer=tokenizer)
    row = replace(
        row,
        sae_feature_ids=list(range(feature_count)),
        sae_feature_mags=[1.0] * feature_count,
    )
    frames = 0
    updates = 0
    original_frame = seed_contract_mod._hash_stage1b_seed_frame
    original_sha256 = seed_contract_mod.hashlib.sha256

    class CountingDigest:
        def __init__(self) -> None:
            self._digest = original_sha256()

        def update(self, payload: Any) -> None:
            nonlocal updates
            updates += 1
            self._digest.update(payload)

        def hexdigest(self) -> str:
            return self._digest.hexdigest()

    def count_frame(digest: Any, label: str, payload: Any) -> None:
        nonlocal frames
        frames += 1
        original_frame(digest, label, payload)

    monkeypatch.setattr(seed_contract_mod.hashlib, "sha256", CountingDigest)
    monkeypatch.setattr(
        seed_contract_mod,
        "_hash_stage1b_seed_frame",
        count_frame,
    )
    assert len(_bulk_identity(seed_rows_to_tensors([row]))) == 64
    assert frames == 18
    assert updates == 72


def test_each_seed_tensor_individually_affects_bulk_identity() -> None:
    tokenizer = CharTokenizer()
    tensors = seed_rows_to_tensors(
        [_row(shard_idx=0, doc_idx=10, tokenizer=tokenizer)]
    )
    baseline = _bulk_identity(tensors)
    for name, _dtype in seed_contract_mod._SEED_TENSOR_SCHEMA:
        mutated = {key: value.clone() for key, value in tensors.items()}
        tensor = mutated[name]
        assert tensor.numel() > 0
        if tensor.dtype == t.float32:
            tensor[0] = t.nextafter(
                tensor[0],
                t.tensor(float("inf"), dtype=t.float32),
            )
        else:
            tensor[0] += 1
        assert _bulk_identity(mutated) != baseline, name


def test_bulk_identity_preserves_row_order_and_offset_only_boundaries() -> None:
    tokenizer = CharTokenizer()
    first = _row(shard_idx=0, doc_idx=10, tokenizer=tokenizer)
    second = replace(
        _row(shard_idx=1, doc_idx=11, tokenizer=tokenizer),
        prefix_text="xy",
        prefix_content_nanochat_tokens=[1_120, 1_121, 1_122],
    )
    ordered = seed_rows_to_tensors([first, second])
    reordered = seed_rows_to_tensors([second, first])
    assert _bulk_identity(ordered) != _bulk_identity(reordered)

    boundary_only = {name: tensor.clone() for name, tensor in ordered.items()}
    raw_bytes = boundary_only["prefix_text_bytes"].clone()
    boundary_only["prefix_text_offsets"][1] += 1
    assert t.equal(boundary_only["prefix_text_bytes"], raw_bytes)
    assert _bulk_identity(boundary_only) != _bulk_identity(ordered)


def test_bulk_identity_uses_float32_bits_unicode_and_empty_feature_boundaries() -> None:
    tokenizer = CharTokenizer()
    first = replace(
        _row(shard_idx=0, doc_idx=2**31 - 1, tokenizer=tokenizer),
        prefix_text="snowman ☃",
        prefix_content_nanochat_tokens=[2**31 - 1, 1, 2],
        sae_feature_ids=[],
        sae_feature_mags=[],
    )
    adjacent = t.nextafter(
        t.tensor(1.0, dtype=t.float32),
        t.tensor(float("inf"), dtype=t.float32),
    ).item()
    second = replace(
        _row(shard_idx=2**31 - 1, doc_idx=11, tokenizer=tokenizer),
        sae_feature_ids=[2, 5],
        sae_feature_mags=[1.0, adjacent],
    )
    tensors = seed_rows_to_tensors([first, second])
    changed = {name: tensor.clone() for name, tensor in tensors.items()}
    changed["sae_feature_mags"][1] = 1.0
    assert _bulk_identity(tensors) != _bulk_identity(changed)


def test_equal_tensors_ignore_safetensors_metadata_and_serialization_layout(
    tmp_path: Path,
) -> None:
    tokenizer = CharTokenizer()
    tensors = seed_rows_to_tensors(
        [_row(shard_idx=0, doc_idx=10, tokenizer=tokenizer)]
    )
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    save_file(tensors, first, metadata={"producer": "first"})
    save_file(
        dict(reversed(tuple(tensors.items()))),
        second,
        metadata={"producer": "second", "extra": "different"},
    )
    assert sha256_file(first) != sha256_file(second)
    assert _bulk_identity(load_file(first)) == _bulk_identity(load_file(second))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda tensors: tensors.pop("doc_idx"),
        lambda tensors: tensors.__setitem__("extra", t.tensor([1])),
        lambda tensors: tensors.__setitem__(
            "doc_idx", tensors["doc_idx"].to(t.int64)
        ),
        lambda tensors: tensors.__setitem__(
            "doc_idx", tensors["doc_idx"].unsqueeze(0)
        ),
        lambda tensors: tensors.__setitem__(
            "sae_feature_ids",
            t.tensor([2, 99, 5, 99], dtype=t.int32)[::2],
        ),
        lambda tensors: tensors.__setitem__(
            "doc_idx", t.empty(1, dtype=t.int32, device="meta")
        ),
    ],
)
def test_bulk_identity_rejects_noncanonical_tensor_buffers_before_hashing(
    mutation: Callable[[dict[str, t.Tensor]], Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = CharTokenizer()
    tensors = seed_rows_to_tensors(
        [_row(shard_idx=0, doc_idx=10, tokenizer=tokenizer)]
    )
    mutation(tensors)
    frame = MagicMock(side_effect=AssertionError("identity frame was published"))
    monkeypatch.setattr(seed_contract_mod, "_hash_stage1b_seed_frame", frame)
    with pytest.raises(AxBenchBaseError):
        _bulk_identity(tensors)
    frame.assert_not_called()


def test_bulk_identity_rejects_big_endian_host_before_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = CharTokenizer()
    tensors = seed_rows_to_tensors(
        [_row(shard_idx=0, doc_idx=10, tokenizer=tokenizer)]
    )
    monkeypatch.setattr(seed_contract_mod.sys, "byteorder", "big")
    frame = MagicMock(side_effect=AssertionError("identity frame was published"))
    monkeypatch.setattr(seed_contract_mod, "_hash_stage1b_seed_frame", frame)
    with pytest.raises(AxBenchBaseError, match="little-endian"):
        _bulk_identity(tensors)
    frame.assert_not_called()


def test_rich_seed_loader_returns_same_rows_and_hashes_loaded_tensors_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = CharTokenizer()
    train_path, _eval_path, expectation = _commit_pair(tmp_path, tokenizer)
    ordinary = load_seed_pool(
        train_path,
        expected_split="train",
        expectation=expectation,
        tokenizer=tokenizer,
    )
    original_load_file = seed_contract_mod.load_file
    original_identity = seed_contract_mod._stage1b_seed_rows_identity
    loaded_objects: list[dict[str, t.Tensor]] = []
    hashed_objects: list[Any] = []

    def tracked_load(path: str) -> dict[str, t.Tensor]:
        tensors = original_load_file(path)
        loaded_objects.append(tensors)
        return tensors

    def tracked_identity(tensors: Any, *, row_count: int) -> str:
        hashed_objects.append(tensors)
        return original_identity(tensors, row_count=row_count)

    monkeypatch.setattr(seed_contract_mod, "load_file", tracked_load)
    monkeypatch.setattr(
        seed_contract_mod,
        "_stage1b_seed_rows_identity",
        tracked_identity,
    )
    loaded = load_seed_pool_with_rows_identity(
        train_path,
        expected_split="train",
        expectation=expectation,
        tokenizer=tokenizer,
    )
    assert loaded.rows == ordinary
    assert loaded.rows_sha256 == (
        "dc40dadc82be40915c0d04f34133c32f63e8a8d082de72471c8931f6f7229c00"
    )
    assert len(loaded_objects) == 1
    assert hashed_objects == loaded_objects


def test_ordinary_loader_does_not_compute_stage1b_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = CharTokenizer()
    train_path, _eval_path, expectation = _commit_pair(tmp_path, tokenizer)
    monkeypatch.setattr(
        seed_contract_mod,
        "_stage1b_seed_rows_identity",
        MagicMock(side_effect=AssertionError("ordinary loader hashed rows")),
    )
    assert load_seed_pool(
        train_path,
        expected_split="train",
        expectation=expectation,
        tokenizer=tokenizer,
    )


@pytest.mark.benchmark
def test_stage1b_bulk_identity_benchmark_records_throughput() -> None:
    tokenizer = CharTokenizer()
    row = replace(
        _row(shard_idx=0, doc_idx=10, tokenizer=tokenizer),
        sae_feature_ids=list(range(100_000)),
        sae_feature_mags=[1.0] * 100_000,
    )
    tensors = seed_rows_to_tensors([row])
    payload_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    started = time.perf_counter()
    digest = _bulk_identity(tensors)
    elapsed = time.perf_counter() - started
    throughput_mib_s = payload_bytes / max(elapsed, 1e-12) / (1024 * 1024)
    assert len(digest) == 64
    assert throughput_mib_s > 0.0
