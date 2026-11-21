"""Exact CPU contracts for the SmolLM3 pretraining cooldown."""

from __future__ import annotations

import json
import math
import os
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch.optim import AdamW
from transformers import (
    PreTrainedTokenizerFast,
    SmolLM3Config,
    SmolLM3ForCausalLM,
    Trainer,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_flash_attention_utils import prepare_fa_kwargs_from_position_ids

import steering.smollm3_cooldown as cooldown
from steering.sft.smollm3 import SMOLLM3_CONTRACT
from steering.smollm3_cooldown import (
    CONTEXT_LENGTH,
    COOLDOWN_STEPS,
    DEVICE_BATCH_SIZE,
    SOURCE_TOKENS,
    TARGET_SHORTFALL,
    TARGET_TOKENS,
    TOKENS_PER_GLOBAL_BATCH,
    WINDOW_LENGTH,
    CooldownCallback,
    CooldownDataset,
    CooldownTrainer,
    PreparedShard,
    atomic_write_bytes,
    build_training_arguments,
    cooldown_collator,
    expected_scheduler_lr,
    experiment_arithmetic,
    gradient_accumulation_steps,
    load_prepared_manifest,
    prepare_dataset,
    resolve_resume_checkpoint,
    semantic_digest,
    semantic_payload,
    sha256_file,
    tokenizer_identity,
    validate_parquet_inventory,
)


def _tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(WordLevel({"<unk>": 0, "a": 1, "b": 2, "c": 3}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")


def _exact_source_config() -> SmolLM3Config:
    return SmolLM3Config(
        vocab_size=128_256,
        hidden_size=2_048,
        intermediate_size=11_008,
        num_hidden_layers=36,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_position_embeddings=4_096,
        bos_token_id=128_000,
        eos_token_id=128_001,
        pad_token_id=None,
        tie_word_embeddings=True,
        rope_parameters={"rope_theta": 50_000.0, "rope_type": "default"},
        no_rope_layer_interval=4,
        architectures=["SmolLM3ForCausalLM"],
    )


def _write_prepared_shard(
    root: Path,
    shard_index: int,
    documents: list[list[int]],
) -> tuple[dict[str, Any], PreparedShard]:
    stem = f"shard_{shard_index:05d}"
    flat: list[int] = []
    ends: list[int] = []
    for document in documents:
        assert document and document[-1] == cooldown.SOURCE_END_OF_TEXT_ID
        flat.extend(document)
        ends.append(len(flat))
    token_path = root / f"{stem}.ds"
    index_path = root / f"{stem}.ds.index"
    np.asarray(flat, dtype=cooldown.TOKEN_DTYPE).tofile(token_path)
    np.asarray(ends, dtype=cooldown.INDEX_DTYPE).tofile(index_path)
    token_sha = sha256_file(token_path)
    index_sha = sha256_file(index_path)
    receipt = {
        "schema_version": cooldown.PREPARED_SCHEMA_VERSION,
        "source": {
            "path": str((root / f"{stem}.parquet").resolve()),
            "size": 1,
            "mtime_ns": 1,
        },
        "tokenizer_sha256": "a" * 64,
        "eos_token_id": cooldown.SOURCE_END_OF_TEXT_ID,
        "window_length": WINDOW_LENGTH,
        "context_stride": CONTEXT_LENGTH,
        "document_count": len(documents),
        "token_count": len(flat),
        "context_count": max(0, (len(flat) - 1) // CONTEXT_LENGTH),
        "files": {
            "tokens": {"bytes": token_path.stat().st_size, "sha256": token_sha},
            "index": {"bytes": index_path.stat().st_size, "sha256": index_sha},
        },
    }
    prepared = PreparedShard(
        stem=stem,
        token_path=token_path,
        index_path=index_path,
        token_count=len(flat),
        document_count=len(documents),
        context_count=max(0, (len(flat) - 1) // CONTEXT_LENGTH),
        token_sha256=token_sha,
        index_sha256=index_sha,
    )
    return receipt, prepared


def _write_manifest(root: Path, receipts: list[dict[str, Any]], required: int) -> Path:
    manifest = {
        "schema_version": cooldown.PREPARED_SCHEMA_VERSION,
        "source_model_id": cooldown.SOURCE_MODEL_ID,
        "source_revision": cooldown.SOURCE_REVISION,
        "tokenizer_sha256": "a" * 64,
        "eos_token_id": cooldown.SOURCE_END_OF_TEXT_ID,
        "context_length": CONTEXT_LENGTH,
        "window_length": WINDOW_LENGTH,
        "context_stride": CONTEXT_LENGTH,
        "required_contexts": required,
        "available_contexts": sum(int(receipt["context_count"]) for receipt in receipts),
        "source_inventory": {
            "parquet_dir": "/source",
            "total_shards": 1_400,
            "train_shards": 1_399,
            "validation_shard": "shard_01399.parquet",
        },
        "shards": receipts,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def _documents_for_two_contexts() -> list[list[int]]:
    lengths = [1_000, 3_000, 3_194, 2_000]
    documents: list[list[int]] = []
    for document_index, length in enumerate(lengths):
        token = document_index + 1
        documents.append([token] * (length - 1) + [cooldown.SOURCE_END_OF_TEXT_ID])
    return documents


def _complete_checkpoint(path: Path, *, step: int, digest: str, world_size: int = 1) -> None:
    path.mkdir(parents=True)
    required = ["optimizer.pt", "scheduler.pt", "trainer_state.json", "training_args.bin"]
    required += (
        ["rng_state.pth"]
        if world_size == 1
        else [f"rng_state_{rank}.pth" for rank in range(world_size)]
    )
    for name in required + ["model.safetensors"]:
        (path / name).write_bytes(name.encode())
    marker = {
        "schema_version": 1,
        "completed_cooldown_step": step,
        "semantic_digest": digest,
        "semantic": {"batch": {"world_size": world_size}},
    }
    (path / cooldown.CHECKPOINT_MARKER).write_text(json.dumps(marker))


class _DecayFixture(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = t.nn.Module()
        self.model.embed_tokens = t.nn.Embedding(8, 4)
        self.model.norm = t.nn.LayerNorm(4)
        self.model.projection = t.nn.Linear(4, 4)


class _ShortDataset(t.utils.data.Dataset[dict[str, t.Tensor]]):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, t.Tensor]:
        input_ids = t.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=t.long)
        labels = input_ids.roll(-1)
        labels[-1] = -100
        return {
            "input_ids": input_ids,
            "position_ids": t.arange(8),
            "labels": labels,
            "shift_labels": labels,
        }


class _SupervisedCounter:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def supervised_items(self, start: int, stop: int) -> int:
        self.calls.append((start, stop))
        return (stop - start) * CONTEXT_LENGTH - 7


def test_exact_experiment_arithmetic_and_complete_batch_choice() -> None:
    arithmetic = experiment_arithmetic()
    assert TOKENS_PER_GLOBAL_BATCH == 2_359_296
    assert SOURCE_TOKENS == 471_859_200_000
    assert arithmetic == {
        "source_step": 200_000,
        "source_tokens": 471_859_200_000,
        "target_tokens": 524_288_000_000,
        "tokens_per_global_batch": 2_359_296,
        "cooldown_steps": 22_222,
        "cooldown_contexts": 12_799_872,
        "cooldown_tokens": 52_428_275_712,
        "final_step": 222_222,
        "final_tokens": 524_287_475_712,
        "target_shortfall": 524_288,
    }
    assert TARGET_TOKENS - arithmetic["final_tokens"] == TARGET_SHORTFALL
    assert arithmetic["final_tokens"] < TARGET_TOKENS
    assert arithmetic["final_tokens"] + TOKENS_PER_GLOBAL_BATCH - TARGET_TOKENS == 1_835_008


@pytest.mark.parametrize(
    ("world_size", "expected"),
    [(1, 192), (2, 96), (3, 64), (4, 48), (8, 24), (16, 12), (32, 6), (64, 3)],
)
def test_gradient_accumulation_preserves_576_contexts(world_size: int, expected: int) -> None:
    assert gradient_accumulation_steps(world_size) == expected
    assert expected * world_size * DEVICE_BATCH_SIZE == 576


@pytest.mark.parametrize("world_size", [0, -1, 5, 7, 128, 193])
def test_gradient_accumulation_rejects_inexact_geometry(world_size: int) -> None:
    with pytest.raises(ValueError, match="world_size"):
        gradient_accumulation_steps(world_size)


def test_scheduler_values_match_transformers_reference_bit_for_bit() -> None:
    parameter = t.nn.Parameter(t.ones(()))
    optimizer = AdamW([parameter], lr=cooldown.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cooldown.WARMUP_STEPS,
        num_training_steps=COOLDOWN_STEPS,
    )
    observed = [optimizer.param_groups[0]["lr"]]
    for _ in range(COOLDOWN_STEPS):
        optimizer.step()
        scheduler.step()
        observed.append(optimizer.param_groups[0]["lr"])
    assert observed == [expected_scheduler_lr(step) for step in range(COOLDOWN_STEPS + 1)]
    assert observed[0] == 0.0
    assert observed[1] == pytest.approx(1e-7)
    assert observed[1_999] == pytest.approx(0.0001999)
    assert observed[2_000] == 0.0002
    assert observed[2_001] == pytest.approx(0.00019999010978142616)
    assert observed[22_221] == pytest.approx(9.890218573830483e-09)
    assert observed[22_222] == 0.0


@pytest.mark.parametrize("step", [-1, COOLDOWN_STEPS + 1])
def test_scheduler_rejects_out_of_horizon_steps(step: int) -> None:
    with pytest.raises(ValueError, match="completed_steps"):
        expected_scheduler_lr(step)


def test_atomic_write_is_exact_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "payload.json"
    atomic_write_bytes(destination, b"first\n")
    assert destination.read_bytes() == b"first\n"
    atomic_write_bytes(destination, b"second\n")
    assert destination.read_bytes() == b"second\n"
    assert list(destination.parent.glob(".*.tmp-*")) == []


def test_sha256_file_streams_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "body"
    path.write_bytes(b"abc" * 1_000_000)
    assert sha256_file(path, chunk_size=17) == sha256_file(path, chunk_size=1_000_000)
    assert sha256_file(path) == "f4096a131e7e6ebfa7a512b5c299e13b065df34d15624ee1202ab394cc4d7e90"


def test_tokenizer_identity_is_complete_and_sensitive() -> None:
    first = _tokenizer()
    second = _tokenizer()
    assert tokenizer_identity(first) == tokenizer_identity(second)
    second.backend_tokenizer.add_tokens(["new-token"])
    assert tokenizer_identity(first) != tokenizer_identity(second)


def test_cooldown_source_config_accepts_every_exact_architecture_field() -> None:
    cooldown.validate_cooldown_source_config(_exact_source_config())


def test_cooldown_still_rejects_sft_export_token_ids() -> None:
    config = _exact_source_config()
    config.eos_token_id = SMOLLM3_CONTRACT.generation_eos_token_id
    config.pad_token_id = SMOLLM3_CONTRACT.generation_pad_token_id
    with pytest.raises(ValueError, match="eos_token_id"):
        cooldown.validate_cooldown_source_config(config)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("rope_parameters", {"rope_theta": 10_000.0, "rope_type": "default"}),
        ("no_rope_layer_interval", 3),
        ("no_rope_layers", [0] * 36),
        ("use_sliding_window", True),
        ("sliding_window", 2_048),
        ("layer_types", ["sliding_attention"] * 36),
        ("attention_bias", True),
        ("mlp_bias", True),
        ("rms_norm_eps", 1.0e-5),
        ("hidden_act", "relu"),
        ("initializer_range", 0.01),
        ("attention_dropout", 0.1),
    ],
)
def test_cooldown_source_config_rejects_every_numeric_architecture_drift(
    name: str, value: Any
) -> None:
    config = _exact_source_config()
    setattr(config, name, value)
    with pytest.raises(ValueError, match=name):
        cooldown.validate_cooldown_source_config(config)


def test_inventory_requires_exact_contiguous_1400_shards(tmp_path: Path) -> None:
    expected = []
    for index in range(1_400):
        path = tmp_path / f"shard_{index:05d}.parquet"
        path.touch()
        expected.append(path)
    assert validate_parquet_inventory(tmp_path) == expected
    expected[99].unlink()
    (tmp_path / "shard_99999.parquet").touch()
    with pytest.raises(ValueError, match="shard_00099"):
        validate_parquet_inventory(tmp_path)


def test_prepare_shard_preserves_row_group_and_document_order(tmp_path: Path) -> None:
    source = tmp_path / "shard_00000.parquet"
    writer = pq.ParquetWriter(source, pa.schema([("text", pa.string())]))
    writer.write_table(pa.table({"text": ["a b", "c"]}))
    writer.write_table(pa.table({"text": ["b a a"]}))
    writer.close()
    output = tmp_path / "prepared"
    output.mkdir()
    tokenizer = _tokenizer()
    receipt = cooldown._prepare_shard(
        source,
        output,
        tokenizer=tokenizer,
        tokenizer_sha256=tokenizer_identity(tokenizer),
        batch_size=1,
    )
    tokens = np.fromfile(output / "shard_00000.ds", dtype=cooldown.TOKEN_DTYPE).tolist()
    ends = np.fromfile(output / "shard_00000.ds.index", dtype=cooldown.INDEX_DTYPE).tolist()
    eos = cooldown.SOURCE_END_OF_TEXT_ID
    assert tokens == [1, 2, eos, 3, eos, 2, 1, 1, eos]
    assert ends == [3, 5, 9]
    assert receipt["document_count"] == 3
    assert receipt["token_count"] == 9
    assert receipt["context_count"] == 0
    assert json.loads((output / "shard_00000.json").read_text()) == receipt
    assert receipt["files"]["tokens"]["sha256"] == sha256_file(output / "shard_00000.ds")


@pytest.mark.parametrize(
    ("token_count", "contexts"),
    [(0, 0), (1, 0), (CONTEXT_LENGTH, 0), (WINDOW_LENGTH, 1), (8_192, 1), (8_193, 2)],
)
def test_context_count_uses_causal_overlap_stride(token_count: int, contexts: int) -> None:
    assert cooldown._context_count(token_count) == contexts


def test_prepare_shard_reuses_only_exact_valid_receipt(tmp_path: Path) -> None:
    source = tmp_path / "shard_00000.parquet"
    pq.write_table(pa.table({"text": ["a b"]}), source)
    output = tmp_path / "prepared"
    output.mkdir()
    tokenizer = _tokenizer()
    identity = tokenizer_identity(tokenizer)
    first = cooldown._prepare_shard(
        source, output, tokenizer=tokenizer, tokenizer_sha256=identity, batch_size=4
    )
    before = (output / "shard_00000.ds").stat().st_mtime_ns
    second = cooldown._prepare_shard(
        source, output, tokenizer=tokenizer, tokenizer_sha256=identity, batch_size=1
    )
    assert second == first
    assert (output / "shard_00000.ds").stat().st_mtime_ns == before
    with (output / "shard_00000.ds").open("r+b") as handle:
        handle.seek(0)
        handle.write(b"bad!")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        cooldown._prepare_shard(
            source, output, tokenizer=tokenizer, tokenizer_sha256=identity, batch_size=1
        )


@pytest.mark.parametrize("missing", ["shard_00000.ds", "shard_00000.ds.index"])
def test_prepare_shard_rejects_incomplete_permanent_output(tmp_path: Path, missing: str) -> None:
    source = tmp_path / "shard_00000.parquet"
    pq.write_table(pa.table({"text": ["a"]}), source)
    output = tmp_path / "prepared"
    output.mkdir()
    present = "shard_00000.ds.index" if missing.endswith(".ds") else "shard_00000.ds"
    (output / present).touch()
    tokenizer = _tokenizer()
    with pytest.raises(RuntimeError, match="incomplete prepared shard"):
        cooldown._prepare_shard(
            source,
            output,
            tokenizer=tokenizer,
            tokenizer_sha256=tokenizer_identity(tokenizer),
            batch_size=1,
        )


def test_prepare_dataset_stops_at_smallest_sufficient_prefix_and_reserves_val(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(1_400):
        path = source / f"shard_{index:05d}.parquet"
        if index < 3:
            pq.write_table(pa.table({"text": ["a " * (WINDOW_LENGTH - 1)]}), path)
        else:
            path.touch()
    output = tmp_path / "prepared"
    monkeypatch.setattr(cooldown, "COOLDOWN_CONTEXTS", 2)
    manifest_path = prepare_dataset(
        parquet_dir=source,
        output_dir=output,
        tokenizer=_tokenizer(),
        batch_size=32,
    )
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["shards"]) == 2
    assert manifest["available_contexts"] == 2
    assert [Path(record["source"]["path"]).stem for record in manifest["shards"]] == [
        "shard_00000",
        "shard_00001",
    ]
    assert not (output / "shard_00002.ds").exists()
    assert not (output / "shard_01399.ds").exists()
    assert manifest["source_inventory"]["validation_shard"] == "shard_01399.parquet"


def test_prepare_dataset_rejects_insufficient_training_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(1_400):
        pq.write_table(pa.table({"text": [""]}), source / f"shard_{index:05d}.parquet")
    monkeypatch.setattr(cooldown, "COOLDOWN_CONTEXTS", 1)
    with pytest.raises(RuntimeError, match="contexts"):
        prepare_dataset(
            parquet_dir=source,
            output_dir=tmp_path / "prepared",
            tokenizer=_tokenizer(),
            batch_size=128,
        )


def test_manifest_validation_checks_hashes_counts_order_and_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, _ = _write_prepared_shard(tmp_path, 0, _documents_for_two_contexts())
    _write_manifest(tmp_path, [receipt], required=2)
    monkeypatch.setattr(cooldown, "COOLDOWN_CONTEXTS", 2)
    manifest, shards = load_prepared_manifest(tmp_path, verify_hashes=True)
    assert manifest["available_contexts"] == 2
    assert len(shards) == 1 and shards[0].context_count == 2
    with (tmp_path / "shard_00000.ds").open("r+b") as handle:
        handle.seek(0)
        original = handle.read(4)
        handle.seek(0)
        handle.write(bytes([original[0] ^ 1]) + original[1:])
    load_prepared_manifest(tmp_path, verify_hashes=False)
    with pytest.raises(ValueError, match="digest mismatch"):
        load_prepared_manifest(tmp_path, verify_hashes=True)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda manifest: manifest.update(context_length=2048), "context_length"),
        (lambda manifest: manifest.update(context_stride=4097), "context_stride"),
        (lambda manifest: manifest.update(source_revision="0" * 40), "source_revision"),
        (lambda manifest: manifest.update(available_contexts=99), "context total"),
        (lambda manifest: manifest["shards"][0].update(context_count=3), "count invariant"),
    ],
)
def test_manifest_rejects_semantic_and_count_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
    message: str,
) -> None:
    receipt, _ = _write_prepared_shard(tmp_path, 0, _documents_for_two_contexts())
    path = _write_manifest(tmp_path, [receipt], required=2)
    manifest = json.loads(path.read_text())
    mutation(manifest)
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(cooldown, "COOLDOWN_CONTEXTS", 2)
    with pytest.raises(ValueError, match=message):
        load_prepared_manifest(tmp_path, verify_hashes=False)


def test_dataset_real_context_size_shift_and_document_boundaries(tmp_path: Path) -> None:
    _, shard = _write_prepared_shard(tmp_path, 0, _documents_for_two_contexts())
    dataset = CooldownDataset([shard], length=2)
    first = dataset[0]
    assert {name: value.shape for name, value in first.items()} == {
        "input_ids": (CONTEXT_LENGTH,),
        "position_ids": (CONTEXT_LENGTH,),
        "labels": (CONTEXT_LENGTH,),
    }
    assert first["position_ids"][:1_005].tolist() == [*range(1_000), *range(5)]
    assert first["position_ids"][3_995:4_005].tolist() == [
        2_995,
        2_996,
        2_997,
        2_998,
        2_999,
        0,
        1,
        2,
        3,
        4,
    ]
    masked = first["labels"].eq(-100).nonzero().flatten().tolist()
    assert masked == [999, 3_999]
    raw = np.fromfile(shard.token_path, dtype=cooldown.TOKEN_DTYPE).astype(np.int64)
    expected_labels = t.from_numpy(raw[1:WINDOW_LENGTH].copy())
    expected_labels[[999, 3_999]] = -100
    assert t.equal(first["input_ids"], t.from_numpy(raw[:CONTEXT_LENGTH].copy()))
    assert t.equal(first["labels"], expected_labels)
    second = dataset[1]
    assert second["position_ids"][0].item() == 0
    assert second["position_ids"][3_097].item() == 3_097
    assert second["position_ids"][3_098].item() == 0
    assert second["labels"].eq(-100).nonzero().flatten().tolist() == [3_097]
    assert second["input_ids"][0].item() == first["labels"][-1].item()
    assert t.equal(
        second["input_ids"],
        t.from_numpy(raw[CONTEXT_LENGTH : 2 * CONTEXT_LENGTH].copy()),
    )
    assert dataset.supervised_items(0, 1) == CONTEXT_LENGTH - 2
    assert dataset.supervised_items(1, 2) == CONTEXT_LENGTH - 1
    assert dataset.supervised_items(0, 2) == 2 * CONTEXT_LENGTH - 3


@pytest.mark.parametrize(("start", "stop"), [(-1, 0), (0, 3), (2, 1)])
def test_dataset_supervised_count_rejects_invalid_interval(
    tmp_path: Path, start: int, stop: int
) -> None:
    _, shard = _write_prepared_shard(tmp_path, 0, _documents_for_two_contexts())
    dataset = CooldownDataset([shard], length=2)
    with pytest.raises(ValueError, match="context interval"):
        dataset.supervised_items(start, stop)


def test_dataset_supervised_count_matches_materialized_labels_across_shards(
    tmp_path: Path,
) -> None:
    _, first = _write_prepared_shard(tmp_path, 0, _documents_for_two_contexts())
    _, second = _write_prepared_shard(tmp_path, 1, _documents_for_two_contexts())
    dataset = CooldownDataset([first, second], length=4)
    for start in range(5):
        for stop in range(start, 5):
            materialized = sum(
                int(dataset[index]["labels"].ne(-100).sum().item())
                for index in range(start, stop)
            )
            assert dataset.supervised_items(start, stop) == materialized


def test_dataset_never_joins_shards_and_closes_previous_mmap(tmp_path: Path) -> None:
    document = [7] * (WINDOW_LENGTH - 1) + [cooldown.SOURCE_END_OF_TEXT_ID]
    _, first = _write_prepared_shard(tmp_path, 0, [document])
    _, second = _write_prepared_shard(
        tmp_path, 1, [[9] * (WINDOW_LENGTH - 1) + [cooldown.SOURCE_END_OF_TEXT_ID]]
    )
    dataset = CooldownDataset([first, second], length=2)
    assert dataset[0]["input_ids"].unique().tolist() == [7]
    assert dataset._tokens is not None
    first_mmap = cast(Any, dataset._tokens)._mmap
    assert dataset[1]["input_ids"].unique().tolist() == [9]
    assert first_mmap.closed
    assert dataset._open_shard_index == 1


def test_dataset_pickle_drops_open_handles_and_reopens_exactly(tmp_path: Path) -> None:
    _, shard = _write_prepared_shard(tmp_path, 0, _documents_for_two_contexts())
    dataset = CooldownDataset([shard], length=2)
    expected = dataset[0]
    assert dataset._tokens is not None
    old_mmap = cast(Any, dataset._tokens)._mmap
    payload = pickle.dumps(dataset)
    assert old_mmap.closed
    restored = pickle.loads(payload)
    assert restored._tokens is None
    actual = restored[0]
    assert all(t.equal(actual[key], expected[key]) for key in expected)


@pytest.mark.parametrize("index", [-1, 2])
def test_dataset_rejects_out_of_range_index(tmp_path: Path, index: int) -> None:
    _, shard = _write_prepared_shard(tmp_path, 0, _documents_for_two_contexts())
    dataset = CooldownDataset([shard], length=2)
    with pytest.raises(IndexError):
        dataset[index]


def test_collator_flattens_three_full_contexts_without_attention_mask(tmp_path: Path) -> None:
    _, shard = _write_prepared_shard(tmp_path, 0, _documents_for_two_contexts() * 2)
    dataset = CooldownDataset([shard], length=3)
    examples = [dataset[index] for index in range(3)]
    batch = cooldown_collator(examples)
    assert set(batch) == {"input_ids", "position_ids", "labels", "shift_labels"}
    assert "attention_mask" not in batch
    assert all(value.shape == (1, 12_288) for value in batch.values())
    assert batch["shift_labels"] is batch["labels"]
    for index, example in enumerate(examples):
        start = index * CONTEXT_LENGTH
        stop = start + CONTEXT_LENGTH
        for key in ("input_ids", "position_ids", "labels"):
            assert t.equal(batch[key][0, start:stop], example[key])


def test_collator_position_resets_produce_exact_transformers_fa_varlen_boundaries(
    tmp_path: Path,
) -> None:
    _, shard = _write_prepared_shard(tmp_path, 0, _documents_for_two_contexts() * 2)
    dataset = CooldownDataset([shard], length=3)
    batch = cooldown_collator([dataset[index] for index in range(3)])
    (cu_q, cu_k), (max_q, max_k) = prepare_fa_kwargs_from_position_ids(batch["position_ids"])
    expected_starts = batch["position_ids"].reshape(-1).eq(0).nonzero().flatten().to(t.int32)
    expected = t.cat([expected_starts, t.tensor([3 * CONTEXT_LENGTH], dtype=t.int32)])
    assert t.equal(cu_q, expected)
    assert t.equal(cu_k, expected)
    assert max_q.item() == expected.diff().max().item()
    assert max_k.item() == expected.diff().max().item()
    assert expected_starts[:5].tolist() == [0, 1_000, 4_000, 4_096, 7_194]


def test_collator_accepts_mask_at_last_shifted_token() -> None:
    examples = []
    for _ in range(3):
        positions = t.arange(CONTEXT_LENGTH)
        labels = t.ones(CONTEXT_LENGTH, dtype=t.long)
        labels[-1] = -100
        examples.append(
            {
                "input_ids": t.ones(CONTEXT_LENGTH, dtype=t.long),
                "position_ids": positions,
                "labels": labels,
            }
        )
    batch = cooldown_collator(examples)
    assert batch["labels"][0, CONTEXT_LENGTH - 1].item() == -100


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_collator_rejects_wrong_microbatch_cardinality(count: int) -> None:
    example = {
        "input_ids": t.ones(CONTEXT_LENGTH, dtype=t.long),
        "position_ids": t.arange(CONTEXT_LENGTH),
        "labels": t.ones(CONTEXT_LENGTH, dtype=t.long),
    }
    with pytest.raises(ValueError, match="exactly 3"):
        cooldown_collator([example] * count)


def test_collator_rejects_wrong_shape_keys_position_start_and_mask() -> None:
    valid = {
        "input_ids": t.ones(CONTEXT_LENGTH, dtype=t.long),
        "position_ids": t.arange(CONTEXT_LENGTH),
        "labels": t.ones(CONTEXT_LENGTH, dtype=t.long),
    }
    wrong_shape = {**valid, "input_ids": t.ones(4, dtype=t.long)}
    with pytest.raises(ValueError, match="shape"):
        cooldown_collator([wrong_shape, valid, valid])
    wrong_keys = {**valid, "extra": t.ones(1)}
    with pytest.raises(ValueError, match="keys"):
        cooldown_collator([wrong_keys, valid, valid])
    wrong_start = {**valid, "position_ids": t.arange(1, CONTEXT_LENGTH + 1)}
    with pytest.raises(ValueError, match="start"):
        cooldown_collator([wrong_start, valid, valid])
    wrong_mask = {**valid, "labels": valid["labels"].clone()}
    wrong_mask["labels"][10] = -100
    with pytest.raises(ValueError, match="boundary"):
        cooldown_collator([wrong_mask, valid, valid])


def test_decay_policy_excludes_only_exact_tied_embedding() -> None:
    model = _DecayFixture()
    trainer = object.__new__(CooldownTrainer)
    decay = CooldownTrainer.get_decay_parameter_names(trainer, model)
    all_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert set(decay) == set(all_names) - {"model.embed_tokens.weight"}
    assert "model.norm.weight" in decay
    assert "model.norm.bias" in decay
    assert "model.projection.bias" in decay


def test_decay_policy_fails_if_exact_embedding_name_is_absent() -> None:
    trainer = object.__new__(CooldownTrainer)
    with pytest.raises(ValueError, match="model.embed_tokens.weight"):
        CooldownTrainer.get_decay_parameter_names(trainer, t.nn.Linear(4, 4))


@pytest.mark.parametrize("norm", [0.0, 0.5, 1.0e20])
def test_gradient_norm_guard_accepts_every_finite_norm(
    norm: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Trainer, "_clip_grad_norm", lambda self, model: t.tensor(norm))
    trainer = object.__new__(CooldownTrainer)
    result = CooldownTrainer._clip_grad_norm(trainer, t.nn.Linear(1, 1))
    assert result.item() == pytest.approx(norm)


@pytest.mark.parametrize("norm", [float("inf"), float("-inf"), float("nan")])
def test_gradient_norm_guard_rejects_nonfinite_before_optimizer(
    norm: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Trainer, "_clip_grad_norm", lambda self, model: t.tensor(norm))
    trainer = object.__new__(CooldownTrainer)
    with pytest.raises(FloatingPointError, match="non-finite"):
        CooldownTrainer._clip_grad_norm(trainer, t.nn.Linear(1, 1))


def test_production_training_arguments_are_exact(tmp_path: Path) -> None:
    args = build_training_arguments(
        output_dir=tmp_path,
        world_size=2,
        logging_steps=10,
        save_steps=2_000,
        dataloader_workers=2,
    )
    assert args.per_device_train_batch_size == 3
    assert args.gradient_accumulation_steps == 96
    assert args.max_steps == 22_222
    assert args.learning_rate == 2e-4
    assert args.warmup_steps == 2_000
    assert str(args.lr_scheduler_type) == "SchedulerType.LINEAR"
    assert str(args.optim) == "OptimizerNames.ADAMW_TORCH_FUSED"
    assert (args.adam_beta1, args.adam_beta2, args.adam_epsilon) == (0.9, 0.95, 1e-8)
    assert args.weight_decay == 0.1
    assert args.max_grad_norm == 1.0
    assert args.bf16 and args.tf32 is False
    assert args.gradient_checkpointing and not args.torch_compile and not args.use_cache
    assert args.logging_nan_inf_filter is False
    assert args.eval_strategy == "no"
    assert args.save_strategy == "steps" and args.save_total_limit == 2
    assert args.ignore_data_skip is False
    assert args.seed == args.data_seed == 6
    assert args.dataloader_drop_last and args.dataloader_persistent_workers
    assert args.remove_unused_columns is False
    assert args.label_names == ["labels", "shift_labels"]
    assert args.train_sampling_strategy == "sequential"
    assert args.ddp_find_unused_parameters is False
    assert args.average_tokens_across_devices


@pytest.mark.parametrize(
    ("logging_steps", "save_steps", "workers"),
    [(0, 1, 0), (1, 0, 0), (1, 1, -1)],
)
def test_training_arguments_reject_invalid_operational_values(
    tmp_path: Path, logging_steps: int, save_steps: int, workers: int
) -> None:
    with pytest.raises(ValueError):
        build_training_arguments(
            output_dir=tmp_path,
            world_size=1,
            logging_steps=logging_steps,
            save_steps=save_steps,
            dataloader_workers=workers,
        )


def test_semantic_digest_is_canonical_and_sensitive() -> None:
    first = {"b": [2, 3], "a": {"x": 1}}
    second = {"a": {"x": 1}, "b": [2, 3]}
    assert semantic_digest(first) == semantic_digest(second)
    second["a"]["x"] = 2
    assert semantic_digest(first) != semantic_digest(second)


def test_semantic_payload_binds_data_code_lock_and_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b'{"complete":true}\n')
    monkeypatch.setattr(
        cooldown.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=""),
    )
    monkeypatch.setattr(
        cooldown,
        "repository_version",
        lambda: {"package": "0.1.0", "git_revision": "a" * 40},
    )
    payload = semantic_payload(manifest_path=manifest, world_size=2, local_world_size=1)
    assert payload["data_manifest_sha256"] == sha256_file(manifest)
    assert payload["repository"]["git_revision"] == "a" * 40
    assert payload["uv_lock_sha256"] == sha256_file(Path(cooldown.__file__).parents[1] / "uv.lock")
    assert payload["batch"] == {
        "contexts_per_device": 3,
        "contexts_per_global_batch": 576,
        "world_size": 2,
        "local_world_size": 1,
        "gradient_accumulation_steps": 96,
    }
    assert payload["optimizer"]["warmup_steps"] == 2_000
    assert payload["execution"] == {"gradient_checkpointing": True, "torch_compile": False}


def test_semantic_payload_rejects_dirty_or_unresolved_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b"{}\n")
    monkeypatch.setattr(
        cooldown.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="?? untracked.py\n"),
    )
    with pytest.raises(RuntimeError, match="clean committed"):
        semantic_payload(manifest_path=manifest, world_size=1, local_world_size=1)
    monkeypatch.setattr(
        cooldown.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=""),
    )
    monkeypatch.setattr(
        cooldown,
        "repository_version",
        lambda: {"package": None, "git_revision": None},
    )
    with pytest.raises(RuntimeError, match="Git revision"):
        semantic_payload(manifest_path=manifest, world_size=1, local_world_size=1)


def test_resume_returns_latest_complete_matching_checkpoint_and_skips_partial(
    tmp_path: Path,
) -> None:
    digest = "d" * 64
    _complete_checkpoint(tmp_path / "checkpoint-10", step=10, digest=digest)
    _complete_checkpoint(tmp_path / "checkpoint-20", step=20, digest=digest)
    (tmp_path / "checkpoint-30.partial").mkdir()
    assert resolve_resume_checkpoint(tmp_path, expected_digest=digest) == tmp_path / "checkpoint-20"


def test_resume_quarantines_unmarked_checkpoint_once_and_preserves_it(tmp_path: Path) -> None:
    partial = tmp_path / "checkpoint-10"
    partial.mkdir()
    (partial / "model.safetensors").write_bytes(b"valuable partial state")
    assert resolve_resume_checkpoint(tmp_path, expected_digest="d" * 64) is None
    quarantined = tmp_path / "checkpoint-10.partial"
    assert (quarantined / "model.safetensors").read_bytes() == b"valuable partial state"
    assert resolve_resume_checkpoint(tmp_path, expected_digest="d" * 64) is None


def test_resume_rejects_quarantine_collision_malformed_path_and_non_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / "checkpoint-1").mkdir()
    (tmp_path / "checkpoint-1.partial").mkdir()
    with pytest.raises(RuntimeError, match="collision"):
        resolve_resume_checkpoint(tmp_path, expected_digest="d" * 64)
    os.replace(tmp_path / "checkpoint-1", tmp_path / "checkpoint-bad")
    with pytest.raises(RuntimeError, match="malformed"):
        resolve_resume_checkpoint(tmp_path, expected_digest="d" * 64)
    (tmp_path / "checkpoint-bad").rename(tmp_path / "not-a-checkpoint")
    (tmp_path / "checkpoint-2").write_text("not a directory")
    with pytest.raises(RuntimeError, match="not a directory"):
        resolve_resume_checkpoint(tmp_path, expected_digest="d" * 64)


def test_resume_rejects_digest_step_and_inventory_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint-10"
    _complete_checkpoint(path, step=10, digest="a" * 64)
    with pytest.raises(RuntimeError, match="semantic identity"):
        resolve_resume_checkpoint(tmp_path, expected_digest="b" * 64)
    marker_path = path / cooldown.CHECKPOINT_MARKER
    marker = json.loads(marker_path.read_text())
    marker["semantic_digest"] = "b" * 64
    marker["completed_cooldown_step"] = 9
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(RuntimeError, match="step mismatch"):
        resolve_resume_checkpoint(tmp_path, expected_digest="b" * 64)
    marker["completed_cooldown_step"] = 10
    marker_path.write_text(json.dumps(marker))
    (path / "optimizer.pt").unlink()
    with pytest.raises(RuntimeError, match="optimizer.pt"):
        resolve_resume_checkpoint(tmp_path, expected_digest="b" * 64)


def test_callback_canary_requests_save_and_stop_at_exact_step() -> None:
    counter = _SupervisedCounter()
    callback = CooldownCallback(
        semantic={"x": 1},
        canary_steps=3,
        flops_per_token=10.0,
        peak_flops=100.0,
        dataset=cast(CooldownDataset, counter),
    )
    control = TrainerControl()
    state = TrainerState(global_step=2)
    empty_args = cast(TrainingArguments, SimpleNamespace())
    assert not callback.on_step_end(empty_args, state, control).should_training_stop
    state.global_step = 3
    result = callback.on_step_end(empty_args, state, control)
    assert result.should_save and result.should_training_stop


def test_callback_writes_marker_last_with_exact_inventory(tmp_path: Path) -> None:
    digest_semantic = {"batch": {"world_size": 1}, "x": 1}
    checkpoint = tmp_path / "checkpoint-7"
    _complete_checkpoint(checkpoint, step=7, digest="temporary")
    (checkpoint / cooldown.CHECKPOINT_MARKER).unlink()
    callback = CooldownCallback(
        semantic=digest_semantic,
        canary_steps=None,
        flops_per_token=10.0,
        peak_flops=100.0,
        dataset=cast(CooldownDataset, _SupervisedCounter()),
    )
    args = cast(
        TrainingArguments,
        SimpleNamespace(output_dir=str(tmp_path), process_index=0, world_size=1),
    )
    callback.on_save(args, TrainerState(global_step=7), TrainerControl())
    marker = json.loads((checkpoint / cooldown.CHECKPOINT_MARKER).read_text())
    assert marker["completed_cooldown_step"] == 7
    assert marker["absolute_step"] == 200_007
    assert marker["nominal_tokens"] == SOURCE_TOKENS + 7 * TOKENS_PER_GLOBAL_BATCH
    assert marker["semantic"] == digest_semantic
    assert marker["semantic_digest"] == semantic_digest(digest_semantic)
    assert set(marker["files"]) == {
        "model.safetensors",
        "optimizer.pt",
        "rng_state.pth",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    }


def test_tiny_real_trainer_uses_shift_labels_decay_and_resume(tmp_path: Path) -> None:
    config = SmolLM3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        no_rope_layer_interval=4,
        tie_word_embeddings=True,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=1,
    )
    model = SmolLM3ForCausalLM(config)
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        do_train=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        max_steps=2,
        learning_rate=1e-3,
        warmup_steps=1,
        optim="adamw_torch",
        weight_decay=0.1,
        max_grad_norm=1.0,
        save_strategy="steps",
        save_steps=1,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels", "shift_labels"],
        train_sampling_strategy="sequential",
        disable_tqdm=True,
    )
    trainer = CooldownTrainer(model=model, args=args, train_dataset=_ShortDataset())
    result = trainer.train()
    assert result.global_step == 2
    assert trainer.state.global_step == 2
    assert (tmp_path / "checkpoint-2" / "optimizer.pt").is_file()
    changed = [
        name
        for name, parameter in model.named_parameters()
        if not t.equal(parameter, initial[name])
    ]
    assert changed
    resumed_model = SmolLM3ForCausalLM(config)
    resumed = CooldownTrainer(model=resumed_model, args=args, train_dataset=_ShortDataset())
    resumed.train(resume_from_checkpoint=str(tmp_path / "checkpoint-1"))
    assert resumed.state.global_step == 2
    for left, right in zip(model.parameters(), resumed_model.parameters(), strict=True):
        assert t.equal(left, right)


def test_explicit_shift_labels_matches_independent_cross_entropy() -> None:
    config = SmolLM3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        no_rope_layer_interval=4,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=1,
    )
    model = SmolLM3ForCausalLM(config).eval()
    input_ids = t.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    position_ids = t.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    shift_labels = t.tensor([[2, 3, 4, -100, 6, 7, 8, -100]])
    with t.no_grad():
        output = model(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=shift_labels,
            shift_labels=shift_labels,
        )
        reference = t.nn.functional.cross_entropy(
            output.logits.float().reshape(-1, config.vocab_size),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )
    assert t.equal(output.loss, reference)


def test_supervised_item_count_uses_shift_labels_without_another_shift(tmp_path: Path) -> None:
    config = SmolLM3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        no_rope_layer_interval=4,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=1,
    )
    args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        per_device_train_batch_size=1,
        average_tokens_across_devices=True,
        remove_unused_columns=False,
        label_names=["labels", "shift_labels"],
        report_to="none",
    )
    trainer = CooldownTrainer(model=SmolLM3ForCausalLM(config), args=args)
    batch = _ShortDataset()[0]
    items = trainer._get_num_items_in_batch([batch, batch], t.device("cpu"))
    assert items is not None
    assert int(items) == 14


def test_mfu_formula_is_dimensionless_and_world_size_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = _SupervisedCounter()
    callback = CooldownCallback(
        semantic={},
        canary_steps=None,
        flops_per_token=100.0,
        peak_flops=1_000.0,
        dataset=cast(CooldownDataset, counter),
    )
    callback.last_log_time = 10.0
    callback.last_log_step = 0
    messages: list[tuple[Any, ...]] = []
    monkeypatch.setattr(cooldown.time, "monotonic", lambda: 12.0)
    monkeypatch.setattr(cooldown.logger, "info", lambda *args: messages.append(args))
    callback.on_log(
        cast(TrainingArguments, SimpleNamespace(world_size=2)),
        TrainerState(global_step=2),
        TrainerControl(),
        logs={"loss": 1.0},
    )
    assert len(messages) == 1
    scheduled_per_second = TOKENS_PER_GLOBAL_BATCH
    expected_mfu = scheduled_per_second * 100.0 / (2 * 1_000.0)
    assert messages[0][-2] == pytest.approx(expected_mfu)
    assert messages[0][5] == pytest.approx((2 * 576 * CONTEXT_LENGTH - 7) / 2)
    assert counter.calls == [(0, 2 * 576)]
    assert callback.last_log_step == 2
    assert callback.last_log_time == 12.0


def test_no_nan_or_infinite_contract_constants() -> None:
    values = [
        cooldown.LEARNING_RATE,
        cooldown.WEIGHT_DECAY,
        cooldown.ADAM_EPSILON,
        cooldown.MAX_GRAD_NORM,
        *cooldown.ADAM_BETAS,
    ]
    assert all(math.isfinite(value) and value > 0 for value in values)
    assert cooldown.ADAM_BETAS[0] < 1 and cooldown.ADAM_BETAS[1] < 1
