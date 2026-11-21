"""Opt-in full SmolLM3-3B and production-data CUDA acceptance gates.

Run on an H200 with the pinned model snapshot already cached:

    RUN_SMOLLM3_COOLDOWN_REAL_GPU=1 uv run pytest -m very_slow \
        tests/steering/test_smollm3_cooldown_real.py -q

The module intentionally has no synthetic fallback. A selected test either
uses the immutable full checkpoint and real production parquet text or fails.

The separate `smollm3_cooldown_ddp_real.py` harness runs the full stock Trainer
on two H200s, first through checkpoint 1 and then by resuming through checkpoint
2 with the production scheduler horizon unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

import steering.smollm3_cooldown as cooldown
from steering.parquet_io import read_parquet_texts
from steering.sft.hf_decoder import validate_loaded_hf_model
from steering.sft.smollm3 import SMOLLM3_CONTRACT

pytestmark = [
    pytest.mark.very_slow,
    pytest.mark.requires_feature_data,
    pytest.mark.skipif(
        os.environ.get("RUN_SMOLLM3_COOLDOWN_REAL_GPU") != "1",
        reason="set RUN_SMOLLM3_COOLDOWN_REAL_GPU=1 for the full H200 acceptance gate",
    ),
]

PRODUCTION_PARQUETS = Path(
    "/orcd/pool/005/henryac/nano-gpt-pretrain-steer/"
    "full3b-clustered-c2048-20260823T194759Z/features/s128/l22/a_res/"
    "p_max_pool/f65k/m_gemma-3-4b-pt/base/context_independent/no_bos/"
    "splice/cut_frac0.15/seed42/_parquets"
)


@pytest.fixture(scope="module")
def source_tokenizer():
    assert t.cuda.is_available(), "the full acceptance gate requires CUDA"
    return cooldown.load_source_tokenizer(local_files_only=True)


@pytest.fixture(scope="module")
def full_model():
    assert t.cuda.is_available(), "the full acceptance gate requires CUDA"
    model = cooldown.load_source_model(local_files_only=True, local_rank=0, local_world_size=1)
    t.nn.Module.to(model, device=t.device("cuda"))
    yield model
    t.nn.Module.to(model, device=t.device("cpu"))
    t.cuda.empty_cache()


def _real_prepared_contexts(tmp_path: Path, tokenizer) -> cooldown.CooldownDataset:
    source = PRODUCTION_PARQUETS / "shard_00000.parquet"
    assert source.is_file(), f"missing production shard: {source}"
    parquet = pq.ParquetFile(source)
    texts: list[str] = []
    token_count = 0
    for row_group_index in range(parquet.num_row_groups):
        group_texts = (
            parquet.read_row_group(row_group_index, columns=["text"]).column("text").to_pylist()
        )
        texts.extend(group_texts)
        encoded = tokenizer(
            group_texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        token_count += sum(len(tokens) + 1 for tokens in encoded)
        if token_count >= cooldown.DEVICE_BATCH_SIZE * cooldown.CONTEXT_LENGTH + 1:
            break
    assert token_count >= cooldown.DEVICE_BATCH_SIZE * cooldown.CONTEXT_LENGTH + 1
    subset = tmp_path / "shard_00000.parquet"
    pq.write_table(pa.table({"text": texts}), subset, row_group_size=max(1, len(texts) // 3))
    output = tmp_path / "prepared"
    output.mkdir()
    receipt = cooldown._prepare_shard(
        subset,
        output,
        tokenizer=tokenizer,
        tokenizer_sha256=cooldown.tokenizer_identity(tokenizer),
        batch_size=64,
    )
    shard = cooldown.PreparedShard(
        stem="shard_00000",
        token_path=output / "shard_00000.ds",
        index_path=output / "shard_00000.ds.index",
        token_count=int(receipt["token_count"]),
        document_count=int(receipt["document_count"]),
        context_count=int(receipt["context_count"]),
        token_sha256=str(receipt["files"]["tokens"]["sha256"]),
        index_sha256=str(receipt["files"]["index"]["sha256"]),
    )
    return cooldown.CooldownDataset([shard], length=cooldown.DEVICE_BATCH_SIZE)


def test_real_production_parquet_subset_matches_canonical_reader_and_tokenizer(
    tmp_path: Path, source_tokenizer
) -> None:
    source = PRODUCTION_PARQUETS / "shard_00000.parquet"
    assert source.is_file()
    canonical = read_parquet_texts(source)
    assert canonical and all(isinstance(text, str) for text in canonical)
    dataset = _real_prepared_contexts(tmp_path, source_tokenizer)
    batch = cooldown.cooldown_collator([dataset[index] for index in range(3)])
    assert batch["input_ids"].shape == (1, 12_288)
    assert batch["position_ids"][0, 0].item() == 0
    assert batch["shift_labels"] is batch["labels"]
    assert batch["labels"].ne(-100).sum().item() > 12_000
    assert batch["input_ids"].min().item() >= 0
    assert batch["input_ids"].max().item() < 128_256


def test_full_checkpoint_exact_parameter_and_decay_contract(full_model) -> None:
    validate_loaded_hf_model(full_model, SMOLLM3_CONTRACT)
    assert sum(parameter.numel() for parameter in full_model.parameters()) == 3_075_098_624
    names = [name for name, parameter in full_model.named_parameters() if parameter.requires_grad]
    assert len(names) == 326
    trainer = object.__new__(cooldown.CooldownTrainer)
    decay = cooldown.CooldownTrainer.get_decay_parameter_names(trainer, full_model)
    assert len(decay) == 325
    assert set(names) - set(decay) == {"model.embed_tokens.weight"}
    assert all(parameter.dtype == t.float32 for parameter in full_model.parameters())
    assert full_model.get_input_embeddings().weight is full_model.get_output_embeddings().weight


def test_full_fa4_bf16_forward_backward_matches_independent_loss(
    tmp_path: Path,
    source_tokenizer,
    full_model,
) -> None:
    dataset = _real_prepared_contexts(tmp_path, source_tokenizer)
    batch = {
        key: value.cuda()
        for key, value in cooldown.cooldown_collator([dataset[i] for i in range(3)]).items()
    }
    assert "attention_mask" not in batch
    t.cuda.reset_peak_memory_stats()
    full_model.zero_grad(set_to_none=True)
    with t.autocast("cuda", dtype=t.bfloat16):
        output = full_model(**batch, use_cache=False)
    reference = t.nn.functional.cross_entropy(
        output.logits.float().reshape(-1, full_model.config.vocab_size),
        batch["shift_labels"].reshape(-1),
        ignore_index=-100,
    )
    t.testing.assert_close(output.loss, reference, rtol=0, atol=0)
    assert t.isfinite(output.loss)
    output.loss.backward()
    norm = t.nn.utils.clip_grad_norm_(full_model.parameters(), cooldown.MAX_GRAD_NORM)
    assert t.isfinite(norm)
    assert norm.item() > 0
    assert t.cuda.max_memory_allocated() < t.cuda.get_device_properties(0).total_memory
    del output, reference, batch
    full_model.zero_grad(set_to_none=True)
    t.cuda.empty_cache()


def test_prepared_tokens_are_little_endian_uint32_and_indices_uint64(
    tmp_path: Path, source_tokenizer
) -> None:
    dataset = _real_prepared_contexts(tmp_path, source_tokenizer)
    shard = dataset.shards[0]
    tokens = np.memmap(shard.token_path, mode="r", dtype="<u4")
    ends = np.memmap(shard.index_path, mode="r", dtype="<u8")
    assert len(tokens) == shard.token_count
    assert len(ends) == shard.document_count
    assert np.all(np.diff(ends) > 0)
    assert int(ends[-1]) == len(tokens)
    assert np.all(tokens[np.asarray(ends, dtype=np.int64) - 1] == cooldown.SOURCE_END_OF_TEXT_ID)
