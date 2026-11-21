"""Minimal Transformers-native SmolLM3 pretraining cooldown primitives."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq
import torch as t
import torch.distributed as dist
from loguru import logger
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from nanochat.common import get_peak_flops
from steering.sft.hf_checkpoint import repository_version
from steering.sft.hf_decoder import (
    HFDecoderModelAdapter,
    validate_hf_source_config,
    validate_hf_source_tokenizer,
    validate_loaded_hf_model,
)
from steering.sft.smollm3 import SMOLLM3_CONTRACT, SOURCE_END_OF_TEXT_ID

SOURCE_MODEL_ID = "HuggingFaceTB/SmolLM3-3B-checkpoints"
SOURCE_REVISION = "8089bf9b71ead724010855dd6fabd1deb90b70ff"
SOURCE_STEP = 200_000
CONTEXT_LENGTH = 4_096
WINDOW_LENGTH = CONTEXT_LENGTH + 1
CONTEXTS_PER_GLOBAL_BATCH = 576
TOKENS_PER_GLOBAL_BATCH = CONTEXTS_PER_GLOBAL_BATCH * CONTEXT_LENGTH
SOURCE_TOKENS = SOURCE_STEP * TOKENS_PER_GLOBAL_BATCH
TARGET_TOKENS = 524_288_000_000
COOLDOWN_STEPS = (TARGET_TOKENS - SOURCE_TOKENS) // TOKENS_PER_GLOBAL_BATCH
COOLDOWN_CONTEXTS = COOLDOWN_STEPS * CONTEXTS_PER_GLOBAL_BATCH
FINAL_TOKENS = SOURCE_TOKENS + COOLDOWN_STEPS * TOKENS_PER_GLOBAL_BATCH
TARGET_SHORTFALL = TARGET_TOKENS - FINAL_TOKENS
DEVICE_BATCH_SIZE = 3
WARMUP_STEPS = 2_000
LEARNING_RATE = 2.0e-4
WEIGHT_DECAY = 0.1
ADAM_BETAS = (0.9, 0.95)
ADAM_EPSILON = 1.0e-8
MAX_GRAD_NORM = 1.0
SEED = 6
ATTENTION_IMPLEMENTATION = "flash_attention_4"
EXPECTED_PARAMETER_COUNT = 3_075_098_624
EXPECTED_PARAMETER_NAMES = 326
PREPARED_SCHEMA_VERSION = 2
CHECKPOINT_MARKER = "cooldown_complete.json"
TOKEN_DTYPE = np.dtype("<u4")
INDEX_DTYPE = np.dtype("<u8")
_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")
_PARTIAL_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)\.partial")


@dataclass(frozen=True)
class PreparedShard:
    """Validated immutable prepared-shard coordinates."""

    stem: str
    token_path: Path
    index_path: Path
    token_count: int
    document_count: int
    context_count: int
    token_sha256: str
    index_sha256: str


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash one regular file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Publish one file atomically and durably in its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def experiment_arithmetic() -> dict[str, int]:
    """Return the single-source-of-truth cooldown arithmetic."""
    return {
        "source_step": SOURCE_STEP,
        "source_tokens": SOURCE_TOKENS,
        "target_tokens": TARGET_TOKENS,
        "tokens_per_global_batch": TOKENS_PER_GLOBAL_BATCH,
        "cooldown_steps": COOLDOWN_STEPS,
        "cooldown_contexts": COOLDOWN_CONTEXTS,
        "cooldown_tokens": COOLDOWN_STEPS * TOKENS_PER_GLOBAL_BATCH,
        "final_step": SOURCE_STEP + COOLDOWN_STEPS,
        "final_tokens": FINAL_TOKENS,
        "target_shortfall": TARGET_SHORTFALL,
    }


def gradient_accumulation_steps(world_size: int) -> int:
    """Derive accumulation while preserving the exact upstream global batch."""
    denominator = DEVICE_BATCH_SIZE * world_size
    if world_size <= 0 or CONTEXTS_PER_GLOBAL_BATCH % denominator:
        raise ValueError(
            f"world_size={world_size} must make {CONTEXTS_PER_GLOBAL_BATCH} divisible by "
            f"device_batch_size*world_size={denominator}"
        )
    return CONTEXTS_PER_GLOBAL_BATCH // denominator


def expected_scheduler_lr(completed_steps: int) -> float:
    """Exact standard-linear LR used by the next optimizer update."""
    if not 0 <= completed_steps <= COOLDOWN_STEPS:
        raise ValueError(f"completed_steps must be in [0, {COOLDOWN_STEPS}]")
    if completed_steps < WARMUP_STEPS:
        multiplier = float(completed_steps) / float(max(1, WARMUP_STEPS))
    else:
        multiplier = max(
            0.0,
            float(COOLDOWN_STEPS - completed_steps) / float(max(1, COOLDOWN_STEPS - WARMUP_STEPS)),
        )
    return LEARNING_RATE * multiplier


def tokenizer_identity(tokenizer: PreTrainedTokenizerBase) -> str:
    """Digest the complete low-level tokenizer serialization."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise TypeError("tokenizer must expose backend_tokenizer")
    serialized = backend.to_str()
    if not isinstance(serialized, str):
        raise TypeError("backend_tokenizer.to_str() must return str")
    return hashlib.sha256(serialized.encode()).hexdigest()


def validate_cooldown_source_config(config: Any) -> None:
    """Validate the SmolLM3 architecture details that affect pretraining numerics."""
    validate_hf_source_config(config)
    expected = {
        # Cooldown continues pretraining; only SFT admits instruct token identities.
        "bos_token_id": SMOLLM3_CONTRACT.source_bos_token_id,
        "eos_token_id": SMOLLM3_CONTRACT.source_eos_token_id,
        "pad_token_id": SMOLLM3_CONTRACT.source_pad_token_id,
        "rope_parameters": {"rope_theta": 50_000.0, "rope_type": "default"},
        "no_rope_layer_interval": 4,
        "no_rope_layers": [1, 1, 1, 0] * 9,
        "use_sliding_window": False,
        "sliding_window": None,
        "layer_types": ["full_attention"] * 36,
        "attention_bias": False,
        "mlp_bias": False,
        "rms_norm_eps": 1.0e-6,
        "hidden_act": "silu",
        "initializer_range": 0.02,
        "attention_dropout": 0.0,
    }
    for name, value in expected.items():
        actual = getattr(config, name, None)
        if actual != value:
            raise ValueError(
                f"SmolLM3 cooldown source config mismatch for {name}: "
                f"expected {value!r}, got {actual!r}"
            )


def load_source_tokenizer(*, local_files_only: bool) -> PreTrainedTokenizerBase:
    """Load and validate the immutable pretraining tokenizer without SFT mutation."""
    config = AutoConfig.from_pretrained(
        SOURCE_MODEL_ID,
        revision=SOURCE_REVISION,
        trust_remote_code=False,
        local_files_only=local_files_only,
    )
    validate_cooldown_source_config(config)
    contract = SMOLLM3_CONTRACT
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(
            SOURCE_MODEL_ID,
            revision=SOURCE_REVISION,
            trust_remote_code=False,
            local_files_only=local_files_only,
        ),
    )
    validate_hf_source_tokenizer(tokenizer, contract)
    if (
        tokenizer.bos_token_id is not None
        or tokenizer.eos_token_id != SOURCE_END_OF_TEXT_ID
        or tokenizer.pad_token_id is not None
    ):
        raise ValueError(
            "pretraining tokenizer structural ids must be "
            f"bos=None, eos={SOURCE_END_OF_TEXT_ID}, pad=None; got "
            f"bos={tokenizer.bos_token_id}, eos={tokenizer.eos_token_id}, "
            f"pad={tokenizer.pad_token_id}"
        )
    return tokenizer


def validate_parquet_inventory(parquet_dir: Path) -> list[Path]:
    """Require the exact 1,400-shard production source inventory."""
    paths = sorted(parquet_dir.glob("*.parquet"))
    expected = [parquet_dir / f"shard_{index:05d}.parquet" for index in range(1_400)]
    if paths != expected:
        missing = [path.name for path in expected if path not in set(paths)]
        extras = [path.name for path in paths if path not in set(expected)]
        raise ValueError(
            "cooldown input must contain exactly shard_00000..shard_01399; "
            f"missing={missing[:5]}, extras={extras[:5]}"
        )
    return paths


def _receipt_path(output_dir: Path, stem: str) -> Path:
    return output_dir / f"{stem}.json"


def _body_paths(output_dir: Path, stem: str) -> tuple[Path, Path]:
    return output_dir / f"{stem}.ds", output_dir / f"{stem}.ds.index"


def _context_count(token_count: int) -> int:
    """Count 4,097-token windows advanced by the 4,096-token causal stride."""
    return max(0, (token_count - 1) // CONTEXT_LENGTH)


def _source_stat(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _validate_receipt(
    source_path: Path,
    output_dir: Path,
    *,
    expected_tokenizer_sha256: str,
    verify_hashes: bool,
) -> dict[str, Any] | None:
    receipt_path = _receipt_path(output_dir, source_path.stem)
    token_path, index_path = _body_paths(output_dir, source_path.stem)
    present = [path.exists() for path in (receipt_path, token_path, index_path)]
    if not any(present):
        return None
    if not all(present):
        raise RuntimeError(f"incomplete prepared shard exists for {source_path.stem}")
    receipt = json.loads(receipt_path.read_text())
    expected = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "source": _source_stat(source_path),
        "tokenizer_sha256": expected_tokenizer_sha256,
        "eos_token_id": SOURCE_END_OF_TEXT_ID,
        "window_length": WINDOW_LENGTH,
        "context_stride": CONTEXT_LENGTH,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(
                f"prepared shard {source_path.stem} has mismatched {key}: "
                f"expected {value!r}, got {receipt.get(key)!r}"
            )
    files = receipt.get("files")
    if not isinstance(files, dict):
        raise RuntimeError(f"prepared shard {source_path.stem} has invalid files receipt")
    for path, key in ((token_path, "tokens"), (index_path, "index")):
        record = files.get(key)
        if not isinstance(record, dict) or record.get("bytes") != path.stat().st_size:
            raise RuntimeError(f"prepared shard {source_path.stem} {key} size mismatch")
        if verify_hashes and record.get("sha256") != sha256_file(path):
            raise RuntimeError(f"prepared shard {source_path.stem} {key} digest mismatch")
    token_count = receipt.get("token_count")
    document_count = receipt.get("document_count")
    context_count = receipt.get("context_count")
    if (
        not isinstance(token_count, int)
        or not isinstance(document_count, int)
        or not isinstance(context_count, int)
        or token_count < document_count
        or context_count != _context_count(token_count)
        or token_path.stat().st_size != token_count * TOKEN_DTYPE.itemsize
        or index_path.stat().st_size != document_count * INDEX_DTYPE.itemsize
    ):
        raise RuntimeError(f"prepared shard {source_path.stem} count invariant failed")
    return receipt


def _prepare_shard(
    source_path: Path,
    output_dir: Path,
    *,
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_sha256: str,
    batch_size: int,
) -> dict[str, Any]:
    existing = _validate_receipt(
        source_path,
        output_dir,
        expected_tokenizer_sha256=tokenizer_sha256,
        verify_hashes=True,
    )
    if existing is not None:
        logger.info(f"Reusing validated prepared shard {source_path.stem}")
        return existing
    token_path, index_path = _body_paths(output_dir, source_path.stem)
    token_tmp = token_path.with_name(f".{token_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    index_tmp = index_path.with_name(f".{index_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    token_count = 0
    document_count = 0
    try:
        parquet = pq.ParquetFile(source_path)
        with token_tmp.open("xb") as token_handle, index_tmp.open("xb") as index_handle:
            for record_batch in parquet.iter_batches(batch_size=batch_size, columns=["text"]):
                texts = record_batch.column("text").to_pylist()
                if any(not isinstance(text, str) for text in texts):
                    raise TypeError(f"{source_path.name} text column contains a non-string")
                encoded = tokenizer(
                    texts,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )["input_ids"]
                if len(encoded) != len(texts):
                    raise RuntimeError("tokenizer changed batch cardinality")
                for token_ids in encoded:
                    values = list(token_ids)
                    if any(
                        isinstance(token_id, bool)
                        or not isinstance(token_id, int)
                        or not 0 <= token_id < SMOLLM3_CONTRACT.tokenizer_vocab_size
                        for token_id in values
                    ):
                        raise ValueError("tokenizer emitted an invalid token id")
                    values.append(SOURCE_END_OF_TEXT_ID)
                    np.asarray(values, dtype=TOKEN_DTYPE).tofile(token_handle)
                    token_count += len(values)
                    document_count += 1
                    np.asarray([token_count], dtype=INDEX_DTYPE).tofile(index_handle)
            token_handle.flush()
            index_handle.flush()
            os.fsync(token_handle.fileno())
            os.fsync(index_handle.fileno())
        os.replace(token_tmp, token_path)
        os.replace(index_tmp, index_path)
        _fsync_directory(output_dir)
    finally:
        token_tmp.unlink(missing_ok=True)
        index_tmp.unlink(missing_ok=True)
    receipt = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "source": _source_stat(source_path),
        "tokenizer_sha256": tokenizer_sha256,
        "eos_token_id": SOURCE_END_OF_TEXT_ID,
        "window_length": WINDOW_LENGTH,
        "context_stride": CONTEXT_LENGTH,
        "document_count": document_count,
        "token_count": token_count,
        "context_count": _context_count(token_count),
        "files": {
            "tokens": {"bytes": token_path.stat().st_size, "sha256": sha256_file(token_path)},
            "index": {"bytes": index_path.stat().st_size, "sha256": sha256_file(index_path)},
        },
    }
    atomic_write_bytes(_receipt_path(output_dir, source_path.stem), _canonical_json(receipt))
    return receipt


def prepare_dataset(
    *,
    parquet_dir: Path,
    output_dir: Path,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
) -> Path:
    """Prepare the smallest deterministic train-shard prefix covering the horizon."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    inventory = validate_parquet_inventory(parquet_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_sha256 = tokenizer_identity(tokenizer)
    receipts: list[dict[str, Any]] = []
    cumulative_contexts = 0
    for source_path in inventory[:-1]:
        receipt = _prepare_shard(
            source_path,
            output_dir,
            tokenizer=tokenizer,
            tokenizer_sha256=tokenizer_sha256,
            batch_size=batch_size,
        )
        receipts.append(receipt)
        cumulative_contexts += int(receipt["context_count"])
        if cumulative_contexts >= COOLDOWN_CONTEXTS:
            break
    if cumulative_contexts < COOLDOWN_CONTEXTS:
        raise RuntimeError(
            f"training shards provide {cumulative_contexts:,} contexts, "
            f"but {COOLDOWN_CONTEXTS:,} are required"
        )
    manifest = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "source_model_id": SOURCE_MODEL_ID,
        "source_revision": SOURCE_REVISION,
        "tokenizer_sha256": tokenizer_sha256,
        "eos_token_id": SOURCE_END_OF_TEXT_ID,
        "context_length": CONTEXT_LENGTH,
        "window_length": WINDOW_LENGTH,
        "context_stride": CONTEXT_LENGTH,
        "required_contexts": COOLDOWN_CONTEXTS,
        "available_contexts": cumulative_contexts,
        "source_inventory": {
            "parquet_dir": str(parquet_dir.resolve()),
            "total_shards": 1_400,
            "train_shards": 1_399,
            "validation_shard": "shard_01399.parquet",
        },
        "shards": receipts,
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing != manifest:
            raise RuntimeError("existing prepared manifest does not match reconstructed manifest")
    else:
        atomic_write_bytes(manifest_path, _canonical_json(manifest))
    return manifest_path


def load_prepared_manifest(
    prepared_dir: Path,
    *,
    verify_hashes: bool,
) -> tuple[dict[str, Any], list[PreparedShard]]:
    """Validate a complete manifest and its selected immutable shard bodies."""
    manifest_path = prepared_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "source_model_id": SOURCE_MODEL_ID,
        "source_revision": SOURCE_REVISION,
        "eos_token_id": SOURCE_END_OF_TEXT_ID,
        "context_length": CONTEXT_LENGTH,
        "window_length": WINDOW_LENGTH,
        "context_stride": CONTEXT_LENGTH,
        "required_contexts": COOLDOWN_CONTEXTS,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"prepared manifest {key} must be {value!r}; got {manifest.get(key)!r}"
            )
    records = manifest.get("shards")
    if not isinstance(records, list) or not records:
        raise ValueError("prepared manifest shards must be a nonempty list")
    prepared: list[PreparedShard] = []
    for index, receipt in enumerate(records):
        if not isinstance(receipt, dict):
            raise ValueError("prepared manifest shard record must be an object")
        stem = f"shard_{index:05d}"
        source = receipt.get("source")
        if not isinstance(source, dict) or Path(str(source.get("path", ""))).stem != stem:
            raise ValueError(f"prepared shard order mismatch at {stem}")
        token_path, index_path = _body_paths(prepared_dir, stem)
        files = receipt.get("files")
        if not isinstance(files, dict):
            raise ValueError(f"prepared shard {stem} has invalid file inventory")
        token_record, index_record = files.get("tokens"), files.get("index")
        if not isinstance(token_record, dict) or not isinstance(index_record, dict):
            raise ValueError(f"prepared shard {stem} has invalid file records")
        for path, record in ((token_path, token_record), (index_path, index_record)):
            if path.stat().st_size != record.get("bytes"):
                raise ValueError(f"prepared body size mismatch: {path}")
            if verify_hashes and sha256_file(path) != record.get("sha256"):
                raise ValueError(f"prepared body digest mismatch: {path}")
        token_count = int(receipt["token_count"])
        document_count = int(receipt["document_count"])
        context_count = int(receipt["context_count"])
        if (
            token_count < document_count
            or context_count != _context_count(token_count)
            or token_path.stat().st_size != token_count * TOKEN_DTYPE.itemsize
            or index_path.stat().st_size != document_count * INDEX_DTYPE.itemsize
        ):
            raise ValueError(f"prepared shard {stem} count invariant failed")
        prepared.append(
            PreparedShard(
                stem=stem,
                token_path=token_path,
                index_path=index_path,
                token_count=token_count,
                document_count=document_count,
                context_count=context_count,
                token_sha256=str(token_record["sha256"]),
                index_sha256=str(index_record["sha256"]),
            )
        )
    available = sum(shard.context_count for shard in prepared)
    if available != manifest.get("available_contexts") or available < COOLDOWN_CONTEXTS:
        raise ValueError("prepared manifest context total is inconsistent or insufficient")
    return manifest, prepared


class CooldownDataset(Dataset[dict[str, t.Tensor]]):
    """Sequential 4,097-token windows with the exact 4,096-token causal stride."""

    def __init__(self, shards: Sequence[PreparedShard], *, length: int = COOLDOWN_CONTEXTS) -> None:
        if length <= 0 or length > sum(shard.context_count for shard in shards):
            raise ValueError("dataset length must be positive and fit the prepared contexts")
        self.shards = tuple(shards)
        self.length = length
        cumulative = 0
        self.cumulative_contexts: list[int] = []
        for shard in self.shards:
            cumulative += shard.context_count
            self.cumulative_contexts.append(cumulative)
        self._open_shard_index: int | None = None
        self._tokens: np.memmap | None = None
        self._document_ends: np.memmap | None = None

    def __len__(self) -> int:
        return self.length

    def _close(self) -> None:
        for array in (self._tokens, self._document_ends):
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self._tokens = None
        self._document_ends = None
        self._open_shard_index = None

    def __getstate__(self) -> dict[str, Any]:
        self._close()
        return self.__dict__.copy()

    def __del__(self) -> None:
        self._close()

    def _open(self, shard_index: int) -> None:
        if self._open_shard_index == shard_index:
            return
        self._close()
        shard = self.shards[shard_index]
        self._tokens = np.memmap(shard.token_path, mode="r", dtype=TOKEN_DTYPE)
        self._document_ends = np.memmap(shard.index_path, mode="r", dtype=INDEX_DTYPE)
        if (
            len(self._tokens) != shard.token_count
            or len(self._document_ends) != shard.document_count
        ):
            self._close()
            raise RuntimeError(f"prepared shard changed after validation: {shard.stem}")
        self._open_shard_index = shard_index

    def __getitem__(self, index: int) -> dict[str, t.Tensor]:
        if not 0 <= index < self.length:
            raise IndexError(index)
        shard_index = bisect_right(self.cumulative_contexts, index)
        prior = 0 if shard_index == 0 else self.cumulative_contexts[shard_index - 1]
        local_context = index - prior
        self._open(shard_index)
        assert self._tokens is not None and self._document_ends is not None
        start = local_context * CONTEXT_LENGTH
        stop = start + WINDOW_LENGTH
        raw = np.asarray(self._tokens[start:stop], dtype=np.int64)
        if raw.shape != (WINDOW_LENGTH,):
            raise RuntimeError("prepared context window is truncated")
        positions = np.arange(WINDOW_LENGTH, dtype=np.int64)
        left = int(np.searchsorted(self._document_ends, start, side="right"))
        right = int(np.searchsorted(self._document_ends, stop - 1, side="right"))
        previous_boundary = 0
        for absolute_end in self._document_ends[left:right]:
            boundary = int(absolute_end) - start
            if 0 < boundary < WINDOW_LENGTH:
                positions[boundary:] -= boundary - previous_boundary
                previous_boundary = boundary
        labels = raw[1:].copy()
        shifted_positions = positions[1:]
        labels[shifted_positions == 0] = -100
        return {
            "input_ids": t.from_numpy(raw[:-1].copy()),
            "position_ids": t.from_numpy(positions[:-1].copy()),
            "labels": t.from_numpy(labels),
        }

    def supervised_items(self, start: int, stop: int) -> int:
        """Count exact non-masked next-token labels in a global context interval."""
        if not 0 <= start <= stop <= self.length:
            raise ValueError(
                f"context interval must satisfy 0 <= {start} <= {stop} <= {self.length}"
            )
        masked = 0
        cursor = start
        while cursor < stop:
            shard_index = bisect_right(self.cumulative_contexts, cursor)
            prior = 0 if shard_index == 0 else self.cumulative_contexts[shard_index - 1]
            local_start = cursor - prior
            local_stop = min(stop - prior, self.shards[shard_index].context_count)
            ends = np.memmap(
                self.shards[shard_index].index_path,
                mode="r",
                dtype=INDEX_DTYPE,
            )
            try:
                left = int(np.searchsorted(ends, local_start * CONTEXT_LENGTH, side="right"))
                right = int(np.searchsorted(ends, local_stop * CONTEXT_LENGTH, side="right"))
                masked += right - left
            finally:
                mmap = getattr(ends, "_mmap", None)
                if mmap is not None:
                    mmap.close()
            cursor = prior + local_stop
        return (stop - start) * CONTEXT_LENGTH - masked


def cooldown_collator(examples: Sequence[Mapping[str, t.Tensor]]) -> dict[str, t.Tensor]:
    """Flatten exactly three independent contexts into the FA4 varlen layout."""
    if len(examples) != DEVICE_BATCH_SIZE:
        raise ValueError(f"collator requires exactly {DEVICE_BATCH_SIZE} contexts")
    expected_keys = {"input_ids", "position_ids", "labels"}
    for example in examples:
        if set(example) != expected_keys:
            raise ValueError(f"collator example keys must be {sorted(expected_keys)}")
        if any(example[key].shape != (CONTEXT_LENGTH,) for key in expected_keys):
            raise ValueError(f"every collator tensor must have shape ({CONTEXT_LENGTH},)")
        if example["position_ids"][0].item() != 0:
            raise ValueError("every context must start with position_id zero")
        if not t.equal(example["labels"][:-1].eq(-100), example["position_ids"][1:].eq(0)):
            raise ValueError("label mask and document-boundary positions disagree")
    batch = {
        key: t.cat([example[key] for example in examples]).unsqueeze(0) for key in expected_keys
    }
    batch["shift_labels"] = batch["labels"]
    if batch["input_ids"].shape != (1, DEVICE_BATCH_SIZE * CONTEXT_LENGTH):
        raise RuntimeError("flattened cooldown batch shape drifted")
    return batch


class CooldownTrainer(Trainer):
    """Trainer with only the upstream decay policy and finite-norm guard changed."""

    def get_decay_parameter_names(self, model: t.nn.Module) -> list[str]:
        names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if len(names) != len(set(names)):
            raise RuntimeError("trainable parameter names are not unique")
        embedding_name = "model.embed_tokens.weight"
        if embedding_name not in names:
            raise ValueError(f"missing exact no-decay parameter {embedding_name!r}")
        return [name for name in names if name != embedding_name]

    def _clip_grad_norm(self, model: t.nn.Module) -> t.Tensor:
        norm = super()._clip_grad_norm(model)
        value = norm if isinstance(norm, t.Tensor) else t.tensor(norm)
        if not bool(t.isfinite(value).all().item()):
            raise FloatingPointError(f"non-finite pre-clip gradient norm: {value}")
        return value


def semantic_payload(
    *, manifest_path: Path, world_size: int, local_world_size: int
) -> dict[str, Any]:
    """Build the fail-closed checkpoint-resume identity."""
    repository_root = Path(__file__).resolve().parent.parent
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise RuntimeError("SmolLM3 cooldown training requires a clean committed repository")
    repository = repository_version()
    if repository.get("git_revision") is None:
        raise RuntimeError("SmolLM3 cooldown could not resolve the repository Git revision")
    lock_path = repository_root / "uv.lock"
    return {
        "format": "smollm3_pretraining_cooldown",
        "format_version": 1,
        "source": {"model_id": SOURCE_MODEL_ID, "revision": SOURCE_REVISION},
        "data_manifest_sha256": sha256_file(manifest_path),
        "arithmetic": experiment_arithmetic(),
        "context_length": CONTEXT_LENGTH,
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "precision": {"parameters": "float32", "autocast": "bfloat16", "tf32": False},
        "execution": {"gradient_checkpointing": True, "torch_compile": False},
        "optimizer": {
            "name": "adamw_torch_fused",
            "learning_rate": LEARNING_RATE,
            "betas": list(ADAM_BETAS),
            "epsilon": ADAM_EPSILON,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "warmup_steps": WARMUP_STEPS,
        },
        "batch": {
            "contexts_per_device": DEVICE_BATCH_SIZE,
            "contexts_per_global_batch": CONTEXTS_PER_GLOBAL_BATCH,
            "world_size": world_size,
            "local_world_size": local_world_size,
            "gradient_accumulation_steps": gradient_accumulation_steps(world_size),
        },
        "seed": SEED,
        "repository": repository,
        "uv_lock_sha256": sha256_file(lock_path),
    }


def semantic_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _checkpoint_inventory(path: Path, *, world_size: int) -> dict[str, int]:
    required = ["optimizer.pt", "scheduler.pt", "trainer_state.json", "training_args.bin"]
    required.extend(
        ["rng_state.pth"]
        if world_size == 1
        else [f"rng_state_{rank}.pth" for rank in range(world_size)]
    )
    missing = [name for name in required if not (path / name).is_file()]
    model_files = sorted(path.glob("model*.safetensors"))
    if not model_files and (path / "pytorch_model.bin").is_file():
        model_files = [path / "pytorch_model.bin"]
    if not model_files:
        missing.append("model weights")
    if missing:
        raise RuntimeError(f"checkpoint {path.name} is incomplete; missing={missing}")
    paths = [path / name for name in required] + model_files
    index = path / "model.safetensors.index.json"
    if index.exists():
        paths.append(index)
    return {candidate.name: candidate.stat().st_size for candidate in sorted(paths)}


class CooldownCallback(TrainerCallback):
    """Production telemetry, bounded canaries, and marker-last checkpoints."""

    def __init__(
        self,
        *,
        semantic: Mapping[str, Any],
        canary_steps: int | None,
        flops_per_token: float,
        peak_flops: float,
        dataset: CooldownDataset,
    ) -> None:
        self.semantic = dict(semantic)
        self.digest = semantic_digest(self.semantic)
        self.canary_steps = canary_steps
        self.flops_per_token = flops_per_token
        self.peak_flops = peak_flops
        self.dataset = dataset
        self.last_log_time = time.monotonic()
        self.last_log_step = 0

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self.last_log_time = time.monotonic()
        self.last_log_step = state.global_step

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        if self.canary_steps is not None and state.global_step >= self.canary_steps:
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        now = time.monotonic()
        step_delta = state.global_step - self.last_log_step
        elapsed = now - self.last_log_time
        if step_delta <= 0 or elapsed <= 0:
            return
        scheduled_per_second = step_delta * TOKENS_PER_GLOBAL_BATCH / elapsed
        supervised = self.dataset.supervised_items(
            self.last_log_step * CONTEXTS_PER_GLOBAL_BATCH,
            state.global_step * CONTEXTS_PER_GLOBAL_BATCH,
        )
        supervised_per_second = supervised / elapsed
        mfu = scheduled_per_second * self.flops_per_token / (args.world_size * self.peak_flops)
        remaining_seconds = (COOLDOWN_STEPS - state.global_step) * elapsed / step_delta
        logger.info(
            "cooldown step={} absolute_step={} nominal_tokens={} scheduled_tok_s={:.1f} "
            "supervised_tok_s={:.1f} supervised_items={} mfu_estimate={:.4f} eta_seconds={:.1f}",
            state.global_step,
            SOURCE_STEP + state.global_step,
            SOURCE_TOKENS + state.global_step * TOKENS_PER_GLOBAL_BATCH,
            scheduled_per_second,
            supervised_per_second,
            supervised,
            mfu,
            remaining_seconds,
        )
        self.last_log_time = now
        self.last_log_step = state.global_step

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if args.process_index == 0:
            if args.output_dir is None:
                raise RuntimeError("Trainer output_dir unexpectedly resolved to null")
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            inventory = _checkpoint_inventory(checkpoint, world_size=args.world_size)
            marker = {
                "schema_version": 1,
                "completed_cooldown_step": state.global_step,
                "absolute_step": SOURCE_STEP + state.global_step,
                "nominal_tokens": SOURCE_TOKENS + state.global_step * TOKENS_PER_GLOBAL_BATCH,
                "semantic_digest": self.digest,
                "semantic": self.semantic,
                "files": inventory,
            }
            atomic_write_bytes(checkpoint / CHECKPOINT_MARKER, _canonical_json(marker))
        if dist.is_available() and dist.is_initialized():
            dist.barrier()


def resolve_resume_checkpoint(output_dir: Path, *, expected_digest: str) -> Path | None:
    """Quarantine partial checkpoints and return the latest matching complete one."""
    complete: list[tuple[int, Path]] = []
    for path in sorted(output_dir.glob("checkpoint-*")):
        if not path.is_dir():
            raise RuntimeError(f"checkpoint-like path is not a directory: {path}")
        if _PARTIAL_CHECKPOINT_RE.fullmatch(path.name):
            continue
        match = _CHECKPOINT_RE.fullmatch(path.name)
        if match is None:
            raise RuntimeError(f"malformed checkpoint directory: {path.name}")
        marker_path = path / CHECKPOINT_MARKER
        if not marker_path.exists():
            destination = path.with_name(f"{path.name}.partial")
            if destination.exists():
                raise RuntimeError(f"partial-checkpoint quarantine collision: {destination}")
            os.replace(path, destination)
            continue
        marker = json.loads(marker_path.read_text())
        if marker.get("semantic_digest") != expected_digest:
            raise RuntimeError(f"checkpoint semantic identity mismatch: {path}")
        step = int(match.group(1))
        if marker.get("completed_cooldown_step") != step:
            raise RuntimeError(f"checkpoint marker step mismatch: {path}")
        _checkpoint_inventory(path, world_size=int(marker["semantic"]["batch"]["world_size"]))
        complete.append((step, path))
    return None if not complete else max(complete)[1]


def build_training_arguments(
    *,
    output_dir: Path,
    world_size: int,
    logging_steps: int,
    save_steps: int,
    dataloader_workers: int,
) -> TrainingArguments:
    """Construct the fixed production Trainer arguments from derived values."""
    if min(logging_steps, save_steps) <= 0 or dataloader_workers < 0:
        raise ValueError("logging/save steps must be positive and workers nonnegative")
    return TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        per_device_train_batch_size=DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=gradient_accumulation_steps(world_size),
        max_steps=COOLDOWN_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="linear",
        warmup_steps=WARMUP_STEPS,
        optim="adamw_torch_fused",
        weight_decay=WEIGHT_DECAY,
        adam_beta1=ADAM_BETAS[0],
        adam_beta2=ADAM_BETAS[1],
        adam_epsilon=ADAM_EPSILON,
        max_grad_norm=MAX_GRAD_NORM,
        bf16=True,
        tf32=False,
        gradient_checkpointing=True,
        torch_compile=False,
        use_cache=False,
        logging_strategy="steps",
        logging_steps=logging_steps,
        logging_first_step=True,
        logging_nan_inf_filter=False,
        report_to="none",
        eval_strategy="no",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        save_only_model=False,
        push_to_hub=False,
        ignore_data_skip=False,
        seed=SEED,
        data_seed=SEED,
        dataloader_drop_last=True,
        dataloader_num_workers=dataloader_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=dataloader_workers > 0,
        dataloader_prefetch_factor=2 if dataloader_workers > 0 else None,
        remove_unused_columns=False,
        label_names=["labels", "shift_labels"],
        train_sampling_strategy="sequential",
        ddp_find_unused_parameters=False,
        average_tokens_across_devices=True,
        disable_tqdm=True,
    )


def load_source_model(
    *, local_files_only: bool, local_rank: int, local_world_size: int
) -> PreTrainedModel:
    """Load one FP32 source model per local-rank slot to cap host-memory spikes."""
    if local_world_size <= 0 or not 0 <= local_rank < local_world_size:
        raise ValueError("invalid local-rank geometry")
    model: PreTrainedModel | None = None
    for slot in range(local_world_size):
        if local_rank == slot:
            model = cast(
                PreTrainedModel,
                AutoModelForCausalLM.from_pretrained(
                    SOURCE_MODEL_ID,
                    revision=SOURCE_REVISION,
                    trust_remote_code=False,
                    local_files_only=local_files_only,
                    dtype=t.float32,
                    low_cpu_mem_usage=True,
                    attn_implementation=ATTENTION_IMPLEMENTATION,
                ),
            )
            model.config.use_cache = False
            validate_loaded_hf_model(model, SMOLLM3_CONTRACT)
            names = [
                name for name, parameter in model.named_parameters() if parameter.requires_grad
            ]
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            if (
                parameter_count != EXPECTED_PARAMETER_COUNT
                or len(names) != EXPECTED_PARAMETER_NAMES
            ):
                raise ValueError(
                    "SmolLM3 cooldown source parameter inventory mismatch: "
                    f"expected {EXPECTED_PARAMETER_COUNT:,} parameters/{EXPECTED_PARAMETER_NAMES} "
                    f"names, got {parameter_count:,}/{len(names)}"
                )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    if model is None:
        raise RuntimeError("source model loading completed without a model")
    model.train()
    return model


def run_training(cfg: Any) -> None:
    """Validate configuration, construct standard Trainer, and train/resume."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not t.cuda.is_available():
        raise RuntimeError("SmolLM3 cooldown production training requires CUDA")
    t.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank() if dist.is_initialized() else 0
    if str(cfg.source.model_id) != SOURCE_MODEL_ID or str(cfg.source.revision) != SOURCE_REVISION:
        raise ValueError("source model identity is immutable")
    prepared_dir = Path(str(cfg.data.prepared_dir)).resolve()
    output_dir = Path(str(cfg.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    verify_hashes = bool(cfg.data.verify_hashes)
    manifest, shards = load_prepared_manifest(
        prepared_dir, verify_hashes=verify_hashes and rank == 0
    )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    tokenizer = load_source_tokenizer(local_files_only=bool(cfg.source.local_files_only))
    if tokenizer_identity(tokenizer) != manifest["tokenizer_sha256"]:
        raise ValueError("runtime tokenizer differs from prepared data tokenizer")
    manifest_path = prepared_dir / "manifest.json"
    semantic = semantic_payload(
        manifest_path=manifest_path,
        world_size=world_size,
        local_world_size=local_world_size,
    )
    digest = semantic_digest(semantic)
    resume = resolve_resume_checkpoint(output_dir, expected_digest=digest) if rank == 0 else None
    if dist.is_available() and dist.is_initialized():
        objects: list[Any] = [None if resume is None else str(resume)]
        dist.broadcast_object_list(objects, src=0)
        resume = None if objects[0] is None else Path(str(objects[0]))
    model = load_source_model(
        local_files_only=bool(cfg.source.local_files_only),
        local_rank=local_rank,
        local_world_size=local_world_size,
    )
    args = build_training_arguments(
        output_dir=output_dir,
        world_size=world_size,
        logging_steps=int(cfg.logging_steps),
        save_steps=int(cfg.save_steps),
        dataloader_workers=int(cfg.data.dataloader_workers),
    )
    flops = HFDecoderModelAdapter(model, max_seq_len=CONTEXT_LENGTH).estimate_flops()
    peak = get_peak_flops(t.cuda.get_device_name(local_rank))
    canary_steps = None if cfg.canary_steps is None else int(cfg.canary_steps)
    if canary_steps is not None and not 0 < canary_steps <= COOLDOWN_STEPS:
        raise ValueError(f"canary_steps must be null or in [1, {COOLDOWN_STEPS}]")
    dataset = CooldownDataset(shards)
    callback = CooldownCallback(
        semantic=semantic,
        canary_steps=canary_steps,
        flops_per_token=flops,
        peak_flops=peak,
        dataset=dataset,
    )
    trainer = CooldownTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=cooldown_collator,
        callbacks=[callback],
    )
    trainer.train(resume_from_checkpoint=None if resume is None else str(resume))
    trainer.save_model(str(output_dir / "final"))


__all__ = [
    "ADAM_BETAS",
    "ADAM_EPSILON",
    "ATTENTION_IMPLEMENTATION",
    "CHECKPOINT_MARKER",
    "CONTEXT_LENGTH",
    "CONTEXTS_PER_GLOBAL_BATCH",
    "COOLDOWN_CONTEXTS",
    "COOLDOWN_STEPS",
    "CooldownCallback",
    "CooldownDataset",
    "CooldownTrainer",
    "DEVICE_BATCH_SIZE",
    "EXPECTED_PARAMETER_COUNT",
    "EXPECTED_PARAMETER_NAMES",
    "FINAL_TOKENS",
    "LEARNING_RATE",
    "MAX_GRAD_NORM",
    "PreparedShard",
    "SEED",
    "SOURCE_MODEL_ID",
    "SOURCE_REVISION",
    "SOURCE_TOKENS",
    "TARGET_SHORTFALL",
    "TARGET_TOKENS",
    "TOKENS_PER_GLOBAL_BATCH",
    "WARMUP_STEPS",
    "WEIGHT_DECAY",
    "WINDOW_LENGTH",
    "atomic_write_bytes",
    "build_training_arguments",
    "cooldown_collator",
    "expected_scheduler_lr",
    "experiment_arithmetic",
    "gradient_accumulation_steps",
    "load_prepared_manifest",
    "load_source_model",
    "load_source_tokenizer",
    "prepare_dataset",
    "resolve_resume_checkpoint",
    "run_training",
    "semantic_digest",
    "semantic_payload",
    "sha256_file",
    "tokenizer_identity",
    "validate_cooldown_source_config",
    "validate_parquet_inventory",
]
