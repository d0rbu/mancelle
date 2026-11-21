"""Native RustBPE SFT prompt steering; no SAE or fitted intervention is used."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import torch as t
from loguru import logger
from omegaconf import DictConfig

from nanochat import flash_attention
from nanochat.atomic_io import atomic_json_dump, atomic_write_parquet
from nanochat.checkpoint_manager import build_model
from nanochat.common import COMPUTE_DTYPE, get_base_dir
from nanochat.tokenizer import RustBPETokenizer
from steering.conditional.encoder_decoder.sm12_fa4 import (
    admit_project_attention,
    uses_native_sm12_fa4,
)
from steering.eval.axbench_upstream.native_contract import (
    FORMAT,
    METHOD,
    POLICY,
    native_input_frame,
    sha256,
)
from steering.eval.chat_generation import ChatGenerationResult, render_messages_for_chat_completion
from steering.eval.inference_engine import InferenceEngine
from steering.immutable_io import install_bytes_once


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def native_runtime_settings(
    *, full_prefix_device: t.device | None = None, cached_device: t.device | None = None,
) -> dict[str, Any]:
    # Match flash_attn_with_kvcache, whose dispatch differs from uncached attention.
    if flash_attention._override_impl == flash_attention.FlashAttnImpl.FA3 and not flash_attention.HAS_FA3:
        raise ValueError("explicit FA3 attention override is unavailable")
    uses_fa3 = flash_attention.HAS_FA3 and flash_attention._override_impl != flash_attention.FlashAttnImpl.SDPA
    device = full_prefix_device or cached_device
    if device is not None and uses_fa3 and (device.type != "cuda" or t.cuda.get_device_capability(device)[0] != 9):
        raise ValueError("selected device cannot use imported FA3 cache backend; launch with NANOCHAT_ATTN_IMPL=sdpa")
    if full_prefix_device is not None:
        admit_project_attention(device=full_prefix_device)
    settings = {"torch_version": t.__version__, "matmul_precision": t.get_float32_matmul_precision(),
                "compute_dtype": str(COMPUTE_DTYPE), "attention_backend": "fa3" if uses_fa3 else "sdpa"}
    if full_prefix_device is not None:
        # Record the resolved full-prefix path, not just KV attention.
        if uses_native_sm12_fa4(full_prefix_device):
            backend = "native_fa4"
        elif flash_attention._use_flex():
            backend = "flex_fa4" if flash_attention.HAS_FA4_BACKEND else "flex_triton"
        else:
            backend = "fa3" if flash_attention._use_fa3() else "sdpa"
        settings["full_prefix_attention_backend"] = backend
    return settings


def validate_native_checkpoint_metadata(meta: Mapping[str, Any], step: int) -> None:
    """Share the native SFT eligibility checks without reading or hashing weights."""
    if type(step) is not int or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if meta.get("step") != step or "sft_loss_mask" not in meta:
        raise ValueError("native decoder evaluation requires the explicit SFT checkpoint step")
    config = meta.get("model_config", {})
    if not config or any(key in meta for key in ("encoder_config", "cross_attention_config")):
        raise ValueError("native decoder evaluation requires decoder-only model_config")
    user = meta.get("user_config", {})
    if user.get("model_backend", "nanochat") != "nanochat" or not user.get("mixture"):
        raise ValueError("native decoder evaluation requires native SFT mixture provenance")


def checkpoint_identity(checkpoint_dir: Path, step: int, tokenizer_path: Path) -> dict[str, Any]:
    if type(step) is not int or step < 0:
        raise ValueError("step must be a nonnegative integer")
    checkpoint_dir = Path(checkpoint_dir).resolve()
    tokenizer_path = Path(tokenizer_path).resolve()
    meta_path = checkpoint_dir / f"meta_{step:06d}.json"
    meta = json.loads(meta_path.read_text())
    validate_native_checkpoint_metadata(meta, step)
    return {
        "checkpoint_dir": str(checkpoint_dir), "step": step,
        "tokenizer_path": str(tokenizer_path),
        "model_sha256": sha256(checkpoint_dir / f"model_{step:06d}.pt"),
        "meta_sha256": sha256(meta_path), "tokenizer_sha256": sha256(tokenizer_path),
    }


class NativeDecoderAdapter:
    """Strict native SFT loader with the trained chat renderer and native KV cache."""

    def __init__(self, checkpoint_dir: Path, step: int, device: str, tokenizer_path: Path):
        self.identity = checkpoint_identity(checkpoint_dir, step, tokenizer_path)
        configured_tokenizer = (Path(get_base_dir()) / "tokenizer" / "tokenizer.pkl").resolve()
        if Path(tokenizer_path).resolve() != configured_tokenizer:
            raise ValueError("tokenizer_path must equal NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl")
        native_runtime_settings(full_prefix_device=t.device(device))
        self.model, self.tokenizer, self.metadata = build_model(
            str(checkpoint_dir), step, t.device(device), "eval",
        )
        if not isinstance(self.tokenizer, RustBPETokenizer):
            raise ValueError("native decoder prompt steering requires the trained RustBPE tokenizer")
        if checkpoint_identity(checkpoint_dir, step, tokenizer_path) != self.identity:
            raise ValueError("checkpoint/tokenizer changed while loading")
        self.engine = InferenceEngine(self.model, self.tokenizer, t.device(device))

    def prompt_tokens(self, user_text: str) -> list[int]:
        """Use the trained user framing without render_for_completion's 2,048-token cap."""
        tokens, _ = self.tokenizer.render_conversation(
            {"messages": [{"role": "user", "content": user_text}]}, max_tokens=None,
        )
        return [*tokens, self.tokenizer.encode_special("<assistant_start>")]

    def prompt_messages(self, messages: Sequence[Mapping[str, str]]) -> list[int]:
        """Render the original complete conversation without truncation."""
        return render_messages_for_chat_completion(self.tokenizer, messages).token_ids

    def generate_messages(
        self, messages: Sequence[Mapping[str, str]], *, seed: int,
        max_new_tokens: int = 128, temperature: float = 1.0, top_k: int = 50,
    ) -> ChatGenerationResult:
        """Generate from complete messages and retain budget-exhaustion evidence."""
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if not math.isfinite(temperature) or temperature <= 0 or type(top_k) is not int or top_k < 1:
            raise ValueError("sampled generation requires positive finite temperature and top_k")
        prompt = render_messages_for_chat_completion(self.tokenizer, messages)
        generated = tuple(self.engine.generate_stream(
            prompt.token_ids, max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=top_k, seed=seed,
            stop_token_ids={self.tokenizer.get_bos_token_id(), prompt.assistant_end_id},
        ))
        return ChatGenerationResult(
            self.tokenizer.decode(list(generated)), generated, len(generated) == max_new_tokens,
        )

    def generate_user_text(
        self, user_text: str, *, seed: int, max_new_tokens: int = 128,
        temperature: float = 1.0, top_k: int = 50,
    ) -> str:
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if not math.isfinite(temperature) or temperature <= 0 or type(top_k) is not int or top_k < 1:
            raise ValueError("sampled generation requires positive finite temperature and top_k")
        tokens = self.prompt_tokens(user_text)
        generated = self.engine.generate_stream(
            tokens, max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=top_k, seed=seed,
            stop_token_ids={self.tokenizer.get_bos_token_id(), self.tokenizer.encode_special("<assistant_end>")},
        )
        return self.tokenizer.decode(list(generated))


def cell_seed(base_seed: int, concept_id: int, input_id: int) -> int:
    fields = (POLICY, METHOD, base_seed, concept_id, input_id, float(1).hex())
    return int.from_bytes(hashlib.sha256(canonical_bytes(fields)).digest()[:8], "big") % (2**31 - 1)


def run_native_decoder(cfg: DictConfig) -> dict[str, Any]:
    """Generate immutable per-cell receipts; a bounded pilot resumes into production."""
    source_path = Path(cfg.source_path).resolve()
    output = Path(cfg.output_dir).resolve()
    if source_path.is_relative_to(output):
        raise ValueError("native output must be disjoint from the frozen source")
    input_sha = sha256(source_path)
    if input_sha != cfg.source_sha256:
        raise ValueError("frozen AxBench source SHA-256 mismatch")
    frame = native_input_frame(pd.read_parquet(source_path), max_concepts=cfg.max_concepts)
    if sha256(source_path) != input_sha:
        raise ValueError("frozen AxBench source changed while loading")
    checkpoint = checkpoint_identity(Path(cfg.checkpoint_dir), cfg.step, Path(cfg.tokenizer_path))
    generation = {key: cfg[key] for key in ("seed", "max_new_tokens", "temperature", "top_k")}
    source = {
        "input_path": str(source_path), "input_sha256": input_sha,
        "max_concepts": cfg.max_concepts,
    }
    source_root = Path(__file__).resolve().parents[3]
    implementation = {
        "source_sha256": {name: sha256(source_root / name) for name in (
            "steering/eval/axbench_upstream/native_decoder.py",
            "steering/eval/axbench_upstream/native_contract.py",
            "steering/eval/inference_engine.py",
            "nanochat/checkpoint_manager.py", "nanochat/gpt.py", "nanochat/engine.py",
            "nanochat/tokenizer.py", "nanochat/sampling.py", "nanochat/flash_attention.py", "nanochat/common.py",
        )},
        **native_runtime_settings(cached_device=t.device(str(cfg.device))),
    }
    identity = {
        "schema_version": 1, "format": FORMAT, "method": METHOD, "factors": [1.0],
        "checkpoint": checkpoint, "generation": generation, "source": source, "implementation": implementation,
        "policy": {"name": POLICY, "template": "frozen boosted steering instruction + \\n\\nQuestion: original_prompt", "renderer": "RustBPE.render_conversation(max_tokens=None) + assistant_start",
                   "conditioning": "natural-language user instruction; no SAE conditioning",
                   "comparability": "single prompt condition, not a numeric feature-strength sweep"},
    }
    cells_dir = output / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    install_bytes_once(canonical_bytes(identity), output / "identity.json", surface="native AxBench identity")
    fingerprint = hashlib.sha256(canonical_bytes(identity)).hexdigest()
    started = time.monotonic()
    generated = 0
    adapter: NativeDecoderAdapter | None = None
    results: list[str] = []
    max_cells = cfg.max_cells
    if max_cells is not None and (type(max_cells) is not int or max_cells < 1):
        raise ValueError("max_cells must be null or a positive integer")
    for row in frame.to_dict(orient="records"):
        seed = cell_seed(cfg.seed, int(row["concept_id"]), int(row["input_id"]))
        key = {"concept_id": int(row["concept_id"]), "input_id": int(row["input_id"]), "factor": 1.0, "seed": seed}
        path = cells_dir / f"c{row['concept_id']:05d}_i{row['input_id']:05d}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if (set(record) != {"identity_sha256", "cell", "generation"}
                    or record.get("identity_sha256") != fingerprint or record.get("cell") != key
                    or not isinstance(record.get("generation"), str)):
                raise ValueError(f"invalid or conflicting native cell receipt: {path}")
            results.append(record["generation"])
            continue
        if max_cells is not None and generated >= max_cells:
            break
        if adapter is None:
            adapter = NativeDecoderAdapter(Path(cfg.checkpoint_dir), cfg.step, cfg.device, Path(cfg.tokenizer_path))
            if adapter.identity != checkpoint:
                raise ValueError("checkpoint changed after resume identity was installed")
        began = time.monotonic()
        text = adapter.generate_user_text(str(row["native_user_prompt"]), seed=seed,
                                max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature, top_k=cfg.top_k)
        install_bytes_once(canonical_bytes({"identity_sha256": fingerprint, "cell": key, "generation": text}), path,
                           surface="native AxBench generation cell")
        results.append(text)
        generated += 1
        progress = {"complete": False, "completed_cells": len(results), "total_cells": len(frame),
                    "new_cells": generated, "elapsed_seconds": time.monotonic() - started,
                    "last_cell_seconds": time.monotonic() - began}
        atomic_json_dump(progress, output / "progress.json")
        logger.info("{}: {}/{} cells; last {:.2f}s", METHOD, len(results), len(frame), progress["last_cell_seconds"])
    complete = len(results) == len(frame)
    manifest = {**identity, "complete": complete, "completed_cells": len(results), "total_cells": len(frame)}
    if complete:
        frame[f"{METHOD}_steered_generation"] = results
        output_path = output / "steering_data.parquet"
        atomic_write_parquet(pa.Table.from_pandas(frame, preserve_index=False), output_path)
        manifest["output"] = {"path": str(output_path), "sha256": sha256(output_path), "rows": len(frame)}
        atomic_json_dump(manifest, output / "run_manifest.json")
    atomic_json_dump(manifest, output / "progress.json")
    return manifest
