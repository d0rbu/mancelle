"""
Unified evaluation script for base models.

Supports three evaluation modes (comma-separated):
  eval=core    : CORE metric (accuracy on ICL tasks)
  eval=bpb     : Bits per byte on train/val splits
  eval=sample  : Generate samples from the model

Default is all three: eval=core,bpb,sample

Examples:

    # Evaluate a HuggingFace model (e.g. GPT-2 124M) using 8 GPUs
    torchrun --nproc_per_node=8 -m scripts.eval.base_eval -- hf_path=openai-community/gpt2

    # Evaluate a nanochat model (e.g. d24) using 8 GPUs
    torchrun --nproc_per_node=8 -m scripts.eval.base_eval -- model_tag=d24 device_batch_size=16

    # Quick/approximate evaluation using a single GPU
    python -m scripts.eval.base_eval model_tag=d24 device_batch_size=16 max_per_task=100 split_tokens=524288
"""

import csv
import math
import os
import shutil
import tempfile
import time
import zipfile
from contextlib import nullcontext
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import hydra
import torch
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM

from errors import FeatureMaskError
from nanochat.checkpoint_manager import ModelPhase, load_model
from nanochat.common import (
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    download_file_with_lock,
    get_base_dir,
)
from nanochat.core_eval import (
    TaskMeta,
    TaskType,
    _sample_fewshot_indices,
    batch_sequences_lm,
    batch_sequences_mc,
    batch_sequences_schema,
    core_model_max_seq_len,
    evaluate_task,
    forward_model,
    render_prompts_lm,
    render_prompts_mc,
    render_prompts_schema,
    should_skip_oversized_core_example,
    stack_sequences,
)
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.dataset import DatasetSplit
from nanochat.loss_eval import evaluate_bpb
from nanochat.report import get_report
from nanochat.tokenizer import HuggingFaceTokenizer, get_token_bytes
from steering.conditional.encoder_decoder.dataloader import precomputed_data_loader
from steering.conditional.encoder_decoder.loss_eval import evaluate_encdec_bpb_conditioned
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_clusters import (
    reject_feature_clusters_config,
    reject_feature_clusters_metadata,
)
from steering.conditional.feature_mask import canonical_dict
from steering.eval.bpb_pipeline import build_feature_pipeline_from_cfg
from steering.eval.chat_generation import (
    empty_assistant_conversation,
    render_prompt_for_chat_completion,
)
from steering.eval.context_cache_io import SafetensorsContextFeatureProvider
from steering.eval.core_dataset import load_selected_core_task_data
from steering.eval.feature_config import (
    ensure_core_eval_has_work,
    parse_feature_configs,
    require_nonempty_feature_configs,
    resolve_next_segment_conditioning_offset,
    validate_inactive_bucket_feature_configs,
)
from steering.eval.inference_engine import (
    InferenceEngine,
    build_engine_for_features,
    build_wrapper_for_features,
)
from steering.eval.segment_mode_guard import require_fixed_checkpoint_segment_mode
from steering.eval.target_conditioning import TargetConditioningMode
from steering.eval.target_real_dispatch import build_target_for_real_config
from steering.eval.target_real_runner import (
    SingleEntryContextProvider,
    assert_target_real_requires_encdec,
    build_shared_context_feature_provider,
    build_target_real_artifacts_for_example,
    build_target_real_payload,
    context_segments_for_target_real_prefix,
    core_row_char_spans,
    lookup_target_cache_entry_or_raise,
    parse_target_real_configs_from_cfg,
    require_shared_context_cache_dir,
    run_target_real_evals,
    should_skip_empty_target_core_example,
)
from steering.overlay_io import read_manifest
from steering.packed_batch_io import compute_selection_hash
from steering.segmentation import token_to_char_offsets

# -----------------------------------------------------------------------------
# Eval mode enum


class EvalMode(StrEnum):
    CORE = "core"
    BPB = "bpb"
    SAMPLE = "sample"


# -----------------------------------------------------------------------------
# HuggingFace loading utilities


class ModelWrapper:
    """Lightweight wrapper to give HuggingFace models a nanochat-compatible interface."""

    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids, targets=None, loss_reduction="mean"):
        logits = self.model(input_ids).logits
        if targets is None:
            return logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )
        return loss

    def get_device(self):
        return next(self.model.parameters()).device


def load_hf_model(hf_path: str, device):
    """Load a HuggingFace model and tokenizer."""
    logger.info(f"Loading HuggingFace model from: {hf_path}")
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()
    max_seq_len = 1024 if "gpt2" in hf_path else None
    model = ModelWrapper(model, max_seq_len=max_seq_len)
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    return model, tokenizer


def get_hf_token_bytes(tokenizer, device="cpu"):
    """Compute token_bytes tensor for a HuggingFace tokenizer."""
    vocab_size = tokenizer.tokenizer.get_vocab_size()
    token_bytes = torch.zeros(vocab_size, dtype=torch.int64, device=device)
    for token_id in range(vocab_size):
        token_str = tokenizer.tokenizer.decode([token_id])
        token_bytes[token_id] = len(token_str.encode("utf-8"))
    return token_bytes


# -----------------------------------------------------------------------------
# CORE evaluation

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def place_eval_bundle(file_path):
    """Unzip eval_bundle.zip and place it in the base directory."""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    logger.info(f"Placed eval_bundle directory at {eval_bundle_dir}")


@dataclass(frozen=True)
class _RenderedCoreExample:
    index: int
    gold: int
    tokens: list[list[int]]
    start_idxs: list[int]
    end_idxs: list[int]


def _render_core_example(
    *,
    idx: int,
    tokenizer,
    data: list[dict],
    task_meta: TaskMeta,
    seed: int,
) -> _RenderedCoreExample:
    item = data[idx]
    fewshot_indices = _sample_fewshot_indices(
        data_len=len(data),
        current_idx=idx,
        num_fewshot=task_meta.num_fewshot,
        seed=seed + idx,
    )
    fewshot = [data[i] for i in fewshot_indices]
    if task_meta.task_type is TaskType.MULTIPLE_CHOICE:
        prompts = render_prompts_mc(item, task_meta.continuation_delimiter, fewshot)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_meta.task_type is TaskType.SCHEMA:
        prompts = render_prompts_schema(item, task_meta.continuation_delimiter, fewshot)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    else:
        prompts = render_prompts_lm(item, task_meta.continuation_delimiter, fewshot)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    return _RenderedCoreExample(
        index=idx,
        gold=int(item.get("gold", 0)),
        tokens=[list(row) for row in tokens],
        start_idxs=list(start_idxs),
        end_idxs=list(end_idxs),
    )


@torch.no_grad()
def evaluate_target_aware_core_task(
    model,
    tokenizer,
    data,
    device,
    task_meta: TaskMeta,
    seed: int = 1234,
    *,
    label: str = "unknown",
    example_batch_size: int = 5,
) -> float | None:
    """Evaluate target-aware CORE examples in independent row batches."""
    if len(data) == 0:
        raise ValueError("core_eval: empty dataset")
    if type(example_batch_size) is not int or example_batch_size < 1:
        raise ValueError(
            f"example_batch_size must be a positive integer, got {example_batch_size!r}"
        )

    t0 = time.perf_counter()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    evaluated = torch.zeros(len(data), dtype=torch.float32, device=device)
    local_indices = list(range(rank, len(data), world_size))
    max_tokens = core_model_max_seq_len(model)

    for batch_start in range(0, len(local_indices), example_batch_size):
        rendered: list[_RenderedCoreExample] = []
        for idx in local_indices[batch_start : batch_start + example_batch_size]:
            example = _render_core_example(
                idx=idx,
                tokenizer=tokenizer,
                data=data,
                task_meta=task_meta,
                seed=seed,
            )
            if should_skip_oversized_core_example(
                example.tokens,
                max_tokens=max_tokens,
                label=label,
                idx=idx,
            ):
                continue
            rendered.append(example)
        if not rendered:
            continue

        flat_tokens = [row for example in rendered for row in example.tokens]
        input_ids = stack_sequences(flat_tokens, tokenizer.get_bos_token_id()).to(device)
        target_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        context_token_ids: list[list[int]] = []
        for example in rendered:
            for tokens, start_idx, end_idx in zip(
                example.tokens,
                example.start_idxs,
                example.end_idxs,
                strict=True,
            ):
                row = len(context_token_ids)
                target_start = max(start_idx - 1, 0)
                target_end = min(max(target_start + 1, end_idx - 1), input_ids.size(1))
                target_mask[row, target_start:target_end] = True
                context_token_ids.append(tokens[: target_start + 1])

        losses, predictions = forward_model(
            model,
            input_ids,
            target_mask=target_mask,
            context_token_ids=context_token_ids,
        )
        row_start = 0
        for example in rendered:
            row_count = len(example.tokens)
            if task_meta.task_type is TaskType.LANGUAGE_MODELING:
                start_idx = example.start_idxs[0]
                end_idx = example.end_idxs[0]
                is_correct = torch.all(
                    predictions[row_start, start_idx - 1 : end_idx - 1]
                    == input_ids[row_start, start_idx:end_idx]
                )
            else:
                mean_losses = torch.stack(
                    [
                        losses[row_start + row, start_idx - 1 : end_idx - 1].mean()
                        for row, (start_idx, end_idx) in enumerate(
                            zip(example.start_idxs, example.end_idxs, strict=True)
                        )
                    ]
                )
                is_correct = mean_losses.argmin() == example.gold
            correct[example.index] = is_correct.to(dtype=torch.float32)
            evaluated[example.index] = 1.0
            row_start += row_count

    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.all_reduce(correct, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(evaluated, op=torch.distributed.ReduceOp.SUM)
    n_evaluated = int(evaluated.sum().item())
    if n_evaluated == 0:
        logger.warning(
            f"evaluate_task: {label} skipped all {len(data)} examples; "
            "no evaluable CORE examples remain"
        )
        return None
    mean_correct = (correct.sum() / evaluated.sum()).item()
    elapsed = time.perf_counter() - t0
    examples_per_sec = n_evaluated / elapsed if elapsed > 0 else 0
    logger.info(
        f"evaluate_task: {elapsed:.1f}s | {n_evaluated} examples scored/"
        f"{len(data)} total | {examples_per_sec:.1f} ex/sec"
    )
    return mean_correct


def evaluate_core(
    model,
    tokenizer,
    device,
    max_per_task=-1,
    seed=1234,
    core_example_batch_size: int = 1,
):
    """
    Evaluate a base model on the CORE benchmark.
    Returns dict with results, centered_results, and core_metric.
    """
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    # Download the eval bundle if needed
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(
            EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle
        )

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tasks = config["icl_tasks"]

    # Load random baseline values
    random_baselines = {}
    with open(eval_meta_data, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row["Eval Task"]
            random_baseline = row["Random baseline"]
            random_baselines[task_name] = float(random_baseline)

    # Evaluate each task
    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task["label"]
        task_meta = TaskMeta(
            task_type=TaskType(task["icl_task_type"]),
            num_fewshot=task["num_fewshot"][0],
            continuation_delimiter=task.get("continuation_delimiter", " "),
        )
        logger.info(
            f"Evaluating: {label} ({task_meta.num_fewshot}-shot, type: {task_meta.task_type})... "
        )

        data_path = os.path.join(data_base_path, task["dataset_uri"])
        data = load_selected_core_task_data(
            data_path,
            max_per_task=max_per_task,
        )

        if getattr(model, "is_target_aware_encdec", False) is True:
            accuracy = evaluate_target_aware_core_task(
                model,
                tokenizer,
                data,
                device,
                task_meta,
                seed=seed,
                label=label,
                example_batch_size=core_example_batch_size,
            )
        else:
            accuracy = evaluate_task(
                model,
                tokenizer,
                data,
                device,
                task_meta,
                seed=seed,
                label=label,
            )
        if accuracy is None:
            elapsed = time.time() - start_time
            logger.warning(
                f"Skipping CORE task {label}: no examples fit model_max_seq_len "
                f"(time: {elapsed:.2f}s)"
            )
            continue
        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        elapsed = time.time() - start_time
        logger.info(
            f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {elapsed:.2f}s"
        )

    if not centered_results:
        raise ValueError("core_eval: no evaluable CORE tasks remain after oversized-row filtering")
    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric,
    }
    return out


def _core_seed_from_cfg(cfg: DictConfig) -> int:
    return int(cfg.get("core_seed", 1234))


def _sync_cfg_model_max_seq_len_with_loaded_model(cfg: DictConfig, sequence_len: int) -> int:
    """Use the loaded checkpoint context length for eval cache resolution."""

    actual = int(sequence_len)
    configured = int(cfg.get("model_max_seq_len", actual))
    if configured != actual:
        logger.warning(
            f"base_eval model_max_seq_len={configured} differs from loaded "
            f"model sequence_len={actual}; using {actual} for eval cache "
            "resolution and metadata validation"
        )
    OmegaConf.update(cfg, "model_max_seq_len", actual, merge=False)
    return actual


def _write_core_csv(output_csv_path: str, core_results: dict) -> None:
    """Write CORE evaluation results to a CSV file."""
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Task", "Accuracy", "Centered"])
        for label in core_results["results"]:
            writer.writerow(
                [
                    label,
                    f"{core_results['results'][label]:.6f}",
                    f"{core_results['centered_results'][label]:.6f}",
                ]
            )
        writer.writerow(["CORE", "", f"{core_results['core_metric']:.6f}"])


# -----------------------------------------------------------------------------
# PR-C Task 4 — BPB conditioning helpers


def _resolve_bpb_selection_dir(
    val_packed_row_dir,
    *,
    selection_hash: str | None,
    feature_selection_raw: object,
    negative_selection_raw=None,
    num_buckets: int = 2,
    feature_mask_canonical=None,
    feature_clusters_descriptor=None,
    splice_batch_size: object = None,
) -> Path | None:
    """Locate the selection-shards subdir under val_packed_row_dir/sel/.

    Selection shards are OPTIONAL — when absent, the precomputed_data_loader
    falls back to its fused-inline path with the real
    ``FeaturePipelineConfig`` reconstructed from cfg.

    ``feature_selection_raw`` is required because the sole production caller
    first reconstructs the fused-inline pipeline from that same config.  When
    ``selection_hash`` is None, the exact hash is computed from the selection
    config (same args as precompute/training; see ANTI-DESYNC comment at the
    call site).  This is mask-aware: ``feature_mask_canonical`` is folded into
    the hash so a masked eval resolves its masked sel/ dir rather than silently
    picking an unrelated one.

    After a sel/ dir is located, its manifest is read and the recorded
    ``feature_mask`` is asserted to equal ``feature_mask_canonical``.  Mismatch
    raises ``FeatureMaskError`` (same guard as training — guards hash collisions
    and moved/mislabelled dirs).  Back-compat (I5): no mask configured + legacy
    manifest with no feature_mask key → None == None → passes, no error.

    Returns:
      - ``Path`` when selection shards exist (sel/<hash>/ resolves)
      - ``None`` when no selection shards exist (sel/ missing or empty)

    Raises ``ValueError`` when ``feature_selection_raw`` is absent or when
    ``selection_hash`` is explicitly pinned and the named subdir doesn't exist.
    Raises ``TypeError`` when ``feature_selection_raw`` is not a plain mapping.
    Raises ``FeatureMaskError`` when the manifest's recorded feature_mask
    does not match the configured mask.
    """
    if feature_selection_raw is None:
        raise ValueError(
            "BPB selection resolution requires conditioning.feature_selection; "
            "build_feature_pipeline_from_cfg validates this before selection "
            "shard resolution."
        )
    if not isinstance(feature_selection_raw, dict):
        raise TypeError(
            "BPB selection resolution requires conditioning.feature_selection "
            f"to resolve to a mapping, got {type(feature_selection_raw).__name__}."
        )

    sel_root = Path(val_packed_row_dir) / "sel"
    if not sel_root.is_dir():
        if selection_hash is not None:
            raise ValueError(
                f"BPB conditioning: bpb.selection_hash={selection_hash!r} "
                f"was pinned but {sel_root} does not exist. Either remove "
                f"the selection_hash override (selection shards are optional) "
                f"or create the sel/ subdirectory."
            )
        return None
    if selection_hash is not None:
        explicit = sel_root / selection_hash
        if not explicit.is_dir():
            raise ValueError(
                f"BPB conditioning: bpb.selection_hash={selection_hash!r} "
                f"resolves to {explicit} which does not exist."
            )
        resolved = explicit
    else:
        # ANTI-DESYNC: this selection-hash is the eval-side ``compute_selection_hash`` call at:
        #   1. scripts/preprocess/precompute_selected_features.py
        #   2. scripts/train/conditional_encdec_train.py:build_shared_state
        #   3. scripts/eval/base_eval.py:_resolve_bpb_selection_dir (THIS site)
        #   4. steering/eval/feature_selection_hash.py:compute_canonical_feature_selection_hash (cache identity)
        # All four must agree on feature-mask, cluster, and splice-batching semantics.
        # Change ALL FOUR or none.
        computed_hash = compute_selection_hash(
            positive_selection=feature_selection_raw,
            negative_selection=negative_selection_raw,
            delta_config=None,
            num_buckets=num_buckets,
            seed=0,
            ddp_rank=0,
            feature_mask=feature_mask_canonical,
            feature_clusters=feature_clusters_descriptor,
            splice_batch_size=splice_batch_size,
        )
        candidate = sel_root / computed_hash
        if not candidate.is_dir():
            return None
        resolved = candidate
    # Defense-in-depth (I4): assert the sel/ manifest records the same feature_mask
    # as the configured one.  Guards hash collisions and moved/mislabeled sel/ dirs.
    sel_manifest = read_manifest(resolved)
    if sel_manifest is not None:
        reject_feature_clusters_metadata(
            sel_manifest.get("metadata", {}),
            f"BPB sel/ manifest {resolved}",
        )

    manifest_mask = sel_manifest.get("metadata", {}).get("feature_mask") if sel_manifest else None
    if manifest_mask != feature_mask_canonical:
        raise FeatureMaskError(
            f"BPB sel/ manifest feature_mask mismatch for {resolved}: "
            f"manifest has {manifest_mask!r}, configured {feature_mask_canonical!r}. "
            "This guards against a hash collision or a moved/mislabeled sel/ dir. "
            "Re-run precompute_selected_features with the current config."
        )
    return resolved


def _run_core_target_real_eval(
    *,
    raw_encdec,
    tokenizer,
    device,
    trc,
    provider,
    max_per_task: int,
    target_mode: TargetConditioningMode | str = TargetConditioningMode.CURRENT,
    seed: int = 1234,
) -> dict:
    """CORE eval driven by per-row target_real conditioning.

    For each task and each example, render rows via the existing CORE
    per-row tokenization (``render_prompts_*`` + ``batch_sequences_*``),
    look up the per-row cache entry, build a per-row target via
    ``build_target_for_real_config``, build a fresh wrapper, and forward
    the row to score the answer. Aggregate accuracy per task.

    Returns dict with ``results``, ``centered_results``, and ``core_metric``
    (mirrors ``evaluate_core``'s output shape).
    """
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(
            EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle
        )
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tasks = config["icl_tasks"]
    random_baselines: dict[str, float] = {}
    with open(eval_meta_data, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            random_baselines[row["Eval Task"]] = float(row["Random baseline"])

    results: dict[str, float] = {}
    centered_results: dict[str, float] = {}
    examples_scored: dict[str, int] = {}
    skipped_empty_target: dict[str, int] = {}
    for task in tasks:
        label = task["label"]
        task_meta = TaskMeta(
            task_type=TaskType(task["icl_task_type"]),
            num_fewshot=task["num_fewshot"][0],
            continuation_delimiter=task.get("continuation_delimiter", " "),
        )
        data_path = os.path.join(data_base_path, task["dataset_uri"])
        data = load_selected_core_task_data(
            data_path,
            max_per_task=max_per_task,
        )

        num_correct = 0
        n_examples = 0
        n_skipped_empty_target = 0
        for idx in range(len(data)):
            item = data[idx]
            fewshot_indices = _sample_fewshot_indices(
                data_len=len(data),
                current_idx=idx,
                num_fewshot=task_meta.num_fewshot,
                seed=seed + idx,
            )
            fewshot = [data[i] for i in fewshot_indices]
            if task_meta.task_type is TaskType.MULTIPLE_CHOICE:
                prompts = render_prompts_mc(item, task_meta.continuation_delimiter, fewshot)
                tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
            elif task_meta.task_type is TaskType.SCHEMA:
                prompts = render_prompts_schema(item, task_meta.continuation_delimiter, fewshot)
                tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
            else:  # LM
                prompts = render_prompts_lm(item, task_meta.continuation_delimiter, fewshot)
                tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)

            if should_skip_oversized_core_example(
                tokens,
                max_tokens=core_model_max_seq_len(raw_encdec),
                label=label,
                idx=idx,
            ):
                continue

            # A row whose continuation decodes to zero characters yields an empty
            # target span, which filter_segments_to_target rejects. That is a
            # property of the benchmark row, not an invariant violation, so skip
            # the example and count it rather than aborting a multi-hour eval.
            # Same policy (and same whole-example granularity) as the oversized-row
            # filter above. Every other target rejection stays loud.
            row_char_spans = core_row_char_spans(tokenizer, tokens, start_idxs, end_idxs)
            if should_skip_empty_target_core_example(row_char_spans):
                n_skipped_empty_target += 1
                continue

            row_losses: list[float] = []
            row_predictions: list[list[int]] = []
            for _row, (tok_seq, si, ei) in enumerate(
                zip(tokens, start_idxs, end_idxs, strict=True)
            ):
                tok_list = list(tok_seq)
                # Cache lookup uses the FULL row tokens.
                entry = lookup_target_cache_entry_or_raise(
                    provider,
                    tok_list,
                    trc_name=trc.name,
                )
                target_char_start, target_char_end = row_char_spans[_row]
                target_token_loss_mask = [0] * len(tok_list)
                target_start_safe = max(si - 1, 0)
                for t_idx in range(target_start_safe, ei):
                    if t_idx < len(target_token_loss_mask):
                        target_token_loss_mask[t_idx] = 1
                target = build_target_for_real_config(
                    cfg=trc,
                    entry=entry,
                    target_char_start=target_char_start,
                    target_char_end=target_char_end,
                    full_token_ids=tok_list,
                    target_token_loss_mask=target_token_loss_mask,
                    tokenizer=tokenizer,
                )
                context_token_ids = tok_list[: target_start_safe + 1]
                context_segments = context_segments_for_target_real_prefix(
                    provider=provider,
                    context_token_ids=context_token_ids,
                    trc_name=trc.name,
                )
                # CORE target_real uses a full-row cache entry for the target
                # span and an exact prompt/context cache row for observed
                # tokens. Target positions are supplied only through the
                # explicit `target` argument.
                wrapper = build_wrapper_for_features(
                    raw_encdec,
                    tokenizer,
                    target=target,
                    context_provider=SingleEntryContextProvider(
                        context_token_ids,
                        context_segments,
                    ),
                    target_mode=target_mode,
                )
                input_ids = torch.tensor([tok_list], dtype=torch.long, device=device)
                target_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                target_start = max(si - 1, 0)
                target_end = min(max(target_start + 1, ei - 1), input_ids.size(1))
                target_mask[0, target_start:target_end] = True
                active_context_token_ids = [context_token_ids]
                with torch.no_grad():
                    target_aware = getattr(wrapper, "is_target_aware_encdec", False)
                    ctx = (
                        wrapper.conditioning_context(
                            target_mask=target_mask,
                            context_token_ids=active_context_token_ids,
                        )
                        if target_aware
                        else nullcontext()
                    )
                    with ctx:
                        outputs = wrapper(input_ids)
                target_ids = torch.roll(input_ids, shifts=-1, dims=1)
                losses = torch.nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    target_ids.view(-1),
                    reduction="none",
                ).view(input_ids.size(0), input_ids.size(1))
                losses[:, -1] = float("nan")
                preds = outputs.argmax(dim=-1)
                row_losses.append(float(losses[0, target_start:target_end].mean().item()))
                row_predictions.append(preds[0].tolist())

            if task_meta.task_type is TaskType.LANGUAGE_MODELING:
                # Defensive: LM tasks always render exactly 1 prompt per
                # example. Slicing ``row_predictions[0]`` / ``tokens[0]``
                # only makes sense for that contract; assert it explicitly
                # so future task-type additions don't silently lock-step
                # against the first row only.
                assert len(start_idxs) == 1, f"LM task expected 1 prompt, got {len(start_idxs)}"
                si0 = start_idxs[0]
                ei0 = end_idxs[0]
                pred_tokens = row_predictions[0][si0 - 1 : ei0 - 1]
                actual_tokens = list(tokens[0])[si0:ei0]
                is_correct = pred_tokens == actual_tokens
            else:
                # Defensive: filter NaN losses (e.g. from a degenerate
                # zero-length target span where the only position in the
                # slice is the NaN'd-out last position). ``min`` returns
                # NaN on a list containing NaN, and ``list.index(NaN)``
                # raises ``ValueError`` (NaN != NaN). Fail loudly with an
                # operator-actionable message instead.
                finite_losses = [
                    (i, loss) for i, loss in enumerate(row_losses) if math.isfinite(loss)
                ]
                if not finite_losses:
                    raise RuntimeError(
                        f"All row losses are non-finite for task {label} "
                        f"example {idx}; check target span computation. "
                        f"row_losses={row_losses}"
                    )
                pred_idx = min(finite_losses, key=lambda x: x[1])[0]
                is_correct = pred_idx == item["gold"]
            num_correct += int(is_correct)
            n_examples += 1

        if n_skipped_empty_target:
            skipped_empty_target[label] = n_skipped_empty_target
            logger.warning(
                f"target_real['{trc.name}'] CORE {label}: skipped "
                f"{n_skipped_empty_target}/{len(data)} example(s) "
                f"({n_skipped_empty_target / len(data):.2%}) whose target span "
                "decodes to zero characters"
            )

        if n_examples == 0:
            logger.warning(
                f"target_real['{trc.name}'] CORE {label}: no examples fit "
                "model_max_seq_len; excluding task from aggregate"
            )
            continue

        examples_scored[label] = n_examples
        accuracy = num_correct / n_examples
        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered_results[label] = (accuracy - 0.01 * random_baseline) / (
            1.0 - 0.01 * random_baseline
        )
        logger.info(
            f"target_real['{trc.name}'] CORE {label}: {accuracy:.4f} "
            f"(centered {centered_results[label]:.4f}, n={n_examples})"
        )

    if not centered_results:
        raise ValueError(
            f"target_real['{trc.name}'] CORE: no evaluable tasks remain after "
            "oversized-row filtering"
        )
    # Surface the empty-target skips at aggregate level too: a metric computed
    # over a heavily-skipped task set is not comparable to a clean run, so the
    # operator must be able to see the rate without grepping per-task logs.
    total_skipped = sum(skipped_empty_target.values())
    total_scored = sum(examples_scored.values())
    if total_skipped:
        logger.warning(
            f"target_real['{trc.name}'] CORE: skipped {total_skipped} example(s) "
            f"with an empty target span across {len(skipped_empty_target)} task(s) "
            f"({total_skipped / (total_skipped + total_scored):.2%} of all examples). "
            "Per-task counts are recorded as n_skipped_empty_target."
        )
    core_metric = sum(centered_results.values()) / len(centered_results)
    return {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric,
        "n_examples": examples_scored,
        "n_skipped_empty_target": skipped_empty_target,
    }


def _run_sample_target_real_eval(
    *,
    raw_encdec,
    tokenizer,
    device,
    trc,
    provider,
    cfg,
    target_mode: TargetConditioningMode | str = TargetConditioningMode.CURRENT,
) -> list[str]:
    """Per-prompt sample generation under target_real conditioning.

    Each ``cfg.sample_prompts`` entry is wrapped as a chat conversation
    with an empty assistant turn. The target cache row is looked up by the
    full conversation; observed-token features come from the exact generation
    prompt cache row and are exposed as the wrapper's context provider.
    """
    samples: list[str] = []
    bos = tokenizer.get_bos_token_id()
    # Detect whether the tokenizer supports chat rendering.
    has_chat_render = hasattr(tokenizer, "render_for_completion") and hasattr(
        tokenizer, "encode_special"
    )
    for prompt in cfg.sample_prompts:
        if has_chat_render:
            # Chat-render: prompt becomes the user turn; empty assistant
            # so render_for_completion ends at <assistant_start>. The
            # cache key uses render_conversation (full conv).
            conv = empty_assistant_conversation(prompt)
            full_token_ids, mask = tokenizer.render_conversation(conv)
            full_token_ids = list(full_token_ids)
            mask = list(mask)
            char_offsets = token_to_char_offsets(tokenizer, full_token_ids)
            mask_1_indices = [i for i, m in enumerate(mask) if m == 1]
            if mask_1_indices:
                target_char_start = int(char_offsets[mask_1_indices[0]])
                target_char_end = int(char_offsets[mask_1_indices[-1] + 1])
            else:
                # Empty assistant content: degenerate range (the assistant_end
                # marker char range). Avoid 0-length by extending by 1 char.
                target_char_start = int(char_offsets[-1]) - 1
                target_char_end = int(char_offsets[-1])
                if target_char_start < 0:
                    target_char_start = 0
            target_token_loss_mask = list(mask)
            try:
                asst_start_id = tokenizer.encode_special("<assistant_start>")
            except (ValueError, RuntimeError, TypeError, AttributeError):
                asst_start_id = None
            if asst_start_id is not None and asst_start_id in full_token_ids:
                # The empty-assistant generation prompt has no mask=1 content
                # positions. Route the final prompt token so generation still
                # has an autoregressive conditioning slot.
                pos = len(full_token_ids) - 1 - full_token_ids[::-1].index(asst_start_id)
                if pos < len(target_token_loss_mask):
                    target_token_loss_mask[pos] = 1
        else:
            # Raw-text BOS-prepended (HF / non-chat tokenizers).
            full_token_ids = list(tokenizer.encode(prompt, prepend=bos))
            char_offsets = token_to_char_offsets(tokenizer, full_token_ids)
            first_target_tok = 1 if full_token_ids and full_token_ids[0] == bos else 0
            target_char_start = int(char_offsets[first_target_tok])
            target_char_end = int(char_offsets[-1])
            if target_char_end <= target_char_start:
                target_char_end = target_char_start + 1
            target_token_loss_mask = [0] * len(full_token_ids)
            for i in range(first_target_tok, len(full_token_ids)):
                target_token_loss_mask[i] = 1
            # The shared builder's ``mask`` parameter is also the fallback
            # source for context-prefix derivation. Raw tokenizers have no
            # render_conversation mask, so the explicit loss mask is the
            # equivalent full-sequence mask for this branch.
            mask = list(target_token_loss_mask)
        if has_chat_render:
            context_token_ids = render_prompt_for_chat_completion(
                tokenizer,
                prompt,
            ).token_ids
        else:
            context_token_ids = list(full_token_ids)
        artifacts = build_target_real_artifacts_for_example(
            model=raw_encdec,
            tokenizer=tokenizer,
            trc=trc,
            provider=provider,
            full_token_ids=full_token_ids,
            mask=mask,
            target_char_range=(target_char_start, target_char_end),
            target_token_loss_mask=target_token_loss_mask,
            context_token_ids=context_token_ids,
            target_mode=target_mode,
        )
        engine = InferenceEngine(artifacts.wrapper, tokenizer, torch.device(device))
        sample_str = engine.generate(
            prompt,
            max_new_tokens=int(cfg.sampling.per_config_max_new_tokens),
            temperature=float(cfg.sampling.per_config_temperature),
            render_as_chat=has_chat_render,
        )
        samples.append(sample_str)
    return samples


def _base_eval_target_real_per_config(
    *,
    raw_encdec,
    tokenizer,
    device,
    trc,
    provider,
    cfg,
    eval_modes,
    target_mode: TargetConditioningMode | str = TargetConditioningMode.CURRENT,
) -> dict:
    """Build the per-target-real-config payload for base_eval.

    Dispatches CORE / SAMPLE based on ``eval_modes``. Returns the payload
    dict (orchestrator writes the yaml).

    CORE-mode wrapper build uses the trc's provider. The provider already
    contains per-segment features for the FULL row
    tokens, so the context-side routing matches the cache entry exactly;
    the explicit ``target`` argument overrides the response slot for
    target_real perturbation.
    """
    body: dict = {}
    if EvalMode.CORE in eval_modes:
        core_out = _run_core_target_real_eval(
            raw_encdec=raw_encdec,
            tokenizer=tokenizer,
            device=device,
            trc=trc,
            provider=provider,
            max_per_task=int(cfg.max_per_task),
            target_mode=target_mode,
            seed=_core_seed_from_cfg(cfg),
        )
        body["results"] = core_out["results"]
        body["centered_results"] = core_out["centered_results"]
        body["core_metric"] = core_out["core_metric"]
        # Coverage accounting travels with the scores. A core_metric computed over
        # a heavily-skipped or thinly-populated task set is not comparable to a
        # clean run, and leaving that only in the logs makes it invisible to every
        # downstream consumer of this yaml.
        body["n_examples"] = core_out["n_examples"]
        body["n_skipped_empty_target"] = core_out["n_skipped_empty_target"]
    if EvalMode.SAMPLE in eval_modes:
        samples = _run_sample_target_real_eval(
            raw_encdec=raw_encdec,
            tokenizer=tokenizer,
            device=device,
            trc=trc,
            provider=provider,
            cfg=cfg,
            target_mode=target_mode,
        )
        body["samples"] = samples
    return build_target_real_payload(trc, body, target_mode=target_mode)


# -----------------------------------------------------------------------------
# Main


@hydra.main(config_path="../../configs/eval", config_name="base_eval", version_base=None)
def main(cfg: DictConfig):
    reject_feature_clusters_config(cfg, "base_eval config")
    reject_feature_clusters_config(
        cfg.get("conditioning", None),
        "base_eval conditioning config",
    )
    # Parse evaluation modes
    eval_modes = {EvalMode(mode.strip()) for mode in cfg.eval.split(",")}
    valid_modes = set(EvalMode)
    invalid = eval_modes - valid_modes
    if invalid:
        raise ValueError(f"Invalid eval modes: {invalid}. Valid: {valid_modes}")

    # Distributed / precision setup
    device_type = autodetect_device_type() if cfg.device_type == "" else cfg.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )

    # Load model and tokenizer
    is_hf_model = cfg.hf_path is not None
    raw_encdec = None  # set when model is EncoderDecoderGPT (for conditioned CORE eval)
    shared_context_provider: SafetensorsContextFeatureProvider | None = None
    feature_configs = parse_feature_configs(cfg.feature_configs)
    if is_hf_model:
        model, tokenizer = load_hf_model(cfg.hf_path, device)
        sequence_len = model.max_seq_len or 1024
        model_name = cfg.hf_path
        model_slug = cfg.hf_path.replace("/", "-")
    else:
        source = cfg.source
        model, tokenizer, meta = load_model(
            source,
            device,
            phase=ModelPhase.EVAL,
            model_tag=cfg.model_tag,
            step=cfg.step,
            eval_conditioning=EvalConditioning.deferred("base_eval"),
        )
        reject_feature_clusters_metadata(meta, "base_eval checkpoint")
        sequence_len = meta["model_config"]["sequence_len"]
        model_name = f"{source} (step {meta['step']})"
        model_slug = f"{source.replace('/', '_')}_{meta['step']:06d}"
        if isinstance(model, EncoderDecoderGPT):
            raw_encdec = model
            require_fixed_checkpoint_segment_mode(meta, surface="base_eval")
            if feature_configs:
                require_nonempty_feature_configs(feature_configs, surface="base_eval")
                require_shared_context_cache_dir(cfg, eval_name="base_eval")
                validate_inactive_bucket_feature_configs(
                    feature_configs,
                    num_buckets=raw_encdec.encoder.config.num_buckets,
                    checkpoint_meta=meta,
                )
    sequence_len = _sync_cfg_model_max_seq_len_with_loaded_model(cfg, sequence_len)
    # Suite name picks the cache subdirectory for null-cache_dir auto-resolution
    # (Task 2). base_eval CORE consumes target caches built from full prompts;
    # the SAMPLE block also uses CORE-style targets (single shared suite).
    target_real_configs = parse_target_real_configs_from_cfg(
        cfg,
        suite="core_target",
        model_max_seq_len=sequence_len,
    )
    target_mode = TargetConditioningMode.from_next_segment_offset(
        resolve_next_segment_conditioning_offset(cfg, None if is_hf_model else meta)
    )
    uses_core_style_feature_configs = bool(eval_modes & {EvalMode.CORE, EvalMode.SAMPLE})
    if uses_core_style_feature_configs:
        encdec_bpb_provides_work = (
            raw_encdec is not None
            and not feature_configs
            and not target_real_configs
            and EvalMode.BPB in eval_modes
        )
        if not encdec_bpb_provides_work:
            feature_configs = ensure_core_eval_has_work(
                feature_configs,
                is_encdec=raw_encdec is not None,
                has_target_real=bool(target_real_configs),
                surface="base_eval",
            )
    if raw_encdec is not None and feature_configs:
        # Build one shared cache provider for CORE / sample-gen using
        # the loaded model's context length.
        shared_context_provider = build_shared_context_feature_provider(
            cfg,
            eval_name="eval",
        )

    # Validate: feature conditioning requires an encoder-decoder model
    has_real_features = any(fc.features for fc in feature_configs)
    if has_real_features and raw_encdec is None:
        raise ValueError(
            "feature_configs with non-null features require an EncoderDecoderGPT model "
            "(use source=encdec or similar). Decoder-only and HuggingFace models do not "
            "support feature conditioning."
        )
    assert_target_real_requires_encdec(target_real_configs, is_encdec=raw_encdec is not None)
    token_bytes = (
        get_hf_token_bytes(tokenizer, device=device)
        if is_hf_model
        else get_token_bytes(device=device)
    )

    logger.info(f"Evaluating model: {model_name}")
    logger.info(f"Eval modes: {', '.join(sorted(eval_modes))}")

    # Results to log
    core_results = None
    bpb_results: dict[DatasetSplit, float] = {}
    samples_by_config: dict[str, list[str]] = {}

    # --- Sampling --- (Phase 2 PR-B: per-config sample blocks)
    if EvalMode.SAMPLE in eval_modes and not is_hf_model:
        logger.info("\n" + "=" * 80)
        logger.info("Model Samples")
        logger.info("=" * 80)
        if ddp_rank == 0:
            for fc in feature_configs:
                logger.info(f"\n--- Sample block: {fc.name} ---")
                provider = shared_context_provider if raw_encdec is not None else None
                samples_for_fc: list[str] = []
                for prompt in cfg.sample_prompts:
                    engine = build_engine_for_features(
                        raw_encdec if raw_encdec is not None else model,
                        tokenizer,
                        device,
                        features=dict(fc.features) if fc.features else None,
                        context_provider=provider,
                        target_mode=target_mode,
                    )
                    with autocast_ctx:
                        sample_str = engine.generate(
                            prompt,
                            max_new_tokens=cfg.sampling.per_config_max_new_tokens,
                            temperature=cfg.sampling.per_config_temperature,
                        )
                    logger.info("-" * 80)
                    logger.info(sample_str)
                    samples_for_fc.append(sample_str)
                samples_by_config[fc.name] = samples_for_fc
    elif EvalMode.SAMPLE in eval_modes and is_hf_model:
        logger.info("\nSkipping sampling for HuggingFace models (not supported)")

    # --- BPB evaluation ---
    bpb_conditioned_results: dict[str, dict] = {}
    if EvalMode.BPB in eval_modes:
        logger.info("\n" + "=" * 80)
        if raw_encdec is None:
            logger.info("BPB Evaluation")
            logger.info("=" * 80)
            # Decoder-only / HuggingFace: NEVER read cfg.bpb.* fields.
            # cfg.bpb.val_packed_row_dir / selection_hash are ??? (mandatory)
            # in the YAML but those are encdec-only knobs.
            tokens_per_step = cfg.device_batch_size * sequence_len * ddp_world_size
            split_tokens = cfg.split_tokens
            if split_tokens % tokens_per_step != 0:
                # Adjust to nearest multiple
                split_tokens = (split_tokens // tokens_per_step) * tokens_per_step
                logger.info(
                    f"Adjusted split_tokens to {split_tokens} (must be divisible by {tokens_per_step})"
                )
            steps = split_tokens // tokens_per_step

            for split_name in DatasetSplit:
                loader = tokenizing_distributed_data_loader_bos_bestfit(
                    tokenizer,
                    cfg.device_batch_size,
                    sequence_len,
                    split_name,
                    device=device,
                )
                with autocast_ctx:
                    bpb = evaluate_bpb(model, loader, steps, token_bytes)
                bpb_results[split_name] = bpb
                logger.info(f"{split_name} bpb: {bpb:.6f}")
        else:
            logger.info("BPB Evaluation (conditioned)")
            logger.info("=" * 80)
            bpb_cfg = cfg.get("bpb", None)
            val_packed_row_dir_raw = (
                bpb_cfg.get("val_packed_row_dir", None) if bpb_cfg is not None else None
            )
            if val_packed_row_dir_raw is None or str(val_packed_row_dir_raw) == "???":
                raise ValueError(
                    "Encoder-decoder BPB eval requires bpb.val_packed_row_dir "
                    "pointing at packed rows with real encoder features."
                )
            feature_pipeline = build_feature_pipeline_from_cfg(cfg)
            selection_hash_raw = (
                bpb_cfg.get("selection_hash", None) if bpb_cfg is not None else None
            )
            _cond_cfg = cfg.get("conditioning", None)
            _fs_raw = (
                OmegaConf.to_container(_cond_cfg.get("feature_selection"), resolve=True)
                if _cond_cfg is not None and _cond_cfg.get("feature_selection") is not None
                else None
            )
            _neg_raw = (
                OmegaConf.to_container(_cond_cfg.get("negative_selection"), resolve=True)
                if _cond_cfg is not None and _cond_cfg.get("negative_selection") is not None
                else None
            )
            _num_buckets = int(_cond_cfg.get("num_buckets", 2)) if _cond_cfg is not None else 2
            selection_dir = _resolve_bpb_selection_dir(
                Path(str(val_packed_row_dir_raw)),
                selection_hash=selection_hash_raw,
                feature_selection_raw=_fs_raw,
                negative_selection_raw=_neg_raw,
                num_buckets=_num_buckets,
                feature_mask_canonical=canonical_dict(feature_pipeline.feature_mask),
                splice_batch_size=cfg.get("selection_splice_batch_size"),
            )
            split_name = DatasetSplit.VAL
            tokens_per_step = cfg.device_batch_size * sequence_len * ddp_world_size
            split_tokens = cfg.split_tokens
            if split_tokens % tokens_per_step != 0:
                split_tokens = (split_tokens // tokens_per_step) * tokens_per_step
            steps = split_tokens // tokens_per_step
            val_loader = precomputed_data_loader(
                packed_row_dir=Path(str(val_packed_row_dir_raw)),
                B=cfg.device_batch_size,
                feature_pipeline=feature_pipeline,
                device=device,
                selection_dir=selection_dir,
                effective_buckets=feature_pipeline.effective_buckets,
                ddp_rank=ddp_rank,
                ddp_world_size=ddp_world_size,
            )
            with autocast_ctx:
                metrics = evaluate_encdec_bpb_conditioned(
                    raw_encdec,
                    val_loader,
                    steps,
                    token_bytes,
                    device,
                )
            bpb_conditioned_results[str(split_name)] = {
                "bpb_conditioned": metrics.bpb_conditioned,
                "avg_features_per_segment": metrics.avg_features_per_segment,
            }
            bpb_results[split_name] = metrics.bpb_conditioned
            logger.info(f"{split_name} bpb_conditioned: {metrics.bpb_conditioned:.6f}")

    # --- CORE evaluation --- (Phase 2 PR-B: per-config conditional)
    if EvalMode.CORE in eval_modes:
        logger.info("\n" + "=" * 80)
        logger.info("CORE Evaluation")
        logger.info("=" * 80)

        if raw_encdec is not None:
            # Encoder-decoder: evaluate each feature config separately
            core_results_by_config = {}
            if feature_configs:
                for fc in feature_configs:
                    logger.info(f"\n--- Feature config: {fc.name} ---")
                    eval_model = build_wrapper_for_features(
                        raw_encdec,
                        tokenizer,
                        features=dict(fc.features) if fc.features else None,
                        context_provider=shared_context_provider,
                        target_mode=target_mode,
                    )
                    with autocast_ctx:
                        core_out = evaluate_core(
                            eval_model,
                            tokenizer,
                            device,
                            max_per_task=cfg.max_per_task,
                            seed=_core_seed_from_cfg(cfg),
                            core_example_batch_size=cfg.core_example_batch_size,
                        )
                    core_results_by_config[fc.name] = core_out
                    logger.info(f"Config '{fc.name}' CORE metric: {core_out['core_metric']:.4f}")
            else:
                logger.info(
                    "No static feature_configs configured; skipping normal encoder-decoder CORE pass"
                )

            # Write a CSV per config
            if ddp_rank == 0 and core_results_by_config:
                base_dir_out = os.path.join(meta["checkpoint_dir"], "base_eval")
                for config_name, core_out in core_results_by_config.items():
                    csv_name = f"{model_slug}_{config_name}.csv"
                    output_csv_path = os.path.join(base_dir_out, csv_name)
                    _write_core_csv(output_csv_path, core_out)
                    logger.info(f"Results written to: {output_csv_path}")

            # Use the first config's results for the report when a static
            # target-delta config was explicitly requested.
            core_results = next(iter(core_results_by_config.values()), None)
        else:
            # Decoder-only / HuggingFace: single evaluation
            with autocast_ctx:
                core_results = evaluate_core(
                    model,
                    tokenizer,
                    device,
                    max_per_task=cfg.max_per_task,
                    seed=_core_seed_from_cfg(cfg),
                )

            if ddp_rank == 0:
                if not is_hf_model:
                    output_csv_path = os.path.join(
                        meta["checkpoint_dir"], "base_eval", f"{model_slug}.csv"
                    )
                else:
                    output_csv_path = os.path.join(get_base_dir(), "base_eval", f"{model_slug}.csv")
                _write_core_csv(output_csv_path, core_results)
                logger.info(f"\nResults written to: {output_csv_path}")

        if core_results is not None:
            logger.info(f"CORE metric: {core_results['core_metric']:.4f}")

    # --- Log to report ---
    report_data: list[dict[str, Any]] = [{"model": model_name}]

    if core_results:
        if raw_encdec is not None and EvalMode.CORE in eval_modes:
            # Multi-config: report all CORE scores
            for config_name, core_out in core_results_by_config.items():
                report_data[0][f"CORE metric ({config_name})"] = core_out["core_metric"]
        else:
            report_data[0]["CORE metric"] = core_results["core_metric"]
        report_data.append(core_results["centered_results"])

    if bpb_results:
        report_data[0]["train bpb"] = bpb_results.get(DatasetSplit.TRAIN)
        report_data[0]["val bpb"] = bpb_results.get(DatasetSplit.VAL)

    if bpb_conditioned_results:
        for split_str, conditioned in bpb_conditioned_results.items():
            for k, v in conditioned.items():
                report_data[0][f"{split_str} {k}"] = v

    if samples_by_config:
        for config_name, sample_list in samples_by_config.items():
            report_data.append(
                {f"sample {i} ({config_name})": s for i, s in enumerate(sample_list)}
            )
    ckpt_dir = meta["checkpoint_dir"] if not is_hf_model else None
    get_report(checkpoint_dir=ckpt_dir).log(section="Base model evaluation", data=report_data)

    # PR-C: per-target-real-config eval loop. Orchestrated by the shared
    # ``run_target_real_evals`` helper — rank-0 barrier, per-config provider
    # construction, output dir creation, and yaml writing all live there.
    if target_real_configs and ddp_rank == 0:
        with autocast_ctx:
            run_target_real_evals(
                target_real_configs=target_real_configs,
                cfg=cfg,
                output_dir_base=os.path.join(meta["checkpoint_dir"], "base_eval"),
                ddp_rank=ddp_rank,
                per_config_eval_fn=lambda trc, provider: _base_eval_target_real_per_config(
                    raw_encdec=raw_encdec,
                    tokenizer=tokenizer,
                    device=device,
                    trc=trc,
                    provider=provider,
                    cfg=cfg,
                    eval_modes=eval_modes,
                    target_mode=target_mode,
                ),
                checkpoint_dir=meta["checkpoint_dir"],
                report_section_prefix=f"Base eval {cfg.source}",
            )

    compute_cleanup()


if __name__ == "__main__":
    main()
