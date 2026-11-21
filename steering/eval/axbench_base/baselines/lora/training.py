"""Per-concept LoRA training loop (Task 24, axbench_base Stage 3b).

Tiny ~100-step, single-device, synchronous fine-tune of
``LoRAAdapter``-wrapped ``nn.Linear`` sublayers at
``cfg.layer_hooks.diffmean_and_probe_layer``. Unlike
``scripts/chat_sft.py`` (full DDP harness with wandb, distributed
optimizers, LR schedules) this module runs one concept at a time on
a decoder-only model and reuses ``GPT.forward(idx, targets=...)``
for the loss (``F.cross_entropy`` with ``ignore_index=-1``).

``mode == "positive"`` trains on ``concept_rich`` rows (adapter
learns to PRODUCE the concept); ``mode == "negative"`` trains on
``concept_avoiding`` rows (learns to SUPPRESS it). Per-concept batch
sampling is seeded at ``cfg.generation.master_seed + concept_id``
for determinism. Save format is safetensors with
``layer_{L}_{parent}_{child}_lora_A`` / ``..._lora_B`` keys, consumed
in lockstep by Task 25's loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch as t
import torch.nn as nn
from loguru import logger
from safetensors.torch import save_file

from nanochat.tokenizer import get_tokenizer
from steering.eval.axbench_base import paths
from steering.eval.axbench_base.artifact_metadata import (
    validate_extraction_tokenizer_metadata,
)
from steering.eval.axbench_base.baselines._capture import (
    _filter_rows_for_concept,
)
from steering.eval.axbench_base.baselines.lora._attach import (
    _ATTACH_PATHS,
    _attach_lora_to_model,
    _AttachedAdapterRef,
    _collect_attached_adapters,
    _detach_lora_from_model,
)
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.positive_qc_contract import (
    validate_positive_qc_contract,
)
from steering.eval.axbench_base.positive_selection_contract import (
    validate_positive_selection_contract,
)
from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.eval.axbench_base.synth_continuation_contract import (
    validate_stage1_continuation_contract,
)
from steering.eval.axbench_base.tokenizer_utils import tokenize_labeled_pair
from steering.eval.axbench_base.trainable_concepts import (
    concept_ids_present_in_rows,
    require_concept_population_coverage,
)
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    Concept,
    SteeringMode,
)

__all__ = (
    "_ATTACH_PATHS",
    "_AttachedAdapterRef",
    "_attach_lora_to_model",
    "_build_lora_batch",
    "_collect_attached_adapters",
    "_detach_lora_from_model",
    "_row_to_sequence_and_target",
    "_sample_batch_indices",
    "_train_lora_single_concept",
    "_train_step",
    "get_tokenizer",
    "train_lora_all",
)


# Cross-entropy ignore-index sentinel. Matches ``GPT.forward``'s
# ``ignore_index=-1``.
_IGNORE_INDEX: int = -1

# GPT.forward has causal attention but no padding mask.  Right-padding is
# therefore invisible to every real token, while left-padding would become
# real prior context and shift the row's RoPE positions.  Token id 0 is only a
# filler at ignored, trailing positions; using BOS here would create additional
# pipeline-owned-looking BOS tokens in the model input.
_RIGHT_PAD_TOKEN_ID: int = 0

_VALID_MODES: frozenset[SteeringMode] = frozenset(
    {SteeringMode.POSITIVE, SteeringMode.NEGATIVE},
)


def _sample_batch_indices(
    *, pool_size: int, batch_size: int, generator: t.Generator,
) -> t.Tensor:
    """Draw ``batch_size`` row indices with replacement via ``generator``."""
    return t.randint(
        low=0,
        high=pool_size,
        size=(batch_size,),
        generator=generator,
        device="cpu",
    )


def _row_to_sequence_and_target(
    row: dict[str, Any], *, tokenizer: Any,
) -> tuple[list[int], list[int]]:
    """Build ``(input_ids, targets)`` for one training row.

    ``input_ids`` is the BOS-prepended tokenization of
    ``prefix_text + output_text``. ``targets`` is the next-token shift
    with prefix and final positions set to ``_IGNORE_INDEX``, so only
    output-token predictions contribute to the loss. If the sequence
    has length ``L`` and the prefix occupies ``P`` tokens, the target
    is ``[IGNORE] * (P - 1) + full[P:L] + [IGNORE]`` (length ``L``).
    """
    prefix_text = row.get("prefix_text")
    output_text = row.get("output_text")
    stored_boundary = row.get("n_prefix_nanochat_tokens")
    if not isinstance(prefix_text, str):
        raise AxBenchBaseError(
            "LoRA synth row prefix_text must be a string, got "
            f"{type(prefix_text).__name__}."
        )
    if not isinstance(output_text, str) or not output_text:
        raise AxBenchBaseError(
            "LoRA synth row output_text must be a non-empty string, got "
            f"{output_text!r}."
        )
    if (
        not isinstance(stored_boundary, int)
        or isinstance(stored_boundary, bool)
        or stored_boundary <= 0
    ):
        raise AxBenchBaseError(
            "LoRA synth row n_prefix_nanochat_tokens must be a positive "
            f"integer, got {stored_boundary!r}."
        )
    full, n_prefix = tokenize_labeled_pair(tokenizer, prefix_text, output_text)
    if n_prefix != stored_boundary:
        raise AxBenchBaseError(
            "Synth labeled-pair boundary mismatch during LoRA training: "
            f"stored={stored_boundary}, exact={n_prefix}. Regenerate the "
            "synth parquet under a fresh run tag."
        )
    targets = [_IGNORE_INDEX] * (n_prefix - 1) + full[n_prefix:] + [_IGNORE_INDEX]
    return full, targets


def _build_lora_batch(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Any,
    device: t.device,
) -> tuple[t.Tensor, t.Tensor]:
    """Right-pad per-row ``(input, target)`` pairs into stacked tensors.

    The causal decoder cannot see trailing input pads from any real token.
    Target pad positions use ``_IGNORE_INDEX``, so trailing pads also
    contribute no loss.  Left-padding is invalid here: without an attention
    mask those tokens would alter the row's context and positional coordinates.
    """
    pair = [_row_to_sequence_and_target(row, tokenizer=tokenizer) for row in rows]
    max_len = max(len(inp) for inp, _ in pair)

    padded_inputs: list[list[int]] = []
    padded_targets: list[list[int]] = []
    for inp, tgt in pair:
        pad = max_len - len(inp)
        padded_inputs.append(inp + [_RIGHT_PAD_TOKEN_ID] * pad)
        padded_targets.append(tgt + [_IGNORE_INDEX] * pad)

    return (
        t.tensor(padded_inputs, dtype=t.long, device=device),
        t.tensor(padded_targets, dtype=t.long, device=device),
    )


def _train_step(
    *,
    model: nn.Module,
    rows_for_step: list[dict[str, Any]],
    tokenizer: Any,
    optimizer: t.optim.Optimizer,
    device: t.device,
    step_index: int,
) -> float:
    """One forward+backward+optimizer step; returns the loss value.

    Uses ``GPT.forward(idx, targets=...)`` which already returns a
    scalar cross-entropy loss with ``ignore_index=-1``. Raises
    ``AxBenchBaseError`` on NaN loss (upstream issue that won't fix
    itself by clipping).
    """
    # Bare-name reference so tests can monkeypatch ``_build_lora_batch``
    # on the module globals (bare names resolve at call time).
    with profile_region("stage3d.lora_build_batch", rows=len(rows_for_step)):
        idx, targets = _build_lora_batch(
            rows=rows_for_step, tokenizer=tokenizer, device=device,
        )
    optimizer.zero_grad(set_to_none=True)
    with profile_region(
        "stage3d.lora_forward_backward",
        rows=len(rows_for_step),
        step_index=step_index,
    ):
        loss = model(idx, targets=targets)
        if t.isnan(loss).any():
            raise AxBenchBaseError(
                f"NaN loss at step {step_index}. This usually means an "
                "upstream training issue (exploded activations, bad batch "
                "construction); inspect the model and the corpus."
            )
        loss.backward()
    with profile_region("stage3d.lora_optimizer_step"):
        optimizer.step()
    return float(loss.detach().cpu().item())


def _select_training_pool(
    rows: list[dict[str, Any]], *, concept_id: int, mode: SteeringMode,
) -> list[dict[str, Any]]:
    """Pick the per-concept training pool — positives or avoidances."""
    pos_rows, neg_rows = _filter_rows_for_concept(rows, concept_id)
    # Caller validates ``mode`` up front, so this is exhaustive.
    return pos_rows if mode == SteeringMode.POSITIVE else neg_rows


def _save_lora_adapter(
    *,
    refs: list[_AttachedAdapterRef],
    out_path: Path,
    metadata: dict[str, str],
) -> None:
    """Save ``lora_A``/``lora_B`` tensors + metadata to safetensors.

    Key format: ``layer_{L}_{parent}_{child}_lora_A`` / ``..._lora_B``,
    consumed in lockstep by Task 25's loader. Each tensor is
    detach-cloned so it has its own storage (safetensors rejects
    aliased storage).
    """
    state: dict[str, t.Tensor] = {}
    for ref in refs:
        prefix = f"layer_{ref.layer}_{ref.parent_name}_{ref.child_name}"
        state[f"{prefix}_lora_A"] = ref.adapter.lora_A.detach().clone().contiguous().cpu()
        state[f"{prefix}_lora_B"] = ref.adapter.lora_B.detach().clone().contiguous().cpu()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(out_path), metadata=metadata)


def _train_lora_single_concept(
    cfg: AxBenchBaseConfig,
    target_model: nn.Module,
    target_model_name: str,
    mode: SteeringMode,
    concept_id: int,
    rows: list[dict[str, Any]],
    tokenizer: Any,
) -> Path:
    """Train one concept's LoRA adapters and save them. Returns the
    output path. The model is restored to its original (no-adapter,
    original train/eval mode) state on return. Raises
    ``AxBenchBaseError`` on ``cfg.run_tag is None``, on a pool smaller
    than ``cfg.lora.min_training_rows``, or on NaN loss.
    """
    if cfg.run_tag is None:
        raise AxBenchBaseError(
            "cfg.run_tag must be set before calling train_lora_all",
        )

    layer = cfg.layer_hooks.diffmean_and_probe_layer
    training_rows = _select_training_pool(rows, concept_id=concept_id, mode=mode)
    if len(training_rows) < cfg.lora.min_training_rows:
        raise AxBenchBaseError(
            f"LoRA concept {concept_id} mode={mode}: only "
            f"{len(training_rows)} training rows survived filtering; "
            f"min_training_rows={cfg.lora.min_training_rows}. "
            "Either increase the corpus size or lower min_training_rows."
        )

    device = next(target_model.parameters()).device
    was_training = target_model.training
    with profile_region("stage3d.lora_attach", concept_id=concept_id):
        original = _attach_lora_to_model(target_model, layer=layer, lora_cfg=cfg.lora)
    try:
        with profile_region("stage3d.lora_collect_adapters", concept_id=concept_id):
            refs = _collect_attached_adapters(target_model, layer=layer)
        lora_params: list[t.Tensor] = [
            p for ref in refs for p in (ref.adapter.lora_A, ref.adapter.lora_B)
        ]
        optimizer = t.optim.AdamW(lora_params, lr=cfg.lora.lr)
        generator = t.Generator(device="cpu").manual_seed(
            cfg.generation.master_seed + concept_id,
        )

        target_model.train()
        with profile_region(
            "stage3d.lora_train_steps",
            concept_id=concept_id,
            rows=cfg.lora.steps,
        ):
            for step_index in range(cfg.lora.steps):
                indices = _sample_batch_indices(
                    pool_size=len(training_rows),
                    batch_size=cfg.lora.batch_size,
                    generator=generator,
                )
                rows_for_step = [training_rows[int(i)] for i in indices.tolist()]
                # Bare-name reference so tests can monkeypatch ``_train_step``
                # on the module globals.
                loss_value = _train_step(
                    model=target_model,
                    rows_for_step=rows_for_step,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    device=device,
                    step_index=step_index,
                )
                logger.debug(
                    f"LoRA concept {concept_id} mode={mode} "
                    f"step {step_index + 1}/{cfg.lora.steps}: loss={loss_value:.4f}",
                )

        out_path = paths.AxBenchBasePaths.from_config(cfg).lora_adapter(
            mode=mode,
            concept_id=concept_id,
        )
        with profile_region("stage3d.lora_save_adapter", concept_id=concept_id):
            _save_lora_adapter(
                refs=refs,
                out_path=out_path,
                metadata={
                    "concept_id": str(concept_id),
                    "mode": mode,
                    "target_model": target_model_name,
                    "layer": str(layer),
                    "n_train": str(len(training_rows)),
                    "rank": str(cfg.lora.rank),
                    "alpha": str(cfg.lora.alpha),
                    "lr": str(cfg.lora.lr),
                    "steps": str(cfg.lora.steps),
                    "batch_size": str(cfg.lora.batch_size),
                },
            )
        logger.info(
            f"LoRA concept {concept_id} mode={mode}: wrote adapter "
            f"(n_train={len(training_rows)}, steps={cfg.lora.steps}) "
            f"to {out_path}"
        )
        return out_path
    finally:
        # Always detach + restore prior train/eval mode, even on error.
        with profile_region("stage3d.lora_detach", concept_id=concept_id):
            _detach_lora_from_model(target_model, original=original)
        if not was_training:
            target_model.eval()


def train_lora_all(
    cfg: AxBenchBaseConfig,
    target_model: nn.Module,
    target_model_name: str,
    mode: SteeringMode,
    concepts: list[Concept],
    *,
    training_corpus_path: Path,
) -> list[Path]:
    """Train per-concept LoRA adapters for every concept in ``concepts``.

    Reads the corpus once, then dispatches to
    ``_train_lora_single_concept`` per concept in order.

    Returns:
        List of output paths (one path per PRESENT concept, in ``concepts``
        order — concepts absent from the synth corpus are skipped (issue #508)).

    Raises:
        AxBenchBaseError: on empty ``concepts``, unknown ``mode``,
            ``cfg.run_tag is None``, or any per-concept failure.
    """
    if not concepts:
        raise AxBenchBaseError(
            "train_lora_all called with empty concepts list",
        )
    if mode not in _VALID_MODES:
        raise AxBenchBaseError(
            f"Unknown mode for LoRA training: {mode!r}. "
            f"Expected one of {sorted(_VALID_MODES)}."
        )

    with profile_region("stage3d.lora_load_training_rows"):
        table = pq.read_table(training_corpus_path)
        validate_extraction_tokenizer_metadata(
            training_corpus_path,
            expected_tokenizer_id=cfg.extraction_tokenizer_id,
            consumer="Stage 3d LoRA",
            artifact_name="synth training parquet",
            schema=table.schema,
        )
        validate_stage1_continuation_contract(
            training_corpus_path,
            consumer="Stage 3d LoRA",
            expected_role=(
                "stage1b_positive"
                if mode == SteeringMode.POSITIVE
                else "stage1c_negative"
            ),
            schema=table.schema,
        )
        if mode == SteeringMode.POSITIVE:
            validate_positive_qc_contract(
                training_corpus_path,
                consumer="Stage 3d LoRA",
                schema=table.schema,
                expected_accept_threshold=cfg.accept_threshold,
            )
            validate_positive_selection_contract(
                training_corpus_path,
                consumer="Stage 3d LoRA",
                schema=table.schema,
                expected_cfg=cfg,
            )
        rows = table.to_pylist()
    logger.info(
        f"LoRA: loaded {len(rows)} training rows from {training_corpus_path} "
        f"for {len(concepts)} concepts (mode={mode}, target={target_model_name})"
    )
    present = concept_ids_present_in_rows(rows)
    require_concept_population_coverage(
        (concept.concept_id for concept in concepts),
        present,
        mode=mode,
        consumer="Stage 3d LoRA",
    )
    tokenizer = get_tokenizer()

    out_paths: list[Path] = []
    for concept in concepts:
        with profile_region(
            "stage3d.lora_train_single",
            concept_id=concept.concept_id,
            mode=mode,
        ):
            out_paths.append(
                _train_lora_single_concept(
                    cfg=cfg,
                    target_model=target_model,
                    target_model_name=target_model_name,
                    mode=mode,
                    concept_id=concept.concept_id,
                    rows=rows,
                    tokenizer=tokenizer,
                )
            )
    record_counter("stage3d.lora_adapters_written", len(out_paths), mode=mode)
    return out_paths
