"""
Utilities for saving and loading model/optim/state checkpoints.
"""

import collections
import glob
import json
import os
import re
import time
from enum import StrEnum

import torch
from loguru import logger

from errors import DataLoadError, FeatureVocabError
from nanochat.atomic_io import atomic_json_dump, atomic_torch_save
from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig, OptimizerKind
from nanochat.tokenizer import RustBPETokenizer, TransformersTokenizer, get_tokenizer
from steering._common.checkpoint_walk import find_last_step_or_none
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    feature_prediction_config_from_metadata,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_pipeline import feature_selection_dirname
from steering.conditional.feature_vocab import (
    META_KEY as FEATURE_VOCAB_META_KEY,
)
from steering.conditional.feature_vocab import (
    feature_vocab_from_state_dict,
)
from steering.profiling.heartbeat import Heartbeat
from steering.profiling.rank_skew import log_rank_duration_skew

# Register ``OptimizerKind`` (a ``StrEnum`` member that appears as
# ``param_group['kind']`` in MuonAdamW optimizer state dicts) with
# ``torch.serialization``'s safe-globals allowlist so that
# ``load_checkpoint(load_optimizer=True)`` can decode the optimizer
# state without falling back to the legacy (unsafe) pickle unpickler.
# Without this, ``torch.load(weights_only=True)`` raises
# ``WeightsUnpickler error: GLOBAL nanochat.gpt.OptimizerKind was not
# an allowed global``. Registering here means every caller of
# ``load_checkpoint`` (training scripts + tests) gets the allowlist
# entry as a side effect of importing the manager.
#
# Moved BELOW the imports (it was interleaved with them, which put six module-level
# imports after executable code): registration happens at module import either way, and
# the project rule is imports-at-top, no exceptions.
torch.serialization.add_safe_globals([OptimizerKind])


# Return contract of the checkpoint loaders below: they yield whichever model / tokenizer kind
# the checkpoint holds. (`nanochat/` is excluded from `ty`, so these aliases document the contract
# for readers and IDEs rather than being type-checked.)
LoadedModel = GPT | EncoderDecoderGPT
LoadedTokenizer = RustBPETokenizer | TransformersTokenizer


class ModelPhase(StrEnum):
    TRAIN = "train"
    EVAL = "eval"


class ModelSource(StrEnum):
    BASE = "base"
    SFT = "sft"
    STANDARD = "standard"
    SAFETY = "safety"
    CAPABILITIES = "capabilities"
    RL = "rl"
    ENCDEC = "encdec"


def log0(message):
    if int(os.environ.get("RANK", 0)) == 0:
        logger.info(message)


def _patch_missing_config_keys(model_config_kwargs):
    """Add default values for new config keys missing in old checkpoints."""
    # Old models were trained with full context (no sliding window)
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
        log0("Patching missing window_pattern in model config to 'L'")


def _patch_missing_keys(model_data, model_config):
    """Add default values for new parameters that may be missing in old checkpoints."""
    n_layer = model_config.n_layer
    # resid_lambdas defaults to 1.0 (identity scaling)
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)
        log0("Patching missing resid_lambdas in model data to 1.0")
    # x0_lambdas defaults to 0.0 (disabled)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)
        log0("Patching missing x0_lambdas in model data to 0.0")


def _fuse_qkv_weights(model_data: dict[str, torch.Tensor]) -> None:
    """Fuse old separate c_q/c_k/c_v weights into c_qkv for backward compat.

    Old checkpoints stored attention projections as three separate linear layers
    (c_q, c_k, c_v). The current model uses a single fused c_qkv projection.
    This mutates *model_data* in-place, concatenating [q, k, v] along dim=0.
    """
    # Collect layer prefixes that have old-style separate keys
    prefixes: list[str] = []
    for key in list(model_data.keys()):
        if key.endswith(".attn.c_q.weight"):
            prefixes.append(key.removesuffix("c_q.weight"))

    for prefix in prefixes:
        q = model_data.pop(f"{prefix}c_q.weight")
        k = model_data.pop(f"{prefix}c_k.weight")
        v = model_data.pop(f"{prefix}c_v.weight")
        model_data[f"{prefix}c_qkv.weight"] = torch.cat([q, k, v], dim=0)
        log0(f"Fused {prefix}c_q/c_k/c_v -> {prefix}c_qkv")


def _recover_model_config(checkpoint_dir: str, meta_data: dict) -> dict:
    """Recover model_config for SFT checkpoints that don't save it.

    Tries two strategies:
    1. Load model_config from the parent checkpoint directory.
    2. Reconstruct from user_config fields.
    """
    # Strategy 1: look in parent directory for a pretrain checkpoint meta.
    parent_dir = os.path.dirname(checkpoint_dir.rstrip("/"))
    parent_step = find_last_step_or_none(parent_dir)
    if parent_step is not None:
        parent_meta_path = os.path.join(parent_dir, f"meta_{parent_step:06d}.json")
        if os.path.exists(parent_meta_path):
            with open(parent_meta_path, encoding="utf-8") as f:
                parent_meta = json.load(f)
            if "model_config" in parent_meta:
                log0(f"Recovered model_config from parent checkpoint: {parent_meta_path}")
                return parent_meta["model_config"]

    # Strategy 2: reconstruct from user_config.
    user_cfg = meta_data.get("user_config", {})
    depth = user_cfg.get("depth", 26)
    aspect_ratio = user_cfg.get("aspect_ratio", 64)
    head_dim = user_cfg.get("head_dim", 128)
    n_embd = depth * aspect_ratio
    n_head = n_embd // head_dim
    model_config = {
        "sequence_len": user_cfg.get("max_seq_len", 2048),
        "vocab_size": 32768,
        "n_layer": depth,
        "n_head": n_head,
        "n_kv_head": n_head,
        "n_embd": n_embd,
        "window_pattern": user_cfg.get("window_pattern", "SSSL"),
    }
    log0(f"Reconstructed model_config from user_config: {model_config}")
    return model_config


def _recover_parent_user_config(checkpoint_dir: str) -> dict:
    """Load user_config from the parent checkpoint directory."""
    parent_dir = os.path.dirname(checkpoint_dir.rstrip("/"))
    parent_step = find_last_step_or_none(parent_dir)
    if parent_step is not None:
        parent_meta_path = os.path.join(parent_dir, f"meta_{parent_step:06d}.json")
        if os.path.exists(parent_meta_path):
            with open(parent_meta_path, encoding="utf-8") as f:
                parent_meta = json.load(f)
            if "user_config" in parent_meta:
                log0(f"Recovered user_config from parent: {parent_meta_path}")
                return parent_meta["user_config"]
    raise FileNotFoundError(
        f"Cannot recover encoder/cross_attention config: no parent "
        f"checkpoint found above {checkpoint_dir}"
    )


def _iter_ancestor_metas(checkpoint_dir: str):
    """Walk upwards from *checkpoint_dir*, yielding each ancestor's meta dict.

    Stops at the filesystem root. Skips directories with no ``meta_*.json``
    files. Each yielded dict is the parsed JSON of the ancestor's latest
    meta file.
    """
    parent_dir = os.path.dirname(checkpoint_dir.rstrip("/"))
    visited: set[str] = set()
    while parent_dir and parent_dir not in visited:
        visited.add(parent_dir)
        if os.path.isdir(parent_dir):
            try:
                parent_step = find_last_step(parent_dir)
            except FileNotFoundError:
                parent_step = None
            if parent_step is not None:
                parent_meta_path = os.path.join(parent_dir, f"meta_{parent_step:06d}.json")
                if os.path.exists(parent_meta_path):
                    try:
                        with open(parent_meta_path, encoding="utf-8") as f:
                            yield parent_meta_path, json.load(f)
                    except (OSError, json.JSONDecodeError):
                        pass
        next_parent = os.path.dirname(parent_dir)
        if next_parent == parent_dir:
            break
        parent_dir = next_parent


def _extract_encdec_fields(meta: dict) -> dict:
    """Pull encoder / cross_attention / bucket metadata out of a meta dict.

    Encoder-decoder checkpoints store these two ways depending on phase:
    pretrain metas keep them under ``user_config.encoder`` /
    ``user_config.cross_attention``; SFT metas instead carry top-level
    ``encoder_config`` / ``xattn_config`` / ``num_buckets``. This helper
    reads either form and returns a dict with whichever keys were found.
    """
    user_cfg = meta.get("user_config") or {}
    out: dict = {}
    encoder = user_cfg.get("encoder")
    if encoder is None:
        encoder = meta.get("encoder_config")
    if encoder is not None:
        out["encoder"] = encoder
    xattn = user_cfg.get("cross_attention")
    if xattn is None:
        xattn = meta.get("xattn_config")
    if xattn is not None:
        out["cross_attention"] = xattn
    num_buckets = user_cfg.get("num_buckets")
    if num_buckets is None:
        num_buckets = meta.get("num_buckets")
    if num_buckets is not None:
        out["num_buckets"] = num_buckets
    negative_selection = user_cfg.get("negative_selection")
    if negative_selection is None:
        negative_selection = meta.get("negative_selection")
    if negative_selection is not None:
        out["negative_selection"] = negative_selection
    feature_prediction = user_cfg.get("feature_prediction")
    if feature_prediction is None:
        feature_prediction = meta.get("feature_prediction_config")
    if feature_prediction is not None:
        out["feature_prediction"] = feature_prediction
    return out


def _recover_encdec_ancestor_config(checkpoint_dir: str) -> dict:
    """Walk ancestor checkpoints, merging the first encdec fields we find.

    Closest ancestor wins for each field. Traversal continues until the required
    encoder-decoder structural fields are populated and any optional
    ``negative_selection`` has also been recovered if present upstream. Raises
    ``FileNotFoundError`` if no ancestor supplies an ``encoder`` config.
    """
    merged: dict = {}
    for meta_path, meta in _iter_ancestor_metas(checkpoint_dir):
        fields = _extract_encdec_fields(meta)
        if not fields:
            continue
        logged = False
        for key, value in fields.items():
            if key not in merged:
                merged[key] = value
                if not logged:
                    log0(f"Recovered encdec fields from ancestor: {meta_path}")
                    logged = True
        if {
            "encoder",
            "cross_attention",
            "num_buckets",
            "negative_selection",
        }.issubset(merged):
            break
    if "encoder" not in merged:
        raise FileNotFoundError(
            f"Cannot recover encoder config: no ancestor of {checkpoint_dir} "
            "carries user_config.encoder or top-level encoder_config"
        )
    return merged


def _report_checkpoint_io_timing(label, elapsed_s, rank, report_rank_skew):
    """Local timing line, or the gathered cross-rank skew report when opted in.

    ``report_rank_skew=True`` runs a collective and therefore requires EVERY
    rank of an initialized process group to make the same call — callers with
    rank-conditional checkpoint calls (e.g. a rank-0-only interval save) must
    leave it False or they will deadlock the group.
    """
    if report_rank_skew:
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 1
        )
        log_rank_duration_skew(label, elapsed_s, rank=rank, world_size=world_size)
    else:
        logger.info(f"{label}: {elapsed_s:.1f}s (this rank)")


def save_checkpoint(
    checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0,
    report_rank_skew=False,
):
    started = time.monotonic()
    with Heartbeat(f"checkpoint save (step {step})"):
        if rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            # Save the model state parameters atomically (tmp + fsync + replace).
            # A crash mid-write can leave unique tmp debris but never a torn
            # model_*.pt, so find_last_step cannot silently return corruption.
            model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
            atomic_torch_save(model_data, model_path)
            logger.info(f"Saved model parameters to: {model_path}")
            # Save the metadata dict as json atomically.
            meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
            atomic_json_dump(meta_data, meta_path)
            logger.info(f"Saved metadata to: {meta_path}")
        # Note that optimizer state is sharded across ranks, so each rank must save its own.
        if optimizer_data is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
            atomic_torch_save(optimizer_data, optimizer_path)
            logger.info(f"Saved optimizer state to: {optimizer_path}")
    _report_checkpoint_io_timing(
        f"checkpoint save (step {step})",
        time.monotonic() - started,
        rank,
        report_rank_skew,
    )


def load_checkpoint(
    checkpoint_dir, step, device, load_optimizer=False, rank=0, report_rank_skew=False,
    broadcast_model=False,
):
    # Resolve step=None to latest checkpoint
    if step is None:
        step = find_last_step(checkpoint_dir)
    started = time.monotonic()
    with Heartbeat(f"checkpoint load (step {step})"):
        # Load the model state.
        # weights_only=True pins torch.load to the safe-globals allowlist and
        # refuses arbitrary pickle opcodes (GLOBAL/REDUCE on disallowed modules).
        # This closes the pickle RCE vector: a malicious .pt file with
        # __reduce__=(os.system, ("...",)) can no longer execute on load.
        # Checkpoint state dicts are tensor-only + primitive scalars, both of
        # which are fully supported by the default safe-globals allowlist.
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        if not (
            broadcast_model
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and (world_size := torch.distributed.get_world_size()) > 1
        ):
            # This is the legitimate plain-load path, not fallback from a
            # failed collective.
            model_data = torch.load(model_path, map_location=device, weights_only=True)
        else:
            distributed_rank = torch.distributed.get_rank()
            model_data = None
            # Phase 1: rank 0 loads and describes the complete state dict.
            if distributed_rank == 0:
                try:
                    model_data = torch.load(
                        model_path,
                        map_location=device,
                        weights_only=True,
                    )
                    schema = []
                    non_tensor_values = {}
                    for key, value in model_data.items():
                        if isinstance(value, torch.Tensor):
                            schema.append(
                                (
                                    key,
                                    str(value.dtype),
                                    tuple(value.shape),
                                    value.requires_grad,
                                )
                            )
                        else:
                            schema.append((key, "", (), False))
                            non_tensor_values[key] = value
                    phase_one = {
                        "schema": schema,
                        "non_tensor_values": non_tensor_values,
                        "ordered_dict": isinstance(model_data, collections.OrderedDict),
                        "state_dict_metadata": getattr(model_data, "_metadata", None),
                    }
                except Exception as exc:
                    load_error = exc
                    phase_one = {
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            else:
                phase_one = None

            phase_one_payload = [phase_one]
            torch.distributed.broadcast_object_list(phase_one_payload, src=0)
            phase_one = phase_one_payload[0]
            if "error" in phase_one:
                if distributed_rank == 0:
                    raise load_error
                raise RuntimeError(
                    f"rank 0 failed to load model checkpoint: {phase_one['error']}"
                )

            schema = phase_one["schema"]
            non_tensor_values = phase_one["non_tensor_values"]
            # Phase 2: receivers allocate everything before any tensor transfer.
            allocation_status = None
            if distributed_rank != 0:
                try:
                    if phase_one["ordered_dict"]:
                        model_data = collections.OrderedDict()
                    else:
                        model_data = {}
                    for key, dtype_str, shape, _requires_grad in schema:
                        if key in non_tensor_values:
                            model_data[key] = non_tensor_values[key]
                        else:
                            dtype = getattr(torch, dtype_str.removeprefix("torch."))
                            model_data[key] = torch.empty(shape, dtype=dtype, device=device)
                    if phase_one["state_dict_metadata"] is not None:
                        model_data._metadata = phase_one["state_dict_metadata"]
                except Exception as exc:
                    allocation_status = (
                        f"rank {distributed_rank}: {type(exc).__name__}: {exc}"
                    )
            allocation_statuses = [None] * world_size
            torch.distributed.all_gather_object(
                allocation_statuses,
                allocation_status,
            )
            if allocation_failures := [
                status for status in allocation_statuses if status is not None
            ]:
                raise RuntimeError(
                    "model checkpoint tensor allocation failed: "
                    + "; ".join(allocation_failures)
                )

            # The token-embedding and lm_head weights are explicitly untied
            # (gpt.py header); receivers are contracted to get value-equal copies.
            # Phase 3: transfer tensors only after every receiver is ready.
            for key, _dtype_str, _shape, requires_grad in schema:
                if key not in non_tensor_values:
                    tensor = model_data[key]
                    if distributed_rank == 0:
                        # Gloo transfers raw storage order while NCCL rejects
                        # non-contiguous collectives, so stage only the send view.
                        broadcast_tensor = (
                            tensor if tensor.is_contiguous() else tensor.contiguous()
                        )
                    else:
                        broadcast_tensor = tensor
                    # A backend failure mid-loop is process-group-fatal, like any
                    # NCCL failure in training, and is outside this error protocol.
                    torch.distributed.broadcast(broadcast_tensor, src=0)
                    if distributed_rank != 0:
                        tensor.requires_grad_(requires_grad)
        # Load the optimizer state if requested. Our optimizer state dicts only
        # contain tensors, ints/floats, strings, tuples, lists and plain dicts -
        # all of which the weights_only=True unpickler accepts. See
        # test_load_optimizer_state_works for the contract regression.
        optimizer_data = None
        if load_optimizer:
            optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
            optimizer_data = torch.load(optimizer_path, map_location=device, weights_only=True)
        # Load the metadata
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, encoding="utf-8") as f:
            meta_data = json.load(f)
    _report_checkpoint_io_timing(
        f"checkpoint load (step {step})",
        time.monotonic() - started,
        rank,
        report_rank_skew,
    )
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.

    ``device`` may be a string accepted by ``torch.device(...)`` (e.g.
    ``"cpu"``, ``"cuda"``, ``"cuda:0"``) or a pre-built ``torch.device``
    object. The argument is coerced to a ``torch.device`` at the top of the
    function.

    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    # Defensive: accept str or torch.device. Inner helpers read
    # ``device.type`` directly, so strings would raise AttributeError
    # deep in the call stack. Operator override of the "nanochat is
    # read-only" convention — see
    # docs/superpowers/specs/2026-05-11-axbench-eval-hardening-design.md (W7).
    device = torch.device(device)
    assert phase in ModelPhase, f"Invalid phase: {phase}. Must be one of {list(ModelPhase)}"
    model_data, optimizer_data, meta_data = load_checkpoint(
        checkpoint_dir, step, device, load_optimizer=False
    )
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config)
    _fuse_qkv_weights(model_data)
    # Phase logs: rank 0 of an axbench-bpb torchrun deadlocked here in 2026-05-13
    # diagnostics — these intermediate prints scoped the hang to a single line.
    # Keeping them in build_model so future bugs in this 5-step sequence are
    # observable rather than silent.
    log0("build_model: building GPT on meta device")
    with torch.device("meta"):
        model = GPT(model_config)
    log0(f"build_model: model.to_empty(device={device})")
    model.to_empty(device=device)
    log0("build_model: model.init_buffers()")
    model.init_buffers()
    log0("build_model: model.load_state_dict(...)")
    model.load_state_dict(model_data, strict=True, assign=True)
    log0(f"build_model: setting phase to {phase}")
    if phase == ModelPhase.EVAL:
        model.eval()
    else:
        model.train()
    log0("build_model: get_tokenizer()")
    tokenizer = get_tokenizer()
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], (
        f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model_config_kwargs['vocab_size']}"
    )
    log0("build_model: done")
    return model, tokenizer, meta_data


def build_encdec_model(
    checkpoint_dir: str,
    step: int,
    device: str | torch.device,
    phase: ModelPhase | str,
    *,
    eval_conditioning: EvalConditioning,
) -> tuple[EncoderDecoderGPT, LoadedTokenizer, dict]:
    """Build an EncoderDecoderGPT from a checkpoint.

    Reconstructs decoder, encoder, and cross-attention configs from the
    saved metadata, infers ``n_sae_features`` from the state dict, and
    loads the model weights.

    ``device`` may be a string accepted by ``torch.device(...)`` (e.g.
    ``"cpu"``, ``"cuda"``, ``"cuda:0"``) or a pre-built ``torch.device``
    object. The argument is coerced to a ``torch.device`` at the top of the
    function.

    ``eval_conditioning`` (#695 load-time contract) is a REQUIRED ``EvalConditioning`` stance that
    says whether/how this load's caller conditions the encoder, so a compacted checkpoint cannot
    silently be evaluated on the wrong feature universe. It is consulted only when ``phase ==
    ModelPhase.EVAL``; for other phases the caller must still declare a stance (e.g.
    ``EvalConditioning.not_evaluating()`` for a training/resume load). See
    ``steering.conditional.eval_conditioning``.

    Returns (model, tokenizer, meta_data) — same interface as ``build_model``.
    """
    device = torch.device(device)  # coerce str → torch.device (mirrors build_model)
    assert phase in ModelPhase, f"Invalid phase: {phase}. Must be one of {list(ModelPhase)}"
    model_data, _, meta_data = load_checkpoint(
        checkpoint_dir,
        step,
        device,
        load_optimizer=False,
    )
    if device.type in {"cpu", "mps"}:
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()
        }
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    # Reconstruct decoder config.
    if "model_config" in meta_data:
        model_config_kwargs = meta_data["model_config"]
    else:
        # SFT checkpoints may not save model_config; reconstruct from
        # user_config or from the parent checkpoint.
        model_config_kwargs = _recover_model_config(checkpoint_dir, meta_data)
        meta_data["model_config"] = model_config_kwargs
    _patch_missing_config_keys(model_config_kwargs)
    dec_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, dec_config)
    # Fuse old-format split c_q/c_k/c_v -> c_qkv for both decoder and
    # encoder attention layers. Without this, strict=True load_state_dict
    # rejects legacy encdec checkpoints that were forked from pre-fusion
    # pretrain runs. (Parity with build_model, which fuses at the same
    # stage of the pipeline.)
    _fuse_qkv_weights(model_data)

    # Reconstruct encoder + cross-attention configs from user_config.
    # SFT checkpoints may not store encoder/cross_attention, and deeper chains
    # (e.g. capabilities_sft nested under safety_sft) need to walk past
    # intermediate SFT metas that also lack them. First try the meta's own
    # top-level encoder_config / xattn_config / num_buckets (the SFT form);
    # then walk ancestors until all three fields are filled.
    user_cfg = meta_data["user_config"]
    if "feature_pred_head.weight" in model_data:
        # Validate the original mirrors before recovery fills any missing fields.
        # Explicit legacy schemas and conflicting mirrors require conversion.
        feature_prediction_config_from_metadata(meta_data)
    own_fields = _extract_encdec_fields(meta_data)
    for key in ("encoder", "cross_attention"):
        if key not in user_cfg and key in own_fields:
            user_cfg[key] = own_fields[key]
    if user_cfg.get("num_buckets") is None and "num_buckets" in own_fields:
        user_cfg["num_buckets"] = own_fields["num_buckets"]
    if user_cfg.get("negative_selection") is None and "negative_selection" in own_fields:
        user_cfg["negative_selection"] = own_fields["negative_selection"]
    if user_cfg.get("feature_prediction") is None and "feature_prediction" in own_fields:
        user_cfg["feature_prediction"] = own_fields["feature_prediction"]

    if (
        "encoder" not in user_cfg
        or "cross_attention" not in user_cfg
        or user_cfg.get("num_buckets") is None
    ):
        # Recovery is best-effort for num_buckets / negative_selection (they have
        # their own downstream error paths). Only the missing-encoder case is
        # truly unrecoverable here, so re-raise FileNotFoundError only then.
        try:
            ancestor = _recover_encdec_ancestor_config(checkpoint_dir)
        except FileNotFoundError:
            if "encoder" not in user_cfg:
                raise
            ancestor = {}
        if "encoder" not in user_cfg:
            user_cfg["encoder"] = ancestor["encoder"]
        if "cross_attention" not in user_cfg and "cross_attention" in ancestor:
            user_cfg["cross_attention"] = ancestor["cross_attention"]
        if user_cfg.get("num_buckets") is None and "num_buckets" in ancestor:
            user_cfg["num_buckets"] = ancestor["num_buckets"]
        if user_cfg.get("negative_selection") is None and "negative_selection" in ancestor:
            user_cfg["negative_selection"] = ancestor["negative_selection"]

    if user_cfg.get("negative_selection") is None:
        for meta_path, meta in _iter_ancestor_metas(checkpoint_dir):
            ancestor_fields = _extract_encdec_fields(meta)
            if "negative_selection" in ancestor_fields:
                user_cfg["negative_selection"] = ancestor_fields["negative_selection"]
                log0(f"Recovered encdec fields from ancestor: {meta_path}")
                break

    enc_cfg = user_cfg["encoder"]
    enc_config = EncoderConfig(
        n_embd=enc_cfg["n_embd"],
        depth=enc_cfg["depth"],
        n_head=enc_cfg.get("n_head"),
        n_kv_head=enc_cfg.get("n_kv_head"),
        head_dim=enc_cfg.get("head_dim", 64),
    )
    xattn_cfg = user_cfg["cross_attention"]
    xattn_config = CrossAttentionConfig(
        frequency=xattn_cfg["frequency"],
        n_head=xattn_cfg.get("n_head"),
        n_kv_head=xattn_cfg.get("n_kv_head"),
    )

    # Recover n_sae_features + the compacted feature vocabulary from the state dict
    # ALONE — the checkpoint is self-describing (issue #695, requirement 1).
    #
    # This MUST NOT go back to reading feature_embed.weight.shape[0] directly: once the
    # encoder's dictionary is compacted that row count is V, not n_sae_features, and
    # every consumer that bounds-checks ORIGINAL feature ids against
    # model.encoder.config.n_sae_features (eval_adapter, target_conditioning,
    # model_detection, harmfulness_eval) would be silently corrupted.
    #
    # A legacy checkpoint carries no vocab buffers; for an identity vocab the old
    # shape-inference is still CORRECT, so legacy loads are unchanged (requirement 5).
    n_sae_features, feature_vocab = feature_vocab_from_state_dict(model_data)

    # meta mirrors the buffers with a fingerprint. The BUFFERS are the record: a compacted
    # checkpoint describes itself, and needs no sidecar to be interpreted (requirement 1).
    #
    # So ABSENCE of the mirror is fine; only DISAGREEMENT is a corruption signal. Raising
    # on absence would make the sidecar mandatory — i.e. requirement 1 would not be
    # delivered — and would contradict scripts/analysis/dump_feature_mask.py, the very tool
    # every #695 error message tells the operator to run, which documents that "ABSENCE is
    # fine. Its *disagreement* is not." (Metas routinely get thinned; that is why
    # _recover_model_config / _iter_ancestor_metas exist at all.)
    meta_vocab = meta_data.get(FEATURE_VOCAB_META_KEY)
    if meta_vocab is None:
        if feature_vocab is not None:
            logger.warning(
                f"{checkpoint_dir}: compacted checkpoint carries no {FEATURE_VOCAB_META_KEY!r} "
                "metadata mirror. Loading from the weight buffers (which are the record); "
                "tooling cannot compare this checkpoint's vocabulary without reading the "
                "weights."
            )
    else:
        expected = None if feature_vocab is None else feature_vocab.metadata_dict()
        if meta_vocab != expected:
            raise FeatureVocabError(
                f"feature vocab metadata in {checkpoint_dir} disagrees with the weights: "
                f"meta records {meta_vocab!r}, the state dict describes {expected!r}."
            )

    if user_cfg.get("num_buckets") is None:
        raise ValueError(
            "Cannot build EncoderDecoderGPT: 'num_buckets' is missing from "
            f"user_config and could not be recovered from a parent checkpoint "
            f"above {checkpoint_dir}. Re-save the checkpoint's meta with "
            "num_buckets, or load from a directory where a parent pretrain "
            "checkpoint carries it."
        )
    num_buckets = user_cfg["num_buckets"]
    log0(f"Building EncoderDecoderGPT: {n_sae_features} SAE features, {num_buckets} buckets")

    # Reconstruct FeaturePredictionConfig if the checkpoint contains the head weights.
    head_in_state = "feature_pred_head.weight" in model_data
    if user_cfg.get("feature_prediction") is None and head_in_state:
        # SFT metas may drop feature_prediction even when the inherited
        # state_dict still carries the pretrain head; walk ancestors.
        for meta_path, meta in _iter_ancestor_metas(checkpoint_dir):
            ancestor_fields = _extract_encdec_fields(meta)
            if "feature_prediction" in ancestor_fields:
                feature_prediction_config_from_metadata(meta)
                user_cfg["feature_prediction"] = ancestor_fields["feature_prediction"]
                log0(f"Recovered feature_prediction from ancestor: {meta_path}")
                break
    fp_cfg = user_cfg.get("feature_prediction")
    if head_in_state and fp_cfg is None:
        raise ValueError(
            "Cannot build EncoderDecoderGPT: checkpoint state_dict contains "
            "'feature_pred_head.weight' but feature_prediction config is missing "
            f"from this checkpoint and all ancestors above {checkpoint_dir}. "
            "Re-save metadata with user_config.feature_prediction or top-level "
            "feature_prediction_config."
        )
    if fp_cfg is not None and head_in_state:
        feature_pred_config = feature_prediction_config_from_metadata(meta_data)
        log0(
            f"Restoring feature_pred_head (layer={feature_pred_config.layer}, weight={feature_pred_config.weight})"
        )
    else:
        feature_pred_config = None

    # Build model on meta device, load weights.
    with torch.device("meta"):
        model = EncoderDecoderGPT(
            dec_config,
            enc_config,
            xattn_config,
            n_sae_features=n_sae_features,
            num_buckets=num_buckets,
            feature_pred_config=feature_pred_config,
            feature_vocab=feature_vocab,
        )
    model.to_empty(device=device)
    model.init_buffers()
    model.load_state_dict(model_data, strict=True, assign=True)
    # assign=True REPLACES the persistent feature_vocab_ids buffer with the checkpoint's.
    # Verify it still describes the configured vocabulary — otherwise every embedding row is
    # silently reinterpreted. (There is no separate derived map; _to_compact computes the
    # remap from feature_vocab_ids per-forward.)
    model.assert_vocab_consistent()

    if phase == ModelPhase.EVAL:
        # Load-time contract (#695 follow-up): an eval must declare how it conditions the encoder,
        # so a compacted checkpoint cannot be evaluated on the wrong feature universe. Reaching here
        # means an encoder-decoder EVAL load; the stance decides. Replaces #703's blanket guard.
        eval_conditioning.assert_loadable(model.feature_vocab, checkpoint_dir=checkpoint_dir)

    if phase == ModelPhase.EVAL:
        model.eval()
    else:
        model.train()

    tokenizer = get_tokenizer()
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], (
        f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match "
        f"model config vocab size {model_config_kwargs['vocab_size']}"
    )
    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [
        f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))
    ]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step


def _find_checkpoint_leaf(root_dir):
    """Find the deepest directory under root_dir that contains model_*.pt files.

    This handles nested checkpoint structures like:
        conditional_encdec_checkpoints/d26/s128/t2048.9/model_000000.pt

    where model_tag resolves to d26 but the actual checkpoint files are
    several levels deeper. Searches depth-first and returns the first
    directory found containing model_*.pt files.

    Raises:
        DataLoadError: If no checkpoint files are found under root_dir.
    """
    # Check if root_dir itself has checkpoint files
    if glob.glob(os.path.join(root_dir, "model_*.pt")):
        return root_dir
    # Otherwise, recurse into subdirectories (sorted for determinism)
    subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    for subdir in subdirs:
        try:
            return _find_checkpoint_leaf(os.path.join(root_dir, subdir))
        except DataLoadError:
            continue
    raise DataLoadError(f"No checkpoint files found under {root_dir}")


# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure


def meta_looks_encdec(meta: dict) -> bool:
    """Return True if a parsed meta dict indicates an ``EncoderDecoderGPT``.

    Encdec checkpoints come in two meta shapes:
      - Pretrain: ``user_config.encoder`` + ``user_config.cross_attention``
        are both present. Written by ``scripts.train.conditional_encdec_train``
        and ``scripts.preprocess.convert_decoder_to_encdec``.
      - SFT: top-level ``encoder_config`` + ``xattn_config`` are both
        present (user_config inherited from the pretrain run; SFT adds
        top-level fields so chained SFT runs can reconstruct without
        walking to the pretrain checkpoint). Written by
        ``scripts.train.conditional_encdec_sft``.

    ``_extract_encdec_fields`` normalizes both shapes. We require BOTH
    ``encoder`` and ``cross_attention`` to be present and truthy — a meta
    with only one is a half-written artifact we shouldn't guess from.
    A decoder-only meta has neither and returns False.

    The heuristic is intentionally strict (AND, not OR): a false positive
    would mis-dispatch a decoder-only checkpoint to ``build_encdec_model``
    and fail obscurely; a false negative falls back to ``build_model``
    whose strict state-dict load emits a clear error if the weights
    actually are encdec.
    """
    fields = _extract_encdec_fields(meta)
    return bool(fields.get("encoder") and fields.get("cross_attention"))


def _is_encdec_checkpoint(checkpoint_dir: str, step: int) -> bool:
    """Return True if a checkpoint on disk has an encdec-shaped meta.

    Thin wrapper around :func:`meta_looks_encdec`: peeks at
    ``meta_{step:06d}.json`` without loading tensor weights. Exists so
    ``load_model`` / ``load_model_from_dir`` can correctly dispatch to
    ``build_encdec_model`` for encdec checkpoints stored under
    ``base_checkpoints/`` (e.g. conditioned-SFT chains whose path lives
    under a decoder-shaped tag but whose weights are encoder-decoder).
    Without the peek, the strict state-dict load in ``build_model`` would
    crash on the extra encoder/cross_attention keys.

    Meta-read errors (missing file, corrupt JSON, invalid UTF-8,
    permission denied, is-a-directory) return False so the caller falls
    back to the decoder-shaped default path. The downstream strict
    ``load_state_dict`` will then be the one to emit a clear error if
    the weights actually don't match. Opens with explicit
    ``encoding="utf-8"`` for determinism across locales, matching the
    rest of this module's JSON reads (see ``load_checkpoint`` at l.291).
    """
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    return meta_looks_encdec(meta)


def load_model_from_dir(
    checkpoints_dir: str,
    device: str | torch.device,
    phase: ModelPhase | str,
    model_tag: str | None = None,
    step: int | None = None,
    *,
    eval_conditioning: EvalConditioning,
) -> tuple[LoadedModel, LoadedTokenizer, dict]:
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # Auto-detect encdec checkpoints whose weights are EncoderDecoderGPT
    # but whose storage path lives under a decoder-shaped tag root. Without
    # this, the strict load in build_model would crash on the extra
    # encoder/cross_attention keys.
    if _is_encdec_checkpoint(checkpoint_dir, step):
        log0(f"Auto-detected EncoderDecoderGPT at {checkpoint_dir} step {step}")
        model, tokenizer, meta_data = build_encdec_model(
            checkpoint_dir, step, device, phase, eval_conditioning=eval_conditioning
        )
    else:
        log0(f"Loading model from {checkpoint_dir} with step {step}")
        model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    meta_data["checkpoint_dir"] = checkpoint_dir
    return model, tokenizer, meta_data


def load_encdec_model_from_dir(
    checkpoints_dir: str,
    device: str | torch.device,
    phase: ModelPhase | str,
    model_tag: str | None = None,
    step: int | None = None,
    *,
    eval_conditioning: EvalConditioning,
) -> tuple[EncoderDecoderGPT, LoadedTokenizer, dict]:
    """Same as ``load_model_from_dir`` but builds an EncoderDecoderGPT.

    Handles nested checkpoint structures (e.g. ``<tag>/s128/t2048.9/``)
    by traversing subdirectories to find the actual checkpoint files.
    """
    if model_tag is None:
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    tag_dir = os.path.join(checkpoints_dir, model_tag)
    # Traverse nested subdirs (s{segment_size}/t{threshold}/) to find checkpoint files
    checkpoint_dir = _find_checkpoint_leaf(tag_dir)
    if step is None:
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    log0(f"Loading EncoderDecoderGPT from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_encdec_model(
        checkpoint_dir, step, device, phase, eval_conditioning=eval_conditioning
    )
    meta_data["checkpoint_dir"] = checkpoint_dir
    return model, tokenizer, meta_data


def _is_encdec_root(source: str) -> bool:
    """Return True if the source chain starts from an encdec root."""
    return source.split("/")[0] == ModelSource.ENCDEC


# Maps segment names to their checkpoint subdirectory names.
_SEGMENT_DIRS = {
    "sft": "standard_sft_checkpoints",
    "standard": "standard_sft_checkpoints",
    "safety": "safety_sft_checkpoints",
    "capabilities": "capabilities_sft_checkpoints",
    "rl": "standard_rl_checkpoints",
    "safety_cond": "safety_sft_conditioned_checkpoints",
    "capabilities_cond": "capabilities_sft_conditioned_checkpoints",
    "standard_cond": "standard_sft_conditioned_checkpoints",
}

# Selection-leaf directory names emitted, in nesting order, by
# FeatureMask.dirname(), append_gradient_leaf(), and
# append_feature_clusters_leaf(). Keep this pattern aligned with those writers.
_SELECTION_LEAF_DIR_RE = re.compile(
    r"^(?:(?:bl|wl)_[0-9a-f]{16}"
    r"|grad_(?:self|future_half_split|future_self_full)_mf"
    r"(?:[2-9]|[1-9][0-9]+)(?:_fp32)?"
    r"|fc_[0-9a-f]{64})$"
)
_MAX_SELECTION_LEAF_DEPTH = 3

# Segments that are roots (not appended as subdirectories of base_checkpoints/).
_ROOT_SEGMENTS = {"base", "encdec"}

# All valid segment names (for validation).
_VALID_SEGMENTS = set(_SEGMENT_DIRS) | _ROOT_SEGMENTS


def _find_segment_through_selection_leaves(path, segment_dir):
    """Return matching ``segment_dir`` paths below up to three selection leaves."""
    candidates = []
    parents = [path]
    for _depth in range(_MAX_SELECTION_LEAF_DEPTH):
        next_parents = []
        for parent in parents:
            try:
                with os.scandir(parent) as scan:
                    entries = sorted(scan, key=lambda entry: entry.name)
            except (FileNotFoundError, NotADirectoryError):
                continue
            for entry in entries:
                if not entry.is_dir() or _SELECTION_LEAF_DIR_RE.fullmatch(entry.name) is None:
                    continue
                leaf_path = entry.path
                candidate = os.path.join(leaf_path, segment_dir)
                if os.path.isdir(candidate):
                    candidates.append(candidate)
                next_parents.append(leaf_path)
        parents = next_parents
    return candidates


def get_checkpoint_dir(
    source, model_tag=None, step=None, segment_size=None, feature_selection=None
):
    """Resolve a slash-separated source chain to a full checkpoint directory path.

    The first segment determines the root directory:
    - "base" or "encdec" are explicit roots.
    - SFT mixture names ("sft"/"standard", "safety", "capabilities") and "rl"
      implicitly start from the "base" root.

    Subsequent segments append subdirectories. Examples:
        "base"            -> base_checkpoints/<tag>/
        "sft"             -> base_checkpoints/<tag>/standard_sft_checkpoints/
        "safety/rl"       -> base_checkpoints/<tag>/safety_sft_checkpoints/standard_rl_checkpoints/
        "encdec"          -> conditional_encdec_checkpoints/<tag>/s<seg>/...
        "encdec/safety"   -> conditional_encdec_checkpoints/<tag>/s<seg>/.../safety_sft_checkpoints/

    Bare "rl" is shorthand for "sft/rl" (backward compatibility).
    """
    segments = source.split("/")
    first = segments[0]

    if first not in _VALID_SEGMENTS:
        raise ValueError(
            f"Unknown source: {source!r}. First segment must be one of {sorted(_VALID_SEGMENTS)}"
        )

    # Determine root directory.
    base_dir = get_base_dir()
    if first == ModelSource.ENCDEC:
        root = "conditional_encdec_checkpoints"
    else:
        root = "base_checkpoints"

    checkpoints_dir = os.path.join(base_dir, root)
    if model_tag is None:
        model_tag = find_largest_model(checkpoints_dir)
    path = os.path.join(checkpoints_dir, model_tag)

    # For encdec root, append segment_size/feature_selection params.
    if first == ModelSource.ENCDEC:
        if segment_size is not None:
            path = os.path.join(path, f"s{segment_size}")
        if feature_selection is not None:
            path = os.path.join(path, feature_selection_dirname(feature_selection))
        # Auto-discover nested structure when not explicitly specified.
        if segment_size is None and feature_selection is None and os.path.isdir(path):
            try:
                path = _find_checkpoint_leaf(path)
            except DataLoadError:
                pass  # keep path as-is; downstream will report missing checkpoints

    # Build the rest of the path from segments.
    if first in _ROOT_SEGMENTS:
        # "base" and "encdec" are pure roots; start appending from segments[1:].
        tail = segments[1:]
    elif first == ModelSource.RL:
        # Bare "rl" is backward-compat shorthand for "sft/rl".
        tail = ["sft", "rl"] + segments[1:]
    else:
        # SFT mixture name as first segment implies base root.
        tail = segments

    for seg in tail:
        if seg not in _SEGMENT_DIRS:
            raise ValueError(
                f"Invalid segment {seg!r} in source {source!r}. "
                f"Valid segments: {sorted(_SEGMENT_DIRS)}"
            )
        segment_path = os.path.join(path, _SEGMENT_DIRS[seg])
        if first == ModelSource.ENCDEC and not os.path.exists(segment_path):
            candidates = _find_segment_through_selection_leaves(path, _SEGMENT_DIRS[seg])
            if len(candidates) == 1:
                segment_path = candidates[0]
            elif len(candidates) > 1:
                raise ValueError(
                    f"Ambiguous checkpoint source {source!r}: segment {seg!r} exists "
                    f"through multiple selection-leaf paths: {candidates}. Pass the "
                    "intended path via load_model(..., checkpoints_dir=...) to "
                    "disambiguate."
                )
        path = segment_path

    return path


def get_report_dir(source, model_tag=None, **kwargs):
    """Return the report directory for a given model source.

    Equivalent to ``os.path.join(get_checkpoint_dir(source, model_tag, **kwargs), "report")``.
    Accepts the same keyword arguments as ``get_checkpoint_dir`` (e.g.
    ``segment_size``, ``feature_selection``).
    """
    return os.path.join(get_checkpoint_dir(source, model_tag=model_tag, **kwargs), "report")


def load_model(
    source: ModelSource | str,
    device: str | torch.device,
    phase: ModelPhase | str,
    model_tag: str | None = None,
    step: int | None = None,
    checkpoints_dir: str | None = None,
    *,
    eval_conditioning: EvalConditioning,
    segment_size=None,
    feature_selection=None,
) -> tuple[LoadedModel, LoadedTokenizer, dict]:
    """Load a model by source key or explicit checkpoints_dir.

    If checkpoints_dir is provided, it is used directly (no source resolution).
    Otherwise, the source string is resolved to the appropriate nested path.
    """
    is_encdec = _is_encdec_root(source)

    if checkpoints_dir is not None:
        # Direct path mode: load from the given directory.
        if is_encdec:
            return load_encdec_model_from_dir(
                checkpoints_dir,
                device,
                phase,
                model_tag=model_tag,
                step=step,
                eval_conditioning=eval_conditioning,
            )
        return load_model_from_dir(
            checkpoints_dir,
            device,
            phase,
            model_tag=model_tag,
            step=step,
            eval_conditioning=eval_conditioning,
        )

    # For "base" source, load_model_from_dir handles tag/step discovery.
    if source == ModelSource.BASE:
        base_dir = get_base_dir()
        return load_model_from_dir(
            os.path.join(base_dir, "base_checkpoints"),
            device,
            phase,
            model_tag=model_tag,
            step=step,
            eval_conditioning=eval_conditioning,
        )

    # For encdec without further chain segments, use load_encdec_model_from_dir
    # which handles nested traversal (s<seg>/t<thr>/) via _find_checkpoint_leaf.
    if source == ModelSource.ENCDEC:
        base_dir = get_base_dir()
        return load_encdec_model_from_dir(
            os.path.join(base_dir, "conditional_encdec_checkpoints"),
            device,
            phase,
            model_tag=model_tag,
            step=step,
            eval_conditioning=eval_conditioning,
        )

    # All other sources (chains, SFT mixtures, RL, encdec/X) resolve to a
    # concrete directory via get_checkpoint_dir, then load directly.
    checkpoint_dir = get_checkpoint_dir(
        source,
        model_tag=model_tag,
        segment_size=segment_size,
        feature_selection=feature_selection,
    )
    if step is None:
        step = find_last_step(checkpoint_dir)
    # Auto-detect encdec checkpoints stored under base_checkpoints/ whose
    # weights are EncoderDecoderGPT (e.g. safety_cond / capabilities_cond
    # chains). `is_encdec` above is True only when the source string starts
    # with `encdec/`, so encdec weights under a base_ root would otherwise
    # fall into `build_model` and crash at strict state-dict load.
    if not is_encdec:
        is_encdec = _is_encdec_checkpoint(checkpoint_dir, step)
    if is_encdec:
        model, tokenizer, meta_data = build_encdec_model(
            checkpoint_dir, step, device, phase, eval_conditioning=eval_conditioning
        )
    else:
        model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    meta_data["checkpoint_dir"] = checkpoint_dir
    return model, tokenizer, meta_data
