"""
Train model. From root directory of the project, run as:

python -m scripts.train.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.train.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.train.base_train depth=4 max_seq_len=512 device_batch_size=1 eval_tokens=512 core_metric_every=-1 total_batch_size=512 num_iterations=20

With torchrun, place Hydra overrides after '--':
torchrun --nproc_per_node=8 -m scripts.train.base_train -- depth=12
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch
import wandb
from hydra import compose, initialize_config_dir
from loguru import logger

import nanochat.flash_attention as fa_module
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.common import (
    DeviceType,
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    configure_wandb_step_metric,
    get_base_dir,
    get_peak_flops,
    print_banner,
)
from nanochat.dataloader import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from nanochat.dataset import DatasetSplit
from nanochat.engine import Engine
from nanochat.flash_attention import FlashAttnImpl
from nanochat.fp8_utils import apply_fp8_training, disable_fp8
from nanochat.gpt import GPT
from nanochat.loss_eval import evaluate_bpb
from nanochat.report import get_report
from nanochat.tokenizer import get_token_bytes, get_tokenizer
from scripts.eval.base_eval import evaluate_core
from steering._common.build_meta import build_gpt_meta
from steering._common.lr_schedules import pretrain_warmup_flat_warmdown
from steering.config_payloads import resolved_config_dict

print_banner()

# -----------------------------------------------------------------------------
# Load Hydra config (compose API for torchrun compatibility)
_config_dir = str(Path(__file__).resolve().parent.parent.parent / "configs" / "training")
_config_name = "base_train"
for _arg in sys.argv[1:]:
    if _arg.startswith("config_name="):
        _config_name = _arg.split("=", 1)[1]
with initialize_config_dir(config_dir=_config_dir, version_base=None):
    _overrides = [
        a
        for a in sys.argv[1:]
        if "=" in a and not a.startswith("-") and not a.startswith("config_name=")
    ]
    cfg = compose(config_name=_config_name, overrides=_overrides)
user_config = resolved_config_dict(cfg)  # for logging
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if cfg.device_type == "" else DeviceType(cfg.device_type)
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
autocast_ctx = (
    torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    if device_type == "cuda"
    else nullcontext()
)
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    logger.info(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float("inf")  # MFU not meaningful for CPU/MPS

# wandb logging init
use_dummy_wandb = cfg.run == "dummy" or not master_process
wandb_run = (
    DummyWandb()
    if use_dummy_wandb
    else wandb.init(project="nanochat", name=cfg.run, config=user_config)
)
configure_wandb_step_metric(wandb_run)

# Flash Attention override
if cfg.attn_impl != FlashAttnImpl.AUTO:
    fa_module._override_impl = FlashAttnImpl(cfg.attn_impl)
    logger.info(f"Attention implementation override: {cfg.attn_impl}")

# Flash Attention status
active_impl = fa_module._active_impl()
if active_impl == FlashAttnImpl.FA4:
    backend = "FLASH (FA4)" if fa_module.HAS_FA4_BACKEND else "Triton"
    logger.info(f"Using FlexAttention ({backend} backend)")
elif active_impl == FlashAttnImpl.FA3:
    logger.info("Using Flash Attention 3 (Hopper GPU detected)")
else:
    logger.warning("Flash Attention not available, using PyTorch SDPA fallback")
    logger.warning("Training will be less efficient without flash attention")
    if cfg.window_pattern != "L":
        logger.warning(
            f"SDPA has limited support for sliding window attention (window_pattern='{cfg.window_pattern}')."
        )

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
logger.info(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

# Build the model, move to device, init the weights
model = build_gpt_meta(
    depth=cfg.depth,
    aspect_ratio=cfg.aspect_ratio,
    head_dim=cfg.head_dim,
    max_seq_len=cfg.max_seq_len,
    window_pattern=cfg.window_pattern,
    vocab_size=vocab_size,
)  # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = asdict(model_config)
logger.info(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(
    device=device
)  # 2) All tensors get storage on target device but with uninitialized (garbage) data

# If we are resuming, overwrite the model parameters with those of the checkpoint.
# Skip the full ``init_weights`` (which would normal_/uniform_-init every parameter
# only to immediately overwrite them via ``load_state_dict(assign=True)``) — only
# the non-persistent rotary buffers (``cos``, ``sin``) need to be (re)computed,
# matching the pattern used by ``nanochat/checkpoint_manager.build_model`` post-load
# (slice 11 audit P1 #8 / consolidation manifest section 1.10).
base_dir = get_base_dir()
output_dirname = cfg.model_tag if cfg.model_tag else f"d{cfg.depth}"  # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = cfg.resume_from_step != -1
if resuming:
    logger.info(f"Resuming optimization from step {cfg.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(
        checkpoint_dir, cfg.resume_from_step, device, load_optimizer=True, rank=ddp_rank
    )
    model.load_state_dict(model_data, strict=True, assign=True)
    model.init_buffers()  # rotary cos/sin only — params come from the checkpoint
    del model_data  # free up this memory after the copy
else:
    model.init_weights()  # 3) All tensors get initialized for a fresh run

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if cfg.fp8:
    if device_type != "cuda":
        logger.warning("FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        apply_fp8_training(model, recipe=cfg.fp8_recipe)

# -----------------------------------------------------------------------------
# Compile the model

orig_model = model  # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
# ``torch.compile`` is annotated as returning an opaque callable, but the
# compiled module preserves the wrapped GPT's attribute surface — keep the
# concrete type so downstream attribute access (``num_scaling_params``,
# ``estimate_flops``, ``train``/``eval``, ``zero_grad``...) typechecks.
model = cast(
    GPT,
    torch.compile(
        model, dynamic=False
    ),  # the inputs to model will never change shape so dynamic=False is safe
)

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

# Get the parameter counts of our model
param_counts = model.num_scaling_params()
logger.info("Parameter counts:")
for key, value in param_counts.items():
    logger.info(f"{key:24s}: {value:,}")
num_params = param_counts["total"]
num_flops_per_token = model.estimate_flops()
logger.info(f"Estimated FLOPs per token: {num_flops_per_token:e}")


# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
def get_scaling_params(m):
    # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    params_counts = m.num_scaling_params()
    scaling_params = params_counts["transformer_matrices"] + params_counts["lm_head"]
    return scaling_params


num_scaling_params = get_scaling_params(model)
target_tokens = int(
    cfg.target_param_data_ratio * num_scaling_params
)  # optimal tokens for the model we are about to train

# Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
d12_ref = build_gpt_meta(
    depth=12,
    aspect_ratio=cfg.aspect_ratio,
    head_dim=cfg.head_dim,
    max_seq_len=cfg.max_seq_len,
    window_pattern=cfg.window_pattern,
    vocab_size=vocab_size,
)  # creates the model on meta device
D_REF = cfg.target_param_data_ratio * get_scaling_params(
    d12_ref
)  # compute-optimal d12 training horizon in tokens (measured empirically)
B_REF = 2**19  # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) Now that we have the token horizon, we can calculate the optimal batch size
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
total_batch_size = cfg.total_batch_size  # user-provided override is possible
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio**0.383
    total_batch_size = 2 ** round(
        math.log2(predicted_batch_size)
    )  # clamp to nearest power of 2 for efficiency
    logger.info(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF  # B/B_ref
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    batch_lr_scale = batch_ratio**0.5  # η ∝ √(B/B_ref)
    logger.info(
        f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})"
    )

# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
weight_decay_scaled = (
    cfg.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
)
if weight_decay_scaled != cfg.weight_decay:
    logger.info(
        f"Scaling weight decay from {cfg.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {cfg.depth}"
    )

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
optimizer = model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=cfg.unembedding_lr * batch_lr_scale,
    embedding_lr=cfg.embedding_lr * batch_lr_scale,
    scalar_lr=cfg.scalar_lr * batch_lr_scale,
    adam_betas=(cfg.adam_beta1, cfg.adam_beta2),
    # Muon hyperparameters
    matrix_lr=cfg.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer,
    cfg.device_batch_size,
    cfg.max_seq_len,
    split=DatasetSplit.TRAIN,
    device=device,
    resume_state_dict=dataloader_resume_state_dict,
)


def build_val_loader():
    return tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer,
        cfg.device_batch_size,
        cfg.max_seq_len,
        split=DatasetSplit.VAL,
        device=device,
    )


x, y, dataloader_state_dict = next(train_loader)  # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert cfg.num_iterations > 0 or cfg.target_param_data_ratio > 0 or cfg.target_flops > 0
if cfg.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = cfg.num_iterations
    logger.info(f"Using user-provided number of iterations: {num_iterations:,}")
elif cfg.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(cfg.target_flops / (num_flops_per_token * total_batch_size))
    logger.info(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif cfg.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    logger.info(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations  # the actual number of tokens we will train for
logger.info(f"Total number of training tokens: {total_tokens:,}")
logger.info(
    f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}"
)  # e.g. Chinchilla was ~20
logger.info(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")


# Momentum scheduler for Muon optimizer (warms up to 0.95 over the first 300 steps)
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum


# Weight decay scheduler for Muon optimizer (linearly decays to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / num_iterations)


# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_bpb = None  # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0  # EMA of training loss
    total_training_time = 0  # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
tokens_per_fwdbwd = (
    cfg.device_batch_size * cfg.max_seq_len
)  # tokens per iteration for a single rank
world_tokens_per_fwdbwd = (
    tokens_per_fwdbwd * ddp_world_size
)  # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
logger.info(
    f"Tokens / micro-batch / rank: {cfg.device_batch_size} x {cfg.max_seq_len} = {tokens_per_fwdbwd:,}"
)
logger.info(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
logger.info(
    f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}"
)

# Go!
while True:
    last_step = (
        step == num_iterations
    )  # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if cfg.eval_every > 0 and (last_step or step % cfg.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = cfg.eval_tokens // (cfg.device_batch_size * cfg.max_seq_len * ddp_world_size)
        with disable_fp8(model), autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        logger.info(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log(
            {
                "step": step,
                "total_training_flops": flops_so_far,
                "total_training_time": total_training_time,
                "val/bpb": val_bpb,
            }
        )
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if cfg.core_metric_every > 0 and (
        last_step or (step > 0 and step % cfg.core_metric_every == 0)
    ):
        model.eval()
        with disable_fp8(orig_model), autocast_ctx:
            results = evaluate_core(
                orig_model, tokenizer, device, max_per_task=cfg.core_metric_max_per_task
            )
        logger.info(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log(
            {
                "step": step,
                "total_training_flops": flops_so_far,
                "core_metric": results["core_metric"],
                "centered_results": results["centered_results"],
            }
        )
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if (
        cfg.sample_every > 0
        and master_process
        and (last_step or (step > 0 and step % cfg.sample_every == 0))
    ):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer)  # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<bos>")
            with disable_fp8(orig_model), autocast_ctx:
                sample, _ = engine.generate_batch(
                    tokens, num_samples=1, max_tokens=16, temperature=0
                )
            logger.info(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (
        step > 0
        and step != cfg.resume_from_step
        and cfg.save_every > 0
        and step % cfg.save_every == 0
    ):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),  # model parameters
            optimizer.state_dict(),  # optimizer state
            {  # metadata saved as json
                "step": step,
                "val_bpb": val_bpb,  # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config,  # inputs to the training script
                "device_batch_size": cfg.device_batch_size,
                "max_seq_len": cfg.max_seq_len,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": {  # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    for _ in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()  # for logging
        loss = loss / grad_accum_steps  # each .backward() is a grad sum => normalize loss here
        loss.backward()
        x, y, dataloader_state_dict = next(
            train_loader
        )  # prefetch the next batch while the GPU is busy with forward/backward
    # step the optimizer
    lrm = pretrain_warmup_flat_warmdown(
        step,
        num_iterations=num_iterations,
        warmup_ratio=cfg.warmup_ratio,
        warmdown_ratio=cfg.warmdown_ratio,
        final_lr_frac=cfg.final_lr_frac,
    )
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group["kind"] == "muon":
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item()  # .item() is a CPU-GPU sync point
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    ema_beta = 0.9  # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = (
        ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    )  # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))  # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    # MFU relative to BF16 peak. With FP8, MFU can exceed 100% (expected behavior).
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt  # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds / 60:.1f}m"
    else:
        eta_str = ""
    epoch = dataloader_state_dict["epoch"]
    logger.info(
        f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time / 60:.2f}m{eta_str}"
    )
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        wandb_run.log(log_data)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == cfg.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect()  # manually collect a lot of garbage from setup
        gc.freeze()  # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable()  # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0:  # every 5000 steps...
        gc.collect()  # manually collect, just to be safe for very, very long runs

# print a few more stats
logger.info(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
logger.info(f"Total training time: {total_training_time / 60:.2f}m")
if val_bpb is not None:
    logger.info(f"Minimum validation bpb: {min_val_bpb:.6f}")

# Log to report
get_report(checkpoint_dir=checkpoint_dir).log(
    section="Base model training",
    data=[
        user_config,  # CLI args
        {  # stats about the training setup
            "Number of parameters": num_params,
            "Number of FLOPs per token": f"{num_flops_per_token:e}",
            "Calculated number of iterations": num_iterations,
            "Number of training tokens": total_tokens,
            "Tokens : Scaling params ratio": total_batch_size * num_iterations / num_scaling_params,
            "DDP world size": ddp_world_size,
            "warmup_ratio": cfg.warmup_ratio,
            "warmdown_ratio": cfg.warmdown_ratio,
            "final_lr_frac": cfg.final_lr_frac,
        },
        {  # stats about training outcomes
            "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
            "Final validation bpb": val_bpb,
            "CORE metric estimate": results.get("core_metric", None),
            "MFU %": f"{mfu:.2f}%",
            "Total training flops": f"{flops_so_far:e}",
            "Total training time": f"{total_training_time / 60:.2f}m",
            "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
        },
    ],
)

# cleanup
wandb_run.finish()  # wandb run finish
compute_cleanup()
