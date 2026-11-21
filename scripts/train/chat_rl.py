"""
Reinforcement learning on GSM8K via "GRPO".

I put GRPO in quotes because we actually end up with something a lot
simpler and more similar to just REINFORCE:

1) Delete trust region, so there is no KL regularization to a reference model
2) We are on policy, so there's no need for PPO ratio+clip.
3) We use DAPO style normalization that is token-level, not sequence-level.
4) Instead of z-score normalization (r - mu)/sigma, only use (r - mu) as the advantage.

1 GPU:
python -m scripts.train.chat_rl

8 GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.train.chat_rl -- run=default
"""

import itertools
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from hydra import compose, initialize_config_dir
from loguru import logger

from nanochat.checkpoint_manager import (
    ModelPhase,
    ModelSource,
    load_model,
    save_checkpoint,
)
from nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_peak_flops,
)
from nanochat.engine import Engine
from nanochat.gpt import GPT
from nanochat.report import get_report
from steering._common.lr_schedules import rl_linear_rampdown
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_clusters import reject_feature_clusters_metadata
from steering.config_payloads import resolved_config_dict
from steering.tokenizer_protocols import require_completion_tokenizer
from tasks.gsm8k import GSM8K


# Deterministic per-rollout seed (slice 11 audit P1 #6).
#
# Python's built-in ``hash()`` is non-deterministic across processes by
# default (PYTHONHASHSEED randomization). Two RL runs with identical cfg
# would otherwise produce different rollouts -> different rewards ->
# different gradients -> unreproducible training. Replace with an
# explicit linear combination of the three index variables, masked to
# the positive int32 range that ``Engine.generate_batch`` accepts.
def _compute_rollout_seed(step: int, example_idx: int, sampling_step: int) -> int:
    """Deterministic non-cryptographic mix of three integer indices.

    Returns a non-negative ``int`` in ``[0, 2**31)``. Used by
    ``get_batch()`` to seed ``Engine.generate_batch`` for each
    (step, example, sampling_step) triple. Determinism guarantee: the
    output depends only on the inputs (no env-dependent state, no
    PYTHONHASHSEED randomization).

    Constants: ``1_000_003`` and ``1_009`` are primes large enough to
    avoid collisions across the index ranges used in practice
    (steps in tens of thousands, examples in tens of thousands,
    sampling_steps in single digits).
    """
    return (step * 1_000_003 + example_idx * 1_009 + sampling_step) & 0x7FFFFFFF


# -----------------------------------------------------------------------------
# Load Hydra config (compose API for torchrun compatibility)
_config_dir = str(Path(__file__).resolve().parent.parent.parent / "configs" / "training")
with initialize_config_dir(config_dir=_config_dir, version_base=None):
    _overrides = [a for a in sys.argv[1:] if "=" in a and not a.startswith("-")]
    cfg = compose(config_name="chat_rl", overrides=_overrides)
user_config = resolved_config_dict(cfg)
# -----------------------------------------------------------------------------

# Init compute/precision
device_type = autodetect_device_type() if cfg.device_type == "" else cfg.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
ptdtype = torch.float32 if cfg.dtype == "float32" else torch.bfloat16
autocast_ctx = (
    torch.amp.autocast(device_type=device_type, dtype=ptdtype)
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
    gpu_peak_flops = float("inf")

# Init model and tokenizer
model, tokenizer, meta = load_model(
    ModelSource.SFT,
    device,
    phase=ModelPhase.EVAL,
    model_tag=cfg.model_tag,
    step=cfg.model_step,
    eval_conditioning=EvalConditioning.deferred("chat_rl"),
)
reject_feature_clusters_metadata(meta, "chat_rl source checkpoint")
if not isinstance(model, GPT):
    raise TypeError(
        f"chat_rl requires a decoder-only GPT source checkpoint; got {type(model).__name__}"
    )
completion_tokenizer = require_completion_tokenizer(tokenizer, surface="chat_rl")

# Create external logging only after the source checkpoint's complete metadata
# contract has been accepted.
use_dummy_wandb = cfg.run == "dummy" or not master_process
wandb_run = (
    DummyWandb()
    if use_dummy_wandb
    else wandb.init(project="nanochat-rl", name=cfg.run, config=user_config)
)

engine = Engine(model, tokenizer)  # for sampling rollouts
num_flops_per_token = model.estimate_flops()

# -----------------------------------------------------------------------------
# Rollout / sampling generator loop that yields batches of examples for training

train_task = GSM8K(subset="main", split="train")
val_task = GSM8K(subset="main", split="test")
num_steps = (len(train_task) // cfg.examples_per_step) * cfg.num_epochs
logger.info(f"Calculated number of steps: {num_steps}")


@torch.no_grad()
def get_batch():
    assistant_end = tokenizer.encode_special(
        "<assistant_end>"
    )  # ok to use this token, it's only for padding and isn't used in the loss.
    rank_indices = range(
        ddp_rank, len(train_task), ddp_world_size
    )  # each rank is responsible for different examples in the training data
    for example_idx in itertools.cycle(rank_indices):
        # First get the full conversation of both user and assistant messages
        conversation = train_task[example_idx]

        # Tokenize the conversation, deleting the last Assistant message and priming the Assistant for a completion instead
        # (i.e. keep the <assistant_start>, but delete everything after it)
        tokens = completion_tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # Generate num_samples samples using batched generation, use loop to avoid OOMs
        model.eval()  # ensure the model is in eval mode
        generated_token_sequences = []
        masks = []
        num_sampling_steps = (
            cfg.num_samples // cfg.device_batch_size
        )  # go sequentially to prevent OOMs
        for sampling_step in range(num_sampling_steps):
            # Deterministic seed (slice 11 audit P1 #6); see helper
            # docstring for rationale and determinism guarantee.
            seed = _compute_rollout_seed(step, example_idx, sampling_step)
            with autocast_ctx:
                generated_token_sequences_batch, masks_batch = engine.generate_batch(
                    tokens,
                    num_samples=cfg.device_batch_size,
                    max_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                    seed=seed,  # must make sure to change the seed for each sampling step
                )
            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)

        # Calculate the rewards for each sample
        rewards = []
        for sample_tokens in generated_token_sequences:
            # Get just the generated tokens (after the prompt)
            generated_tokens = sample_tokens[prefix_length:]
            # Decode the generated response
            generated_text = tokenizer.decode(generated_tokens)
            # Calculate the reward
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)

        # Pad the sequences so that their lengths (in time) match
        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_generated_token_sequences = [
            seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences
        ]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        # Stack up the sequences and masks into PyTorch tensors
        ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        # Generate autoregressive inputs and targets to the Transformer
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()  # clone to avoid in-place modification:
        targets[
            mask_ids[:, 1:] == 0
        ] = -1  # <-- inplace modification right here. -1 is the ignore index
        # NOTE also that the Engine returns mask=0 for BOTH the prompt tokens AND the tool use tokens.
        # So we will (correctly) end up not training on the prompt tokens, or the tool use forced tokens.
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        # Calculate the advantages by simply subtracting the mean (instead of z-score (x-mu)/sigma)
        mu = rewards.mean()
        advantages = rewards - mu
        # yield inputs/targets as (B, T) of ids and rewards as (B,) of floats
        yield generated_token_sequences, inputs, targets, rewards, advantages


# -----------------------------------------------------------------------------
# Simple evaluation loop for GSM8K pass@k
def run_gsm8k_eval(
    task,
    tokenizer,
    engine,
    max_examples=None,
    num_samples=1,
    max_completion_tokens=256,
    temperature=0.0,
    top_k=50,
):
    """
    Evaluates GSM8K task and returns a list of records of evaluation outcomes.
    In a distributed setting, all ranks cooperate but this function will NOT
    do the reduction across ranks. This is the responsibility of the caller.
    Because the evaluation can take a while, this function will yield records one by one.
    """
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        # Generate k samples using batched generation inside the Engine
        assert (
            num_samples <= cfg.device_batch_size
        )  # usually this is true. we can add a loop if not...
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        # Check each sample for correctness
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({"is_correct": is_correct})
        # A bit bloated because I wanted to do more complex logging at one point.
        record = {
            "idx": idx,
            "outcomes": outcomes,
        }
        yield record


# -----------------------------------------------------------------------------
# Training loop

# Init the optimizer
optimizer = model.setup_optimizer(
    unembedding_lr=cfg.unembedding_lr,
    embedding_lr=cfg.embedding_lr,
    matrix_lr=cfg.matrix_lr,
    weight_decay=cfg.weight_decay,
)

# Set the initial learning rate as a fraction of the base learning rate
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * cfg.init_lr_frac
    group["initial_lr"] = group["lr"]


# Calculate the number of examples each rank handles to achieve the desired examples_per_step
logger.info(
    f"Total sequences per step: {cfg.examples_per_step * cfg.num_samples}"
)  # total batch size in sequences/step
assert cfg.examples_per_step % ddp_world_size == 0, (
    "Desired examples per step must be divisible by the number of ranks"
)
examples_per_rank = cfg.examples_per_step // ddp_world_size  # per GPU
logger.info(f"Calculated examples per rank: {examples_per_rank}")

# Kick off the training loop
batch_iterator = get_batch()
total_training_time = 0
smooth_dt = 0
ema_beta = 0.9
for step in range(num_steps):
    # Evaluate the model once in a while and log to wandb
    if step % cfg.eval_every == 0:
        model.eval()
        passk = torch.zeros(
            cfg.device_batch_size, device=device
        )  # pass@k for k=1..device_batch_size
        with autocast_ctx:
            records_iter = run_gsm8k_eval(
                val_task,
                tokenizer,
                engine,
                num_samples=cfg.device_batch_size,
                max_examples=cfg.eval_examples,
                temperature=1.0,
            )
            records = list(records_iter)  # collect all records
        for k in range(1, cfg.device_batch_size + 1):
            passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)
        passk = passk / num_records.item()  # normalize by the total number of records
        print_passk = [
            f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, cfg.device_batch_size + 1)
        ]
        logger.info(f"Step {step} | {', '.join(print_passk)}")
        log_passk = {f"pass@{k}": passk[k - 1].item() for k in range(1, cfg.device_batch_size + 1)}
        wandb_run.log(
            {
                "step": step,
                **log_passk,
            }
        )

    # Forward/Backward on rollouts over multiple examples in the dataset
    synchronize()
    t0 = time.time()
    rewards_list = []
    sequence_lengths = []
    for example_step in range(examples_per_rank):
        # Get one batch corresponding to one example in the training dataset
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)
        # Evaluate the loss and gradients
        model.train()  # ensure the model is in train mode
        # We need one more loop because we can never exceed the device_batch_size
        assert inputs_all.size(0) % cfg.device_batch_size == 0
        num_passes = inputs_all.size(0) // cfg.device_batch_size
        for pass_idx in range(num_passes):
            # Pluck out the batch for this pass
            b0, b1 = (
                pass_idx * cfg.device_batch_size,
                (pass_idx + 1) * cfg.device_batch_size,
            )
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            rewards = rewards_all[b0:b1]
            advantages = advantages_all[b0:b1]
            # Calculate log probabilities. Note that the loss calculates NLL = -logp, so we negate
            with autocast_ctx:
                logp = -model(inputs, targets, loss_reduction="none").view_as(inputs)  # (B, T)
            # Calculate the PG objective. Note that ignore_index=-1 ensures that invalid tokens have loss 0.
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()
            # normalize by the number of valid tokens, number of passes, and examples_per_rank
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            # Note, there is no need to add PPO ratio+clip because we are on policy
            # Finally, formulate the loss that we want to minimize (instead of objective we wish to maximize)
            loss = -pg_obj
            loss.backward()
            logger.info(
                f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} | loss: {loss.item():.6f} | Average reward: {rewards.mean().item()}"
            )
        # For logging
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences_all)

    # A bunch of logging for how the rollouts went this step
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
    if ddp:  # aggregate across ranks
        mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
        mean_sequence_length_tensor = torch.tensor(
            mean_sequence_length, dtype=torch.float, device=device
        )
        dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_sequence_length_tensor, op=dist.ReduceOp.AVG)
        mean_reward = mean_reward_tensor.item()
        mean_sequence_length = mean_sequence_length_tensor.item()
    logger.info(
        f"Step {step}/{num_steps} | Average reward: {mean_reward} | Average sequence length: {mean_sequence_length:.2f}"
    )
    wandb_run.log(
        {
            "step": step,
            "reward": mean_reward,
            "sequence_length": mean_sequence_length,
        }
    )

    # Update the model parameters
    lrm = rl_linear_rampdown(step, num_steps=num_steps)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0

    # Timing metrics
    total_tokens_this_step = sum(sequence_lengths) * ddp_world_size
    tok_per_sec = int(total_tokens_this_step / dt) if dt > 0 else 0
    flops_per_sec = num_flops_per_token * total_tokens_this_step / dt if dt > 0 else 0
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt
    smooth_dt = ema_beta * smooth_dt + (1 - ema_beta) * dt
    logger.info(
        f"Step {step}/{num_steps} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f}% | total time: {total_training_time / 60:.2f}m"
    )

    wandb_run.log(
        {
            "step": step,
            "lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        }
    )

    # Master process saves the model once in a while. Skip first step. Save last step.
    if master_process and ((step > 0 and step % cfg.save_every == 0) or step == num_steps - 1):
        checkpoint_dir = os.path.join(meta["checkpoint_dir"], "standard_rl_checkpoints")
        # Use the dataclass-aware serializer instead of ``.__dict__`` so
        # nested-dataclass fields survive the JSON round-trip if
        # ``GPTConfig`` ever grows them (slice 11 audit P1 #7 / P3 #20).
        model_config_kwargs = asdict(model.config)
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None,  # note: we don't bother to save the optimizer state
            {
                "model_config": model_config_kwargs,
                # Persist the user_config too (slice 11 audit P1 #7) so
                # downstream tooling that walks RL ancestors via
                # ``_recover_*_user_config`` can fetch encoder /
                # cross_attention / feature_prediction settings.
                "user_config": user_config,
            },
        )
        logger.info(f"Saved model checkpoint to {checkpoint_dir}")

logger.info(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
logger.info(f"Total training time: {total_training_time / 60:.2f}m")

# Log to report
rl_checkpoint_dir = os.path.join(meta["checkpoint_dir"], "standard_rl_checkpoints")
get_report(checkpoint_dir=rl_checkpoint_dir).log(
    section="Chat RL",
    data=[
        user_config,  # CLI args
    ],
)

wandb_run.finish()  # wandb run finish
compute_cleanup()
