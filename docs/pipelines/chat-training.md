# Chat Training Pipeline (SFT + RL)

Fine-tune a pre-trained base model for conversational use via Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL).

## Overview

Each fine-tuning stage nests its checkpoints inside the parent stage's directory, forming a tree of model lineages:

```
Base model ─┬─ Standard SFT ──── RL
            ├─ Safety SFT ──┬─── RL
            │               └─── Capabilities SFT
            └─ Capabilities SFT
```

**Scripts:** `scripts/train/chat_sft.py`, `scripts/train/chat_rl.py`, `scripts/eval/chat_eval.py`
**Configs:** `configs/training/chat_sft.yaml`, `configs/training/chat_rl.yaml`, `configs/eval/chat_eval.yaml`

## SFT (Supervised Fine-Tuning)

### Quick Start

```bash
# Standard SFT (8 GPUs)
torchrun --standalone --nproc_per_node=8 \
    -m scripts.train.chat_sft -- \
    device_batch_size=16 \
    run=my-sft-run

# Single GPU
python -m scripts.train.chat_sft
```

### SFT Mixtures

Five pre-defined training mixtures:

#### Standard (`mixture=standard`)
General-purpose chat training:
- SmolTalk (general conversation)
- Custom identity conversations (personality)
- MMLU (1 epoch, knowledge)
- GSM8K (1 epoch, math)
- SimpleSpelling + SpellingBee

#### Capabilities (`mixture=capabilities`)
Knowledge and reasoning focused:
- SmolTalk (120k rows), 2× custom identity, MMLU (2 epochs),
  OpenCoder SFT stage 2 (capped at 50k rows), OrcaMath, MetaMath (GSM-only)
- SimpleSpelling, SpellingBee

#### Safety (`mixture=safety`)
Refusal and safety training:
- WildJailbreak (2 epochs), WildGuardMix (3 epochs), CoCoNot (2 epochs)
- SmolTalk (120k), Nemotron SFT Safety, 2× identity conversations

#### Steered mixtures (`mixture=standard_steered`, `mixture=standard_steered_balanced`)

Both splice a **steered dataset** — single-turn conversations sampled from a
steered model — into the standard mixture. Point them at the file with
`steered_jsonl=/path/to/steered.jsonl`; the key is **required** for these two
mixtures and **rejected** for every other one (a silently ignored steered
dataset would be a training bug, not a nuisance).

**JSONL format.** One conversation per non-blank line, holding **exactly two**
message objects — a `user` prompt and a **non-blank** `assistant` completion —
either as a bare JSON list or under the `"messages"` key of a JSON object (the
shape the steered-dataset export writes; its other keys such as `id` are
provenance and are ignored):

```jsonl
[{"role": "user", "content": "Tell me about bridges"}, {"role": "assistant", "content": "Bridges span ..."}]
{"id": "0_975", "messages": [{"role": "user", "content": "And tunnels?"}, {"role": "assistant", "content": "Tunnels bore ..."}]}
```

Anything else — malformed JSON, an object without a `messages` list, a
different message count, wrong roles, non-string or blank assistant content, a
missing file, zero rows — raises `errors.DataLoadError` naming the file and the
1-based line number. Extra per-message fields (e.g. the steering feature id)
are preserved verbatim.

- **`standard_steered`** appends the steered task to the standard train tasks.
- **`standard_steered_balanced`** additionally down-caps the standard tasks to
  the steered row count, giving a 50/50 mix. The cap is a largest-remainder
  apportionment proportional to each task's length (ties go to the lower
  index); a task whose share rounds to zero is kept at length 0 so the mixture
  fingerprint still enumerates the whole catalogue. Requires
  `steered_rows <= standard_rows`.

**Exact even spacing.** The steered rows are not shuffled into the mixture.
Steered row `k` (in the seed-42 shuffled steered order) is placed at combined
position `((2k + 1) * L) // (2M)` for `M` steered rows out of `L` total; the
base rows keep their usual seed-42 order and fill the gaps. Exactly one
steered row therefore lands in every half-open interval `[kL/M, (k+1)L/M)`.
(That is weaker than "every window of `L/M` positions": for `L=10, M=3` the
positions are `[1, 5, 8]`, and the 3-wide window `(2, 3, 4)` carries none.)

> **DDP caveat.** The spacing is exact at the DATASET level only. The
> conditioned SFT generator strides dataset positions by `world_size`, so a
> rank's steered share can be uneven — a spacing of 6 lands only on odd ranks
> at world size 2, 4 and 8. The averaged gradient still carries the global
> steered fraction.

**Validation family.** The steered mixtures have no val split of their own:
they validate on **standard's**. Convert them train-only
(`split=train`) and convert `mixture=standard split=val` once for the val side.

### SFT Commands

```bash
# Standard mixture
torchrun --standalone --nproc_per_node=8 \
    -m scripts.train.chat_sft -- \
    mixture=standard \
    device_batch_size=16

# Safety mixture
torchrun --standalone --nproc_per_node=8 \
    -m scripts.train.chat_sft -- \
    mixture=safety \
    device_batch_size=16

# Capabilities mixture
torchrun --standalone --nproc_per_node=8 \
    -m scripts.train.chat_sft -- \
    mixture=capabilities \
    device_batch_size=16

# Custom parameters
torchrun --standalone --nproc_per_node=8 \
    -m scripts.train.chat_sft -- \
    mixture=standard \
    mmlu_epochs=5 \
    gsm8k_epochs=6 \
    num_iterations=2000

# Specific model tag
torchrun --standalone --nproc_per_node=8 \
    -m scripts.train.chat_sft -- \
    model_tag=d26 \
    model_step=5000

# Dry run (train without persisting checkpoints or writing a report)
python -m scripts.train.chat_sft dry_run=true
```

To count tokens without training, run `uv run python -m scripts.preprocess.count_tokens`
with `configs/training/count_tokens.yaml`; it dispatches to
`count_sft_tokens`, `count_capabilities_sft_tokens`, or `count_safety_sft_tokens`
for the selected mixture. SFT estimates use the same conversation preparation as
`convert_sft_to_parquet`: rows are truncated after the final assistant turn and
rows with no assistant turn or an empty final response are excluded. The count
remains the full nanochat render, including its BOS token; parquet conversion
strips BOS only when writing the stored text.

### SFT Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_backend` | nanochat | `nanochat` or `huggingface` |
| `huggingface` | null | Generic immutable-source/AdamW/export block used only by the HF presets |
| `mixture` | standard | standard, capabilities, safety, standard_steered, or standard_steered_balanced |
| `steered_jsonl` | null | Steered single-turn JSONL; required by (and only accepted for) the steered mixtures |
| `max_seq_len` | 1024 | Maximum sequence length |
| `device_batch_size` | 32 | Per-GPU batch size |
| `total_batch_size` | 524288 | Total tokens per gradient step |
| `num_iterations` | -1 | Auto-computed from data |
| `embedding_lr` | 0.3 | Embedding learning rate |
| `matrix_lr` | 0.02 | Weight matrix learning rate |
| `weight_decay` | 0.0 | No weight decay for SFT |
| `eval_every` | 150 | Evaluation frequency (steps) |
| `sft_loss_mask` | full_sequence | `full_sequence` (default) or `completion_only`; the latter supervises assistant text and tool calls while excluding user turns and tool outputs |

### SFT Dataloader

Uses BOS-aligned best-fit packing (same as pretraining) but:
- Target supervision follows `sft_loss_mask`. The default, `full_sequence`, supervises all non-padding targets. Optional `completion_only` supervises assistant text and tool calls (`<python_start>` through `<python_end>`) while excluding user turns and injected tool outputs (`<output_start>` through `<output_end>`).
- Pads instead of cropping when documents don't fit
- Conversations rendered with special tokens: `<user_start>`, `<user_end>`, `<assistant_start>`, `<assistant_end>`

The training loop, not the generator, owns the optimizer-step horizon. It
commits a produced conversation cursor only after `optimizer.step()` succeeds,
so a bounded run executes exactly `num_iterations` updates and never pulls an
untrained batch after the final microbatch. HF distributed runs build one
complete accumulation window before forward/backward so a rank-local data
failure propagates before any training-model collective. The committed HF
state includes the next striped logical cursor and ordered pending best-fit
positions; resume re-renders that exact buffer instead of approximating it with
a scalar conversation count.

### Pinned Hugging Face decoder SFT

The `chat_sft_smollm3_stage1_step_40000`, `chat_sft_pythia_1_4b`, and
`chat_sft_pythia_2_8b` presets train native Transformers
`SmolLM3ForCausalLM` and `GPTNeoXForCausalLM`
instances from immutable revisions. SmolLM3 uses
`524d818535ee98e56a4f0d621c84dd5c8fb8d3d4`; Pythia-1.4B uses
`fedc38a16eea3bd36a96b906d78d11d2ce18ed79`; and Pythia-2.8B uses
`2a259cdd96a4beb1cdf467512e3904197345f6a9`.
They share the
standard task mixture, deterministic shuffle, BOS best-fit packing, validation
BPB, batch schedule, flat-80%-then-decay LR schedule, and logging loop with
native decoder SFT. They do not convert either model into `nanochat.GPT`.
The [Pythia model card](https://huggingface.co/EleutherAI/pythia-2.8b)
identifies the final checkpoint with training revision `step143000`; the preset
pins the immutable `main` commit containing that final safetensors export.

HF model parameters and Adam moments remain FP32; forward/backward uses BF16
autocast and non-reentrant gradient checkpointing. SDPA is the default;
`flash_attention_4` is an explicit validated alternative. The default
`dist_muon_adamw` strategy keeps the repository's
`MuonAdamW`/`DistMuonAdamW` one-Adam-group path and optimizer-owned gradient
reduction. The alternative `ddp_fused_adamw` optionally compiles the adapter,
wraps it in DDP, suppresses accumulation reductions with `no_sync()`, and uses
fused CUDA AdamW. SmolLM3's input/output embedding is tied and appears once;
Pythia's input/output embeddings remain distinct.

For either strategy, the optimized objective is the exact global mean over
`targets != -1`, not the mean of rank or microbatch means. Each rank
backpropagates its summed loss multiplied by `world_size / global_valid_tokens`;
the strategy's AVG gradient reduction then produces the concatenated-batch
oracle exactly. Logs distinguish scheduled, non-padding, and supervised target
counts and throughput.

SmolLM3 conversations export as simple ChatML using
`<|im_start|>`/`<|im_end|>`.
Repository Python calls map to `<tool_call>...</tool_call>` and injected output
to `<tool_response>...</tool_response>`. Completion-only masks exclude BOS,
user material, assistant headers, and tool responses while supervising
assistant text, tool calls, the assistant `im_end`, and its trailing newline.
Consecutive supervised multipart pieces share one generation span, preventing
a text/tool-call boundary from splitting supervision inside a byte-fragmented
Unicode character.
Full-sequence mode remains the preset default.

Pythia's tokenizer has no ChatML controls, so its template uses plain
`system:`, `user:`, and `assistant:` headers and the existing
`<|endoftext|>` token as the message terminator. Tool calls and responses keep
the same literal tags. Newlines separate tool-response regions because
Pythia's ByteLevel vocabulary otherwise has a `><` piece that crosses a
supervised-call/unsupervised-response mask boundary. The payload bytes are not
changed. Those separator newlines are supervised structural material; only the
tool-response tags and payload are excluded. Consecutive supervised multipart
content is accumulated into one Jinja generation span ending at a separator or
EOS, so a span never ends partway through a byte-fragmented Unicode character.
The exported ByteLevel post-processor also retains complete token offsets
(`trim_offsets=false`). Together these rules make leading whitespace and
multipart boundaries map to whole tokens without changing rendered text or
token IDs.

Neither pinned base tokenizer has a chat template. The adapter therefore
installs the family contract's Jinja template, then delegates
rendering and assistant-mask generation to Transformers'
`apply_chat_template`. Its `{% generation %}` blocks are necessary to express
the repository contract that tool calls are supervised while tool responses
are not. The same installed template is saved with every export, so training,
completion prompting, and downstream Hugging Face generation share one wire
format rather than separate renderers.

Each save is an atomic BF16 Hugging Face directory with tokenizer, installed
Jinja chat template, generation config, and `sft_metadata.json`; it contains no
optimizer, RNG, dataloader, or resume state and records `resumable: false`.
Exports can be sources for subsequent SFT stages; see the exact family
[BOS/EOS/PAD chaining contract](hf-decoder-sft.md).
Separate `<output>/restart/` artifacts store the FP32 model, strategy-specific
optimizer state, Torch CPU/CUDA RNG, and rank-local ordered best-fit state.
They are loadable only after a complete marker binds the exact inventory,
semantic identity, strategy, and world/local-world topology. Save with
`huggingface.restart_save_every=N`; resume only an explicit complete step with
`huggingface.restart_from_step=N`. The initial restored boundary is never saved
again; persistence resumes only after a new optimizer update commits. When both
artifact types are due, restart publishes first, allowing an exact-final resume
to backfill a missing portable export without overwriting completed state.
The standalone harmfulness evaluator can load one concrete exported step using
`hf_checkpoint_dir=.../step_NNNNNN`; it renders exactly this saved template and
binds the entire export to response-cache provenance. General chat evaluation,
RL, arbitrary external HF architectures, HF steering/progression, automatic
restart discovery, and topology-changing restart remain outside this backend's
scope.

The 32-H200 benchmark presets and fail-closed Slurm launcher are documented in
the [HF decoder SFT H200 runbook](../performance/hf-decoder-sft-h200.md). The
launcher freezes git/lock/data identities, runs NCCL preflight, and retains one
synchronized outer `paid_step` timing row per rank/update. Fine-grained timing
and Chrome traces are separate diagnostic arms.

The pinned source is the first 40,000-step Stage-1 release (about 94.4B
pretraining tokens), so weak initial instruction following is a source-model
quality property, not by itself an integration failure.

### SFT LR Schedule

Flat for 80% of training → linear decay to 0.

## RL (Reinforcement Learning)

### Quick Start

```bash
# GRPO on GSM8K (8 GPUs)
torchrun --standalone --nproc_per_node=8 \
    -m scripts.train.chat_rl -- \
    run=my-rl-run

# Single GPU
python -m scripts.train.chat_rl
```

### RL Pipeline

Uses a simplified GRPO (Group Relative Policy Optimization) on GSM8K math problems:

1. **Sample:** Generate `num_samples` completions per problem
2. **Score:** Check if answer matches ground truth
3. **Advantage:** `reward - mean(rewards)` (no sigma normalization)
4. **Update:** REINFORCE-style policy gradient

### RL Commands

```bash
# Default RL training
torchrun --standalone --nproc_per_node=8 \
    -m scripts.train.chat_rl

# Custom sampling
torchrun --standalone --nproc_per_node=8 \
    -m scripts.train.chat_rl -- \
    num_samples=32 \
    temperature=0.8 \
    max_new_tokens=512

# Custom training
torchrun --standalone --nproc_per_node=8 \
    -m scripts.train.chat_rl -- \
    examples_per_step=32 \
    num_epochs=2 \
    eval_every=30
```

### RL Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 1 | Training epochs over GSM8K |
| `examples_per_step` | 16 | Problems per gradient step |
| `num_samples` | 16 | Rollouts per problem |
| `max_new_tokens` | 256 | Max generated tokens |
| `temperature` | 1.0 | Sampling temperature |
| `top_k` | 50 | Top-k sampling |
| `device_batch_size` | 8 | Per-GPU batch size |
| `eval_every` | 60 | Evaluation frequency |
| `eval_examples` | 400 | Validation examples |

### RL LR Schedule

Linear rampdown to 0 over total steps.

## Chat Evaluation

```bash
# Evaluate standard SFT model (base_checkpoints/<tag>/standard_sft_checkpoints/)
torchrun --standalone --nproc_per_node=8 \
    -m scripts.eval.chat_eval -- \
    source=sft

# Evaluate RL model (base_checkpoints/<tag>/standard_sft_checkpoints/standard_rl_checkpoints/)
torchrun --standalone --nproc_per_node=8 \
    -m scripts.eval.chat_eval -- \
    source=rl

# Evaluate safety SFT (base_checkpoints/<tag>/safety_sft_checkpoints/)
torchrun --standalone --nproc_per_node=8 \
    -m scripts.eval.chat_eval -- \
    source=safety

# Evaluate safety→RL chain (base_checkpoints/<tag>/safety_sft_checkpoints/standard_rl_checkpoints/)
torchrun --standalone --nproc_per_node=8 \
    -m scripts.eval.chat_eval -- \
    source=safety/rl

# Evaluate specific tasks
torchrun --standalone --nproc_per_node=8 \
    -m scripts.eval.chat_eval -- \
    source=sft \
    'task_name=GSM8K|ARC-Easy|MMLU'

# Single task with custom generation
python -m scripts.eval.chat_eval \
    source=sft \
    task_name=HumanEval \
    temperature=0.8 \
    num_samples=10
```

### Evaluation Tasks

| Task | Type | Metric |
|------|------|--------|
| ARC-Easy / ARC-Challenge | Categorical (logit ranking) | Accuracy |
| MMLU | Categorical (logit ranking) | Accuracy |
| GSM8K | Generative (sampling) | pass@k |
| HumanEval | Generative (sampling) | pass@k |
| SpellingBee | Generative (sampling) | Exact match |

**ChatCORE** — The aggregate metric: mean centered accuracy across all tasks.

For the public instruction-tuned Qwen2.5-3B and SmolLM3-3B baselines, use
`--config-name=chat_eval_qwen2_5_3b_instruct` or
`--config-name=chat_eval_smollm3_3b` with a fresh `output_dir` and no `source`.
See [public HF ChatCORE evaluation](evaluation.md#public-instruction-tuned-hf-baselines)
for pinned revisions, non-thinking SmolLM3 rendering, distributed examples,
strict grading, and per-example evidence. These use the public release's
template; repository SFT exports use the training templates described above.

## Harmfulness Evaluation

```bash
# Safety SFT model (base_checkpoints/d26/safety_sft_checkpoints/)
python -m scripts.eval.harmfulness_eval \
    source=safety \
    model_tag=d26

# Safety→RL chain (base_checkpoints/d26/safety_sft_checkpoints/standard_rl_checkpoints/)
python -m scripts.eval.harmfulness_eval \
    source=safety/rl \
    model_tag=d26

# Judge only (skip generation, reuse existing responses)
python -m scripts.eval.harmfulness_eval \
    source=safety \
    model_tag=d26 \
    judge_only=true

# Encoder-decoder model with feature configs (file or directory)
python -m scripts.eval.harmfulness_eval \
    source=encdec \
    feature_configs=out/data/common_features/safety_sft/absolute/k32/seed_42/r_mean_pct_incl_zeros/m0.5/features.yaml

# Encoder-decoder: auto-discover all configs in a directory
python -m scripts.eval.harmfulness_eval \
    source=encdec \
    feature_configs=out/data/common_features/safety_sft/

# Encoder-decoder after safety SFT
python -m scripts.eval.harmfulness_eval \
    source=encdec/safety \
    model_tag=d26

# One immutable local repository Hugging Face SFT export
python -m scripts.eval.harmfulness_eval \
    hf_checkpoint_dir=/path/to/step_000123
```

Uses separate harmfulness and benign-compliance LLM judgments via OpenRouter
(default `openai/gpt-5.6-luna`). **Requires `OPENROUTER_API_KEY`**. Outputs the
harmful-prompt attack success rate, exact-score-1 safe response rate, overall
mean harmfulness score, and benign non-refusal compliance rate with auditable
per-pair reasoning. The required 1--5 helpfulness score is persisted separately
as a diagnostic and does not affect compliance.

HF mode accepts the shared repository export contract used by SmolLM3 and
Pythia/GPT-NeoX. It does not accept Hub IDs, external checkpoints, mutable
parent directories, feature configs, or steering. Results and response caches
live below the export's `report/harmfulness_results*/` directories; `report/`
is excluded from the checkpoint content digest.

## Steering Evaluation

Evaluate SAE feature steering on encoder-decoder models with pairwise judging. Also uses OpenRouter for the judge (**requires `OPENROUTER_API_KEY`**).

```bash
# Run steering eval with a config
python -m scripts.eval.steering_eval \
    source=encdec \
    steering_config=path/to/steering_config.yaml \
    model_tag=my_model

# Extract absolute conditioning features first
python -m scripts.preprocess.extract_common_features \
    feature_dir=out/data/features \
    top_k=32 \
    num_buckets=10 \
    output=features.yaml
```

## Evaluation Visualization

```bash
# Auto-discover checkpoints and plot comparison charts
python -m scripts.eval.plot_evals

# Custom output directory
python -m scripts.eval.plot_evals output_dir=fig/my_experiment
```

Generates PNG charts comparing harmfulness, base_eval, and steering metrics across model lineages.

## Checkpoint Hierarchy

Downstream checkpoints are **nested inside their parent's directory**. Each fine-tuning stage writes its checkpoints as a subdirectory of the checkpoint it was initialized from. This naturally supports multi-stage training and branching pipelines.

### Structure

```
$NANOCHAT_BASE_DIR/
├── base_checkpoints/
│   └── d26/                                    # model_tag (by depth)
│       ├── model_005000.pt                     # base model weights
│       ├── meta_005000.json
│       ├── standard_sft_checkpoints/           # source="sft"
│       │   ├── model_000300.pt
│       │   ├── standard_rl_checkpoints/        # source="sft/rl"  or  "rl"
│       │   │   └── model_000120.pt
│       │   └── report/
│       ├── safety_sft_checkpoints/             # source="safety"
│       │   ├── model_000300.pt
│       │   ├── capabilities_sft_checkpoints/   # source="safety/capabilities"
│       │   │   └── model_000300.pt
│       │   ├── standard_rl_checkpoints/        # source="safety/rl"
│       │   │   └── model_000120.pt
│       │   └── report/
│       └── capabilities_sft_checkpoints/       # source="capabilities"
│           └── model_000300.pt
│
└── conditional_encdec_checkpoints/
    └── d26/                                    # model_tag
        └── s128/                               # segment_size
            ├── t2048.9/                        # threshold mode
            │   ├── model_005000.pt             # source="encdec"
            │   ├── safety_sft_checkpoints/     # source="encdec/safety"
            │   │   └── model_000300.pt
            │   └── report/
            └── dist_geometric/                 # sampling mode
                └── mean_5.0/
                    └── std_5.0/
                        ├── model_005000.pt
                        └── report/
```

### Source Resolution

The `source` parameter is a `/`-separated chain that resolves to a nested path. The first segment determines the root; subsequent segments append subdirectories.

| Source string | Resolves to |
|---------------|-------------|
| `base` | `base_checkpoints/<tag>/` |
| `sft` or `standard` | `base_checkpoints/<tag>/standard_sft_checkpoints/` |
| `safety` | `base_checkpoints/<tag>/safety_sft_checkpoints/` |
| `capabilities` | `base_checkpoints/<tag>/capabilities_sft_checkpoints/` |
| `rl` | `base_checkpoints/<tag>/standard_sft_checkpoints/standard_rl_checkpoints/` |
| `safety/rl` | `base_checkpoints/<tag>/safety_sft_checkpoints/standard_rl_checkpoints/` |
| `safety/capabilities` | `base_checkpoints/<tag>/safety_sft_checkpoints/capabilities_sft_checkpoints/` |
| `encdec` | `conditional_encdec_checkpoints/<tag>/s<seg>/<feature_selection>/` |
| `encdec/safety` | `conditional_encdec_checkpoints/<tag>/s<seg>/<feature_selection>/safety_sft_checkpoints/` |

**Key rules:**
- Bare `rl` is shorthand for `sft/rl` (backward compatibility)
- SFT mixture names (`sft`/`standard`, `safety`, `capabilities`) implicitly start from the `base` root
- `encdec` checkpoints have extra nesting levels for `s{segment_size}/` plus feature selection dirs: `t{threshold}` (threshold mode) or `dist_{dist}/mean_{mean}/std_{std}` (sampling mode)
- Chains can be arbitrarily deep (e.g., `safety/capabilities/rl`)
