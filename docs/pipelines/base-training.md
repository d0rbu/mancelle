# Base Model Training Pipeline

Pre-train a GPT-2 class causal language model from scratch on FineWeb-Edu data.

## Overview

```
Data download → Tokenizer training → Base pretraining → Evaluation
```

**Scripts:** `scripts/train/base_train.py`, `scripts/eval/base_eval.py`, `scripts/preprocess/tok_train.py`, `scripts/preprocess/tok_eval.py`
**Configs:** `configs/training/base_train.yaml`, `configs/eval/base_eval.yaml`, `configs/data/tok_train.yaml`

## End-to-End: Speedrun (8xH100, ~3 hours)

The fastest way to train a complete model:

```bash
bash runs/speedrun.sh
```

This runs the full pipeline: data download → tokenizer training → pretraining → evaluation → SFT → chat evaluation.

With wandb logging:
```bash
WANDB_RUN=my-run screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
```

## Step-by-Step

### 1. Download Data

```bash
# Download 8 shards for tokenizer training (~800MB)
python -m nanochat.dataset -n 8

# Download 370 shards for full pretraining (~37GB)
python -m nanochat.dataset -n 370
```

Data is stored in `$NANOCHAT_BASE_DIR/base_data/` (default: `~/.cache/nanochat/base_data/`).

### 2. Train Tokenizer

```bash
# Train BPE tokenizer (vocab_size=32768, ~2B chars)
python -m scripts.preprocess.tok_train

# Evaluate compression ratio
python -m scripts.preprocess.tok_eval
```

Outputs tokenizer to `$NANOCHAT_BASE_DIR/tokenizer/`.

### 3. Pre-train

```bash
# Single GPU
python -m scripts.train.base_train

# 8 GPUs (distributed)
torchrun --standalone --nproc_per_node=8 -m scripts.train.base_train -- depth=26 device_batch_size=16

# With FP8 (Hopper/Blackwell GPUs)
torchrun --standalone --nproc_per_node=8 -m scripts.train.base_train -- depth=26 device_batch_size=16 fp8=true

# Custom training horizon
torchrun --standalone --nproc_per_node=8 -m scripts.train.base_train -- target_param_data_ratio=10.5
```

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `depth` | 26 | Number of transformer layers |
| `aspect_ratio` | 64 | Numeric per-layer width ratio; `n_embd = depth * aspect_ratio` rounded up to a multiple of `head_dim` (fractional ratios are allowed for targeting a fixed width) |
| `head_dim` | 128 | Per-head dimension |
| `max_seq_len` | 1024 | Context length |
| `device_batch_size` | 32 | Per-GPU batch size |
| `total_batch_size` | -1 | Auto-computed from scaling laws |
| `target_param_data_ratio` | 8.25 | Data tokens / model params ratio |
| `fp8` | false | Enable FP8 training |
| `attn_impl` | auto | Flash attention backend (auto/fa4/fa3/sdpa) |
| `window_pattern` | "SSSL" | Sliding window pattern |
| `resume_from_step` | -1 | Resume from checkpoint step (-1 = start fresh) |

### Automatic Scaling

When `num_iterations: -1` and `total_batch_size: -1`, the training script auto-computes:
- Total tokens from `target_param_data_ratio * num_params`
- Batch size from scaling laws
- Learning rate schedule with warmdown

### 4. Evaluate

```bash
# CORE metric + BPB + sampling (distributed)
torchrun --standalone --nproc_per_node=8 -m scripts.eval.base_eval -- device_batch_size=16

# Specific evaluation modes
torchrun --standalone --nproc_per_node=8 -m scripts.eval.base_eval -- eval=core
torchrun --standalone --nproc_per_node=8 -m scripts.eval.base_eval -- eval=bpb
torchrun --standalone --nproc_per_node=8 -m scripts.eval.base_eval -- eval=sample

# Evaluate specific model tag/step
torchrun --standalone --nproc_per_node=8 -m scripts.eval.base_eval -- model_tag=d24 step=5000

# Evaluate HuggingFace model for comparison
torchrun --standalone --nproc_per_node=8 -m scripts.eval.base_eval -- hf_path=openai-community/gpt2
```

### Evaluation Modes

| Mode | Metric | Description |
|------|--------|-------------|
| `core` | CORE centered accuracy | DCLM in-context learning tasks |
| `bpb` | Bits per byte | Tokenizer-agnostic loss on train/val |
| `sample` | Generated text | Model samples from prompts |

## Checkpoints

Checkpoints are saved to `$NANOCHAT_BASE_DIR/base_checkpoints/`:

```
base_checkpoints/
├── model_000500.pt       # Model weights at step 500
├── model_001000.pt
├── optim_000500_rank0.pt # Optimizer state (per-rank)
├── meta_000500.json      # Training metadata
└── ...
```

## Hardware-Specific Configs

| Config | Hardware | Batch Size | Notes |
|--------|----------|------------|-------|
| `base_train.yaml` | 8xH100 | 32 | Default |
| `conditional_encdec_4090.yaml` | RTX 4090 | 4 | Reduced depth=12, seq_len=1024 |

## Logging

- **Wandb**: Set `run=my-run-name` (or `WANDB_RUN` env var). Use `run=dummy` to disable.
- **Report**: Markdown report generated to `$NANOCHAT_BASE_DIR/report/` via `python -m nanochat.report generate`.
