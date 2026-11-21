# nano-gpt-pretrain-steer

Pre-train GPT-2 class language models, then **steer** them with Sparse Autoencoder (SAE) features via an encoder-decoder cross-attention architecture. Built on a vendored copy of [Karpathy's nanochat](https://github.com/karpathy/nanochat).

---

## 1-minute quickstart

```bash
git clone <this repo> && cd nano-gpt-pretrain-steer
uv venv && uv sync
huggingface-cli login           # downloads Gemma + GemmaScope
bash runs/extract_features_quick.sh
```

If that finishes with `==> SUCCESS`, your install works. ~5 minutes, single GPU.

---

## What this is

The big idea: at training time a bidirectional **feature encoder** consumes per-segment SAE activations and feeds them into the decoder via cross-attention. At inference time you can **steer** generations by choosing which features the encoder sees.

```
text  ──►  decoder ──► next-token logits
              ▲
SAE features ─► feature encoder
```

| Phase | Status |
|---|---|
| Decoder pretraining (nanochat-style) | Production |
| SAE feature extraction pipeline | Production |
| Encoder-decoder conditional pretraining | Active |
| Conditioned encoder-decoder SFT | Active |
| Structured-SFT self-conditioning GRPO | Experimental; hardware qualification required |
| Verified-reward encoder-decoder RLVR | Experimental; generated rollouts, feature-policy gradients, default WandB. [Guide](docs/pipelines/encoder-decoder-rlvr.md) |
| RL fine-tuning | Implemented |
| Steering evals (AxBench-Base / AxBench-BPB / harmfulness) | Active |

**Hardware:** single GPU works for the quickstart and small experiments. Full pipeline assumes 8×H100 / 4×H200 / 2×B200. CPU/MPS works only for the educational `runs/runcpu.sh`.

---

## Setup (first time)

```bash
# 1. Install uv (one-time, system-wide)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Set up the project venv
uv venv && uv sync

# 3. Configure secrets
cp .env.example .env && $EDITOR .env

# 4. Authenticate
huggingface-cli login           # for Gemma + GemmaScope downloads
wandb login                     # optional, for run logging
```

See [`.env.example`](.env.example) for what each environment variable does.

---

## Where to go next

| You want to... | Read |
|---|---|
| See every CLI command | [`docs/cli-reference.md`](docs/cli-reference.md) |
| Understand the architecture | [`docs/reference/architecture.md`](docs/reference/architecture.md) |
| See every config knob | [`docs/reference/configuration.md`](docs/reference/configuration.md) |
| Look up what a file does | [`docs/reference/file-reference.md`](docs/reference/file-reference.md) |
| Run the full pipeline (8×H100) | [`runs/speedrun.sh`](runs/speedrun.sh) |
| Run feature extraction | [`docs/pipelines/feature-extraction.md`](docs/pipelines/feature-extraction.md) |
| Train an encoder-decoder model | [`docs/pipelines/encoder-decoder-training.md`](docs/pipelines/encoder-decoder-training.md) |
| Run an evaluation | [`docs/pipelines/evaluation.md`](docs/pipelines/evaluation.md) |
| Profile a bottleneck | [`docs/performance/profiling.md`](docs/performance/profiling.md) |
| Pull project checkpoints + feature shards | [`docs/pipelines/artifacts-sync.md`](docs/pipelines/artifacts-sync.md) |
| Modify `nanochat/` | Don't — see [`nanochat/VENDORED.md`](nanochat/VENDORED.md) |

---

## Repo layout

```
steering/    SAE feature extraction + encoder-decoder conditioning (the research code)
nanochat/    Vendored training framework — DO NOT MODIFY (see nanochat/VENDORED.md)
vendor/      Vendored projects with isolated environments and patch policies
scripts/     Pipeline entry points (training, extraction, eval)
configs/     Hydra YAML configs
tasks/       SFT task definitions (GSM8K, MMLU, ARC, safety datasets)
tests/       pytest suite
runs/        End-to-end pipeline shell scripts (speedrun.sh, runcpu.sh, ...)
docs/        Documentation
```

---

## Testing

```bash
uv run pre-commit install              # install the standard local checks
uv run pre-commit run --all-files      # ruff + ty + vulture + caller ratchet
uv run pytest tests/                  # fast, default
uv run pytest tests/code_quality/test_production_callers.py  # test-only production API ratchet
uv run pytest tests/ -m "slow"        # GPU-required
uv run pytest tests/ -m benchmark     # perf tests
uv run ruff check .                   # lint
uv run ty check                       # type check (the type checker is `ty`)
uv run vulture                        # high-confidence dead-code scan
```

Markers: `slow`, `very_slow`, `benchmark`, `api`, `requires_feature_data`.

Vulture's pre-commit gate uses its high-confidence tier. Run
`uv run vulture --min-confidence 60` for the broader review surface; findings
at that tier require manual verification because framework callbacks,
dataclass fields, and reflective APIs are common in this repository.

---

## Conventions

- `import torch as t` everywhere
- `from loguru import logger` for all logging (rank-aware in DDP)
- Hydra for all CLI configs; imports at the top of every file
- Package management via `uv` (not pip/poetry)

## License

MIT — see [`LICENSE`](LICENSE). Vendored `nanochat/` is also MIT (see
`nanochat/LICENSE`); vendored AxBench is Apache-2.0 (see
`vendor/axbench/LICENSE`).
