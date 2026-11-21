# Getting started

Detailed walkthrough for a new contributor. The [README](../../README.md) has the 1-minute version. This is the 30-minute version: install, authenticate, run an end-to-end loop, learn where things live.

## 1. Hardware check

| What you have | What you can do |
|---|---|
| 8×H100 / 4×H200 / 2×B200 | Full pipeline, FP8 training, large sweeps. The reference is `runs/speedrun.sh`. |
| 1×H100 / H200 | Most experiments work. Reduce `device_batch_size` and bump `total_batch_size`'s grad-accum. |
| 1×RTX 4090 (24 GB) | Floor for encoder-decoder training. See `runs/run_4090.sh`. |
| Single 8 GB consumer GPU | Quickstarts only (`base_train_tiny_1gpu`, `extract_features_quick`). |
| CPU / Apple Silicon | Educational only. `runs/runcpu.sh` runs but won't produce a usable model. |

CUDA setup: PyTorch 2.13 with CUDA 13.0 wheels (the `[[tool.uv.index]] url = "https://download.pytorch.org/whl/cu130"` in `pyproject.toml`). The repo auto-detects FA4 (Hopper+/Blackwell) → FA3 (Hopper) → SDPA fallback.

## 2. Install

```bash
# Once per machine
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# In the repo
git clone git@github.com:Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer.git
cd nano-gpt-pretrain-steer
uv venv
uv sync
```

`uv sync` is the canonical install. If you see "package X not found," delete `.venv/` and re-sync rather than reaching for pip.

## 3. Authenticate

Copy the env template and fill it in:

```bash
cp .env.example .env
$EDITOR .env
```

| Service | Why you need it | How to get a token |
|---|---|---|
| Hugging Face | Download Gemma-2-2b weights and GemmaScope SAEs | https://huggingface.co/settings/tokens (read access is enough) |
| Weights & Biases | Run logging, metrics dashboard | `wandb login` (or set `WANDB_API_KEY`) |
| OpenRouter | LLM-as-judge for `steering_eval`, `harmfulness_eval`, AxBench | https://openrouter.ai/keys |
| Neuronpedia | SAE feature interpretation / labels | https://www.neuronpedia.org/ |

You can skip OpenRouter / Neuronpedia until you actually need them. HF and wandb are needed for almost everything.

`NANOCHAT_BASE_DIR` (defaults to `~/.cache/nanochat`) is where checkpoints, downloaded data, the tokenizer, and intermediate artifacts go. Make sure it has space — feature extraction can produce tens of GB per dataset.

Tests that need the real BPE tokenizer **skip cleanly** (they do not fail or error) when this cached `tokenizer/tokenizer.pkl` is absent — build it with `python -m scripts.preprocess.tok_train`. The shared `tests.conftest.get_tokenizer_or_skip()` helper enforces this.

## 4. First end-to-end loop

The smallest meaningful path: **download data → extract one shard of features → train a tiny model → eval it**. ~20 minutes on a single GPU.

### 4a. Validate the install (one command)

```bash
bash runs/extract_features_quick.sh
```

This downloads one pretraining shard and extracts SAE features from it. You should see:

```
==> SUCCESS: features extracted to out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base
    1 shard(s) written
```

If you don't see `==> SUCCESS`, jump to [Common errors](#common-errors).

### 4b. Train a tiny base model

```bash
python -m scripts.train.base_train config_name=base_train_tiny_1gpu
```

You should see loss decrease over ~100 iterations, with eval every 25 steps. Output goes to `$NANOCHAT_BASE_DIR/base_checkpoints/d6/` (or `model_tag` if you set one). With `save_every: -1` (default for the tiny config) no checkpoint is written — that's intentional for a smoke test.

### 4c. Quick eval

```bash
python -m scripts.eval.base_eval --config-name=base_eval_quick source=base model_tag=d6
```

This reports BPB only (CORE is disabled in the quick config). Expect a high BPB number — a 100-iteration smoke model isn't a real model.

You've now exercised the three core pipelines (extract / train / eval).

### 4d. Using Gemma 3 (gemma-scope-2)

The default extraction targets Gemma-2-2b + GemmaScope (v1). To extract against Gemma 3 + the gemma-scope-2 SAE family instead, switch to the gemma3 config:

```bash
python -m steering.extract_features --config-name extract_features_gemma3
```

Gemma 3 repos are gated on Hugging Face, so `HF_TOKEN` must have access (accept the license on the relevant `google/gemma-3-*` model page first). The full v2 pipeline — model list, L0 tiers, valid layers — is documented in [`docs/pipelines/gemma-scope-2.md`](../pipelines/gemma-scope-2.md).

## 5. What lives where

Inside `$NANOCHAT_BASE_DIR`:

```
base_checkpoints/<tag>/    Pretrained model checkpoints
sft_checkpoints/<tag>/     SFT model checkpoints
conditional_encdec_checkpoints/<tag>/   Encoder-decoder checkpoints
tokenizer/                 Trained BPE tokenizer
base_data/                 Downloaded FineWeb-Edu parquet shards
sft_data/                  SFT conversation data
eval_bundle/               Cached eval data (DCLM CORE)
report/                    Auto-generated training report
```

Inside the repo (gitignored):

```
out/data/features/<feature_path>/     Extracted SAE features (CSR safetensors shards)
out/profiling/                        Profiling outputs
outputs/                              Hydra run dirs (configs + logs per run)
fig/                                  Generated figures
logs/                                 Run logs
wandb/                                Local wandb run dirs
```

Feature directory layout is documented in `steering/feature_paths.py` and follows the convention `{root}/s{seg_size}/l{layer}/a_{act}/p_{pool}/f{width}/m_{model}/{phase}`.

## 6. Common errors

### `OSError: huggingface_hub.errors.HfHubHTTPError: 401 Client Error`

You need to log in: `huggingface-cli login`, or set `HF_TOKEN` in `.env`.

### `RuntimeError: CUDA out of memory`

Lower `device_batch_size`. Keep `total_batch_size` constant — gradient accumulation will fill the gap. Order of magnitude per GPU:

| Hardware | depth=12 typical `device_batch_size` |
|---|---|
| H100 80GB | 16-32 |
| H200 141GB | 32-64 |
| RTX 4090 24GB | 4-8 |

Encoder-decoder training takes ~2× the memory of decoder-only at the same depth.

### `FileNotFoundError: ...features.../offsets`

Feature shards aren't where the training script expects them. Check `feature_dir=` resolves to a real path. The default auto-resolves from SAE config; pass an explicit `feature_dir=` to override.

### `AssertionError: encoder_strict ... has_raw ^ has_precomputed`

You're feeding an encoder-decoder model both raw feature inputs AND a pre-computed encoder output. Pass exactly one. See `steering/conditional/encoder_decoder/model.py:314-332`.

### `RuntimeError: tokenizer mismatch / vocab_size`

The project uses two tokenizers: Gemma (256k) for feature extraction, RustBPE (32,768) for nanochat training. They are not interchangeable. See [`docs/reference/architecture.md`](../reference/architecture.md#tokenizer-warning).

### `wandb.errors.UsageError: api_key not configured`

Run `wandb login` once, or set `WANDB_API_KEY` in `.env`. To skip wandb entirely, set `run=dummy` in your config or override.

### Hangs at "downloading model" with no output

Hugging Face is rate-limiting or your `HF_TOKEN` is missing/wrong. Check the token has read access to `google/gemma-2-2b` (you may need to accept the license on the HF model page first).

## 7. Where to go next

| Goal | Read |
|---|---|
| See every CLI command | [`docs/cli-reference.md`](../cli-reference.md) |
| Understand the architecture | [`docs/reference/architecture.md`](../reference/architecture.md) |
| Learn the domain vocabulary | [`docs/onboarding/glossary.md`](glossary.md) |
| Do common dev tasks | [`docs/onboarding/workflows.md`](workflows.md) |
| All Hydra config knobs | [`docs/reference/configuration.md`](../reference/configuration.md) |
| File-by-file map | [`docs/reference/file-reference.md`](../reference/file-reference.md) |
| Pipeline deep dives | [`docs/pipelines/`](../pipelines/) |
| Modify nanochat/ | Don't — see [`nanochat/VENDORED.md`](../../nanochat/VENDORED.md) |
