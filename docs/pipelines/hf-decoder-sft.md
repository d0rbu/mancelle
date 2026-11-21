# Hugging Face decoder SFT exports and chaining

The HF backend in `scripts/train/chat_sft.py` supports the pinned Pythia-1.4B,
Pythia-2.8B, and SmolLM3-3B architecture contracts. See
[chat training](chat-training.md)
for training and [the H200 runbook](../performance/hf-decoder-sft-h200.md)
for distributed qualification. A portable `step_NNNNNN/` export is a model
source for a new SFT stage; it is not an optimizer restart checkpoint.

## Source config contract

`HFDecoderContract.accepted_source_token_ids` admits exactly the upstream
or repository SFT BOS/EOS/PAD tuple for each family:

| Family | Upstream BOS / EOS / PAD | SFT export BOS / EOS / PAD | Tied embeddings |
|---|---|---|---|
| Pythia | `0 / 0 / null` | `0 / 0 / 1` | No |
| SmolLM3 | `128000 / 128001 / null` | `128000 / 128012 / 128004` | Yes |

The tuples are indivisible: mixing upstream EOS with SFT PAD (or the reverse)
is rejected. Token IDs must have integer types; booleans, floats, foreign IDs,
and mismatched architectures, dimensions, vocabularies, or embedding ownership
remain errors. SmolLM3 cooldown separately requires the pristine pretraining
token identities.

Pythia has one tokenizer/template/token-ID contract and a model-ID-keyed source
geometry table. `EleutherAI/pythia-1.4b` selects hidden/intermediate dimensions
`2048/8192`, 24 layers, and 16 attention heads;
`EleutherAI/pythia-2.8b` selects `2560/10240`, 32 layers, and 32 heads. Both use
context 2048, parallel residuals, model vocabulary 50304, and tokenizer
vocabulary 50277. Any other Hub Pythia model ID is rejected before tokenizer or
weight loading. A local chained export whose immediate source ID is another
export path is admitted only when its loaded config matches exactly one known
Pythia geometry; an unknown geometry still fails loudly.

The immutable source presets are:

| Config | Model ID | Revision |
|---|---|---|
| `chat_sft_pythia_1_4b` | `EleutherAI/pythia-1.4b` | `fedc38a16eea3bd36a96b906d78d11d2ce18ed79` |
| `chat_sft_pythia_2_8b` | `EleutherAI/pythia-2.8b` | `2a259cdd96a4beb1cdf467512e3904197345f6a9` |

Select `mixture=standard_steered`, provide `steered_jsonl`, and select
`sft_loss_mask=completion_only` for the requested steered instruct run. Those
choices are independent of Pythia size and use the same task and masking code.

## Why the export keeps its instruct token identities

`configure_generation_contract` sets BOS/EOS/PAD in both the model config and
generation config. The tokenizer adapter installs the same identities and chat
template without resizing the vocabulary. Repository HF evaluation requires
these three token identities to agree across config, generation config, and
tokenizer. SmolLM3 also reads config PAD when constructing its embedding's
`padding_idx`. Pythia's causal-LM embedding does not use config PAD, but changing
only its export would violate the existing evaluation contract. Therefore the
exporter remains unchanged; source validation accepts its existing output.

The other config changes are deliberate: training disables `use_cache`;
Transformers serialization records the actual architecture and version;
publication records `dtype=bfloat16` and removes legacy `torch_dtype`.
Neither vocabulary/dimensions nor `tie_word_embeddings` changes. Source
validation constrains architecture and embedding ownership, while allowing
training cache settings and serialized precision/version annotations.

## Existing artifacts and identity

Existing exports require no edits, migration, or re-upload. In particular,
the reported Pythia export at
`checkpoints/sft/pythia-2.8b-standard-steered-mask-20260911/step_001091`
has the admitted SFT PAD value `1`. Its bytes and any content/config digest
remain unchanged. The validator does not normalize or rewrite config objects.

HF ChatCORE hashes the loaded config, and repository harmfulness evaluation
binds the checkpoint file inventory. Neither hash input nor algorithm changes.
The SFT restart identity binds the resolved run config, repository version,
source, tokenizer contract, mixture fingerprints, and topology; this fix does
not change that schema. A run using a new repository commit still has a new
repository identity as designed, so this does not promise cross-commit optimizer
restart compatibility.

## Frozen standard-steered data manifest

The externally produced `pythia-standard-steered-mask-manifest.json` is not
reusable as the strict data identity for a 1.4B run. Its tokenizer block records
the 2.8B Hub model ID and revision, so strict whole-manifest verification binds
that source even though the two sizes have the same tokenizer bytes and chat
template. It does not bind hidden size. Build a new, separately named 1.4B
manifest from the same frozen task inputs and the same measured epoch geometry;
the task hashes and step counts should remain equal, while tokenizer source
identity changes to the pinned 1.4B model and revision.

The builder's `sft_config_name` selects that source: it names the base training
config (`chat_sft_pythia_2_8b`, the default, or `chat_sft_pythia_1_4b`) whose pinned
Hub model and revision are recorded in the manifest. Build and verify with the same
selector, and launch with the matching `standard_steered` preset; the production
launcher derives the selector from `CONFIG_NAME` and passes it to `mode=verify`
(`scripts/slurm/hf_decoder_sft_preflight.sh`), so a 1.4B manifest launched with the
2.8B preset, or vice versa, fails preflight instead of training on the wrong identity.

```bash
.venv/bin/python -m scripts.utils.build_hf_sft_data_manifest \
  output_path=$DATA_MANIFEST sft_loss_mask=completion_only sft_config_name=chat_sft_pythia_1_4b
CONFIG_NAME=chat_sft_pythia_1_4b_standard_steered SFT_LOSS_MASK=completion_only \
  NUM_ITERATIONS=<epoch.num_iterations> ... sbatch scripts/slurm/hf_decoder_sft_h200_1x4.sbatch
```

## ChatCORE on a repository export

The HF path in `scripts/eval/chat_eval.py` accepts one concrete local repository
export without conversion. The directory must be an immutable
`step_NNNNNN/` export with valid `sft_metadata.json`, safetensors weights, saved
tokenizer/template, and matching generation config. Local exports require
`hf_revision=null` and BF16; Pythia source geometry is selected from the source
model ID recorded in export metadata.

```bash
.venv/bin/python -m scripts.eval.chat_eval \
  hf_path=/absolute/path/to/step_001102 \
  hf_revision=null \
  output_dir=out/chatcore/pythia-1.4b-sft
```

At the audited main revision `fbb59c4e1`, there is no separate HF SFT stage
manifest builder. The next-stage gate is `load_hf_config_and_tokenizer`, which
reuses `validate_hf_source_config`. Regression tests exercise actual tiny-model
exports and that loader through two stages for every registered family, keeping
the real vocabulary/token IDs and reducing only architecture dimensions.

## Pythia-2.8B standard_steered on Engaging (one node, four H200s)

Full-sequence loss, SDPA, checkpointing, LR `2e-5`, and AdamW-only `dist_muon_adamw`; sequence length
is 2048; global batch is 524288 tokens; micro-batch 1 gives 64 accumulation steps.
HF distributed SFT rejects `num_iterations=-1`. The CPU manifest builder replays
Pythia's pinned tokenizer/template and the actual best-fit packer for each rank,
then sets `num_iterations=max(ceil(rank_micro_batches/64))` at first-epoch consumption. Packing lookahead, rank imbalance, and final-update rounding can
retain/repeat boundary conversations; this is a measured dataset horizon, not
an exactly-once sampler. The 2026-09-11 snapshot gives `[1099,1102,1100,1100]`,
so the preset pins **1102 steps / 577765376 tokens**, not the encoder-decoder's 1116 steps.

```bash
# Local pythia-sft worktree, after committing; transfer without pushing.
git bundle create /tmp/pythia-sft.bundle origin/main..HEAD
scp /tmp/pythia-sft.bundle engaging:/tmp/pythia-sft.bundle
ssh engaging
# Engaging: stage this bundle's exact branch head in a new worktree.
export W=/orcd/data/dylanhm/001/nano-gpt-pretrain-steer/claude-sft3b-steered-20260905
cd /home/henryac/claude-turnseg-val-20260904/repo
git fetch origin main
git fetch /tmp/pythia-sft.bundle HEAD
git worktree add -b feat/pythia-sft-standard-steered "$W/pythia-sft" FETCH_HEAD
cd "$W/pythia-sft"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
uv sync --locked                         # creates $W/pythia-sft/.venv
export NANOCHAT_BASE_DIR="$W/nanochat_cache"
export HF_HOME=/home/henryac/.cache/huggingface
export STEERED_JSONL="$W/data/messages_steered.jsonl"
# Pre-download on a networked login node BEFORE switching to offline execution.
uv run --no-sync hf download EleutherAI/pythia-2.8b --revision 2a259cdd96a4beb1cdf467512e3904197345f6a9 --include '*.json' '*.safetensors'
# Builds/hashes actual task bodies, including validation and identity twice.
export DATA_MANIFEST="$W/pythia-standard-steered-manifest.json"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
uv run --no-sync python -m scripts.utils.build_hf_sft_data_manifest "output_path=$DATA_MANIFEST"
export NUM_ITERATIONS="$(.venv/bin/python -c 'import json,os; print(json.load(open(os.environ["DATA_MANIFEST"]))["epoch"]["num_iterations"])')"
export EXPECTED_GIT_REVISION="$(git rev-parse HEAD)"
export EXPECTED_UV_LOCK_SHA256="$(sha256sum uv.lock | cut -d' ' -f1)"
export DATA_MANIFEST_SHA256="$(sha256sum "$DATA_MANIFEST" | cut -d' ' -f1)"
export OUTPUT_DIR="$W/pythia-standard-steered-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTPUT_DIR/logs"                # only logs may pre-exist for a fresh run
export SBATCH_OUTPUT="$OUTPUT_DIR/logs/slurm_%j.out" SBATCH_ERROR="$OUTPUT_DIR/logs/slurm_%j.err"
sbatch -p mit_normal_gpu -A mit_amf_advanced_gpu -q mit_amf_advanced_gpu scripts/slurm/hf_decoder_sft_h200_1x4.sbatch
# Restart: retain every variable/path/config above; choose an existing complete marker.
ls "$OUTPUT_DIR"/restart/complete_*.json
export RESTART_FROM_STEP=500               # only if complete_000500.json exists
sbatch -p mit_normal_gpu -A mit_amf_advanced_gpu -q mit_amf_advanced_gpu scripts/slurm/hf_decoder_sft_h200_1x4.sbatch
```

Identity was verified present on 2026-09-11; if missing, the upstream producer is
`curl -fL -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl`.
The old venv lock matches main; use the fresh worktree-local sync to certify installed state. The trainer ignores `sft_data/standard_steered`
parquet; stage SmolTalk/MMLU/GSM8K HF caches by running the builder online first
if offline loading fails. The builder writes `.json.sha256`; verification rejects
content/order/geometry drift. Rebuild to a new manifest after any data change.
Exports land in `$OUTPUT_DIR/step_XXXXXX/` (BF16, model-only); restart state is in
`$OUTPUT_DIR/restart/`. Eval is every 150 steps; exports/restarts every 500 and at
completion. For completion-only, build a new manifest with `sft_loss_mask=completion_only`
and export `SFT_LOSS_MASK=completion_only` for this same launcher.
