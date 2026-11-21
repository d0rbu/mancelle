# Self-conditioned ChatCORE

`chat_eval_self_conditioned` evaluates fixed or turn encoder-decoder checkpoints
using real features from the independently rendered completion prefix and the
model's FP head thereafter. It requires one process, batch size 1, one sample,
and temperature 0. Categorical tasks score the assistant marker's scalar KV
prefill logits under S0; generative tasks use the scalar calculator driver.
Reference answers and full-conversation cache rows never enter inference.

The checkpoint decides the family. Turn K comes only from
`turn_self_cond_segment_size_rustbpe`; fixed K comes only from
`fixed_self_cond_segment_size_rustbpe` and its measured provenance. No CLI
cadence override is accepted. `self_conditioned.fixed_segment_size_assertion` optionally
checks equality for a fixed checkpoint. These cadences approximate source
segmentation in RustBPE input positions; they do not establish that a raw SFT
checkpoint trained with predicted conditioning.

## Numerical behavior

Selection uses the shared AxBench pure-model endpoint: checkpoint positive and
negative count distributions, delta-free positive selection, MAX_POOL over FP
rows, float32 logit ranking at interpolation weight zero, and disjoint positive
and negative features in buckets 0 and 1. Compact FP rows map back to original
SAE IDs through the exact checkpoint vocabulary.

S0 pools the final observed prompt source slot. Fixed generation retains the
K-input boundary replay; turn generation counts the assistant marker once and
commits each later input once. Calculator output is segmented from the causal
prefix plus known forced output using the source tokenizer, including Gemma
`><` straddles. A partial assistant window closes before tool input, and a
completed tool window supplies the next slot. Forced tokens consume budget;
terminal tokens and the final budgeted token cause no further forward.
Exceptions abort the evaluation without publishing a result.

## Benchmark identity interpolation

`chat_eval_benchmark_steering` extends this same evaluator with one optional
`self_conditioned.benchmark_steering: {targets: PATH, strength: NUMBER}` block.
Strength is continuous from 0 through 6; the existing AxBench interpolation
uses weight `strength / 6`. Zero preserves pure-model FP ranking exactly;
six ranks only the fitted target. Intermediate strengths interpolate the
signed sigmoid FP scores with that target. Positive/negative count sampling,
disjoint selection, buckets, per-example seeds, prompt routing and cadence
remain owned by the existing self-conditioning implementation. The task's
target applies to S0 and subsequent generated feature slots, including the
existing calculator transition. Real prompt cache slots retain their features.

Each target represents complete correct-reference examples of one benchmark
versus **the same fixed background corpus**. It does not distinguish correct
from incorrect answers. The fit reuses AxBench's dense extraction: max pool
each complete source document, excluding source BOS; map each allowed feature
through the checkpoint statistics; max sibling percentiles within each
document; average in float64; subtract the background mean and divide by 100;
gather the exact FP-head vocabulary into CPU float32. No sparse top-k feature
bag, additional scaling, normalization, or threshold is introduced.

The offline builder uses these pinned sources:

| Target | Fit candidates | Reserved calibration candidates |
|---|---|---|
| MMLU | `all/dev` | `all/validation` |
| ARC-Easy, ARC-Challenge | Their respective `train` | Their respective `validation` |
| GSM8K | `main/train`, 80% of deterministic input groups | Remaining 20% |
| HumanEval | MBPP `sanitized/train` | MBPP `sanitized/validation` |
| SpellingBee | First 10,000 training seeds, 80% of word groups | Remaining 20% |

The background is the first 1,024 eligible distinct first exchanges in the
native SmolTalk training order (including an initial system message). Native
task shuffling and conversation rendering are preserved. Multiple-choice
references remain native answer labels, with the question and choices present
in the fitting document. GSM8K retains its calculator markers and reference
outputs. MBPP fits only its prompt and code after the supplied reference tests
pass; assertions and test imports are excluded from fitting text.

Before caps, the builder excludes normalized input overlaps with **all 19,329
evaluation examples** across the six tasks. It groups by question stem or
SpellingBee word, reserves calibration groups before fit selection, rejects
conflicting references for identical rendered inputs, and excludes duplicate
inputs. The background additionally excludes embedded candidate/evaluation
inputs and reviewed coding exclusions in either side of the first exchange.
Single SpellingBee words are not a global prose blacklist. Coding overlap
review is recorded explicitly in `coding_exclusions`; exact normalization is
not a guarantee against every semantic paraphrase. Review that list when
changing the pinned coding sources. The full HumanEval test split remains
intact. Supplied dataset references are trusted except for MBPP's executable
qualification; this is not an independent correctness audit of every annotation.

Complete conversations are rendered without the tokenizer's usual length cap,
then rejected if their source-token count plus BOS exceeds the authenticated
statistics context window. Documents are never silently truncated. Up to
1,024 eligible positive documents per task are retained after exclusions and
length checks. Smaller native splits use all eligible examples. Reserved
calibration texts are written separately and never fitted; this command does
not tune strength or add a calibration-scoring runner.

```bash
# Inspect counts, exclusions, references and complete fitting text first.
uv run python -m scripts.eval.build_chat_benchmark_targets \
  checkpoint_meta_path=/path/to/meta_001116.json \
  stats_path=/path/to/exact/training/feature_statistics.safetensors \
  output_dir=/path/to/benchmark-targets

# Same configuration and source checkout: fit with the existing Gemma/SAE stack.
uv run python -m scripts.eval.build_chat_benchmark_targets \
  checkpoint_meta_path=/path/to/meta_001116.json \
  stats_path=/path/to/exact/training/feature_statistics.safetensors \
  output_dir=/path/to/benchmark-targets dry_run=false

uv run python -m scripts.eval.chat_eval --config-name=chat_eval_benchmark_steering \
  model_tag=YOUR_TAG step=YOUR_STEP model_max_seq_len=2048 \
  stats_path=/path/to/exact/training/feature_statistics.safetensors \
  self_conditioned.cache_dir=/path/to/prefix-cache \
  self_conditioned.benchmark_steering.targets=/path/to/benchmark-targets/targets.safetensors \
  self_conditioned.benchmark_steering.strength=2.5 \
  self_conditioned_output_dir=/path/to/results/strength-2.5
```

Use the checkpoint family's configuration assertions from the runbooks below.
The self-conditioned path derives feature identity from the checkpoint.
`NANOCHAT_BASE_DIR` must identify the
matching decoder tokenizer and checkpoint tree for preparation and evaluation.
Dry runs write immutable `preparation.json` and `corpus.json`; fitting adds
`targets.safetensors`. Reusing identical artifacts is allowed; changed bytes
require a fresh output directory. The builder checks the actual dataset cache
revision, including offline fallback, and detects source-code or spelling-asset
changes during preparation. Extraction source-weight revision is explicitly
unrecorded by the existing loader; the artifact records its resolved SAE,
backend, dtype, tokenizers, statistics and source-code identity.

Evaluation snapshots and validates the entire bundle before any forward,
including six exact head-row tensors and the checkpoint's statistics,
source geometry, mask, clusters and vocabulary. It compares requested prompts
at their original indices with the full exclusion population. A changed prompt
or task size fails before evaluation. Receipt v3 adds the full bundle SHA-256,
strength and effective ranking policy; the existing merger therefore rejects
mixed strengths or bundles. All task shards must use the same six-target file.
The descriptive preparation/provenance sections are retained and hashed;
runtime admission does not re-execute the offline data audit.

The default extraction batch is 1, qualified on a 24 GB RTX 4090. Complete
background exchanges can require more memory than AxBench's shorter concept
responses. Larger devices can override `extraction_batch_size` in **both**
preparation and fitting commands. The batch size is recorded, and fitting has
no automatic OOM retry or truncation.

GPU qualification is opt-in through
`tests/steering/test_chat_benchmark_targets_gpu.py`. Stage an untouched baseline
checkout (for example `fd45aa81daf1af51a81bf94cad9fc7ddc561c6f6`), the real
checkpoint, its authenticated prefix cache, statistics, and a fitted bundle.
Write an operator YAML with `baseline_tree`, `builder` (the three fields
`checkpoint_meta_path`, `stats_path`, `output_dir`) and `chat_eval` overrides
(`model_tag`, `step`, `model_max_seq_len`, `stats_path`, and the
`self_conditioned.cache_dir` / `self_conditioned.benchmark_steering` block).
Then run:

```bash
NANOCHAT_BASE_DIR=/runtime \
CHATCORE_BENCHMARK_GPU_CONFIG=/path/to/qualification.yaml \
CHATCORE_BENCHMARK_GPU_EVIDENCE=/path/to/qualification-results.json \
uv run pytest tests/steering/test_chat_benchmark_targets_gpu.py -m slow
```

The tests require a model with at least 2.5 billion parameters and the actual
65,536-feature source dictionary. They compare all six task shapes at strengths
0, 2.5 and 6; compare 500 MMLU examples with the untouched evaluator; and refit
the complete prepared corpus against the untouched AxBench numerical code and
the published tensors. Every generated slot checks the task target, strength,
original feature IDs and bucket assignments; strength 6 also checks both
selection orders against an independent Python oracle. Captured timings include trace hashing and, for the
first pass, checkpoint loading; they are not production throughput estimates.
Run each checkpoint family with its own qualified artifacts and operator YAML.

## Turn checkpoint runbook

Use an authenticated schema-5 prefix cache prepared with the checkpoint
lineage through the [turn cache runbook](evaluation.md#turn-delimited-target-real-context-caches).
The checkpoint must record `turn_self_cond_segment_size_rustbpe`; the CLI
resolves it without a numerical override. For example:

```bash
uv run python -m scripts.eval.chat_eval --config-name=chat_eval_self_conditioned \
  source=encdec model_tag=YOUR_TAG step=YOUR_STEP model_max_seq_len=2048 \
  task_name=GSM8K problem_start=0 max_problems=500 \
  stats_path=/local/path/to/feature_statistics.safetensors \
  self_conditioned.cache_dir=/path/to/cache self_conditioned.seed=1234 \
  self_conditioned_output_dir=/path/to/results/GSM8K
```

For relocated runs (for example, an Engaging checkpoint evaluated locally),
pass `stats_path` to cache conversion, validation, and evaluation. Otherwise the
consumer opens `statistics_path` recorded in the online-SAE lineage. The override
must contain the exact training statistics bytes: admission verifies their raw
SHA-256 against the lineage before any forward. Moving the file does not change
the turn contract or require rebuilding an already authenticated cache.

Schema 5 binds the mask, canonical clusters, cluster descriptor, and feature
vocabulary through `turn_contract`. The consumer compares that entire semantic
payload with the checkpoint-derived contract, alongside cache identity and
provenance checks. Turn caches do not require the fixed-cache
`feature_mask_canonical` metadata field; use the production writer's output
without adding fixed-cache metadata.

Ranges are exact half-open requests. An out-of-range request, missing prefix,
or overlength prompt aborts before any forward. Different jobs may evaluate
disjoint task/problem ranges, but each process is scalar. Seeds derive from the
base seed, canonical task, original example index, exact prompt tokens and
sample index, independently for token sampling and feature selection.

## Fixed checkpoint runbook

First record fixed cadence in a fresh evaluation leaf, using the actual packed
TRAIN source, paired parquet and decoder tokenizer. The source checkpoint and
model bytes stay unchanged; the leaf links the original weights. The estimator
authenticates the inventories, evenly samples up to `max_rows` packed rows,
reconstructs complete source windows, excludes clipped/prediction-only windows
and the held-out shard, and records the exact rational mean and ties-to-even
rounded K. Admission and publication readback require the scalar, measurement
cadence and rounded rational mean to agree, validate version/TRAIN/source width
and a positive window count, and match the original model digest to the loaded weights.
Its measured RustBPE count is distinct from the Gemma source width.
Offline preparation also accepts raw pretraining metadata: feature width comes
from the recorded SAE and bucket count from `user_config` when their exported
top-level fields are absent.

```bash
uv run python -m scripts.utils.record_fixed_cadence \
  checkpoint_meta_path=/path/to/original/meta_250000.json \
  feature_dir=/path/to/training/features parquet_dir=/path/to/paired/parquet \
  tokenizer_dir=/path/to/decoder/tokenizer max_rows=4096 \
  output_dir=/runtime/conditional_encdec_checkpoints/measured_base
```

Fixed preparation is a separate CPU/GPU/CPU sequence. The utility derives SAE,
statistics, mask, cluster, vocabulary and selection authority from the
measured checkpoint copy. `plan` needs source and decoder tokenizers, but no GPU.
It writes `requested.jsonl`, `missing.jsonl`, `extraction.jsonl`, extraction
parquet, `extract.sh`, and an authenticated `plan.json` under
`<output_dir>.prepare` . It reports requested and missing
counts separately: the production 500/500/500/500/164/256 population has 2,420
prefixes and the old six-by-100 cache lacks 1,820.

```bash
uv run python -m scripts.eval.prepare_chat_self_conditioned_cache \
  stage=plan checkpoint_meta_path=/path/to/measured/meta_250000.json \
  legacy_cache_dir=/path/to/schema3/cache \
  legacy_contexts_file=/path/to/legacy.jsonl \
  legacy_feature_dir=/path/to/legacy/extracted \
  legacy_parquet_dir=/path/to/legacy/parquet \
  output_dir=/path/to/core_chat_self_conditioned_v4 chat_max_problems=500
# On a GPU, with the same checkout and environment:
bash /path/to/core_chat_self_conditioned_v4.prepare/extract.sh
# Repeat the first command with stage=publish and identical remaining inputs.
```

Legacy inputs are optional. With authenticated legacy JSONL, parquet and raw
features, selection replays **every** old context in its original enumeration
order before the missing prefixes. Identical selected payloads are preserved;
changed payload hashes are recorded explicitly. Raw shards must carry producer
metadata for the original dictionary width, L0 tier, layer, model and resolved
SAE release/ID, plus a digest of the actual ordered extraction documents. The
converter matches each document slice against the planned enumeration before
feature selection. Missing or incompatible identity requires rebuilding all
requested prefixes, with the reason recorded in the plan. Historical shards are
never relabeled with inferred provenance.
Publication uses the existing schema-4 writer in a temporary root, validates
coverage/routing and input stability, then installs a fresh cache root. It
never edits the old cache or upgrades just a manifest integer. An existing authenticated
schema-5 turn cache needs no fixed-cache conversion.

After publication, point `NANOCHAT_BASE_DIR` at the runtime root containing the
measured leaf and matching tokenizer, then run:

```bash
NANOCHAT_BASE_DIR=/runtime uv run python -m scripts.eval.chat_eval \
  --config-name=chat_eval_self_conditioned source=encdec \
  model_tag=measured_base step=250000 model_max_seq_len=2048 \
  task_name=GSM8K problem_start=0 max_problems=500 \
  self_conditioned.cache_dir=/path/to/core_chat_self_conditioned_v4 \
  self_conditioned.seed=1234 self_conditioned_output_dir=/results/fixed/GSM8K
```

`next_segment_conditioning_offset=auto` reads checkpoint timing; an explicit
value is an equality assertion. This path requires current-segment real prompt
routing. An explicitly blank output directory fails before loading.

## Receipts and merges

Results live at `<self_conditioned_output_dir>/self_conditioned/result.yaml`
(or `<checkpoint_dir>/chat_eval/self_conditioned/result.yaml`). Receipt version
3 binds checkpoint/model bytes, the effective decoder/encoder/cross-attention
architecture digest and `COMPUTE_DTYPE` (in addition to requested `dtype`), recorded K and measurement provenance, source
geometry, tokenizer, the effective two-sided FP interpolation policy, canonical count
distributions, mask/cluster/vocabulary hashes, seeds,
loaded cache manifest/shards/population/preparation, effective generation and
code identity. Per-task ranges and integer counts are outside shared identity.
An existing result file is not overwritten.

The historical merger name also accepts the new result kind:

```bash
uv run python -m scripts.eval.merge_chat_eval_target_real \
  'input_files=[/results/gsm0/self_conditioned/result.yaml,/results/gsm1/self_conditioned/result.yaml]' \
  'task_totals={GSM8K:500}' output_file=/results/merged/self_conditioned/result.yaml
```

`task_totals` declares the exact `[0,N)` union for each task. The merger rejects
mixed kinds/versions, changed identity, missing or overlapping ranges,
inconsistent task sizes, and accuracy/count disagreement. It adds integer
numerators/denominators and recomputes the metric; it never averages shard
accuracies. Target-real receipt versions 1 and 2 keep their existing contract.

## Cadence output compatibility

The ongoing `eb5aec39a` measurement job's `FIXED_LEAF/meta_250000.json` remains
readable without conversion or rerunning the job. The reader accepts
`fixed_self_cond_segment_size_rustbpe` with `fixed_self_cond_cadence_provenance`,
version `packed_train_fixed_cadence_v1`, TRAIN split, source width 128, the
measurement's integer cadence and rational mean (`numerator`/`denominator`),
a positive window count, and the original model SHA-256. Cache preparation
also uses the recorded tokenizer checksum and statistics inventory. Additional
census, sample-policy and inventory fields from that writer remain accepted.
Compatibility was exercised on an artifact emitted by the archived `eb5aec39a`
CPU fixture; the running Engaging job was not accessed or modified.

## Verification and limits

The retained AxBench config/RNG goldens and the fixed/turn tiny-model ChatCORE
goldens check deterministic CPU behavior. The latter record all six tasks'
predictions, accuracies, decoder-input/slot/logit hashes and selected-feature
hashes captured from `eb5aec39a` with the repository's CPU encoder reference.
They do not establish CUDA numerical equality or 3B model quality.

The operator's post-shrink equality check remains pending: rerun MMLU (500) and
one GSM8K shard on the RTX 4090, with identical checkpoint/cache bytes, indices
and settings; compare accuracy and every per-problem prediction with the
`eb5aec39a` run. Any difference requires investigation and explicit justification.
Peak memory and throughput are not measured by the CPU suite.
