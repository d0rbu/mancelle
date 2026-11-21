# Campaign 2: instruct → safety → capabilities

Authoring only: no GPU qualification, cluster contact, submission, or paid judging was performed.
Start from the **completion-only standard_steered SFT** checkpoints, never an RL/selfcond export.
Both families retain all intermediate (every 500 steps) and final checkpoints; each stage is a fresh SFT epoch.

## Resource plan and critical path

| Pool | Before stage 2 | During SFT | After checkpoints finish |
|---|---|---|---|
| Priority 4×H200 | Four one-GPU safety extraction workers | Encdec safety → capabilities, four GPUs/job | Independent one-GPU eval cells |
| Preemptable 4×H200 | Capabilities extraction while Pythia instruct finishes | Pythia safety → capabilities, four GPUs/job | One-GPU DiffMean fits/evals |
| Local 4090 | Extraction overflow if its mixture is not assigned to H200 | Pythia instruct eval, then other available eval cells | Remaining evals, one cell at a time |
| CPU partition/local CPU | Both conversions + HF manifest/epoch replays in parallel | Statistics/validation, cadence views | Paid judges only after separate authorization |

Do not allocate extraction and four-GPU SFT simultaneously on the same four-GPU lane.
Stage 3 data can be prepared before stage 2 finishes; models within each family are sequential.
Critical paths: encdec instruct + safety artifacts → safety SFT → capabilities SFT;
Pythia instruct export + safety manifest → safety SFT → capabilities SFT. Families are independent.
Run each instruct/safety evaluation as soon as its checkpoint is complete; it need not await capabilities.
`eval_matrix.sh` submits three baseline cells, or nine safety strength cells, when all three checkpoints exist;
direct `eval.sbatch` supports earlier individual cells.
Use `afterok` dependencies. An interrupted job fails its successors; resume/re-submit that stage, then replace successors.
Do not submit the same stage on both lanes: filesystem locks reject duplicate writers.

## Dataset sizes and provisional budgets

The converter and both trainers use **chat_sft.yaml defaults**. In particular, capabilities uses 200k/80k spelling,
not the 300k/120k overrides of the separate `chat_sft_capabilities.yaml` preset.
Offline CPU inspection of the locally cached task sources on 2026-09-11 gave:
WildJailbreak 261,559; response-bearing WildGuardMix 37,934; CoCoNot 11,476;
MMLU auxiliary 99,842; OrcaMath train 199,035; GSM-only MetaMath train 239,000.
Let `I` be identity JSONL rows and `N` be normalized NemotronSafety rows (Nemotron is not cached locally).

| Mixture | Config composition / conversations | Approximate s32 feature bytes |
|---|---|---|
| Instruct reference | standard_steered ≈1.2M, owner-supplied | 39 GB |
| Safety | 2×261559 + 3×37934 + 2×11476 + 120000 + N + 2I = **779872 + N + 2I** | ≈25.3 GB + 32.5 GB per extra million rows |
| Capabilities | 120000 + 50000 + 2×99842 + 199035 + 239000 + 200000 + 80000 + 2I = **1087719 + 2I** | ≈35.4 GB |

These are cached-source counts, not a claim about the staged campaign snapshot. Conversion may drop invalid rows.
The actual CPU manifests include every normalized task body, fingerprint, and four-rank packed epoch measurement.
Feature sizes scale only approximately with conversation count; lengths and activation sparsity can change them substantially.

| Operation, per mixture/cell | Rough GPU-hours for planning (unmeasured assumptions) |
|---|---|
| Convert, manifest replay, stats, cadence view | 0 GPU-h; provision 1–12 CPU wall-hours each |
| H200 extraction | 6–24 GPU-h per million conversations; safety ×(0.780+N/1M), capabilities ×1.088 |
| 4090 extraction, conservative batch | 35–140 GPU-h per million; use overflow, not the critical safety path |
| Encdec SFT, either stage | 9–28 GPU-h per 1000 optimizer steps (assume 8–25 s/step across four H200s) |
| Pythia SFT, either stage | 7–23 GPU-h per 1000 steps (assume 6–20 s/step across four H200s) |
| Harmfulness, one checkpoint/variant | 0.5–4 H200 GPU-h or 1–8 4090 GPU-h per 1200 prompts, ≤512 generated tokens |
| Pythia DiffMean fit | 0.1–1 GPU-h/checkpoint for a bounded 128-pair fit, plus generation above |

Use measured manifest `epoch.num_iterations` for HF; encdec ends at its committed dataset boundary.
For a provisional 800–1600 steps/stage, budget encdec 7–45 and Pythia 6–37 GPU-h/stage.
Budgets exclude queueing, preemption/replay, downloads, shared-storage reads, and paid judge latency.
Replace these broad provisioning ranges with first-shard/first-50-step timings before reserving the full campaign.

## Launch contracts

From the clean campaign checkout, export absolute `CAMPAIGN_ROOT`, `NANOCHAT_BASE_DIR`, `HF_HOME`,
`EXPECTED_GIT_REVISION=$(git rev-parse HEAD)`, and `CAMPAIGN_REPO=$PWD`.
Pre-stage root `.venv`, the separate `vendor/axbench` environment (`uv sync --project vendor/axbench --frozen`),
Gemma/SAE caches, RustBPE tokenizer, identity JSONL, spelling words, and **all** mixture datasets including Nemotron.
Every launcher is offline. Run `sbatch` from this checkout (Slurm copies scripts; `SLURM_SUBMIT_DIR` selects the code).
No paths to an operator's old HOME checkout are embedded. Pass site-specific preemptable account/QoS via `SBATCH_*` if needed.

```bash
MIXTURE=safety bash scripts/slurm/campaign2/submit_data.sh priority
MIXTURE=capabilities bash scripts/slurm/campaign2/submit_data.sh preemptable
# Local alternative: submit cpu.sbatch with --export=ALL,CAMPAIGN_CPU_TASK=convert, then
# run extract_local.sh, then submit cpu.sbatch with --export=ALL,CAMPAIGN_CPU_TASK=stats.
# Export MIXTURE first. Never run both local and Slurm extraction on the same mixture concurrently.
```

Submit `cpu.sbatch` directly with `--export=ALL,CAMPAIGN_CPU_TASK=convert`, `stats`,
or `manifest` and the CPU resources shown in `submit_data.sh` / `submit_pythia.sh`.

Extraction is Gemma-3-4b-pt L22/65k, BF16, context-independent/no-BOS, turn-delimited s32, context window 4096.
H200 batches: 248 segments/128 CPU documents. 4090 batches: **2 segments/8 documents**; expect ≥64 GB host RAM.
**Expected to fit 24 GB**, following the existing preset's CPU-offloaded TransformerLens companion and BF16 model/SAE.
Weights are roughly 8 GB + <1 GB SAE, with short isolated forwards; allocator/runtime peaks remain unqualified without a GPU test.
Statistics run on CPU over every training shard (`exclude_last_shard=false`); train/val validators are mandatory.

For encdec export `ENCDEC_INSTRUCT_DIR`, `ENCDEC_INSTRUCT_STEP`, exact `FEATURE_MASK`, `FEATURE_CLUSTERS`,
`NODE_STAGE_ROOT` (writable node-local disk), `SAFETY_STATS_JOB`, `CAPABILITIES_STATS_JOB`, optional `INSTRUCT_JOB`.
Then `bash scripts/slurm/campaign2/submit_encdec.sh priority` queues the two stages.
Each `encdec/<stage>/base` holds only symlinked prior model/meta; its `<mixture>_sft_conditioned_checkpoints` stays shared.
Inputs are checksummed into job-specific local directories with a 20 GiB reserve; staging failure aborts. Reclaim those directories after jobs.
The original s32 recipe's `missing_percentile_policy=lowest` is preserved: these are pilot-policy BPB values, not `error`-policy comparisons.
Resume finds exact model/meta + ranks 0–3; completion publishes `encdec/<stage>/final_step` only after trainer success.

For HF export `PYTHIA_INSTRUCT_EXPORT` (concrete `step_XXXXXX`), `EXPECTED_UV_LOCK_SHA256`, optional `PYTHIA_INSTRUCT_JOB`.
Then `bash scripts/slurm/campaign2/submit_pythia.sh preemptable`; CPU manifests precede each stage.
Local parent paths use the parent's 40-hex source revision (Transformers resolves local files independently of revision).
Stage manifests hash every parent file, verify the tokenizer contract, and bind the frozen task/epoch manifest.
HF restarts require explicit `RESTART_FROM_STEP` backed by a complete restart marker; model-only exports start the next stage.

## Harmfulness and steering

The legacy `training_progression.py`/standalone evaluator reject turn checkpoints; progression also lacks an HF checkpoint interface.
`harmfulness_eval_full3b.yaml` and the historical `progression_chat_results.json` were unavailable in this checkout/scratchpad.
Campaign 2 reuses the canonical BeaverTails/WildGuard prompt pool and judges, plus shared turn-aware AxBench context/self-conditioning code.
Do not merge historical results without matching prompt and generation contracts. Generation is per-prompt resumable; outputs bind code/locks/checkpoint bytes.
For each raw encdec checkpoint, run the CPU `scripts.utils.campaign2_eval_view` with `metadata`, `feature_dir`, `parquet_dir`,
`output_dir`; instruct additionally needs `steered_jsonl`. Submit `cpu.sbatch` directly with
`--export=ALL,CAMPAIGN_CPU_TASK=eval-view` and the eval-view resources (`mit_normal`, advanced CPU account/QoS,
16 CPUs, 128 GB, six hours). It measures RustBPE cadence; it never assumes Gemma s32 means RustBPE 32.
Then use `eval_local.sh` or `eval.sbatch` with `stage`, `variant`, `checkpoint_dir`, `output_dir`, `mode=generate`;
encdec additionally needs `step` and its exact train `statistics`. `mode=preflight` performs CPU checks only.
`eval_matrix.sh` accepts `EVAL_POOL`, `EVAL_VARIANT`, `EVAL_OUTPUT_ROOT`, all three `EVAL_{INSTRUCT,SAFETY,CAPABILITIES}_CHECKPOINT`
and (encdec) corresponding `_STEP`/`_STATISTICS` variables. It preflights every cell before submitting any.
Encdec baseline uses static interpolation weight 0, real turn contexts and measured cadence; Gemma/SAE are released before target loading on 4090.
Pythia `pythia_prompt` uses the configured safety instruction inside the exported chat template and vendored PromptSteering.
`pythia_diffmean` uses **separate training data only**: its frozen `safety.training_pairs`
artifact. Evaluation references are never an input. Supply `safety.training_pairs`
at launch: the JSONL and its sibling `.manifest.json` must travel together.
Select these cells directly with `variant=encdec_safety`, `variant=pythia_diffmean`,
or `variant=pythia_prompt`.

Build the frozen CPU artifact (no model weights or provider calls):

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m steering.eval.campaign2_training_pairs \
  output_dir=/home/d0rb/axbench_sft_local/campaign2_local/training_pairs
# Once the pinned datasets are cached, HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 works too.
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m scripts.eval.campaign2 \
  variant=pythia_diffmean mode=preflight device=cpu \
  stage=instruct checkpoint_dir=/path/to/step_001091 output_dir=/path/to/cell \
  safety.training_pairs=/home/d0rb/axbench_sft_local/campaign2_local/training_pairs/pairs.jsonl
```

The source is `PKU-Alignment/BeaverTails`, default config, **`330k_train`** split
(the full training split, not the smaller `30k_train` subset). For each exact raw
prompt, retain the first `is_safe=True` and first `is_safe=False` response in native
row order. Exclude all BeaverTails-Evaluation test and WildGuardTest prompts using
the shared safety-target/ChatCORE normalization: Unicode NFKC, case fold, collapsed
whitespace. Exclusion precedes pairing eligibility and selection, including unpaired
prompts. Reject malformed source fields; exclude prompts missing either class or
whose first responses are identical after stripping surrounding whitespace. Do not
substitute later responses. Among eligible pairs sharing a normalized prompt, keep
the first raw prompt in native order; never combine responses across raw spellings.
Shuffle that native-order prompt list with Python `random.Random(42).shuffle`, then
take the first `cap` pairs (default and hard maximum **128**). Empty output fails.

Every JSONL row has exactly `prompt`, `safe_response`, `unsafe_response`, all nonblank
strings with distinct responses. The manifest records the selection rule, seed,
cap, revisions/configs/splits and source row counts, exclusion counts, selected row
count and SHA-256 of the exact UTF-8 JSONL bytes. Identical reruns are idempotent;
changed output at the same destination is refused. Preflight checks strict JSONL
schema, normalized uniqueness, 1–128 rows, count consistency and manifest hash. It
reloads both complete pinned evaluation splits, verifies their cache revisions,
and independently checks **zero normalized overlap**, regardless of runtime eval
sampling. Both artifact hashes enter the cell identity.

Vendor DiffMean fits safe-minus-unsafe at layer 25 over rendered full conversations
≤1024 tokens, excluding initial BOS; the worker refuses truncation. It normalizes
the direction and calibrates positive max activation on those training texts,
independently per checkpoint. Factor multiplies that scale. The frozen pairs do
not contain a fitted direction; fitting still runs against each selected checkpoint.

After separately authorizing provider calls, submit `eval.sbatch` directly with
`mode=judge allow_paid_judge=true`, `OPENROUTER_API_KEY`, `CUDA_VISIBLE_DEVICES=''`, and CPU resource overrides
(`--partition=mit_normal --account=mit_amf_advanced_cpu --qos=mit_amf_advanced_cpu --gres=none --cpus-per-task=4 --mem=32G --time=06:00:00`).
Paid judging still requires the explicit gate.
Judging atomically saves each harmfulness and benign-compliance result under
`judge_records/{harmfulness,compliance}/<prompt-index>.json` as it completes.
Records bind the exact response text, prompt, pool/category, original prompt index,
cell identity, judge model, prompt-template hashes, and result schema versions;
a checksum and the canonical score parser validate cached content on resume.
These cell-local records are authoritative: the legacy shared SQLite judge cache
cannot establish this provenance and is not reused by Campaign 2.
Missing, failed, corrupt, or mismatched judgments are requested again; valid
successes are retained. Each model route gets at most three attempts, with exponential
backoff of one then two seconds. Exhausted failures retain their reason, and the
cell fails loudly with the remaining original prompt indexes for each judge.
Rerun the same judge command to resume. `results.json` is emitted only when both
judges have complete coverage, using the unchanged aggregation in prompt order.
Collect judged cells with `scripts.eval.campaign2_progression input_root=... output_file=... 'variants=[encdec,pythia,pythia_prompt]'`.
Set `judge_refusal_fallback_model=x-ai/grok-latest` to route only primary-provider
policy refusals to that secondary model. Transport and parsing failures never
trigger the fallback. Per-pair judge-model assignments and the number of pairs
scored by the fallback are recorded in `results.json`; a fallback refusal or
exhausted fallback parse failure fails the cell without a third model or skip.

### Encoder-decoder safety strengths

`variant=encdec_safety` runs the merged harmfulness
[`self_conditioned.safety_steering` runtime](harmfulness-interpolation.md), explicitly through `generate_encdec`.
Pass `safety.targets=/path/to/targets.safetensors`, finite `safety.strength` in `[0,6]`,
and `conditioning.context_cache_dir=/path/to/cache`.
The judged study uses **0, 3, 6** (interpolation weights 0, 0.5, 1). Strength zero uses the shared raw-logit
ranking endpoint. The existing `variant=encdec` extraction, zero-target policy, generation and cell identity are unchanged.
The Campaign interface intentionally does not expose the underlying target-score interpolation knobs.

Each safety invocation accepts one instruct, safety or capabilities checkpoint and one strength:

```bash
export NANOCHAT_BASE_DIR=/path/to/runtime NANOCHAT_DTYPE=bfloat16
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m scripts.eval.campaign2 \
  variant=encdec_safety mode=preflight stage=instruct \
  checkpoint_dir=/path/to/instruct-eval-view step=1125 \
  statistics=/path/to/instruct/feature_statistics.safetensors \
  safety.targets=/path/to/instruct/targets.safetensors safety.strength=3 \
  conditioning.context_cache_dir=/path/to/instruct/prompt-cache \
  output_dir=/path/to/results/instruct/encdec_safety
# Generate with the same arguments and mode=generate, with CUDA visible.
```

CPU preflight loads and validates the checkpoint weights without forwarding. Provision host memory for the
model loader (the supplied masked checkpoint has about 10 GB of serialized weights). Admission binds:

- Model/metadata bytes, effective architecture and feature prediction, decoder tokenizer files,
  source s32 turn geometry, measured RustBPE cadence, and cadence evidence. A campaign evaluation view can
  use `campaign2_cadence.json`; an existing ordinary-SFT export can use its embedded
  `turn_self_cond_cadence_provenance`. Counts, range and exact mean must agree with the recorded cadence.
- The exact `target.safety` bundle path and SHA-256, signed dense values and original feature-row order,
  pinned WildGuardTrain response provenance, **128** selected positives, and the complete pinned
  BeaverTails-Evaluation/WildGuardTest exclusion population. CPU admission reloads both full evaluation
  sources and checks their pinned revisions and prompt order, even for a small runtime slice.
- Statistics digest, SAE release/site/layer/width/L0, extraction tokenizer/BOS/turn geometry,
  missing-percentile policy, canonical mask/clusters and compact vocabulary, using the merged target loader.
  The existing bundle's explicit unrecorded Gemma weight revision remains a provenance limitation.
- The real schema-5 cache manifest and compressed shard bytes, full requested prompt coverage and generation
  capacity, ordered prompt digest, original prompt indexes and token/selection seeds, code bytes/revision,
  dependency locks, generation settings and precision. No Gemma extraction is run for these safety cells.

`output_dir` is the variant root. The runner appends `strength-0`, `strength-3`, or `strength-6` automatically
(other strengths use an unrounded, round-trippable numeric spelling). Each leaf contains `identity.json`,
`prompts.json`, `cell.lock`, `records/000000.json` onward, `responses.json`, and after judging `results.json`.
Every safety response record embeds a receipt binding cell digest, stable index, prompt seeds and response text.
Resume validates existing records before any new forward; `responses.json` binds the complete ordered responses.
Changed target/checkpoint/cache/code/settings at the same leaf fail loudly; use a new output root for a new contract.
Strengths always occupy separate leaves. Keep `device` and `NANOCHAT_DTYPE` consistent for generation and judging;
judge admission still loads exclusively on CPU with CUDA hidden.

For the full nine-cell matrix, export the five stage-specific inputs below for **each** stage:

| Stage prefix | Required suffixes |
|---|---|
| `EVAL_INSTRUCT_` | `CHECKPOINT`, `STEP`, `STATISTICS`, `TARGETS`, `PROMPT_CACHE` |
| `EVAL_SAFETY_` | `CHECKPOINT`, `STEP`, `STATISTICS`, `TARGETS`, `PROMPT_CACHE` |
| `EVAL_CAPABILITIES_` | `CHECKPOINT`, `STEP`, `STATISTICS`, `TARGETS`, `PROMPT_CACHE` |

```bash
export CAMPAIGN_REPO=$PWD EXPECTED_GIT_REVISION=$(git rev-parse HEAD)
export EVAL_POOL=priority EVAL_VARIANT=encdec_safety EVAL_OUTPUT_ROOT=/path/to/results
export EVAL_SAFETY_STRENGTHS='0 3 6' # default; custom lists must be nonempty and distinct
bash scripts/slurm/campaign2/eval_matrix.sh
```

The launcher preflights **every** stage/strength on CPU before its first submission. It never infers a shared
target, statistics or cache: pass the appropriate artifact for each stage. A target may be reused across
checkpoints only when their complete fit contract and original feature row order agree.
`EVAL_POOL=local` runs cells sequentially; the operator must reserve the shared GPU first. Check compute PIDs,
wait 60 seconds while busy, never terminate another job, and bound local qualification runs below 15 minutes/12 GB.

After separately authorizing provider calls, use the CPU-overridden `eval.sbatch` command described above with
`mode=judge allow_paid_judge=true`, the same cell arguments and strength, and `OPENROUTER_API_KEY`.
Generation and preflight never call a paid judge.
Collect and plot the judged strength lines using the shared harmfulness plotting functions:

```bash
.venv/bin/python -m scripts.eval.campaign2_progression \
  input_root=/path/to/results output_file=/path/to/progression.json \
  'variants=[encdec,encdec_safety]' 'safety_strengths=[0,3,6]' plots_dir=/path/to/figures
```

Collection requires all three stages for every requested variant/strength, identical ordered prompts, valid
response/judge receipts, and matching checkpoint/target/generation contracts across strengths within a stage.
The four plots show harmful ASR, harmful safe-response rate, benign compliance and overall mean harmfulness,
with one line per variant/strength. Omit `plots_dir` to collect JSON only.

For a future bounded GPU qualification, put the campaign checkpoint/step/statistics/targets/cache overrides
in a YAML file and run:

```bash
CAMPAIGN2_SAFETY_GPU_CONFIG=/path/to/smoke.yaml timeout 840 .venv/bin/python -m pytest \
  tests/scripts/test_campaign2_safety_gpu.py -m slow -q -rs
```

This checks all configured prompts on CPU, then generates the first harmful and first benign prompt at
strength 3 with a 40-token budget and their original indexes. It rechecks compute PIDs before GPU transfer,
caps the allocator at 10 GB, verifies finite forward logits and response receipts, and records memory/timing
evidence in pytest's temporary directory. If another process remains after 60 seconds, it skips.
