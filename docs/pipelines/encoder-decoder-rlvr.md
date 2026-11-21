# Encoder-decoder RLVR

This experimental trainer learns feature self-conditioning from verified final
answers on fully generated conversations. It is separate from
[structured-SFT GRPO](sft-self-conditioning-grpo.md), whose reward is detached
teacher-forced next-token loss. The shared model, action likelihood, native
calculator generator, checkpoint admission, and optimizer remain authoritative.

## Supported checkpoint and data

Use a conditioned-SFT-shaped encoder-decoder checkpoint with turn-delimited
features, a recorded generated-assistant RustBPE cadence, final-layer MAX_POOL
feature prediction, two buckets, zero label dropout, recoverable geometric
positive/negative count distributions, and exact mask/cluster/vocabulary
metadata. Fixed-token checkpoints are rejected. The raw SFT checkpoint can be
used directly; an additional self-conditioning training stage is unnecessary.
Missing cadence must be measured from the original SFT artifacts using the
existing [cadence workflow](selfcond-post-training.md), never guessed.

The intended 3B starting point is step **1116** of
`sft3b-steered-turnseg32-20260905`, beneath
`nanochat_cache/conditional_encdec_checkpoints/full3b_clustered_c2048_long_w32_bf16_b14_fp/standard_steered_sft_conditioned_checkpoints/`.
Its model SHA-256 is
`a91b0ffb0fefe6d34c2954f72d57bdbe33f73c306177cfc8e23128c7b9cead27`;
its measured RustBPE cadence is **33**, source Gemma window size **32**, context
length **2048**, and effective feature vocabulary **844** of **65536**.
Bounded real-model head-only qualification on source revision `a617fc1b`
covers one and two H200 GPUs. Separately, bounded full-policy MC-only
qualification completed on one and two 96 GB RTX PRO 6000 Blackwell Server
Edition GPUs with native FA4, including checkpoint execs and standalone
evaluation. See the [SM12 results, frozen source snapshots, and reproducibility limits](../performance/fa4-sm12-rlvr-validation.md).
These qualifications apply to their recorded source snapshots and training
scopes; they do not establish a minimum GPU capacity.

| Task | Training | Validation | Retained train / validation |
|---|---|---|---:|
| ARC-Easy | ARC-Easy/train | ARC-Easy/test | 2241 / 2376 |
| ARC-Challenge | ARC-Challenge/train | ARC-Challenge/test | 1112 / 1172 |
| MMLU | auxiliary_train/train | all/test | 99807 / 14042 |
| GSM8K | main/train | main/test | 7473 / 1319 |

Dataset revisions are fixed in `configs/data/prepare_encdec_rlvr.yaml` and
validated against the resolved Hugging Face dataset. Preparation excludes any
training question matching any heldout question after NFKC, case folding, and
whitespace normalization, before applying optional qualification caps. It
preserves duplicates and source order. These official test splits are used for
model selection; there is no untouched final test set.

The bundle contains exact question-only RustBPE prompts, source identifiers,
minimal answer keys, a manifest, and the existing schema-5 prompt feature cache.
Reference answers and reasoning never enter the conditioning cache. Extraction
uses the immutable starting checkpoint's SAE lineage and statistics; training
encodes the cached raw selected features through the evolving model weights.
The bundle binds both tokenizers, source revisions, rows, verifier semantics,
base checkpoint, statistics, and cache manifest/shards. Resume uses the complete
bundle identity. An RL checkpoint's changed weights do not invalidate its
original prompt cache.

## Preparation

Select the active RustBPE tokenizer with `NANOCHAT_BASE_DIR`. Supply the exact
base model, its metadata, and the locally available copy of its recorded
statistics. Use a fresh output directory.

```bash
uv run python -m scripts.preprocess.prepare_encdec_rlvr \
  stage=prepare output_dir=/absolute/path/to/rlvr-data \
  base_checkpoint_path=/absolute/path/to/model_001116.pt \
  base_checkpoint_sha256=a91b0ffb0fefe6d34c2954f72d57bdbe33f73c306177cfc8e23128c7b9cead27 \
  checkpoint_meta_path=/absolute/path/to/meta_001116.json \
  stats_path=/absolute/path/to/feature_statistics.safetensors
```

This writes `train.jsonl`, `validation.jsonl`, `contexts.jsonl`,
`context_parquet/`, and `preparation.json`. For bounded qualification, add
explicit `train_caps` and `validation_caps` mappings containing positive counts
for all four task names. Full-source exclusion and reference checks still run.

Extract features from `context_parquet/` with the checkpoint's frozen SAE
settings, then use `scripts.eval.build_eval_context_cache`
with `--config-name=build_eval_context_cache_turn stage=convert`. Reuse the
[turn extraction and conversion workflow](evaluation.md#turn-delimited-target-real-context-caches),
substituting the RLVR preparation's contexts/parquet and writing the cache to
`<bundle>/context_cache`. Do not re-enumerate `chat_target` examples: RLVR's
`chat_rlvr` records contain only question prefixes. Pass these three metadata
pins to the existing cache converter:

| Converter override | Value from `preparation.json` |
|---|---|
| `+metadata.rlvr_data_sha256` | `bundle_sha256` |
| `+metadata.rlvr_base_model_sha256` | `base_checkpoint.model_sha256` |
| `+metadata.rlvr_base_metadata_sha256` | `base_checkpoint.metadata_sha256` |

Run the same preparation command with `stage=finalize` after cache conversion.
It checks all prepared artifacts, exact prompt coverage, and producer pins,
then publishes the final `manifest.json`. An unfinalized preparation is never
accepted by the trainer. The final bundle can be relocated without changing
the recorded producer paths; runtime checks compare content identity.

## Reward and feature objective

GSM8K uses the repository's exact `extract_answer` behavior: find the first
`#### ` number and remove commas, then compare strings. For example,
`#### 1,000` equals `#### 1000`, while `#### 10.0` differs from `#### 10`.
Extra whitespace immediately after the marker, a leading plus sign, or a
different final-answer format can yield zero reward. Preparation rejects
unparseable references. This is intentionally evaluator parity, not a general
mathematical-equivalence checker or an LLM judge.

Multiple-choice completions are stripped at their edges and must then be one
of that row's exact labels. Invalid labels, explanations, punctuation, and
lowercase alternatives receive zero. Numeric ARC labels and three/five-choice
rows are supported. Verification sees the entire generated suffix, including
calculator outputs. Explicit invalid structural generation terminates with
zero reward; integrity errors and nonfinite computations abort the run.
Stray or nested calculator delimiters are invalid. Ending inside a Python block
or truncating a forced calculator response also receives zero reward, while
retaining the native tokens and EOS/budget stop reason. Generation identity v2
records these structural checks; evaluation rejects earlier v1 RLVR checkpoints.

Each training prompt is sampled uniformly from `tasks` and then uniformly by
row within that task, with replacement. The default ordered list is
`[ARC-Easy, ARC-Challenge, MMLU, GSM8K]`, which preserves the original sampled
rows exactly for the same seed, step, and global prompt ordinal. `tasks` must
be a nonempty list of distinct exact task names; order affects sampling and is
pinned for resume. Each group generates its own tokens and positive/negative
feature actions. Token, count, and feature-row RNG streams are separate and
stable by step, global prompt ordinal, source question, rollout, and decision.
Defaults are temperature 1, no top-k, 16 output tokens for multiple choice, and
512 for GSM8K. Calculator output tokens consume this same budget. Questions
are never truncated or silently replaced.

For each prompt, advantages use the population standard deviation of its
binary group rewards. The loss is

```text
- mean_prompt mean_rollout [ advantage * mean_live_decision exp(new_logp - old_logp) ]
```

The likelihood is the exact ordered positive-then-negative feature sampling
likelihood without replacement. Only causally live decisions enter the mean:
the feature slot must have been committed before logits used by a real sampled
token draw. EOS counts as a real draw; dummy draws for forced calculator tokens
do not. Unused trailing actions remain in the transcript for faithful replay.

By default, all connected encoder, decoder, cross-attention, and feature-head
parameters receive this feature-policy gradient. The untied LM head is frozen.
There is no next-token loss or token log-probability gradient, KL penalty, entropy bonus,
critic, PPO epoch, or replay buffer. Because decoder parameters also affect
token generation, this is deliberately a partial-gradient, length-normalized
surrogate; it is not the full episodic token-and-feature policy gradient.

Set `train_feature_head_only=true` to train only the complete feature-prediction
head, including any nested residual blocks. Encoder, decoder, embeddings,
cross-attention, residual scalars, and the LM head remain frozen and receive
neither gradients nor optimizer state. Replay runs the encoder, decoder, and
head-input normalization under `no_grad`, matching collection's gradient mode.
It then applies the complete head with autograd enabled. This matters for
compiled kernels whose outputs can differ by gradient mode even when every
trunk parameter is frozen. Head likelihood gradients flow through every head layer.
This narrows the same partial-gradient surrogate to the head. It does not add a
token-likelihood objective or differentiate through sampled feature IDs.

Feature action collection and backward replay use the same batch-one,
full-prefix, no-KV forward. Raw prompt encoder shapes/order are preserved;
previous sampled segments are encoded separately and appended. Full-policy
replay differentiates through that encoding; head-only replay keeps it frozen.
The current action and future tokens never enter that action's state. Cached
KV decoding supplies token logits only. Reentrant activation checkpointing
limits full-policy replay memory. During its gradient-enabled recomputation,
each history encoder call also uses a nonreentrant checkpoint. Backward
recomputes its intermediates, including BF16 weight casts, so those model-sized
casts do not remain live for every history segment at once. This adds one
encoder recomputation per history segment while preserving each call's shape,
order, and compiler context. Initial prompt encoding runs directly;
no-gradient action collection does not use these inner checkpoints. Head-only
replay runs the trunk under `no_grad` and skips both levels of checkpointing.
Both modes share identical S0 policy probes across a group. See the
[3B memory measurements](../performance/fa4-sm12-rlvr-validation.md#history-encoder-memory)
for the measured reduction and qualification limits.
Recorded likelihood terms must replay within the existing `2e-5` tolerances
before the optimizer is allowed to run. Generic self-conditioned ChatCORE
evaluation uses a different cached feature policy; use the RLVR evaluator for
this trained policy.

Head-only mode reduces optimizer state and removes trunk backward work and
checkpoint recomputation. It still runs the trunk for generation and canonical
policy probes and retains the head's hidden states for each live prefix. Actual memory
and step-time savings depend on rollout lengths, head size, and hardware.

## Training, validation, and recovery

```bash
NCCL_TIMEOUT_SEC=3600 uv run torchrun --standalone --nproc_per_node=1 \
  -m scripts.train.conditional_encdec_rlvr -- \
  base_checkpoint_dir=/absolute/path/to/sft-checkpoint model_step=1116 \
  bundle_dir=/absolute/path/to/rlvr-data \
  stats_path=/absolute/path/to/feature_statistics.safetensors \
  output_dir=/absolute/path/to/rlvr-run device_type=cuda \
  global_prompts_per_step=4 group_size=4 num_iterations=100
```

These batch sizes are illustrative, not a qualified performance preset. Even a
single-rank training run uses torchrun so DistMuonAdamW owns averaging. Each
rank accumulates its local prompt mean, clips locally, and the optimizer
averages exactly once. This preserves the existing optimizer convention;
clipped single-rank and distributed updates need not be equal. All-zero
advantages still perform a normal optimizer step, so existing momentum can
move weights.

The RLVR default is `train_feature_head_only=false`; append
`train_feature_head_only=true` to the command to opt in. Startup logs report the
resolved scope and trainable/total parameter counts. Clipping and optimizer-owned
averaging include only trainable parameters in either mode.

For multiple-choice-only training, append this quoted Hydra override:

```bash
'tasks=[ARC-Easy,ARC-Challenge,MMLU]'
```

This changes training selection only. Keep the complete four-task bundle and
prompt cache: admission still verifies every task and every prompt's token
budget, and both in-training and standalone validation still evaluate all four
tasks. Gemma's source tokenizer must also be available in the active Hugging
Face cache (`HF_HOME` when relocated). Excluded tasks retain a zero training
count in telemetry. Prompt text, native answer formats, token budgets, and
cache identity are unchanged. To change task membership or order, start a fresh
run from the original SFT checkpoint with a new output directory; existing RLVR
checkpoints cannot be resumed with a different task list.

Use an explicit `NCCL_TIMEOUT_SEC` for distributed runs. Independent task draws
can give one rank long GSM8K rollouts while another finishes multiple-choice
groups early; the latter waits at the next synchronized stage. The example
allows one hour. Size this timeout to the planned group and prompt counts.

Validation runs at startup, every 10 completed steps, and the final step using
one fixed seeded panel of 32 rows per task, including tasks excluded from
training. At startup, rank zero warns for each task whose panel has zero parsed
answers, reporting its sample count and whether training includes it. Training
continues. Inspect `validation/samples` and the native formats described above;
a zero parse rate on this finite panel does not establish a population-wide
failure. The warning is not repeated during periodic validation or checkpoint
recovery. Validation shares collection and grading
with training and partitions rows without duplication across ranks. Evaluate
all heldout rows with `validation_per_task=null`:

```bash
uv run python -m scripts.eval.encdec_rlvr \
  base_checkpoint_dir=/absolute/path/to/sft-checkpoint model_step=1116 \
  bundle_dir=/absolute/path/to/rlvr-data \
  stats_path=/absolute/path/to/feature_statistics.safetensors \
  rlvr_checkpoint_dir=/absolute/path/to/rlvr-run rlvr_step=100 \
  device_type=cuda validation_per_task=null \
  output_path=/absolute/path/to/full-heldout.json
```

Omit `rlvr_checkpoint_dir`/`rlvr_step` to evaluate the SFT starting model under
the same canonical RLVR feature policy. Evaluation output paths must be fresh.

Checkpoints contain weights, each rank's optimizer state, rank RNG states,
scientific resume identity, completed optimizer step, and WandB identity. A
completion marker is written only after every rank saves. Resume explicitly
with `resume_from_step=N`; incomplete checkpoints and identity/topology drift
fail. Admission fences the model, exact metadata bytes, and this rank's optimizer
across loading, requires the saved per-rank RNG envelope, and restores RNG in
lockstep before rollout. The runtime identity records both configured attention
and the resolved token KV-cache backend (FA3 or SDPA), including effective CUDA
SDPA flags when that fallback is active. On exact SM90, RLVR disables cuDNN SDPA
when FA4 is active, token KV uses SDPA, and hard deterministic algorithms are
disabled. A
paired H200 diagnostic produced exact agreement across six evaluations of the
eight-example panel, including after backward and compiler reset. This bounded
evidence does not identify a kernel cause or guarantee reproducibility for all
workloads matching the predicate. Nonfinal saves restart workers using the
existing deterministic runtime procedure. Resumed distributed CUDA workers
initialize NCCL in a fresh store namespace derived from the checkpoint step and
the torchrun restart generation, so bootstrap data from the pre-exec process
cannot be reused.

On SM12 consumer Blackwell (for example RTX PRO 6000), decoder full-prefix
attention uses FA4's native CuTe causal/sliding-window interface. The locked
FA4 release rejects the block masks used by the FlexAttention interface on this
architecture. A project-owned adapter is installed at shared GRPO/RLVR worker
admission, including evaluation and checkpoint restarts; it preserves GQA,
the inclusive left window, tensor layout, and autograd. KV-cache token decoding
retains its existing SDPA path, while packed encoder and cross-attention retain
their Triton kernels. SM12 native FA4 backward requires deterministic algorithms
to be disabled, as SM90 FA4 does. The runtime identity records this choice;
the existing `2e-5` policy-likelihood replay checks remain mandatory. This is
not a bitwise-deterministic training/resume guarantee. Other SDPA/SM100 exactness
and SM90 FA4 numerical limitations are inherited from
[structured-SFT GRPO](sft-self-conditioning-grpo.md).
See the [SM12 kernel checks and measured attention timings](../performance/fa4-sm12-rlvr-validation.md)
for the qualified shapes and performance limits.
Resume requires the recorded Git commit and unchanged Python sources. Keep a
dedicated checkout fixed for a run, including across checkpoint worker restarts.
The scientific identity binds the ordered trainable-parameter manifest. Exact
resume, including `resume_checkpoint_dir`, rejects changes of training scope
before restoring optimizer state. An absent flag resolves to `false` and adds
no configuration-identity difference; the existing source-revision guard still
applies. Head-only checkpoints use the same model format and evaluation loader.

Canonical prefix replay needs more compiled attention variants than Torch's
default limit. Admission raises both compiler limits to at least
`1 + 2 * distinct_decoder_windows * context_length` (8193 for the 3B checkpoint),
preserves larger ambient limits, and records the effective settings for resume.
Suppressed compiler errors are rejected, and exhausting either limit raises
instead of silently switching to eager attention. Dynamic packed cross-attention
has no proven specialization bound; a long run can still exhaust this budget.
Checkpoint replay carries the caller's context into CUDA autograd workers,
including Torch's context-local compiler settings.
Standalone evaluation may use a different ceiling, but retains the same
fail-on-exhaustion behavior.

Head-only training retains compiled kernels across ordinary optimizer steps,
avoiding repeated tracing of the frozen trunk. Full-policy training still
resets compiler state each step. Both scopes retain the mandatory worker
re-exec after nonfinal checkpoints, the compiler limits above, and every
behavior-likelihood replay check. Dynamic packed attention can still exhaust
the compiler budget between checkpoints; exhaustion raises rather than
changing execution paths. See the
[real-GPU performance measurements](../performance/rlvr-head-only-compiler.md).

To branch from an older checkpoint, set `resume_checkpoint_dir` to its source,
use a fresh `output_dir`, and set `wandb.new_run=true`. The new run records its
parent. It must retain the same scientific training configuration.

### Frozen decoder weight copies

Single-rank CUDA head-only runs can reuse temporary BF16 copies of frozen
decoder Linear weights by setting `frozen_decoder_max_shadow_bytes=2147483648`
(a 2 GiB budget). The default budget is zero, which disables this optimization.
The complete feature head, encoder, and cross-attention keep their existing
implementation. FP32 parameters, optimizer ownership, and checkpoint schemas
remain unchanged.

Copies are built after checkpoint restoration, reused through collection,
replay and validation, and released before each worker re-exec. Every resumed
worker rebuilds them. Startup and cleanup logs report admission, copy bytes,
setup time, and setup cost per completed worker update. Benchmarks report setup
and cleanup separately from complete-update samples; campaign costs must also
include checkpoint/restart and validation.

`frozen_decoder_min_free_bytes=4294967296` requires an estimated 4 GiB of
headroom after staging. Available memory includes unused allocator reserves;
fragmentation, stream use, and competing allocations can change what is usable.
This check does not guarantee future allocations will fit. Insufficient budget
or headroom, unsupported dtype/device, multiple ranks, and staging-only OOM
leave native forwards active. Full-policy training also uses native forwards.

Distinct registered parameters or buffers sharing selected storage disable
staging. Versioned writes or replacement of a selected parameter invalidate its
copy and use the current native weight. During the worker lifetime, keep the
model registry and training scope fixed; `.data` writes or writes through
unregistered aliases with independent version counters are unsupported. The
trainer performs all checkpoint loading before entry and optimizes only the
feature head while copies are active. Both memory settings remain part of the
normal exact-resume configuration identity.
Cleanup releases the owned copies and removes only the forward overrides it
installed, preserving a caller's later replacement or deletion.

See the [4090 performance report](../performance/rlvr-frozen-decoder.md) for
complete-update gains, exact numerical comparisons, setup/memory costs, and
the measured qualification boundary.

### Hidden replay cache

Set `hidden_replay_max_cache_bytes=268435456` to allow up to 256 MiB of
group-local hidden states and prompt snapshots during single-rank CUDA
head-only training. The default budget is zero. Behavior collection still
generates the same tokens and feature actions. The cache retains the complete
normalized input to the feature head for admitted canonical states; replay
applies the complete live head, including residual blocks, before the existing
position selection and pooling. Causal history, behavior likelihood, reward,
loss, gradients, and optimizer checks remain active.

The optional `hidden_replay_history_encoder=true` also reuses frozen encoder
outputs for repeated feature actions within canonical behavior collection.
Its default is false and it requires a positive cache budget. Each key contains
the exact ordered feature IDs, magnitudes, and mask. The cache owns all five
packed output tensors, including padding and layout, and returns independent
copies with native dynamic annotations. The generation provider retains its
native computation. A fresh frozen-state check covers
each nonempty history operation; new entries become reusable only after the
canonical forward's existing successful state check. Native action tensor
construction and segment append still run on hits.

The additional `hidden_replay_prompt_encoder=true` requires history reuse and
defaults to false. It retains the first admitted canonical prompt encoding in
each group, including independent snapshots of the three actual encoder inputs
and all five packed output tensors. Nonempty-history calls perform the fresh
frozen-state check before prompt lookup and compare input contents and metadata,
including changes through `.data` or aliases. Hits use independent output
copies through the synchronous history append loop; empty-history calls always
compute the prompt natively. New entries become reusable only after the same
successful canonical postcheck. Changed prompts use native computation, and
the first saved prompt remains eligible if its exact inputs recur.
If hidden-state admission evicts that prompt, it is not admitted again within
the group.

Each training group owns a separate session spanning collection and backward.
Sessions release their owned tensor references on success or error before any
optimizer step, validation, checkpoint save, or worker restart. An escaping
exception traceback or retained consumer can temporarily keep a tensor object
alive after that release; weak-reference diagnostics report those survivors
without replacing the native exception. Successful groups require zero
survivors. Validation and standalone
evaluation do not use this cache. Worker completion logs report aggregate
hits, fallbacks, allocation bounds, and cleanup; checkpoints contain no cache
tensors and retain their existing model/optimizer/RNG schemas. All cache
settings are pinned on exact resume.

Hits require the same model and group, unchanged frozen registered tensors,
model structure/configuration, native compute bindings, execution mode,
device, stream, and decoder-staging ownership. Head values remain live, while
head structure and trainability must stay fixed. The cache compares the fresh
contents of all four prompt tensors, including changes through `.data` or
independent aliases. Frozen-model `.data` writes, unregistered aliases with
independent version counters, concurrent mutation, and deliberate changes to
framework internals or operational caches are outside this contract.
Normal Dynamo constructor/deserializer tagging remains allowed: it does not
change the existing model's forward computation. The guard inspects live state
without requiring installed Torch source files.

Unsupported execution or framework inspection, mutation, missing states, and
memory refusal fall back to native computation. `hidden_replay_min_free_bytes`
defaults to `4294967296`; available-memory estimates include unused allocator
reserves and do not guarantee an allocation will fit. Only OOM from optional
cache copies is caught; errors from native model or head computation propagate.
The byte budget covers owned tensor storage, excluding Python metadata and
allocator reservations. Separate per-group limits allow at most 512 entries,
262144 key elements, and 8 MiB of serialized keys. Optional history and canonical
prompt entries share these limits, including pending copies and temporary
copies on hits. Both yield capacity to hidden-state admission and are released
when collection ends. No state is reused across groups or optimizer updates.

Complete-update benchmark timings include admission, copies, guards, content
comparisons, live-head replay, and group cleanup. When the cache is enabled,
the optional profiler retains its collection/replay scopes and native events
without replacing model forward methods, which would invalidate the cache.
See the [4090 performance report](../performance/rlvr-hidden-replay.md) for
complete-update gains, exact numerical comparisons, memory costs, and the
qualified source and workload boundaries.
The [history encoder report](../performance/rlvr-history-encoder.md) records
the additional opt-in history reuse results: a full-workload gain and a
short-workload regression, with exact parity in both.
The [canonical prompt encoder report](../performance/rlvr-prompt-encoder.md)
records two full-workload pairs in opposite process orders and the short-workload
regression. Prompt reuse remains disabled by default.

## WandB and qualification

WandB is enabled by default in project `nanochat-encdec-rlvr`. Only rank zero
initializes it, after admission. Set `wandb.mode=offline` explicitly for local
logging or `wandb.enabled=false` to disable it. Initialization/logging failures
are visible and synchronized across ranks. Logging uses completed optimizer
steps and suppresses already logged steps on recovery. Offline worker sessions
are separate local WandB files carrying the persisted run ID; they do not
contact or resume a server run until explicitly synced.

Metrics include reward and parse rates, zero-variance groups, action and token
counts, stop reasons, replay ratios, local gradient norms, learning rates,
timing, and peak memory. Each validation table has at most eight samples, two
per task, with bounded text. Checkpoints are never uploaded automatically.
Parse rate measures answer-format parsing; structural-invalid completions can
still contain a parseable answer, but their reward is always zero and their
explicit invalid-token stops are counted separately. Incomplete calculator
transcripts retain their EOS/budget stop classification.

Qualification must distinguish deterministic unit/contract tests, CPU tests
using the test-owned attention reference, real CUDA kernels, and the actual
3B model on one and at least two suitable GPUs. Test equality against the
native graders and legacy generator, causal boundary/tool cuts, compact feature
order, live-action normalization, all trainable gradient families, frozen LM
weights, distributed clipping/averaging, and fresh-process checkpoint recovery.
Tiny-model tests do not establish 3B memory use, throughput, or reward diversity;
a short successful run does not establish improved benchmark accuracy.

## Depth-2 step-1125 experiment

Issue #1691 uses `conditional_encdec_rlvr_depth2_s1125_full_h200` and
`runs/rlvr_depth2_s1125_full_h200.sh`. The four tasks, sampler, seeds, learning
rates, generation budgets and validation panel match
[control 4zib2dr5](https://wandb.ai/d0rb/nanochat-encdec-rlvr/runs/4zib2dr5).
The depth-2 checkpoint has 3,032,489,474 total parameters;
2,977,963,522 parameters in 364 tensors are trainable after freezing the LM
head. The retained projection and both residual blocks contribute six
trainable head tensors. Its feature width is 844 original-ID representatives
from a 65,536-feature SAE, and its recorded RustBPE cadence is 33.

Use the authenticated cadence-bearing view of step 1125. Preserve its original
SFT metadata and measurement receipt beside the derived metadata:

| File | SHA-256 |
|---|---|
| `model_001125.pt` | `d404424912a03b876cc89a318ac994a606c13f4031986b86b6a997ca327e468a` |
| `meta_001125.json` | `783853412eaddb5b0d9762093d423e50ec85ae67ab4e996b13444da27447cf4b` |
| Original SFT metadata | `de5f122bdda2f30ad61c39e06a7cfa325aab4e00ddbc0db6f9e88411be46ff2a` |
| `campaign2_cadence.json` | `ff2b7f8b9d8023370eac1be0126ee655e9a508fff8d3b177b881c524dd29ed12` |

Run the normal preparation pipeline against that view with both caps null.
Before reusing the existing raw extraction, require exact ordered equality of
110,633 training rows, 18,909 validation rows and 125,029 distinct contexts,
including prompts, answer keys, extraction inputs and tokenizer/statistics/
vocabulary/selection geometry. Convert into a fresh cache using the new
preparation digest and both new base pins, finalize normally, and compare all
decoded entries with the control. The bundle's final identity and cache
producer provenance change with the base checkpoint.

Set these operational inputs from the experiment's authenticated staging
receipt. Keep the source checkout fixed through all checkpoint re-execs:

```bash
export RLVR_SOURCE_REVISION=<reviewed-full-git-sha>
export RLVR_PYTHON=<qualified-environment>/bin/python
export RLVR_BASE_DIR=<staging>/base
export RLVR_BUNDLE_DIR=<staging>/bundle
export RLVR_STATS_PATH=<staging>/inputs/feature_statistics.safetensors
export NANOCHAT_BASE_DIR=<staging>/inputs/nanochat
export RLVR_OUTPUT_DIR=<staging>/smoke

# A separate two-step run with the full scientific batch and initial validation.
sbatch --partition=mit_normal_gpu \
  --account=mit_amf_advanced_gpu --qos=mit_amf_advanced_gpu \
  --gres=gpu:h200:1 --cpus-per-task=8 --mem=128G --time=06:00:00 \
  --output=<staging>/logs/smoke-%j.log \
  runs/rlvr_depth2_s1125_full_h200.sh \
  num_iterations=2 save_every=1 wandb.name=rlvr-depth2-s1125-full-h200-smoke

# After qualification, start production from SFT in a fresh directory.
export RLVR_OUTPUT_DIR=<staging>/production
sbatch --partition=mit_normal_gpu \
  --account=mit_amf_advanced_gpu --qos=mit_amf_advanced_gpu \
  --gres=gpu:h200:1 --cpus-per-task=8 --mem=128G --time=06:00:00 \
  --output=<staging>/logs/production-%j.log \
  runs/rlvr_depth2_s1125_full_h200.sh
```

Reserve storage for ten complete model/optimizer sets plus temporary writes.
Size the allocation using measured qualification throughput. Resume a complete
checkpoint with the same command and `resume_from_step=<completed-step>`;
keep the scientific configuration, source, environment and runtime identities
fixed. Full-bundle admission, nonzero learning, unchanged replay tolerances,
complete checkpoint markers, a frozen LM head and canonical standalone reload
are qualification gates. Report H200 FA4 backward numerical behavior separately
from deterministic tensor-equality tests.

The recorded step-1125 smoke and a subsequent 64-group natural-reward search
both produced only zero rewards. A separate, discarded controlled mixed-reward
update established finite nonzero head/trunk gradients and updates on the H200.
That fixture qualifies numerical update mechanics, not learning from verified
benchmark rewards. Its explicit reward provenance and the original unsuccessful
smoke learning check are retained in the experiment evidence; production keeps
the native grader and starts again from SFT.

Evaluate the final checkpoint with the canonical RLVR evaluator:

```bash
"$RLVR_PYTHON" -m scripts.eval.encdec_rlvr \
  "base_checkpoint_dir=$RLVR_BASE_DIR" model_step=1125 \
  "bundle_dir=$RLVR_BUNDLE_DIR" "stats_path=$RLVR_STATS_PATH" \
  "rlvr_checkpoint_dir=$RLVR_OUTPUT_DIR" rlvr_step=100 \
  device_type=cuda validation_per_task=32 output_path=<fresh-final-evaluation.json>
```

Publication uses fresh bucket prefixes
`rlvr/bundles/sft3b-head2-depth2-20260918-s1125-full` and
`rlvr/runs/rlvr-depth2-s1125-full-h200/`. Publish the cadence-bearing view and
receipt under a fresh checkpoint prefix. Verify checkpoint bodies remotely
before uploading their completion marker. The
[published experiment artifacts](https://huggingface.co/buckets/d0rb/nano-gpt-pretrain-steer-artifacts/tree/rlvr/runs/rlvr-depth2-s1125-full-h200)
include source, artifact and runtime identities, qualification evidence and
verified checkpoints. Publish the completed trajectory and comparison report
under the same run prefix, documenting the distinct SFT histories and
hardware/backends.
