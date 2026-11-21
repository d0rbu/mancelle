# Hugging Face decoder SFT on 32 H200s

This document is the implementation and acceptance specification for distributed
supervised fine-tuning of the repository's pinned Pythia-2.8B and SmolLM3-3B
Hugging Face decoder backends. It intentionally extends the existing
`scripts/train/chat_sft.py` path instead of introducing a second trainer.

The production target is four eight-GPU H200 nodes. Correctness gates are
mandatory at every smaller topology before a 32-GPU benchmark is meaningful.
Performance measurements are diagnostic until the exact model, dataset snapshot,
software lock, topology, and timing window are recorded together.

## Non-negotiable invariants

1. **The optimized objective is the exact global supervised-token mean.** Padding
   and `sft_loss_mask` may create unequal valid-token counts across micro-batches
   and ranks. The summed loss numerator and valid-token denominator therefore
   define the objective; neither micro-batches nor ranks receive equal weight by
   accident.
2. **Only a successful optimizer update commits input progress.** The iterator
   publishes the state that produced a complete batch. The loop promotes the
   final state for an accumulation window only after `optimizer.step()` succeeds.
3. **Best-fit packing resumes exactly.** A scalar cursor is insufficient because
   the packer pops conversations out of order. A checkpoint records the next
   striped logical cursor and the ordered logical positions still in the buffer.
4. **Distributed HF runs have an explicit positive optimizer-step horizon.** A
   dataset horizon is rank-asymmetric and can truncate work when ranks finish at
   different times. `num_iterations=-1` remains available only to the legacy
   single-process path.
5. **Restart is fail closed.** The topology, complete resolved training config,
   pinned source, tokenizer/template contract, task-mixture fingerprints, optional
   dataset snapshot digest, and batch/optimizer geometry must match.
6. **Portable exports and restart checkpoints are different artifacts.** Portable
   exports contain BF16 model/tokenizer/metadata for inference. Restart checkpoints
   contain FP32 model, optimizer, RNG, and rank-local iterator state.
7. **Measurements name their denominator.** Scheduled tokens, non-padding target
   tokens, and supervised tokens are reported separately. Algorithmic MFU uses the
   repository FLOP estimator and the configured H200 peak convention; it does not
   claim to include recomputation FLOPs from activation checkpointing.

## Minimal implementation shape

The implementation owns no new training framework. It changes only these seams:

- `steering/sft/iteration.py` owns deterministic best-fit state publication and
  restoration plus token-count metadata for each yielded batch.
- `steering/sft/hf_decoder.py` owns the supported attention/optimizer strategies,
  summed-loss adapter call, optional compilation, and DDP wrapping helpers.
- `steering/sft/hf_restart.py` owns canonical semantic identity and atomic,
  complete-marker restart save/load.
- `scripts/train/chat_sft.py` owns orchestration, exact loss normalization,
  commit boundaries, profiling scopes, and artifact scheduling.
- Existing portable export code remains the only portable model writer.
- Existing profiling primitives (`TimingRecorder`, `timed_region`,
  `timing_tags`, `StartupTrace`) remain the only instrumentation primitives.

No code under `nanochat/` is modified. No prefetcher, custom optimizer, custom
collective scheduler, asynchronous checkpointer, or automatic topology migration
is introduced.

## Configuration contract

The shared HF block adds only knobs that select materially different behavior:

```yaml
huggingface:
  attention_implementation: sdpa       # sdpa | flash_attention_4
  distributed_implementation: dist_muon_adamw  # dist_muon_adamw | ddp_fused_adamw
  torch_compile: false
  restart_from_step: -1
  restart_save_every: 0
  data_snapshot_sha256: null
```

Profiling is off by default and uses a small top-level surface shared with the
repository's existing terminology:

```yaml
profile_timing: false
profile_trace_steps: []
profile_trace_ranks: []
profile_output_dir: out/profile/chat_sft
```

`profile_trace_steps` is an explicit list of completed optimizer-step numbers,
each within the configured positive horizon; this avoids ambiguous
warmup-relative arithmetic and unreachable selections. A trace is created only
on a selected rank and step; both lists must be empty or both nonempty. Timing
CSVs are rank-local. With `profile_timing=true`, each rank records an accurate
`chat_sft/paid_step` row from the loop's existing outer synchronization; nested
rows are unsynchronized host-launch timings. A fully instrumentation-free
control keeps timing disabled too.

The implementation accepts only `sdpa` and `flash_attention_4`, only the two
optimizer/distribution strategies above, strict booleans for compile/timing,
non-negative restart/save steps, and a lowercase SHA-256 dataset digest
when supplied. The H200 presets keep gradient clipping disabled because the
sharded optimizer's existing local-before-reduction clipping semantics are not
equivalent to DDP's globally averaged gradient.

## Startup and object ownership

Initialization is ordered to minimize peak host memory and make restart semantics
unambiguous:

1. Compose and strictly validate config; initialize distributed CUDA.
2. Load and validate the pinned source config and tokenizer.
3. Load one FP32 HF model per node-local-rank slot, move it to its GPU, and
   release transient host weight storage before building the resident datasets.
4. Resolve batch geometry, token-byte accounting, task mixtures, and immutable
   fingerprints; reject distributed dataset-horizon runs.
5. Resolve the output root, build the canonical semantic identity, and require
   its digest to agree across every rank.
6. If restarting, load the FP32 model state before DDP construction.
7. Optionally compile the adapter, then optionally wrap it in DDP.
8. Construct the strategy-specific optimizer and restore its state.
9. Validate and restore Torch CPU/CUDA RNG and the rank-local iterator state.
10. Initialize external logging and construct train/validation generators.

The unwrapped adapter owns parameters and FLOP estimation. The raw Transformers
model owns portable export. The training object is either the adapter, compiled
adapter, or DDP wrapper. Evaluation and portable export use the unwrapped adapter
or raw model and never a DDP-prefixed state dict.

## Exact global supervised-token loss

For one optimizer step, construct all accumulation micro-batches under one
rank-synchronized data-construction stage. This cuts the normal success-path
control collective from one per micro-batch to one per optimizer step while
preserving fail-loud data errors before model collectives begin.

Let `n_r` be the total count of `targets != -1` across all local micro-batches on
rank `r`, and `S_r` the corresponding differentiable summed cross entropy. One
all-reduce computes `N = sum_r n_r`; `N == 0` is an error. Each rank backpropagates

```text
S_r * world_size / N
```

Both supported distributed strategies average gradients across ranks: DDP does
so during backward, and `DistMuonAdamW` does so in optimizer collectives. The
resulting gradient is therefore exactly

```text
sum_r grad(S_r) / N
```

There is no separate division by gradient-accumulation steps. DDP enters
`no_sync()` for every micro-batch except the last so only the final backward
reduces accumulated gradients. Detached summed numerators are all-reduced for the
logged global loss; logging never reuses only the last micro-batch loss.

Each optimizer step records:

- scheduled tokens: `B * T * world_size * grad_accum_steps`;
- non-padding targets: targets before completion masking but after row padding;
- supervised targets: final `targets != -1`;
- globally normalized loss numerator and denominator.

Each CSV duration is one rank's monotonic `perf_counter` interval between the
existing pre/post CUDA synchronizations. For a whole-job step metric, align all
32 rows by completed step and use the maximum `paid_step.elapsed_ns`; never sum
rank durations or select the fastest rank. Recompute cluster throughput and MFU
from that maximum when ranks differ materially.

## Deterministic best-fit state

Every fetched conversation has a monotonic logical position. Dataset access is
`logical_position % dataset_size`; each rank increments its logical position by
`world_size`. Buffer entries retain `(logical_position, ids, loss_mask)`.

The JSON-safe published state contains only the state that cannot be derived
from the fenced run identity and topology:

```text
next cursor, ordered buffered positions, consumed, micro-batch count
```

On restore, validation combines those fields with the fenced rank, world size,
dataset identity, and fixed buffer size. It checks rank ownership modulo world
size, strictly increasing unique buffered positions, every buffered position
preceding the next cursor, and accounting consistency. Skipped conversations and
current epoch are derived from that state. The buffer is deterministically
re-rendered in saved order. State is published only after every output tensor and
count has been constructed. A failed forward, backward, clipping, or optimizer
step does not update the loop's committed copy.

Python objects for rendered conversations are deliberately not serialized. At a
buffer size of 100, deterministic re-rendering is simpler, safer, and small
relative to full-model restart loading.

## Restart artifacts

Restart files live under `<hf-output>/restart/`:

```text
model_000500.pt
meta_000500.json
optim_000500_rank0.pt ... optim_000500_rank31.pt
complete_000500.json
```

The repository's atomic Torch/JSON writers are reused. Rank zero writes the FP32
model state and metadata. Every rank writes one rank file. With
`dist_muon_adamw`, every rank file includes its optimizer shard and runtime state.
With `ddp_fused_adamw`, rank zero includes the full optimizer state and every
other rank file contains runtime state only. Runtime state contains the committed
iterator snapshot and Torch CPU/CUDA RNG state.

After all writes succeed, a barrier separates data persistence from publication.
Rank zero removes the in-progress lock, every rank confirms that cleanup, then
rank zero checks the exact expected inventory and writes `complete_<step>.json`
as the final filesystem mutation. Only a checkpoint with a valid complete marker
is loadable. The marker binds step, world size, strategy, semantic-identity
digest, and expected filenames. The publication stage prevents any rank from
proceeding before the marker exists.

Restart loading is explicit by step. It validates the marker and inventory before
loading, uses node-local-rank slots for large model/optimizer reads, validates the
semantic identity and topology, and then restores model, optimizer, iterator, and
RNG. The just-loaded boundary is observational: the trainer does not try to
rewrite its portable or restart artifacts before executing the next update.
When both artifact types are due, restart publication happens first. If the
following portable publication fails at the final boundary, exact-final resume
may backfill the absent portable directory from the completed restart; an
existing directory is validated and never overwritten.
There is no latest-step guessing, pruning, topology migration, or recovery of
incomplete files in the first implementation.

## Semantic identity

The canonical JSON identity includes:

- repository commit;
- resolved Hydra training config, with `restart_from_step` normalized to `-1`
  and non-semantic profiling keys omitted;
- pinned source ID/revision and validated architecture contract;
- tokenizer vocabulary, structural token IDs, and chat-template digest;
- train and validation mixture fingerprints;
- optional operator-supplied dataset snapshot SHA-256;
- world and local-world topology.

The digest and canonical payload are both stored. Profiling can be disabled on
resume without changing training identity; all training-semantic config remains
fenced.

## Distribution and optimizer strategies

`dist_muon_adamw` preserves the existing one-Adam-group `DistMuonAdamW` behavior.
It reduces gradients during `optimizer.step()` and does not overlap communication
with backward.

`ddp_fused_adamw` optionally compiles the adapter, wraps it with
`DistributedDataParallel(device_ids=[local_rank], output_device=local_rank,
broadcast_buffers=False, gradient_as_bucket_view=True,
find_unused_parameters=False)`, and uses `torch.optim.AdamW(fused=True)` on CUDA.
No bucket-size tuning or `static_graph=True` is used until a trace proves a need.
The CPU test helper uses non-fused AdamW because fused AdamW is CUDA-only.

Every optimizer group stores `initial_lr` and `kind="adamw"` so the existing LR
loop stays strategy-independent. Parameter identity and exact coverage are checked
at construction.

## Attention and compilation

The selected attention implementation is passed to the initial HF model load and
recorded in training/restart metadata. Portable export validation always reloads
with SDPA to prove that the output is portable and not dependent on FlashAttention.

Compilation is opt-in and uses the existing repository-compatible automatic
dynamic-shape behavior. A compile benchmark is accepted only after numerical
parity, absence of recompiles in the measured window, and a repeatable useful
throughput gain. It is not coupled to the DDP or attention choice.

## Profiling and metric contract

`StartupTrace` gains only an optional event prefix; the existing `encdec_*`
default remains byte-for-byte compatible. Decoder startup records source-config
preflight, distributed compute initialization, model load, mixture identity,
restart load, DDP/compile, optimizer, first batch, first optimizer update, and
steady state. The trace begins after Hydra composition/backend validation and
before process-group/CUDA initialization, so its time-to-first-step includes
the distributed rendezvous.

Rank-local `TimingRecorder` scopes use stable names for:

- `chat_sft/batch_construction`
- `chat_sft/forward_backward`
- `chat_sft/gradient_clip`
- `chat_sft/optimizer_step`
- `chat_sft/whole_step`

The whole-step CUDA-synchronized wall time is authoritative for throughput and
MFU. Region timing is unsynchronized host-side diagnostic data. Trace collection
is a separate run: initially ranks 0, 8, 16, and 24, two steps
after at least ten warmup updates, without Python stacks unless a trace requires
them. External DCGM/`nvidia-smi` capture owns clocks, power, thermals, and link
counters; no in-process monitoring thread is added.

## Dataset and mixture discipline

Dataset cache contents are staged before allocation and described by an
operator-generated manifest whose SHA-256 is supplied as
`data_snapshot_sha256`. Existing mixture construction is unchanged.

No asynchronous prefetch is added initially. If synchronized traces later show
batch construction above 5--10 percent of paid step time, a bounded prefetch may
be proposed separately with exact iterator/RNG commit semantics.

## Testing pyramid

### Pure and property tests

- Strict config-domain boundaries and inert/unknown values.
- Summed-loss/count helpers against a concatenated independent oracle.
- Best-fit state round trips, epoch wrap, completion-only skips, truncation,
  malformed/tampered state, and random conversation-length properties.
- Semantic identity canonicalization and strict mismatch rejection.
- Complete-marker inventory, ordering, and injected write failures.
- Strategy-specific optimizer coverage and metadata.
- Profiling-off zero-work behavior, selected rank/step traces, and lifecycle.

### Multiprocess integration tests

- Two-rank Gloo exact global objective with unequal rank/micro-batch masks for
  both optimizer strategies, compared with one concatenated-process update.
- DDP `no_sync()` accumulation and tied/untied parameter coverage.
- Uninterrupted versus save/restart runs with exact next batches, optimizer
  states, RNG, and final parameters.
- Fault injection before model, rank, metadata, and marker publication; no
  incomplete checkpoint is loadable.
- Rank-local data errors fail before the training model's first collective.

### CUDA and production acceptance tests

Opt-in marked tests use the pinned full Pythia-2.8B and SmolLM3-3B sources and the
real selected dataset cache. They perform full FP32 model load, BF16 autocast
forward/backward, optimizer update, intentional interruption, exact restart
restoration into the next update, portable BF16 export/reload/generation, and
finite loss/output assertions. The tests fail rather than silently shrink the
model when their explicit environment gate is enabled. Separate topology runs in
the production sequence cover two-rank NCCL, one-node eight-H200, four-node
smoke, and four-node sustained benchmarking.

Tests that require unavailable H200s, scheduler allocation, or a private model
checkpoint remain present and explicitly unrun locally; they are never replaced
by claims based only on tiny models.

### Concrete production commands

The one-H200 gate deliberately runs both full pinned sources, a real standard
mixture, one fused-AdamW update followed by an intentional interruption, exact
FP32 restart restoration and a second update, BF16 portable
publication/reload/generation, synchronized region timings, and one selected
trace. Use a fresh output root with at least roughly 100 GiB free per family
and retain it as evidence:

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
RUN_HF_DECODER_SFT_FULL_GPU=1 \
HF_DECODER_SFT_DATA_MANIFEST=/shared/manifests/standard-sft.json \
HF_DECODER_SFT_FULL_OUTPUT_ROOT=/shared/results/hf-sft-one-h200-<run-id> \
uv run pytest -m very_slow \
  tests/steering/test_hf_decoder_sft_full_gpu.py -q -s
```

Before submitting the 4x8 canary, commit the exact tree and freeze every
required identity. The data manifest is operator-owned; it should enumerate
the cache objects/revisions or exact local bodies used by all constituent task
datasets, not merely contain a human label.

```bash
export EXPECTED_GIT_REVISION="$(git rev-parse HEAD)"
export EXPECTED_UV_LOCK_SHA256="$(sha256sum uv.lock | awk '{print $1}')"
export DATA_MANIFEST=/shared/manifests/standard-sft.json
export DATA_MANIFEST_SHA256="$(sha256sum "${DATA_MANIFEST}" | awk '{print $1}')"
export NANOCHAT_BASE_DIR=/shared/nanochat-cache
export HF_HOME=/shared/huggingface-cache
export OUTPUT_DIR=/shared/results/pythia-sdpa-dist-<run-id>
export CONFIG_NAME=chat_sft_pythia_2_8b_h200_4x8_benchmark
export ATTENTION_IMPLEMENTATION=sdpa
export DISTRIBUTED_IMPLEMENTATION=dist_muon_adamw
export DRY_RUN=true
export PROFILE_TIMING=true

mkdir -p experiments/logs
sbatch --account="${ACCOUNT}" --qos="${QOS}" \
  --export=ALL scripts/slurm/hf_decoder_sft_h200_4x8.sbatch
```

The 60-update throughput canary is dry by default: it writes the immutable run
contract, copied data manifest, Slurm allocation, NCCL preflight, and 32
rank-local timing CSVs, but no model/optimizer files. It also records the
actually imported Python, Torch, Transformers, CUDA, cuDNN, and NCCL versions;
the FlashAttention 4 and kernels package versions are retained as well, and the
lockfile digest alone is not treated as proof that the environment was synced.
TorchInductor compile workers default to eight per rank so an opt-in compile arm
cannot oversubscribe the 64 CPUs allocated per node. The authoritative throughput
rows are `name=chat_sft/paid_step`;
discard steps 1--10 as warmup and analyze steps 11--60.
Nested regions are unsynchronized host timings and do not perturb the
synchronized outer row.

For a persistence acceptance run, use a fresh output and set
`DRY_RUN=false`, `RESTART_SAVE_EVERY=30`, `VALIDATE_EXPORT=true`, and
`PROFILE_TIMING=false`. This is storage-intensive: rank-local DistMuon optimizer
shards, an FP32 model, and a BF16 portable export are retained. To continue an
interrupted persistence run, reuse the exact environment and existing
`OUTPUT_DIR`, set `RESTART_FROM_STEP` to a retained complete step, and keep both
profiling modes disabled. The launcher verifies the requested marker and retained
data manifest before loading; every Slurm invocation gets a distinct evidence
directory under `OUTPUT_DIR/invocations/`. For a trace arm, use another fresh
output and pass explicit lists such as `PROFILE_TRACE_STEPS='[12]'` and
`PROFILE_TRACE_RANKS='[0,8,16,24]'`; do not use that arm for throughput.

## H200 benchmark sequence

1. Freeze repository commit, uv lock, model revisions/checkpoint digest, dataset
   snapshot digest, H200 SKU, clocks/power mode, node list, network interfaces,
   NCCL environment, and output root.
2. Run topology preflight: one collective GPU per node, then all eight per node;
   verify bus IDs, peer links, IB/RoCE devices, all-reduce bandwidth, clock state,
   thermals, ECC/Xid health, local cache capacity, and output quota.
3. Pass one-GPU exact-model update/export/reload.
4. Pass one-node eight-GPU loss/resume and short sustained run.
5. Pass four nodes with one GPU each to isolate inter-node behavior.
6. Pass four nodes with eight GPUs each for loss/resume, then benchmark.
7. Benchmark one variable at a time in this order: attention, activation
   checkpointing and micro-batch geometry, optimizer strategy, compile, world
   size. Carry only a correctness-passing winner forward.
8. Use at least ten warmup updates, 30--50 measured updates, three repeats, and an
   ABBA order for close comparisons. Run heavy traces separately.

For every trial retain raw per-rank step rows, startup summary, config, environment,
external telemetry, NCCL logs when enabled, portable/restart artifact inventories,
median/p95 step time, scheduled/non-padding/supervised throughput, MFU convention,
HBM peak, and failure health signals.

If one 32-rank job scales below roughly 75 percent of the eight-rank reference and
training rounds are independent, compare aggregate useful throughput from multiple
smaller jobs before committing the limited allocation to a single world-size-32
run.

## Acceptance gates

- Exact global loss and uninterrupted/resume parity pass at each topology.
- Final portable export reloads with SDPA and produces finite logits/generation.
- No hang, nonfinite loss/gradient, CUDA Xid/ECC fault, or restart ambiguity.
- At least 15 percent HBM headroom under the selected steady-state geometry.
- Measured p95 step time is at most about 1.2 times median unless the retained
  trace identifies an expected periodic operation.
- No compile recompile occurs in the measured window.
- A complexity-increasing option is retained only for a repeatable useful
  throughput improvement (normally at least about 3 percent) with no correctness
  or reliability regression.
- No unexplained region accounts for more than about 10 percent of paid step time.

## Explicit non-goals

- Modifying vendored `nanochat/`.
- Supporting arbitrary Transformers architectures or unpinned revisions.
- Automatic restart discovery, deletion, pruning, or topology migration.
- Serializing rendered conversations or adding a state-schema framework.
- Asynchronous checkpointing or data prefetch before measurement justifies it.
- FSDP, tensor parallelism, pipeline parallelism, or a custom communication
  scheduler.
- Treating an algorithmic MFU proxy as hardware-measured FLOPs.

## Implementation addenda

Addenda are appended here when current-main code, tests, or real H200 evidence
changes an assumption. Each entry must state whether it changes behavior, tests,
or only the benchmark runbook.

- **2026-08-26 -- current-main anchor.** Implementation starts from
  `5bdf6c54e192028b143296a48e66a6c7492cd695`; the final review rebased it onto
  `d1e2dc94233cca440944d674bce670ea9cd34e43`. Intervening current-main changes
  do not alter decoder-only HF SFT. No behavior change to this specification.
- **2026-08-27 -- final restart ordering audit.** Restart RNG restoration now
  occurs after external logging setup and immediately before iterator creation.
  Rank-local restart argument, artifact, optimizer, runtime, and RNG checks are
  aligned before later collectives; two-rank fault tests prove that one malformed
  rank aborts every rank before checkpoint IO or continued training. Local-world
  environment parsing is likewise aligned before topology collectives, and
  selected trace export plus final profiling publication failures are aligned
  before another training collective.
- **2026-08-27 -- minimal exact iterator state.** The restart payload stores
  only the logical cursor, ordered buffered positions, consumed count, and
  micro-batch count. Dataset size is already fenced by the run identity and
  skipped conversations are derived from
  `attempted - consumed - buffered`, so persisting either would duplicate
  state. Malformed-state and replay tests cover the reduced contract without
  serializing rendered conversations.
- **2026-08-26 -- DDP restart ownership.** Fused-AdamW rank zero stores the one
  full optimizer state, all other rank files store runtime only, and every rank
  reads rank zero's optimizer payload through serialized node-local-rank slots.
  DistMuonAdamW continues to store one local optimizer shard per rank. Tests
  cover both layouts under two-process Gloo.
- **2026-08-26 -- distributed identity consensus.** After each rank builds its
  canonical run identity, a fixed-size integer collective requires every digest
  byte to agree before restart IO, compilation, or optimizer construction.
  Two-process tests cover both agreement and all-rank fail-closed disagreement.
- **2026-08-26 -- loaded-boundary scheduling.** A resumed interval is never
  republished at the initial loop boundary. This prevents the strict
  no-overwrite checkpoint contract from rejecting a valid continuation.
  Restart publishes before a same-boundary portable export, so exact-final
  resume can backfill an absent portable directory after export failure without
  rewriting restart state. The schedule, directory probe, and publication order
  are covered directly, and the one-H200 gate intentionally interrupts after
  update one, restores that full checkpoint, then completes update two through
  the real trainer.
- **2026-08-26 -- benchmark timing.** Rank-local timing CSVs now include one
  `chat_sft/paid_step` row per committed update, measured by the already-present
  pre/post CUDA synchronization. This permits undistorted raw step rows with
  nested synchronization disabled; trace and synchronized-attribution arms
  remain separate.
- **2026-08-26 -- production gates.** The checked-in one-H200 pytest gate has
  no reduced fallback and requires both full pinned sources, the real standard
  mixture cache, an operator data-manifest digest, interrupted optimizer/restart
  continuation, export IO, and an explicit offline H200 environment. The 4x8
  Slurm launcher additionally fences git and `uv.lock`, requires a new absolute
  output root for a fresh run or an exact complete marker for continuation,
  retains per-invocation evidence, records every runtime/profile axis, runs NCCL
  preflight, and admits only the two canary configs. These hardware gates remain
  unexecuted until the allocation exists.
- **2026-08-26 -- literal marker-last publication.** The in-progress lock is now
  removed and synchronized before the complete marker is written, making that
  marker the final filesystem mutation. Fault-injection tests cover data-write,
  lock-cleanup, and marker-write failure boundaries.
- **2026-08-26 -- private SmolLM3 source boundary.** The repository supports the
  pinned public SmolLM3 source and validated HF-compatible source identity. The
  exact path/revision or content manifest for the operator's private checkpoint
  was not supplied, so no speculative private preset or unverified digest was
  added. Add that preset only after those immutable inputs are available.
