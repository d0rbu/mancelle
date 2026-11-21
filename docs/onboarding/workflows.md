# Workflows

How to do common dev tasks. For a full command reference see [`docs/cli-reference.md`](../cli-reference.md). For deep pipeline docs see [`docs/pipelines/`](../pipelines/).

---

## Run a fast end-to-end smoke test

Validates that extraction + training + eval all work on your machine. ~20 min on a single GPU.

```bash
bash runs/extract_features_quick.sh                                          # 1 shard of features
python -m scripts.train.base_train config_name=base_train_tiny_1gpu                # 100-iter tiny train
python -m scripts.eval.base_eval --config-name=base_eval_quick source=base model_tag=d6
```

If all three finish, you're set up.

---

## Train a custom configuration

The pattern is: pick a base config in `configs/training/`, override the knobs you care about on the CLI.

```bash
# Example: train a depth-12 base model on 4×H100 with FP8
torchrun --standalone --nproc_per_node=4 -m scripts.train.base_train -- \
    depth=12 device_batch_size=16 fp8=true run=d12_v2

# Example: train an encoder-decoder with a smaller encoder
torchrun --standalone --nproc_per_node=2 -m scripts.train.conditional_encdec_train -- \
    depth=12 encoder.n_embd=0.25 encoder.depth=0.25 \
    feature_dir=out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base \
    run=encdec_small_v1
```

If your machine differs from the canonical hardware, put per-machine batch sizes in `configs/local/<config_name>.yaml` (gitignored) instead of overriding on every command.

---

## Produce clustered selections from remote bodies

Both supported modes first need packed inventory v2:

```bash
uv run python -m scripts.preprocess.publish_remote_packed_inventory \
  packed_row_dir=<local-packed-dir>
```

For ordinary clustered sampling (`feature_delta=null`), upload those packed
bodies and the inventory to their canonical `selection_prefix` mapping, keep
the local statistics/packed-manifest/provenance controls, and run with
`selection_num_workers=1`. No raw/percentile body or source inventory is used:

```bash
uv run python -m scripts.preprocess.precompute_selected_features \
  --config-name=<clustered-remote-packed-config> \
  feature_dir=<local-controls> packed_row_dir=<local-packed-controls> \
  feature_delta=null feature_selection.delta_weights=null \
  selection_num_workers=1 \
  'remote_preselection={staging_dir:/dedicated/scratch,max_output_shard_bytes:1000000000}'
```

The producer streams one packed body plus one lookahead. A partial receipt
prefix is replayed byte-for-byte to recover the continuous sampling RNG;
complete receipts finalize without replay. Selection body count equals packed
inventory count, not raw feature-shard count.

The splice-CF path additionally requires its authenticated source family:

This path supports only static clusters, precomputed sampling, and
`feature_delta.mode=splice_counterfactual`. Keep feature statistics, packed
manifests/provenance, percentile manifests, and `_splice_provenance` sidecars
local. Raw, percentile, and packed bodies are streamed; there is no local or
live fallback.

Existing remote training still accepts packed inventory v1. The remote
pre-selection producer requires v2. Clustered splice pre-selection trusts the
completed percentile family already present at the configured remote prefix;
it builds a short-lived source contract from remote listings and local layout,
statistics, manifest, and sidecar controls. No source inventory is downloaded,
recomputed, uploaded, or retained.
For a smoke, use a dedicated `remote_training.selection_prefix` containing a
small packed/source family and compare its selection body bytes with a local
run over the same artifacts.

Run or restart with the exact same semantic selection config:

```bash
uv run python -m scripts.preprocess.precompute_selected_features \
  --config-name=<clustered-remote-packed-config> \
  feature_dir=<local-derived-controls> \
  packed_row_dir=<local-packed-controls> \
  'remote_preselection={staging_dir:/dedicated/scratch,max_output_shard_bytes:1000000000}'
```

The producer validates admission before creating staging/output state, uploads
each body followed by its receipt, and uploads timestamp-free `manifest.json`
last. Restart with the same command. Valid receipts skip work; body-only orphans
are deterministically recomputed and verified; receipt/body conflicts fail.
The producer never overwrites conflicting objects or deletes remote debris.

Before a full launch, retain evidence from a one-packed-shard parity smoke and
an interrupted several-shard resume smoke. Completion requires exact expected
body/receipt counts, final manifest presence, empty owned staging, and successful
`RemoteSelectionMaterializer` discovery. A synthetic green suite is not a
full-corpus production proof.

---

## Launch verified remote packed training

Use the dedicated preset only after publishing the signed packed inventory and
selection families documented in the encoder-decoder pipeline:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m scripts.train.conditional_encdec_train -- \
  config_name=cluster_fp/conditional_encdec_splice_cf_gemma3pt_c1024_remote_packed \
  feature_dir=<derived-c1024-feature-dir>
```

That hardware-independent preset deliberately leaves decoded-batch CUDA staging
disabled. The exact eight-H100 B=16/BF16 production spelling is:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m scripts.train.conditional_encdec_train -- \
  config_name=cluster_fp/conditional_encdec_splice_cf_gemma3pt_c1024_remote_packed_8xh100
```

The target preset pins the production derived-feature root, the 524,288-token
global batch, `fp8=false`, `steering_eval_every=0`, a 512 MiB pageable host
queue, and three pinned slots. It inherits the existing `run` and `model_tag`;
staging is operational and does not create a new checkpoint identity.

The preset requires `body_prefetch.lookahead_shards_per_rank=2`; do not add an
enable flag. The protocol supports multi-node torchrun when every rank sees
the same cache bytes and `flock` namespace, holds a cache-root session lock,
and admits the full signed byte peak before spawning one isolated downloader
per rank. Run concurrent jobs with separate `feature_root` cache roots.

For signed inventory position `i`, only rank `i % world_size` may transfer the
packed/selection pair. All ranks still descriptor-pin and decode every body
needed by their existing row stripes. A worker failure aborts; deleting a
demand, changing owners, or letting a consumer fetch is not a recovery path.

Body prefetch stops at verified compressed bytes. Decode already runs on the
batch-producer thread, but the unstaged B=16 route buffers only four
microbatches (one optimizer step). The H100 target instead lets the existing
byte-bounded staging producer advance through decode, selection validation, and
batch reconstruction far enough to enter the next shard while current-shard
batches remain consumable. The same producer naturally pulls issue-#826 cache
verification ahead; it does not weaken that integrity gate. Issue #867 peer
status polling remains independent and is supplied by the issue-#868 stack
base, PR #870, rather than by CUDA staging.

Before an eight-GPU production launch, run the focused 2/4/8-process suite and
the target-host cold-cache acceptance procedure in
[Strict Rank-Striped Remote Body Prefetch](../pipelines/remote-cache-materialization.md).
Keep Xet performance settings in the launch environment and benchmark them as
separate runs; do not use deprecated `hf_transfer`.

---

## Evaluate an existing checkpoint

```bash
# BPB + CORE + samples (slow but complete)
torchrun --standalone --nproc_per_node=8 -m scripts.eval.base_eval -- \
    source=base model_tag=d12 step=10000

# BPB only (fast)
python -m scripts.eval.base_eval --config-name=base_eval_quick source=base model_tag=d12 step=10000

# Eval an HF model directly (no checkpoint needed)
python -m scripts.eval.base_eval source=hf hf_path=openai-community/gpt2 eval=bpb

# Encoder-decoder eval (full-context conditioned)
torchrun --standalone --nproc_per_node=8 -m scripts.eval.base_eval -- \
    source=encdec model_tag=encdec_d12 step=54600 \
    conditioning.context_cache_dir=out/eval_caches/core \
    bpb.val_packed_row_dir=out/packed_val_rows \
    feature_configs=configs/eval/feature_filter/<config>.yaml
```

For steering-specific evals (AxBench, harmfulness, completion steering) see `docs/cli-reference.md` → Evaluation, and the deep dives in `docs/axbench_*.md`.

---

## Tune autointerp filter judge concurrency

The autointerp filter is single-process: `judge.max_concurrent` (default 8)
bounds how many uncached intruder trials are judged concurrently via an
in-process thread pool, itself bounded by `judge.requests_per_minute`
(default 60). One model/SAE/tokenizer stays resident in VRAM for the whole
run — there is no process sharding, no plan/prepare/launch/combine step, and
no shared on-disk lock coordination to manage.

```bash
uv run -m scripts.preprocess.precompute_autointerp_filter -- \
  feature_dir=path/to/features \
  n_sae_features=65536 \
  output_dir=out/autointerp/my_run \
  source_index.enabled=true \
  token_examples.materialization_batch_size=1 \
  judge.max_concurrent=8
```

The generic command uses the CPU/TransformerLens reference backend, so it must
set materialization batch size `1`. The repository default is the fixed
production slot-block size `128`, with no `auto` mode; this is not the model
batch dimension. It is accepted only for native-CUDA Gemma residual-post,
context-independent/no-BOS, fixed-token singleton targets. Each real slot runs
at its true serial length on one of two CUDA streams. Use the Gemma 3 4B PT
l22/f65k preset for that validated contract. Unsupported combinations,
runtime-fingerprint/profile-digest mismatches, and OOMs fail rather than
silently falling back or reducing concurrency. Block materialization completes
before judge concurrency begins.

Serial (`judge.max_concurrent=1`) and concurrent runs over the same targets
produce byte-identical artifacts (see
`tests/steering/autointerp_filter/test_concurrency_parity.py`), so raising
`max_concurrent` is purely a throughput knob — it never changes scoring
output. `max_judge_calls` still caps total admitted uncached trials for the
whole run. Successfully parsed trial results are committed incrementally from
the runner thread as completions are observed, so an interrupted run reuses
every completed SQLite commit; worker threads never write the cache. Canonical
outcome ordering still determines final scoring and which error is raised.

Before a long job, sanity-check concurrency throughput with the no-model/no-API
synthetic harness:

```bash
python -m scripts.benchmark.benchmark_autointerp_concurrency \
  --n-targets 12 --n-trials 8 --max-concurrent 8
```

and check the real VRAM footprint of the production model/SAE stack (useful
for sizing `max_concurrent` against available VRAM) with:

```bash
PYTHONPATH=. uv run python -m scripts.benchmark.measure_autointerp_model_vram
```

See [`autointerp-filter.md`](../pipelines/autointerp-filter.md) for the full
caching, judge-budget, and validation contracts.

---

## Run the pinned SmolLM3 cooldown

Stage the immutable source snapshot and prepare data once on shared HenryAC
pool storage. Preparation resumes only complete receipt-backed shards and stops
at the first deterministic shard prefix with enough 4,097-token windows at the
standard 4,096-token causal stride:

```bash
uv run python -m scripts.preprocess.prepare_smollm3_cooldown \
  data.prepared_dir=/orcd/pool/005/henryac/<unique>/smollm3-cooldown-data \
  output_dir=/orcd/pool/005/henryac/<unique>/smollm3-cooldown-run
```

Submit `scripts/slurm/smollm3_cooldown.sbatch` with the current approved
account/QoS and explicit `PREPARED_DATA_DIR`/`OUTPUT_DIR`. Set
`CANARY_STEPS=2` for the first save/resume proof; leave it unset for the
22,222-update run. The launcher is offline and never installs packages.
Checkpoint markers fail closed on model/data/code/lock/topology drift. See the
full contract and acceptance gates in
[`docs/pipelines/smollm3-cooldown.md`](../pipelines/smollm3-cooldown.md).

## Add a new SFT task

Tasks subclass `Task` from `tasks/common.py`. Look at `tasks/gsm8k.py` as a reference.

```python
# tasks/my_task.py
from tasks.common import Task

class MyTask(Task):
    eval_type = "generative"   # or "categorical"
    size = 1000

    def __iter__(self):
        for example in self._load():
            yield {"prompt": example["q"], "answer": example["a"]}
```

Then register in `tasks/mixtures.py` if you want it included in an SFT mix. To eval against it, point `chat_eval` at the task name.

---

## Add a new evaluation

Pattern: `scripts/<name>_eval.py` + `configs/eval/<name>.yaml` + helpers in `steering/eval/<name>/` if needed.

1. Copy `scripts/eval/shuffled_conditioning_eval.py` as a starting template (compose-API entry, follows the codebase pattern).
2. Add a Hydra config under `configs/eval/<name>.yaml` with `defaults: [_self_]` and required `???` fields.
3. Add tests under `tests/<name>/` — at minimum a fast unit test that runs against a mocked model.
4. Add an entry to `docs/cli-reference.md` under "Evaluation."
5. If it needs API access, mark its tests with `@pytest.mark.api`.

For an LLM-judge eval, look at `steering/eval/harmfulness_judge.py` for the OpenRouter-backed judge pattern.

---

## Profile a slow step

```bash
# Generic CUDA profiling — capture 3 steps starting at step N (chrome trace output)
python -m scripts.train.conditional_encdec_train profile_trace=100  # captures steps 100-102

# Self-conditioning attribution and trace are separate runs. The step is the
# zero-based absolute optimizer step, including after resume.
python -m scripts.train.conditional_encdec_selfcond \
  base_checkpoint_dir=/abs/path/to/base-leaf \
  profile_timing=true profile_trace=null
python -m scripts.train.conditional_encdec_selfcond \
  base_checkpoint_dir=/abs/path/to/base-leaf \
  profile_timing=false profile_trace=10 profile_trace_steps=1

# AxBench-Base profiling
python -m scripts.benchmark.profile_axbench_base

# Stage 0c verification hot-path profile: stops after 50 _verify_candidate calls
STAGE0C_PROFILE_VERIFY_CANDIDATES=50 \
python -m scripts.benchmark.profile_stage0c_verification \
  stage=0c verification_cross_prefix_batch_size=1 ...

# Compare/roll back to the pre-schema-8 repeated-prefill algorithm.
# Use a fresh run_tag: shared and repeated checkpoints are incompatible.
python -m scripts.benchmark.profile_stage0c_verification \
  stage=0c verification_prefill_mode=repeated ...

# Encoder-decoder microbenchmark registry entry
python -m scripts.profile_progression_microbench.encdec_build_conditioning
```

Output goes to `out/profiling/`. Open chrome traces in [Perfetto UI](https://ui.perfetto.dev/). For end-to-end attribution across pipeline phases, see `docs/performance/profiling.md`.

When you find a hotspot:
- Annotate it with `torch.profiler.record_function("<name>")` so it shows up clearly in the trace.
- Compare backends (FA4 vs FA3 vs SDPA) with `attn_impl=fa3` etc.
- Check `compile=true` is on and `compile_static=auto` (the default) is using
  a static outer model when loader shapes are fixed. Packed encoder and
  cross-attention work uses separate dynamic graphs for the content-dependent
  token length.

---

## Debug a CUDA OOM

1. **Lower `device_batch_size` first**, keep `total_batch_size` constant. Gradient accumulation will preserve the effective batch.
2. **Disable FP8 temporarily** — `fp8=false`. FP8 has overhead in some shapes.
3. **Reduce `max_seq_len`** for a smoke run (e.g. 512 instead of 2048).
4. **Encoder-decoder specific**: smaller `encoder.depth` (try `0.25` instead of `0.5`), increase `cross_attention.frequency` (every 8th layer instead of every 4th), disable `feature_prediction` (`feature_prediction: null`).
5. **Inspect the actual peak**: set `PYTORCH_NO_CUDA_MEMORY_CACHING=1` for a faithful peak-memory reading (slow; use only for diagnosis).

If OOM happens *during* a known-good config, suspect a leak — reproduce with `gc_mem_threshold=0.5` (extraction) or `torch.cuda.memory_summary()` calls bracketing the suspect block.

---

## Debug packed-row CPU memory growth

`precompute_packed_rows` is CPU-heavy. Its bounded feature cache and bounded
compression queue can still expose allocator retention: process memory rises
across source-shard transitions, old anonymous pages move into swap, and
throughput eventually collapses even though the Python data structures are not
growing without bound. `torch.cuda.empty_cache()` does not help this case; the
large tensors and Arrow buffers are in host memory.

First attribute the memory to the exact producer process. `VmRSS + VmSwap` is
the useful per-process commitment; `smaps_rollup` distinguishes private
anonymous memory from shared libraries and filesystem page cache.

```bash
PACKED_ROWS_PID=$(pgrep -f 'scripts.preprocess.precompute_packed_rows' | head -n1)
awk '/^(VmPeak|VmSize|VmHWM|VmRSS|VmSwap):/ {print}' \
    "/proc/$PACKED_ROWS_PID/status"
awk '/^(Rss|Pss|Private_Dirty|Swap|SwapPss):/ {print}' \
    "/proc/$PACKED_ROWS_PID/smaps_rollup"
vmstat 1
swapon --show
```

If `Pss` is close to `Rss`, `Private_Dirty` dominates, and the same process's
`VmRSS + VmSwap` keeps increasing after multiple source-shard transitions,
retry with Arrow's system allocator and a small glibc arena cap. Both variables
must be present when the process starts:

```bash
ARROW_DEFAULT_MEMORY_POOL=system \
MALLOC_ARENA_MAX=4 \
uv run python -m scripts.preprocess.precompute_packed_rows \
    feature_dir=<feature_dir> resume=true compress_workers=1
```

PyArrow's default allocator is build-dependent; inspect it with:

```bash
uv run python -c 'import pyarrow as pa; print(pa.default_memory_pool().backend_name)'
```

The allocator variables are resource controls, not artifact-identity inputs.
Normal resume identity, packed/provenance completeness, atomic writes, and
manifest/checksum validation still apply. Do not delete a partial trailing
shard: `resume=true` keeps the contiguous complete packed + provenance prefix
and regenerates the first incomplete shard. Resume replays every kept packed
row to recover deterministic packing state, so choose the worker count before a
long replay when possible.

Start with one compression worker while establishing a stable memory plateau.
If memory is reclaimed or reused across source-shard transitions and swap I/O
stays quiet, test two workers; each worker permits another large shard assembly
to be in flight. Do not jump directly to high concurrency. Stop the test if the
process commitment keeps rising, `MemAvailable` approaches the host reserve, or
disk-backed swap I/O persists. The allocator override is a mitigation for
retention, not evidence that every memory-growth bug is allocator-only.
On hosts with zram and high `vm.swappiness`, a small `VmSwap` value can be an
intentional compressed-RAM trade; sustained reads from a disk-backed swapfile
are the stronger throughput warning.

`MALLOC_ARENA_MAX` is a glibc/Linux control and may be ignored on other
platforms. It can trade allocator parallelism for a smaller footprint, so
compare steady-state shard throughput as well as peak memory.

### Stream remote raw features and percentiles into local packed rows

Keep statistics, token overlays, and parquets under the logical local feature
tree, then provide a separate scratch directory. The percentile family is read
unconditionally from `<remote-prefix>/pct/<pct16>/`; no FUSE mount or local
percentile symlink publication is needed. Production must pin the full
1,400-shard raw and percentile inventories and should begin with one compression
worker:

```bash
uv run python -m scripts.preprocess.precompute_packed_rows \
  feature_dir=<production-l22-logical-feature-dir> \
  data_dir=<local-parquet-dir> \
  remote.enabled=true \
  remote.staging_dir=<dedicated-scratch> \
  remote.expected_shard_count=1400 \
  compress_workers=1 \
  profile=true \
  profile_summary_output=out/traces/packed_rows_remote.json
```

Before a full run, check that scratch has at least the two largest raw plus two
largest percentile bodies plus 1 GiB, and that the packed-output volume has room
for both train and validation outputs. Monitor `VmRSS + VmSwap`, output bytes/rate, and
`remote_transport.{downloaded_bytes,critical_path_wait_seconds,staging_high_water_bytes,
staging_high_water_bodies,decoded_selected_shards,
final_fence_status,status}` and the matching percentile transport counters under
`remote_transport.percentile`. `max_shards` limits packed parquets but does not
weaken either sealed full-family identity: only the selected prefix is downloaded
from each.

After interruption, rerun with `resume=true`; do not use
`resume_from_shard`. The existing contiguous packed+provenance prefix is replayed
and the first incomplete suffix is regenerated. Do not start a consumer while
the sibling `.sN.remote-packing-state.json`/`.sall...` marker exists. Completion
requires that marker to be absent and both train/validation manifests to carry
valid FP source certificates. Remote mode performs no upload or persistent body
cache, so publication remains a separate explicit operation. Re-running an
already completed artifact downloads no shard bodies (only the verified
percentile manifest control object) and repairs missing mirrors idempotently.

Packing completeness is enforced by the producer itself: empty input fails and
the final lookahead buffer is fully drained, so every loaded document is
consumed. Resume uses the same complete drain. Readers authenticate bodies,
provenance, manifests, and inventories; they do not rely on a dedicated
packing-version metadata marker.

---

## Resume an interrupted training run

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.train.base_train -- \
    depth=12 run=d12 resume_from_step=5000
```

`resume_from_step=N` loads the flat files
`model_NNNNNN.pt`, `meta_NNNNNN.json`, and
`optim_NNNNNN_rankR.pt` from
`$NANOCHAT_BASE_DIR/base_checkpoints/<tag>/`; there is no `step_N/`
subdirectory. The default tag is `d{depth}`; override with
`model_tag=<name>` if you used a custom one.

For encoder-decoder runs, `scripts.train.conditional_encdec_train` uses the same `resume_from_step=` knob — checkpoints live under `$NANOCHAT_BASE_DIR/conditional_encdec_checkpoints/`.

### Resume the full3B run after W31→W32 topology migration

Ordinary resume intentionally rejects a changed world size. For the canonical
full3B handoff, first run the offline migrator against the complete W31 flat
checkpoint leaf and the exact local packed/selection pair:

```bash
uv run python -m steering.tools.migrate_encdec_train_state \
    source_checkpoint_dir=<w31-checkpoint-leaf> \
    source_step=<NNNNNN> \
    packed_row_dir=<packed-row-dir> \
    selection_dir=<selection-dir> \
    destination_dir=<fresh-w32-checkpoint-leaf> \
    max_replay_rows=<operator-limit>
```

Dry-run is the default. Review the unchanged step, W31/W32 batch arithmetic,
`sqrt(32/31)` LR-scale ratio, replay boundary, exact repeated-row/token count,
and `skipped_rows: 0`. Then repeat the identical command with `dry_run=false`.
The destination must be the checkpoint leaf resolved by
`full3b_clustered_c2048_long_w32_bf16_b14`; the trainer does not search an
arbitrary migration output directory.

Execute mode installs exactly 32 optimizer shards plus the byte-identical model,
updated metadata, and `topology_migration_manifest.json`. Resume with the W32
training config and the same raw step:

```bash
torchrun --nnodes=4 --nproc_per_node=8 \
    --node_rank=<0..3> --master_addr=<host> --master_port=<port> \
    -m scripts.train.conditional_encdec_train -- \
    config_name=full3b_clustered_c2048_long_w32_bf16_b14 \
    resume_from_step=<NNNNNN>
```

Use the repository's production launcher to supply the concrete rendezvous
values while retaining total `WORLD_SIZE=32`. Before releasing the full
continuation, run the owner-gated 20-step canary and check loss
continuity, finite/sane gradient norms, and that the next complete checkpoint
contains exactly 32 optimizer shards. The migration's dataloader policy is an
audited replay from the previous packed-shard boundary, not an exact cursor
translation; do not proceed if the reported repeat exceeds the reviewed bound.

---

## Resume AxBench-BPB-Adv Faster-GCG candidate building

Rerun the same production `build_candidates` command with the same `run_tag`.
There is no checkpoint option: completed non-control Faster-GCG groups are
restored automatically from
`candidates/universal_gcg_checkpoint/manifest.json` and its bucketed group
shards. Startup logs `expected_groups`, `restored_groups`, `pending_groups`, and
the current runtime hash. Only the group that was in flight without an installed
shard is recomputed; GEPA reruns, and no-op/additive-direction rows are rebuilt.

If startup reports a manifest or shard mismatch, do not edit or delete the old
state. The error names the differing field/path: preserve the run and select a
fresh `run_tag`. A runtime-hash warning is different—it permits resume but means
byte-identical parity is guaranteed only for groups produced on the same numeric
runtime. A legacy completed run with no checkpoint directory remains readable;
re-running `build_candidates` recomputes Faster-GCG once and starts checkpointing
rather than treating old final Parquet files as partial state. A process already
running older code cannot gain this feature retroactively.

---

## Run a hyperparameter sweep

Sweep cells that enable feature prediction against a compact vocabulary need
precomputed packed rows with a completed `fp_target_source_integrity`
certificate. New packed-row production writes it automatically; certify an
existing family with `scripts.preprocess.validate_fp_target_source`. See
[encoder-decoder-training.md § Compact feature-prediction targets and source integrity](../pipelines/encoder-decoder-training.md#compact-feature-prediction-targets-and-source-integrity).

```bash
# Generate the full config grid + launch
bash scripts/launch_sweep.sh 4 --all --output-dir out/sweep_h200

# Faster variant (amortized startup)
python -m scripts.train.run_encdec_sweep_fast

# Analyze results once it's done
python -m scripts.eval.analyze_sweep
```

`launch_sweep.sh N` forks N processes onto N GPUs, partitioning configs by index. The amortized variant shares startup cost across many configs in one process.

---

## Build a clustered CORE/sample context cache

Enumerate the exact CORE/sample prompts, extract their Gemma SAE features, then
convert with the same mask, schema-v1 clusters, bucket count, and statistics file
as training. Prompt caches are delta-free: omit `delta_weights` from the positive
selection and set `feature_delta=null`.

```bash
uv run python -m scripts.eval.build_eval_context_cache stage=enumerate \
  evals=core,samples contexts_file=<contexts.jsonl> \
  context_parquet_dir=<context-parquet> model_max_seq_len=<N>

# Run steering.extract_features on <context-parquet>, then:
uv run python -m scripts.eval.build_eval_context_cache stage=convert \
  contexts_file=<contexts.jsonl> feature_dir=<extracted-context-features> \
  stats_path=<feature_statistics.safetensors> \
  stats_source_sha256=<exact-64-hex-sha256> output_dir=<cache-root> \
  n_sae_features=<N_SAE> num_buckets=<B> model_max_seq_len=<N> \
  seed=<real-training-seed> \
  'feature_mask={mode: whitelist, path: <accepted-feature-ids.txt>}' \
  'feature_clusters={path: <clusters.yaml>}' \
  feature_delta=null identity.feature_selection_hash=null \
  'feature_selection={distribution: geometric, mean: 12, std: 12, min_k: 1, max_k: 49}' \
  'negative_selection={distribution: geometric, mean: 3, min_k: 0, max_k: 15}'

# Decoder and extractor context lengths use different tokenizers. For the
# clustered L22/Gemma-3 production c1024 cache on one H200, extract with:
uv run python -m steering.extract_features \
  --config-name=extract_eval_context_features_gemma3_4bpt_l22_f65k_ci_nobos_c1024_h200 \
  data_dir=<context-parquet> \
  output_dir=<extracted-context-features>

uv run python -m scripts.eval.build_eval_context_cache stage=validate \
  output_dir=<cache-root> contexts_file=<contexts.jsonl> \
  stats_path=<feature_statistics.safetensors> \
  stats_source_sha256=<exact-64-hex-sha256> \
  n_sae_features=<N_SAE> num_buckets=<B> model_max_seq_len=<N> \
  seed=<real-training-seed> \
  'feature_mask={mode: whitelist, path: <accepted-feature-ids.txt>}' \
  'feature_clusters={path: <clusters.yaml>}' \
  feature_delta=null identity.feature_selection_hash=<derived-hash>
```

The purpose-specific extraction preset keeps `model_max_seq_len=1024` for
cache identity and evaluation, but uses `context_window=2048` on the Gemma
side. A c1024 RustBPE context can otherwise exceed the Gemma extraction
window. Always run the standalone `validate_eval_context_cache` gate below: it
fails if any RustBPE context token falls outside the extracted Gemma segment
spans.

`seed=<real-training-seed>` is required (strict, no silent default) for
clustered builds: both the cache's identity hash
(`steering/eval/cache_identity.py`) and its sampled content
(`build_feature_pipeline_from_cfg_block`'s RNG) are pinned to this seed, which
must equal the checkpoint's actual `seed` (the same value the packed-row
training selection uses) — omitting it raises before any cache I/O.
Unclustered builds ignore it.

Use the resulting root as `conditioning.context_cache_dir` and pin the full
derived hash in `conditioning.identity.feature_selection_hash`. Training
re-derives it from the top-level policy and fails before the loop if it
differs — including checking `conditioning.seed` matches the pinned hash, so
`conditioning.seed` must also be set in the training config (not just the
`build_eval_context_cache` CLI invocation above) whenever
`conditioning.feature_clusters` is set. On a failure, do not edit manifest
metadata: fix the mask/cluster/statistics/seed or selection inputs and
rebuild the hash-scoped cache directory. Missing prompt keys mean the
enumeration inputs changed and require enumerate/extract/convert again.

For the canonical issue-#768 production run, preflight the real artifact copies
before launch:

```bash
sha256sum \
  out/autointerp_filter/gemma3_4bpt_l22_f65k_ci_nobos_freqmask_n40x40_root_20260710_145543/manual_feature_filter_with_descriptions.yaml \
  out/artifacts/feature_clusters_two_stage_schema_v1.yaml \
  out/data/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base/context_independent/no_bos/feature_statistics.safetensors

uv run pytest -m slow \
  tests/steering/test_issue768_production_artifacts.py -q

uv run python -m scripts.preprocess.validate_fp_target_source \
  packed_row_dir=<derived-train-packed-dir> n_sae_features=65536

uv run python -m scripts.eval.build_eval_context_cache stage=validate \
  output_dir=out/data/eval_context_features/gemma3_4bpt_l22_clustered \
  contexts_file=<contexts.jsonl> stats_path=<feature_statistics.safetensors> \
  stats_source_sha256=dadf6d4f8d3b56ba1b40e0f49f4bb92bfb635c64f78a530f04f624b1d5773b5d \
  n_sae_features=65536 num_buckets=2 model_max_seq_len=1024 seed=42 \
  'feature_mask={mode: whitelist, path: <manual-filter.yaml>}' \
  'feature_clusters={path: <clusters.yaml>}' feature_delta=null \
  identity.feature_selection_hash=9b2085a6f814a6a63128ea186cfaa83a1369c6658a1bcdced1f5c97143d6434f
```

The artifact test pins the three file SHA-256 values, `V=2,074`, vocabulary
fingerprint, derived selection `e9e39573…`, base selection `6be4095a…`, prompt
cache `9b2085a6…` (D-seed migration, 2026-07-24: pinned to the real
`conditioning.seed=42` rather than the legacy hardcoded seed=0; the
pre-migration value was `63aeb505…`), and compact head shape. Training then
validates the complete selection/certificate/cache manifests at startup —
including re-deriving this same prompt-cache identity from the top-level
training config's own seed, so `conditioning.seed` must equal the trainer's
top-level `seed`. If clustered steering fails, fix or regenerate the named
source artifact; do not edit manifests. The trainer restores RNG, probe
offload state, and training mode on an ordinary evaluation exception, so the
failed checkpoint can be diagnosed and resumed after the artifact issue is
corrected.

---

## Validate BPB-Adv output contracts

Production BPB-Adv accepts only unclustered source-contract v6 and clustered
v7. Validate the agreed output root after producing or archiving runs:

```bash
uv run python -m scripts.eval.validate_axbench_bpb_adv_output_root
```

The command inventories every direct-child run, rejects missing or legacy
contracts, and fails if migration control-plane entries remain. Historical
v4/v5 and incomplete smoke outputs are retained in the dated issue-1232
archive outside the production root; they are not rewritten or deleted.

---

## Validate that data on disk is consistent

```bash
# CSR invariants on extracted feature shards
python -m steering.validate_features

# Token boundary overlays
python -m scripts.utils.validate_token_boundary_overlays

# Token boundary overlay validator regression test
uv run pytest tests/scripts/test_validate_token_boundary_overlays.py -q

# Eval context cache against live extraction code
python -m scripts.eval.validate_eval_context_cache

# BF16 vs FP32 numerical deltas (for precision regression checks)
python -m scripts.utils.verify_bf16_threshold
```

Run these after any change to extraction / overlay / packing code, especially before kicking off long training runs.
The token-boundary validator regression test creates its own temporary nanochat tokenizer cache.

---

## Add a quickstart for a new pipeline

Pattern that worked for the existing `*_quick.yaml` configs:

1. Inherit from the production config:
   ```yaml
   defaults:
     - <production_config>
     - _self_
   ```
2. Override only the knobs that matter for being fast: depth, iterations, batch size, eval frequency, save frequency (`-1` to disable).
3. Add a docstring at the top showing the canonical invocation (compose-API scripts use `config_name=foo`, `@hydra.main` scripts use `--config-name=foo`).
4. Reference it from `docs/cli-reference.md` and `README.md` if it's a top-line install-validation flow.
5. If you need a wrapper shell script, model it on `runs/extract_features_quick.sh`.
