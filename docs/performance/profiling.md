# Profiling SAE Feature Extraction

This guide covers profiling, benchmarking, and interpreting performance data for the SAE feature extraction pipeline.

## AxBench Stage 1 tokenizer construction

Stage 1b and Stage 1c use a process-local cache for the nanochat RustBPE
tokenizer. A normal profile may therefore show one cold-start
`nanochat.tokenizer.get_tokenizer` / `RustBPETokenizer.from_directory` /
`tiktoken.Encoding` construction, but those frames should not recur once row
assembly begins. Repeated tokenizer construction is a regression signal and
should be diagnosed with an underlying-loader call count before comparing
throughput. The cache is operational only; profiling must still verify the
unchanged tokenizer checksum and exact `n_prefix_nanochat_tokens` boundaries.

## Quick Start

### Self-conditioning startup trace (issue #1479)

`scripts.train.conditional_encdec_selfcond` emits an always-on, rank-0
`selfcond_startup {json}` trace. It is intentionally separate from
`torch.profiler` and the opt-in benchmark harnesses: timings use host-wall
`time.perf_counter()` and do not synchronize CUDA. The first completed
optimizer step is reported separately from the steady-state mean so graph or
backend warmup is visible without contaminating throughput comparisons.

Use the `kind=summary` record's `total_s`, `named_phase_s`, and
`unattributed_s` for a quick startup budget, then inspect `kind=phase` records
for checkpoint/model restore, SFT artifact/index work, alignment, and the
first optimizer step. `kind=event` records carry artifact counts and lazy
boundaries. These records are liveness and attribution diagnostics, not
synchronized GPU performance claims; use a CUDA-synchronized benchmark or
profiler when device-time attribution is required. The exact schema and lazy
validation boundary are documented in the
[self-conditioning pipeline guide](../pipelines/selfcond-post-training.md#startup-trace-and-timing).

### Self-conditioning live-loop profiling

`profile_timing=true` writes `selfcond_timing_rank<R>.csv` with synchronized
main-thread regions and rank/step/micro-step/slot columns. It perturbs CUDA
overlap and must not be used as the throughput arm. `profile_trace=<step>` plus
`profile_trace_steps=<count>` instead writes one CPU/CUDA Chrome trace per rank
with shapes, stacks, and all-thread collection enabled. The two diagnostics are
mutually exclusive, and the entire absolute-step trace window must execute in
the current fresh or resumed run.

Keep the first capture to one step: on-policy rollout repeats a model forward
for each live continuation slot, so trace retention scales with operations and
slot count. A real 100M, sequence-length-1024, batch-size-one H200 capture with
shapes, stacks, CUDA, and all-thread collection produced a 2,724,199,573-byte
trace and a 35,065,260-KiB process high-water mark. Treat trace mode as a
forensic arm: start with one rank and one step, provision host memory and disk,
and do not enable it across a large world size merely to measure throughput.
Use the rank-local files to distinguish a common kernel bottleneck from rank
skew, and compare them with a third run where both diagnostics are disabled.
Commands, file names, marker semantics, and the SFT pending-batch caveat are in
the [pipeline profiling section](../pipelines/selfcond-post-training.md#profiling-and-benchmark-arms).

### Activation-Collector Benchmark

```bash
uv run python scripts/benchmark/benchmark_activation_methods.py smoke=true
```

Use `smoke=true` for a CPU wiring/correctness run. Use the default
`smoke=false` only on a machine with the configured Gemma models and CUDA.

### pytest-benchmark

```bash
uv run pytest tests/steering/test_performance.py -m benchmark --model-scale=tiny
```

### Backend Comparison (TransformerLens vs nnterp)

```bash
uv run python scripts/benchmark/benchmark_activation_methods.py batch_size=16
```

### Chrome Trace

The extraction code is annotated with `torch.profiler.record_function`
regions. There is no checked-in standalone trace CLI; wrap
`steering.extract_features.process_shard` or the benchmark harness with
`torch.profiler` for ad-hoc traces and open the trace in
[Perfetto UI](https://ui.perfetto.dev) or `chrome://tracing`.

### TensorBoard Trace

Use the same `torch.profiler` wrapper with a TensorBoard trace handler when
you need TensorBoard output.

### Encoder-decoder startup trace

For the standalone conditional encoder-decoder trainer, startup observability
is emitted as structured INFO lines rather than a profiler trace. The trace
begins before Hydra composition and covers the expensive artifact/model/data
resolution seams through the first completed optimizer update. It is free of
new CUDA synchronizations, DDP barriers, artifact scans, and hash computation;
values are rank-0-local wall-clock timings.

The records have four stable prefixes:

| Prefix | Purpose |
|---|---|
| `encdec_startup_phase` | Named phase, `ok`/`skipped`/`error`, elapsed and cumulative seconds |
| `encdec_startup_data_path` | Final packed/assembled route and selection source, plus discovery/decision timing |
| `encdec_startup_first_step` | Invocation-local first update, exact legacy step `dt`, compile-configured flag |
| `encdec_startup_summary` | Completion/no-update/error status, first/steady metrics, and phase-seconds JSON |

Interpret `first_step_s` as cold-start cost: it includes the existing first
forward/backward/optimizer work and the post-optimizer device synchronization,
including any actual compile tracing. `compile_wrapper` measures only creation
of the `torch.compile` wrapper, so wrapper construction and first-use tracing
remain distinguishable. Steady-state is the mean of invocation-local updates
after the first whose numeric step is strictly greater than 10, matching the
trainer's existing ETA convention. A resumed invocation therefore reports its
own first update even when its checkpoint step is nonzero.

The route event is the authoritative runtime decision. `precomputed` plus
`selection_shards` is the exact packed/selection path; `precomputed` plus
`fused_base_csr` is packed rows with inline base-CSR selection; `assembled` plus
`live_feature_pipeline` is a live fallback. `feature_delta` is explicitly
reported when a non-splice delta makes discovered packed rows inapplicable.
Do not compare rank-local phase numbers as a distributed critical path without
an external synchronized measurement.

The checked-in production-shaped gate is opt-in because feature trees and model
weights are environment-specific. With a real full-size artifact and an
appropriate GPU allocation, run for example:

```bash
ENCDEC_STARTUP_PRODUCTION=1 \
ENCDEC_STARTUP_PRODUCTION_FEATURE_DIR=/path/to/real/features \
ENCDEC_STARTUP_PRODUCTION_CONFIG_NAME=conditional_encdec_4090 \
uv run pytest -m very_slow tests/scripts/test_conditional_encdec_startup_production.py
```

Set `ENCDEC_STARTUP_PRODUCTION_DATA_DIR` when the config does not resolve the
matching corpus automatically. The test performs one real update, validates the
complete phase set and route event (with optional CORE, sample, and steering
cadences disabled for this startup-focused run), and fails if the supplied
artifact/model cannot reach a completed first step. Without these explicit inputs it skips;
the deterministic model-backed parity test is not a production performance
claim.

### AxBench-BPB-Adv Faster-GCG shared-prefix scoring

Run the real eval CLI with `stage=build_candidates`, a fresh `run_tag`, and
`+profile.enabled=true`; do not reuse an immutable artifact tree across candidate
evaluation implementations or batch shapes. Shared-prefix scoring records nested
regions named `axbench_bpb_adv.candidate_eval.shared_prefix.{layout_grouping,
encoder_preparation,prefill,materialized_kv_copy,suffix_forward_bpb}`. Counters
record compatibility groups/prefills, full and suffix token positions, copied K/V
bytes, maximum source/destination cache dimensions, and effective candidate
device batch size. The algorithm counter is fixed to
`shared_prefix_kv`; there is no production mode switch.

For performance comparisons, use identical checkpoints, group inputs, seed,
attention configuration, and candidate order. Compare the committed test-only
full-sequence oracle with shared-prefix runs in fresh processes, warm up one
group, and measure at least three repetitions.
Report paired medians, a 95% bootstrap confidence interval, p95, peak reserved
memory as a fraction of physical VRAM, and post-group live-allocation behavior.
Candidate batch-size 32 is acceptable only after exact discrete-trajectory and
emitted-row parity, no OOM, peak reserved memory at most 80%, at least 15% paired
median `candidate_eval` reduction, and a time-ratio confidence-interval upper
bound below 1.0. This exact requirement applies to changing the production
shared-prefix batch shape; it is not a full-sequence-oracle parity requirement.

Treat the oracle comparison as a counterfactual search-quality study. In
addition to per-sample drift, report the oracle rank and BPB gap of each
shared-prefix choice on the same candidate pool, reciprocal rank, top-k overlap,
and the final retained trigger's improvement under both scorers. Exact legacy
argmin or trajectory equality is not required. Production acceptance requires
deterministic shared-prefix reruns, finite bounded drift, explicitly reviewed
search-quality evidence, at least 2.0x paired-median speedup (95% lower bound at
least 1.5x), p95 at least 1.5x, no monotonic live-allocation growth, and the same
80% memory ceiling.

### AxBench-BPB-Adv round-trip score-stage benchmark

Use `scripts.benchmark.benchmark_axbench_bpb_adv_roundtrip` for whole-stage
round-trip comparisons, including generation, target-to-probe handoff, probe
load, and probe scoring. The issue #1260 results and artifact map are recorded in
[`axbench-bpb-adv.md`](../evaluation/axbench-bpb-adv.md#round-trip-performance-characteristics-issue-1260-campaign).

Performance pairs must use fresh processes, the same pinned stratified subset
and subset hash, identical checkpoints and generation settings, and paired
repetitions reported by median. Keep each run in a separate benchmark directory
so checkpoint restore or warm process state cannot contaminate its mate. For an
attention-backend A/B, run the identical revision and subset once with the
default environment and once with `NANOCHAT_ATTN_IMPL=sdpa`; compare both the
wall/region profiles and score tables. Backend-induced numerical differences are
measured drift and must be reported, even when SDPA removes flex/Triton compile
and Dynamo-recompilation costs.

### Standalone ChatCORE ragged decode

For changes to `steering.eval.greedy_token_batch`, compare the real
`--chatcore-only --source sft` suite on one GPU and retain per-task scores, wall
time, peak allocated/reserved CUDA bytes, and the exact commit or complete source
manifest. Profile a
shorter 64-token generative arm separately so profiler overhead does not enter
the production timing. Ragged attention comparisons must cover full-prefix and
sliding-window masks, GQA when supported, exact K/V updates, and an explicit
BF16 output tolerance.

Triton cold start is part of standalone latency. Measure one candidate arm with
a fresh `TRITON_CACHE_DIR`, then a warm arm, bracketing them with unchanged
references. Do not report only the warm-cache result or pool compilation time
into CUDA operator time. Completion bytes and RNG order need not match for this
suite, but every task score and any numerical drift must be reported.

`uv run python -m scripts.benchmark.benchmark_chatcore checkpoint_dir=... step=999
output_dir=... grading=candidate` runs the six-task, 50-problem-per-task suite
three times on one CUDA GPU. Use `grading=reference` for the original nanochat
HumanEval result-manager path. Both variants use the same model, generation,
code-execution worker, limits, and grading rules. Interleave separate reference
and candidate processes and use fresh `TRITON_CACHE_DIR` and
`TORCHINDUCTOR_CACHE_DIR` directories to distinguish first-run latency from warm
repetitions. Keep the GPU free of other work throughout each measurement.

The output includes source/input hashes, scores, exact completion and grading
records, synchronized wall times, and CUDA memory peaks. `profile=true` adds a
separate 64-token generative trace and host profile after the timed suites.
`record_logits=true` records full-vocabulary logits for four rows every 64
ragged decode calls; run those numerical audits separately, since their CPU
copies instrument the measured path. The
[HumanEval grading experiment](chatcore-grading-20260920.md) records the first
optimization found with this harness.
The [residual-blend experiment](chatcore-residual-blend-20260920.md) compares
separate source snapshots with `grading=candidate` in both arms, preserving the
grading optimization while qualifying a decoder change. The harness records
the GPU UUID and Triton helper hashes; an absent helper in an older reference
snapshot is recorded as `null`.
The [bounded parallel grading experiment](chatcore-parallel-grading-20260920.md)
uses the same harness in parent/candidate snapshots and captures both scalar
and batch program hashes/verdicts. Independent scalar counts, batch sizes, and
worker bounds expose route selection. CPU affinity/allocation and lifetime
parent/child RSS are recorded separately from CUDA memory peaks; lifetime RSS
does not measure an exact aggregate grading-memory peak.

The [FP8 cache-compaction experiment](chatcore-cache-compaction-20260920.md)
uses a resident ABBA/BAAB screen followed by fresh-process qualification with
the unchanged canonical harness. Both candidate process geometric means and
medians must beat both reference processes. Complete manifests bind the new
helper and tests, while separate excluded audits check exact completion/grade
bytes and sampled logits. Its report distinguishes the qualified runtime/test
snapshot from later documentation-only packaging and reports cold reserved
memory separately from warm peaks.

The [4,096-byte compaction-tile follow-up](chatcore-cache-compaction-tile-20260921.md)
retains that protocol and changes only the kernel's byte tile. Fresh-process
warm complete-suite latency falls by 0.79%; both candidate processes beat the
faster reference by 0.45–0.60% across geometric means and medians. The report
retains all 22 calls and documents reference drift that affects the magnitude.

### AxBench-Base Stage 0c process memory

Stage 0c completion heartbeats are emitted after the active concept's
candidate list, API definitions, Python garbage, and cached CUDA allocator
blocks have been released. On Linux the heartbeat reads the inexpensive
`/proc/self/status` summary and includes:

| Field | Meaning |
|---|---|
| `current_rss_kb` | Current resident set (`VmRSS`) |
| `anonymous_rss_kb` | Current anonymous resident memory (`RssAnon`) |
| `swap_kb` | Process swap (`VmSwap`) |
| `anonymous_total_kb` | `RssAnon + VmSwap`, the resident-plus-swapped anonymous footprint |
| `peak_rss_kb` | Process resident high-water mark (`VmHWM`) |

Available values are also written as machine-readable profiler counters named
`stage0c.memory.<field>`, tagged with `candidate_index` and
`candidates_visited`. Missing, restricted, or malformed procfs data falls back
to the fields available from `psutil` and `resource`; instrumentation never
fails the evaluation. The heartbeat intentionally does not read
`smaps_rollup`, because walking VMAs at every concept boundary is more
expensive.

Interpret `anonymous_total_kb` after the initial shard/model load as the
candidate-retention signal. It may reach the largest active-concept high-water,
but a concept-linear slope after cleanup indicates that another strong
reference is retaining concept state.

### AxBench-Base Stage 1 post-provider QC

Provider calls run in the existing request-chain worker pool, while parsing,
Gemma/SAE QC, retries, tracker/checkpoint updates, and row assembly stay on the
caller thread. When `synth.max_concurrent > 1` and `qc_batch_size > 1`, Stage
1b handles a deterministic frontier of up to `qc_batch_size` stable chain
indices at once. Higher-index provider requests continue concurrently, but a
fast response cannot change the frontier's GPU batch membership. It records
`stage1b.compute_qc_percentiles_batch` for rich continuations and
`stage1b.avoid_strict_zero_qc_batch` for the combined natural and LLM-avoiding
pass. Parallel Stage 1c records `stage1c.strict_zero_qc_batch`. The `rows`
attribute is the number of continuations that reached that GPU pass; provider
errors, parser refusals, and no-token rows excluded before forwarding do not
inflate it.

The shared scheduler retains the per-request lifecycle counters and adds:

| Counter | Meaning |
|---|---|
| `synth.chain_pool.ready_results_observed` | Distinct completed futures observed for the first time; each result contributes at most once |
| `synth.chain_pool.batch_results_handled` | Results sent through this caller callback, capped by `qc_batch_size` |
| `synth.chain_pool.completed_results_backlogged` | Distinct completed futures that waited behind an earlier stable-index frontier; each result contributes at most once |

A frontier waits for all of its selected requests, so a slow low-index provider
response can delay QC without making the batch shape timing-dependent. The
effective post-provider cap is
`min(qc_batch_size, max(1, synth.max_concurrent))`; an effective cap of one is
the singleton-shaped reference through the same application state machine.
Small BF16 drift across different configured effective shapes is expected,
including possible decision changes next to a QC threshold. For GPU-memory
pressure, lower `qc_batch_size` before a fresh Stage 1b transaction. That knob
is semantic identity: changing it while an authoritative Stage 1b partial
exists is rejected, so an OOMing partial must either resume under the original
value or be explicitly restarted under a smaller one. Stage 1 does not catch
an OOM and silently retry with a different shape. Compare
accepted pairs, attempts per acceptance, provider errors/429s, GPU utilization,
and peak allocated/reserved CUDA memory when evaluating throughput.

### AxBench-Base Stage 1b seed identity

The default regression gate is structural rather than timing-sensitive. The
`stage1b_seed_rows_v2` helper hashes exactly 18 frames and performs 72
`hashlib.update` calls regardless of row, token, or feature-element counts. It
operates on `memoryview(tensor.numpy()).cast("B")`; tests prohibit regressions
to per-row/per-element framing and cover the frozen one-row digest. Run the
non-gating throughput record with:

```bash
uv run pytest tests/axbench_base/unit/test_seed_contract.py -m benchmark
```

For production acceptance, warm the pages through the ordinary strict seed
loader, time only the bulk identity over the same loaded tensors, and compare
separate matched subprocesses for ordinary versus identity-returning full
loads using `/usr/bin/time -v`. Record the digest and `Maximum resident set
size`; the rich-load delta must remain below 128 MiB. Also verify the legacy
digest once on reconstructed rows when testing migration, rather than using the
old serializer as a current-path benchmark.

On the 23,953-row production train artifact (142,093,387 feature IDs and
magnitudes; 11,177,194 prefix tokens), the original legacy measurement was
151.11 s and the final acceptance run measured 158.31 s. The bulk identity
measured 0.57 s on warm pages, a 278x speedup against the acceptance run, and
produced `c271d1b2fe99e2871c3eed71e7f47cbf63f84bed9b85a0988dbdb2af69daca9d`.

## Activation Collection Backends

The pipeline supports two backends for GPT-2 forward pass:

| Backend | Description | Speedup | Output Match |
|---------|-------------|---------|--------------|
| `transformerlens` | TransformerLens `stop_at_layer` | 1.6x vs full forward | Bit-for-bit identical |
| `nnterp` (auto) | HuggingFace GPT-2 + nnterp `tracer.stop()` with TL weight transforms | 2.9x vs full forward | allclose(1e-3), cosine_sim=1.0 |

The `auto` backend (default) uses nnterp on CUDA if available, falling back to TransformerLens.

Configure via Hydra: `backend=auto | nnterp | transformerlens`

### Why nnterp is faster

1. **HuggingFace forward pass** has no hook infrastructure overhead (~2x faster per layer)
2. **`tracer.stop()`** halts after layer 8, skipping layers 9-11 (saves ~33% of transformer compute)
3. **Weight transforms** from TransformerLens (fold_ln, center_writing_weights) applied at load time ensure SAE-compatible activations

## record_function Labels

The extraction pipeline is annotated with `torch.profiler.record_function` for zero-overhead tracing:

| Label | Location | What it measures |
|-------|----------|-----------------|
| `gpt2_forward_batched` | `batched_forward` | GPT-2 forward pass (batched, early-stop) |
| `sae_encode_batched` | `batched_forward` | SAE encoder on batched activations |
| `masked_mean_pool` | `batched_forward` | Masked mean pooling across token positions |
| `gpt2_forward` | `_get_sae_activations` | GPT-2 forward pass (single doc, with cache) |
| `sae_encode` | `_get_sae_activations` | SAE encoder (single doc) |
| `tokenize` | `extract_document_features` | GPT-2 tokenizer encoding |
| `read_row_group` | `process_shard` | Parquet row group I/O |
| `save_shard` | `process_shard` | CSR assembly + validation + torch.save |

### Typical Time Distribution (CUDA, batch_size=16)

- `gpt2_forward_batched`: ~55-70% (nnterp backend reduces this)
- `sae_encode_batched`: ~15-25%
- `masked_mean_pool`: ~2-5%
- Sparse extraction: ~3-5%
- Tokenize / I/O: ~2-5%

## Running Benchmarks

### Activation Method Comparison

```bash
uv run python scripts/benchmark/benchmark_activation_methods.py batch_size=8 n_iters=15
```

### pytest-benchmark

```bash
# Basic run (tiny model, CPU)
uv run pytest tests/steering/test_performance.py -m benchmark -v --model-scale=tiny

# CUDA benchmarks (requires GPU)
uv run pytest tests/steering/test_performance.py -m "benchmark and slow" -v

# Save baseline and compare
uv run pytest tests/steering/test_performance.py -m benchmark --benchmark-autosave --model-scale=tiny
uv run pytest tests/steering/test_performance.py -m benchmark --benchmark-compare --model-scale=tiny

# Skip benchmarks in normal test runs
uv run pytest tests/ --benchmark-disable
```

### Benchmark Classes

| Class | What it benchmarks |
|-------|-------------------|
| `TestExtractDocumentPerformance` | `extract_document_features` across pooling modes and text lengths |
| `TestBatchedForwardPerformance` | `batched_forward` at batch sizes 1, 4, 16 |
| `TestCUDAPerformance` | Real-model CUDA benchmarks (requires GPU) |
| `TestGetSAEActivationsPerformance` | `_get_sae_activations` at different sequence lengths |
| `TestProcessShardPerformance` | `process_shard` on a 20-document synthetic parquet |
| `TestComputeStatisticsPerformance` | `compute_statistics` on 3 shards x 100 documents |

### Token-Boundary Precompute (issue #529)

`scripts/preprocess/precompute_token_boundaries profile=true` writes a JSON sidecar
(`profile_summary_output`, default `out/traces/tok_boundary_profile.json`) alongside the
existing chrome trace:

```json
{
  "manifest_pre_pass": {"seconds": 0.0, "bytes_read": 0, "cache_hit": true, "n_shards_hashed": 370},
  "worker_pool": {"spawn_count": 1, "total_init_seconds": 2.1},
  "per_shard": [{"stem": "...", "load_features_seconds": 9.1, "tokenize_seconds": 7.2,
                 "offset_paths": {"fast": {"count": 0, "total_ms": 0.0},
                                  "slow": {"count": 0, "total_ms": 0.0},
                                  "small_n": {"count": 370, "total_ms": 12.4}},
                 "worker_compute_seconds": 7.3, "map_wait_seconds": 0.0,
                 "scatter_seconds": 0.1, "overlay_save_seconds": 1.1,
                 "load_bytes": 8500000000, "overlay_bytes": 240000}],
  "totals": {"wall_seconds": 5400.0, "n_processed": 370, "n_skipped": 0}
}
```

`offset_paths` reports the per-document segmentation-path counts and cumulative milliseconds
for `fast`, `slow`, and `small_n`. `worker_compute_seconds` is the sum of the per-document
worker wall times. `map_wait_seconds` measures result availability waits in the process pool
(zero for the sequential branch), while `scatter_seconds` covers copying each document's
boundaries into shard-order tensors. These counters are aggregated for every run; only this
sidecar write is gated by `profile=true`.

In compact remote mode, `load_features_seconds` and `load_bytes` measure the
stable compact-shard load in the caller; boundary computation retains the same
timing fields and shared code path as local raw mode. `remote_transport`
reports sealed raw identity metadata, compact transfer count/bytes/digest,
compact staging high-water, manifest identity, and the raw/compact/manifest
final fence. It contains no direct raw-body transfer because the consumer never
performs one.

Benchmark the one-time publisher separately from repeated compact consumers.
Publisher reports should cover raw download/decompression, compact encode and
upload, peak RSS, staging high-water, wall time, and idempotent-rerun behavior.
Consumer reports should cover manifest authentication, compact transfer/load,
the ordinary parquet/tokenization/boundary/save phases, peak RSS, and wall time.
Keep observed subset measurements separate from extrapolated full-family values.

`scripts/benchmark/benchmark_token_boundaries.py` runs named variants (the defaults are `cold`
then `warm` on the same `feature_dir`) in sequence and emits a Markdown table of the manifest
pre-pass + per-stage median/p90. The `warm` row shows the base-hash cache win:
`bytes_read=0`, `cache_hit=true`. See
[feature-extraction.md](../pipelines/feature-extraction.md#token-boundary-overlay-precomputation).

### Percentile Precompute (issue #1269)

`scripts/preprocess/precompute_percentiles profile=true` writes a JSON sidecar
(`profile_summary_output`, default `out/traces/pct_profile.json`) after the run. In a
multi-rank run, the default path is suffixed with `.rank{rank}` so each rank keeps its own
records; an explicit output path is used as given.

```json
{
  "config": {"feature_dir": "...", "device": "cpu", "resume": false,
             "force": false, "zstd_level": 12, "world_size": 1, "rank": 0},
  "per_shard": [{"stem": "shard_00000", "n_values": 900000000,
                 "load_seconds": 120.0, "load_bytes": 8500000000,
                 "compute_seconds": 240.0, "save_seconds": 75.0,
                 "overlay_bytes": 240000000}],
  "totals": {"wall_seconds": 5400.0, "n_processed": 370, "n_skipped": 0}
}
```

`load_seconds`/`load_bytes` cover the base feature shard read, `compute_seconds` covers
the direct-gather percentile calculation, and `save_seconds`/`overlay_bytes` cover the
compressed overlay write. Upload and post-write verification happen outside this producer;
their cost appears only as inter-shard gaps in the wall-clock totals.

The CUDA producer delta-decodes storage-width feature IDs in complete-segment chunks and
keeps exactly one CPU chunk ahead of the current CUDA lookup. It writes returned bytes into
one preallocated output tensor. Consequently, `load_seconds` includes the summed chunk-decode
work and `compute_seconds` includes the summed lookup work, but those two phase counters can
overlap and need not add up to wall time. CPU mode retains the synchronous full-decoder path.

With `remote.enabled=true`, the local schema above is intentionally unchanged
and a separate schema-v1 remote sidecar is written (default
`out/traces/pct_remote_profile.json`). Its top-level fields are `mode: remote`,
`config`, `inventory`, `staging`, `per_shard`, and `totals`. `inventory`
records sealed count, bytes, and digest. `staging` records admission required,
observed free/recoverable bytes, and observed stable-file high-water bytes/body
count. Each shard records source identity/size and downloaded-vs-reused
disposition, whether computation occurred, whether upload was scheduled and
later verified, or whether an orphan payload was adopted, plus value count and
elapsed wall time. `totals.status` is `complete` or
`failed` and includes wall time and processed-shard count.

For CUDA remote runs, `config.decompression_lookahead_enabled` records whether
the one-source preparation pipeline was selected. `staging` also records its
host-memory requirement and the admission-time host/cgroup headroom. The source
worker's download, hashing, and zstd time overlaps current-shard compute, so a
shard's `elapsed_seconds` is wall time rather than a sum of exclusive phases.

Transfer, Xet verification, computation, and upload overlap, so per-shard wall
times are descriptive and are not additive. The stable-file high-water does not
measure OS page cache, HF/Xet caches, wire bytes, or atomic temporaries that
exist entirely between observations. Use OS network and filesystem telemetry
for those acceptance gates. An incomplete resume still transfers every raw
source body for the schema-v2 base hash even when receipts skip most compute and
upload work.

### Feature Statistics

`steering.compute_statistics profile=true` writes an atomic JSON timing sidecar
(`profile_summary_output`, default
`out/traces/compute_statistics_profile.json`) without changing the reducer or
statistics artifact. This instrumentation targets the exact reducer and therefore
requires `streaming=true`:

```bash
uv run python -m steering.compute_statistics \
    --config-name compute_statistics_gemma3_l22_f65k \
    profile=true \
    profile_summary_output=out/traces/statistics_baseline.json
```

The schema-v2 sidecar records the resolved numeric runtime, whether bounded
read+zstd prefetch actually ran (`config.read_zstd_prefetch_enabled`), whole-run regions, aggregate
shard-phase seconds, per-shard input bytes/activation counts/timings, and
end-to-end throughput. Per-shard phases are:

| Phase | Timed work |
|---|---|
| `load` | critical-path shard load: complete read + zstd + safetensors decode when sequential; wait for prepared bytes + safetensors decode when read+zstd prefetch is active |
| `validate` | base-shard invariants, metadata initialization, feature-ID bounds |
| `exact_counts` | Float16-key unique/count reduction on the resolved device, sparse result transfer when CUDA is used, and CPU dense-table update |
| `moments` | CPU or CUDA occurrence/sum/squared-sum/min/max reduction, including transfers |
| `segment_stats` | exact segment/document weighted-distribution updates |
| `bookkeeping` | corpus totals and p99 subsample copy |
| `checkpoint` | checkpoint-state assembly plus durable generation commit when due |

`run_regions_seconds` separately exposes preflight, checkpoint restore, count-table
allocation, complete pass-1 wall time, complete finalization, exact magnitude
finalization, output
inventory fencing, serialization, and checkpoint cleanup. Regions can be nested:
`statistics.pass1.wall` contains the per-shard phases and pass-1 setup/finish, so
do not add it to those phase totals. Likewise, `statistics.finalize.wall`
contains `statistics.finalize.exact_magnitudes`.

For an optimization comparison, keep the shard inventory, revision, device,
GPU chunk size, Torch thread count, and checkpoint cadence identical. Use fresh
processes and report end-to-end wall time plus per-shard phase median/p90; the
sidecar throughput fields are descriptive, not a substitute for paired repeated
runs. The first parallelism experiment should be chosen from measured phase
shares rather than assumed I/O or GPU bottlenecks.

With `remote.enabled=true`, the otherwise unchanged schema additionally carries
`remote_transport` (`schema_version: 1`). It records sealed full/statistics
inventory counts, bytes, and digests; excluded identity; admission/free/
recoverable bytes; per-shard schedule, verified-ready, consumer-request, and
consumer-ready timestamps; disposition, transfer/wait durations, attempts,
retries, and materialized bytes; verified downloaded/reused totals; staging
byte/body high-water; cleanup/join duration; critical-path wait; and
post-bootstrap overlap.

`materialized_bytes` is bytes observed in local attempt targets, including
partial/retried attempts, not network-wire traffic. Measure OS network RX
separately for acceptance. With read+zstd prefetch, the background preparation
duration and remote transfer/wait detail can overlap reducer phases and are not
additive; `load` reports only the remaining main critical path, and top-level
wall time is authoritative. Overlap fraction is
`sum(max(0, transfer - consumer_wait)) / sum(transfer)` after bootstrap.

Successful sidecars use `remote_transport.status=complete`. Any remote-mode
`BaseException` writes an atomic partial sidecar with `status=failed` and
terminal stage/type/message/index/path when available. A secondary sidecar
failure is attached to, and never replaces, the computation failure. Local
failure behavior remains unchanged and emits no partial sidecar. As before, a
profile-write failure after successful statistics publication raises while
retaining the valid artifact.

### Packed-Row Precompute (issue #531)

`scripts/preprocess/precompute_packed_rows profile=true` runs the full (`max_shards`-bounded)
packing and writes a JSON sidecar (`profile_summary_output`, default
`out/traces/packed_rows_profile.json`). Packing is one continuous stream (rows span base-shard
boundaries), so per-stage costs are run-aggregate; the save side is per output shard:

```json
{
  "manifest_pre_pass": {"seconds": 0.0, "bytes_read": 0, "cache_hit": true, "n_shards_hashed": 370},
  "stage_totals": {"parquet": 0.0, "lookup": 90.0, "tokenize": 40.0, "seg_to_raw": 60.0,
                   "truncate_assert_append": 5.0, "pack": 20.0, "csr_build": 45.0},
  "row_build_detail": {"row_to_csr_seconds": 39.0, "row_provenance_seconds": 6.0,
                       "build_segments_seconds": 12.0, "segment_boundaries_seconds": 3.0,
                       "feature_concat_seconds": 20.0, "feature_ids_concat_seconds": 3.0,
                       "percentiles_concat_seconds": 4.0, "feature_counts_seconds": 1.0,
                       "row_tensor_finalize_seconds": 4.0},
  "base_shard_loads": {"count": 369, "total_seconds": 9400.0},
  "startup": {"feature_index_scan_seconds": 27.1},
  "feature_index_loads": [{"phase": "resident_load", "shard_stem": "shard_00000",
                            "total_seconds": 24.8, "compressed_bytes": 1929000000,
                            "decompressed_bytes": 6100000000, "n_segments": 1810000,
                            "n_features": 921000000, "n_docs": 83000,
                            "detail_seconds": {"compressed_read": 1.1, "zstd_decompress": 4.8,
                              "safetensors_deserialize": 0.2, "tensor_upcast": 1.7,
                              "feature_id_delta_decode": 2.9, "percentile_overlay": 1.4,
                              "token_boundary_overlay": 0.3, "doc_index": 11.5}}],
  "compression": {"backpressure_seconds": 380.0, "max_inflight_shards": 2},
  "per_output_shard": [{"shard_idx": 0, "n_rows": 16384, "n_segments": 282399,
                        "n_features": 1784766275, "assemble_seconds": 1.2,
                        "save_seconds": 74.5, "worker_seconds": 76.0,
                        "queue_wait_seconds": 66.2,
                        "save_bytes": 1537696820}],
  "resume": {"mode": {"train": "none", "val": "none"}, "replay_seconds": 0.0,
             "rows_replayed": 0, "checkpoint_seek_seconds": 0.0,
             "checkpoint_write_seconds": 0.0},
  "totals": {"wall_seconds": 5400.0, "total_rows": 6850000, "rows_per_sec": 70.0, "n_shards_written": 418}
}
```

`base_shard_loads.total_seconds` is the lookup-visible critical-path delay, including any wait
to join an in-flight prefetch. `feature_index_loads` instead records the work performed by the
scan/load thread: compressed read, zstd, safetensors decode, dtype restoration, feature-ID
delta decode, base postprocessing, percentile/token overlays, validation, and eager document
index construction. Those load details can overlap packing and therefore must not be added to
wall time. The startup scan is separate because it validates the first shard with the generic
full loader before the packed-row skip-magnitudes index begins resident loads.
`stage_totals.csr_build` remains the aggregate around both row operations;
`row_build_detail` separates packed CSR conversion from provenance construction, then divides
`row_to_csr_seconds` into segment routing, boundary derivation, feature concatenation,
and final dtype packing. The nested details are already contained by their parent
and must not be added to it when reconstructing wall time. `feature_concat_seconds` is further
split into feature-ID concatenation, percentile concatenation, and per-segment feature-count
construction.

`compression.backpressure_seconds` and each output shard's `queue_wait_seconds` measure how
long the serial producer is blocked on the bounded compression semaphore;
`max_inflight_shards` verifies the configured memory bound was actually exercised. This
distinguishes compression that is merely expensive in a worker from compression that delays
the producer. Per-shard `save_seconds` covers packed-payload serialization/compression/write;
`worker_seconds` additionally contains assembly, provenance save, max-dimension reduction, and
the dimension companion, matching the interval for which the semaphore permit is held.

The bounded local benchmark requires `data_dir` and creates a fresh private feature tree for
every variant. It links exactly the bounded raw/percentile/token inputs, computes their base
content hash once, rewrites private schema-v2 overlay manifests for that bounded family, seeds a
matching private stat cache for every variant, stat-fences every source before and after every
run, and writes all packed products below the benchmark run root. This avoids both production
writes and an irrelevant full-family rehash during every bounded variant. It publishes a report
only after exact SHA-256 equality of train/val
packed shards, provenance, dimension companions, and timestamp-normalized manifests against
the first variant. Inode/timestamp fingerprint caches are derived accelerators and are excluded
from parity. Variants that intentionally change output bytes (for example `zstd_level`) are
therefore rejected; use it for output-invariant implementation or worker-count comparisons.
The report includes RSS, and PSS/swap where the host exposes them.
The first row is labeled `reference`; only subsequent variants that passed the complete artifact
digest comparison are labeled `exact`. A one-variant run therefore measures resources without
claiming cross-variant parity. For example:

```bash
uv run python -m scripts.benchmark.benchmark_packed_rows \
  feature_dir=out/data/features/... \
  data_dir=$NANOCHAT_BASE_DIR/base_data \
  segment_size=128 sae_width=65k context_len=2048 max_shards=4 \
  report_md=out/bench/packed_rows.md \
  'variants=[{name: serial, overrides: {compress_workers: 1}}, {name: parallel, overrides: {compress_workers: 2}}]'
```

Packed preprocessing now has one unconditional encoded-percentile path. The
former float round trip was removed only after recursive artifact digests matched
exactly on the bounded corpus. Continue using exact artifact parity for future
output-invariant changes. A faster producer can increase
`compression.backpressure_seconds` even while reducing wall time, because the
unchanged save worker becomes the new bottleneck; interpret the row-to-CSR and
backpressure timings together.

On bounded four-shard c1024 runs (103,732 rows), removing the encoded carrier's
unused bucket field and disabling bucket collection reduced `seg_to_raw` from
14.50s to 5.21s (64.1%). The corresponding one-worker wall times were 313.0s and
247.6s, but differing shard-load/save timings mean that end-to-end delta is not
an isolated allocation measurement. On back-to-back runs of the final no-field
implementation, two save workers reduced wall time from 247.6s to 186.1s, raised
throughput from 418.9 to 557.4 rows/s, and reduced compression backpressure from
136.1s to 48.4s. PSS rose from 52.19 GiB to 53.97 GiB and neither run swapped.
All runs had the same sealed input-inventory SHA-256 and complete artifact-map
SHA-256. These are single fresh-process samples, not a default-setting decision:
repeat at production shard count/density and monitor PSS/swap before raising
concurrency, because each worker can retain another large shard assembly.

The generated run directory contains `input_inventory.json`, bounded-content identity/timing in
`run_metadata.json`, one profile JSON and artifact digest map and resolved config per variant,
isolated variant feature trees, and the report.
RSS/PSS/swap are absolute peaks for the benchmark process tree. For the cleanest memory number,
run one heavyweight variant per CLI process; a multi-variant parity sweep is intentionally
sequential in one interpreter and a prior run may leave allocator-reserved pages resident. See
[feature-extraction.md](../pipelines/feature-extraction.md#packed-row-precomputation).

With `remote.enabled=true`, the local fields remain and `remote_transport` adds
the sealed raw inventory count/bytes/SHA-256, joint admission required/free bytes under
`joint_admission_required_bytes` and `joint_admission_observed_free_bytes`,
downloaded/materialized bytes, transfer and critical-path wait seconds, staging
high-water bodies/bytes, header processing totals, selected decode count, final-fence status,
resume classification, contiguous kept train/validation shard counts, and
terminal status/error. `manifest_pre_pass.cache_hit` is always false;
`seconds`, `bytes_read`, and `n_shards_hashed` are all zero because remote
finalization performs no manifest hash pass. Inventory size/count remain under
`remote_transport.inventory_*`. A failure sidecar is diagnostic only and never
turns an unauthenticated packed output into a success.

`remote_transport.percentile` repeats the transport payload shape for the
checksum-addressed percentile subfamily: its own inventory count/bytes/SHA-256,
downloaded/materialized bytes, transfer/wait timings, staging high-water,
per-body attempts, list timings, final fence, and terminal error. The parent
The parent `admission_required_bytes` and `admission_observed_free_bytes` remain
the raw transport's own guard. The `joint_admission_*` pair describes the
two-largest-raw plus two-largest-percentile plus 1 GiB envelope. The nested
percentile transport's admission fields describe only that stream's internal guard.

`remote_transport.bodies_processed` counts bodies consumed by this invocation;
`remote_transport.cursor_position` is the absolute next position in the sealed inventory.
They differ after checkpoint seek resume.

When profiling is enabled, `remote_transport.per_body` additionally reports transport wait,
header validation, selected classification, and activation time. Selected bodies'
decompression and index construction use the same top-level `feature_index_loads` breakdown as
local mode. `transport_wait_seconds` may overlap the transport's download counters.

`remote_transport.per_body_transport` records every ATTEMPTED download (not just consumed
bodies): position, `status` (`ok`/`failed`) with the failure's exception type under `error`,
the successful attempt's download time, cumulative SHA-256/Xet hash time (re-verification
passes after tolerated identity drift — timestamps, or a transiently stale size stat on the
download path — accumulate here), total transfer time, attempt count,
and materialized bytes — so a failure sidecar still carries the failing body's record.
For every `download_verified_object` consumer, including token-boundary,
percentile, statistics, preselection, and packed-row transport, SHA-256 is
skipped when no committed content SHA-256 is supplied; `sha256_seconds` is then
`0.0`, while size and Xet identity against the sealed inventory remain the
authoritative content check. Consumers with an explicit committed content
SHA-256 continue to verify it.
`total_attempts` sums attempts over all attempted downloads. `snapshot_list_seconds` and
`fence_list_seconds` time the two inventory listings, while `list_calls` records how many ran.
Each `per_body` header and activation wall timer has a matching
`*_cpu_seconds` field. Each `feature_index_loads` entry similarly carries
`detail_cpu_seconds`, keyed exactly like `detail_seconds`. These are CALLER-THREAD CPU times:
wall minus thread-CPU includes both genuine waiting (I/O, lock, GIL) and any compute PyTorch
runs on intra-op worker threads, so treat the difference as "not this thread's CPU", not as a
pure stall measurement.

### Feature-Shard Splice

`scripts/preprocess/splice_feature_shards.py profile=true` writes a JSON sidecar
with both coarse `phase_seconds` and finer `detail_seconds`. The coarse phase
keys remain stable for dashboards and regression tests; the detail keys split
source loading, recombination gates, derived-shard construction, provenance, and
the save path into profiler-friendly substeps.

Use the bounded benchmark harness for comparable measurements without writing
inside the production feature directory:

```bash
uv run python -m scripts.benchmark.benchmark_splice_feature_shards \
  feature_dir=out/data/features/.../base \
  data_dir=$NANOCHAT_BASE_DIR/base_data \
  max_shards=4 \
  cut_fraction=0.15 \
  report_md=out/bench/splice_feature_shards.md
```

The report includes the original per-phase median/p90 table plus a detail table
sorted by total detailed seconds. Treat benchmark deltas as actionable only
after the canonical artifact comparator proves exact parity: loaded feature
tensors/metadata, parquet bytes/text, and provenance bytes/JSON must all match.
To measure save-side pipeline parallelism, run the same isolated inputs with
explicit variants such as
`variants='[{name: sync, overrides: {}}, {name: async_save_d1, overrides: {async_feature_save_depth: 1}}]'`;
`async_feature_save_depth>0` is serial-only and is rejected with
`shard_parallelism>1`.

## Common Performance Patterns

### Pattern: `gpt2_forward_batched` dominates
- **What it means**: The transformer forward pass is the bottleneck (expected).
- **Action**: Use nnterp backend (`backend=nnterp`), increase batch size, or reduce context window.

### Pattern: `sae_encode_batched` is significant
- **What it means**: SAE encoding is non-trivial at large batch sizes (6GB for full [B, 1024, 24576]).
- **Action**: The pipeline already pools per-window to avoid materializing full SAE activations.

### Pattern: Low GPU utilization
- **What it means**: Batch size too small, kernel launch overhead dominates.
- **Action**: Increase `batch_size` (use `sweep_batch_sizes=true` to find optimum).

### Pattern: OOM errors
- **What it means**: Batch size too large for available VRAM.
- **Action**: Reduce `batch_size`. At batch_size=64 on 24GB GPU, expect ~10GB usage.

### Pattern: `read_row_group` / `save_shard` dominates
- **What it means**: I/O bottleneck.
- **Action**: Use SSD storage, or consider async I/O pipeline.

## Runtime Assertions

The pipeline includes runtime assertions that validate data integrity:

- **`validate_shard`** (`steering/validate_features.py`): Checks required fields, dtype correctness, offset monotonicity, segment metadata consistency, and feature ID ranges.
- **`_get_sae_activations`**: Verifies SAE output is 2D, matches input sequence length, and has no grad.
- **`extract_document_features`**: Verifies output dtypes, magnitude positivity, and descending sort order.

These assertions use Python `assert`. To disable for maximum throughput: `python -O`.

## Statistical Comparison Output

When benchmark scripts compare implementations, prefer
`steering.profiling.statistics.compare_timings()` for statistical comparisons:

| Field | Meaning |
|-------|---------|
| Speedup | Ratio: baseline_mean / candidate_mean |
| 95% CI | Bootstrap confidence interval for the speedup ratio |
| Significant | Whether the difference is statistically significant (Mann-Whitney U, p < 0.05) |

**Interpreting results:**
- A speedup of 2.0x with CI [1.8, 2.2] and "significant=YES" means the candidate is reliably ~2x faster.
- A speedup of 1.05x with CI [0.95, 1.15] and "significant=no" means there's no meaningful difference.
- Always check the CI width. Narrow CIs = reliable measurement. Wide CIs = need more iterations (`--n-runs`).

## Running Each Profiling Script

### Encoder-Decoder Progression Profiling
```bash
uv run python -m scripts.profile_progression
uv run python -m scripts.profile_progression_microbench.encdec_build_conditioning
```
**Metrics:** progression eval/report timings and encoder-decoder conditioning
construction microbenchmarks.

Static-conditioned progression now enumerates the exact selected suites and
seeds after loading each checkpoint, then checks every unique nanochat token
context against the identity- and vocabulary-validated cache before the first
scored example. Macro CSVs expose this as the `context_coverage` stage.
This is a coverage fence, not a successful-profile substitute: an incomplete
cache must be rebuilt before timing the evaluation itself.

A real RTX 4090 check with the 400M l22 step-54600 checkpoint found a cache
whose feature-vocabulary identity matched but which contained only 45 of 120
required CORE contexts. Before the fence, five HellaSwag examples consumed
45.24 seconds before the first missing Jeopardy row failed the run. With the
fence, all 75 missing rows were reported in 11.03 seconds wall time and no eval
forward ran. See
[`progression-context-coverage-evidence/rtx4090-l22-incomplete-cache.json`](progression-context-coverage-evidence/rtx4090-l22-incomplete-cache.json).

The exact missing rows were subsequently enumerated and extracted with the
real Gemma-3 4B PT / Gemma Scope L22 SAE, then converted under the checkpoint's
pinned mask, clusters, statistics bytes, preselection seed, and compact
vocabulary. Production validation reopened all 120 unique rows. On the same
RTX 4090, three measured Path-C CORE macro iterations after one warmup had a
6.198s median end-to-end time: 4.863s suite, 0.361s checkpoint load, and 0.331s
coverage. The 110 wrapper calls per iteration spent a median 1.005s building
conditioning and 4.412s in wrapper forwards (the latter includes the former),
so the coverage fence is not the next bottleneck. A suite-level trace showed
CPU/launch overhead rather than one dominant GPU kernel: 162,139 kernel
launches, 50,827 copies, 9,385 stream synchronizations, and 1,287
`FlexAttentionAutogradOp` calls. See
[`progression-context-coverage-evidence/rtx4090-l22-complete-cache.json`](progression-context-coverage-evidence/rtx4090-l22-complete-cache.json).

Target-aware CORE now renders examples independently but scores up to five of
them in one forward. The completion rows, target masks, and context-cache keys
remain per-example; only the physical model call is coalesced. A first batched
FlexAttention prototype was rejected because CORE's varying static shapes hit
Dynamo's recompilation limit and the compiled-to-eager transition changed real
BF16 task results. The retained path scopes decoder self-attention to SDPA for
each target-aware forward and restores the prior backend afterward; packed
encoder and cross-attention kernels remain unchanged.

On the same RTX 4090, cache, checkpoint, 22 tasks, and 110-example workload,
batch sizes 1 and 5 produced exactly equal serialized result payloads with
SHA-256 `428ae959830caeb6cd2bebd0acf6677a388fe9c7ac16bd2fbf4de9f5f450e4eb`.
Three measured batch-5 macro iterations after one warmup had a 2.849s median
end-to-end time and 1.347s suite time. Relative to the prior 110-forward
profile, that is 2.18x end-to-end and 3.61x suite speedup; wrapper calls fell
to 22. Cold suite time also fell from 32.363s to 12.891s. See
[`progression-core-batching-evidence/rtx4090-l22.json`](progression-core-batching-evidence/rtx4090-l22.json).

The post-batching trace then exposed 264 nested-vmap mask-construction calls
using 541.8ms of self CPU. Packed encoder and cross-attention routing now
materialize the same owner-equality matrix directly, classify its partial/full
128×128 tiles, and use `BlockMask.from_kv_blocks`; the exact per-token
`mask_mod` remains attached. Differential CUDA tests compare every sparse-mask
tensor against `create_block_mask`, including empty tails, irregular partial
tiles, and full tiles.

On the same RTX 4090 workload, the nested-vmap events fell to zero. Three
measured iterations after one warmup reduced median suite time from 1.347s to
1.132s (**15.9%**) and end-to-end time from 2.849s to 2.662s (**6.6%**).
Conditioning-build totals fell 35.4%; CUDA time was effectively unchanged,
confirming a CPU construction win. See
[`progression-block-mask-evidence/rtx4090-l22.json`](progression-block-mask-evidence/rtx4090-l22.json).

The resulting trace made the exact context-coverage fence the largest isolated
CPU stage. Its CORE enumerator parsed all 91,037 JSONL rows even though
`max_per_task=5` selected only 110 examples. Capped CORE loading now scans row
byte offsets, reproduces the historical seed-1337 positional shuffle, and
JSON-decodes only the selected rows. Evaluation, target-real evaluation, and
both context-enumeration paths share that loader; uncapped runs retain the full
sequential decode and shuffle. Detached-base parity covered all 22 selected
task lists and produced the same 255 context records, 120 unique keys, and
token-ID SHA-256.

On the same RTX 4090 checkpoint workload, three measured iterations after one
warmup reduced median coverage from 0.346s to 0.182s (**47.4%**), suite time
from 1.132s to 0.961s (**15.2%**), and end-to-end time from 2.662s to 2.358s
(**11.4%**). See
[`progression-core-loader-evidence/rtx4090-l22.json`](progression-core-loader-evidence/rtx4090-l22.json).

The next trace showed that every one of the 12 cross-attention layers repeated
the same three eager CUDA `seg_map` guards on every wrapper call. The model now
validates that shared routing state once before the decoder-layer loop and uses
a prevalidated internal layer path; direct layer callers still take the fully
checked public path. Across 22 wrapper calls this removed exactly 726 repeated
guards, reducing stream synchronizations from 4,389 to 3,663, async copies from
7,369 to 6,643, and kernel launches from 27,244 to 24,340. CUDA attention time
was effectively unchanged, as expected for a host-synchronization optimization.

On the same RTX 4090 workload, three measured iterations after one warmup
reduced median wrapper-forward time from 0.737s to 0.692s (**6.1%**), suite time
from 0.961s to 0.917s (**4.6%**), and end-to-end time from 2.358s to 2.257s
(**4.3%**). The retained path was bit-exact against a checked-per-layer test
oracle. See
[`progression-xattn-validation-evidence/rtx4090-l22.json`](progression-xattn-validation-evidence/rtx4090-l22.json).

The next trace localized most remaining conditioning synchronization to encoder
feature packing. Context features arrive as CPU tuples and dictionaries, but
the wrapper previously allocated its padded encoder tensors on CUDA and wrote
each feature slot into them separately. It now assembles the complete IDs,
magnitudes, and mask tensors on CPU, then submits three non-blocking transfers
on the encoder's current CUDA stream. This path performs no sampling, so the
batching change does not alter RNG behavior.

On the same RTX 4090 workload, the retained trace removed 1,948 stream
synchronizations (**53.2%**), 3,830 async copies (**57.7%**), 1,040 kernel
launches, and 974 scalar materializations. Every removed event was inside
`encdec_build_conditioning`; its trace time fell **14.2%**, while the 1,800
remaining `aten::is_nonzero` calls were unchanged and remain attributable to
the separate target-mask/segment-map path. Three measured macro iterations
after one warmup reduced median conditioning-build time from 0.220s to 0.201s
(**8.7%**), wrapper-forward time from 0.692s to 0.671s (**3.1%**), and suite
time from 0.917s to 0.892s (**2.7%**). End-to-end time moved only **0.3%**
because checkpoint-load variation dominates that aggregate. See
[`progression-conditioning-staging-evidence/rtx4090-l22.json`](progression-conditioning-staging-evidence/rtx4090-l22.json).

Representative local dataloader measurements from 2026-06-05 (not CI
thresholds):

| Path | Before | After | Notes |
|---|---:|---:|---|
| Live row-group tokenization | 2.13s | 0.68s | 8 real FineWeb row groups, RustBPE, byte-identical token lists; batched tokenizer call is 3.1x faster than per-doc calls |
| Live dataloader row-group refill | 1.10s | 0.80s | l12/f16k real features, `B=4`, `T=1024`, `feature_delta=null`; first-batch shard load excluded |
| Packed FP-target scatter | 14.3ms | 9.4ms | f65k packed row shard, `B=8`, `T=2048`; direct flat scatter is bit-identical to the old `fp_flat` two-step reference |
| Fused packed reconstruction with FP targets | 29.7ms | 20.4ms | same f65k shard/config; avoids a duplicate CSR traversal before selection |

Additional non-tiny Gemma 3 4B PT checks from the same local run used a fixed
view over the first 41 f65k shards, `B=4`, `T=2048`, and a single RTX 4090.
Deterministic CUDA was required for exact GPU equality
(`CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic algorithms enabled, TF32
disabled).

| Check | Result |
|---|---|
| 4-layer and 6-layer non-FP pretraining equality | Exact batches, logits, losses, gradients, AdamW model state, and AdamW optimizer state for 3 steps |
| 4-layer and 6-layer FP-head equality | Exact multi-step equality with nonzero dense FP targets for 2 steps; 4-layer used paired models, 6-layer used sequential model lifetimes to fit VRAM |
| Loader-only timing | old scalar 38.65s, new batch 34.14s over two 40-batch runs; 1.13x total speedup |
| 4-layer training timing | old scalar 39.59s, new batch 36.18s over two 20-step runs; 1.09x total speedup, 1.18x post-first-batch speedup, 5.22GB peak |
| 6-layer training timing | old scalar 37.74s, new batch 38.19s over two 20-step runs; 0.99x total speedup, 1.15x post-first-batch speedup, 6.74GB peak |
| Completion smoke | 2 real-data training steps followed by 3 greedy encoder-decoder completion calls with 12 generated tokens each |

The full percentile-overlay FP path was not regenerated for all 41 shards in
this run: CPU statistics were stopped at 15 of 40 pass-1 shards because the ETA
exceeded the GPU reservation. The FP-head equality check therefore used nonzero
live-path dense targets derived from the real Gemma 3 4B PT feature rows rather
than the full all-extracted percentile overlay.

### Packed FP-target CSR gather

`build_fp_targets_from_shard` rebases each segment's CSR start by its cumulative
feature count before repeating it. This removes one feature-sized
`repeat_interleave` while preserving the public builder's scatter and validation.
On one canonical T1024/V2074 shard, each fresh process traversed all 512
rank-striped B32 batches with native 16/32 CPU thread settings. Eight measured
arms in ABBABAAB order, after separate A/B warmups, reduced mean component wall
time from **3.04079 to 2.52696 s (16.90%)** and process CPU from **48.58807 to
40.48426 s (16.68%)**. Both halves, both adjacent positions, and all four pairs
improved; the smallest pair improved wall/CPU by 8.16%/8.32%. All 5,120 outputs
including warmups matched retained goldens and an independent max-reduction
oracle, with input and RNG state unchanged. These are local CPU component
measurements, with no training-throughput or GPU-speed claim. The retained local
receipts are under `out/splice_perf_122/fp_gather_rebase_benchmark_v1`.

### Attention Backends
```bash
torchrun --standalone --nproc_per_node=1 -m scripts.benchmark.benchmark_attention -- impl=sdpa
torchrun --standalone --nproc_per_node=1 -m scripts.benchmark.benchmark_attention -- impl=fa3
torchrun --standalone --nproc_per_node=1 -m scripts.benchmark.benchmark_attention -- impl=fa4
torchrun --standalone --nproc_per_node=1 -m scripts.benchmark.benchmark_attention -- impl=flex
```
**Metrics:** tokens/sec and loss over a short non-cached decoder training loop.
`impl=flex` explicitly requests compiled FlexAttention and uses its FLASH backend
when available or Triton otherwise; `impl=fa4` requires the FLASH backend. The
JSON records `attention_routes.configured`, `non_cached`, and `kv_cache`
separately. Cached generation has no FlexAttention branch in the current
decoder wrapper and routes independently to FA3 or SDPA; that field is routing
instrumentation, not a cached-decode timing measurement.

### Feature Extraction GPU-Idle Sidecar (issue #564)

```bash
# Enable the sidecar (free-when-off by default)
torchrun --standalone --nproc_per_node=7 -m steering.extract_features \
    --config-name extract_features_8xpro6000 \
    profile=true \
    profile_output=out/traces/extraction_profile \
    num_shards=370
```

When `profile=true` the extraction loop writes a per-rank CSV + JSON sidecar at
`{profile_output}_rank{r}.csv` / `{profile_output}_rank{r}.json` (local-rank 0
also runs a `GPUUtilizationMonitor`). The CSV has one row per `timed_region`
invocation (columns: `name`, `elapsed_ns`, plus any active `timing_tags`). The
JSON summary contains:

| Field | Meaning |
|---|---|
| `gpu_idle_fraction` | Fraction of GPU-utilization samples below the idle threshold (default 5%). **Primary metric for issue #564.** |
| `gpu_longest_idle_s` | Longest contiguous idle streak in seconds. |
| `gpu_util_mean` | Mean GPU utilization across all samples. |
| `gpu_sample_count` | Total number of GPU-utilization samples. |
| `shards_completed` | Number of shards finished during this run. |
| `shards_per_hour` | Throughput derived from `shards_completed / wall_seconds × 3600`. |
| `wall_seconds` | Total elapsed seconds for the extraction loop. |
| `rank` | DDP rank that wrote this sidecar. |

The `input_stall` and `compute` `timed_region` labels in the per-rank CSV break
down where time is spent: `input_stall` is the wait for the next prefetched shard
when input staging is on (`input_depth >= 1`) — with `input_depth=0` there is no
prefetcher and the inline parquet read happens *inside* `compute`, so `input_stall`
is ~0; `compute` covers that inline read (when applicable), the GPU forward pass,
SAE encode, pool, and shard save dispatch.

**Free-when-off.** With `profile=false` (the default) the `timed_region` context
manager yields immediately — no `record_function` marker, no `cuda.synchronize`,
no timing rows. The `extraction_profiler` context yields `None` and installs
nothing. There is no overhead on production extraction runs.

### Feature Extraction Activation Collection
```bash
uv run python scripts/benchmark/benchmark_activation_methods.py smoke=true
uv run python scripts/benchmark/benchmark_activation_methods.py run_gpt2=true
```
**Metrics:** activation-collection latency with a pre-timing correctness gate.

### Native Compile Gate
```bash
uv run python scripts/benchmark/benchmark_native_compile.py smoke=true
uv run python scripts/benchmark/benchmark_native_compile.py n_iters=30
```
**Metrics:** eager vs compiled native collector latency after the oracle gate.

### Token Boundaries and Packed Rows
```bash
uv run python -m scripts.benchmark.benchmark_token_boundaries \
  feature_dir=out/data/features/... segment_size=128 report_md=out/bench/tok_boundary.md
uv run python -m scripts.benchmark.benchmark_packed_rows \
  feature_dir=out/data/features/... data_dir=$NANOCHAT_BASE_DIR/base_data \
  segment_size=128 sae_width=16k \
  max_shards=4 report_md=out/bench/packed_rows.md
```
**Metrics:** overlay pre-pass/cache, per-stage token-boundary timing, packed-row
stage totals, FeatureIndex subphases, compression backpressure, exact artifact
parity, rows/sec, CPU/threads, and peak RSS/PSS/swap.

### Feature-Shard Splice
```bash
uv run python -m scripts.benchmark.benchmark_splice_feature_shards \
  feature_dir=out/data/features/.../base \
  data_dir=$NANOCHAT_BASE_DIR/base_data \
  max_shards=4 \
  report_md=out/bench/splice_feature_shards.md
```
**Metrics:** per-phase median/p90 for load, parquet read, source alignment,
recombination, derived-shard build, provenance build, parquet/provenance writes,
feature save, and total wall; plus shards/hour, output bytes, CPU seconds, and
high-water RSS. The harness mirrors only the selected shard/parquet pairs into
`out/bench/splice_feature_shards/run_*` and runs variants from that isolated
input tree, so it never writes inside the production feature dir.

Before accepting a splice optimization, run the same `feature_dir`, `data_dir`,
`max_shards`, `cut_fraction`, seed, and variant list on the same machine and
include absolute and percent deltas against the baseline report. Full artifact
parity is required: decompressed feature tensors and metadata must match,
derived parquets and provenance must be byte-identical, and failed-shard/resume
behavior must remain loud.

The generic feature decoder reuses the owned `int32` buffer created when
upcasting `uint16`/`int16` IDs; unchanged ID dtypes still clone before delta
decoding. On one Gemma-3-4b-pt feature shard, eight complete decoder calls in
ABBA BAAB order measured median wall time **223.15 → 210.71 ms** (5.58% lower)
and process CPU time **26.91% lower**. A separate fresh-process pair observed
immediate post-decode high-water RSS **1,988,448,256 → 1,788,256,256 bytes**;
their pre-decode high-water marks differed, and this single pair does not
establish a universal memory reduction. Read/decompression, output checks and
returned-output destruction were outside the decoder timers. These are local
CPU component measurements; no pipeline or GPU throughput gain was measured.
See the [decoder ownership evidence](evidence/feature-decoder-owned-buffer-local-cpu.json)
for every sample, both order groups, exact-value checks and memory scope.

Opposite-percentile splice pairing reuses its constructor's finite-table evidence
for an owned native CPU table that remains read-only by contract. Generic lookups
and groups whose assembled feature tensors are subclasses retain their
table-NaN checks; missing-row
diagnostics and percentile arithmetic are unchanged. Eight complete warm pairer
calls on one captured workload measured mean wall **1.414 → 1.229 s (13.04% lower)**
and process CPU **17.034 → 14.525 s (14.73% lower)**. Both halves and adjacent-order
aggregates improved, while one adjacent pair regressed **7.84% wall / 8.04% CPU**.
Construction, workload capture and correctness checks were outside these timers;
this is a local CPU component result, with no pipeline or GPU speedup claim. The
[evidence capsule](evidence/splice-owned-finite-table-local-cpu.json) retains all
samples and the earlier incomplete 120-second attempt. Focused validation passed
388 tests (1 CUDA skip, 8 deselected), with scoped Ruff/ty/diff checks passing;
41 synthetic controls are counted separately. One production splice reproduced
all 19 decoded fields and raw metadata, plus byte-identical parquet/provenance,
for 8,240 documents and 50,689,690 feature values. Its owned outputs and test
temporary files were removed after independent custody checks; raw evidence remains.

The percentile lookup omits a redundant Boolean AND when updating the search upper
bound: inactive lanes already have `lo == hi == mid`. A local CPU study of 76
actual splice queries used one common baseline warmup followed by eight complete
calls per query in balanced orders. Summing every arm’s four-call mean across
queries reduced wall time by **12.874 ms (2.532%)** and process CPU by **166.891 ms
(2.067%)**. All 608 timed outputs and 76 warmups matched exactly; the fixed warm
component gates passed. First-position calls on reference queries were 1.423%
slower, and three of ten blocks regressed. An earlier order-sensitive screen remains deferred. These
measurements cover deliberately warm inputs and retained result buffers, not
native once-per-query or pipeline throughput, GPU execution, or statistical
significance. The [evidence capsule](evidence/percentile-converged-lanes-local-cpu.json)
retains every timing, estimator, threshold, negative subgroup, and scope limit.

Long offset-helper decodes pass exact native integer lists through the same public
Hugging Face decode method using an object array, retaining its list-copy boundary,
current defaults and private decoder behavior. Custom public methods and other
input types keep the original argument. On 8,240 actual Gemma-3-4b-pt documents,
eight complete helper traversals in ABBA BAAB order measured mean wall time
**2.15669 → 2.02098 s (6.29% lower)** and process CPU **6.25% lower**. Both halves
and both adjacent order positions improved, but the final pair was **4.28% slower**
in wall time and **4.42% slower** in CPU time. Each arm had an untimed warmup;
every traversal started with a fresh singleton cache and matched all ordered
offsets, cache contents, route counts and final text lengths. Guard scans and
array construction were timed; input setup and output checks were outside clocks
and can condition later traversals. These are warm local CPU helper measurements,
with no pipeline or startup speedup, GPU throughput, memory reduction or
statistical significance claim.
A separate candidate-production splice run matched all 19 decoded fields, raw
feature metadata and parquet/provenance bytes against the canonical baseline;
that run establishes correctness only. The
[evidence capsule](evidence/hf-public-offset-decode-local-cpu.json) retains all
eight samples, fixed gates, negative pair and validation scope.


For standard Gemma tokenizers with the native `Replace(▁ → space), ByteFallback,
Fuse` decoder and cleanup disabled, progressive offsets reuse decoded pending byte
runs while counting ordinary-piece lengths directly. The same native decoder owns
invalid and incomplete UTF-8 behavior, and the existing offset shrink repair remains
unchanged. Custom methods, changed defaults/configuration and unsupported inputs
retain progressive public decoding. On all **8,240 actual documents**, eight complete
helper traversals in ABBABAAB order measured mean wall **1.93926 → 1.03549 s
(46.60% lower)** and process CPU **46.58% lower**, including guard and conversion
costs. Each arm had a warmup and every traversal used a fresh singleton cache;
all offsets, ordered cache contents, route counts and input/state checks matched.
Both halves, both adjacent positions and all four adjacent pairs improved.
These are warm CPU component measurements on one cohort, with no pipeline, startup,
GPU, memory or significance claim. The native primitive passed 82,200 synthetic
prefix-vector cases and 37 small Gemma controls passed. Focused CPU validation
passed **453 tests**, with **12 skips** for unavailable real-tokenizer fixtures and
2 deselections. An import-format correction resolved the initial Ruff I001;
Ruff, ty and diff checks then passed without repeating pytest. A separate actual
production SFS run matched all **19 decoded fields**, raw feature metadata and
parquet/provenance bytes against the canonical baseline, establishing correctness
only. The [evidence capsule](evidence/gemma-native-prefix-runs-local-cpu.json)
retains all samples, fixed gates, limits and receipt pins.

## Hugging Face decoder-SFT timing and traces

`scripts.train.chat_sft` exposes HF-only diagnostics through the shared decoder
config:

- `profile_timing=true` writes `chat_sft_timing_rank<R>.csv`. The
  `chat_sft/paid_step` row reuses the loop's existing CUDA synchronization and
  is the authoritative paid wall time. It includes exact scheduled,
  non-padding, and supervised token counts plus globally normalized loss;
- nested timing rows are intentionally unsynchronized host launch/control
  durations and must not be summed as device time. Use a Chrome trace for GPU
  attribution;
- nonempty `profile_trace_steps` and `profile_trace_ranks` lists enable tracing.
  Each selected rank/step pair writes one Chrome trace. Empty or half-enabled
  selections fail before model work.

HF runs also emit lightweight `chat_sft_startup_phase`, first-step, and summary
lines without a config knob or CUDA synchronization. Startup phases cover the
pinned source config/tokenizer, distributed compute initialization, serialized
node-local model load, mixture build and fingerprint, semantic/restart load,
compile/DDP, optimizer, and first batch. Time-to-first-step begins before the
process-group/CUDA initialization and therefore includes the rendezvous.
Use separate processes for trace, timing CSV, and instrumentation-free control
runs. The exact 4x8 procedure and retained artifact set are in
[`hf-decoder-sft-h200.md`](hf-decoder-sft-h200.md).
For a distributed job, align paid rows by completed step and use the maximum
rank duration as cluster step time; do not sum rank durations or report the
fastest rank's throughput.

## Conditioned-SFT live-loop attribution

`scripts.train.conditional_encdec_sft` has independent opt-in controls for the
live real-text/real-feature path:

- `profile_timing=true` writes `sft_timing_rank<R>.csv`; the default
  `profile_timing_sync_cuda=true` makes main-thread regions complete their
  device work but perturbs overlap, so this arm is for attribution only.
  Prefetch-worker phase rows remain unsynchronized CPU wall times. Each such
  row carries the matching rank, step, and microstep and enters the rank
  recorder only when its batch is delivered; discarded speculative batches
  therefore publish no rows;
- `profile_trace=<step>` plus `profile_trace_steps=<count>` captures an
  all-thread CPU/CUDA Chrome trace from the selected optimizer steps. A
  trace-only run does not propagate the profiler-listener context used by
  `timed_region` into the SFT prefetch thread, so its fine-grained markers are
  absent. `profile_timing=true` supplies the worker-local listener and records
  those rows (and trace markers when both diagnostics are enabled), but changes
  overlap. Retained events scale with microbatches and operations: the
  issue-#1475 four-step B32/accumulation-8 capture exhausted a 256-GiB cgroup
  while finalizing, so start with the smallest capture and prefer the timing
  diagnostics.

Use a separate fresh-process run with all timing/trace controls disabled for
throughput. Compare the same source tree, checkpoint, feature/parquet identity,
node/GPU shape, batch geometry, attention backend, and compiler-cache policy.
`sft_prefetch_batches` is a complete-batch FIFO depth, not a feature-shard
depth; `0` is the synchronous control. The producer is bounded to one optimizer
step so it cannot cross validation or checkpoint RNG/state boundaries. See the
SFT pipeline's
[batch-prefetch section](../pipelines/encoder-decoder-training.md#conditioned-sft-batch-prefetch)
for output interpretation and correctness invariants.

For the exact issue-#1475 2xH200 B32/T1024/accumulation-8 workload, depth 2
improved stable throughput by 3.58-8.53% across two nodes and reduced the
feature-transition step by about 3.5-3.9 seconds. Depth 8 was slightly slower.
These are geometry-specific measurements; keep the global default at `0` and
re-measure before enabling depth 2 elsewhere.

## Conditioned-SFT startup trace

`conditional_encdec_sft` also emits an always-on, rank-aware startup trace at
`INFO`; it does not require `profile_timing` or `profile_trace` and it does not
create profiler markers. The boundary starts at the first executable line of
`main()` (Python import and process-launch time are outside it) and ends after
the first completed optimizer step. Timings use the monotonic
`time.perf_counter_ns()` clock. The normal rank-aware Loguru sink makes the
phase lines rank-0 visible; the closing summary includes `ddp_world_size` so
the wall-clock result is not mistaken for an aggregate across ranks.

Each completed or failed leaf phase has the form:

```text
SFT startup phase=<name> status=<completed|failed|skipped> elapsed_s=<seconds> key=value ...
```

The current phase names are `hydra_config`, `compute_init`,
`base_checkpoint_load`, `selection_identity`, `sft_resume_checkpoint_load`,
`checkpoint_identity`,
`validation_artifact_paths`, `tokenizer_load`, `mixture_build`,
`wandb_setup`, `model_construct`, `model_materialize`, `model_state_load`, `compile_setup`,
`optimizer_and_runtime_setup`, `statistics_load`, `feature_pipeline_setup`,
`percentile_table_setup`, `sft_feature_store_load`,
`sft_percentile_document_coverage`, `sft_percentile_id_scan`,
`sft_percentile_coverage`, `sft_artifact_identity`,
`sft_document_coverage`, `sft_feature_row_coverage`, `sft_alignment_probe`,
`sft_initial_batch`, and `first_optimizer_step`. Conditional phases are
omitted when their work is not part of the run; the percentile scan and
coverage lines remain with `sources=0`/`tables=0` so a disabled consumer is
explicit. Train/validation and repeated table loads carry a `source` or
`table` field rather than being merged into an ambiguous total.

`sft_percentile_id_scan` measures the rank-once ID-only union (issue #1495):
rank 0's phase time is the actual shard scan plus the bitmap broadcast, while
every other rank's phase time is its wait for that broadcast. Rank 0 also
logs one `conditioned SFT <source> ID-only percentile scan: ...` line per
covered source (train, plus val when periodic eval is enabled) with
shard/document/segment/compressed-byte totals; when the scan falls back to
the reference document walk it logs a WARNING naming the anomaly instead,
and the phase's `elapsed_s` reflects the slower walk. The
`raw_ids`/`masked_ids` fields are unchanged.

Artifact phases report only facts already observed by the existing validation
path: feature/parquet shard counts, document counts, missing-document counts,
alignment candidates/requested/sampled/matched rows, and raw versus
post-feature-mask percentile IDs. A weak-cache reuse is marked `cache=hit`;
the observer does not perform an additional artifact scan.

The first-step line is deliberately separate from steady state. The closing
summary has `summary=time_to_first_step`, `time_to_first_step_s`,
`startup_phase_sum_s`, `unattributed_startup_s`, `first_step_warmup_s`,
`post_first_step_mean_s`, and `post_first_step_steps`. If the loop reaches no
optimizer step, the summary has `status=not_reached` and null first/steady-state
fields. The residual is expected to include orchestration and other work that
does not belong to a measured leaf phase; it is not silently attributed to a
nearby phase. First-step timing includes the existing CUDA synchronization,
forward/backward, compiler/autotune work, and optimizer update. No artificial
warmup forward is inserted, and existing live-loop `step ... ms` semantics are
unchanged.

## Optimization Targets by Metric

| Target | Script | What to look at |
|--------|--------|----------------|
| Model forward pass | `scripts.benchmark.benchmark_attention` | tokens/sec and loss by attention backend |
| Progression eval | `scripts.profile_progression` | eval/report timings |
| Progression CORE numerical parity | `scripts.benchmark.benchmark_progression_core_numerics` | scored-row identity, target-position logit/loss drift, ranking overlap, cosine similarity, and final metric drift across revisions |
| Repository-HF harmfulness generation | `scripts.benchmark.benchmark_hf_harmfulness_generation` | serial-vs-batched synchronized latency, deterministic replay, greedy parity, and optional operator traces without judge calls |
| Conditioning build | `scripts.profile_progression_microbench.encdec_build_conditioning` | encoder-decoder conditioning construction |
| Feature extraction | `scripts/benchmark/benchmark_activation_methods.py` | activation collection latency and correctness gate |
| Native compile | `scripts/benchmark/benchmark_native_compile.py` | eager vs compiled native collector |
| Token-boundary overlays | `scripts.benchmark.benchmark_token_boundaries` | manifest cache, tokenization, overlay save |
| Packed rows | `scripts.benchmark.benchmark_packed_rows` | row/FeatureIndex subphases, worker/backpressure cost, exact parity, rows/sec, RSS/PSS/swap |
| Feature-shard splice | `scripts.benchmark.benchmark_splice_feature_shards` | per-phase splice timings, save cost, throughput, RSS/HWM |
| Encoder-decoder l22 training | `scripts.benchmark.profile_encdec_l22_training` | real packed/pre-selected data scaling, exact-checkpoint GPU phases/kernels, measured CUDA/NCCL training including target geometry and FP8 A/Bs, opt-in forward-component diagnostics with an eager control, distributed-optimizer collective inventory, and topology sensitivity |
| Conditioned-SFT live loop | `scripts.train.conditional_encdec_sft` | separate synchronous/prefetched throughput arms plus opt-in per-rank timing (including delivered worker rows) and a bounded CPU/CUDA Chrome trace on real SFT artifacts |
| NCCL launch preflight | `scripts.benchmark.preflight_nccl_transport` | bounded all-reduce/reduce-scatter/all-gather correctness and rank-critical timing before a reserved multi-node run |

The encoder-decoder profiler keeps measured and modeled results separate. See
the [encoder-decoder component profile](encoder-decoder-training-components.md)
for the repeated encoder/decoder/cross-attention/head attribution, topology
comparison, and packed FlexAttention A/B. Component diagnostics are opt-in and
do not replace the ordinary compiled throughput samples.

For the planned 4 x 8 or 8 x 8 H200 run, use the
[launch-readiness runbook](h200-8x8-launch-readiness.md). Its checked-in 3B
3B preset reports analytical optimizer evidence in `topology` mode and measures
the exact target geometry on synthetic rows in `distributed` mode. The
allocation-local transport probe supplies separately labeled NCCL measurements
and selects the host-staged fallback if the default path fails.

## Known Limitations

- **torch.compile**: Hooks are not attached to the compiled throughput graph. The encoder-decoder profiler instead brackets opt-in eager component samples with an adjacent eager control and records a separate compiled operator ranking.
- **Multi-GPU**: `profile_encdec_l22_training distributed` measures full compiled optimizer steps under torchrun, while `topology` inventories exact messages and models 64-rank sensitivity. The distributed mode does not isolate collective-only time or exercise remote data staging, so use an NCCL microbenchmark/trace for transport attribution.
- **Warmup**: Scripts now include warmup iterations. The first timed iteration may still have JIT compilation overhead from new code paths. Use `--n-runs` for reliable comparisons.

## Warm statistics-loader reconstruction

The loader releases each consumed row from the privately owned native `Tensor.tolist()` percentile result after the existing rounding loop. Eligibility is automatic, with the original reconstruction retained for custom mappings, keys, tensors, `tolist` overrides, instance attributes and TorchFunctionMode. It adds no configuration option and keeps the rounding operations unchanged. A full candidate-loader SFS correctness run passed: all 19 decoded typed fields and raw metadata matched the canonical oracles, and parquet/provenance bytes matched exactly across 8,240 documents, 24,107 segments, 50,689,690 features and 2,380 switches. Only the script’s public stats-loader alias was substituted and restored; the exact candidate source has since been copied into the working tree. The production source is byte-identical to that qualified candidate.

On the canonical 82,241,744-byte statistics artifact (SHA-256 `d957803f6da9f53c5307eb1e0d60796249dafef0c053349542973545d720bf93`), complete public CPU `load_statistics` averaged **2.215658 → 2.108248 s wall (4.8478% lower)** and **2.215282 → 2.107778 s process CPU (4.8528% lower)**, saving approximately **0.107 s** per load. The baseline source was `921d0e05dc18fb42c3eb99c19a08bd3e06c6d1cc7c406b6d32a8e965595738f3`; candidate source was `3d7f1590cd8ed22900269278e996e8e49b1f12cefb545e4e97b3d9d4ef05e43c`.

Two warmups preceded eight retained ABBABAAB calls. Both mean improvements exceeded the fixed 3% thresholds, and both halves and adjacent-order position classes improved in wall and CPU time. All four adjacent pairs also improved. All ten complete protocol-5 output graphs matched SHA-256 `ac48d388b0a5fff77400691ec20f6849bab2acd734f35b592794920bd499bdc9` (109,282,153 bytes; 65,471 statistics rows). Thirty-two synthetic controls passed, including fallback, callback, ownership and error cases.

The process used native Torch threads 16/inter-op 32, 32-CPU affinity, observed nice 6, all five CPU/tokenizer environment overrides unset, and enabled GC with thresholds `(2000, 10, 10)`. File reading, parsing and reconstruction were timed; collection, output hashing and graph destruction ran outside clocks. Candidate warmup route profiling was excluded from all eight timing samples. This is a warm-filesystem, same-process loader component measurement; it establishes no quantitative memory benefit, causal GC saving, cold-cache, pipeline or GPU improvement. Compact timings and exact evidence pins are in [statistics-row-release-local-cpu.json](evidence/statistics-row-release-local-cpu.json).

The three owned qualification outputs (112,254,945 logical bytes) were removed after independent custody reviews; an independent postcheck confirmed all planned paths absent and raw evidence retained.

Repository validation was **479 passed, 1 failed, 4 CUDA-unavailable skips and 11 deselected** (18 warnings); all 11 added row-release tests passed. The failure was the existing remote-percentile shared processing/cleanup error test: a distinct background download-cancellation note was attached to the shared error. Ruff and ty initially found test-only import/annotation issues; the minimal fixes passed Ruff, ty and `git diff --check` before pytest.

A fixed eight-process diagnostic of the unchanged failing node then passed four times with the original loader and four times with the candidate, leaving the original failure's cause inconclusive. A separate two-process Events witness forced the original download worker to observe cancellation; both loaders produced the same native cancellation error and teardown note while preserving the shared primary/close error identity. This establishes baseline reachability under that controlled schedule, not the precise original cause or a green repository test gate. Raw failed/setup-invalid attempts and all diagnostic outcomes are retained in the evidence capsule's receipt pins.


## Eager statistics quantile rows

`compute_per_feature_percentiles` converts each eligible native quantile row to a Python list at its existing result-construction point, replacing individual tensor indexing and `.item()` calls for each percentile. Quantile arithmetic, rounding, histograms and defaults remain unchanged. The scalar path remains available for custom inputs or callbacks, active Torch modes, gradient-bearing intermediates and unsupported owners; eligibility is checked again for each row.

On one unchanged real feature shard (98,347,010 compressed bytes; 8,239 documents, 24,107 segments, 50,689,690 activations and 44,441 active features), complete public **eager CPU** `compute_statistics` averaged **15.302442 → 9.575165 s wall (37.4272% lower)** and **51.029470 → 44.620386 s process CPU (12.5596% lower)**. The timed operation included reading, computation, saving, logging and the original post-save sanity check. Two warmups preceded eight ABBABAAB measurements; the fixed 3% wall/CPU gates passed, both halves and adjacent-position classes improved, and all four adjacent pairs improved. Only the candidate warmup was profiled, confirming 44,441 converted rows; the eight timing calls had no profiler.

All ten returned graphs matched the strict original ordered JSON oracle, and all ten saved files matched all 22 tensor payloads, dtypes and shapes plus nonvolatile metadata. Reloaded graphs matched after sorting **only** their `health_metrics` keys: the safetensors metadata map has no stable insertion-order guarantee. Every other graph order and value remained strict. The earlier attempt stopped on precisely that health-map order difference before any timed samples; its failed check and outputs remain retained. The corrected oracle also preserves each raw reload hash and observed health-key order.

The matched process used Torch threads 16/inter-op 32, 32-CPU affinity, observed nice 19, enabled GC `(2000, 10, 10)`, unset CPU/tokenizer environment overrides and hidden/unavailable CUDA. Explicit collection, result destruction, reload validation and hashing ran outside clocks. This is a warm-filesystem, same-process supported eager-producer measurement; it does not measure default streaming, GPU or complete pipeline speed, or establish a memory/GC benefit. Forty-five bounded semantic controls passed, including callbacks, mode changes between rows, input state, errors and forward saved-tensor payloads. Focused CPU validation passed **175 tests, with 20 skips and three explicit CUDA-only deselections** (19 warnings); all eleven added regression cases passed. Ruff, ty and `git diff --check` passed. The initial run retained **173 passed, five failed and 20 skipped**: two failures were an empty-percentile scalar-reference test-oracle error, corrected with a tuple subclass while retaining the same cases and assertions; three came from unchanged reference tests that default to CUDA. Only those three node IDs were excluded from the CPU rerun. Their unchanged test/reference source and failure traces were checked; no baseline test execution is claimed. The measured source differs from the adopted source only by import ordering, with identical non-import AST and import bindings. The earlier statistics-loader remote-test failure and its limited diagnostic conclusions above remain unchanged. Exact settings, timings and receipt pins are in [statistics-quantile-rows-local-cpu.json](evidence/statistics-quantile-rows-local-cpu.json).

After three custody audits, the ten successful-run statistics files (558,277,760 bytes) and eleven private directories were removed exactly once; both independent postchecks passed. The original three baseline files, earlier failed attempt and all raw evidence remain retained.


## Feature blacklist statistics snapshot

With implicit width, the blacklist CLI reuses the fully parsed statistics graph from width resolution for that invocation; each later invocation rereads the file. Explicit-width and standalone calls keep their existing loader path. The full parse, validation, frequency bounds and artifact format remain unchanged.

On the unchanged 82,241,744-byte statistics artifact (65,536 dictionary features, 65,471 active rows and 130,534,655 segments), complete public `precompute_feature_blacklist.run(cfg)` measured:

| Metric | Baseline mean | Candidate mean | Reduction |
| --- | ---: | ---: | ---: |
| Wall | 5.796217 s | 3.571787 s | 38.3773% |
| Process CPU | 4.881221 s | 2.299627 s | 52.8883% |

Two warmups preceded eight fixed ABBABAAB measurements. Every call used a fresh identical physical statistics copy with absent outputs and unchanged default bounds, implicit width and `force=false`. Parsing, generation, writes, reload and logging were timed; setup, collection and verification were outside clocks. Warmup-only profiling confirmed two loader calls became one; all eight measured calls had no profiler. The predeclared 3% wall/CPU gates passed, both halves and adjacent-position classes improved, and all four adjacent pairs improved.

All ten calls matched all 3,701 expected IDs, exact text, typed semantic metadata and input hash/identity; only the new output timestamp value was volatile and retained raw. Twenty-seven bounded controls passed, including full malformed-field parsing, cache/force behavior, fresh reads between invocations, explicit/disabled paths and generic hook/barrier behavior. The first parsed graph is deliberately reused if a logger callback replaces the file during that invocation; this is not a claim of identical concurrent-writer or duplicate-callback observation.

The process used native Torch threads 16/inter-op 32, 32-CPU affinity, observed nice 6, enabled GC `(2000, 10, 10)`, unset CPU/tokenizer environment overrides and hidden/unavailable CUDA. This is a warm-filesystem, same-process complete CLI component measurement, not startup, training/pipeline/GPU throughput or memory evidence. Scoped Ruff/ty and `git diff --check` passed; focused validation passed **219 tests**, with eight configured-marker deselections and 18 warnings, including all eleven added cases and one normal-import full-artifact cache-hit check (one load, unchanged inputs and outputs). Projectwide ty still fails with the same **607 diagnostics (484 errors, 123 warnings)** on the exact parent and current trees. Their complete diagnostic text matched after normalizing only owned root paths and 25 secondary references to the same shared environment. The initial strict comparison failed on that path spelling and remains recorded; this is not a projectwide typing pass or a full-repository test claim. Exact samples and receipts are in [blacklist-statistics-snapshot-local-cpu.json](evidence/blacklist-statistics-snapshot-local-cpu.json).

After three custody audits, the 30 benchmark scratch files (822,635,320 bytes) and 21 private directories were removed exactly once; both independent postchecks passed. Original statistics, the earlier profile baseline and all raw evidence remain retained.


## Common-feature occurrence counts

The common-feature extractor reuses a per-shard CPU integer count vector for active-feature coverage and occurrence counts. Percentile arithmetic, full statistics parsing, ranking and output YAML are unchanged; other tensor ranks and devices retain the previous path.

On one unchanged real shard (8,239 documents, 24,107 segments, 50,689,690 activations and 65,536 dictionary features), complete public absolute extraction with `top_k=64` measured:

| Metric | Baseline mean | Candidate mean | Reduction |
| --- | ---: | ---: | ---: |
| Wall | 7.381775 s | 6.560871 s | 11.1207% |
| Process CPU | 45.027631 s | 33.308117 s | 26.0274% |

Two warmups preceded eight fixed ABBABAAB measurements. The predeclared 3% wall/CPU gates and both halves/position classes passed. The fourth adjacent wall pair was 0.4542% slower and remains included; adjacent pairs were report-only. Warm-only profiling confirmed the native count route; all eight primary calls had no profiler. All ten calls matched the full dense sum/count payloads bit for bit, all 64 ordered selections and YAML bytes, independent bucket checks, and unchanged input/statistics hashes. Candidate-namespace validation passed 134 tests: 22 paired controls and 112 existing-suite tests (21 warnings). Normal-namespace focused validation passed 120 tests (112 existing and eight added; 18 warnings), with scoped Ruff and ty passing. The initial run had 118 passes and two new fixture failures: base writers reject empty segments before the changed scanner executes. Only those fixtures were repaired; their failed logs and XML remain retained. Projectwide ty still fails with 607 diagnostics (484 errors, 123 warnings), byte-identical to the retained parent diagnostic log under the matched project/PYTHONPATH settings. The current tool is ty 0.0.91; the historical version was not recorded. An earlier unmatched-environment 603-diagnostic attempt is also retained. This is not a projectwide typing or full-repository test pass.

The complete operation included both statistics reads, table construction, full activation scan, ranking, buckets, logging and YAML writes. Equivalent delegating observers retained results in both arms with overhead inside clocks; imports, collection, destruction and verification were outside clocks. The process used Torch threads 16/inter-op 32, all 32 CPUs, observed nice 6, ordinary GC and unset CPU/tokenizer environment overrides with CUDA hidden. This is a warm-process CPU component result, without a training/pipeline, GPU or memory claim. Exact samples, the adverse pair and receipts are in [common-feature-counts-local-cpu.json](evidence/common-feature-counts-local-cpu.json).


## Common-feature statistics snapshot

When the discovered source and boundary statistics paths are equal, common-feature extraction reuses the first fully parsed graph for that invocation. Different paths, including unresolved aliases, still load separately; delta baseline statistics retain their separate load, and each later invocation reads its source afresh. The second discovery and missing-file check remain in place. A mid-invocation replacement after the first parse intentionally uses that snapshot rather than promising identical duplicate-read or callback observation.

On the same full real cohort (8,239 documents, 24,107 segments, 50,689,690 activations and 65,536 dictionary features), complete public absolute extraction with the same source/boundary path and `top_k=64` measured:

| Metric | Baseline mean | Candidate mean | Reduction |
| --- | ---: | ---: | ---: |
| Wall | 7.202017 s | 4.924297 s | 31.6261% |
| Process CPU | 39.827700 s | 33.889891 s | 14.9087% |

Two warmups preceded eight fixed ABBABAAB measurements. All four predeclared gates passed: pooled wall and CPU improved by at least 3%, and both halves and position classes improved. All four report-only adjacent pairs improved. Warmup profiling confirmed two native statistics loads became one; all eight primary calls were unprofiled. Every call matched full dense sum/count bits, ordered selections and YAML bytes, independent ranking/bucket checks, and unchanged statistics/input hashes. All percentile arithmetic, scan chunks and occurrence counting stayed unchanged.

Candidate-namespace controls passed 136 tests, including 16 paired controls and 120 existing cases (21 warnings). The initial attempt had 135 passes and one incorrect delta-fixture expectation: its positive-delta pool contained only feature 0. The corrected test requires the exact feature/bucket and YAML bytes; the failed attempt remains retained. Normal-namespace validation passed 125 tests (120 existing and five added; 18 warnings), with scoped Ruff, ty and `git diff --check` passing. Projectwide ty still fails with the same 607 diagnostics (484 errors, 123 warnings), with complete raw output byte-identical to the retained parent log under explicit matched project/PYTHONPATH settings. The current tool is ty 0.0.91; the historical version was not recorded. This is not a projectwide typing or full-repository test pass.

The process used native Torch threads 16/inter-op 32, all 32 CPUs, observed nice 6, ordinary GC, unset CPU/tokenizer environment overrides and hidden CUDA. Full public parsing, tables, scan, ranking, buckets, logging and YAML writes were timed, with equivalent result observers in both arms. Imports, configuration, collection, destruction and verification were outside clocks. This is a warm-process CPU component result for equal discovered statistics paths, without a training/pipeline, GPU or memory claim. Exact samples and receipts are in [common-feature-statistics-snapshot-local-cpu.json](evidence/common-feature-statistics-snapshot-local-cpu.json).


## Exact statistics histogram-bin lookup

Exact streaming finalization reuses one lazy, call-local histogram-bin lookup over the 31,743 positive finite float16 values. Unrounded shared edges, clamping, lower-bin interior-edge ties, quantiles and count validation retain their existing semantics; production allocation and memory admission are unchanged.

On the original full cohort (8,239 documents, 24,107 segments and 50,689,690 activations), complete public producer means were **21.514165 → 15.064948 s wall (29.9766% lower)** and **208.156727 → 52.026200 s process CPU (75.0062% lower)**. Two warmups preceded eight fixed ABBABAAB samples; all four predeclared gates passed, including both halves and position classes. The first adjacent pair regressed 3.9203% in wall time; all eight samples remain included. Every one of ten saved outputs matched all 22 tensor payloads, dtypes/shapes and nonvolatile metadata of the original golden.

This measurement uses the explicitly controlled low-RAM CPU regime: identical NumPy-backed count storage and benchmark-only shape/type admission in both arms, Torch threads 16/inter-op 32, owned threads normalized to nice 19 before clocks, hidden CUDA and unset CPU/tokenizer overrides. Full allocation, updates, finalization and save were timed; imports, priority control and verification were outside clocks. Process CPU includes all threads and spin. The observed peak was 3.332 GiB in this regime; it does not establish production allocator RAM, eager-backing performance, cold-cache, GPU or training throughput. All samples, gates, runtime/configuration, adverse pair and source/oracle receipts are in [statistics-histogram-lookup-local-cpu.json](evidence/statistics-histogram-lookup-local-cpu.json).

Focused CPU validation retained **519 passed, three failed, 42 skipped and four marker deselections**; all three added cases passed. The three existing CUDA-reference failures were reproduced individually against the unchanged whole baseline with identical `No CUDA GPUs are available` errors. No clean-suite or full-scope paired-baseline pass is claimed. Scoped Ruff/ty and whitespace checks passed; the compact evidence retains raw JUnit and source-binding receipts.

## Exact statistics direct CPU count updates

The CPU exact-count reducer adds integer ones directly at each validated composite key instead of sorting keys to derive multiplicities. Duplicate keys accumulate in the same bounded chunks; input validation, partial-update checkpoints, total-count checks and the CUDA path remain unchanged.

On the complete genuine 8,239-document / 24,107-segment / 50,689,690-activation streaming workload, warm AB was excluded and all eight fixed ABBABAAB samples were retained. Mean wall time fell from 10.77394 to 9.85394 seconds (**8.539%**) and mean process CPU from 38.90285 to 25.14237 seconds (**35.371%**). Both chronological halves and pair positions improved for each metric; all adjacent report-only pairs also improved. All ten outputs matched the original 22-field golden and static metadata.

This uses the declared low-RAM NumPy-backed count-storage adapter in both arms, native Torch threads 16/inter-op 32, and owned threads normalized to nice 19 before clocks. Production allocation/admission remain unchanged. Process CPU includes every target thread and spin; the native exact-count wall phase also includes validation and updates, so these measurements do not isolate sorting CPU or establish production allocator RAM, eager-backing, cold-cache, GPU or training performance. All samples, strata, gates, runtime/configuration and immutable receipts are in [statistics-direct-count-update-local-cpu.json](evidence/statistics-direct-count-update-local-cpu.json).


## Percentile-overlay table certificate

The local CPU overlay producer relies on the table builder’s monotonicity check to reject NaNs, so it omits a separate whole-table NaN scan and repeated lookup NaN checks. Lookup arithmetic, missing-row errors, uint8 encoding, borrowed defaults and remote/CUDA paths remain unchanged. The timings below measured the earlier implementation with the whole-table scan; they do not establish a speedup from removing it.

For the complete genuine 8,239-document / 24,107-segment / 50,689,690-activation operation, warm AB preceded eight fixed ABBABAAB samples. Mean wall time fell **6.685230 → 6.215319 s (7.0291%)** and process CPU **49.820657 → 44.344473 s (10.9918%)**. All four gates passed, including both halves and position classes. All samples remain included: adjacent report-only pair 2 regressed 1.2641% wall and 8.2658% CPU; pair 0 regressed 0.1751% CPU.

All ten full uint8 overlays, all 22 unchanged statistics tensor payloads, checksum-only metadata updates and static manifest/base-hash checks matched. The process used the ordinary overlay allocator, Torch 16/inter-op 32 and owned threads normalized to nice 19 before clocks. Full public statistics loading, table construction, certificate scan, lookup, compression and writes were timed; imports, private fixture copies, priority control and verification were outside clocks. Process CPU includes worker threads and spin. A separate coarse lookup diagnostic supported the target scope but also showed earlier variation; no pure NaN-check CPU or cold-cache, GPU, training or memory claim follows. Exact samples, adverse pairs, configuration, controls and validation receipts are in [percentile-overlay-table-certificate-local-cpu.json](evidence/percentile-overlay-table-certificate-local-cpu.json).

Scoped CPU validation passed **121 tests**, including all eight added boundary cases, with eight existing marker deselections and 18 warnings. Scoped Ruff 0.16.0, ty 0.0.65 and `git diff --check` passed. This is a five-file regression result, not a full-repository or GPU pass.


## Clustered-selection host block certificate

Clustered selection validation reuses its existing private feature-ID lists to form owned uint32 keys from local row and feature ID. Sorting these keys certifies that a block has no duplicates or positive/negative overlap. Any collision runs the original complete row loop, preserving earliest-row, positive-duplicate, negative-duplicate and minimum-overlap error ordering. Both original tensor-to-list checkpoints and all preceding validation remain unchanged.

For one complete genuine 282,343-segment serialized-selection publication call, excluded warm AB preceded all eight fixed ABBABAAB samples. Mean wall time fell **0.670590 → 0.330975 s (50.6442%)** and process CPU **1.767627 → 0.850561 s (51.8812%)**. All four predeclared gates passed, including both halves and position classes. Report-only adjacent CPU pair 2 regressed **1.1243%** and remains included. All ten calls matched the full six-field independent semantic oracle, integrity/digest tuple, actual width 94 versus family bound 96, immutable inputs and RNG/state.

The ordinary CPU component used Torch threads 16/inter-op 32 and owned threads at nice 19. Whole serialized load, validation, fingerprinting and width computation were timed; setup and verification were outside clocks. Process CPU includes target workers and spin. Scratch scales with block ID count, including a temporary Python concatenation list and owned NumPy arrays; no memory, GPU, full-producer, training or universal caller improvement follows. The earlier Tensor certificate implementation remains rejected. Exact samples, adverse metrics and immutable qualification receipts are in [clustered-selection-host-certificate-local-cpu.json](evidence/clustered-selection-host-certificate-local-cpu.json). The five scoped caller suites produced **471 passes, two failures, three skips and one deselection**; all four new cases passed. Both exact `dev` signature failures reproduced with the original whole cluster module installed before imports; only those two baseline cases were repeated, with no test fixes or exclusions. Scoped Ruff 0.16.0, ty 0.0.65 and `git diff --check` passed. This is not a clean-suite, full-repository, full baseline-suite or GPU validation claim.


## Packed reconstruction row endpoints

The shared `_build_batch_index` helper derives first and last segment indices from cumulative row counts when every selected row has segments. It retains the original row scan when a row is empty, preserving the existing error behavior; all-zero batches still return defined zero segment maps and document padding. Both precomputed reconstruction and live selection use this helper.

A complete natural 16,384-row packed/selection shard (1,024 batches of 16, context 2,048, segment padding 23, feature padding 96, one effective positive bucket) passed exact eight-field independent references and input/RNG checks. The first fixed panel excluded only warm AB and retained all eight ABBABAAB samples: pooled wall time fell from 1.475717041 to 1.094919272 seconds (**25.80%**) and process CPU from 23.290011580 to 17.266761072 seconds (**25.86%**). Both halves, both positions and every report-only pair improved; the baseline 2.259154485-second sample remains included. Native Torch threads were 16/inter-op 32 and owned threads used nice 19. These are CPU reconstruction component results with retained outputs, including observed baseline variation, not endpoint-only, full-family, training, GPU or universal caller speed claims. The five scoped caller suites passed 152 tests with one skip, including all eight new regression cases. Scoped Ruff/ty, Vulture and `git diff --check` passed. Whole-project checks retain baseline failures: two Ruff RUF007 findings, 415 ty diagnostics and four production-caller symbols (one failed ratchet test, four passes). These reproduced on the pristine parent checkout. The original strict ty log comparison failed on 13 absolute-versus-relative headers for the same dependency environment; both independent audits confirmed every remaining byte equal after canonicalizing only those source-location headers. This is not a clean whole-project validation claim.
