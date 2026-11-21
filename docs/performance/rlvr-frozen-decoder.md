# Frozen decoder weight staging for head-only RLVR

Optional worker-scoped BF16 copies let frozen decoder linears reuse their
compute weights across token generation and replay. FP32 parameters, optimizer
ownership, and checkpoint tensor entries stay unchanged. The default budget is
zero; set `frozen_decoder_max_shadow_bytes=2147483648` to allow up to 2 GiB of copies.
The complete feature head, encoder, and cross-attention use their existing
implementations. See the [training contract](../pipelines/encoder-decoder-rlvr.md#frozen-decoder-weight-copies)
for admission, lifetime, and mutation constraints.

## Measured result

Both arms include [compiler retention](rlvr-head-only-compiler.md) and
[CPU SDPA positions](rlvr-sdpa-kv-positions.md). On 2026-09-20, the admitted 3B
step-1116 checkpoint generated both tokens and feature actions using native
RLVR collection, rewards, replay, and optimization. Only its linear feature
head was trainable. Measurements used one RTX 4090, PyTorch 2.13.0+cu130,
BF16 compute, and deterministic SDPA.

| Workload | Native weight casts | Reused BF16 weights | Median latency reduction |
|---|---:|---:|---:|
| Four prompts, G=2, MC=16 / GSM8K=64 | 2.715 s | 2.502 s | 7.82% |
| Four prompts, G=4, MC=16 / GSM8K=512 | 33.103 s | 30.058 s | 9.20% |

Each arm starts in a fresh process with one real warmup update. The short
workload has five measured updates; the full-budget workload has three.
Every paired update is faster. Arm order is reversed between workloads:
short reference first, full-budget candidate first. Both arms use production
source `4779b3dd743868a0fac0c7999a332c56093dc7d1`, changing only the staging
budget and output directory. A separate prototype screen found similar gains;
its results are recorded as historical evidence, separately from these native
production measurements.

All eight pairs match exactly at zero tolerance: generated trajectories and
rewards, 192 tensors, and 454 Python floating scalars, including clipped head
gradients, updated parameters, and full optimizer state. Token IDs, other
scalar types, and container structure are also checked recursively but are not
included in those counts. Three updates per workload have nonzero gradients.
The workloads generate 638 and 4,575 measured tokens, respectively.

The 123 selected linears require 1,903,308,160 bytes of persistent copies
(1.903 GB, 1.773 GiB). Peak allocated memory increases by 1.902 GB in the short
workload and 1.860 GB in the full-budget workload. Synchronized staging takes
7.44 / 7.65 ms; total context entry takes 11.97 / 12.48 ms. Amortized across
all six / four worker updates, including warmup, entry costs 2.00 / 3.12 ms per
update. Cleanup takes 0.37 / 0.50 ms. Both candidate runs report admission,
completion, cleanup, and zero fallback calls or invalidated modules.

[Machine-readable evidence](evidence/rlvr-frozen-decoder-4090.json) preserves
samples, exact comparisons, source/runtime/input identities, staging reports,
ownership observations, and final-source qualification. One-second polling
observed no other compute PID during any measured update. This cannot exclude
subsecond activity, graphics work, or host contention.

The timer includes collection, reward, replay/backward, clipping, and optimizer
work with outer CUDA synchronization. It excludes staging, cleanup, model
loading, checkpoint/restart, validation, telemetry, and receipt serialization.
These are scoped complete compute-update measurements. One process per arm
and workload does not provide confidence intervals or a randomized campaign
noise estimate. No full-suite, Hopper, multi-GPU, or campaign-wide performance
claim is made.

## Trace evidence

A separate native CPU/CUDA profile of logical update 4 generates 134 tokens
and 14 feature actions. Its non-timing/non-memory receipt fields match the
previous CPU-position profile, including rewards, advantages, replay metrics,
and gradient norm. The benchmark update function's AST is identical. The
control trace comes from the earlier CPU-position implementation; the candidate
uses the pinned production source above. These diagnostic runs are separate
from the fresh production timing pairs.

Native eager BF16 weight-copy kernels fall from 20,994 to 350. Remaining copies
have the feature-head weight/bias shapes: 336 carry feature-policy annotations,
and 14 bias-shaped copies lack those annotations. All 20,644 decoder-shaped
copies disappear (20,496 block projections and 148 LM-head projections). This
confirms the intended mechanism. Candidate staging remains admitted, with zero fallback/invalidation
and successful cleanup.

The post-change trace records 1.7972 seconds of actual GPU kernel work. Compiled
encoder casts still account for 9,028 kernels / 0.3689 seconds, and compiled
cross-attention casts for 13,608 / 0.2698 seconds. Scalar-attributed D2H copies
and stream synchronizations remain at 1,989 each, with 7,627 `aten::item` calls.
These residual costs are candidates for subsequent experiments.

The profiled update takes 5.5057 seconds with instrumentation; that duration is
not a throughput sample. Inclusive CPU regions overlap, and annotation spans
include launch gaps. [Profile evidence](evidence/rlvr-frozen-decoder-profile-4090.json)
pins both traces, attribution scripts, workload matching, source differences,
and kernel counts without treating instrumented duration changes as gains.

## Correctness and operational limits

Focused checks passed 128 CPU tests; the pinned final source passed 37 CUDA
tests. Four Gloo cases initially failed because the sandbox prohibited TCP sockets; all four
passed when rerun with networking permitted and CUDA hidden. The CUDA checks
include native linear output/input-gradient parity, noncontiguous inputs,
biases, parameter/state-dict/optimizer identity, storage aliases, versioned
mutation and replacement, dtype fallback, budget/headroom refusal, injected
partial staging OOM, cross-stream allocator use, inference-mode entry, and
cleanup after exceptions.

A separate depth-2 head test compares three native updates with scripted token
choices and mixed rewards, ensuring all six head tensors have nonzero gradients
and change each step. Logits, features, gradients, parameters, optimizer state,
and frozen model contents match exactly. The enabled real-tokenizer CLI test
checks checkpoint-every-step reexec, resume forks, and uninterrupted training;
model and optimizer comparisons pass across all three paths, and RNG state
matches between the resumed fork and checkpointed run. Every worker admits
staging and reports no fallback. That CLI test does not require nonzero gradients;
the depth-2 test provides that coverage separately. Ruff and targeted production
and new-test type checks pass. The existing runtime test file retains 18 type
diagnostics reproduced on its untouched version.

Staging is admitted only for single-rank CUDA head-only BF16 execution with
eligible contiguous FP32 decoder weights. Budget/headroom refusal, unsupported
execution, or staging-only allocation failure leaves native forwards active.
Admission includes reusable allocator reserves and is an estimate, not a
promise that future activation or KV allocations will fit. The 4 GiB default
headroom must be assessed for each workload.

Distinct registered parameters or buffers sharing selected storage disable
staging. Direct versioned mutation or replacement invalidates a copy and falls
back to the native forward. Model registrations and trainable scope must remain
fixed during the context; unversioned `.data` writes and writes through
unregistered independently versioned aliases are unsupported. Context entry
and exit must not race model use. Copies are ordinary tensors so input
gradients remain supported, and cross-stream use is registered with the CUDA
allocator. Checkpoint loading precedes entry; cleanup precedes reexec and
ordinary/error teardown. Copies never become checkpoint or optimizer entries.

## Reproduce

Use the [native benchmark and exact receipt comparison](rlvr-head-only-compiler.md#reproduce)
with identical admitted inputs and fresh output directories. Keep
`compiler_reset=null` and `train_feature_head_only=true` in both arms.
Run the control with `frozen_decoder_max_shadow_bytes=0`; run the candidate
with `frozen_decoder_max_shadow_bytes=2147483648`. Keep
`frozen_decoder_min_free_bytes=4294967296` in both.

For the short pair use `'row_indices=[0,8,16,24]'`, `group_size=2`,
`mc_max_tokens=16`, `gsm8k_max_tokens=64`, `warmup_steps=1`, `measure_steps=5`,
and `'parity_steps=[0,1,2,3,4]'`. For the full-budget pair use
`'row_indices=[0,10,16,26]'`, `group_size=4`, `gsm8k_max_tokens=512`,
`measure_steps=3`, and `'parity_steps=[0,1,2]'`. Compare every saved trajectory,
clipped gradient, updated parameter, and optimizer state at zero tolerance,
then verify the candidate staging report says applied/completed/cleaned up
with zero fallback calls and invalidations.
