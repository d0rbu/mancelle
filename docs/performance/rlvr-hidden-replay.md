# Hidden-state reuse during head-only RLVR replay

An optional cache reuses the frozen trunk's complete normalized feature-head
inputs within one collected group. Replay evaluates the complete live head
before native position selection and pooling. The default budget is zero;
`hidden_replay_max_cache_bytes=268435456` permits up to 256 MiB of owned hidden
states and prompt snapshots. Model parameters, optimizer entries, and
checkpoint tensor schemas stay unchanged. See the
[training contract](../pipelines/encoder-decoder-rlvr.md#hidden-replay-cache).

## Measured result

On 2026-09-20, production source
`32f439d8f214a2f22e2234c1e53f82d2fcfc6760` ran the admitted 3B step-1116
checkpoint on one RTX 4090 with PyTorch 2.13.0+cu130, BF16 compute, and
deterministic SDPA. Only the linear feature head was trainable. Both arms
retain compiler state, use CPU SDPA positions, and enable identical frozen
decoder staging with a 2 GiB budget and 4 GiB headroom. Only the hidden-cache
budget and output directory differ.

| Workload | Native replay median | Cached replay median | Median reduction | Mean reduction |
|---|---:|---:|---:|---:|
| Four prompts, G=2, MC=16 / GSM8K=64 | 2.623616 s | 2.584727 s | 1.48% | 5.30% |
| Four prompts, G=4, MC=16 / GSM8K=512 | 31.842968 s | 27.228881 s | 14.49% | 13.93% |

Each arm starts in a fresh process with one real warmup update. Short uses
five measured updates, candidate first; full uses three, reference first.
All eight paired updates are faster. Three updates per workload have nonzero
gradients. The short/full workloads generate 638/4,575 measured tokens and
53/176 live feature actions.

All eight saved payloads match at zero tolerance: trajectories and rewards,
192 tensors, and 454 Python floating scalars, including clipped head gradients,
updated parameters, and full optimizer state. Token IDs, other scalar types,
and container structure are also checked recursively but are not counted in
those totals. Warmup non-timing receipts match; full warmup tensor snapshots
are not saved.

Including warmup, the short/full candidates admit and close all 24/16 groups
and hit all 39/191 canonical replay calls. They capture 39/192 behavior
states; one full-workload state has no live replay. There are no misses,
invalidations, skipped captures, or optional-copy OOMs. Peak owned tensor
payload is 2,307,170/51,258,240 bytes, including prompt snapshots. Peak allocated
CUDA memory rises by 798,720/43,527,168 bytes. Owned payload, allocator
reservation, and peak CUDA allocation are distinct measures. Every successful
group reports zero retained owned tensor objects at cleanup.

[Machine-readable evidence](evidence/rlvr-hidden-replay-4090.json) preserves
samples, exact comparisons, source/runtime/input identities, lifetime counters,
qualification, and ownership observations. Approximately one-second polling
observed no other compute PID during either measured workload, with recorded
coverage within 1.5 seconds. It cannot exclude subsecond processes, graphics
work, or host contention.

Outer CUDA synchronization bounds collection, native reward, replay/backward,
clipping, and optimization. Cache admission, copies, guards, fresh prompt
comparisons, live-head replay, and cleanup are inside the timer. Persistent
decoder staging setup/cleanup, loading, checkpoints/restart, validation,
telemetry, and receipt serialization are excluded. One process per arm and
workload provides neither confidence intervals nor a randomized noise
estimate; these are complete compute-update measurements.

## Mechanism evidence

An earlier diagnostic on source `1c89fe318a115205ea3097a1ce6d7c22c2bf5e9d`
executed the original canonical bodies without skipping work. All 39 observed
behavior/replay head-input pairs and pooled logits matched exactly in values
and bits, with matching native receipts and saved parity payloads. It observed
complete contiguous BF16 head inputs before selection and pooling. Its CPU
copies and hooks perturb execution, so its durations are not performance
evidence.

A separately qualified ignored prototype reduced short/full medians by
3.28%/10.55%, with exact saved parity. Those historical screening results are
separate from the production measurements above. Production uses explicit
group sessions and a shared canonical readout path, without the prototype's
global dispatcher or observation hooks. Its hit and lifetime receipts confirm
actual reuse; no production kernel-duration reduction is inferred.

## Correctness and operational limits

Focused CPU checks passed 342 unique cases before the final source freeze;
affected cases were rerun after the final fixes. The four-case orchestration
and one-case settings reruns repeat cases within that total. One policy Gloo
case initially encountered sandbox TCP permission denial and passed its
isolated retry with CUDA hidden.

The pinned production source passed 74 CUDA cases and one enabled real-RustBPE
CLI restart case. A separate ignored CUDA profiler-compatibility case passed
with cache hits, exact head gradients and RNG, and cleanup before profiler
exit, without materializing a FunctionEvent tree. It establishes that
fixture's compatibility, not full-model profiling performance.

Coverage includes native output/gradient parity and S0 sharing; prompt
mutations through ordinary writes, `.data`, and independent aliases; frozen
state/binding invalidation; allocation/key bounds; injected optional-copy OOM;
unsupported execution; ownership; and exception cleanup. A depth-2 test
compares three native generated updates with scripted token choices and mixed
rewards. All six head tensors have nonzero gradients and change each step;
gradients, parameters, optimizer state, and frozen contents match exactly,
with decoder staging enabled in both arms.

The real CLI test compares checkpoint-every-step reexec, a resume fork, and
uninterrupted training. Fork state/optimizer/RNG match the checkpointed run;
uninterrupted model/optimizer state also matches. It verifies standalone
evaluation and per-worker session cleanup. It does not require nonzero
gradients; the depth-2 test supplies that coverage separately.

Admission requires the supported single-rank CUDA head-only native model and
frozen-state contract. Unsupported framework inspection, execution, states,
mutations, or memory availability use native replay. Zero budget performs no
framework admission or cache work. Frozen-model `.data` writes, unregistered
independently versioned aliases, concurrent mutation, and deliberate framework
or operational-cache tampering remain unsupported. Fresh contents of all four
prompt tensors are checked independently of version counters.

The 4 GiB headroom estimate includes reusable allocator reserves and is not an
OOM guarantee. Only OOM from optional copies is caught; native computation
errors propagate. No natural OOM recovery experiment was run. Sessions release
ownership before optimizer, checkpoint, or restart; an escaping traceback or
retained consumer may temporarily retain a borrowed tensor and is reported
without masking the original error. Both settings are pinned on exact resume.
No full-suite, Hopper, multi-GPU, full-policy acceleration, depth-2 production
throughput, or cross-version exact-resume claim is made.

## Reproduce

Use the [native benchmark and exact comparison](rlvr-head-only-compiler.md#reproduce)
with identical admitted inputs and fresh output directories. Keep
`compiler_reset=null`, `train_feature_head_only=true`,
`frozen_decoder_max_shadow_bytes=2147483648`, and
`frozen_decoder_min_free_bytes=4294967296` in both arms. Use
`hidden_replay_max_cache_bytes=0` for reference and `268435456` for candidate;
keep `hidden_replay_min_free_bytes=4294967296` in both.

Short uses `'row_indices=[0,8,16,24]'`, `group_size=2`, `mc_max_tokens=16`,
`gsm8k_max_tokens=64`, `warmup_steps=1`, `measure_steps=5`, and
`'parity_steps=[0,1,2,3,4]'`. Full uses `'row_indices=[0,10,16,26]'`,
`group_size=4`, `gsm8k_max_tokens=512`, `measure_steps=3`, and
`'parity_steps=[0,1,2]'`. Keep `measure_phases=false` and `profile=false`.
Compare every saved trajectory, clipped gradient, updated parameter, and
optimizer state at zero tolerance. Verify every candidate group is admitted,
every canonical replay hits, and cleanup leaves zero misses, invalidations,
optional-copy failures, or retained owned tensor objects.
