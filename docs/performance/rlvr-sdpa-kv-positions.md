# SDPA KV-cache position placement

Project `InferenceEngine` CUDA caches keep their position counter on CPU when
the resolved cached-attention backend is SDPA. K/V tensors remain on the model
device. Native FA3 keeps its device-resident counter. The ordinary `KVCache`
still owns advance, rewind, reset, and prefill; there is no duplicate host
position to keep synchronized. Stacked and compacted caches use the same
allocation policy. Attention backend selection must remain fixed for the
lifetime of a cache, as it does in the production entrypoints.

## Measured result

This follows [compiler retention](rlvr-head-only-compiler.md): both arms already
retain compiled kernels. On 2026-09-20, the same admitted 3B step-1116 model ran
native autoregressive token/feature RLVR with only its linear feature head
trainable, on one RTX 4090 with PyTorch 2.13.0+cu130 and deterministic SDPA.

| Workload | CUDA position counter | CPU position counter | Latency reduction |
|---|---:|---:|---:|
| Four prompts, G=2, MC=16 / GSM8K=64 | 2.938 s | 2.706 s | 7.91% |
| Four prompts, G=4, MC=16 / GSM8K=512 | 35.057 s | 33.173 s | 5.37% |

The short workload uses five measured complete updates after one real warmup
update in each fresh process. All five paired receipts match exactly:
generated trajectories and rewards, 120 tensors, and 163 floating scalars.
Three updates have nonzero gradients. Every paired update is faster; the
reported delta uses the two medians. Cold updates take 27.996 / 28.036 seconds.
The control's cold warmup briefly overlapped a second 450–452 MiB GPU process,
so these cold timings are not an isolated startup comparison. That process
ended before the measured updates; one-second ownership polling observed no
competing compute process during any measured sample in either workload.
Polling cannot exclude subsecond activity, graphics work, or host contention.

The full-budget workload uses three measured updates, all with nonzero
gradients. It also matches exactly: generated trajectories/rewards, 72 tensors,
and 291 floating scalars. Generated-token counts are `[1306, 1609, 1660]` in
both arms. Cold updates take 61.895 / 59.663 seconds. Arm order is reversed:
the short control runs first, while the full-budget candidate runs first.
Every paired full-budget update is faster, too.

[Machine-readable evidence](evidence/rlvr-sdpa-kv-positions-4090.json) preserves
all samples, numerical comparisons, source/runtime/input hashes, and trace
attribution. One initial control attempt failed during model loading because
another GPU process started after the availability check; it supplied no
performance samples and is excluded.

The benchmark includes collection, reward, replay/backward, clipping, and the
optimizer, with outer CUDA synchronization. Startup, checkpoint/restart,
validation, and telemetry overhead are excluded. These are scoped compute
measurements, not campaign-wide speedups or confidence intervals. The control
is PR #1728 at `58a68a74f`; the candidate is `9147b00e9`.

## Trace evidence

A separate warmed CPU/CUDA trace recorded 7,627 `aten::item` calls occupying
0.2090 seconds of inclusive host time. Direct operator attribution identified
6,425 scalar-triggered `cudaStreamSynchronize` calls occupying 0.1551 seconds,
and 6,425 scalar D2H copies transferring only 28,077 bytes. The native SDPA
cached-attention path reads the position once in every decoder layer; model
and self-conditioning guards also read it. The scalar totals also include
other guards. Keeping position metadata on CPU
removes those device synchronization boundaries without changing attention
arithmetic or cache mutation order. In a matched candidate profile of the same
logical update, scalar-attributed D2H copies and synchronizations fell from
6,425 to 1,989: 4,436 fewer of each. The scalar operator counts stayed at
7,627. Total stream synchronizations fell from 8,835 to 4,399. This confirms
the intended mechanism; these profile durations are excluded from the timing
comparison above.

The control's profiled update took 5.8317 seconds; actual GPU kernel duration was 2.6151
seconds. Inclusive CPU regions overlap, and GPU annotation spans include
launch gaps. These diagnostic times are separate from the uninstrumented
throughput samples. Weight casts remain a larger measured kernel cost and
are a separate optimization candidate.

## Verification and reproduction

Native CUDA tests compare CPU-position and ordinary CUDA-position caches
through prefill, decode, rewind, reset, prefill-copy, stacking, and compaction.
Outputs and K/V contents match exactly, and heterogeneous positions still fail
before writes. Fixed-token and turn-delimited boundary failures restore the
position before retry. Backend-policy tests execute the actual dispatcher,
including forced SDPA on an FA3-capable configuration and FA4's cached-attention
fallback. The focused CPU suite passed 225 tests, and five native CUDA cache
and rollback cases passed. Final-source CUDA qualification also passed a
depth-2 head's three updates comparing compiler reset with retention; scripted
token choices and mixed rewards ensure nonzero gradients while logits, feature
sampling, backward, and optimization remain native. Real-tokenizer CLI
checkpoint/reexec/resume equivalence also passed with exact model and optimizer
state; that test does not require nonzero gradients. The benchmark pairs above
separately establish exact pre/post-change parity. These tests do not claim
native Hopper FA3 execution or multi-GPU qualification.

Use the [complete-update benchmark and exact receipt comparison](rlvr-head-only-compiler.md#reproduce)
in fresh control/candidate checkouts. Keep `compiler_reset=null` in both arms;
use `'row_indices=[0,8,16,24]'`, `group_size=2`, `gsm8k_max_tokens=64`,
`warmup_steps=1`, `measure_steps=5`, and `'parity_steps=[0,1,2,3,4]'`.
Every saved trajectory, clipped gradient, updated parameter, and optimizer
state is compared at zero tolerance.
For the full-budget pair, use `'row_indices=[0,10,16,26]'`, `group_size=4`,
`gsm8k_max_tokens=512`, `measure_steps=3`, and `'parity_steps=[0,1,2]'`.
