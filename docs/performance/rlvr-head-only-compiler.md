# Head-only RLVR compiler lifetime

The head-only RLVR trainer retains compiled kernels between optimizer steps.
Full-policy RLVR and structured-SFT GRPO retain their existing per-step resets.
The model forward, feature sampling, generated tokens, reward, loss, clipping,
and optimizer are unchanged.

## Measured results

On 2026-09-20, a single RTX 4090 used PyTorch 2.13.0+cu130, BF16 compute,
deterministic SDPA, and the 2,988,187,138-parameter step-1116 checkpoint. Only
the 1,405,260 parameters in its linear feature head were trainable.

| Workload | Reset each step | Retain compiled kernels | Latency reduction | Speedup |
|---|---:|---:|---:|---:|
| Four prompts, G=2, MC=16 / GSM8K=64 | 26.962 s | 2.920 s | 89.17% | 9.23x |
| Four prompts, G=4, MC=16 / GSM8K=512 | 59.967 s | 35.106 s | 41.46% | 1.71x |

Values are medians after one real warmup update in each fresh process. The
short workload measured five updates. All five paired receipts matched exactly, including
generated token/feature trajectories, rewards, likelihood records, 120 tensors
and 163 floating scalars. Three measured updates had nonzero gradients.
Generated-token counts were `[134, 134, 87, 134, 149]` in both arms. The first
cold update took 28.566 / 28.965 seconds; peak measured CUDA allocation was
9.8703 / 9.8711 GiB. Checkpoint worker restarts still incur cold compilation,
so this steady-state reduction is not a campaign-wide speedup claim.

The full-budget workload measured three updates, all with nonzero gradients.
Its three paired receipts also matched exactly: generated trajectories, rewards,
72 tensors, and 291 floating scalars. Generated-token counts were
`[1306, 1609, 1660]`; gradient norms were `[10.3392, 23.1193, 10.8312]` in both
arms. Cold updates took 61.681 / 61.818 seconds, and peak measured allocation
was 9.9200 / 9.9198 GiB. The control and candidate shared final production
source `93f70c0c9`, with only the benchmark's compiler-reset override differing.
The short comparison used the earlier unchanged-production harness commit
`e75d81dce` with explicit reset/retain overrides. These sequential samples
establish the measured deltas on this device; they are not confidence intervals.

[Machine-readable evidence](evidence/rlvr-head-only-compiler-4090.json) contains
every timing sample, parity result, exact rows, configurations, runtime and
artifact pins, source hashes, and separate host/GPU trace summaries. Two initial
load attempts overlapped another GPU process and failed with OOM; they are
excluded from these successful matched comparisons.
One-second compute-process polling observed only each benchmark's own worker
throughout all four successful warmup/measurement windows. This does not
exclude subsecond transients, graphics activity, or host contention.

## Bottleneck evidence

A CPU/CUDA trace on the local RTX 4090 used the authenticated step-1116 3B
checkpoint, real RustBPE/Gemma tokenizers, and the bounded four-task RLVR bundle.
It covered native autoregressive feature and token generation, native reward,
head-only replay/backward, and the one-rank distributed optimizer. Four prompts
used two trajectories each, 16-token multiple-choice budgets, and a 64-token
GSM8K budget. Generated-assistant feature cadence was 33.

After the per-step compiler reset, 11 `entire_frame_compile` regions occupied
26.68 seconds. Encoder host regions occupied 25.06 seconds, versus 0.143 seconds
in their GPU annotation spans. Replay/backward host regions occupied 0.649
seconds. These regions overlap and must not be added together. The trace
therefore selected repeated compiler setup as the first optimization target.

The complete Chrome trace was retained. Its optional PyTorch operator-table
aggregation was interrupted after export because it remained CPU-bound for
several minutes. The benchmark now exports the trace directly without that
aggregation. Traced and internally synchronized diagnostic steps are excluded
from throughput samples.

## Reproduce

Use an admitted checkpoint, its exact RLVR bundle and statistics, and one free
CUDA device. Paths in the example are placeholders for those existing inputs.
Run the two arms in fresh processes with identical rows, seeds, warmup count,
and measured-step count:

```bash
uv run torchrun --standalone --nproc_per_node=1 \
  -m scripts.benchmark.benchmark_encdec_rlvr -- \
  base_checkpoint_dir=/path/to/base model_step=1116 \
  bundle_dir=/path/to/bundle stats_path=/path/to/statistics.safetensors \
  output_dir=out/rlvr-reference 'row_indices=[0,8,16,24]' \
  group_size=2 gsm8k_max_tokens=64 warmup_steps=1 measure_steps=5 \
  'parity_steps=[0,1,2,3,4]' compiler_reset=true
```

Repeat with a fresh `output_dir` and `compiler_reset=false`. The default
`compiler_reset=null` follows the trainer's selected scope: retain for
head-only, reset for full-policy. Warmup performs real optimizer updates, so
both arms must start from the same checkpoint and include identical warmup.
For the full-budget pair, use `'row_indices=[0,10,16,26]'`, `group_size=4`,
`gsm8k_max_tokens=512`, `measure_steps=3`, and `'parity_steps=[0,1,2]'` in both
arms. All workloads include ARC-Easy, ARC-Challenge, MMLU, and GSM8K.

`results.json` records synchronized complete-update times, exact prompt rows,
runtime/source/artifact identities, rewards, replay ratios, gradient norms,
and peak allocated memory. `parity-*.pt` files contain semantic trajectories,
clipped gradients, updated trainable parameters, and optimizer state. Load
them with `torch.load(..., weights_only=True)` and compare corresponding
steps. Timing and serialization receipts are outside the parity contract.
Frozen parameter checks use mutation versions and absent gradients, rather
than a second full-model snapshot.

For the exact comparison used here, run this with the two output directories:

```python
from pathlib import Path

import torch as t


def identical(left, right):
    assert type(left) is type(right)
    if isinstance(left, t.Tensor):
        assert left.dtype == right.dtype and t.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            identical(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            identical(a, b)
    else:
        assert left == right


reference, candidate = Path("out/rlvr-reference"), Path("out/rlvr-candidate")
files = sorted(reference.glob("parity-*.pt"))
assert files and [p.name for p in files] == sorted(p.name for p in candidate.glob("parity-*.pt"))
for source in files:
    left = t.load(source, weights_only=True, map_location="cpu")
    right = t.load(candidate / source.name, weights_only=True, map_location="cpu")
    for key in ("groups", "clipped_gradients", "updated_parameters", "optimizer"):
        identical(left[key], right[key])
```

For attribution, add `measure_phases=true` or `profile=true`. These perform
additional optimizer steps after the measured samples and write separate
diagnostics. Chrome traces can be large; begin with a bounded generated-token
budget. Profiling leaves native callables unchanged and records native operator
events plus the outer `rlvr/collection`, `rlvr/replay_backward`, and
`rlvr/clip_optimizer` scopes, with the same behavior whether replay caching is
enabled or disabled. It does not add inner encoder or canonical-policy wrappers.
The measured boundary includes compiler reset when enabled,
collection, reward, replay/backward, clipping, and optimizer at a fixed learning
rate. It excludes
startup/artifact admission, validation, checkpoint I/O, W&B, receipt writing,
and the trainer's task-selection/telemetry bookkeeping. It is a complete
compute-update measurement, not an end-to-end campaign duration.

## Correctness boundary

The added CUDA depth-2 regression compares three reset/retain updates with
scripted token choices that force mixed binary rewards. Native token logits,
feature sampling, packed encoding, cross-attention, and the fused optimizer
remain active. Every head tensor has a nonzero gradient and changes each step;
token logits, feature trajectories, gradients, updated parameters, and optimizer
state match exactly. Frozen parameters remain unchanged. This controlled test
qualifies depth-2 numerical behavior separately from the native-generation 3B
performance measurements above.

The real-tokenizer, one-GPU, depth-2 CLI test also passed. It checks exact model,
optimizer, and RNG agreement for a fork from step 1, exact model/optimizer
agreement between three uninterrupted updates and checkpoint-every-step worker
restarts, frozen trunk ownership, and standalone evaluation on all four tasks.
Three additional CUDA compiler/replay checks passed, including fail-on-exhaustion.
The focused CPU/runtime and code-quality run reported 112 passed and 36
deselected. Its sole failure was the production-caller audit's four unrelated
symbols, reproduced identically on untouched main `967c25dc8`. Ruff, targeted
Ty, and the documented exact receipt comparison passed. The full repository
suite and multi-GPU qualification were not run.

The original reset protected against uncheckpointed compiler fallback state.
RLVR now rejects suppressed compiler errors and fails on recompilation-budget
exhaustion. Every nonfinal checkpoint still restarts the worker process, so
ordinary continuation and explicit recovery both rebuild process-local state.
Head-only replay keeps the trunk in collection's no-gradient mode. These
controls make retention a candidate for numerical qualification; they do not
replace measured trajectory, gradient, optimizer, and restart comparisons.

The dynamic packed-attention specialization count has no proved global bound.
The configured limits and fail-closed behavior remain unchanged. GPU/runtime,
head architecture, rollout length, and checkpoint cadence remain qualification
dimensions; a short 4090 result does not establish multi-GPU or Hopper parity.
