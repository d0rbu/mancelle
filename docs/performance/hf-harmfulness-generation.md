# Repository-HF Harmfulness Generation Performance

This note records the first optimization loop for the standalone
repository-HF harmfulness generator. The checked-in evidence is
[`hf-harmfulness-generation-evidence/rtx4090-smollm3-3b.json`](hf-harmfulness-generation-evidence/rtx4090-smollm3-3b.json).

## Experiment and bottleneck

The fixed workload used a strict local SmolLM3 3B BF16 export, four harmful
and four benign prompts, 32 generated tokens, temperature 0.7, top-k 50, and
seed 4242 on an RTX 4090. Judge/API work was excluded.

The former `generate_batch` implementation called `model.generate()` once per
prompt. Its trace contained 64,768 `aten::mm` calls and 35,968 single-row GEMV
kernels for only 256 output tokens. GEMV accounted for 1.74 seconds of 2.28
seconds total CUDA self-time. The model was therefore launch- and
memory-bandwidth-bound on serial row-one decoder forwards, not on tokenization,
sampling, or attention.

## First-loop implementation

The first batched implementation rendered every prompt first, checked each
real prompt length, left-pads the outer batch, and calls Transformers generation
once. Greedy generation uses the ordinary greedy loop. Sampled generation uses
one `torch.Generator` per row, seeded with `base_seed + absolute_prompt_index`;
a logits processor applies that row's temperature and top-k before selecting
one token for the common batched loop.

Batching changed BF16 GEMM shape and could therefore produce numerically close
but non-bitwise-identical sampled trajectories. The exact batch size was
already response-cache identity. The distinct
`base_plus_absolute_prompt_index_hf_batch_v1` algorithm descriptor also
invalidated sequential HF response caches while leaving native and
self-conditioned caches untouched.

## Measured result

The matched-process benchmark measured these three synchronized samples:

| Shape | Samples (seconds) | Median |
|---|---|---:|
| Serial reference | 3.3901, 3.3157, 3.3221 | 3.3221 s |
| Batched production | 0.5813, 0.5813, 0.5816 | 0.5813 s |

That is a **5.72x speedup** and **82.5% lower latency**. A separate unprofiled
pair around the two profiler captures measured 2.5750 seconds versus 0.4607
seconds (5.59x). Peak allocated memory rose from 5,990.3 MiB to 6,037.6 MiB,
a 47.3 MiB increase.

The optimized trace contains 8,096 matrix multiplies, exactly one eighth of the
reference count, and no single-row GEMV kernels. CUDA self-time fell from 2.275
seconds to 0.400 seconds.

## Correctness evidence

Across the full 256-token greedy trajectory on the real 3B model:

- all eight decoded responses were byte-identical;
- next-token top-1 agreement was 100%;
- top-50 overlap averaged 49.33/50 and never fell below 46/50;
- logit mean absolute error was 0.0242, RMS error was 0.0360, and maximum
  absolute error was 0.4451;
- mean cosine similarity was 0.999929 and the minimum was 0.996872.

Both sampled serial and batched configurations repeated exactly across three
runs. Their sampled texts differ from each other because the BF16 forward batch
shape differs; this is the reason for the versioned algorithm/cache identity.
Unit tests separately prove that identical logits retain the same independent
absolute-prompt RNG streams across outer batch partitions.

## Reproduction

```bash
uv run python -m scripts.benchmark.benchmark_hf_harmfulness_generation \
  --checkpoint-dir /absolute/path/to/step_NNNNNN \
  --device cuda:0 \
  --iterations 3 \
  --num-harmful 4 \
  --num-benign 4 \
  --max-new-tokens 32 \
  --temperature 0.7 \
  --top-k 50 \
  --seed 4242 \
  --output out/profiling/hf-harmfulness/benchmark.json
```

Add `--profile-dir out/profiling/hf-harmfulness/traces` to capture one serial
and one batched Chrome trace plus key-average tables. Profiler timing is
diagnostic; use the synchronized unprofiled samples for headline latency.

## Follow-up: outer-batch occupancy

The next loop held one 32-prompt pool and the production 512-token generation
settings fixed while changing only the outer batch. The machine-readable
evidence is
[`rtx4090-smollm3-3b-batch-scaling.json`](hf-harmfulness-generation-evidence/rtx4090-smollm3-3b-batch-scaling.json).

| Outer batch | Median for 32 prompts | Prompts/s | Peak allocated |
|---:|---:|---:|---:|
| 8 | 32.2255 s | 0.9930 | 6,411.8 MiB |
| 32 | 14.5634 s | 2.1973 | 7,693.2 MiB |

Both shapes reproduced byte-identical outputs across two runs. Batch 32
therefore provides **2.21x throughput** at the default 512-token budget while
remaining far below the 24 GiB device limit on the measured SmolLM3 3B export.
At that stage, `repository_hf_batch_size` defaulted to 32; the native `batch_size`
default remains 8, and encoder-decoder self-conditioning still requires 1.
The effective backend batch is recorded in response provenance.

Reproduce the fixed-prompt comparison with:

```bash
uv run python -m scripts.benchmark.benchmark_hf_harmfulness_generation \
  --checkpoint-dir /absolute/path/to/step_NNNNNN \
  --device cuda:0 \
  --iterations 2 \
  --num-harmful 16 \
  --num-benign 16 \
  --max-new-tokens 512 \
  --temperature 0.7 \
  --top-k 50 \
  --seed 4242 \
  --batch-sizes 8 32 \
  --output out/profiling/hf-harmfulness/batch-scaling.json
```

## Follow-up: vectorized sampling transforms

At batch 32, the next trace showed that the independent-row sampling contract
was applying temperature, top-k, and softmax in 32 separate calls per decode
step. Top-k and softmax alone accounted for 2,048 calls and about 86.8 ms of
CUDA self-time over a 32-token decode.

At that stage, temperature, top-k, and softmax were applied to the complete
score matrix. Only `torch.multinomial` remained row-wise, with the same
per-prompt generators and the same rank-2 row slices. A row-wise oracle test
proved exact selection parity for identical logits, and the real model retained
the exact sampled response hashes before and after the change.

On the fixed 32-prompt, 512-token workload, median latency fell from 14.5634
seconds to 12.9746 seconds: **10.91% lower latency** (1.12x speedup). Peak
allocation changed by only 0.0625 MiB. The corresponding 32-token profile
reduced top-k and softmax from 1,024 calls each to 32 each and reduced measured
CUDA self-time from 0.8320 seconds to 0.7440 seconds. Machine-readable evidence
is
[`rtx4090-smollm3-3b-sampling-vectorization.json`](hf-harmfulness-generation-evidence/rtx4090-smollm3-3b-sampling-vectorization.json).

A full 1,200-prompt judged evaluation remains a separately authorized,
cost-bearing acceptance step and was not run in this loop.

## Follow-up: batch-level sampling and batch 64

The row-wise multinomial remained one launch per prompt per decode step. The
current sampler retains the matrix-wide temperature, top-k, and softmax work,
but now draws all rows with one matrix-wide `torch.multinomial` and one
batch-level generator. The generator is seeded with
`base_seed + batch_start_index`; changing the outer batch is allowed to change
the sampled trajectory. The `base_plus_batch_start_hf_batch_v2` response-cache
descriptor records that algorithm change, while repeating one fixed
configuration remains exact.

On the same 32-prompt, 512-token workload used above, the batch-level sampler
reduced median latency from 12.9746 to 11.9975 seconds: **7.53% lower latency**
(1.08x). The output changed as expected under the new RNG trajectory, and each
configuration reproduced byte-identically across two runs.

A fixed 64-prompt production-budget comparison then measured:

| Outer batch | Median for 64 prompts | Prompts/s | Peak allocated |
|---:|---:|---:|---:|
| 32 | 23.1039 s | 2.7701 | 7,721.1 MiB |
| 64 | 20.8140 s | 3.0749 | 9,568.9 MiB |

Batch 64 is **11.0% faster** than batch 32 on this workload. A separate
128-prompt, 32-token occupancy sweep ruled out batch 128: it reached only 29.20
prompts/s versus 42.99 for batch 64 and raised peak allocation to 13,506.1 MiB.
`repository_hf_batch_size` therefore defaults to 64. Native generation remains
at 8 and encoder-decoder self-conditioning still requires 1. The complete
machine-readable evidence is
[`rtx4090-smollm3-3b-batch-level-sampling.json`](hf-harmfulness-generation-evidence/rtx4090-smollm3-3b-batch-level-sampling.json).

No judge/API calls or full 1,200-prompt evaluation were run.
