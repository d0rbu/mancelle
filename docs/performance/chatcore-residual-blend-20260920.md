# ChatCORE residual-blend performance — 2026-09-20

The Ada ragged decoder now combines its two residual scalar multiplications,
addition, and final compute-dtype cast into one Triton kernel. Eligibility is
checked once per decode call. Unsupported devices, activation layouts/dtypes,
or scalar parameter devices/dtypes retain the eager expression. Vendored code
is unchanged.

## Measured target and arithmetic

A separate 64-token generative profile of the real d26 checkpoint on an RTX
4090 identified 3,276 residual blends. Correlating both operator external IDs
and CUDA launch IDs identified exactly four kernels per blend: 13,104 kernels
with 30.597 ms of kernel execution. Their first-to-last-kernel spans totaled
77.273 ms, including 46.676 ms of gaps. Host blend ranges totaled 161.442 ms,
including 32.671 ms in CUDA launch calls. These profiling numbers identify the
target; they are not measured complete-suite savings. The
[discovery attribution](evidence/chatcore-residual-blend-discovery-20260920.json)
records the trace hash, correlation method, and separate counts.

The eager rounding boundaries matter. CUDA RMSNorm autocast leaves the initial
embedding `x0` in FP32. The first layer's residual stream is also FP32; later
layers use BF16 or FP16. For those later layers, the CUDA scalar is first cast
to the residual stream's dtype, and its product is rounded to that dtype before
the FP32 addition with the `x0` product. The final result is cast to the compute
dtype. The fused kernel preserves those steps and disables floating-point
contraction. It reads the live learned scalar arrays on every call and writes
fresh output storage without mutating `x0` when the first-layer inputs alias. It does not
specialize on scalar values or assume identity/default coefficients.

## Complete-suite qualification

The reference is parent PR #1727 at
`c022c23afd683f53998ce42b103926242018f568`. Both arms retain its shared-boolean
HumanEval grader (`grading=candidate`). Full source manifests bind 3,515 parent
files plus the candidate's new helper/test. The only runtime differences are
the decoder integration and new residual helper; both snapshots use the same
benchmark instrumentation.

Engaging job `23289885` ran on one NVIDIA L40S on `node3402`, using
`mit_normal_gpu` with account and QoS `mit_amf_advanced_gpu`. Software was
PyTorch `2.13.0+cu130`, Triton `3.7.1`, and NVIDIA driver `590.48.01`.
The job completed with exit `0:0` in 8:45. Its separate post-comparison profile
recorded 1,638 calls to the fused residual kernel (4.295 ms of kernel time),
confirming that the candidate exercised the optimized path.
The checkpoint is d26 SFT step 999: 26 layers, width 1664, 13 attention/KV
heads, and context 2048. Its SHA-256 is
`d3d320a5b4e3985a6915d70d51f63f4500ace6c12bc7bbd867fec74ab50f7371`.

Each complete suite uses 50 problems per task across ARC-Easy, ARC-Challenge,
MMLU, GSM8K, HumanEval, and SpellingBee, with the production 512-token
generation budget. The process order is reference, candidate, candidate,
reference. Each process uses fresh Triton/Inductor cache directories and runs
five suites: one first suite with compilation and four warm repetitions.
CUDA synchronization brackets every suite. Model loading, input hashing,
artifact writes, profiling, and numerical capture are outside the warm timing.
Dataset loading uses the same offline caches.

| Measurement | Reference | Candidate |
|---|---:|---:|
| Median warm latency, eight suites per arm | 10.2943 s | 10.1338 s |
| Warm range | 10.2357–10.3447 s | 10.0941–10.2012 s |
| Warm medians of the two separate processes | 10.2709 / 10.3208 s | 10.1620 / 10.1165 s |
| First suite in each fresh-cache process | 16.8580 / 16.3813 s | 16.3714 / 16.6397 s |
| Peak allocated CUDA memory | 17.30 GiB | 17.30 GiB |

Warm latency fell **0.1606 seconds, or 1.56%**. Every candidate warm sample was
faster than every reference warm sample in this experiment. First-suite samples
overlap and are reported separately; they are not a general cold-start claim.
The result qualifies this standalone BF16 suite on this L40S. RTX 4090 and FP16
kernel correctness tests do not establish complete-suite speedups on those
other configurations. DDP performance was not measured.

All 22 timing/audit suites produced identical six-task scores, 42,089 generated
tokens and completion bytes, and 50 HumanEval program hashes/verdicts. ChatCORE
was 0.2867. Separate instrumented runs compared 1,048,576 full-vocabulary logit
values from four rows at decode calls 0, 64, 128, 192, 256, 320, 384, and 448:
maximum absolute difference was zero (`atol=rtol=0`), and input IDs matched.
Their instrumented timings are excluded from the performance comparison.
All 382 GPU process samples were free of unrelated work, and CUDA/nvidia-smi
UUIDs identified the same monitored GPU.

The [raw comparison](evidence/chatcore-residual-blend-20260920.json) retains all
individual timing/memory records, scores, output hashes, source/input hashes,
and verification counts. Its SHA-256 is
`586fa784b13b5dadf99799f088c5db566e11ccc44a63c3469a41506b98403839`.
Full artifacts and the controllers remain under
`/home/henryac/chatcore-perf-20260920/loop3_job_23289885` and its parent campaign
directory on Engaging.

## Tests and reproduction

All 36 kernel tests passed on both RTX 4090 and L40S. They compare directly with
eager CUDA arithmetic for both dtype branches and cover trained coefficients,
cancellation, FMA sensitivity, strided and mutated scalar arrays, input
aliasing/immutability, tails, empty shapes, and invalid inputs. Two additional
dispatch/integration tests verify fallback admission and decoder/cache parity.
The full CPU decoder test file reports 41 passes and three failures also
reproduced on untouched baseline code. The production-caller check retains its
four existing baseline findings. Changed-file Ruff/ty, repository Vulture, and
diff checks pass; broad repository checks are not claimed green.

Use isolated reference/candidate source snapshots, the same checkpoint/assets,
and the benchmark harness from this change in both snapshots:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
TRITON_CACHE_DIR=/tmp/chatcore-arm-triton \
TORCHINDUCTOR_CACHE_DIR=/tmp/chatcore-arm-inductor \
uv run python -m scripts.benchmark.benchmark_chatcore \
  checkpoint_dir=/path/to/d26/standard_sft_checkpoints step=999 \
  output_dir=out/chatcore/arm grading=candidate repeats=5
```

Set `NANOCHAT_BASE_DIR` to the matching tokenizer assets. Use fresh output and
cache directories for each ABBA process. Run separate
`record_logits=true repeats=1` audits and compare all completion/grade records,
input/source hashes, and sampled logits. Add `profile=true` only to a separate
profile arm. Source hashes explicitly record an absent new helper as `null` in
the reference snapshot.
