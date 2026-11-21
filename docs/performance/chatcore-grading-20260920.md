# ChatCORE HumanEval grading performance — 2026-09-20

HumanEval previously started a multiprocessing manager and an execution worker
for every program, although its caller consumes only the success boolean.
The project adapter now passes a shared boolean to the unchanged nanochat
execution worker. It retains the temporary directory, reliability guard, memory
limit, execution alarm, and parent timeout grace. The parent also reaps the
worker before closing its process handle. No vendored or model code changes.

## Evidence and experiment

The baseline is `967c25dc8c82cde266c2b591ff1d018de06133bf`. A CPU/CUDA profile of
the real d26 SFT checkpoint on an RTX 4090 identified the grading overhead:
the separate 64-token generative profile took 7.588 seconds, including 5.519
seconds in 50 HumanEval evaluations and 100 process starts. This motivated
removing the result-manager process while keeping the execution worker.

The final implementation was compared on one NVIDIA L40S in Engaging job
`23284891`, node `node3006`, using `mit_normal_gpu` with account and QoS
`mit_amf_advanced_gpu`. PyTorch was `2.13.0+cu130`, Triton `3.7.1`, and the NVIDIA
driver `590.48.01`.

Each suite evaluates 50 problems from each of ARC-Easy, ARC-Challenge, MMLU,
GSM8K, HumanEval, and SpellingBee, with the production 512-token generation
budget. The checkpoint is d26 SFT step 999, with 26 layers, width 1664,
13 attention/KV heads, and context length 2048. Its SHA-256 is
`d3d320a5b4e3985a6915d70d51f63f4500ace6c12bc7bbd867fec74ab50f7371`.

The process order was reference, candidate, candidate, reference. Every process
used fresh Triton and Inductor cache directories and ran three complete suites.
The first suite includes compilation; the other two are warm repetitions.
CUDA synchronization brackets every suite. Model loading, input hashing,
artifact writes, profiling, and numerical capture are outside the reported warm
timings. GPU process telemetry checks for unrelated GPU work throughout each
arm. A local attempt interrupted by unrelated GPU work was excluded.

| Measurement | Reference | Candidate |
|---|---:|---:|
| Median warm complete-suite latency, four repetitions | 10.9683 s | 10.1263 s |
| Warm range | 10.9415–11.0507 s | 10.0896–10.1883 s |
| First suite in each fresh-cache process | 24.9557 / 17.2023 s | 16.3107 / 16.1611 s |
| Peak allocated CUDA memory | 17.30 GiB | 17.30 GiB |

The warm latency reduction is **0.8420 seconds, or 7.68%**. First-suite results
are reported separately; they include compilation and are not a model-loading
or universal cold-start estimate. This qualifies the measured standalone suite
on this L40S, not every checkpoint, task population, GPU, or distributed setup.

## Correctness and checks

The comparison also runs separate instrumented reference/candidate suites for
full-vocabulary logit samples from four rows every 64 ragged decode calls.
Their timings are excluded from the latency comparison. Raw result artifacts
include complete generated token lists, each HumanEval program's hash/verdict,
source and input hashes, sampled logits, scores, and memory peaks.

All 14 complete suites had identical six-task scores, 42,089 completion tokens,
completion bytes, and 50 per-program grading verdicts. ChatCORE was 0.2867.
The separate audit compared 1,048,576 logit values over eight sampled decode
calls with **zero absolute difference** (`atol=rtol=0`). All 320 GPU process
telemetry samples were free of unrelated work. The job completed with exit
`0:0`. The [machine-readable evidence](evidence/chatcore-grading-20260920.json)
retains all individual timing/memory records, score/output hashes, source/input
hashes, and verification counts. Full raw artifacts remain under
`/home/henryac/chatcore-perf-20260920/job_23284891` on Engaging; the verified
comparison JSON has SHA-256
`eb3ce83f8fe7ea7ab136242b20cdfcff12987b7c9db361a84193e92ab7f3323f`.

An additional replay compares all 164 canonical HumanEval solutions and all 50
RTX-baseline generated programs with the original grader: 214 exact verdict
matches, with all 164 canonical programs passing. The 66 focused tests cover
worker success/failure, stdout/stderr-producing programs, syntax/assertion errors, memory limits,
execution alarms, a disabled alarm requiring the parent deadline, worker exit,
cleanup failure after a successful verdict, supported multiprocessing start
methods, task parsing, and ChatCORE integration.

Changed-file Ruff and ty checks and repository-wide Vulture pass. Unchanged
baseline failures were reproduced on an untouched export of the baseline:
two Ruff findings in `test_session_rewind.py`, the same 442 ty diagnostics,
and four existing production-caller findings. These broad checks are not claimed
green.

## Reproduction

Point `NANOCHAT_BASE_DIR` at the checkpoint's RustBPE tokenizer assets and use
the same cached task revisions. The measured cached datasets were ARC
`210d026faf9955653af8916fad021475a3f00453`, MMLU
`c30699e8356da336a370243923dbaf21066bb9fe`, GSM8K
`740312add88f781978c0658806c59bc2815b9866`, and HumanEval
`7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544`; SpellingBee uses its deterministic
production test generator.

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
TRITON_CACHE_DIR=/tmp/chatcore-reference-triton \
TORCHINDUCTOR_CACHE_DIR=/tmp/chatcore-reference-inductor \
uv run python -m scripts.benchmark.benchmark_chatcore \
  checkpoint_dir=/path/to/d26/standard_sft_checkpoints step=999 \
  output_dir=out/chatcore/reference grading=reference
```

Repeat with `grading=candidate`, distinct output/cache directories, and an
interleaved process order. `profile=true` adds a separate short generative
profile. `record_logits=true repeats=1` records the separate numerical audit;
its measured times must not be pooled with uninstrumented timing runs.
