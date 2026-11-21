# AxBench-Adv latent counter decoder weight casts

This follow-up to [issue #1759](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/issues/1759)
uses the real 400M step-1013 encoder-decoder checkpoint on a local RTX 4090.
The workload is one `SelfConditioningMasked` concept (36), a shared zero and
both counter families at raw magnitude 256, ten responses per point, and a
deterministic local HTTP judge. The judge makes no paid provider calls.

## Bottleneck and rejected broader cast

A one-response CPU/CUDA profile recorded 128 generated tokens. It saw 12,477
`aten::copy_` calls and 76,808 CUDA kernel launches. These counts include more
than weights, but the vendored `Linear.forward` casts its FP32 weight to the
activation dtype on every call. The [CPU](evidence/axbench-adv-latent-decoder-bf16-4090-profile-cpu.txt)
and [CUDA](evidence/axbench-adv-latent-decoder-bf16-4090-profile-cuda.txt)
key-average tables are the raw profile. Profiler overhead makes their total
wall time unsuitable for a speedup claim.

Casting all 147 target `Linear` modules once reduced the same slice from
81.997 to 79.509 seconds, but changed generated text and published point
artifacts. That broader change was rejected. Casting only the 54 decoder-block
`Linear` modules converted 339,743,232 bytes of FP32 weight storage to
169,871,616 bytes of BF16 weight storage; a complete probe published the
same 48 of 49 files byte for byte. The only differing file was the timestamped
final-worker log. The decoder-only cast leaves the encoder, cross-attention,
LM head, feature-prediction head, and checkpoint on their previous path.

## Matched cell replay

A single loaded runtime replayed the same ten-input masked DiffMean counter
cell in baseline/candidate/candidate/baseline order. Before each pass it
selected the original FP32 decoder weights or their BF16 copies, then used
the same model, fitted direction, seeds, inputs, zero-delay judge, and RTX 4090.
All four `RawEvaluation` tuples and generated-text hashes matched exactly.
The [raw paired report](evidence/axbench-adv-latent-decoder-bf16-4090-paired.json)
contains each timed pass.

| Pass | Cell wall time |
|---|---:|
| First baseline, includes cold generation | 13.961 s |
| Candidate 1 | 6.374 s |
| Candidate 2 | 6.303 s |
| Warm baseline | 6.567 s |

The warm baseline divided by the candidate median is **1.036×**, a 0.228-second
gain for ten responses. This measures one cell after runtime startup, with
zero judge latency; it is not a whole-campaign or live-provider speedup.

An additional [parity matrix](evidence/axbench-adv-latent-decoder-bf16-4090-matrix.json)
replayed ten outputs for **SelfConditioning, SelfConditioningMasked, and
NLSteering**, each with DiffMean and LinearProbe counter directions, on the same
loaded RTX 4090 runtime. Every baseline/candidate pair had the same generated
text hash and exact `RawEvaluation` tuples. Excluding the first cold baseline
cell, the other five baseline cells totaled 31.541 seconds versus 30.320
seconds with BF16 decoder weights, or **1.040×**. This matrix uses a zero-delay
local judge and does not include final publication.

## Production pipeline replay

The opt-in `project_bf16_decoder_linears` setting defaults to `false` and is
admitted only for counter-only CUDA BF16 project runs. The pipeline manifest
records it when enabled. The source checkpoint and scientific inputs remain
unchanged. The checked-in
[`benchmark_axbench_adv_judge_lifecycle.py`](../../scripts/benchmark/benchmark_axbench_adv_judge_lifecycle.py)
ran the complete slice through extraction, model generation, the real framed
judge worker against a deterministic local HTTP endpoint, final judging,
aggregation, and manifest publication. The candidate used the implemented
setting rather than a monkeypatch. Each pass made 90 local HTTP calls.

| Run | Source validation through publication |
|---|---:|
| PR #1771 baseline | 81.997 s |
| BF16 decoder option | 79.817 s |

This is **2.180 seconds saved, 1.027× end-to-end** on the bounded slice.
All 46 non-log, non-manifest files were byte-identical. The two timestamped
logs differed; the sole manifest field difference was
`project_bf16_decoder_linears: true`. The [comparison report](evidence/axbench-adv-latent-decoder-bf16-4090-pipeline.json)
and the [baseline](evidence/axbench-adv-latent-decoder-bf16-4090-baseline-lifecycle.json)
and [candidate](evidence/axbench-adv-latent-decoder-bf16-4090-candidate-lifecycle.json)
lifecycle reports retain the raw timings and file counts. The worker peak
concurrency varied between runs, so the matched same-runtime cell replay above
is stronger evidence for the generation-path speedup.

This work has no live-provider rate-limit or latency measurement and no
50-concept wall-time claim. The production-code pipeline replay covers one
method and concept at one nonzero magnitude; the default-off option preserves
the existing source and output contract for other workloads.

Targeted checks passed: 67 run-config/counter-pipeline tests and 181
self-conditioning/counter-resume tests. Ruff and targeted Ty passed. The full
test suite was not run.
