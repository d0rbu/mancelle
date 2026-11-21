# AxBench-Adv latent counter judge overlap

Issue [#1759](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/issues/1759)
targets the project encoder-decoder `SelfConditioning`, `SelfConditioningMasked`,
and `NLSteering` counter cells. The baseline code at `99c81d18f` generates one
response and then waits for its concept and fluency judgments before starting
the next response. The two judge requests are sequential in the single framed
worker. The target model stays resident during those waits.

## Observed baseline

The completed local RTX 4090 concepts 0–1 shard used source `99c81d18f`, the
step-1013 400M SFT checkpoint, factor 6, and both seven-strength counter
families. Its resolved config SHA-256 is
`99140a3aa8ea0771e6db2a48e04a1a1aaa8c38c4a06167da94c65af725727673`;
its complete manifest SHA-256 is
`6166e958f654d0480972565d043a492741fbdb02c95663f01b355dab1d8f76bd`.
All 261 files in that manifest were rehashed successfully. The shard is at
`out/latent-adv400m-20260923/jobs/encdec400-c000-002/output/` in the main
checkout.

| Observed interval or count | Value |
|---|---:|
| Launch receipt to first completed control point | 312 s; includes startup and first generation/judging |
| First to last nonzero point | 5,827 s for 84 committed points; includes later control cells |
| Last nonzero point to published manifest | 136 s; includes final judging, aggregation, plots, and serialization |
| Unique inner-judged responses | 900: 840 counter plus 60 shared controls |
| Inner concept/fluency calls | 1,800: 1,798 paid misses, 2 cache hits |
| Final judge calls | 2,700: 900 paid misses, 1,800 cache hits |

The local process has one framed judge client; its concept and fluency calls
are sequential. The final stage is already much shorter than the inner point
loop in this shard. Point timestamps do not isolate GPU generation time from
the inner judge wait, and the first control timestamp does not isolate cold
startup. These are bounded artifact observations, not a before/after speedup.

The completed shard's `run.log` records 15 sequential final evaluation
invocations (one shared control and seven strengths for each counter family).
They span 129.007 s, from the first `Evaluating generations` line at
19:37:53.379 to the last `Evaluation completed` line at 19:40:02.386.
Across those invocations, 67.505 s elapsed from evaluation start to the
`Total calls` summary, 7.758 s from `Generating final plot` to evaluation
completion, and 53.744 s fell between invocations. The published-manifest
message followed at 19:40:04. These log intervals include worker startup,
dispatch, aggregation, and file operations; the 67.505 s is not an isolated
API-latency measurement. Per-invocation timestamps and durations are in
[`evidence/axbench-adv-latent-final-stage-baseline.csv`](evidence/axbench-adv-latent-final-stage-baseline.csv).

An independent 27-second `nvidia-smi` sample during NLSteering counter work
contains 54 half-second samples: mean GPU utilization 7.48%, 44 samples at
0%, maximum 47%, with a constant 11,917 MiB resident. The raw sample is
[`evidence/axbench-adv-latent-counter-4090-baseline.csv`](evidence/axbench-adv-latent-counter-4090-baseline.csv).
Its short window cannot establish campaign-wide average utilization or a
precise fraction of wall time spent in the judge.

## Paired qualification

`scripts.benchmark.benchmark_axbench_adv_judge_overlap` loads the validated
400M source and direction fit once on a real CUDA GPU. It replays one control
and both counter families for each of the three project methods in `1,2,2,1`
serial/overlapped order. A deterministic two-call judge uses the same delay and
fresh cache state in every run; it makes no provider calls. The harness checks
exact generated UTF-8 text and `RawEvaluation` equality, and records source
validation, direction preflight, runtime startup, generation, judge, complete
cell wall time, call/cache counts, peak CUDA allocation, peak host RSS, and
per-response stage timestamps. A separate `nvidia-smi` process samples GPU
utilization and residency every 500 ms.

The provider-free local RTX 4090 run used the same resolved config hash above,
factor 6, raw counter magnitude 256, and all ten inputs per point. Its source
validation took 14.969 s, direction-fit preflight 0.011 s, and shared runtime
startup 181.818 s. Each response incurred two sequential fake-judge calls
with a combined 2.0 s delay. No live provider was contacted. The harness
completed 36 cell replays (two serial and two overlapped per condition) and
asserted exact `RawEvaluation` parity; all four generation hashes matched in
each of the nine method/pressure groups. Every cell made 20 judge calls with
zero cache hits and 20 synthetic paid misses.

| Method | Pressure | Warm serial, s | Overlap median, s | Speedup |
|---|---|---:|---:|---:|
| SelfConditioning | control | 27.749 | 20.764 | 1.336× |
| SelfConditioning | DiffMean | 27.285 | 20.771 | 1.314× |
| SelfConditioning | LinearProbe | 27.677 | 20.778 | 1.332× |
| SelfConditioningMasked | control | 27.802 | 20.776 | 1.338× |
| SelfConditioningMasked | DiffMean | 27.477 | 20.797 | 1.321× |
| SelfConditioningMasked | LinearProbe | 27.701 | 20.777 | 1.333× |
| NLSteering | control | 27.778 | 20.767 | 1.338× |
| NLSteering | DiffMean | 27.748 | 20.772 | 1.336× |
| NLSteering | LinearProbe | 27.718 | 20.766 | 1.335× |

The nine warm serial cells total 248.937 s; the corresponding overlap medians
total 186.967 s. That is 61.970 s saved, or **1.331× cell throughput**, on
this bounded workload. The first serial control took 56.926 s, including a
36.919 s generation interval, whereas its final warm serial replay took
27.749 s. The [raw report](evidence/axbench-adv-latent-overlap-4090-full10-raw-report.json)
still contains the original 1.409× headline, which averaged this cold control
with the warm replay. The corrected harness uses the final warm serial replay
for its summary; the raw timed rows were not changed.

Across timed cells, [500 ms GPU samples](evidence/axbench-adv-latent-overlap-4090-full10-gpu.csv)
showed mean utilization 11.89% for serial and 16.67% for overlap, with zero
utilization on 70.4% and 61.8% of samples respectively. The event trace
records 123.605 s of simultaneous generation and judging in overlap mode and
none in serial mode. Peak allocated CUDA memory was 9.153 GiB for serial and
9.137 GiB for overlap; `nvidia-smi` reported 11,909–11,920 MiB resident during
timed cells. Peak process host RSS was 64.949 GiB. These utilization windows
exclude source validation and runtime startup; the two modes ran in paired
order on the same GPU, checkpoint, inputs, and fake-judge cache conditions.

The raw report SHA-256 is
`2793af11a8cedd7bdc71573a48d8b7dfec927a344dd6b3ac0ac776e763cef1c6`;
the GPU trace SHA-256 is
`1ab5eee2c918d159edc036c1972672e15da9a1a55bc933013a1c077612836b88`.

An additional overlap-first, serial-second comparison used separate fresh
processes on the same GPU and source. From source validation through all nine
cells, [overlap](evidence/axbench-adv-latent-overlap-4090-direct-overlap-report.json)
took 416.013 s and [serial](evidence/axbench-adv-latent-overlap-4090-direct-serial-report.json)
took 545.478 s; timed cells accounted for 213.945 s and 279.760 s. Each run
made 180 judge calls with zero hits, and total GPU generation time differed by
less than 0.8 s. This comparison is diagnostic: all nine generation hashes
differed across the two processes, and serial runtime startup took 249.190 s
versus 185.489 s for overlap. Consequently its 1.311× whole-run wall ratio is
not the exact-output performance claim; the paired warm-cell comparison above
is. These bounded runs exclude final judging, aggregation, and publication.
The two direct report SHA-256 values are
`07ecd20b697a458bc9e717fe0104b876148ed55f6e87e995cc991a6c34122192`
and `35a9d630f37c2e2422ee2c37492352841996cb156d8b8ede6f8426aefcf5c972`.

The initial Engaging fallback job `23600479` was canceled while still pending
after local qualification succeeded.
