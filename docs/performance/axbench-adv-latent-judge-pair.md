# AxBench-Adv project judge pair concurrency

This continuation is stacked on the generation/judging overlap change in
[#1760](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1760).
The [paired RTX 4090 replay](axbench-adv-latent-judge-overlap.md) used all ten
inputs for each of the three project encoder-decoder methods under the shared
control and both latent counter families. With a synthetic two-second judge
per response, the warmed overlapped cells took about 20.8 seconds each. Their
GPU generation totaled about 7.7 seconds per cell; the judge still sent the
independent concept and fluency requests sequentially.

`project_judge_pair_concurrency=2` lets the existing single judge subprocess
send those two requests together, then waits for both before returning the
same ordered ratings and rationale. It keeps the same model, prompts, parser,
SQLite cache keys, retry policy, and one-response worker. The setting defaults
to `1`, is restricted to project methods, and is recorded in the manifest when
enabled. At most two provider requests can be active. The generation thread
and counter hooks remain unchanged.

## Provider-free validation

A direct fake-client probe of AxBench's `LanguageModel` took 0.455 seconds for
two serial 200-ms requests and 0.246 seconds for the concurrent pair (1.844×
for the request unit). The ordered completions and cache accounting matched.
This CPU probe is a feasibility check, not the GPU performance result.

The actual framed Python 3.12 worker was also exercised against a local fake
OpenAI-compatible HTTP endpoint. Both pair modes returned identical ratings,
text, rationale, and paid-miss counts; the server observed one versus two
maximum concurrent requests. Focused checks passed: 236 project unit/functional
tests and 75 vendored-worker tests. The worker tests include reversed response
completion order, cache replay, a transient failure after the other request
was cached, and an invalid rating that fails without publishing a result.

## Real-GPU replay

The provider-free Engaging replay used source commit
`de615ab097101dfc8e5cb37b1c9ecf17247f6b4a`, an NVIDIA L40S, and the
completed 400M step-1013 AxBench-Adv source configuration. Job `23623309`
completed successfully. It called the real framed judge worker against a local
fake HTTP endpoint, ran ten inputs under all nine method/pressure conditions,
and compared pair modes in `1,2,2,1` order with fresh SQLite caches. All 36
replays passed exact `RawEvaluation` parity; generation hashes matched within
each four-replay group. Every replay made 20 judge calls and 20 synthetic paid
misses. The local endpoint observed peak concurrency of one in serial mode and
two in pair mode.

| Method | Pressure | Warm serial, s | Pair median, s | Speedup |
|---|---|---:|---:|---:|
| SelfConditioning | control | 21.299 | 12.885 | 1.653× |
| SelfConditioning | DiffMean | 21.301 | 12.916 | 1.649× |
| SelfConditioning | LinearProbe | 21.316 | 12.711 | 1.677× |
| SelfConditioningMasked | control | 21.325 | 12.904 | 1.653× |
| SelfConditioningMasked | DiffMean | 21.290 | 12.904 | 1.650× |
| SelfConditioningMasked | LinearProbe | 21.288 | 12.716 | 1.674× |
| NLSteering | control | 21.339 | 12.950 | 1.648× |
| NLSteering | DiffMean | 21.327 | 12.856 | 1.659× |
| NLSteering | LinearProbe | 21.293 | 12.859 | 1.656× |

The nine warm serial cells totaled 191.778 s; the corresponding pair medians
totaled 115.701 s. This is **1.658× warmed cell throughput** and 76.077 s
saved across these nine conditions with the deliberately delayed local judge.
The first cold serial control took 42.805 s, so the comparison uses each
group's final warm serial replay. Source validation took 25.904 s and shared
runtime startup 272.693 s on this allocation; those phases are excluded from
the cell ratio. The job elapsed time was 26:19, including import prewarming,
startup, and all four replays per group.

Peak allocated CUDA memory was 9.137 GiB and peak process host RSS was 64.091
GiB. The [500 ms GPU sample](evidence/axbench-adv-latent-judge-pair-l40s-full10-gpu.csv)
has 2,779 observations, mean utilization 18.72%, and a maximum resident
footprint of 11,573 MiB. The sample spans startup and timed cells, so its mean
is not a serial-versus-pair utilization comparison. The
[raw report](evidence/axbench-adv-latent-judge-pair-l40s-full10-raw-report.json)
and GPU sample have SHA-256 values
`981b2d6480466185b299ae5b2a0bfb1393a1bf0b5d623345f6725a72e11ec8a9`
and `f9d3b9c711af8d73cb1d83a798904d7b3f3a74130c511806eeabac19765726bc`.

The first isolated attempt, job `23622401`, stopped after its first replay
because the benchmark reused an exclusive worker stderr filename. The
corrected harness uses a fresh filename per replay. Its completed rerun is the
source of every result above.

Live-provider latency and rate-limit behavior are not measured here. Higher
request concurrency needs provider-capacity qualification before rollout to a
paid campaign.
