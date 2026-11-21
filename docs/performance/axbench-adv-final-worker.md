# AxBench-Adv persistent final evaluator

This opt-in follow-up to the [bounded judge queue](axbench-adv-inflight3.md)
for [#1759](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/issues/1759)
reuses one isolated vendored Python 3.12 process for sequential final judged
points. The default continues to start a fresh evaluator for each point.
`reuse_final_judge_worker: true` changes process lifetime only: every point
still has its own input, output, SQLite cache accounting, validation, and
atomic publication. The worker uses the existing framed transport, closes on
success, and aborts on evaluation or protocol failure.

## Startup evidence

The completed 400M step-1013 one-concept publication slice spends about
43 seconds from its last generated point to its manifest, including five
final evaluator invocations, aggregation, and publication. Its evaluator log
shows several seconds between one evaluation finishing and the next starting.
Each invocation imports the vendored evaluator in a new Python process.

A provider-free, cache-only replay held the five published point inputs, the
original URL-bound SQLite cache, evaluator config, output paths, and order
fixed. Five fresh `uv run` processes took **29.224 s total**. One process
calling the same vendored evaluator five times took **8.520 s total**; its
five evaluation calls took 0.589–0.845 s each after import. All **40 output
files** (eight per point), including call accounting, were byte-identical
between the two modes. The [raw screening report](evidence/axbench-adv-final-worker-cache-only.json)
contains timings and file hashes. This is a CPU/cache-only startup screen,
not an end-to-end speedup claim.

The project-owned framed worker separately replayed those five points and
matched all 40 one-off output files. An intentional cache-only miss produced
an error, terminated the worker, and did not publish a completed manifest.
No paid provider was called.

## Real-GPU publication comparison

The [full-slice replay](../../scripts/benchmark/benchmark_axbench_adv_judge_workers.py)
keeps the 400M step-1013 source, all three project methods, both counter
families, raw strengths 256 and 1024, and ten inputs per point fixed. Both
arms use two inner judge workers, in-flight depth three, paired inner
requests, mask reuse, repeated DiffMean text reuse, selected-concept fitting,
and fresh output and cache paths. The only changed setting is final evaluator
process reuse. The local deterministic HTTP judge has a one-second delay per
request; no paid provider was called.

The candidate-first pass published complete manifests in both modes. Reuse
took **237.345 s** versus **245.046 s** with fresh evaluator processes:
**7.701 s saved, 1.032× end-to-end**. Each made 450 local HTTP calls and had
identical complete call accounting. All 78 non-log files excluding the
manifest were byte-identical, manifest file inventories matched, and both
downstream counter slices loaded. The manifest differed only in the enabled
`reuse_final_judge_worker` field. From point-file timestamps, the last point
to manifest interval fell from 43.199 s to 23.763 s; the candidate spent
about 11.7 s longer before that interval, so this one run understates the
isolated final-stage saving. These timestamp intervals are diagnostic, not
independent phase timers. The [candidate-first report](evidence/axbench-adv-final-worker-4090-candidate-first.json)
contains the exact timing and output checks.

A second full-slice replay reversed the order. The baseline took **253.352 s**
and reuse took **235.556 s**: **17.795 s saved, 1.076× end-to-end**. It again
made 450 local HTTP calls per arm, matched complete call accounting and all
78 non-log published files excluding the manifest, and loaded both downstream
slices. Its last-point-to-manifest interval fell from 41.187 s to 27.073 s.
The [baseline-first report](evidence/axbench-adv-final-worker-4090-baseline-first.json)
records those checks.

Across the two orders, mean wall time was **236.451 s with reuse** versus
**249.199 s with fresh processes**: **12.748 s saved, 1.054× end-to-end**.
Both individual comparisons favored reuse. These are local RTX 4090 replays
with a deterministic HTTP judge, one concept, and all three project methods;
they do not establish live-provider throughput or a larger campaign speedup.
The opt-in setting preserves the default behavior until that rollout is
chosen. Focused tests covering final publication, config parsing, pipeline
manifest behavior, resume, and framed transport passed (**177 tests**); the
full suite was not run.
