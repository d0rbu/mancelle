# AxBench-Adv bounded judge queue depth three

This opt-in follow-up to the [two-worker qualification](axbench-adv-judge-workers.md)
for [#1759](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/issues/1759)
allows `project_judge_inflight: 3` when `project_judge_workers: 2`. One generated
response can wait while both framed judge workers are busy. GPU generation,
counter hooks, and result commit order still have one owner. At pair
concurrency two, the inner judge remains bounded to four simultaneous HTTP
requests. The default remains one in-flight response and one judge worker.

## Paired real-GPU cells

The [paired benchmark](../../scripts/benchmark/benchmark_axbench_adv_judge_pair.py)
used the real 400M step-1013 extraction and target stacks on an uncontended
local RTX 4090, the real framed judge workers and vendored evaluator, and a
deterministic local HTTP endpoint. Each endpoint request waited one second;
the concept and fluency pair would take two seconds if sent sequentially. The
three project methods each ran shared control, DiffMean counter, and LinearProbe
counter conditions at ten inputs and raw strength 256. Every condition ran
depth two/three/three/two with fresh SQLite caches and fixed two judge workers,
paired requests, packed-mask reuse, repeated DiffMean text reuse, and
selected-concept fitting. No paid provider was called.

All 36 passes completed with exact `RawEvaluation` equality within each
condition, including generated text, ratings, judge detail, and paid-miss
accounting. Every pass made 20 local HTTP calls and 20 paid misses. Peak inner
HTTP concurrency was four at both depths.

| Method | Condition | Warm depth two | Depth three median | Speedup |
|---|---|---:|---:|---:|
| SelfConditioning | Control | 9.387 s | 8.210 s | 1.143× |
| SelfConditioning | DiffMean | 9.667 s | 7.977 s | 1.212× |
| SelfConditioning | LinearProbe | 9.683 s | 8.507 s | 1.138× |
| SelfConditioningMasked | Control | 9.813 s | 8.544 s | 1.148× |
| SelfConditioningMasked | DiffMean | 9.631 s | 8.191 s | 1.176× |
| SelfConditioningMasked | LinearProbe | 9.639 s | 8.365 s | 1.152× |
| NLSteering | Control | 9.862 s | 8.313 s | 1.186× |
| NLSteering | DiffMean | 9.780 s | 8.335 s | 1.173× |
| NLSteering | LinearProbe | 9.719 s | 8.399 s | 1.157× |

The nine warm depth-two times totaled **87.180 s**, versus **74.842 s** for
the depth-three medians: **12.338 s saved, 1.165× faster** for these cells.
One-time source validation (15.881 s) and runtime startup (42.863 s) are
excluded from this cell comparison. Maximum CUDA allocation was the same at
both depths (9,811,083,776 bytes). The main process high-water RSS was
68,012,344 KiB; this is process RSS, not unique host memory across workers.
The [raw cell report](evidence/axbench-adv-inflight3-4090-cells.json) records
every pass and comparison.

## Published end-to-end slice

The [publication benchmark](../../scripts/benchmark/benchmark_axbench_adv_judge_workers.py)
ran depth three first and depth two second, with fresh outputs and caches for
both. It used all three project methods, both counter families, strengths 256
and 1024, and all ten inputs for concept 36. The wall time includes source
validation, runtime startup, generation, inner and final judging, aggregation,
and manifest publication.

| Depth | Complete wall time | Local HTTP calls | Published non-log files excluding manifest |
|---|---:|---:|---:|
| Three | 255.231 s | 450 | 78 |
| Two | 273.625 s | 450 | 78 |

Depth three saved **18.394 s, a 1.072× end-to-end speedup** in this bounded
slice. All 78 non-log files outside `manifest.json` were byte-identical.
Complete call accounting and the manifest's file inventory matched. The only
manifest field difference was `project_judge_inflight` (`3` versus `2`), and
both downstream counter slices loaded. The run-wide observed HTTP peaks were
22 and 17; these include the existing parallel final evaluator and do not
represent the inner judge's four-request bound. The [raw publication report](evidence/axbench-adv-inflight3-4090-fullslice.json)
records the timings and exact checks.

## Rollout and limits

Keep the default depth one for existing campaigns. A fresh, source-qualified
project counter run may use depth three only with two judge workers. As with
the two-worker option, check live provider capacity before rollout. These
measurements use a local deterministic endpoint on one GPU; they do not
estimate full-campaign wall time or live-provider rate-limit behavior. The
completed 400M campaign used frozen source and has not been migrated to this
option.

Targeted tests in `test_run_config.py`, `test_self_conditioning.py`,
`test_counter_pipeline.py`, and `test_counter_resume.py` passed (239 cases).
Ruff, targeted Ty, and `git diff --check` passed. The full test suite was not
run. The SHA-256 digests of the cell report, publication report, paired
benchmark script, and publication benchmark script are respectively
`e0ddc4c9a1dbcea40969649f51c5f1d25f7ceb79a0da3316f7016dc74d137916`,
`7bd09b0116775c8fa3dff886a80638a1a4f881a676a8f9aa237a6d9a8d9271ca`,
`2c3c3a8a0ee3da0b25c0bdea172a8d3f5662da04bff52aaaedcbdcac075cb84d`,
and `7ed6fdc13fb8c1d3f0cc041392a37e3fc01c5bfc86373a8ca6865cae4fbb587f`.

The script digests above identify the scripts at PR #1767 head
`fedd12e4ed58799f70b70c681dcf41f7dc3d5d34`; a later final-evaluator
comparison extends the publication benchmark.
