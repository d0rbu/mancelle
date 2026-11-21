# AxBench-Adv project judge-worker qualification

This is an opt-in follow-up to the [combined project qualification](axbench-adv-latent-combined-qualification.md) for [#1759](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/issues/1759). It keeps one GPU generation owner and permits two independent framed judge workers per active project concept cell. `project_judge_workers: 2` requires `project_judge_inflight: 2`; the default remains one worker. The two workers use separate SQLite connections to the same judge cache. Identical generated text is judged in sequence so its second occurrence sees the first response's committed cache entries.

The local RTX 4090 replay used the real 400M step-1013 extraction and target stacks, all ten inputs per point, the real framed worker and vendored evaluator, and a deterministic local HTTP judge. The local endpoint delayed each request by one second, making a concept/fluency pair take about two seconds when sequential. Both modes held pair concurrency at two, packed-mask reuse, repeated DiffMean text reuse, and selected-concept fitting fixed. No paid provider was called.

## Paired steady-state cells

The [paired benchmark](../../scripts/benchmark/benchmark_axbench_adv_judge_pair.py) replayed each of the three project methods against shared control, DiffMean counter, and LinearProbe counter conditions in one loaded runtime. Each condition ran one/two/two/one workers with a fresh SQLite cache per pass. All 36 passes completed with exact `RawEvaluation` equality within each condition, including generated text, ratings, judge detail, and paid-miss accounting. Every pass made 20 local HTTP calls and 20 paid cache misses. Peak request concurrency was two with one worker and four with two workers.

| Method | Condition | Warm one worker | Two workers, median | Speedup |
|---|---|---:|---:|---:|
| SelfConditioning | Control | 11.121 s | 9.254 s | 1.202× |
| SelfConditioning | DiffMean | 11.072 s | 9.590 s | 1.155× |
| SelfConditioning | LinearProbe | 11.073 s | 9.528 s | 1.162× |
| SelfConditioningMasked | Control | 11.134 s | 9.586 s | 1.162× |
| SelfConditioningMasked | DiffMean | 11.048 s | 9.519 s | 1.161× |
| SelfConditioningMasked | LinearProbe | 11.061 s | 9.487 s | 1.166× |
| NLSteering | Control | 11.108 s | 9.383 s | 1.184× |
| NLSteering | DiffMean | 11.062 s | 9.320 s | 1.187× |
| NLSteering | LinearProbe | 11.088 s | 9.626 s | 1.152× |

The sum of warm one-worker times was **99.766 s**, versus **85.293 s** for the two-worker medians: **14.473 s saved, 1.170× faster** across nine ten-response cells. This excludes one-time source validation (14.892 s) and runtime startup (40.085 s). Maximum CUDA allocation was the same in both modes (9,811,083,776 bytes). The main process's high-water RSS was 68,007,468 KiB; an active two-worker pass had two additional framed judge processes of about 832,000 KiB RSS each. These are separate process RSS samples and must not be added as unique physical memory.

The [raw cell report](evidence/axbench-adv-judge-workers-4090-cells.json) records all pass times, generation hashes, provider concurrency, calls, and memory readings.

## Published end-to-end slices

The [publication benchmark](../../scripts/benchmark/benchmark_axbench_adv_judge_workers.py) ran two workers first and one worker second, each with a fresh output and cache. Both runs used concept 36, both counter families, and raw strengths 256 and 1024. With all three project methods, **two workers took 262.824 s versus 271.022 s** for one worker: **8.198 s saved, 1.031× end-to-end speedup** from source validation through final judging, aggregation, and manifest publication. Both runs made 450 local HTTP calls. All 78 non-log output files other than the manifest were byte-identical; the manifest file inventory and complete call accounting also matched, and both downstream counter slices loaded. Direct comparison found `project_judge_workers: 2` to be the only manifest difference.

A smaller one-method slice at strength 256 also published equal files and call accounting with 90 calls, but its candidate-first total was **120.034 s versus 114.677 s**. This pair does not show an end-to-end win: a one-concept run spends much of its time in one-time startup and final judging. The paired cell benchmark isolates the repeated work, and the larger published slice shows the saved point time can exceed that overhead. Neither published comparison estimates full-campaign wall time or live-provider rate-limit effects.

The [three-method report](evidence/axbench-adv-judge-workers-4090-all3-fullslice.json) and [one-method report](evidence/axbench-adv-judge-workers-4090-one-method-fullslice.json) contain exact timing, call, file, and accounting checks. The SHA-256 digests of the cell report, three-method report, one-method report, paired benchmark script, and publication benchmark script are respectively `5031884e8c3aecb1ee29a2899b4569e4ead1d1d527a18ab3b915efd362a3ac36`, `f2c9e356292e3ab818514533bff143011ba98767bd4d5a500a726842dce3531c`, `d603e64e4a23c642ff3e6f60e47ad57f6dba63be58625cfa136c93b440d6df49`, `86b0cdd3ae08177b6e8c4f65a166fa0d27b531a9db866e31cc2d47c8f67ef778`, and `7e520dbc4b384db9e575bafad36192afeaade16ec8919992efbd558aad795b08`.

The benchmark-script digests above identify the scripts at PR #1766 head
`f067c54119b54de7923a8d633c6f0ac9f0e73323`; the depth-three follow-up
extends those scripts while preserving the recorded #1766 evidence.

## Rollout

Keep `project_judge_workers: 1` for existing and resumed campaigns. For a fresh, source-qualified project counter run, set `project_judge_inflight: 2` and then opt into `project_judge_workers: 2` after checking live provider capacity. With `project_judge_pair_concurrency: 2`, the new maximum is four simultaneous inner-judge requests per concept cell. The current evidence is local-provider qualification; it does not establish live-provider quota behavior or justify migrating sealed outputs.
