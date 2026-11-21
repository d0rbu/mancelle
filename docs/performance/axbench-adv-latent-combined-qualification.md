# AxBench-Adv project counter stack qualification

This qualification sits above [#1764](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1764). It measures the three project encoder-decoder methods against the 400M step-1013 source for [#1759](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/issues/1759). Both counter families and the shared zero control are included. The proposed new startup option fits dense DiffMean directions only for a fresh execution slice's owned concepts; source validation, all prompt preparations, and the selected directions' original fitting order remain intact. The default remains the full fit.

## Combined cell replay

The [real-GPU replay](../../scripts/benchmark/benchmark_axbench_adv_judge_pair.py) compared serial defaults with generation/judge overlap, concept/fluency concurrency, and packed-mask reuse enabled together. It used the real 400M checkpoint and target model, the real framed judge worker, and a deterministic local HTTP judge with two seconds of sequential latency per response. Each of the nine method/control/counter conditions generated and judged ten responses in off/on/on/off order with fresh SQLite caches. The script required exact `RawEvaluation` equality across all four passes, which includes generated UTF-8 text, ratings, judge detail, and paid-miss accounting. All 36 passes completed; each condition had one generation hash across its four passes, 20 judge calls, and 20 paid cache misses. The enabled passes reached two concurrent provider requests; serial passes reached one.

The [raw report](evidence/axbench-adv-combined-stack-4090-full10.json) records the individual cell timings, generated-text hashes, call counts, peak CUDA allocation, and process peak RSS. On the local RTX 4090, the sum of nine warm serial cell times divided by the sum of the enabled cell times was **2.529×**. Warm serial cells were roughly 27.5–28.1 seconds and enabled cells roughly 11.0 seconds each. The first control cell included a 37.2-second cold pass and was excluded from the warm comparison. Source validation took 15.0 seconds and the one-time runtime startup took 118.5 seconds; neither is included in the cell speedup. The maximum CUDA allocation reported across cells was 9.811 GB, and the process high-water RSS was 68,034,688 KiB. These are process measurements, not a GPU-utilization time series.

This is a bounded, local-judge workload. It does not measure live-provider latency or rate limits, whole-campaign wall time, final instruction judging, or publication. The sampler could not read NVML during the nine-condition replay; a separately sampled representative cell follows. The completed 400M campaign was produced by its frozen source; these new flags have not been deployed into that campaign.

## Sampled GPU duty cycle

A second [real-GPU replay](evidence/axbench-adv-combined-stack-4090-sampled-cell.json) held the same source, judge delay, ten inputs, and fresh-cache off/on/on/off order, and selected the `SelfConditioningMasked` DiffMean counter cell. A host-permitted `nvidia-smi` process recorded [477 half-second samples](evidence/axbench-adv-combined-stack-4090-gpu-samples.csv) across startup and all four passes. The report's Unix timestamps delimit each timed pass; the comparisons below use only samples within those windows.

| Pass | Cell time | GPU samples | Mean utilization | Samples at 0% | Resident VRAM |
|---|---:|---:|---:|---:|---:|
| Cold serial | 37.354 s | 73 | 10.16% | 78.1% | 12,021–12,023 MiB |
| Enabled 1 | 11.067 s | 22 | 36.36% | 18.2% | 12,023 MiB |
| Enabled 2 | 11.055 s | 22 | 32.82% | 27.3% | 12,023 MiB |
| Warm serial | 27.867 s | 54 | 10.61% | 68.5% | 12,023 MiB |

The enabled passes' sample-weighted mean was 34.59%, versus 10.61% for the warm serial pass. Their zero-utilization share was 22.7%, versus 68.5%. The warm cell speedup was 2.519×. All four passes returned exactly equal `RawEvaluation` tuples, one generation hash, 20 judge calls, and 20 paid misses. These short windows measure this local fake-judge workload on one GPU; they are not a campaign-wide utilization estimate or a live-provider throughput claim.

## Selected-concept startup fit

The project runtime receives a validated 50-concept training set even for a two-concept execution slice. With `project_fit_selected_concepts_only=true`, it validates that same full source and prepares the same prompt contexts, then passes only the owned concept IDs to the dense DiffMean fit. The [paired CUDA benchmark](../../scripts/benchmark/benchmark_axbench_adv_selected_fit.py) compares a full 50-concept fit with a two-concept fit on one loaded extraction stack, with repeated-text reuse enabled in both. It checks selected direction tensors with `torch.equal` after each pass.

The [raw paired report](evidence/axbench-adv-selected-fit-4090-paired.json) used the actual local shard IDs 36 and 37. Full/selected/selected/full fits took 86.109, 4.687, 4.653, and 82.018 seconds. Both selected directions were bitwise equal to the first full fit on all four passes, with maximum absolute difference 0.0. The final warm full fit versus the median selected fit is 82.018 versus 4.670 seconds: **77.348 seconds saved in this fit, 17.56× faster**. Peak allocated CUDA memory was 9.444 GB on the last full fit versus 9.028 GB on the selected fits. This isolates the fit; it does not measure a complete runtime startup or campaign.

## Whole-runtime startup and output parity

The [paired whole-runtime benchmark](../../scripts/benchmark/benchmark_axbench_adv_selected_runtime.py) created four fresh extraction/target runtimes in full/selected/selected/full order. It restored the validated source's prompt-sampling generator to the same starting state before each construction. This matters because the setup object shares a mutable generator: an initial uncontrolled same-process comparison advanced that state and changed generated text even though the fitted directions matched. The controlled comparison held the source, model, selected concepts 36–37, ten inputs of concept 36, masked DiffMean counter, raw magnitude 256, judge concurrency, and fresh-cache local judge fixed. It required bitwise direction equality and exact `RawEvaluation` equality after every pass.

| Runtime | Startup | Judged cell | Exact directions and judged outputs |
|---|---:|---:|---|
| First full fit | 117.451 s | 13.880 s | Yes |
| Selected fit 1 | 40.542 s | 6.486 s | Yes |
| Selected fit 2 | 41.116 s | 6.745 s | Yes |
| Final full fit | 118.562 s | 6.591 s | Yes |

The [raw runtime report](evidence/axbench-adv-selected-runtime-4090-paired.json) records the identical prompt-RNG-state and generation hashes, 20 judge calls and misses per cell, and memory high-water marks. The final full startup versus the median selected startup was 118.562 versus 40.829 seconds: **77.733 seconds saved, a 2.904× whole-runtime startup speedup** on this source and shard. Source validation took a separate 14.728 seconds in the benchmark and is excluded from that ratio. Warm judged-cell time was effectively unchanged (6.591 seconds full versus a 6.616-second selected median); the first full cell's extra time was a cold-pass effect. This is startup and one counter-cell evidence, not a whole-campaign wall-time claim.

The nine-condition report, selected-fit report, sampled-cell report, GPU sample, whole-runtime report, combined benchmark, selected-fit benchmark, and whole-runtime benchmark have SHA-256 digests `f18562af56b520014d6793db046406811972bff8ccc54bb62b02b3e7a2e6ac27`, `fa552113c089839810b19650d93b4ca2b5342861cdf91b7fc4b6de2abebc25c7`, `21f6f64052f582a1bf3ac009cd22d19068d4cd2560ca868ccff765af3e87c3ff`, `c51bd6b40799df68b840643508765b04ae0cfab5b790268329104c8fa03c87c0`, `de8311ada61ff9724c5505a4f7d03471c725593ff737a6aa7764867ec303f7ea`, `0a377e1f93cd64d20bbda99838c06b4f0e23b6bc05c3eb34e3874b83610332f6`, `9aa4fa9b84770486cbcf8f8592383f83f0e8895fa1eb4a4bec65367239d8d312`, and `99a03019e80f76f50af3f9226c7e45c7fa573c66e2d7af614e5ad68c1c0cb9db`, respectively.

The selected-fit option is recorded in the published manifest and is opt-in for a fresh, qualified slice. Existing completed points, sealed resume snapshots, and the live campaign are not migrated.

The benchmark-script digest above refers to the script at PR #1765 head `0b977dee68b1556d7d51f0fdaf3bb6324049544f`. A later opt-in two-worker comparison extends the script; its current digest and results are in the [judge-worker qualification](axbench-adv-judge-workers.md).

## Bounded published pipeline replay

The [full-slice benchmark](../../scripts/benchmark/benchmark_axbench_adv_fullslice.py)
ran the actual project pipeline on the local RTX 4090 with the 400M step-1013
source. It selected concept 36, `SelfConditioningMasked`, both counter families,
one raw nonzero magnitude (256), and all ten inputs per point. The framed
inner judge and real vendored final evaluator called a deterministic local
HTTP endpoint with two seconds of sequential delay per response pair. Each
run used fresh output and SQLite cache paths. The enabled run preceded the
serial run; all time from source validation through final judging, aggregation,
and manifest publication is included.

| Mode | Complete wall time | Local HTTP calls | Published files |
|---|---:|---:|---:|
| Combined options enabled | 120.685 s | 90 | 47 |
| Serial defaults | 316.496 s | 90 | 47 |

This bounded end-to-end workload was **2.623× faster** with the combined
options. Both manifests had the same call accounting and 45-file inventory.
All 46 other output files were byte-identical, including the generated point
receipts, final judged JSONL, aggregates, and plots. Only `manifest.json`
differed, in the five opt-in setting fields. The downstream slice loader
accepted both counter families in both runs. The [paired report](evidence/axbench-adv-fullslice-4090-paired.json)
records the timings, scope, source config digest, accounting, option differences,
and an aggregate digest of the identical files.

The local endpoint's observed peak of 7 versus 10 concurrent requests includes
the pre-existing parallel final evaluator; it is not the project inner-judge
concurrency limit. This single-concept, one-method replay proves publication
and exact output parity under the new flags, not live-provider capacity or a
50-concept campaign speedup. It also does not cover a sealed resume of this
new output.

The same checked-in script also published one enabled slice with **all three
project methods**, both counter families, concept 36, and raw magnitude 256.
This complete run took **198.033 s**, made 270 local HTTP calls, and published
61 files, including the final judged outputs and verified manifest. Both
downstream counter slices loaded successfully. All four
`SelfConditioningMasked` point receipts shared with the one-method enabled run
were byte-identical. The [three-method report](evidence/axbench-adv-fullslice-all3-4090.json)
records its scope, accounting, and manifest digest. There is no matched serial
three-method full-pipeline run, so the 2.623× end-to-end comparison above
applies only to the one-method slice. The local endpoint's peak 20 concurrent
requests again includes final judging and is not the inner-judge bound.
