# AxBench-Adv repeated DiffMean text reuse

This optimization stacks on [#1763](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1763) and applies to project encoder-decoder `SelfConditioning`, `SelfConditioningMasked`, and `NLSteering` startup. The 400M step-1013 source manifest sets DiffMean SAE extraction batch size to one.

## Bottleneck

A synchronized local RTX 4090 [startup profile](evidence/axbench-adv-diffmean-startup-profile.json) of the #1763 source measured 182.011 s for `SelfConditioningMethodRuntime.__init__`, including 147.710 s fitting all 50 dense DiffMean directions. Extraction model loading took 25.589 s, and 373 prompt preparations took 7.722 s. Source contract validation was a separate 14.502 s. These are one cold-run measurements, not steady-state campaign timing.

The exact source training set has 50 concepts, each with 72 positive and 72 negative texts: 7,200 rows but only 3,816 distinct strings. Only 216 strings repeat, accounting for 3,384 redundant batch-one SAE forwards. The benchmark logs counts and timings, not training text content.

A [prototype screen](evidence/axbench-adv-diffmean-reuse-4090-screen.json) confirmed 7,200 versus 3,816 actual extraction rows, 3,384 cache hits, and bitwise direction parity in off/on/on/off order. It used a local benchmark wrapper before the option was wired into the runtime; the integrated run below is the qualification result.

## Change

`project_reuse_diffmean_texts=true` stores the float32 max-pooled activation for a text only when that exact string occurs more than once in the same runtime fit. Later positive or negative occurrences use that row in their original order before the unchanged percentile reduction. The cache lasts only for the 50-concept fit and requires source `extraction_batch_size=1`; the default is `false`. The option is recorded in the published manifest when enabled. It changes no source texts, percentile tables, direction normalization, generation, counter hooks, or judging.

## Real-GPU qualification

The checked-in [paired benchmark](../../scripts/benchmark/benchmark_axbench_adv_diffmean_reuse.py) uses the validated 400M step-1013 source and runs batch-one fits with reuse off/on/on/off against one loaded Gemma/SAE stack on the local RTX 4090. It compares every fitted direction tensor against the first baseline with `torch.equal`, checks the maximum absolute difference, and captures peak allocated CUDA memory. No judge or paid provider is used.

The [raw paired report](evidence/axbench-adv-diffmean-reuse-4090-paired.json) contains the individual timings and exactness results. Across all 50 concepts, every direction is bitwise equal in all replays (maximum absolute difference 0.0).

The integrated benchmark ran against product commit
`ce94e984f477b933e262c741878dda3eb224774d` in this stack:

| Pass | Reuse | Fit wall time | Exact directions | Peak CUDA allocated |
|---|---|---:|---|---:|
| 1 | Off | 150.553 s | Yes | 9.285 GiB |
| 2 | On | 82.206 s | Yes | 8.803 GiB |
| 3 | On | 81.654 s | Yes | 8.803 GiB |
| 4 | Off | 147.485 s | Yes | 9.285 GiB |

The final warm baseline versus the median of the two enabled passes is
147.485 versus 81.930 s: **65.555 s saved, a 1.800× fit speedup**. The first
off pass is excluded from that comparison because it includes first-pass
effects. The observed peak CUDA allocation fell by 0.482 GiB. The cache held
216 repeated-text rows, or 54 MiB of float32 activations, in the prototype
screen; the integrated benchmark does not separately measure cache residency.

The startup profile, prototype screen, integrated paired report, and benchmark
script have SHA-256 values, respectively:
`46fdacfbd4428192395e85922d7123cbff3b3c58e44e832d8b7d3111bdf7466c`,
`372826d09c361340c70d22599a659f37ac8c39e4cd8fc694c6de8ef738055589`,
`12090efd763174e71f6da006a1912f0e1e9c3c4ffce7781d4599329eaaba7f55`,
and `057b556141bb4681e9cbf0091e1da0632b6fa999bb8e8977fd308defc272a962`.

The paired fit isolates the identified startup component. It does not measure a complete campaign, provider latency, final judging, or a fresh whole-runtime startup speedup. Enable the setting only for a fresh qualified source after the campaign owner chooses a rollout; the completed encoder-decoder campaign is not migrated.
