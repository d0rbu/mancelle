# Prospective production packing comparison with per-trial clock checks

Declared before any trial in this comparison. This is one finite comparison of
the unchanged production V2 packing implementation. It does not repair timing
from job 23377981. That campaign's numerical proofs remain valid within their
recorded scopes, but its timing inconsistencies remain unexplained. The clean
baseline diagnostic 23380833 is bounded non-reproduction on another H200, with
different firmware/settings and instrumentation; it is not clock calibration.

## Immutable implementation and admission

- Require source manifest
  `0e89bf87f2fcbea0e77fe8d3eb43d8c9b5071e9cf45666ecf85c272ea7e9c96f`,
  including production packing `d995ec99` and profiler `5a8cf492`.
- Reuse the authenticated successful optimizer sub-gates and strict Triton
  independent model gates from the closed 23377981 evidence chain, as enumerated
  in `production-clock-comparison-reuse.json`. Require their exact receipt bytes,
  source identity, pass fields, runtime identity and final integrity evidence.
  Native independent forward/backward remains unqualified; the original native
  baseline rejection remains a rejection. Numerical tolerances do not change.
- Require the same eight historical Python/Torch runtime hashes, Python 3.13.14,
  Torch 2.13.0+cu130, native FLASH route, BF16, compilation enabled, OMP threads 8,
  shared Inductor cache and recorded non-device controls. Hash the two additional
  CUDA wrapper files before and after, without claiming historical pins for them.
- One H200 allocation; no GPU UUID selected based on observed performance.
  Record GPU UUID, firmware, driver/settings, CPU affinity and Slurm resources.
  All sixteen processes must observe the same GPU and runtime identity.
- Verify all source files, imported helpers, submitted launcher, protocols,
  numerical receipt originals/copies and runtime files before and after, including
  failure paths. Any missing, mismatched or failed admission rejects the campaign.

## Workloads and fixed order

Both workloads use width 1664, encoder/decoder depth 27, context 2048, feature
vocabulary 844, transformer FP head depth 10, 64 padded segments, 24 padded
features, segment size/static maximum 32, no FP8, the frozen model seed 42 and
batch seed 20260815. Small is B2/GA1 (4096 tokens/update); large is B8/GA8
(131072 tokens/update). These are resident synthetic batches and randomly
initialized models with evolving warmup/measured updates.

Each trial is a fresh process with 20 warmup updates and exactly 40 measured
updates. All 60 records remain in the artifact; all 40 measured records contribute
to the run median. A is the upstream optimizer; B is the production packing
opt-in. No alternative candidate, warmup adjustment or outlier exclusion.

The sixteen trials execute in this exact order:

1. large-A1, large-B1, large-B2, large-A2
2. small-A1, small-B1, small-B2, small-A2
3. large-A3, large-B3, large-B4, large-A4
4. small-A3, small-B3, small-B4, small-A4

Pairs are Ai/Bi within each workload for i=1,2,3,4. The two ABBA blocks provide
four run-level comparisons per workload. Step samples are not independent
replicate runs and will not be used to manufacture a confidence interval.

## Timer and prospective validity rules

Reuse the immutable `diagnose_production_timing_clocks.py` instrumented step
(`1f78eb13`) without changing model math, optimizer math, CUDA record order or
the original post-synchronization wall stop. Generalize only the classifier's
event inventory to GA1/GA8: 3*GA+2 records, 2*GA+1 original elapsed queries,
3*GA+1 supplemental offsets and GA supplemental gap queries. Supplemental CUDA
queries occur after the original wall stop. Host perf/raw anchors and device/
stream observations stay as in the clock diagnostic.

Unlike the diagnostic, hold rows in memory and save before collection and at
completion/error/signal, with no per-step JSON writes. This changes between-step
cadence; both arms use the same new wrapper. Forced termination can lose unsaved
rows and is incomplete evidence. Anchor overhead is included in the primary wall
interval. Report the result as instrumented complete-step latency, with no
subtraction or inferred uninstrumented speedup.

Every trial must pass the generalized full raw-record classifier. Keep the
original diagnostic CUDA consistency allowance of 1 ms. Warmup host bounds stay
max(2 ms, 0.001*whole-CUDA-ms), including compilation. For measured records, use a
new, prospective fixed 1 ms host bound for paired host clock agreement, logical
anchor/legacy endpoint order, anchor width and host/device/host containment.
Malformed, incomplete or nonfinite evidence, changed streams/devices within a
step, negative anchor brackets, uncertainty or any violation rejects that trial
and campaign. Different coherent streams between separately synchronized steps
are allowed; the allocated GPU must remain identical across the entire campaign.
These are admission tolerances, not a calibration or an absolute error bound.

Retain all extrema, including coherent forward spikes. Do not repair, reorder,
subtract from or selectively drop samples. Independently recompute classification
and binding to the original profiler's four raw arrays after transfer.

## Prospective descriptive performance decision

Only after all sixteen trials and final integrity pass:

- Compute each trial's full-precision median of all 40 original wall samples.
  For every pair, report A-B milliseconds and 100*(A-B)/A percent together.
- Require wall saving strictly greater than 2 ms in all four pairs of each
  workload. This is a conservative, predeclared descriptive margin chosen as
  twice the measured admission allowance, not a confidence bound or claim that
  true timing error is bounded by 1 ms. Any failure makes this screen inconclusive.
- Report optimizer-phase medians separately and require positive optimizer
  savings in all pairs to corroborate the intended mechanism. They cannot
  substitute for complete-step savings. Also report whole-CUDA and host-clock
  coherence distributions and all trial/run variation.
- Preserve the old campaign, its contradiction, the clean diagnostic and every
  new trial regardless of direction. Four positive pairs remain limited
  descriptive replication, not universal, real-data, distributed or complete
  SFT pipeline throughput. No automatic default enablement or merge.

## Finite resources and stopping rule

One non-preemptible `mit_normal_gpu` allocation with account and QoS
`mit_amf_advanced_gpu`, one H200, 8 CPUs, 128 GiB requested memory, four hours.
Shared work deadline is 210 minutes after batch start, leaving 30 minutes for
final verification. Each trial is capped at 2100 seconds plus 30 seconds kill
grace, reduced if necessary by the shared work deadline. Smaller phases are
separately capped. The caps are maxima, not a promise all trials fit.

No requeue, retry, replacement allocation or fallback is authorized by this
protocol. Stop on an admission/process/clock error; retain all partial evidence
and attempt final verification. Evaluate the direction/margin rule only after
all sixteen trials and final integrity checks; an early negative pair never
shortens the fixed schedule. A failed direction/margin screen ends this comparison
as inconclusive; it does not trigger more samples. Any later investigation or
different candidate needs a new, explicitly documented finite design before
running. This protocol is frozen by SHA before staging/submission.
