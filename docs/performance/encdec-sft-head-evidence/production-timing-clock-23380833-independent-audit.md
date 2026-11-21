# Independent audit: clock diagnostic 23380833

**The earlier timing contradiction was not reproduced within the frozen bounds in this instrumented baseline run.** All 60 records pass independently recomputed clock/event invariants, and collection, source, runtime and artifact closure are complete. This does not establish clock accuracy, explain job 23377981, validate its timing retroactively, or qualify throughput.

Job 23380833 completed `0:0` on node4602 at 07:06:19 EDT on September 21, 2026 (32m10s). The probe process finished in 1752 s under its 1800 s cap; the short 48 s remaining margin is relevant when setting prospective cold-start budgets. No new GPU/network/source operations were performed for this audit.

## Coverage and admission

The retained workload is baseline `MuonAdamW`, packing=false, one H200, BF16/native explicit FLASH, B8/GA8/T2048 (131072 tokens/update), width1664, encoder/decoder27, transformer head10, vocabulary844. Seeds42/20260815 and 20 warmup + 40 measured updates are unchanged. The model has 401 parameter tensors. Every measured profile scalar equals its corresponding raw record in both the critical and rank-zero arrays (160 associations).

Independent stdlib arithmetic, without importing or calling the frozen classifier, checked all 1,560 event records, 3,480 host-anchor triples, 1,500 CUDA phase/gap edges, legacy float endpoints, both host clocks, record ordering and stream identities. Every saved per-sample classifier dictionary was reproduced exactly using Python3.13.2; the remote helper used 3.13.14. All 60 rows use device 0/current stream 0, with matching event devices, ordered usable anchors, no violations and no uncertainties. Original/replacement function-source hashes and the original function AST were also independently reproduced.

The unchanged bounds apply to every row: CUDA interval/telescoping allowance 1 ms; host epsilon=max(2ms,0.001×whole CUDA envelope). No observations were removed.

## Recorded ranges and retained extremes

Times below are milliseconds. The JSON companion retains every per-sample value and stage-specific extrema/index.

| Quantity | Warmup min…max | Measured min / median / max |
|---|---:|---:|
| Host wall minus phase sum | 0.153132…0.301704 | 0.162070 / 0.207258 / 0.280412 |
| Host wall minus whole CUDA envelope | −0.019250…0.180736 | 0.137733 / 0.183944 / 0.258638 |
| Pre-query perf wall minus envelope | −0.101390…0.087384 | 0.050831 / 0.086449 / 0.137841 |
| Pre-query RAW wall minus envelope | 0.108816…1.443858 | 0.109519 / 0.144505 / 0.196681 |
| Original 17-query block, perf clock | 0.077937…0.106324 | 0.077637 / 0.096909 / 0.119063 |
| Largest anchor bracket per row | 0.001796…0.013197 | 0.001637 / 0.002088 / 0.006127 |
| Largest CUDA edge residual per row | 0.000114…0.006592 | 0.000122 / 0.000267 / 0.000443 |

The first warmup wall is 127496.152625 ms. Its −0.019250ms wall/envelope and −0.101390ms synchronized perf/envelope deficits remain visible; neither exceeds that row's 127.496172 ms predeclared host allowance. Its largest paired-host span difference is 1.550631 ms. All 40 measured walls and synchronized pre-query walls exceed their CUDA envelopes even without the allowance; measured host epsilons are 4.512719–5.105340ms.

The last measured row (zero-based index 39/global 59) remains in the data: wall 5105.525527 ms, CUDA envelope 5105.340332 ms, forward 2057.918427 ms versus measured forward median1460.339745ms. Its first forward microbatch is 770.664490 ms; the longer duration appears in both GPU and host observations. It is not a containment violation and no cause is established. The measured wall sample CV is 2.0479%, including that row; this is descriptive variability, not a performance estimate or a reason to filter it.

The relative perf-minus-RAW clock offset changes by−2.332784ms over 182.181083 s of measured anchors (descriptive fitted rate difference−12.873ppm), and−5.032125ms over 396.763341 s across all anchors. Per-measured-step synchronized perf/RAW differences are only−0.061998…−0.054710ms, within the frozen bounds. These two clocks may share hardware; the relative rate observation is not calibration or an explanation of the old discrepancy.

Measured CPU observation spans are also retained: event-record sites total 2.025045–3.096879ms/row, and outer-call minus original wall 2.128195–2.912561ms. The 17-query block includes endpoint/anchor bookkeeping. None of these is a counterfactual instrumentation-overhead estimate, and no subtraction repairs old timings. Supplemental queries and per-step JSON writes also change between-step cadence.

## Integrity and environment

Independent provenance review matched all 59 clock-tree files plus its adjacent batch log to identical before/after remote inventories. All 19 physical `.exit` files are0 (including the finalizer); incoming/final status0, `EXIT_STATUS`, `COMPLETED` and all three final artifact hashes agree. Before/after checks cover 3,535 source files, 6 staged and 6 executed helper files, 3 launcher copies, the sealed manifest, and 10 runtime files. The original eight runtime pins match 23377981 exactly; `cuda/__init__.py` and `cuda/streams.py` are two additional current-run pins with no historical same-byte claim. All 1,005 runtime source/config entries and kernel source text match the prior large-A1 receipt; resolved config differs only in trial ID/output path.

The node/allocation differs: node4602, `mit_preemptable`, account/QOS `mit_general/normal`, versus node3401-c1, `mit_normal_gpu`, `mit_amf_advanced_gpu/mit_amf_advanced_gpu`. Both requested 8 CPUs / 128G / one H200 and report driver 590.48.01, driver CUDA capability 13.1, Torch CUDA 13.0, 700W and MIG disabled. GPU UUID and VBIOS differ (`96.00.DA.00.11` versus `96.00.A5.00.03`); persistence is Enabled versus Disabled, and addressing mode None versus HMM. Current affinity records 16 logical IDs; prior affinity was not recorded. These differences are covariates, not causal conclusions.

The audit found no material admission or raw-record blocker to a **new, separately reviewed fixed-count comparison with prospective coherence checks**. Such a comparison must freeze arms/order/count/warmup, retain all samples and reject coherence failures; this baseline probe cannot predict its outcome or authorize retries until favorable. The original 23377981 timing validity remains unresolved.

## Portable evidence bindings

All filenames are relative to their named result bundle, independent of this audit's location.

- Original bundle: `production-timing-clock-23380833/`; raw `clock-diagnostic.json` SHA `6ac03aa058d72c53a84fd3b59c2a6ab4fc6bf1718ef883a9281bf6bb49ece3c5`.
- Final `exit-receipt.json`: `cf7e0c045140c98f041b4e489c5511afac64776f4a35c15cafbfe0c678052f33`.
- Before/after inventory: `bae8306f12ca88112036b50318c41e59688511893c649e0eee79bae01f43226e`; transfer receipt `verification-jobs-23380833-23376183-terminal-transfer.json`: `c4e13ef69e25746aa8e8e5739c7117b062caf034d661f08e99fdb1db4d9b1baf`.
- Frozen helper `1f78eb13888cde2a481c1022c2c6cdb820dc8865216aafd3546f9aecbf149c62`; protocol `edafa5474ccd1bb01a8755a8e26a4072544ae1747ecf0cef41cb64a126c36b2c`; submitted launcher `ad954166cd27c24965ab61e5a6140de9d7c678c32e34243b8f64bf193e5a7faa`.
- Sealed runtime source manifest `0e89bf87f2fcbea0e77fe8d3eb43d8c9b5071e9cf45666ecf85c272ea7e9c96f`, base revision `de4c6772c11b1a7360897a1f8beb1ee563e40700`.
- JSON companion `production-timing-clock-23380833-independent-audit.json`: `85532d9840f13507efa17b5a2a207a81f8667a38dba344270c45457d0c1ca247`.
