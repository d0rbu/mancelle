# AxBench-Adv statistics reconstruction startup

This follow-up to [#1770](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1770)
for [#1759](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/issues/1759)
converts each scalar safetensors column to a Python list once before rebuilding
`PerFeatureStat` rows. The raw statistics SHA-256 check, row order, duplicate-id
behavior, scalar casts, rounding, histogram lists, and returned dataclasses
remain unchanged. The change is in the shared statistics reader; no new run
setting is needed.

## Bottleneck and direct check

On the 400M step-1013 source, source validation took **15.874 s** and
`load_method_setup` took **13.623 s**. A second profile attributed **11.495 s**
to loading the pinned 79 MB statistics snapshot. The
[source](evidence/axbench-adv-stats-reconstruct-source-phases-4090.json) and
[setup](evidence/axbench-adv-stats-reconstruct-setup-phases-4090.json) reports
record those timings. `cProfile` attributed 9.758 s of self time to
[`_reconstruct_statistics`](evidence/axbench-adv-stats-reconstruct-cprofile.txt),
whose old loop indexed and converted individual
tensor scalars for each feature and percentile.

Fresh-process loads of the same pinned 65,316-feature snapshot, with the
candidate run first, produced the **same pickled statistics object SHA-256**:
`bb680b9e14d9e1f530b663fe98e86c329c4c756bf013f284909c220d688e4252`.
The [baseline](evidence/axbench-adv-stats-reconstruct-cpu-baseline.json) took
**10.965 s**; the [candidate](evidence/axbench-adv-stats-reconstruct-cpu-candidate.json)
took **2.629 s**. Peak process RSS rose from 1,680,552 to 1,955,952 KiB,
an increase of about **269 MiB** while the temporary Python lists coexist
with the final statistics object. This measured memory cost should be considered
for larger feature dictionaries.

## Complete published slice

The checked-in [lifecycle harness](../../scripts/benchmark/benchmark_axbench_adv_judge_lifecycle.py)
ran the same RTX 4090, 400M step-1013 source, concept 36,
`SelfConditioningMasked`, shared control plus both counter families, raw
strength 256, and ten responses per point. Both runs used fresh output/cache
paths and a deterministic local HTTP judge. The committed #1770 baseline ran
first and the committed candidate ran second. The baseline code is the parent
of this change; #1770's later commit only corrected a documentation count.

| Path | Wall time |
|---|---:|
| Parent statistics reader | 94.901 s |
| Bulk scalar reconstruction | 86.940 s |
| Saving | **7.961 s (1.092× complete speedup)** |

Each run made **90 local HTTP calls**. All **45 non-log files excluding the
manifest** matched byte for byte. The manifest differed only in its recorded
implementation commit; its file inventory and call accounting matched. Both
downstream counter slices loaded from the committed candidate output. The
[baseline](evidence/axbench-adv-stats-reconstruct-baseline-4090.json),
[candidate](evidence/axbench-adv-stats-reconstruct-candidate-4090.json), and
[comparison](evidence/axbench-adv-stats-reconstruct-comparison-4090.json)
reports contain timings and every matching file hash.

Targeted statistics, turn-context, AxBench config, and pipeline tests passed
(**415 passed, 16 skipped**), as did Ruff, targeted Ty, and diff checks. The
full suite was not run. No paid provider was called. The complete comparison
is one concept and one method; it does not establish a 50-concept or
live-provider speedup.
