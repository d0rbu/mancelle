# AxBench-Adv project judge-worker reuse

This opt-in follow-up to [#1768](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1768)
for [#1759](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/issues/1759)
keeps the two project judge processes alive across sequential method and
concept cells. Before each new cell, the vendored worker closes its previous
judge and creates one from the new validated method, concept, factor, cache,
and endpoint payload. The parent verifies the returned identity. SQLite cache
connections and per-cell accounting are recreated; GPU generation, prompts,
judge model, request bound, and point publication are unchanged. Any rebind or
cell failure aborts the pool. The default continues to start fresh workers
for each cell.

## Bottleneck measurement

The [lifecycle profile](evidence/axbench-adv-judge-rebind-lifecycle-4090.json)
from the [checked-in harness](../../scripts/benchmark/benchmark_axbench_adv_judge_lifecycle.py)
ran the published pipeline on the local RTX 4090 with the 400M step-1013
source, `SelfConditioningMasked`, one concept, shared control, both counter
families, raw strength 256, and ten inputs per point. The real two-worker
Python 3.12 judge used a deterministic local HTTP endpoint; no paid provider
was called. Opening the two workers took **7.215 s** and closing them took
**1.931 s** in a **111.716 s** complete slice. The same worker pair stayed
open for all three control/counter points. With all three project methods,
fresh processes repeat that lifecycle at the next method boundary.

## Matched real-GPU publication comparison

The [publication benchmark](../../scripts/benchmark/benchmark_axbench_adv_judge_workers.py)
held the 400M source, concept 36, all three project methods, both counter
families, raw strength 256, and ten inputs per point fixed. Both arms used
two inner judge workers, in-flight depth three, paired concept/fluency
requests, packed-mask reuse, repeated DiffMean-text reuse, selected-concept
fitting, and the persistent final evaluator. Each arm had fresh output and
SQLite cache paths. The local HTTP judge delayed each request by one second.

| Run order | Fresh workers | Rebound workers | Saved | Complete speedup |
|---|---:|---:|---:|---:|
| Rebind first | 170.610 s | 164.167 s | 6.443 s | 1.039× |
| Fresh first | 174.079 s | 151.101 s | 22.978 s | 1.152× |
| Paired mean | 172.344 s | 157.634 s | 14.711 s | 1.093× |

Both comparisons made **270 local HTTP calls per arm** and matched complete
call accounting, the manifest file inventory, and all **54 non-log published
files excluding the manifest** byte for byte. Both downstream counter slices
loaded. Direct manifest comparison found only the conditional
`project_reuse_judge_workers: true` field. The
[rebind-first](evidence/axbench-adv-judge-rebind-4090-candidate-first.json) and
[fresh-first](evidence/axbench-adv-judge-rebind-4090-baseline-first.json)
reports contain exact timings and parity checks.

Point-file timestamps show a **60.981 s** mean first-to-last point span with
rebind versus **76.371 s** with fresh workers, a **15.390 s** saving in the
repeated method work. Startup-to-first-point variation explains much of the
whole-run difference between orders. These file intervals are diagnostics,
not independent phase timers.

A separate provider-free worker smoke used the real Python 3.12 process to
judge before and after a method rebind, then sent an invalid judge model on a
second rebind. The first two evaluations made four local HTTP calls; the
invalid rebind failed, terminated the child, and left no live worker process.

The focused `test_self_conditioning.py`, `test_worker_client.py`,
`test_run_config.py`, `test_counter_pipeline.py`, and
`test_counter_resume.py` subset passed (**310 tests**). Ruff, targeted Ty for
project files, and `git diff --check` passed. The vendored worker has two
pre-existing Ty diagnostics in `_exact_keys`; the same diagnostics reproduce
on the untouched #1768 head. The full suite was not run.

This is a bounded one-concept local-judge replay, not a measured live-provider
or 50-concept campaign speedup. Rebind does not raise the existing four-request
inner-judge concurrency bound. Keep the option off for existing campaigns and
enable it only for a fresh qualified source after the campaign owner chooses
to integrate the stack.

The SHA-256 digests of the fresh-first report, rebind-first report, lifecycle
report, publication benchmark, and lifecycle harness are respectively
`054d4bf22c4bc69aa51ff646205a50d258c76803e5a16716228d07c62cd91caf`,
`c2258ee0f14f4258ba80816eae888ece4daf47af81913256e05e1cd7adda701b`,
`9b27fdb91218cfb393e38f56ab823521c6a39bf38dd3627dcab35ab5dc303c27`,
`fcf1b4a40b67c3ae49ea62ab379f56b1c44482a2cdefac0f5177404de7b212c3`,
and `7e0d56006fe611dd3cf284ad66587ff8b36254e14af00e3b0822bfdd3ed5d3d9`.
