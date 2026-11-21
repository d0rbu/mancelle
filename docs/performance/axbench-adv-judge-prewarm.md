# AxBench-Adv project judge prewarm

This follow-up to [#1773](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1773)
for [#1759](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/issues/1759)
starts the existing reusable project judge pool while the first runtime loads
its extraction stack, prepares prompts, and fits the selected directions. It
uses the first pending cell's validated method, concept, factor, and cache
identity. The first cell then rebinds the pool through the existing identity
check. No judge request is sent until generated text is ready. The option is
controlled by the existing `project_reuse_judge_workers` setting; its default
remains off. Runtime or worker-startup failure aborts every opened worker.

## Profile and comparison

On the local RTX 4090, runtime initialization on the #1773 head spent 7.374 s
loading the weight-free Gemma/SAE stack, 6.411 s preparing 373 prompts,
4.347 s fitting the selected dense DiffMean directions, and 0.341 s loading
the target model. The first reusable two-worker pool then took about 5.9 s to
start before the first cell could enter. Prewarm overlaps that pool startup
with the existing runtime work.

The real-GPU publication replays held the 400M step-1013 source, concept 36,
shared control, both counter families, raw magnitude 256, and ten responses
per point fixed. Both paths used the same two-worker pool, in-flight depth
three, native extraction, and all preceding stack options. Each used fresh
output and SQLite cache paths. The real framed Python 3.12 judge called a
deterministic local HTTP endpoint with a one-second request delay. No paid
provider was called.

| Workload | Parent wall | Prewarm wall | First cell entry, parent → prewarm | Complete speedup |
|---|---:|---:|---:|---:|
| One method, three points | 71.011 s | 64.088 s | 5.877 → 0.012 s | 1.108× |
| Three methods, nine points; parent first | 111.480 s | 107.850 s | 5.947 → 0.012 s | 1.034× |
| Three methods, nine points; prewarm first | 110.712 s | 107.698 s | 5.904 → 0.012 s | 1.028× |
| Three-method paired mean | 111.096 s | 107.774 s | — | 1.031× |

The one-method pair matched all **46 non-log output files**, including the
manifest, byte for byte; both made 90 local HTTP calls. Each three-method pair
matched all **58 non-log output files**, including the manifest, byte for byte;
each run made 270 local HTTP calls. These comparisons cover generated text,
per-point judge results, final evaluation, call accounting, and published
outputs. The candidate replays preceded their commit, so both arms recorded
the same parent implementation SHA in the manifests. A post-commit replay
would record this PR's SHA there. The first-cell entry reduction is a phase
timer; the complete wall gain is smaller because startup overlaps work on the critical
path. Both three-method run orders favored prewarm by 3.014–3.630 s.

The [comparison report](evidence/axbench-adv-judge-prewarm-4090-comparison.json)
records both run orders, individual timings, call counts, file-inventory
digests, and links to the six raw lifecycle reports. The checked-in
[`benchmark_axbench_adv_judge_lifecycle.py`](../../scripts/benchmark/benchmark_axbench_adv_judge_lifecycle.py)
accepts `--all-three-methods` to replay the larger slice.
The one-method comparison is an additional bounded check. These are
one-concept local-judge measurements, not a measured 50-concept campaign or
live-provider speedup.

Focused `test_self_conditioning.py`, `test_worker_client.py`, `test_run_config.py`,
`test_counter_pipeline.py`, and `test_counter_resume.py` tests passed
(**319 passed**). The tests include prewarm before first cell and worker cleanup
after runtime or partial pool-startup failure. Ruff, targeted Ty, and diff
checks passed. The full test suite was not run.
