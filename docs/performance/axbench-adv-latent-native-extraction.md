# AxBench-Adv latent counter native extraction

This follow-up to [issue #1759](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/issues/1759)
uses the 400M full SFT step-1013 encoder-decoder checkpoint and its Gemma-3-4B
residual-post SAE on a local RTX 4090. The bounded workload is concept 36,
`SelfConditioningMasked`, the shared zero plus DiffMean and LinearProbe
counters at raw magnitude 256, ten responses per point, and a deterministic
local HTTP judge. It makes no paid provider calls.

## Profile and change

The [phase trace](evidence/axbench-adv-native-extraction-4090-phase-profile.json)
recorded 373 prompt preparations in a complete published slice. Their Gemma
and SAE forward path took 6.055 seconds; the enclosing extraction and selection
steps took 7.762 and 0.990 seconds respectively. Sparse extraction took only
0.110 seconds. The prompt preparations cover the complete validated source so
the shared feature-selection RNG reaches the same state before the owned
concept is evaluated. Skipping other concepts would change selected features.

`project_native_extraction` is a default-off option for weight-free CUDA
counter runs with a Gemma-3 residual-post SAE. It uses the repository's
structurally truncated native collector for both prompt and dense DiffMean
SAE forwards instead of the nnterp tracer. The option is recorded in the
published manifest. The original source order, selection RNG, target
checkpoint, counter directions, generated text, and judging path remain the
same. The default path stays on nnterp.

## Production pipeline replay

The checked-in
[`benchmark_axbench_adv_judge_lifecycle.py`](../../scripts/benchmark/benchmark_axbench_adv_judge_lifecycle.py)
ran the complete slice from source validation through prompt extraction,
generation, the real framed judge worker against a deterministic local HTTP
endpoint, final instruction judging, aggregation, and manifest publication.
Both runs used the same branch and config except for the new option, fresh
output and SQLite cache paths, and 90 local HTTP calls. The candidate ran
before the default-off replay.

| Run | Complete wall time |
|---|---:|
| Native extraction | 70.974 s |
| Default-off nnterp | 78.427 s |

The measured gain is **7.453 seconds, or 1.105× end-to-end**. All 46
non-log, non-manifest files were byte-identical, including point receipts,
final judged outputs, aggregates, and plots. Two timestamped logs differed;
the sole manifest field difference was `project_native_extraction: true`.
The [comparison](evidence/axbench-adv-native-extraction-4090-comparison.json)
and raw [baseline](evidence/axbench-adv-native-extraction-4090-baseline-after.json)
and [candidate](evidence/axbench-adv-native-extraction-4090-candidate.json)
lifecycle reports retain timings, file counts, and accounting. An earlier
[baseline trace](evidence/axbench-adv-native-extraction-4090-baseline-before.json)
took 79.613 seconds, consistent with the default-off replay.

## Three-method parity replay

A second complete production-code slice covered `SelfConditioning`,
`SelfConditioningMasked`, and `NLSteering`, both counter families, the shared
zero, raw magnitude 256, and ten responses per point. It used the same local
judge and fresh caches. Native extraction took **127.618 seconds** versus
**140.918 seconds** for default-off nnterp: **13.300 seconds saved, 1.104×**.
Both runs made 270 local HTTP calls. All 63 non-log, non-manifest files were
byte-identical, including every point, final judged output, aggregate, and
plot. Only the timestamped final-worker log and the manifest option field
differed. The [comparison](evidence/axbench-adv-native-extraction-4090-all3-comparison.json)
and raw [baseline](evidence/axbench-adv-native-extraction-4090-all3-baseline.json)
and [candidate](evidence/axbench-adv-native-extraction-4090-all3-candidate.json)
reports retain the timings, file inventory, and call accounting.

Targeted checks passed: 233 tests covering the config, runtime dispatch,
native fit forwarding, and counter publication. Ruff and targeted Ty passed.
The full test suite was not run.

These slices measure one concept and one nonzero strength. They do not
establish a live-provider or 50-concept campaign speedup. The option is
intended for a fresh qualified run; existing three-method campaign outputs
are not migrated or replayed by this change.
