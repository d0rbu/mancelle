# AxBench-Adv project extraction startup

This opt-in follow-up to [#1769](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1769)
for [#1759](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/issues/1759)
avoids loading unused TransformerLens Gemma-3 weights in the project
self-conditioning runtime. The nnterp HuggingFace decoder and the same SAE
still run extraction. The weight-free path reads TransformerLens's model config,
including its 8,192-token context cap, and the registered HuggingFace tokenizer.
The flag `project_weight_free_extraction` defaults to `false` and is recorded
in the published manifest only when enabled. Other extraction callers retain
their existing loader.

## Profile

The local RTX 4090 profile used the 400M step-1013 source, concept 36,
`SelfConditioningMasked`, both counter families, raw strength 256, and ten
responses per point. The real extraction-stack load took **27.523 s**:
`HookedTransformer.from_pretrained` consumed **19.481 s**, SAE load **0.875 s**,
and the separate nnterp decoder **6.997 s**. Source validation took **15.469 s**.
The [component profile](evidence/axbench-adv-weight-free-load-components-4090.json)
and [startup profile](evidence/axbench-adv-weight-free-baseline-startup-4090.json)
record these measurements. The weight-free prototype spent 0.0004 s reading
the TransformerLens config, 3.270 s resolving the tokenizer, 0.818 s loading
the SAE, and 7.549 s loading nnterp; see its
[phase report](evidence/axbench-adv-weight-free-prototype-phases-4090.json).
Using the HuggingFace context limit directly would change truncation: it
advertises 131,072 tokens for this model, whereas the existing TransformerLens
runtime uses 8,192.

## Complete published slice

The checked-in [lifecycle harness](../../scripts/benchmark/benchmark_axbench_adv_judge_lifecycle.py)
ran the full source-validation → preparation → three judged points → final
judging → aggregation → manifest path, with a deterministic local HTTP judge.
Both runs used the same source, settings, hardware, fresh output/cache paths,
and 90 local HTTP calls. The only production setting changed was
`project_weight_free_extraction`.

| Path | Wall time |
|---|---:|
| Original loader | 111.534 s |
| Weight-free loader | 94.940 s |
| Saving | **16.595 s (1.175× complete speedup)** |

The [baseline](evidence/axbench-adv-weight-free-baseline-lifecycle-4090.json)
and [candidate](evidence/axbench-adv-weight-free-candidate-lifecycle-4090.json)
reports hold the raw timings. All **45 non-log files excluding the manifest**
were byte-identical, including judged point receipts, final judged outputs,
and aggregates. Manifest file inventory and call accounting matched; its only
changed key was `project_weight_free_extraction: true`. Both downstream
counter-slice loaders accepted the candidate. The
[comparison report](evidence/axbench-adv-weight-free-comparison-4090.json)
contains every matching file hash. A preliminary in-process prototype took
96.138 s and matched all 46 non-log files, including the manifest, against
the original path; the production run is the performance claim above.

Focused config, self-conditioning, and pipeline tests passed (**215 tests**),
as did Ruff, targeted Ty, and diff checks. The full suite was not run. No paid
provider was called. This is one concept and one method, not a measured
50-concept or live-provider speedup. Keep the flag off for existing campaigns;
use it only for a fresh run after source/replay qualification by the campaign
owner.
