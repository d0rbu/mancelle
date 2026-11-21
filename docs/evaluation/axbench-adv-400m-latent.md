# Calibrated latent AxBench-Adv at 400M scale

These presets rerun **DiffMean and LinearProbe counter-steering**, retaining
the selected positive steering methods. They do not run GEPA, ChatCORE, or new
standard AxBench baselines. Standard100 outputs remain authenticated inputs and
supply native zero controls. The evaluation population is concepts **0–49**,
with the original ten AxBench-Adv prompts per concept.

| Preset in `configs/eval/axbench_adv/` | Target and positive methods | Nonzero raw counter magnitudes |
|---|---|---|
| `encdec400m_sft_latent_c50` | Encoder-decoder SFT step 1013: SelfConditioning, SelfConditioningMasked, NLSteering, each at factor 6 | 256, 1024, 4096, 8192, 16384, 32768, 65536 |
| `native_counter_d14_sft_latent_c50` | D14/W896 decoder SFT step 1001: PromptSteering | 64, 256, 512, 1024, 2048, 4096 |
| `native_methods_d14_sft_latent_c50` | Same decoder: DiffMean, LinearProbe, LoRA at their authenticated c50-selected factors | 64, 256, 512, 1024, 2048, 4096 |

The decoder has **399,115,836 parameters**. The old D12 decoder is not this
baseline. The encoder-decoder has one MLP feature-prediction head block. Its
counter fit uses decoder layer 9, at the saved encoder-decoder intervention
site; the native decoder counter uses layer 11 at decoder block output. A grid
is specific to the checkpoint, fitted vectors and site, not transferable by
parameter count alone.

## Calibration and interpretation

The September 23 pilot generated 128-token sampled continuations with the real
checkpoint and saved directions, then used the existing AxBench concept,
instruction and fluency judge prompts. It covered eight concepts per 400M model
and two independent calibration prompts per concept. Coarse and refined sweeps
use disjoint four-concept subsets for SelfConditioning and the native methods;
NLSteering's refinement covers all eight concepts. Each reported point therefore
has 8 or 16 outputs per positive method. This is exploratory scale calibration,
not a completed c50 benchmark or a precise estimate of a universal threshold.

The D14 sweep retained substantial coherence through 1024, degraded around 2048,
and had only 0–6.25% adequate outputs across the eight positive-method/counter
combinations at 4096. The encoder-decoder was largely usable at 4096–8192,
degraded at 16384, and had 0–6.25% adequate outputs at 32768. The 65536 endpoint
confirms the broad collapse region; finite-sample scores are not monotonic.
The original `[0.25, 0.5, 1, 2]` grid is retained as the legacy default, but does
not probe these models' measured coherence boundaries.

Use the **same grid for every positive method on a model**, and judge both
counter families independently. Every coefficient means `residual -= alpha * d`
with the saved normalized vector `d`. Do not multiply by signed fit calibration,
maximum activation, positive steering factor, or a model-width factor. Zero is
already supplied by the shared control and must not appear in the YAML list.

Coherence collapse determines the upper calibration bound; incoherent text is
not a successful attack. Retain the existing fluency/length/refusal adequacy
checks and show concept score alongside fluency, instruction score, adequate
fraction and changes relative to each method's zero control. Some controls are
already imperfect, so an absolute fluency threshold alone is insufficient for
comparing robustness.

The raw pilot outputs, judge receipts, exact inputs, source SHA, summary tables
and curves are archived under
`hf://buckets/d0rb/nano-gpt-pretrain-steer-artifacts/evals/axbench_adv/counter_scale_calibration_20260923/`.
The exploratory NLSteering factor-5 cohort is retained there for audit and
excluded from the primary calibration; the preset uses the verified factor 6.

## Encoder-decoder launch

Supply the relocated historical six-method source, its exact operative target
metadata/receipt, the completed fit root, and the separate NL attachment. Paths
are mandatory rather than tied to a workstation. The inherited SmolLM source
identity describes historical main/aux inputs; the Hydra method filter limits
execution to the three encoder-decoder methods. No SmolLM generation is launched.

```bash
# REF: authenticated encdec400 reference source; NL: NL attachment + judgments.
# INPUTS: historical main/aux source root; FITS: completed checkpoint-own fit root.
# STATS: feature_statistics.safetensors; NANOCHAT_BASE_DIR: trained tokenizer root.
uv run python -m scripts.eval.axbench_adv.run \
  --config-name=encdec400m_sft_latent_c50 \
  source.prepared_dir="$REF/prepared_dir" \
  source.main_parquet="$INPUTS/main/inference/steering_data.parquet" \
  source.aux_parquet="$INPUTS/aux/inference/steering_data.parquet" \
  source.evaluated_jsonl="$REF/source/evaluated_six_method.jsonl" \
  source.project_parquet="$REF/project/inference/steering_data.parquet" \
  source.target_meta="$REF/source/meta_001013.json" \
  source.target_receipt="$REF/source/remote_complete_001013.json" \
  source.feature_statistics="$STATS" \
  source.nl_attachment.paths.project_parquet="$NL/nl-adv50-attachment/inference/steering_data.parquet" \
  source.nl_attachment.paths.project_manifest="$NL/nl-adv50-attachment/run_manifest.json" \
  source.nl_attachment.paths.evaluated_jsonl="$NL/nl100-final/steering.jsonl" \
  project_counter_direction_root="$FITS" \
  output_dir="$RUN/output" judge_cache="$RUN/judge.sqlite" \
  reflection_cache="$RUN/unused-reflection.sqlite"
```

Use a fresh output directory. The fit authenticates operative metadata SHA-256
`81e0af5b64b9dfcf8d6aa4ec80aacbf086a8885e86b6d8a247e40789119eda31`;
substituting a later metadata stamp requires a compatible fit/source receipt,
even when model weights match. The model hash is
`9e7fb23e4f39b2094c175fb7e80b91e6df7a30be6e8f3ecfe4d90b6eaa8f1842`.
For an authenticated relocated prepared tree, use the existing
`AXBENCH_ADV_ALLOW_RELOCATED_PREPARED=1` relocation mechanism. A GPU is required;
the project runner also performs final judging using the existing authorized
provider configuration.

Concept workers can use disjoint `AXBENCH_ADV_CONCEPT_LO`/`HI` intervals.
Method workers can override `hydra.job.env_set.AXBENCH_ADV_METHODS` to a subset
of the three named methods. Give each project worker a fresh output and separate
caches; merge with the existing counter slice merger.

## Native decoder launches

Restore the exact D14 step-1001 checkpoint/tokenizer and standard100 parents.
Set `NANOCHAT_BASE_DIR` to the trained tokenizer root and retain the producer's
precision/backend environment (`NANOCHAT_DTYPE=bfloat16` for this campaign).
The archived c0–49 standalone controls were generated on H100. A 4090 pilot with
the same checkpoint, seeds and recorded `sdpa` setting did not reproduce those
strings exactly; that backend label alone does not pin the underlying GPU
arithmetic. The default `zero_control_source=archived` preserves the exact-replay
guard and should be qualified with `phase=pilot` on the intended lane.

For a reviewed D12 or D14 standalone-method run,
`zero_control_source=runtime` instead generates the positive-only shared zero
with the same adapter, selected factor, prompt rendering, per-cell seed and
sampling settings used by the attacks. It stores fresh concept/fluency
judgments through the durable generation cache and runs the ordinary final
evaluator so instruction judgments are fresh too. This is a separately
identified same-runtime control lineage: authenticated standard100 still
supplies factor-selection and fit provenance, but its generations and ratings
are not copied and no full baseline reevaluation is implied. Changing the
policy requires a fresh run directory.

```bash
uv run python -m scripts.eval.axbench_adv.run_native_counter \
  --config-name=native_counter_d14_sft_latent_c50 \
  checkpoint_dir="$CKPT" tokenizer_path="$TOKENIZER" \
  source_manifest="$PROMPT_SOURCE/run_manifest.json" \
  source_parquet="$PROMPT_SOURCE/steering_data.parquet" \
  prepared_dir="$PREPARED" output_dir="$RUN/prompt-counter"

uv run python -m scripts.eval.axbench_adv.run_native_methods \
  --config-name=native_methods_d14_sft_latent_c50 \
  checkpoint_dir="$CKPT" tokenizer_path="$TOKENIZER" \
  final_root="$UNION" "archive_roots=$ARCHIVE_ROOTS" \
  zero_control_source=runtime \
  output_dir="$RUN/method-counter" mailbox_dir="$RUN/provider-mailbox"
```

`ARCHIVE_ROOTS` is a Hydra list of the original native shard roots, not their
parent directory. The prompt preset validates explicit hashes and D14 geometry;
the standalone-method preset revalidates checkpoint identities and selected
factors through the union's authenticated parents. Do not substitute the old D12
standard100 union. `phase=pilot concept_ids=[0]` on the standalone-method command
runs a provider-free check at the first configured magnitude.

Prompt counter workers can split disjoint `point_methods`/`point_magnitudes`
after `phase=fit`; these are placement filters within one immutable full-grid
identity. Standalone-method workers can split disjoint `concept_ids` with their
own output/mailbox directories. A separately authorized
`run_native_provider` process consumes method mailboxes. Final judging uses
`run_native_evaluation` for each completed prompt counter point and
`judge_native_method_points` for the standalone methods. The default archived
policy reuses authenticated zero-control judgments. Runtime controls use the
normal final judge and its cache; `final_root` remains an authenticated
generation input but is not needed by final-judge recovery.

Changing a grid requires a new run directory. Both native paths bind the full
configured grid into their identities and reject a mismatched resume. Existing
legacy magnitude labels and default grids remain compatible.
