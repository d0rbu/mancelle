# D14 native AxBench evaluation handoff

Implementation commit: `9cd29174d7408a8c4fdd1cb0b9948ac2e2dc716c`

Base checkout: `1b72cfadd4ab2de92e5609cfdb31d8aa0183ee59`

Status: CPU implementation and contract qualification complete. GPU execution,
live provider work, artifact upload, PR creation, and merge were not performed.

## Contract implemented

- The native standalone stack now admits exactly two reviewed decoder contracts:
  D12/768/1024 and D14/896/1024. Other depth/width/context combinations fail
  before fit, generation, or provider work.
- D14 model loading additionally requires exactly `399115836` parameters.
- Layer adaptation reuses the existing encoder-decoder baseline resolver:
  `floor(target_n_layer * qwen_layer / 36 + 0.5)` (round-half-up).
  Qwen layer 28 therefore maps to D12 layer 9 and D14 layer 11. Qwen LoRA
  layers `[7,14,21,28]` map to D12 `[2,5,7,9]` and D14 `[3,5,8,11]`.
  This preserves the reviewed relative-depth choice; it is not a layer sweep or
  a claim of empirical optimality.
- D12 keeps the existing format/schema, constants, default hashes, step 1003,
  and provider-request bytes. D14 has separate Hydra presets with no fabricated
  final step or hashes.
- The native PromptSteering loader and full-model checkpoint loading path are
  unchanged.
- The authenticated zero-control judge replay remains intact and was included
  in the focused CPU gate.

## Changed files

- `steering/eval/axbench_upstream/native_baselines.py`
- `steering/eval/axbench_adv/native_methods.py`
- `steering/eval/axbench_adv/native_mailbox.py`
- `steering/eval/axbench_adv/native_provider.py`
- `configs/eval/axbench_upstream/native_baselines_d14.yaml`
- `configs/eval/axbench_adv/native_methods_d14.yaml`
- `tests/scripts/eval/test_axbench_native_baselines.py`
- `tests/axbench_adv/unit/test_native_methods.py`
- `tests/axbench_adv/unit/test_native_method_provider.py`
- `docs/evaluation/axbench-upstream.md`
- `docs/evaluation/axbench-adv.md`
- `docs/cli-reference.md`
- `docs/reference/file-reference.md`

## Identity propagation

1. The D14 standard producer requires the actual final SFT step and exact
   model, metadata, and RustBPE tokenizer SHA-256 values. It checks metadata
   geometry, reviewed layers, loaded geometry, and loaded parameter count.
2. Fit contracts bind the producer identity digest, concept/training source,
   D14 width, D14 intervention layer, and D14 LoRA layers. Fit manifests hash
   every direction/adapter artifact. Generation cells bind the fit hash and run
   identity; run manifests bind the exact checkpoint/tokenizer identity.
3. The standard union remains a relocation index. Adv reauthenticates the
   union inventory, the declared parent archive bytes, every extracted shard,
   shard identity, fit/run manifest, generation cell, fitted tensor shape, and
   judged standard output. All shards must share one scientific identity.
4. The Adv runtime rehashes its checkpoint and tokenizer and matches step plus
   all three hashes to the standard producer identity before model/provider
   work. D14 provider judge/reflection requests also include the exact Adv run
   identity digest. Legacy D12 request bytes are unchanged.
5. Every Adv saved generation, GEPA/counter cell, point input, emission file,
   and judge receipt is bound to the Adv run identity. Final judging validates
   every point before calls. The shared zero control reauthenticates the exact
   union provenance/files and replays the original standard judgments with zero
   new calls. Archive validation retains this lineage.

These links reject old D12 directions (768-wide), LoRA state/layers, standard
union parents, runtime checkpoint/tokenizer files, generation outputs, and
judged outputs when presented as D14.

## D14 standard baseline configuration

The preset intentionally contains unresolved values. Substitute the real final
SFT values; do not reuse the example token names as literals.

```bash
uv run --no-sync python -m scripts.eval.axbench_upstream.run_native_baselines \
  --config-name=native_baselines_d14 \
  phase=fit \
  source_path=/ABS/STANDARD100/steering_data.parquet \
  source_sha256=<STANDARD100_PARQUET_SHA256> \
  prepared_dir=/ABS/AXBENCH_PREPARED \
  checkpoint_dir=/ABS/D14_FINAL_SFT_CHECKPOINTS \
  step=<ACTUAL_FINAL_SFT_STEP> \
  tokenizer_path=/ABS/TOKENIZER/tokenizer.pkl \
  expected.model_sha256=<ACTUAL_MODEL_SHA256> \
  expected.meta_sha256=<ACTUAL_META_SHA256> \
  expected.tokenizer_sha256=<ACTUAL_TOKENIZER_SHA256> \
  output_dir=/ABS/FRESH/D14_NATIVE_STANDARD

uv run --no-sync python -m scripts.eval.axbench_upstream.run_native_baselines \
  --config-name=native_baselines_d14 \
  phase=generate \
  source_path=/ABS/STANDARD100/steering_data.parquet \
  source_sha256=<STANDARD100_PARQUET_SHA256> \
  prepared_dir=/ABS/AXBENCH_PREPARED \
  checkpoint_dir=/ABS/D14_FINAL_SFT_CHECKPOINTS \
  step=<ACTUAL_FINAL_SFT_STEP> \
  tokenizer_path=/ABS/TOKENIZER/tokenizer.pkl \
  expected.model_sha256=<ACTUAL_MODEL_SHA256> \
  expected.meta_sha256=<ACTUAL_META_SHA256> \
  expected.tokenizer_sha256=<ACTUAL_TOKENIZER_SHA256> \
  output_dir=/ABS/FRESH/D14_NATIVE_STANDARD
```

Concept/method slices may use the existing `concept_lo`, `concept_hi`, and
`methods` controls with disjoint output roots. Build and judge the complete
standard100 product using the existing shard/archive process, then publish its
verified union only after all parent archives and method grids validate.

## D14 AxBench-Adv configuration

The Adv preset takes the actual step but deliberately does not expose duplicate
hash overrides: hashes come from the authenticated standard shard identities
and must match the runtime files.

Provider-free one-concept qualification:

```bash
uv run --no-sync python -m scripts.eval.axbench_adv.run_native_methods \
  --config-name=native_methods_d14 \
  phase=pilot concept_ids='[0]' \
  final_root=/ABS/D14_STANDARD100_UNION \
  archive_roots='[/ABS/D14_SHARD_0/native,/ABS/D14_SHARD_1/native]' \
  checkpoint_dir=/ABS/D14_FINAL_SFT_CHECKPOINTS \
  step=<ACTUAL_FINAL_SFT_STEP> \
  tokenizer_path=/ABS/TOKENIZER/tokenizer.pkl \
  output_dir=/ABS/FRESH/D14_ADV_PILOT
```

Full Adv generation (only after separately authorizing GPU and provider work):

```bash
uv run --no-sync python -m scripts.eval.axbench_adv.run_native_methods \
  --config-name=native_methods_d14 \
  phase=run \
  final_root=/ABS/D14_STANDARD100_UNION \
  archive_roots='[/ABS/D14_SHARD_0/native,/ABS/D14_SHARD_1/native]' \
  checkpoint_dir=/ABS/D14_FINAL_SFT_CHECKPOINTS \
  step=<ACTUAL_FINAL_SFT_STEP> \
  tokenizer_path=/ABS/TOKENIZER/tokenizer.pkl \
  output_dir=/ABS/FRESH/D14_ADV \
  mailbox_dir=/ABS/PRIVATE/D14_ADV_MAILBOX
```

Final judging, preserving authenticated control replay:

```bash
uv run --no-sync python -m scripts.eval.axbench_adv.judge_native_method_points \
  run_dir=/ABS/FRESH/D14_ADV \
  final_root=/ABS/D14_STANDARD100_UNION \
  judge_cache=/ABS/PRIVATE/D14_ADV-JUDGE-CACHE.sqlite \
  cache_only=true
```

Use `cache_only=false` only with separate provider authorization. The mailbox
provider remains a separately operated CPU service; this checkout did not run it.

## CPU evidence

Focused native baseline/Adv contract gate:

```text
164 passed, 2 skipped, 19 warnings in 16.27s
```

The two skips are the opt-in authentic recovery-artifact cases requiring
`NATIVE_ADV_RECOVERY_ARTIFACT_ROOT`. The passing set covered standalone fit and
generation contracts, D12 compatibility, D14 geometry/layer/parameter-count
admission, wrong step/hash/tokenizer rejection, D12-artifact substitution
rejection, standard judging, Adv source/union/counter identity, D14 provider
identity propagation, final judging, and zero-control replay.

Additional checks:

```text
Ruff (touched Python files): passed
ty (touched Python files): passed
git diff --check: passed
D14 baseline and Adv Hydra composition with a supplied step/hash fixture: passed
```

The warnings were existing dependency deprecations plus unavailable NVML in the
CPU sandbox; no test failed. No full repository suite was run.

## Remaining qualification owned by root

1. After SFT completion, independently compute the final model, metadata, and
   tokenizer hashes and record the actual final SFT step. Confirm the metadata
   says D14/896/1024 and the loaded model count is 399,115,836.
2. Run a minimal D14 standard fit/generation GPU slice, verify layer-11 capture,
   `[3,5,8,11]` LoRA state, nonzero LoRA updates, finite generations, and exact
   resume on the same identity. Then complete and judge standard100.
3. Build the verified standard union and retain every original shard archive and
   extracted native root. Re-run Adv source validation after relocation.
4. Run the provider-free D14 Adv pilot and compare its zero-control generation
   to the authenticated standard output. This is the first real-GPU check of the
   generalized positive and counter scopes.
5. With separate provider authorization, run the full Adv mailbox workflow and
   final judging. Verify all 13 point contracts, the D14 run digest on mailbox
   requests, zero new calls for shared-control replay, aggregates, plots, and
   final archive validation.
6. Run the remaining owner-required D14 evaluations outside this native
   standalone blocker, archive evidence, and open the PR. Do not treat the CPU
   evidence here as GPU, provider, convergence, or result qualification.
