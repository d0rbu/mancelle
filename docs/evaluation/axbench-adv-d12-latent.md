# Native D12 SFT latent counter-steering grid

Use **`[64, 256, 512, 1024, 2048, 4096]`**, plus the existing shared zero
control, for the **D12/W768 decoder-only SFT checkpoint at step 1003**.
This is an exploratory grid supported by the completed coarse calibration.
The 512 and 2048 points fill the two largest relevant gaps; their refinement
judgments are incomplete. This is not a completed 50-concept robustness result.

| Preset in `configs/eval/axbench_adv/` | Positive methods | Source population |
|---|---|---|
| `native_methods_d12_sft_latent_c50` | DiffMean, LinearProbe, LoRA | Authenticated D12 standard100 union; evaluate concepts 0–49 |
| `native_counter_d12_sft_latent_c50` | PromptSteering | Authenticated D12 standard50 producer; see historical-runtime requirement below |

Both presets use DiffMean and LinearProbe counters at **decoder block-output
layer 9**, with `residual -= alpha * saved_normalized_direction`. There is no
activation, positive-factor, or width multiplier. Zero is supplied by the
authenticated control and must not be included in either magnitude list.
The standalone preset selects only the two latent pressures. Positive factors
remain selected by the authenticated c0–49 source judgments: DiffMean **0.4**,
LinearProbe **1.5**, and LoRA **1** for this source. PromptSteering uses factor 1
and its separately fitted canonical counter directions.

The inherited prompt checkpoint identities are:

| Artifact | SHA-256 |
|---|---|
| Model, step 1003 | `1247ed969508ad8b2449a353c95d6cbc1063a8572dfd4af720c9d81660cb3ffb` |
| Metadata | `2c8ba8a022889e796bff6c4078f90697903d219f633cafa4c33e698c99a78226` |
| Trained RustBPE tokenizer | `c3945190ac5c38a6ac941480e21895763775619e756cde4d108cafb5a081e185` |

Standalone checkpoint hashes flow through the authenticated union parents.
These presets are specific to this D12 target and intervention site. The
[D14/encoder-decoder presets](axbench-adv-400m-latent.md) have separate calibration.

## Completed calibration evidence

The September 23 coarse sweep contains **416 generated and fully judged
samples**: concepts 0–3, two independently authored prompts per concept, four
positive methods, both counter directions, and raw magnitudes
`[0, 16, 64, 256, 1024, 4096, 16384]`. Zero was generated once per
method/concept/prompt and reused across counter families. Each nonzero table
entry therefore has **eight observations**, with paired zero controls.

Generation used the actual SFT checkpoint, BF16/SDPA on RTX 4090, 128 new tokens,
temperature 1, top-k 50, and paired `cell_seed(42, concept_id, input_id)` seeds.
These exploratory seeds are shared across positive methods; the standalone
benchmark retains its existing method/factor-specific seed schedule. The two
calibration prompts do not occur in either canonical source prompt population:

1. Explain in three sentences why a metal spoon feels colder than a wooden spoon
   in the same room.
2. Write a short, polite message asking a neighbor to return a borrowed book next week.

Cached judgments use `openai/gpt-5.6-luna`, temperature 0, with the unchanged
AxBench concept, fluency and instruction prompts. The judge sees neither method
nor coefficient labels. A fresh audit checked every generation against its
completion inventory, rebuilt the exact judge request hashes, parsed the cached
provider responses, and matched all three ratings to each saved judgment. It
also checked all **400 fitted artifact hashes**, revalidated the D12 source and
checkpoint, and confirmed that the pilot judge templates equal the vendored
AxBench templates.

The following percentages pass the existing fluency/length/refusal adequacy
gate. The [complete coarse summary](calibration/d12_sft_counter_grid_coarse.csv) also
contains concept, fluency and instruction means, adequacy-gated fitness, and
paired changes from zero.

| Positive method | Counter | Zero | 64 | 256 | 1024 | 4096 |
|---|---|---:|---:|---:|---:|---:|
| PromptSteering | DiffMean | 87.5% | 75% | 100% | 62.5% | 0% |
| PromptSteering | LinearProbe | 87.5% | 62.5% | 87.5% | 50% | 0% |
| DiffMean | DiffMean | 75% | 75% | 87.5% | 75% | 0% |
| DiffMean | LinearProbe | 75% | 75% | 50% | 62.5% | 0% |
| LinearProbe | DiffMean | 62.5% | 87.5% | 75% | 75% | 0% |
| LinearProbe | LinearProbe | 62.5% | 62.5% | 100% | 100% | 12.5% |
| LoRA | DiffMean | 100% | 100% | 87.5% | 87.5% | 12.5% |
| LoRA | LinearProbe | 100% | 87.5% | 87.5% | 62.5% | 0% |

The grid retains 64 and 256 to resolve early effects, 512 and 1024 to sample
usable intermediate strengths, 2048 to probe the transition, and 4096 as the
measured collapse anchor. At 16384 every coarse output fails adequacy, so that
extra endpoint adds little to the proposed sweep. The legacy `[0.25, 0.5, 1, 2]`
default does not span the observed transition. It remains unchanged for older
run identities.

Low concept scores alone do not establish a successful attack: report adequacy
and paired changes alongside concept scores. Baseline concept expression is
already weak in this small authored-prompt cohort, and baseline fluency differs
across methods. These results select a useful range; they do not estimate a
universal collapse threshold or rank method robustness.

The disjoint refinement cohort (concepts 4–7, two other prompts, 480 generations)
has incomplete cached judgments and is **excluded** from the CSV and table.
Completing its 1,108 missing requests was blocked by the execution environment's
external-destination policy. No new paid receipts were produced by this audit.
The 512 and 2048 points are interpolation choices pending that refinement.

Raw generations, fit identities and cached provider receipts are retained at
`out/counter-calibration-d12-20260923/` in the campaign workspace. The review
audit is at `out/calibration-audit/` in the PR worktree. Source revision:
`99c81d18f2b97560744cf6eb7e4c993e7374dc9d`.

| Coarse artifact | SHA-256 |
|---|---|
| `identity.json` | `93187e26e970bc6bb8550d5f14d9f6484841952dc4fe1f2bb259e4c455dfe002` |
| `GENERATION_COMPLETE.json` | `4577b7da08fd206dcc260b98626c760977e633ffcaf977d600d937b8e2b415a8` |
| Checked-in summary CSV | `0a76668fc52368e6f9b54345fe56ade67a0a36a021a83e40c952e69448db1492` |

## Standalone launch and qualification

Use the D12 standard100 union and its original shard roots, the checkpoint above,
and the trained tokenizer. Set `NANOCHAT_BASE_DIR` to the tokenizer cache root.
The following is a provider-free qualification of the preset:

```bash
NANOCHAT_DTYPE=bfloat16 NANOCHAT_ATTN_IMPL=sdpa \
uv run python -m scripts.eval.axbench_adv.run_native_methods \
  --config-name=native_methods_d12_sft_latent_c50 \
  final_root="$D12_UNION" "archive_roots=$D12_ARCHIVE_ROOTS" \
  checkpoint_dir="$D12_CKPT" tokenizer_path="$TOKENIZER" \
  output_dir="$RUN/pilot" phase=pilot 'concept_ids=[0]'
```

On RTX 4090, this reproduced all **three archived zero controls exactly** and
generated six nonzero outputs at 64, covering all three positive methods and
both counter families, with no provider calls. The completed coarse sweep is
the evidence for the other measured coefficients; this production pilot does
not qualify all concepts or the entire grid.

For a later full run, use a fresh output directory, `phase=run`, omit the pilot's
`concept_ids=[0]` override (null selects 0–49), and supply a separate
`mailbox_dir`. The provider worker and final three-axis judging follow the
[native standalone workflow](axbench-adv.md#native-decoder-standalone-methods).
Generation alone does not constitute a judged benchmark result.

### Fresh runtime-control opt-in

The standalone DiffMean, LinearProbe, and LoRA preset also supports the
explicit `zero_control_source=runtime` policy. It retains the authenticated
D12 checkpoint, original fits, selected positive factors, prompts, and per-cell
seeds, while generating a new positive-only shared-zero cohort on the attack
runtime. Use a fresh output and mailbox; the policy is bound into the run
identity, so an archived-policy directory cannot be resumed as a runtime-policy
run:

```bash
NANOCHAT_DTYPE=bfloat16 NANOCHAT_ATTN_IMPL=sdpa \
uv run --no-sync python -m scripts.eval.axbench_adv.run_native_methods \
  --config-name=native_methods_d12_sft_latent_c50 \
  final_root="$D12_UNION" "archive_roots=$D12_ARCHIVE_ROOTS" \
  checkpoint_dir="$D12_CKPT" tokenizer_path="$TOKENIZER" \
  zero_control_source=runtime phase=run \
  output_dir="$RUN/runtime-c50" mailbox_dir="$RUN/runtime-c50-mailbox"
```

For the D12 c0–49 latent grid this produces **50 × 3 × 10 = 1,500** fresh
control generations and **50 × 3 × 10 × 2 × 6 = 18,000** nonzero generations.
The durable generation records carry fresh concept/fluency judgments, and
`judge_native_method_points` applies the ordinary final judge to the runtime
control as well as the nonzero points. **No archived control generation or old
concept, fluency, or instruction rating is reused.** This is a new judged
control cohort, not a parity claim about the archived outputs or ratings.

`phase=pilot concept_ids=[0]` is provider-free and exercises only input `i0` for
each of the three positive methods (plus the first configured magnitude for
each counter family). It does not exercise `i1` through `i9` and cannot qualify
the fresh-control judgment or persistence path. Under the archived policy,
passing the three c0/i0 replays also does not establish parity for later inputs;
the full run can still reject a mismatch at c0/i1.

Before a c0–49 launch, use a separate fresh root with
`phase=run concept_ids=[0]`, run the separately authorized mailbox provider and
final judge, and require all ten c0 inputs to generate, receive fresh judgments,
persist durable receipts, and resume without new calls. Only that all-ten-input
slice qualifies the runtime-control path beyond the one-input pilot. Then omit
`concept_ids=[0]` for the 50-concept command above.

## Historical PromptSteering runtime

The archived D12 standard50 PromptSteering source binds producer revision
`e912044e3b3520478fd507411118d6e1543f2710`. Current main deliberately rejects it
because two producer files differ. The new prompt preset does **not** bypass
that admission check and is not a current-main launch recipe for this archive.
The standalone `zero_control_source=runtime` option above does not apply to
PromptSteering; changing its historical-runtime path is out of scope.

To reuse that historical source, retain its matching producer runtime, BF16 and
unset `NANOCHAT_ATTN_IMPL`, as recorded in the source manifest. Its old counter
runner also fixes `COUNTER_MAGNITUDES` in code: adapt only that coefficient tuple
to `(64.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0)` in an isolated copy before
using this preset. Producer implementation files and source admission remain
unchanged. A new grid requires fresh output and fresh fit receipts (`phase=all`
or `phase=fit` followed by `phase=generate`), because fit identity includes the
grid. Do not copy old fit receipts into the new run.

The retained historical-runtime qualification reproduced two archived prompt
controls and two authored-prompt calibration controls exactly, and checked token
rendering for all 20 unique calibration user texts. This supports the pilot's
intervention semantics; a complete prompt run with the new grid remains unrun.
Alternatively, the current-main preset can consume a compatible current
producer source for the exact D12 checkpoint after ordinary source qualification.
