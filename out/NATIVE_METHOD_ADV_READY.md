# Native decoder standalone-method Adv handoff

Status: CPU-qualified implementation ready for root's provider-free GPU pilot.
The commit is the commit containing this file; resolve its immutable SHA with
`git rev-parse HEAD` after checkout (a commit cannot contain its own SHA).

## Exact scope

- Positive steering methods are exactly `DiffMean`, `LinearProbe`, and `LoRA`.
  They are distinct from the pressures, which are exactly `gepa`,
  `linear_probe_counter`, and `diffmean_counter`.
- DiffMean and LinearProbe apply `selected_factor * fit.max_activation *
  saved_fp32_unit_direction`. LoRA attaches the authenticated saved rsLoRA
  state at layers `[2, 5, 7, 9]`; it is not converted to a direction.
- Both counter families subtract `[0.25, 0.5, 1.0, 2.0] *
  saved_fp32_unit_direction` from their separately fitted canonical TRAIN
  DiffMean or LinearProbe fit. Positive calibration is not applied to counters.
- Best positive factors are recomputed from exactly the judged standard-method
  population `concept_id=0..49`; all 100 judged concepts must authenticate, and
  missing, duplicate, or concept-100 substitutions fail closed.
- The standard judged generation at the selected factor is reused as the shared
  zero point. No baseline generation or PromptSteering evaluation is launched.
- GEPA keeps proposal counts `[1, 2, 4, 8]`, seed `42`, the existing seed
  suffix/reflection protocol, and the native mailbox protocol.
- Original standalone shard identities, fit/run manifests, fit bytes, cell
  inventories and contents, source hashes, concepts/methods, checkpoint,
  tokenizer, RustBPE policy, geometry, and generation policy authenticate before
  any provider request. `union-provenance.json` is only the verified standard100
  relocation index; it is not a synthesized native run manifest.

Existing `NativeDecoderPromptSteering` requests remain accepted byte-for-byte by
the provider. Tagged requests are restricted to the three standalone methods and
their fixed factor grids; provider judging still uses the unchanged method-agnostic
inner concept/fluency evaluator.

## Root first-concept provider-free recipe

Run only after `../decoder100-final/` contains all three completed method
directories plus the combined `union-provenance.json`, and after all original
Adv50 fit archives have been read back. The archive glob deliberately names the
original extracted native roots; source validation will fail before model loading
if any `(method, concept 0..49)` fit is absent or duplicated.

```zsh
native_archives=(
  ../jobs/decoder-*/native(N/)
  ../decoder-readback/*/contents/native(N/)
)
native_archive_csv="${(j:,:)native_archives}"

NANOCHAT_BASE_DIR=/home/d0rb/Documents/Github/nano-gpt-pretrain-steer/nanochat_cache \
uv run python -m scripts.eval.axbench_adv.run_native_methods \
  final_root=../decoder100-final \
  "archive_roots=[${native_archive_csv}]" \
  checkpoint_dir=../../local400m/nanochat/base_checkpoints/d12_base/standard_steered_sft_checkpoints \
  step=1003 \
  tokenizer_path=/home/d0rb/Documents/Github/nano-gpt-pretrain-steer/nanochat_cache/tokenizer/tokenizer.pkl \
  output_dir=../decoder-method-adv-pilot-c000 \
  phase=pilot \
  concept_ids='[0]' \
  device=cuda:0
```

This phase creates `pilot.json` and `status.json`, evaluates all three positive
interventions and both counter families at magnitude `0.25`, constructs no
mailbox, and performs no provider call. Use a fresh output directory on retry;
identity-bound outputs fail closed rather than being silently overwritten.

For a full run, retain every default grid, change `phase=run`, provide a disjoint
root-owned `mailbox_dir`, and set the intended `concept_ids`. The separate final
judge entry point is:

```zsh
uv run python -m scripts.eval.axbench_adv.judge_native_method_points \
  run_dir=PATH_TO_COMPLETED_RUN \
  judge_cache=PATH_TO_ROOT_OWNED_CACHE \
  cache_only=true
```

Root remains responsible for changing `cache_only`, credentials, provider
consumption, GPU execution, judging, and uploads.

## Reusable source and fit identities

Shared scientific source identity authenticated on the actual local concept 0
and concept 1 archives:

- common identity SHA-256:
  `e93b0d441718512d13e90df713371bbd084935d93887d7c4598797fd250c2390`
- model: `1247ed969508ad8b2449a353c95d6cbc1063a8572dfd4af720c9d81660cb3ffb`
- checkpoint metadata:
  `2c8ba8a022889e796bff6c4078f90697903d219f633cafa4c33e698c99a78226`
- RustBPE tokenizer:
  `c3945190ac5c38a6ac941480e21895763775619e756cde4d108cafb5a081e185`
- standard source:
  `5e8cca19c31d7924c39792f20f4d50ce74e1acb9e265ce3dd80a1dd079c33b0f`
- prepared metadata:
  `435aad14e202402b0989d62b279f63dfc63aa2aff0e488cf62652c0700a1efdc`
- canonical TRAIN data:
  `6e4a00f39e9a11c1f6c74ed0b3241d1122621e6b55ce063eb3ad071b075b977b`
- decoder geometry: 12 layers, width 768, sequence length 1024; direction
  intervention is decoder block output layer 9.

Actual authenticated fit SHA-256 values used by the CPU pilot contract:

| concept | DiffMean | LinearProbe | LoRA |
|---|---|---|---|
| 0 | `3b49226c37e394dee3dd8d9d9779a046394fa17e196c2a2ea158b31a996e7fcf` | `6feff7a617c2e1dff5ee082e92dc108089d13c93ab59c301ca77d82afbd9a5ef` | `4a48e86bfeb29d96cd1875e7a67ef197bfa9646c87907031c9d6ae8bbc3d3574` |
| 1 | `e2a34a1b77f45b70903d9edd178d7f38a592c3bae41ab2109af8956e6836b14b` | `64f1209e74109d0e85a55956b4ff8346f79586c0d9e294b32d2da76b6da93048` | `d5fa18fbc575f030c86e14e1879f7db0ae0fce614b9c57b9079f661e53849cdf` |

The final three best factors are intentionally not hard-coded here. They are
selected at preflight from the completed judged concepts 0..49 and recorded in
the immutable run identity.

## CPU qualification and remaining limits

- Focused method/provider/legacy-native tests: `74 passed`.
- Ruff format/check: clean. Targeted `ty check`: clean. `git diff --check`: clean.
- Actual concept 0/1 DMLP, LinearProbe, and LoRA archives passed full identity,
  union-bound parent-tar member/proof/extracted-byte inventory, manifest,
  fit-byte, cell, source-grid, and policy authentication.
- Mutation coverage includes source/hash mismatch, exact-50 selection,
  calibrated positive versus raw negative arithmetic, actual LoRA state use,
  immutable run/cell identity, loaded-adapter identity, output-manifest hashes,
  interrupted judge-receipt recovery, all three tagged provider routes, and
  preservation of legacy PromptSteering requests.
- The owner-required read-only Codex CLI review found five blockers across two
  cycles; all were fixed, mutation-tested, and the helper's final closure found
  no remaining concrete blocker in the reviewed scope.

No real-GPU generation, CUDA numerical comparison, full Adv50 union preflight,
provider request, final instruction judge, aggregate publication, or convergence
claim was made in this worktree. Root's fresh concept-0 GPU pilot is the next
required gate; the completed three-method standard100 union and remaining
readback archives were not available during this CPU qualification.
