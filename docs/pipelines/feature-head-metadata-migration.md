# Explicit feature-head metadata conversion

Checkpoint readers require `block` (`mlp` or `transformer`) and nonnegative
integer `head_layers`. Writers persist the effective configuration in both
`user_config.feature_prediction` and `feature_prediction_config`. Both mirrors
must agree. Depth zero still constructs the linear head, but records both fields.
Fresh model construction may use the defaults `mlp` and `0`.

Checkpoints predating issue #1719 need an explicit metadata conversion before
inference, SFT, pretraining resume, topology migration, self-conditioning, GRPO or RLVR. Runtime
readers reject the removed `mlp_layers` field and missing architecture fields.
Headless checkpoints need no conversion. The removed SFT override
`feature_prediction_mlp_layers` is also rejected, including Hydra `+key` syntax.
The reader tolerates the historical SFT record of that option as integer zero
only when both metadata locations describe no head. It constructs no head and
does not accept the removed option in fresh configuration or CLI input.

The utility inspects the paired weight archive on CPU with memory-mapped,
metadata-only tensor loading. It interprets a missing old depth as zero only
inside the converter. It validates the projection and every declared residual
MLP tensor's keys, shapes and dtype, then emits `block: mlp`, `head_layers: N`.
It updates both mirrors and renames any recorded old SFT override. Conflicting
mirrors, mixed/partial schemas, depth/tensor disagreement, or an existing output
path are errors. Unrelated metadata is preserved; weights and optimizer files
are never rewritten. Publication of the new metadata file is atomic and refuses
overwrite, including concurrent destination creation.
If old metadata omits `loss_mode`, conversion records the legacy canonical
reader's `best_token` default explicitly; fresh construction still defaults to
`max_pool`.

Create a new checkpoint view, using real copies or reflinks for files that may
be resumed. Keep historical checkpoints and receipts immutable:

```bash
export SOURCE=/absolute/path/to/historical/checkpoint
export VIEW=/absolute/path/to/new-checkpoint-view
export STEP=001125
mkdir "$VIEW"
cp --reflink=auto "$SOURCE/model_$STEP.pt" "$VIEW/model_$STEP.pt"
uv run python -m steering.tools.migrate_feature_head_metadata \
  "source_metadata=$SOURCE/meta_$STEP.json" \
  "weights=$VIEW/model_$STEP.pt" \
  "output_metadata=$VIEW/meta_$STEP.json"
sha256sum "$SOURCE/meta_$STEP.json" "$VIEW/meta_$STEP.json" \
  "$SOURCE/model_$STEP.pt" "$VIEW/model_$STEP.pt"
```

Record both metadata hashes and require identical source/view model hashes.
For a resumable view, also copy every required optimizer shard and associated
checkpoint completion/state files into the isolated directory. Do not link
writable files back to a historical run. The depth-2 instruct checkpoint has
four optimizer shards; loading its weights alone is not a resumable SFT state.
For the original pretrained base use step `250000`; for the linear and depth-2
controls use step `001125`.

Conversion changes metadata identity. Regenerate metadata-bound receipts with
their existing producers, including the cadence view and any RLVR preparation
bundle. Editing old receipt hash strings is not a valid regeneration. Reuse
feature data and ChatCORE caches only after their normal compatibility checks
pass. Preserve old scores as historical measurements and label fresh evaluation
separately. See the [experiment runbook](../performance/residual-feature-head.md).

Fresh SFT may strictly load a converted linear head and add a requested positive
depth. Expanded checkpoints must match the requested block kind and depth.
Every resume must match both saved fields exactly before optimizer state is
applied. For example, a new run can use
`feature_prediction_block=transformer feature_prediction_head_layers=6`;
resuming that run requires the same values. There is no special sweep interface.
