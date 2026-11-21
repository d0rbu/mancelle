# CLI reference

Every command-line entry point in the repo, grouped by purpose. For deep-dives on a pipeline, follow the link in each section header.

**Conventions used below**

- `python -m X` is shorthand for `uv run python -m X` if you haven't activated the venv.
- Scripts marked **DDP** are launched with `torchrun --standalone --nproc_per_node=N`. The `--` separator is required between torchrun args and Hydra overrides.
- Configs use [Hydra](https://hydra.cc/). Overrides are `key=value` pairs (compose-API scripts) or `--config-name=...` plus overrides (scripts using `@hydra.main`). When a command uses the optional `--` separator, put Hydra flags such as `--config-name=...` **before** it and only `key=value` overrides after it. Both forms are documented per-entry.
- Quickstart configs (small/single-GPU variants) are listed where they exist.

---

## Setup & data

| Command | Purpose |
|---|---|
| `python -m nanochat.dataset -n 8` | Download N FineWeb-Edu pretraining shards (~250M chars each) into `$NANOCHAT_BASE_DIR` |
| `python -m scripts.preprocess.tok_train` | Train BPE tokenizer (vocab = 2¹⁵) on downloaded corpus. Config: `configs/data/tok_train.yaml` |
| `python -m scripts.preprocess.tok_eval` | Evaluate tokenizer compression ratio. Config: `configs/data/tok_eval.yaml` |
| `python -m scripts.preprocess.setup_hf_tokenizer` | Set up the HuggingFace Gemma tokenizer (used by feature extraction) |
| `python -m scripts.preprocess.count_tokens` | Estimate total token count for a planned pretraining run |

---

## Pretraining — see [`docs/pipelines/base-training.md`](pipelines/base-training.md)

| Command | Purpose | DDP |
|---|---|---|
| `python -m scripts.train.base_train` | Pretrain a decoder-only GPT (the nanochat-style base model). Config: `configs/training/base_train.yaml`. Quickstart: `config_name=base_train_tiny_1gpu` | ✓ |
| `python -m scripts.train.training_progression` | Pretrain while emitting per-step metrics for plotting later | ✓ |

**Common invocations**

```bash
# Full run on 8×H100
torchrun --standalone --nproc_per_node=8 -m scripts.train.base_train -- \
    depth=26 target_param_data_ratio=8.25 device_batch_size=16 fp8=true run=d26

# Tiny smoke test on 1 GPU (~5 min)
python -m scripts.train.base_train config_name=base_train_tiny_1gpu

# Resume from step
torchrun --standalone --nproc_per_node=8 -m scripts.train.base_train -- \
    resume_from_step=10000 run=d26
```

---

## SFT & RL — see [`docs/pipelines/chat-training.md`](pipelines/chat-training.md), [`docs/pipelines/sft-mixing.md`](pipelines/sft-mixing.md), [`docs/pipelines/sft-self-conditioning-grpo.md`](pipelines/sft-self-conditioning-grpo.md)

| Command | Purpose | DDP |
|---|---|---|
| `python -m scripts.train.chat_sft` | SFT a base model on chat-formatted conversations (standard / safety / capabilities mixes). Config: `configs/training/chat_sft*.yaml` | ✓ |
| `python -m scripts.train.chat_rl` | RL fine-tune via GRPO on GSM8K (or another verifiable task). Config: `configs/training/chat_rl.yaml` | ✓ |
| `python -m scripts.train.conditional_encdec_sft` | SFT the encoder-decoder model with feature conditioning. Config: `configs/training/conditional_encdec_sft.yaml` | ✓ |
| `sbatch scripts/slurm/conditional_encdec_sft_residual_head.sbatch` | Stage one instruct-SFT run with `RESIDUAL_HEAD_PRESET=mlp` (depth 2) or `transformer` (depth 4), optional `RESIDUAL_HEAD_LAYERS`, and required `RESIDUAL_HEAD_MIGRATED_BASE`. Uses four H200s, BF16 and FA4. Run the alignment audit first; see the [runbook](performance/residual-feature-head.md). | |
| `torchrun -m scripts.train.conditional_encdec_sft_grpo` | Experimental sequential self-conditioning GRPO over authoritative structured-SFT conversations. Runtime geometry is mandatory; `none` is the default initial conditioning. Even one-rank runs use torchrun so DistMuonAdamW owns the update. Config: `configs/training/conditional_encdec_sft_grpo.yaml`. | ✓ |
| `python -m scripts.preprocess.prepare_encdec_rlvr` | Prepare pinned question-only ARC/MMLU/GSM8K JSONL and extraction inputs; after schema-5 prompt-cache conversion, `stage=finalize` authenticates the complete bundle. Config: `configs/data/prepare_encdec_rlvr.yaml`. [Workflow](pipelines/encoder-decoder-rlvr.md). | |
| `torchrun -m scripts.train.conditional_encdec_rlvr` | Generated verified-reward rollouts with feature-policy gradients and a frozen LM head; `train_feature_head_only=true` trains only the complete feature head. Turn-delimited checkpoints only; geometry and hardware are explicit. `tasks=[ARC-Easy,ARC-Challenge,MMLU]` selects MC-only training; validation remains all four tasks. WandB enabled by default. Config selection uses `config_name=...`. | ✓ |
| `bash runs/rlvr_depth2_s1125_full_h200.sh` | Pinned step-1125 residual-head experiment: one H200, BF16/FA4, four tasks, 8 prompts × 4 rollouts × 100 steps. Requires the authenticated base, finalized bundle, statistics, tokenizer root, and immutable source revision; see the [RLVR runbook](pipelines/encoder-decoder-rlvr.md#depth-2-step-1125-experiment). | ✓ |
| `python -m scripts.eval.encdec_rlvr` | Evaluate the same canonical feature policy and verifier as RLVR training. Default fixed 32/task panel; `validation_per_task=null` evaluates every prepared heldout row. Config: `configs/eval/encdec_rlvr.yaml`; selection uses `config_name=...`. | ✓ |
| `python -m scripts.train.run_encdec_sft_pipeline` | Orchestrate chained encoder-decoder SFT phases; set `phase_order=capabilities,safety` for reverse order | ✓ |
| `python -m scripts.preprocess.convert_sft_to_parquet` | Convert SFT JSON / JSONL data into split-tagged train/val parquet shards. Legacy `full_nanochat_text` remains default; set `artifact_format=structured_nanochat_rollout_v1` for the separate GRPO artifact. The steered mixtures need `steered_jsonl=<path>` and accept `split=train` only (their val family is `standard`). |  |
| `python -m scripts.preprocess.mix_sft_into_pretraining` | Mix SFT conversations into pretraining shards (used during late pretraining) |  |
| `python -m scripts.preprocess.prepare_smollm3_cooldown` | Tokenize the smallest deterministic production-shard prefix needed for the pinned SmolLM3-3B cooldown; writes manifest-last `.ds`/`.ds.index` artifacts |  |
| `python -m scripts.train.smollm3_cooldown` | Continue the pinned public SmolLM3-3B step-200k weights to the repository's 524.288B-token target with standard Transformers Trainer DDP | ✓ |

For self-conditioning GRPO focused on ARC-Easy, ARC-Challenge, MMLU, and GSM8K,
convert with `mixture=chatcore artifact_format=structured_nanochat_rollout_v1`
and train with `config_name=conditional_encdec_sft_grpo_chatcore`. This defaults to
[head-only adaptation](pipelines/sft-self-conditioning-grpo.md#training-scope).
Add `train_feature_head_only=false` for full-policy training, auxiliary NTP, or
resuming an older full-policy checkpoint. See
[benchmark subset sources and commands](pipelines/sft-self-conditioning-grpo.md#benchmark-focused-subset).

Pinned Hugging Face decoder presets use the same entrypoint and loop:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m scripts.train.chat_sft -- \
  config_name=chat_sft_smollm3_stage1_step_40000

torchrun --standalone --nproc_per_node=8 \
  -m scripts.train.chat_sft -- \
  config_name=chat_sft_pythia_2_8b

torchrun --standalone --nproc_per_node=8 \
  -m scripts.train.chat_sft -- \
  config_name=chat_sft_pythia_1_4b
```

The presets load `HuggingFaceTB/SmolLM3-3B-checkpoints` only at commit
`524d818535ee98e56a4f0d621c84dd5c8fb8d3d4`, or final
`EleutherAI/pythia-2.8b` only at commit
`2a259cdd96a4beb1cdf467512e3904197345f6a9`, or
`EleutherAI/pythia-1.4b` only at commit
`fedc38a16eea3bd36a96b906d78d11d2ce18ed79`. Portable BF16 Hugging Face exports
remain model-only and explicitly non-resumable. Independently scheduled FP32
restart artifacts support exact same-topology continuation with
`huggingface.restart_save_every=N` and explicit
`huggingface.restart_from_step=N`.

The correctness-first four-node/eight-H200-per-node canaries are
`chat_sft_{pythia_1_4b,pythia_2_8b,smollm3_3b}_h200_4x8_benchmark`. Use the fail-closed
`scripts/slurm/hf_decoder_sft_h200_4x8.sbatch` launcher; it requires exact git,
lockfile, and data-manifest digests, a new absolute shared output root for fresh
runs or an exact retained complete marker for profiling-free continuation, runs
the repository NCCL preflight, records every benchmark/profile axis in distinct
per-invocation evidence, and writes rank-local paid-step rows. See the
[H200 decoder-SFT runbook](performance/hf-decoder-sft-h200.md).

For completion-only instruct → safety → capabilities chains, local/Slurm data preparation,
and harmfulness/steering evaluation, see [Campaign 2](pipelines/campaign2.md).

---

## Encoder-decoder pretraining — see [`docs/pipelines/encoder-decoder-training.md`](pipelines/encoder-decoder-training.md)

The encoder-decoder pipeline has a **5-step precomputation chain** before training:

1. **Extract** features → 2. **Compute statistics** → 3. **Precompute percentiles** → 4. **Precompute token boundaries** → 5. **Precompute packed rows**

| Command | Purpose | DDP |
|---|---|---|
| `python -m steering.extract_features` | Step 1: extract SAE features from a corpus into per-shard CSR safetensors. Config: `configs/extraction/extract_features.yaml`. Quickstart: `--config-name=extract_features_quick` | ✓ |
| `python -m steering.compute_statistics` | Step 2: aggregate corpus-wide feature statistics with a one-read exact float16 count reducer (zero magnitude-spill scratch; 15.4995 GiB CPU count table at 65k width; bounded CUDA key/count reduction when CUDA resolves). Optional `checkpoint_dir=... checkpoint_interval_shards=50` adds crash-safe compatible resume; `remote.enabled=true remote.staging_dir=... remote.expected_shard_count=N` streams an identity-sealed HF family with fixed one-body lookahead and a final inventory fence; `profile=true` writes the performance sidecar. |  |
| `python -m scripts.preprocess.precompute_percentiles` | Step 3: precompute magnitude→percentile overlays (uint8) for fast online lookup. Local mode supports DDP; opt-in `remote.enabled=true` is single-rank, streams a sealed HF raw family through bounded staging, publishes immutable receipts/statistics, and commits the schema-v2 manifest last. | local only |
| `python -m scripts.preprocess.publish_token_boundary_source` | One-time preparation for remote Step 4: stream the complete sealed HF raw family and publish deterministic compact v1 sidecars, then commit them with `manifest.json` last. Production sets explicit `raw_prefix`, `staging_dir`, and `expected_shard_count=1400`. Reruns verify immutable sidecars; no production upload is implied by the token-boundary runner. |  |
| `python -m scripts.preprocess.precompute_token_boundaries` | Step 4: precompute char↔token boundary maps for cross-tokenizer alignment. `remote.enabled=true` treats the configured HF raw prefix as the identity root, authenticates its derived compact v1 source, and streams only compact bodies through fixed one-shard lookahead while keeping exact paired parquets and overlays local. There is no raw fallback; production pins `remote.expected_shard_count=1400`. Local mode is unchanged. |  |
| `python -m scripts.preprocess.precompute_packed_rows` | Step 5: precompute packed training rows (best-fit document packing; records segment provenance natively and drains every loaded document through the final lookahead buffer). `remote.enabled=true remote.staging_dir=... remote.expected_shard_count=1400` lockstep-streams selected parquet-matched prefixes from sealed HF raw-feature and checksum-addressed percentile inventories; statistics, token overlays, parquets, packed outputs, provenance, certificates, and profiles stay local. Remote resume uses `resume=true`; `resume_from_shard` is rejected. |  |
| `python -m scripts.preprocess.validate_fp_target_source` | Stream and certify an existing packed-row family for compact FP targets: exact shard fingerprint/inventory, CSR structure, original-ID bounds, and per-segment uniqueness. New packed-row production certifies automatically. |  |
| `python -m scripts.preprocess.publish_remote_packed_inventory` | After final compact-FP certification, validate and atomically write the selection-independent schema-v2 issue-#806 packed-row inventory, including semantic layouts and the packed base hash. `selection_checksum=<64-hex>` is only an optional compatibility cross-check against an existing selection. Upload remains an explicit `scripts.utils.hf_push` step. `remote.enabled=true` instead signs an OFFLOADED family (issue #1463) by streaming each published body back once, and publishes the inventory beside the bodies itself. Config: `configs/extraction/publish_remote_packed_inventory.yaml`. |  |
| `python -m scripts.preprocess.publish_remote_inline_eval_source` | Against the complete producer-side raw family, validate the global schema-v2 token overlay and atomically certify one exact raw/token-leaf/parquet source for issue-#821 remote-packed clustered inline steering. Upload remains explicit. Config: `configs/extraction/publish_remote_inline_eval_source.yaml`. |  |
| `python -m scripts.preprocess.splice_feature_shards` | Optional (post-`compute_statistics`): recombine a base shard's segments into a derived feature dir (features verbatim, no model re-run) at `cut_fraction` (fraction of splice points selected as cuts). `pairing_mode=random` preserves the seeded permutation; `pairing_mode=opposite_percentile pairing_candidate_pool_size=64` favors incoming segments opposite to the continuations they replace, using local base statistics and bounded partner swaps. CPU-only. Supports opt-in serial async feature saves locally, or bounded verified HF-backed staging/publication with `remote.enabled=true`. Config: `configs/extraction/splice_feature_shards.yaml`. See [feature-shard-splice.md](pipelines/feature-shard-splice.md) | ✓ |
| `python -m scripts.preprocess.precompute_gradient_attribution` | Optional (gradient ranking, step G1): precompute gradient-attribution score overlays (`grad/<hash>/`) — full-model forward + per-segment backwards. Config: `configs/extraction/precompute_gradient_attribution.yaml` | ✓ |
| `python -m scripts.preprocess.precompute_selected_features` | Optional: precompute thresholded/sampled feature selections. Ordinary static `feature_clusters.path` selection applies mask then representative max-percentile; clustered splice-CF independently applies that transform to current/reference sides before computing representative deltas. Gradient ranking is mutually unsupported with clustering. `selection_resume=true` (default) reuses only exact-byte, stats, current-semantic, and post-serialization-validation-certified clustered progress. Each splice shard is reloaded and validated when published, and its signed running maximum CSR width must agree with the remote-loader certificate, so completion and unchanged resume skip the former large packed-body validation replay. Splice-CF resource controls include `selection_source_cache_shards`, `selection_sidecar_cache_shards`, `selection_num_workers`, `selection_shared_store` / `_capacity`, and disk-backed `selection_packed_spool_dir`; they never enter selection identity. `selection_splice_batch_size=64` is the default separately-addressed multi-segment ordinary-row batching mode with tensor-native CSR assembly; the exact size enters the hash/manifest because RNG grouping changes. Set it to null/1 explicitly to preserve legacy bytes and discover/resume a legacy selection family. Values `>1` worker run fail-closed host/cgroup/filesystem admission before output creation and emit `SPLICE_RESOURCE_ADMISSION` JSON. Opt-in `remote_preselection={staging_dir:...,max_output_shard_bytes:...}` requires clustered precomputed sampling and `remote_training.train_packed`. Splice-CF streams authenticated raw/percentile/packed bodies; one-worker `feature_delta=null` streams packed bodies only, replays a receipt prefix to restore RNG state, and uses no source inventory. Both publish body → receipt → timestamp-free manifest without changing selection identity. See [remote/streaming preselection](pipelines/encoder-decoder-training.md#remotestreaming-clustered-pre-selection), [resource-safe splice preselection](pipelines/encoder-decoder-training.md#resource-safe-parallel-splice-preselection-issue-758), [Static Feature Clusters](pipelines/encoder-decoder-training.md#static-feature-clusters-preselection-only), and [gradient attribution](pipelines/gradient-attribution-selection.md). |  |
| `python -m scripts.train.conditional_encdec_train` | Train the encoder-decoder model. Config: `configs/training/conditional_encdec_train.yaml`. Quickstart: `config_name=conditional_encdec_simple` | ✓ |
| `python -m steering.tools.migrate_encdec_train_state` | Offline native-checkpoint world-size migration. The canonical W31→W32 config preserves BF16/B14/accumulation-2 and the raw step, exactly reshards optimizer tensors, and restarts precomputed data at an audited replay-only shard boundary. Config: `configs/tools/migrate_encdec_train_state.yaml`. |  |
| `python -m scripts.train.conditional_encdec_selfcond` | On-policy self-conditioning post-training from an eligible FP-head checkpoint, using pretraining packed rows or SFT conversations via `selfcond.data_source`. A turn-delimited SFT source measures the generated-assistant cadence from its TRAIN artifacts and records it top-level as `turn_self_cond_segment_size_rustbpe`; there is no config knob, and the output checkpoint owns that inference cadence. Supports torchrun, rank-0 startup diagnostics, and opt-in per-rank synchronized timing or bounded CPU/CUDA traces; SFT spelling mixtures use the vendored offline word-list asset. Config: `configs/training/conditional_encdec_selfcond.yaml`. See [selfcond-post-training.md](pipelines/selfcond-post-training.md). | ✓ |
| `python -m scripts.preprocess.convert_decoder_to_encdec` | Convert a decoder-only checkpoint into an encoder-decoder one (encoder + cross-attn initialized from scratch) |  |
| `python -m scripts.analysis.dump_feature_mask <ckpt_dir>` | Emit a checkpoint's exact `feature_mask` as a paste-ready config. Handles unclustered, current effective-vocab clustered, and pre-effective clustered checkpoints; refuses ambiguous thin clustered metadata. **Needed to evaluate or resume any masked checkpoint** — see below. |  |
| `python -m steering.validate_features` | Validate CSR invariants on extracted feature shards (offsets monotonic, mags positive, every base segment non-empty / offsets strictly increasing, etc.) |  |
| `python -m scripts.utils.validate_token_boundary_overlays` | Validate precomputed token boundary overlays |  |
| `python -m scripts.preprocess.backfill_packed_row_shapes` | Backfill `max_n_seg` and raw observed `max_seg_size` metadata into older packed-row manifests (CUDA uses the raw value only as a semantic bound; CPU-reference FP8 alignment remains consumer-internal) |  |
| `python -m scripts.preprocess.extract_common_features` | Build a baseline-delta feature map (compares activations across two datasets, e.g. `safety_sft` vs `base`) |  |

If packed-row CPU memory grows across source-shard transitions, use the
[allocator-retention diagnostic and mitigation](onboarding/workflows.md#debug-packed-row-cpu-memory-growth)
before increasing `compress_workers`.

**Common invocations**

```bash
# 1-GPU validation extract (~5 min)
bash runs/extract_features_quick.sh

# 4×H200 production extract
torchrun --nproc_per_node=4 -m steering.extract_features num_shards=300

# Steered SFT mixture: convert train-only, then borrow standard's val family
python -m scripts.preprocess.convert_sft_to_parquet \
    mixture=standard_steered split=train steered_jsonl=/path/to/steered.jsonl
python -m scripts.preprocess.convert_sft_to_parquet mixture=standard split=val
torchrun --nproc_per_node=4 -m steering.extract_features \
    dataset_phase=standard_steered_sft
torchrun --nproc_per_node=4 -m steering.extract_features \
    dataset_phase=standard_sft_val

# PT remote percentile preflight: metadata only, no downloads or writes
uv run python -m scripts.preprocess.precompute_percentiles \
  --config-name=precompute_percentiles \
  feature_dir=out/data/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base/context_independent/no_bos \
  segment_size=128 device=cuda zstd_level=1 resume=true force=false \
  profile=true profile_summary_output=/persistent/precompute-percentiles-profile.json \
  dry_run=true \
  remote.enabled=true \
  remote.bucket_id=henryac/nano-gpt-pretrain-steer \
  remote.local_features_root=out/data/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base/context_independent/no_bos \
  remote.remote_features_root=l22_ci_nobos_raw \
  remote.expected_shard_count=1400 \
  remote.max_output_shard_bytes=1000000000 \
  remote.max_decompressed_shard_bytes=4000000000 \
  remote.staging_dir=<persistent-local-staging>

# Authorized production launch: identical contract, only dry_run changes
uv run python -m scripts.preprocess.precompute_percentiles \
  --config-name=precompute_percentiles \
  feature_dir=out/data/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base/context_independent/no_bos \
  segment_size=128 device=cuda zstd_level=1 resume=true force=false \
  profile=true profile_summary_output=/persistent/precompute-percentiles-profile.json \
  dry_run=false \
  remote.enabled=true \
  remote.bucket_id=henryac/nano-gpt-pretrain-steer \
  remote.local_features_root=out/data/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base/context_independent/no_bos \
  remote.remote_features_root=l22_ci_nobos_raw \
  remote.expected_shard_count=1400 \
  remote.max_output_shard_bytes=1000000000 \
  remote.max_decompressed_shard_bytes=4000000000 \
  remote.staging_dir=<persistent-local-staging>

# Pull features and materialize checksum-bearing percentile statistics.
# Existing root tokenizer metadata is preserved; dry-run performs no merge.
uv run python -m scripts.utils.hf_pull category=features

# Smoke-train the encoder-decoder
torchrun --standalone --nproc_per_node=1 -m scripts.train.conditional_encdec_train -- \
    config_name=conditional_encdec_simple \
    feature_dir=out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base

# Audit a complete flat W31 checkpoint leaf. The destination must not exist.
python -m steering.tools.migrate_encdec_train_state \
    source_checkpoint_dir=<checkpoint-leaf> \
    source_step=<NNNNNN> \
    packed_row_dir=<exact-packed-row-dir> \
    selection_dir=<exact-selection-dir> \
    destination_dir=<new-w32-checkpoint-leaf> \
    max_replay_rows=<operator-limit>

# Execute only after reviewing the emitted plan; all other overrides stay exact.
python -m steering.tools.migrate_encdec_train_state \
    source_checkpoint_dir=<checkpoint-leaf> \
    source_step=<NNNNNN> \
    packed_row_dir=<exact-packed-row-dir> \
    selection_dir=<exact-selection-dir> \
    destination_dir=<new-w32-checkpoint-leaf> \
    max_replay_rows=<reviewed-repeat-count> \
    dry_run=false

# Verified remote packed run on generic hardware. Byte transport is prefetched,
# but decoded-batch CUDA staging remains opt-in in this preset.
torchrun --standalone --nproc_per_node=8 -m scripts.train.conditional_encdec_train -- \
    config_name=cluster_fp/conditional_encdec_splice_cf_gemma3pt_c1024_remote_packed \
    feature_dir=<derived-c1024-feature-dir>

# Exact 8xH100 B=16/BF16 production launch. This target preset pins the
# production feature root and enables a 512 MiB/three-slot decoded-batch stage.
torchrun --standalone --nproc_per_node=8 -m scripts.train.conditional_encdec_train -- \
    config_name=cluster_fp/conditional_encdec_splice_cf_gemma3pt_c1024_remote_packed_8xh100

# Post-train an eligible FP-head checkpoint (unwired eval cadences must be off)
python -m scripts.train.conditional_encdec_selfcond \
    config_name=conditional_encdec_selfcond \
    model_tag=d12_fp_uncompacted \
    core_metric_every=0 sample_every=0

# Post-train on SFT conversations (the explicit base must be an SFT leaf)
python -m scripts.train.conditional_encdec_selfcond \
    config_name=conditional_encdec_selfcond \
    base_checkpoint_dir=/abs/.../safety_sft_conditioned_checkpoints \
    selfcond.data_source=sft \
    selfcond.train_scope=head_only selfcond.ste_form=dense
```

Run these from the primary repository root, or make both feature-root arguments
absolute paths to the same PT logical directory so the local statistics artifact
resolves. The dry run lists remote state but deliberately downloads no control
objects. The reviewed fresh level-1 launch therefore requires the reported output
prefix to be empty; if it is nonempty, stop for explicit contract inspection.
Lowering `expected_shard_count` does not create a canary because the source family
must be exactly contiguous. A canary needs a separately published source prefix.

Remote body mode requires one shared cache/lock namespace (it may span multiple
torchrun nodes). Signed inventory index `i` is downloaded only by spawned owner
`i % world_size`; every rank still reads the bodies required by its unchanged
row stripe. A custom remote mapping without `body_prefetch` is rejected rather
than selecting the removed consumer downloader. Concurrent jobs must use
separate `feature_root` cache namespaces. See
[Strict Rank-Striped Remote Body Prefetch](pipelines/remote-cache-materialization.md).
Compressed-body prefetch and decoded-batch staging are separate bounded layers:
the generic preset leaves staging disabled, while the target H100 preset enables
it without changing the semantic run/checkpoint fingerprint. Issue #867's peer
status polling is a separate control-plane cost and is not fixed by staging;
the issue-#868 change is stacked on PR #870, which supplies that independent
polling fix.

**Recovering a checkpoint's `feature_mask` (issue #695).** The default frequency band (`feature_mask_freq_min: 1e-5` / `feature_mask_freq_max: 0.5`) auto-produces a blacklist, and a resolved mask now **compacts the encoder's embedding tables**. A compacted checkpoint must be evaluated / resumed with the *same* mask — but the mask lives in a **generated** `{feature_dir}/blacklist/freq_min*_max*.txt` that a `feature_dir` move or a statistics recompute destroys. Recover it from the checkpoint itself:

```bash
# Paste-ready Hydra block (stdout; provenance to stderr)
uv run python -m scripts.analysis.dump_feature_mask \
    out/encdec_checkpoints/.../bl_1a2b3c4d5e6f7a8b

# Or as an ids .txt to reference via feature_mask.path
uv run python -m scripts.analysis.dump_feature_mask <ckpt_dir> --format txt > mask_ids.txt
```

For an unclustered checkpoint it cross-checks `meta.feature_mask_canonical` against `encoder.feature_vocab_ids` through the mask-only constructor. For a clustered checkpoint it strictly rebuilds the canonical consolidation and accepts either the current effective vocabulary or the recognizable pre-effective mask-only migration shape, labeling which one it found. If cluster evidence exists but the mask metadata was thinned, effective ids cannot distinguish mask-excluded from consolidated-away members, so the command **refuses to emit anything** and tells you to recover the config/W&B record. See [encoder-decoder-training.md § Encoder Feature-Dictionary Compaction](pipelines/encoder-decoder-training.md#encoder-feature-dictionary-compaction).

### Gemma 3 / gemma-scope-2 sibling configs

Each pipeline/eval config has a `_gemma3` sibling that overlays the Gemma 3 +
gemma-scope-2 (v2) deltas onto its base config. Select with the same
`--config-name=` / `config_name=` form as the base config. See
[`docs/pipelines/gemma-scope-2.md`](pipelines/gemma-scope-2.md).

| Sibling config | Base config it overlays |
|---|---|
| `extract_features_gemma3` | `extract_features` |
| `compute_statistics_gemma3` | `compute_statistics` |
| `precompute_percentiles_gemma3` | `precompute_percentiles` |
| `precompute_token_boundaries_gemma3` | `precompute_token_boundaries` |
| `precompute_packed_rows_gemma3` | `precompute_packed_rows` |
| `validate_token_boundary_overlays_gemma3` | `validate_token_boundary_overlays` |
| `conditional_encdec_train_gemma3` | `conditional_encdec_train` |
| `base_eval_gemma3` | `base_eval` |
| `chat_eval_gemma3` | `chat_eval` |
| `completion_steering_eval_gemma3` | `completion_steering_eval` |
| `axbench_bpb_eval_gemma3` | `axbench_bpb_eval` |

---

## Evaluation — see [`docs/pipelines/evaluation.md`](pipelines/evaluation.md)

### Base / chat model evals

| Command | Purpose | DDP |
|---|---|---|
| `python -m scripts.eval.base_eval` | Unified base-model eval: CORE metric, BPB on train/val, qualitative samples. Config: `configs/eval/base_eval.yaml`. Quickstart: `--config-name=base_eval_quick` | ✓ |
| `python -m scripts.eval.chat_eval` | ChatCORE for native checkpoints, pinned public Qwen2.5-3B-Instruct/SmolLM3-3B, or one strict local repository HF `step_NNNNNN` export. Config: `configs/eval/chat_eval.yaml`; HF presets: `chat_eval_qwen2_5_3b_instruct`, `chat_eval_smollm3_3b` (require fresh `output_dir`). Local exports use `hf_revision=null`. [HF protocol](pipelines/evaluation.md#public-instruction-tuned-hf-baselines). `--config-name=chat_eval_turn` admits single-process static target-real turn checkpoints with matching schema-5 caches; `problem_start=125 max_problems=125` selects canonical problems `[125,250)` after checking cache coverage before any forward. No cadence required. See the [s32 runbook](pipelines/evaluation.md#turn-delimited-target-real-context-caches). | Native fixed / HF only |
| `python -m scripts.eval.chat_eval --config-name=chat_eval_self_conditioned` | Scalar fixed/turn ChatCORE with checkpoint-measured cadence and prefix-only real features; [both runbooks](pipelines/evaluation.md#self-conditioned-chatcore). | |
| `python -m scripts.eval.chat_eval --config-name=chat_eval_benchmark_steering` | Same scalar evaluator with per-benchmark targets and continuous `self_conditioned.benchmark_steering.strength=0..6`; [runbook](pipelines/self-conditioned-chatcore.md#benchmark-identity-interpolation). | |
| `python -m scripts.eval.build_chat_benchmark_targets` | Prepare six isolated fitting/calibration populations and one shared background; `dry_run=false` fits dense AxBench percentile contrasts. Requires checkpoint metadata and output directory. | |
| `python -m scripts.eval.build_harmfulness_safety_target` | Fit 128 WildGuard v2 safe harmful-prompt responses and safe benign non-refusals against the shared SmolTalk background; defaults to preparation-only, extraction batch one. [Safety interpolation](pipelines/harmfulness-interpolation.md). | |
| `python -m scripts.utils.record_fixed_cadence` | Measure fixed K from authenticated packed TRAIN windows into a fresh evaluation checkpoint leaf. Config: `configs/eval/record_fixed_cadence.yaml`. | |
| `python -m scripts.eval.prepare_chat_self_conditioned_cache stage=plan` | Plan schema-4 fixed prefix cache from checkpoint provenance; run emitted `extract.sh`, then repeat with `stage=publish`. Config: `configs/eval/prepare_chat_self_conditioned_cache.yaml`. | |
| `python -m scripts.eval.merge_chat_eval_target_real` | Merge identity-matched target-real or self-conditioned ChatCORE task YAMLs with recomputed metrics and atomic publication. For problem shards, declare the exact task set and coverage with `task_totals={GSM8K:500,HumanEval:164}`; omitted declarations retain the complete six-task merge. Config: `configs/eval/merge_chat_eval_target_real.yaml`; required `input_files=[...]` and `output_file=...`. | |
| `python -m scripts.eval.harmfulness_eval` | Standalone harmfulness + benign-compliance eval. `generate_only=true` writes response receipts without a judge. Encoder-decoder uses dense `self_conditioned.safety_steering` and rejects sparse `feature_configs`; [runbook](pipelines/harmfulness-interpolation.md). Config: `configs/eval/harmfulness_eval.yaml` |  |
| `python -m scripts.eval.harmfulness_eval hf_checkpoint_dir=/path/to/step_000123` | Generate and judge with one concrete local repository Hugging Face SFT export. The export must satisfy the safetensors/BF16/SDPA metadata and saved-chat-template contract; Hub IDs and latest-step discovery are unsupported. |  |
| `python -m scripts.eval.harmfulness_eval --config-name=harmfulness_eval_l22` | Run the compact L22 harmfulness preset with real prompt caches and authenticated checkpoint cadence. Optionally set a fitted safety target; sparse feature maps are rejected. |  |
| `python -m steering.eval.dclm_eval` | DCLM CORE eval driver for encoder-decoder models |  |

### Steering evals

| Command | Purpose |
|---|---|
| `python -m scripts.eval.steering_eval` | Pairwise judge-based steering eval (does the steered output reflect the feature?) |
| `python -m scripts.eval.completion_steering_eval` | Ablation/insertion completion eval — measures steering causality |
| `python -m scripts.eval.shuffled_conditioning_eval` | Shuffled-feature control eval (random vs. real conditioning) |
| `python -m scripts.eval.axbench_upstream.run_baselines` | Run one vendored AxBench `train`, `latent`, or `steering` stage through its frozen uv environment. See [`docs/evaluation/axbench-upstream.md`](evaluation/axbench-upstream.md) |
| `python -m scripts.eval.axbench_upstream.build_project_input` | Build a model-free, input-only AlpacaEval grid directly from pinned prepared AxBench data. The c5 default contains five concepts, ten prompts per concept, and the full factor grid; it loads no Qwen or other generation model and publishes a required companion provenance manifest. Config: `configs/eval/axbench_upstream/project_input.yaml`. |
| `python -m scripts.eval.axbench_upstream.run_methods` | Run the two default corrected project methods over a validated legacy main frame or model-free project input. Supplying the strict `nl_target_source_manifest_path`, or using inherited config `methods_nl`, explicitly adds `NLSteering`; null preserves the two-method output and identity. Turn checkpoints use checkpoint-lineage prompt extraction and metadata-owned RustBPE refresh. Config: `configs/eval/axbench_upstream/methods.yaml`; there is no masked scale knob. World-size one and `torchrun --standalone --nproc_per_node=N -m ... -- device=cuda ...` share the same rank-striped, exact-resume executor. |
| `python -m scripts.eval.axbench_upstream.run_encdec_baselines` | Fit and run `EncDecDiffMean`, `EncDecLinearProbe`, `EncDecPromptSteering`, and `EncDecLoRA` under real fit/prompt conditioning and pure self-conditioned generation. Fixed-mode safe cells use physical batching; turn mode uses exact conditioned-SFT prompt geometry and scalar metadata-cadenced refresh. Config: `configs/eval/axbench_upstream/encdec_baselines.yaml`; fields use ordinary Hydra overrides. Supports the same one-node/shared-filesystem torchrun topology and changed-world-size logical-cell resume. |
| `python -m scripts.benchmark.benchmark_axbench_upstream compare\|profile\|throughput\|attention\|cost-model` | Strictly compare project/encoder-decoder bundles; profile a fresh normal producer run with per-rank regions and fixed selected-cell CPU/CUDA traces; collect one fresh low-overhead complete-run throughput sample without selected traces or per-region CUDA synchronization, optionally adding deferred current-stream CUDA timing for exact named coarse regions; benchmark historical/candidate packed-attention layouts against eager/compiled Flex and a legal isolated SDPA reference; or build explicitly labeled c2/c500 projections from at least three complete-run samples. See [`docs/evaluation/axbench-upstream.md`](evaluation/axbench-upstream.md#compare-profile-and-model-campaign-cost). |
| `python -m scripts.eval.axbench_upstream.merge_methods` | Validate and atomically merge main, aux, encdec-main, encdec-aux, and project Parquets into the ten-method frame. Uses native flags `--main`, `--aux`, `--encdec-main`, `--encdec-aux`, `--project`, `--output`, and `--max-concepts`; an existing output requires explicit `--force`. |
| `python -m scripts.eval.axbench_upstream.merge_baseline_project_methods` | Merge the four vendored baselines with the two default project methods. `--include-nl-steering` explicitly selects the seven-method product and requires the complete `NLSteering` grid; column presence alone never selects it. |
| `python -m scripts.eval.axbench_upstream.run_baseline_project_evaluation` | Judge the six-method baseline-plus-project frame, or pass `--include-nl-steering` to require and judge the dedicated seven-method NL product. Cache, cache-only, and API-base flags match `run_evaluation`. |
| `python -m scripts.eval.axbench_upstream.run_evaluation` | Run the vendored AxBench judge on the merged ten-method frame with an explicit SQLite cache. Uses `--input`, `--output-dir`, `--cache`, optional `--cache-only`, and optional `--api-base-url`. |
| `python -m scripts.eval.plot_axbench_histograms` | Render offline histograms of per-concept mean concept, fluency, and instruction scores. `pipeline=upstream` takes an evaluated directory and explicit `+method_factors={...}`; `pipeline=adv` takes a completed run and uses its recorded factors. GEPA is the default p0/p1/p2/p4/p8 pressure; add `+pressure=gcg` or `+pressure=soft_prompt` for the selected pressure's k-grid. Config: `configs/plot/axbench_histograms.yaml`. See [histogram usage](evaluation/axbench-upstream.md#across-concept-score-histograms). |
| `python -m scripts.eval.axbench_upstream.run_roundtrip` | Score every steered generation in the default ten-method merged frame by re-encoding it through its concept's own gemma-scope SAE feature. `--include-nl-steering` selects the exact seven-method baseline-plus-project NL frame. Uses `--input`, `--output-dir`, and optional `--device` (default `auto`). See [`docs/evaluation/axbench-upstream.md`](evaluation/axbench-upstream.md#roundtrip-metric) |
| `python -m scripts.eval.axbench_adv.fit_directions --config-name=fit_directions` | Fit project DiffMean and LinearProbe directions from full rendered AxBench training rows with real SAE conditioning and post-cross-attention capture. Requires prepared data, self-contained operative target metadata, feature statistics, a new/empty output root, concept count, layer, seed, and device. Publishes direction safetensors followed by the fit manifest; nonpositive calibration on a nonzero direction is counter-only and rejected by scaled consumers. Inputs must remain unchanged; no metadata-repair receipt or final rehash is required. |
| `python -m scripts.eval.axbench_adv.run --config-name=smoke` | Run the fixed two-concept, six-method AxBench-Adv campaign. `pressures` selects GEPA, additive counters, GCG, or opt-in `soft_prompt`; `[gepa]` preserves the legacy run. Soft prompts use `soft_prompt.learning_rate`, `max_steps`, and `optimized_token_counts`; their retained attacks are tensors. Requires explicit prepared/main/aux/evaluated/project Parquets, target metadata, feature statistics, fresh output, and separate judge/reflection caches. `AXBENCH_ADV_CONCEPT_LO`/`AXBENCH_ADV_CONCEPT_HI` select a checked half-open concept slice; `AXBENCH_ADV_METHODS` selects a checked method subset without weakening full source validation. See [`docs/evaluation/axbench-adv.md`](evaluation/axbench-adv.md). |
| `python -m scripts.eval.axbench_adv.merge_slices --input-root=<run> --input-root=<run> --output-root=<fresh-dir>` | Validate completed concept/method slices and rebuild pressure-specific aggregates and plots. GCG and soft prompts retain separate `gcg/` and `soft_prompt/` roots; source identities, effective config, factors, grids, controls, and final per-method concept sets must match. Soft producer tensors and receipts are authenticated in the input roots; merged outputs retain aggregate and source lineage. |
| `python -m scripts.eval.axbench_base_eval` | AxBench-Base full pipeline (Stages 0a-3e, exact context-cache boundary, then 4-6). DDP Stages 0b and 0c: `torchrun --standalone --nproc_per_node=N -m scripts.eval.axbench_base_eval stage=0b\|0c …`; other stages run rank-0-only. Clustered runs require paired path-backed `conditioning.feature_mask`/`feature_clusters`, an explicit method list, and one stage at a time. Supported clustered stages are `0a`, `0b`, `0c`, `1b`, `1c`, `2`, `3a`, `3b`, `3e`, and `4`; there is no dedicated clustered preset on main. Stage 3e enumerates exact encoder-decoder Stage 4 prompts; extract/convert them before generation. Stage 4 preflights every selected DiffMean/probe identity and training layer before output or generation. This launcher defaults `NCCL_TIMEOUT_SEC=1800` to tune its PG timeout. See [`docs/pipelines/axbench-base-eval.md`](pipelines/axbench-base-eval.md), [`docs/evaluation/axbench-base.md`](evaluation/axbench-base.md), [`docs/evaluation/axbench-base-advanced.md`](evaluation/axbench-base-advanced.md) |
| `python -m scripts.eval.axbench_bpb_eval` | AxBench-BPB: offline, BPB-based variant of AxBench-Base. Selected DiffMean/probe coordinates and their target residual layer are validated before DDP/model initialization. See [`docs/evaluation/axbench-bpb.md`](evaluation/axbench-bpb.md), [`docs/evaluation/axbench-bpb-advanced.md`](evaluation/axbench-bpb-advanced.md) |
| `python -m scripts.eval.axbench_negative_prepare` | Prepare the pinned first-N UltraChat test conversations with original messages, source hashes and grouped test identities. Config: `eval/axbench_negative/prepare.yaml`. |
| `python -m scripts.eval.axbench_negative_build` | Mine included-history feature activity, collect fixed neutral API draws, and publish complete rich/avoiding references with full-response SAE and independent quality checks. Checks the intersection of declared target context budgets before sampling. Config: `eval/axbench_negative/build.yaml`. See [natural-chat negative AxBench](evaluation/axbench-negative.md). |
| `python -m scripts.eval.axbench_negative_eval` | Evaluate a frozen negative bank with fresh target controls, native DiffMean/LinearProbe or encoder-decoder SelfConditioning, full-history judges and explicit failure coverage. Config: `eval/axbench_negative/eval.yaml`. |
| `python -m scripts.eval.axbench_bpb_adv_eval` | AxBench-BPB-Adv: adversarial candidate search over AxBench-Base Stage 0-3 artifacts with configurable steering y-axis. Default `steering_metric=roundtrip` generates steered completions and probes the original SAE feature; `steering_metric=bpb_class_margin` preserves the signed per-prefix BPB class margin. Rank-0 source preflight covers clustered and unclustered runs and installs immutable source-contract v6/v7 only after validating the one direction training layer and every runtime-required baseline/counter coordinate; failures are broadcast before model/output work. Production model-backed `build_candidates` automatically checkpoints and restores each completed non-control Faster-GCG concept group; no resume flag is needed, mismatched semantic identity fails loudly, and GEPA still reruns. `prepare_context_cache` can be launched with `torchrun --standalone --nproc_per_node=N`; other stages are rank-0-only. Implemented stages: `validate_source`, `prepare_context_cache`, `prepare_attack_rows`, `build_candidates`, `score`, `score_general_val`, `aggregate`, `plot`. The full Faster-GCG grid can be impractical, so for a cheap unclustered spot-check pass `--config-name axbench_bpb_adv_timing_probe`. See [`docs/evaluation/axbench-bpb-adv.md`](evaluation/axbench-bpb-adv.md) |
| `python -m scripts.eval.clean_axbench_source` | Build a filter-derived clean AxBench-BPB source sibling for BPB-Adv fail-loud source validation. Rewrites only synth parquets, copies or hardlinks all other artifacts including seed-v2 files/manifest, preserves the validated direction layer and identities, writes `cleaning_manifest.json`, and self-verifies under source-contract v6/v7. Legacy seeds are rejected rather than rewritten. |
| `python -m scripts.eval.validate_axbench_bpb_adv_output_root` | Validate the fixed BPB-Adv output root contains only current v6/v7 contracts and no migration control-plane entries. |
| `python -m scripts.eval.axbench_bpb_adv_generate_samples` | AxBench-BPB-Adv per-datapoint sampled completions against a FINISHED run. Legacy/unclustered behavior samples decoder-only families. A clustered config may additionally replay `encdec` families from the exact scored context cache and persisted fair/roofline bags, with the feature/source identity stamped into immutable YAML. Config: `generate_samples` (`generation:` block + top-level `counter_direction_family`); keep target blocks, cluster selection identity, and `steering_metric` identical to the finished eval. Do not mirror score-stage `roundtrip_steering.enabled` for sample generation. See [`docs/evaluation/axbench-bpb-adv.md`](evaluation/axbench-bpb-adv.md#per-datapoint-sampled-completions) |
| `python -m scripts.eval.plot_steering_strength_families` | Standalone replotter for AxBench-BPB-Adv steering-strength family plots. Reads finished `macro.parquet` artifacts for one or more run tags and writes `steering_strength_plots/<mode>/steering_strength_families.png` (or `--plots-dir` for concatenated runs). Use `--metric roundtrip` or `--metric bpb_class_margin`. |

Faster-GCG candidate loss evaluation uses exact-grouped shared-prefix KV reuse.
Use a fresh run tag for the new implementation. The optional scoped batch
override does not change gradient or GEPA batch composition:

```bash
uv run python -m scripts.eval.axbench_bpb_adv_eval \
  --config-name axbench_bpb_adv_timing_probe \
  stage=build_candidates \
  run_tag=gcg_shared_prefix_probe_001 \
  adversarial_methods.0.candidate_eval_device_batch_size=16 \
  +profile.enabled=true
```

Use `candidate_eval_device_batch_size=null` to inherit the top-level
`device_batch_size`.

For BPB-Adv, fresh `stage=all` runs fail before expensive work when their
selected modes lack the fair encoder-decoder Feature-DiffMean target. Direct
`stage=plot` validates the finished macro artifact and preserves existing plots
when that target is missing.

Clustered AxBench-Base has no `stage=all` mode and no dedicated Hydra profile on
main. Add the absent nested cluster key with `+`, pass the paired path-backed
mask, and select the stage's exact allowlist. The source-stage allowlist is the
six positive/negative decoder-only DiffMean, linear-probe, and control methods;
Stages `3e`/`4` additionally allow the encoder-decoder controls and ordinary
`ours_*` context-editing methods listed in
`steering/eval/cluster_config.py`. Preserve these overrides across each stage:

```bash
uv run python -m scripts.eval.axbench_base_eval \
  stage=0a \
  enabled_methods='[diffmean_decoder_only,neg_diffmean_decoder_only,linear_probe_decoder_only,neg_linear_probe_decoder_only,control_decoder_only,neg_control_decoder_only]' \
  conditioning.feature_mask='{mode:whitelist,path:/abs/path/accepted_feature_ids.txt}' \
  +conditioning.feature_clusters.path=/abs/path/feature_clusters.yaml \
  concept_feature_mode=canonical \
  <the ordinary feature/model/run overrides>
```

BPB-Adv already declares its two top-level keys, so override them as complete
mappings and keep `base_methods` explicit. This currently exercises clustered
preflight and artifact stages, but scoring reaches the known intervention
blockers documented above:

```bash
uv run python -m scripts.eval.axbench_bpb_adv_eval \
  feature_mask='{mode:whitelist,path:/abs/path/accepted_feature_ids.txt}' \
  feature_clusters='{path:/abs/path/feature_clusters.yaml}' \
  base_methods='[diffmean_decoder_only,neg_diffmean_decoder_only,linear_probe_decoder_only,neg_linear_probe_decoder_only,control_decoder_only,neg_control_decoder_only,feature_diffmean_encdec,neg_feature_diffmean_encdec,ours_gcg_encdec,neg_ours_gcg_encdec]' \
  <the ordinary source/checkpoint/run overrides>
```

Clustered AxBench-Base, BPB-Adv, and BPB-Adv sample-generation preflights accept
`force_cluster_shard_revalidation=true`. The default certified path still
rehashes compressed selection bytes but reuses a signature-valid packed
fingerprint cache. The force flag performs the full packed/selection body
validation and atomically repairs the cache; it is rejected for unclustered
runs. Under `torchrun`, rank 0 alone performs and publishes the forced packed
body pass.

AxBench-Base Stage 0c defaults to shared-prefix prefill. To select the old
repeated-prefill algorithm for rollback or debugging, start a fresh Stage 0c
artifact set and pass:

```bash
uv run python -m scripts.eval.axbench_base_eval \
  stage=0c verification_prefill_mode=repeated ...
```

The mode is fingerprinted; it cannot resume artifacts created in the other
mode. The removed `batched_generation_mode` key is intentionally unsupported.

Stage 0c generates verification continuations through the existing OpenRouter
synth client by default, while retaining local SAE rejection scoring. The
shipped model can also be selected explicitly:

```bash
uv run python -m scripts.eval.axbench_base_eval \
  stage=0c verification_api_model=inclusionai/ling-2.6-flash \
  verification_api_prefix_wave_size=100 \
  synth.max_concurrent=100 synth.requests_per_minute=6000 ...
```

The default is `inclusionai/ling-2.6-flash`. The API branch derives its own
operational definition from raw Neuronpedia activation examples and never
sends the published Stage 0a description to the generator. It requires both
`OPENROUTER_API_KEY` and `NEURONPEDIA_API_KEY` and uses checkpoint schema 9.
The synth defaults are 100 concurrent calls and 6000 admissions/minute per
process. API mode speculatively proposes up to
`verification_api_prefix_wave_size=100` candidate prefixes per wave, capped by
candidate-attempt/pool and zero-hit-bail limits but intentionally not by the
remaining accepted-prefix target. With five seeds, a full wave contains 500
logical calls and keeps at most 100 workers busy. Every proposal is locally
SAE-scored; only the first target passes in candidate order are retained. The
continuation request repeats the independently validated operational definition
as quoted data in a dedicated literal-format system instruction and uses a
contract-first user prompt; definition inference remains user-only. It
authenticates a 96-character local suffix, and later attempts receive a fixed
failure-class correction without reflecting rejected provider text. A hard
applicability gate tells the provider to leave an incompatible prefix
realistically unsteered rather than inventing a new actor or domain merely to
activate the feature.
The separate `verification_cross_prefix_batch_size=4` knob controls only local
Gemma generation.
Set `verification_api_model=null` to select the exact schema-8 local
compatibility path. Switch modes only with a fresh Stage 0 artifact set or
`run_tag`. See
[`verification_api_model`](reference/configuration.md#verification_api_model--axbench-base-stage-0c-api-continuations).

Restrict AxBench-Base Stage 0a concept candidates to a curated whitelist
(concept-candidate filtering, distinct from `conditioning.feature_mask`)
before the Neuronpedia description fetch — point
`concept_feature_whitelist_path` at either whitelist artifact the
[autointerp filter](pipelines/autointerp-filter.md) produces:

```bash
uv run python -m scripts.eval.axbench_base_eval ... \
  concept_feature_whitelist_path=out/autointerp_filter/.../manual_feature_filter.yaml
```

See [`docs/pipelines/axbench-base-eval.md`](pipelines/axbench-base-eval.md)
and [`docs/reference/configuration.md`](reference/configuration.md#concept_feature_whitelist_path--axbench-base-stage-0a-concept-whitelist).

### Eval support / context caches

| Command | Purpose |
|---|---|
| `python -m scripts.eval.build_eval_context_cache` | Enumerate or convert/validate feature contexts for eval. For AxBench Stage 3e, set `required_contexts_manifest` and `extraction_tokenizer_id`; conversion validates committed JSONL/parquet hashes/order, derives BOS decoded length, and stamps the exact contexts fingerprint into cache metadata. Static-cluster CORE/sample caches additionally accept `feature_mask`, schema-v1 `feature_clusters`, and exact `stats_source_sha256`; they require sampling, `feature_delta=null`, and no `delta_weights` key. |
| `python -m scripts.eval.build_eval_context_cache --config-name=build_harmfulness_context_cache_l22 stage=enumerate` | Enumerate the exact 700 harmful + 500 benign phase-A L22 prompt contexts (`num_seeds=1`, `base_seed=42`) and their extraction parquet. |
| `python -m steering.extract_features --config-name=extract_eval_context_features_gemma3_4bpt_l22_f65k_ci_nobos_c1024_h200` | GPU-extract Gemma-3/Gemma-scope-2 L22 features from the enumerated harmfulness parquet; the SFT training feature shard is not a substitute. |
| `python -m scripts.eval.build_eval_context_cache --config-name=build_harmfulness_context_cache_l22 stage=convert` | Convert the 1,200 extracted prompt rows into the checkpoint-pinned clustered cache. |
| `python -m scripts.eval.build_eval_context_cache --config-name=build_harmfulness_context_cache_l22 stage=validate` | Re-open the L22 harmfulness cache and validate its identity, compact vocabulary, and 1,200-row count. |
| `python -m scripts.eval.validate_eval_context_cache` | Validate an eval context cache against live extraction. Supplying the same cluster/mask/statistics contract additionally verifies the cluster descriptor, effective-vocabulary fingerprint, representative-only rows, and clustered selection semantic marker. |
| `python -m scripts.eval.build_eval_context_cache --config-name=build_eval_context_cache_turn stage=enumerate` | Enumerate independent full/prefix `chat_target` rows with turn annotations. Use `stage=convert` after extraction, then `stage=validate`; conversion/validation require exact `checkpoint_meta_path`, JSONL/parquet inputs, raw shards, and checkpoint training statistics. |
| `python -m scripts.eval.validate_eval_context_cache --config-name=validate_eval_context_cache_turn` | Deep schema-5 validation: authenticate renderer-supervised target bounds, reconstruct every source segment, replay lineage-seeded selection, validate exact input provenance, and exercise production pooled/lockstep target construction. |

The mandatory fresh AxBench encoder-decoder order is: run `axbench_base_eval
stage=3e`, extract the emitted `stage4_contexts_extraction` parquet with
`steering.extract_features`, run `build_eval_context_cache stage=convert` and
`stage=validate` with the Stage 3e manifest, then run `axbench_base_eval
stage=4`. See the pipeline guide for the full override set. Old seed pools,
context caches, or generation batches do not satisfy this contract.

### Sanity-check evals — `steering/eval/sanity_check/`

These are minimal, fast-running variants used to validate eval pipelines themselves (e.g., when modifying judges or inference engines).

| Command | Purpose |
|---|---|
| `python -m steering.eval.sanity_check.run_all` | Run all sanity-check evals back-to-back |
| `python -m steering.eval.sanity_check.run_all --chatcore-only --source sft` | Run only standalone ChatCORE on the SFT checkpoint; no judge API key is required |
| `torchrun --standalone --nproc_per_node=N -m steering.eval.sanity_check.run_all --chatcore-only --source sft` | Shard standalone ChatCORE examples across `N` local GPUs; task totals are all-reduced and only rank 0 writes `results.yaml` |
| `python -m steering.eval.sanity_check.dclm` | Sanity-check the DCLM CORE eval |
| `python -m steering.eval.sanity_check.harmfulness` | Sanity-check the harmfulness judge |
| `python -m steering.eval.sanity_check.inference` | Sanity-check the inference engine |
| `python -m steering.eval.sanity_check.steering` | Sanity-check the steering-eval baseline |
| `bash steering/eval/sanity_check/run_chatcore_delayed.sh` | Run a chat+CORE sanity-check eval with a delayed start (sleep, then launch) |

Standalone ChatCORE first pools ARC-Easy, ARC-Challenge, and MMLU prompts across
task boundaries, sorts them by length, and assigns whole globally formed chunks
to ranks. The target is one chunk per rank with a hard cap of 150 rows; measured
padded-token and quadratic-attention budgets split unusually long prompts into
smaller chunks. The default 50-problem run stays within both budgets and uses
width 150 on one GPU, 75 on two GPUs, and 19 on eight GPUs. The decoder still
computes every prompt hidden state, but the output
projection receives only each row's answer position and the union of valid
answer-token rows instead of materializing `(B, T, vocab)` logits. Task identity
is restored before scoring. CUDA allocator cache from this high-width phase is
released before generative staging admission.

The same run pools HumanEval, GSM8K, and SpellingBee prompts by generation
budget, sorts each pool by prompt length, and assigns whole globally formed
microbatches to ranks. CUDA targets three former per-task chunks, capped at 150;
for the default 50-problem run that is width 150 on one GPU, 75 on two GPUs, and
21 on eight GPUs. CPU keeps the portable two-chunk cap of 100. Task identity and
original row position are restored before scoring. On admitted pre-Hopper CUDA
runs, the 150-row path derives a shorter initial K/V allocation from current free
VRAM after reserving compute-dtype linear shadows and fixed headroom. It grows
layerwise to the final context length only after finished-row compaction makes
that fit within the initial row-token budget; failed admission or growth retries
the same ordered work in chunks of at most 100. The suite sizes persistent weight
staging against the admitted initial K/V projections and retains one
compute-dtype shadow set across all of its chunks; nested batch contexts reuse
that mapping. Hopper-or-newer CUDA uses the full 150-row cache without weight
staging. Before this admission check, CUDA runs collect unreachable categorical
temporaries and release their allocator cache so the device-wide free-memory
measurement reflects the live model rather than the preceding high-width phase.
On compute-capability 8.x CUDA, the already length-sorted mixed prompts are
prefilled in right-padded groups of at most 4,096 scheduled token positions.
Every padding position follows its row's real prompt, so causal attention keeps
it out of every real token's context; the per-row cache length excludes those
slots and decode overwrites them before they become visible. This batching can
change batch-shape numerics and completion bytes, while task scores remain the
correctness contract. On measured Ada devices (compute capability 8.9), these
ragged K/V rows use E4M3 storage while queries, new K/V inputs, accumulation, and
outputs remain BF16/FP32 as before. E4M3 halves K/V bytes and is another explicit
numerical boundary: it need not preserve completion bytes, but standalone task
scores must stay within the benchmark tolerance. Other CUDA devices and CPU
partition mixed prompts into exact-length, padding-free groups. Each exact-length source group stays within
the single-source row-token cache envelope already covered by admission. Both
paths assemble the real K/V prefixes into one cache, then decode uses
per-row RoPE positions with native FA3 ragged-cache support or an explicit
prefix/window mask. The measured Ada path also applies the vendored MLP's
ReLU-squared activation in place with one fused Triton kernel on the fresh
MLP-up tensor; the BF16/FP16 result is unchanged while one intermediate and one
kernel launch are removed. Its one-token decode path also reads the strided Q/K
projection views directly and fuses RoPE plus RMSNorm. That fusion retains
BF16/FP16 outputs within the documented task-score tolerance but need not retain
completion bytes. Other devices retain the vendored activation and Q/K paths.
Compute-capability 8.x CUDA consumes the attention mask in a fused
one-token Triton attention/cache-update kernel; other non-FA3 devices retain
SDPA. Equal-length batches retain the
ordinary batched prefill and uniform-position path, which tracks cache position on
the host and checks generated tokens in short chunks instead of synchronizing
after every token. These batching choices
intentionally do not preserve completion bytes, categorical batch numerics, or
evaluation/RNG ordering; task scores are the correctness contract. On CUDA systems without
Flash Attention 3, the uniform SDPA fallback's position tensor also stays on CPU
because that backend only uses it for host-side uniform-position indexing; K/V
storage remains on GPU. The ragged SDPA fallback casts the one-token query to the
cache storage dtype and attends to the K/V cache directly, avoiding a full-cache
dtype round trip on every layer and decode step. The compute-capability 8.x
Triton path uses the same mask and cache positions, reads the selected cache
storage dtype into FP32 accumulation, supports GQA, and updates K/V before its
online-softmax attention without materializing an attention matrix. GSM8K and SpellingBee use the
same CUDA ragged groups with per-step calculator handling: sampled
`<python_start>` / `<python_end>` spans
are evaluated through the Engine's calculator, and `<output_start>` / result /
`<output_end>` tokens are forced independently per row. On pre-Hopper CUDA with
the Triton/SDPA ragged fallback, rows that have emitted a terminal token are
packed out of the live K/V batch at the next host check; the cache storage is
reused in place, and results retain original input order. This can change later rows'
floating-point/RNG schedule because GEMM batch shapes shrink, consistent with
the task-score correctness contract. Hopper-or-newer GPUs and native FA3 keep
the original batch width because smaller live batches regress their measured
throughput. Any shared batch containing GSM8K or SpellingBee enables per-step
calculator handling for every row; a tool-free batch retains chunked host checks
and restarts through the exact Engine path if it emits `<python_start>`.

### Feature filtering — see [`docs/pipelines/feature-filter.md`](pipelines/feature-filter.md)

| Command | Purpose |
|---|---|
| `python -m steering.eval.feature_filter.cli` | Filter SAE features by interpretability / quality before running steering evals |
| `uv run -m scripts.preprocess.precompute_autointerp_filter -- feature_dir=... output_dir=... token_examples.materialization_batch_size=1` | Generic CPU/reference invocation: score local singleton feature targets and emit `accepted_feature_ids.txt`. Slot-block size `128` is the repository default but requires the native-CUDA Gemma CI/no-BOS two-stream singleton contract (the production preset satisfies it); `1` is the unchanged serial reference. Judge `max_concurrent` defaults to 8. See [`docs/pipelines/autointerp-filter.md`](pipelines/autointerp-filter.md). |
| `uv run -m scripts.preprocess.compute_manual_filter_coverage -- feature_dir=... selection_yaml_path=...` | Validate a manually uncommented autointerp YAML list and print exact per-segment selected-feature coverage. |
| `uv run -m scripts.eval.autointerp_prompt_calibration -- trial_audit_path=... judge.prompt_variant=delphi judge.few_shot_set=delphi` | Ad hoc replay of an existing autointerp `trial_audit.yaml` against prompt/few-shot variants without re-forwarding SAE examples |

### Static feature-cluster preselection

`feature_clusters` consumes a strict static schema-v1 YAML/JSON file; it does
not compute clusters. Use the same mask, cluster path, feature selection, delta,
and bucket settings for precomputation and training:

```bash
uv run python -m scripts.preprocess.precompute_selected_features \
    feature_dir=<feature_dir> packed_row_dir=<packed_row_dir> \
    +feature_clusters.path=<clusters.yaml> feature_delta=null \
    feature_selection.delta_weights=null
```

The leading `+` is required because the shipped Hydra config declares
`feature_clusters: null`. The producer writes a content-addressed `sel/<hash>/`
directory and marks it complete only after validating and fingerprinting every
compressed shard. Clustered training requires
`use_precomputed_batches=true`; it never falls back to live/fused selection.
The same resolved clusters size the encoder vocabulary: only mask-allowed
representatives and unclustered allowed ids receive physical rows; members
consolidated into a representative are absent and hit the ordinary out-of-vocabulary
guard. With no clusters, construction is the literal mask-only path.
For splice-counterfactual training, pass the same cluster override to the
derived-train, derived-validation, and delta-free base-validation selection
precomputations. Positive `steering_eval_every` is supported only on this
splice-counterfactual precomputed path: trainer startup validates the derived
validation selection/provenance and constructs the bounded in-memory clustered
inline payload. It has no separate CLI or persistent steering-cache artifact.
See
[encoder-decoder-training.md § Static Feature Clusters](pipelines/encoder-decoder-training.md#static-feature-clusters-preselection-only)
for the exact support matrix and failure boundaries.

The completion TUI does not accept a `feature_clusters` override. It discovers
an encoder-decoder checkpoint, resolves the checkpoint's canonical mask and
cluster metadata, and reproduces supported ordinary percentile selection from
the matching base feature shards. Clustered launches require the matching
percentile overlay and reject delta, gradient-ranked, and raw-magnitude
selection semantics.

---

## Serving & chat — see [`docs/pipelines/serving.md`](pipelines/serving.md)

| Command | Purpose |
|---|---|
| `python -m scripts.chat.chat_cli -p "your prompt"` | Interactive CLI chat with a decoder-only chat model. Omit `-p` for an interactive loop |
| `python -m scripts.chat.chat_web` | Launch the FastAPI ChatGPT-style web UI |
| `python -m scripts.chat.steer_chat_cli` | Interactive CLI chat with the encoder-decoder model + SAE feature steering. Encoder-decoder checkpoints must use `fixed_tokens`; turn-delimited checkpoints fail before generation setup. |
| `python -m scripts.completion_tui` | Textual TUI for SAE-guided completions, including compact-mask validation and clustered-checkpoint composite rows, member inspection, rep-space resampling, and rep-only generation. Checkpoint metadata is authoritative; there is no TUI cluster config key. The TUI requires a `fixed_tokens` checkpoint and matching fixed-token feature statistics; turn-delimited geometry fails closed before feature-row loading or generation. A non-lazy launch may already have loaded model weights when the checkpoint check fires (`lazy=true` rejects first). |
| `python -m scripts.chat_tui target_meta_path=/path/to/meta.json` | Textual multi-turn self-conditioned Chat with conversation selection, rewind/editing, and AxBench steering. Fixed checkpoints retain continuous-stream semantics. Turn-delimited post-training checkpoints use atomic full-history extraction/prefill epochs and their recorded RustBPE cadence; raw turn-SFT parents are unsupported. Both require matching SHA-bound statistics. See [serving](pipelines/serving.md#chat-tui). |

In the Chat TUI, `/regen` increments the seed and replaces the latest assistant
response using the current steering settings, preserving earlier turns and user
text. It also works on a cancelled response and uses fresh sampling even with
deterministic rewind enabled. A new seed can still produce the same text
(for example, with greedy decoding). A pending `/seed N` is advanced and consumed
by `/regen`.

---

## Profiling & benchmarking — see [`docs/performance/profiling.md`](performance/profiling.md)

| Command | Purpose |
|---|---|
| `python -m scripts.benchmark.benchmark_activation_methods` | Compare SAE activation extraction methods (forward-hook vs. nnsight, etc.) |
| `torchrun --standalone --nproc_per_node=1 -m scripts.benchmark.benchmark_encdec_rlvr -- ...` | Benchmark authenticated generated-token/feature RLVR updates, including replay and the optimizer. Saves synchronized samples and numerical parity receipts; optional traces and phase diagnostics run separately. See [head-only RLVR performance](performance/rlvr-head-only-compiler.md). |
| `python -m scripts.benchmark.benchmark_attention` | Microbenchmark compiled FlexAttention / FA4 / FA3 / SDPA on the non-cached decoder training path and record the distinct cached-generation route. |
| `uv run python -m scripts.benchmark.benchmark_chatcore checkpoint_dir=... step=999 output_dir=... grading=candidate` | Repeated six-task standalone ChatCORE on one CUDA GPU, with `grading=reference` for the original HumanEval result-manager path. Records synchronized latency, memory, source/input hashes, completion bytes, and per-program verdicts. `profile=true` adds a separate short decode trace; `record_logits=true` is a separate instrumented numerical audit. See [profiling](performance/profiling.md#standalone-chatcore-ragged-decode). |
| `python -m scripts.benchmark.benchmark_axbench_adv_gcg_hf input=... output=... method=DiffMean` | Replay AxBench-Adv baseline GCG gradients and scores, retaining full tensors, synchronized timings, and optional CUDA traces. `paired_state=true` compares the original state guard in alternating pairs. Uses the vendored environment with benchmark-only Hydra/Loguru dependencies; see [GCG performance](performance/axbench-adv-gcg.md). |
| `python -m scripts.benchmark.benchmark_axbench_adv_gcg_project input=... output=... method=SelfConditioning` | Replay project GCG through normal checkpoint/feature preparation in the root environment. Retains full tensors, timings, peak allocations, and optional compressed CUDA traces. `paired=true` compares `paired_reference=logits` (default) or `paired_reference=state`; `full_logits=true` restores the full LM head in an unpaired run. See [GCG performance](performance/axbench-adv-gcg.md). |
| `python -m scripts.benchmark.benchmark_axbench_adv_judge_overlap --config=/path/to/resolved-config.yaml --output=/fresh/path` | Provider-free real-CUDA AxBench-Adv project control and counter replay with paired serial/overlapped judging, exact text/result parity, stage timings, cache-call counts, and memory peaks. |
| `python -m scripts.benchmark.benchmark_axbench_adv_judge_pair --config=/path/to/resolved-config.yaml --output=/fresh/path` | Real-CUDA project control and counter replay with the actual judge worker against a local fake API; compares serial and concurrent concept/fluency calls with exact result parity and bounded request counts. |
| `python -m scripts.benchmark.bench_generate_samples --device {cuda,cpu} --iterations N [--stop-mode {none,compaction}]` | Load the canonical `d12_base` decoder, warm up five times on a fixed ~400-token prompt, then benchmark `InferenceEngine.generate_samples` at N=4 and a 32-token budget. JSON includes synchronized aggregate timing, per-iteration wall times with median/p95, end-to-end tokens/s, decode-region seconds, and profiler counters. `none` is the default full-budget path; `compaction` uses seed 1260, temperature 1.5, and stop ids divisible by 32 (calibrated against the pinned `d12_base` checkpoint for staggered finishes), and fails unless rows finish and the active batch compacts. No run directory is created. |
| `python -m scripts.benchmark.benchmark_hf_harmfulness_generation --checkpoint-dir /path/to/step_NNNNNN --device cuda:0` | Generation-only real-prompt repository-HF harmfulness benchmark. By default it compares the former serial forward shape with the production left-padded batch and can write traces with `--profile-dir`. `--batch-sizes 32 64 128` instead measures fixed-prompt outer-batch scaling, throughput, repeatability, and peak allocation without running the serial reference. It never constructs a judge client. |
| `python -m scripts.benchmark.benchmark_progression_core_numerics capture\|compare ...` | Capture the target logits and losses actually scored by a real target-aware CORE run, then tolerance-gate two captures from different revisions. The comparator also checks scored-row identity, top-1 agreement, top-50 overlap, cosine similarity, and final metric drift. |
| `python -m scripts.benchmark.benchmark_autointerp_concurrency --n-targets 12 --n-trials 8 --max-concurrent 8` | Single-process synthetic serial-vs-concurrent (`judge.max_concurrent`) normalized final-artifact parity and timing harness, proving concurrency changes throughput without changing output bytes. Uses a fake provider/rate-limited-sleep judge; it is not a real GPU/data/API benchmark. |
| `uv run python -m scripts.benchmark.benchmark_autointerp_token_examples --output-dir ... --report ... --targets ... --instrumentation off` | Real-data/model token-example materialization benchmark. `--targets` are reference/parity anchors; optional `--materialization-targets` is their block-filler superset. `off` is required for headline throughput; tier 1/2 are diagnostics. See the [`autointerp-filter` pipeline guide](pipelines/autointerp-filter.md). |
| `PYTHONPATH=. uv run python -m scripts.benchmark.measure_autointerp_model_vram` | Load the real production autointerp model + SAE stack once and record peak/resident VRAM, backing the single-resident-model design (in-process concurrency shares one loaded model instead of one per shard worker). |
| `python -m scripts.benchmark.benchmark_splice_feature_shards` | Benchmark post-extraction feature-shard splicing on a bounded, isolated mirror of real shard/parquet inputs; emits per-phase and detailed profile medians/p90 plus throughput |
| `python -m scripts.benchmark.bench_encoder_vocab_compaction` | Compare the canonical 8xPRO6000 model at full versus compact feature vocabulary, including the FP head/objective. Uses five warmups and 30 CUDA-synchronized iterations; reports median/p95/IQR step time, initialization and step peak allocated/reserved VRAM, parameter counts, and `/proc/<pid>/smaps_rollup` RSS. |
| `python -m scripts.benchmark.profile_encdec_l22_training mode={data,train,distributed,topology}` | Profile the audited spliced/pre-selected l22 training path. `data` measures real-shard rank-striped CPU process scaling, `train` measures exact-checkpoint CUDA phases, `distributed` measures compiled H100/NCCL optimizer steps and optional controlled component/operator diagnostics under torchrun, and `topology` reports an exact optimizer collective inventory plus a clearly labeled hierarchical-ring sensitivity model for either the audited default or an explicit shape-only geometry. See the [encoder-decoder component profile](performance/encoder-decoder-training-components.md). |
| `torchrun ... -m scripts.benchmark.preflight_nccl_transport` | Run the bounded NCCL correctness/timing gate used by the [8 x 8 H200 launch runbook](performance/h200-8x8-launch-readiness.md). Requires an explicit `NCCL_TIMEOUT_SEC`; the Slurm wrapper tries defaults and then a host-staged fallback. |
| `python -m scripts.utils.storage_recovery_gate --checkpoint-file <model_NNNNNN.pt> --selection-dir <sel/hash>` | Inside an existing Slurm allocation, run the bounded compute-node checkpoint/selection storage gate before one retry attempt. Healthy or 3-probe exhaustion returns 0; malformed invocation, missing allocation/tooling, or unexpected worker failure returns 2. See the [8 x 8 H200 launch runbook](performance/h200-8x8-launch-readiness.md#allocation-local-storage-recovery-gate). |
| `python -m scripts.analysis.benchmark_splice_preselection` | Controlled splice-preselection benchmark matrix (serial, W8/C12, W12/C16, W16/C20 plus source-cache sweeps). Records exact commands/config, raw process-tree resource JSONL (PSS/USS/SwapPss, faults, I/O, CPU, PSI, cgroup, tmpfs, deleted mappings), per-stage/cache profiles, summaries, and output-tree parity inputs. |
| `python -m scripts.benchmark.profile_axbench_base` | Profile the AxBench-Base eval pipeline (find slow stages) |
| `python -m scripts.benchmark.benchmark_axbench_bpb_adv_roundtrip --output-root ... --run-tag ... --benchmark-dir ... [--metrics both\|bpb_class_margin\|roundtrip] [--sample-size N] [--sample-seed S]` | Run one or both production AxBench-BPB-Adv metrics. `--metrics` defaults to both; `--sample-size` deterministically stratifies by method/target/mode and gives each metric an isolated hardlinked derived run, while `--sample-seed` defaults to 0. Remaining arguments are Hydra overrides. |
| `uv run python -m scripts.benchmark.compare_roundtrip_score_tables A.parquet B.parquet [--json OUT] [--expect-control-drift | --expect-reseeding]` | Compare exact-schema AxBench-BPB-Adv roundtrip score tables by ordered `(candidate_id, sample_index)`. Row-order changes and columns outside the selected drift allowlist exit 1. `--expect-control-drift` is control-only. `--expect-reseeding` reports both sides' seed, generated text/token-count/stop-reason, and probe-score drift while still exact-gating identity, prompt/digest, editable-span, reference-count, target-feature, and teacher-forced-loss columns. The two flags are mutually exclusive. Non-finite drift values, missing keys, and schema mismatches are hard errors. |
| `python -m scripts.benchmark.profile_stage0c_verification` | Profile the AxBench Stage 0c serial `_verify_candidate` hot path with torch profiler + cProfile; set `STAGE0C_PROFILE_VERIFY_CANDIDATES=50` and pass `stage=0c verification_cross_prefix_batch_size=1` |
| `python -m scripts.profile_progression` | Profile training metrics across checkpoints in a run |
| `python -m scripts.profile_progression_microbench.encdec_build_conditioning` | Microbenchmark encoder-decoder conditioning construction |
| `python -m scripts.utils.verify_bf16_threshold` | Verify BF16 numerical-precision violations against an FP32 baseline |
| `bash scripts/bench_compare.sh` | Compare benchmark results between two runs |
| `bash scripts/setup_fa4.sh [--fa3]` | Sync and verify the locked FA4 production forward/backward path on Hopper+; optionally verify the FA3 KV-cache fallback |

---

## Sweeps & analysis

Compact FP sweep cells require precomputed packed rows with a completed
`fp_target_source_integrity` certificate, just like ordinary compact-FP
pretraining.

| Command | Purpose |
|---|---|
| `python -m scripts.train.run_encdec_sweep` | Hyperparameter sweep over encoder-decoder configs |
| `python -m scripts.train.run_encdec_sweep_fast` | Faster sweep variant (amortizes startup; reuses warmed model state) |
| `bash scripts/launch_sweep.sh N --all --output-dir out/sweep_h200` | Launch a multi-GPU sweep — N parallel processes, each on a different GPU, partitioning the configs |
| `python -m scripts.eval.analyze_sweep` | Analyze the results of a sweep (best-config selection, plots) |
| `python -m scripts.eval.plot_evals` | Auto-plot eval comparison charts across runs |
| `python -m scripts.analysis.dump_feature_mask <ckpt_dir>` | Emit a checkpoint's exact `feature_mask` as a paste-ready config (`--format yaml\|txt`, `--step N`). Cluster-aware round trips distinguish current effective and pre-effective mask-only shapes; thin clustered metadata is refused as ambiguous. |

---

## Artifacts (HF bucket sync) — see [`docs/pipelines/artifacts-sync.md`](pipelines/artifacts-sync.md)

| Command | Purpose |
|---|---|
| `python -m scripts.utils.hf_pull category=<name>` | Pull a category's files from the project's HF bucket |
| `python -m scripts.utils.hf_push category=<name>` | Push local files for a category up to the bucket |
| `bash runs/download_artifacts.sh` | One-command pull of all standard categories |
| `bash runs/upload_artifacts.sh` | One-command push of all standard categories |

Standard categories: `base_checkpoints`, `sft_checkpoints`, `encdec_checkpoints`, `features`. Defined in `configs/artifacts/default.yaml`.

---

## Validation & utilities

| Command | Purpose |
|---|---|
| `python -m steering.validate_features` | Validate CSR feature-shard invariants (offsets, mags, doc_ids) |
| `python -m scripts.utils.validate_token_boundary_overlays` | Validate token boundary overlays |
| `python -m scripts.eval.validate_eval_context_cache` | Validate eval context cache against live extraction code |
| `python -m scripts.utils.verify_bf16_threshold` | Verify BF16 vs FP32 precision deltas |
| `python -m scripts.preprocess.backfill_packed_row_shapes` | Backfill raw packed-row shape metadata into older manifests |
| `python -m scripts.utils.nnterp_weight_transform` | Apply TransformerLens-style weight transformations to a checkpoint |
| `python -m steering.tools.migrate_feature_head_metadata source_metadata=... weights=... output_metadata=...` | Validate an old linear/MLP head's tensor layout and write new canonical metadata without changing weights or optimizer files. Refuses overwrite and ambiguous schemas. Config: `configs/utils/migrate_feature_head_metadata.yaml`; see [migration instructions](pipelines/feature-head-metadata-migration.md). |
| `python -m scripts.test.capture_neuronpedia_fixtures` | Capture Neuronpedia API fixtures for offline tests |
| `python -m scripts.validation.check_neuronpedia_schema` | Validate Neuronpedia-derived data schemas |

---

## Reporting

| Command | Purpose |
|---|---|
| `python -m nanochat.report reset` | Clear the markdown training report and write a new header |
| `python -m nanochat.report generate` | Generate the final markdown training report from accumulated sections |

---

## Dev / data generation

| Command | Purpose |
|---|---|
| `python -m dev.gen_synthetic_data` | Generate synthetic identity / capability conversations |
| `python -m dev.repackage_data_reference` | Reference script showing how the FineWeb-Edu shards were originally repackaged |

---

## Shell scripts — `runs/`

End-to-end orchestration. Each script is heavily annotated; read the header before running.

| Script | What it does | Hardware |
|---|---|---|
| `runs/runcpu.sh` | Educational: tokenizer + tiny base model + tiny SFT on CPU/MPS (~30 min) | CPU / Apple Silicon |
| `runs/run_4090.sh` | Tokenizer + base + SFT on a single RTX 4090 | 1×4090 (24 GB) |
| `runs/extract_features_quick.sh` | One-shard extraction validation (~5 min) | 1×GPU |
| `runs/speedrun.sh` | Full pipeline: tokenizer → pretrain (d26) → eval → SFT → eval | 8×H100 (~3 h) |
| `runs/pretrain_4xh200.sh` | Speedrun variant for 4×H200 | 4×H200 |
| `runs/train_370.sh` | Pretrain d26 on 370 shards (~1.4B params) | Multi-GPU |
| `runs/train_370_l.sh` | Pretrain d27 on 370 shards (~1.98B params) | Multi-GPU |
| `runs/encdec_2xh200.sh` | Encoder-decoder training on 2×H200 | 2×H200 |
| `runs/encdec_4xh200.sh` | Full encoder-decoder pipeline (extract → stats → train) on 4×H200 | 4×H200 |
| `runs/sft_safety.sh` | Safety SFT pipeline | Multi-GPU |
| `runs/sft_d12_10k.sh` | Chained SFT (safety → capabilities) on a d12 base at step 10k | Multi-GPU |
| `runs/scaling_laws.sh` | Scaling-laws sweep (parallel depth scan) | Multi-GPU |
| `runs/miniseries.sh` | Mini experiment series — trains d12-d26 and logs metrics to CSV | Multi-GPU |

### Slurm submission scripts

| File | Purpose |
|---|---|
| `scripts/slurm/extract.sbatch` | Slurm template for feature extraction |
| `scripts/slurm/preflight_nccl_transport.sh` | Allocation-local bounded default/GDR0 NCCL transport selector for the 8 x 8 H200 runbook |
| `scripts/slurm/train.sbatch` | Slurm template for pretraining |
| `scripts/slurm/smollm3_cooldown.sbatch` | Two-node H200 torchrun launcher for the pinned SmolLM3 cooldown; requires explicit shared data/output paths and submission-time account/QoS |

---

## Distributed launch — quick reference

For DDP commands, the canonical launcher is `torchrun`:

```bash
# 8 GPUs, single node
torchrun --standalone --nproc_per_node=8 -m scripts.train.base_train -- depth=26

# 4 GPUs, single node
torchrun --standalone --nproc_per_node=4 -m scripts.train.base_train -- depth=20

# Manual rank set (rare — used in extract_features.py docstring for split-process extraction)
RANK=0 WORLD_SIZE=2 LOCAL_RANK=0 python -m steering.extract_features &
RANK=1 WORLD_SIZE=2 LOCAL_RANK=1 python -m steering.extract_features &
```

The `--` separator is mandatory between torchrun's own flags and the Hydra overrides passed to your script.

---

## Hydra config selection

Two patterns are used depending on the script:

**Pattern A — `@hydra.main`-decorated scripts** (e.g., `steering.extract_features`, `scripts.eval.base_eval`)

```bash
python -m steering.extract_features --config-name=extract_features_quick num_shards=2
```

**Pattern B — compose-API scripts** (e.g., `scripts.train.base_train`, `scripts.train.conditional_encdec_train`, all `chat_*`, all `*_sft`, `*_rl`, etc. — i.e., anything that runs under `torchrun`)

```bash
python -m scripts.train.base_train config_name=base_train_tiny_1gpu device_batch_size=8
```

The difference: `--config-name=` (with leading `--`) for pattern A vs. `config_name=` (no dashes) for pattern B. The former is Hydra's native CLI flag; the latter is parsed manually because torchrun-launched scripts can't use `@hydra.main`.

`scripts.train.conditional_encdec_train` forwards Hydra's ordinary, append,
force-add, and delete operators (`key=value`, `+key=value`, `++key=value`,
and `~key`). Quote delete overrides such as `'~feature_selection'` so shells
such as zsh do not interpret the leading tilde before Python receives it. The
manual parser continues to ignore torchrun launcher/positional tokens, but
fails loudly on Hydra's native `--config-name` / `--config_name` spellings
because this compose-API entrypoint requires `config_name=<name>`.

---

## Where to put new entry points

| Kind of code | Goes in |
|---|---|
| New training loop variant | `scripts/<name>.py` + `configs/training/<name>.yaml` |
| New evaluation | `scripts/<name>_eval.py` + `configs/eval/<name>.yaml` |
| New precomputation step | `scripts/precompute_<name>.py` |
| New steering eval helper | `steering/eval/<name>.py` |
| New end-to-end runner | `runs/<scenario>.sh` |
| New SFT task | `tasks/<name>.py` (subclass `Task` from `tasks/common.py`) |

For everything else, see [`docs/reference/file-reference.md`](reference/file-reference.md).

### Native decoder-only AxBench SFT evaluation

- `uv run python -m scripts.eval.axbench_upstream.run_native_decoder`: generate
  `NativeDecoderPromptSteering` using `configs/eval/axbench_upstream/native_decoder.yaml`.
  Requires the exact frozen aux input and SHA-256, explicit SFT checkpoint step,
  and its RustBPE tokenizer. `max_cells` bounds a resumable production pilot.
- `uv run python -m scripts.eval.axbench_upstream.run_native_evaluation`: target-only
  generation judging using `configs/eval/axbench_upstream/native_evaluate.yaml`.
  Requires `input_path`, producer `manifest_path`, `output_dir`, and `judge_cache`.
- `uv run python -m scripts.eval.axbench_upstream.run_native_baselines`: fit and/or
  generate native decoder `DiffMean`, `LinearProbe`, and real per-concept `LoRA`
  using `configs/eval/axbench_upstream/native_baselines.yaml`; the D14 preset is
  `native_baselines_d14.yaml` and requires the actual final step and all checkpoint/tokenizer
  hashes. Set
  `phase=fit|generate|all`; methods, concept range, checkpoint/input hashes,
  training data, factors, seeds, fit semantics, and runtime are authenticated.
- `uv run python -m scripts.eval.axbench_upstream.run_native_baseline_evaluation`:
  target-only judging for exactly one completed standalone method using
  `configs/eval/axbench_upstream/native_baseline_evaluate.yaml`.

See [native decoder prompt-steering semantics](evaluation/axbench-upstream.md#native-decoder-only-sft-prompt-steering)
and [native standalone baseline semantics](evaluation/axbench-upstream.md#native-decoder-only-standalone-baselines).

Native decoder-only adversarial generation and provider handoff:

- `python -m scripts.eval.axbench_adv.run_native_decoder`: target-only real GEPA
  over a completed native AxBench source; config `eval/axbench_adv/native_decoder.yaml`.
- `python -m scripts.eval.axbench_adv.run_native_counter`: fit checkpoint-own
  DiffMean and LinearProbe residual directions from canonical AxBench training
  pairs, then generate additive counter points at configured `magnitudes`
  (default 0.25, 0.5, 1 and 2).
  Config `eval/axbench_adv/native_counter.yaml` pins D12 decoder block-output
  layer 9; `native_counter_d14.yaml` requires the actual D14 SFT step and three
  hashes, pins D14/896/1024 with 399,115,836 parameters and layer 11, and consumes
  only canonical concepts 0..49 from the authenticated standard100 source.
  `phase=fit|generate|all`, `max_cells`,
  `point_methods`, and `point_magnitudes` change placement only; exact artifacts
  and cells resume under one immutable full-grid identity. Concurrent workers
  must select disjoint points.
- `python -m scripts.eval.axbench_adv.run_native_provider`: separately authorized
  CPU consumption of durable judge/reflection artifacts; config `native_provider.yaml`.
- `python -m scripts.eval.axbench_adv.judge_native_points`: resumable final AxBench
  judging of every declared native Adv point and concept slice; config `native_final_judge.yaml`.
- `python -m scripts.eval.axbench_adv.run_native_methods`: authenticated native
  DiffMean, LinearProbe, and LoRA Adv generation; configs `native_methods.yaml`
  (D12) and `native_methods_d14.yaml` (actual final step required). Both reviewed
  geometries accept the explicit `zero_control_source=runtime` fresh-control
  policy; archived replay remains the default.
  `phase=pilot` is provider-free and requires one concept; `phase=run` stages
  GEPA plus both raw-direction counter families through a separate mailbox.
- `python -m scripts.eval.axbench_adv.judge_native_method_points`: final judging
  and separate GEPA/counter aggregation for the staged standalone-method points;
  config `native_method_final_judge.yaml`.
- `python -m scripts.eval.axbench_adv.run_encdec_baseline_counter`: explicit
  four-method encoder-decoder counter lane; config `encdec_baseline_counter.yaml`.
  Use `phase=preflight|pilot|generate|judge|finalize|merge`. The producer's own
  judged c50 factor sweep and authenticated fits are prerequisites. See
  [encoder-decoder baseline counters](evaluation/axbench-adv-encdec-baselines.md).

See [native decoder-only AxBench-Adv](evaluation/axbench-adv.md#native-decoder-only-sft-prompt-steering)
for method semantics, source fences, accounting, and the shared search population.
For the calibrated encoder-decoder step-1013 and D14 step-1001 latent-only presets,
see [400M latent AxBench-Adv](evaluation/axbench-adv-400m-latent.md).
For native D12 step 1003, use `native_methods_d12_sft_latent_c50` or
`native_counter_d12_sft_latent_c50`; see the
[D12 grid evidence and historical prompt-runtime requirement](evaluation/axbench-adv-d12-latent.md).

### Counter partial-snapshot continuation

The counter-only replay workflow and its strict snapshot contract are documented
in [Counter continuation](evaluation/axbench-adv.md#counter-continuation-from-an-immutable-partial-snapshot).
Use `scripts.eval.axbench_adv.seal_counter_snapshot` to seal an operator-created
snapshot, `scripts.eval.axbench_adv.preflight_counter_resume` for CPU-only
admission, and the normal `scripts.eval.axbench_adv.run` to finish missing points.
`scripts.eval.axbench_adv.merge_counter_continuations` adds original-campaign
lineage and complete c50 coverage before the native merge.
